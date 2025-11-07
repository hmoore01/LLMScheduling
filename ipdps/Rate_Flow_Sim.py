"""
Rate-Flow LLM Scheduling Simulator (Epoch Aggregated)
----------------------------------------------------
This module reworks the simulator to a *fluid / rate-based* flow. Instead of
routing individual requests, we aggregate total work (e.g., tokens) per epoch
and drain that work through datacenter capacities. The design preserves your
power/energy/carbon/water accounting while drastically speeding up simulation.

Key Ideas
~~~~~~~~~
- Work unit: tokens (or any scalar work unit) per model_type per source DC.
- Schedule plan: For each (src_dc, model_type), a mapping to {tgt_dc: fraction}
  or absolute tokens. Fractions are normalized per (src_dc, model_type).
- Power plan: Per-DC instructions setting node or processor power states.
- Capacity: Each DC exposes tokens/sec by model, derived from GPU perf tables.
- Drain: In one step per epoch, each DC consumes up to capacity*E tokens, the
  remainder becomes leftover carried to the next epoch.
- TTFT surrogate: queueing-style function of utilization + network latency.

Integration
~~~~~~~~~~~
- Replace your old per-request entry with LLM_Simulator(..., mode="rate").
- Provide epoch_work_df with columns [src_dc, model_type, total_tokens].
- Keep your existing CSV readers; populate Processor.model_perf fields.

This file defines: Processor, Node, Datacenter, GeoNetwork, LLM_Simulator
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Iterable
from typing import Callable, Union
import math
import collections

# ---- Debug helpers (near imports) ----
import os

# ---- Rate-Flow defaults (solar) ----
DEFAULT_SOLAR_KW_CAPACITY = 1000.0  # same for all DCs
# simple bell-shaped 24h profile; values are fractions of capacity
DEFAULT_SOLAR_PROFILE_24H = [
    0.00, 0.00, 0.00, 0.00, 0.00,  # 0–4
    0.05, 0.20, 0.50, 0.75, 0.90,  # 5–9
    1.00, 0.95, 0.90, 0.80, 0.60,  # 10–14
    0.40, 0.20, 0.05,               # 15–17
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00  # 18–23
]

RFS_DEBUG = os.environ.get("RFS_DEBUG", "").strip().lower() not in ("", "0", "false", "no")

def _dprint(*args, force=False):
    if RFS_DEBUG or force:
        passthrough = args

def _norm_model_name(name: str) -> str:
    s = str(name).strip().lower().replace("-", "").replace("_", "")
    if "70b" in s:
        return "Llama70b"
    return "Llama7b"

# Sensible defaults used when only request-level ms exists
_DEFAULT_AVG_TOKENS = {"Llama7b": 512.0, "Llama70b": 256.0}

def _safe_float(x):
    if x in (None, "", "None", "NA", "NaN"):
        return None
    try:
        return float(x)
    except Exception:
        return None

def _finalize_perf_entry(ms_per_token=None, ms_per_request=None, avg_tokens_per_request=None, model_key=None):
    """
    Build a per-model perf dict, deriving ms_per_token from request-level data when needed.
    Accepts positional or keyword args; 'model_key' used only for selecting defaults.
    """
    mspt  = _safe_float(ms_per_token)
    mspre = _safe_float(ms_per_request)
    avgt  = _safe_float(avg_tokens_per_request)

    # If only request-level data is present, derive ms/token
    if mspt is None and mspre is not None:
        if (avgt is None or avgt <= 0) and model_key:
            avgt = _DEFAULT_AVG_TOKENS.get(model_key)
            _dprint(f"[PERF] Deriving avg_tokens for {model_key} via default -> {avgt}")
        if avgt is not None and avgt > 0:
            mspt = mspre / avgt
            _dprint(f"[PERF] Derived ms_per_token for {model_key}: {mspt:.6f} from ms_per_request={mspre} / avg_tokens={avgt}")

    out = {}
    if mspt is not None and mspt > 0:
        out["ms_per_token"] = float(mspt)
    if mspre is not None and mspre > 0:
        out["ms_per_request"] = float(mspre)
    if avgt is not None and avgt > 0:
        out["avg_tokens_per_request"] = float(avgt)
    return out



@dataclass
class ProcNode:
    """
    Combined Node + Processor:
      - Holds HW traits (accel_type, gpu_config), power (tdp_kw/idle_kw),
        and per-model perf (ms_per_request, ms_per_token).
      - Tracks availability (available_at_ms) for queueing.
    """
    __slots__ = (
        "node_id", "type_id", "accel_type", "gpu_config",
        "tdp_kw", "idle_kw", "model_perf", "available_at_ms"
    )

    def __init__(
        self,
        node_id: int,
        type_id: Optional[int],
        accel_type: str,
        gpu_config: str,
        tdp_kw: float,
        idle_kw: float,
        model_perf: Dict[str, Dict[str, float]],
    ):
        self.node_id = int(node_id)
        self.type_id = int(type_id) if type_id is not None else None
        self.accel_type = str(accel_type)
        self.gpu_config = str(gpu_config)
        self.tdp_kw = float(tdp_kw)
        self.idle_kw = float(idle_kw)
        self.model_perf = model_perf or {}
        self.available_at_ms: float = 0.0

    # --- simple helpers ---
    def estimate_exec_ms(self, model: str, rec: Optional[Dict[str, Any]] = None) -> float:
        """
        Estimate execution time. If you later pass tokens in `rec`, you can
        use ms_per_token here; for now we use ms_per_request from CSVs.
        """
        perf = self.model_perf.get(model) or self.model_perf.get(model.capitalize())
        if perf and "ms_per_request" in perf:
            return float(perf["ms_per_request"])
        return 1000.0  # fallback (shouldn’t hit with proper CSVs)



# -----------------------------
# Datacenter definition
# -----------------------------

@dataclass
class Datacenter:
    """
    Holds a flat list of ProcNode 'units' and dispatches workload to the least-busy eligible unit.

    Public API (unchanged to callers):
      add_node(node: ProcNode)
      apply_power_plan(plan_slice)
      schedule_request(model, arrival, net_latency_ms, source_dc, target_dc, **kw) -> dict
      report_global_stats() -> dict
      report_utilization()  -> float in [0,1]
      reset_epoch()
    """

    def __init__(
        self,
        dc_id: int,
        carbon_intensity_g_per_kwh: float,
        time_of_use_24h: Optional[List[float]] = None,
        cop_profile_24h: Optional[List[float]] = None,
        blowdown_ratio: float = 0.0,
        water_cycling_density: float = 0.0,
        potable_energy_intensity: float = 0.0,
        wastewater_energy_intensity: float = 0.0,
        water_static: float = 0.0,
        epoch_length: int = 900,
        debug: bool = False,
    ):
        self.dc_id = int(dc_id)
        self.debug = debug

        self.carbon_intensity_g_per_kwh = float(carbon_intensity_g_per_kwh)
        self.time_of_use_24h = time_of_use_24h or None
        self.cop_profile_24h = cop_profile_24h or None
        self.blowdown_ratio = float(blowdown_ratio)

        self.water_cycling_density = float(water_cycling_density)
        self.potable_energy_intensity = float(potable_energy_intensity)
        self.wastewater_energy_intensity = float(wastewater_energy_intensity)
        self.water_static = float(water_static)

        self.epoch_length_s = int(epoch_length)
        self.epoch_length_ms = float(self.epoch_length_s * 1000)

        # Flat list of ProcNode execution units
        self.units: List[ProcNode] = []

        # Eligibility controls
        self._enabled_node_ids: Optional[set[int]] = None
        self._enabled_type_ids: Optional[set[int]] = None

        # Per-epoch aggregates
        self._busy_ms: float = 0.0
        self._energy_kwh: float = 0.0
        self._carbon_g: float = 0.0
        self._water_m3: float = 0.0
        self._ttft_sum_s: float = 0.0
        self._ttft_count: int = 0
        self.energy_cost_agg: float = 0.0

    # ---------- inventory ----------
    def add_node(self, node: ProcNode) -> None:
        self.units.append(node)

    # ---------- power plan ----------
    def apply_power_plan(self, plan_slice: Optional[Dict[str, Any]]) -> None:
        if not isinstance(plan_slice, dict) or not plan_slice:
            self._enabled_node_ids = None
            self._enabled_type_ids = None
            return

        en_nodes = plan_slice.get("enable_nodes")
        en_types = plan_slice.get("enable_types")
        off_nodes = plan_slice.get("off_nodes")
        off_types = plan_slice.get("off_types")

        if en_nodes is not None:
            self._enabled_node_ids = set(int(x) for x in en_nodes)
        else:
            self._enabled_node_ids = set(u.node_id for u in self.units)
            if off_nodes:
                self._enabled_node_ids -= set(int(x) for x in off_nodes)

        if en_types is not None:
            self._enabled_type_ids = set(int(x) for x in en_types)
        else:
            inv_types = {u.type_id for u in self.units if u.type_id is not None}
            self._enabled_type_ids = None
            if off_types:
                self._enabled_type_ids = set(inv_types)
                self._enabled_type_ids -= set(int(x) for x in off_types)

        if self.debug:
            nn = "ALL" if self._enabled_node_ids is None else len(self._enabled_node_ids)
            tt = "ALL" if self._enabled_type_ids is None else len(self._enabled_type_ids)
            print(f"[DC {self.dc_id}] apply_power_plan -> enabled_nodes={nn} enabled_types={tt}")

    # ---------- scheduling ----------
    def _eligible_unit_iter(self):
        for u in self.units:
            if self._enabled_node_ids is not None and u.node_id not in self._enabled_node_ids:
                continue
            if self._enabled_type_ids is not None and (u.type_id is not None) and (u.type_id not in self._enabled_type_ids):
                continue
            yield u

    def _pick_least_busy_unit(self, arrival_ms: float) -> Optional[ProcNode]:
        best = None
        best_avail = math.inf
        for u in self._eligible_unit_iter():
            if u.available_at_ms < best_avail:
                best_avail = u.available_at_ms
                best = u
        return best

    def _energy_for_exec_kwh(self, u: ProcNode, exec_ms: float) -> float:
        return float(u.tdp_kw) * (float(exec_ms) / 3_600_000.0)

    def _carbon_for_energy_g(self, energy_kwh: float) -> float:
        return (self.carbon_intensity_g_per_kwh * energy_kwh)

    def _water_for_energy_m3(self, energy_kwh: float) -> float:
        return (self.water_static * energy_kwh) if self.water_static > 0.0 else 0.0

    def _cop_for_ms(self, t_ms: float) -> float:
        """
        Return COP for a given millisecond timestamp.
        - If a 24-value hourly profile is present, use hour-of-day.
        - Else, fall back to a default (3.0 if unset).
        """
        # Fallback COP if no profile/invalid entries
        cop_default = getattr(self, "cop_default", 3.0)
        try:
            prof = self.cop_profile_24h
            if prof and len(prof) == 24:
                # Derive hour-of-day (0..23) from milliseconds into a notional day.
                # If you track absolute epoch start time elsewhere, you can pass it
                # through and add here; this uses local time-of-day from t_ms alone.
                sec = (float(t_ms) / 1000.0) % 86400.0
                h = int(sec // 3600)  # 0..23
                cop = float(prof[h])
                # Guard against zeros/negatives in input
                return cop if cop > 0.0 else cop_default
        except Exception:
            pass
        return cop_default

    def _time_for_ms(self, t_ms: float) -> float:
        """
        Return COP for a given millisecond timestamp.
        - If a 24-value hourly profile is present, use hour-of-day.
        - Else, fall back to a default (3.0 if unset).
        """
        # Fallback COP if no profile/invalid entries
        time_default = getattr(self, "time_default", 0.15)
        try:
            prof = self.time_of_use_24h
            if prof and len(prof) == 24:
                # Derive hour-of-day (0..23) from milliseconds into a notional day.
                # If you track absolute epoch start time elsewhere, you can pass it
                # through and add here; this uses local time-of-day from t_ms alone.
                sec = (float(t_ms) / 1000.0) % 86400.0
                h = int(sec // 3600)  # 0..23
                time = float(prof[h])
                # Guard against zeros/negatives in input
                return time if time > 0.0 else time_default
        except Exception:
            pass
        return time_default

    def schedule_request(
        self,
        model: str,
        arrival: int,
        net_latency_ms: float,
        source_dc: int,
        target_dc: int,
        **kwargs,
    ) -> Dict[str, Any]:
        arrival_ms = float(arrival)
        u = self._pick_least_busy_unit(arrival_ms)
        if u is None:
            # No enabled capacity
            return {
                "ttft_s": float(net_latency_ms) / 1000.0,
                "start_ms": arrival_ms,
                "finish_ms": arrival_ms,
                "queue_delay_ms": 0.0,
                "exec_ms": 0.0,
                "energy_kwh": 0.0,
                "carbon_g": 0.0,
                "water_m3": 0.0,
                "node_id": None,
            }

        start_ms = max(arrival_ms, u.available_at_ms)
        queue_ms = max(0.0, start_ms - arrival_ms)
        exec_ms = u.estimate_exec_ms(model, kwargs)
        finish_ms = start_ms + exec_ms

        # Update unit state + busy
        u.available_at_ms = finish_ms
        self._busy_ms += exec_ms

        # Accounting
        energy_kwh = self._energy_for_exec_kwh(u, exec_ms)
        carbon_g  = self._carbon_for_energy_g(energy_kwh)
        # --- IT (compute) energy from exec time (kWh) ---
        it_energy_kwh = self._energy_for_exec_kwh(u, exec_ms)

        # ------------------------------------------------------------------
        # COP-AWARE COOLING, HEAT REJECTION & WATER
        # ------------------------------------------------------------------
        # Cooling electrical energy (kWh) needed to remove the IT heat:
        #   cooling_elec_kwh = it_energy_kwh / COP
        cop = self._cop_for_ms(start_ms)
        tou = self._time_for_ms(start_ms)
        cooling_elec_kwh = it_energy_kwh / max(cop, 0.1)  # safety floor

        # Total heat rejected to atmosphere (kWh thermal) ≈ all electric energy ends as heat:
        #   heat_rejected_kwh = it_energy_kwh + cooling_elec_kwh
        heat_rejected_kwh = it_energy_kwh + cooling_elec_kwh

        # Water modeling (per kWh of HEAT rejected):
        # - water_static: baseline m^3/kWh_heat (site fixed losses, drift, etc.)
        # - water_cycling_density: m^3/kWh_heat (evaporation in towers)
        static_m3 = heat_rejected_kwh * max(0.0, self.water_static)
        evap_m3 = heat_rejected_kwh * max(0.0, self.water_cycling_density)

        # Blowdown via cycles-of-concentration (CoC): blowdown = evap / (CoC - 1)
        if self.blowdown_ratio and self.blowdown_ratio > 1.0:
            blowdown_m3 = evap_m3 / (self.blowdown_ratio - 1.0)
        else:
            blowdown_m3 = 0.0

        # Total make-up water drawn from supply:
        makeup_m3 = static_m3 + evap_m3 + blowdown_m3

        # Water-related electricity (kWh):
        #  - potable/make-up water delivery & treatment
        #  - wastewater treatment for blowdown
        water_energy_kwh = (
                makeup_m3 * max(0.0, self.potable_energy_intensity) +
                blowdown_m3 * max(0.0, self.wastewater_energy_intensity)
        )
        ttft_s     = (float(net_latency_ms) + queue_ms + exec_ms) / 1000.0

        energy_costs = energy_kwh * tou

        # Aggregates
        self._ttft_sum_s += ttft_s
        self._ttft_count += 1
        self._energy_kwh += energy_kwh
        self._carbon_g  += carbon_g
        self._water_m3   += makeup_m3
        self.energy_cost_agg += energy_costs


        if self.debug:
            print(f"[DC {self.dc_id}] model={model} arr={arrival_ms:.0f}ms net={net_latency_ms:.0f}ms "
                  f"queue={queue_ms:.0f}ms exec={exec_ms:.0f}ms node={u.node_id} TTFT={ttft_s:.3f}s")

        return {
            "ttft_s": float(ttft_s),
            "start_ms": float(start_ms),
            "finish_ms": float(finish_ms),
            "queue_delay_ms": float(queue_ms),
            "exec_ms": float(exec_ms),
            "energy_kwh": float(energy_kwh),
            "carbon_g": float(carbon_g),
            "water_m3": float(makeup_m3),
            "it_energy_kwh": float(it_energy_kwh),
            "cooling_elec_kwh": float(cooling_elec_kwh),
            "water_energy_kwh": float(water_energy_kwh),
            "static_m3": float(static_m3),
            "evap_m3": float(evap_m3),
            "blowdown_m3": float(blowdown_m3),
            "node_id": int(u.node_id),
            "energy_costs": float(energy_costs),
        }

    # ---------- reporting ----------
    def report_global_stats(self) -> Dict[str, Any]:
        avg_ttft = (self._ttft_sum_s / self._ttft_count) if self._ttft_count > 0 else 0.0
        return {
            "avg_ttft": float(avg_ttft),
            "energy_cost": float(self.energy_cost_agg),
            "carbon_emissions": float(self._carbon_g),
            "water_usage": float(self._water_m3),
            "total_energy": float(self._energy_kwh),
        }

    def report_utilization(self) -> float:
        if not self.units or self.epoch_length_ms <= 0.0:
            return 0.0
        cap_ms = len(self.units) * self.epoch_length_ms
        u = self._busy_ms / cap_ms
        return max(0.0, min(1.0, float(u)))

    def reset_epoch(self) -> None:
        self._busy_ms = 0.0
        self._energy_kwh = 0.0
        self._carbon_g = 0.0
        self._water_m3 = 0.0
        self._ttft_sum_s = 0.0
        self._ttft_count = 0
        self.energy_cost_agg = 0.0
        for unit in self.units:
            unit.available_at_ms = 0.0



# -----------------------------
# GeoNetwork (global routing)
# -----------------------------
# -----------------------------

@dataclass
class _RingEdge:
    u: int
    v: int
    w_ms: float  # latency u->v for one hop along the ring


class Geo_Network:
    """
    Holds all Datacenters and the latency matrix, stores a ring topology,
    and accounts for network latency when routing workload between DCs.

    Topology: ring over DC ids [0..N-1].
    Path cost: sum of per-hop latencies along the ring (min of CW/CCW).

    Public API:
      - apply_schedule_plan(epoch_idx, workload_df, schedule_plan, power_plan) -> List[dict]
      - report_global_stats() -> dict
      - report_dc_utilization() -> dict[int, float]  (if DCs expose utilization)
    """

    def __init__(self, datacenters: Dict[int, "Datacenter"], latency_matrix: List[List[float]], debug: bool = True):
        self.debug = debug
        self.datacenters: Dict[int, "Datacenter"] = dict(sorted(datacenters.items()))
        self.lat = latency_matrix
        self.dc_ids: List[int] = list(self.datacenters.keys())
        self.num_dc = len(self.dc_ids)

        # Build ring topology and precompute per-direction edge weights
        self.ring_edges_cw: List[_RingEdge] = []
        if self.num_dc > 0:
            for i in range(self.num_dc):
                u = self.dc_ids[i]
                v = self.dc_ids[(i + 1) % self.num_dc]
                # Edge weight uses latency_matrix[u][v] directly (ms)
                w = float(self.lat[u][v])
                self.ring_edges_cw.append(_RingEdge(u=u, v=v, w_ms=w))

        if self.debug:
            print("=== Geo_Network ===")
            print(f"Datacenters : {self.num_dc} -> {self.dc_ids}")
            if self.num_dc:
                sample = ", ".join(f"{e.u}->{e.v}:{e.w_ms:.1f}ms" for e in self.ring_edges_cw[:min(6, len(self.ring_edges_cw))])
                print(f"Ring edges  : {sample}{' ...' if len(self.ring_edges_cw)>6 else ''}")
            print()

        # scratch containers (reset per epoch)
        self._last_epoch_results: List[Dict[str, Any]] = []
        self._last_epoch_metrics: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Core routing helper: ring-path latency between src and dst
    # ------------------------------------------------------------------
    def _ring_path_latency_ms(self, src_dc: int, dst_dc: int) -> float:
        if src_dc == dst_dc or self.num_dc <= 1:
            return 0.0

        # Map dc_id -> position in ring order
        pos = {dc_id: i for i, dc_id in enumerate(self.dc_ids)}
        i_src = pos[src_dc]; i_dst = pos[dst_dc]

        # Clockwise: sum edges from i_src -> i_dst moving +1 each step
        cw_ms = 0.0
        i = i_src
        while i != i_dst:
            u = self.dc_ids[i]
            j = (i + 1) % self.num_dc
            v = self.dc_ids[j]
            cw_ms += float(self.lat[u][v])
            i = j

        # Counter-clockwise: sum edges going -1 each step
        ccw_ms = 0.0
        i = i_src
        while i != i_dst:
            j = (i - 1 + self.num_dc) % self.num_dc
            u = self.dc_ids[j]
            v = self.dc_ids[i]
            ccw_ms += float(self.lat[u][v])
            i = j

        return min(cw_ms, ccw_ms)

    # ------------------------------------------------------------------
    # Optional hook: apply per-DC power plan if DC implements it
    # power_plan can be any structure; we try to pass per-DC slices
    # ------------------------------------------------------------------
    def _apply_power_plan(self, power_plan: Dict[str, Any] | None):
        if not power_plan:
            return
        for dc_id, dc in self.datacenters.items():
            plan_slice = power_plan.get(dc_id) if isinstance(power_plan, dict) else None
            if hasattr(dc, "apply_power_plan") and callable(getattr(dc, "apply_power_plan")):
                try:
                    dc.apply_power_plan(plan_slice)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Public: apply schedule + power plans to a workload for one epoch
    # Returns a list of per-request dicts (detailed results)
    # ------------------------------------------------------------------
    def apply_schedule_plan(
        self,
        epoch_idx: int,
        workload_df,                         # pd.DataFrame
        schedule_plan: Dict[str, Any],
        power_plan: Dict[str, Any] | None,
    ) -> List[Dict[str, Any]]:
        """
        Workload columns expected (rename below to your actual columns if different):
          - 'source_dc' : int
          - 'arrival_ms': int (or 'arrival' in ms)
          - 'model'     : str ('Llama7b' or 'Llama70b')
          - (optionally tokens, sizes, etc., which DCs may use)

        schedule_plan formats supported (examples):
          - {'default_target_dc': 0}  -> send all to DC 0
          - {'route': {'Llama7b': 1, 'Llama70b': 3}}  -> per-model target
          - {'map': {request_index: dc_id, ...}}  -> explicit mapping by row index
        """
        # apply power plan first (DC can pre-toggle capacity, etc.)
        self._apply_power_plan(power_plan)

        details: List[Dict[str, Any]] = []

        # Helpers to choose a target DC
        def choose_target_dc(row_idx: int, model: str, src_dc: int) -> int:
            # 1) explicit map by index
            mp = schedule_plan.get("map") if isinstance(schedule_plan, dict) else None
            if isinstance(mp, dict) and row_idx in mp:
                return int(mp[row_idx])

            # 2) per-model routing
            rt = schedule_plan.get("route") if isinstance(schedule_plan, dict) else None
            if isinstance(rt, dict) and model in rt:
                return int(rt[model])

            # 3) default DC
            if "default_target_dc" in schedule_plan:
                return int(schedule_plan["default_target_dc"])

            # 4) fallback: local
            return int(src_dc)

        # Main loop
        for row_idx, row in enumerate(workload_df.itertuples(index=False), start=0):
            # NOTE: adapt these attribute names if your dataframe columns differ
            # For example, if your columns are 'src_dc' and 'arrival', adjust accordingly.
            try:
                src_dc = int(getattr(row, "source_dc"))
            except AttributeError:
                src_dc = int(getattr(row, "src_dc"))

            model = str(getattr(row, "model"))
            arrival_ms = int(getattr(row, "arrival_ms", getattr(row, "arrival", 0)))

            tgt_dc = choose_target_dc(row_idx, model, src_dc)
            net_ms = self._ring_path_latency_ms(src_dc, tgt_dc)

            dc = self.datacenters.get(tgt_dc)
            result: Dict[str, Any] = {
                "epoch": int(epoch_idx),
                "request_idx": int(row_idx),
                "source_dc": int(src_dc),
                "target_dc": int(tgt_dc),
                "model": model,
                "arrival_ms": float(arrival_ms),
                "net_latency_ms": float(net_ms),
            }

            # Delegate to DC if it supports detailed scheduling
            if dc and hasattr(dc, "schedule_request") and callable(getattr(dc, "schedule_request")):
                try:
                    dc_ret = dc.schedule_request(
                        model=model,
                        arrival=arrival_ms,
                        net_latency_ms=net_ms,
                        source_dc=src_dc,
                        target_dc=tgt_dc,
                        # You may add tokens/size if present in the row, e.g.:
                        # prefill_tokens=getattr(row, "prefill_tokens", None),
                        # gen_tokens=getattr(row, "gen_tokens", None),
                    )
                    # Merge DC fields into our result (DC may include ttft_s, energy_kwh, carbon_g, water_m3, etc.)
                    if isinstance(dc_ret, dict):
                        result.update(dc_ret)
                except Exception:
                    # Keep minimal record if DC threw
                    pass
            else:
                # Minimal placeholder if DC lacks scheduling method
                # TTFT := net latency only (for debugging); energy/carbon/water remain 0
                result.update({
                    "ttft_s": float(net_ms) / 1000.0,
                    "energy_cost": 0.0,
                    "energy_kwh": 0.0,
                    "carbon_g": 0.0,
                    "water_m3": 0.0,
                })

            details.append(result)

        # Store last epoch snapshot for reporting
        self._last_epoch_results = details
        self._last_epoch_metrics = self._aggregate_epoch_metrics(details)
        return details

    # ------------------------------------------------------------------
    # Summarize epoch-level metrics from details
    # ------------------------------------------------------------------
    def _aggregate_epoch_metrics(self, details: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not details:
            return {
                "avg_ttft": 0.0,
                "energy_cost": 0.0,
                "carbon_emissions": 0.0,
                "water_usage": 0.0,
                "total_energy": 0.0,
            }

        # Average TTFT
        ttft_sum = 0.0
        ttft_cnt = 0
        energy_kwh = 0.0
        carbon_g = 0.0
        water_m3 = 0.0
        energy_cost = 0.0

        for r in details:
            # TTFT
            v = r.get("ttft_s")
            if v is None:
                v = r.get("TTFT") or r.get("time_to_first_token_s")
            try:
                ttft_sum += float(v)
                ttft_cnt += 1
            except Exception:
                pass

            # Energy
            try: energy_kwh += float(r.get("energy_kwh", 0.0))
            except Exception: pass

            # Carbon
            try: carbon_g += float(r.get("carbon_g", r.get("carbon_emissions", 0.0)))
            except Exception: pass

            # Water
            try: water_m3 += float(r.get("water_m3", r.get("water_usage", 0.0)))
            except Exception: pass

            try: energy_cost += float(r.get("energy_costs", r.get("energy_cost", 0.0)))
            except Exception: pass

        avg_ttft = (ttft_sum / max(1, ttft_cnt)) if ttft_cnt > 0 else 0.0

        # If you track energy_cost elsewhere (e.g., within DCs), you can fold it in here.
        # For now, compute cost from DCs if they expose report_global_stats().

        return {
            "avg_ttft": float(avg_ttft),
            "energy_cost": float(energy_cost),
            "carbon_emissions": float(carbon_g),
            "water_usage": float(water_m3),
            "total_energy": float(energy_kwh),
        }

    # ------------------------------------------------------------------
    # Public: epoch summary
    # ------------------------------------------------------------------
    def report_global_stats(self) -> Dict[str, Any]:
        # Prefer DC rollups if available; otherwise return our own aggregation
        total = dict(self._last_epoch_metrics)
        return total

    # ------------------------------------------------------------------
    # Public: per-DC utilization (0..1), if DCs expose it; else we return {}
    # LLM_Simulator will fall back to computing it from detailed results.
    # ------------------------------------------------------------------
    def report_dc_utilization(self) -> Dict[int, float]:
        util = {}
        for dc_id, dc in self.datacenters.items():
            if hasattr(dc, "report_utilization") and callable(getattr(dc, "report_utilization")):
                try:
                    u = float(dc.report_utilization())
                    # clamp to [0,1]
                    util[dc_id] = max(0.0, min(1.0, u))
                except Exception:
                    pass
        return util



# -----------------------------
# Public entry point + CSV builders
# -----------------------------

# Assumes the helper functions and loaders we added earlier exist in this module:
# - load_dc_specs(...)
# - _load_node_type_templates(...)
# - _gpu_table_by_key(...), _finalize_perf_from_gpu_row(...)
# - _parse_type_counts(...), _parse_24h(...)
# - build_world_from_csvs(...)  -> (dc_specs, node_recs, lat_mat, gpu_tables)
#
# And these classes already exist and keep their public behavior:
# - Datacenter, Node, Processor, Geo_Network
#   with methods used below:
#     Datacenter.add_node(Node)
#     Node.add_processor(Processor)
#     Geo_Network.__init__(datacenters: Dict[int, Datacenter], latency_matrix: List[List[float]])
#     Geo_Network.apply_schedule_plan(epoch_idx: int, workload_df: pd.DataFrame,
#                                     schedule_plan: Dict[str, Any], power_plan: Dict[str, Any]) -> List[Dict]
#     Geo_Network.report_global_stats() -> Dict[str, Any]

# expects these helpers are available (exact-header versions):
#   load_dc_specs_exact
#   load_node_type_templates_exact
#   build_gpu_tables_exact
#   load_latency_matrix_exact
#   load_epoch_length_exact
#   build_world_from_csvs_exact

class LLM_Simulator:
    """
    Outside interaction point for the simulator.

    Loads all CSV specs (exact headers), builds:
      - Datacenters (carbon/profile/water params),
      - Nodes (with GPU-based perf/power),
      - Processors,
      - Geo network (latency matrix),
    and exposes run_epoch(...) to apply schedule + power plans to a workload.

    run_epoch returns: (metrics, detailed_results, dc_usage)
      - metrics: dict incl. avg_ttft, energy_cost, carbon_emissions, water_usage, total_energy
      - detailed_results: list of per-request dicts
      - dc_usage: { dc_id: {"utilization": float 0..1} }
    """

    def __init__(
        self,
        spec_dir: str = "sim_specs",
        dc_specs_csv: Optional[str] = None,
        node_specs_csv: Optional[str] = None,
        latency_csv: Optional[str] = None,
        a100_csv: Optional[str] = None,
        h100_csv: Optional[str] = None,
        epoch_length: Optional[int] = None,
        debug: bool = True,          # default True to print verification
    ) -> None:
        self.debug = debug
        self.spec_dir = spec_dir

        # Resolve file paths (exact header CSVs)
        self.dc_specs_csv   = dc_specs_csv   or os.path.join(spec_dir, "Datacenter_specs.csv")
        self.node_specs_csv = node_specs_csv or os.path.join(spec_dir, "Node_Specs.csv")
        self.latency_csv    = latency_csv    or os.path.join(spec_dir, "Geo_Latencies.csv")
        self.a100_csv       = a100_csv       or os.path.join(spec_dir, "A100_GPU.csv")
        self.h100_csv       = h100_csv       or os.path.join(spec_dir, "H100_GPU.csv")

        # Epoch length (optional exact-header loader)
        if epoch_length is not None:
            self.epoch_length = int(epoch_length)
        else:
            gran_csv = os.path.join(spec_dir, "Workload_Granularity.csv")
            try:
                self.epoch_length = int(load_epoch_length_exact(gran_csv))
            except Exception:
                self.epoch_length = 900

        if self.debug:
            print("=== LLM_Simulator init ===")
            print(f"spec_dir               : {self.spec_dir}")
            print(f"Datacenter_specs.csv   : {self.dc_specs_csv}")
            print(f"Node_Specs.csv         : {self.node_specs_csv}")
            print(f"Geo_Latencies.csv      : {self.latency_csv}")
            print(f"A100_GPU.csv           : {self.a100_csv}")
            print(f"H100_GPU.csv           : {self.h100_csv}")
            print(f"Epoch length (s)       : {self.epoch_length}")

        # === Load CSVs (exact headers) & Build world ===
        # If you prefer a single call, you can use build_world_from_csvs_exact(...)
        dc_specs   = load_dc_specs_exact(self.dc_specs_csv)
        templates  = load_node_type_templates_exact(self.node_specs_csv)
        gpu_tables = build_gpu_tables_exact(self.a100_csv, self.h100_csv)
        lat_mat    = load_latency_matrix_exact(self.latency_csv)

        if self.debug:
            # Datacenter overview
            print(f"\n[VERIFY] DC specs loaded: {len(dc_specs)} datacenters")
            for dc_id, p in list(dc_specs.items())[:5]:  # show a few
                print(f"  DC {dc_id}: CI={p['carbon_intensity_g_per_kwh']:.2f} g/kWh "
                      f"Nodes={p['Total_Nodes']} counts={p['node_type_counts_str'][:60]}{'...' if len(p['node_type_counts_str'])>60 else ''}")

            # Node templates overview
            print(f"[VERIFY] Node type templates: {len(templates)} types")
            for tid, t in sorted(templates.items())[:6]:
                print(f"  Type {tid}: accel={t['accel_type']} cfg={t['gpu_config']} procs={t['processor_count']} tdp_kw={t['tdp_kw']} idle_kw={t['idle_kw']}")


            # GPU tables overview
            a100_keys = list(gpu_tables['A100'].keys())
            h100_keys = list(gpu_tables['H100'].keys())
            print(f"[VERIFY] GPU table A100 keys: {a100_keys}")
            print(f"[VERIFY] GPU table H100 keys: {h100_keys}")

            # Latency matrix shape
            n = len(lat_mat)
            print(f"[VERIFY] Latency matrix size: {n}x{n}")



        # Expand nodes & perf/power using the exact-header world builder
        dc_specs, node_recs, lat_mat, gpu_tables = build_world_from_csvs_exact(
            self.dc_specs_csv, self.node_specs_csv, self.latency_csv, self.a100_csv, self.h100_csv
        )

        if self.debug:
            # node and perf sanity
            print(f"\n[VERIFY] Expanded node records: {len(node_recs)}")
            if node_recs:
                s = node_recs[0]
                mp7 = s["model_perf"]["Llama7b"]
                mp70 = s["model_perf"]["Llama70b"]
                print("  example node:",
                      f"dc={s['dc_id']} type={s['type_id']} accel={s['accel_type']} cfg={s['gpu_config']} "
                      f"procs={s['processor_count']} tdp_kw={s['tdp_kw']:.3f} idle_kw={s['idle_kw']:.3f}")
                print("  perf 7b : ms/request=", mp7["ms_per_request"], "  ms/token=", mp7["ms_per_token"])
                print("  perf 70b: ms/request=", mp70["ms_per_request"], "  ms/token=", mp70["ms_per_token"])

        # ===== Build objects =====
        self.datacenters: Dict[int, Datacenter] = {}
        for dc_id, params in dc_specs.items():
            dc = Datacenter(
                dc_id=dc_id,
                carbon_intensity_g_per_kwh=params["carbon_intensity_g_per_kwh"],
                time_of_use_24h=params["time_of_use_24h"],
                cop_profile_24h=params["cop_profile_24h"],
                blowdown_ratio=params["blowdown_ratio"],
                water_cycling_density=params["water_cycling_density_m3_per_kwh_heat"],
                potable_energy_intensity=params["potable_EI_kWh_per_m3"],
                wastewater_energy_intensity=params["wastewater_EI_kWh_per_m3"],
                water_static=params["water_static_m3_per_kwh_heat"],
                # epoch length from simulator (optional, pass through if desired):
                epoch_length=self.epoch_length,
            )
            self.datacenters[dc_id] = dc

        total_nodes = 0
        for rec in node_recs:
            dc = self.datacenters[rec["dc_id"]]
            unit = ProcNode(
                node_id=rec["node_id"],
                type_id=rec.get("type_id"),
                accel_type=rec["accel_type"],
                gpu_config=rec["gpu_config"],
                tdp_kw=float(rec["tdp_kw"]),
                idle_kw=float(rec["idle_kw"]),
                model_perf=rec["model_perf"],  # {"Llama7b": {...}, "Llama70b": {...}}
            )
            dc.add_node(unit)
            total_nodes += 1

        if self.debug:
            print(f"\n[VERIFY] Built world (combined Node+Processor):")
            print(f"  Datacenters : {len(self.datacenters)}")
            print(f"  Exec units  : {total_nodes}")
            for dc_id, dc in sorted(self.datacenters.items()):
                print(f"    DC {dc_id}: units={len(dc.units)}")
        # Build the geo network
        self.network = Geo_Network(self.datacenters, lat_mat)

        if self.debug:
            print("[VERIFY] Geo_Network built and ready.\n")

            self._debug_dump_cooling_params()
            self._sanity_probe_water_1kwh()

    # ---------------------------------------------------------------------
    # Public entry-point: run ONE epoch with a schedule + power plan
    # Returns: (metrics, detailed_results, dc_usage) where dc_usage has utilization in [0,1]
    # ---------------------------------------------------------------------
    # Rate_Flow_Sim.py  (inside LLM_Simulator)

    def _debug_dump_cooling_params(self) -> None:
        """
        Prints one line per DC with the fields that drive water & cooling.
        Requires self.debug = True to print (but we’ll also print if values look zero).
        """
        print("\n=== Cooling/Water params by DC ===")
        hdr = ("DC  CI(g/kWh)  water_static(m3/kWh_heat)  water_cycling_density(m3/kWh_heat)  "
               "blowdown_ratio  potable_EI(kWh/m3)  wastewater_EI(kWh/m3)  COP_profile[0..3]/default")
        print(hdr)
        zeroish = False
        for dc_id in sorted(self.datacenters):
            dc = self.datacenters[dc_id]
            prof = getattr(dc, "cop_profile_24h", None)
            prof_head = None
            if prof and len(prof) == 24:
                prof_head = ",".join(f"{float(x):.2f}" for x in prof[:4])
            cop_def = getattr(dc, "cop_default", 3.0)
            line = (f"{dc_id:2d}  {dc.carbon_intensity_g_per_kwh:8.1f}  "
                    f"{dc.water_static:10.6f}  {dc.water_cycling_density:10.6f}  "
                    f"{dc.blowdown_ratio:6.2f}        {dc.potable_energy_intensity:6.4f}            "
                    f"{dc.wastewater_energy_intensity:6.4f}        "
                    f"{(prof_head if prof_head else '—') or '—'} / {cop_def:.2f}")
            print(line)

            if (dc.water_static == 0.0 and dc.water_cycling_density == 0.0 and
                    dc.potable_energy_intensity == 0.0 and dc.wastewater_energy_intensity == 0.0):
                zeroish = True

        if zeroish:
            print("[DIAG] Many DC water parameters are zero. If this is unexpected, "
                  "verify Datacenter_specs.csv headers & units match the loader mapping.")

    def _sanity_probe_water_1kwh(self) -> None:
        """
        Quick synthetic probe: emulate 1 kWh IT at t=0 in each DC to see water result math.
        Prints the expected makeup water if schedule_request math is used.
        """
        print("\n=== Sanity probe: 1.0 kWh IT → water (by DC) ===")
        for dc_id in sorted(self.datacenters):
            dc = self.datacenters[dc_id]
            cop = dc._cop_for_ms(0.0)
            cooling_elec = 1.0 / max(cop, 0.1)
            heat_rej = 1.0 + cooling_elec  # kWh_heat
            static_m3 = heat_rej * max(0.0, dc.water_static)
            evap_m3 = heat_rej * max(0.0, dc.water_cycling_density)
            if dc.blowdown_ratio and dc.blowdown_ratio > 1.0:
                blowdown_m3 = evap_m3 / (dc.blowdown_ratio - 1.0)
            else:
                blowdown_m3 = 0.0
            makeup_m3 = static_m3 + evap_m3 + blowdown_m3
            print(f"DC {dc_id:2d}: COP={cop:.2f}  heat_rej={heat_rej:.3f} kWh  "
                  f"static={static_m3:.6f} m³  evap={evap_m3:.6f} m³  blowdown={blowdown_m3:.6f} m³  "
                  f"makeup={makeup_m3:.6f} m³")

    def run_epoch(
        self,
        epoch_idx: int,
        workload_df: pd.DataFrame,
        schedule_plan: Dict[str, Any],
        power_plan: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[int, Dict[str, float]]]:
        # --- ensure deterministic runs (consecutive runs identical) ---
        for dc in self.datacenters.values():
            if hasattr(dc, "reset_epoch"):
                dc.reset_epoch()

        if self.debug:
            print(f"=== run_epoch: epoch={epoch_idx} | workload={len(workload_df)} rows ===")
        # Apply plans and get per-request results
        detailed_results: List[Dict[str, Any]] = self.network.apply_schedule_plan(
            epoch_idx=epoch_idx,
            workload_df=workload_df,
            schedule_plan=schedule_plan,
            power_plan=power_plan,
        )

        # Aggregate epoch-level metrics
        metrics: Dict[str, Any] = self.network.report_global_stats()
        metrics.setdefault("avg_ttft", 0.0)
        metrics.setdefault("energy_cost", 0.0)
        metrics.setdefault("carbon_emissions", 0.0)
        metrics.setdefault("water_usage", 0.0)
        metrics.setdefault("total_energy", 0.0)

        # Per-DC utilization (0..1)
        dc_usage = self._get_dc_utilization(detailed_results)
        metrics["by_datacenter"] = dc_usage

        if self.debug:
            used = ", ".join(f"dc{d}:{u['utilization']:.2f}" for d,u in sorted(dc_usage.items()))
            print(f"[EPOCH {epoch_idx}] "
                  f"TTFT={metrics['avg_ttft']:.6f}s  "
                  f"C={metrics['carbon_emissions']:.6f}  "
                  f"W={metrics['water_usage']:.6f}  "
                  f"E={metrics['total_energy']:.6f}  "
                  f"EC={metrics['energy_cost']:.6f}  "
                  f"| Util: {used}")
            print(f"[EPOCH {epoch_idx}] detailed results: {len(detailed_results)}\n")

        return metrics, detailed_results, dc_usage

    # ---------------------------------------------------------------------
    # Utilization helpers (same as before)
    # ---------------------------------------------------------------------
    def _get_dc_utilization(self, detailed_results: List[Dict[str, Any]]) -> Dict[int, Dict[str, float]]:
        if hasattr(self.network, "report_dc_utilization") and callable(getattr(self.network, "report_dc_utilization")):
            try:
                util = self.network.report_dc_utilization()
                return self._normalize_util_map(util)
            except Exception:
                pass

        per_dc = {}
        hook_found = False
        for dc_id, dc in self.datacenters.items():
            if hasattr(dc, "report_utilization") and callable(getattr(dc, "report_utilization")):
                try:
                    u = dc.report_utilization()
                    per_dc[dc_id] = u
                    hook_found = True
                except Exception:
                    continue
        if hook_found:
            return self._normalize_util_map(per_dc)

        return self._aggregate_dc_utilization(detailed_results)

    def _normalize_util_map(self, raw: Dict[Any, Any]) -> Dict[int, Dict[str, float]]:
        out: Dict[int, Dict[str, float]] = {}
        for k, v in (raw or {}).items():
            try:
                dc_id = int(k)
            except Exception:
                try:
                    dc_id = int(float(k))
                except Exception:
                    continue
            if isinstance(v, dict):
                u = v.get("utilization", v.get("util", v.get("usage", 0.0)))
            else:
                u = v
            try:
                u = float(u)
            except Exception:
                u = 0.0
            u = max(0.0, min(1.0, u))
            out[dc_id] = {"utilization": u}
        return out

    def _aggregate_dc_utilization(self, detailed_results: List[Dict[str, Any]]) -> Dict[int, Dict[str, float]]:
        from collections import defaultdict

        # --- 1) Capacity (units per DC) ---
        # Prefer exec_units (list), else processors/units fallbacks, else 0
        procs_per_dc: Dict[int, int] = {}
        for dc_id, dc in self.datacenters.items():
            units = 0
            # common names in your codebase
            for attr in ("exec_units", "units", "processors"):
                if hasattr(dc, attr) and getattr(dc, attr) is not None:
                    try:
                        units = len(getattr(dc, attr))  # if it's a list-like
                    except TypeError:
                        # if it's a scalar count
                        try:
                            units = int(getattr(dc, attr))
                        except Exception:
                            pass
                    if units:
                        break
            # last resort: known scalar fields you sometimes keep
            if not units:
                for attr in ("processor_count", "unit_count", "num_executors"):
                    if hasattr(dc, attr) and getattr(dc, attr) is not None:
                        try:
                            units = int(getattr(dc, attr))
                            break
                        except Exception:
                            pass
            procs_per_dc[int(dc_id)] = max(0, int(units))

        epoch_ms = float(self.epoch_length) * 1000.0
        cap_ms: Dict[int, float] = {dc: float(procs) * epoch_ms for dc, procs in procs_per_dc.items()}

        # --- 2) Busy time (sum of exec ms per DC) ---
        busy_ms: Dict[int, float] = defaultdict(float)

        def _extract_dc_id(rec: Dict[str, Any]):
            for k in ("dc_id", "target_dc", "assigned_dc", "dc", "datacenter"):
                if k in rec and rec[k] is not None:
                    try:
                        return int(rec[k])
                    except Exception:
                        try:
                            return int(float(rec[k]))
                        except Exception:
                            pass
            return None

        def _extract_busy_ms(rec: Dict[str, Any]) -> float:
            # try explicit durations first
            for k in ("exec_ms", "proc_ms", "service_ms", "gpu_time_ms", "process_ms"):
                if k in rec and rec[k] is not None:
                    try:
                        return max(0.0, float(rec[k]))
                    except Exception:
                        continue
            # fallback: end - start
            s = rec.get("start_ms", rec.get("start_time_ms"))
            e = rec.get("end_ms", rec.get("end_time_ms"))
            if s is not None and e is not None:
                try:
                    return max(0.0, float(e) - float(s))
                except Exception:
                    pass
            return 0.0

        for rec in detailed_results:
            dc_id = _extract_dc_id(rec)
            if dc_id is None:
                continue
            busy_ms[dc_id] += _extract_busy_ms(rec)

        # --- 3) Utilization = busy / capacity ---
        util: Dict[int, Dict[str, float]] = {}
        for dc_id in procs_per_dc.keys():
            cap = cap_ms.get(dc_id, 0.0)
            if cap <= 0.0:
                u = 0.0
            else:
                u = busy_ms.get(dc_id, 0.0) / cap
            util[dc_id] = {"utilization": max(0.0, min(1.0, u))}

        # ensure any DC that showed busy time but didn't appear in capacity map is present
        for dc_id in busy_ms.keys():
            util.setdefault(dc_id, {"utilization": 0.0})

        return util

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    def _load_epoch_length_default(self) -> int:
        """
        Load Epoch_Length from Workload_Granularity.csv if present; else default 900.
        """
        path = os.path.join(self.spec_dir, "Workload_Granularity.csv")
        try:
            df = pd.read_csv(path)
            val = int(df.iloc[0].get("Epoch_Length", 900))
            return val if val > 0 else 900
        except Exception:
            return 900




# =========================
# EXACT-HEADER CSV HELPERS
# =========================
# Assumes your CSVs have the following exact columns:
# Datacenter_specs.csv:
#   ['DC_Num','DC_Name','Location','Carbon_Intensity','Water_Static','Total_Nodes',
#    'Node_Types','Time_of_Use(24_Hours)','COP_Profile(24_Hours)',
#    'Water_Cycling_Density','Solids_Ratio','Potable_Energy_Intensity','Wastewater_Energy_Intensity']
#
# A100_GPU.csv / H100_GPU.csv:
#   ['num_GPUs','Mem_Size','Llama7b_Process','Llama70b_Process',
#    'prefill_token_size','gen_token_size','batch_size','TDP']
#
# Geo_Latencies.csv:
#   ['Datacenter_Dest','0','1','2','3','4','5','6','7','8','9','10','11']
#
# Workload_Granularity.csv: at least 'Epoch_Length'
#
# Node_Specs.csv:
#   ['Node_Num','Node_Type','Inter_GPU_Bandwidth','Load_Bandwidth_PCIE','Load_Delay_NVLinkNum_GPUs']
#
# NOTE: These helpers do no defensive checking by request—headers & data must match.

from typing import Dict, Any, List, Tuple
import pandas as pd
from collections import Counter


# -------- small utilities --------

def _parse_24h_exact(series_str: str) -> List[float]:
    # Accept ';' or ',' separated 24 numbers (no guards)
    parts = [p.strip() for p in str(series_str).replace(",", ";").split(";")]
    return [float(x) for x in parts]

def _parse_gpu_from_node_type_exact(s: str) -> Tuple[str, str]:
    """ '8_A100s' -> ('A100','8_A100'), '4_H100' -> ('H100','4_H100') """
    t = s.strip().upper().replace("-", "_")
    t = t[:-1] if t.endswith("S") else t  # drop trailing 'S' in '8_A100S'
    # Split like '8_A100'
    n, g = t.split("_", 1)
    return (g, f"{n}_{g}")  # accel_type, gpu_config

def _counts_from_nodetypes_exact(nodetypes_str: str) -> Dict[int, int]:
    """ '0;0;1;5;5' -> {0:2,1:1,5:2} """
    seq = [int(x.strip()) for x in nodetypes_str.split(";") if x.strip()]
    return dict(sorted(Counter(seq).items()))

def _parse_24h_semicolon(val):
    """
    Parse a semicolon-separated string of 24 floats (e.g., '1.0;1.1;...;0.9').
    Returns a list[float] of length 24 or None if missing/invalid.
    """
    if val is None:
        return None
    # Pandas may already give a float(NaN) for empty cells
    try:
        import math
        if isinstance(val, float) and math.isnan(val):
            return None
    except Exception:
        pass

    if isinstance(val, (list, tuple)) and len(val) == 24:
        try:
            return [float(v) for v in val]
        except Exception:
            return None

    if not isinstance(val, str):
        # Might be a single number or unexpected type
        s = str(val)
    else:
        s = val

    tokens = [t.strip() for t in s.split(';') if t.strip() != ""]
    if len(tokens) != 24:
        return None
    try:
        return [float(t) for t in tokens]
    except Exception:
        return None


def _parse_node_type_counts_from_row(row, expected_types=6):
    """
    Accepts several header names and formats:
      - 'Node_Type_Counts' or 'node_type_counts' or 'NodeTypeCounts' etc.
      - Value can be:
          * semicolon or comma separated 'k:v' pairs, e.g. "0:167;1:167;...;5:166"
          * OR a flat list of N ints in order of type id 0..N-1, e.g. "167;167;167;167;166;166"
    Returns (counts_dict, counts_str).
    """
    # Try multiple header spellings
    candidates = [
        "Node_Type_Counts", "node_type_counts", "NodeTypeCounts",
        "NodeTypeCounts(0-5)", "NodeTypeCounts_0to5", "NodeTypeCounts_0_5"
    ]
    raw = None
    for k in candidates:
        if k in row and row[k] is not None and str(row[k]).strip() != "":
            raw = str(row[k]).strip()
            break
    if not raw:
        return {}, ""

    # Split on ';' first (common), fallback to ','.
    parts = [p.strip() for p in raw.replace(",", ";").split(";") if p.strip() != ""]
    counts = {}

    # Case A: "k:v" style
    if ":" in parts[0]:
        for item in parts:
            if ":" not in item:
                continue
            k, v = item.split(":", 1)
            try:
                k_i = int(k.strip())
                v_i = int(float(v.strip()))
                counts[k_i] = v_i
            except Exception:
                # ignore bad tokens
                pass
    else:
        # Case B: flat list like "a;b;c;d;e;f"
        ints = []
        for item in parts:
            try:
                ints.append(int(float(item)))
            except Exception:
                # ignore bad tokens
                pass
        if ints:
            # Map sequentially to type ids [0..len-1]
            counts = {i: val for i, val in enumerate(ints)}

    # Keep only non-negative ints; trim/sort by key for a stable string
    counts = {int(k): int(v) for k, v in counts.items() if int(v) >= 0}
    counts_str = ",".join(f"{k}:{counts[k]}" for k in sorted(counts.keys()))
    return counts, counts_str


def load_dc_specs_exact(self) -> dict[int, dict]:
    """
    Load Datacenter_specs.csv using the CURRENT headers, including 24-hour
    COP and TOU profiles and water/cooling fields. Returns:
      { dc_id: {
          'dc_id', 'carbon_intensity_g_per_kwh',
          'water_static_m3_per_kwh_heat', 'water_cycling_density_m3_per_kwh_heat',
          'blowdown_ratio', 'potable_EI_kWh_per_m3', 'wastewater_EI_kWh_per_m3',
          'tou_24h', 'cop_24h', 'node_type_counts'
        }, ... }
    """
    import pandas as pd
    import os

    path = "sim_specs/Datacenter_specs.csv"
    df = pd.read_csv(path)

    # Expected column set (per the CSV you sent)
    COL_DC      = "DC_Num"
    COL_CI      = "Carbon_Intensity"
    COL_WS      = "Water_Static"
    COL_WCD     = "Water_Cycling_Density"
    COL_SR      = "Solids_Ratio"
    COL_PEI     = "Potable_Energy_Intensity"
    COL_WWEI    = "Wastewater_Energy_Intensity"
    COL_TOU24   = "Time_of_Use(24_Hours)"
    COL_COP24   = "COP_Profile(24_Hours)"
    COL_TYPES   = "Node_Types"
    COL_TOTAL   = "Total_Nodes"

    missing = [c for c in [COL_DC, COL_CI, COL_WS, COL_WCD, COL_SR, COL_PEI, COL_WWEI, COL_TOU24, COL_COP24, COL_TYPES]
               if c not in df.columns]
    if missing:
        raise ValueError(f"[DC Specs] Missing required columns: {missing}\nFound: {list(df.columns)}")

    out: dict[int, dict] = {}
    for _, row in df.iterrows():
        did = int(_safe_float(row[COL_DC]))
        if did < 0:
            continue

        carbon = _safe_float(row[COL_CI])  # g CO2 per kWh

        # Water/cooling fields
        water_static = _safe_float(row[COL_WS])                     # m^3 per kWh_heat (constant adder)
        water_cycle  = _safe_float(row[COL_WCD])                    # m^3 per kWh_heat evaporated
        blowdown     = _safe_float(row[COL_SR])                    # ratio (e.g., 0.30)
        potable_ei   = _safe_float(row[COL_PEI])                    # kWh per m^3
        waste_ei     = _safe_float(row[COL_WWEI])                   # kWh per m^3

        # 24-hour arrays
        tou_24  = _parse_24h_semicolon(row[COL_TOU24])
        cop_24  = _parse_24h_semicolon(row[COL_COP24])

        # Node types list → counts
        type_counts, node_type_counts_str = _parse_node_type_counts_from_row(row[COL_TYPES])

        total_nodes = _safe_float(row[COL_TOTAL])

        out[did] = {
            "dc_id": did,
            "carbon_intensity_g_per_kwh": carbon,
            "water_static_m3_per_kwh_heat": water_static,
            "water_cycling_density_m3_per_kwh_heat": water_cycle,
            "blowdown_ratio": blowdown,
            "potable_EI_kWh_per_m3": potable_ei,
            "wastewater_EI_kWh_per_m3": waste_ei,
            "time_of_use_24h": tou_24,
            "cop_profile_24h": cop_24,
            "node_type_counts": type_counts,
            "node_type_counts_str": node_type_counts_str,
            "Total_Nodes": total_nodes,
        }

        # Guardrails + helpful logs if values look suspicious
        if (water_static == 0.0 and water_cycle == 0.0) or (cop_24 is None):
            self._logger.warning(
                "[DC %d] Cooling/water fields may be missing: "
                "water_static=%.6f, water_cycle=%.6f, blowdown=%.2f, potable_EI=%.4f, wastewater_EI=%.4f, cop_24h=%s",
                did, water_static, water_cycle, blowdown, potable_ei, waste_ei,
                "None" if cop_24 is None else "OK"
            )

    return out


# -------- Node type templates (from Node_Specs) --------

def load_node_type_templates_exact(node_specs_csv: str) -> Dict[int, Dict[str, Any]]:
    """
    Returns: dict[Type_ID] -> template dict with:
      accel_type, gpu_config, processor_count, tdp_kw, idle_kw
    (This CSV doesn’t provide power; we leave tdp/idle at 0.0 so GPU tables can fill them.)
    """
    df = pd.read_csv(node_specs_csv)
    templates: Dict[int, Dict[str, Any]] = {}

    for _, row in df.iterrows():
        tid = int(row["Node_Num"])  # use Node_Num as the Type_ID
        accel_type, gpu_config = _parse_gpu_from_node_type_exact(row["Node_Type"])
        templates[tid] = {
            "accel_type": accel_type,   # "A100" or "H100"
            "gpu_config": gpu_config,   # e.g., "8_A100"
            "processor_count": 1,       # default = 1 processor per node
            "tdp_kw": 0.0,              # filled later from GPU tables
            "idle_kw": 0.0,             # filled later
        }

    return templates


# -------- GPU perf/power tables --------

def load_gpu_table_exact(gpu_csv_path: str, chip: str) -> Dict[str, Dict[str, Any]]:
    """
    Build a lookup keyed by '<num>_<chip>' (e.g., '8_A100', '4_H100') → full row dict
    with an added '_meta': {'tdp_kw': TDP/1000}.
    """
    t = pd.read_csv(gpu_csv_path)
    table: Dict[str, Dict[str, Any]] = {}
    for _, row in t.iterrows():
        num = int(row["num_GPUs"])
        key = f"{num}_{chip}"
        d = row.to_dict()
        d["_meta"] = {"tdp_kw": float(row["TDP"]) / 1000.0}
        table[key] = d
    return table

def build_gpu_tables_exact(a100_csv: str, h100_csv: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
    return {
        "A100": load_gpu_table_exact(a100_csv, "A100"),
        "H100": load_gpu_table_exact(h100_csv, "H100"),
    }


# -------- Latency matrix --------

def load_latency_matrix_exact(lat_csv: str) -> List[List[float]]:
    """
    Reads Geo_Latencies.csv and returns an NxN float matrix (drops the 'Datacenter_Dest' label column).
    """
    df = pd.read_csv(lat_csv)
    df_num = df.drop(columns=["Datacenter_Dest"])
    return df_num.astype(float).values.tolist()


# -------- Epoch length (optional helper) --------

def build_world_from_csvs_exact(
    dc_specs_csv: str,
    node_specs_csv: str,
    latency_csv: str,
    a100_csv: str,
    h100_csv: str,
) -> tuple[Dict[int, Dict[str, Any]], List[Dict[str, Any]], List[List[float]], Dict[str, Dict[str, Dict[str, Any]]]]:
    """
    Returns:
      dc_specs : dict[dc_id] -> per-DC parameters (already normalized)
      nodes    : list of node records, each with model_perf and power fields
      lat_mat  : NxN list of floats (ms)
      gpu_tbls : {'A100': {...}, 'H100': {...}}
    """
    # --- small local helpers (kept here to avoid changing other files) ---
    def _parse_counts_str_to_dict(s: str) -> Dict[int, int]:
        """
        Accept '0:167;1:167;2:166;3:166;4:167;5:167' OR same with commas.
        Returns {0:167, 1:167, ...}. Ignores malformed fragments.
        """
        if not s:
            return {}
        parts = [p.strip() for p in s.replace(",", ";").split(";") if p.strip()]
        out: Dict[int, int] = {}
        for item in parts:
            if ":" not in item:
                continue
            k, v = item.split(":", 1)
            try:
                ki = int(k.strip())
                vi = int(float(v.strip()))
                if ki >= 0 and vi >= 0:
                    out[ki] = vi
            except Exception:
                # ignore malformed piece
                pass
        return out

    def _even_split_counts(total_nodes: int, ntypes: int) -> Dict[int, int]:
        total = max(0, int(total_nodes or 0))
        base, rem = divmod(total, ntypes)
        d = {i: base for i in range(ntypes)}
        for i in range(rem):
            d[i] += 1
        return d

    def _num(x):
        """Convert numeric strings to float; treat None/'None'/'nan' as None."""
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip().lower()
        if s in ("none", "nan", ""):
            return None
        try:
            return float(s)
        except Exception:
            return None

    # --- load the four CSV-driven artifacts with your existing helpers ---
    dc_specs = load_dc_specs_exact(dc_specs_csv)
    templates = load_node_type_templates_exact(node_specs_csv)
    gpu_tbls = build_gpu_tables_exact(a100_csv, h100_csv)
    lat_mat  = load_latency_matrix_exact(latency_csv)

    # Determine number of node types from templates (expected 6: ids 0..5)
    ntypes = len(templates)

    # --- Expand nodes from counts and attach perf/power ---
    nodes: List[Dict[str, Any]] = []

    for dc_id, params in dc_specs.items():
        # Preferred field from the updated DC loader:
        counts_str = params.get("node_type_counts_str", "")

        # Backcompat: if empty, try any legacy aliases the CSV may have used
        if not counts_str:
            counts_str = params.get("Node_Type_Counts", "") or params.get("node_types", "")

        counts = _parse_counts_str_to_dict(counts_str)

        # If still empty, fall back to even-splitting Total_Nodes
        if not counts:
            total_nodes = params.get("Total_Nodes", params.get("total_nodes", 0))
            try:
                total_nodes = int(total_nodes)
            except Exception:
                total_nodes = 0
            if total_nodes > 0:
                counts = _even_split_counts(total_nodes, ntypes)

        # No nodes for this DC if counts still empty
        if not counts:
            continue

        # Build per-type
        next_local_node_id = 0
        for type_id in sorted(counts):
            num = int(counts.get(type_id, 0))
            if num <= 0:
                continue

            tmpl = templates[type_id]
            accel = tmpl["accel_type"]         # "A100" or "H100"
            cfg   = tmpl["gpu_config"]         # e.g., "8_A100"
            procs = int(tmpl.get("processor_count", 1))

            # Power metadata from GPU tables
            row  = gpu_tbls[accel][cfg]
            meta = row.get("_meta", {})
            tdp_kw  = float(meta.get("tdp_kw", 0.0))
            idle_kw = float(meta.get("idle_kw", 0.1 * tdp_kw))

            # Perf per model (handle missing/None cells gracefully)
            pre_sz = _num(row.get("prefill_token_size"))
            gen_sz = _num(row.get("gen_token_size"))
            denom_tokens = (pre_sz or 0.0) + (gen_sz or 0.0)
            if denom_tokens <= 0:
                denom_tokens = 1.0  # avoid div-by-zero; not used if ms_per_request is None

            ms7  = _num(row.get("Llama7b_Process"))
            ms70 = _num(row.get("Llama70b_Process"))

            perf_7b = {
                "ms_per_request": ms7,
                "ms_per_token":   (ms7 / denom_tokens) if ms7 is not None else None,
            }
            perf_70b = {
                "ms_per_request": ms70,
                "ms_per_token":   (ms70 / denom_tokens) if ms70 is not None else None,
            }

            for _ in range(num):
                nodes.append({
                    "dc_id": int(dc_id),
                    "node_id": next_local_node_id,
                    "type_id": int(type_id),
                    "accel_type": accel,
                    "gpu_config": cfg,
                    "processor_count": procs,
                    "tdp_kw": tdp_kw,
                    "idle_kw": idle_kw,
                    "model_perf": {
                        "Llama7b":  perf_7b,
                        "Llama70b": perf_70b,
                    },
                })
                next_local_node_id += 1

    return dc_specs, nodes, lat_mat, gpu_tbls




