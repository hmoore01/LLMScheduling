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
        print(*args)

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
class Processor:
    proc_id: int
    node_id: int
    epoch_length: int
    power_state: str = "on"  # "on", "idle", or "off"
    tdp_kw: float = 0.0       # active power (kW)
    idle_kw: float = 0.0      # idle power (kW)
    # Model performance table per processor configuration.
    # Example entry:
    #   model_perf["Llama7b"] = {
    #       "ms_per_token": 1.2
    #       # OR
    #       "ms_per_request": 1187.0, "avg_tokens_per_request": 512
    #   }
    model_perf: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Optional: per-processor coefficient for TTFT surrogate (seconds)
    ttft_alpha_sec: Optional[float] = None

    # ------------- power helpers -------------
    def is_active(self) -> bool:
        return self.power_state.lower() == "on"

    def is_off(self) -> bool:
        return self.power_state.lower() == "off"

    def active_kW(self) -> float:
        return float(self.tdp_kw)

    def idle_kW(self) -> float:
        return float(self.idle_kw)

    # ------------- throughput -------------
    def estimate_tokens_per_sec(self, model_type: str) -> float:
        key = _norm_model_name(model_type)
        cfg = (self.model_perf or {}).get(key) or (self.model_perf or {}).get(str(model_type))
        if not cfg:
            _dprint(
                f"[CAP] No perf entry for model='{model_type}' (norm='{key}') on proc {self.node_id}:{self.proc_id}")
            return 0.0

        mspt = cfg.get("ms_per_token")
        if mspt is not None and mspt > 0:
            return 1000.0 / float(mspt)

        mspre = cfg.get("ms_per_request")
        avgtok = cfg.get("avg_tokens_per_request") or _DEFAULT_AVG_TOKENS.get(key)
        if mspre is not None and avgtok and avgtok > 0:
            sec = max(1e-6, float(mspre) / 1000.0)
            return float(avgtok) / sec

        exec_ms = cfg.get("exec_ms")
        default_tokens = cfg.get("default_tokens")
        if exec_ms is not None and default_tokens and default_tokens > 0:
            sec = max(1e-6, float(exec_ms) / 1000.0)
            return float(default_tokens) / sec

        _dprint(
            f"[CAP] Perf entry lacks usable fields for model='{model_type}' (norm='{key}') on proc {self.node_id}:{self.proc_id} -> 0 TPS")
        return 0.0

    def estimate_duty_cycle(self, dc_utilization: float) -> float:
        """Map DC utilization to processor duty. Simple passthrough/clip.
        If you don’t have per-proc routing, use DC-level utilization.
        """
        return max(0.0, min(1.0, float(dc_utilization)))


@dataclass
class Node:
    node_id: int
    type_id: int
    processors: List[Processor] = field(default_factory=list)


# -----------------------------
# Datacenter definition
# -----------------------------

@dataclass
class Datacenter:
    def __init__(
        self,
        dc_id: int,
        nodes: list,
        *,
        # Cooling & other hardware
        cop: float = 3.0,
        other_hw_overhead: float = 0.13,
        cooling_overhead_multiplier: float = 3.0,
        # Environmental intensities
        carbon_intensity_kg_per_kwh: float = 0.4,
        water_evap_m3_per_kwh: float = 0.0004,
        blowdown_ratio: float = 0.25,
        # Solar model
        solar_kw_capacity: float = 0.0,
        solar_profile_24h: list | None = None,
        # Battery model
        battery_kwh_capacity: float = 0.0,
        battery_max_kw_charge: float = 0.0,
        battery_max_kw_discharge: float = 0.0,
        battery_roundtrip_eff: float = 0.92,
        battery_embodied_kgco2e: float = 0.0,
        battery_cycle_life: float = 2500.0,
        # TTFT shaping
        ttft_alpha_sec: float = 0.20,
        ttft_beta_ms_per_token: float = 0.0,
    ):
        self.dc_id = int(dc_id)
        self.nodes = nodes or []

        # Cooling / other HW
        self.cop = float(cop)
        self.other_hw_overhead = float(other_hw_overhead)
        self.cooling_overhead_multiplier = float(cooling_overhead_multiplier)

        # Environmental
        self.carbon_intensity_kg_per_kwh = float(carbon_intensity_kg_per_kwh)
        self.water_evap_m3_per_kwh = float(water_evap_m3_per_kwh)
        self.blowdown_ratio = float(blowdown_ratio)

        # Solar
        self.solar_kw_capacity = float(solar_kw_capacity)
        self.solar_profile_24h = list(solar_profile_24h) if solar_profile_24h else None

        # Battery
        self.battery_kwh_capacity = float(battery_kwh_capacity)
        self.battery_max_kw_charge = float(battery_max_kw_charge)
        self.battery_max_kw_discharge = float(battery_max_kw_discharge)
        self.battery_roundtrip_eff = float(battery_roundtrip_eff)
        self.battery_embodied_kgco2e = float(battery_embodied_kgco2e)
        self.battery_cycle_life = float(battery_cycle_life)

        # Keep SoC both with and without underscore to satisfy old/new call-sites
        start_soc = 0.5 * self.battery_kwh_capacity if self.battery_kwh_capacity > 0 else 0.0
        self._battery_soc_kwh = start_soc
        self.battery_soc_kwh = start_soc   # legacy alias
        self._battery_cycles_used = 0.0

        # TTFT parameters
        self.ttft_alpha_sec = float(ttft_alpha_sec)
        self.ttft_beta_ms_per_token = float(ttft_beta_ms_per_token)

    def __repr__(self) -> str:
        return (f"Datacenter(dc_id={self.dc_id}, nodes={len(self.nodes)}, "
                f"cop={self.cop}, carbon_intensity={self.carbon_intensity_kg_per_kwh}, "
                f"solar_kw={self.solar_kw_capacity}, batt_kwh={self.battery_kwh_capacity}, "
                f"ttft_alpha={self.ttft_alpha_sec}, ttft_beta_mspt={self.ttft_beta_ms_per_token})")


    def apply_power_plan(self, plan: Dict):
        """Apply power states. Plan may specify per-processor or per-node states.
        Example plan formats:
            {"processors": { (node_id, proc_id): "on"/"idle"/"off" }}
            {"nodes": { node_id: "on"/"idle"/"off" }}
        """
        procs_set = plan.get("processors", {})
        nodes_set = plan.get("nodes", {})
        for node in self.nodes:
            node_state = nodes_set.get(node.node_id)
            for proc in node.processors:
                state = procs_set.get((node.node_id, proc.proc_id), node_state)
                if state:
                    proc.power_state = str(state).lower()

    # ---------- capacity aggregation ----------
    def tokens_per_sec_by_model(self, model_type: str) -> float:
        total = 0.0
        for node in self.nodes:
            for proc in node.processors:
                if not proc.is_active():
                    continue
                total += proc.estimate_tokens_per_sec(model_type)
        return total

    def process_rate_epoch(self, assigned_tokens: Dict[str, float], epoch_length: int,
                           avg_net_ms_by_model: Optional[Dict[str, float]] = None,
                           warmup_sec_by_model: Optional[Dict[str, float]] = None,
                           epoch_hour: Optional[int] = None) -> Dict:
        """Drain assigned tokens for the epoch and compute energy/carbon/water & DERs.
        Returns a metrics dict with per-model details and energy breakdown.
        """
        E = float(epoch_length)
        hours = E / 3600.0
        per_model = {}

        # 1) capacity & drain
        total_processed_tokens = 0.0
        total_capacity_tokens = 0.0
        for model, assigned in assigned_tokens.items():
            tps = max(0.0, self.tokens_per_sec_by_model(model))
            capacity = tps * E
            processed = min(float(assigned), capacity)
            leftover = max(0.0, float(assigned) - processed)
            util = 0.0 if capacity <= 0 else (processed / capacity)

            total_processed_tokens += processed
            total_capacity_tokens += max(0.0, capacity)

            per_model[model] = {
                "assigned_tokens": float(assigned),
                "processed_tokens": processed,
                "leftover_tokens": leftover,
                "capacity_tokens": capacity,
                "utilization": util,
                "avg_net_ms": (avg_net_ms_by_model or {}).get(model, 0.0),
                "avg_warmup_sec": (warmup_sec_by_model or {}).get(model, 0.0),
                "ttft_alpha_sec": self.ttft_alpha_sec,
            }

        # 2) DC-level utilization for energy splitting
        dc_util = 0.0 if total_capacity_tokens <= 1e-9 else (
            total_processed_tokens / total_capacity_tokens
        )

        # 3) IT+facility energy (demand before DER offsets)
        demand = self._it_and_facility_kwh(dc_util, epoch_length)

        # 4) Solar production this epoch
        solar_kwh = self._solar_generation_kwh(epoch_length, epoch_hour)

        # 5) Battery dispatch (greedy: self-consume solar, store surplus, then discharge to shave grid)
        batt = self._battery_dispatch_kwh(demand, solar_kwh, hours)
        # batt dict contains: {"solar_used_kwh","solar_charged_kwh","batt_discharge_kwh",
        #                      "batt_charge_from_solar_kwh","grid_import_kwh",
        #                      "embodied_carbon_kg","soc_kwh_after"}

        # 6) Water usage from total onsite energy demand (independent of source)
        water_m3 = self._water_from_energy(demand)

        # 7) Carbon: grid import * intensity + battery embodied amortization
        carbon = batt["grid_import_kwh"] * float(self.carbon_intensity_kg_per_kwh) + batt["embodied_carbon_kg"]

        return {
            "dc_id": self.dc_id,
            "energy_kwh": demand,
            "carbon_emissions": carbon,
            "water_usage": water_m3,
            "dc_utilization": dc_util,
            "per_model": per_model,
            # breakdown
            "energy_breakdown": {
                "demand_total_kwh": demand,
                "solar_used_kwh": batt["solar_used_kwh"],
                "solar_curtailed_kwh": max(0.0, solar_kwh - batt["solar_used_kwh"] - batt["batt_charge_from_solar_kwh"]),
                "battery_discharge_kwh": batt["batt_discharge_kwh"],
                "battery_charge_from_solar_kwh": batt["batt_charge_from_solar_kwh"],
                "grid_import_kwh": batt["grid_import_kwh"],
            },
        }

    # ---------- energy/carbon/water (demand before DERs) ----------
    def _it_and_facility_kwh(self, dc_utilization: float, epoch_length: int) -> float:
        hours = float(epoch_length) / 3600.0
        active_proc_kwh = 0.0
        idle_proc_kwh = 0.0
        for node in self.nodes:
            for proc in node.processors:
                if proc.is_off():
                    continue
                duty = proc.estimate_duty_cycle(dc_utilization if proc.is_active() else 0.0)
                active_proc_kwh += proc.active_kW() * duty * hours
                idle_frac = 0.0 if proc.is_off() else (1.0 - duty)
                idle_proc_kwh += proc.idle_kW() * idle_frac * hours

        processor_kwh = active_proc_kwh + idle_proc_kwh
        other_hw_kwh = self.other_hw_overhead * processor_kwh
        # Cooling power: processor energy through A/C with COP plus overhead (your prior rule)
        cooling_kwh = (processor_kwh / max(1e-6, self.cop)) * self.cooling_overhead_multiplier
        total_kwh = processor_kwh + other_hw_kwh + cooling_kwh
        return total_kwh

    # ---------- Solar model ----------
    def _solar_generation_kwh(self, epoch_length: int, epoch_hour: Optional[int]) -> float:
        if self.solar_kw_capacity <= 1e-6:
            return 0.0
        hours = float(epoch_length) / 3600.0
        if self.solar_profile_24h and epoch_hour is not None:
            cf = float(self.solar_profile_24h[int(epoch_hour) % 24])
        else:
            # simple bell-shaped default: 0 at night, peak at noon
            cf = 0.0
        return max(0.0, self.solar_kw_capacity * cf * hours)

    def _battery_dispatch_kwh(self, demand_kwh: float, solar_kwh: float, hours: float) -> dict:
        """
        Charge-then-discharge heuristic with complete accounting.

        Returns keys:
          grid_import_kwh, solar_used_kwh,
          battery_charge_kwh,            # legacy alias
          batt_charge_from_solar_kwh,    # preferred
          batt_charge_from_grid_kwh,     # 0.0 (we don't grid-charge here)
          battery_discharge_kwh,         # preferred
          batt_discharge_kwh,            # alias (caller compatibility)
          embodied_carbon_kg
        """
        if not hasattr(self, "_battery_soc_kwh"):
            self._battery_soc_kwh = float(getattr(self, "battery_soc_kwh", 0.0))
        if not hasattr(self, "battery_soc_kwh"):
            self.battery_soc_kwh = float(self._battery_soc_kwh)

        solar_used = min(max(solar_kwh, 0.0), max(demand_kwh, 0.0))
        demand_after_solar = max(0.0, demand_kwh - solar_used)
        excess_solar = max(0.0, solar_kwh - solar_used)

        charge_power_cap = max(0.0, float(self.battery_max_kw_charge)) * max(0.0, hours)
        can_charge_kwh = min(charge_power_cap,
                             max(0.0, float(self.battery_kwh_capacity) - float(self._battery_soc_kwh)))
        charge_kwh = min(excess_solar, can_charge_kwh)

        eff = float(getattr(self, "battery_roundtrip_eff", 1.0))
        ceff = eff ** 0.5 if eff > 0 else 0.0
        deff = ceff

        stored_kwh = charge_kwh * ceff
        self._battery_soc_kwh += stored_kwh
        self.battery_soc_kwh = self._battery_soc_kwh

        discharge_power_cap = max(0.0, float(self.battery_max_kw_discharge)) * max(0.0, hours)
        can_discharge_kwh_out = min(discharge_power_cap, demand_after_solar)
        soc_draw_kwh = min(self._battery_soc_kwh, (can_discharge_kwh_out / deff) if deff > 0 else 0.0)
        discharge_kwh = soc_draw_kwh * deff

        self._battery_soc_kwh -= soc_draw_kwh
        if self._battery_soc_kwh < 0:
            self._battery_soc_kwh = 0.0
        self.battery_soc_kwh = self._battery_soc_kwh

        grid_import = max(0.0, demand_after_solar - discharge_kwh)

        batt_cap = float(getattr(self, "battery_kwh_capacity", 0.0))
        batt_co2e = float(getattr(self, "battery_embodied_kgco2e", 0.0))
        batt_cycles = float(getattr(self, "battery_cycle_life", 0.0))
        if batt_cap > 0.0 and batt_co2e > 0.0 and batt_cycles > 0.0:
            per_kwh_delivered_kg = batt_co2e / (batt_cycles * batt_cap)
            embodied_carbon_kg = discharge_kwh * per_kwh_delivered_kg
        else:
            embodied_carbon_kg = 0.0

        return {
            "grid_import_kwh": grid_import,
            "solar_used_kwh": solar_used,
            "battery_charge_kwh": charge_kwh,  # alias
            "batt_charge_from_solar_kwh": charge_kwh,  # preferred
            "batt_charge_from_grid_kwh": 0.0,  # not used in this heuristic
            "battery_discharge_kwh": discharge_kwh,  # preferred
            "batt_discharge_kwh": discharge_kwh,  # alias for caller
            "embodied_carbon_kg": embodied_carbon_kg,
        }

    # ---------- Water model ----------
    def _water_from_energy(self, total_kwh: float) -> float:
        evap_m3 = total_kwh * float(self.water_evap_m3_per_kwh)
        blowdown_m3 = evap_m3 / max(1e-6, float(self.blowdown_ratio))
        total_water_m3 = evap_m3 + blowdown_m3
        return total_water_m3


# -----------------------------
# GeoNetwork (global routing)
# -----------------------------
# -----------------------------

class GeoNetwork:
    def __init__(self, datacenters: List[Datacenter],
                 dc_latency_ms: Union[Iterable[Iterable[float]], Callable[[int, int], float]]):
        self.datacenters: List[Datacenter] = datacenters
        self.dc_by_id: Dict[int, Datacenter] = {dc.dc_id: dc for dc in datacenters}
        self.dc_latency_ms = dc_latency_ms  # matrix or callable (src, tgt) -> ms

    # ---------------- TTFT helpers ----------------
    def _lat_ms(self, src: int, tgt: int) -> float:
        if callable(self.dc_latency_ms):
            return float(self.dc_latency_ms(src, tgt))
        return float(self.dc_latency_ms[src][tgt])

    # --------------- scheduling (rate) ---------------
    def apply_schedule_plan_rate(
        self,
        epoch_work_rows: Iterable[Tuple[int, str, float]],
        schedule_plan: Dict[Tuple[int, str], Dict[int, float]],
        power_plan: Dict[int, Dict],
        epoch_length: int,
        leftover_carry_in: Optional[Dict[Tuple[int, str, int], float]] = None,
    ) -> Tuple[Dict, Dict[int, Dict], Dict[Tuple[int, str, int], float]]:
        """Route aggregated epoch work and drain per-DC capacities.
        - epoch_work_rows: iterable of (src_dc, model_type, total_tokens)
        - schedule_plan: {(src_dc, model): {tgt_dc: frac_or_tokens}}
        - power_plan: {dc_id: {...}}  (forwarded to Datacenter.apply_power_plan)
        - leftover_carry_in: {(src, model, tgt): tokens}

        Returns (global_stats, per_dc_metrics, leftover_carry_out)
        """
        E = float(epoch_length)

        # 1) Start with base work rows
        work_rows: List[Tuple[int, str, float]] = []
        for src, model, toks in epoch_work_rows:
            work_rows.append((int(src), str(model), float(toks)))
        # Carry-in workloads are already routed to a specific tgt; we will add them post-routing

        # 2) Build assigned tokens per (tgt, model) and track avg net latency per tgt+model
        assigned: Dict[Tuple[int, str], float] = collections.defaultdict(float)
        # For TTFT surrogate, compute average network latency for tokens arriving to (tgt, model)
        net_ms_sum: Dict[Tuple[int, str], float] = collections.defaultdict(float)

        for (src, model, total_tokens) in work_rows:
            plan = schedule_plan.get((src, model), {})
            if not plan:
                # default: stay local
                tgt_tokens = {src: float(total_tokens)}
            else:
                # Determine if plan is absolute or fractional
                has_abs = any(v > 1.0 for v in plan.values())
                if has_abs:
                    tgt_tokens = {int(t): float(v) for t, v in plan.items()}
                    # If under-specified, keep remainder local
                    remainder = float(total_tokens) - sum(tgt_tokens.values())
                    if remainder > 1e-9:
                        tgt_tokens[src] = tgt_tokens.get(src, 0.0) + remainder
                else:
                    denom = sum(float(f) for f in plan.values())
                    if denom <= 1e-12:
                        tgt_tokens = {src: float(total_tokens)}
                    else:
                        tgt_tokens = {int(t): float(total_tokens) * float(f) / denom for t, f in plan.items()}

            for tgt, tok in tgt_tokens.items():
                assigned[(tgt, model)] += tok
                net_ms_sum[(tgt, model)] += tok * self._lat_ms(src, tgt)

        # 2b) Merge carry-in leftovers (already routed to a target)
        if leftover_carry_in:
            for (src, model, tgt), tok in leftover_carry_in.items():
                assigned[(int(tgt), str(model))] += float(tok)
                net_ms_sum[(int(tgt), str(model))] += float(tok) * self._lat_ms(int(src), int(tgt))

        # 3) Apply power plan
        for dc_id, plan in power_plan.items():
            dc = self.dc_by_id.get(int(dc_id))
            if dc:
                dc.apply_power_plan(plan)

        # 4) Partition per-DC assignments and compute per-model avg net latencies
        per_dc_assign: Dict[int, Dict[str, float]] = collections.defaultdict(lambda: collections.defaultdict(float))
        per_dc_avg_net_ms: Dict[int, Dict[str, float]] = collections.defaultdict(dict)
        for (tgt, model), tokens in assigned.items():
            per_dc_assign[int(tgt)][str(model)] += float(tokens)
        for (tgt, model), tok in assigned.items():
            avg_ms = 0.0 if tok <= 1e-12 else (net_ms_sum[(tgt, model)] / tok)
            per_dc_avg_net_ms[int(tgt)][str(model)] = avg_ms

        # 5) Drain at each DC
        per_dc_metrics: Dict[int, Dict] = {}
        total_energy = total_carbon = total_water = 0.0
        weighted_ttft_sum = 0.0
        total_processed_tokens = 0.0
        leftover_carry_out: Dict[Tuple[int, str, int], float] = {}

        for dc in self.datacenters:
            assigned_map = per_dc_assign.get(dc.dc_id, {})
            avg_ms_map = per_dc_avg_net_ms.get(dc.dc_id, {})
            result = dc.process_rate_epoch(assigned_map, epoch_length, avg_net_ms_by_model=avg_ms_map)
            per_dc_metrics[dc.dc_id] = result

            total_energy += result["energy_kwh"]
            total_carbon += result["carbon_emissions"]
            total_water += result["water_usage"]

            # Compute TTFT surrogate and leftovers
            for model, m in result["per_model"].items():
                processed = float(m["processed_tokens"])
                total_processed_tokens += processed
                rho = max(0.0, min(0.999, float(m["utilization"])) )
                alpha = float(m.get("ttft_alpha_sec") or (epoch_length / 2.0))
                avg_net_ms = float(m.get("avg_net_ms", 0.0))
                warm = float(m.get("avg_warmup_sec", 0.0))
                avg_ttft = (avg_net_ms / 1000.0) + warm + (alpha * rho / max(1e-6, (1.0 - rho)))
                weighted_ttft_sum += avg_ttft * processed

                leftover = float(m["leftover_tokens"])
                if leftover > 1e-6:
                    # Keep original src unknown here; record as (tgt->tgt) to preserve location
                    leftover_carry_out[(dc.dc_id, str(model), dc.dc_id)] = leftover

        avg_ttft_sec = (weighted_ttft_sum / total_processed_tokens) if total_processed_tokens > 0 else 0.0
        global_stats = {
            "epoch_length": epoch_length,
            "processed_tokens": total_processed_tokens,
            "avg_ttft_sec": avg_ttft_sec,
            "energy_kwh": total_energy,
            "carbon_emissions": total_carbon,
            "water_usage": total_water,
        }
        return global_stats, per_dc_metrics, leftover_carry_out


# -----------------------------
# Public entry point + CSV builders
# -----------------------------

def LLM_Simulator(
    epoch_idx: int,
    epoch_work_df,                 # pandas-like with columns: src_dc, model_type, total_tokens
    schedule_plan: Dict[Tuple[int, str], Dict[int, float]],
    power_plan: Dict[int, Dict],
    node_properties: Dict,
    epoch_length: int,
    dc_latency_ms,
    datacenters: List[Datacenter],
    mode: str = "rate",
    leftover_carry_in: Optional[Dict[Tuple[int, str, int], float]] = None,
    epoch_hour: Optional[int] = None,
):
    """Wrapper consistent with your previous simulator signature.
    - mode="rate": uses rate-flow path.
    - epoch_work_df: must support .itertuples() or .to_records(); we only read src_dc, model_type, total_tokens.
    - datacenters: constructed externally (using node_properties & CSVs) and passed in.
    - epoch_hour: optional (0..23) for solar profile lookup.
    """
    if mode != "rate":
        raise NotImplementedError("This module implements the rate-flow path only.")

    # Extract rows efficiently (supports pandas or any iterable of dict-like)
    rows: List[Tuple[int, str, float]] = []
    if hasattr(epoch_work_df, "itertuples"):
        for r in epoch_work_df.itertuples(index=False):
            src = getattr(r, "src_dc")
            model = getattr(r, "model_type")
            toks = getattr(r, "total_tokens")
            rows.append((int(src), str(model), float(toks)))
    elif hasattr(epoch_work_df, "to_dict"):
        for rec in epoch_work_df.to_dict("records"):
            rows.append((int(rec["src_dc"]), str(rec["model_type"]), float(rec["total_tokens"])) )
    else:
        # Assume already an iterable of (src, model, tokens)
        rows = [(int(a), str(b), float(c)) for (a, b, c) in epoch_work_df]

    net = GeoNetwork(datacenters, dc_latency_ms)
    stats, per_dc, leftovers = net.apply_schedule_plan_rate(
        rows, schedule_plan, power_plan, epoch_length, leftover_carry_in=leftover_carry_in
    )

    # Inject epoch_hour into per-DC energy (recompute solar/battery with epoch hour if provided)
    if epoch_hour is not None:
        # Re-run only the energy partitioning to attach solar/battery breakdowns with hour
        for dc_id, m in per_dc.items():
            dc = net.dc_by_id[dc_id]
            # Recompute with same utilization but proper hour
            dc_util = float(m.get("dc_utilization", 0.0))
            demand = dc._it_and_facility_kwh(dc_util, epoch_length)
            solar_kwh = dc._solar_generation_kwh(epoch_length, epoch_hour)
            batt = dc._battery_dispatch_kwh(demand, solar_kwh, float(epoch_length)/3600.0)
            water_m3 = dc._water_from_energy(demand)
            carbon = batt["grid_import_kwh"] * float(dc.carbon_intensity_kg_per_kwh) + batt["embodied_carbon_kg"]
            m["energy_kwh"] = demand
            m["carbon_emissions"] = carbon
            m["water_usage"] = water_m3
            m["energy_breakdown"] = {
                "demand_total_kwh": demand,
                "solar_used_kwh": batt["solar_used_kwh"],
                "solar_curtailed_kwh": max(0.0, solar_kwh - batt["solar_used_kwh"] - batt["batt_charge_from_solar_kwh"]),
                "battery_discharge_kwh": batt["batt_discharge_kwh"],
                "battery_charge_from_solar_kwh": batt["batt_charge_from_solar_kwh"],
                "grid_import_kwh": batt["grid_import_kwh"],
            }
        # Re-aggregate global stats
        stats["energy_kwh"] = sum(m["energy_kwh"] for m in per_dc.values())
        stats["carbon_emissions"] = sum(m["carbon_emissions"] for m in per_dc.values())
        stats["water_usage"] = sum(m["water_usage"] for m in per_dc.values())

    return stats, per_dc, leftovers


# ---------- CSV Builders ----------
# These helpers assume CSVs similar to your previous setup.
# - node_specs_csv: maps node_id -> dc_id, type_id, and processor count/tdp/idle
# - gpu_perf_csvs: dict of model perf tables per node type (e.g., A100/H100 variants)
# - dc_specs_csv: per-DC infra (COP, carbon intensity, water params, solar/battery)

import csv

def load_gpu_perf_tables(gpu_perf_csvs: Dict[str, str], debug: bool = False) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    """
    Returns:
      tables[accel_type][config_name][model_name] -> perf dict

    Supports:
      • WIDE:   columns like 'Llama7b_Process', 'Llama70b_Process' (treated as ms_per_request)
                plus optional 'prefill_token_size' + 'gen_token_size' to derive avg_tokens_per_request
      • NARROW: config, Model_Name, (ms_per_token OR (ms_per_request + avg_tokens_per_request)), optional Tokens/sec
    """
    import csv
    local_dprint = (lambda *a, **k: _dprint(*a, **k)) if (debug or RFS_DEBUG) else (lambda *a, **k: None)

    tables: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}

    for accel_type, path in gpu_perf_csvs.items():
        per_accel: Dict[str, Dict[str, Dict[str, float]]] = {}
        best_tps: Dict[str, float] = {}

        local_dprint(f"\n[GPU-LOAD] Accel={accel_type} File='{path}'")
        try:
            with open(path, "r", newline="") as f:
                reader = csv.DictReader(f)
                headers = [h for h in (reader.fieldnames or [])]
                local_dprint(f"[GPU-LOAD] Headers: {headers}")

                def _norm(h: str) -> str:
                    return h.lower().replace(" ", "").replace("_", "")

                norm_headers = {_norm(h): h for h in headers}

                # Optional config column
                cfg_key = None
                for kk in ("config", "gpuconfig", "gpu_cfg"):
                    if kk in norm_headers:
                        cfg_key = norm_headers[kk]
                        break
                if cfg_key:
                    local_dprint(f"[GPU-LOAD] Using config column: '{cfg_key}'")
                else:
                    local_dprint("[GPU-LOAD] No config column; defaulting to cfg='default'")

                # Detect WIDE columns: *_Process names
                wide_model_cols = []
                for h in headers:
                    hn = _norm(h)
                    if "llama7bprocess" in hn or "llama70bprocess" in hn:
                        wide_model_cols.append(h)
                if wide_model_cols:
                    local_dprint(f"[GPU-LOAD] Detected WIDE layout with model columns: {wide_model_cols}")
                else:
                    # also accept 'Llama7b'/'Llama70b' bare columns as ms_per_request
                    for h in headers:
                        hn = _norm(h)
                        if hn in ("llama7b", "llama70b"):
                            wide_model_cols.append(h)
                    if wide_model_cols:
                        local_dprint(f"[GPU-LOAD] Detected WIDE layout with model columns: {wide_model_cols}")
                    else:
                        local_dprint("[GPU-LOAD] Detected NARROW layout (model per row)")

                # Possible aux columns for avg tokens
                prefill_col = norm_headers.get("prefilltokensize")
                gen_col     = norm_headers.get("gentokensize")
                if prefill_col or gen_col:
                    local_dprint(f"[GPU-LOAD] Will derive avg_tokens_per_request from columns: prefill='{prefill_col}', gen='{gen_col}'")

                row_idx = 0
                for row in reader:
                    row_idx += 1
                    cfg = "default"
                    if cfg_key:
                        v = str(row.get(cfg_key, "")).strip()
                        if v:
                            cfg = v

                    if wide_model_cols:
                        # WIDE layout: numeric model cols are ms_per_request
                        # Avg tokens per request, if present
                        avgt = None
                        if prefill_col or gen_col:
                            p = _safe_float(row.get(prefill_col)) if prefill_col else None
                            g = _safe_float(row.get(gen_col)) if gen_col else None
                            if (p or 0) > 0 or (g or 0) > 0:
                                avgt = (p or 0.0) + (g or 0.0)

                        for mh in wide_model_cols:
                            val_raw = row.get(mh, "")
                            mspre = _safe_float(val_raw)
                            if mspre is None:
                                continue
                            model = _norm_model_name(mh)  # 'Llama7b' or 'Llama70b'
                            perf = _finalize_perf_entry(
                                ms_per_token=None,
                                ms_per_request=mspre,
                                avg_tokens_per_request=avgt,
                                model_key=model,
                            )
                            per_accel.setdefault(cfg, {})
                            existing = per_accel[cfg].setdefault(model, {})
                            if "ms_per_token" in perf or "ms_per_token" not in existing:
                                existing.update(perf)
                                local_dprint(f"[GPU-LOAD][{accel_type}] row#{row_idx} cfg='{cfg}' model='{model}' "
                                             f"mspre={mspre} avgt={avgt} -> stored")
                        continue

                    # NARROW layout
                    low = {k.lower(): v for k, v in row.items()}
                    def lk(*names, default=None):
                        for name in names:
                            if name in low and str(low[name]).strip() != "":
                                return low[name]
                        return default

                    raw_model = lk("model_name", "model", "modeltype", "name")
                    if not raw_model:
                        local_dprint(f"[GPU-LOAD][{accel_type}] row#{row_idx}: no model column; skipping")
                        continue
                    model = _norm_model_name(raw_model)

                    mspt   = lk("ms_per_token", "mspertoken")
                    mspre  = lk("ms_per_request", "msperrequest", "exec_ms", "execms", "ms")
                    avgtok = lk("avg_tokens_per_request", "avgtokensperrequest", "avg_tokens", "tokens_per_request")

                    perf = _finalize_perf_entry(mspt, mspre, avgtok, model_key=model)
                    if perf:
                        per_accel.setdefault(cfg, {})
                        existing = per_accel[cfg].setdefault(model, {})
                        if "ms_per_token" in perf or "ms_per_token" not in existing:
                            existing.update(perf)
                            local_dprint(f"[GPU-LOAD][{accel_type}] row#{row_idx} cfg='{cfg}' model='{model}' "
                                         f"mspt={perf.get('ms_per_token')} mspre={perf.get('ms_per_request')} "
                                         f"avg={perf.get('avg_tokens_per_request')} -> stored")
                    else:
                        local_dprint(f"[GPU-LOAD][{accel_type}] row#{row_idx} cfg='{cfg}' model='{model}' -> no usable perf")

                    # optional tokens/sec support
                    tps = lk("output_tokens_per_second", "tokens_per_second", "toks_per_sec")
                    if tps is not None:
                        val = _safe_float(tps)
                        if val:
                            best_tps[model] = max(best_tps.get(model, 0.0), val)
                            local_dprint(f"[GPU-LOAD][{accel_type}] row#{row_idx} model='{model}' tokens/sec={val}")

        except FileNotFoundError:
            local_dprint(f"[GPU-LOAD][{accel_type}] File not found: {path}; proceeding with empty table")
            per_accel = {}

        # finalize tokens/sec derivations
        if best_tps:
            local_dprint(f"[GPU-LOAD][{accel_type}] Finalizing {len(best_tps)} tokens/sec entries")
        for cfg, by_model in per_accel.items():
            for model, tps_val in list(best_tps.items()):
                if tps_val and tps_val > 0:
                    entry = by_model.setdefault(model, {})
                    if "ms_per_token" not in entry or not entry["ms_per_token"]:
                        entry["ms_per_token"] = 1000.0 / tps_val
                        local_dprint(f"[GPU-LOAD][{accel_type}] cfg='{cfg}' model='{model}' "
                                     f"derived ms_per_token from tps={tps_val:.3f} -> {entry['ms_per_token']:.6f} ms/token")

        # summary
        n_cfg = len(per_accel)
        n_models = sum(len(m) for m in per_accel.values())
        with_mspt = sum(
            1 for cfg in per_accel.values() for perf in cfg.values()
            if perf.get("ms_per_token", 0) > 0
        )
        local_dprint(f"[GPU-LOAD][{accel_type}] Summary: configs={n_cfg}, models={n_models}, with_ms_per_token={with_mspt}")

        tables[accel_type] = per_accel

    return tables



def build_datacenters_from_csv(
    node_specs_csv: str,
    gpu_perf_csvs: Dict[str, str],
    dc_specs_csv: str,
    epoch_length: int,
    debug: bool = False,
) -> List["Datacenter"]:
    import csv

    # ---- Local debug printer ----
    local_dprint = (lambda *a, **k: _dprint(*a, **k)) if (debug or RFS_DEBUG) else (lambda *a, **k: None)

    # ---- Solar defaults (uniform across DCs, regardless of CSV) ----
    # Use module-level constants if they exist; otherwise fall back to these.
    solar_kw_default = globals().get("DEFAULT_SOLAR_KW_CAPACITY", 1000.0)
    solar_prof_default = globals().get("DEFAULT_SOLAR_PROFILE_24H", [
        0.00, 0.00, 0.00, 0.00, 0.00,
        0.05, 0.20, 0.50, 0.75, 0.90,
        1.00, 0.95, 0.90, 0.80, 0.60,
        0.40, 0.20, 0.05,
        0.00, 0.00, 0.00, 0.00, 0.00, 0.00
    ])

    # 1) Load DC infra/specs (we’ll still read values, but enforce our solar defaults)
    dc_params: Dict[int, Dict] = {}
    try:
        with open(dc_specs_csv, "r", newline="") as f:
            reader = csv.DictReader(f)
            count = 0
            for row in reader:
                count += 1
                did = int(row.get("DC_Num") or row.get("dc_id") or row.get("dc") or 0)

                # Optional CSV solar profile; we will override to defaults below anyway.
                prof = str(row.get("solar_profile_24h", "")).strip()
                solar_prof_csv = [float(x) for x in prof.split(";") if x.strip()] if prof else None

                dc_params[did] = {
                    "cop": float(row.get("cop", 3.0)),
                    "other_hw_overhead": float(row.get("other_hw_overhead", 0.13)),
                    "cooling_overhead_multiplier": float(row.get("cooling_overhead_multiplier", 3.0)),
                    "carbon_intensity_kg_per_kwh": float(row.get("carbon_intensity_kg_per_kwh", 0.4)),
                    "water_evap_m3_per_kwh": float(row.get("water_evap_m3_per_kwh", 0.0004)),
                    "blowdown_ratio": float(row.get("blowdown_ratio", 0.25)),
                    # read but ignore for consistency; we’ll use uniform defaults for all DCs:
                    "solar_kw_capacity": float(row.get("solar_kw_capacity", 0.0)),
                    "solar_profile_24h": solar_prof_csv,
                    "battery_kwh_capacity": float(row.get("battery_kwh_capacity", 0.0)),
                    "battery_max_kw_charge": float(row.get("battery_max_kw_charge", 0.0)),
                    "battery_max_kw_discharge": float(row.get("battery_max_kw_discharge", 0.0)),
                    "battery_roundtrip_eff": float(row.get("battery_roundtrip_eff", 0.92)),
                    "battery_embodied_kgco2e": float(row.get("battery_embodied_kgco2e", 0.0)),
                    "battery_cycle_life": float(row.get("battery_cycle_life", 2500.0)),
                    "ttft_alpha_sec": float(row.get("ttft_alpha_sec", 0.20)),
                    "ttft_beta_ms_per_token": float(row.get("ttft_beta_ms_per_token", 0.0)),
                }
            local_dprint(f"[BUILD] Loaded {count} DC spec rows from '{dc_specs_csv}'")
            if count and debug:
                sample_k = next(iter(dc_params))
                local_dprint(f"[BUILD] DC sample {sample_k}: {dc_params[sample_k]}")
    except FileNotFoundError:
        local_dprint(f"[BUILD] dc_specs file not found: '{dc_specs_csv}' (proceeding with defaults)")

    # 2) GPU perf tables (includes per-config meta.tdp_kw if available)
    perf_tables = load_gpu_perf_tables(gpu_perf_csvs, debug=debug)

    # 3) Build DCs/Nodes/Processors
    dcs: Dict[int, "Datacenter"] = {}
    nodes_by_dc: Dict[int, Dict[int, "Node"]] = {}

    # Helper: get best available TDP kW from perf meta
    def _perf_meta_tdp_kw(accel: str, cfg: str) -> Optional[float]:
        cfg_map = perf_tables.get(accel, {})
        # exact cfg first
        exact = cfg_map.get(cfg, {})
        if isinstance(exact, dict) and "_meta" in exact and "tdp_kw" in exact["_meta"]:
            return exact["_meta"]["tdp_kw"]
        # else take max across cfgs
        best = None
        for _cfg, per_model in cfg_map.items():
            if isinstance(per_model, dict) and "_meta" in per_model and "tdp_kw" in per_model["_meta"]:
                val = per_model["_meta"]["tdp_kw"]
                best = max(best or 0.0, val)
        return best

    with open(node_specs_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        row_idx = 0
        for row in reader:
            row_idx += 1
            dc_id   = int(row.get("dc_id") or row.get("DC_Num") or row.get("dc") or 0)
            node_id = int(row.get("node_id") or row.get("Node_ID") or row.get("node") or 0)
            type_id = int(row.get("type_id") or row.get("Type_ID") or row.get("type") or 0)

            pcount  = int(row.get("processor_count") or row.get("Processor_Count") or 1)

            # Prefer kW if present; else watts → kW conversion if value looks large
            def _to_kw(v) -> float:
                try:
                    x = float(v)
                except (TypeError, ValueError):
                    return 0.0
                return (x / 1000.0) if x > 50.0 else x

            tdp_kw  = _to_kw(row.get("proc_tdp_kw") or row.get("TDP_kW") or row.get("tdp_kw") or row.get("TDP"))
            idle_kw = _to_kw(row.get("proc_idle_kw") or row.get("Idle_kW") or row.get("idle_kw") or 0.0)

            accel_type = str(row.get("accel_type") or row.get("GPU_Type") or row.get("gpu_type") or "A100").strip()
            gpu_cfg    = str(row.get("gpu_config") or row.get("GPU_Config") or row.get("config") or "default").strip() or "default"

            # Create DC if needed (force same solar across all DCs)
            if dc_id not in dcs:
                dp = dc_params.get(dc_id, {})
                # enforce uniform solar
                dp_solar_kw   = float(solar_kw_default)
                dp_solar_prof = list(solar_prof_default)
                dcs[dc_id] = Datacenter(
                    dc_id=dc_id,
                    nodes=[],
                    cop=dp.get("cop", 3.0),
                    other_hw_overhead=dp.get("other_hw_overhead", 0.13),
                    cooling_overhead_multiplier=dp.get("cooling_overhead_multiplier", 3.0),
                    carbon_intensity_kg_per_kwh=dp.get("carbon_intensity_kg_per_kwh", 0.4),
                    water_evap_m3_per_kwh=dp.get("water_evap_m3_per_kwh", 0.0004),
                    blowdown_ratio=dp.get("blowdown_ratio", 0.25),
                    solar_kw_capacity=dp_solar_kw,
                    solar_profile_24h=dp_solar_prof,
                    battery_kwh_capacity=dp.get("battery_kwh_capacity", 0.0),
                    battery_max_kw_charge=dp.get("battery_max_kw_charge", 0.0),
                    battery_max_kw_discharge=dp.get("battery_max_kw_discharge", 0.0),
                    battery_roundtrip_eff=dp.get("battery_roundtrip_eff", 0.92),
                    battery_embodied_kgco2e=dp.get("battery_embodied_kgco2e", 0.0),
                    battery_cycle_life=dp.get("battery_cycle_life", 2500.0),
                    ttft_alpha_sec=dp.get("ttft_alpha_sec", 0.20),
                    ttft_beta_ms_per_token=dp.get("ttft_beta_ms_per_token", 0.0),
                )
                local_dprint(f"[BUILD] Created DC {dc_id} from specs (solar={dp_solar_kw} kW, profile=uniform)")

            # Create Node if needed
            if dc_id not in nodes_by_dc:
                nodes_by_dc[dc_id] = {}
            if node_id not in nodes_by_dc[dc_id]:
                nodes_by_dc[dc_id][node_id] = Node(node_id=node_id, type_id=type_id, processors=[])
                dcs[dc_id].nodes.append(nodes_by_dc[dc_id][node_id])
                local_dprint(f"[BUILD] Added Node {node_id} to DC {dc_id} (type_id={type_id})")

            # Select model perf for accel+cfg; if not found, merge all cfgs (skipping _meta)
            config_map = perf_tables.get(accel_type, {})
            model_perf = config_map.get(gpu_cfg)
            merged_used = False
            if model_perf is None:
                merged_used = True
                merged: Dict[str, Dict[str, float]] = {}
                for _cfg, per_model in config_map.items():
                    if not isinstance(per_model, dict):
                        continue
                    for m, perf in per_model.items():
                        if m == "_meta":
                            continue
                        merged.setdefault(m, {})
                        # prefer entries that already include ms_per_token
                        if ("ms_per_token" in perf) or ("ms_per_token" not in merged[m]):
                            merged[m].update(perf)
                model_perf = merged

            # Power fallbacks: use perf meta TDP if node row is zero/missing
            meta_tdp = _perf_meta_tdp_kw(accel_type, gpu_cfg)
            if (tdp_kw is None) or (tdp_kw <= 0.0):
                if meta_tdp:
                    tdp_kw = float(meta_tdp)
                else:
                    tdp_kw = 0.8  # conservative default kW if nothing available
            if (idle_kw is None) or (idle_kw <= 0.0):
                idle_kw = 0.1 * tdp_kw  # default idle = 10% of TDP

            local_dprint(
                f"[BUILD] row#{row_idx} DC {dc_id} Node {node_id}: accel={accel_type} cfg='{gpu_cfg}' "
                f"pcount={pcount} TDP={tdp_kw:.3f}kW idle={idle_kw:.3f}kW "
                f"perf={'merged' if merged_used else 'exact'} models={len([k for k in (model_perf or {}) if k != '_meta'])}"
            )

            # Create processors
            for pid in range(pcount):
                proc = Processor(
                    proc_id=pid,
                    node_id=node_id,
                    epoch_length=epoch_length,
                    power_state="on",
                    tdp_kw=tdp_kw,
                    idle_kw=idle_kw,
                    model_perf=model_perf or {},
                )
                nodes_by_dc[dc_id][node_id].processors.append(proc)

    # per-DC summary
    for dc_id, dc in dcs.items():
        n_nodes = len(dc.nodes)
        n_procs = sum(len(n.processors) for n in dc.nodes)
        local_dprint(f"[BUILD] DC {dc_id} summary: nodes={n_nodes}, procs={n_procs}")

    return [dcs[k] for k in sorted(dcs.keys())]



