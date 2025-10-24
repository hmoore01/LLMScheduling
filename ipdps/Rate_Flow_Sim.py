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
import math
import collections

# -----------------------------
# Processor / Node definitions
# -----------------------------

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
        cfg = self.model_perf.get(str(model_type))
        if not cfg:
            return 0.0
        if "ms_per_token" in cfg:
            mspt = max(1e-6, float(cfg["ms_per_token"]))
            return 1000.0 / mspt
        mr = cfg.get("ms_per_request")
        tav = cfg.get("avg_tokens_per_request")
        if mr is not None and tav is not None:
            sec = max(1e-6, float(mr) / 1000.0)
            return float(tav) / sec
        # Fallbacks
        exec_ms = cfg.get("exec_ms")
        default_tokens = cfg.get("default_tokens")
        if exec_ms is not None and default_tokens is not None:
            sec = max(1e-6, float(exec_ms) / 1000.0)
            return float(default_tokens) / sec
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
    dc_id: int
    nodes: List[Node]

    # Environment / infra parameters
    cop: float = 3.0                               # Coefficient of performance
    other_hw_overhead: float = 0.13                # 13% of processor energy
    cooling_overhead_multiplier: float = 3.0       # supporting equipment
    carbon_intensity_kg_per_kwh: float = 0.4       # site-specific grid intensity

    # Water parameters (example placeholders; customize to your model)
    potable_kwh_per_m3: float = 0.6
    wastewater_kwh_per_m3: float = 0.5
    water_evap_m3_per_kwh: float = 0.0004         # m^3 per kWh (example)
    blowdown_ratio: float = 0.25                   # blowdown = evap / ratio

    # Optional per-DC TTFT constant (seconds)
    ttft_alpha_sec: Optional[float] = None

    # --- On-site Solar (simple capacity factor curve) ---
    solar_kw_capacity: float = 0.0                 # nameplate DC solar (kW)
    solar_profile_24h: Optional[List[float]] = None  # 24 hourly capacity factors (0..1)

    # --- Battery (Li-ion) ---
    battery_kwh_capacity: float = 0.0
    battery_max_kw_charge: float = 0.0
    battery_max_kw_discharge: float = 0.0
    battery_roundtrip_eff: float = 0.92            # AC-to-AC round-trip efficiency
    battery_soc_kwh: float = 0.0                   # current state of charge (kWh)
    battery_embodied_kgco2e: float = 0.0           # total embodied carbon
    battery_cycle_life: float = 2500.0             # full cycles over life

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
                "soc_kwh_after": batt["soc_kwh_after"],
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

    # ---------- Battery model ----------
    def _battery_dispatch_kwh(self, demand_kwh: float, solar_kwh: float, hours: float) -> Dict[str, float]:
        # Use solar first to meet demand
        solar_used = min(demand_kwh, solar_kwh)
        remaining_demand = demand_kwh - solar_used
        solar_surplus = solar_kwh - solar_used

        # Charge battery with solar surplus (power & capacity limited)
        can_charge_kwh = min(self.battery_max_kw_charge * hours, max(0.0, self.battery_kwh_capacity - self.battery_soc_kwh))
        charge_from_solar = min(solar_surplus, can_charge_kwh)
        # Round-trip efficiency: store only eff*charge as available for discharge later? Commonly losses split.
        eff = max(0.1, min(1.0, self.battery_roundtrip_eff))
        # Assume charge-side losses; energy added to SOC = charge_from_solar * sqrt(eff)
        eta_c = math.sqrt(eff)
        eta_d = math.sqrt(eff)
        self.battery_soc_kwh += charge_from_solar * eta_c
        solar_surplus -= charge_from_solar

        # Discharge battery to reduce remaining demand
        can_discharge_kwh = min(self.battery_max_kw_discharge * hours, self.battery_soc_kwh)
        discharge = min(remaining_demand / max(1e-6, eta_d), can_discharge_kwh)
        discharge_to_load = discharge * eta_d
        self.battery_soc_kwh -= discharge
        remaining_demand -= discharge_to_load

        grid_import = max(0.0, remaining_demand)

        # Embodied carbon amortization per discharged kWh (to load)
        embodied_per_full_cycle_kwh = max(1e-6, self.battery_kwh_capacity * self.battery_cycle_life)
        embodied_per_kwh = float(self.battery_embodied_kgco2e) / embodied_per_full_cycle_kwh
        embodied_carbon = embodied_per_kwh * discharge_to_load

        return {
            "solar_used_kwh": solar_used,
            "batt_charge_from_solar_kwh": charge_from_solar,
            "batt_discharge_kwh": discharge_to_load,
            "grid_import_kwh": grid_import,
            "embodied_carbon_kg": embodied_carbon,
            "soc_kwh_after": self.battery_soc_kwh,
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
    def __init__(self, datacenters: List[Datacenter], dc_latency_ms: Iterable[Iterable[float]] | callable):
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
                "soc_kwh_after": batt["soc_kwh_after"],
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

def build_datacenters_from_csv(
    node_specs_csv: str,
    gpu_perf_csvs: Dict[str, str],  # e.g., {"A100": "A100_GPU.csv", "H100": "H100_GPU.csv"}
    dc_specs_csv: str,
    epoch_length: int,
) -> List[Datacenter]:
    """Construct Datacenter/Node/Processor graph from CSVs.

    Expected node_specs headers (example):
      dc_id,node_id,type_id,processor_count,proc_tdp_kw,proc_idle_kw,accel_type

    Expected gpu perf CSVs (one per accel_type). Headers (example):
      config,model_type,ms_per_token,ms_per_request,avg_tokens_per_request
      (rows for each config used by type_id; use a mapping below)

    Expected dc_specs headers (example):
      dc_id,cop,carbon_intensity_kg_per_kwh,water_evap_m3_per_kwh,blowdown_ratio,
      solar_kw_capacity,solar_profile_24h (semicolon-separated 24 floats),
      battery_kwh_capacity,battery_max_kw_charge,battery_max_kw_discharge,
      battery_roundtrip_eff,battery_embodied_kgco2e,battery_cycle_life

    Assumptions: type_id implicitly maps to an accel_type key.
    """
    # 1) Load DC infra specs
    dc_params: Dict[int, Dict] = {}
    with open(dc_specs_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            did = int(row["DC_Num"])  # required
            prof = row.get("solar_profile_24h", "").strip()
            solar_prof = [float(x) for x in prof.split(";") if x.strip()] if prof else None
            dc_params[did] = {
                "cop": float(row.get("cop", 3.0)),
                "carbon_intensity_kg_per_kwh": float(row.get("carbon_intensity_kg_per_kwh", 0.4)),
                "water_evap_m3_per_kwh": float(row.get("water_evap_m3_per_kwh", 0.0004)),
                "blowdown_ratio": float(row.get("blowdown_ratio", 0.25)),
                "solar_kw_capacity": float(row.get("solar_kw_capacity", 0.0)),
                "solar_profile_24h": solar_prof,
                "battery_kwh_capacity": float(row.get("battery_kwh_capacity", 0.0)),
                "battery_max_kw_charge": float(row.get("battery_max_kw_charge", 0.0)),
                "battery_max_kw_discharge": float(row.get("battery_max_kw_discharge", 0.0)),
                "battery_roundtrip_eff": float(row.get("battery_roundtrip_eff", 0.92)),
                "battery_embodied_kgco2e": float(row.get("battery_embodied_kgco2e", 0.0)),
                "battery_cycle_life": float(row.get("battery_cycle_life", 2500.0)),
            }

    # 2) Prepare GPU perf per accel_type
    accel_perf: Dict[str, Dict[str, Dict[str, float]]] = {}
    for accel_type, path in gpu_perf_csvs.items():
        table: Dict[str, Dict[str, float]] = {}
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                model = str(row["Model_Name"]).strip()
                entry = {}
                if row.get("ms_per_token"):
                    entry["ms_per_token"] = float(row["ms_per_token"]) if row["ms_per_token"] else None
                if row.get("ms_per_request"):
                    entry["ms_per_request"] = float(row["ms_per_request"]) if row["ms_per_request"] else None
                if row.get("avg_tokens_per_request"):
                    entry["avg_tokens_per_request"] = float(row["avg_tokens_per_request"]) if row["avg_tokens_per_request"] else None
                # You can extend with exec_ms/default_tokens if available
                table[model] = entry
        accel_perf[accel_type] = table

    # 3) Build DCs, Nodes, Procs
    dcs: Dict[int, Datacenter] = {}
    nodes_by_dc: Dict[int, Dict[int, Node]] = collections.defaultdict(dict)

    with open(node_specs_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dc_id = int(row["dc_id"]) ; node_id = int(row["node_id"]) ; type_id = int(row["type_id"]) ;
            pcount = int(row.get("processor_count", 1))
            tdp_kw = float(row.get("proc_tdp_kw", 0.0))
            idle_kw = float(row.get("proc_idle_kw", 0.0))
            accel_type = str(row.get("accel_type", "A100")).strip()

            # Ensure DC exists
            if dc_id not in dcs:
                dp = dc_params.get(dc_id, {})
                dcs[dc_id] = Datacenter(
                    dc_id=dc_id,
                    nodes=[],
                    cop=dp.get("cop", 3.0),
                    carbon_intensity_kg_per_kwh=dp.get("carbon_intensity_kg_per_kwh", 0.4),
                    water_evap_m3_per_kwh=dp.get("water_evap_m3_per_kwh", 0.0004),
                    blowdown_ratio=dp.get("blowdown_ratio", 0.25),
                    solar_kw_capacity=dp.get("solar_kw_capacity", 0.0),
                    solar_profile_24h=dp.get("solar_profile_24h"),
                    battery_kwh_capacity=dp.get("battery_kwh_capacity", 0.0),
                    battery_max_kw_charge=dp.get("battery_max_kw_charge", 0.0),
                    battery_max_kw_discharge=dp.get("battery_max_kw_discharge", 0.0),
                    battery_roundtrip_eff=dp.get("battery_roundtrip_eff", 0.92),
                    battery_embodied_kgco2e=dp.get("battery_embodied_kgco2e", 0.0),
                    battery_cycle_life=dp.get("battery_cycle_life", 2500.0),
                )

            # Ensure Node exists
            if node_id not in nodes_by_dc[dc_id]:
                nodes_by_dc[dc_id][node_id] = Node(node_id=node_id, type_id=type_id, processors=[])
                dcs[dc_id].nodes.append(nodes_by_dc[dc_id][node_id])

            # Add processors
            for pid in range(pcount):
                proc = Processor(
                    proc_id=pid,
                    node_id=node_id,
                    epoch_length=epoch_length,
                    power_state="on",
                    tdp_kw=tdp_kw,
                    idle_kw=idle_kw,
                    model_perf=accel_perf.get(accel_type, {}),
                )
                nodes_by_dc[dc_id][node_id].processors.append(proc)

    # Return list ordered by dc_id
    return [dcs[k] for k in sorted(dcs.keys())]
