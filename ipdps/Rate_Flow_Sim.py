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
# ===========================================================
# === Simulation Constants (hardcoded, no CSV dependency) ===
# ===========================================================

CONSTANTS = {
    # --- Temperature / cooling ---
    "TEMP_REF_C": 20.0,          # reference temperature baseline (°C)
    "TEMP_SETPOINT_C": 29.0,     # supply temperature setpoint (°C)
    "IT_POWER_TEMP_ALPHA": 0.0,  # IT power flat vs. temperature
    "EXEC_MS_TEMP_ALPHA": 0.0,   # no perf degradation (can set +0.005 per °C if needed)
    "COP_TEMP_ALPHA_PER_C": 0.16,  # +4% COP per °C above ref
    "PUE_TEMP_ALPHA_PER_C": -0.04, # -0.01 PUE per °C above ref

    # --- Cooling defaults ---
    "DEFAULT_COP": 3.0,          # mechanical cooling COP baseline
    "DEFAULT_PUE": 1.18,         # for liquid/oil systems
    "OTHER_IT_OVERHEAD_FRAC": 0.13,  # non-CPU IT overhead fraction

    # --- Solar / battery defaults ---
    "SOLAR_KW_CAPACITY": 0.0,   # typical DC-scale PV array
    "SOLAR_PROFILE_24H": [0.0, 0.0, 0.0, 0.0, 0.05, 0.15, 0.35, 0.55,
                          0.75, 0.9, 1.0, 0.9, 0.75, 0.55, 0.35, 0.15,
                          0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # normalized day curve

    # --- Battery defaults ---
    "BATTERY_CAP_KWH": 0.0,
    "BATTERY_SOC_INIT": 0.0,
    "BATTERY_MAX_CHARGE_KW": 0.0,
    "BATTERY_MAX_DISCHARGE_KW": 0.0,
    "BATTERY_ROUNDTRIP_EFF": 0.92,
    "BATTERY_EMBODIED_CO2_PER_KWH": 0.05,  # kg CO2 per kWh throughput

    # --- ToU default ---
    "TOU_PRICE_24H": [0.12, 0.12, 0.12, 0.12, 0.14, 0.15, 0.18, 0.20,
                      0.22, 0.25, 0.25, 0.23, 0.22, 0.20, 0.18, 0.16,
                      0.14, 0.13, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12],
}

SPEC_CPU_WORKLOADS = [
    "bwaves_r",
    "namd_r",
    "povray_r",
    "cactusBSSN_r",
    "parest_r",
    "fotonik3d_r",
    "perlbench_r",
    "gcc_r",
    "mcf_r",
    "deepsjeng_r",
    "exchange2_r",
]

SPECINT_POWER_DATA = {
    "perlbench_r":  {"Temp": [30, 36, 39, 45, 48], "Power": [260, 256, 258, 260, 261]},
    "gcc_r":        {"Temp": [28, 35, 38, 49, 58], "Power": [256, 253, 254, 256, 260]},
    "mcf_r":        {"Temp": [28, 36, 39, 50, 60], "Power": [275, 270, 271, 274, 278]},
    "omnetpp_r":    {"Temp": [28, 34, 37, 49, 59], "Power": [246, 241, 241, 246, 251]},
    "xalancbmk_r":  {"Temp": [31, 37, 40, 50, 60], "Power": [289, 285, 282, 286, 291]},
    "x264_r":       {"Temp": [30, 35, 39, 49, 58], "Power": [258, 255, 255, 258, 262]},
    "deepsjeng_r":  {"Temp": [30, 35, 39, 50, 59], "Power": [248, 247, 248, 248, 253]},
    "leela_r":      {"Temp": [29, 34, 38, 49, 57], "Power": [232, 231, 231, 232, 236]},
    "exchange2_r":  {"Temp": [29, 34, 38, 49, 58], "Power": [238, 237, 237, 238, 242]},
    "xz_r":         {"Temp": [28, 33, 37, 48, 57], "Power": [229, 228, 228, 230, 234]},
}

SPECFP_POWER_DATA = {
    "bwaves_r":     {"Temp": [32, 38, 44, 50, 60], "Power": [310, 307, 307, 308, 312]},
    "cactuBSSN_r":  {"Temp": [30, 36, 42, 50, 60], "Power": [281, 278, 279, 281, 286]},
    "namd_r":       {"Temp": [30, 36, 42, 50, 60], "Power": [273, 271, 270, 271, 277]},
    "parest_r":     {"Temp": [28, 34, 40, 49, 59], "Power": [256, 253, 254, 257, 262]},
    "povray_r":     {"Temp": [32, 38, 44, 50, 60], "Power": [293, 291, 290, 291, 295]},
    "lbm_r":        {"Temp": [27, 33, 38, 49, 56], "Power": [239, 236, 237, 240, 244]},
    "wrf_r":        {"Temp": [28, 33, 40, 50, 59], "Power": [252, 250, 250, 252, 258]},
    "blender_r":    {"Temp": [30, 35, 42, 50, 60], "Power": [273, 270, 269, 271, 276]},
    "cam4_r":       {"Temp": [30, 36, 43, 50, 60], "Power": [296, 293, 293, 294, 299]},
    "imagick_r":    {"Temp": [29, 35, 42, 50, 60], "Power": [258, 256, 255, 256, 261]},
    "nab_r":        {"Temp": [29, 35, 42, 50, 60], "Power": [281, 278, 278, 279, 283]},
    "fotonik3d_r":  {"Temp": [26, 31, 38, 48, 56], "Power": [228, 226, 226, 228, 232]},
    "roms_r":       {"Temp": [27, 32, 40, 49, 58], "Power": [246, 244, 244, 247, 252]},
}

# Merge int + fp tables for easy lookup
SPEC_CPU_POWER_TABLES = {**SPECINT_POWER_DATA, **SPECFP_POWER_DATA}


def interp_piecewise_linear(x, xs, ys):
    """Piecewise linear interpolation."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i+1]:
            x0, x1 = xs[i], xs[i+1]
            y0, y1 = ys[i], ys[i+1]
            t = (x - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return ys[-1]


def spec_cpu_temp_multiplier(workload, temp_c):
    """
    Compute multiplier from raw SPEC CPU power tables.
    Returns a normalized multiplier relative to mid-temperature (~40°C).
    """
    if workload not in SPEC_CPU_POWER_TABLES:
        return None

    tab = SPEC_CPU_POWER_TABLES[workload]
    xs = tab["Temp"]
    ys = tab["Power"]

    raw = interp_piecewise_linear(temp_c, xs, ys)

    # Normalize around the middle point (~40°C) to produce a multiplier.
    # We pick the value closest to 40°C in the table:
    idx = min(range(len(xs)), key=lambda i: abs(xs[i] - 40))
    ref = ys[idx]

    return raw / ref



MANUAL_NODE_TYPE_COUNTS: Optional[Dict[int, Dict[int, int]]] = None


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
class Battery:
    cap_kwh: float = 0.0
    soc_kwh: float = 0.0
    max_charge_kw: float = 0.0
    max_discharge_kw: float = 0.0
    roundtrip_eff: float = 0.92
    embodied_co2_per_kwh_throughput: float = 0.0  # kgCO2 per kWh throughput

    # per-epoch accounting
    charged_kwh: float = 0.0
    discharged_kwh: float = 0.0

    def reset_epoch(self):
        self.charged_kwh = 0.0
        self.discharged_kwh = 0.0

    def can_use(self) -> bool:
        return (
            self.cap_kwh > 0.0
            and self.max_charge_kw > 0.0
            and self.max_discharge_kw > 0.0
            and self.roundtrip_eff > 0.0
        )

    def charge(self, want_kwh: float, hours: float) -> float:
        """Return stored kWh added to SOC (post-efficiency). Throughput counted at input."""
        if not self.can_use() or want_kwh <= 0.0:
            return 0.0
        hours = max(hours, 1e-9)
        limit_kwh = self.max_charge_kw * hours
        headroom_in_kwh = max(0.0, self.cap_kwh - self.soc_kwh) / self.roundtrip_eff
        in_kwh = min(want_kwh, limit_kwh, headroom_in_kwh)
        stored = in_kwh * self.roundtrip_eff
        if stored <= 0.0:
            return 0.0
        self.soc_kwh += stored
        self.charged_kwh += in_kwh
        return stored

    def discharge(self, want_kwh: float, hours: float) -> float:
        """Return delivered kWh to load; throughput counted at output."""
        if not self.can_use() or want_kwh <= 0.0:
            return 0.0
        hours = max(hours, 1e-9)
        limit_kwh = self.max_discharge_kw * hours
        deliverable = min(want_kwh, limit_kwh, self.soc_kwh)
        if deliverable <= 0.0:
            return 0.0
        self.soc_kwh -= deliverable
        self.discharged_kwh += deliverable
        return deliverable

    def embodied_carbon_kg(self) -> float:
        throughput = self.charged_kwh + self.discharged_kwh
        return throughput * max(0.0, self.embodied_co2_per_kwh_throughput)



@dataclass
class ProcNode:
    def __init__(
        self,
        node_id: int,
        model_perf: dict,
        *,
        # old/builder-style args:
        type_id: int | None = None,
        accel_type: str | None = None,
        gpu_config: str | None = None,
        tdp_kw: float | None = None,
        idle_kw: float | None = None,
        # optional modern args:
        tdp_w: float | None = None,
        idle_w: float | None = None,
        base_idle_frac: float | None = None,
        workload_class: str = "generic",
    ):
        self.node_id = int(node_id)
        self.type_id = type_id
        self.accel_type = accel_type
        self.gpu_config = gpu_config

        # Normalize power inputs
        # Prefer explicit Watts if given; else convert from kW.
        tdp_w_norm = float(tdp_w) if tdp_w is not None else (
            float(tdp_kw) * 1000.0 if tdp_kw is not None else 0.0
        )
        idle_w_norm = float(idle_w) if idle_w is not None else (
            float(idle_kw) * 1000.0 if idle_kw is not None else None
        )

        self.tdp_w = tdp_w_norm
        # Derive idle fraction if not explicitly provided
        if base_idle_frac is not None:
            self.base_idle_frac = float(base_idle_frac)
        elif idle_w_norm is not None and self.tdp_w > 0.0:
            self.base_idle_frac = max(0.0, min(1.0, idle_w_norm / self.tdp_w))
        else:
            # safe default
            self.base_idle_frac = 0.13

        self.model_perf = dict(model_perf)  # {"Llama7b": {...}, "Llama70b": {...}}
        self.state = "IDLE"                 # "ON" | "IDLE" | "OFF"
        self.available_at_ms = 0.0
        self.dc_ref = None
        self.busy_ms_epoch = 0.0

        self.workload_class = str(workload_class).lower()

    def attach_dc(self, dc):
        self.dc_ref = dc

    # --- IT power fraction by state (IT side only) ---
    def it_frac_for_state(self) -> float:
        s = self.state
        if s == "OFF":
            return 0.0
        if s == "IDLE":
            return self.base_idle_frac
        return 1.0  # ON

    def _base_it_power_w(self) -> float:
        """
        Returns baseline IT power draw in Watts before temp scaling.
        GPUs use hardware TDP. CPUs use SPEC per-workload baseline power
        (mid-point of provided power curve).
        """
        # Default = GPU / accelerator path
        base = float(self.tdp_w or 0.0)

        # Only override if CPU
        if self.accel_type not in ("CPU", "cpu"):
            return base

        # workload classification
        wc = (self.workload_class or "").lower()

        # resolve SPEC workload key
        workload_key = None
        for key in SPEC_CPU_POWER_TABLES.keys():
            if key.replace("_r", "") in wc:  # matching without _r suffix
                workload_key = key
                break

        if workload_key:
            ptab = SPEC_CPU_POWER_TABLES[workload_key]
            powers = ptab["Power"]
            mid = powers[len(powers) // 2]  # midpoint as baseline
            base = float(mid)

        return base

    # --- temperature multipliers coming from DC constants ---
    # --- temperature multipliers coming from DC constants + ITD-aware model ---
    # --- temperature multipliers coming from DC constants + ITD-aware model ---
    def _it_power_temp_mult(self) -> float:
        """
        Temperature-based multiplier. Uses SPEC CPU piecewise-linear multipliers
        normalized relative to nearest value to 40°C. GPUs use DC alpha model.
        """

        dc = self.dc_ref
        if not dc:
            return 1.0

        temp_c = float(dc.temp_c_setpoint)

        # GPU or accelerator → old behavior
        if self.accel_type not in ("CPU", "cpu"):
            dT = temp_c - dc.temp_ref_c
            return max(0.0, 1.0 + dc.it_power_temp_alpha * dT)

        wc = (self.workload_class or "").lower()

        # match to SPEC workload
        workload_key = None
        for key in SPEC_CPU_POWER_TABLES.keys():
            if key.replace("_r", "") in wc:
                workload_key = key
                break

        if workload_key:
            tab = SPEC_CPU_POWER_TABLES[workload_key]
            xs = tab["Temp"]
            ys = tab["Power"]

            raw = interp_piecewise_linear(temp_c, xs, ys)

            # normalize using closest point to 40°C
            idx = min(range(len(xs)), key=lambda i: abs(xs[i] - 40))
            ref = ys[idx]
            return raw / ref

        # Fallback generic CPU curve
        def _generic_cpu_mult(T: float) -> float:
            T_opt = 35.0
            d = (T - T_opt) / 25.0
            m = 1.0 + 0.04 * (d * d) + 0.08 * (d ** 4)
            return max(0.85, min(1.20, m))

        return _generic_cpu_mult(temp_c)


    def _exec_ms_temp_mult(self) -> float:
        dc = self.dc_ref
        if not dc:
            return 1.0
        dT = float(dc.temp_c_setpoint) - dc.temp_ref_c
        return max(0.0, 1.0 + dc.exec_ms_temp_alpha * dT)   # usually 0.0 unless you enable it

    # --- performance model ---
    def estimate_exec_ms(self, tokens, model: str, kwargs) -> float:
        rec = self.model_perf.get(model, {})
        base_ms = float(rec.get("ms_per_request") or 0.0)
        base_ms  = base_ms * tokens
        if base_ms <= 0.0:
            ms_per_tok = float(rec.get("ms_per_token") or 0.0)
            toks = float(kwargs.get("tokens") or kwargs.get("avg_tokens") or 0.0)
            base_ms = ms_per_tok * toks
        return base_ms * self._exec_ms_temp_mult()

    # --- IT energy for execution window (kWh) ---
    def it_energy_kwh_for_exec(self, exec_ms: float) -> float:
        if self.state == "OFF" or self.tdp_w <= 0.0:
            return 0.0

        # For the duration of this exec window, treat the node as ON
        self.state = "ON"
        it_frac = self.it_frac_for_state()  # will be 1.0 when state == "ON"

        # Use workload-specific base power for CPUs, TDP for GPUs
        base_power_w = self._base_it_power_w()

        # Apply temperature multiplier (ITD for CPUs, linear alpha for GPUs)
        power_w = base_power_w * it_frac * self._it_power_temp_mult()  # W

        hours = max(0.0, float(exec_ms)) / 3_600_000.0  # ms -> hours
        self.busy_ms_epoch += float(exec_ms)

        # Return to IDLE so idle accounting can pick it up separately
        self.state = "IDLE"

        # Convert Wh -> kWh
        return power_w * hours / 1000.0





# -----------------------------
# Datacenter definition
# -----------------------------

@dataclass
class Datacenter:
    """
    Cooling modes:
      - "MECH_COP"          : cooling energy via COP (with temp-sensitive COP)
      - "LIQUID_WATER_PUE"  : facility overhead via PUE (temp-sensitive PUE)
      - "LIQUID_OIL_PUE"    : same PUE path (you can set different DEFAULT_PUE if desired)

    New features:
      - Temperature setpoint affects COP/PUE (NOT IT power by default)
      - Solar PV + Battery with embodied carbon per kWh throughput
      - ToU pricing for grid energy cost
      - Per-epoch energy/carbon/cost accounting
      - Per-epoch busy time and a utilization() helper

    Compatible with:
      - add_node(...), apply_power_plan(...)
      - settle_and_score(node, exec_ms, start_ms)  -> dict
      - reset_epoch(), finalize_epoch()
    """

    def __init__(
            self,
            dc_id: int,
            carbon_intensity_g_per_kwh: float,
            time_of_use_24h: list[float] | None = None,
            cop_profile_24h: list[float] | None = None,
            blowdown_ratio: float | None = None,
            water_cycling_density_m3_per_kwh_heat: float | None = None,
            potable_EI_kWh_per_m3: float | None = None,
            wastewater_EI_kWh_per_m3: float | None = None,
            water_static_m3_per_kwh_heat: float | None = None,
            cooling_mode: str = "MECH_COP",  # "MECH_COP", "LIQUID_WATER_PUE", "LIQUID_OIL_PUE"
            epoch_length: int | None = None,
            debug: bool = False,
    ):
        self.id = int(dc_id)
        self.carbon_intensity_g_per_kwh = float(carbon_intensity_g_per_kwh)
        self.cooling_mode = str(cooling_mode)
        self.debug = bool(debug)

        # --- constants wiring (no CSV dependency) ---
        self.temp_c_setpoint = CONSTANTS["TEMP_SETPOINT_C"]
        self.temp_ref_c = CONSTANTS["TEMP_REF_C"]
        self.it_power_temp_alpha = CONSTANTS["IT_POWER_TEMP_ALPHA"]
        self.exec_ms_temp_alpha = CONSTANTS["EXEC_MS_TEMP_ALPHA"]
        self.cop_temp_alpha_per_C = CONSTANTS["COP_TEMP_ALPHA_PER_C"]
        self.pue_temp_alpha_per_C = CONSTANTS["PUE_TEMP_ALPHA_PER_C"]

        self.cop_default = CONSTANTS["DEFAULT_COP"]
        self.cop_profile_24h = list(cop_profile_24h) if cop_profile_24h else None
        self.pue_value = CONSTANTS["DEFAULT_PUE"]
        self.other_it_overhead_frac = CONSTANTS["OTHER_IT_OVERHEAD_FRAC"]

        # --- Water/cooling parameters from CSV (stored for use elsewhere) ---
        # These are retained so your existing water accounting/printing paths keep working.
        self.blowdown_ratio = float(blowdown_ratio) if blowdown_ratio is not None else 0.30
        self.water_cycling_density = (
            float(water_cycling_density_m3_per_kwh_heat) if water_cycling_density_m3_per_kwh_heat is not None else 0.10
        )
        self.potable_energy_intensity = (
            float(potable_EI_kWh_per_m3) if potable_EI_kWh_per_m3 is not None else 0.005
        )
        self.wastewater_energy_intensity = (
            float(wastewater_EI_kWh_per_m3) if wastewater_EI_kWh_per_m3 is not None else 0.010
        )
        self.water_static = (
            float(water_static_m3_per_kwh_heat) if water_static_m3_per_kwh_heat is not None else 5.0
        )

        self.solar_kw_capacity = CONSTANTS["SOLAR_KW_CAPACITY"]
        self.solar_profile_24h = list(CONSTANTS["SOLAR_PROFILE_24H"])

        self.battery = Battery(
            cap_kwh=CONSTANTS["BATTERY_CAP_KWH"],
            soc_kwh=CONSTANTS["BATTERY_SOC_INIT"],
            max_charge_kw=CONSTANTS["BATTERY_MAX_CHARGE_KW"],
            max_discharge_kw=CONSTANTS["BATTERY_MAX_DISCHARGE_KW"],
            roundtrip_eff=CONSTANTS["BATTERY_ROUNDTRIP_EFF"],
            embodied_co2_per_kwh_throughput=CONSTANTS["BATTERY_EMBODIED_CO2_PER_KWH"],
        )

        self.tou_price = list(time_of_use_24h) if time_of_use_24h else None

        # inventory
        self.units: list[ProcNode] = []

        # epoch counters (reset each epoch)
        self._busy_ms = 0.0
        self.energy_grid_kwh = 0.0
        self.energy_solar_kwh = 0.0
        self.energy_batt_discharge_kwh = 0.0
        self.energy_batt_charge_kwh = 0.0
        self.embodied_battery_co2_kg = 0.0
        self.energy_other_kwh = 0.0
        self.energy_cooling_kwh = 0.0
        self.energy_it_kwh = 0.0
        self.cost_usd = 0.0

        # ---- Water accounting (epoch) ----
        self.water_evap_m3 = 0.0
        self.water_blowdown_m3 = 0.0
        self.water_static_m3 = 0.0
        self.water_makeup_m3 = 0.0
        self.water_energy_potable_kwh = 0.0
        self.water_energy_wastewater_kwh = 0.0
        self.water_energy_total_kwh = 0.0
        self.water_carbon_g = 0.0

        self.energy_cost_usd = 0.0  # mirrors self.cost_usd for reporting
        self.carbon_g = 0.0  # full DC carbon for the epoch
        self.water_usage_m3 = 0.0  # mirrors self.water_makeup_m3

        self._epoch_len_s = float(epoch_length)
        self.last_used_unit = 0

    # ---------- inventory ----------
    def add_node(self, unit: ProcNode):
        unit.attach_dc(self)
        self.units.append(unit)

    # ---------- power plan ----------
    def apply_power_plan(self, plan_slice: dict | None):
        if not plan_slice:
            return
        mode_all = plan_slice.get("all")
        if mode_all in ("ON", "IDLE", "OFF"):
            for u in self.units:
                u.state = mode_all
        unit_modes = plan_slice.get("unit") or {}
        if unit_modes:
            id_map = {u.node_id: u for u in self.units}
            for node_id, state in unit_modes.items():
                uid = int(node_id)
                if uid in id_map and state in ("ON", "IDLE", "OFF"):
                    id_map[uid].state = state

    def schedule_request(
            self,
            *,
            model: str,
            arrival: int | float,
            net_latency_ms: float = 0.0,
            source_dc: int | None = None,
            target_dc: int | None = None,
            tokens: int | None = None,
            **kwargs,
    ) -> dict:
        """
        Minimal queue-free scheduling: pick any non-OFF unit, estimate exec_ms,
        and account energy/carbon/cost/water. Returns fields used by Geo_Network.
        """
        # choose a unit (prefer ON > IDLE > OFF)
        available_units = [u for u in self.units if u.state in ("ON", "IDLE")]

        if not available_units:
            raise RuntimeError("No units available for scheduling request.")

        # Round-robin selection
        self.last_used_unit = (self.last_used_unit + 1) % len(available_units)
        unit = available_units[self.last_used_unit]

        # If nothing usable, return network latency only (debug-safe)
        if unit is None:
            return {
                "dc_id": int(self.id),
                "start_ms": float(arrival),
                "end_ms": float(arrival),
                "exec_ms": 0.0,
                "ttft_s": float(net_latency_ms) / 1000.0,
                "energy_kwh": 0.0,
                "carbon_g": 0.0,
                "cost_usd": 0.0,
                "water_m3": 0.0,
            }

        # Estimate execution time from node perf (tokens optional)
        exec_ms = float(unit.estimate_exec_ms(tokens, model, kwargs or {}))
        start_ms = float(arrival)
        end_ms = start_ms + exec_ms

        # Energy/carbon/cost/water accounting
        score = self.settle_and_score(unit, exec_ms, start_ms)

        # Simple ttft_s: network + service (no queue)
        ttft_s = float(net_latency_ms) / 1000.0 + (exec_ms / 1000.0 if exec_ms > 0 else 0.0)

        return {
            "dc_id": int(self.id),
            "start_ms": start_ms,
            "end_ms": end_ms,
            "exec_ms": exec_ms,
            "ttft_s": ttft_s,
            "energy_kwh": float(score.get("energy_kwh", 0.0)),
            "carbon_g": float(score.get("carbon_g", 0.0)),
            "cost_usd": float(score.get("cost_usd", 0.0)),
            "water_m3": float(score.get("water_m3", 0.0)),
        }

    # ---------- time helpers ----------
    def _hour_of_day(self, ms: float) -> int:
        sec = (float(ms) / 1000.0) % 86400.0
        return int(sec // 3600)

    def _tou_price(self, ms: float) -> float:
        if not self.tou_price:
            return 0.0
        h = self._hour_of_day(ms)
        return float(self.tou_price[h % len(self.tou_price)])

    # ---------- cooling helpers (temp-aware) ----------
    def _cop_for_ms(self, ms: float) -> float:
        # Start from hourly COP profile if provided; else baseline constant.
        if self.cop_profile_24h and len(self.cop_profile_24h) >= 24:
            base = float(self.cop_profile_24h[self._hour_of_day(ms) % 24])
        else:
            base = self.cop_default

        # Apply temperature sensitivity (positive alpha increases COP with higher setpoint)
        dT = float(self.temp_c_setpoint) - self.temp_ref_c
        cop = base * max(0.0, 1.0 + self.cop_temp_alpha_per_C * dT)
        return max(1.0, cop)

    def _pue_for_ms(self, ms: float) -> float:
        base = self.pue_value
        dT = float(self.temp_c_setpoint) - self.temp_ref_c
        pue = base + self.pue_temp_alpha_per_C * dT
        return max(1.0, pue)

    def _solar_kw_at_ms(self, ms: float) -> float:
        if self.solar_kw_capacity <= 0.0 or not self.solar_profile_24h:
            return 0.0
        h = self._hour_of_day(ms)
        frac = float(self.solar_profile_24h[h % len(self.solar_profile_24h)])
        return max(0.0, self.solar_kw_capacity * frac)

    # ---------- per-exec energy ----------
    def _energy_for_exec_kwh(self, u: ProcNode, exec_ms: float, start_ms: float) -> float:
        """
        Total DC energy for this execution BEFORE PV/battery offset:
          IT + 'other IT' overhead + cooling (via COP or PUE).
        """
        it_kwh = u.it_energy_kwh_for_exec(exec_ms)
        other_kwh = it_kwh * max(0.0, self.other_it_overhead_frac)

        if self.cooling_mode == "MECH_COP":
            cop = max(0.1, self._cop_for_ms(start_ms))
            cooling_kwh = it_kwh / cop
            infra_kwh = it_kwh + other_kwh + cooling_kwh
        else:
            # Liquid cooling via PUE: facility energy = IT * (PUE - 1)
            pue = max(1.0, self._pue_for_ms(start_ms))
            non_it_facility_kwh = it_kwh * (pue - 1.0)
            # Treat "other_kwh" as part of non-IT facility; remainder is cooling
            cooling_kwh = max(0.0, non_it_facility_kwh - other_kwh)
            infra_kwh = it_kwh + other_kwh + cooling_kwh

        # accumulate components (for diagnostics)
        self.energy_it_kwh += it_kwh
        self.energy_other_kwh += other_kwh
        self.energy_cooling_kwh += cooling_kwh

        # utilization (busy time)
        self._busy_ms += max(0.0, float(exec_ms))

        return infra_kwh

    def _apply_solar_battery_offset(self, gross_kwh: float, start_ms: float, end_ms: float) -> float:
        """
        Priority: PV -> load, then battery discharge. Charge battery with PV surplus.
        Returns kWh that must be drawn from the grid.
        """
        hours = max(1e-9, (end_ms - start_ms) / 3_600_000.0)
        pv_kw = self._solar_kw_at_ms(start_ms)
        pv_kwh = pv_kw * hours

        # PV to load
        pv_to_load = min(pv_kwh, gross_kwh)
        self.energy_solar_kwh += pv_to_load
        remaining = gross_kwh - pv_to_load

        # Battery discharge to cover remaining
        batt_deliver = self.battery.discharge(remaining, hours) if self.battery else 0.0
        remaining -= batt_deliver
        self.energy_batt_discharge_kwh += batt_deliver

        # PV surplus -> battery charge
        surplus = max(0.0, pv_kwh - pv_to_load)
        if surplus > 0.0 and self.battery:
            stored = self.battery.charge(surplus, hours)
            if stored > 0.0:
                self.energy_batt_charge_kwh += stored

        return max(0.0, remaining)  # grid draw


    def _account_water_from_it(self, it_kwh: float, start_ms: float, cop: float) -> float:
        """
        Returns makeup water (m^3) for this execution and updates epoch-level water counters.
        Assumptions (same as before):
          - Heat rejection = IT + Cooling; Cooling = IT/COP (mechanical only)
          - static water use: water_static_m3_per_kwh_heat * heat_rej_kwh
          - evaporative:     water_cycling_density_m3_per_kwh_heat * heat_rej_kwh
          - blowdown rule (ModelSet #80): total draw = evap / blowdown_ratio; blowdown = draw - evap
          - potable energy intensity applies to (evap + static); wastewater EI to blowdown
          - added water energy produces additional carbon (tracked separately as water_carbon_g),
            but DOES NOT alter your existing E/EC/cost paths.
        """
        # heat rejected for mechanical AC
        heat_rej_kwh = it_kwh + (it_kwh / max(1e-9, cop))

        static_m3 = float(self.water_static) * heat_rej_kwh
        evap_m3 = float(self.water_cycling_density) * heat_rej_kwh

        # blowdown
        ratio = max(1e-9, float(self.blowdown_ratio))  # avoid div-by-zero
        total_draw_m3 = evap_m3 / ratio
        blowdown_m3 = max(0.0, total_draw_m3 - evap_m3)

        makeup_m3 = static_m3 + total_draw_m3

        # energy for water processing
        potable_kwh = float(self.potable_energy_intensity) * (evap_m3 + static_m3)
        wastewater_kwh = float(self.wastewater_energy_intensity) * blowdown_m3
        total_water_kwh = potable_kwh + wastewater_kwh

        # carbon from water processing energy (separate; not mixed into main carbon_g)
        water_co2_g = total_water_kwh * float(self.carbon_intensity_g_per_kwh)

        # accumulate epoch counters
        self.water_static_m3 += static_m3
        self.water_evap_m3 += evap_m3
        self.water_blowdown_m3 += blowdown_m3
        self.water_makeup_m3 += makeup_m3
        self.water_energy_potable_kwh += potable_kwh
        self.water_energy_wastewater_kwh += wastewater_kwh
        self.water_energy_total_kwh += total_water_kwh
        self.water_carbon_g += water_co2_g

        return makeup_m3

    def account_energy_carbon_cost(
            self,
            u: ProcNode,
            exec_ms: float,
            start_ms: float
    ) -> tuple[float, float, float, float]:
        """
        Returns (energy_kwh_total, carbon_g, cost_usd, water_m3) for THIS execution.

        - energy_kwh_total: IT + other + cooling before PV/battery (gross facility kWh)
        - carbon_g       : incremental carbon for this exec from grid energy only
        - cost_usd       : incremental cost for this exec from grid energy only
        - water_m3       : incremental makeup water (if MECH_COP)

        NOTE:
          - Water processing carbon is tracked in self.water_carbon_g and not added
            into carbon_g here (to preserve existing reporting semantics).
        """
        # Total DC facility energy for this execution (IT + other + cooling)
        gross_kwh = self._energy_for_exec_kwh(u, exec_ms, start_ms)
        end_ms = start_ms + exec_ms

        # PV + battery offset → grid energy for THIS execution
        grid_kwh = self._apply_solar_battery_offset(gross_kwh, start_ms, end_ms)
        self.energy_grid_kwh += grid_kwh

        # --- Water processing (mechanical cooling only) ---
        water_m3 = 0.0
        if self.cooling_mode == "MECH_COP" and grid_kwh > 0.0:
            cop = max(0.1, self._cop_for_ms(start_ms))
            # IT energy for this execution
            it_kwh_this = u.it_energy_kwh_for_exec(exec_ms)
            water_m3 = self._account_water_from_it(it_kwh_this, start_ms, cop)

        # --- Per-exec carbon and cost from grid energy only ---
        carbon_g = grid_kwh * float(self.carbon_intensity_g_per_kwh)

        price = self._tou_price(start_ms) if self.tou_price else 0.0
        cost_usd = price * grid_kwh

        return gross_kwh, carbon_g, cost_usd, water_m3

    def settle_and_score(self, u: ProcNode, exec_ms: float, start_ms: float) -> dict:
        gross_kwh, carbon_g, cost_usd, water_m3 = self.account_energy_carbon_cost(
            u, exec_ms, start_ms
        )

        # Accumulate epoch-level cost
        self.cost_usd += cost_usd

        return {
            "energy_kwh": gross_kwh,   # facility kWh (before PV/battery)
            "carbon_g": carbon_g,      # per-exec grid carbon only
            "cost_usd": cost_usd,      # per-exec energy cost
            "water_m3": water_m3,
        }

    # ---------- epoch lifecycle ----------
    def reset_epoch(self):
        self._busy_ms = 0.0
        self.energy_grid_kwh = 0.0
        self.energy_solar_kwh = 0.0
        self.energy_batt_discharge_kwh = 0.0
        self.energy_batt_charge_kwh = 0.0
        self.embodied_battery_co2_kg = 0.0
        self.energy_other_kwh = 0.0
        self.energy_cooling_kwh = 0.0
        self.energy_it_kwh = 0.0
        self.cost_usd = 0.0
        # water
        self.water_evap_m3 = 0.0
        self.water_blowdown_m3 = 0.0
        self.water_static_m3 = 0.0
        self.water_makeup_m3 = 0.0
        self.water_energy_potable_kwh = 0.0
        self.water_energy_wastewater_kwh = 0.0
        self.water_energy_total_kwh = 0.0
        self.water_carbon_g = 0.0

        self.energy_cost_usd = 0.0
        self.carbon_g = 0.0
        self.water_usage_m3 = 0.0

        for u in self.units:
            try:
                u.busy_ms_epoch = 0.0
            except AttributeError:
                pass
        if self.battery:
            self.battery.reset_epoch()

    def finalize_epoch(self):
        """
        Accrue all idle IT energy, cooling overhead, and facility loads
        for the portion of the epoch in which each unit is not busy.

        Uses:
          - workload-specific CPU base power (SPEC curves)
          - GPU TDP for accelerators
          - idle_frac
          - temperature-scaled ITD multipliers
        """

        epoch_ms = max(0.0, float(self._epoch_len_s) * 1000.0)
        idle_it_kwh = 0.0

        for u in self.units:

            # Skip units that never turned ON and never worked
            if getattr(u, "state", "IDLE") == "OFF" and getattr(u, "busy_ms_epoch", 0.0) <= 0.0:
                continue

            busy = min(epoch_ms, max(0.0, getattr(u, "busy_ms_epoch", 0.0)))
            idle_ms = max(0.0, epoch_ms - busy)

            if idle_ms <= 0.0:
                continue

            # Nodes with no TDP cannot contribute to idle IT energy
            if u.tdp_w <= 0.0:
                continue

            # --- Key fix: workload-aware idle power ---
            # CPU SPEC workloads use per-workload base power
            # GPUs / accelerators fall back to TDP
            base_power_w = u._base_it_power_w()

            # Idle IT power:
            #   base_power * idle_frac * temperature_multiplier
            it_idle_w = base_power_w * u.base_idle_frac * u._it_power_temp_mult()

            # Accumulate idle IT energy (kWh)
            idle_it_kwh += (it_idle_w / 1000.0) * (idle_ms / 3_600_000.0)

        # --- No idle energy? Then nothing more to do this epoch ---
        if idle_it_kwh <= 0.0:
            if self.battery:
                self.embodied_battery_co2_kg = self.battery.embodied_carbon_kg()
            return

        # "Other IT" overhead (fans, PSU losses, control systems)
        other_kwh = idle_it_kwh * max(0.0, self.other_it_overhead_frac)

        # Cooling/Facility overhead
        if self.cooling_mode == "MECH_COP":
            # COP-based (e.g., chilled-water plant)
            cop = max(0.1, self._cop_for_ms(0.0))
            cooling_kwh = idle_it_kwh / cop
        else:
            # PUE-based model
            pue = max(1.0, self._pue_for_ms(0.0))
            non_it_facility_kwh = idle_it_kwh * (pue - 1.0)
            cooling_kwh = max(0.0, non_it_facility_kwh - other_kwh)

        # Total facility idle energy
        gross_idle_kwh = idle_it_kwh + other_kwh + cooling_kwh

        # Accumulate into DC accounting fields
        self.energy_it_kwh += idle_it_kwh
        self.energy_other_kwh += other_kwh
        self.energy_cooling_kwh += cooling_kwh

        # Route idle energy through PV/battery → grid mix
        start_ms, end_ms = 0.0, epoch_ms
        grid_kwh = self._apply_solar_battery_offset(gross_idle_kwh, start_ms, end_ms)
        self.energy_grid_kwh += grid_kwh

        # Battery embodied carbon (per epoch)
        if self.battery:
            self.embodied_battery_co2_kg = self.battery.embodied_carbon_kg()

    # ---------- optional: utilization helper ----------
    def utilization(self, epoch_len_s: float) -> float:
        """
        Returns DC utilization in [0,1] as (busy_ms) / (units * epoch_ms).
        Busy time is incremented per execution in _energy_for_exec_kwh().
        """
        if not self.units:
            return 0.0
        epoch_ms = max(1e-9, float(epoch_len_s) * 1000.0)
        cap_ms = epoch_ms * len(self.units)
        return max(0.0, min(1.0, self._busy_ms / cap_ms))

    def report_utilization(self) -> float:
        return self.utilization(self._epoch_len_s)




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

            tokens = int(getattr(row, "tokens", getattr(row, "tokens", 0)))

            dc = self.datacenters.get(tgt_dc)
            result: Dict[str, Any] = {
                "epoch": int(epoch_idx),
                "request_idx": int(row_idx),
                "source_dc": int(src_dc),
                "target_dc": int(tgt_dc),
                "model": model,
                "arrival_ms": float(arrival_ms),
                "net_latency_ms": float(net_ms),
                "tokens": tokens,
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
                        tokens=tokens,
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

        for dc in self.datacenters.values():
            # 1) Snapshot pre-finalize totals
            pre_grid_kwh = float(getattr(dc, "energy_grid_kwh", 0.0))
            pre_cost_usd = float(getattr(dc, "cost_usd", 0.0))
            pre_emb_kg = float(getattr(dc, "embodied_battery_co2_kg", 0.0))

            # 2) Finalize once per epoch (adds idle IT/other/cooling and battery embodied CO2)
            if hasattr(dc, "finalize_epoch"):
                dc.finalize_epoch()

            # 3) Compute deltas added by finalize
            post_grid_kwh = float(getattr(dc, "energy_grid_kwh", 0.0))
            post_emb_kg = float(getattr(dc, "embodied_battery_co2_kg", 0.0))

            dE_kwh = max(0.0, post_grid_kwh - pre_grid_kwh)
            dEmb_kg = max(0.0, post_emb_kg - pre_emb_kg)

            if dE_kwh <= 0.0 and dEmb_kg <= 0.0:
                # Nothing added by finalize for this DC; skip the synthetic record
                continue

            # 4) Price finalize energy at t=0 (matches finalize_epoch’s ToU choice)
            tou_price = 0.0
            if getattr(dc, "tou_price", None) is not None and hasattr(dc, "_tou_price"):
                try:
                    tou_price = float(dc._tou_price(0.0))
                except Exception:
                    tou_price = 0.0

            dCost_usd = dE_kwh * tou_price

            # 5) Carbon from energy + battery embodied CO₂
            ci_g_per_kwh = float(getattr(dc, "carbon_intensity_g_per_kwh", 0.0))
            dCarbon_g = dE_kwh * ci_g_per_kwh + (dEmb_kg * 1000.0)

            # 6) Commit the finalize cost into the DC’s running cost total
            if hasattr(dc, "cost_usd"):
                dc.cost_usd = float(getattr(dc, "cost_usd", 0.0)) + dCost_usd

            # 7) Append a synthetic detail row so aggregations include finalize additions
            details.append({
                "dc_id": int(getattr(dc, "id", -1)),
                "start_ms": 0.0,
                "end_ms": 0.0,
                "exec_ms": 0.0,
                "ttft_s": 0.0,
                "energy_kwh": dE_kwh,
                "carbon_g": dCarbon_g,
                "cost_usd": dCost_usd,
                "water_m3": 0.0,  # finalize currently doesn’t add water
                "tag": "epoch_finalize_idle",  # helpful for debugging/plots
            })

        # Store last epoch snapshot for reporting
        self._last_epoch_results = details
        self._last_epoch_metrics = self._aggregate_epoch_metrics(tokens, details)
        return details

    # ------------------------------------------------------------------
    # Summarize epoch-level metrics from details
    # ------------------------------------------------------------------
    def _aggregate_epoch_metrics(self, tokens, details: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Combine:
          - Per-request TTFT stats from 'details'
          - Datacenter-level totals (energy, carbon, cost, water)

        'tokens' arg is kept for backward compatibility but is no longer used;
        we compute normalization from the details themselves.
        """

        # -----------------------------
        # 1. TTFT (per-request metric)
        # -----------------------------
        ttft_sum = 0.0
        ttft_cnt = 0
        token_sum = 0

        for r in details:
            v = r.get("ttft_s", r.get("TTFT", r.get("time_to_first_token_s")))
            if v is not None:
                try:
                    ttft_sum += float(v)
                    ttft_cnt += 1
                except Exception:
                    pass

            # Try to accumulate tokens for per-token normalization if present
            t = r.get("tokens", r.get("total_tokens", None))
            if t is not None:
                try:
                    token_sum += max(0, int(t))
                except Exception:
                    pass

        if token_sum > 0:
            avg_ttft = ttft_sum / float(token_sum)
        elif ttft_cnt > 0:
            avg_ttft = ttft_sum / float(ttft_cnt)
        else:
            avg_ttft = 0.0

        # -------------------------------------------------------
        # 2. Pull COMPLETE energy/carbon/water/cost from DCs
        # -------------------------------------------------------
        total_energy_kwh = 0.0
        total_it_energy_kwh = 0.0
        total_cooling_energy_kwh = 0.0
        total_carbon_g = 0.0
        total_water_m3 = 0.0
        total_cost_usd = 0.0

        for dc_id, dc in self.datacenters.items():
            # Energy terms
            grid_kwh = float(getattr(dc, "energy_grid_kwh", 0.0))
            it_kwh = float(getattr(dc, "energy_it_kwh", 0.0))
            cool_kwh = float(getattr(dc, "energy_cooling_kwh", 0.0))

            total_energy_kwh += grid_kwh
            total_it_energy_kwh += it_kwh
            total_cooling_energy_kwh += cool_kwh

            # Water usage (we treat makeup as "usage")
            water_m3 = float(getattr(dc, "water_makeup_m3", 0.0))
            total_water_m3 += water_m3

            # Cost (busy + idle)
            cost_usd = float(getattr(dc, "cost_usd", 0.0))
            total_cost_usd += cost_usd

            # Carbon:
            #  - energy-based: grid_kwh * CI
            #  - plus water_processing carbon tracked separately
            #  - plus battery embodied carbon (kg → g)
            ci = float(getattr(dc, "carbon_intensity_g_per_kwh", 0.0))
            water_carbon = float(getattr(dc, "water_carbon_g", 0.0))
            batt_emb_kg = float(getattr(dc, "embodied_battery_co2_kg", 0.0))

            energy_carbon = grid_kwh * ci
            carbon_total_dc = energy_carbon + water_carbon + batt_emb_kg * 1000.0

            total_carbon_g += carbon_total_dc

            # Backwards-compat aliases on the DC itself
            try:
                dc.energy_cost_usd = cost_usd
                dc.water_usage_m3 = water_m3
                dc.carbon_g = carbon_total_dc
            except Exception:
                pass

        # -------------------------------------------------------
        # 3. Return global metrics
        # -------------------------------------------------------
        return {
            "avg_ttft": float(avg_ttft),
            "energy_cost": float(total_cost_usd),
            "carbon_emissions": float(total_carbon_g),
            "water_usage": float(total_water_m3),
            "total_energy": float(total_energy_kwh),
            "total_it_energy_kwh": float(total_it_energy_kwh),
            "total_cooling_energy_kwh": float(total_cooling_energy_kwh),
        }

    # ------------------------------------------------------------------
    # Public: epoch summary
    # ------------------------------------------------------------------
    def report_global_stats(self) -> Dict[str, Any]:
        """
        Return a full global rollup of:
            - avg_ttft     (from last epoch aggregate)
            - total_energy (kWh)
            - total_it_energy_kwh (kWh)
            - total_cooling_energy_kwh (kWh)
            - energy_cost  ($)
            - carbon_emissions (g)
            - water_usage (m^3)

        Values are taken from _last_epoch_metrics, which is populated by
        _aggregate_epoch_metrics at the end of apply_schedule_plan.
        """

        if not isinstance(self._last_epoch_metrics, dict):
            # Fallback: recompute from scratch if something went wrong
            self._last_epoch_metrics = self._aggregate_epoch_metrics(
                tokens=0,
                details=self._last_epoch_results or [],
            )

        m = self._last_epoch_metrics or {}

        return {
            "avg_ttft": float(m.get("avg_ttft", 0.0)),
            "energy_cost": float(m.get("energy_cost", 0.0)),
            "carbon_emissions": float(m.get("carbon_emissions", 0.0)),
            "water_usage": float(m.get("water_usage", 0.0)),
            "total_energy": float(m.get("total_energy", 0.0)),
            "total_it_energy_kwh": float(m.get("total_it_energy_kwh", 0.0)),
            "total_cooling_energy_kwh": float(m.get("total_cooling_energy_kwh", 0.0)),
        }

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
        self.cpu_csv = os.path.join(spec_dir, "POVRay_CPU.csv")

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
            self.dc_specs_csv, self.node_specs_csv, self.latency_csv, self.a100_csv, self.h100_csv, self.cpu_csv
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
                water_cycling_density_m3_per_kwh_heat=params["water_cycling_density_m3_per_kwh_heat"],
                potable_EI_kWh_per_m3=params["potable_EI_kWh_per_m3"],
                wastewater_EI_kWh_per_m3=params["wastewater_EI_kWh_per_m3"],
                water_static_m3_per_kwh_heat=params["water_static_m3_per_kwh_heat"],
                cooling_mode=params.get("cooling_mode", "MECH_COP"),
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
                workload_class=rec.get("workload_class", "generic"),
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

    def _debug_dump_cooling_params(self):
        print("\n\n=== Cooling/Water params by DC ===")
        print("DC  CI(g/kWh)  water_static(m3/kWh_heat)  water_cycling_density(m3/kWh_heat)  blowdown_ratio  "
              "potable_EI(kWh/m3)  wastewater_EI(kWh/m3)  COP_profile[0..3]/default")

        for dc_id, dc in self.datacenters.items():
            # Pull values with safe fallbacks and ensure they are floats (not lists)
            ci = float(getattr(dc, "carbon_intensity_g_per_kwh", 0.0))
            ws = float(getattr(dc, "water_static", 0.0))
            wcd = float(getattr(dc, "water_cycling_density", 0.0))
            br = float(getattr(dc, "blowdown_ratio", 0.0))
            pei = float(getattr(dc, "potable_energy_intensity", 0.0))
            wei = float(getattr(dc, "wastewater_energy_intensity", 0.0))

            # COP display: sample first 4 hours if profile exists; else show default
            cop_prof = getattr(dc, "cop_profile_24h", None)
            if cop_prof and len(cop_prof) >= 1:
                sample = cop_prof[:4] if len(cop_prof) >= 4 else cop_prof
                cop_str = ",".join(f"{float(x):.2f}" for x in sample)
            else:
                cop_default = float(getattr(dc, "cop_default", 0.0))
                cop_str = f"{cop_default:.2f}"

            line = (f"{dc_id:2d}  {ci:8.1f}    {ws:10.6f}                 {wcd:10.6f}               {br:5.2f}        "
                    f"{pei:6.4f}                {wei:6.4f}        {cop_str}")
            print(line)

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
        metrics.setdefault("total_it_energy_kwh", 0.0)
        metrics.setdefault("total_cooling_energy_kwh", 0.0)

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
                  f"E_IT={metrics['total_it_energy_kwh']:.6f}  "
                  f"E_COOL={metrics['total_cooling_energy_kwh']:.6f}  "
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

def build_gpu_tables_exact(
    a100_csv: str,
    h100_csv: str,
    cpu_csv: str | None = None,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    tables: Dict[str, Dict[str, Dict[str, Any]]] = {
        "A100": load_gpu_table_exact(a100_csv, "A100"),
        "H100": load_gpu_table_exact(h100_csv, "H100"),
    }
    if cpu_csv is not None and os.path.exists(cpu_csv):
        tables["CPU"] = load_gpu_table_exact(cpu_csv, "CPU")
    return tables


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
    cpu_csv: str | None = None,
) -> tuple[
        Dict[int, Dict[str, Any]],
        List[Dict[str, Any]],
        List[List[float]],
        Dict[str, Dict[str, Dict[str, Any]]]
    ]:
    """
    Returns:
      dc_specs : dict[dc_id] -> per-DC parameters (already normalized)
      nodes    : list of node records, each with model_perf+power
      lat_mat  : NxN list of floats (ms)
      gpu_tbls : {'A100': {...}, 'H100': {...}, 'CPU': {...}}
    """

    # ---------------------------
    # Local utility functions
    # ---------------------------
    def _parse_counts_str_to_dict(s: str) -> Dict[int, int]:
        if not s:
            return {}
        parts = [p.strip() for p in s.replace(",", ";").split(";") if p.strip()]
        out: Dict[int, int] = {}
        for piece in parts:
            if ":" not in piece:
                continue
            k, v = piece.split(":", 1)
            try:
                ki = int(k.strip())
                vi = int(float(v.strip()))
                if ki >= 0 and vi >= 0:
                    out[ki] = vi
            except Exception:
                pass
        return out

    def _even_split_counts(total_nodes: int, ntypes: int) -> Dict[int, int]:
        t = max(0, int(total_nodes or 0))
        base, rem = divmod(t, ntypes)
        out = {i: base for i in range(ntypes)}
        for i in range(rem):
            out[i] += 1
        return out

    def _num(x):
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

    # ---------------------------
    # Load CSV-driven structures
    # ---------------------------
    dc_specs  = load_dc_specs_exact(dc_specs_csv)
    templates = load_node_type_templates_exact(node_specs_csv)
    gpu_tbls  = build_gpu_tables_exact(a100_csv, h100_csv, cpu_csv=cpu_csv)
    lat_mat   = load_latency_matrix_exact(latency_csv)

    # ---------------------------
    # Global overrides for node counts
    # ---------------------------
    global MANUAL_NODE_TYPE_COUNTS
    overrides = MANUAL_NODE_TYPE_COUNTS or {}

    ntypes = len(templates)
    nodes: List[Dict[str, Any]] = []

    FORCE_DC_ID = 0

    for dc_id, params in dc_specs.items():

        # When overrides are active, skip all other DCs
        if overrides and dc_id != FORCE_DC_ID:
            continue

        # ---------------------------------------------
        # Load CSV-based counts (existing code)
        # ---------------------------------------------
        counts_str = params.get("node_type_counts_str", "")
        if not counts_str:
            counts_str = params.get("Node_Type_Counts", "") or params.get("node_types", "")

        counts = _parse_counts_str_to_dict(counts_str)
        if not counts:
            total_nodes = params.get("Total_Nodes", params.get("total_nodes", 0))
            try:
                total_nodes = int(total_nodes)
            except Exception:
                total_nodes = 0
            if total_nodes > 0:
                counts = _even_split_counts(total_nodes, ntypes)

        # ---------------------------------------------
        # APPLY OVERRIDES, but only for DC 0
        # ---------------------------------------------
        if overrides and dc_id in overrides:
            # overrides replace counts for those type_ids
            for tid, manual_count in overrides[dc_id].items():
                if 0 <= tid < ntypes:
                    counts[tid] = int(manual_count)

        if not counts:
            continue

        # ---------------------------
        # 3) Expand nodes based on final counts
        # ---------------------------
        next_local_node_id = 0

        for type_id in sorted(counts.keys()):
            num = int(counts[type_id])
            if num <= 0:
                continue

            tmpl = templates[type_id]
            accel = tmpl["accel_type"]          # "A100", "H100", or "CPU"
            cfg   = tmpl["gpu_config"]
            procs = int(tmpl.get("processor_count", 1))

            # Lookup GPU/CPU power tables
            row  = gpu_tbls[accel][cfg]
            meta = row.get("_meta", {})

            tdp_kw  = float(meta.get("tdp_kw", 0.0))
            idle_kw = float(meta.get("idle_kw", 0.15 * tdp_kw))

            # ---------------------------
            # Build performance tables
            # ---------------------------
            pre_sz = _num(row.get("prefill_token_size"))
            gen_sz = _num(row.get("gen_token_size"))
            denom  = (pre_sz or 0) + (gen_sz or 0) or 1.0

            ms7  = _num(row.get("Llama7b_Process"))
            ms70 = _num(row.get("Llama70b_Process"))

            if accel in ("A100", "H100"):
                model_perf = {
                    "Llama7b": {
                        "ms_per_request": ms7,
                        "ms_per_token":   ms7 / denom if ms7 else None,
                    },
                    "Llama70b": {
                        "ms_per_request": ms70,
                        "ms_per_token":   ms70 / denom if ms70 else None,
                    },
                }

            elif accel == "CPU":
                # load SPEC workload ms_per_request values
                model_perf = {}
                for wl in SPEC_CPU_WORKLOADS:
                    val = _num(row.get(wl))
                    if val is not None:
                        model_perf[wl] = {
                            "ms_per_request": val,
                            "ms_per_token": val,
                        }

                # fallback to POVRAY if nothing inserted yet
                if not model_perf:
                    if ms7 is not None:
                        model_perf = {
                            "povray_r": {
                                "ms_per_request": ms7,
                                "ms_per_token": ms7,
                            }
                        }
                    else:
                        model_perf = {}

            else:
                model_perf = {}

            # ---------------------------
            # Instantiate node entries
            # ---------------------------
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
                    "model_perf": model_perf,
                })
                next_local_node_id += 1

    return dc_specs, nodes, lat_mat, gpu_tbls





