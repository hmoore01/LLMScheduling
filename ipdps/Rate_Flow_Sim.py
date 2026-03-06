from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Iterable, Any
from typing import Callable, Union
import math
import collections
import heapq
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

# ---- Debug helpers (near imports) ----
import os
import pandas as pd

# ===========================================================
# === Simulation Constants (hardcoded, no CSV dependency) ===
# ===========================================================

CONSTANTS = {
    "TEMP_REF_C": 20.0,
    "TEMP_SETPOINT_C": 29.0,
    "IT_POWER_TEMP_ALPHA": 0.0,
    "EXEC_MS_TEMP_ALPHA": 0.0,
    "COP_TEMP_ALPHA_PER_C": 0.16,
    "PUE_TEMP_ALPHA_PER_C": -0.04,
    "DEFAULT_COP": 3.0,
    "DEFAULT_PUE": 1.18,
    "OTHER_IT_OVERHEAD_FRAC": 0.13,
    "SOLAR_KW_CAPACITY": 0.0,
    "SOLAR_PROFILE_24H": [0.0, 0.0, 0.0, 0.0, 0.05, 0.15, 0.35, 0.55,
                          0.75, 0.9, 1.0, 0.9, 0.75, 0.55, 0.35, 0.15,
                          0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "BATTERY_CAP_KWH": 0.0,
    "BATTERY_SOC_INIT": 0.0,
    "BATTERY_MAX_CHARGE_KW": 0.0,
    "BATTERY_MAX_DISCHARGE_KW": 0.0,
    "BATTERY_ROUNDTRIP_EFF": 0.92,
    "BATTERY_EMBODIED_CO2_PER_KWH": 0.05,
    "TOU_PRICE_24H": [0.12, 0.12, 0.12, 0.12, 0.14, 0.15, 0.18, 0.20,
                      0.22, 0.25, 0.25, 0.23, 0.22, 0.20, 0.18, 0.16,
                      0.14, 0.13, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12],
}

SPEC_CPU_WORKLOADS = [
    "bwaves_r", "namd_r", "povray_r", "cactusBSSN_r", "parest_r",
    "fotonik3d_r", "perlbench_r", "gcc_r", "mcf_r", "deepsjeng_r", "exchange2_r",
]

SPECINT_POWER_DATA = {
    "perlbench_r": {"Temp": [30, 36, 39, 45, 48], "Power": [260, 256, 258, 260, 261]},
    "gcc_r": {"Temp": [28, 35, 38, 49, 58], "Power": [256, 253, 254, 256, 260]},
    "mcf_r": {"Temp": [28, 36, 39, 50, 60], "Power": [275, 270, 271, 274, 278]},
    "omnetpp_r": {"Temp": [28, 34, 37, 49, 59], "Power": [246, 241, 241, 246, 251]},
    "xalancbmk_r": {"Temp": [31, 37, 40, 50, 60], "Power": [289, 285, 282, 286, 291]},
    "x264_r": {"Temp": [30, 35, 39, 49, 58], "Power": [258, 255, 255, 258, 262]},
    "deepsjeng_r": {"Temp": [30, 35, 39, 50, 59], "Power": [248, 247, 248, 248, 253]},
    "leela_r": {"Temp": [29, 34, 38, 49, 57], "Power": [232, 231, 231, 232, 236]},
    "exchange2_r": {"Temp": [29, 34, 38, 49, 58], "Power": [238, 237, 237, 238, 242]},
    "xz_r": {"Temp": [28, 33, 37, 48, 57], "Power": [229, 228, 228, 230, 234]},
}

SPECFP_POWER_DATA = {
    "bwaves_r": {"Temp": [32, 38, 44, 50, 60], "Power": [310, 307, 307, 308, 312]},
    "cactuBSSN_r": {"Temp": [30, 36, 42, 50, 60], "Power": [281, 278, 279, 281, 286]},
    "namd_r": {"Temp": [30, 36, 42, 50, 60], "Power": [273, 271, 270, 271, 277]},
    "parest_r": {"Temp": [28, 34, 40, 49, 59], "Power": [256, 253, 254, 257, 262]},
    "povray_r": {"Temp": [32, 38, 44, 50, 60], "Power": [293, 291, 290, 291, 295]},
    "lbm_r": {"Temp": [27, 33, 38, 49, 56], "Power": [239, 236, 237, 240, 244]},
    "wrf_r": {"Temp": [28, 33, 40, 50, 59], "Power": [252, 250, 250, 252, 258]},
    "blender_r": {"Temp": [30, 35, 42, 50, 60], "Power": [273, 270, 269, 271, 276]},
    "cam4_r": {"Temp": [30, 36, 43, 50, 60], "Power": [296, 293, 293, 294, 299]},
    "imagick_r": {"Temp": [29, 35, 42, 50, 60], "Power": [258, 256, 255, 256, 261]},
    "nab_r": {"Temp": [29, 35, 42, 50, 60], "Power": [281, 278, 278, 279, 283]},
    "fotonik3d_r": {"Temp": [26, 31, 38, 48, 56], "Power": [228, 226, 226, 228, 232]},
    "roms_r": {"Temp": [27, 32, 40, 49, 58], "Power": [246, 244, 244, 247, 252]},
}

SPEC_CPU_POWER_TABLES = {**SPECINT_POWER_DATA, **SPECFP_POWER_DATA}


def interp_piecewise_linear(x, xs, ys):
    if x <= xs[0]: return ys[0]
    if x >= xs[-1]: return ys[-1]
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            x0, x1 = xs[i], xs[i + 1]
            y0, y1 = ys[i], ys[i + 1]
            t = (x - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return ys[-1]


MANUAL_NODE_TYPE_COUNTS: Optional[Dict[int, Dict[int, int]]] = None


def set_manual_node_type_counts(counts_json: Optional[str] = None, num_dcs: int = 12) -> None:
    global MANUAL_NODE_TYPE_COUNTS
    if counts_json is None:
        MANUAL_NODE_TYPE_COUNTS = None
        return
    import json
    try:
        raw_counts = {int(k): int(v) for k, v in json.loads(counts_json).items()}
    except Exception as e:
        print(f"[WARNING] Failed to parse node type counts: {e}")
        MANUAL_NODE_TYPE_COUNTS = None
        return
    MANUAL_NODE_TYPE_COUNTS = {dc_id: dict(raw_counts) for dc_id in range(num_dcs)}


def _safe_float(x):
    if x in (None, "", "None", "NA", "NaN"): return None
    try:
        return float(x)
    except Exception:
        return None


@dataclass
class Battery:
    cap_kwh: float = 0.0
    soc_kwh: float = 0.0
    max_charge_kw: float = 0.0
    max_discharge_kw: float = 0.0
    roundtrip_eff: float = 0.92
    embodied_co2_per_kwh_throughput: float = 0.0

    charged_kwh: float = 0.0
    discharged_kwh: float = 0.0

    def reset_epoch(self):
        self.charged_kwh = 0.0
        self.discharged_kwh = 0.0

    def can_use(self) -> bool:
        return (
                    self.cap_kwh > 0.0 and self.max_charge_kw > 0.0 and self.max_discharge_kw > 0.0 and self.roundtrip_eff > 0.0)

    def charge(self, want_kwh: float, hours: float) -> float:
        if not self.can_use() or want_kwh <= 0.0: return 0.0
        hours = max(hours, 1e-9)
        limit_kwh = self.max_charge_kw * hours
        headroom_in_kwh = max(0.0, self.cap_kwh - self.soc_kwh) / self.roundtrip_eff
        in_kwh = min(want_kwh, limit_kwh, headroom_in_kwh)
        stored = in_kwh * self.roundtrip_eff
        if stored <= 0.0: return 0.0
        self.soc_kwh += stored
        self.charged_kwh += in_kwh
        return stored

    def discharge(self, want_kwh: float, hours: float) -> float:
        if not self.can_use() or want_kwh <= 0.0: return 0.0
        hours = max(hours, 1e-9)
        limit_kwh = self.max_discharge_kw * hours
        deliverable = min(want_kwh, limit_kwh, self.soc_kwh)
        if deliverable <= 0.0: return 0.0
        self.soc_kwh -= deliverable
        self.discharged_kwh += deliverable
        return deliverable

    def embodied_carbon_kg(self) -> float:
        throughput = self.charged_kwh + self.discharged_kwh
        return throughput * max(0.0, self.embodied_co2_per_kwh_throughput)


@dataclass
class ProcNode:
    def __init__(self, node_id: int, model_perf: dict, *, type_id: int | None = None, accel_type: str | None = None,
                 gpu_config: str | None = None, tdp_kw: float | None = None, idle_kw: float | None = None,
                 tdp_w: float | None = None, idle_w: float | None = None, base_idle_frac: float | None = None,
                 workload_class: str = "generic"):
        self.node_id = int(node_id)
        self.type_id = type_id
        self.accel_type = accel_type
        self.gpu_config = gpu_config

        tdp_w_norm = float(tdp_w) if tdp_w is not None else (float(tdp_kw) * 1000.0 if tdp_kw is not None else 0.0)
        idle_w_norm = float(idle_w) if idle_w is not None else (
            float(idle_kw) * 1000.0 if idle_kw is not None else None)

        self.tdp_w = tdp_w_norm
        if base_idle_frac is not None:
            self.base_idle_frac = float(base_idle_frac)
        elif idle_w_norm is not None and self.tdp_w > 0.0:
            self.base_idle_frac = max(0.0, min(1.0, idle_w_norm / self.tdp_w))
        else:
            self.base_idle_frac = 0.13

        self.model_perf = dict(model_perf)
        self.state = "IDLE"
        self.dc_ref = None
        self.busy_ms_epoch = 0.0
        self.workload_class = str(workload_class).lower()
        self.next_available_ms = 0.0

    def attach_dc(self, dc):
        self.dc_ref = dc

    def it_frac_for_state(self) -> float:
        s = self.state.upper() if self.state else "IDLE"
        if s == "OFF": return 0.0
        if s == "IDLE": return self.base_idle_frac
        return 1.0

    def _base_it_power_w(self) -> float:
        base = float(self.tdp_w or 0.0)
        if self.accel_type not in ("CPU", "cpu"): return base
        wc = (self.workload_class or "").lower()
        workload_key = None
        for key in SPEC_CPU_POWER_TABLES.keys():
            if key.replace("_r", "") in wc:
                workload_key = key
                break
        if workload_key:
            ptab = SPEC_CPU_POWER_TABLES[workload_key]
            powers = ptab["Power"]
            mid = powers[len(powers) // 2]
            base = float(mid)
        return base

    def _it_power_temp_mult(self) -> float:
        dc = self.dc_ref
        if not dc: return 1.0
        temp_c = float(dc.temp_c_setpoint)
        if self.accel_type not in ("CPU", "cpu"):
            dT = temp_c - dc.temp_ref_c
            return max(0.0, 1.0 + dc.it_power_temp_alpha * dT)

        wc = (self.workload_class or "").lower()
        workload_key = None
        for key in SPEC_CPU_POWER_TABLES.keys():
            if key.replace("_r", "") in wc:
                workload_key = key
                break

        if workload_key:
            tab = SPEC_CPU_POWER_TABLES[workload_key]
            xs, ys = tab["Temp"], tab["Power"]
            raw = interp_piecewise_linear(temp_c, xs, ys)
            idx = min(range(len(xs)), key=lambda i: abs(xs[i] - 40))
            ref = ys[idx]
            return raw / ref

        def _generic_cpu_mult(T: float) -> float:
            T_opt = 35.0
            d = (T - T_opt) / 25.0
            m = 1.0 + 0.04 * (d * d) + 0.08 * (d ** 4)
            return max(0.85, min(1.20, m))

        return _generic_cpu_mult(temp_c)

    def _exec_ms_temp_mult(self) -> float:
        dc = self.dc_ref
        if not dc: return 1.0
        dT = float(dc.temp_c_setpoint) - dc.temp_ref_c
        return max(0.0, 1.0 + dc.exec_ms_temp_alpha * dT)

    def estimate_exec_ms(self, tokens, model: str, kwargs) -> float:
        if not hasattr(self, "_model_cache"):
            self._model_cache = {}

        if model not in self._model_cache:
            rec = self.model_perf.get(model)
            if not rec:
                if model in ["Llama7b", "Llama70b"]:
                    for k in self.model_perf:
                        if model in k and "FP16" in k and "B1" in k:
                            rec = self.model_perf[k]
                            break
                if not rec:
                    for k, v in self.model_perf.items():
                        if k in model or model in k:
                            rec = v
                            break
            self._model_cache[model] = rec

        rec = self._model_cache[model]
        if not rec: return 0.0

        if "ms_per_token" in rec and rec["ms_per_token"] > 0:
            val = float(rec["ms_per_token"]) * float(tokens)
            return val * self._exec_ms_temp_mult()
        if "ms_per_request" in rec:
            return float(rec["ms_per_request"]) * self._exec_ms_temp_mult()
        return 0.0

    def it_energy_kwh_for_exec(self, exec_ms: float) -> float:
        if self.state and self.state.upper() == "OFF": return 0.0
        if self.tdp_w <= 0.0: return 0.0
        power_w = self._base_it_power_w() * 1.0 * self._it_power_temp_mult()
        hours = max(0.0, float(exec_ms)) / 3_600_000.0
        self.busy_ms_epoch += float(exec_ms)
        return power_w * hours / 1000.0


@dataclass
class Datacenter:
    def __init__(self, dc_id: int, carbon_intensity_g_per_kwh: float, time_of_use_24h: list[float] | None = None,
                 cop_profile_24h: list[float] | None = None, blowdown_ratio: float | None = None,
                 water_cycling_density_m3_per_kwh_heat: float | None = None, potable_EI_kWh_per_m3: float | None = None,
                 wastewater_EI_kWh_per_m3: float | None = None, water_static_m3_per_kwh_heat: float | None = None,
                 cooling_mode: str = "MECH_COP", epoch_length: int | None = None, debug: bool = True):
        self.id = int(dc_id)
        self.carbon_intensity_g_per_kwh = float(carbon_intensity_g_per_kwh)
        self.cooling_mode = str(cooling_mode)
        self.debug = bool(debug)

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

        self.blowdown_ratio = float(blowdown_ratio) if blowdown_ratio is not None else 0.30
        self.water_cycling_density = float(
            water_cycling_density_m3_per_kwh_heat) if water_cycling_density_m3_per_kwh_heat is not None else 0.10
        self.potable_energy_intensity = float(potable_EI_kWh_per_m3) if potable_EI_kWh_per_m3 is not None else 0.005
        self.wastewater_energy_intensity = float(
            wastewater_EI_kWh_per_m3) if wastewater_EI_kWh_per_m3 is not None else 0.010
        self.water_static = float(water_static_m3_per_kwh_heat) if water_static_m3_per_kwh_heat is not None else 5.0

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
        self.units: list[ProcNode] = []

        self._epoch_len_s = float(epoch_length) if epoch_length else 900.0
        self.reset_epoch()  # Initialize counters

    def add_node(self, unit: ProcNode):
        unit.attach_dc(self)
        self.units.append(unit)

    def apply_power_plan(self, plan_slice: dict | None):
        if not plan_slice: return

        mode_all = plan_slice.get("all")
        if mode_all and str(mode_all).upper() in ("ON", "IDLE", "OFF"):
            target_state = str(mode_all).upper()
            for u in self.units: u.state = target_state
            return

        unit_modes = plan_slice.get("unit") or {}
        if not unit_modes: return

        keys = list(unit_modes.keys())
        max_key = max(int(k) for k in keys) if keys else 0

        if max_key <= 6 and len(keys) <= 7:
            for u in self.units:
                type_id = getattr(u, "type_id", None)
                target_state = None
                if type_id is not None and type_id in unit_modes:
                    target_state = unit_modes[type_id]
                elif type_id is not None and str(type_id) in unit_modes:
                    target_state = unit_modes[str(type_id)]
                if target_state and str(target_state).upper() in ("ON", "IDLE", "OFF"):
                    u.state = str(target_state).upper()
        else:
            id_map = {u.node_id: u for u in self.units}
            for node_id, state in unit_modes.items():
                uid = int(node_id)
                if uid in id_map and state and str(state).upper() in ("ON", "IDLE", "OFF"):
                    id_map[uid].state = str(state).upper()

    def schedule_request(self, model, arrival, net_latency_ms=0.0, tokens=None, **kwargs):
        if not hasattr(self, "_node_heap"):
            self._node_heap = []
            self._heap_valid = False
            self._model_cache = {}

        if not self._heap_valid:
            self._node_heap = [(u.next_available_ms, u.node_id, u) for u in self.units if u.state != "OFF"]
            heapq.heapify(self._node_heap)
            self._heap_valid = True

        if not self._node_heap:
            return {
                "dc_id": self.id, "dropped": True, "ttft_s": 0.0,
                "energy_kwh": 0.0, "carbon_g": 0.0, "cost_usd": 0.0, "water_m3": 0.0
            }

        avail_ms, node_id, unit = heapq.heappop(self._node_heap)

        if model not in self._model_cache:
            rec = unit.model_perf.get(model)
            if not rec:
                for k, v in unit.model_perf.items():
                    if k in model or model in k:
                        rec = v
                        break
            self._model_cache[model] = rec

        exec_ms = float(unit.estimate_exec_ms(tokens, model, kwargs))
        rec = getattr(unit, "_model_cache", {}).get(model)
        if rec is None:
            rec = self._model_cache.get(model)
        start_ms = max(float(arrival), unit.next_available_ms)
        wait_ms = start_ms - float(arrival)

        if wait_ms > 300000.0:
            heapq.heappush(self._node_heap, (unit.next_available_ms, unit.node_id, unit))
            return {
                "dc_id": self.id, "dropped": True, "ttft_s": 0.0,
                "energy_kwh": 0.0, "carbon_g": 0.0, "cost_usd": 0.0, "water_m3": 0.0
            }

        unit.next_available_ms = start_ms + exec_ms
        end_ms = start_ms + exec_ms
        heapq.heappush(self._node_heap, (unit.next_available_ms, unit.node_id, unit))

        score = self.settle_and_score(unit, exec_ms, start_ms)

        prefill_ms = 0.0
        if rec:
            if "ms_per_request" in rec:
                prefill_ms = float(rec["ms_per_request"]) * unit._exec_ms_temp_mult()
            elif "ms_per_token" in rec:
                prefill_ms = float(rec["ms_per_token"]) * float(tokens or 1) * unit._exec_ms_temp_mult()

        ttft_s = (float(net_latency_ms) + wait_ms + prefill_ms) / 1000.0

        return {
            "dc_id": int(self.id),
            "start_ms": start_ms, "end_ms": end_ms, "exec_ms": exec_ms,
            "ttft_s": ttft_s, "dropped": False,
            "energy_kwh": float(score.get("energy_kwh", 0.0)),
            "carbon_g": float(score.get("carbon_g", 0.0)),
            "cost_usd": float(score.get("cost_usd", 0.0)),
            "water_m3": float(score.get("water_m3", 0.0)),
        }

    def _hour_of_day(self, ms: float) -> int:
        sec = (float(ms) / 1000.0) % 86400.0
        return int(sec // 3600)

    def _tou_price(self, ms: float) -> float:
        if not self.tou_price: return 0.0
        h = self._hour_of_day(ms)
        return float(self.tou_price[h % len(self.tou_price)])

    def _cop_for_ms(self, ms: float) -> float:
        if self.cop_profile_24h and len(self.cop_profile_24h) >= 24:
            base = float(self.cop_profile_24h[self._hour_of_day(ms) % 24])
        else:
            base = self.cop_default
        dT = float(self.temp_c_setpoint) - self.temp_ref_c
        return max(1.0, base * max(0.0, 1.0 + self.cop_temp_alpha_per_C * dT))

    def _pue_for_ms(self, ms: float) -> float:
        dT = float(self.temp_c_setpoint) - self.temp_ref_c
        return max(1.0, self.pue_value + self.pue_temp_alpha_per_C * dT)

    def _solar_kw_at_ms(self, ms: float) -> float:
        if self.solar_kw_capacity <= 0.0 or not self.solar_profile_24h: return 0.0
        h = self._hour_of_day(ms)
        frac = float(self.solar_profile_24h[h % len(self.solar_profile_24h)])
        return max(0.0, self.solar_kw_capacity * frac)

    def _energy_for_exec_kwh(self, u: ProcNode, exec_ms: float, start_ms: float) -> float:
        it_kwh = u.it_energy_kwh_for_exec(exec_ms)
        other_kwh = it_kwh * max(0.0, self.other_it_overhead_frac)

        if self.cooling_mode == "MECH_COP":
            cop = max(0.1, self._cop_for_ms(start_ms))
            cooling_kwh = it_kwh / cop
            infra_kwh = it_kwh + other_kwh + cooling_kwh
        else:
            pue = max(1.0, self._pue_for_ms(start_ms))
            non_it_facility_kwh = it_kwh * (pue - 1.0)
            cooling_kwh = max(0.0, non_it_facility_kwh - other_kwh)
            infra_kwh = it_kwh + other_kwh + cooling_kwh

        self.energy_it_kwh += it_kwh
        self.energy_other_kwh += other_kwh
        self.energy_cooling_kwh += cooling_kwh
        self._busy_ms += max(0.0, float(exec_ms))
        return infra_kwh

    def _apply_solar_battery_offset(self, gross_kwh: float, start_ms: float, end_ms: float) -> float:
        hours = max(1e-9, (end_ms - start_ms) / 3_600_000.0)
        pv_kwh = self._solar_kw_at_ms(start_ms) * hours

        pv_to_load = min(pv_kwh, gross_kwh)
        self.energy_solar_kwh += pv_to_load
        remaining = gross_kwh - pv_to_load

        batt_deliver = self.battery.discharge(remaining, hours) if self.battery else 0.0
        remaining -= batt_deliver
        self.energy_batt_discharge_kwh += batt_deliver

        surplus = max(0.0, pv_kwh - pv_to_load)
        if surplus > 0.0 and self.battery:
            stored = self.battery.charge(surplus, hours)
            if stored > 0.0: self.energy_batt_charge_kwh += stored

        return max(0.0, remaining)

    def _account_water_from_it(self, it_kwh: float, start_ms: float, cop: float) -> float:
        heat_rej_kwh = it_kwh + (it_kwh / max(1e-9, cop))
        static_m3 = float(self.water_static) * heat_rej_kwh
        evap_m3 = float(self.water_cycling_density) * heat_rej_kwh

        ratio = max(1e-9, float(self.blowdown_ratio))
        total_draw_m3 = evap_m3 / ratio
        blowdown_m3 = max(0.0, total_draw_m3 - evap_m3)
        makeup_m3 = static_m3 + total_draw_m3

        potable_kwh = float(self.potable_energy_intensity) * (evap_m3 + static_m3)
        wastewater_kwh = float(self.wastewater_energy_intensity) * blowdown_m3
        total_water_kwh = potable_kwh + wastewater_kwh

        water_co2_g = total_water_kwh * float(self.carbon_intensity_g_per_kwh)

        self.water_static_m3 += static_m3
        self.water_evap_m3 += evap_m3
        self.water_blowdown_m3 += blowdown_m3
        self.water_makeup_m3 += makeup_m3
        self.water_energy_potable_kwh += potable_kwh
        self.water_energy_wastewater_kwh += wastewater_kwh
        self.water_energy_total_kwh += total_water_kwh
        self.water_carbon_g += water_co2_g
        return makeup_m3

    def account_energy_carbon_cost(self, u: ProcNode, exec_ms: float, start_ms: float) -> tuple[
        float, float, float, float]:
        pre_it_kwh = self.energy_it_kwh
        gross_kwh = self._energy_for_exec_kwh(u, exec_ms, start_ms)
        end_ms = start_ms + exec_ms
        it_kwh_this = self.energy_it_kwh - pre_it_kwh

        grid_kwh = self._apply_solar_battery_offset(gross_kwh, start_ms, end_ms)
        self.energy_grid_kwh += grid_kwh

        water_m3 = 0.0
        if self.cooling_mode == "MECH_COP":
            cop = max(0.1, self._cop_for_ms(start_ms))
            water_m3 = self._account_water_from_it(it_kwh_this, start_ms, cop)

        carbon_g = grid_kwh * float(self.carbon_intensity_g_per_kwh)
        price = self._tou_price(start_ms) if self.tou_price else 0.0
        cost_usd = price * grid_kwh

        return gross_kwh, carbon_g, cost_usd, water_m3

    def settle_and_score(self, u: ProcNode, exec_ms: float, start_ms: float) -> dict:
        gross_kwh, carbon_g, cost_usd, water_m3 = self.account_energy_carbon_cost(u, exec_ms, start_ms)
        self.cost_usd += cost_usd
        return {"energy_kwh": gross_kwh, "carbon_g": carbon_g, "cost_usd": cost_usd, "water_m3": water_m3}

    def reset_epoch(self):
        self._heap_valid = False
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
        self.water_evap_m3 = 0.0
        self.water_blowdown_m3 = 0.0
        self.water_static_m3 = 0.0
        self.water_makeup_m3 = 0.0
        self.water_energy_potable_kwh = 0.0
        self.water_energy_wastewater_kwh = 0.0
        self.water_energy_total_kwh = 0.0
        self.water_carbon_g = 0.0

        for u in self.units:
            try:
                u.busy_ms_epoch = 0.0
                u.next_available_ms = 0.0
            except AttributeError:
                pass
        if self.battery: self.battery.reset_epoch()

    def finalize_epoch(self, epoch_start_ms: float = 0.0):
        start_ms = float(epoch_start_ms)
        epoch_ms = max(0.0, float(self._epoch_len_s) * 1000.0)
        idle_it_kwh = 0.0

        for u in self.units:
            state_upper = u.state.upper() if u.state else "IDLE"
            if state_upper == "OFF" and getattr(u, "busy_ms_epoch", 0.0) <= 0.0: continue
            busy = min(epoch_ms, max(0.0, getattr(u, "busy_ms_epoch", 0.0)))
            idle_ms = max(0.0, epoch_ms - busy)
            if idle_ms <= 0.0 or u.tdp_w <= 0.0: continue

            base_power_w = u._base_it_power_w()
            it_idle_w = base_power_w * u.base_idle_frac * u._it_power_temp_mult()
            if state_upper == "OFF": it_idle_w = 0.0
            idle_it_kwh += (it_idle_w / 1000.0) * (idle_ms / 3_600_000.0)

        if idle_it_kwh <= 0.0:
            if self.battery: self.embodied_battery_co2_kg = self.battery.embodied_carbon_kg()
            return

        other_kwh = idle_it_kwh * max(0.0, self.other_it_overhead_frac)
        if self.cooling_mode == "MECH_COP":
            cop = max(0.1, self._cop_for_ms(start_ms))
            cooling_kwh = idle_it_kwh / cop
            self._account_water_from_it(idle_it_kwh, start_ms, cop)
        else:
            pue = max(1.0, self._pue_for_ms(start_ms))
            non_it_facility_kwh = idle_it_kwh * (pue - 1.0)
            cooling_kwh = max(0.0, non_it_facility_kwh - other_kwh)

        gross_idle_kwh = idle_it_kwh + other_kwh + cooling_kwh
        self.energy_it_kwh += idle_it_kwh
        self.energy_other_kwh += other_kwh
        self.energy_cooling_kwh += cooling_kwh

        grid_kwh = self._apply_solar_battery_offset(gross_idle_kwh, start_ms, start_ms + epoch_ms)
        self.energy_grid_kwh += grid_kwh

        if self.battery: self.embodied_battery_co2_kg = self.battery.embodied_carbon_kg()

    def report_utilization(self):
        if not self.units: return 0.0
        epoch_ms = self._epoch_len_s * 1000.0
        total_busy = sum(u.busy_ms_epoch for u in self.units)
        capacity = len(self.units) * epoch_ms
        return total_busy / max(1.0, capacity)


# -----------------------------
# GeoNetwork (global routing)
# -----------------------------

class Geo_Network:
    def __init__(self, datacenters: Dict[int, "Datacenter"], latency_matrix: List[List[float]], debug: bool = True,
                 parallel_dc_workers: int = 1):
        self.debug = debug
        self.parallel_dc_workers = max(1, int(parallel_dc_workers or 1))
        self.datacenters: Dict[int, "Datacenter"] = dict(sorted(datacenters.items()))
        self.lat = latency_matrix
        self.dc_ids: List[int] = list(self.datacenters.keys())
        self.num_dc = len(self.dc_ids)

        self._shortest_path_cache = {}
        for src in self.dc_ids:
            self._shortest_path_cache[src] = {}
            for dst in self.dc_ids:
                self._shortest_path_cache[src][dst] = self._calculate_ring_path(src, dst)

        self._last_epoch_results: List[Dict[str, Any]] = []
        self._last_epoch_metrics: Dict[str, Any] = {}

    def _calculate_ring_path(self, src_dc: int, dst_dc: int) -> float:
        if src_dc == dst_dc or self.num_dc <= 1: return 0.0
        pos = {dc_id: i for i, dc_id in enumerate(self.dc_ids)}
        i_src, i_dst = pos[src_dc], pos[dst_dc]

        cw_ms = 0.0
        i = i_src
        while i != i_dst:
            u, v = self.dc_ids[i], self.dc_ids[(i + 1) % self.num_dc]
            cw_ms += float(self.lat[u][v])
            i = (i + 1) % self.num_dc

        ccw_ms = 0.0
        i = i_src
        while i != i_dst:
            j = (i - 1 + self.num_dc) % self.num_dc
            u, v = self.dc_ids[j], self.dc_ids[i]
            ccw_ms += float(self.lat[u][v])
            i = j
        return min(cw_ms, ccw_ms)

    def _ring_path_latency_ms(self, src_dc: int, dst_dc: int) -> float:
        return self._shortest_path_cache[src_dc][dst_dc]

    def _apply_power_plan(self, power_plan: Dict[str, Any] | None):
        if not power_plan: return
        for dc_id, dc in self.datacenters.items():
            plan_slice = power_plan.get(dc_id) if isinstance(power_plan, dict) else None
            if hasattr(dc, "apply_power_plan") and callable(getattr(dc, "apply_power_plan")):
                try:
                    dc.apply_power_plan(plan_slice)
                except Exception:
                    pass

    def _process_requests_for_dc(self, dc: "Datacenter", tasks: List[Tuple[Any, ...]], epoch_idx: int,
                                 epoch_start_ms: float) -> List[Tuple[int, Dict[str, Any]]]:
        out: List[Tuple[int, Dict[str, Any]]] = []
        for row_idx, src_dc, tgt_dc, model, arrival_rel_ms, arrival_abs_ms, net_ms, tokens in tasks:
            result = {
                "epoch": int(epoch_idx), "request_idx": int(row_idx),
                "source_dc": int(src_dc), "target_dc": int(tgt_dc), "model": str(model),
                "arrival_ms": float(arrival_rel_ms), "net_latency_ms": float(net_ms), "tokens": int(tokens),
            }
            try:
                dc_ret = dc.schedule_request(model=model, arrival=arrival_abs_ms, net_latency_ms=net_ms, tokens=tokens)
                if isinstance(dc_ret, dict):
                    dc_ret = dict(dc_ret)
                    if "start_ms" in dc_ret:
                        dc_ret["start_ms"] = float(dc_ret["start_ms"]) - epoch_start_ms
                    if "end_ms" in dc_ret:
                        dc_ret["end_ms"] = float(dc_ret["end_ms"]) - epoch_start_ms
                    result.update(dc_ret)
            except Exception:
                result.update({
                    "ttft_s": float(net_ms) / 1000.0, "dropped": True,
                    "energy_cost": 0.0, "cost_usd": 0.0, "energy_kwh": 0.0, "carbon_g": 0.0, "water_m3": 0.0,
                })
            out.append((int(row_idx), result))
        return out

    def apply_schedule_plan(self, epoch_idx: int, workload_df, schedule_plan: Dict[str, Any],
                            power_plan: Dict[str, Any] | None) -> List[Dict[str, Any]]:
        self._apply_power_plan(power_plan)
        details: List[Optional[Dict[str, Any]]] = []
        if workload_df is None or len(workload_df) == 0:
            details = []
            total_tokens_for_agg = 0
        else:
            mp = schedule_plan.get("map", {}) if isinstance(schedule_plan, dict) else {}
            rt = schedule_plan.get("route", {}) if isinstance(schedule_plan, dict) else {}
            if isinstance(schedule_plan, dict):
                default_raw = schedule_plan.get("default_target_dc", -1)
                try:
                    default_dc = int(default_raw)
                except Exception:
                    default_dc = -1
            else:
                default_dc = -1

            src_series = workload_df.get("source_dc_id")
            if src_series is None: src_series = workload_df.get("source_dc")
            if src_series is None: src_series = workload_df.get("src_dc")
            if src_series is None:
                raise KeyError("workload_df missing source DC column (source_dc_id/source_dc/src_dc)")
            model_series = workload_df.get("model_type")
            if model_series is None: model_series = workload_df.get("model")
            if model_series is None:
                raise KeyError("workload_df missing model column (model_type/model)")

            src_dcs = src_series.to_numpy(copy=False)
            models = model_series.to_numpy(copy=False)
            n_rows = len(src_dcs)

            arrivals_col = workload_df.get("arrival_ms", workload_df.get("arrival"))
            arrivals = arrivals_col.to_numpy(copy=False) if arrivals_col is not None else [0.0] * n_rows
            tokens_col = workload_df.get("num_tokens", workload_df.get("tokens"))
            tokens_arr = tokens_col.to_numpy(copy=False) if tokens_col is not None else [0] * n_rows

            epoch_len_s = 900.0
            if self.datacenters:
                try:
                    epoch_len_s = float(next(iter(self.datacenters.values()))._epoch_len_s)
                except Exception:
                    epoch_len_s = 900.0
            epoch_start_ms = float(epoch_idx) * epoch_len_s * 1000.0

            details = [None] * n_rows
            tasks_by_dc: Dict[int, List[Tuple[Any, ...]]] = collections.defaultdict(list)
            total_tokens_for_agg = 0

            for row_idx in range(n_rows):
                src_dc = int(src_dcs[row_idx])
                model = str(models[row_idx])
                arrival_rel_ms = float(arrivals[row_idx])
                arrival_abs_ms = epoch_start_ms + arrival_rel_ms
                tokens = int(tokens_arr[row_idx])
                total_tokens_for_agg += tokens

                tgt_dc = mp.get(row_idx)
                if tgt_dc is None:
                    tgt_dc = mp.get(str(row_idx))
                if tgt_dc is None:
                    tgt_dc = rt.get(model, default_dc if default_dc != -1 else src_dc)
                tgt_dc = int(tgt_dc)

                try:
                    net_ms = float(self._ring_path_latency_ms(src_dc, tgt_dc))
                except Exception:
                    net_ms = 0.0
                dc = self.datacenters.get(tgt_dc)
                if dc is None:
                    details[row_idx] = {
                        "epoch": int(epoch_idx), "request_idx": row_idx,
                        "source_dc": src_dc, "target_dc": tgt_dc, "model": model,
                        "arrival_ms": arrival_rel_ms, "net_latency_ms": net_ms, "tokens": tokens,
                        "ttft_s": net_ms / 1000.0, "dropped": True,
                        "energy_cost": 0.0, "cost_usd": 0.0, "energy_kwh": 0.0, "carbon_g": 0.0, "water_m3": 0.0,
                    }
                else:
                    tasks_by_dc[tgt_dc].append(
                        (row_idx, src_dc, tgt_dc, model, arrival_rel_ms, arrival_abs_ms, net_ms, tokens))

            if tasks_by_dc:
                if self.parallel_dc_workers > 1 and len(tasks_by_dc) > 1:
                    max_workers = min(self.parallel_dc_workers, len(tasks_by_dc))
                    with ThreadPoolExecutor(max_workers=max_workers) as ex:
                        future_map = {
                            ex.submit(self._process_requests_for_dc, self.datacenters[dc_id], dc_tasks, epoch_idx,
                                      epoch_start_ms): (dc_id, dc_tasks)
                            for dc_id, dc_tasks in tasks_by_dc.items()
                        }
                        for fut in as_completed(future_map):
                            dc_id, dc_tasks = future_map[fut]
                            try:
                                updates = fut.result()
                            except Exception:
                                updates = self._process_requests_for_dc(self.datacenters[dc_id], dc_tasks, epoch_idx,
                                                                        epoch_start_ms)
                            for row_idx, result in updates:
                                details[row_idx] = result
                else:
                    for dc_id, dc_tasks in tasks_by_dc.items():
                        updates = self._process_requests_for_dc(self.datacenters[dc_id], dc_tasks, epoch_idx,
                                                                epoch_start_ms)
                        for row_idx, result in updates:
                            details[row_idx] = result

            for idx, rec in enumerate(details):
                if rec is None:
                    details[idx] = {
                        "epoch": int(epoch_idx), "request_idx": idx,
                        "source_dc": int(src_dcs[idx]), "target_dc": int(src_dcs[idx]), "model": str(models[idx]),
                        "arrival_ms": float(arrivals[idx]), "net_latency_ms": 0.0, "tokens": int(tokens_arr[idx]),
                        "ttft_s": 0.0, "dropped": True,
                        "energy_cost": 0.0, "cost_usd": 0.0, "energy_kwh": 0.0, "carbon_g": 0.0, "water_m3": 0.0,
                    }

        epoch_len_s = 900.0
        if self.datacenters:
            try:
                epoch_len_s = float(next(iter(self.datacenters.values()))._epoch_len_s)
            except Exception:
                epoch_len_s = 900.0
        epoch_start_ms = float(epoch_idx) * epoch_len_s * 1000.0

        final_details: List[Dict[str, Any]] = [d for d in details if d is not None]
        for dc in self.datacenters.values():
            pre_grid_kwh = float(getattr(dc, "energy_grid_kwh", 0.0))
            pre_emb_kg = float(getattr(dc, "embodied_battery_co2_kg", 0.0))

            if hasattr(dc, "finalize_epoch"):
                dc.finalize_epoch(epoch_start_ms=epoch_start_ms)

            post_grid_kwh = float(getattr(dc, "energy_grid_kwh", 0.0))
            post_emb_kg = float(getattr(dc, "embodied_battery_co2_kg", 0.0))
            dE_kwh = max(0.0, post_grid_kwh - pre_grid_kwh)
            dEmb_kg = max(0.0, post_emb_kg - pre_emb_kg)

            if dE_kwh <= 0.0 and dEmb_kg <= 0.0: continue

            tou_price = 0.0
            if getattr(dc, "tou_price", None) is not None and hasattr(dc, "_tou_price"):
                try:
                    tou_price = float(dc._tou_price(epoch_start_ms))
                except Exception:
                    tou_price = 0.0

            dCost_usd = dE_kwh * tou_price
            ci_g_per_kwh = float(getattr(dc, "carbon_intensity_g_per_kwh", 0.0))
            dCarbon_g = dE_kwh * ci_g_per_kwh + (dEmb_kg * 1000.0)

            if hasattr(dc, "cost_usd"): dc.cost_usd = float(getattr(dc, "cost_usd", 0.0)) + dCost_usd

            final_details.append({
                "dc_id": int(getattr(dc, "id", -1)),
                "start_ms": 0.0, "end_ms": 0.0, "exec_ms": 0.0, "ttft_s": 0.0,
                "energy_kwh": dE_kwh, "carbon_g": dCarbon_g, "cost_usd": dCost_usd, "water_m3": 0.0,
                "tag": "epoch_finalize_idle",
            })

        self._last_epoch_results = final_details
        self._last_epoch_metrics = self._aggregate_epoch_metrics(total_tokens_for_agg, final_details)
        return final_details

    def _aggregate_epoch_metrics(self, tokens, details: List[Dict[str, Any]]) -> Dict[str, Any]:
        ttft_sum, ttft_cnt, token_sum = 0.0, 0, 0
        reqs_completed, reqs_dropped = 0, 0

        for r in details:
            if r.get("tag") == "epoch_finalize_idle": continue
            if r.get("dropped", False):
                reqs_dropped += 1
                continue
            reqs_completed += 1

            v = r.get("ttft_s", r.get("TTFT", r.get("time_to_first_token_s")))
            if v is not None:
                try:
                    ttft_sum += float(v); ttft_cnt += 1
                except Exception:
                    pass
            t = r.get("tokens", r.get("total_tokens", None))
            if t is not None: token_sum += max(0, int(t))

        avg_ttft = (ttft_sum / float(ttft_cnt)) if ttft_cnt > 0 else 0.0
        total_energy_kwh, total_it_energy_kwh, total_cooling_energy_kwh = 0.0, 0.0, 0.0
        total_carbon_g, total_water_m3, total_cost_usd = 0.0, 0.0, 0.0

        for dc_id, dc in self.datacenters.items():
            grid_kwh = float(getattr(dc, "energy_grid_kwh", 0.0))
            total_energy_kwh += grid_kwh
            total_it_energy_kwh += float(getattr(dc, "energy_it_kwh", 0.0))
            total_cooling_energy_kwh += float(getattr(dc, "energy_cooling_kwh", 0.0))
            total_water_m3 += float(getattr(dc, "water_makeup_m3", 0.0))
            total_cost_usd += float(getattr(dc, "cost_usd", 0.0))

            ci = float(getattr(dc, "carbon_intensity_g_per_kwh", 0.0))
            water_carbon = float(getattr(dc, "water_carbon_g", 0.0))
            batt_emb_kg = float(getattr(dc, "embodied_battery_co2_kg", 0.0))
            total_carbon_g += (grid_kwh * ci) + water_carbon + (batt_emb_kg * 1000.0)

        return {
            "avg_ttft": float(avg_ttft), "requests_completed": reqs_completed,
            "requests_dropped": reqs_dropped, "energy_cost": float(total_cost_usd),
            "carbon_emissions": float(total_carbon_g), "water_usage": float(total_water_m3),
            "total_energy": float(total_energy_kwh), "total_it_energy_kwh": float(total_it_energy_kwh),
            "total_cooling_energy_kwh": float(total_cooling_energy_kwh), "processed_tokens": float(token_sum)
        }

    def report_global_stats(self) -> Dict[str, Any]:
        if not isinstance(self._last_epoch_metrics, dict):
            self._last_epoch_metrics = self._aggregate_epoch_metrics(0, self._last_epoch_results or [])
        m = self._last_epoch_metrics or {}
        return {
            "avg_ttft": float(m.get("avg_ttft", 0.0)), "energy_cost": float(m.get("energy_cost", 0.0)),
            "carbon_emissions": float(m.get("carbon_emissions", 0.0)), "water_usage": float(m.get("water_usage", 0.0)),
            "total_energy": float(m.get("total_energy", 0.0)),
            "total_it_energy_kwh": float(m.get("total_it_energy_kwh", 0.0)),
            "total_cooling_energy_kwh": float(m.get("total_cooling_energy_kwh", 0.0)),
            "processed_tokens": float(m.get("processed_tokens", 0.0)),
            "requests_completed": int(m.get("requests_completed", 0)),
            "requests_dropped": int(m.get("requests_dropped", 0)),
        }

    def report_dc_utilization(self) -> Dict[int, float]:
        util = {}
        for dc_id, dc in self.datacenters.items():
            if hasattr(dc, "report_utilization") and callable(getattr(dc, "report_utilization")):
                try:
                    util[dc_id] = max(0.0, min(1.0, float(dc.report_utilization())))
                except Exception:
                    pass
        return util


# -----------------------------
# Public entry point + CSV builders (Unchanged)
# -----------------------------

class LLM_Simulator:
    def __init__(self, spec_dir: str = "sim_specs", dc_specs_csv: Optional[str] = None,
                 node_specs_csv: Optional[str] = None, latency_csv: Optional[str] = None,
                 a100_csv: Optional[str] = None, h100_csv: Optional[str] = None, epoch_length: Optional[int] = None,
                 debug: bool = True, parallel_dc_workers: int = 1) -> None:
        self.debug = debug
        self.parallel_dc_workers = max(1, int(parallel_dc_workers or 1))
        self.spec_dir = spec_dir
        self.dc_specs_csv = dc_specs_csv or os.path.join(spec_dir, "Datacenter_specs.csv")
        self.node_specs_csv = node_specs_csv or os.path.join(spec_dir, "Node_Specs.csv")
        self.latency_csv = latency_csv or os.path.join(spec_dir, "Geo_Latencies.csv")
        self.a100_csv = a100_csv or os.path.join(spec_dir, "A100_GPU.csv")
        self.h100_csv = h100_csv or os.path.join(spec_dir, "H100_GPU.csv")
        self.cpu_csv = os.path.join(spec_dir, "POVRay_CPU.csv")

        if epoch_length is not None:
            self.epoch_length = int(epoch_length)
        else:
            gran_csv = os.path.join(spec_dir, "Workload_Granularity.csv")
            try:
                self.epoch_length = int(load_epoch_length_exact(gran_csv))
            except Exception:
                self.epoch_length = 900

        dc_specs, node_recs, lat_mat, gpu_tables = build_world_from_csvs_exact(self.dc_specs_csv, self.node_specs_csv,
                                                                               self.latency_csv, self.a100_csv,
                                                                               self.h100_csv, self.cpu_csv)

        self.datacenters: Dict[int, Datacenter] = {}
        for dc_id, params in dc_specs.items():
            dc = Datacenter(dc_id=dc_id, carbon_intensity_g_per_kwh=params["carbon_intensity_g_per_kwh"],
                            time_of_use_24h=params["time_of_use_24h"], cop_profile_24h=params["cop_profile_24h"],
                            blowdown_ratio=params["blowdown_ratio"],
                            water_cycling_density_m3_per_kwh_heat=params["water_cycling_density_m3_per_kwh_heat"],
                            potable_EI_kWh_per_m3=params["potable_EI_kWh_per_m3"],
                            wastewater_EI_kWh_per_m3=params["wastewater_EI_kWh_per_m3"],
                            water_static_m3_per_kwh_heat=params["water_static_m3_per_kwh_heat"],
                            cooling_mode=params.get("cooling_mode", "MECH_COP"), epoch_length=self.epoch_length)
            self.datacenters[dc_id] = dc

        total_nodes = 0
        for rec in node_recs:
            dc = self.datacenters[rec["dc_id"]]
            unit = ProcNode(node_id=rec["node_id"], type_id=rec.get("type_id"), accel_type=rec["accel_type"],
                            gpu_config=rec["gpu_config"], tdp_kw=float(rec["tdp_kw"]), idle_kw=float(rec["idle_kw"]),
                            model_perf=rec["model_perf"], workload_class=rec.get("workload_class", "generic"))
            dc.add_node(unit)
            total_nodes += 1
        self.network = Geo_Network(self.datacenters, lat_mat, parallel_dc_workers=self.parallel_dc_workers)

    def run_epoch(self, epoch_idx: int, workload_df: pd.DataFrame, schedule_plan: Dict[str, Any],
                  power_plan: Dict[str, Any]) -> Tuple[
        Dict[str, Any], List[Dict[str, Any]], Dict[int, Dict[str, float]]]:
        for dc in self.datacenters.values():
            if hasattr(dc, "reset_epoch"): dc.reset_epoch()
        detailed_results: List[Dict[str, Any]] = self.network.apply_schedule_plan(epoch_idx=epoch_idx,
                                                                                  workload_df=workload_df,
                                                                                  schedule_plan=schedule_plan,
                                                                                  power_plan=power_plan)
        metrics: Dict[str, Any] = self.network.report_global_stats()
        for k in ("avg_ttft", "energy_cost", "carbon_emissions", "water_usage", "total_energy", "total_it_energy_kwh",
                  "total_cooling_energy_kwh", "processed_tokens", "requests_completed", "requests_dropped"):
            metrics.setdefault(k, 0.0)
        dc_usage = self._get_dc_utilization(detailed_results)
        metrics["by_datacenter"] = dc_usage
        return metrics, detailed_results, dc_usage

    def run_epochs_parallel(self, workload_df: pd.DataFrame, schedule_plan: Dict[str, Any],
                            power_plan: Dict[str, Any], epoch_indices: Optional[List[int]] = None,
                            max_workers: Optional[int] = None, include_details: bool = False) -> Dict[int, Dict[str, Any]]:
        if workload_df is None or len(workload_df) == 0:
            return {}
        if "epoch" not in workload_df.columns:
            raise KeyError("workload_df must include an 'epoch' column for batch parallel execution")

        epoch_vals = workload_df["epoch"].astype(int)
        if epoch_indices is None:
            selected_epochs = sorted(epoch_vals.unique().tolist())
        else:
            selected_set = {int(e) for e in epoch_indices}
            selected_epochs = [int(e) for e in sorted(set(epoch_vals.tolist())) if int(e) in selected_set]
        if not selected_epochs:
            return {}

        groups: Dict[int, pd.DataFrame] = {}
        for e in selected_epochs:
            groups[e] = workload_df[epoch_vals == e].copy()

        sim_kwargs = {
            "spec_dir": self.spec_dir,
            "dc_specs_csv": self.dc_specs_csv,
            "node_specs_csv": self.node_specs_csv,
            "latency_csv": self.latency_csv,
            "a100_csv": self.a100_csv,
            "h100_csv": self.h100_csv,
            "epoch_length": self.epoch_length,
            "debug": self.debug,
            "parallel_dc_workers": self.parallel_dc_workers,
        }

        if max_workers is None:
            max_workers = max(1, (os.cpu_count() or 2) - 1)
        max_workers = max(1, min(int(max_workers), len(selected_epochs)))

        results: Dict[int, Dict[str, Any]] = {}
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            future_map = {}
            for e in selected_epochs:
                payload = (
                    sim_kwargs, int(e), groups[e].to_dict(orient="records"),
                    schedule_plan, power_plan, bool(include_details),
                )
                future_map[ex.submit(_run_epoch_worker, payload)] = int(e)

            for fut in as_completed(future_map):
                e = future_map[fut]
                out = fut.result()
                if include_details:
                    epoch_idx, metrics, details, dc_usage = out
                    results[int(epoch_idx)] = {"metrics": metrics, "details": details, "by_datacenter": dc_usage}
                else:
                    epoch_idx, metrics, dc_usage = out
                    results[int(epoch_idx)] = {"metrics": metrics, "by_datacenter": dc_usage}
        return results

    def _get_dc_utilization(self, detailed_results: List[Dict[str, Any]]) -> Dict[int, Dict[str, float]]:
        if hasattr(self.network, "report_dc_utilization") and callable(getattr(self.network, "report_dc_utilization")):
            try:
                return self._normalize_util_map(self.network.report_dc_utilization())
            except Exception:
                pass
        per_dc, hook_found = {}, False
        for dc_id, dc in self.datacenters.items():
            if hasattr(dc, "report_utilization") and callable(getattr(dc, "report_utilization")):
                try:
                    per_dc[dc_id] = dc.report_utilization(); hook_found = True
                except Exception:
                    continue
        if hook_found: return self._normalize_util_map(per_dc)
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
            u = v.get("utilization", v.get("util", v.get("usage", 0.0))) if isinstance(v, dict) else v
            try:
                u = float(u)
            except Exception:
                u = 0.0
            out[dc_id] = {"utilization": max(0.0, min(1.0, u))}
        return out

    def _aggregate_dc_utilization(self, detailed_results: List[Dict[str, Any]]) -> Dict[int, Dict[str, float]]:
        from collections import defaultdict
        procs_per_dc: Dict[int, int] = {}
        for dc_id, dc in self.datacenters.items():
            units = 0
            for attr in ("exec_units", "units", "processors"):
                if hasattr(dc, attr) and getattr(dc, attr) is not None:
                    try:
                        units = len(getattr(dc, attr))
                    except TypeError:
                        try:
                            units = int(getattr(dc, attr))
                        except Exception:
                            pass
                    if units: break
            if not units:
                for attr in ("processor_count", "unit_count", "num_executors"):
                    if hasattr(dc, attr) and getattr(dc, attr) is not None:
                        try:
                            units = int(getattr(dc, attr)); break
                        except Exception:
                            pass
            procs_per_dc[int(dc_id)] = max(0, int(units))

        cap_ms: Dict[int, float] = {dc: float(procs) * float(self.epoch_length) * 1000.0 for dc, procs in
                                    procs_per_dc.items()}
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
            for k in ("exec_ms", "proc_ms", "service_ms", "gpu_time_ms", "process_ms"):
                if k in rec and rec[k] is not None:
                    try:
                        return max(0.0, float(rec[k]))
                    except Exception:
                        continue
            s, e = rec.get("start_ms", rec.get("start_time_ms")), rec.get("end_ms", rec.get("end_time_ms"))
            if s is not None and e is not None:
                try:
                    return max(0.0, float(e) - float(s))
                except Exception:
                    pass
            return 0.0

        for rec in detailed_results:
            dc_id = _extract_dc_id(rec)
            if dc_id is not None: busy_ms[dc_id] += _extract_busy_ms(rec)

        util: Dict[int, Dict[str, float]] = {}
        for dc_id in procs_per_dc.keys():
            cap = cap_ms.get(dc_id, 0.0)
            u = 0.0 if cap <= 0.0 else busy_ms.get(dc_id, 0.0) / cap
            util[dc_id] = {"utilization": max(0.0, min(1.0, u))}
        for dc_id in busy_ms.keys(): util.setdefault(dc_id, {"utilization": 0.0})
        return util


def _run_epoch_worker(payload: Tuple[Dict[str, Any], int, List[Dict[str, Any]], Dict[str, Any], Dict[str, Any], bool]):
    sim_kwargs, epoch_idx, records, schedule_plan, power_plan, include_details = payload
    sim = LLM_Simulator(**sim_kwargs)
    workload_df = pd.DataFrame(records)
    metrics, details, dc_usage = sim.run_epoch(int(epoch_idx), workload_df, schedule_plan or {}, power_plan or {})
    if include_details:
        return int(epoch_idx), metrics, details, dc_usage
    return int(epoch_idx), metrics, dc_usage


# =========================
# EXACT-HEADER CSV HELPERS
# =========================

def _parse_24h_exact(series_str: str) -> List[float]:
    parts = [p.strip() for p in str(series_str).replace(",", ";").split(";")]
    return [float(x) for x in parts]


def _parse_gpu_from_node_type_exact(s: str) -> Tuple[str, str]:
    t = s.strip().upper().replace("-", "_")
    t = t[:-1] if t.endswith("S") else t
    n, g = t.split("_", 1)
    return (g, f"{n}_{g}")


def _counts_from_nodetypes_exact(nodetypes_str: str) -> Dict[int, int]:
    seq = [int(x.strip()) for x in nodetypes_str.split(";") if x.strip()]
    return dict(sorted(Counter(seq).items()))


def _parse_24h_semicolon(val):
    if val is None: return None
    try:
        import math
        if isinstance(val, float) and math.isnan(val): return None
    except Exception:
        pass
    if isinstance(val, (list, tuple)) and len(val) == 24:
        try:
            return [float(v) for v in val]
        except Exception:
            return None
    s = str(val) if not isinstance(val, str) else val
    tokens = [t.strip() for t in s.split(';') if t.strip() != ""]
    if len(tokens) != 24: return None
    try:
        return [float(t) for t in tokens]
    except Exception:
        return None


def _parse_node_type_counts_from_row(row, expected_types=6):
    candidates = ["Node_Type_Counts", "node_type_counts", "NodeTypeCounts", "NodeTypeCounts(0-5)",
                  "NodeTypeCounts_0to5", "NodeTypeCounts_0_5"]
    raw = None
    for k in candidates:
        if k in row and row[k] is not None and str(row[k]).strip() != "":
            raw = str(row[k]).strip()
            break
    if not raw: return {}, ""
    parts = [p.strip() for p in raw.replace(",", ";").split(";") if p.strip() != ""]
    counts = {}
    if ":" in parts[0]:
        for item in parts:
            if ":" not in item: continue
            k, v = item.split(":", 1)
            try:
                counts[int(k.strip())] = int(float(v.strip()))
            except Exception:
                pass
    else:
        ints = []
        for item in parts:
            try:
                ints.append(int(float(item)))
            except Exception:
                pass
        if ints: counts = {i: val for i, val in enumerate(ints)}
    counts = {int(k): int(v) for k, v in counts.items() if int(v) >= 0}
    counts_str = ",".join(f"{k}:{counts[k]}" for k in sorted(counts.keys()))
    return counts, counts_str


def load_dc_specs_exact(csv_path) -> dict[int, dict]:
    import pandas as pd
    df = pd.read_csv(csv_path)
    out: dict[int, dict] = {}
    for _, row in df.iterrows():
        did = int(_safe_float(row["DC_Num"]))
        if did < 0: continue
        type_counts, node_type_counts_str = _parse_node_type_counts_from_row(row)
        if not type_counts:
            raw_node_types = row.get("Node_Types", None)
            if raw_node_types is not None and str(raw_node_types).strip() != "":
                try:
                    type_counts = _counts_from_nodetypes_exact(str(raw_node_types))
                except Exception:
                    type_counts = {}
        if not node_type_counts_str and type_counts:
            node_type_counts_str = ",".join(f"{k}:{type_counts[k]}" for k in sorted(type_counts.keys()))
        out[did] = {
            "dc_id": did,
            "carbon_intensity_g_per_kwh": _safe_float(row["Carbon_Intensity"]),
            "water_static_m3_per_kwh_heat": _safe_float(row["Water_Static"]),
            "water_cycling_density_m3_per_kwh_heat": _safe_float(row["Water_Cycling_Density"]),
            "blowdown_ratio": _safe_float(row["Solids_Ratio"]),
            "potable_EI_kWh_per_m3": _safe_float(row["Potable_Energy_Intensity"]),
            "wastewater_EI_kWh_per_m3": _safe_float(row["Wastewater_Energy_Intensity"]),
            "time_of_use_24h": _parse_24h_semicolon(row["Time_of_Use(24_Hours)"]),
            "cop_profile_24h": _parse_24h_semicolon(row["COP_Profile(24_Hours)"]),
            "node_type_counts": type_counts,
            "node_type_counts_str": node_type_counts_str,
            "Total_Nodes": _safe_float(row["Total_Nodes"]),
        }
    return out


def load_node_type_templates_exact(node_specs_csv: str) -> Dict[int, Dict[str, Any]]:
    df = pd.read_csv(node_specs_csv)
    templates: Dict[int, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        tid = int(row["Node_Num"])
        accel_type, gpu_config = _parse_gpu_from_node_type_exact(row["Node_Type"])
        templates[tid] = {
            "accel_type": accel_type, "gpu_config": gpu_config,
            "processor_count": 1, "tdp_kw": 0.0, "idle_kw": 0.0,
        }
    return templates


def load_gpu_table_exact(gpu_csv_path: str, chip: str) -> Dict[str, List[Dict[str, Any]]]:
    t = pd.read_csv(gpu_csv_path)
    table: Dict[str, List[Dict[str, Any]]] = {}
    for _, row in t.iterrows():
        key = f"{int(row['num_GPUs'])}_{chip}"
        d = row.to_dict()
        d["_meta"] = {"tdp_kw": float(row["TDP"]) / 1000.0}
        if key not in table: table[key] = []
        table[key].append(d)
    return table


def build_gpu_tables_exact(a100_csv: str, h100_csv: str, cpu_csv: str | None = None) -> Dict[
    str, Dict[str, List[Dict[str, Any]]]]:
    tables = {"A100": load_gpu_table_exact(a100_csv, "A100"), "H100": load_gpu_table_exact(h100_csv, "H100")}
    if cpu_csv is not None and os.path.exists(cpu_csv): tables["CPU"] = load_gpu_table_exact(cpu_csv, "CPU")
    return tables


def load_latency_matrix_exact(lat_csv: str) -> List[List[float]]:
    return pd.read_csv(lat_csv).drop(columns=["Datacenter_Dest"]).astype(float).values.tolist()


def load_epoch_length_exact(path):
    val = int(pd.read_csv(path).iloc[0].get("Epoch_Length", 900))
    return val if val > 0 else 900


def build_world_from_csvs_exact(dc_specs_csv: str, node_specs_csv: str, latency_csv: str, a100_csv: str, h100_csv: str,
                                cpu_csv: str | None = None) -> tuple[
    Dict[int, Dict[str, Any]], List[Dict[str, Any]], List[List[float]], Dict[str, Dict[str, Any]]]:
    def _parse_counts_str_to_dict(s: str) -> Dict[int, int]:
        if not s: return {}
        out = {}
        for piece in [p.strip() for p in str(s).replace(",", ";").split(";") if p.strip()]:
            if ":" not in piece: continue
            k, v = piece.split(":", 1)
            try:
                ki, vi = int(k.strip()), int(float(v.strip()))
                if ki >= 0 and vi >= 0: out[ki] = vi
            except:
                pass
        return out

    def _even_split_counts(total_nodes: int, ntypes: int) -> Dict[int, int]:
        base, rem = divmod(max(0, int(total_nodes or 0)), max(1, ntypes))
        out = {i: base for i in range(ntypes)}
        for i in range(rem): out[i] += 1
        return out

    def _num(x):
        if x is None: return None
        if isinstance(x, (int, float)): return float(x)
        s = str(x).strip().lower()
        if s in ("none", "nan", ""): return None
        try:
            return float(s)
        except:
            return None

    dc_specs = load_dc_specs_exact(dc_specs_csv)
    templates = load_node_type_templates_exact(node_specs_csv)
    gpu_tbls = build_gpu_tables_exact(a100_csv, h100_csv, cpu_csv=cpu_csv)
    lat_mat = load_latency_matrix_exact(latency_csv)

    global MANUAL_NODE_TYPE_COUNTS
    overrides = MANUAL_NODE_TYPE_COUNTS or {}
    ntypes = len(templates)
    nodes: List[Dict[str, Any]] = []

    for dc_id, params in dc_specs.items():
        counts_str = params.get("node_type_counts_str", "") or params.get("Node_Type_Counts", "")
        counts = _parse_counts_str_to_dict(counts_str)
        if not counts:
            total = int(params.get("Total_Nodes", 0))
            if total > 0: counts = _even_split_counts(total, ntypes)

        if overrides and dc_id in overrides:
            for tid, val in overrides[dc_id].items():
                if 0 <= tid < ntypes: counts[tid] = int(val)

        if not counts: continue

        next_local_node_id = 0
        for type_id in sorted(counts.keys()):
            num = int(counts[type_id])
            if num <= 0: continue
            tmpl = templates[type_id]
            accel, cfg, procs = tmpl["accel_type"], tmpl["gpu_config"], int(tmpl.get("processor_count", 1))
            rows_list = gpu_tbls.get(accel, {}).get(cfg, [])
            if isinstance(rows_list, dict): rows_list = [rows_list]
            if not rows_list:
                tdp_kw, idle_kw, model_perf = 0.0, 0.0, {}
            else:
                meta = rows_list[0].get("_meta", {})
                tdp_kw = float(meta.get("tdp_kw", 0.0))
                idle_kw = float(meta.get("idle_kw", 0.15 * tdp_kw))
                model_perf = {}
                if accel == "CPU":
                    row = rows_list[0]
                    for wl in SPEC_CPU_WORKLOADS:
                        val = _num(row.get(wl))
                        if val is not None: model_perf[wl] = {"ms_per_request": val, "ms_per_token": val}
                    if not model_perf:
                        ms7 = _num(row.get("Llama7b_Process"))
                        if ms7: model_perf["povray_r"] = {"ms_per_request": ms7, "ms_per_token": ms7}
                else:
                    for row in rows_list:
                        if "Model_Variant" in row:
                            variant = str(row.get("Model_Variant", "Base")).replace("(70B)", "").strip()
                            scenario = str(row.get("Scenario_Type", "Standard"))
                            batch = int(float(row.get("batch_size", "1")))
                            for base_model in ["Llama7b", "Llama70b"]:
                                val = _num(row.get(f"{base_model}_Process"))
                                if val is not None and val > 0:
                                    entry = {"ms_per_token": val / (batch * 1000.0), "batch_size": batch,
                                             "variant": variant, "scenario": scenario}
                                    model_perf[f"{base_model}_{scenario}_{variant}_B{batch}"] = entry
                                    relaxed_key = f"{base_model}_{variant}_B{batch}"
                                    if relaxed_key not in model_perf: model_perf[relaxed_key] = entry
                        else:
                            denom = (_num(row.get("prefill_token_size")) or 0) + (
                                        _num(row.get("gen_token_size")) or 0) or 1.0
                            ms7, ms70 = _num(row.get("Llama7b_Process")), _num(row.get("Llama70b_Process"))
                            if ms7: model_perf["Llama7b"] = {"ms_per_request": ms7, "ms_per_token": ms7 / denom}
                            if ms70: model_perf["Llama70b"] = {"ms_per_request": ms70, "ms_per_token": ms70 / denom}
            for _ in range(num):
                nodes.append({
                    "dc_id": int(dc_id), "node_id": next_local_node_id, "type_id": int(type_id),
                    "accel_type": accel, "gpu_config": cfg, "processor_count": procs,
                    "tdp_kw": tdp_kw, "idle_kw": idle_kw, "model_perf": model_perf,
                })
                next_local_node_id += 1
    return dc_specs, nodes, lat_mat, gpu_tbls
