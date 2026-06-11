"""Verify the fast path is bit-identical to the reference implementation."""
import os, sys, importlib, time
import numpy as np
import pandas as pd
from profile_sim import build_specs, build_workload, SPEC

build_specs()

def fresh_sim(mod):
    return mod.LLM_Simulator(debug=False, spec_dir=SPEC)

def run_case(mod, fast, case, collect_details):
    os.environ["RFS_FAST"] = "1" if fast else "0"
    importlib.reload(mod)
    sim = fresh_sim(mod)
    name, wl, plan, power = case
    if name == "battery":
        for dc in sim.datacenters.values():
            if hasattr(mod, "Battery"):
                dc.battery = mod.Battery()
            dc.solar_pv_area_m2 = 5000.0
            dc.solar_pv_efficiency = 0.2
            dc.solar_irradiance_24h = [max(0.0, 800.0*np.sin((h-6)/12*np.pi)) for h in range(24)]
    if name == "lmp":
        sim.apply_lmp_override({d: 0.31 for d in sim.datacenters})
    t0 = time.perf_counter()
    stats, details, dc_usage = sim.run_epoch(7, wl, plan, power, collect_details=collect_details)
    dt = time.perf_counter() - t0
    state = {}
    for d, dc in sim.datacenters.items():
        state[d] = {a: getattr(dc, a, None) for a in (
            "energy_it_kwh","energy_other_kwh","energy_cooling_kwh","energy_grid_kwh",
            "energy_solar_kwh","energy_batt_discharge_kwh","energy_batt_charge_kwh",
            "energy_solar_curtailed_kwh","water_static_m3","water_evap_m3","water_blowdown_m3",
            "water_makeup_m3","water_energy_potable_kwh","water_energy_wastewater_kwh",
            "water_energy_total_kwh","water_carbon_g","cost_usd","_busy_ms")}
        state[d]["node_avail"] = sorted((u.node_id, u.next_available_ms, u.busy_ms_epoch) for u in dc.units)
    return stats, details, dc_usage, state, dt

def cmp_vals(a, b, path):
    if isinstance(a, dict):
        assert isinstance(b, dict) and set(a) == set(b), f"{path}: keys {set(a)^set(b)}"
        for k in a: cmp_vals(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), f"{path}: len {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)): cmp_vals(x, y, f"{path}[{i}]")
    elif isinstance(a, float):
        assert (a == b) or (np.isnan(a) and np.isnan(b)), f"{path}: {a!r} != {b!r}"
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"

import Rate_Flow_Sim_v2 as mod

wl_normal   = build_workload(60_000, seed=1)
wl_overload = build_workload(60_000, seed=2)
wl_overload["num_tokens"] = (wl_overload["num_tokens"] * 40).clip(upper=2_000_000)  # force boundary drops
wl_unknown  = build_workload(20_000, seed=3)
wl_unknown.loc[::7, "model_type"] = "Mystery13b"   # exercise model-fallback misses

cases = [
    ("plain",    wl_normal,   {},                                          {"all": "ON"}),
    ("route",    wl_normal,   {"route": {"Llama70b": 2, "Llama7b": 5}},    {"all": "ON"}),
    ("maparr",   wl_normal,   {"map_array": np.arange(len(wl_normal)) % 8}, {"all": "ON"}),
    ("overload", wl_overload, {"route": {"Llama70b": 1, "Llama7b": 1}},    {"all": "ON"}),
    ("poweroff", wl_normal,   {},                                          {str(d): ("OFF" if d % 3 == 0 else "ON") for d in range(8)}),
    ("battery",  wl_normal,   {},                                          {"all": "ON"}),
    ("lmp",      wl_normal,   {},                                          {"all": "ON"}),
    ("unknown",  wl_unknown,  {},                                          {"all": "ON"}),
]

speed = []
for case in cases:
    for cd in (True, False):
        s_f, d_f, u_f, st_f, t_f = run_case(mod, True,  case, cd)
        s_r, d_r, u_r, st_r, t_r = run_case(mod, False, case, cd)
        cmp_vals(s_r, s_f, f"{case[0]}/cd={cd}/stats")
        cmp_vals(u_r, u_f, f"{case[0]}/cd={cd}/dc_usage")
        cmp_vals(st_r, st_f, f"{case[0]}/cd={cd}/dc_state")
        if cd:
            assert len(d_r) == len(d_f)
            for i, (rr, rf) in enumerate(zip(d_r, d_f)):
                cmp_vals(rr, rf, f"{case[0]}/details[{i}]")
        speed.append((case[0], cd, t_r, t_f))
        print(f"  {case[0]:<9} collect_details={cd!s:<5} IDENTICAL  "
              f"(ref {t_r:5.2f}s -> fast {t_f:5.2f}s, {t_r/max(t_f,1e-9):4.1f}x)")

print("\nALL CASES BIT-IDENTICAL")