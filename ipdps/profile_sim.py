"""Profiling + regression harness for Rate_Flow_Sim_v2.

Builds a minimal-but-valid sim_specs dir, a realistic synthetic workload,
runs run_epoch, and profiles it.  Also used to verify optimized versions
produce EXACTLY identical outputs.
"""
import os, sys, cProfile, pstats, io, time
import numpy as np
import pandas as pd

# Synthetic specs go in their OWN directory so the harness can never clobber
# a real sim_specs/.  Point RFS_SPEC_DIR at your real spec dir to run the
# harness/equivalence tests against the actual world instead (no files are
# written in that case).
SPEC = os.environ.get("RFS_SPEC_DIR", "sim_specs_synth")
_USING_REAL_SPECS = "RFS_SPEC_DIR" in os.environ

def build_specs(num_dcs=8, nodes_per_dc=40):
    if _USING_REAL_SPECS:
        return  # never write into a user-supplied spec dir
    if os.path.isdir(SPEC):
        expected = {"Datacenter_specs.csv", "Node_Specs.csv", "Geo_Latencies.csv",
                    "H100_GPU.csv", "A100_GPU.csv"}
        existing = set(os.listdir(SPEC))
        if existing - expected:
            raise SystemExit(
                f"Refusing to write synthetic specs: {SPEC}/ contains unexpected "
                f"files ({sorted(existing - expected)[:5]}...). Set RFS_SPEC_DIR "
                f"to use real specs, or point SPEC elsewhere.")
    os.makedirs(SPEC, exist_ok=True)
    tou = ";".join(f"{0.08+0.04*np.sin(h/24*6.283):.4f}" for h in range(24))
    cop = ";".join(f"{3.5+0.5*np.sin(h/24*6.283):.3f}" for h in range(24))
    rows = []
    for d in range(num_dcs):
        rows.append({
            "DC_Num": d, "Carbon_Intensity": 200 + 40*d, "Water_Static": 0.00018,
            "Water_Cycling_Density": 0.0009, "Solids_Ratio": 0.2,
            "Potable_Energy_Intensity": 0.4, "Wastewater_Energy_Intensity": 0.7,
            "Time_of_Use(24_Hours)": tou, "COP_Profile(24_Hours)": cop,
            "Node_Type_Counts": f"0:{nodes_per_dc//2};1:{nodes_per_dc - nodes_per_dc//2}",
            "Total_Nodes": nodes_per_dc, "Cooling_Mode": "MECH_COP",
        })
    pd.DataFrame(rows).to_csv(f"{SPEC}/Datacenter_specs.csv", index=False)
    pd.DataFrame([
        {"Node_Num": 0, "Node_Type": "8_H100s"},
        {"Node_Num": 1, "Node_Type": "8_A100s"},
    ]).to_csv(f"{SPEC}/Node_Specs.csv", index=False)
    for chip, ms70 in [("H100", 28.0), ("A100", 55.0)]:
        pd.DataFrame([{
            "num_GPUs": 8, "TDP": 5600 if chip == "H100" else 3200,
            "Model_Variant": "FP16", "Scenario_Type": "Standard", "batch_size": 1,
            "Llama7b_Process": ms70/6.0, "Llama70b_Process": ms70,
        }]).to_csv(f"{SPEC}/{chip}_GPU.csv", index=False)
    lat = pd.DataFrame(
        [[0.0 if i == j else 20.0 + 5.0*abs(i-j) for j in range(num_dcs)] for i in range(num_dcs)])
    lat.insert(0, "Datacenter_Dest", range(num_dcs))
    lat.to_csv(f"{SPEC}/Geo_Latencies.csv", index=False)

def build_workload(n=300_000, num_dcs=8, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "source_dc_id": rng.integers(0, num_dcs, n),
        "model_type": rng.choice(["Llama70b", "Llama7b"], n, p=[0.75, 0.25]),
        "num_tokens": rng.lognormal(6.0, 0.8, n).astype(np.int64).clip(10, 30_000),
        "arrival_ms": np.sort(rng.uniform(0, 900_000, n)),
        "base_num_tokens": rng.lognormal(5.5, 0.7, n).astype(np.int64).clip(10, 5000),
    })

def run(module_name, n_rows, collect_details, profile=False):
    sys.path.insert(0, ".")
    mod = __import__(module_name)
    sim = mod.LLM_Simulator(debug=False, spec_dir=SPEC)
    wl = build_workload(n_rows)
    plan = {"route": {"Llama70b": 2, "Llama7b": 5}}   # exercise routing path
    if profile:
        pr = cProfile.Profile(); pr.enable()
    t0 = time.perf_counter()
    stats, details, dc_usage = sim.run_epoch(3, wl, plan, {"all": "ON"}, collect_details=collect_details)
    dt = time.perf_counter() - t0
    if profile:
        pr.disable()
        s = io.StringIO()
        pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(22)
        print(s.getvalue())
    return stats, details, dc_usage, dt

if __name__ == "__main__":
    build_specs()
    mode = sys.argv[1] if len(sys.argv) > 1 else "profile"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 300_000
    if mode == "profile":
        stats, details, _, dt = run("Rate_Flow_Sim_v2", n, collect_details=True, profile=True)
        print(f"rows={n}  wall={dt:.2f}s  completed={stats['requests_completed']} "
              f"dropped={stats['requests_dropped']} avg_ttft={stats['avg_ttft']:.4f}")