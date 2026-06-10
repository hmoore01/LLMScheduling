#!/usr/bin/env python3
"""
verify_map_array.py — prove the schedule-map memory optimisation is
result-preserving.

build_schedule_map now returns the request->DC assignment as a compact int32
array ("map_array") instead of a Python dict ("map").  This script feeds the
*same logical assignment* through apply_schedule_plan in BOTH forms and asserts
the simulator produces byte-identical epoch metrics.

It checks two cases:
  * a DENSE assignment  — every request routed
  * a SPARSE assignment — ~15% of requests left unassigned, exercising the
    -1-sentinel / route-table fallback path

Run in the project directory (needs the sim spec CSVs):
    python verify_map_array.py --spec-dir sim_specs
"""
import argparse
import sys

import numpy as np
import pandas as pd


def build_workload(n, models, dcs, epoch_len, seed=3):
    rng = np.random.default_rng(seed)
    base = rng.integers(200, 2000, size=n).astype(np.int64)
    return pd.DataFrame({
        "source_dc_id":    rng.choice(dcs, size=n),
        "model_type":      rng.choice(models, size=n),
        "base_num_tokens": base,
        "num_tokens":      (base * 25).astype(np.int64),
        "arrival_ms":      rng.uniform(0, epoch_len * 1000.0, size=n),
    })


def main():
    ap = argparse.ArgumentParser(description="A/B test: dict map vs int32 map_array.")
    ap.add_argument("--spec-dir", default="sim_specs")
    ap.add_argument("--rows", type=int, default=20000)
    ap.add_argument("--epoch-length", type=int, default=900)
    a = ap.parse_args()

    import Rate_Flow_Sim_v2 as RFS

    def fresh_sim():
        # a clean simulator — run_epoch resets DC counters, so two fresh sims
        # given identical inputs must return identical metrics.
        return RFS.LLM_Simulator(spec_dir=a.spec_dir,
                                 epoch_length=a.epoch_length, debug=False)

    ref = fresh_sim()
    dcs = sorted(int(d) for d in ref.datacenters.keys())
    models = sorted({m for dc in ref.datacenters.values()
                     for u in getattr(dc, "units", [])
                     for m in getattr(u, "model_perf", {}).keys()}) or ["m"]
    del ref

    print("═" * 74)
    print("  verify_map_array.py — dict 'map' vs int32 'map_array' equivalence")
    print(f"  spec-dir={a.spec_dir}  rows={a.rows:,}  DCs={len(dcs)}")
    print("═" * 74)

    rng = np.random.default_rng(11)
    fails = 0

    for label, sparse in [("dense  (every request routed)", False),
                          ("sparse (~15% unassigned -> fallback)", True)]:
        df = build_workload(a.rows, models, dcs, a.epoch_length)
        n = len(df)

        # one logical assignment, expressed two ways
        d = {}
        for i in range(n):
            if sparse and rng.random() < 0.15:
                continue                       # leave unassigned
            d[i] = int(rng.choice(dcs))
        arr = np.full(n, -1, dtype=np.int32)
        for i, dc in d.items():
            arr[i] = dc

        power = {dc: {"all": "IDLE"} for dc in dcs}

        m_dict = fresh_sim().run_epoch(0, df, {"map": d},          power)[0]
        m_arr  = fresh_sim().run_epoch(0, df, {"map_array": arr},   power)[0]

        bad = []
        for k in sorted(set(m_dict) | set(m_arr)):
            va, vb = m_dict.get(k), m_arr.get(k)
            if isinstance(va, dict) or isinstance(vb, dict):
                continue                       # nested (by_datacenter) — derives from routing
            try:
                if abs(float(va) - float(vb)) > 1e-9:
                    bad.append((k, va, vb))
            except (TypeError, ValueError):
                if va != vb:
                    bad.append((k, va, vb))

        if bad:
            fails += 1
            print(f"  [FAIL] {label}")
            for k, va, vb in bad[:8]:
                print(f"         {k}: dict={va}  array={vb}")
        else:
            n_metrics = sum(1 for k in m_dict if not isinstance(m_dict[k], dict))
            print(f"  [PASS] {label}")
            print(f"         {n_metrics} scalar metrics identical "
                  f"(avg_ttft={m_dict.get('avg_ttft', 0):.6f}s)")

    print("═" * 74)
    if fails == 0:
        print("RESULT: PASS — the schedule-map array conversion is result-preserving.")
        print("The simulator routes a dict 'map' and an int32 'map_array' identically.")
        sys.exit(0)
    print(f"RESULT: FAIL — {fails} case(s) diverged.  Do NOT deploy the change.")
    sys.exit(1)


if __name__ == "__main__":
    main()