#!/usr/bin/env python3
"""
profile_simulator.py — find where the simulator actually spends its time.

Runs ONE realistic autoscaled epoch through Rate_Flow_Sim.run_epoch under
cProfile and reports the hottest functions by cumulative and total time.
This tells us which lines to optimize — measured, not guessed.

Usage:
    python3 profile_simulator.py
    python3 profile_simulator.py --trace simulator_ready_trace.csv --epoch 81
    python3 profile_simulator.py --count-mult 4000   # match the real sweep scale

The default count-mult matches a realistic sweep epoch so the profile reflects
real run conditions, not a toy workload.
"""
from __future__ import annotations
import argparse
import cProfile
import io
import pstats
import sys
import time
import numpy as np
import pandas as pd

from Rate_Flow_Sim_v2 import LLM_Simulator


def load_epoch(trace_path: str, epoch_idx: int):
    trace = pd.read_csv(trace_path)
    ecol = None
    for c in ("epoch", "epoch_idx", "epoch_index"):
        if c in trace.columns:
            ecol = c
            break
    if ecol is None:
        raise ValueError(f"No epoch column in {trace_path}")
    grp = trace.groupby(ecol)
    if epoch_idx not in grp.groups:
        # pick the busiest epoch instead
        epoch_idx = max(grp.groups.keys(), key=lambda e: len(grp.get_group(e)))
        print(f"[profile] requested epoch not found; using busiest epoch {epoch_idx}")
    return grp.get_group(epoch_idx).copy(), epoch_idx


def replicate_epoch(epoch_df: pd.DataFrame, count_mult: int, token_scale: float,
                    epoch_length_s: int = 900):
    """Build a scaled epoch the same way the autoscaler would — so the profile
    reflects real sweep conditions."""
    if count_mult <= 1:
        out = epoch_df.copy()
    else:
        n_base = len(epoch_df)
        tiled = {c: np.tile(epoch_df[c].to_numpy(), count_mult) for c in epoch_df.columns}
        out = pd.DataFrame(tiled)
        base_arr = np.tile(
            pd.to_numeric(epoch_df["arrival_ms"], errors="coerce").fillna(0.0).to_numpy(),
            count_mult)
        dup = np.repeat(np.arange(count_mult, dtype=float), n_base)
        win = float(epoch_length_s) * 1000.0
        slot = win / float(count_mult)
        rng = np.random.default_rng(12345)
        jit = rng.uniform(0.0, slot, size=len(out))
        out["arrival_ms"] = (base_arr + dup * slot + jit) % win
    out["num_tokens"] = (
        pd.to_numeric(out["num_tokens"], errors="coerce").fillna(0.0) * token_scale
    ).round().astype(int)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default="simulator_ready_trace.csv")
    ap.add_argument("--spec-dir", default="sim_specs")
    ap.add_argument("--epoch", type=int, default=81)
    ap.add_argument("--epoch-length", type=int, default=900)
    ap.add_argument("--count-mult", type=int, default=4000,
                    help="row replication — match the real sweep (~4000)")
    ap.add_argument("--token-scale", type=float, default=1.6)
    ap.add_argument("--top", type=int, default=25, help="hottest N functions to show")
    args = ap.parse_args()

    print("=" * 74)
    print("SIMULATOR PROFILE")
    print("=" * 74)

    print("[1] Building simulator ...")
    t0 = time.time()
    sim = LLM_Simulator(spec_dir=args.spec_dir,
                        epoch_length=args.epoch_length, debug=False)
    print(f"    built in {time.time()-t0:.2f}s")

    print("[2] Loading + scaling epoch ...")
    base_df, ep = load_epoch(args.trace, args.epoch)
    scaled = replicate_epoch(base_df, args.count_mult, args.token_scale,
                             args.epoch_length)
    print(f"    epoch {ep}: {len(base_df):,} base rows -> {len(scaled):,} scaled "
          f"(count_mult={args.count_mult}, token_scale={args.token_scale})")

    # Empty schedule_plan = simulator default routing; empty power_plan = default.
    schedule_plan: dict = {}
    power_plan: dict = {}

    print("[3] Profiling run_epoch ...")
    prof = cProfile.Profile()
    t0 = time.time()
    prof.enable()
    sim.run_epoch(ep, scaled, schedule_plan, power_plan)
    prof.disable()
    wall = time.time() - t0
    print(f"    run_epoch wall time: {wall:.2f}s for {len(scaled):,} requests "
          f"({1000*wall/max(1,len(scaled)):.3f} ms/request)")

    # ── Report: cumulative time (where time is spent including callees) ──────
    print("\n" + "=" * 74)
    print(f"TOP {args.top} BY CUMULATIVE TIME (includes sub-calls)")
    print("=" * 74)
    s = io.StringIO()
    ps = pstats.Stats(prof, stream=s).sort_stats("cumulative")
    ps.print_stats(args.top)
    print(s.getvalue())

    # ── Report: total time (time IN the function itself, the real hotspots) ──
    print("=" * 74)
    print(f"TOP {args.top} BY TOTAL/SELF TIME (the actual hot lines to optimize)")
    print("=" * 74)
    s = io.StringIO()
    ps = pstats.Stats(prof, stream=s).sort_stats("tottime")
    ps.print_stats(args.top)
    print(s.getvalue())

    # ── Extrapolation ────────────────────────────────────────────────────────
    print("=" * 74)
    print("EXTRAPOLATION")
    print("=" * 74)
    per_epoch = wall
    print(f"  1 epoch  @ this scale : {per_epoch:.1f}s")
    print(f"  82 epochs (1 config)  : {per_epoch*82/60:.1f} min")
    print(f"  1530 configs          : {per_epoch*82*1530/3600:.0f} hours "
          f"({per_epoch*82*1530/86400:.1f} days)")
    print()
    print("  The TOTAL-TIME table above lists the functions to optimize first.")
    print("  A function high in tottime with a huge call count is usually a")
    print("  per-request Python loop that can be vectorised.")


if __name__ == "__main__":
    main()