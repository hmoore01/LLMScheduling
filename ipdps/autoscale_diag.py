#!/usr/bin/env python3
"""
autoscale_isolate.py — run ONLY the autoscaler step, with nothing to swallow
errors.  The overnight harness catches exceptions per-run and prints a clean
"Done", hiding the real failure.  This script calls _build_global_peak_plan
directly so any exception prints in full.

Usage:
    python3 autoscale_isolate.py
    python3 autoscale_isolate.py --trace simulator_ready_trace.csv --spec-dir sim_specs

Run it from the same directory the overnight sweep runs in, with the SAME
trace file the sweep uses for the failing config.
"""
from __future__ import annotations
import argparse
import sys
import traceback
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default="simulator_ready_trace.csv")
    ap.add_argument("--spec-dir", default="sim_specs")
    ap.add_argument("--epoch-length", type=int, default=900)
    ap.add_argument("--target-util", type=float, default=0.95)
    ap.add_argument("--max-rows", type=int, default=16_000_000)
    ap.add_argument("--max-mult", type=float, default=1_500_000.0)
    ap.add_argument("--max-drop", type=float, default=0.02)
    ap.add_argument("--search-steps", type=int, default=9)
    ap.add_argument("--n-epochs", type=int, default=None,
                    help="Override number_of_epoch passed to _build_global_peak_plan. "
                         "The harness passes args.epoch (default 96) — which may differ "
                         "from the real distinct-epoch count in a sparse trace. Pass 96 "
                         "here to reproduce exactly what the harness does.")
    args = ap.parse_args()

    print("=" * 70)
    print("AUTOSCALE ISOLATION — running _build_global_peak_plan directly")
    print("=" * 70)
    print(f"trace={args.trace} spec_dir={args.spec_dir} "
          f"epoch_length={args.epoch_length}")
    print(f"target_util={args.target_util} max_rows={args.max_rows} "
          f"max_mult={args.max_mult} max_drop={args.max_drop}")
    print()

    # ── Step 1: import the real autoscaler ───────────────────────────────────
    print("[1] Importing simulator_LLM ...")
    try:
        from simulator_LLM import _build_global_peak_plan
        print("    OK — _build_global_peak_plan imported")
    except Exception:
        print("    FAILED to import simulator_LLM:")
        traceback.print_exc()
        sys.exit(1)

    # ── Step 2: build the dry-run simulator ──────────────────────────────────
    print("[2] Building LLM_Simulator (dry sim) ...")
    try:
        from Rate_Flow_Sim_v2 import LLM_Simulator
        dry_sim = LLM_Simulator(spec_dir=args.spec_dir,
                                epoch_length=args.epoch_length, debug=False)
        n_dc = len(dry_sim.datacenters)
        n_units = sum(len(dc.units) for dc in dry_sim.datacenters.values())
        print(f"    OK — {n_dc} DCs, {n_units} units")
    except Exception:
        print("    FAILED to build simulator:")
        traceback.print_exc()
        sys.exit(1)

    # ── Step 3: load + group the trace ───────────────────────────────────────
    print("[3] Loading trace ...")
    try:
        trace = pd.read_csv(args.trace)
        print(f"    OK — {len(trace):,} rows, columns: {list(trace.columns)}")
    except Exception:
        print("    FAILED to load trace:")
        traceback.print_exc()
        sys.exit(1)

    # Group into epochs the way the harness does.
    epoch_col = None
    for c in ("epoch", "epoch_idx", "epoch_index"):
        if c in trace.columns:
            epoch_col = c
            break
    if epoch_col is None:
        for c in ("time_index", "time", "timestamp", "seconds"):
            if c in trace.columns:
                trace["epoch"] = (pd.to_numeric(trace[c], errors="coerce").fillna(0)
                                  // args.epoch_length).astype(int)
                epoch_col = "epoch"
                print(f"    Derived epoch column from '{c}'")
                break
    if epoch_col is None:
        print("    FAILED — no epoch or time column in trace.")
        print(f"    Columns present: {list(trace.columns)}")
        sys.exit(1)

    grouped = trace.groupby(epoch_col)
    n_epochs_real = len(grouped.groups)
    group_keys = sorted(grouped.groups.keys())
    print(f"    Grouped into {n_epochs_real} DISTINCT epochs on column '{epoch_col}'")
    print(f"    Epoch key range: min={group_keys[0]} max={group_keys[-1]} "
          f"-> max+1 = {int(group_keys[-1]) + 1}")
    if int(group_keys[-1]) + 1 != n_epochs_real:
        missing = sorted(set(range(int(group_keys[0]), int(group_keys[-1]) + 1))
                         - set(int(k) for k in group_keys))
        print(f"    *** SPARSE TRACE: {len(missing)} epoch indices in the range "
              f"have NO rows: {missing}")
        print(f"    *** The harness prints 'max+1' = {int(group_keys[-1]) + 1} epochs, "
              f"but only {n_epochs_real} actually exist.")

    # number_of_epoch: replicate the harness (args.epoch, default 96) unless
    # overridden.  This is THE value the harness passes — and it may not match
    # the real distinct-epoch count above.
    number_of_epoch = args.n_epochs if args.n_epochs is not None else n_epochs_real
    if args.n_epochs is not None and args.n_epochs != n_epochs_real:
        print(f"    Using number_of_epoch={number_of_epoch} (OVERRIDE — "
              f"reproducing harness; real distinct count is {n_epochs_real})")
    else:
        print(f"    Using number_of_epoch={number_of_epoch}")

    # ── Step 4: call the autoscaler — THIS is where the swallowed error is ───
    print("[4] Calling _build_global_peak_plan (the step that fails silently) ...")
    print("    --- any exception below is the REAL error the harness hides ---")
    try:
        plan = _build_global_peak_plan(
            dry_sim=dry_sim,
            grouped_trace=grouped,
            number_of_epoch=number_of_epoch,
            target_util=args.target_util,
            max_multiplier=args.max_mult,
            max_rows=args.max_rows,
            max_drop=args.max_drop,
            search_steps=args.search_steps,
        )
    except Exception:
        print()
        print("    *** EXCEPTION IN _build_global_peak_plan — THIS IS THE BUG ***")
        traceback.print_exc()
        sys.exit(1)

    # ── Step 5: report the plan ──────────────────────────────────────────────
    print()
    print("[5] _build_global_peak_plan RETURNED (no exception). Plan contents:")
    if not isinstance(plan, dict):
        print(f"    Unexpected return type: {type(plan)} -> {plan!r}")
        sys.exit(1)
    for k, v in plan.items():
        print(f"      {k:24s} = {v}")

    if not plan.get("enabled", False):
        print()
        print("    >>> Plan is DISABLED. reason =", plan.get("reason", "<none>"))
        print("    >>> The autoscaler returned cleanly but produced no scaling.")
        print("    >>> THIS is why the run does nothing and reports Done.")
    else:
        print()
        print("    >>> Plan enabled — autoscaler itself is fine.")
        print("    >>> If the sweep still no-ops, the failure is AFTER autoscale")
        print("    >>> (in the epoch loop / framework call). Tell me and we look there.")


if __name__ == "__main__":
    main()