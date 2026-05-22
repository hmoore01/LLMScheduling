#!/usr/bin/env python3
"""
ttft_diagnostic.py — isolate WHY LA_Hyper's TTFT is far higher than the heuristics'.

The approach: take ONE real epoch of workload, build a schedule + power plan with
each strategy, run them all through the SAME simulator, and decompose the
resulting TTFT so the cause is visible rather than guessed.

TTFT in Rate_Flow_Sim_v2 is, per request:
    ttft_s = (net_latency_ms + wait_ms + prefill_ms) / 1000
where
    wait_ms    = queue waiting time  (start_ms - arrival_ms)
    prefill_ms = model compute time  (ms_per_token * tokens)
    net_latency_ms = inter-DC network hop

If LA_Hyper's TTFT is high because of wait_ms  -> it's a ROUTING / queue-depth problem.
If it's high because of net_latency_ms         -> it's a LOCALITY problem (routing far).
If it's high because of prefill_ms              -> it's a MODEL-VARIANT problem.
If it's high because of DROPPED requests        -> it's an OVERSUBSCRIPTION problem.

The script prints a per-strategy, per-component breakdown plus per-DC queue depth
so you can see exactly which bucket the gap lives in.

Usage:
    python ttft_diagnostic.py                      # default: epoch 0, 8 DCs
    python ttft_diagnostic.py --epoch 5 --num-dcs 6
    python ttft_diagnostic.py --epochs 0,1,2       # average across several epochs

Requires (same dir): Rate_Flow_Sim_v2.py, simulator_ready_trace.csv, sim_specs/,
and the framework files LA_Hyper_DDQN.py, Helix.py.
"""

from __future__ import annotations
import argparse
import os
import sys
import numpy as np
import pandas as pd

import Rate_Flow_Sim_v2 as Rate_Flow_Sim

# Import the REAL autoscale logic from the harness so the diagnostic scales the
# workload exactly the way the overnight runs do.  simulator_LLM.py guards all
# heavy code under `if __name__ == "__main__"`, so importing it is side-effect free.
try:
    from simulator_LLM import _apply_autoscale_multiplier, _build_global_peak_plan
    _HAVE_REAL_AUTOSCALE = True
except Exception as _e:   # pragma: no cover
    _HAVE_REAL_AUTOSCALE = False
    print(f"[WARN] Could not import real autoscale from simulator_LLM: {_e}")
    print("[WARN] Falling back to approximate scaling — numbers may not match overnight runs.")


# ─────────────────────────────────────────────────────────────────────────────
# Workload loading  (mirrors simulator_LLM.py)
# ─────────────────────────────────────────────────────────────────────────────
def _derive_arrival_ms(df: pd.DataFrame, epoch_length_s: int = 900) -> pd.Series:
    epoch_max_ms = float(max(1, int(epoch_length_s))) * 1000.0
    if "arrival_ms" in df.columns:
        return pd.to_numeric(df["arrival_ms"], errors="coerce").fillna(0.0).clip(0, epoch_max_ms)
    for col in ("time_index", "time", "timestamp"):
        if col in df.columns:
            t = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            if float(t.max()) <= float(epoch_length_s) + 1e-9:
                return (t * 1000.0).clip(0, epoch_max_ms)
            return t.clip(0, epoch_max_ms)
    # No time column — spread uniformly
    return pd.Series(np.linspace(0, epoch_max_ms, len(df)), index=df.index)


def load_epoch(trace_path: str, epoch_idx: int, epoch_length: int) -> pd.DataFrame:
    if not os.path.exists(trace_path):
        raise FileNotFoundError(f"Workload CSV not found: {trace_path}")
    trace = pd.read_csv(trace_path)
    # Identify the epoch column
    epoch_col = None
    for c in ("epoch", "epoch_idx", "epoch_index"):
        if c in trace.columns:
            epoch_col = c
            break
    if epoch_col is None:
        # Derive epoch from a time column
        for c in ("time_index", "time", "timestamp"):
            if c in trace.columns:
                trace["epoch"] = (pd.to_numeric(trace[c], errors="coerce").fillna(0)
                                  // epoch_length).astype(int)
                epoch_col = "epoch"
                break
    if epoch_col is None:
        raise ValueError("Could not find an epoch or time column in the trace.")

    epoch_df = trace[trace[epoch_col] == epoch_idx].copy()
    if epoch_df.empty:
        raise ValueError(f"No rows for epoch {epoch_idx}. "
                         f"Available: {sorted(trace[epoch_col].unique())[:10]}...")
    epoch_df["arrival_ms"] = _derive_arrival_ms(epoch_df, epoch_length)
    return epoch_df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Workload-scaling  — uses the REAL autoscaler from simulator_LLM.py
# ─────────────────────────────────────────────────────────────────────────────
def autoscale(epoch_df: pd.DataFrame, multiplier: float, epoch_idx: int,
              epoch_length: int, count_cap: int = None) -> pd.DataFrame:
    """Scale an epoch's workload using the harness's actual autoscale logic.

    `multiplier` is the SINGLE total multiplier the overnight run computed
    (e.g. 7937.5).  The real _apply_autoscale_multiplier splits it internally
    into a row-replication count and a per-row token scale — exactly as the
    overnight runs do — so the TTFT this diagnostic measures is comparable to
    the chart.

    Pass multiplier=1.0 to skip scaling.
    """
    if multiplier <= 1.0:
        return epoch_df

    # The v2 sim / autoscaler expects num_tokens; normalize the column name.
    df = epoch_df.copy()
    if "num_tokens" not in df.columns and "tokens" in df.columns:
        df = df.rename(columns={"tokens": "num_tokens"})

    if _HAVE_REAL_AUTOSCALE:
        scaled, count_mult, remainder_scale = _apply_autoscale_multiplier(
            df, multiplier, epoch_idx,
            epoch_length_s=epoch_length, count_cap=count_cap,
        )
        print(f"  [Auto-Scale] multiplier {multiplier:.1f} → "
              f"Requests x{count_mult}, Tokens x{remainder_scale:.3f}")
        return scaled

    # Fallback (only if import failed) — approximate, flagged loudly upstream.
    count_mult = max(1, int(np.floor(multiplier * 0.95)))
    remainder = multiplier / count_mult
    scaled = pd.concat([df] * count_mult, ignore_index=True)
    scaled["num_tokens"] = (pd.to_numeric(scaled["num_tokens"], errors="coerce")
                            .fillna(1.0) * remainder).round().astype(int)
    return scaled


# ─────────────────────────────────────────────────────────────────────────────
# Per-DC capacity  (the SAME formula Helix / the patched LA_Hyper use)
# ─────────────────────────────────────────────────────────────────────────────
def dc_capacity_tokens(sim, epoch_length: int) -> dict:
    epoch_ms = float(epoch_length) * 1000.0
    caps = {}
    for dc_id, dc in sim.datacenters.items():
        total = 0.0
        for unit in getattr(dc, "units", []):
            best_tpm = 0.0
            for rec in getattr(unit, "model_perf", {}).values():
                mpt = float(rec.get("ms_per_token", 0.0))
                if mpt > 0:
                    best_tpm = max(best_tpm, 1.0 / mpt)
            total += best_tpm * epoch_ms
        caps[int(dc_id)] = max(total, 1.0)
    return caps


# ─────────────────────────────────────────────────────────────────────────────
# Routing strategies — each returns (schedule_plan, power_plan, request_df)
# ─────────────────────────────────────────────────────────────────────────────
FIXED_VARIANT = "_FP16 (Base)_B16"


def _request_rows(epoch_df: pd.DataFrame) -> list:
    rows = []
    src_col = "source_dc_id" if "source_dc_id" in epoch_df.columns else "source_dc"
    mdl_col = "model_type" if "model_type" in epoch_df.columns else "model"
    tok_col = "num_tokens" if "num_tokens" in epoch_df.columns else "tokens"
    for r in epoch_df.itertuples(index=False):
        rows.append({
            "source_dc": int(getattr(r, src_col)),
            "model": f"{getattr(r, mdl_col)}{FIXED_VARIANT}",
            "arrival_ms": float(getattr(r, "arrival_ms", 0.0)),
            "tokens": max(1, int(getattr(r, tok_col))),
        })
    return rows


def route_helix(epoch_df, dc_ids, caps):
    """Greedy least-loaded-relative-to-capacity, per request. All DCs ON."""
    pending = {d: 0.0 for d in dc_ids}
    plan_map = {}
    rows = _request_rows(epoch_df)
    for idx, req in enumerate(rows):
        best = min(dc_ids, key=lambda d: pending[d] / caps.get(d, 1.0))
        pending[best] += req["tokens"]
        plan_map[idx] = best
    active = {d for d in dc_ids if pending[d] > 0}
    power = {d: {"all": "ON" if d in active else "OFF"} for d in dc_ids}
    return {"map": plan_map}, power, pd.DataFrame(rows), pending


def route_lahyper_style(epoch_df, dc_ids, caps, power_fraction=1.0,
                        capacity_aware=True):
    """Static proportional split — the LA_Hyper build_schedule_map approach.

    capacity_aware=True   -> weight by real token throughput  (patched LA_Hyper)
    capacity_aware=False  -> weight by node count             (original bug)
    power_fraction        -> fraction of DCs powered on (1.0 = all; <1 = eco)
    """
    rows = _request_rows(epoch_df)
    n_req = len(rows)
    dc_arr = np.array(dc_ids)

    # Decide which DCs are powered on (highest-capacity first)
    n_on = max(1, int(round(len(dc_ids) * power_fraction)))
    on_order = sorted(dc_ids, key=lambda d: caps.get(d, 0.0), reverse=True)
    powered = set(on_order[:n_on])

    if capacity_aware:
        weights = np.array([caps.get(d, 1.0) if d in powered else 0.0 for d in dc_ids])
    else:
        # Node-count proxy: every powered DC counts equally (the original bug)
        weights = np.array([1.0 if d in powered else 0.0 for d in dc_ids])

    if weights.sum() <= 0:
        weights = np.ones(len(dc_ids))
    weights = weights / weights.sum()

    # Largest-remainder integer allocation
    raw = weights * n_req
    counts = np.floor(raw).astype(np.int64)
    rem = n_req - int(counts.sum())
    counts[np.argsort(raw - counts)[::-1][:rem]] += 1

    plan_map = {}
    ptr = 0
    pending = {d: 0.0 for d in dc_ids}
    for di, cnt in enumerate(counts):
        for _ in range(int(cnt)):
            if ptr < n_req:
                plan_map[ptr] = int(dc_arr[di])
                pending[int(dc_arr[di])] += rows[ptr]["tokens"]
                ptr += 1
    power = {d: {"all": "ON" if d in powered else "OFF"} for d in dc_ids}
    return {"map": plan_map}, power, pd.DataFrame(rows), pending


# ─────────────────────────────────────────────────────────────────────────────
# REAL LA_Hyper pipeline probe
# ─────────────────────────────────────────────────────────────────────────────
def probe_real_lahyper(epoch_df_unscaled: pd.DataFrame, epoch_idx: int,
                       multiplier: float, epoch_length: int, spec_dir: str,
                       dc_ids: list, count_cap: int = None) -> list:
    """Call LA_Hyper's ACTUAL milp_optimizer and capture what it records.

    This is the decisive test.  The diagnostic's other strategies use
    *idealized* routing; this one runs the real trained agent.  We monkey-patch
    the Pareto tracker's record_solution so every (mode, metrics, schedule_plan,
    power_plan) the agent produces is captured, then we re-run each captured
    plan through a CLEAN sim.run_epoch and compare:

        agent-reported TTFT   vs   clean-sim TTFT of the identical plan

    If they match and both are high  -> the agent generates bad plans.
    If agent-reported is high but clean-sim of the same plan is low
                                      -> the agent's reporting/eval path is wrong.

    Returns a list of dicts, one per captured solution.
    """
    try:
        import LA_Hyper_DDQN as LAH
    except Exception as e:
        print(f"[probe] Could not import LA_Hyper_DDQN: {e}")
        return []

    # The harness passes UNSCALED epoch_data to milp_optimizer; the framework
    # itself does NOT autoscale — autoscaling happens upstream in simulator_LLM.
    # So feed milp_optimizer the SAME scaled frame the diagnostic uses, to match.
    scaled = autoscale(epoch_df_unscaled, multiplier, epoch_idx,
                       epoch_length, count_cap=count_cap)

    # ── Monkey-patch record_solution to capture every recorded solution ──────
    captured = []
    tracker = getattr(LAH, "_PARETO_TRACKER", None)
    if tracker is None:
        # _PARETO_TRACKER missing from the imported module.  This is almost
        # never a real bug in the file — it's usually a stale import.  Report
        # enough detail to tell which.
        import os as _os
        mod_path = getattr(LAH, "__file__", "<unknown>")
        has_attr = "_PARETO_TRACKER" in dir(LAH)
        print("[probe] LA_Hyper has no _PARETO_TRACKER — cannot capture.")
        print(f"[probe]   imported module file : {mod_path}")
        print(f"[probe]   '_PARETO_TRACKER' in dir(module): {has_attr}")
        print(f"[probe]   module attributes (sample): "
              f"{[a for a in dir(LAH) if not a.startswith('__')][:12]}")
        print("[probe] LIKELY CAUSES:")
        print("[probe]   1. Stale bytecode cache — run: rm -rf __pycache__")
        print("[probe]   2. A different LA_Hyper_DDQN.py earlier on sys.path")
        print("[probe]   3. Import died partway — run: "
              "python3 -c 'import LA_Hyper_DDQN' to see the real error")
        print(f"[probe]   (check the file at the path above actually defines "
              f"_PARETO_TRACKER at module level)")
        return []

    orig_record = tracker.record_solution

    # Signature-agnostic wrapper: record_solution's signature may grow new
    # keyword args (e.g. `force`).  Accept *args/**kwargs and forward verbatim
    # so the probe never breaks when LA_Hyper's API changes.
    def _capturing_record(metrics, weights, power_plan, mode_name="Scan",
                          *args, **kwargs):
        captured.append({
            "mode": mode_name,
            "metrics": dict(metrics) if metrics else {},
            "power_plan": power_plan,
        })
        return orig_record(metrics, weights, power_plan, mode_name,
                           *args, **kwargs)

    tracker.record_solution = _capturing_record

    # ── Call the real milp_optimizer ─────────────────────────────────────────
    node_properties = {int(d): {"id": int(d)} for d in dc_ids}
    epoch_summary = {
        "node_types": [0, 1, 2, 3, 4, 5],
        "datacenters": dc_ids,
        "avg_input_tokens": 100,
        "avg_output_tokens": 100,
        "spec_dir": spec_dir,
        "epoch_length": epoch_length,
    }

    try:
        LAH.milp_optimizer(
            epoch_data=scaled, epoch_idx=epoch_idx,
            node_properties=node_properties, epoch_summary=epoch_summary,
        )
    except Exception as e:
        import traceback
        print(f"[probe] milp_optimizer raised: {e}")
        traceback.print_exc()
        tracker.record_solution = orig_record
        return []
    finally:
        tracker.record_solution = orig_record

    print(f"[probe] LA_Hyper attempted to record {len(captured)} solutions "
          f"for epoch {epoch_idx}")
    print(f"[probe] (this list includes solutions the drop-rate filter may "
          f"have REJECTED from the actual front — the crosscheck below shows "
          f"every attempt so you can still see what the agent generated)")
    return captured


def crosscheck_captured(captured: list, scaled_epoch_df: pd.DataFrame,
                        spec_dir: str, epoch_length: int, epoch_idx: int) -> None:
    """For each captured solution, re-run its power_plan through a clean sim
    and compare the clean TTFT to what LA_Hyper reported."""
    if not captured:
        return
    print("\n" + "=" * 78)
    print("REAL LA_Hyper PIPELINE — agent-reported vs clean-sim crosscheck")
    print("=" * 78)
    print(f"\n{'Mode':<20}{'agent TTFT':>13}{'clean-sim TTFT':>16}{'match?':>10}")
    print("-" * 59)

    rows = _request_rows(scaled_epoch_df)
    req_df = pd.DataFrame(rows)

    for sol in captured:
        agent_ttft = float(sol["metrics"].get("avg_ttft",
                            sol["metrics"].get("avg_ttft_sec", 0.0)))
        power_plan = sol["power_plan"]
        # Re-run the SAME power plan through a clean sim with default routing
        # (empty schedule_plan = simulator's own default placement).
        try:
            sim = Rate_Flow_Sim.LLM_Simulator(spec_dir=spec_dir,
                                              epoch_length=epoch_length, debug=False)
            m, _, _ = sim.run_epoch(epoch_idx, req_df, {}, power_plan or {})
            clean_ttft = float(m.get("avg_ttft", 0.0))
        except Exception as e:
            print(f"{sol['mode']:<20}{agent_ttft:>13.3f}   (clean run failed: {e})")
            continue
        delta = abs(agent_ttft - clean_ttft)
        match = "yes" if delta < 0.5 else "NO"
        print(f"{sol['mode']:<20}{agent_ttft:>13.3f}{clean_ttft:>16.3f}{match:>10}")

    print("\n  If agent TTFT >> clean-sim TTFT for the same power plan:")
    print("    → the agent's schedule_plan (routing) is the culprit, not power.")
    print("  If agent TTFT ≈ clean-sim TTFT and both are high:")
    print("    → the power plan itself (DCs off / wrong variant) drives TTFT.")
    print("  If both are low (~4s) but the CHART shows 20s:")
    print("    → the chart is stale or the harness parse is wrong, NOT the agent.")


# ─────────────────────────────────────────────────────────────────────────────
# TTFT decomposition from detailed_results
# ─────────────────────────────────────────────────────────────────────────────
def decompose(detailed_results: list, request_df: pd.DataFrame) -> dict:
    """Break the epoch's TTFT into wait / prefill+net / drops, per DC.

    TTFT in Rate_Flow_Sim is: ttft_s = (net_latency_ms + wait_ms + prefill_ms)/1000
    The detailed record gives ttft_s and start_ms; wait = start - arrival.
    We therefore derive the (prefill + net) component as ttft_s - wait, rather
    than reading exec_ms — exec_ms is the FULL request execution time (all
    tokens, generation included), which is NOT the time-to-first-token and
    massively overstates the prefill share.
    """
    arrivals = request_df["arrival_ms"].to_numpy() if "arrival_ms" in request_df else None

    n_total = len(detailed_results)
    n_drop = 0
    wait_ms, prefill_net_ms, ttft_s, exec_ms_list = [], [], [], []
    per_dc = {}   # dc_id -> list of wait_ms

    for i, rec in enumerate(detailed_results):
        if not isinstance(rec, dict):
            continue
        if rec.get("dropped", False):
            n_drop += 1
            continue
        dc = rec.get("dc_id")
        t  = float(rec.get("ttft_s", 0.0))
        ttft_s.append(t)
        s  = rec.get("start_ms")
        ex = rec.get("exec_ms")
        if ex is not None:
            exec_ms_list.append(float(ex))

        # wait = start - arrival
        w = None
        if s is not None and arrivals is not None and i < len(arrivals):
            w = max(0.0, float(s) - float(arrivals[i]))
        if w is not None:
            wait_ms.append(w)
            per_dc.setdefault(dc, []).append(w)
            # prefill+net is whatever's left of TTFT after the queue wait
            prefill_net_ms.append(max(0.0, t * 1000.0 - w))

    def _stat(arr):
        if not arr:
            return dict(mean=0.0, p50=0.0, p95=0.0, max=0.0)
        a = np.array(arr, dtype=np.float64)
        return dict(mean=float(a.mean()), p50=float(np.percentile(a, 50)),
                    p95=float(np.percentile(a, 95)), max=float(a.max()))

    return {
        "n_total": n_total,
        "n_drop": n_drop,
        "drop_pct": 100.0 * n_drop / max(1, n_total),
        "ttft_s": _stat(ttft_s),
        "wait_ms": _stat(wait_ms),
        "prefill_ms": _stat(prefill_net_ms),   # now prefill+net, derived from TTFT
        "exec_ms": _stat(exec_ms_list),        # full request compute (for reference)
        "per_dc_wait": {dc: _stat(v) for dc, v in per_dc.items()},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Run one strategy through the simulator
# ─────────────────────────────────────────────────────────────────────────────
def run_strategy(name, schedule_plan, power_plan, request_df,
                 spec_dir, epoch_length, epoch_idx, pending):
    sim = Rate_Flow_Sim.LLM_Simulator(spec_dir=spec_dir, epoch_length=epoch_length,
                                      debug=False)
    metrics, details, dc_usage = sim.run_epoch(epoch_idx, request_df,
                                               schedule_plan, power_plan)
    decomp = decompose(details, request_df)
    decomp["name"] = name
    decomp["sim_avg_ttft"] = float(metrics.get("avg_ttft", 0.0))
    decomp["carbon"] = float(metrics.get("carbon_emissions", 0.0))
    decomp["energy_cost"] = float(metrics.get("energy_cost", 0.0))
    decomp["pending_tokens"] = dict(pending)
    decomp["n_dcs_used"] = sum(1 for v in pending.values() if v > 0)
    return decomp


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────
def print_report(results: list, caps: dict):
    print("\n" + "=" * 78)
    print("TTFT DIAGNOSTIC REPORT")
    print("=" * 78)

    # ── Headline comparison ──────────────────────────────────────────────────
    print(f"\n{'Strategy':<26}{'avg TTFT(s)':>13}{'wait p50(s)':>13}"
          f"{'wait p95(s)':>13}{'drop%':>9}{'DCs used':>10}")
    print("-" * 84)
    for r in results:
        print(f"{r['name']:<26}{r['sim_avg_ttft']:>13.3f}"
              f"{r['wait_ms']['p50']/1000:>13.3f}{r['wait_ms']['p95']/1000:>13.3f}"
              f"{r['drop_pct']:>8.1f}%{r['n_dcs_used']:>10}")

    # ── TTFT component decomposition ─────────────────────────────────────────
    print(f"\n{'TTFT component breakdown (mean, seconds)':<50}")
    print(f"{'Strategy':<26}{'wait':>11}{'prefill+net':>13}{'= ttft':>10}"
          f"{'(exec_ms)':>12}")
    print("-" * 72)
    for r in results:
        w  = r['wait_ms']['mean'] / 1000.0
        pn = r['prefill_ms']['mean'] / 1000.0
        ex = r['exec_ms']['mean'] / 1000.0
        print(f"{r['name']:<26}{w:>11.3f}{pn:>13.3f}{r['ttft_s']['mean']:>10.3f}"
              f"{ex:>12.3f}")
    print("  wait        = queue time (start - arrival)")
    print("  prefill+net = ttft - wait  (first-token compute + network hop)")
    print("  exec_ms     = FULL request execution (all tokens) — reference only,")
    print("                NOT part of TTFT.  If exec_ms >> ttft, requests are")
    print("                large and TTFT is a small fraction of total service.")
    print("  → If wait dominates: routing/queue problem.")
    print("  → If prefill+net dominates with wait≈0: compute/hardware problem,")
    print("    routing can only help by landing requests on faster nodes.")

    # ── Per-DC queue depth ───────────────────────────────────────────────────
    for r in results:
        print(f"\nPer-DC mean queue wait — {r['name']}:")
        pdw = r["per_dc_wait"]
        if not pdw:
            print("   (no per-DC data)")
            continue
        for dc in sorted(pdw.keys(), key=lambda x: (x is None, x)):
            cap = caps.get(dc, 0.0)
            load = r["pending_tokens"].get(dc, 0.0)
            util = (load / cap * 100.0) if cap > 0 else 0.0
            print(f"   DC {str(dc):<4} wait_mean={pdw[dc]['mean']/1000:>8.2f}s  "
                  f"wait_p95={pdw[dc]['p95']/1000:>8.2f}s  "
                  f"load/cap={util:>7.1f}%")

    # ── Verdict ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    by_name = {r["name"]: r for r in results}
    helix = by_name.get("Helix (per-req balanced)")
    lah   = by_name.get("LA_Hyper (capacity-aware)")
    if helix and lah:
        gap = lah["sim_avg_ttft"] - helix["sim_avg_ttft"]
        print(f"\nLA_Hyper - Helix TTFT gap: {gap:+.2f}s")
        wait_gap = (lah['wait_ms']['mean'] - helix['wait_ms']['mean']) / 1000.0
        pn_gap   = (lah['prefill_ms']['mean'] - helix['prefill_ms']['mean']) / 1000.0
        drop_gap = lah['drop_pct'] - helix['drop_pct']
        print(f"  ├─ attributable to queue wait      : {wait_gap:+.2f}s")
        print(f"  ├─ attributable to prefill+network : {pn_gap:+.2f}s")
        print(f"  └─ drop-rate difference            : {drop_gap:+.1f} pct points")

        # How big is TTFT relative to full request service time?
        lah_ttft  = lah['ttft_s']['mean']
        lah_exec  = lah['exec_ms']['mean'] / 1000.0
        if lah_exec > 0:
            ttft_frac = 100.0 * lah_ttft / lah_exec
            print(f"\n  TTFT is {ttft_frac:.0f}% of full request execution time "
                  f"(ttft={lah_ttft:.1f}s vs exec={lah_exec:.1f}s).")

        if abs(wait_gap) > abs(pn_gap) * 2 and abs(wait_gap) > 0.5:
            print("\n  → DOMINANT CAUSE: queue wait. LA_Hyper builds deeper queues.")
            if lah["n_dcs_used"] < helix["n_dcs_used"]:
                print(f"    LA_Hyper used {lah['n_dcs_used']} DCs vs Helix's "
                      f"{helix['n_dcs_used']} — concentrating load.")
                print("    FIX DIRECTION: power on more DCs / floor the power sliders.")
            else:
                lah_utils = [lah['pending_tokens'].get(d,0)/caps.get(d,1) for d in caps]
                hel_utils = [helix['pending_tokens'].get(d,0)/caps.get(d,1) for d in caps]
                ls, hs = np.std(lah_utils), np.std(hel_utils)
                print(f"    Same DC count, utilization spread: "
                      f"LA_Hyper σ={ls:.2f} vs Helix σ={hs:.2f}.")
                if ls > hs * 1.5:
                    print("    FIX DIRECTION: static split is uneven — make routing "
                          "adaptive (iterative, like Helix).")
                else:
                    print("    Spread comparable — gap is ordering/arrival timing.")
        elif abs(pn_gap) > abs(wait_gap) * 2 and abs(pn_gap) > 0.5:
            print("\n  → DOMINANT CAUSE: prefill+network, with queue wait ≈ 0.")
            print("    This is NOT a load-balancing problem.  Requests are landing")
            print("    on slower nodes (higher ms_per_token) or routing farther")
            print("    (higher network hop).  Capacity-proportional routing sends")
            print("    a share of traffic to low-throughput DCs by design.")
            print("    FIX DIRECTION: route by per-request SPEED — pick the DC that")
            print("    executes THIS model fastest — not by capacity proportion.")
        elif drop_gap > 20:
            print("\n  → DOMINANT CAUSE: drops. Workload exceeds capacity.")
        else:
            print("\n  → No single dominant cause; gap is small / spread across "
                  "components.  If the gap here (<~2s) is far smaller than the")
            print("    gap in the overnight chart (10-20s), the chart difference")
            print("    is NOT explained by routing — look at how each framework")
            print("    chooses model variants or reports avg_ttft.")

    # Also report the bug-vs-fixed comparison if present
    bug = by_name.get("LA_Hyper (node-count BUG)")
    fix = by_name.get("LA_Hyper (capacity-aware)")
    if bug and fix:
        d = bug["sim_avg_ttft"] - fix["sim_avg_ttft"]
        print(f"\nCapacity-aware fix impact: {d:+.2f}s "
              f"({bug['sim_avg_ttft']:.2f}s → {fix['sim_avg_ttft']:.2f}s)")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Diagnose LA_Hyper vs heuristic TTFT gap")
    ap.add_argument("--trace", default="simulator_ready_trace.csv")
    ap.add_argument("--spec-dir", default="sim_specs")
    ap.add_argument("--epoch", type=int, default=0)
    ap.add_argument("--epochs", default=None,
                    help="comma-separated epoch list; overrides --epoch, averages results")
    ap.add_argument("--epoch-length", type=int, default=900)
    ap.add_argument("--multiplier", type=float, default=1.0,
                    help="SINGLE total autoscale multiplier (the value the overnight "
                         "run printed, e.g. 7937.5).  Split internally into "
                         "row-replication and token-scale by the real autoscaler.")
    ap.add_argument("--count-cap", type=int, default=None,
                    help="optional cap on row-replication count (matches harness)")
    ap.add_argument("--eco-fraction", type=float, default=0.5,
                    help="fraction of DCs the LA_Hyper-eco variant powers on")
    ap.add_argument("--probe-real", action="store_true",
                    help="ALSO call LA_Hyper's real milp_optimizer and crosscheck "
                         "the plans it generates against a clean sim. This is the "
                         "decisive test for whether the agent's own plans cause "
                         "the high TTFT. Slow — runs on the first epoch only.")
    args = ap.parse_args()

    epochs = ([int(x) for x in args.epochs.split(",")]
              if args.epochs else [args.epoch])

    # Probe sim once for DC list + capacities
    probe = Rate_Flow_Sim.LLM_Simulator(spec_dir=args.spec_dir,
                                        epoch_length=args.epoch_length, debug=False)
    dc_ids = sorted(int(d) for d in probe.datacenters.keys())
    caps = dc_capacity_tokens(probe, args.epoch_length)
    del probe

    print(f"Discovered {len(dc_ids)} DCs: {dc_ids}")
    print(f"Per-DC capacity (tokens/epoch): "
          f"{ {d: f'{caps[d]:.2e}' for d in dc_ids} }")
    cap_spread = np.std(list(caps.values())) / max(1e-9, np.mean(list(caps.values())))
    print(f"Capacity heterogeneity (CV): {cap_spread:.2f}  "
          f"({'HIGH — capacity-aware routing matters' if cap_spread > 0.3 else 'low'})")

    all_runs = {}
    for ep in epochs:
        print(f"\n--- Loading epoch {ep} ---")
        epoch_df_raw = load_epoch(args.trace, ep, args.epoch_length)
        epoch_df = autoscale(epoch_df_raw, args.multiplier, ep,
                             args.epoch_length, count_cap=args.count_cap)
        print(f"Epoch {ep}: {len(epoch_df)} requests after scaling "
              f"(total multiplier={args.multiplier})")

        strategies = [
            ("Helix (per-req balanced)",
             *route_helix(epoch_df, dc_ids, caps)),
            ("LA_Hyper (capacity-aware)",
             *route_lahyper_style(epoch_df, dc_ids, caps,
                                  power_fraction=1.0, capacity_aware=True)),
            ("LA_Hyper (node-count BUG)",
             *route_lahyper_style(epoch_df, dc_ids, caps,
                                  power_fraction=1.0, capacity_aware=False)),
            (f"LA_Hyper-eco ({int(args.eco_fraction*100)}% DCs on)",
             *route_lahyper_style(epoch_df, dc_ids, caps,
                                  power_fraction=args.eco_fraction,
                                  capacity_aware=True)),
        ]

        for name, sched, power, req_df, pending in strategies:
            r = run_strategy(name, sched, power, req_df,
                             args.spec_dir, args.epoch_length, ep, pending)
            all_runs.setdefault(name, []).append(r)

        # ── Real LA_Hyper pipeline probe (only on the first epoch — it is
        #    slow and the point is to localize the cause, not benchmark) ──────
        if args.probe_real and ep == epochs[0]:
            captured = probe_real_lahyper(
                epoch_df_raw, ep, args.multiplier, args.epoch_length,
                args.spec_dir, dc_ids, count_cap=args.count_cap,
            )
            crosscheck_captured(captured, epoch_df, args.spec_dir,
                                args.epoch_length, ep)

    # Average across epochs
    results = []
    for name, runs in all_runs.items():
        if len(runs) == 1:
            results.append(runs[0])
        else:
            # Average the scalar fields; keep first run's per-DC detail
            avg = dict(runs[0])
            avg["sim_avg_ttft"] = np.mean([r["sim_avg_ttft"] for r in runs])
            avg["drop_pct"]     = np.mean([r["drop_pct"] for r in runs])
            for comp in ("wait_ms", "prefill_ms", "ttft_s", "exec_ms"):
                for stat in ("mean", "p50", "p95", "max"):
                    avg[comp][stat] = np.mean([r[comp][stat] for r in runs])
            results.append(avg)

    print_report(results, caps)


if __name__ == "__main__":
    main()