#!/usr/bin/env python3
"""
sweep_thresholds.py — choose the eco-agent drop gate and TTFT SLA offline.

The full simulator takes ~90 min per run, so sweeping a grid of
(drop_gate x TTFT_SLA) with real runs is infeasible.  But those two
thresholds only affect *selection*, not the candidates themselves — so this
script replays every threshold combination against a candidate pool that was
dumped once during a normal run.  Fast (seconds) and faithful.

USAGE
-----
1. Produce the candidate dump (one normal LAHyper run, any epoch count):

       LAHYPER_DUMP_CANDIDATES=candidates.jsonl \\
           python3 -u simulator_LLM.py --framework lahyper --epoch 24 ...

   Each line is one (epoch, eco-mode) record with every candidate's
   drop_frac, ttft and metric value.

2. Sweep:

       python3 sweep_thresholds.py candidates.jsonl

   Flags:
     --drop-gates   comma list, percent     (default 5,10,15,20,30)
     --slas         comma list, seconds     (default 3,5,8,12,20,35,50)
     --sla-hit-target   fraction 0..1       (default 0.90; reference line)
     --recommend    also print a single recommended cell
                    (default: full grid only — you pick)
     --csv PATH     also write the full grid as CSV

WHAT IT REPORTS
---------------
For each (drop_gate, SLA) cell, across all dumped epochs and the three eco
modes:
  * sla_hit_rate   — fraction of (epoch,mode) selections whose chosen plan
                     actually meets the SLA without relaxation
  * mean_ttft      — mean TTFT of the chosen plans
  * eco_quality    — geometric mean across modes of
                     (best_attainable_metric / chosen_metric);  1.0 means the
                     gate/SLA cost nothing vs the unconstrained optimum,
                     lower means the constraints forced a worse plan
  * relax_rate     — fraction of selections that had to relax past the gate
                     or SLA because nothing qualified

The full grid is always printed; cells clearing --sla-hit-target are flagged.
Pass --recommend for a single suggested (drop_gate, SLA) cell.
"""
import argparse
import json
import sys
from collections import defaultdict


# Drop-gate tiers must mirror LA_Hyper_DDQN.py's eco-mode selection.  The
# sweep substitutes the *primary* tier; the fallback tiers below it are kept.
def _select(cands, metric_key, drop_gate, sla, own_ideal_idx):
    """Replay the eco-mode selection for one (epoch, mode) candidate pool.

    Returns (chosen_index, relaxed_bool, sla_met_bool).
    Mirrors LA_Hyper_DDQN.py: tiered drop fallback, own-ideal injection
    (gated by the active drop tier), then SLA constraint with graceful
    relaxation to the lowest-TTFT tier when no candidate meets the SLA.
    """
    tiers = (drop_gate, max(drop_gate, 0.30), 1.01)

    # ── drop gate, tiered ────────────────────────────────────────────────
    viable, used_tier, relaxed = [], tiers[-1], False
    for ti, tier in enumerate(tiers):
        viable = [i for i, c in enumerate(cands) if c["drop_frac"] <= tier]
        if viable:
            used_tier = tier
            relaxed = relaxed or (ti > 0)
            break
    if not viable:
        viable = list(range(len(cands)))
        relaxed = True

    # own-ideal injection, gated by the active drop tier
    for i in own_ideal_idx:
        if i not in viable and cands[i]["drop_frac"] <= used_tier:
            viable.append(i)

    # ── SLA constraint ───────────────────────────────────────────────────
    sla_ok = [i for i in viable if cands[i]["ttft"] <= sla]
    if sla_ok:
        chosen = min(sla_ok, key=lambda i: cands[i]["metric_value"])
        return chosen, relaxed, True

    # No candidate meets the SLA — relax: pick the metric-minimiser among the
    # lowest-TTFT tier (closest to the SLA we can get), so the agent still
    # specialises but its TTFT row honestly shows the SLA was missed.
    relaxed = True
    if viable:
        ttft_sorted = sorted(viable, key=lambda i: cands[i]["ttft"])
        cutoff = cands[ttft_sorted[max(0, len(ttft_sorted) // 4)]]["ttft"]
        low_tier = [i for i in viable if cands[i]["ttft"] <= cutoff]
        chosen = min(low_tier, key=lambda i: cands[i]["metric_value"])
        return chosen, relaxed, False
    return None, True, False


def load_records(path):
    recs = []
    with open(path) as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  warning: skipped malformed line {ln}: {e}", file=sys.stderr)
    return recs


def best_attainable(cands, metric_key):
    """Lowest metric value over ALL candidates — the unconstrained optimum."""
    vals = [c["metric_value"] for c in cands
            if c["metric_value"] not in (None, float("inf"))]
    return min(vals) if vals else None


def geomean(xs):
    xs = [x for x in xs if x and x > 0]
    if not xs:
        return 0.0
    import math
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def main():
    ap = argparse.ArgumentParser(description="Sweep eco-agent drop gate and TTFT SLA.")
    ap.add_argument("dump", help="candidates.jsonl produced via LAHYPER_DUMP_CANDIDATES")
    ap.add_argument("--drop-gates", default="5,10,15,20,30",
                    help="comma list, percent (default: 5,10,15,20,30)")
    ap.add_argument("--slas", default="3,5,8,12,20,35,50",
                    help="comma list, seconds (default: 3,5,8,12,20,35,50)")
    ap.add_argument("--sla-hit-target", type=float, default=0.90,
                    help="SLA hit rate flagged as a reference line in the grid")
    ap.add_argument("--recommend", action="store_true",
                    help="also print a single recommended cell "
                         "(default: report the full grid only)")
    ap.add_argument("--csv", default=None, help="optional: write full grid to CSV")
    args = ap.parse_args()

    drop_gates = [float(x) / 100.0 for x in args.drop_gates.split(",")]
    slas       = [float(x) for x in args.slas.split(",")]

    recs = load_records(args.dump)
    if not recs:
        print("No candidate records found. Did the run set "
              "LAHYPER_DUMP_CANDIDATES and exercise the eco modes?")
        sys.exit(1)

    eco_recs = [r for r in recs
                if r.get("mode") in ("carbon_agent", "water_agent", "cost_agent")]
    n_epochs = len({r["epoch"] for r in eco_recs})
    print(f"Loaded {len(recs)} records — {len(eco_recs)} eco-mode "
          f"(epoch,mode) pools across {n_epochs} epoch(s).\n")

    grid = []  # (drop_gate, sla, sla_hit_rate, mean_ttft, eco_quality, relax_rate)
    for dg in drop_gates:
        for sla in slas:
            sla_hits, relaxes, ttfts = 0, 0, []
            quality_by_mode = defaultdict(list)
            n = 0
            for r in eco_recs:
                cands = r["candidates"]
                if not cands:
                    continue
                mk = r["metric_key"]
                own_ideal = [i for i, c in enumerate(cands)
                             if c.get("origin_mode") == r["mode"]
                             and c.get("origin_variant") == "ideal"]
                chosen, relaxed, sla_met = _select(cands, mk, dg, sla, own_ideal)
                if chosen is None:
                    continue
                n += 1
                sla_hits += int(sla_met)
                relaxes  += int(relaxed)
                ttfts.append(cands[chosen]["ttft"])
                opt = best_attainable(cands, mk)
                chosen_val = cands[chosen]["metric_value"]
                if opt and chosen_val and chosen_val > 0:
                    quality_by_mode[r["mode"]].append(opt / chosen_val)
            if n == 0:
                continue
            sla_rate   = sla_hits / n
            relax_rate = relaxes / n
            mean_ttft  = sum(ttfts) / len(ttfts)
            per_mode_q = [geomean(v) for v in quality_by_mode.values()]
            eco_q      = geomean(per_mode_q) if per_mode_q else 0.0
            grid.append((dg, sla, sla_rate, mean_ttft, eco_q, relax_rate))

    # ── print the grid ──────────────────────────────────────────────────
    # The right-hand flag marks cells whose SLA hit rate clears the target —
    # a quick visual filter when picking a cell by hand.
    print(f"{'drop%':>6} {'SLA(s)':>7} {'SLA_hit':>8} {'mean_TTFT':>10} "
          f"{'eco_qual':>9} {'relax':>7}   flag")
    print("-" * 60)
    last_dg = None
    for dg, sla, sr, mt, eq, rr in grid:
        if last_dg is not None and dg != last_dg:
            print()
        last_dg = dg
        flag = "<- meets SLA target" if sr >= args.sla_hit_target else ""
        print(f"{dg*100:>6.0f} {sla:>7.0f} {sr*100:>7.0f}% {mt:>9.2f}s "
              f"{eq:>9.3f} {rr*100:>6.0f}%   {flag}")

    print()
    print("Columns: SLA_hit = fraction of selections meeting the SLA without "
          "relaxing;")
    print("         eco_qual = best-attainable metric / chosen metric "
          "(1.0 = constraints cost nothing);")
    print("         relax = fraction of selections that fell back past the "
          "gate or SLA.")
    print("Pick the cell that balances an acceptable SLA_hit against eco_qual "
          "for your SLA.")

    # ── recommendation (opt-in) ─────────────────────────────────────────
    if args.recommend:
        eligible = [g for g in grid if g[2] >= args.sla_hit_target]
        print()
        if eligible:
            best = max(eligible, key=lambda g: (round(g[4], 4), -g[1]))
            dg, sla, sr, mt, eq, rr = best
            print(f"RECOMMENDED:  drop_gate = {dg*100:.0f}%   "
                  f"TTFT_SLA = {sla:.0f}s")
            print(f"  -> SLA met {sr*100:.0f}%, mean TTFT {mt:.2f}s, "
                  f"eco-quality {eq:.3f}, relaxed {rr*100:.0f}%.")
        else:
            best = max(grid, key=lambda g: g[2]) if grid else None
            print(f"No cell reaches the {args.sla_hit_target*100:.0f}% "
                  f"SLA-hit target.")
            if best:
                print(f"  Highest achievable is {best[2]*100:.0f}% at "
                      f"drop={best[0]*100:.0f}%, SLA={best[1]:.0f}s.")
            print("  => The SLA may be physically too tight for this "
                  "workload; consider widening the eco DC footprint.")

    if args.csv:
        with open(args.csv, "w") as f:
            f.write("drop_gate_pct,sla_s,sla_hit_rate,mean_ttft_s,"
                    "eco_quality,relax_rate\n")
            for dg, sla, sr, mt, eq, rr in grid:
                f.write(f"{dg*100:.0f},{sla:.0f},{sr:.4f},{mt:.4f},"
                        f"{eq:.4f},{rr:.4f}\n")
        print(f"\nFull grid written to {args.csv}")


if __name__ == "__main__":
    main()