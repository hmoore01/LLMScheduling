#!/usr/bin/env python3
"""
Parse GTARL epoch tables from one or more log files and compute convergence
speed per ablation variant on the Balanced scheme.

Usage:
    python analyze_convergence.py logs/dc_sweep_*.log logs/new_ablation_*.log
    python analyze_convergence.py experiment_results/*.log

Convergence is defined as the first epoch where the 10-epoch rolling average
of each metric stays within CONV_THRESHOLD (default 5%) of its final
10-epoch average for all subsequent epochs.
"""

import re
import sys
import os
import argparse
import numpy as np
from collections import defaultdict

# ─── Configuration ────────────────────────────────────────────────────────────
CONV_THRESHOLD = 0.05  # 5% of final rolling mean
ROLLING_WINDOW = 10  # epochs for rolling average
SCHEME = "Balanced"  # scheme to analyze


# ─── Parsers ──────────────────────────────────────────────────────────────────

def parse_log_files(filepaths):
    """Parse multiple log files, return list of run dicts.

    Each run dict:
        {
            "ablation": str,   # "" for full GTARL, "no-film" etc for ablations
            "num_dcs": int,
            "run_num": int,
            "epochs": {epoch_idx: {"ttft": float, "carbon": float, "water": float, "cost": float}}
        }
    """
    all_runs = []

    for fpath in filepaths:
        if not os.path.exists(fpath):
            print(f"  [WARN] File not found: {fpath}")
            continue
        print(f"  Parsing {fpath}...")
        with open(fpath, "r", errors="replace") as f:
            text = f.read()
        runs = _parse_single_log(text, fpath)
        all_runs.extend(runs)
        print(f"    Found {len(runs)} parliament runs")

    return all_runs


def _parse_single_log(text, source_file=""):
    """Parse a single log file's text into run dicts."""
    runs = []

    # Split into runs by looking for the FW= header line
    # Pattern: [timestamp] FW=PARLIAMENT  DCs=8  Util=95%  RUN=3/5
    run_header_pat = re.compile(
        r"FW=PARLIAMENT\s+DCs=(\d+)\s+Util=\d+%\s+RUN=(\d+)/(\d+)"
    )

    # Find ablation from command line in the log
    # Pattern: --ablation no-film
    ablation_pat = re.compile(r"--ablation\s+(\S+)")

    # Epoch table row for a specific scheme
    # Pattern: │ Balanced         0.695★     378.4      240.8      152.3     16662  │
    # Also handle without box-drawing chars:
    #   Balanced         0.695     378.4      240.8      152.3     16662
    epoch_header_pat = re.compile(r"EPOCH\s+(\d+)")
    scheme_row_pat = re.compile(
        r"(?:│\s*)?" + re.escape(SCHEME) + r"\s+"
                                           r"([0-9eE+\-\.★]+)\s+"  # TTFT
                                           r"([0-9eE+\-\.★]+)\s+"  # Carbon
                                           r"([0-9eE+\-\.★]+)\s+"  # Water
                                           r"([0-9eE+\-\.★]+)\s+"  # Cost
                                           r"([0-9]+)"  # Served
    )

    # Split text into chunks per run
    # Find all run header positions
    headers = list(run_header_pat.finditer(text))
    if not headers:
        return runs

    for h_idx, header_match in enumerate(headers):
        num_dcs = int(header_match.group(1))
        run_num = int(header_match.group(2))

        # Extract the text for this run (up to next run header or end)
        start = header_match.start()
        end = headers[h_idx + 1].start() if h_idx + 1 < len(headers) else len(text)
        run_text = text[start:end]

        # Also look backwards from the header for ablation info
        lookback_start = max(0, start - 2000)
        lookback_text = text[lookback_start:end]

        # Detect ablation
        ablation = ""
        abl_matches = list(ablation_pat.finditer(lookback_text))
        if abl_matches:
            # Use the last match before this run's header
            ablation = abl_matches[-1].group(1)

        # Also check for ablation in the run banner
        # Pattern: [PARLIAMENT @ 8 DCs [no-film]  run 1/5]
        banner_pat = re.compile(r"PARLIAMENT\s+@\s+\d+\s+DCs\s+\[([^\]]+)\]")
        banner_m = banner_pat.search(lookback_text[-1500:])
        if banner_m:
            ablation = banner_m.group(1)

        # Parse epoch tables within this run
        epochs = {}
        current_epoch = None

        for line in run_text.split("\n"):
            # Check for epoch header
            ep_m = epoch_header_pat.search(line)
            if ep_m:
                current_epoch = int(ep_m.group(1))
                continue

            # Check for scheme row
            if current_epoch is not None:
                sr_m = scheme_row_pat.search(line)
                if sr_m:
                    epochs[current_epoch] = {
                        "ttft": float(sr_m.group(1).replace("★", "")),
                        "carbon": float(sr_m.group(2).replace("★", "")),
                        "water": float(sr_m.group(3).replace("★", "")),
                        "cost": float(sr_m.group(4).replace("★", "")),
                        "served": int(sr_m.group(5)),
                    }

        if epochs:
            runs.append({
                "ablation": ablation,
                "num_dcs": num_dcs,
                "run_num": run_num,
                "epochs": epochs,
                "source": os.path.basename(source_file),
            })

    return runs


# ─── Convergence Computation ──────────────────────────────────────────────────

def compute_convergence(epoch_dict, metric, window=ROLLING_WINDOW, threshold=CONV_THRESHOLD):
    """Compute convergence epoch for a single metric in a single run.

    Returns (convergence_epoch, final_rolling_mean) or (None, None) if not converged.
    """
    if len(epoch_dict) < window * 2:
        return None, None

    # Sort epochs and extract metric values
    sorted_epochs = sorted(epoch_dict.keys())
    values = np.array([epoch_dict[e][metric] for e in sorted_epochs])

    if len(values) < window:
        return None, None

    # Compute rolling average
    rolling = np.convolve(values, np.ones(window) / window, mode="valid")
    # rolling[i] corresponds to epochs[i] through epochs[i + window - 1]
    # The "center" epoch for rolling[i] is epochs[i + window - 1]

    if len(rolling) < 2:
        return None, None

    # Final rolling mean = last value of the rolling average
    final_mean = rolling[-1]

    if abs(final_mean) < 1e-10:
        return sorted_epochs[window - 1], final_mean

    band = abs(final_mean * threshold)

    # Find first epoch where ALL subsequent rolling values stay within band
    converged_idx = None
    for i in range(len(rolling)):
        # Check if all values from i to end are within band
        if np.all(np.abs(rolling[i:] - final_mean) <= band):
            converged_idx = i
            break

    if converged_idx is not None:
        # Map back to actual epoch index
        conv_epoch = sorted_epochs[converged_idx + window - 1]
        return conv_epoch, final_mean

    return None, final_mean


def analyze_runs(all_runs, target_dcs=8):
    """Compute convergence stats grouped by ablation variant."""
    # Filter to target DC count and parliament
    filtered = [r for r in all_runs if r["num_dcs"] == target_dcs]

    if not filtered:
        print(f"\n  [WARN] No parliament runs found for {target_dcs} DCs")
        return {}

    # Group by ablation
    groups = defaultdict(list)
    for r in filtered:
        abl = r["ablation"] if r["ablation"] else "GTARL-Full"
        groups[abl].append(r)

    results = {}
    metrics = ["ttft", "carbon", "water", "cost"]
    metric_labels = {"ttft": "TTFT", "carbon": "Carbon", "water": "Water", "cost": "Cost"}

    for abl, runs in sorted(groups.items()):
        conv_data = {m: [] for m in metrics}
        n_epochs_list = []

        for run in runs:
            n_epochs_list.append(len(run["epochs"]))
            for metric in metrics:
                ce, fm = compute_convergence(run["epochs"], metric)
                if ce is not None:
                    conv_data[metric].append(ce)

        # Compute stats
        abl_result = {"n_runs": len(runs), "n_epochs_avg": np.mean(n_epochs_list)}
        for metric in metrics:
            vals = conv_data[metric]
            if vals:
                abl_result[f"{metric}_conv_avg"] = np.mean(vals)
                abl_result[f"{metric}_conv_min"] = np.min(vals)
                abl_result[f"{metric}_conv_max"] = np.max(vals)
                abl_result[f"{metric}_conv_n"] = len(vals)
            else:
                abl_result[f"{metric}_conv_avg"] = None
                abl_result[f"{metric}_conv_n"] = 0

        # Overall convergence = max across all 4 metrics (all must converge)
        all_conv = []
        for metric in metrics:
            if conv_data[metric]:
                all_conv.append(np.mean(conv_data[metric]))
        abl_result["overall_conv"] = max(all_conv) if all_conv else None

        results[abl] = abl_result

    return results


# ─── Output ───────────────────────────────────────────────────────────────────

def print_results(results):
    metrics = ["ttft", "carbon", "water", "cost"]
    labels = {"ttft": "TTFT", "carbon": "Carbon", "water": "Water", "cost": "Cost"}

    print(f"\n{'═' * 90}")
    print(f"  CONVERGENCE ANALYSIS — {SCHEME} Scheme @ {CONV_THRESHOLD * 100:.0f}% threshold, "
          f"{ROLLING_WINDOW}-epoch rolling window")
    print(f"{'═' * 90}")

    # Table 1: Per-metric convergence epoch
    print(f"\n  Convergence epoch (avg across runs):")
    print(f"  {'Variant':<22} {'Runs':>5} {'TTFT':>8} {'Carbon':>8} {'Water':>8} "
          f"{'Cost':>8} {'Overall':>10}")
    print(f"  {'─' * 22} {'─' * 5} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 10}")

    # Sort: GTARL-Full first, then alphabetically
    sorted_keys = sorted(results.keys(), key=lambda k: ("" if k == "GTARL-Full" else k))

    for abl in sorted_keys:
        r = results[abl]
        parts = [f"  {abl:<22} {r['n_runs']:>5}"]
        for m in metrics:
            avg = r.get(f"{m}_conv_avg")
            n = r.get(f"{m}_conv_n", 0)
            if avg is not None:
                parts.append(f"{avg:>7.0f}")
            else:
                parts.append(f"{'n/c':>8}")
        ov = r.get("overall_conv")
        parts.append(f"{ov:>9.0f}" if ov is not None else f"{'n/c':>10}")
        print(" ".join(parts))

    # Table 2: Δ from baseline
    if "GTARL-Full" in results:
        bl = results["GTARL-Full"]
        bl_overall = bl.get("overall_conv")

        print(f"\n  Δ convergence epochs from GTARL-Full (positive = slower):")
        print(f"  {'Variant':<22} {'ΔTTFT':>8} {'ΔCarbon':>8} {'ΔWater':>8} "
              f"{'ΔCost':>8} {'ΔOverall':>10}")
        print(f"  {'─' * 22} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 10}")

        for abl in sorted_keys:
            if abl == "GTARL-Full":
                continue
            r = results[abl]
            parts = [f"  {abl:<22}"]
            for m in metrics:
                bl_avg = bl.get(f"{m}_conv_avg")
                ab_avg = r.get(f"{m}_conv_avg")
                if bl_avg is not None and ab_avg is not None:
                    d = ab_avg - bl_avg
                    parts.append(f"{d:>+7.0f}")
                else:
                    parts.append(f"{'n/a':>8}")
            ov = r.get("overall_conv")
            if bl_overall is not None and ov is not None:
                d = ov - bl_overall
                parts.append(f"{d:>+9.0f}")
            else:
                parts.append(f"{'n/a':>10}")
            print(" ".join(parts))

    # Table 3: Epoch range per run (shows how many epochs were available)
    print(f"\n  Data coverage:")
    print(f"  {'Variant':<22} {'Runs':>5} {'Avg Epochs':>12} {'Conv Rate':>12}")
    print(f"  {'─' * 22} {'─' * 5} {'─' * 12} {'─' * 12}")
    for abl in sorted_keys:
        r = results[abl]
        conv_rates = []
        for m in metrics:
            n = r.get(f"{m}_conv_n", 0)
            conv_rates.append(n / r["n_runs"] if r["n_runs"] > 0 else 0)
        avg_rate = np.mean(conv_rates)
        print(f"  {abl:<22} {r['n_runs']:>5} {r['n_epochs_avg']:>11.0f} "
              f"{avg_rate:>11.0%}")

    print(f"\n{'═' * 90}")
    print(f"  n/c = did not converge within available epochs")
    print(f"  Conv Rate = fraction of (runs × metrics) that converged")
    print(f"{'═' * 90}")


def export_csv(results, output_path):
    """Export convergence results to CSV."""
    import csv
    metrics = ["ttft", "carbon", "water", "cost"]

    fieldnames = ["variant", "n_runs", "avg_epochs",
                  "ttft_conv", "carbon_conv", "water_conv", "cost_conv",
                  "overall_conv"]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for abl in sorted(results.keys(), key=lambda k: ("" if k == "GTARL-Full" else k)):
            r = results[abl]
            row = {
                "variant": abl,
                "n_runs": r["n_runs"],
                "avg_epochs": f"{r['n_epochs_avg']:.0f}",
                "overall_conv": f"{r['overall_conv']:.0f}" if r.get("overall_conv") else "",
            }
            for m in metrics:
                avg = r.get(f"{m}_conv_avg")
                row[f"{m}_conv"] = f"{avg:.0f}" if avg is not None else ""
            writer.writerow(row)
    print(f"\n  CSV exported: {output_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyze convergence speed from GTARL log files")
    parser.add_argument("logfiles", nargs="+", help="Log file paths (supports glob)")
    parser.add_argument("--dcs", type=int, default=8,
                        help="DC count to analyze (default: 8)")
    parser.add_argument("--threshold", type=float, default=0.05,
                        help="Convergence threshold (default: 0.05 = 5%%)")
    parser.add_argument("--window", type=int, default=10,
                        help="Rolling window size (default: 10)")
    parser.add_argument("--scheme", type=str, default="Balanced",
                        help="Scheme to analyze (default: Balanced)")
    parser.add_argument("--csv", type=str, default="",
                        help="Export results to CSV")
    args = parser.parse_args()

    global CONV_THRESHOLD, ROLLING_WINDOW, SCHEME
    CONV_THRESHOLD = args.threshold
    ROLLING_WINDOW = args.window
    SCHEME = args.scheme

    print(f"Convergence Analysis")
    print(f"  Scheme: {SCHEME}")
    print(f"  DCs: {args.dcs}")
    print(f"  Threshold: {CONV_THRESHOLD * 100:.0f}%")
    print(f"  Window: {ROLLING_WINDOW} epochs")
    print(f"  Log files: {len(args.logfiles)}")

    all_runs = parse_log_files(args.logfiles)

    if not all_runs:
        print("\nNo parliament runs found in the provided log files.")
        print("Expected format: epoch tables with '┌ EPOCH N' headers and scheme rows.")
        sys.exit(1)

    print(f"\nTotal parliament runs parsed: {len(all_runs)}")
    ablations = defaultdict(int)
    for r in all_runs:
        abl = r["ablation"] if r["ablation"] else "GTARL-Full"
        ablations[abl] += 1
    for abl, count in sorted(ablations.items()):
        print(f"  {abl}: {count} runs ({all_runs[0]['num_dcs'] if all_runs else '?'} DCs)")

    results = analyze_runs(all_runs, target_dcs=args.dcs)

    if not results:
        print(f"\nNo results for {args.dcs} DCs.")
        sys.exit(1)

    print_results(results)

    if args.csv:
        export_csv(results, args.csv)


if __name__ == "__main__":
    main()