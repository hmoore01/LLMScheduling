"""
hybrid_experiments.py
=====================
Overnight experiment runner for the Hybrid framework covering three sweeps:

  1. DC Sweep      — vary [4, 6, 8, 12] DCs at fixed 95% utilisation
  2. Util Sweep    — vary [0.75, 0.85, 0.95, 1.05] util at fixed 8 DCs
  3. PHV Baseline  — per-epoch Pareto Hypervolume at the (8 DC, 95%) baseline

The (8 DC, 95%) configuration is shared between sweeps 1 and 2, so it is
executed only once and its results count in both analysis tables.

Total unique configs: 4 (DC sweep) + 3 extra util points = 7
Total runs: 7 configs × NUM_RUNS = 35 (default)
"""

import csv
import os
import re
import subprocess
import sys
import time
import numpy as np
from datetime import datetime
from itertools import product

# ── Experiment parameters ─────────────────────────────────────────────────────

FRAMEWORK = "hybrid"

DC_SWEEP_VALUES      = [4, 6, 8, 12]   # DCs to sweep
DC_SWEEP_UTIL        = 0.95            # fixed util for DC sweep

UTIL_SWEEP_VALUES    = [0.95]  # utils to sweep
UTIL_SWEEP_DCS       = 8              # fixed DC count for util sweep

# Baseline config (shared between both sweeps — run only once)
BASELINE_DCS         = 8
BASELINE_UTIL        = 0.95

NUM_RUNS             = 1

# Autoscale settings
AUTOSCALE_MODE       = "global_peak"
MAX_DROP             = 0.02
MAX_MULT             = 1_500_000
MAX_ROWS             = 300_000
SEARCH_STEPS         = 9

# ── CSV schema ────────────────────────────────────────────────────────────────

FIELDNAMES = [
    "framework", "num_dcs", "target_util", "run",
    "start_time", "end_time", "elapsed_min", "exit_code", "epochs",
    "avg_ttft_s", "total_carbon_kg", "total_water_l",
    "total_energy_usd", "total_energy_kwh",
    "avg_epoch_phv", "command",
]

METRICS = ["avg_ttft_s", "total_carbon_kg", "total_water_l", "total_energy_usd"]
LABELS  = ["TTFT(s)",    "Carbon(kg)",       "Water(L)",      "Cost($)"]


# ── Pareto Hypervolume ────────────────────────────────────────────────────────

def _dominated(a, b):
    """True if point a dominates point b (minimisation, all ≤ and one <)."""
    return all(ai <= bi for ai, bi in zip(a, b)) and any(ai < bi for ai, bi in zip(a, b))


def _pareto_front(points):
    n = len(points)
    dominated = [False] * n
    for i in range(n):
        if dominated[i]:
            continue
        for j in range(n):
            if i == j or dominated[j]:
                continue
            if _dominated(points[j], points[i]):
                dominated[i] = True
                break
    return [i for i in range(n) if not dominated[i]]


def _hypervolume_mc(points, ref, n_samples=200_000):
    """Monte Carlo hypervolume approximation for arbitrary dimensions."""
    pts = np.array(points)
    ref = np.array(ref)
    valid = np.all(pts < ref, axis=1)
    pts = pts[valid]
    if len(pts) == 0:
        return 0.0
    lo = pts.min(axis=0)
    box_vol = np.prod(ref - lo)
    if box_vol <= 0:
        return 0.0
    rng = np.random.default_rng(42)
    samples = rng.uniform(lo, ref, size=(n_samples, len(ref)))
    dominated_count = sum(1 for s in samples if any(np.all(p <= s) for p in pts))
    return box_vol * dominated_count / n_samples


def compute_hypervolume(points, ref):
    """Hypervolume of a point set relative to ref (minimisation)."""
    if not points:
        return 0.0
    front_idx = _pareto_front(points)
    front = [points[i] for i in front_idx]
    return _hypervolume_mc(front, ref)


# ── Output parsers ────────────────────────────────────────────────────────────

def _parse_final_report(output_lines):
    """Extract aggregate metrics from the simulator's === Final Report === block."""
    text = "".join(output_lines)
    metrics = {k: "" for k in [
        "epochs", "avg_ttft_s", "total_carbon_kg",
        "total_water_l", "total_energy_usd", "total_energy_kwh",
    ]}
    patterns = {
        "epochs":           r"Epochs:\s*([0-9]+)",
        "avg_ttft_s":       r"Average TTFT \(s\):\s*([0-9eE+\-\.]+)",
        "total_carbon_kg":  r"Total Carbon \(kg\):\s*([0-9eE+\-\.]+)",
        "total_water_l":    r"Total Water \(L\):\s*([0-9eE+\-\.]+)",
        "total_energy_usd": r"Total Energy \(\$\):\s*([0-9eE+\-\.]+)",
        "total_energy_kwh": r"Total Energy \(kWh\):\s*([0-9eE+\-\.]+)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, text)
        if m:
            metrics[key] = m.group(1)
    return metrics


def _parse_epoch_phv(output_lines):
    """
    Parse per-epoch Pareto points from [HYBRID-FRONT] tagged lines emitted by
    Hybrid_Scheduler_LLM.milp_optimizer() and compute the average per-epoch PHV.

    Each epoch emits exactly one line of the form:
        [HYBRID-FRONT] epoch=N ttft=X.X carbon=X.X water=X.X cost=X.X

    PHV for a single-point front is the hypervolume of that point relative to
    the run's own reference (worst value per objective × 1.10 across all epochs).

    Returns a formatted string (e.g. "1234.56") or "" if fewer than 5 epochs
    were parsed (not enough data for a meaningful average).
    """
    # Self-contained tagged lines — no dependence on epoch header ordering.
    front_pat = re.compile(
        r"\[HYBRID-FRONT\]\s+epoch=(\d+)"
        r"\s+ttft=([0-9eE+\-\.]+)"
        r"\s+carbon=([0-9eE+\-\.]+)"
        r"\s+water=([0-9eE+\-\.]+)"
        r"\s+cost=([0-9eE+\-\.]+)",
        re.IGNORECASE,
    )

    epoch_points = {}
    for line in output_lines:
        m = front_pat.search(line)
        if m:
            ep = int(m.group(1))
            pt = [float(m.group(i)) for i in range(2, 6)]  # ttft, carbon, water, cost
            epoch_points[ep] = pt  # last write wins if duplicate epoch (shouldn't happen)

    if len(epoch_points) < 5:
        return ""

    all_pts = list(epoch_points.values())
    arr = np.array(all_pts)
    ref_point = (arr.max(axis=0) * 1.10).tolist()

    epoch_hvs = [compute_hypervolume([pt], ref_point) for pt in all_pts]
    return f"{np.mean(epoch_hvs):.2f}" if epoch_hvs else ""


# ── Command builder ───────────────────────────────────────────────────────────

def _build_command(num_dcs, target_util):
    return [
        sys.executable, "-u", "simulator_LLM.py",
        "--framework",            FRAMEWORK,
        "--num-dcs",              str(num_dcs),
        "--target-util",          str(target_util),
        "--autoscale-mode",       AUTOSCALE_MODE,
        "--autoscale-max-drop",   str(MAX_DROP),
        "--autoscale-max-mult",   str(MAX_MULT),
        "--autoscale-max-rows",   str(MAX_ROWS),
        "--autoscale-search-steps", str(SEARCH_STEPS),
    ]


# ── Single-run executor ───────────────────────────────────────────────────────

def _run_one(num_dcs, target_util, run_num, script_dir, log_file, writer, summary_file):
    cmd        = _build_command(num_dcs, target_util)
    start_dt   = datetime.now()
    start_time = time.time()

    header = (f"\n[{start_dt}] HYBRID  DCs={num_dcs}  "
              f"Util={target_util*100:.0f}%  RUN={run_num}/{NUM_RUNS}")
    print(header)
    log_file.write(header + "\n")
    log_file.write(f"Command: {' '.join(cmd)}\n")
    log_file.write("-" * 60 + "\n")
    log_file.flush()

    output_lines, exit_code, process = [], -1, None
    try:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, cwd=script_dir,
        )
        try:
            for line in process.stdout:
                print(line, end="")
                log_file.write(line)
                output_lines.append(line)
                log_file.flush()
        except (BrokenPipeError, IOError, ValueError):
            pass
        process.wait(timeout=30)
        exit_code = int(process.returncode)
        if exit_code != 0:
            warn = f"\n[{datetime.now()}] WARNING: HYBRID exited code {exit_code}\n"
            print(warn, end="")
            log_file.write(warn)
            log_file.flush()
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        err = f"\n[{datetime.now()}] TIMEOUT: HYBRID killed after wait\n"
        print(err)
        log_file.write(err)
        log_file.flush()
    except Exception as exc:
        err = f"\nERROR running hybrid: {exc}\n"
        print(err)
        log_file.write(err)
        if process and process.poll() is None:
            try:
                process.kill()
                process.wait(timeout=10)
            except Exception:
                pass
        exit_code = (int(process.returncode)
                     if process and process.returncode is not None else -1)

    elapsed_min = (time.time() - start_time) / 60.0
    finish = (f"[{datetime.now()}] Finished HYBRID "
              f"(dcs={num_dcs}, util={target_util}, run={run_num}) "
              f"in {elapsed_min:.2f} min.\n")
    print(finish)
    log_file.write("-" * 60 + "\n" + finish)
    log_file.flush()

    parsed      = _parse_final_report(output_lines)
    avg_ep_phv  = _parse_epoch_phv(output_lines)
    if avg_ep_phv:
        print(f"  [PHV] hybrid @ {num_dcs}DCs/{target_util} "
              f"avg per-epoch PHV = {avg_ep_phv}")

    row = {
        "framework":        FRAMEWORK,
        "num_dcs":          num_dcs,
        "target_util":      f"{target_util:.2f}",
        "run":              run_num,
        "start_time":       start_dt.isoformat(timespec="seconds"),
        "end_time":         datetime.now().isoformat(timespec="seconds"),
        "elapsed_min":      f"{elapsed_min:.2f}",
        "exit_code":        str(exit_code),
        "epochs":           parsed["epochs"],
        "avg_ttft_s":       parsed["avg_ttft_s"],
        "total_carbon_kg":  parsed["total_carbon_kg"],
        "total_water_l":    parsed["total_water_l"],
        "total_energy_usd": parsed["total_energy_usd"],
        "total_energy_kwh": parsed["total_energy_kwh"],
        "avg_epoch_phv":    avg_ep_phv,
        "command":          " ".join(cmd),
    }
    writer.writerow(row)
    summary_file.flush()
    return row


# ── Summary helpers ───────────────────────────────────────────────────────────

def _stats(vals):
    from scipy import stats as scipy_stats
    v = vals.dropna()
    n = len(v)
    if n == 0:
        return {"avg": None, "min": None, "max": None, "ci95": None, "n": 0}
    avg, mn, mx = v.mean(), v.min(), v.max()
    ci95 = 0.0
    if n >= 2:
        ci95 = scipy_stats.t.ppf(0.975, df=n - 1) * v.std(ddof=1) / (n ** 0.5)
    return {"avg": avg, "min": mn, "max": mx, "ci95": ci95, "n": n}


def _fmt(s):
    if s["avg"] is None:
        return "n/a"
    return f"{s['avg']:.3f}±{s['ci95']:.3f} [{s['min']:.3f}, {s['max']:.3f}]"


def _fmt_phv(s):
    if s["avg"] is None:
        return "n/a"
    return f"{s['avg']:.1f}±{s['ci95']:.1f}"


# ── Main ──────────────────────────────────────────────────────────────────────

def run_batch_experiments():
    # Build deduplicated experiment list.
    # (BASELINE_DCS, BASELINE_UTIL) appears in both sweeps — run it once.
    seen   = set()
    configs = []

    for num_dcs in DC_SWEEP_VALUES:
        key = (num_dcs, DC_SWEEP_UTIL)
        if key not in seen:
            seen.add(key)
            configs.append(key)

    for util in UTIL_SWEEP_VALUES:
        key = (UTIL_SWEEP_DCS, util)
        if key not in seen:
            seen.add(key)
            configs.append(key)

    total_configs = len(configs)
    total_runs    = total_configs * NUM_RUNS

    script_dir   = os.path.dirname(os.path.abspath(__file__))
    ts           = datetime.now().strftime("%Y%m%d_%H%M")
    out_dir      = os.path.join(script_dir, "experiment_results")
    os.makedirs(out_dir, exist_ok=True)
    log_path     = os.path.join(out_dir, f"hybrid_sweep_{ts}.log")
    summary_path = os.path.join(out_dir, f"hybrid_sweep_{ts}.csv")

    print(f"{'='*70}")
    print(f"  HYBRID EXPERIMENT SUITE")
    print(f"  DC sweep  : {DC_SWEEP_VALUES} DCs @ {DC_SWEEP_UTIL*100:.0f}% util")
    print(f"  Util sweep: {[int(u*100) for u in UTIL_SWEEP_VALUES]}% util @ {UTIL_SWEEP_DCS} DCs")
    print(f"  Baseline  : {BASELINE_DCS} DCs @ {BASELINE_UTIL*100:.0f}% (shared, run once)")
    print(f"  Unique configs: {total_configs}  |  Runs/config: {NUM_RUNS}  |  Total: {total_runs}")
    print(f"  Log : {log_path}")
    print(f"  CSV : {summary_path}")
    print(f"{'='*70}\n")

    with open(log_path, "w") as log_file, \
         open(summary_path, "w", newline="") as summary_file:

        writer = csv.DictWriter(summary_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        log_file.write(f"=== STARTED: {datetime.now()} ===\n")
        log_file.write(f"Configs={total_configs} Runs/config={NUM_RUNS} "
                       f"Total={total_runs}\n\n")

        run_counter = 0
        for num_dcs, target_util in configs:
            label = (f"HYBRID @ {num_dcs} DCs, "
                     f"Util={target_util*100:.0f}%"
                     + (" [BASELINE]"
                        if num_dcs == BASELINE_DCS and target_util == BASELINE_UTIL
                        else ""))
            for run_num in range(1, NUM_RUNS + 1):
                run_counter += 1
                print(f"\n{'='*60}")
                print(f"Run {run_counter}/{total_runs}  "
                      f"[{label}  run {run_num}/{NUM_RUNS}]")
                print(f"{'='*60}")
                log_file.write(f"\n{'='*60}\n")
                log_file.write(f"Run {run_counter}/{total_runs} "
                               f"[{label} run {run_num}/{NUM_RUNS}]\n")
                try:
                    _run_one(num_dcs, target_util, run_num,
                             script_dir, log_file, writer, summary_file)
                except Exception as exc:
                    err = (f"\n[{datetime.now()}] FATAL: HYBRID @ {num_dcs} DCs "
                           f"util={target_util} run {run_num} crashed: {exc}\n")
                    print(err)
                    log_file.write(err)
                    log_file.flush()

        log_file.write(f"\n=== COMPLETED: {datetime.now()} ===\n")

    # ── Analysis & reporting ──────────────────────────────────────────────────
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED")
    print("="*80)
    print(f"Log : {log_path}")
    print(f"CSV : {summary_path}\n")

    try:
        import pandas as pd

        df = pd.read_csv(summary_path)
        for col in METRICS + ["total_energy_kwh", "avg_epoch_phv"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["target_util"] = pd.to_numeric(df["target_util"], errors="coerce")
        df["num_dcs"]     = pd.to_numeric(df["num_dcs"],     errors="coerce")

        W = 36  # column width

        # ── 1. DC Sweep ───────────────────────────────────────────────────────
        dc_df = df[df["target_util"] == DC_SWEEP_UTIL].copy()

        print(f"\n{'═'*80}")
        print(f"  DC SWEEP  (util={DC_SWEEP_UTIL*100:.0f}%, {NUM_RUNS} runs per config)")
        print(f"  Format: avg±95%CI [min, max]")
        print(f"{'═'*80}")

        for metric, label in zip(METRICS, LABELS):
            print(f"\n  {label}:")
            print(f"  {'DCs':<8}", end="")
            for dc in DC_SWEEP_VALUES:
                print(f"  {'DCs='+str(dc):^{W}}", end="")
            print()
            print(f"  {'─'*8}", end="")
            for _ in DC_SWEEP_VALUES:
                print(f"  {'─'*W}", end="")
            print()
            print(f"  {'hybrid':<8}", end="")
            for dc in DC_SWEEP_VALUES:
                vals = dc_df[dc_df["num_dcs"] == dc][metric]
                print(f"  {_fmt(_stats(vals)):^{W}}", end="")
            print()

        # ── 2. Util Sweep ─────────────────────────────────────────────────────
        util_df = df[df["num_dcs"] == UTIL_SWEEP_DCS].copy()

        print(f"\n{'═'*80}")
        print(f"  UTIL SWEEP  ({UTIL_SWEEP_DCS} DCs, {NUM_RUNS} runs per config)")
        print(f"  Format: avg±95%CI [min, max]")
        print(f"{'═'*80}")

        for metric, label in zip(METRICS, LABELS):
            print(f"\n  {label}:")
            print(f"  {'Util':<8}", end="")
            for util in UTIL_SWEEP_VALUES:
                print(f"  {'Util='+str(int(util*100))+'%':^{W}}", end="")
            print()
            print(f"  {'─'*8}", end="")
            for _ in UTIL_SWEEP_VALUES:
                print(f"  {'─'*W}", end="")
            print()
            print(f"  {'hybrid':<8}", end="")
            for util in UTIL_SWEEP_VALUES:
                vals = util_df[util_df["target_util"] == util][metric]
                print(f"  {_fmt(_stats(vals)):^{W}}", end="")
            print()

        # ── 3. PHV — Baseline and both sweeps ────────────────────────────────
        print(f"\n{'═'*80}")
        print(f"  PER-EPOCH PARETO HYPERVOLUME (PHV)")
        print(f"  1 objective point per epoch (TTFT, Carbon, Water, Cost)")
        print(f"  Reference = worst value per objective × 1.10 across all epochs in run")
        print(f"  Format: avg±95%CI across {NUM_RUNS} runs")
        print(f"{'═'*80}")

        # PHV across DC sweep
        print(f"\n  DC Sweep PHV (util={DC_SWEEP_UTIL*100:.0f}%):")
        print(f"  {'DCs':<8}", end="")
        for dc in DC_SWEEP_VALUES:
            print(f"  {'DCs='+str(dc):^24}", end="")
        print()
        print(f"  {'─'*8}", end="")
        for _ in DC_SWEEP_VALUES:
            print(f"  {'─'*24}", end="")
        print()
        print(f"  {'hybrid':<8}", end="")
        for dc in DC_SWEEP_VALUES:
            vals = dc_df[dc_df["num_dcs"] == dc]["avg_epoch_phv"]
            s = _stats(vals)
            print(f"  {_fmt_phv(s):^24}", end="")
        print()

        # PHV across util sweep
        print(f"\n  Util Sweep PHV ({UTIL_SWEEP_DCS} DCs):")
        print(f"  {'Util':<8}", end="")
        for util in UTIL_SWEEP_VALUES:
            print(f"  {'Util='+str(int(util*100))+'%':^24}", end="")
        print()
        print(f"  {'─'*8}", end="")
        for _ in UTIL_SWEEP_VALUES:
            print(f"  {'─'*24}", end="")
        print()
        print(f"  {'hybrid':<8}", end="")
        for util in UTIL_SWEEP_VALUES:
            vals = util_df[util_df["target_util"] == util]["avg_epoch_phv"]
            s = _stats(vals)
            print(f"  {_fmt_phv(s):^24}", end="")
        print()

        # Highlight baseline explicitly
        bl = df[(df["num_dcs"] == BASELINE_DCS) &
                (df["target_util"] == BASELINE_UTIL)]["avg_epoch_phv"]
        s_bl = _stats(bl)
        print(f"\n  Baseline PHV ({BASELINE_DCS} DCs @ {BASELINE_UTIL*100:.0f}%): "
              f"{_fmt_phv(s_bl)}  (n={s_bl['n']})")

        print(f"\n{'═'*80}")
        print(f"  Raw data: {summary_path}")
        print(f"{'═'*80}")

    except Exception as e:
        print(f"\n(Summary error: {e})")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_batch_experiments()