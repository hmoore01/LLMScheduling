"""
overnight_sweep.py
==================
Overnight experiment runner covering three trace sources and five sweeps.

Traces
──────
  1. BurstGPT     — BurstGPT_without_fails_2.csv
  2. Azure-Code   — AzureLLMInferenceTrace_code_1week.csv   (--day-offset 0)
  3. Azure-Conv   — AzureLLMInferenceTrace_conv_1week.csv   (--day-offset 0)

Sweeps (per trace, each pivots around the shared baseline)
──────────────────────────────────────────────────────────
  1. DC Sweep     — num_dcs        [4, 6, 8, 12]
  2. Util Sweep   — target_util    [65%, 75%, 85%, 95%, 105%]
  3. Mix Sweep    — large_frac     [0.0, 0.25, 0.50, 0.75, 1.0]
  4. Origin Sweep — distribution   [even, population, time]
  5. Noise Sweep  — prediction_noise [0.0, 0.10, 0.20, 0.30]

Baseline: 8 DCs / 95% util / 0.75 large_frac / even / 0 noise — shared across
all sweeps and executed only once per (trace, framework) pair.  Each sweep
varies ONE axis and holds the others at baseline, so the baseline point is the
common anchor of every sweep table.

Which sweeps run is controlled by ENABLED_SWEEPS, so the run can be phased:
e.g. run {baseline, dc, util, mix} first, analyse, then a later invocation with
{origin, noise} appends to the same CSV (the baseline anchor rows from phase 1
are reused by the analysis automatically).

Frameworks
──────────
  Every entry in FRAMEWORKS is benchmarked.  parliament (Game_Theoretic_RL)
  runs online-only — no --offline-train flag is ever passed.  NOTE: the output
  parsers also recognise helix / splitwise / perllm / hybrid / nsga2 FRONT
  tags; if any of those are intended baselines, add them to FRAMEWORKS — they
  are NOT swept unless listed there.

Robustness
──────────
  • Resume-safe: rows already marked "ok*" in the CSV are skipped on restart,
    so a crash mid-sweep costs only the unfinished runs (see RERUN_FAILED).
  • Idle hang guard (PER_RUN_IDLE_TIMEOUT_MIN, OFF by default): only a run
    that goes fully silent for that many minutes is killed; a slow run that is
    still printing epoch lines is never interrupted, so runs always finish.
  • Preflight: verifies the sweep-defining CLI flags actually exist in the
    target scripts, so an origin/noise sweep can't silently be a no-op.
  • Coverage audit: after the run, every intended (trace, framework, config,
    run) cell is checked against the CSV and missing/failed cells are listed.

Runtime estimate
────────────────
  total runs = len(TRACES) × len(configs) × len(FRAMEWORKS) × NUM_RUNS.
  With the defaults below that is 3 × 17 × 5 × 5 = 1275 runs — plan a weekend,
  and rely on the resume logic if it does not finish in one sitting.
"""

import csv
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import numpy as np
from datetime import datetime
import autopush_results   # local module -- must sit beside this script

# ── Trace definitions ─────────────────────────────────────────────────────────
TRACES = [
    {
        "name":       "burstgpt",
        "label":      "BurstGPT",
        "inputs":     ["BurstGPT_without_fails_2.csv"],
        "extra_args": [],
    },
    {
        "name":       "azure_code",
        "label":      "Azure-Code",
        "inputs":     ["AzureLLMInferenceTrace_code_1week.csv"],
        "extra_args": ["--day-offset", "0"],
    },
    {
        "name":       "azure_conv",
        "label":      "Azure-Conv",
        "inputs":     ["AzureLLMInferenceTrace_conv_1week.csv"],
        "extra_args": ["--day-offset", "0"],
    },
]

# ── Frameworks ────────────────────────────────────────────────────────────────
# Every entry here is swept on all traces and all configs.  The output parsers
# additionally recognise these FRONT tags: helix, splitwise, perllm, hybrid,
# nsga2.  If any of those are intended baselines, add their CLI name here — they
# are NOT run unless listed.
FRAMEWORKS = [f.strip() for f in os.environ.get("SWEEP_FRAMEWORKS", "nsga2").split(",") if f.strip()]

# ── Sweep axes ────────────────────────────────────────────────────────────────
BASELINE_DCS        = 8
BASELINE_UTIL       = 0.95
BASELINE_LARGE_FRAC = 0.75

DC_SWEEP_VALUES         = [4, 6, 8, 10, 12]
UTIL_SWEEP_VALUES       = [0.65, 0.75, 0.85, 0.95, 1.05]
LARGE_FRAC_SWEEP_VALUES = [0.0, 0.25, 0.50, 0.75, 1.0]

# Origin pattern sweep — source-DC assignment distribution
BASELINE_DISTRIBUTION   = "even"
ORIGIN_SWEEP_VALUES     = ["even", "population", "time"]

# Workload prediction inaccuracy sweep — noise fraction in [0, 1]
BASELINE_PRED_NOISE     = 0.0
PRED_NOISE_SWEEP_VALUES = [0.0, 0.10, 0.20, 0.30]

NUM_RUNS   = 5      # per (trace, config, framework) — CI needs n >= 2
NUM_EPOCHS = 96     # 96 x 15 min = 24 hours

# ── Robustness & phasing knobs ────────────────────────────────────────────────
# Which sweeps to run this invocation.  Drop entries to phase the work (e.g.
# run {baseline, dc_sweep, util_sweep, mix_sweep} first, then a later run with
# {origin_sweep, noise_sweep}).  The CSV is shared, so analysis still sees the
# baseline anchors written by an earlier phase.
ENABLED_SWEEPS = {
    "baseline", "dc_sweep", "util_sweep", "mix_sweep",
    "origin_sweep", "noise_sweep",
}

# Resume: a run whose CSV row has status starting with "ok" is considered done
# and skipped on restart.  With RERUN_FAILED=True, "crashed"/"no-data" rows are
# retried (their stale rows are ignored by the analysis, which only averages
# "ok*" rows).  Set False to also treat any prior attempt as final.
RERUN_FAILED = True

# Optional hang guard.  This is an IDLE timeout, not a total-runtime cap: it
# only fires when a run produces NO output for this many minutes, which means a
# genuine hang.  A slow-but-progressing run keeps printing epoch lines and is
# never interrupted, so legitimate long runs always finish.  None = disabled
# (the default — runs are never killed).  If you enable it, size it well above
# the longest silent stretch a healthy run can have (e.g. one slow epoch).
PER_RUN_IDLE_TIMEOUT_MIN = None

# Preflight: if a sweep-defining CLI flag is missing from the target script,
# warn (False) or abort (True).  Catches an origin/noise sweep silently being a
# no-op because the simulator never learned the flag.
STRICT_PREFLIGHT = False

# ── Trace processor ───────────────────────────────────────────────────────────
TRACE_SCRIPT   = "trace_process.py"
ACTIVE_TRACE   = "simulator_ready_trace.csv"
TRACE_TEMPLATE = "trace_{name}_lf{lf:.2f}.csv"

# ── Autoscale ─────────────────────────────────────────────────────────────────
AUTOSCALE_MODE = "global_peak"
MAX_DROP       = 0.02
MAX_MULT       = 1_500_000
# Row-replication budget for autoscaling.  Token inflation is now hard-capped
# (AUTOSCALE_MAX_TOKEN_SCALE=2.0 in simulator_LLM.py) so request sizes stay
# realistic — which means ROW REPLICATION must supply the load needed to reach
# target utilisation.  At ~3.8k base requests/epoch and a ~7.9k-x peak
# multiplier, count_mult must reach ~4k, so the budget needs to be in the tens
# of millions.  This makes each epoch process many requests (slow, memory-
# heavy); lower this if runs become infeasible — the autoscaler will then warn
# and honestly reach a lower utilisation rather than inflate tokens.
MAX_ROWS       = 16_000_000
SEARCH_STEPS   = 9

# ── TTFT-band token-scale calibration ─────────────────────────────────────────
# Static token scaling lands at very different average TTFTs depending on the
# trace's token distribution.  When TTFT_CALIBRATE is on, the first run of each
# (trace, config) is preceded by ONE calibration pass: simulator_LLM.py bisects
# the autoscale token-scale target until the HELIX baseline's request-weighted
# average TTFT falls inside TTFT_BAND.  The calibrated value is then passed to
# EVERY framework of that config via --autoscale-target-token-scale, so all
# frameworks (on this machine and any other) see byte-identical workloads and
# remain directly comparable.  Only the rows-vs-tokens split moves; the total
# multiplier — and therefore the achieved utilisation — is untouched.
#
# Results are cached in CALIB_FILE so resume never recalibrates.  MULTI-MACHINE
# NOTE: because each machine runs a different FRAMEWORKS list, all machines
# must use the SAME calibrated value for the same config.  Run one machine
# first (or a quick helix-only calibration phase), commit CALIB_FILE to the
# shared repo, and pull it on the others — a key found in the file is always
# reused as-is, never recomputed.
TTFT_CALIBRATE    = True
TTFT_BAND         = (2.0, 5.0)     # helix avg-TTFT target band, seconds
TTFT_CALIB_EPOCHS = 2              # representative epochs per calibration probe
CALIB_FRAMEWORK   = "helix"
CALIB_FILE        = "experiment_results/ttft_calibration.json"

# Global autoscale plans are identical across frameworks and runs of the same
# (trace, config); caching them skips the dry-run peak search + probe on every
# run after the first.  The calibration pass seeds this cache, so by the time
# real runs start the plan is usually already a cache hit.
PLAN_CACHE_FILE   = "experiment_results/autoscale_plans.json"

# "--calibrate-only" on the command line: walk every (trace, config) in the
# sweep, run the TTFT calibration (which also seeds the autoscale plan cache)
# for each, and exit WITHOUT executing any experiments.  Run this once on one
# machine, commit experiment_results/ttft_calibration.json and
# autoscale_plans.json, pull on the others — then every machine starts its
# experiments immediately with identical, pre-shared scaling.
CALIBRATE_ONLY = "--calibrate-only" in sys.argv

# ── CSV schema ────────────────────────────────────────────────────────────────
# Two CSVs are written, each at its natural granularity so each stays readable:
#
#   RUNS csv  — one row per run (~25 columns); every column is meaningful for
#               every framework, so there are no blank cells.
#   MODES csv — one row per (run, lahyper mode).  The 10x5 per-mode breakdown
#               used to sit as 50 columns inside the runs csv — blank for every
#               non-lahyper framework — which is what made the file unreadable.
#               It now lives here, linked back to the run row by `run_id`.
#
# The runs csv carries a `status` column (see _extract_run_metrics) so a run
# that crashed mid-way is an explicit row, not a row of silent blank metrics.

# A SINGLE CSV in long ("tidy") form — one file, every metric in it.
#
#   * Most rows are run-level: scope="run", carrying that run's headline
#     metrics in the metric columns.
#   * A lahyper run additionally emits one row per multi-agent mode
#     (scope="time_agent", "carbon_agent", ...) carrying that mode's metrics
#     in the SAME metric columns.  Those rows sit directly beneath their run
#     (same run_id).
#
# To read headline results: filter scope=="run".  For the lahyper per-agent
# breakdown: the scope=<mode> rows.  No second file, no 50-column wide block.
FIELDNAMES = [
    "run_id", "trace", "framework", "sweep",
    "num_dcs", "target_util", "large_frac", "distribution", "prediction_noise",
    "run", "scope",
    "status", "epochs_completed", "epochs_expected", "metric_source",
    "avg_ttft_s", "total_carbon_kg", "total_water_m3",
    "total_energy_usd", "total_energy_kwh", "avg_epoch_phv",
    "elapsed_min", "exit_code", "start_time", "end_time", "command",
]

# lahyper per-mode modes — must stay in sync with LA_Hyper_DDQN.py.
PER_MODE_NAMES = [
    "time_agent", "carbon_agent", "water_agent", "cost_agent",
    "Balanced", "green_perf", "cost_guard", "water_saver", "peak_power_guard",
    "Pareto_Sample",
]

METRICS = ["avg_ttft_s", "total_carbon_kg", "total_water_m3", "total_energy_usd"]
LABELS  = ["TTFT(s)",    "Carbon(kg)",       "Water(m3)",      "Cost($)"]


# ── Pareto Hypervolume helpers ────────────────────────────────────────────────

def _dominated(a, b):
    return all(ai <= bi for ai, bi in zip(a, b)) and any(ai < bi for ai, bi in zip(a, b))


def _pareto_front(points):
    n          = len(points)
    dominated  = [False] * n
    for i in range(n):
        if dominated[i]:
            continue
        for j in range(n):
            if i != j and not dominated[j] and _dominated(points[j], points[i]):
                dominated[i] = True
                break
    return [i for i in range(n) if not dominated[i]]


def _hypervolume_mc(points, ref, n_samples=200_000):
    pts = np.array(points)
    ref = np.array(ref)
    pts = pts[np.all(pts < ref, axis=1)]
    if len(pts) == 0:
        return 0.0
    lo      = pts.min(axis=0)
    box_vol = np.prod(ref - lo)
    if box_vol <= 0:
        return 0.0
    rng     = np.random.default_rng(42)
    samples = rng.uniform(lo, ref, size=(n_samples, len(ref)))
    cnt     = sum(1 for s in samples if any(np.all(p <= s) for p in pts))
    return box_vol * cnt / n_samples


def compute_hypervolume(points, ref):
    if not points:
        return 0.0
    front = [points[i] for i in _pareto_front(points)]
    return _hypervolume_mc(front, ref)


# ── Output parsers ────────────────────────────────────────────────────────────

def _parse_epoch_fronts(lines):
    """Return {epoch: [(ttft, carbon, water, cost, served), ...]} from per-epoch
    [*-FRONT] lines.  `served` is None for lines that predate the served=
    field.  Used by the partial-run fallback (PHV has its own parser)."""
    pat = re.compile(
        r"\[(?:HYBRID|NSGA2|LAHYPER|PARLIAMENT|PERLLM|HELIX|SPLITWISE)-FRONT\]"
        r".*?epoch=(\d+).*?ttft=([0-9eE+\-\.]+).*?carbon=([0-9eE+\-\.]+)"
        r".*?water=([0-9eE+\-\.]+).*?cost=([0-9eE+\-\.]+)"
        r"(?:.*?served=([0-9eE+\-\.]+))?",
        re.IGNORECASE,
    )
    epoch_pts = {}
    for line in lines:
        m = pat.search(line)
        if not m:
            continue
        try:
            epoch  = int(m.group(1))
            vals   = [float(m.group(i)) for i in range(2, 6)]
            served = float(m.group(6)) if m.group(6) is not None else None
        except (ValueError, TypeError):
            continue
        epoch_pts.setdefault(epoch, []).append(tuple(vals) + (served,))
    return epoch_pts


def _extract_run_metrics(lines, epochs_expected):
    """Pull the headline run metrics, robustly.

    Primary source is the simulator's '=== Final Report ===' block — but that
    only prints if the run completes.  If it is missing or incomplete (the run
    crashed or was killed mid-way), this aggregates whatever per-epoch
    [*-FRONT] lines DID print: average TTFT, summed carbon/water/cost.  So a
    partial run yields real numbers instead of a row of blank cells.

    Returns the six metric strings plus 'status', 'epochs_completed' and
    'metric_source', so a partial run is always flagged explicitly.
    """
    text = "".join(lines)
    out  = {k: "" for k in ("avg_ttft_s", "total_carbon_kg", "total_water_m3",
                            "total_energy_usd", "total_energy_kwh")}

    # Fast path: the Final Report block.
    report = {}
    for key, pat in {
        "report_epochs":    r"Epochs:\s*([0-9]+)",
        "avg_ttft_s":       r"Average TTFT\s*\([^)]*\):\s*([0-9eE+\-\.]+)",
        "total_carbon_kg":  r"Total Carbon\s*\([^)]*\):\s*([0-9eE+\-\.]+)",
        "total_water_m3":   r"Total Water\s*\([^)]*\):\s*([0-9eE+\-\.]+)",
        "total_energy_usd": r"Total Energy\s*\(\s*\$\s*\):\s*([0-9eE+\-\.]+)",
        "total_energy_kwh": r"Total Energy\s*\(\s*kWh\s*\):\s*([0-9eE+\-\.]+)",
    }.items():
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            report[key] = m.group(1)

    fronts           = _parse_epoch_fronts(lines)
    epochs_seen      = sorted(fronts.keys())
    epochs_completed = len(epochs_seen)

    metric_keys = ("avg_ttft_s", "total_carbon_kg", "total_water_m3",
                   "total_energy_usd", "total_energy_kwh")
    have_report = all(k in report for k in metric_keys)

    if have_report:
        for k in metric_keys:
            out[k] = report[k]
        if report.get("report_epochs"):
            epochs_completed = int(report["report_epochs"])
        out["metric_source"] = "final-report"
        if epochs_expected and epochs_completed < epochs_expected:
            out["status"] = f"ok-short({epochs_completed}/{epochs_expected})"
        else:
            out["status"] = "ok"
    elif epochs_completed > 0:
        # Crashed / killed before the Final Report — aggregate the epochs that
        # ran.  Per epoch: mean the front points.  Across epochs: TTFT is
        # request-weighted (matching the Final Report's final_avg_ttft) when the
        # FRONT lines carry served=; otherwise a plain mean.  carbon/water/cost
        # are summed across epochs.  kWh is not on the FRONT line -> blank.
        per_ep_ttft, per_ep_w = [], []
        sum_carbon = sum_water = sum_cost = 0.0
        for ep in epochs_seen:
            pts = fronts[ep]
            n   = len(pts)
            per_ep_ttft.append(sum(p[0] for p in pts) / n)
            sum_carbon += sum(p[1] for p in pts) / n
            sum_water  += sum(p[2] for p in pts) / n
            sum_cost   += sum(p[3] for p in pts) / n
            served = [p[4] for p in pts if p[4] is not None]
            per_ep_w.append(sum(served) / len(served) if served else 0.0)
        if sum(per_ep_w) > 0.0:
            ttft = (sum(t * w for t, w in zip(per_ep_ttft, per_ep_w))
                    / sum(per_ep_w))
            out["metric_source"] = "epoch-fallback-weighted"
        else:
            ttft = sum(per_ep_ttft) / len(per_ep_ttft)
            out["metric_source"] = "epoch-fallback-unweighted"
        out["avg_ttft_s"]       = f"{ttft:.6f}"
        out["total_carbon_kg"]  = f"{sum_carbon:.3f}"
        out["total_water_m3"]   = f"{sum_water:.4f}"
        out["total_energy_usd"] = f"{sum_cost:.3f}"
        out["total_energy_kwh"] = ""
        out["status"]           = (f"crashed(ep{epochs_seen[-1]},"
                                   f"{epochs_completed}/{epochs_expected})")
    else:
        out["metric_source"] = "none"
        out["status"]        = "no-data"

    out["epochs_completed"] = str(epochs_completed)
    return out


def _parse_epoch_phv(lines):
    pat = re.compile(
        r"\[(?:HYBRID|NSGA2|LAHYPER|PARLIAMENT|PERLLM|HELIX|SPLITWISE)-FRONT\]"
        r".*?epoch=(\d+).*?ttft=([0-9eE+\-\.]+).*?carbon=([0-9eE+\-\.]+)"
        r".*?water=([0-9eE+\-\.]+).*?cost=([0-9eE+\-\.]+)",
        re.IGNORECASE,
    )
    epoch_pts = {}
    for line in lines:
        m = pat.search(line)
        if m:
            epoch = int(m.group(1))
            pt    = [float(m.group(i)) for i in range(2, 6)]
            epoch_pts.setdefault(epoch, []).append(pt)
    if len(epoch_pts) < 5:
        return ""
    all_pts   = [pt for pts in epoch_pts.values() for pt in pts]
    ref_point = (np.array(all_pts).max(axis=0) * 1.10).tolist()
    hvs       = [compute_hypervolume(pts, ref_point) for pts in epoch_pts.values()]
    return f"{np.mean(hvs):.2f}" if hvs else ""


# ── Per-mode summary table parser ─────────────────────────────────────────────
# simulator_LLM.py prints a clean "=== LA_HYPER MULTI-AGENT SUMMARY (Run Totals) ==="
# table at the end of every lahyper run with rows like:
#
#   time_agent         | 1.5434      | 146.460         | 335.623       | 68.823       | 327.727
#
# Format: Mode | Avg TTFT(s) | Total Carb(kg) | Total Wat(m³) | Total Cost($) | Total Energy(kWh)
# (TTFT is averaged across epochs; everything else is summed.)
#
# Parsing this instead of the per-epoch tables means we get exactly the
# aggregation the simulator already computed — no double-averaging, no
# off-by-one on epochs, and per-mode rows that survive even when the framework
# crashes mid-run (as long as the final report was printed).
_SUMMARY_ROW_RE = re.compile(
    r"^([A-Za-z_][A-Za-z_0-9]*)\s+\|\s+"
    r"([0-9eE+\-\.]+)\s+\|\s+"           # Avg TTFT(s)
    r"([0-9eE+\-\.]+)\s+\|\s+"           # Total Carb(kg)
    r"([0-9eE+\-\.]+)\s+\|\s+"           # Total Wat(m³)
    r"([0-9eE+\-\.]+)\s+\|\s+"           # Total Cost($)
    r"([0-9eE+\-\.]+)\s*$"                # Total Energy(kWh)
)


def _parse_per_mode(lines):
    """Parse the LA_HYPER MULTI-AGENT SUMMARY table.

    Returns {"<mode>_<metric>": value} where metric ∈ {ttft, carbon, water,
    cost, total_energy}.  Returns an empty dict if the summary table is absent
    (e.g. non-lahyper run, or run that crashed before final report).
    """
    # Find the summary table header — scan forward from there.
    in_table = False
    out = {}
    for line in lines:
        stripped = line.strip()
        if "LA_HYPER MULTI-AGENT SUMMARY" in stripped:
            in_table = True
            continue
        if not in_table:
            continue
        # Blank line or new section ends the table.
        if not stripped or stripped.startswith("==="):
            if out:                       # we've already started collecting → done
                break
            continue
        m = _SUMMARY_ROW_RE.match(stripped)
        if not m:
            continue                       # header line or separator — skip
        mode = m.group(1)
        if mode not in PER_MODE_NAMES:
            continue                       # unknown / future mode
        try:
            out[f"{mode}_ttft"]         = f"{float(m.group(2)):.4f}"
            out[f"{mode}_carbon"]       = f"{float(m.group(3)):.4f}"
            out[f"{mode}_water"]        = f"{float(m.group(4)):.4f}"
            out[f"{mode}_cost"]         = f"{float(m.group(5)):.4f}"
            out[f"{mode}_total_energy"] = f"{float(m.group(6)):.4f}"
        except ValueError:
            continue
    return out


# ── Trace management ──────────────────────────────────────────────────────────

def generate_trace(trace_cfg, large_frac, script_dir, log_file):
    fname = TRACE_TEMPLATE.format(name=trace_cfg["name"], lf=large_frac)
    fpath = os.path.join(script_dir, fname)

    if os.path.exists(fpath):
        msg = f"[TRACE] Reuse {fname}\n"
        print(msg, end=""); log_file.write(msg)
        return fpath

    msg = (f"\n[TRACE] Generating {fname}  "
           f"(trace={trace_cfg['label']}  lf={large_frac:.2f})\n")
    print(msg, end=""); log_file.write(msg); log_file.flush()

    cmd = [sys.executable, "-u", TRACE_SCRIPT]
    for inp in trace_cfg["inputs"]:
        cmd += ["--input", inp]
    cmd += ["--output", fname, "--max-epochs", str(NUM_EPOCHS),
            "--large-frac", str(large_frac)]
    cmd += trace_cfg.get("extra_args", [])

    result = subprocess.run(cmd, capture_output=True, text=True,
                            cwd=script_dir, timeout=600)
    log_file.write(result.stdout)
    if result.stderr:
        log_file.write(result.stderr)
    log_file.flush()

    if result.returncode != 0 or not os.path.exists(fpath):
        raise RuntimeError(
            f"Trace generation failed (exit {result.returncode}) "
            f"for {trace_cfg['name']} lf={large_frac:.2f}\n"
            f"stderr: {result.stderr[:500]}"
        )

    size_mb = os.path.getsize(fpath) / 1_048_576
    done = f"[TRACE] Done {fname}  ({size_mb:.1f} MB)\n"
    print(done, end=""); log_file.write(done)
    return fpath


def activate_trace(trace_path, script_dir):
    shutil.copy2(trace_path, os.path.join(script_dir, ACTIVE_TRACE))


# ── Command builder ───────────────────────────────────────────────────────────

def _build_command(framework, num_dcs, target_util,
                   distribution="even", prediction_noise=0.0,
                   token_scale_target=None, plan_key=None):
    cmd = [
        sys.executable, "-u", "simulator_LLM.py",
        "--framework",              framework,
        "--epoch",                  str(NUM_EPOCHS),
        "--num-dcs",                str(num_dcs),
        "--target-util",            str(target_util),
        "--autoscale-mode",         AUTOSCALE_MODE,
        "--autoscale-max-drop",     str(MAX_DROP),
        "--autoscale-max-mult",     str(MAX_MULT),
        "--autoscale-max-rows",     str(MAX_ROWS),
        "--autoscale-search-steps", str(SEARCH_STEPS),
        "--distribution",           distribution,
    ]
    if prediction_noise > 0.0:
        cmd += ["--prediction-noise", str(prediction_noise)]
    if token_scale_target is not None and float(token_scale_target) > 0.0:
        cmd += ["--autoscale-target-token-scale", f"{float(token_scale_target):.6f}"]
    if plan_key:
        cmd += ["--autoscale-plan-cache", PLAN_CACHE_FILE,
                "--autoscale-plan-key",   str(plan_key)]
    return cmd


# ── TTFT calibration (one shared token scale per trace+config) ────────────────

_CALIB_RESULT_RE = re.compile(
    r"\[TTFT-Calib\]\s+RESULT\s+token_scale_target=([0-9eE+\-\.]+)"
    r"\s+measured_ttft=([0-9eE+\-\.]+|nan)"
    r".*?status=([\w\-]+)",
    re.IGNORECASE,
)

_CALIB_MEM = {}          # in-process cache: {key: token_scale or None}


def _calib_key(trace_name, num_dcs, target_util, large_frac,
               distribution, prediction_noise):
    return (f"{trace_name}|dcs{int(num_dcs)}|util{float(target_util):.2f}"
            f"|lf{float(large_frac):.2f}|{distribution}"
            f"|noise{float(prediction_noise):.2f}")


def _calib_path(script_dir):
    return os.path.join(script_dir, CALIB_FILE)


def _load_calibrations(script_dir):
    try:
        with open(_calib_path(script_dir)) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def _save_calibration(script_dir, key, entry):
    path = _calib_path(script_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = _load_calibrations(script_dir)
    data[key] = entry
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _ensure_token_scale(trace_cfg, num_dcs, target_util, large_frac,
                        distribution, prediction_noise, script_dir, log_file):
    """Return the calibrated token-scale target for this (trace, config),
    calibrating once with the helix baseline if no cached value exists.

    The matching trace file MUST already be active (simulator_ready_trace.csv)
    — the caller activates it before the first real run of the config, which is
    exactly when this is invoked.  Returns None when calibration is disabled or
    fails, in which case runs fall back to the simulator's static split (the
    pre-calibration behaviour) and a warning is logged."""
    if not TTFT_CALIBRATE:
        return None
    key = _calib_key(trace_cfg["name"], num_dcs, target_util, large_frac,
                     distribution, prediction_noise)
    if key in _CALIB_MEM:
        return _CALIB_MEM[key]
    disk = _load_calibrations(script_dir)
    if key in disk:
        ts = float(disk[key]["token_scale"])
        _CALIB_MEM[key] = ts
        msg = (f"[CALIB] Reusing cached token scale {ts:.4f}x for {key} "
               f"(measured helix TTFT {disk[key].get('measured_ttft', '?')} s, "
               f"status={disk[key].get('status', '?')})\n")
        print(msg, end=""); log_file.write(msg); log_file.flush()
        return ts

    cmd = _build_command(CALIB_FRAMEWORK, num_dcs, target_util,
                         distribution=distribution,
                         prediction_noise=prediction_noise,
                         plan_key=key) + [
        "--ttft-calibrate-only",
        "--ttft-band-low",     str(TTFT_BAND[0]),
        "--ttft-band-high",    str(TTFT_BAND[1]),
        "--ttft-calib-epochs", str(TTFT_CALIB_EPOCHS),
    ]
    hdr = (f"\n[CALIB] {trace_cfg['label']}: calibrating token scale for "
           f"helix avg TTFT in [{TTFT_BAND[0]:.1f}, {TTFT_BAND[1]:.1f}] s "
           f"(dcs={num_dcs} util={target_util:.2f} lf={large_frac:.2f} "
           f"dist={distribution} noise={prediction_noise:.2f})\n")
    print(hdr, end=""); log_file.write(hdr)
    log_file.write(f"  cmd: {' '.join(cmd)}\n"); log_file.flush()

    lines = []
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                bufsize=1, cwd=script_dir)
        for line in proc.stdout:
            print(line, end=""); log_file.write(line)
            lines.append(line)
        proc.wait(timeout=30)
        log_file.flush()
    except Exception as exc:
        msg = f"[CALIB] WARNING: calibration subprocess failed ({exc})\n"
        print(msg, end=""); log_file.write(msg); log_file.flush()
        _CALIB_MEM[key] = None
        return None

    result = None
    for line in lines:
        m = _CALIB_RESULT_RE.search(line)
        if m:
            result = m            # keep the LAST result line
    if result is None or proc.returncode != 0:
        msg = (f"[CALIB] WARNING: no calibration result for {key} "
               f"(exit {proc.returncode}); runs for this config will use the "
               f"simulator's static token scaling.\n")
        print(msg, end=""); log_file.write(msg); log_file.flush()
        _CALIB_MEM[key] = None
        return None

    ts = float(result.group(1))
    entry = {
        "token_scale":   ts,
        "measured_ttft": result.group(2),
        "status":        result.group(3),
        "band":          list(TTFT_BAND),
        "framework":     CALIB_FRAMEWORK,
        "calibrated_at": datetime.now().isoformat(timespec="seconds"),
        "machine":       autopush_results.MACHINE_ID,
    }
    _save_calibration(script_dir, key, entry)
    _CALIB_MEM[key] = ts
    msg = (f"[CALIB] Done: token scale {ts:.4f}x "
           f"(helix avg TTFT {entry['measured_ttft']} s, "
           f"status={entry['status']}) — cached in {CALIB_FILE}\n")
    print(msg, end=""); log_file.write(msg); log_file.flush()
    return ts


# ── Single-run executor ───────────────────────────────────────────────────────

def _run_one(trace_cfg, framework, sweep_name, num_dcs, target_util,
             large_frac, run_num, script_dir, log_file, writer, summary_file,
             run_id, distribution="even", prediction_noise=0.0,
             token_scale_target=None):
    plan_key   = _calib_key(trace_cfg["name"], num_dcs, target_util,
                            large_frac, distribution, prediction_noise)
    cmd        = _build_command(framework, num_dcs, target_util,
                                distribution=distribution,
                                prediction_noise=prediction_noise,
                                token_scale_target=token_scale_target,
                                plan_key=plan_key)
    start_dt   = datetime.now()
    start_time = time.time()

    hdr = (f"\n[{start_dt}] {trace_cfg['label']:<12}|{framework.upper():<12}|"
           f"sweep={sweep_name} DCs={num_dcs} util={target_util*100:.0f}% "
           f"lf={large_frac:.2f} run={run_num}/{NUM_RUNS}")
    print(hdr); log_file.write(hdr + "\n")
    log_file.write(f"  cmd: {' '.join(cmd)}\n" + "-"*70 + "\n")
    log_file.flush()

    output_lines, exit_code, process = [], -1, None
    timed_out   = {"flag": False}        # set by the monitor if the run goes idle
    last_output = {"t": time.time()}     # updated on every line the run prints

    try:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, cwd=script_dir)

        # Idle/no-output hang guard.  The stdout read loop below blocks until the
        # process closes its pipe, so a truly hung run (no output, never exits)
        # would otherwise stall the whole batch.  This monitor kills the process
        # ONLY after PER_RUN_IDLE_TIMEOUT_MIN of complete silence — a run that is
        # still printing epoch lines resets the clock and is never interrupted,
        # so slow-but-healthy runs always finish.  Disabled when the knob is None.
        monitor      = None
        stop_monitor = threading.Event()
        if PER_RUN_IDLE_TIMEOUT_MIN:
            idle_sec = PER_RUN_IDLE_TIMEOUT_MIN * 60.0

            def _watch_idle(proc):
                poll = max(1.0, min(idle_sec / 4.0, 30.0))
                while not stop_monitor.wait(poll):
                    if time.time() - last_output["t"] > idle_sec:
                        timed_out["flag"] = True
                        try: proc.kill()
                        except Exception: pass
                        return

            monitor = threading.Thread(target=_watch_idle, args=(process,),
                                       daemon=True)
            monitor.start()

        try:
            for line in process.stdout:
                last_output["t"] = time.time()
                print(line, end=""); log_file.write(line)
                output_lines.append(line); log_file.flush()
        except (BrokenPipeError, IOError, ValueError):
            pass
        finally:
            stop_monitor.set()
            if monitor:
                monitor.join(timeout=5)

        process.wait(timeout=30)
        exit_code = int(process.returncode)
        if timed_out["flag"]:
            log_file.write(f"\n[IDLE-TIMEOUT] {framework} killed after "
                           f"{PER_RUN_IDLE_TIMEOUT_MIN} min with no output\n")
        elif exit_code != 0:
            log_file.write(f"\n[WARN] {framework} exit code {exit_code}\n")
    except subprocess.TimeoutExpired:
        process.kill(); process.wait()
        log_file.write(f"\n[TIMEOUT] {framework} killed\n")
    except Exception as exc:
        log_file.write(f"\n[ERROR] {framework}: {exc}\n")
        if process and process.poll() is None:
            try: process.kill(); process.wait(timeout=10)
            except Exception: pass
        exit_code = (int(process.returncode)
                     if process and process.returncode is not None else -1)

    elapsed_min = (time.time() - start_time) / 60.0
    finish = (f"[{datetime.now()}] Done {framework} "
              f"({trace_cfg['name']} dcs={num_dcs} util={target_util:.2f} "
              f"lf={large_frac:.2f} run={run_num}) in {elapsed_min:.2f}min\n")
    print(finish); log_file.write("-"*70 + "\n" + finish); log_file.flush()

    info       = _extract_run_metrics(output_lines, NUM_EPOCHS)
    avg_ep_phv = _parse_epoch_phv(output_lines)
    per_mode   = _parse_per_mode(output_lines)
    # A killed run is recorded as a timeout regardless of how many epochs it
    # logged, so the audit and analysis treat it as a failure, not "ok".  Status
    # still starts with "timeout" so the coverage audit buckets it correctly.
    if timed_out["flag"]:
        info["status"] = (f"timeout-idle({info['epochs_completed']}/{NUM_EPOCHS},"
                          f"{PER_RUN_IDLE_TIMEOUT_MIN}min-silent)")
    if avg_ep_phv:
        print(f"  [PHV] {framework}@{trace_cfg['label']} = {avg_ep_phv}")
    # Surface the run's health on the console — a crash is no longer a silent
    # row of blank metric cells.
    if not info["status"].startswith("ok"):
        print(f"  [STATUS] {framework}@{trace_cfg['label']} run{run_num}: "
              f"{info['status']}  (metrics from {info['metric_source']})")

    row = {
        "run_id":           run_id,
        "trace":            trace_cfg["name"],
        "framework":        framework,
        "sweep":            sweep_name,
        "num_dcs":          num_dcs,
        "target_util":      f"{target_util:.2f}",
        "large_frac":       f"{large_frac:.2f}",
        "distribution":     distribution,
        "prediction_noise": f"{prediction_noise:.2f}",
        "run":              run_num,
        "scope":            "run",
        "status":           info["status"],
        "epochs_completed": info["epochs_completed"],
        "epochs_expected":  str(NUM_EPOCHS),
        "metric_source":    info["metric_source"],
        "avg_ttft_s":       info["avg_ttft_s"],
        "total_carbon_kg":  info["total_carbon_kg"],
        "total_water_m3":   info["total_water_m3"],
        "total_energy_usd": info["total_energy_usd"],
        "total_energy_kwh": info["total_energy_kwh"],
        "avg_epoch_phv":    avg_ep_phv,
        "elapsed_min":      f"{elapsed_min:.2f}",
        "exit_code":        str(exit_code),
        "start_time":       start_dt.isoformat(timespec="seconds"),
        "end_time":         datetime.now().isoformat(timespec="seconds"),
        "command":          " ".join(cmd),
    }
    writer.writerow(row); summary_file.flush()

    # lahyper per-mode breakdown -> additional rows in the SAME csv
    # (scope=<mode>), reusing the shared metric columns.  Baseline frameworks
    # emit no extra rows.  csv.DictWriter blanks any field not supplied here,
    # so run-level-only columns (timing, command, phv) stay empty on these.
    ident = {
        "run_id": run_id, "trace": trace_cfg["name"], "framework": framework,
        "sweep": sweep_name, "num_dcs": num_dcs,
        "target_util": f"{target_util:.2f}", "large_frac": f"{large_frac:.2f}",
        "distribution": distribution,
        "prediction_noise": f"{prediction_noise:.2f}", "run": run_num,
        "status": info["status"], "epochs_completed": info["epochs_completed"],
        "epochs_expected": str(NUM_EPOCHS),
    }
    for mode in PER_MODE_NAMES:
        if f"{mode}_ttft" in per_mode:
            writer.writerow({
                **ident, "scope": mode, "metric_source": "multi-agent-summary",
                "avg_ttft_s":       per_mode.get(f"{mode}_ttft", ""),
                "total_carbon_kg":  per_mode.get(f"{mode}_carbon", ""),
                "total_water_m3":   per_mode.get(f"{mode}_water", ""),
                "total_energy_usd": per_mode.get(f"{mode}_cost", ""),
                "total_energy_kwh": per_mode.get(f"{mode}_total_energy", ""),
            })
    summary_file.flush()
    return row


# ── Summary helpers ───────────────────────────────────────────────────────────

def _stats(vals):
    from scipy import stats as scipy_stats
    v  = vals.dropna()
    n  = len(v)
    if n == 0:
        return {"avg": None, "min": None, "max": None, "ci95": None, "n": 0}
    avg, mn, mx = float(v.mean()), float(v.min()), float(v.max())
    ci95 = 0.0
    if n >= 2:
        ci95 = float(
            scipy_stats.t.ppf(0.975, df=n-1) * float(v.std(ddof=1)) / (n**0.5)
        )
    return {"avg": avg, "min": mn, "max": mx, "ci95": ci95, "n": n}


def _fmt(s):
    if s["avg"] is None:
        return "n/a"
    ci = s["ci95"] if s["ci95"] is not None else 0.0
    return f"{s['avg']:.3f}+/-{ci:.3f} [{s['min']:.3f},{s['max']:.3f}]"


def _fmt_phv(s):
    if s["avg"] is None:
        return "n/a"
    ci = s["ci95"] if s["ci95"] is not None else 0.0
    return f"{s['avg']:.1f}+/-{ci:.1f}"


def _print_sweep_table_str(df, sweep_col, sweep_values, metric, label, fw_list, width=34):
    """Variant of _print_sweep_table for string-valued sweep columns (e.g. distribution)."""
    print(f"\n  {label}:")
    hdr = f"  {'Framework':<14}"
    for v in sweep_values:
        hdr += f"  {str(v):^{width}}"
    print(hdr)
    print(f"  {'':->14}", end="")
    for _ in sweep_values:
        print(f"  {'':->{width}}", end="")
    print()
    for fw in fw_list:
        fw_df   = df[df["framework"] == fw]
        row_str = f"  {fw:<14}"
        for v in sweep_values:
            sub = fw_df[fw_df[sweep_col].astype(str) == str(v)]
            row_str += f"  {_fmt(_stats(sub[metric])):^{width}}"
        print(row_str)


def _print_sweep_table(df, sweep_col, sweep_values, metric, label, fw_list, width=34):
    print(f"\n  {label}:")
    hdr = f"  {'Framework':<14}"
    for v in sweep_values:
        col_lbl = (f"{v*100:.0f}%" if isinstance(v, float) and sweep_col != "num_dcs"
                   else str(v))
        hdr += f"  {col_lbl:^{width}}"
    print(hdr)
    print(f"  {'':->14}", end="")
    for _ in sweep_values:
        print(f"  {'':->{width}}", end="")
    print()
    for fw in fw_list:
        fw_df   = df[df["framework"] == fw]
        row_str = f"  {fw:<14}"
        for v in sweep_values:
            if sweep_col == "num_dcs":
                sub = fw_df[fw_df[sweep_col] == v]
            else:
                sub = fw_df[np.isclose(fw_df[sweep_col].astype(float), v, atol=1e-4)]
            row_str += f"  {_fmt(_stats(sub[metric])):^{width}}"
        print(row_str)


# ── Resume / preflight / coverage helpers ─────────────────────────────────────

def _run_key(trace_name, framework, num_dcs, target_util,
             large_frac, distribution, prediction_noise, run_num):
    """Canonical identity of a single run, formatted exactly as the CSV stores
    it so a row read back from disk matches a config built in memory."""
    return (
        str(trace_name), str(framework), int(num_dcs),
        f"{float(target_util):.2f}", f"{float(large_frac):.2f}",
        str(distribution), f"{float(prediction_noise):.2f}", int(run_num),
    )


def _load_run_status(csv_path):
    """Read the existing results CSV and return {run_key: status} for every
    scope=='run' row.  Empty dict if the file is absent or unreadable."""
    status = {}
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return status
    try:
        with open(csv_path, newline="") as fh:
            for r in csv.DictReader(fh):
                if r.get("scope") != "run":
                    continue
                try:
                    key = _run_key(
                        r["trace"], r["framework"], r["num_dcs"],
                        r["target_util"], r["large_frac"],
                        r.get("distribution") or BASELINE_DISTRIBUTION,
                        r.get("prediction_noise") or 0.0, r["run"],
                    )
                except (KeyError, ValueError):
                    continue
                prev = status.get(key, "")
                # Keep the best status we've seen for this key (ok* wins).
                if prev.startswith("ok"):
                    continue
                status[key] = r.get("status", "")
    except Exception as exc:
        print(f"[RESUME] Could not read existing CSV ({exc}); starting fresh.")
    return status


def _is_done(status):
    """A run counts as done (skippable) iff it produced a final report."""
    s = status or ""
    if s.startswith("ok"):
        return True
    return (not RERUN_FAILED) and s != ""


def _preflight_flags(script_dir):
    """Verify the sweep-defining CLI flags actually appear in the target
    scripts.  A flag passed but unknown to the script means that whole sweep is
    silently a no-op, which a coverage count alone would never reveal."""
    checks = [
        ("simulator_LLM.py", "--distribution",
         "origin/distribution sweep" if len(ORIGIN_SWEEP_VALUES) > 1 else None),
        ("simulator_LLM.py", "--prediction-noise",
         "prediction-noise sweep" if len(PRED_NOISE_SWEEP_VALUES) > 1 else None),
        ("simulator_LLM.py", "--num-dcs",  "DC sweep"),
        ("simulator_LLM.py", "--target-util", "utilisation sweep"),
        ("simulator_LLM.py", "--ttft-calibrate-only",
         "TTFT calibration" if TTFT_CALIBRATE else None),
        ("simulator_LLM.py", "--autoscale-target-token-scale",
         "TTFT calibration" if TTFT_CALIBRATE else None),
        ("simulator_LLM.py", "--autoscale-plan-cache", "autoscale plan cache"),
        (TRACE_SCRIPT,       "--large-frac",  "model-mix sweep"),
    ]
    # --day-offset only matters if some trace uses it.
    if any("--day-offset" in t.get("extra_args", []) for t in TRACES):
        checks.append((TRACE_SCRIPT, "--day-offset", "Azure day-offset"))

    missing = []
    cache = {}
    for fname, flag, why in checks:
        if why is None:
            continue
        path = os.path.join(script_dir, fname)
        if fname not in cache:
            try:
                with open(path) as fh:
                    cache[fname] = fh.read()
            except OSError:
                cache[fname] = None
        src = cache[fname]
        if src is None:
            missing.append((fname, flag, why, "script not found"))
        elif flag not in src:
            missing.append((fname, flag, why, "flag not referenced"))

    if missing:
        print("\n[PREFLIGHT] WARNING — sweep flags not found in target scripts:")
        for fname, flag, why, reason in missing:
            print(f"    {flag:<20} ({why}) -> {fname}: {reason}")
        print("    Affected sweep(s) may run but produce baseline-identical "
              "results.")
        if STRICT_PREFLIGHT:
            print("[PREFLIGHT] STRICT_PREFLIGHT=True -> aborting.")
            sys.exit(2)
        print("[PREFLIGHT] Continuing (STRICT_PREFLIGHT=False).\n")
    else:
        print("[PREFLIGHT] OK — all sweep flags are recognised by their "
              "target scripts.\n")
    return missing


def _audit_coverage(csv_path, traces, frameworks, configs, num_runs):
    """Cross-check every intended (trace, framework, config, run) cell against
    the CSV and print a clear completeness report.  This is the guarantee that
    the matrix is actually covered — not just that the loop ran."""
    status = _load_run_status(csv_path)
    buckets = {"ok": [], "short": [], "timeout": [], "crashed": [],
               "no-data": [], "missing": []}

    for tr in traces:
        for sweep, dcs, util, lf, dist, noise in configs:
            for fw in frameworks:
                for run_num in range(1, num_runs + 1):
                    key = _run_key(tr["name"], fw, dcs, util, lf, dist,
                                   noise, run_num)
                    s = status.get(key)
                    label = (f"{tr['name']}|{fw}|dcs{dcs}|util{util:.2f}|"
                             f"lf{lf:.2f}|{dist}|noise{noise:.2f}|run{run_num}")
                    if s is None:
                        buckets["missing"].append(label)
                    elif s.startswith("ok-short"):
                        buckets["short"].append(label)
                    elif s.startswith("ok"):
                        buckets["ok"].append(label)
                    elif s.startswith("timeout"):
                        buckets["timeout"].append(label)
                    elif s.startswith("crashed"):
                        buckets["crashed"].append(label)
                    else:
                        buckets["no-data"].append(label)

    total = sum(len(v) for v in buckets.values())
    print(f"\n{'='*80}")
    print(f"  COVERAGE AUDIT  —  {total} intended cells "
          f"({len(traces)} traces x {len(configs)} configs x "
          f"{len(frameworks)} frameworks x {num_runs} runs)")
    print(f"{'='*80}")
    print(f"  ok (usable)      : {len(buckets['ok'])}")
    print(f"  ok-short         : {len(buckets['short'])}  (final report, fewer epochs)")
    print(f"  timeout          : {len(buckets['timeout'])}")
    print(f"  crashed          : {len(buckets['crashed'])}")
    print(f"  no-data          : {len(buckets['no-data'])}")
    print(f"  MISSING (unrun)  : {len(buckets['missing'])}")

    incomplete = (buckets["timeout"] + buckets["crashed"]
                  + buckets["no-data"] + buckets["missing"])
    if not incomplete:
        print("\n  ✓ Full coverage — every intended cell has a usable result.")
    else:
        print(f"\n  {len(incomplete)} cell(s) need attention "
              f"(re-run the script to retry them):")
        CAP = 40
        for lbl in incomplete[:CAP]:
            print(f"    - {lbl}")
        if len(incomplete) > CAP:
            print(f"    ... and {len(incomplete) - CAP} more "
                  f"(see status!='ok*' rows in the CSV).")
    print(f"{'='*80}")
    return buckets


# ── Main ──────────────────────────────────────────────────────────────────────

def run_experiments():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir    = os.path.join(script_dir, "experiment_results")
    os.makedirs(out_dir, exist_ok=True)
    log_path   = os.path.join(out_dir, "LAHyper_Full_Experiments.log")
    csv_path   = os.path.join(out_dir, "LAHyper_Results.csv")     # single results file

    # Start the background GitHub uploader: pushes THIS machine's results
    # under a unique per-host filename, at most once every 30 min. Set
    # AUTOPUSH=0 in the environment to disable (e.g. for local test runs).
    # The preflight check warns immediately if pushing isn't possible, so a
    # misconfigured machine is caught now -- before you walk away overnight.
    if os.environ.get("AUTOPUSH", "1") != "0":
        autopush_results.check_can_push(script_dir)
        autopush_results.start_background_pusher(script_dir, min_interval=1800)
        print(f"  AUTOPUSH: uploader started (machine={autopush_results.MACHINE_ID})")

    # Append mode — safe to restart: existing rows are preserved, the header is
    # written only when the CSV does not yet exist (or is empty).
    csv_is_new = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0

    # Validate inputs
    for tr in TRACES:
        for inp in tr["inputs"]:
            if not os.path.exists(os.path.join(script_dir, inp)):
                print(f"ERROR: input not found: {inp}\n"
                      f"       Update the TRACES list at the top of this file.",
                      file=sys.stderr)
                sys.exit(1)

    # Build deduped config list (baseline shared across all sweeps).  Only the
    # sweeps named in ENABLED_SWEEPS contribute configs, so the run can be phased.
    # Config tuple: (sweep_name, num_dcs, target_util, large_frac, distribution, pred_noise)
    seen, configs = set(), []
    def _add(sweep, dcs, util, lf, dist=BASELINE_DISTRIBUTION, noise=BASELINE_PRED_NOISE):
        if sweep not in ENABLED_SWEEPS:
            return
        key = (dcs, util, lf, dist, noise)
        if key not in seen:
            seen.add(key)
            configs.append((sweep, dcs, util, lf, dist, noise))

    # Baseline (shared anchor for every sweep)
    _add("baseline",  BASELINE_DCS,  BASELINE_UTIL,       BASELINE_LARGE_FRAC)
    # DC sweep
    for dcs  in DC_SWEEP_VALUES:         _add("dc_sweep",   dcs,          BASELINE_UTIL, BASELINE_LARGE_FRAC)
    # Util sweep
    for util in UTIL_SWEEP_VALUES:       _add("util_sweep", BASELINE_DCS, util,          BASELINE_LARGE_FRAC)
    # Model-mix sweep
    for lf   in LARGE_FRAC_SWEEP_VALUES: _add("mix_sweep",  BASELINE_DCS, BASELINE_UTIL, lf)
    # Origin-pattern sweep (population & time; even = baseline already added)
    for dist in ORIGIN_SWEEP_VALUES:
        if dist != BASELINE_DISTRIBUTION:
            _add("origin_sweep", BASELINE_DCS, BASELINE_UTIL, BASELINE_LARGE_FRAC, dist=dist)
    # Prediction-noise sweep (0.0 = baseline already added)
    for noise in PRED_NOISE_SWEEP_VALUES:
        if noise != BASELINE_PRED_NOISE:
            _add("noise_sweep", BASELINE_DCS, BASELINE_UTIL, BASELINE_LARGE_FRAC, noise=noise)

    # An origin/noise-only phase still needs the baseline anchor for its tables;
    # those rows come from an earlier phase's CSV, so warn if neither is present.
    if not configs:
        print("ERROR: ENABLED_SWEEPS produced no configs — nothing to run.",
              file=sys.stderr)
        sys.exit(1)

    unique_lf_all     = sorted({lf for _, _, _, lf, _, _ in configs})
    total_trace_files = len(TRACES) * len(unique_lf_all)
    total_runs        = len(TRACES) * len(configs) * len(FRAMEWORKS) * NUM_RUNS

    # Resume: how many of these cells already have a usable result on disk?
    prior_status = _load_run_status(csv_path)
    already_done = 0
    for tr in TRACES:
        for _, dcs, util, lf, dist, noise in configs:
            for fw in FRAMEWORKS:
                for run_num in range(1, NUM_RUNS + 1):
                    k = _run_key(tr["name"], fw, dcs, util, lf, dist, noise, run_num)
                    if _is_done(prior_status.get(k, "")):
                        already_done += 1
    remaining = total_runs - already_done

    print("=" * 76)
    print(f"  OVERNIGHT SWEEP  —  {len(TRACES)} TRACES x {len(configs)} CONFIGS x "
          f"{len(FRAMEWORKS)} FRAMEWORKS x {NUM_RUNS} RUNS = {total_runs}")
    print("-" * 76)
    for tr in TRACES:
        print(f"  {tr['label']:<14}  {', '.join(tr['inputs'])}")
    print("-" * 76)
    print(f"  Enabled sw.: {', '.join(sorted(ENABLED_SWEEPS))}")
    print(f"  Frameworks : {', '.join(FRAMEWORKS)}")
    print(f"  parliament : online-only (no --offline-train)")
    print(f"  DC sweep   : {DC_SWEEP_VALUES} @ util={BASELINE_UTIL*100:.0f}%  lf={BASELINE_LARGE_FRAC:.2f}")
    print(f"  Util sweep : {[int(u*100) for u in UTIL_SWEEP_VALUES]}% @ {BASELINE_DCS}DCs  lf={BASELINE_LARGE_FRAC:.2f}")
    print(f"  Mix sweep  : lf={LARGE_FRAC_SWEEP_VALUES} @ {BASELINE_DCS}DCs  util={BASELINE_UTIL*100:.0f}%")
    print(f"  Origin sw. : dist={ORIGIN_SWEEP_VALUES} @ baseline config")
    print(f"  Noise sw.  : pred_noise={PRED_NOISE_SWEEP_VALUES} @ baseline config")
    print(f"  TTFT calib : helix avg TTFT -> [{TTFT_BAND[0]:.1f}, {TTFT_BAND[1]:.1f}] s, "
          f"cache={CALIB_FILE}" if TTFT_CALIBRATE
          else "  TTFT calib : disabled (static token scaling)")
    print(f"  Trace files: {total_trace_files}  |  NUM_RUNS={NUM_RUNS}  (need >=2 for 95% CI)")
    print(f"  Hang guard : idle timeout {PER_RUN_IDLE_TIMEOUT_MIN} min"
          if PER_RUN_IDLE_TIMEOUT_MIN
          else "  Hang guard : disabled (runs never interrupted)")
    print(f"  Resume     : {already_done} done, {remaining} remaining "
          f"(retry-failed={RERUN_FAILED})")
    print(f"  Log : {log_path}")
    print(f"  CSV : {csv_path}")
    print("=" * 76 + "\n")

    # Preflight: make sure each sweep's flag is actually understood downstream.
    _preflight_flags(script_dir)

    with open(log_path, "a") as log_file, \
         open(csv_path,  "a", newline="") as summary_file:

        writer = csv.DictWriter(summary_file, fieldnames=FIELDNAMES)
        if csv_is_new:
            writer.writeheader()
        log_file.write(f"\n=== STARTED {datetime.now()} ===\n"
                       f"Traces={len(TRACES)} Configs={len(configs)} "
                       f"Frameworks={len(FRAMEWORKS)} Runs={NUM_RUNS} "
                       f"Total={total_runs}\n\n")
        log_file.flush()

        # ── Phase 1: Pre-generate all traces ─────────────────────────────────
        print("[PHASE 1] Pre-generating traces ...")
        trace_paths = {}      # {(trace_name, lf): file_path}
        for tr in TRACES:
            for lf in unique_lf_all:
                try:
                    path = generate_trace(tr, lf, script_dir, log_file)
                    trace_paths[(tr["name"], lf)] = path
                except RuntimeError as exc:
                    err = f"\n[FATAL] {exc}\n"
                    print(err); log_file.write(err)
                    sys.exit(1)
        print(f"[PHASE 1] Done — {total_trace_files} trace file(s) ready.\n")

        # ── Phase 2: Run experiments ──────────────────────────────────────────
        print("[PHASE 2] Running experiments ...")
        run_counter  = 0            # position in the full matrix (1..total_runs)
        executed     = 0            # actually launched this session
        skipped      = 0            # already done on disk
        active_key   = None         # (trace_name, lf) currently active
        elapsed_hist = []           # minutes per executed run, for ETA

        for tr in TRACES:
            print(f"\n{'#'*70}")
            print(f"  TRACE: {tr['label']}")
            print(f"{'#'*70}")
            log_file.write(f"\n{'#'*70}\nTRACE: {tr['label']}\n{'#'*70}\n")

            for sweep_name, num_dcs, target_util, large_frac, distribution, pred_noise in configs:
                config_label = (f"{sweep_name:<12} DCs={num_dcs} "
                                f"util={target_util*100:.0f}% lf={large_frac:.2f} "
                                f"dist={distribution} noise={pred_noise:.2f}")

                for framework in FRAMEWORKS:
                    for run_num in range(1, NUM_RUNS + 1):
                        run_counter += 1
                        key = _run_key(tr["name"], framework, num_dcs,
                                       target_util, large_frac, distribution,
                                       pred_noise, run_num)

                        # Resume: skip cells that already have a usable result.
                        # (In calibrate-only mode nothing executes anyway, so
                        # never skip — every config must still get calibrated.)
                        if not CALIBRATE_ONLY and _is_done(prior_status.get(key, "")):
                            skipped += 1
                            print(f"[SKIP {run_counter}/{total_runs}] "
                                  f"{tr['name']}|{framework}|{config_label}|run{run_num} "
                                  f"(status={prior_status.get(key)})")
                            continue

                        # Activate the right trace only when a run will actually
                        # execute (avoids needless copies on a full resume).
                        tk = (tr["name"], large_frac)
                        if tk != active_key:
                            activate_trace(trace_paths[tk], script_dir)
                            active_key = tk
                            msg = f"\n[TRACE] Active: {os.path.basename(trace_paths[tk])}\n"
                            print(msg, end=""); log_file.write(msg)

                        # One calibrated token scale per (trace, config) —
                        # measured against the helix baseline, cached on disk,
                        # then shared by every framework of this config.  Runs
                        # lazily here (not up-front) because the right trace
                        # must be active, and a fully-resumed config should
                        # never trigger a needless calibration.
                        ts_target = _ensure_token_scale(
                            tr, num_dcs, target_util, large_frac,
                            distribution, pred_noise, script_dir, log_file)

                        # --calibrate-only: the calibration above (which also
                        # seeds the autoscale plan cache) is the whole job for
                        # this cell — both caches are keyed per (trace, config),
                        # so the in-memory cache makes the framework x run
                        # repeats of the same config instant no-ops.
                        if CALIBRATE_ONLY:
                            continue

                        eta = ""
                        if elapsed_hist:
                            mean_min = sum(elapsed_hist) / len(elapsed_hist)
                            left     = total_runs - run_counter + 1
                            eta_h    = mean_min * left / 60.0
                            eta = f"  ~{mean_min:.1f} min/run, ETA {eta_h:.1f} h"

                        print(f"\n{'='*70}")
                        print(f"Run {run_counter}/{total_runs}  "
                              f"(exec {executed+1}, skipped {skipped}){eta}")
                        print(f"  [{tr['label']}|{framework}|{config_label}|run{run_num}]")
                        print(f"{'='*70}")
                        log_file.write(f"\n{'='*70}\n"
                                       f"Run {run_counter}/{total_runs} "
                                       f"[{tr['name']}|{framework}|{config_label}|{run_num}]\n")

                        t0 = time.time()
                        try:
                            _run_one(tr, framework, sweep_name, num_dcs,
                                     target_util, large_frac, run_num,
                                     script_dir, log_file, writer, summary_file,
                                     run_counter,
                                     distribution=distribution,
                                     prediction_noise=pred_noise,
                                     token_scale_target=ts_target)
                            executed += 1
                            elapsed_hist.append((time.time() - t0) / 60.0)
                        except Exception as exc:
                            err = (f"\n[{datetime.now()}] FATAL: "
                                   f"{tr['label']}|{framework}|{config_label} "
                                   f"run{run_num}: {exc}\n")
                            print(err); log_file.write(err); log_file.flush()

        if CALIBRATE_ONLY:
            print(f"\n[PHASE 2] CALIBRATE-ONLY done — no experiments executed. "
                  f"Commit {CALIB_FILE} and {PLAN_CACHE_FILE} to the shared repo "
                  f"and pull on the other machines before starting their sweeps.")
        print(f"\n[PHASE 2] Done — {executed} run(s) executed, "
              f"{skipped} skipped (already done).")
        log_file.write(f"\n=== COMPLETED {datetime.now()}  "
                       f"executed={executed} skipped={skipped} ===\n")

    # ── Phase 3: Analysis & reporting ────────────────────────────────────────
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED")
    print("="*80)
    print(f"Log : {log_path}")
    print(f"CSV : {csv_path}\n")

    try:
        import pandas as pd

        df = pd.read_csv(csv_path)
        # Long-form CSV: the summary tables operate on run-level rows only;
        # the scope=<mode> rows are the lahyper per-agent breakdown.
        if "scope" in df.columns:
            df = df[df["scope"] == "run"].copy()
        # Only average runs that produced a final report.  crashed/timeout/
        # no-data rows stay in the CSV for the coverage audit but must not enter
        # the means (their summed metrics span fewer epochs and aren't
        # comparable).  Keep "ok" and "ok-short".
        if "status" in df.columns:
            usable = df["status"].fillna("").str.startswith("ok")
            n_drop = int((~usable).sum())
            if n_drop:
                print(f"  (excluding {n_drop} non-ok run row(s) from the "
                      f"sweep tables; see coverage audit below)")
            df = df[usable].copy()
        for col in METRICS + ["total_energy_kwh", "avg_epoch_phv",
                               "target_util", "num_dcs", "large_frac", "prediction_noise"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        if "distribution" not in df.columns:
            df["distribution"] = BASELINE_DISTRIBUTION

        # ── Per-trace sweep tables ────────────────────────────────────────────
        for tr in TRACES:
            tname  = tr["name"]
            tlabel = tr["label"]
            tdf    = df[df["trace"] == tname].copy()

            print(f"\n{'#'*80}")
            print(f"  TRACE: {tlabel}")
            print(f"{'#'*80}")

            # DC Sweep (even dist, no noise)
            dc_df = tdf[
                np.isclose(tdf["target_util"],  BASELINE_UTIL,       atol=1e-4) &
                np.isclose(tdf["large_frac"],   BASELINE_LARGE_FRAC, atol=1e-4) &
                (tdf["distribution"] == BASELINE_DISTRIBUTION) &
                np.isclose(tdf["prediction_noise"].fillna(0.0), BASELINE_PRED_NOISE, atol=1e-4)
            ].copy()
            print(f"\n  -- DC SWEEP  (util={BASELINE_UTIL*100:.0f}%  "
                  f"lf={BASELINE_LARGE_FRAC:.2f}  dist=even  noise=0  {NUM_RUNS} run(s))")
            print(f"  Format: avg+/-95%CI [min, max]")
            for metric, mlabel in zip(METRICS, LABELS):
                _print_sweep_table(dc_df, "num_dcs", DC_SWEEP_VALUES,
                                   metric, mlabel, FRAMEWORKS)

            # Util Sweep
            util_df = tdf[
                (tdf["num_dcs"] == BASELINE_DCS) &
                np.isclose(tdf["large_frac"], BASELINE_LARGE_FRAC, atol=1e-4) &
                (tdf["distribution"] == BASELINE_DISTRIBUTION) &
                np.isclose(tdf["prediction_noise"].fillna(0.0), BASELINE_PRED_NOISE, atol=1e-4)
            ].copy()
            print(f"\n  -- UTIL SWEEP  ({BASELINE_DCS} DCs  "
                  f"lf={BASELINE_LARGE_FRAC:.2f}  dist=even  noise=0  {NUM_RUNS} run(s))")
            print(f"  Format: avg+/-95%CI [min, max]")
            for metric, mlabel in zip(METRICS, LABELS):
                _print_sweep_table(util_df, "target_util", UTIL_SWEEP_VALUES,
                                   metric, mlabel, FRAMEWORKS)

            # Model Mix Sweep
            mix_df = tdf[
                (tdf["num_dcs"] == BASELINE_DCS) &
                np.isclose(tdf["target_util"], BASELINE_UTIL, atol=1e-4) &
                (tdf["distribution"] == BASELINE_DISTRIBUTION) &
                np.isclose(tdf["prediction_noise"].fillna(0.0), BASELINE_PRED_NOISE, atol=1e-4)
            ].copy()
            print(f"\n  -- MODEL MIX SWEEP  ({BASELINE_DCS} DCs  "
                  f"util={BASELINE_UTIL*100:.0f}%  dist=even  noise=0  {NUM_RUNS} run(s))")
            print(f"  lf: 0.0=all small  1.0=all large")
            print(f"  Format: avg+/-95%CI [min, max]")
            for metric, mlabel in zip(METRICS, LABELS):
                _print_sweep_table(mix_df, "large_frac", LARGE_FRAC_SWEEP_VALUES,
                                   metric, mlabel, FRAMEWORKS)

            # Origin Pattern Sweep
            origin_df = tdf[
                (tdf["num_dcs"] == BASELINE_DCS) &
                np.isclose(tdf["target_util"],  BASELINE_UTIL,       atol=1e-4) &
                np.isclose(tdf["large_frac"],   BASELINE_LARGE_FRAC, atol=1e-4) &
                np.isclose(tdf["prediction_noise"].fillna(0.0), BASELINE_PRED_NOISE, atol=1e-4)
            ].copy()
            print(f"\n  -- ORIGIN PATTERN SWEEP  ({BASELINE_DCS} DCs  "
                  f"util={BASELINE_UTIL*100:.0f}%  lf={BASELINE_LARGE_FRAC:.2f}  noise=0  {NUM_RUNS} run(s))")
            print(f"  Distributions: even (round-robin) | population (pop-weighted) | time (tz-shifted)")
            print(f"  Format: avg+/-95%CI [min, max]")
            for metric, mlabel in zip(METRICS, LABELS):
                _print_sweep_table_str(origin_df, "distribution", ORIGIN_SWEEP_VALUES,
                                       metric, mlabel, FRAMEWORKS)

            # Prediction Noise Sweep
            noise_df = tdf[
                (tdf["num_dcs"] == BASELINE_DCS) &
                np.isclose(tdf["target_util"],  BASELINE_UTIL,       atol=1e-4) &
                np.isclose(tdf["large_frac"],   BASELINE_LARGE_FRAC, atol=1e-4) &
                (tdf["distribution"] == BASELINE_DISTRIBUTION)
            ].copy()
            print(f"\n  -- PREDICTION NOISE SWEEP  ({BASELINE_DCS} DCs  "
                  f"util={BASELINE_UTIL*100:.0f}%  lf={BASELINE_LARGE_FRAC:.2f}  dist=even  {NUM_RUNS} run(s))")
            print(f"  noise=0.0 is perfect forecast; 0.30 = ±30% token/volume/origin error")
            print(f"  Format: avg+/-95%CI [min, max]")
            for metric, mlabel in zip(METRICS, LABELS):
                _print_sweep_table(noise_df, "prediction_noise", PRED_NOISE_SWEEP_VALUES,
                                   metric, mlabel, FRAMEWORKS)

        # ── Cross-trace baseline comparison ───────────────────────────────────
        bl = df[
            (df["num_dcs"] == BASELINE_DCS) &
            np.isclose(df["target_util"],  BASELINE_UTIL,       atol=1e-4) &
            np.isclose(df["large_frac"],   BASELINE_LARGE_FRAC, atol=1e-4) &
            (df["distribution"].fillna(BASELINE_DISTRIBUTION) == BASELINE_DISTRIBUTION) &
            np.isclose(df["prediction_noise"].fillna(0.0), BASELINE_PRED_NOISE, atol=1e-4)
        ].copy()

        print(f"\n{'='*80}")
        print(f"  CROSS-TRACE BASELINE COMPARISON")
        print(f"  {BASELINE_DCS} DCs  {BASELINE_UTIL*100:.0f}% util  "
              f"large_frac={BASELINE_LARGE_FRAC:.2f}  ({NUM_RUNS} run(s))")
        print(f"  Format: avg+/-95%CI [min, max]")
        print(f"{'='*80}")

        W            = 34
        trace_labels = [tr["label"] for tr in TRACES]

        for metric, mlabel in zip(METRICS, LABELS):
            print(f"\n  {mlabel}:")
            hdr = f"  {'Framework':<14}"
            for tlabel in trace_labels:
                hdr += f"  {tlabel:^{W}}"
            print(hdr)
            print(f"  {'':->14}", end="")
            for _ in trace_labels:
                print(f"  {'':->{W}}", end="")
            print()
            for fw in FRAMEWORKS:
                row_str = f"  {fw:<14}"
                for tr in TRACES:
                    sub = bl[(bl["trace"] == tr["name"]) & (bl["framework"] == fw)]
                    row_str += f"  {_fmt(_stats(sub[metric])):^{W}}"
                print(row_str)

        print(f"\n  PHV (per-epoch Pareto Hypervolume):")
        hdr = f"  {'Framework':<14}"
        for tlabel in trace_labels:
            hdr += f"  {tlabel:^24}"
        print(hdr)
        print(f"  {'':->14}", end="")
        for _ in trace_labels:
            print(f"  {'':->24}", end="")
        print()
        for fw in FRAMEWORKS:
            row_str = f"  {fw:<14}"
            for tr in TRACES:
                sub = bl[(bl["trace"] == tr["name"]) & (bl["framework"] == fw)]
                row_str += f"  {_fmt_phv(_stats(sub['avg_epoch_phv'])):^24}"
            print(row_str)

        print(f"\n{'='*80}")
        print(f"  Raw data: {csv_path}")
        print(f"{'='*80}")

    except Exception as e:
        print(f"\n(Summary error: {e})")
        import traceback; traceback.print_exc()

    # Coverage audit runs regardless of whether the pandas summary succeeded —
    # it depends only on the CSV and the intended matrix.
    try:
        _audit_coverage(csv_path, TRACES, FRAMEWORKS, configs, NUM_RUNS)
    except Exception as e:
        print(f"\n(Coverage audit error: {e})")
        import traceback; traceback.print_exc()

    # Final push so the very last results land regardless of the throttle.
    if os.environ.get("AUTOPUSH", "1") != "0":
        try:
            autopush_results.push_once(min_interval=0)
        except Exception as e:
            print(f"(final push error: {e})")


if __name__ == "__main__":
    run_experiments()