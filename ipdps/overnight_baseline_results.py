"""
overnight_sweep.py
==================
Overnight experiment runner covering three trace sources and three sweeps.

Traces
──────
  1. BurstGPT     — BurstGPT_1.csv
  2. Azure-Code   — AzureLLMInferenceTrace_code_1week.csv
  3. Azure-Conv   — AzureLLMInferenceTrace_conv_1week.csv

Sweeps (per trace)
──────────────────
  1. DC Sweep    — [4, 6, 8, 12] DCs @ baseline util & model mix
  2. Util Sweep  — [65%, 75%, 85%, 95%, 105%] util @ 8 DCs & baseline mix
  3. Mix Sweep   — large_frac [0.0, 0.25, 0.50, 0.75, 1.0] @ 8 DCs & 95% util

Baseline: 8 DCs / 95% util / 0.75 large_frac — shared across all sweeps and
executed only once per (trace, framework) pair to avoid redundant runs.

Frameworks
──────────
  All FRAMEWORKS are benchmarked.  parliament (Game_Theoretic_RL) runs in
  online-only mode — no --offline-train flag is ever passed.

Runtime estimate
────────────────
  NUM_RUNS=1:  3 traces × 12 configs × 7 frameworks × 1 = 252 total runs
  NUM_RUNS=3:  3 × 12 × 7 × 3 = 756 runs  (CI is meaningful; plan a weekend)
"""

import csv
import os
import re
import shutil
import subprocess
import sys
import time
import numpy as np
from datetime import datetime

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
FRAMEWORKS = [   # online-only — no --offline-train
    "lahyper",
    "parliament",
    "hybrid",
    "helix",
    "nsga2",
    "perllm",
    "splitwise",
    "qlearning",
    "ddqn",
    "actorcritic",
]

# ── Sweep axes ────────────────────────────────────────────────────────────────
BASELINE_DCS        = 8
BASELINE_UTIL       = 0.95
BASELINE_LARGE_FRAC = 0.75

DC_SWEEP_VALUES         = [4, 6, 8, 12]
UTIL_SWEEP_VALUES       = [0.65, 0.75, 0.85, 0.95, 1.05]
LARGE_FRAC_SWEEP_VALUES = [0.0, 0.25, 0.50, 0.75, 1.0]

# Origin pattern sweep — source-DC assignment distribution
BASELINE_DISTRIBUTION   = "even"
ORIGIN_SWEEP_VALUES     = ["even", "population", "time"]

# Workload prediction inaccuracy sweep — noise fraction in [0, 1]
BASELINE_PRED_NOISE     = 0.0
PRED_NOISE_SWEEP_VALUES = [0.0, 0.10, 0.20, 0.30]

NUM_RUNS   = 3      # per (trace, config, framework) — CI needs n >= 2
NUM_EPOCHS = 96     # 96 x 15 min = 24 hours

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

# ── CSV schema ────────────────────────────────────────────────────────────────
# Top-level summary fields per run.
BASE_FIELDNAMES = [
    "trace", "framework", "sweep", "num_dcs", "target_util", "large_frac",
    "distribution", "prediction_noise",
    "run", "start_time", "end_time", "elapsed_min", "exit_code", "epochs",
    "avg_ttft_s", "total_carbon_kg", "total_water_m3",
    "total_energy_usd", "total_energy_kwh",
    "avg_epoch_phv",
]

# Per-mode aggregate fields, sourced from the LA_HYPER MULTI-AGENT SUMMARY
# table at the end of every lahyper run.  TTFT is averaged across epochs;
# carbon/water/cost/total_energy are summed across epochs (matching the
# simulator's own aggregation).
#
# Keep this list in sync with the modes defined in LA_Hyper_DDQN.py.
PER_MODE_NAMES = [
    "time_agent", "carbon_agent", "water_agent", "cost_agent",
    "Balanced", "green_perf", "cost_guard", "water_saver", "peak_power_guard",
    "Pareto_Sample",
]
PER_MODE_METRICS = ["ttft", "carbon", "water", "cost", "total_energy"]

PER_MODE_FIELDS = [
    f"{mode}_{metric}"
    for mode in PER_MODE_NAMES
    for metric in PER_MODE_METRICS
]

FIELDNAMES = BASE_FIELDNAMES + PER_MODE_FIELDS + ["command"]

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

def _parse_final_report(lines):
    text    = "".join(lines)
    metrics = {k: "" for k in [
        "epochs", "avg_ttft_s", "total_carbon_kg",
        "total_water_m3", "total_energy_usd", "total_energy_kwh",
    ]}
    for key, pat in {
        "epochs":           r"Epochs:\s*([0-9]+)",
        "avg_ttft_s":       r"Average TTFT\s*\([^)]+\):\s*([0-9eE+\-\.]+)",
        "total_carbon_kg":  r"Total Carbon\s*\([^)]+\):\s*([0-9eE+\-\.]+)",
        "total_water_m3":   r"Total Water\s*\([^)]+\):\s*([0-9eE+\-\.]+)",
        "total_energy_usd": r"Total Energy\s*\(\$\):\s*([0-9eE+\-\.]+)",
        "total_energy_kwh": r"Total Energy\s*\(kWh\):\s*([0-9eE+\-\.]+)",
    }.items():
        m = re.search(pat, text)
        if m:
            metrics[key] = m.group(1)
    return metrics


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
                   distribution="even", prediction_noise=0.0):
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
    return cmd


# ── Single-run executor ───────────────────────────────────────────────────────

def _run_one(trace_cfg, framework, sweep_name, num_dcs, target_util,
             large_frac, run_num, script_dir, log_file, writer, summary_file,
             distribution="even", prediction_noise=0.0):
    cmd        = _build_command(framework, num_dcs, target_util,
                                distribution=distribution,
                                prediction_noise=prediction_noise)
    start_dt   = datetime.now()
    start_time = time.time()

    hdr = (f"\n[{start_dt}] {trace_cfg['label']:<12}|{framework.upper():<12}|"
           f"sweep={sweep_name} DCs={num_dcs} util={target_util*100:.0f}% "
           f"lf={large_frac:.2f} run={run_num}/{NUM_RUNS}")
    print(hdr); log_file.write(hdr + "\n")
    log_file.write(f"  cmd: {' '.join(cmd)}\n" + "-"*70 + "\n")
    log_file.flush()

    output_lines, exit_code, process = [], -1, None
    try:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, cwd=script_dir)
        try:
            for line in process.stdout:
                print(line, end=""); log_file.write(line)
                output_lines.append(line); log_file.flush()
        except (BrokenPipeError, IOError, ValueError):
            pass
        process.wait(timeout=30)
        exit_code = int(process.returncode)
        if exit_code != 0:
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

    parsed     = _parse_final_report(output_lines)
    avg_ep_phv = _parse_epoch_phv(output_lines)
    per_mode   = _parse_per_mode(output_lines)
    if avg_ep_phv:
        print(f"  [PHV] {framework}@{trace_cfg['label']} = {avg_ep_phv}")

    row = {
        "trace":            trace_cfg["name"],
        "framework":        framework,
        "sweep":            sweep_name,
        "num_dcs":          num_dcs,
        "target_util":      f"{target_util:.2f}",
        "large_frac":       f"{large_frac:.2f}",
        "distribution":     distribution,
        "prediction_noise": f"{prediction_noise:.2f}",
        "run":              run_num,
        "start_time":       start_dt.isoformat(timespec="seconds"),
        "end_time":         datetime.now().isoformat(timespec="seconds"),
        "elapsed_min":      f"{elapsed_min:.2f}",
        "exit_code":        str(exit_code),
        "epochs":           parsed["epochs"],
        "avg_ttft_s":       parsed["avg_ttft_s"],
        "total_carbon_kg":  parsed["total_carbon_kg"],
        "total_water_m3":   parsed["total_water_m3"],
        "total_energy_usd": parsed["total_energy_usd"],
        "total_energy_kwh": parsed["total_energy_kwh"],
        "avg_epoch_phv":    avg_ep_phv,
        "command":          " ".join(cmd),
    }
    # Fill per-mode columns; missing modes get "".
    for field in PER_MODE_FIELDS:
        row[field] = per_mode.get(field, "")
    writer.writerow(row); summary_file.flush()
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
        print(f"  {'':->width}", end="")
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
        print(f"  {'':->width}", end="")
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


# ── Main ──────────────────────────────────────────────────────────────────────

def run_experiments():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir    = os.path.join(script_dir, "experiment_results")
    os.makedirs(out_dir, exist_ok=True)
    log_path   = os.path.join(out_dir, "LAHyper_Full_Experiments.log")
    csv_path   = os.path.join(out_dir, "LAHyper_Full_Experiments.csv")

    # Append mode — safe to restart: existing rows are preserved, header is
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

    # Build deduped config list (baseline shared across all sweeps)
    # Config tuple: (sweep_name, num_dcs, target_util, large_frac, distribution, pred_noise)
    seen, configs = set(), []
    def _add(sweep, dcs, util, lf, dist=BASELINE_DISTRIBUTION, noise=BASELINE_PRED_NOISE):
        key = (dcs, util, lf, dist, noise)
        if key not in seen:
            seen.add(key)
            configs.append((sweep, dcs, util, lf, dist, noise))

    # Baseline (shared across all non-noise, non-origin sweeps)
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

    unique_lf_all     = sorted({lf for _, _, _, lf, _, _ in configs})
    total_trace_files = len(TRACES) * len(unique_lf_all)
    total_runs        = len(TRACES) * len(configs) * len(FRAMEWORKS) * NUM_RUNS

    print("=" * 76)
    print(f"  OVERNIGHT SWEEP  —  {len(TRACES)} TRACES x {len(configs)} CONFIGS x "
          f"{len(FRAMEWORKS)} FRAMEWORKS x {NUM_RUNS} RUNS = {total_runs}")
    print("-" * 76)
    for tr in TRACES:
        print(f"  {tr['label']:<14}  {', '.join(tr['inputs'])}")
    print("-" * 76)
    print(f"  Frameworks : {', '.join(FRAMEWORKS)}")
    print(f"  parliament : online-only (no --offline-train)")
    print(f"  DC sweep   : {DC_SWEEP_VALUES} @ util={BASELINE_UTIL*100:.0f}%  lf={BASELINE_LARGE_FRAC:.2f}")
    print(f"  Util sweep : {[int(u*100) for u in UTIL_SWEEP_VALUES]}% @ {BASELINE_DCS}DCs  lf={BASELINE_LARGE_FRAC:.2f}")
    print(f"  Mix sweep  : lf={LARGE_FRAC_SWEEP_VALUES} @ {BASELINE_DCS}DCs  util={BASELINE_UTIL*100:.0f}%")
    print(f"  Origin sw. : dist={ORIGIN_SWEEP_VALUES} @ baseline config")
    print(f"  Noise sw.  : pred_noise={PRED_NOISE_SWEEP_VALUES} @ baseline config")
    print(f"  Trace files: {total_trace_files}  |  NUM_RUNS={NUM_RUNS}  (need >=2 for 95% CI)")
    print(f"  Log : {log_path}")
    print(f"  CSV : {csv_path}")
    print("=" * 76 + "\n")

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
        run_counter  = 0
        active_key   = None   # (trace_name, lf) currently active

        for tr in TRACES:
            print(f"\n{'#'*70}")
            print(f"  TRACE: {tr['label']}")
            print(f"{'#'*70}")
            log_file.write(f"\n{'#'*70}\nTRACE: {tr['label']}\n{'#'*70}\n")

            for sweep_name, num_dcs, target_util, large_frac, distribution, pred_noise in configs:
                config_label = (f"{sweep_name:<12} DCs={num_dcs} "
                                f"util={target_util*100:.0f}% lf={large_frac:.2f} "
                                f"dist={distribution} noise={pred_noise:.2f}")

                tk = (tr["name"], large_frac)
                if tk != active_key:
                    activate_trace(trace_paths[tk], script_dir)
                    active_key = tk
                    msg = f"\n[TRACE] Active: {os.path.basename(trace_paths[tk])}\n"
                    print(msg, end=""); log_file.write(msg)

                for framework in FRAMEWORKS:
                    for run_num in range(1, NUM_RUNS + 1):
                        run_counter += 1
                        print(f"\n{'='*70}")
                        print(f"Run {run_counter}/{total_runs}  "
                              f"[{tr['label']}|{framework}|{config_label}|run{run_num}]")
                        print(f"{'='*70}")
                        log_file.write(f"\n{'='*70}\n"
                                       f"Run {run_counter}/{total_runs} "
                                       f"[{tr['name']}|{framework}|{config_label}|{run_num}]\n")
                        try:
                            _run_one(tr, framework, sweep_name, num_dcs,
                                     target_util, large_frac, run_num,
                                     script_dir, log_file, writer, summary_file,
                                     distribution=distribution,
                                     prediction_noise=pred_noise)
                        except Exception as exc:
                            err = (f"\n[{datetime.now()}] FATAL: "
                                   f"{tr['label']}|{framework}|{config_label} "
                                   f"run{run_num}: {exc}\n")
                            print(err); log_file.write(err); log_file.flush()

        log_file.write(f"\n=== COMPLETED {datetime.now()} ===\n")

    # ── Phase 3: Analysis & reporting ────────────────────────────────────────
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED")
    print("="*80)
    print(f"Log : {log_path}\nCSV : {csv_path}\n")

    try:
        import pandas as pd

        df = pd.read_csv(csv_path)
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
                print(f"  {'':->W}", end="")
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


if __name__ == "__main__":
    run_experiments()