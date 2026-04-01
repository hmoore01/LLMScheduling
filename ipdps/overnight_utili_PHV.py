import csv
import os
import re
import subprocess
import sys
import time
import numpy as np
from datetime import datetime
from itertools import product

# ── Experiment Matrix ─────────────────────────────────────────────────────────

FRAMEWORKS = [
    "parliament",
    "nsga2",
    "helix",
    "splitwise",
    "actorcritic",
    "ddqn",
    "qlearning",
    "perllm",
]

# Sweep: vary utilization, fix DC count at 8
UTIL_SWEEP_VALUES = [0.75, 0.85, 0.95, 1.05]
FIXED_NUM_DCS = 8

# Parliament model directory
PARLIAMENT_MODEL_DIR = "models/"
PARLIAMENT_TRAINED_DCS = 12

# Statistical validation
NUM_RUNS = 5

# Shared autoscale / run settings
AUTOSCALE_MODE = "global_peak"
MAX_DROP = 0.02
MAX_MULT = 1_500_000
MAX_ROWS = 300_000
SEARCH_STEPS = 9

# ── CSV fieldnames ────────────────────────────────────────────────────────────

FIELDNAMES = [
    "framework", "scheme", "num_dcs", "target_util", "run",
    "start_time", "end_time", "elapsed_min", "exit_code", "epochs",
    "avg_ttft_s", "total_carbon_kg", "total_water_l",
    "total_energy_usd", "total_energy_kwh", "avg_epoch_phv", "command",
]

PARLIAMENT_SCHEMES = ["MinCost", "Balanced", "MinLatency", "MinCarbon", "MinWater"]


def _parse_epoch_phv(output_lines, fw):
    """Parse per-epoch objective points from output and compute average PHV.

    Parliament: 5 scheme rows per epoch from the epoch table.
    NSGA2: [NSGA2-FRONT] lines per epoch (re-executed on fresh sims).
    Others: single metrics line per epoch.

    Returns average PHV across epochs, or "" if not enough data.
    """
    text = "".join(output_lines)

    # Collect points per epoch: {epoch_idx: [[ttft, carbon, water, cost], ...]}
    epoch_points = {}

    if fw == "parliament":
        # Parse epoch tables: each has 5 scheme rows
        # │ MinCost          0.998      354.5      237.5      128.9★    16662  │
        epoch_pat = re.compile(r"EPOCH\s+(\d+)")
        scheme_pat = re.compile(
            r"(?:│\s*)?(MinCost|Balanced|MinLatency|MinCarbon|MinWater)\s+"
            r"([0-9eE+\-\.★]+)\s+"
            r"([0-9eE+\-\.★]+)\s+"
            r"([0-9eE+\-\.★]+)\s+"
            r"([0-9eE+\-\.★]+)\s+"
            r"([0-9]+)"
        )
        cur_epoch = None
        for line in text.split("\n"):
            ep_m = epoch_pat.search(line)
            if ep_m:
                cur_epoch = int(ep_m.group(1))
                continue
            if cur_epoch is not None:
                sm = scheme_pat.search(line)
                if sm:
                    pt = [float(sm.group(i).replace("★", "")) for i in range(2, 6)]
                    epoch_points.setdefault(cur_epoch, []).append(pt)

    elif fw == "nsga2":
        # Parse [NSGA2-FRONT] lines
        front_pat = re.compile(
            r"\[NSGA2-FRONT\]\s+epoch=(\d+)\s+member=\d+\s+"
            r"ttft=([0-9eE+\-\.]+)\s+carbon=([0-9eE+\-\.]+)\s+"
            r"water=([0-9eE+\-\.]+)\s+cost=([0-9eE+\-\.]+)"
        )
        for m in front_pat.finditer(text):
            ep = int(m.group(1))
            pt = [float(m.group(i)) for i in range(2, 6)]
            epoch_points.setdefault(ep, []).append(pt)

    else:
        # Single-solution frameworks: parse epoch metrics from log
        # [EPOCH N] ... avg_ttft=X.XX carbon=XX water=XX cost=XX
        # Or from the epoch table (single row for the framework)
        epoch_pat = re.compile(r"EPOCH\s+(\d+)")
        # Try to find per-epoch metric lines
        metrics_pat = re.compile(
            r"avg_ttft.*?([0-9]+\.[0-9]+).*?"
            r"carbon.*?([0-9]+\.[0-9]+).*?"
            r"water.*?([0-9]+\.[0-9]+).*?"
            r"(?:cost|energy_cost).*?([0-9]+\.[0-9]+)"
        )
        cur_epoch = None
        for line in text.split("\n"):
            ep_m = epoch_pat.search(line)
            if ep_m:
                cur_epoch = int(ep_m.group(1))
                continue
            if cur_epoch is not None:
                mm = metrics_pat.search(line)
                if mm:
                    pt = [float(mm.group(i)) for i in range(1, 5)]
                    epoch_points.setdefault(cur_epoch, []).append(pt)
                    cur_epoch = None  # Only one result per epoch

    if len(epoch_points) < 5:
        return ""

    # Compute reference point from ALL points across ALL epochs
    all_pts = [pt for pts in epoch_points.values() for pt in pts]
    if not all_pts:
        return ""
    all_pts_arr = np.array(all_pts)
    ref_point = (all_pts_arr.max(axis=0) * 1.10).tolist()

    # Compute PHV per epoch, then average
    epoch_hvs = []
    for ep in sorted(epoch_points.keys()):
        pts = epoch_points[ep]
        if pts:
            hv = compute_hypervolume(pts, ref_point)
            epoch_hvs.append(hv)

    if not epoch_hvs:
        return ""
    return f"{np.mean(epoch_hvs):.2f}"


# Storage for NSGA2 Pareto front points per run (populated during execution)
# Key: (target_util, run_num) → list of [ttft, carbon, water, cost] points
_NSGA2_FRONTS = {}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_final_report(output_lines):
    """Parse the simulator's final report.  Returns a dict of aggregate metrics
    plus a list of per-scheme dicts (empty list for non-parliament frameworks)."""
    text = "".join(output_lines)
    metrics = {k: "" for k in [
        "epochs", "avg_ttft_s", "total_carbon_kg",
        "total_water_l", "total_energy_usd", "total_energy_kwh"]}
    patterns = {
        "epochs": r"Epochs:\s*([0-9]+)",
        "avg_ttft_s": r"Average TTFT \(s\):\s*([0-9eE+\-\.]+)",
        "total_carbon_kg": r"Total Carbon \(kg\):\s*([0-9eE+\-\.]+)",
        "total_water_l": r"Total Water \(L\):\s*([0-9eE+\-\.]+)",
        "total_energy_usd": r"Total Energy \(\$\):\s*([0-9eE+\-\.]+)",
        "total_energy_kwh": r"Total Energy \(kWh\):\s*([0-9eE+\-\.]+)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, text)
        if m:
            metrics[key] = m.group(1)

    scheme_rows = []
    summary_start = text.rfind("RUN SUMMARY")
    if summary_start >= 0:
        summary_text = text[summary_start:]
        scheme_pat = re.compile(
            r"^\s+(MinCost|Balanced|MinLatency|MinCarbon|MinWater)\s+"
            r"([0-9eE+\-\.★]+)\s+"
            r"([0-9eE+\-\.★]+)\s+"
            r"([0-9eE+\-\.★]+)\s+"
            r"([0-9eE+\-\.★]+)\s+"
            r"([0-9]+)\s+"
            r"([0-9]+)",
            re.MULTILINE
        )
        for m in scheme_pat.finditer(summary_text):
            scheme_rows.append({
                "scheme": m.group(1),
                "epochs": m.group(7),
                "avg_ttft_s": m.group(2).replace("★", ""),
                "total_carbon_kg": m.group(3).replace("★", ""),
                "total_water_l": m.group(4).replace("★", ""),
                "total_energy_usd": m.group(5).replace("★", ""),
                "total_energy_kwh": "",
            })

    return metrics, scheme_rows


def _build_command(fw, target_util, num_dcs, script_dir=""):
    cmd = [
        sys.executable, "-u", "simulator_LLM.py",
        "--framework", fw, "--num-dcs", str(num_dcs),
        "--target-util", str(target_util),
        "--autoscale-mode", AUTOSCALE_MODE,
        "--autoscale-max-drop", str(MAX_DROP),
        "--autoscale-max-mult", str(MAX_MULT),
        "--autoscale-max-rows", str(MAX_ROWS),
        "--autoscale-search-steps", str(SEARCH_STEPS),
    ]
    if fw == "parliament":
        if num_dcs == PARLIAMENT_TRAINED_DCS:
            model_dir = PARLIAMENT_MODEL_DIR
        else:
            model_dir = f"{PARLIAMENT_MODEL_DIR}_{num_dcs}dc"
        abs_dir = os.path.join(script_dir, model_dir) if script_dir \
            else os.path.abspath(model_dir)
        model_path = os.path.join(abs_dir, "gtarl_agents.pt")

        if os.path.exists(model_path):
            cmd.extend(["--model-dir", model_dir, "--load-model"])
            print(f"  [PARLIAMENT] {num_dcs} DCs — pre-trained + heuristic + online sims")
        else:
            abs_base = os.path.join(script_dir, PARLIAMENT_MODEL_DIR) if script_dir \
                else os.path.abspath(PARLIAMENT_MODEL_DIR)
            base_path = os.path.join(abs_base, "gtarl_agents.pt")
            if num_dcs != PARLIAMENT_TRAINED_DCS and os.path.exists(base_path):
                cmd.extend(["--model-dir", PARLIAMENT_MODEL_DIR,
                            "--transfer-from", base_path])
                print(f"  [PARLIAMENT] {num_dcs} DCs — transfer from base + heuristic + online sims")
            else:
                print(f"  [PARLIAMENT] {num_dcs} DCs — heuristic bootstrap + online sims (no model)")
    return cmd


def _run_one(fw, target_util, num_dcs, script_dir, log_file, writer,
             summary_file, run=1):
    cmd = _build_command(fw, target_util, num_dcs, script_dir=script_dir)
    start_dt = datetime.now()
    start_time = time.time()

    header = (f"\n[{start_dt}] FW={fw.upper()}  DCs={num_dcs}  "
              f"Util={target_util * 100:.0f}%  RUN={run}/{NUM_RUNS}")
    print(header)
    log_file.write(header + "\n")
    log_file.write(f"Command: {' '.join(cmd)}\n")
    log_file.write("-" * 60 + "\n")
    log_file.flush()

    output_lines, exit_code, process = [], -1, None
    try:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, cwd=script_dir)
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
            warn = f"\n[{datetime.now()}] WARNING: {fw.upper()} exited code {exit_code}\n"
            print(warn, end="");
            log_file.write(warn);
            log_file.flush()
    except subprocess.TimeoutExpired:
        process.kill();
        process.wait()
        err = f"\n[{datetime.now()}] TIMEOUT: {fw.upper()} killed after wait\n"
        print(err);
        log_file.write(err);
        log_file.flush()
    except Exception as exc:
        err = f"\nERROR running {fw}: {exc}\n"
        print(err);
        log_file.write(err)
        if process and process.poll() is None:
            try:
                process.kill();
                process.wait(timeout=10)
            except Exception:
                pass
        exit_code = int(process.returncode) if process and process.returncode is not None else -1

    elapsed_min = (time.time() - start_time) / 60.0
    finish = (f"[{datetime.now()}] Finished {fw.upper()} "
              f"(util={target_util}, run={run}) in {elapsed_min:.2f} min.\n")
    print(finish)
    log_file.write("-" * 60 + "\n" + finish)
    log_file.flush()

    parsed, scheme_rows = _parse_final_report(output_lines)

    # Compute per-epoch PHV for this run
    # Parliament: 5 scheme points/epoch, NSGA2: N front members/epoch, others: 1 point/epoch
    avg_epoch_phv = _parse_epoch_phv(output_lines, fw)
    if avg_epoch_phv:
        print(f"  [PHV] {fw} avg per-epoch PHV = {avg_epoch_phv}")

    base_row = {
        "framework": fw, "num_dcs": num_dcs,
        "target_util": f"{target_util:.2f}", "run": run,
        "start_time": start_dt.isoformat(timespec="seconds"),
        "end_time": datetime.now().isoformat(timespec="seconds"),
        "elapsed_min": f"{elapsed_min:.2f}", "exit_code": str(exit_code),
        "command": " ".join(cmd),
    }

    if fw == "parliament" and scheme_rows:
        for sr in scheme_rows:
            row = {**base_row, "scheme": sr["scheme"],
                   "epochs": sr["epochs"], "avg_ttft_s": sr["avg_ttft_s"],
                   "total_carbon_kg": sr["total_carbon_kg"],
                   "total_water_l": sr["total_water_l"],
                   "total_energy_usd": sr["total_energy_usd"],
                   "total_energy_kwh": sr["total_energy_kwh"],
                   "avg_epoch_phv": avg_epoch_phv}
            writer.writerow(row)
    else:
        row = {**base_row, "scheme": "",
               "epochs": parsed["epochs"], "avg_ttft_s": parsed["avg_ttft_s"],
               "total_carbon_kg": parsed["total_carbon_kg"],
               "total_water_l": parsed["total_water_l"],
               "total_energy_usd": parsed["total_energy_usd"],
               "total_energy_kwh": parsed["total_energy_kwh"],
               "avg_epoch_phv": avg_epoch_phv}
        writer.writerow(row)

    summary_file.flush()
    return base_row


# ── Pareto Hypervolume ────────────────────────────────────────────────────────

def _dominated(a, b):
    """True if point a dominates point b (all <= and at least one <)."""
    return all(ai <= bi for ai, bi in zip(a, b)) and any(ai < bi for ai, bi in zip(a, b))


def _pareto_front(points):
    """Return indices of non-dominated points."""
    n = len(points)
    is_dominated = [False] * n
    for i in range(n):
        if is_dominated[i]:
            continue
        for j in range(n):
            if i == j or is_dominated[j]:
                continue
            if _dominated(points[j], points[i]):
                is_dominated[i] = True
                break
    return [i for i in range(n) if not is_dominated[i]]


def _hypervolume_2d(points, ref):
    """Exact 2D hypervolume computation."""
    pts = sorted(points, key=lambda p: p[0])
    hv = 0.0
    prev_y = ref[1]
    for p in pts:
        if p[0] < ref[0] and p[1] < ref[1]:
            hv += (ref[0] - p[0]) * (prev_y - p[1])
            prev_y = min(prev_y, p[1])
    return hv


def _hypervolume_mc(points, ref, n_samples=100_000):
    """Monte Carlo hypervolume approximation for arbitrary dimensions."""
    pts = np.array(points)
    ref = np.array(ref)

    # Filter to points that are within the reference
    valid = np.all(pts < ref, axis=1)
    pts = pts[valid]
    if len(pts) == 0:
        return 0.0

    # Find bounding box: [min per dim, ref per dim]
    lo = pts.min(axis=0)
    box_vol = np.prod(ref - lo)
    if box_vol <= 0:
        return 0.0

    # Sample random points in the bounding box
    rng = np.random.default_rng(42)
    samples = rng.uniform(lo, ref, size=(n_samples, len(ref)))

    # Count how many samples are dominated by at least one point
    dominated_count = 0
    for s in samples:
        for p in pts:
            if np.all(p <= s):
                dominated_count += 1
                break

    return box_vol * dominated_count / n_samples


def compute_hypervolume(points, ref):
    """Compute hypervolume of a set of points w.r.t. a reference point.
    All objectives are minimization."""
    if len(points) == 0:
        return 0.0
    dim = len(points[0])
    # Filter to Pareto front
    front_idx = _pareto_front(points)
    front = [points[i] for i in front_idx]
    if dim == 2:
        return _hypervolume_2d(front, ref)
    else:
        return _hypervolume_mc(front, ref, n_samples=200_000)


# ── Main ──────────────────────────────────────────────────────────────────────

def run_batch_experiments():
    experiments = [
        (fw, FIXED_NUM_DCS, util, "")
        for util, fw in product(UTIL_SWEEP_VALUES, FRAMEWORKS)
    ]

    total_configs = len(experiments)
    total_runs = total_configs * NUM_RUNS

    script_dir = os.path.dirname(os.path.abspath(__file__))
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = os.path.join(script_dir, "experiment_results")
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, f"util_sweep_{ts}.log")
    summary_path = os.path.join(out_dir, f"util_sweep_{ts}.csv")

    print(f"Starting: {total_configs} configs × {NUM_RUNS} runs = {total_runs} total")
    print(f"  Utilizations: {UTIL_SWEEP_VALUES}")
    print(f"  Frameworks: {len(FRAMEWORKS)}")
    print(f"  DCs: {FIXED_NUM_DCS}")
    print(f"Log: {log_path}\nCSV: {summary_path}\n")

    with open(log_path, "w") as log_file, \
            open(summary_path, "w", newline="") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=FIELDNAMES)
        writer.writeheader()

        log_file.write(f"=== STARTED: {datetime.now()} ===\n")
        log_file.write(f"Configs={total_configs} Runs/config={NUM_RUNS} "
                       f"Total={total_runs}\n\n")

        run_counter = 0
        for fw, num_dcs, target_util, _ in experiments:
            label = f"{fw.upper()} @ {num_dcs} DCs, Util={target_util * 100:.0f}%"
            for run_num in range(1, NUM_RUNS + 1):
                run_counter += 1
                print(f"\n{'=' * 60}")
                print(f"Run {run_counter}/{total_runs}  "
                      f"[{label}  run {run_num}/{NUM_RUNS}]")
                print(f"{'=' * 60}")
                log_file.write(f"\n{'=' * 60}\n")
                log_file.write(f"Run {run_counter}/{total_runs} [{label} "
                               f"run {run_num}/{NUM_RUNS}]\n")
                try:
                    _run_one(fw, target_util, num_dcs, script_dir, log_file,
                             writer, summary_file, run_num)
                except Exception as exc:
                    err = (f"\n[{datetime.now()}] FATAL: {fw.upper()} "
                           f"run {run_num} crashed: {exc}\n")
                    print(err)
                    log_file.write(err)
                    log_file.flush()

        log_file.write(f"\n=== COMPLETED: {datetime.now()} ===\n")

    # ══════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETED")
    print("=" * 80)
    print(f"Log: {log_path}")
    print(f"CSV: {summary_path}")

    try:
        import pandas as pd
        from scipy import stats as scipy_stats

        df = pd.read_csv(summary_path)
        for col in ["avg_ttft_s", "total_carbon_kg", "total_water_l",
                    "total_energy_usd", "total_energy_kwh"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["target_util"] = pd.to_numeric(df["target_util"], errors="coerce")

        METRICS = ["avg_ttft_s", "total_carbon_kg", "total_water_l", "total_energy_usd"]
        LABELS = ["TTFT(s)", "Carbon(kg)", "Water(L)", "Cost($)"]

        def _stats(vals):
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
            return (f"{s['avg']:.3f}±{s['ci95']:.3f} "
                    f"[{s['min']:.3f}, {s['max']:.3f}]")

        # Build row labels
        row_labels = []
        for fw in FRAMEWORKS:
            if fw == "parliament":
                for scheme in PARLIAMENT_SCHEMES:
                    row_labels.append((fw, scheme, f"parliament/{scheme}"))
            else:
                row_labels.append((fw, "", fw))

        # ── Utilization Sweep Table ──────────────────────────────────────
        print(f"\n{'═' * 100}")
        print(f"  UTILIZATION SWEEP — {FIXED_NUM_DCS} DCs, {NUM_RUNS} runs per config")
        print(f"  Format: avg±95%CI [min, max]")
        print(f"{'═' * 100}")

        for metric, label in zip(METRICS, LABELS):
            print(f"\n  {label}:")
            print(f"  {'Framework':<22}", end="")
            for util in UTIL_SWEEP_VALUES:
                print(f" | {'Util=' + str(int(util * 100)) + '%':^36}", end="")
            print(f" |")
            print(f"  {'─' * 22}", end="")
            for _ in UTIL_SWEEP_VALUES:
                print(f"-+-{'─' * 36}", end="")
            print(f"-+")

            for fw, scheme, display_name in row_labels:
                print(f"  {display_name:<22}", end="")
                for util in UTIL_SWEEP_VALUES:
                    if scheme:
                        vals = df[(df["framework"] == fw) &
                                  (df["target_util"] == util) &
                                  (df["scheme"] == scheme)][metric]
                    else:
                        vals = df[(df["framework"] == fw) &
                                  (df["target_util"] == util)][metric]
                    s = _stats(vals)
                    print(f" | {_fmt(s):^36}", end="")
                print(f" |")
            print()

        # ── Best per metric per utilization ──────────────────────────────
        print(f"{'═' * 100}")
        print(f"  BEST FRAMEWORK PER METRIC (by mean of {NUM_RUNS} runs)")
        print(f"{'═' * 100}")
        for metric, label in zip(METRICS, LABELS):
            print(f"\n  {label}:")
            for util in UTIL_SWEEP_VALUES:
                best_avg, best_label = None, None
                for fw in FRAMEWORKS:
                    if fw == "parliament":
                        continue
                    vals = df[(df["framework"] == fw) &
                              (df["target_util"] == util)][metric].dropna()
                    if len(vals) > 0:
                        avg = vals.mean()
                        if best_avg is None or avg < best_avg:
                            best_avg, best_label = avg, fw
                for scheme in PARLIAMENT_SCHEMES:
                    vals = df[(df["framework"] == "parliament") &
                              (df["target_util"] == util) &
                              (df["scheme"] == scheme)][metric].dropna()
                    if len(vals) > 0:
                        avg = vals.mean()
                        if best_avg is None or avg < best_avg:
                            best_avg, best_label = avg, f"parliament/{scheme}"
                if best_label:
                    print(f"    Util={int(util * 100):>3}%: {best_label:<22}  avg={best_avg:.4f}")

        # ══════════════════════════════════════════════════════════════════
        # PER-EPOCH PARETO HYPERVOLUME (PHV)
        # ══════════════════════════════════════════════════════════════════
        # PHV computed per-epoch during each run from epoch-level objectives:
        #   Parliament: 5 scheme points per epoch
        #   NSGA2: N Pareto front members per epoch (re-executed on fresh sims)
        #   Others: 1 point per epoch
        # Self-referencing: each run's reference point = worst per metric + 10%
        # across that run's own epochs.

        df["avg_epoch_phv"] = pd.to_numeric(df["avg_epoch_phv"], errors="coerce")

        print(f"\n{'═' * 100}")
        print(f"  PER-EPOCH PARETO HYPERVOLUME (PHV)")
        print(f"  Parliament: 5 scheme points/epoch | NSGA2: Pareto front/epoch | Others: 1 point/epoch")
        print(f"  Mean ± 95%CI across {NUM_RUNS} runs (self-referenced per run)")
        print(f"{'═' * 100}")

        print(f"\n  {'Framework':<22}", end="")
        for util in UTIL_SWEEP_VALUES:
            print(f" {'Util=' + str(int(util * 100)) + '%':>24}", end="")
        print()
        print(f"  {'─' * 22}", end="")
        for _ in UTIL_SWEEP_VALUES:
            print(f" {'─' * 24}", end="")
        print()

        for fw in FRAMEWORKS:
            print(f"  {fw:<22}", end="")
            for util in UTIL_SWEEP_VALUES:
                # For parliament, PHV is the same across all scheme rows of a run
                # (computed once per run), so deduplicate by run
                util_fw = df[(df["framework"] == fw) &
                             (df["target_util"] == util)].copy()
                if fw == "parliament":
                    # Take one PHV value per run (all scheme rows have same value)
                    phv_vals = util_fw.drop_duplicates(subset=["run"])["avg_epoch_phv"].dropna()
                else:
                    phv_vals = util_fw["avg_epoch_phv"].dropna()

                s = _stats(phv_vals)
                if s["avg"] is not None:
                    print(f" {s['avg']:>13.1f}±{s['ci95']:<9.1f}", end="")
                else:
                    print(f" {'n/a':>24}", end="")
            print()

        print(f"\n{'═' * 100}")
        print(f"  Raw data: {summary_path}")
        print(f"{'═' * 100}")

    except Exception as e:
        print(f"\n(Summary error: {e})")
        import traceback;
        traceback.print_exc()


if __name__ == "__main__":
    run_batch_experiments()