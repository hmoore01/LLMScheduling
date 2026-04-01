#!/usr/bin/env python3
"""Run the 3 new ablation studies: no-phase2, no-sgd, no-exploration.
Each runs parliament @ 8 DCs × 5 runs = 15 runs total."""

import subprocess
import sys
import os
import time
import csv
import re
from datetime import datetime

VARIANTS = ["no-phase2", "no-sgd", "no-exploration"]
NUM_RUNS = 5
NUM_DCS = 8
TARGET_UTIL = 0.95

# Autoscale settings (match overnight_baseline_results.py)
AUTOSCALE_MODE = "global_peak"
MAX_DROP = 0.02
MAX_MULT = 1_500_000
MAX_ROWS = 300_000
SEARCH_STEPS = 9

FIELDNAMES = [
    "framework", "scheme", "ablation", "num_dcs", "target_util", "run",
    "start_time", "end_time", "elapsed_min", "exit_code", "epochs",
    "avg_ttft_s", "total_carbon_kg", "total_water_l",
    "total_energy_usd", "total_energy_kwh",
]


def _parse_scheme_rows(output_text):
    """Parse per-scheme rows from RUN SUMMARY."""
    rows = []
    summary_start = output_text.rfind("RUN SUMMARY")
    if summary_start < 0:
        return rows
    summary_text = output_text[summary_start:]
    pat = re.compile(
        r"^\s+(MinCost|Balanced|MinLatency|MinCarbon|MinWater)\s+"
        r"([0-9eE+\-\.★]+)\s+"
        r"([0-9eE+\-\.★]+)\s+"
        r"([0-9eE+\-\.★]+)\s+"
        r"([0-9eE+\-\.★]+)\s+"
        r"([0-9]+)\s+"
        r"([0-9]+)",
        re.MULTILINE
    )
    for m in pat.finditer(summary_text):
        rows.append({
            "scheme": m.group(1),
            "epochs": m.group(7),
            "avg_ttft_s": m.group(2).replace("★", ""),
            "total_carbon_kg": m.group(3).replace("★", ""),
            "total_water_l": m.group(4).replace("★", ""),
            "total_energy_usd": m.group(5).replace("★", ""),
            "total_energy_kwh": "",
        })
    return rows


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "experiment_results")
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    csv_path = os.path.join(results_dir, f"new_ablation_{timestamp}.csv")
    log_path = os.path.join(results_dir, f"new_ablation_{timestamp}.log")

    total = len(VARIANTS) * NUM_RUNS
    print(f"Running {len(VARIANTS)} new ablation variants × {NUM_RUNS} runs = {total} total")
    print(f"CSV: {csv_path}")
    print(f"Log: {log_path}")

    with open(csv_path, "w", newline="") as csv_file, \
         open(log_path, "w") as log_file:

        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        writer.writeheader()

        run_counter = 0
        for variant in VARIANTS:
            for run_num in range(1, NUM_RUNS + 1):
                run_counter += 1
                label = f"parliament [{variant}] run {run_num}/{NUM_RUNS}"
                print(f"\n{'='*60}")
                print(f"Run {run_counter}/{total}  [{label}]")
                print(f"{'='*60}")
                log_file.write(f"\n{'='*60}\n{label}\n")

                cmd = [
                    sys.executable, "-u", "simulator_LLM.py",
                    "--framework", "parliament",
                    "--num-dcs", str(NUM_DCS),
                    "--target-util", str(TARGET_UTIL),
                    "--autoscale-mode", AUTOSCALE_MODE,
                    "--autoscale-max-drop", str(MAX_DROP),
                    "--autoscale-max-mult", str(MAX_MULT),
                    "--autoscale-max-rows", str(MAX_ROWS),
                    "--autoscale-search-steps", str(SEARCH_STEPS),
                    "--ablation", variant,
                ]

                start_dt = datetime.now()
                start_time = time.time()
                output_lines = []
                exit_code = -1

                try:
                    process = subprocess.Popen(
                        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, bufsize=1, cwd=script_dir)
                    try:
                        for line in process.stdout:
                            print(line, end="")
                            log_file.write(line)
                            output_lines.append(line)
                    except (BrokenPipeError, IOError, ValueError):
                        pass
                    process.wait(timeout=30)
                    exit_code = int(process.returncode)
                    if exit_code != 0:
                        print(f"  WARNING: exited code {exit_code}")
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()
                    print(f"  TIMEOUT")
                except Exception as exc:
                    print(f"  ERROR: {exc}")
                    if process and process.poll() is None:
                        try: process.kill(); process.wait(timeout=10)
                        except: pass
                    exit_code = process.returncode if process and process.returncode else -1

                elapsed = (time.time() - start_time) / 60.0
                print(f"  Finished in {elapsed:.1f} min (exit={exit_code})")
                log_file.write(f"  Finished in {elapsed:.1f} min (exit={exit_code})\n")
                log_file.flush()

                # Parse scheme rows
                text = "".join(output_lines)
                scheme_rows = _parse_scheme_rows(text)

                base = {
                    "framework": "parliament", "ablation": variant,
                    "num_dcs": NUM_DCS, "target_util": f"{TARGET_UTIL:.2f}",
                    "run": run_num,
                    "start_time": start_dt.isoformat(timespec="seconds"),
                    "end_time": datetime.now().isoformat(timespec="seconds"),
                    "elapsed_min": f"{elapsed:.2f}", "exit_code": str(exit_code),
                }

                if scheme_rows:
                    for sr in scheme_rows:
                        row = {**base, **sr}
                        writer.writerow(row)
                else:
                    # Fallback: parse aggregate
                    epochs_m = re.search(r"Epochs:\s*([0-9]+)", text)
                    row = {**base, "scheme": "",
                           "epochs": epochs_m.group(1) if epochs_m else "",
                           "avg_ttft_s": "", "total_carbon_kg": "",
                           "total_water_l": "", "total_energy_usd": "",
                           "total_energy_kwh": ""}
                    writer.writerow(row)
                csv_file.flush()

    print(f"\n{'='*60}")
    print(f"ALL DONE — {total} runs")
    print(f"CSV: {csv_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()