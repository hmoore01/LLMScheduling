import os
import subprocess
import shutil
import re
import pandas as pd
import time

# --- Experiment Configuration ---
FRAMEWORKS = ["parliament", "helix", "hybrid", "ddqn"]
OUTPUT_DIR = "experiment_results"
MASTER_CSV = os.path.join(OUTPUT_DIR, "master_experiment_results.csv")

# Define the scenarios to avoid running duplicate combinations
scenarios = []

# 1. DC Scaling at 95% Utilization (Updated to 4, 6, 8, 12)
for dc in [4, 6, 8, 12]:
    scenarios.append({"dc": dc, "util": 0.95, "exp_name": "dc_scaling"})

# 2. Utilization Scaling at 12 DCs (Skipping 0.95 since it's covered above)
for u in [0.75, 0.85, 1.05]:
    scenarios.append({"dc": 12, "util": u, "exp_name": "util_scaling"})


def parse_metrics(stdout: str) -> dict:
    """Extracts final report metrics using regex."""
    ttft = re.search(r"Average TTFT \(s\):\s+([\d.]+)", stdout)
    carbon = re.search(r"Total Carbon \(kg\):\s+([\d.]+)", stdout)
    water = re.search(r"Total Water \(m.\):\s+([\d.]+)", stdout)
    cost = re.search(r"Total Energy \(\$\):\s+([\d.]+)", stdout)
    energy_kwh = re.search(r"Total Energy \(kWh\):\s+([\d.]+)", stdout)

    return {
        "TTFT_s": float(ttft.group(1)) if ttft else None,
        "Carbon_kg": float(carbon.group(1)) if carbon else None,
        "Water_m3": float(water.group(1)) if water else None,
        "Cost_USD": float(cost.group(1)) if cost else None,
        "Energy_kWh": float(energy_kwh.group(1)) if energy_kwh else None
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_results = []

    total_runs = len(FRAMEWORKS) * len(scenarios)
    current_run = 1

    print(f"=== Starting MARLIN Experimental Matrix ({total_runs} total runs) ===")
    start_time = time.time()

    for scenario in scenarios:
        dc = scenario["dc"]
        util = scenario["util"]
        exp_name = scenario["exp_name"]

        for fw in FRAMEWORKS:
            print(f"[{current_run}/{total_runs}] Running: FW={fw.upper():<10} | DCs={dc:<2} | Util={util:<4} ...",
                  end="", flush=True)

            # Construct the base command (Added --load-model for inference!)
            cmd = [
                "python", "simulator_LLM.py",
                "--framework", fw,
                "--num-dcs", str(dc),
                "--target-util", str(util)
            ]

            # Dynamically point Parliament to the correct trained_models subfolder
            if fw.lower() == "parliament":
                cmd.extend(["--model-dir", f"trained_models/dc_{dc}"])

            # Execute the simulator and capture the output
            run_start = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True)
            run_time = time.time() - run_start

            # Enhanced Error Catching
            if result.returncode != 0:
                print(f" ERROR in {run_time:.1f}s (Exit Code: {result.returncode})")
                if result.stderr:
                    print(f"Tail of error:\n{result.stderr[-500:]}")
                else:
                    print("No standard error output. (If Exit Code is 137, your machine ran out of RAM!)")
                current_run += 1
                continue

            # Parse the metrics
            metrics = parse_metrics(result.stdout)

            # Save the specific utility trace file
            trace_source = "marlin_utility_trace.csv"
            trace_dest = os.path.join(OUTPUT_DIR, f"trace_{fw}_dc{dc}_util{int(util * 100)}.csv")
            if os.path.exists(trace_source):
                shutil.move(trace_source, trace_dest)

            # Record the data
            run_data = {
                "Experiment": exp_name,
                "Framework": fw,
                "Num_DCs": dc,
                "Target_Util": util,
                **metrics,
                "Runtime_sec": round(run_time, 1),
                "Trace_File": trace_dest
            }
            all_results.append(run_data)

            # Print quick summary to console
            print(
                f" Done in {run_time:.1f}s -> Cost: ${metrics.get('Cost_USD', 0):.2f} | TTFT: {metrics.get('TTFT_s', 0):.3f}s")

            # Continuously save to CSV in case the script is interrupted
            df = pd.DataFrame(all_results)
            df.to_csv(MASTER_CSV, index=False)

            current_run += 1

    total_time = (time.time() - start_time) / 60
    print(f"\n=== All Experiments Completed in {total_time:.1f} minutes ===")
    print(f"Results compiled in: {MASTER_CSV}")
    print(f"Time-series traces saved in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()