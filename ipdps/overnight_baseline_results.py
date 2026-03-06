import csv
import os
import re
import subprocess
import sys
import time
from datetime import datetime


def _parse_final_report(output_lines):
    text = "".join(output_lines)
    metrics = {
        "epochs": "",
        "avg_ttft_s": "",
        "total_carbon_kg": "",
        "total_water_l": "",
        "total_energy_usd": "",
        "total_energy_kwh": "",
    }
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
    return metrics


def run_batch_experiments():
    # The exact order you requested
    frameworks = [
        "helix",
        "splitwise",
        "actorcritic",
        "nsga2",
        "ddqn",
        "qlearning",
        "perllm",
    ]

    target_util = 0.95
    autoscale_mode = "global_peak"
    max_drop = 0.02
    max_mult = 1_500_000
    max_rows = 300_000
    search_steps = 9

    script_dir = os.path.dirname(os.path.abspath(__file__))
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = os.path.join(script_dir, "experiment_results")
    os.makedirs(out_dir, exist_ok=True)
    log_filename = os.path.join(out_dir, f"overnight_results_{ts}.log")
    summary_filename = os.path.join(out_dir, f"overnight_summary_{ts}.csv")

    print(f"Starting overnight batch run for {len(frameworks)} frameworks.")
    print(f"Target Utilization (max target): {target_util * 100:.1f}%")
    print(f"Auto-Scale Mode: {autoscale_mode}")
    print(f"Live output + full logs: {log_filename}")
    print(f"Summary CSV: {summary_filename}\n")

    with open(log_filename, "w", encoding="utf-8") as log_file, \
            open(summary_filename, "w", newline="", encoding="utf-8") as summary_file:
        writer = csv.DictWriter(
            summary_file,
            fieldnames=[
                "framework",
                "start_time",
                "end_time",
                "elapsed_min",
                "exit_code",
                "epochs",
                "avg_ttft_s",
                "total_carbon_kg",
                "total_water_l",
                "total_energy_usd",
                "total_energy_kwh",
                "command",
            ],
        )
        writer.writeheader()

        log_file.write(f"=== OVERNIGHT RUN STARTED: {datetime.now()} ===\n")
        log_file.write(f"Target Utilization: {target_util}\n\n")
        log_file.write(f"Autoscale Mode: {autoscale_mode}\n")
        log_file.write(f"Max Drop: {max_drop}\n")
        log_file.write(f"Max Multiplier: {max_mult}\n")
        log_file.write(f"Max Rows: {max_rows}\n")
        log_file.write(f"Search Steps: {search_steps}\n\n")

        for fw in frameworks:
            start_time = time.time()
            start_dt = datetime.now()

            # Construct the command array
            cmd = [
                sys.executable, "-u", "simulator_LLM.py",
                "--framework", fw,
                "--target-util", str(target_util),
                "--autoscale-mode", autoscale_mode,
                "--autoscale-max-drop", str(max_drop),
                "--autoscale-max-mult", str(max_mult),
                "--autoscale-max-rows", str(max_rows),
                "--autoscale-search-steps", str(search_steps),
            ]

            msg = f"\n[{datetime.now()}] Starting framework: {fw.upper()}..."
            print(msg)
            log_file.write(msg + "\n")
            log_file.write(f"Command: {' '.join(cmd)}\n")
            log_file.write("-" * 50 + "\n")
            log_file.flush()  # Ensure it writes to disk immediately

            process = None
            output_lines = []
            exit_code = -1
            try:
                # Run the simulator and capture the output
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    cwd=script_dir,
                )

                # Stream output to both the console and the log file in real-time
                for line in process.stdout:
                    print(line, end="")
                    log_file.write(line)
                    output_lines.append(line)
                    log_file.flush()

                process.wait()
                exit_code = int(process.returncode)
                if exit_code != 0:
                    err_msg = f"\n[{datetime.now()}] WARNING: {fw.upper()} exited with code {exit_code}\n"
                    print(err_msg, end="")
                    log_file.write(err_msg)
                    log_file.flush()

            except Exception as e:
                error_msg = f"\nERROR running {fw}: {str(e)}\n"
                print(error_msg)
                log_file.write(error_msg)
                if process and process.poll() is None:
                    process.kill()
                    process.wait()
                exit_code = int(process.returncode) if process else -1

            end_dt = datetime.now()
            elapsed_time = (time.time() - start_time) / 60.0
            finish_msg = f"[{datetime.now()}] Finished {fw.upper()} in {elapsed_time:.2f} minutes.\n"
            print(finish_msg)
            log_file.write("-" * 50 + "\n")
            log_file.write(finish_msg)
            log_file.flush()

            parsed = _parse_final_report(output_lines)
            writer.writerow(
                {
                    "framework": fw,
                    "start_time": start_dt.isoformat(timespec="seconds"),
                    "end_time": end_dt.isoformat(timespec="seconds"),
                    "elapsed_min": f"{elapsed_time:.2f}",
                    "exit_code": str(exit_code),
                    "epochs": parsed["epochs"],
                    "avg_ttft_s": parsed["avg_ttft_s"],
                    "total_carbon_kg": parsed["total_carbon_kg"],
                    "total_water_l": parsed["total_water_l"],
                    "total_energy_usd": parsed["total_energy_usd"],
                    "total_energy_kwh": parsed["total_energy_kwh"],
                    "command": " ".join(cmd),
                }
            )
            summary_file.flush()

        log_file.write(f"\n=== OVERNIGHT RUN COMPLETED: {datetime.now()} ===\n")

    print("\nAll frameworks completed.")
    print(f"Full log: {log_filename}")
    print(f"Summary CSV: {summary_filename}")


if __name__ == "__main__":
    run_batch_experiments()
