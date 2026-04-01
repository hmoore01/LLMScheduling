import os
import glob
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------
# Configuration & Paths
# ---------------------------------------------------
# Point this to your actual sim_results folder
BASE_RESULTS_DIR = "/mnt/c/Users/hmoor/Downloads/GreenEdgeCloudSim/GreenEdgeCloudSim/sim_results"

FRAMEWORKS_TO_TEST = ["CLOUD_ONLY", "EDGE_ONLY", "HYBRID", "DDQN"]

# --- Log Parsing Column Indices (Zero-Indexed based on ';') ---
LOC_COL = 8  # Execution Location ("1"=Cloud, "2"=Edge, "0"=Mobile)
NET_DELAY_COL = 9  # Network Delay
COMP_DELAY_COL = 10  # Compute Delay

# ---------------- IEEE Access Plot Styling ----------------
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "lines.linewidth": 2.0,
    "figure.autolayout": True
})


# ---------------------------------------------------
# Advanced Log Parser
# ---------------------------------------------------
def parse_framework_logs(framework):
    framework_dir = os.path.join(BASE_RESULTS_DIR, framework)
    print(f"📂 Parsing logs from: {framework_dir}")

    stats = {
        "assignments": {"Cloud": 0, "Edge": 0, "Mobile": 0},
        "delays": {"total_network": 0.0, "total_compute": 0.0, "task_count": 0},
        "sustainability": {"total_energy": 0.0, "total_carbon": 0.0},
        "failures": 0
    }

    if not os.path.exists(framework_dir):
        print(f"   ⚠️ Directory not found for {framework}. Returning zeros.")
        return stats

    # 1. Parse Success Logs (Assignments & Latency)
    success_files = glob.glob(os.path.join(framework_dir, "**", "*_SUCCESS.log"), recursive=True)
    for log_file in success_files:
        with open(log_file, "r") as f:
            for line in f:
                if line.startswith("#") or not line.strip(): continue
                parts = line.split(";")
                if len(parts) > max(LOC_COL, NET_DELAY_COL, COMP_DELAY_COL):
                    loc = parts[LOC_COL].strip()
                    if loc == "1":
                        stats["assignments"]["Cloud"] += 1
                    elif loc == "2":
                        stats["assignments"]["Edge"] += 1
                    elif loc == "0":
                        stats["assignments"]["Mobile"] += 1

                    try:
                        stats["delays"]["total_network"] += float(parts[NET_DELAY_COL].strip())
                        stats["delays"]["total_compute"] += float(parts[COMP_DELAY_COL].strip())
                        stats["delays"]["task_count"] += 1
                    except ValueError:
                        pass

                        # 2. Parse Fail Logs (Failed Tasks)
    fail_files = glob.glob(os.path.join(framework_dir, "**", "*_FAIL.log"), recursive=True)
    for log_file in fail_files:
        with open(log_file, "r") as f:
            for line in f:
                if not line.startswith("#") and line.strip():
                    stats["failures"] += 1

    # 3. Parse Power Logs (Carbon & Energy)
    power_files = glob.glob(os.path.join(framework_dir, "**", "*_POWER_METRICS.log"), recursive=True)
    for log_file in power_files:
        with open(log_file, "r") as f:
            lines = [line.strip() for line in f if not line.startswith("#") and line.strip()]
            if lines:
                last_line = lines[-1]
                parts = last_line.split(";")
                if len(parts) >= 10:
                    try:
                        stats["sustainability"]["total_energy"] += float(parts[4]) + float(parts[5]) + float(parts[6])
                        stats["sustainability"]["total_carbon"] += float(parts[7]) + float(parts[8]) + float(parts[9])
                    except ValueError:
                        pass

    if stats["delays"]["task_count"] == 0 and stats["failures"] == 0:
        print(f"   ⚠️ WARNING: Zero tasks found for {framework}. Did the simulation run?")

    return stats


# ---------------------------------------------------
# Automated Plotting
# ---------------------------------------------------
def generate_plots(data_dict):
    print("\n📈 Generating plots...")
    frameworks = list(data_dict.keys())

    # Extract Data Arrays
    carbon_emissions = [data_dict[f]["sustainability"]["total_carbon"] for f in frameworks]
    energy_consumption = [data_dict[f]["sustainability"]["total_energy"] for f in frameworks]

    avg_latencies = []
    for f in frameworks:
        tc = data_dict[f]["delays"]["task_count"]
        if tc > 0:
            avg_latencies.append(
                (data_dict[f]["delays"]["total_network"] + data_dict[f]["delays"]["total_compute"]) / tc)
        else:
            avg_latencies.append(0)

    cloud_tasks = [data_dict[f]["assignments"]["Cloud"] for f in frameworks]
    edge_tasks = [data_dict[f]["assignments"]["Edge"] for f in frameworks]
    mobile_tasks = [data_dict[f]["assignments"]["Mobile"] for f in frameworks]

    failed_tasks = [data_dict[f]["failures"] for f in frameworks]

    # --- Plot 1: Carbon Emissions ---
    plt.figure(figsize=(7, 5))
    plt.bar(frameworks, carbon_emissions, color=['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728'], edgecolor='black')
    plt.ylabel("Total Carbon Emission (kg CO2e)")
    plt.title("Carbon Footprint Comparison")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig("plot_1_carbon_emissions.pdf", format='pdf')
    plt.close()

    # --- Plot 2: Energy Consumption ---
    plt.figure(figsize=(7, 5))
    plt.bar(frameworks, energy_consumption, color=['#9467bd', '#8c564b', '#e377c2', '#7f7f7f'], edgecolor='black')
    plt.ylabel("Total Energy Consumption (kWh)")
    plt.title("Energy Consumption Comparison")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig("plot_2_energy_consumption.pdf", format='pdf')
    plt.close()

    # --- Plot 3: Average Latency ---
    plt.figure(figsize=(7, 5))
    plt.bar(frameworks, avg_latencies, color=['#7f7f7f', '#bcbd22', '#17becf', '#1f77b4'], edgecolor='black')
    plt.ylabel("Average Total Delay (Seconds)")
    plt.title("Task Latency Comparison")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig("plot_3_avg_latency.pdf", format='pdf')
    plt.close()

    # --- Plot 4: Task Assignments (Stacked Bar) ---
    plt.figure(figsize=(8, 5))
    bar_width = 0.5
    p1 = plt.bar(frameworks, cloud_tasks, width=bar_width, label='Cloud', color='#1f77b4', edgecolor='black')
    p2 = plt.bar(frameworks, edge_tasks, width=bar_width, bottom=cloud_tasks, label='Edge', color='#2ca02c',
                 edgecolor='black')
    bottom_mobile = np.add(cloud_tasks, edge_tasks).tolist()
    p3 = plt.bar(frameworks, mobile_tasks, width=bar_width, bottom=bottom_mobile, label='Mobile', color='#ff7f0e',
                 edgecolor='black')

    plt.ylabel("Number of Tasks Executed")
    plt.title("Task Placement by Framework")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig("plot_4_task_assignments.pdf", format='pdf')
    plt.close()

    # --- Plot 5: Failed Tasks ---
    plt.figure(figsize=(7, 5))
    plt.bar(frameworks, failed_tasks, color='#d62728', edgecolor='black')
    plt.ylabel("Number of Failed Tasks")
    plt.title("Task Failure Comparison")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig("plot_5_task_failures.pdf", format='pdf')
    plt.close()

    print("✅ Successfully generated 5 plots as PDF files in the current directory!")


# ---------------------------------------------------
# Main Execution
# ---------------------------------------------------
if __name__ == "__main__":
    print(f"🔍 Searching for data in: {BASE_RESULTS_DIR}\n")

    experiment_data = {}
    for framework in FRAMEWORKS_TO_TEST:
        experiment_data[framework] = parse_framework_logs(framework)

    generate_plots(experiment_data)