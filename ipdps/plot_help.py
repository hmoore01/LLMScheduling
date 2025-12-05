import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Patch

# Use DejaVu Serif font
plt.rcParams["font.family"] = "DejaVu Serif"

# Workload mapping from uploaded CSVs (assumes they are in the working dir)
workload_files = {
    "bwaves_r": "Temperature_Sweep_bwaves_r.csv",
    "deepsjeng_r": "Temperature_Sweep_deepsjeng_r.csv",
    "exchange2_r": "Temperature_Sweep_exchange2_r.csv",
    "fotonik3d_r": "Temperature_Sweep_exchange2_r.csv",
    "gcc_r": "Temperature_Sweep_gcc_r.csv",
    "mcf_r": "Temperature_Sweep_mcf_r.csv",
    "namd_r": "Temperature_Sweep_namd_r.csv",
    "parest_r": "Temperature_Sweep_parest_r.csv",
    "perlbench_r": "Temperature_Sweep_perlbench_r.csv",
    "povray_r": "Temperature_Sweep_povray_r.csv",
}

TEMP_COL = "Temp_C"
IT_COL = "Total_IT_Energy_kWh"
COOL_COL = "Total_Cooling_Energy_kWh"
TOTAL_COL = "Total_Energy_kWh"

workloads = list(workload_files.keys())

all_temps = None
data_it = {}
data_cool = {}
data_total = {}
data_other = {}

# Load data
for w, fname in workload_files.items():
    df = pd.read_csv(Path(fname)).sort_values(TEMP_COL)
    temps = df[TEMP_COL].values
    if all_temps is None:
        all_temps = temps
    else:
        if not np.array_equal(all_temps, temps):
            raise ValueError(f"Temperature mismatch in {fname}")
    data_it[w] = df[IT_COL].values
    data_cool[w] = df[COOL_COL].values
    data_total[w] = df[TOTAL_COL].values

temps = all_temps
n_temps = len(temps)
n_workloads = len(workloads)

# Average total energy per temperature
avg_per_temp = np.zeros(n_temps)
for i in range(n_temps):
    avg_per_temp[i] = np.mean([data_total[w][i] for w in workloads])
baseline_avg = avg_per_temp[0]

# Create figure
fig, ax = plt.subplots(figsize=(12, 5), dpi=350)

group_width = 0.80
bar_width = group_width / n_workloads
indices = np.arange(n_temps)

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# Plot stacked bars (IT solid, Cooling hatched gray)
for wi, w in enumerate(workloads):
    x = indices - group_width / 2 + (wi + 0.5) * bar_width

    it_vals = data_it[w]
    cool_vals = data_cool[w]
    total_vals = data_total[w]

    other_vals = total_vals - it_vals - cool_vals

    data_other[w] = other_vals

    # IT energy (solid color)
    ax.bar(
        x, it_vals, width=bar_width, color=colors[wi % len(colors)],
        edgecolor="black", linewidth=0.4
    )

    # Cooling energy (hatched gray)
    ax.bar(
        x, cool_vals, width=bar_width, bottom=it_vals,
        color="lightgrey", edgecolor="black", linewidth=0.4, hatch="//"
    )

    ax.bar(
        x, other_vals, width=bar_width, bottom=cool_vals + it_vals,
        color="lightblue", edgecolor="black", linewidth=0.4, hatch="xx"
    )
    # Label (total energy) above each bar
    for xi, tv, itv, othv, cv in zip(x, total_vals, it_vals, cool_vals, other_vals):
        ax.text(
            xi, itv + cv + + othv + baseline_avg * 0.004,
            f"{tv:.0f}", ha="center", va="bottom",
            fontsize=10, rotation=90
        )

# --- Average segments: one dashed horizontal segment per temperature group, on top of bars ---
for i, avg in enumerate(avg_per_temp):
    x_center = indices[i]
    x_left = x_center - group_width / 2.0 - 0.1
    x_right = x_center + group_width / 2.0 + 0.1

    # Segment at the average value
    y_seg = avg

    ax.hlines(
        y=y_seg - 50,
        xmin=x_left,
        xmax=x_right,
        colors="red",
        linestyles="--",
        linewidth=2.0,
    )

# Baseline label on the bar region, centered at the baseline segment
baseline_x_center = indices[0]
offset = (max(avg_per_temp) * 0.03)
baseline_y = baseline_avg - offset
ax.text(
    baseline_x_center,
    baseline_y - 50,
    f"{baseline_avg:.0f} (baseline)",
    ha="center",
    va="center",
    fontsize=10,
    bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.85),
    color="white",
)

SMALL_CHANGE_THRESH = 150

# --- Vertical arrows and percent change labels between segments ---
for i in range(1, n_temps):
    x_prev = indices[i - 1]
    x_curr = indices[i]
    x_mid = (x_prev + x_curr) / 2.0

    prev_avg = avg_per_temp[i - 1]
    curr_avg = avg_per_temp[i]

    diff = curr_avg - prev_avg
    abs_diff = abs(diff)

    y0 = prev_avg - 50
    y1 = curr_avg - 50

    if abs_diff >= SMALL_CHANGE_THRESH:
        # Draw the arrow exactly as before
        y_bottom = min(y0, y1)
        y_top = max(y0, y1)

        ax.annotate(
            "",
            xy=(x_mid, y_bottom),
            xytext=(x_mid, y_top),
            arrowprops=dict(arrowstyle="<->", color="red", lw=1.4),
        )

    # Label centered on the arrow
    x_center = indices[i]
    pct_drop = (baseline_avg - avg_per_temp[i]) / baseline_avg * 100.0
    y_text = avg_per_temp[i] - 125
    ax.text(
        x_center,
        y_text,
        f"{avg_per_temp[i]:.0f} ({pct_drop:.0f}%)",
        ha="center",
        va="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.85),
        color="white",
    )

# Main title (unchanged)
ax.set_xlabel("Temperature (°C)", fontsize=12)
ax.set_ylabel("Total Energy (kWh)", fontsize=12)

ax.set_xticks(indices)
ax.set_xticklabels([f"{t:.0f}" for t in temps], fontsize=10)
ax.tick_params(axis="y", labelsize=10)

# Legends: workloads and stack components, placed in upper-right inside axes
workload_handles = [
    Patch(facecolor=colors[i % len(colors)], edgecolor="black", label=workloads[i])
    for i in range(n_workloads)
]

type_handles = [
    Patch(facecolor="black", edgecolor="black", alpha=0.6, label="Compute Energy"),
    Patch(facecolor="lightgrey", edgecolor="black", hatch="//", label="Cooling Energy"),
    Patch(facecolor="lightblue", edgecolor="black", hatch="xx", label="Support Energy"),
]

# First legend: workloads
leg1 = ax.legend(
    handles=workload_handles,
    loc="upper right",
    bbox_to_anchor=(0.850, 1.1),
    fontsize=8,
    title="Workloads",
    framealpha=0.9,
    ncol=2,
)
ax.add_artist(leg1)

# Second legend: stack components, below workloads legend
leg2 = ax.legend(
    handles=type_handles,
    loc="upper right",
    bbox_to_anchor=(0.995, 1.1),
    fontsize=9,
    title="Components",
    framealpha=0.9,
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()

# Save
out_path = Path("Total_Energy_vs_Temp_10SPEC_stacked_avgsegments_overlap.png")
fig.savefig(out_path, dpi=350, bbox_inches="tight")
plt.close(fig)

print(f"Saved to {out_path.resolve()}")

