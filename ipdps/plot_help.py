import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Patch

# Use DejaVu Serif font (common for academic papers)
plt.rcParams["font.family"] = "DejaVu Serif"

# Workload mapping from uploaded CSVs (assumes they are in the working dir)
workload_files = {
    "bwaves_r": "Temperature_Sweep_bwaves_r.csv",
    "deepsjeng_r": "Temperature_Sweep_deepsjeng_r.csv",
    "mcf_r": "Temperature_Sweep_mcf_r.csv",
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

# Target temperatures to plot
target_temps = [20, 30, 40]

# Load data
for w, fname in workload_files.items():
    df = pd.read_csv(Path(fname)).sort_values(TEMP_COL)

    # Filter the dataframe to only include target temperatures
    df = df[df[TEMP_COL].isin(target_temps)]

    temps = df[TEMP_COL].values
    if all_temps is None:
        all_temps = temps
    else:
        if not np.array_equal(all_temps, temps):
            raise ValueError(f"Temperature mismatch in {fname}. Ensure {target_temps} exist in all files.")
    data_it[w] = df[IT_COL].values
    data_cool[w] = df[COOL_COL].values
    data_total[w] = df[TOTAL_COL].values

temps = all_temps
n_temps = len(temps)
n_workloads = len(workloads)

# Average total energy per temperature (Still calculated for baseline label offset, but lines not drawn)
avg_per_temp = np.zeros(n_temps)
for i in range(n_temps):
    avg_per_temp[i] = np.mean([data_total[w][i] for w in workloads])
baseline_avg = avg_per_temp[0]

# Create figure sized for a single IEEE column (~3.5 inches wide)
fig, ax = plt.subplots(figsize=(3.5, 2.8), dpi=350)

group_width = 0.80
bar_width = group_width / n_workloads
indices = np.arange(n_temps)

# Use a different color palette (Set2) to visually separate from the old plot
cmap = plt.get_cmap("Set2")
colors = cmap.colors

# Plot stacked bars
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

    # --- NEW: Cooling energy (pattern changed to "oo" small circles) ---
    ax.bar(
        x, cool_vals, width=bar_width, bottom=it_vals,
        color="gainsboro", edgecolor="black", linewidth=0.4, hatch="////"
    )

    # --- NEW: Support energy (pattern changed to "++" crosses) ---
    ax.bar(
        x, other_vals, width=bar_width, bottom=cool_vals + it_vals,
        color="aliceblue", edgecolor="black", linewidth=0.4, hatch="...."
    )

    # Label (total energy) above each bar - smaller font to fit
    for xi, tv, itv, othv, cv in zip(x, total_vals, it_vals, cool_vals, other_vals):
        ax.text(
            xi, itv + cv + othv + baseline_avg * 0.015,
            f"{tv:.0f}", ha="center", va="bottom",
            fontsize=6, rotation=90
        )

# Main title / axis labels scaled down
ax.set_xlabel("Temperature (°C)", fontsize=8)
ax.set_ylabel("Total Energy (kWh)", fontsize=8)

ax.set_xticks(indices)
ax.set_xticklabels([f"{t:.0f}" for t in temps], fontsize=7)
ax.tick_params(axis="y", labelsize=7)

# --- SMALL TOP-RIGHT LEGENDS ---
short_workloads = [w.replace('_r', '') for w in workloads]

workload_handles = [
    Patch(facecolor=colors[i % len(colors)], edgecolor="black", label=short_workloads[i], linewidth=0.5)
    for i in range(n_workloads)
]

# --- NEW: Update legend handles to match new patterns ---
type_handles = [
    Patch(facecolor="black", edgecolor="black", alpha=0.6, label="Compute", linewidth=0.5),
    Patch(facecolor="gainsboro", edgecolor="black", hatch="////", label="Cooling", linewidth=0.5),
    Patch(facecolor="aliceblue", edgecolor="black", hatch="....", label="Support", linewidth=0.5),
]

# First legend: workloads (Top Right corner, inside the plot area boundaries)
leg1 = ax.legend(
    handles=workload_handles,
    loc="upper right",
    bbox_to_anchor=(1.0, 1.12),
    fontsize=5,
    title="Workloads",
    title_fontsize=6,
    framealpha=0.9,
    ncol=3,
    columnspacing=0.8,
    handletextpad=0.4,
    handlelength=1.2,
    labelspacing=0.2
)
ax.add_artist(leg1)

# Second legend: stack components (Below workloads legend)
leg2 = ax.legend(
    handles=type_handles,
    loc="upper right",
    bbox_to_anchor=(1.0, 0.95),
    fontsize=5,
    title="Components",
    title_fontsize=6,
    framealpha=0.9,
    ncol=3,
    columnspacing=0.8,
    handletextpad=0.4,
    handlelength=1.2,
    labelspacing=0.2
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Slight top margin adjustment
plt.subplots_adjust(top=0.85)

# Save
out_path = Path("Total_Energy_vs_Temp_10SPEC_IEEE_simplified.png")
fig.savefig(out_path, dpi=350, bbox_inches="tight")
plt.close(fig)

print(f"Saved to {out_path.resolve()}")