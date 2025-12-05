import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "DejaVu Serif"

# ------------------------------------------------
# Data definition (from your message)
# ------------------------------------------------

temps = np.array([20.0, 25.0, 30.0, 35.0, 40.0, 45.0])

node_configs = ["500 nodes", "1000 nodes", "1500 nodes", "2000 nodes"]

# For convenience, store each metric as a dict[node_label] -> np.array([...])

total_energy = {
    "500 nodes": np.array([
        2697.110947,
        2205.984868,
        2017.090223,
        1916.680374,
        1854.261975,
        1811.993931,
    ]),
    "1000 nodes": np.array([
        2862.918035,
        2341.599581,
        2141.092484,
        2034.132632,
        1967.402468,
        1922.376909,
    ]),
    "1500 nodes": np.array([
        3024.024588,
        2473.369696,
        2261.579352,
        2148.255187,
        2077.335493,
        2029.630594,
    ]),
    "2000 nodes": np.array([
        3185.131140,
        2605.139810,
        2382.066221,
        2262.377742,
        2187.268519,
        2136.884278,
    ]),
}

it_energy = {
    "500 nodes": np.array([
        1408.917938,
        1408.917938,
        1408.917938,
        1408.618969,
        1408.220344,
        1408.070859,
    ]),
    "1000 nodes": np.array([
        1495.532314,
        1495.532314,
        1495.532314,
        1494.937732,
        1494.144957,
        1493.847667,
    ]),
    "1500 nodes": np.array([
        1579.691222,
        1579.691222,
        1579.691222,
        1578.809409,
        1577.633658,
        1577.192751,
    ]),
    "2000 nodes": np.array([
        1663.850130,
        1663.850130,
        1663.850130,
        1662.681085,
        1661.122358,
        1660.537836,
    ]),
}

cooling_energy = {
    "500 nodes": np.array([
        1105.033677,
        613.907598,
        425.012953,
        324.940939,
        262.972987,
        220.873860,
    ]),
    "1000 nodes": np.array([
        1172.966521,
        651.648067,
        451.140969,
        344.852995,
        279.018666,
        234.329046,
    ]),
    "1500 nodes": np.array([
        1238.973507,
        688.318615,
        476.528272,
        364.200556,
        294.609460,
        247.402785,
    ]),
    "2000 nodes": np.array([
        1304.980494,
        724.989163,
        501.915575,
        383.548116,
        310.200254,
        260.476523,
    ]),
}

# Define "Other" as the remaining portion: Total - IT - Cooling
other_energy = {}
for cfg in node_configs:
    other_energy[cfg] = total_energy[cfg] - it_energy[cfg] - cooling_energy[cfg]

# ------------------------------------------------
# Compute averages across node configs per temperature
# ------------------------------------------------

n_temps = len(temps)
n_cfgs = len(node_configs)

avg_per_temp = np.zeros(n_temps)
for i in range(n_temps):
    avg_per_temp[i] = np.mean([total_energy[cfg][i] for cfg in node_configs])

baseline_avg = avg_per_temp[0]  # 20°C baseline

# ------------------------------------------------
# Plot
# ------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

group_width = 0.8
bar_width = group_width / n_cfgs
indices = np.arange(n_temps)

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

for ci, cfg in enumerate(node_configs):
    x = indices - group_width / 2.0 + (ci + 0.5) * bar_width

    it_vals = it_energy[cfg]
    other_vals = other_energy[cfg]
    cool_vals = cooling_energy[cfg]
    total_vals = total_energy[cfg]

    # 1) IT energy: base
    ax.bar(
        x, it_vals,
        width=bar_width,
        color=colors[ci % len(colors)],
        edgecolor="black",
        linewidth=0.4,
    )

    # 3) Cooling energy: top, // hatch
    ax.bar(
        x, cool_vals,
        width=bar_width,
        bottom=it_vals,
        color="lightgrey",
        edgecolor="black",
        linewidth=0.4,
        hatch="//",
    )

    # 2) Other energy: middle, X hatch
    ax.bar(
        x, other_vals,
        width=bar_width,
        bottom=it_vals + cool_vals,
        color="lightblue",
        edgecolor="black",
        linewidth=0.4,
        hatch="xx",
    )

    # Total labels (vertical) above each bar
    for xi, tv, itv, othv, cv in zip(x, total_vals, it_vals, other_vals, cool_vals):
        ax.text(
            xi,
            itv + othv + cv + baseline_avg * 0.004,
            f"{tv:.0f}",
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=90,
        )

# ------------------------------------------------
# Axes, spines, legends
# ------------------------------------------------
ax.set_xlabel("Temperature (°C)", fontsize=11)
ax.set_ylabel("Total Energy (kWh)", fontsize=11)

ax.set_xticks(indices)
ax.set_xticklabels([f"{t:.0f}" for t in temps], fontsize=9)
ax.tick_params(axis="y", labelsize=9)

# Optional: remove some spines for a cleaner look
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

from matplotlib.patches import Patch

# Legend: node configurations
cfg_handles = [
    Patch(facecolor=colors[i % len(colors)], edgecolor="black", label=node_configs[i])
    for i in range(n_cfgs)
]

leg1 = ax.legend(
    handles=cfg_handles,
    loc="upper right",
    bbox_to_anchor=(0.850, 0.995),
    fontsize=7,
    title="Datacenter Size",
    framealpha=0.9,
)

# Legend: stack components
comp_handles = [
    Patch(facecolor="black", edgecolor="black", alpha=0.6, label="Compute Energy"),
    Patch(facecolor="lightgrey", edgecolor="black", hatch="//", label="Cooling Energy"),
    Patch(facecolor="lightblue", edgecolor="black", hatch="xx", label="Support Energy"),
]

leg2 = ax.legend(
    handles=comp_handles,
    loc="upper right",
    bbox_to_anchor=(0.995, 0.995),
    fontsize=7,
    title="Components",
    framealpha=0.9,   # two-column legend for components
)

ax.add_artist(leg1)  # keep both legends

fig.tight_layout()
fig.savefig("Total_Energy_vs_Temp_nodes_stacked.png", dpi=300, bbox_inches="tight")
plt.close(fig)

