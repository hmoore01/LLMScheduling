import pandas as pd
import numpy as np

# 1. Load data and define objectives
df = pd.read_csv('experiment_results/util_sweep_20260322_0718.csv')
obj_cols = ['avg_ttft_s', 'total_carbon_kg', 'total_water_l', 'total_energy_usd']
df = df.dropna(subset=obj_cols)

# 2. Get global min/max for normalization across the ENTIRE dataset
global_min = df[obj_cols].min().values
global_max = df[obj_cols].max().values

# Prevent division by zero if an objective happens to be perfectly constant
range_vals = global_max - global_min
range_vals[range_vals == 0] = 1.0

# Normalize the data: (value - min) / (max - min)
df_norm = df.copy()
df_norm[obj_cols] = (df[obj_cols] - global_min) / range_vals

# 3. Establish Normalized Global Min and Reference Point
norm_global_min = np.zeros(len(obj_cols))  # All minimums are now perfectly 0.0
norm_ref_point = np.ones(len(obj_cols)) * 1.05  # All maximums are 1.0, add 5% buffer -> 1.05
bounding_box_volume = np.prod(norm_ref_point - norm_global_min)  # 1.05^4 = ~1.2155


def get_non_dominated(points):
    """Filters a set of points to return only the strictly non-dominated Pareto front."""
    if len(points) == 0:
        return points
    is_efficient = np.ones(points.shape[0], dtype=bool)
    for i, c in enumerate(points):
        if is_efficient[i]:
            strictly_better = (points < c).any(axis=1)
            is_efficient[is_efficient] = strictly_better[is_efficient]
            is_efficient[i] = True
    return points[is_efficient]


def estimate_hypervolume_mc(front, ref_point, global_min_pt, num_samples=2000000):
    """Monte Carlo estimation of Hypervolume using highly optimized pure NumPy."""
    if len(front) == 0:
        return 0.0

    dims = len(ref_point)
    samples = np.random.uniform(low=global_min_pt, high=ref_point, size=(num_samples, dims))

    is_dominated = np.zeros(num_samples, dtype=bool)
    for p in front:
        dominates = np.all(p <= samples, axis=1)
        is_dominated |= dominates

    fraction = is_dominated.sum() / num_samples
    return fraction * bounding_box_volume


results = {}

# 4. Create groups for each framework (using NORMALIZED data)
groups = {fw: df_norm[df_norm['framework'] == fw] for fw in df_norm['framework'].unique()}
groups['MARLIN/parliament (Collective)'] = df_norm[df_norm['framework'].str.lower().isin(['marlin', 'parliament'])]

# 5. Extract front and compute Hypervolume
print("Computing Normalized Pareto fronts and estimating Hypervolume...\n")
for name, group in groups.items():
    points = group[obj_cols].values
    if len(points) == 0:
        continue

    # Extract front from normalized points
    front = get_non_dominated(points)

    # Calculate HV using normalized bounds
    hv_estimate = estimate_hypervolume_mc(front, norm_ref_point, norm_global_min, num_samples=2000000)

    results[name] = {'Total Runs': len(points), 'Front Size': len(front), 'Norm. Est. PHV': hv_estimate}

# 6. Display results formatted nicely
results_df = pd.DataFrame(results).T
results_df = results_df.sort_values(by='Norm. Est. PHV', ascending=False)
pd.options.display.float_format = '{:.4f}'.format
print(results_df)