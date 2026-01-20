import pandas as pd
import numpy as np
import os
from MultiAgentRL_Broken import ResourceEnv

# --- Helper Functions (Copied from simulator_LLM.py) ---

def _map_model_to_llama(m: str) -> str:
    s = str(m).strip().lower()
    if ("chatgpt" in s) or ("gpt-3.5" in s) or ("gpt3.5" in s):
        return "Llama7b"
    if ("gpt-4" in s) or ("gpt4" in s):
        return "Llama70b"
    if "70" in s or "70b" in s or "llama-2-70b" in s or "llama2-70b" in s:
        return "Llama70b"
    if "7" in s or "7b" in s or "llama-2-7b" in s or "llama2-7b" in s:
        return "Llama7b"
    return str(m)

def _even_src_dc(df: pd.DataFrame, num_dcs: int) -> pd.Series:
    """Round-robin assign source DC if missing."""
    if "source_dc_id" in df.columns:
        return pd.to_numeric(df["source_dc_id"], errors="coerce").fillna(0).astype(int)
    out = np.zeros(len(df), dtype=int)
    if "epoch" not in df.columns:
        out = np.arange(len(df)) % max(1, num_dcs)
        return pd.Series(out, index=df.index, dtype=int)
    for ep, idx in df.groupby("epoch").indices.items():
        n = len(idx)
        out[idx] = np.arange(n) % max(1, num_dcs)
    return pd.Series(out, index=df.index, dtype=int)

def _ensure_num_tokens(df: pd.DataFrame, default_tokens: int = 400) -> pd.Series:
    if "num_tokens" in df.columns:
        return pd.to_numeric(df["num_tokens"], errors="coerce").fillna(0).astype(int)
    if "total_tokens" in df.columns:
        return pd.to_numeric(df["total_tokens"], errors="coerce").fillna(0).astype(int)
    if "tokens" in df.columns:
        return pd.to_numeric(df["tokens"], errors="coerce").fillna(0).astype(int)
    if "prompt_tokens" in df.columns:
        return pd.to_numeric(df["prompt_tokens"], errors="coerce").fillna(0).astype(int)
    return pd.Series(default_tokens, index=df.index, dtype=int)

# --- Main Baseline Calculation Logic ---

def calculate_average_baselines(num_epochs=10, workload_path="simulator_ready_trace.csv"):
    print(f"Loading trace from {workload_path}...")
    try:
        trace = pd.read_csv(workload_path)
    except FileNotFoundError:
        print("Error: 'simulator_ready_trace.csv' not found. Please provide the correct path.")
        return

    # Basic preprocessing
    if "epoch" not in trace.columns: trace["epoch"] = 0
    trace["source_dc_id"] = _even_src_dc(trace, 12) # Defaulting to 12 DCs
    trace["model_type"] = trace["model_type"].astype(str).map(_map_model_to_llama)
    trace["num_tokens"] = _ensure_num_tokens(trace)

    # Pre-calculate totals for summary
    trace['is_7b'] = trace["model_type"].str.contains("7b", case=False)
    trace['is_70b'] = trace["model_type"].str.contains("70b", case=False)

    grouped_trace = trace.groupby("epoch")
    available_epochs = sorted(grouped_trace.groups.keys())

    # Select random sample
    selected_epochs = np.random.choice(available_epochs, size=min(num_epochs, len(available_epochs)), replace=False)
    selected_epochs = sorted(selected_epochs)
    print(f"Running baseline calculation on {len(selected_epochs)} epochs: {selected_epochs}\n")

    totals = {"carbon": 0.0, "cost": 0.0, "water": 0.0, "total_energy": 0.0}

    for epoch_idx in selected_epochs:
        df = grouped_trace.get_group(epoch_idx).copy()

        # Calculate summary stats for this specific epoch
        l7b = df[df['is_7b']]["num_tokens"].sum()
        l70b = df[df['is_70b']]["num_tokens"].sum()

        config = {
            "epoch_df": df,
            "epoch_idx": epoch_idx,
            "num_datacenters": 12,
            "node_properties": {},
            "agent_specs": {"dummy": {"weights": {}}},
            "active_agent_profile": "dummy",
            "epoch_summary": {
                "llama7b_total": l7b,
                "llama70b_total": l70b,
                "spec_dir": "sim_specs",
                "a100_csv": "sim_specs/A100_GPU.csv",
                "h100_csv": "sim_specs/H100_GPU.csv",
                # Ensure dummy values for required fields
                "dc_carbon_intensity": [400]*12,
                "dc_water_intensity": [1.0]*12,
                "dc_energy_price": [0.10]*12
            }
        }

        # Initialize Env and Extract Baseline
        # Note: ResourceEnv calculates baseline in reset()
        env = ResourceEnv(config)
        env.reset()

        metrics = env.baseline_metrics

        print(f"Epoch {epoch_idx}: Carbon={metrics['carbon']:.2f}, Cost=${metrics['cost']:.2f}, Water={metrics['water']:.2f}")

        totals["carbon"] += metrics['carbon']
        totals["cost"] += metrics['cost']
        totals["water"] += metrics['water']
        totals["total_energy"] += metrics['total_energy']

    # Calculate Averages
    avgs = {k: v / len(selected_epochs) for k, v in totals.items()}

    print("\n" + "="*40)
    print("RECOMMENDED CONSTRAINT BUDGETS (PER EPOCH)")
    print("="*40)
    print(f"Average Baseline Carbon: {avgs['carbon']:.2f} g")
    print(f"  -> Recommended Budget (80%): {avgs['carbon'] * 0.8:.2f}")

    print(f"Average Baseline Cost:   ${avgs['cost']:.2f}")
    print(f"  -> Recommended Budget (80%): {avgs['cost'] * 0.8:.2f}")

    print(f"Average Baseline Water:  {avgs['water']:.2f} L")
    print(f"  -> Recommended Budget (80%): {avgs['water'] * 0.8:.2f}")
    print("="*40)

    print(f"Average Baseline Energy: {avgs['total_energy']:.2f} eV")
    print(f" -> Recommended Budget (80%): {avgs['total_energy'] * 0.8:.2f}")
    print("="*40)

if __name__ == "__main__":
    calculate_average_baselines()