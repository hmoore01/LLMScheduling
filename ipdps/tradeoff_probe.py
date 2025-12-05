#!/usr/bin/env python3
"""
Probe the tradeoff surface of the Rate_Flow_Sim + ResourceEnv stack by
sampling random actions and plotting the resulting metrics.

Usage example:

    python tradeoff_probe.py \
        --epoch-csv path/to/epoch_data.csv \
        --num-dcs 12 \
        --epoch-idx 0 \
        --spec-dir sim_specs \
        --profile time_agent

Make sure:
  - Your ResourceEnv class is in resource_env.py (or fix the import).
  - spec_dir points at the directory where LLM_Simulator expects its specs.
"""

import argparse
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Adjust this import if your file/module name is different
from MultiAgentRL import ResourceEnv


# ---------------------------------------------------------------------
# 1. Build a ResourceEnv config similar to your training setup
# ---------------------------------------------------------------------
def build_agent_specs() -> Dict[str, Dict[str, Any]]:
    """
    Simple default agent_specs. You can modify this to match your
    actual MARL training setup.
    """
    return {
        "time_agent": {
            "weights": {"ttft": 1.0},
            "constraints": {},
            "include_duals_in_obs": False,
            "penalty_weight": 1.0,
        },
        "carbon_agent": {
            "weights": {"carbon": 1.0},
            "constraints": {},
            "include_duals_in_obs": False,
            "penalty_weight": 1.0,
        },
        "water_agent": {
            "weights": {"water": 1.0},
            "constraints": {},
            "include_duals_in_obs": False,
            "penalty_weight": 1.0,
        },
        "cost_agent": {
            "weights": {"cost": 1.0},
            "constraints": {},
            "include_duals_in_obs": False,
            "penalty_weight": 1.0,
        },
    }


def load_epoch_df(path: str, epoch_idx: int) -> pd.DataFrame:
    """
    Load a CSV and restrict it to a single epoch (epoch_idx) if there
    is an 'epoch' column. If no epoch column is present, we treat the
    whole file as a single epoch.
    """
    df = pd.read_csv(path)

    if "epoch" in df.columns:
        if epoch_idx not in df["epoch"].unique():
            raise ValueError(
                f"Requested epoch_idx={epoch_idx}, but available epochs are: "
                f"{sorted(df['epoch'].unique())}"
            )
        df = df[df["epoch"] == epoch_idx].copy()
    else:
        # Add a dummy epoch column so ResourceEnv._epoch_pool logic is happy
        df = df.copy()
        df["epoch"] = epoch_idx

    return df


def make_env(
    epoch_csv: str,
    epoch_idx: int,
    num_dcs: int,
    spec_dir: str,
    epoch_length: int,
    active_profile: str,
    debug: bool = False,
) -> ResourceEnv:
    """
    Construct a ResourceEnv instance using a single-epoch dataframe and
    simple agent_specs.
    """
    epoch_df = load_epoch_df(epoch_csv, epoch_idx)
    agent_specs = build_agent_specs()

    if active_profile not in agent_specs:
        raise KeyError(
            f"active_profile '{active_profile}' not in agent_specs keys: "
            f"{list(agent_specs.keys())}"
        )

    # Minimal epoch_summary; ResourceEnv will fall back for missing fields
    epoch_summary: Dict[str, Any] = {
        "spec_dir": spec_dir,
        "epoch_length": int(epoch_length),
        # Optional: you can add dc_carbon_intensity, dc_water_intensity, etc. here
    }

    # node_properties isn't used by the pasted ResourceEnv logic directly,
    # but we pass an empty dict so the interface is satisfied.
    node_properties: Dict[str, Any] = {}

    config: Dict[str, Any] = {
        "epoch_df": epoch_df,
        "node_properties": node_properties,
        "epoch_idx": int(epoch_idx),
        "num_datacenters": int(num_dcs),
        "epoch_summary": epoch_summary,
        "max_steps": 1,
        "agent_specs": agent_specs,
        "active_agent_profile": active_profile,
        "debug": debug,
    }

    env = ResourceEnv(config)

    # Optional: Make sure we only sample this epoch (should already be true
    # because epoch_df only contains one epoch, but this is extra safety).
    env._available_epochs = [int(epoch_idx)]

    return env


# ---------------------------------------------------------------------
# 2. Sample random actions and record metrics
# ---------------------------------------------------------------------
def sample_random_policies(
    env: ResourceEnv,
    n_samples: int = 200,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """
    For a fixed env configuration (fixed epoch), sample random actions,
    run 1-step episodes, and collect metrics.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    records = []

    for i in range(n_samples):
        obs, infos = env.reset()

        # Build a random action dict: one random action per agent from its action_space
        actions: Dict[str, np.ndarray] = {}
        for agent in env.agents:
            actions[agent] = env.action_space(agent).sample()

        # Single step (episode is 1-step by design)
        obs_next, rewards, terminations, truncations, infos_step = env.step(actions)

        # Just grab metrics from the first agent (they're identical across agents)
        any_agent = env.agents[0]
        info = infos_step[any_agent]
        raw_metrics: Dict[str, float] = info["raw_metrics"]

        record = {
            "sample_id": i,
            "reward_raw": info.get("reward_raw", np.nan),
            "reward_scaled": info.get("reward_scaled", np.nan),
            "ttft": raw_metrics.get("ttft", np.nan),
            "carbon_emissions": raw_metrics.get("carbon_emissions", np.nan),
            "water_usage": raw_metrics.get("water_usage", np.nan),
            "energy_cost": raw_metrics.get("energy_cost", np.nan),
            "total_energy": raw_metrics.get("total_energy", np.nan),
            "network_load": raw_metrics.get("network_load", np.nan),
        }

        records.append(record)

    df = pd.DataFrame(records)
    return df


# ---------------------------------------------------------------------
# 3. Plot pairwise tradeoffs
# ---------------------------------------------------------------------
def plot_tradeoffs(df: pd.DataFrame, title_prefix: str = "") -> None:
    """
    Make simple scatter plots to visualize relationships between metrics.
    """
    metrics_pairs = [
        ("ttft", "carbon_emissions"),
        ("ttft", "energy_cost"),
        ("ttft", "water_usage"),
        ("carbon_emissions", "energy_cost"),
        ("carbon_emissions", "water_usage"),
        ("energy_cost", "water_usage"),
    ]

    for x, y in metrics_pairs:
        if x not in df.columns or y not in df.columns:
            continue

        plt.figure()
        plt.scatter(df[x], df[y], alpha=0.6)
        plt.xlabel(x)
        plt.ylabel(y)
        plt.title(f"{title_prefix}{x} vs {y}")
        plt.tight_layout()

    plt.show()


# ---------------------------------------------------------------------
# 4. CLI + main
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe ResourceEnv tradeoff surface.")
    parser.add_argument(
        "--epoch-csv",
        type=str,
        required=True,
        help="Path to CSV containing epoch data (with columns like epoch, source_dc_id, model_type, num_tokens).",
    )
    parser.add_argument(
        "--num-dcs",
        type=int,
        required=True,
        help="Number of datacenters in the simulation.",
    )
    parser.add_argument(
        "--epoch-idx",
        type=int,
        default=0,
        help="Epoch index to probe (if epoch column exists in CSV).",
    )
    parser.add_argument(
        "--spec-dir",
        type=str,
        default="sim_specs",
        help="Directory containing simulator spec files (for LLM_Simulator).",
    )
    parser.add_argument(
        "--epoch-length",
        type=int,
        default=900,
        help="Epoch length in seconds (passed to LLM_Simulator).",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="time_agent",
        help="Active agent profile to use (time_agent, carbon_agent, water_agent, cost_agent).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=300,
        help="Number of random policies to sample.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output from ResourceEnv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    env = make_env(
        epoch_csv=args.epoch_csv,
        epoch_idx=args.epoch_idx,
        num_dcs=args.num_dcs,
        spec_dir=args.spec_dir,
        epoch_length=args.epoch_length,
        active_profile=args.profile,
        debug=args.debug,
    )

    print(
        f"[TRADEOFF PROBE] NUM_DATACENTERS={env.NUM_DATACENTERS}, "
        f"profile={args.profile}, epoch_idx={args.epoch_idx}"
    )

    df = sample_random_policies(env, n_samples=args.samples)
    print("\n[TRADEOFF PROBE] Summary statistics:")
    print(df.describe())

    plot_tradeoffs(df, title_prefix=f"profile={args.profile} | epoch={args.epoch_idx} | ")


if __name__ == "__main__":
    main()
