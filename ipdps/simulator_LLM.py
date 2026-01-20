import random
import copy
import numpy as np
import math
# import matplotlib.pyplot as plt
import time
import pickle
import math
import argparse
import os
import csv
from sklearn.cluster import KMeans
import pandas as pd
import hashlib
from typing import Dict, Any, List, Optional, Callable, Union, Literal


def train_marl_constrained_profiles(
        epoch_data: pd.DataFrame,
        epoch_idx: int,
        num_datacenters: int,
        node_properties,
        total_timesteps: int = 100_000,
        num_envs: int = 1,
        overwrite_existing: bool = False,
        model_dir: str = "trained_models/sb3_agents",
        # Add new args to match call in __main__
        spec_dir: str = "sim_specs",
        epoch_length: int = 900,
):
    """
    Train MARL profiles using the multi-agent infrastructure in MultiAgentRL_Broken.py.
    """
    import MultiAgentRL

    # 1) Build full agent_specs from this file's helper
    agent_specs = build_agent_specs(num_datacenters=num_datacenters)

    # ... (Selection logic remains same) ...
    constrained_specs = {k: v for k, v in agent_specs.items() if v.get("constraints")}
    single_metric_specs = {k: v for k, v in agent_specs.items() if
                           not v.get("constraints") and len(v.get("weights", {})) == 1}

    if not constrained_specs and not single_metric_specs:
        return

    training_specs = {}
    training_specs.update(single_metric_specs)
    training_specs.update(constrained_specs)

    # 3) Build a minimal epoch_summary for ResourceEnv / MultiAgentRL
    # UPDATE: Calculate totals robustly
    is_7b = epoch_data["model_type"].astype(str).str.lower().str.contains("7b")
    is_70b = epoch_data["model_type"].astype(str).str.lower().str.contains("70b")

    llama7b_total = float(epoch_data[is_7b]["num_tokens"].sum())
    llama70b_total = float(epoch_data[is_70b]["num_tokens"].sum())

    # --- [NEW] Load Real Carbon Intensity from Specs ---
    # We need the agents to know the REAL carbon values, not the 1.0 fallback.
    import os
    dc_specs_path = os.path.join(spec_dir, "Datacenter_specs.csv")
    if os.path.exists(dc_specs_path):
        dc_df = pd.read_csv(dc_specs_path)
        # Sort by DC_Num to ensure alignment
        if "DC_Num" in dc_df.columns:
            dc_df = dc_df.sort_values("DC_Num")
        # Extract Carbon Intensity column
        # Matches the CSV header from Rate_Flow_Sim.py (usually 'Carbon_Intensity')
        if "Carbon_Intensity" in dc_df.columns:
            real_ci = dc_df["Carbon_Intensity"].astype(float).tolist()
        else:
            print("[WARNING] 'Carbon_Intensity' column not found in specs. Using default.")
            real_ci = [400.0] * num_datacenters  # Safer default than 1.0
    else:
        print("[WARNING] Datacenter_specs.csv not found. Using default 400.0.")
        real_ci = [400.0] * num_datacenters

    epoch_summary = {
        "llama7b_total": llama7b_total,
        "llama70b_total": llama70b_total,
        "num_datacenters": num_datacenters,
        "spec_dir": spec_dir,
        "epoch_length": epoch_length,
        # Pass the loaded values to the Env
        "dc_carbon_intensity": real_ci,
    }

    # 4) Delegate to the multi-agent trainer
    MultiAgentRL.train_with_multi_epoch(
        trace_path="simulator_ready_trace.csv",
        num_datacenters=12,
        agent_specs=build_agent_specs(12),
        node_properties=node_properties,
        total_timesteps=2_000_000,
        sampling_strategy="stratified",  # or "uniform", "curriculum"
        use_domain_randomization=True,
    )
    print("[MARL TRAIN] Finished training all selected MARL profiles.")


def write_epoch_stats(tag: str, epoch_index: int, stats_dict: dict, tag2: Optional[str] = None) -> None:
    outdir = "LLM_Results"
    os.makedirs(outdir, exist_ok=True)
    fname = f"{outdir}/{tag}{('_' + tag2) if tag2 else ''}_epoch_{epoch_index}.txt"
    with open(fname, "w") as f:
        f.write(f"Epoch {epoch_index}\n")
        for k, v in stats_dict.items():
            f.write(f"{k}: {v}\n")


def get_deterministic_perturbation(epoch_idx, error_rate):
    key = f"epoch_{epoch_idx}"
    hash_bytes = hashlib.sha256(key.encode()).digest()
    raw = int.from_bytes(hash_bytes[:4], 'big')  # first 4 bytes

    frac = (raw % 10 ** 6) / 10 ** 6  # → [0, 1)
    perturbation = 1.0 + (2 * frac - 1.0) * error_rate  # → [1 - e, 1 + e]

    return perturbation


def apply_rate_error(summary, epoch_idx, error_rate):
    if error_rate == 0.0:
        return summary

    perturbation = get_deterministic_perturbation(epoch_idx, error_rate)

    for key in ["llama7b_total", "llama70b_total"]:
        true_val = summary[key]
        summary[key] = int(max(0, round(true_val * perturbation)))

    # Recalculate rates after perturbation (assuming 900-second epochs)
    summary["llama7b_rate"] = summary["llama7b_total"] / 900
    summary["llama70b_rate"] = summary["llama70b_total"] / 900

    return summary


Number = Union[int, float]
MergePolicy = Literal["min", "max", "sum", "override"]


def _always() -> Callable[[int], bool]:
    return lambda _: True


def schedule_only_epochs(epochs: List[int]) -> Callable[[int], bool]:
    S = set(int(e) for e in epochs)
    return lambda e: e in S


def schedule_in_range(start_incl: int, end_incl: int) -> Callable[[int], bool]:
    s, t = int(start_incl), int(end_incl)
    return lambda e: s <= e <= t


def schedule_every_k(k: int, phase: int = 0, active_residue: Optional[List[int]] = None) -> Callable[[int], bool]:
    """Looping: active when (epoch - phase) % k is in residues (default {0})."""
    k = int(k);
    phase = int(phase)
    residues = {0} if not active_residue else set(int(r) for r in active_residue)
    return lambda e: ((e - phase) % k) in residues


def _merge_constraints(dst: Dict[str, Any], src: Dict[str, Any], policy: MergePolicy = "override", priority: int = 0):
    """
    Merge constraint dicts per-key. Supports min/max/sum/override.
    If you want priority to matter for override, put higher priority rules later in the list
    or pass policy="override" and call this in priority order.
    """
    for k, v in src.items():
        if k not in dst:
            dst[k] = v
            continue
        if policy == "override":
            dst[k] = v
        elif policy == "min":
            dst[k] = min(dst[k], v)
        elif policy == "max":
            dst[k] = max(dst[k], v)
        elif policy == "sum":
            dst[k] = (dst[k] + v)
        else:
            raise ValueError(f"Unknown merge policy: {policy}")


def resolve_epoch_constraints(agent_spec: Dict[str, Any], epoch_idx: int) -> Dict[str, Any]:
    """
    Returns a concrete constraints dict for the agent at epoch `epoch_idx`.
    Backward compatible:
      - if agent_spec has 'constraints' (old schema), they’re always active.
      - new schema: 'epoch_rules': list of {when: {...}, constraints: {...}, merge: "..."}
    Merge order:
      1) base 'constraints' (always)
      2) epoch_rules in list order (later rules can override earlier ones)
    """
    out: Dict[str, Any] = {}

    # 1) legacy: always-on constraints
    base = agent_spec.get("constraints", None)
    if isinstance(base, dict) and base:
        _merge_constraints(out, base, policy="override")

    # 2) epoch rules
    rules = agent_spec.get("epoch_rules", []) or []
    for r in rules:
        when = r.get("when", {"type": "always"})
        merge: MergePolicy = r.get("merge", "override")
        constraints: Dict[str, Any] = r.get("constraints", {}) or {}

        kind = when.get("type", "always")
        if kind == "always":
            active = True
        elif kind == "epochs":
            active = schedule_only_epochs(when.get("list", []))(epoch_idx)
        elif kind == "range":
            active = schedule_in_range(when.get("start", 0), when.get("end", 0))(epoch_idx)
        elif kind == "loop":
            active = schedule_every_k(
                when.get("k", 24),
                when.get("phase", 0),
                when.get("residues", None),
            )(epoch_idx)
        else:
            raise ValueError(f"Unknown schedule type: {kind}")

        if active:
            _merge_constraints(out, constraints, policy=merge)

    return out


def build_agent_specs(num_datacenters: int = 12) -> Dict[str, Dict[str, Any]]:
    """
    Build agent specifications with tuned constraints.

    Budget level: moderate
    Based on baseline metrics:
      Carbon: 8,000 kg
      Water: 5,000 L
      Cost: $2,800
      Energy: 22,774 kWh
    """
    return {
        # Single-objective agents (no constraints, pure optimization)
        "time_agent": {
            "weights": {"ttft": 10},
            "constraints": {},
        },
        "carbon_agent": {
            "weights": {"carbon": 10},
            "constraints": {},
        },
        "water_agent": {
            "weights": {"water": 10},
            "constraints": {},
        },
        "cost_agent": {
            "weights": {"cost": 10},
            "constraints": {},
        },

        # Hybrid agents (balanced objectives with constraints)
        "green_perf": {
            "weights": {"ttft": 6, "carbon": 3, "cost": 1},
            "constraints": {
                "carbon": {
                    "budget": 4000,  # 50% of baseline
                    "scope": "global",
                    "penalty": 0.5,
                },
            },
        },
        "cost_guard": {
            "weights": {"ttft": 7, "cost": 3},
            "constraints": {
                "cost": {
                    "budget": 2800,  # 100% of baseline
                    "scope": "global",
                    "penalty": 0.5,
                },
            },
        },
        "water_saver": {
            "weights": {"ttft": 7, "water": 3},
            "constraints": {
                "water": {
                    "budget": 2500,  # 50% of baseline
                    "scope": "global",
                    "penalty": 0.5,
                },
            },
        },
        "peak_power_guard": {
            "weights": {"ttft": 8, "total_energy": 2},
            "constraints": {
                "total_energy": {
                    "budget": 25051,  # 110% of baseline
                    "scope": "global",
                    "penalty": 0.3,
                },
            },
        },
    }


def _unwrap_to_resource_env(env):
    # Reuse the robust unwrap you already have in the logger
    def recursive_find(e, max_depth=20):
        visited = set()
        stack = [(e, 0)]
        while stack:
            cur, depth = stack.pop()
            if id(cur) in visited or depth > max_depth:
                continue
            visited.add(id(cur))
            if isinstance(cur, ResourceEnv):
                return cur
            for attr in dir(cur):
                if attr.startswith("__"):
                    continue
                try:
                    sub = getattr(cur, attr)
                    if isinstance(sub, (list, tuple)):
                        stack.extend((item, depth + 1) for item in sub)
                    elif hasattr(sub, "__class__"):
                        stack.append((sub, depth + 1))
                except Exception:
                    continue
        return None

    return recursive_find(env)


def rollout_once_collect_leftovers(env_config, model_dir, profile_name):
    import supersuit
    from supersuit import black_death_v3
    from stable_baselines3 import PPO

    cfg = dict(env_config)
    cfg["active_agent_profile"] = profile_name

    raw_env = ResourceEnv(cfg)
    death_wrapped = black_death_v3(raw_env)
    vec_env = supersuit.pettingzoo_env_to_vec_env_v1(death_wrapped)
    venv = supersuit.concat_vec_envs_v1(vec_env, num_vec_envs=1, base_class="stable_baselines3")

    model_path = os.path.join(model_dir, profile_name, "final_model.zip")
    model = PPO.load(model_path, env=venv, device="cpu")

    obs = venv.reset()
    # Roll until env says done (max_steps in config)
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = venv.step(action)
        # vectorized dones: they come as arrays; consider done when all environments report done
        if isinstance(dones, dict):
            done = bool(dones.get("__all__", False))
        else:
            # sb3 vec env returns array; end when all True
            try:
                done = bool(np.all(dones))
            except Exception:
                done = False

    # Unwrap and fetch leftovers from the inner env
    inner_env = _unwrap_to_resource_env(venv)
    leftovers = inner_env.get_last_leftovers() if inner_env is not None else None

    # Cleanup
    venv.close()
    del model, venv, vec_env, death_wrapped, raw_env

    return leftovers or []


if __name__ == "__main__":
    import argparse, os
    import pandas as pd
    import numpy as np
    from typing import List

    # ---------- CLI ----------
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--laxity', type=int, default=10)
    parser.add_argument('-s', '--slo', type=float, default=0.25)
    parser.add_argument('-e', '--epoch', type=int, default=96)
    parser.add_argument('-t', '--time', type=int, default=110)
    parser.add_argument('-n', '--node', type=int, default=8)
    parser.add_argument('-d', '--duration', type=int, default=22)
    parser.add_argument('-r', '--request', type=int, default=1)
    parser.add_argument('-f', '--framework', type=str, default='Helix',
                        choices=['Helix', 'NSGA2', 'PerLLM', 'Splitwise', 'Hybrid', 'MARL'])

    # Scaling
    parser.add_argument('--freq-scale', type=float, default=1.0)
    parser.add_argument('--token-scale', type=float, default=10.0)
    parser.add_argument('--count-scale', type=int, default=2)
    parser.add_argument('--error-rate', type=float, default=0.0)

    # Optional: override # of DCs used for default distribution when src DC is missing
    parser.add_argument('--num-dcs', type=int, default=12)
    parser.add_argument(
        '--train-marl',
        action='store_true',
        help='If set and framework==MARL, train all constrained MARL profiles instead of running evaluation.'
    )
    parser.add_argument(
        '--marl-timesteps',
        type=int,
        default=100_000,
        help='Total PPO timesteps per constrained MARL profile.'
    )
    parser.add_argument(
        '--marl-num-envs',
        type=int,
        default=1,
        help='Number of parallel vector envs to use for MARL training.'
    )
    parser.add_argument(
        '--marl-overwrite',
        action='store_true',
        help='Retrain and overwrite existing MARL models if they already exist.'
    )

    # Distribution mode for request origins
    parser.add_argument(
        '--distribution',
        type=str,
        default='even',
        choices=['even', 'population', 'time'],
        help='Request origin distribution mode: even (round-robin), population (weighted by region), or time (diurnal pattern following activity hours)'
    )

    # Population weights can be specified as JSON or use defaults
    parser.add_argument(
        '--population-weights',
        type=str,
        default=None,
        help='JSON string of population weights per DC (e.g., \'{"0":0.15,"1":0.10,...}\')'
    )

    # Time zone offsets for time-based distribution
    parser.add_argument(
        '--timezone-offsets',
        type=str,
        default=None,
        help='JSON string of UTC offset hours per DC (e.g., \'{"0":-5,"1":-8,...}\') for time-based distribution'
    )

    # Spec directory override
    parser.add_argument(
        '--spec-dir',
        type=str,
        default='sim_specs',
        help='Directory containing simulation spec CSV files'
    )

    # Custom node type counts (for scalability experiments)
    parser.add_argument(
        '--node-type-counts',
        type=str,
        default=None,
        help='JSON string of node type counts per type (e.g., \'{"0":167,"1":167,...}\')'
    )

    args = parser.parse_args()


    # ---------- Helpers ----------
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
        # fallback: pass-through
        return str(m)


    # Default population weights for major regions (normalized)
    # Based on approximate global internet user distribution
    DEFAULT_POPULATION_WEIGHTS = {
        0: 0.12,  # US East
        1: 0.10,  # US West
        2: 0.08,  # US Central
        3: 0.15,  # Europe West
        4: 0.10,  # Europe Central
        5: 0.05,  # Europe North
        6: 0.18,  # Asia Pacific (China region)
        7: 0.08,  # Asia Pacific (Japan/Korea)
        8: 0.06,  # Asia Pacific (Southeast)
        9: 0.04,  # South America
        10: 0.02,  # Middle East
        11: 0.02,  # Africa
    }

    # Default timezone offsets (UTC hours) for time-based distribution
    # Maps DC regions to their approximate UTC offsets
    DEFAULT_TIMEZONE_OFFSETS = {
        0: -5,  # US East (EST/EDT)
        1: -8,  # US West (PST/PDT)
        2: -6,  # US Central (CST/CDT)
        3: 0,  # Europe West (GMT/WET)
        4: 1,  # Europe Central (CET)
        5: 2,  # Europe North (EET)
        6: 8,  # Asia Pacific - China (CST)
        7: 9,  # Asia Pacific - Japan/Korea (JST/KST)
        8: 7,  # Asia Pacific - Southeast (ICT)
        9: -3,  # South America (BRT)
        10: 3,  # Middle East (AST)
        11: 2,  # Africa (CAT)
    }

    # Base population for time-weighted distribution (used as multiplier)
    DEFAULT_BASE_POPULATION = {
        0: 0.12,  # US East
        1: 0.10,  # US West
        2: 0.08,  # US Central
        3: 0.15,  # Europe West
        4: 0.10,  # Europe Central
        5: 0.05,  # Europe North
        6: 0.18,  # Asia Pacific (China region)
        7: 0.08,  # Asia Pacific (Japan/Korea)
        8: 0.06,  # Asia Pacific (Southeast)
        9: 0.04,  # South America
        10: 0.02,  # Middle East
        11: 0.02,  # Africa
    }


    def _even_src_dc(df: pd.DataFrame, num_dcs: int) -> pd.Series:
        """Round-robin assign source DC if missing; per-epoch so distribution is even each epoch."""
        if "source_dc_id" in df.columns:
            return pd.to_numeric(df["source_dc_id"], errors="coerce").fillna(0).astype(int)
        # build per-epoch RR
        out = np.zeros(len(df), dtype=int)
        if "epoch" not in df.columns:
            # single-epoch fallback
            out = np.arange(len(df)) % max(1, num_dcs)
            return pd.Series(out, index=df.index, dtype=int)
        for ep, idx in df.groupby("epoch").indices.items():
            # indices arrives as numpy array of row positions
            n = len(idx)
            out[idx] = np.arange(n) % max(1, num_dcs)
        return pd.Series(out, index=df.index, dtype=int)


    def _population_weighted_src_dc(df: pd.DataFrame, num_dcs: int, weights: dict = None) -> pd.Series:
        """Assign source DC based on population weights; per-epoch for reproducibility."""
        if weights is None:
            weights = DEFAULT_POPULATION_WEIGHTS

        # Normalize weights to available DCs
        available_weights = {k: v for k, v in weights.items() if k < num_dcs}
        total = sum(available_weights.values())
        if total > 0:
            available_weights = {k: v / total for k, v in available_weights.items()}
        else:
            # Fallback to even distribution
            available_weights = {i: 1.0 / num_dcs for i in range(num_dcs)}

        dc_ids = list(available_weights.keys())
        dc_probs = list(available_weights.values())

        out = np.zeros(len(df), dtype=int)
        if "epoch" not in df.columns:
            out = np.random.choice(dc_ids, size=len(df), p=dc_probs)
            return pd.Series(out, index=df.index, dtype=int)

        # Per-epoch assignment with fixed seed for reproducibility
        for ep, idx in df.groupby("epoch").indices.items():
            np.random.seed(int(ep) * 42)  # Deterministic per epoch
            n = len(idx)
            out[idx] = np.random.choice(dc_ids, size=n, p=dc_probs)

        return pd.Series(out, index=df.index, dtype=int)


    def _time_based_src_dc(
            df: pd.DataFrame,
            num_dcs: int,
            timezone_offsets: dict = None,
            base_population: dict = None,
            epoch_length_sec: int = 900,
            simulation_start_hour: int = 0
    ) -> pd.Series:
        """
        Assign source DC based on time-of-day activity patterns.

        This simulates realistic diurnal traffic patterns where request origins
        shift based on local time at each DC's region. Traffic follows the sun,
        with more requests originating from regions during their active hours.

        Activity model:
        - Peak activity: 9 AM - 9 PM local time (business + evening hours)
        - Low activity: 12 AM - 6 AM local time (night hours)
        - Medium activity: 6 AM - 9 AM and 9 PM - 12 AM (transition periods)

        Args:
            df: DataFrame with 'epoch' column
            num_dcs: Number of datacenters
            timezone_offsets: Dict mapping DC ID to UTC offset in hours
            base_population: Dict mapping DC ID to base population weight
            epoch_length_sec: Duration of each epoch in seconds (default 900 = 15 min)
            simulation_start_hour: Starting hour of simulation in UTC (default 0 = midnight)

        Returns:
            Series of source DC IDs
        """
        if timezone_offsets is None:
            timezone_offsets = DEFAULT_TIMEZONE_OFFSETS
        if base_population is None:
            base_population = DEFAULT_BASE_POPULATION

        def get_activity_multiplier(local_hour: float) -> float:
            """
            Return activity multiplier based on local hour (0-24).
            Models typical human activity patterns:
            - Night (0-6): Very low activity (0.1-0.3)
            - Morning transition (6-9): Rising activity (0.3-0.8)
            - Day/Evening peak (9-21): High activity (0.8-1.0)
            - Night transition (21-24): Declining activity (0.5-0.3)
            """
            hour = local_hour % 24

            if 0 <= hour < 6:
                # Night: very low, slight increase toward morning
                return 0.1 + 0.03 * hour  # 0.1 to 0.28
            elif 6 <= hour < 9:
                # Morning ramp-up
                return 0.3 + 0.23 * (hour - 6)  # 0.3 to 0.99
            elif 9 <= hour < 12:
                # Morning peak
                return 0.9 + 0.03 * (hour - 9)  # 0.9 to 0.99
            elif 12 <= hour < 14:
                # Lunch dip
                return 0.85
            elif 14 <= hour < 18:
                # Afternoon peak
                return 0.95
            elif 18 <= hour < 21:
                # Evening peak (often highest for consumer services)
                return 1.0
            elif 21 <= hour < 23:
                # Evening decline
                return 0.8 - 0.2 * (hour - 21)  # 0.8 to 0.4
            else:  # 23-24
                # Late night
                return 0.3

        def compute_epoch_weights(epoch: int, num_dcs: int) -> dict:
            """Compute DC weights for a specific epoch based on local time activity."""
            # Calculate UTC hour for this epoch
            # Each epoch is epoch_length_sec seconds, starting from simulation_start_hour
            hours_elapsed = (epoch * epoch_length_sec) / 3600.0
            utc_hour = (simulation_start_hour + hours_elapsed) % 24

            weights = {}
            for dc_id in range(num_dcs):
                # Get timezone offset for this DC
                tz_offset = timezone_offsets.get(dc_id, 0)

                # Calculate local hour at this DC
                local_hour = (utc_hour + tz_offset) % 24

                # Get activity multiplier based on local time
                activity = get_activity_multiplier(local_hour)

                # Get base population weight for this DC
                base_pop = base_population.get(dc_id, 1.0 / num_dcs)

                # Final weight = base_population * activity_multiplier
                weights[dc_id] = base_pop * activity

            # Normalize weights to sum to 1
            total = sum(weights.values())
            if total > 0:
                weights = {k: v / total for k, v in weights.items()}
            else:
                weights = {i: 1.0 / num_dcs for i in range(num_dcs)}

            return weights

        out = np.zeros(len(df), dtype=int)

        if "epoch" not in df.columns:
            # Single epoch, use epoch 0
            weights = compute_epoch_weights(0, num_dcs)
            dc_ids = list(weights.keys())
            dc_probs = list(weights.values())
            out = np.random.choice(dc_ids, size=len(df), p=dc_probs)
            return pd.Series(out, index=df.index, dtype=int)

        # Per-epoch assignment based on time-of-day
        for ep, idx in df.groupby("epoch").indices.items():
            np.random.seed(int(ep) * 42)  # Deterministic per epoch

            # Compute weights for this epoch
            weights = compute_epoch_weights(int(ep), num_dcs)
            dc_ids = list(weights.keys())
            dc_probs = list(weights.values())

            n = len(idx)
            out[idx] = np.random.choice(dc_ids, size=n, p=dc_probs)

        return pd.Series(out, index=df.index, dtype=int)


    def _assign_src_dc(
            df: pd.DataFrame,
            num_dcs: int,
            distribution: str,
            weights: dict = None,
            timezone_offsets: dict = None
    ) -> pd.Series:
        """Unified source DC assignment based on distribution mode."""
        if "source_dc_id" in df.columns:
            # Already has source_dc_id, but may need to apply distribution
            existing = pd.to_numeric(df["source_dc_id"], errors="coerce").fillna(0).astype(int)
            # If trace already has valid assignments, respect them unless forcing redistribution
            if existing.max() > 0:
                return existing

        if distribution == "population":
            return _population_weighted_src_dc(df, num_dcs, weights)
        elif distribution == "time":
            return _time_based_src_dc(df, num_dcs, timezone_offsets, weights)
        else:  # 'even' or default
            return _even_src_dc(df, num_dcs)


    def _ensure_num_tokens(df: pd.DataFrame, default_tokens: int = 400) -> pd.Series:
        if "num_tokens" in df.columns:
            return pd.to_numeric(df["num_tokens"], errors="coerce").fillna(0).astype(int)
        if "total_tokens" in df.columns:
            return pd.to_numeric(df["total_tokens"], errors="coerce").fillna(0).astype(int)
        if "tokens" in df.columns:
            return pd.to_numeric(df["tokens"], errors="coerce").fillna(0).astype(int)
        if {"prompt_tokens", "gen_tokens"}.issubset(df.columns):
            vals = pd.to_numeric(df["prompt_tokens"], errors="coerce").fillna(0) + \
                   pd.to_numeric(df["gen_tokens"], errors="coerce").fillna(0)
            return vals.astype(int)
        if "prompt_tokens" in df.columns:
            return pd.to_numeric(df["prompt_tokens"], errors="coerce").fillna(0).astype(int)
        return pd.Series(default_tokens, index=df.index, dtype=int)


    def summarize_epoch_rate(df: pd.DataFrame):
        """Simple summary printout for visibility (not used by simulator)."""
        grp = df.groupby(["source_dc_id", "model_type"], as_index=False)["num_tokens"].sum()
        grp.rename(columns={"num_tokens": "tokens"}, inplace=True)
        return grp


    # ---------- Load workload ----------
    workload_path = "simulator_ready_trace.csv"
    if not os.path.exists(workload_path):
        raise FileNotFoundError(f"Could not find workload CSV: {workload_path}")
    trace = pd.read_csv(workload_path)

    # Ensure epoch present and int
    if "epoch" not in trace.columns:
        # treat everything as epoch 0 if missing
        trace["epoch"] = 0
    trace["epoch"] = pd.to_numeric(trace["epoch"], errors="coerce").fillna(0).astype(int)

    # Parse population weights if provided
    population_weights = None
    if args.population_weights:
        import json as _json

        try:
            population_weights = {int(k): float(v) for k, v in _json.loads(args.population_weights).items()}
        except Exception as e:
            print(f"[WARNING] Failed to parse population weights: {e}. Using defaults.")
            population_weights = None

    # Parse timezone offsets if provided (for time-based distribution)
    timezone_offsets = None
    if hasattr(args, 'timezone_offsets') and args.timezone_offsets:
        import json as _json

        try:
            timezone_offsets = {int(k): float(v) for k, v in _json.loads(args.timezone_offsets).items()}
        except Exception as e:
            print(f"[WARNING] Failed to parse timezone offsets: {e}. Using defaults.")
            timezone_offsets = None

    # Source DC assignment based on distribution mode
    if "src_dc" in trace.columns and "source_dc_id" not in trace.columns:
        trace = trace.rename(columns={"src_dc": "source_dc_id"})

    # Use unified distribution function
    trace["source_dc_id"] = _assign_src_dc(
        trace,
        args.num_dcs,
        distribution=args.distribution,
        weights=population_weights,
        timezone_offsets=timezone_offsets
    )

    print(f"[INIT] Distribution mode: {args.distribution}")
    if args.distribution == "population":
        dist_summary = trace.groupby("source_dc_id").size()
        print(f"[INIT] Request distribution:\n{dist_summary}")
    elif args.distribution == "time":
        # Show distribution summary for a few sample epochs
        sample_epochs = [0, 24, 48, 72]  # ~0h, 6h, 12h, 18h if 15-min epochs
        print(f"[INIT] Time-based distribution (request counts by source DC):")
        for ep in sample_epochs:
            if ep <= trace["epoch"].max():
                ep_dist = trace[trace["epoch"] == ep].groupby("source_dc_id").size()
                print(f"  Epoch {ep}: {dict(ep_dist)}")

    # Model mapping (ChatGPT/GPT-4 → Llama7b/Llama70b)
    if "model_type" not in trace.columns:
        trace["model_type"] = "Llama7b"
    trace["model_type"] = trace["model_type"].astype(str).map(_map_model_to_llama)

    # Tokens
    trace["num_tokens"] = _ensure_num_tokens(trace, default_tokens=400)

    # All requests arrive at epoch start (the Helix path also enforces this; harmless to set here)
    trace["arrival_ms"] = 0

    # Compatibility: create a time_index column for legacy code paths (not used in request-mode)
    if "time_index" not in trace.columns:
        trace["time_index"] = 0

    # Enforce types
    trace["source_dc_id"] = pd.to_numeric(trace["source_dc_id"], errors="coerce").fillna(0).astype(int)
    trace["num_tokens"] = pd.to_numeric(trace["num_tokens"], errors="coerce").fillna(0).astype(int)

    # Group by epoch
    grouped_trace = trace.groupby("epoch")
    max_epoch = int(trace["epoch"].max())
    print(f"[INIT] Loaded workload with {len(trace)} entries across {max_epoch + 1} epochs")

    framework = args.framework
    number_of_epoch = args.epoch
    node_properties: List[dict] = []  # leave empty; Helix tolerates this

    print(f"[INIT] Running {framework} for {number_of_epoch} epochs")

    # ---------- Metrics ----------
    cumulative_ttft = 0.0
    cumulative_carbon = 0.0
    cumulative_water = 0.0
    cumulative_energy = 0.0
    cumulative_total_energy = 0.0
    epoch_counter = 0

    marl_scheme_sums: Dict[str, Dict[str, float]] = {}

    if framework.lower() == "marl" and getattr(args, "train_marl", False):
        # ----------------------------------------------------------
        # Build a pool of epochs to train on
        # ----------------------------------------------------------
        all_epochs = sorted(grouped_trace.groups.keys())

        if number_of_epoch is not None and number_of_epoch > 0:
            max_train = min(number_of_epoch, len(all_epochs))
            train_epochs = all_epochs[:max_train]
        else:
            train_epochs = all_epochs

        print(
            f"[MARL TRAIN] Building training pool from {len(train_epochs)} epochs: "
            f"{int(train_epochs[0])} .. {int(train_epochs[-1])}"
        )

        # Concatenate all chosen epochs into one dataframe, keeping 'epoch'
        epoch_data = pd.concat(
            [grouped_trace.get_group(e).copy() for e in train_epochs],
            ignore_index=True,
        )

        # ----------------------------------------------------------
        # Apply the same scaling as the evaluation path
        # ----------------------------------------------------------
        epoch_data["time_index"] = (
                epoch_data["time_index"] * args.freq_scale
        ).clip(upper=899).astype(int)

        if args.token_scale != 1.0:
            epoch_data["num_tokens"] = (
                    epoch_data["num_tokens"] * args.token_scale
            ).round().astype(int)

        if args.count_scale > 1:
            epoch_data = pd.concat(
                [epoch_data] * args.count_scale,
                ignore_index=True,
            )

        # All requests arrive at t=0 in the rate-based simulator
        epoch_data["arrival_ms"] = 0

        # Use the first training epoch as the "representative" idx
        # for summary / perturbation purposes; the env will override
        # self.epoch_idx per episode during reset().
        train_epoch_idx = int(train_epochs[0])

        train_marl_constrained_profiles(
            epoch_data=epoch_data,
            epoch_idx=train_epoch_idx,
            num_datacenters=args.num_dcs,
            node_properties=node_properties,
            total_timesteps=args.marl_timesteps,
            num_envs=args.marl_num_envs,
            overwrite_existing=args.marl_overwrite,
            model_dir="trained_models/sb3_agents",
            spec_dir="sim_specs",
            epoch_length=900,
        )

        print("[MARL TRAIN] Completed training; exiting without running evaluation.")
        exit(0)


    # ---------- Framework Import ----------
    def get_framework(framework):
        fw = framework.lower()
        if fw == 'helix':
            from Helix import Helix;
            return Helix
        elif fw == 'nsga2':
            from NSGA2 import NSGA2;
            return NSGA2
        elif fw == 'perllm':
            from PerLLM import PerLLM;
            return PerLLM
        elif fw == 'splitwise':
            from Splitwise import Splitwise;
            return Splitwise
        elif fw == 'hybrid':
            from Hybrid_Scheduler_LLM import Hybrid_Scheduler_LLM;
            return Hybrid_Scheduler_LLM
        elif fw == 'marl':
            import MultiAgentRL;
            return MultiAgentRL
        else:
            raise ValueError(f"Framework '{framework}' not found")


    FW = get_framework(framework)

    # ---------- Epoch loop ----------
    for epoch_idx in range(number_of_epoch):
        if epoch_idx not in grouped_trace.groups:
            continue

        # Make a copy and apply scaling
        epoch_data = grouped_trace.get_group(epoch_idx).copy()

        # NOTE: request-mode ignores runtime position; the column is kept for legacy
        epoch_data["time_index"] = (epoch_data["time_index"] * args.freq_scale).clip(upper=899).astype(int)

        # Scale tokens
        if args.token_scale != 1.0:
            epoch_data["num_tokens"] = (epoch_data["num_tokens"] * args.token_scale).round().astype(int)

        # Replicate rows (count-scale)
        if args.count_scale > 1:
            epoch_data = pd.concat([epoch_data] * args.count_scale, ignore_index=True)

        # All requests arrive at t=0
        epoch_data["arrival_ms"] = 0

        # Summary (for logging only)
        epoch_summary = summarize_epoch_rate(epoch_data)
        epoch_counter += 1
        print(f"\n--- Epoch {epoch_idx} ({framework}) ---")
        print(epoch_summary.head())

        # --- Call framework (per-request) ---
        stats, results, leftovers = FW.milp_optimizer(
            epoch_data=epoch_data,
            epoch_idx=epoch_idx,
            node_properties=node_properties,
            epoch_summary={
                "node_types": [0, 1, 2, 3, 4, 5],
                # Datacenter ids used by the Splitwise algorithm to build DC index maps
                "datacenters": list(range(args.num_dcs)),
                # Rough token split used by Splitwise for prompt vs generation;
                # frameworks that do not use these fields will ignore them.
                "avg_input_tokens": int(max(1, epoch_data["num_tokens"].mean() * 0.7)),
                "avg_output_tokens": int(max(1, epoch_data["num_tokens"].mean() * 0.3)),
                # Spec directory for simulator initialization (used by all frameworks)
                "spec_dir": args.spec_dir,
                "epoch_length": 900,
                # Input/output fraction for Splitwise phase splitting
                "in_frac": 0.7,
                "out_frac": 0.3,
            }
        )

        print(f"\n--- Epoch {epoch_idx} Dispatched Requests (First 5) ---")
        if results and isinstance(results, list):
            # Convert to DF for pretty printing
            disp_df = pd.DataFrame(results)
            if "model" in disp_df.columns:
                # Group by the compound key to see the variants
                summary = disp_df.groupby("model")["tokens"].count().reset_index(name="count")
                print(summary)
            else:
                print(disp_df[["model", "target_dc", "tokens"]].head())
        elif results and isinstance(results, dict):
            # Handle MARL return format if it's a dict of profiles
            first_key = next(iter(results))
            print(f"Stats for profile '{first_key}': {results[first_key]}")

        # --- Aggregate results ---
        cumulative_ttft += float(stats.get("avg_ttft", stats.get("avg_ttft_sec", 0.0)))
        cumulative_carbon += float(stats.get("carbon_emissions", 0.0)) / 1000.0
        cumulative_water += float(stats.get("water_usage", 0.0)) / 100  # m³→L
        cumulative_energy += float(stats.get("energy_cost", 0.0))
        cumulative_total_energy += float(stats.get('total_energy', 0.0))

        if framework.lower() == "marl" and isinstance(results, dict) and results:
            for scheme_name, m in results.items():
                if not m:
                    continue
                ttft_val = float(m.get("avg_ttft", m.get("avg_ttft_sec", 0.0)))
                carbon_val = float(m.get("carbon_emissions", 0.0)) / 1000.0
                water_val = float(m.get("water_usage", 0.0)) / 100.0
                energy_cost = float(m.get("energy_cost", 0.0))
                total_energy_kwh = float(m.get("total_energy", m.get("energy_kwh", 0.0)))

                agg = marl_scheme_sums.setdefault(
                    scheme_name,
                    {
                        "ttft_sum": 0.0,
                        "carbon_sum": 0.0,
                        "water_sum": 0.0,
                        "energy_sum": 0.0,
                        "total_energy_sum": 0.0,
                        "epochs": 0,
                    },
                )
                agg["ttft_sum"] += ttft_val
                agg["carbon_sum"] += carbon_val
                agg["water_sum"] += water_val
                agg["energy_sum"] += energy_cost
                agg["total_energy_sum"] += total_energy_kwh
                agg["epochs"] += 1

        # --- Log epoch ---
        os.makedirs("LLM_Results", exist_ok=True)
        with open(f"LLM_Results/{framework}_epoch_{epoch_idx}.txt", "w") as f:
            for k, v in stats.items():
                f.write(f"{k}: {v}\n")

    # ---------- Final report ----------
    print("\n=== Final Report ===")
    print(f"Average TTFT (s): {cumulative_ttft / max(1, epoch_counter):.6f}")
    print(f"Carbon (kg): {cumulative_carbon:.3f}")
    print(f"Water (L): {cumulative_water:.3f}")
    print(f"Energy ($): {cumulative_energy:.3f}")
    print(f"Total Energy (kWh): {cumulative_total_energy:.3f}")

    out = f"LLM_Results/{framework}_final.txt"
    with open(out, "w") as f:
        f.write("=== Final Results ===\n")
        f.write(f"Epochs: {epoch_counter}\n")
        f.write(f"Average TTFT (s): {cumulative_ttft / max(1, epoch_counter):.6f}\n")
        f.write(f"Total Carbon (g): {cumulative_carbon:.3f}\n")
        f.write(f"Total Water (L): {cumulative_water:.3f}\n")
        f.write(f"Total Energy ($): {cumulative_energy:.3f}\n")
        f.write(f"Total Energy (kWh): {cumulative_energy:.3f}\n")

        if framework.lower() == "marl" and marl_scheme_sums:
            print("\n=== Per-scheme MARL Results ===")
            for scheme_name, agg in marl_scheme_sums.items():
                ep = max(1, agg["epochs"])
                avg_ttft = agg["ttft_sum"] / ep

                print(f"[{scheme_name}]")
                print(f"  Epochs: {ep}")
                print(f"  Average TTFT (s): {avg_ttft:.6f}")
                print(f"  Carbon (kg): {agg['carbon_sum']:.3f}")
                print(f"  Water (L): {agg['water_sum']:.3f}")
                print(f"  Energy ($): {agg['energy_sum']:.3f}")
                print(f"  Total Energy (kWh): {agg['total_energy_sum']:.3f}")

                scheme_path = f"LLM_Results/MARL_{scheme_name}_final.txt"
                with open(scheme_path, "w") as f:
                    f.write("=== Final Results (MARL scheme) ===\n")
                    f.write(f"Scheme: {scheme_name}\n")
                    f.write(f"Epochs: {ep}\n")
                    f.write(f"Average TTFT (s): {avg_ttft:.6f}\n")
                    f.write(f"Total Carbon (kg): {agg['carbon_sum']:.3f}\n")
                    f.write(f"Total Water (L): {agg['water_sum']:.3f}\n")
                    f.write(f"Total Energy ($): {agg['energy_sum']:.3f}\n")
                    f.write(f"Total Energy (kWh): {agg['total_energy_sum']:.3f}\n")

                print(f"[DONE] Wrote per-scheme summary for {scheme_name} to {scheme_path}")

    print(f"[DONE] Results written to {out}")