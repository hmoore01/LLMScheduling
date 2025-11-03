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

def write_epoch_stats(tag: str, epoch_index: int, stats_dict: dict, tag2: Optional[str] = None) -> None:
    outdir = "LLM_Results"
    os.makedirs(outdir, exist_ok=True)
    fname = f"{outdir}/{tag}{('_' + tag2) if tag2 else ''}_epoch_{epoch_index}.txt"
    with open(fname, "w") as f:
        f.write(f"Epoch {epoch_index}\n")
        for k, v in stats_dict.items():
            f.write(f"{k}: {v}\n")


def summarize_epoch_rate(epoch_data):
    total_duration = 900  # assuming each epoch is 900 seconds
    summary = {
        "llama7b_total": (epoch_data["model_type"] == "Llama7b").sum(),
        "llama70b_total": (epoch_data["model_type"] == "Llama70b").sum(),
    }
    summary["llama7b_rate"] = summary["llama7b_total"] / total_duration
    summary["llama70b_rate"] = summary["llama70b_total"] / total_duration
    return summary


def get_deterministic_perturbation(epoch_idx, error_rate):
    key = f"epoch_{epoch_idx}"
    hash_bytes = hashlib.sha256(key.encode()).digest()
    raw = int.from_bytes(hash_bytes[:4], 'big')  # first 4 bytes

    frac = (raw % 10**6) / 10**6  # → [0, 1)
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
    k = int(k); phase = int(phase)
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

def build_agent_specs(num_datacenters: int):
    # Helper caps for power-based hard constraints (same meaning as before)
    global_power_cap = 0.6 * float(num_datacenters)  # sum of power scalars across DCs per step
    per_dc_power_cap = 0.85

    agent_specs = {
        # ---- Single-objective (no constraints) ----
        "time_agent":   {"weights": {"ttft": 10}, "constraints": {}, "include_duals_in_obs": False},
        "carbon_agent": {"weights": {"carbon": 10}, "constraints": {}, "include_duals_in_obs": False},
        "water_agent":  {"weights": {"water": 10},  "constraints": {}, "include_duals_in_obs": False},
        "cost_agent":   {"weights": {"cost": 10},   "constraints": {}, "include_duals_in_obs": False},

        # ---- Practical, constrained profiles ----

        # 1) Green performance: prefer low latency, keep carbon under budget (episode window)
        #    (unchanged; this is always-on under the old schema)
        "green_perf": {
            "weights": {"ttft": 6, "carbon": 3, "cost": 1},
            "constraints": {
                "carbon": {"budget": 2.2e5, "scope": "global", "window": "episode", "hard": False, "budget_units": "raw"}
            },
            "lambda_lr": {"carbon": 5e-4},
            "include_duals_in_obs": True
        },

        # 2) Cost guard: fast service but constrained by energy cost budget (episode window)
        "cost_guard": {
            "weights": {"ttft": 7, "cost": 3},
            "constraints": {
                "energy_cost": {"budget": 120.0, "scope": "global", "window": "episode", "hard": False, "budget_units": "raw"}
            },
            "lambda_lr": {"energy_cost": 5e-4},
            "include_duals_in_obs": True
        },

        # 3) Water saver: prioritize performance with a water cap (episode window)
        "water_saver": {
            "weights": {"ttft": 7, "water": 3},
            "constraints": {
                "water_usage": {"budget": 2.0e4, "scope": "global", "window": "episode", "hard": False, "budget_units": "raw"}
            },
            "lambda_lr": {"water_usage": 5e-4},
            "include_duals_in_obs": True
        },

        # 4) Peak power guard (step window): enforce instantaneous power guardrails (hard)
        #    Base rule (always): hard per-step caps (unchanged)
        #    Plus: epoch-aware modifiers below (examples)
        "peak_power_guard": {
            "weights": {"ttft": 10},
            "constraints": {
                "per_dc_power_max": {"rule": "per_dc_power_max", "max": per_dc_power_cap, "scope": "global", "window": "step", "hard": True},
                "global_power_max_sum": {"rule": "global_power_max_sum", "max_sum": global_power_cap, "scope": "global", "window": "step", "hard": True},
                "per_dc_share_max_70b": {"rule": "per_dc_share_max_70b", "max": 0.5, "scope": "global", "window": "step", "hard": True}
            },
            "epoch_rules": [
                # (A) Looping “peak hours” daily: tighten the global sum cap during 5–8pm every 24-step cycle
                {
                    "when": {"type": "loop", "k": 24, "phase": 0, "residues": [17, 18, 19, 20]},
                    "merge": "override",
                    "constraints": {
                        "global_power_max_sum": {"rule": "global_power_max_sum", "max_sum": max(1.0, 0.8 * global_power_cap), "scope": "global", "window": "step", "hard": True}
                    }
                },
                # (B) Maintenance window for epochs 96..99: freeze per-DC to a very low ceiling (range schedule)
                {
                    "when": {"type": "range", "start": 96, "end": 99},
                    "merge": "override",
                    "constraints": {
                        "per_dc_power_max": {"rule": "per_dc_power_max", "max": 0.10, "scope": "global", "window": "step", "hard": True}
                    }
                },
                # (C) Specific epochs list: relax share cap (e.g., launch test waves) on chosen epochs only
                {
                    "when": {"type": "epochs", "list": [12, 36, 60]},
                    "merge": "override",
                    "constraints": {
                        "per_dc_share_max_70b": {"rule": "per_dc_share_max_70b", "max": 0.7, "scope": "global", "window": "step", "hard": True}
                    }
                },
            ],
            "include_duals_in_obs": False
        },
    }
    return agent_specs

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
                        choices=['Helix','NSGA2','PerLLM','Splitwise','Hybrid','MARL'])

    # Scaling
    parser.add_argument('--freq-scale', type=float, default=0.5)
    parser.add_argument('--token-scale', type=float, default=50.0)
    parser.add_argument('--count-scale', type=int, default=30)
    parser.add_argument('--error-rate', type=float, default=0.0)
    args = parser.parse_args()

    # ---------- Simple helper ----------
    def summarize_epoch_rate(df: pd.DataFrame):
        """Summarize total tokens per source_dc_id & model_type."""
        grp = df.groupby(["source_dc_id", "model_type"], as_index=False)["num_tokens"].sum()
        grp.rename(columns={"num_tokens": "tokens"}, inplace=True)
        return grp

    # ---------- Load workload ----------
    trace = pd.read_csv("simulator_ready_trace.csv")
    grouped_trace = trace.groupby("epoch")
    max_epoch = int(trace["epoch"].max())

    framework = args.framework
    number_of_epoch = args.epoch
    node_properties = []  # used for signature consistency

    print(f"[INIT] Running {framework} for {number_of_epoch} epochs")

    # ---------- Metrics ----------
    cumulative_ttft = 0.0
    cumulative_carbon = 0.0
    cumulative_water = 0.0
    cumulative_energy = 0.0
    epoch_counter = 0

    # ---------- Framework Import ----------
    def get_framework(framework):
        fw = framework.lower()
        if fw == 'helix':
            from Helix import Helix; return Helix
        elif fw == 'nsga2':
            from NSGA2 import NSGA2; return NSGA2
        elif fw == 'perllm':
            from PerLLM import PerLLM; return PerLLM
        elif fw == 'splitwise':
            from Splitwise import Splitwise; return Splitwise
        elif fw == 'hybrid':
            from Hybrid_Scheduler_LLM import Hybrid_Scheduler_LLM; return Hybrid_Scheduler_LLM
        elif fw == 'marl':
            import MultiAgentRL; return MultiAgentRL
        else:
            raise ValueError(f"Framework '{framework}' not found")

    FW = get_framework(framework)

    # ---------- Epoch loop ----------
    for epoch_idx in range(number_of_epoch):
        if epoch_idx not in grouped_trace.groups:
            continue
        epoch_data = grouped_trace.get_group(epoch_idx).copy()

        # --- Scaling ---
        epoch_data["time_index"] = (epoch_data["time_index"] * args.freq_scale).clip(upper=899).astype(int)
        epoch_data["num_tokens"] = (epoch_data["num_tokens"] * args.token_scale).astype(int)
        if args.count_scale > 1:
            epoch_data = pd.concat([epoch_data] * args.count_scale, ignore_index=True)

        # --- Summary ---
        epoch_summary = summarize_epoch_rate(epoch_data)
        epoch_counter += 1
        print(f"\n--- Epoch {epoch_idx} ({framework}) ---")
        print(epoch_summary.head())

        # --- Call framework ---
        stats, results, leftovers = FW.milp_optimizer(
            epoch_data=epoch_data,
            epoch_idx=epoch_idx,
            node_properties=node_properties,
            epoch_summary=epoch_summary
        )

        # --- Aggregate results ---
        cumulative_ttft += float(stats.get("avg_ttft", stats.get("avg_ttft_sec", 0.0)))
        cumulative_carbon += float(stats.get("carbon_emissions", 0.0)) * 1000.0  # kg→g
        cumulative_water += float(stats.get("water_usage", 0.0)) * 1000.0         # m³→L
        cumulative_energy += float(stats.get("energy_cost", 0.0))

        # --- Log epoch ---
        os.makedirs("LLM_Results", exist_ok=True)
        with open(f"LLM_Results/{framework}_epoch_{epoch_idx}.txt", "w") as f:
            for k, v in stats.items():
                f.write(f"{k}: {v}\n")

    # ---------- Final report ----------
    print("\n=== Final Report ===")
    print(f"Average TTFT (s): {cumulative_ttft / max(1, epoch_counter):.6f}")
    print(f"Carbon (g): {cumulative_carbon:.3f}")
    print(f"Water (L): {cumulative_water:.3f}")
    print(f"Energy ($): {cumulative_energy:.3f}")

    out = f"LLM_Results/{framework}_final.txt"
    with open(out, "w") as f:
        f.write("=== Final Results ===\n")
        f.write(f"Epochs: {epoch_counter}\n")
        f.write(f"Average TTFT (s): {cumulative_ttft / max(1, epoch_counter):.6f}\n")
        f.write(f"Total Carbon (g): {cumulative_carbon:.3f}\n")
        f.write(f"Total Water (L): {cumulative_water:.3f}\n")
        f.write(f"Total Energy ($): {cumulative_energy:.3f}\n")

    print(f"[DONE] Results written to {out}")





