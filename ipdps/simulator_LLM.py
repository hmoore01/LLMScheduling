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

def write_epoch_stats(framework, epoch_idx, stats, tag="balanced"):
    if not os.path.exists('LLM_Results/Epoch_Stats'):
        os.makedirs('LLM_Results/Epoch_Stats')

    filename = f'LLM_Results/Epoch_Stats/{framework}_{tag}_epoch_stats.csv'
    write_header = not os.path.exists(filename)

    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(["Epoch", "TTFT", "Carbon", "Water, Energy"])
        writer.writerow([
            epoch_idx,
            stats["avg_ttft"],
            stats["carbon_emissions"],
            stats["water_usage"],
            stats["energy_cost"]
        ])


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


# ---- epoch_constraints.py (or top of the MARL file) ----
from typing import Dict, Any, List, Optional, Callable, Union, Literal

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
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--laxity', type=int, help='deadline laxity', default=10)
    parser.add_argument('-s', '--slo', type=float, help='SLO constraint', default=0.25)
    parser.add_argument('-e', '--epoch', type=int, help='number of epochs', default=96)
    parser.add_argument('-t', '--time', type=int, help='decision time', default=110)
    parser.add_argument('-n', '--node', type=int, help='number of nodes', default=8)
    parser.add_argument('-d', '--duration', type=int, help='duration time', default=22)
    parser.add_argument('-r', '--request', type=int, help='number of requests', default=1)
    parser.add_argument('-f', '--framework', type=str, help='framework', default='Helix', choices=[
                        'SLO', 'Load', 'Ideal', 'Back', 'Hybrid', 'Score', 'Binary',
                        'Mscore', 'DSLO', 'Qtrain', 'Qtest', 'Search', 'MARL', 'Helix',
                        'Train_RL', 'SARL_Eval', 'Splitwise', 'Swarm', 'PerLLM', 'NSGA2'])

    #LLM specific args
    parser.add_argument('--freq-scale', type=float, default=0.5, help='Timeline compression factor')
    parser.add_argument('--token-scale', type=float, default=50.0, help='Multiply token size')
    parser.add_argument('--count-scale', type=float, default=30, help='Duplicate requests')
    parser.add_argument('--error-rate', type=float, default=0.0, help='Error rate')

    args = parser.parse_args()

    # Load and process the trace file
    trace = pd.read_csv("simulator_ready_trace.csv")
    grouped_trace = trace.groupby("epoch")
    max_epoch = trace["epoch"].max()

    ddl_laxity = args.laxity
    slo_constraint = args.slo
    number_of_epoch = args.epoch
    decision_time = args.time
    number_of_node = args.node
    time_of_duration = args.duration
    time_of_request = args.request
    framework = args.framework

    df = pd.read_csv("sim_specs/Node_Specs.csv")
    node_properties = []
    node_id_counter = 0
    for dc_id in range(12):
        nodes_per_type = 1000 // 6
        for node_type in range(6):
            for _ in range(nodes_per_type):
                gpu_type = "A100" if "A100" in df.iloc[node_type]["Node_Type"] else "H100"
                node_properties.append({
                    "node_id": node_id_counter,
                    "datacenter_id": dc_id,
                    "node_type": node_type,
                    "gpu_type": gpu_type
                })
                node_id_counter += 1


    print(f"Initialized with {number_of_node} nodes, {time_of_duration} duration, "
          f"{time_of_request} requests, framework: {framework}")

    DEPLOY_PROFILE = "green_perf"

    cumulative_carbon = 0.0
    cumulative_water = 0.0
    cumulative_ttft = 0.0
    cumulative_energy_costs = 0.0
    total_invocations = 0
    epoch_counter = 0
    ttft_carbon = 0.0
    ttft_water = 0.0
    ttft_ttft = 0.0
    ttft_energy = 0.0
    carbon_carbon = 0.0
    carbon_water = 0.0
    carbon_ttft = 0.0
    carbon_energy = 0.0
    water_carbon = 0.0
    water_ttft = 0.0
    water_water = 0.0
    water_energy = 0.0
    energy_ttft = 0.0
    energy_carbon = 0.0
    energy_water = 0.0
    energy_energy = 0.0
    epoch_summaries = []
    network_load_history = []
    leftover_pool = []
    if framework =="MARL":
        import MultiAgentRL
        from MultiAgentRL import ResourceEnv

        all_agents = list(ResourceEnv.agent_reward_weights.keys())
        cumulative_sums = {
            agent_id: {"carbon_emissions": 0, "water_usage": 0, "avg_ttft": 0, "energy_cost": 0}
            for agent_id in all_agents
        }

    for epoch_idx in range(0,1):
        if epoch_idx not in grouped_trace.groups:
            continue

        epoch_data = grouped_trace.get_group(epoch_idx)

        # Frequency scaling
        epoch_data["time_index"] = (epoch_data["time_index"] * args.freq_scale).clip(upper=899).astype(int)

        # Token count scaling
        epoch_data["num_tokens"] = (epoch_data["num_tokens"] * args.token_scale).astype(int)

        # Count duplication scaling
        if args.count_scale != 1.0:
            multiplier = int(args.count_scale)
            copies = [epoch_data.copy() for _ in range(multiplier - 1)]
            epoch_data = pd.concat([epoch_data] + copies, ignore_index=True)

        llama_7b = (epoch_data["model_type"] == "Llama7b").sum()
        llama_70b = (epoch_data["model_type"] == "Llama70b").sum()

        epoch_summary = summarize_epoch_rate(epoch_data)
        epoch_summary = apply_rate_error(epoch_summary, epoch_idx, args.error_rate)
        epoch_summaries.append(epoch_summary)
        epoch_summary_df = pd.DataFrame(epoch_summaries)
        epoch_summary_df.to_csv("epoch_summary.csv",index=False)
        print(epoch_summary)

        print(f"\n--- Epoch {epoch_idx} ---")
        print(f"Llama_8B: {llama_7b}, Llama_70B: {llama_70b}")
        epoch_counter += 1

        if framework == 'Train_RL':
            import MultiAgentRL
            import multiprocessing as mp

            if mp.get_start_method(allow_none=True) != "spawn":
                mp.set_start_method("spawn", force=True)

            agent_specs = build_agent_specs(num_datacenters=12)
            base_env_config = {
                "epoch_df": epoch_data,
                "epoch_summary": epoch_summary,
                "epoch_idx": epoch_idx,
                "node_properties": node_properties,
                "num_datacenters": 12,
                "max_steps": 50,
                "agent_specs": agent_specs,
            }

            MultiAgentRL.train_all_schemes(
                epoch_data, epoch_summary, epoch_idx, node_properties,
                agent_specs=agent_specs, num_datacenters=12,
                total_timesteps=10_000
            )

            leftovers = rollout_once_collect_leftovers(
                env_config=base_env_config,
                model_dir="trained_models/sb3_agents",
                profile_name=DEPLOY_PROFILE
            )
            if leftovers:
                print(f"[Epoch {epoch_idx}] Carrying over {len(leftovers)} leftover requests to next epoch.")
                # They already have time_index set to 0 inside the simulator; keep that.
                leftover_pool.extend(leftovers)

        if framework =="MARL":
            import MultiAgentRL
            from MultiAgentRL import ResourceEnv  # to get the agent list

            # === Prepare agent list ===
            all_agents = list(ResourceEnv.agent_reward_weights.keys())

            # === Run multi-agent inference ===
            metrics = MultiAgentRL.run_multiagent(epoch_data, epoch_summary, epoch_idx, node_properties)

            # === Process results per agent ===
            for agent_id in all_agents:
                stats = metrics.get(agent_id,
                                    {"carbon_emissions": 0, "water_usage": 0, "avg_ttft": 0, "energy_cost": 0})
                cumulative_sums[agent_id]["carbon_emissions"] += stats["carbon_emissions"]
                cumulative_sums[agent_id]["water_usage"] += stats["water_usage"]
                cumulative_sums[agent_id]["avg_ttft"] += stats["avg_ttft"]
                cumulative_sums[agent_id]["energy_cost"] += stats["energy_cost"]

                if "network_load" in stats:
                    net_load = stats["network_load"].copy()
                    net_load["epoch"] = epoch_idx
                    network_load_history.append(net_load)

                # Optionally write individual epoch CSVs (if needed for debugging)
                write_epoch_stats("RL", epoch_idx, stats, tag=agent_id.replace("_agent", ""))



        if framework == 'Helix':
            import Helix
            stats, results, leftover_requests = Helix.milp_optimizer(epoch_data, epoch_idx, node_properties, epoch_summary)
            cumulative_carbon += stats["carbon_emissions"]
            cumulative_water += stats["water_usage"]
            cumulative_ttft += stats["avg_ttft"]
            cumulative_energy_costs += stats["energy_cost"]
            total_invocations += len(results)

            write_epoch_stats("Helix", epoch_idx, stats)

            if "network_load" in stats:
                net_load = stats["network_load"].copy()
                net_load["epoch"] = epoch_idx
                network_load_history.append(net_load)


        elif framework == 'NSGA2':
            import NSGA2

            plans, stats_list, leftover_requests = NSGA2.nsga2_scheduler(epoch_data, epoch_idx, node_properties, epoch_summary)

            print(stats_list)
            stats = stats_list[0]
            print(f"Carbon Emissions:")
            print(stats["carbon_emissions"])
            cumulative_carbon += stats["carbon_emissions"]
            cumulative_water += stats["water_usage"]
            cumulative_ttft += stats["avg_ttft"]
            cumulative_energy_costs += stats["energy_cost"]
            total_invocations += len(epoch_data)

            if "network_load" in stats:
                net_load = stats["network_load"].copy()
                net_load["epoch"] = epoch_idx
                network_load_history.append(net_load)

            write_epoch_stats("NSGA2", epoch_idx, stats)

        elif framework == 'PerLLM':
            import PerLLM

            stats_list, results_list, leftover_requests = PerLLM.perllm_scheduler(epoch_data, epoch_idx, node_properties, epoch_summary)
            results = results_list[0]

            print(f"Carbon Emissions:")
            print(results["carbon_emissions"])
            cumulative_carbon += results["carbon_emissions"]
            cumulative_water += results["water_usage"]
            cumulative_ttft += results["avg_ttft"]
            cumulative_energy_costs += results["energy_cost"]
            total_invocations += len(epoch_data)

            if "network_load" in stats:
                net_load = stats["network_load"].copy()
                net_load["epoch"] = epoch_idx
                network_load_history.append(net_load)

            write_epoch_stats("PerLLM", epoch_idx, results)


        elif framework == 'Splitwise':
            import Splitwise

            stats, results, leftover_requests = Splitwise.splitwise_scheduler(epoch_data, epoch_idx, node_properties, epoch_summary)
            print(f"Carbon Emissions:")
            print(stats["carbon_emissions"])
            cumulative_carbon += stats["carbon_emissions"]
            cumulative_water += stats["water_usage"]
            cumulative_ttft += stats["avg_ttft"]
            cumulative_energy_costs += stats["energy_cost"]
            total_invocations += len(results)

            if "network_load" in stats:
                net_load = stats["network_load"].copy()
                net_load["epoch"] = epoch_idx
                network_load_history.append(net_load)

            write_epoch_stats("Splitwise", epoch_idx, stats)

        elif framework == 'Hybrid':
            import Hybrid_Scheduler_LLM

            pareto_population, pareto_objectives = Hybrid_Scheduler_LLM.hybrid_scheduler(
                epoch_data, epoch_idx, node_properties
            )


            def dict_to_vector(obj):
                return [
                    obj["avg_ttft"],
                    obj["carbon_emissions"],
                    obj["water_usage"],
                    obj["energy_cost"]
                ]


            best_ttft_idx = min(range(len(pareto_objectives)), key=lambda i: pareto_objectives[i]["avg_ttft"])
            best_carbon_idx = min(range(len(pareto_objectives)), key=lambda i: pareto_objectives[i]["carbon_emissions"])
            best_water_idx = min(range(len(pareto_objectives)), key=lambda i: pareto_objectives[i]["water_usage"])
            best_energy_idx =min(range(len(pareto_objectives)), key=lambda i: pareto_objectives[i]["energy_cost"])

            weights = [0.25, 0.25, 0.25, 0.25]


            def weighted_score(obj):
                v = dict_to_vector(obj)
                return v[0] * weights[0] + v[1] * weights[1] + v[2] * weights[2] + v[3] * weights[3]


            best_balanced_idx = min(range(len(pareto_objectives)), key=lambda i: weighted_score(pareto_objectives[i]))

            balanced_stats = pareto_objectives[best_balanced_idx]
            ttft_stats = pareto_objectives[best_ttft_idx]
            carbon_stats = pareto_objectives[best_carbon_idx]
            water_stats = pareto_objectives[best_water_idx]
            energy_stats = pareto_objectives[best_energy_idx]

            # Accumulate Balanced Plan Stats
            cumulative_carbon += balanced_stats["carbon_emissions"]
            cumulative_water += balanced_stats["water_usage"]
            cumulative_ttft += balanced_stats["avg_ttft"]
            cumulative_energy_costs += balanced_stats["energy_cost"]

            # Accumulate TTFT-Optimized Plan Stats
            ttft_carbon += ttft_stats["carbon_emissions"]
            ttft_water += ttft_stats["water_usage"]
            ttft_ttft += ttft_stats["avg_ttft"]
            ttft_energy += ttft_stats["energy_cost"]

            # Accumulate Carbon-Optimized Plan Stats
            carbon_carbon += carbon_stats["carbon_emissions"]
            carbon_water += carbon_stats["water_usage"]
            carbon_ttft += carbon_stats["avg_ttft"]
            carbon_energy += carbon_stats["energy_cost"]

            # Accumulate Water-Optimized Plan Stats
            water_carbon += water_stats["carbon_emissions"]
            water_water += water_stats["water_usage"]
            water_ttft += water_stats["avg_ttft"]
            water_energy += water_stats["energy_cost"]

            energy_carbon += energy_stats["carbon_emissions"]
            energy_water += energy_stats["water_usage"]
            energy_ttft += energy_stats["avg_ttft"]
            energy_energy += energy_stats["energy_cost"]

            write_epoch_stats("Hybrid", epoch_idx, balanced_stats, tag="balanced")
            write_epoch_stats("Hybrid", epoch_idx, ttft_stats, tag="ttft")
            write_epoch_stats("Hybrid", epoch_idx, carbon_stats, tag="carbon")
            write_epoch_stats("Hybrid", epoch_idx, water_stats, tag="water")
            write_epoch_stats("Hybrid", epoch_idx, energy_stats, tag="energy")

    # Final results
    # ave_violation_rate = np.mean(objective_arr[:, 0])
    # cumulative_carbon = np.sum(objective_arr[:, 1])
    # cumulative_water = np.sum(objective_arr[:, 2]) / 100
    print("\n=== Final Report ===")
    print(f"Average Time to first Token (s): {cumulative_ttft / epoch_counter}")
    print(f"Carbon (g): {cumulative_carbon}")
    print(f"Water (L): {cumulative_water}")
    if not os.path.exists('LLM_Results'):
        os.makedirs('LLM_Results')

    output_path = f'LLM_Results/{framework}_l{ddl_laxity}_n{number_of_node}_d{time_of_duration}_r{time_of_request}_e{number_of_epoch}.txt'

    with open(output_path, 'w') as f:
        # Write overall summary
        f.write("=== Overall Aggregate Results ===\n")
        f.write(
            f"Number of 15 minute epochs: {epoch_counter}\n"
            f"Average Time to First Token (s): {cumulative_ttft / epoch_counter:.4f}\n"
            f"Cumulative Carbon Emissions (g): {cumulative_carbon:.2f}\n"
            f"Cumulative Water Usage (L): {cumulative_water:.2f}\n"
            f"Cumulative Energy Costs ($): {cumulative_energy_costs:.2f}\n\n"
        )

        # If Hybrid or MARL, include all agent breakdowns
        if framework in ('Hybrid', 'MARL'):
            f.write("=== Per-Agent Cumulative Results ===\n")
            hybrid_outputs = {
                "sustainable_ttft": (ttft_ttft, ttft_carbon, ttft_water, ttft_energy),
                "sustainable_carbon": (carbon_ttft, carbon_carbon, carbon_water, carbon_energy),
                "sustainable_water": (water_ttft, water_carbon, water_water, water_energy),
                "sustainable_energy": (energy_ttft, energy_carbon, energy_water, energy_energy)
            }

            for agent_name, (ttft, carbon, water, energy) in hybrid_outputs.items():
                f.write(
                    f"[{agent_name}]\n"
                    f"  Average Time to First Token (s): {ttft / epoch_counter:.4f}\n"
                    f"  Cumulative Carbon Emissions (g): {carbon:.2f}\n"
                    f"  Cumulative Water Usage (L): {water:.2f}\n"
                    f"  Cumulative Energy Costs ($): {energy:.2f}\n\n"
                )

            # Also loop through any additional agents captured automatically
            for agent_id, totals in cumulative_sums.items():
                f.write(
                    f"[{agent_id}]\n"
                    f"  Average Time to First Token (s): {totals['avg_ttft'] / epoch_counter:.4f}\n"
                    f"  Cumulative Carbon Emissions (g): {totals['carbon_emissions']:.2f}\n"
                    f"  Cumulative Water Usage (L): {totals['water_usage']:.2f}\n"
                    f"  Cumulative Energy Costs ($): {totals['energy_cost']:.2f}\n\n"
                )


        if network_load_history:
            f.write("=== Coarse Network Load by Epoch ===\n")
            f.write(f"{'Epoch':<6} {'Avg Load':>10} {'Active (s)':>14} {'Capacity (s)':>16}\n")
            f.write("-" * 50 + "\n")
            for entry in network_load_history:
                f.write(f"{entry['epoch']:<6} {entry['avg_load_ratio'] * 100:>9.2f}% "
                        f"{entry['total_active_seconds']:>14} {entry['total_capacity_seconds']:>16}\n")
            f.write("\n")

            f.write("=== Detailed Datacenter Load by Epoch ===\n")
            for entry in network_load_history:
                epoch = entry["epoch"]
                per_dc = entry.get("per_datacenter", [])
                f.write(f"Epoch {epoch}:\n")
                f.write(f"{'DC ID':<8} {'Location':<15} {'Load %':>8} {'Active (s)':>12} {'Capacity (s)':>14}\n")
                f.write("-" * 60 + "\n")
                for dc in per_dc:
                    f.write(f"{dc['datacenter_id']:<8} {dc['location']:<15} "
                            f"{dc['load_ratio'] * 100:>7.2f}% "
                            f"{dc['active_seconds']:>12} "
                            f"{dc['capacity_seconds']:>14}\n")
                f.write("\n")


