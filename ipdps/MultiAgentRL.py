import gymnasium as gym
from gymnasium import spaces
import numpy as np
import copy
import os
import supersuit
from pettingzoo.utils.env import ParallelEnv
from stable_baselines3.a2c import MlpPolicy
from supersuit import pettingzoo_env_to_vec_env_v1, concat_vec_envs_v1, black_death_v3
from pettingzoo.utils import parallel_to_aec
from pettingzoo.utils.wrappers import BaseWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from simulation import LLM_Simulator
from multiprocessing import Process, Queue
from typing import Dict, Any, List, Tuple, Optional

import pandas as pd

import joblib

NUM_NODE_TYPES = 6
latency_df = pd.read_csv("/mnt/c/Users/epiclab/Desktop/HPDC-LLM-v4/HPDC-LLM-v3/ipdps/sim_specs/Geo_Latencies.csv", index_col=0)
METRIC_ORDER = ["ttft", "carbon", "water", "cost"]

# estimator_model = joblib.load("checkpoint_epoch_1050.pkl")

from train_predictor import STATIC_FEATURES

def extract_features(epoch_df, schedule_plan, power_plan, num_datacenters=12):
    total_requests = len(schedule_plan)
    avg_tokens = epoch_df["num_tokens"].mean()
    max_tokens = epoch_df["num_tokens"].max()
    pct_llama70b = (epoch_df["model_type"] == "Llama70b").mean()
    avg_batch = epoch_df["batch_size"].mean()
    std_batch = epoch_df["batch_size"].std()

    # Token-to-batch ratio
    token_batch_ratio = avg_tokens / avg_batch if avg_batch > 0 else 0

    # Datacenter usage features
    dc_ids = [req["target_dc_id"] for req in schedule_plan]
    dc_counts = pd.Series(dc_ids).value_counts(normalize=True)
    active_dc_ratio = len(dc_counts) / num_datacenters
    std_dc_usage = dc_counts.std() if len(dc_counts) > 1 else 0.0

    # Average latency between source and target DCs
    latencies = []
    for req, sched in zip(epoch_df.to_dict("records"), schedule_plan):
        src = int(req["source_dc_id"])
        tgt = int(sched["target_dc_id"])
        lat = latency_df.iloc[src, tgt]
        if lat > 0:
            latencies.append(lat)
    avg_route_latency = np.mean(latencies) if latencies else 0.0

    # Power plan features
    total_idle = sum(1 for dc in power_plan.values() for state in dc.values() if state == "Idle")
    total_off = sum(1 for dc in power_plan.values() for state in dc.values() if state == "Off")
    num_active_nodes = num_datacenters * NUM_NODE_TYPES - total_off
    peak_batch = epoch_df["batch_size"].max()

    # Weighted model intensity
    model_weights = epoch_df["model_type"].apply(lambda x: 2 if x == "Llama70b" else 1)
    model_load_intensity = model_weights.mean()

    dynamic = [
        total_requests, avg_tokens, max_tokens, pct_llama70b,
        avg_batch, std_batch, active_dc_ratio, std_dc_usage, total_idle,
        num_active_nodes, peak_batch, token_batch_ratio,
        avg_route_latency, model_load_intensity
    ]

    return dynamic + STATIC_FEATURES


 # def estimate_metrics(epoch_df, schedule_plan, power_plan):
 #   features = extract_features(epoch_df, schedule_plan, power_plan)
 #   prediction = estimator_model.predict([features])[0]
 #   return {
 #       "avg_ttft": prediction[0],
 #       "energy_cost": prediction[1],
 #       "carbon_emissions": prediction[2],
 #       "water_usage": prediction[3],
 #       "total_energy": prediction[4]
 #   }


class ResourceEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "ResourceEnv"}

    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        # === Core config from your existing setup ===
        self.epoch_df = config["epoch_df"]
        self.node_properties = config["node_properties"]
        self.epoch_idx = config["epoch_idx"]
        self.NUM_DATACENTERS = config["num_datacenters"]
        self.max_steps = config.get("max_steps", 1)
        self.epoch_summary = config["epoch_summary"]

        # === New: profile-driven reward/constraints spec ===
        # config must include:
        #   - "active_agent_profile": str (key in agent_specs)
        #   - "agent_specs": dict(profile_name -> spec dict)
        self.active_agent_profile = config["active_agent_profile"]
        self.agent_specs: Dict[str, Dict[str, Any]] = config["agent_specs"]
        assert self.active_agent_profile in self.agent_specs, \
            f"Profile '{self.active_agent_profile}' not found in agent_specs"

        self.profile = self.agent_specs[self.active_agent_profile]

        # Normalized reward weights (or empty dict if none given)
        self.reward_weights: Dict[str, float] = self._normalize_weights(
            self.profile.get("weights", None)
        )

        # Constraints setup
        self.constraints: Dict[str, Dict[str, Any]] = self.profile.get("constraints", {})
        self.include_duals_in_obs: bool = bool(self.profile.get("include_duals_in_obs", True))
        self.lambda_lr: Dict[str, float] = self.profile.get("lambda_lr", {})

        # Dual variables (Lagrange multipliers)
        self.duals: Dict[str, float] = {}
        lambda_init = self.profile.get("lambda_init", {})
        for cname in self.constraints.keys():
            self.duals[cname] = float(lambda_init.get(cname, 0.0))

        # === Agents (one per DC) ===
        self.agents = [f"dc_{i}" for i in range(self.NUM_DATACENTERS)]
        self.possible_agents = self.agents[:]
        self.current_step = 0
        self.num_power_controls = self.NUM_DATACENTERS * 6  # 6 node types per DC

        # === Workload snapshot ===
        self.llama7b_total = int(self.epoch_summary.get("llama7b_total", 0))
        self.llama70b_total = int(self.epoch_summary.get("llama70b_total", 0))

        # === Spaces ===
        self.action_spaces: Dict[str, spaces.Space] = {}
        self.observation_spaces: Dict[str, spaces.Space] = {}
        self._build_spaces()

        # === Runtime state tracking ===
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.dones["__all__"] = False
        self.infos = {agent: {} for agent in self.agents}
        self.current_actions = {}
        self.collected_plans = {}
        self.render_mode = "human"
        self.power_plan: Dict[str, Any] = {}
        self.schedule_plan: Dict[str, Any] = {}

        # Metric normalization trackers
        self.metric_max_tracker = {
            "carbon": 1.0,
            "ttft": 1.0,
            "water": 1.0,
            "cost": 1.0,
        }

        # Episodic budget tracking
        self.episodic_cost_totals: Dict[str, float] = {k: 0.0 for k in self.constraints.keys()}
        self.episodic_cost_counts: Dict[str, int] = {k: 0 for k in self.constraints.keys()}
        # print(f"[ENV] Current agents: {self.agents} (len={len(self.agents)})", flush=True)

    def _build_spaces(self) -> None:
        # --- Action space (shared shape for all DC agents) ---
        act_space = spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)
        self.action_spaces = {agent: act_space for agent in self.agents}

        # --- Observation size computation ---
        base_obs_size = 8 + len(self.agents)

        constraint_extra = 0
        self._obs_has_duals = bool(self.include_duals_in_obs and len(self.constraints) > 0)
        self._obs_has_headroom = False

        if self._obs_has_duals:
            # one slot per constrained metric for λ (dual)
            constraint_extra += len(self.constraints)

            # add headroom slots for episodic-window constraints
            episodic_cnt = sum(1 for c in self.constraints.values()
                               if c.get("window", "episode") == "episode")
            if episodic_cnt > 0:
                self._obs_has_headroom = True
                constraint_extra += episodic_cnt

        obs_size = base_obs_size + constraint_extra

        obs_space = spaces.Box(low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32)
        self.observation_spaces = {agent: obs_space for agent in self.agents}

    @staticmethod
    def _normalize_weights(weights: Optional[Dict[str, float]]) -> Dict[str, float]:
        if not weights:
            return {}

        total = sum(float(weights.get(m, 0.0)) for m in METRIC_ORDER)
        if total <= 0:
            # No positive weights, treat as equal 10 / num_metrics
            equal_weight = 10.0 / len(METRIC_ORDER)
            return {m: equal_weight for m in METRIC_ORDER}

        scale_factor = 10.0 / total
        return {m: float(weights.get(m, 0.0)) * scale_factor for m in METRIC_ORDER}

    def _strong_schedule(self, profile_name: Optional[str] = None) -> np.ndarray:

        persona = profile_name or self.active_agent_profile

        def _base_plan_for(base: str) -> np.ndarray:
            dist_7b = np.ones(self.NUM_DATACENTERS, dtype=np.float32)
            dist_70b = np.ones(self.NUM_DATACENTERS, dtype=np.float32)

            if base == "carbon_agent":
                # Lower carbon intensity → higher weight
                carbon_intensity = np.array([
                    343.27, 505.88, 220.25, 205.31, 21.00, 444.15,
                    230.33, 230.33, 350.40, 332.12, 516.20, 549.72
                ], dtype=np.float32)
                inverse = 1.0 / (carbon_intensity + 1e-6)
                dist_7b = inverse / inverse.sum()
                dist_70b = dist_7b.copy()

            elif base == "water_agent":
                water_intensity = np.array([
                    0.005, 0.012, 0.021, 0.037, 0.042, 0.043,
                    0.064, 0.080, 0.101, 0.107, 0.120, 0.160
                ], dtype=np.float32)
                inverse = 1.0 / (water_intensity + 1e-6)
                dist_7b = inverse / inverse.sum()
                dist_70b = dist_7b.copy()

            elif base == "cost_agent":
                avg_energy_cost = np.array([
                    0.128777, 0.114167, 0.271030, 0.130333, 0.054000,
                    0.062917, 0.112918, 0.151667, 0.159167, 0.069625,
                    0.155417, 0.157000
                ], dtype=np.float32)
                inverse = 1.0 / (avg_energy_cost + 1e-6)
                dist_7b = inverse / inverse.sum()
                dist_70b = dist_7b.copy()

            elif base == "time_agent":
                # Uniform preference (or plug a latency-aware prior if you have one)
                dist_7b[:] = 1.0 / self.NUM_DATACENTERS
                dist_70b[:] = 1.0 / self.NUM_DATACENTERS

            else:
                # Default to uniform if unknown base is requested
                dist_7b[:] = 1.0 / self.NUM_DATACENTERS
                dist_70b[:] = 1.0 / self.NUM_DATACENTERS

            return np.concatenate([dist_7b, dist_70b], dtype=np.float32)

        # If persona is one of the base agents, just return that base plan.
        if persona in {"carbon_agent", "water_agent", "cost_agent", "time_agent"}:
            return _base_plan_for(persona)

        # Otherwise, build a mixed plan using the normalized weights (sum = 10).
        # Map metrics → base personas.
        metric_to_base = {
            "carbon": "carbon_agent",
            "ttft": "time_agent",
            "water": "water_agent",
            "cost": "cost_agent",
        }

        # If no weights provided ({}), fall back to time_agent (or choose a default you prefer).
        if not self.reward_weights:
            return _base_plan_for("time_agent")

        combined_7b = np.zeros(self.NUM_DATACENTERS, dtype=np.float32)
        combined_70b = np.zeros(self.NUM_DATACENTERS, dtype=np.float32)

        # Linearly combine base plans according to weights (they already sum to 10).
        weight_sum = 0.0
        for metric, w in self.reward_weights.items():
            if w <= 0.0:
                continue
            base = metric_to_base.get(metric)
            if base is None:
                continue
            base_plan = _base_plan_for(base)
            base_7b = base_plan[:self.NUM_DATACENTERS]
            base_70b = base_plan[self.NUM_DATACENTERS:]
            combined_7b += float(w) * base_7b
            combined_70b += float(w) * base_70b
            weight_sum += float(w)

        # Normalize; if degenerate, default to uniform
        if weight_sum <= 0.0 or combined_7b.sum() <= 0.0 or combined_70b.sum() <= 0.0:
            dist_7b = np.full(self.NUM_DATACENTERS, 1.0 / self.NUM_DATACENTERS, dtype=np.float32)
            dist_70b = dist_7b.copy()
        else:
            dist_7b = combined_7b / combined_7b.sum()
            dist_70b = combined_70b / combined_70b.sum()

        return np.concatenate([dist_7b, dist_70b], dtype=np.float32)

    def _strong_power(self, profile_name: Optional[str] = None) -> np.ndarray:
        persona = profile_name or self.active_agent_profile

        # Number of discrete power patterns (e.g., 0..7 scaled to 0..1)
        pattern_levels = 8
        pattern_indices = {
            "carbon_agent": 1,  # Only 2_A100s
            "water_agent": 1,  # Only 2_A100s
            "cost_agent": 1,  # Only 2_A100s
            "time_agent": 6  # Only 8_H100s (second strongest)
        }

        # Base agent case
        if persona in pattern_indices:
            idx = pattern_indices[persona]
            return np.array([idx / (pattern_levels - 1)], dtype=np.float32)

        # Mixed agent case — use the normalized weights
        if not self.reward_weights:
            # No weights? Default to time_agent lever
            return self._strong_power("time_agent")

        # Metric → base agent mapping
        metric_to_base = {
            "carbon": "carbon_agent",
            "ttft": "time_agent",
            "water": "water_agent",
            "cost": "cost_agent",
        }

        # Get base lever values in the same metric order
        levers = []
        weights = []
        for metric, w in self.reward_weights.items():
            if w <= 0.0:
                continue
            base_agent = metric_to_base.get(metric)
            if not base_agent:
                continue
            base_val = self._strong_power(base_agent)[0]  # recursive call for base
            levers.append(base_val)
            weights.append(w)

        if not levers:
            return self._strong_power("time_agent")

        # Weighted average (weights already sum to 10 in your setup)
        mixed_lever = np.dot(weights, levers) / sum(weights)
        return np.array([mixed_lever], dtype=np.float32)

    def _default_schedule(self, agent):
        return self._strong_schedule(agent)

    def _default_power(self, agent):
        return self._strong_power(agent)

    def get_default_action(self, agent: str) -> np.ndarray:
        # Which agent index?
        try:
            dc_idx = int(agent.split("_")[1])
        except Exception:
            dc_idx = 0

        # Global prior distributions for 7B/70B
        prior = self._strong_schedule(self.active_agent_profile)  # shape = 2 * NUM_DATACENTERS
        n = self.NUM_DATACENTERS
        dist_7b = prior[:n]
        dist_70b = prior[n:]

        # Power prior
        power_scalar = float(self._strong_power(self.active_agent_profile)[0])

        # Return within Box(low=0, high=1)
        a0 = float(np.clip(dist_7b[dc_idx], 0.0, 1.0))
        a1 = float(np.clip(dist_70b[dc_idx], 0.0, 1.0))
        a2 = float(np.clip(power_scalar, 0.0, 1.0))
        return np.array([a0, a1, a2], dtype=np.float32)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        # --- Core episode state ---
        self.current_step = 0
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.dones["__all__"] = False
        self.infos = {agent: {} for agent in self.agents}
        self.collected_plans = {}
        self.power_plan = {}
        self.schedule_plan = {}

        # --- Metric normalization trackers (per-episode reset) ---
        self.metric_max_tracker = {
            "carbon": 1.0,
            "ttft": 1.0,
            "water": 1.0,
            "cost": 1.0,
        }

        # --- Constraint episodic accumulators ---
        for k in self.episodic_cost_totals.keys():
            self.episodic_cost_totals[k] = 0.0
            self.episodic_cost_counts[k] = 0

        # --- Default actions per agent ---
        # Prefer user's helper if present; otherwise fall back to strong schedule/power prior
        if hasattr(self, "get_default_action") and callable(getattr(self, "get_default_action")):
            self.current_actions = {agent: self.get_default_action(agent) for agent in self.agents}
        else:
            # Fallback: derive from strong schedule (global DC shares) + strong power (scalar)
            prior = self._strong_schedule(self.active_agent_profile)  # [dist_7b..., dist_70b...]
            dist_7b = prior[:self.NUM_DATACENTERS]
            dist_70b = prior[self.NUM_DATACENTERS:]
            power_scalar = float(self._strong_power(self.active_agent_profile)[0])
            self.current_actions = {
                f"dc_{i}": np.array([float(dist_7b[i]), float(dist_70b[i]), power_scalar], dtype=np.float32)
                for i in range(self.NUM_DATACENTERS)
            }

        # --- Build observations with validation ---
        obs: Dict[str, np.ndarray] = {}
        shapes: Dict[str, Tuple[int, ...]] = {}
        expected_shape: Optional[Tuple[int, ...]] = None

        for agent in self.agents:
            # Use provided observe(...) if the project already has it; else use our builder
            if hasattr(self, "observe") and callable(getattr(self, "observe")):
                o = self.observe(agent)  # existing user method
            else:
                o = self._build_observation(agent)  # new builder
            obs[agent] = o
            shapes[agent] = o.shape
            if expected_shape is None:
                expected_shape = o.shape
            elif o.shape != expected_shape:
                print(f"[reset ⚠️] Observation shape mismatch for {agent}: got {o.shape}, expected {expected_shape}",
                      flush=True)

        if any(shape != expected_shape for shape in shapes.values()):
            print(f"[reset ⚠️] Inconsistent observation shapes at reset: {shapes}", flush=True)

        info = {agent: {} for agent in self.agents}
        return obs, info

    def _run_metrics_backend(self):
        metrics, results, leftover_requests_out = LLM_Simulator(
            self.epoch_idx,
            self.epoch_df,
            self.schedule_plan,
            self.power_plan,
        )
        self._last_metrics = dict(metrics)  # cache for callbacks
        self._last_results = results
        self._last_leftovers = leftover_requests_out
        return metrics, results, leftover_requests_out

    def get_last_metrics(self):
        return getattr(self, "_last_metrics", None)

    def get_last_leftovers(self):
        return getattr(self, "_last_leftovers", None)


    def step(self, actions):
        import time
        assert set(actions.keys()) == set(self.agents), (
            f"[step ❌] Agent mismatch in actions: {list(actions.keys())} vs expected {self.agents}"
        )
        self.current_step += 1
        agents = list(self.agents)

        # --- Parse & normalize cross-agent distributions (softmax over DCs) ---
        raw = {a: np.asarray(actions[a], dtype=np.float32).copy() for a in agents}

        def _softmax(vec):
            v = np.clip(np.asarray(vec, dtype=np.float64), -50.0, 50.0)
            v -= np.max(v)
            ex = np.exp(v)
            s = ex.sum()
            return (ex / s) if (s > 0 and np.isfinite(s)) else np.full_like(ex, 1.0 / len(ex))

        logits_7b = [raw[a][0] for a in agents]
        logits_70b = [raw[a][1] for a in agents]
        dist_7b = _softmax(logits_7b).astype(np.float32)  # sum(dist_7b)=1
        dist_70b = _softmax(logits_70b).astype(np.float32)  # sum(dist_70b)=1
        power_scalars = [float(np.clip(raw[a][2], 0.0, 1.0)) for a in agents]

        normalized_actions = {
            a: np.array([dist_7b[i], dist_70b[i], power_scalars[i]], dtype=np.float32)
            for i, a in enumerate(agents)
        }

        # --- Safety layer (hard step constraints) ---
        projected_actions, projection_flags = self._apply_safety_layer(normalized_actions)

        # --- Build schedule plan from distributions ---
        df_7b = self.epoch_df[self.epoch_df['model_type'] == 'Llama7b']
        df_70b = self.epoch_df[self.epoch_df['model_type'] == 'Llama70b']
        self.schedule_plan = self._build_schedule_plan(df_7b, df_70b, dist_7b, dist_70b)

        # --- Build power plan from scalar lever -> discrete node-type pattern ---
        power_patterns = [
            [0, 0, 0, 0, 0, 0],  # All Off
            [0, 0, 0, 0, 1, 0],  # Only 2_A100s
            [0, 0, 0, 0, 0, 1],  # Only 2_H100s
            [0, 0, 1, 0, 0, 0],  # Only 4_A100s
            [0, 0, 0, 1, 0, 0],  # Only 4_H100s
            [1, 0, 0, 0, 0, 0],  # Only 8_A100s
            [0, 1, 0, 0, 0, 0],  # Only 8_H100s
            [1, 1, 1, 1, 1, 1],  # All On
        ]
        num_patterns = len(power_patterns)
        self.power_plan = {}
        for i, agent in enumerate(agents):
            dc_id = int(agent.split("_")[1])
            lever_val = float(np.clip(projected_actions[agent][2], 0.0, 1.0))
            idx = min(int(lever_val * num_patterns), num_patterns - 1)
            node_pattern = power_patterns[idx]
            self.power_plan[dc_id] = {
                node_type: ("On" if on else "Off") for node_type, on in enumerate(node_pattern)
            }

        # --- Run simulator & expose ALL raw metrics ---
        t0 = time.time()
        metrics, results, leftover_requests_out = self._run_metrics_backend()
        metric_time = time.time() - t0

        # Raw metrics (domain units)
        ttft = float(metrics.get("avg_ttft", 0.0))
        cost = float(metrics.get("energy_cost", 0.0))
        carbon = float(metrics.get("carbon_emissions", 0.0))
        water = float(metrics.get("water_usage", 0.0))
        total_energy = float(metrics.get("total_energy", 0.0))
        network_load, network_load_detail = self._extract_network_load(metrics)

        # --- Update normalizers (invert-lower-is-better for reward view) ---
        self.metric_max_tracker["ttft"] = max(self.metric_max_tracker["ttft"], ttft)
        self.metric_max_tracker["cost"] = max(self.metric_max_tracker["cost"], cost)
        self.metric_max_tracker["carbon"] = max(self.metric_max_tracker["carbon"], carbon)
        self.metric_max_tracker["water"] = max(self.metric_max_tracker["water"], water)
        # Optional extras for logging
        if "total_energy" not in self.metric_max_tracker:
            self.metric_max_tracker["total_energy"] = 1.0
        if "network_load" not in self.metric_max_tracker:
            self.metric_max_tracker["network_load"] = 1.0
        self.metric_max_tracker["total_energy"] = max(self.metric_max_tracker["total_energy"], total_energy)
        self.metric_max_tracker["network_load"] = max(self.metric_max_tracker["network_load"], network_load)

        def inv_norm(x, key):
            denom = max(self.metric_max_tracker[key], 1e-12)
            return float(np.clip(1.0 - (x / denom), 0.0, 1.0))

        global_norm = {
            "ttft": inv_norm(ttft, "ttft"),
            "carbon": inv_norm(carbon, "carbon"),
            "water": inv_norm(water, "water"),
            "cost": inv_norm(cost, "cost"),
            # extras if you later want them in reward:
            "total_energy": inv_norm(total_energy, "total_energy"),
            "network_load": inv_norm(network_load, "network_load"),
        }

        # --- Reward: weights dict (sum=10), "good = high" normalized metrics ---
        rw = 0.0
        for m, w in self.reward_weights.items():
            rw += float(w) * float(global_norm.get(m, 0.0))

        # --- Constraints: per-step costs + episodic accumulation ---
        if hasattr(self, "_compute_constraint_costs"):
            costs_step = self._compute_constraint_costs({
                "global": {
                    "avg_ttft": ttft,
                    "energy_cost": cost,
                    "carbon_emissions": carbon,
                    "water_usage": water,
                    "total_energy": total_energy,
                    "network_load": network_load,
                },
                "global_norm": global_norm,
            })
        else:
            costs_step = {}

        for cname, cdef in self.constraints.items():
            if cdef.get("window", "episode") == "episode":
                self.episodic_cost_totals[cname] += float(costs_step.get(cname, 0.0))
                self.episodic_cost_counts[cname] += 1

        # --- Collected plans (store all RAW metrics + leftovers) ---
        self.collected_plans[self.current_step] = {
            agent: {
                "metrics_raw": {
                    "avg_ttft": ttft,
                    "energy_cost": cost,
                    "carbon_emissions": carbon,
                    "water_usage": water,
                    "total_energy": total_energy,
                    "network_load": network_load,
                },
                "metrics_normalized": global_norm,
                "schedule_plan": self.schedule_plan,
                "power_plan": self.power_plan,
                "metric_time": metric_time,
                "projected": bool(projection_flags.get(agent, False)),
                "leftovers": leftover_requests_out,
                "results_sample_size": len(self._last_results) if hasattr(self, "_last_results") else None,
            } for agent in agents
        }

        # --- Observations / rewards / dones / infos ---
        if hasattr(self, "observe") and callable(getattr(self, "observe")):
            obs = {agent: self.observe(agent) for agent in agents}
        else:
            obs = {agent: self._build_observation(agent) for agent in agents}

        rewards = {agent: float(rw) for agent in agents}
        done_now = (self.current_step >= self.max_steps)
        terminations = {agent: done_now for agent in agents}
        terminations["__all__"] = done_now
        truncations = {agent: False for agent in agents}
        truncations["__all__"] = False

        if hasattr(self, "_current_constraint_status"):
            constraints_status = self._current_constraint_status()
        else:
            constraints_status = {}

        infos = {
            agent: {
                "metrics": {
                    "raw": {
                        "avg_ttft": ttft,
                        "energy_cost": cost,
                        "carbon_emissions": carbon,
                        "water_usage": water,
                        "total_energy": total_energy,
                        "network_load": network_load,
                    },
                    "norm": global_norm,
                },
                "costs": costs_step,
                "constraints": constraints_status,
                "duals": getattr(self, "duals", {}).copy(),
                "projected": bool(projection_flags.get(agent, False)),
                "leftovers": {
                    "count": len(self._last_leftovers) if hasattr(self, "_last_leftovers") else 0
                },
                "step_idx": self.current_step,
                "metric_time": metric_time,
            } for agent in agents
        }

        if done_now and hasattr(self, "_update_duals_after_episode"):
            self._update_duals_after_episode()

        self.rewards = rewards
        self.dones = terminations.copy()
        self.infos = infos
        return obs, rewards, terminations, truncations, infos

    def _apply_safety_layer(self, actions):
        # Fast path: nothing to enforce
        has_hard_step = any(
            c.get("hard", False) and c.get("window", "episode") == "step"
            for c in self.constraints.values()
        )
        if not has_hard_step:
            return actions, {a: False for a in self.agents}

        import numpy as np

        agents = list(self.agents)
        n = self.NUM_DATACENTERS

        # Extract current distributions and power scalars
        dist7b = np.array([float(actions[a][0]) for a in agents], dtype=np.float64)
        dist70b = np.array([float(actions[a][1]) for a in agents], dtype=np.float64)
        power = np.array([float(np.clip(actions[a][2], 0.0, 1.0)) for a in agents], dtype=np.float64)

        # Helpers
        def _mask_and_renorm(dist, mask):
            dist = np.where(mask, dist, 0.0)
            s = dist.sum()
            if s > 1e-12:
                return dist / s, True
            # If all mass was masked out, fallback to uniform over allowed entries
            allowed = np.count_nonzero(mask)
            if allowed == 0:
                # No feasible DCs -> return original (will be caught by caller)
                return dist, False
            out = np.zeros_like(dist)
            out[mask] = 1.0 / float(allowed)
            return out, True

        def _cap_simplex_with_upper_bounds(p, ub):
            p = np.asarray(p, dtype=np.float64).clip(0.0, None)
            ub = np.asarray(ub, dtype=np.float64).clip(0.0, 1.0)

            if ub.sum() < 1.0 - 1e-12:
                # Infeasible caps; best we can do is saturate the caps and report infeasible
                return ub.copy(), False

            # Iterative water-filling under upper bounds
            x = np.minimum(p, ub)
            total = x.sum()
            if abs(total - 1.0) < 1e-9:
                return x, True
            if total > 1.0:
                # Scale down proportionally, then re-cap to ensure x<=ub
                x = x * (1.0 / total)
                x = np.minimum(x, ub)
                # Final renorm (there is room because sum(ub)>=1)
                s = x.sum()
                if s < 1.0 - 1e-12:
                    # Distribute deficit to coordinates with slack
                    for _ in range(5):  # a few passes usually suffice
                        slack = ub - x
                        room = slack.sum()
                        if room <= 1e-12:
                            break
                        add = min(1.0 - s, room)
                        # Proportional to slack
                        x += (slack / max(room, 1e-12)) * add
                        s = x.sum()
                        if abs(s - 1.0) < 1e-9:
                            break
                else:
                    x /= s
                return x, True
            else:
                # We have deficit; add mass in proportion to remaining headroom (ub - x)
                for _ in range(10):
                    s = x.sum()
                    if abs(s - 1.0) < 1e-9:
                        break
                    deficit = 1.0 - s
                    slack = ub - x
                    room = slack.sum()
                    if room <= 1e-12:
                        # No room left; fallback to proportional renorm (shouldn't happen if sum(ub)>=1)
                        x = x / max(x.sum(), 1e-12)
                        break
                    x += (slack / room) * deficit
                    # ensure we don't exceed ub due to numerical issues
                    x = np.minimum(x, ub)
                # final tiny renorm
                x = x / max(x.sum(), 1e-12)
                return x, True

        # Defaults: everything allowed, no caps
        mask_7b = np.ones(n, dtype=bool)
        mask_70b = np.ones(n, dtype=bool)
        cap_7b = np.ones(n, dtype=np.float64)  # per-DC upper bound on 7B share
        cap_70b = np.ones(n, dtype=np.float64)  # per-DC upper bound on 70B share
        per_dc_power_max = 1.0
        global_power_max_sum = None

        # Parse hard step constraints
        for cname, cdef in self.constraints.items():
            if not (cdef.get("hard", False) and cdef.get("window", "episode") == "step"):
                continue
            rule = cdef.get("rule", None)

            if rule == "mask_7b":
                allowed = np.array(cdef.get("allowed_dcs", []), dtype=int)
                m = np.zeros(n, dtype=bool)
                m[np.clip(allowed, 0, n - 1)] = True
                mask_7b &= m
            elif rule == "mask_70b":
                allowed = np.array(cdef.get("allowed_dcs", []), dtype=int)
                m = np.zeros(n, dtype=bool)
                m[np.clip(allowed, 0, n - 1)] = True
                mask_70b &= m
            elif rule == "per_dc_share_max_7b":
                cap = float(cdef.get("max", 1.0))
                cap_7b = np.minimum(cap_7b, cap)
            elif rule == "per_dc_share_max_70b":
                cap = float(cdef.get("max", 1.0))
                cap_70b = np.minimum(cap_70b, cap)
            elif rule == "per_dc_power_max":
                per_dc_power_max = float(cdef.get("max", 1.0))
            elif rule == "global_power_max_sum":
                global_power_max_sum = float(cdef.get("max_sum", None))
            else:
                # Ignore unknown rules; they might be cost-only constraints handled elsewhere
                continue

        # Apply masks
        changed = False
        dist7b_new, ok7b = _mask_and_renorm(dist7b, mask_7b)
        dist70b_new, ok70b = _mask_and_renorm(dist70b, mask_70b)
        changed |= (not np.allclose(dist7b, dist7b_new) or not np.allclose(dist70b, dist70b_new))
        dist7b, dist70b = dist7b_new, dist70b_new

        # Apply per-DC share caps (project onto capped simplex)
        if cap_7b.min() < 1.0 - 1e-12:
            proj7b, feas7b = _cap_simplex_with_upper_bounds(dist7b, cap_7b)
            changed |= not np.allclose(dist7b, proj7b)
            dist7b = proj7b
            if not feas7b:
                # Infeasible caps; we relaxed by setting x=ub (sum<1). Renormalize softly to keep scheduler happy.
                s = dist7b.sum()
                if s > 0:
                    dist7b = dist7b / s

        if cap_70b.min() < 1.0 - 1e-12:
            proj70b, feas70b = _cap_simplex_with_upper_bounds(dist70b, cap_70b)
            changed |= not np.allclose(dist70b, proj70b)
            dist70b = proj70b
            if not feas70b:
                s = dist70b.sum()
                if s > 0:
                    dist70b = dist70b / s

        # Per-DC power cap
        if per_dc_power_max < 1.0 - 1e-12:
            power_capped = np.minimum(power, per_dc_power_max)
            changed |= not np.allclose(power, power_capped)
            power = power_capped

        # Global power cap (scale down proportionally if exceeded)
        if global_power_max_sum is not None:
            s = power.sum()
            if s > global_power_max_sum + 1e-12 and s > 1e-12:
                scale = global_power_max_sum / s
                power_scaled = power * scale
                changed |= True
                power = power_scaled

        # Rebuild projected actions and flags
        projected_actions = {}
        projection_flags = {}
        for i, a in enumerate(agents):
            before = actions[a]
            after = np.array([dist7b[i], dist70b[i], power[i]], dtype=np.float32)
            projected_actions[a] = after
            projection_flags[a] = (changed or not np.allclose(before, after))

        return projected_actions, projection_flags

    def _compute_constraint_costs(self, metrics_pack: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        costs: Dict[str, float] = {}

        g_raw = metrics_pack.get("global", {})
        g_norm = metrics_pack.get("global_norm", {})

        # Canonical mapping from constraint names to metric keys
        # You can extend this as needed (e.g., "slo_violation_rate": "slo_violation_rate")
        name_to_metric_raw = {
            "avg_ttft": "avg_ttft",
            "ttft": "avg_ttft",
            "energy_cost": "energy_cost",
            "cost": "energy_cost",
            "carbon_emissions": "carbon_emissions",
            "carbon": "carbon_emissions",
            "water_usage": "water_usage",
            "water": "water_usage",
            "total_energy": "total_energy",
            "network_load": "network_load",
        }

        # For normalized space, use keys that match your reward normalization
        name_to_metric_norm = {
            "ttft": "ttft",
            "carbon": "carbon",
            "water": "water",
            "cost": "cost",
            "total_energy": "total_energy",
            "network_load": "network_load",
        }

        for cname, cdef in self.constraints.items():
            units = cdef.get("budget_units", "raw")  # "raw" or "norm"

            if units == "raw":
                # Look up raw metric by name or alias; if missing, fall back to 0.0
                raw_key = name_to_metric_raw.get(cname, cname)
                c_val = float(g_raw.get(raw_key, 0.0))
            else:
                # Normalized with "good = high" (1 = best). Convert to cost where higher = worse.
                norm_key = name_to_metric_norm.get(cname, cname)
                good = float(g_norm.get(norm_key, 0.0))
                c_val = float(1.0 - np.clip(good, 0.0, 1.0))  # turn "good" into "cost"

            costs[cname] = c_val

        return costs

    def _current_constraint_status(self) -> Dict[str, Dict[str, float]]:
        status: Dict[str, Dict[str, float]] = {}
        for cname, cdef in self.constraints.items():
            budget = float(cdef.get("budget", 0.0))
            window = cdef.get("window", "episode")
            if window == "episode":
                avg = self._episodic_avg_cost(cname)
                status[cname] = {
                    "avg_cost": avg,
                    "budget": budget,
                    "margin": budget - avg,
                }
            else:
                status[cname] = {
                    "avg_cost": float('nan'),
                    "budget": budget,
                    "margin": float('nan'),
                }
        return status

    def _episodic_avg_cost(self, cname: str) -> float:
        cnt = max(int(self.episodic_cost_counts.get(cname, 0)), 1)
        tot = float(self.episodic_cost_totals.get(cname, 0.0))
        return tot / cnt

    def _update_duals_after_episode(self) -> None:
        for cname, cdef in self.constraints.items():
            if cdef.get("window", "episode") != "episode":
                continue

            budget = float(cdef.get("budget", 0.0))
            alpha = float(self.lambda_lr.get(cname, 1e-3))
            avg_cost = self._episodic_avg_cost(cname)

            new_lam = self.duals.get(cname, 0.0) + alpha * (avg_cost - budget)
            # Non-negativity
            new_lam = max(0.0, new_lam)

            # Optional clipping range for stability
            lam_clip = cdef.get("lambda_clip", None)
            if isinstance(lam_clip, (list, tuple)) and len(lam_clip) == 2:
                lo, hi = float(lam_clip[0]), float(lam_clip[1])
                if hi < lo:
                    lo, hi = hi, lo
                new_lam = min(max(new_lam, lo), hi)

            self.duals[cname] = float(new_lam)

    def _build_schedule_plan(self, df_7b, df_70b, dist_7b, dist_70b):

        local_pref_threshold = 0.20  # same spirit as your original heuristic

        def _normalize_proportions(w):
            w = np.asarray(w, dtype=np.float64)
            w = np.clip(w, 0.0, None)
            s = w.sum()
            if s <= 0.0 or not np.isfinite(s):
                return np.full(self.NUM_DATACENTERS, 1.0 / self.NUM_DATACENTERS, dtype=np.float64)
            return w / s

        def _integer_quotas(n_items, proportions):
            raw = proportions * float(n_items)
            floors = np.floor(raw).astype(int)
            rema = raw - floors
            remaining = int(n_items - floors.sum())
            if remaining > 0:
                order = np.argsort(-rema)  # descending remainder
                for i in range(remaining):
                    floors[order[i % len(order)]] += 1
            return floors  # length = num_dcs, sum == n_items

        def _assign_requests(requests_df, proportions):
            if requests_df is None or len(requests_df) == 0:
                return []

            proportions = _normalize_proportions(proportions)
            n = len(requests_df)
            quotas = _integer_quotas(n, proportions)
            remaining = quotas.copy()

            max_share = float(proportions.max())
            meaningful = proportions >= (local_pref_threshold * max_share)

            assignments = []
            # Iterate in row order (time_index already present in df)
            for row in requests_df.itertuples():
                src = int(row.source_dc_id)

                # Prefer local if it still has quota and its share is meaningful
                if 0 <= src < self.NUM_DATACENTERS and remaining[src] > 0 and meaningful[src]:
                    target_dc = src
                else:
                    # Pick DC with largest remaining quota (ties → lowest index)
                    if remaining.sum() > 0:
                        target_dc = int(np.argmax(remaining))
                    else:
                        # Shouldn't happen, but fallback to source
                        target_dc = src if (0 <= src < self.NUM_DATACENTERS) else 0

                remaining[target_dc] -= 1

                assignments.append({
                    "target_dc_id": int(target_dc),
                    "model_type": str(row.model_type),
                    "num_tokens": int(row.num_tokens),
                    "batch_size": int(getattr(row, "batch_size", 1)),
                    "source_dc_id": int(row.source_dc_id),
                    "time_index": int(row.time_index),
                })

            return assignments

        # Build per-model plans and concatenate
        dist_7b = _normalize_proportions(dist_7b)
        dist_70b = _normalize_proportions(dist_70b)

        plan_7b = _assign_requests(df_7b, dist_7b)
        plan_70b = _assign_requests(df_70b, dist_70b)

        return plan_7b + plan_70b

    def observe(self, agent: str):
        # --- Shapes & one-hot ---
        obs_dim = self.observation_spaces[agent].shape[0]
        onehot_dim = len(self.agents)
        try:
            agent_idx = self.agents.index(agent)
            agent_onehot = np.eye(onehot_dim, dtype=np.float32)[agent_idx]
        except ValueError:
            agent_onehot = np.zeros(onehot_dim, dtype=np.float32)

        # --- If no data yet this step, return zeros + onehot (matches declared shape) ---
        if (
                self.current_step == 0
                or self.current_step not in self.collected_plans
                or agent not in self.collected_plans[self.current_step]
        ):
            vec = np.zeros(obs_dim, dtype=np.float32)
            # place onehot starting at index 8
            start = 8
            end = start + onehot_dim
            if end <= obs_dim:
                vec[start:end] = agent_onehot[: max(0, min(onehot_dim, obs_dim - start))]
            return vec

        try:
            # --- Pull latest per-step artifacts ---
            agent_data = self.collected_plans[self.current_step][agent]
            # Prefer the new raw metrics dict; fall back to legacy "metrics"
            m_raw = agent_data.get("metrics_raw") or agent_data.get("metrics", {})
            power_plan = agent_data.get("power_plan", {})
            dc_id = int(agent.split("_")[1])

            # Raw simulator metrics (domain units)
            carbon = float(m_raw.get("carbon_emissions", 0.0))
            ttft = float(m_raw.get("avg_ttft", 0.0))
            water = float(m_raw.get("water_usage", 0.0))
            cost = float(m_raw.get("energy_cost", 0.0))

            # --- Dynamic normalization (per-episode max trackers) ---
            max_tracker = getattr(self, "metric_max_tracker", {
                "carbon": max(carbon, 1.0),
                "ttft": max(ttft, 1.0),
                "water": max(water, 1.0),
                "cost": max(cost, 1.0),
            })

            carbon_norm = float(np.clip(carbon / max(max_tracker["carbon"], 1e-12), 0.0, 1.0))
            ttft_norm = float(np.clip(ttft / max(max_tracker["ttft"], 1e-12), 0.0, 1.0))
            water_norm = float(np.clip(water / max(max_tracker["water"], 1e-12), 0.0, 1.0))
            cost_norm = float(np.clip(cost / max(max_tracker["cost"], 1e-12), 0.0, 1.0))

            # Workload scalars (normalize to a rough [0,1] range; adjust if desired)
            load_7b = float(np.clip(self.llama7b_total / 10000.0, 0.0, 1.0))
            load_70b = float(np.clip(self.llama70b_total / 10000.0, 0.0, 1.0))

            # Power summary for this DC: ratio of "On" nodes (6 types total)
            on_count = sum(1 for s in power_plan.get(dc_id, {}).values() if s == "On")
            power_ratio = float(np.clip(on_count / 6.0, 0.0, 1.0))

            # Progress in episode
            step_ratio = float(np.clip(self.current_step / max(1, self.max_steps), 0.0, 1.0))

            scalar_features = np.array([
                carbon_norm, ttft_norm, water_norm, cost_norm,
                load_7b, load_70b, power_ratio, step_ratio
            ], dtype=np.float32)

            # --- Optional: duals and episodic headroom features ---
            extras = []
            if getattr(self, "_obs_has_duals", False) and len(self.constraints) > 0:
                # Duals (sorted by constraint name for determinism)
                for cname in sorted(self.constraints.keys()):
                    lam = self.duals.get(cname, 0.0)
                    extras.append(self._normalize_dual(lam))
                # Episodic headroom (only for episode-window constraints)
                if getattr(self, "_obs_has_headroom", False):
                    for cname in sorted(self.constraints.keys()):
                        cdef = self.constraints[cname]
                        if cdef.get("window", "episode") == "episode":
                            extras.append(self._episodic_headroom(cname))

            # --- Assemble observation: [8 scalars | onehot | extras] ---
            obs_vec = np.concatenate([scalar_features, agent_onehot, np.array(extras, dtype=np.float32)],
                                     dtype=np.float32)

            # Shape guard: pad or truncate to declared shape if needed
            if obs_vec.shape[0] != obs_dim:
                if obs_vec.shape[0] < obs_dim:
                    pad = np.zeros(obs_dim - obs_vec.shape[0], dtype=np.float32)
                    obs_vec = np.concatenate([obs_vec, pad], dtype=np.float32)
                else:
                    obs_vec = obs_vec[:obs_dim]

            return obs_vec

        except Exception as e:
            print(f"[observe ❌] Error for agent {agent} at step {self.current_step}: {e}", flush=True)
            # Fallback: zeros + onehot
            vec = np.zeros(obs_dim, dtype=np.float32)
            start = 8
            end = start + onehot_dim
            if end <= obs_dim:
                vec[start:end] = agent_onehot[: max(0, min(onehot_dim, obs_dim - start))]
            return vec

    def _normalize_dual(self, lam: float) -> float:
        # tanh squashing: 0 -> 0.5, grows toward 1.0 as λ→∞
        return float(np.tanh(float(lam)) * 0.5 + 0.5)

    def _episodic_headroom(self, cname: str) -> float:
        c_def = self.constraints.get(cname, {})
        budget = float(c_def.get("budget", 0.0))
        if budget <= 0.0:
            return 0.0
        avg_cost = self._episodic_avg_cost(cname)  # defined earlier
        margin = budget - avg_cost
        return float(np.clip(margin / budget, 0.0, 1.0))

    def _extract_network_load(self, metrics: dict):
        net = metrics.get("network_load", 0.0)
        if isinstance(net, dict):
            scalar = float(net.get("avg_load_ratio", 0.0))
            detail = {
                "total_active_seconds": float(net.get("total_active_seconds", 0.0)),
                "total_capacity_seconds": float(net.get("total_capacity_seconds", 0.0)),
                "per_datacenter": [
                    {
                        "datacenter_id": int(d.get("datacenter_id", 0)),
                        "location": d.get("location", ""),
                        "load_ratio": float(d.get("load_ratio", 0.0)),
                        "active_seconds": float(d.get("active_seconds", 0.0)),
                        "capacity_seconds": float(d.get("capacity_seconds", 0.0)),
                    }
                    for d in net.get("per_datacenter", [])
                ],
            }
            return scalar, detail
        return float(net), None

    def _to_scalar(self, x, reduce: str = "sum") -> float:
        if isinstance(x, dict):
            vals = list(x.values())
        elif isinstance(x, (list, tuple)):
            vals = list(x)
        else:
            try:
                return float(x)
            except Exception:
                return 0.0

        arr = np.array(vals, dtype=np.float64)
        if reduce == "mean":
            return float(np.nanmean(arr)) if arr.size else 0.0
        if reduce == "max":
            return float(np.nanmax(arr)) if arr.size else 0.0
        # default: sum
        return float(np.nansum(arr)) if arr.size else 0.0

    def render(self, mode="human"):
        print(f"[Step {self.current_step}] Agent outputs:")
        for agent, result in self.collected_plans[self.current_step].items():
            m = result["metrics"]
            print(f"  {agent}: TTFT={m['avg_ttft']:.3f}s, Carbon={m['carbon_emissions']:.2f}g, Water={m['water_usage']:.2f}L, Cost=${m['energy_cost']:.4f}")

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def get_schedule_plan(self, agent_id):
        return self.schedule_plan.get(agent_id, [])

    def get_power_plan(self, agent_id):
        return self.power_plan.get(agent_id, {})

    def get_collected_plans(self):
        return self.collected_plans

    def get_rewards(self):
        return self.rewards

    def get_current_step(self):
        return self.current_step

    def get_agents(self):
        return self.agents

    def observe_all(self):
        return {agent: self.observe(agent) for agent in self.agents}


import os
import json
import csv
import matplotlib.pyplot as plt
import psutil
import gc
import time
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
import threading
import datetime
from torch.utils.tensorboard import SummaryWriter
import traceback
import pandas as pd
from pathlib import Path

class CustomLoggerCallback(BaseCallback):
    def __init__(self, scheme_id, epoch_idx, log_dir, agent_list, verbose=0):
        super().__init__(verbose)
        self.scheme_id = scheme_id
        self.epoch_idx = epoch_idx
        self.log_dir = log_dir
        self.agent_list = agent_list
        self.step = 0

        # Time series we summarize at rollout end
        self.episode_rewards = []
        self.episode_metrics = {
            "ttft": [], "carbon": [], "water": [], "cost": [],
            "total_energy": [], "network_load": [],
            "metric_time": [], "leftovers": [], "proj_frac": []
        }

        # Paths
        self.step_log_file = os.path.join(log_dir, f"{scheme_id}_epoch_{epoch_idx}_steps.csv")
        self.summary_file = os.path.join(log_dir, "summary", f"{scheme_id}_epoch_{epoch_idx}_summary.csv")
        self.combined_summary_file = os.path.join(log_dir, "summary", f"combined_epoch_{epoch_idx}_summary.csv")  # legacy
        self.combined_summary_file_v2 = os.path.join(log_dir, "summary", f"combined_epoch_{epoch_idx}_summary_v2.csv")
        self.plan_dir = os.path.join(log_dir, "plans")
        self.plot_dir = os.path.join(log_dir, "plots")
        self.details_file = os.path.join(log_dir, f"{scheme_id}_epoch_{epoch_idx}_details.jsonl")
        self.writer = SummaryWriter(log_dir=os.path.join(log_dir, "tensorboard", scheme_id))

        # Ensure subfolders exist
        for path in [self.plan_dir, self.plot_dir, os.path.dirname(self.summary_file)]:
            os.makedirs(path, exist_ok=True)

        # Initialize step CSV (now includes new columns)
        if not os.path.exists(self.step_log_file):
            with open(self.step_log_file, "w") as f:
                f.write("step,reward,ttft,carbon,water,cost,total_energy,network_load,metric_time,leftovers,proj_frac\n")

        # Legacy combined summary (kept as-is for backward compatibility)
        if not os.path.exists(self.combined_summary_file):
            with open(self.combined_summary_file, "w") as f:
                f.write("agent,epoch,avg_reward,avg_ttft,avg_carbon,avg_water,avg_cost\n")

        # New combined summary V2 with extra metrics
        if not os.path.exists(self.combined_summary_file_v2):
            with open(self.combined_summary_file_v2, "w") as f:
                f.write("agent,epoch,avg_reward,avg_ttft,avg_carbon,avg_water,avg_cost,avg_total_energy,avg_network_load,avg_metric_time,avg_leftovers,avg_proj_frac\n")

    # --- unwrap helper (unchanged) ---
    def _unwrap_to_resource_env(self, env):
        def recursive_find(env, max_depth=20):
            visited = set()
            stack = [(env, 0)]
            while stack:
                current, depth = stack.pop()
                if id(current) in visited or depth > max_depth:
                    continue
                visited.add(id(current))
                if isinstance(current, ResourceEnv):
                    return current
                for attr in dir(current):
                    if attr.startswith("__"):
                        continue
                    try:
                        sub = getattr(current, attr)
                        if isinstance(sub, (list, tuple)):
                            stack.extend((item, depth + 1) for item in sub)
                        elif hasattr(sub, "__class__"):
                            stack.append((sub, depth + 1))
                    except Exception:
                        continue
            return None
        return recursive_find(env)

    def _on_step(self) -> bool:
        try:
            env = self._unwrap_to_resource_env(self.model.get_env())
            if env is None or env.current_step not in env.collected_plans:
                return True

            metrics_list = []
            rewards_list = []
            metric_times = []
            proj_flags = []
            # costs/constraints/duals (take from first agent if consistent across agents)
            constraints_status = None
            duals_snapshot = None
            costs_snapshot = None

            leftovers_count = 0
            if hasattr(env, "get_last_leftovers"):
                last_leftovers = env.get_last_leftovers()
                leftovers_count = len(last_leftovers) if last_leftovers is not None else 0

            for agent_id in env.agents:
                step_data = env.collected_plans[env.current_step].get(agent_id)
                if step_data is None:
                    continue

                # Prefer new metrics_raw; fallback to legacy "metrics"
                metrics = step_data.get("metrics_raw") or step_data.get("metrics", {})
                reward = env.rewards.get(agent_id, 0.0)
                metric_time = step_data.get("metric_time", None)
                projected = step_data.get("projected", False)

                # Also peek into env.infos for constraints/duals/costs
                info = env.infos.get(agent_id, {})
                if constraints_status is None:
                    constraints_status = info.get("constraints", None)
                if duals_snapshot is None:
                    duals_snapshot = info.get("duals", None)
                if costs_snapshot is None:
                    costs_snapshot = info.get("costs", None)

                # Standardize metric keys for averaging
                m_row = {
                    "avg_ttft": metrics.get("avg_ttft", metrics.get("ttft", 0.0)),
                    "carbon_emissions": metrics.get("carbon_emissions", metrics.get("carbon", 0.0)),
                    "water_usage": metrics.get("water_usage", metrics.get("water", 0.0)),
                    "energy_cost": metrics.get("energy_cost", metrics.get("cost", 0.0)),
                    "total_energy": metrics.get("total_energy", 0.0),
                    "network_load": 0.0
                }
                metrics_list.append(m_row)
                rewards_list.append(float(reward))
                if metric_time is not None:
                    metric_times.append(float(metric_time))
                proj_flags.append(1.0 if projected else 0.0)

            if not metrics_list:
                return True

            # Averages across agents for this env step
            avg_reward = float(np.mean(rewards_list))
            avg_ttft = float(np.mean([m["avg_ttft"] for m in metrics_list]))
            avg_carbon = float(np.mean([m["carbon_emissions"] for m in metrics_list]))
            avg_water = float(np.mean([m["water_usage"] for m in metrics_list]))
            avg_cost = float(np.mean([m["energy_cost"] for m in metrics_list]))
            avg_total_energy = float(np.mean([m["total_energy"] for m in metrics_list]))
            avg_network_load = float(np.mean([m["network_load"] for m in metrics_list]))
            avg_time = float(np.mean(metric_times)) if metric_times else 0.0
            proj_frac = float(np.mean(proj_flags)) if proj_flags else 0.0

            # Accumulate for rollout summary
            self.episode_rewards.append(avg_reward)
            self.episode_metrics["ttft"].append(avg_ttft)
            self.episode_metrics["carbon"].append(avg_carbon)
            self.episode_metrics["water"].append(avg_water)
            self.episode_metrics["cost"].append(avg_cost)
            self.episode_metrics["total_energy"].append(avg_total_energy)
            self.episode_metrics["network_load"].append(avg_network_load)
            self.episode_metrics["metric_time"].append(avg_time)
            self.episode_metrics["leftovers"].append(float(leftovers_count))
            self.episode_metrics["proj_frac"].append(proj_frac)

            # Step CSV (new columns appended)
            with open(self.step_log_file, "a") as f:
                f.write(f"{self.step},{avg_reward:.4f},{avg_ttft:.4f},{avg_carbon:.2f},{avg_water:.2f},{avg_cost:.4f},{avg_total_energy:.4f},{avg_network_load:.4f},{avg_time:.4f},{leftovers_count},{proj_frac:.4f}\n")

            # Per-step details JSONL (constraints/duals/costs; one line per step)
            try:
                details_row = {
                    "step": int(self.step),
                    "reward_avg": avg_reward,
                    "metrics_avg": {
                        "avg_ttft": avg_ttft,
                        "carbon_emissions": avg_carbon,
                        "water_usage": avg_water,
                        "energy_cost": avg_cost,
                        "total_energy": avg_total_energy,
                        "network_load": avg_network_load,
                        "metric_time": avg_time,
                        "leftovers": int(leftovers_count),
                        "proj_frac": proj_frac,
                    },
                    "constraints": constraints_status,
                    "duals": duals_snapshot,
                    "costs": costs_snapshot,
                }
                with open(self.details_file, "a") as jf:
                    jf.write(json.dumps(details_row) + "\n")
            except Exception as _:
                pass  # don't interrupt training for logging hiccups

            # Save a representative plan every 1000 steps (now with constraints + duals)
            if self.step % 1000 == 0:
                rep_agent = env.agents[0]
                rep_data = env.collected_plans[env.current_step].get(rep_agent, {})
                plan_path = os.path.join(self.plan_dir, f"{self.scheme_id}_epoch_{self.epoch_idx}_step_{self.step}.json")
                with open(plan_path, "w") as f:
                    json.dump({
                        "step": int(self.step),
                        "schedule_plan": rep_data.get("schedule_plan", []),
                        "power_plan": rep_data.get("power_plan", {}),
                        "constraints": constraints_status,
                        "duals": duals_snapshot,
                        "leftovers": env.get_last_leftovers() if hasattr(env, "get_last_leftovers") else None,
                    }, f, indent=2)

            # TensorBoard scalars (new)
            self.writer.add_scalar("step/avg_total_energy", avg_total_energy, self.num_timesteps)
            self.writer.add_scalar("step/avg_network_load", avg_network_load, self.num_timesteps)
            self.writer.add_scalar("step/metric_time", avg_time, self.num_timesteps)
            self.writer.add_scalar("step/leftovers", leftovers_count, self.num_timesteps)
            self.writer.add_scalar("step/proj_frac", proj_frac, self.num_timesteps)

            # Keep your existing TB metrics too
            self.writer.add_scalar("step/avg_reward", avg_reward, self.num_timesteps)
            self.writer.add_scalar("step/avg_ttft", avg_ttft, self.num_timesteps)
            self.writer.add_scalar("step/avg_carbon", avg_carbon, self.num_timesteps)
            self.writer.add_scalar("step/avg_water", avg_water, self.num_timesteps)
            self.writer.add_scalar("step/avg_cost", avg_cost, self.num_timesteps)

            self.step += 1

        except Exception as e:
            print(f"[CustomLogger ❌] Step logging error: {e}", flush=True)

        return True

    def _on_rollout_end(self) -> None:
        try:
            n = len(self.episode_rewards)
            if n == 0:
                print("[CustomLogger] No steps recorded.")
                return

            def _avg(key): return float(np.mean(self.episode_metrics[key])) if self.episode_metrics[key] else 0.0

            avg_reward = float(np.mean(self.episode_rewards))
            avg_ttft = _avg("ttft")
            avg_carbon = _avg("carbon")
            avg_water = _avg("water")
            avg_cost = _avg("cost")
            avg_total_energy = _avg("total_energy")
            avg_network_load = _avg("network_load")
            avg_time = _avg("metric_time")
            avg_leftovers = _avg("leftovers")
            avg_proj_frac = _avg("proj_frac")

            # Per-profile summary CSV (single-row)
            with open(self.summary_file, "w") as f:
                f.write("epoch,avg_reward,avg_ttft,avg_carbon,avg_water,avg_cost,avg_total_energy,avg_network_load,avg_metric_time,avg_leftovers,avg_proj_frac\n")
                f.write(f"{self.epoch_idx},{avg_reward:.4f},{avg_ttft:.4f},{avg_carbon:.2f},{avg_water:.2f},{avg_cost:.4f},{avg_total_energy:.4f},{avg_network_load:.4f},{avg_time:.4f},{avg_leftovers:.2f},{avg_proj_frac:.4f}\n")

            # Legacy combined (kept to avoid breaking downstream tools)
            with open(self.combined_summary_file, "a") as f:
                f.write(f"{self.scheme_id},{self.epoch_idx},{avg_reward:.4f},{avg_ttft:.4f},{avg_carbon:.2f},{avg_water:.2f},{avg_cost:.4f}\n")

            # New combined V2 with extra metrics
            with open(self.combined_summary_file_v2, "a") as f:
                f.write(f"{self.scheme_id},{self.epoch_idx},{avg_reward:.4f},{avg_ttft:.4f},{avg_carbon:.2f},{avg_water:.2f},{avg_cost:.4f},{avg_total_energy:.4f},{avg_network_load:.4f},{avg_time:.4f},{avg_leftovers:.2f},{avg_proj_frac:.4f}\n")

            # TensorBoard rollout summaries (new + legacy)
            self.writer.add_scalar("rollout/avg_reward", avg_reward, self.num_timesteps)
            self.writer.add_scalar("rollout/avg_ttft", avg_ttft, self.num_timesteps)
            self.writer.add_scalar("rollout/avg_carbon", avg_carbon, self.num_timesteps)
            self.writer.add_scalar("rollout/avg_water", avg_water, self.num_timesteps)
            self.writer.add_scalar("rollout/avg_cost", avg_cost, self.num_timesteps)
            self.writer.add_scalar("rollout/avg_total_energy", avg_total_energy, self.num_timesteps)
            self.writer.add_scalar("rollout/avg_network_load", avg_network_load, self.num_timesteps)
            self.writer.add_scalar("rollout/avg_metric_time", avg_time, self.num_timesteps)
            self.writer.add_scalar("rollout/avg_leftovers", avg_leftovers, self.num_timesteps)
            self.writer.add_scalar("rollout/avg_proj_frac", avg_proj_frac, self.num_timesteps)

            # Plots (keep your existing behavior; add plots for new metrics)
            metric_map = {
                "reward": self.episode_rewards,
                "ttft": self.episode_metrics["ttft"],
                "carbon": self.episode_metrics["carbon"],
                "water": self.episode_metrics["water"],
                "cost": self.episode_metrics["cost"],
                "total_energy": self.episode_metrics["total_energy"],
                "network_load": self.episode_metrics["network_load"],
                "metric_time": self.episode_metrics["metric_time"],
                "leftovers": self.episode_metrics["leftovers"],
                "proj_frac": self.episode_metrics["proj_frac"],
            }

            window = 20
            for key, values in metric_map.items():
                if not values:
                    continue
                steps = list(range(len(values)))
                raw_series = pd.Series(values, dtype=float)
                smoothed = raw_series.rolling(window=window).mean()

                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(steps, values, label=f"{key.capitalize()} (Raw)", alpha=0.4)
                ax.plot(steps, smoothed, label=f"{key.capitalize()} (Smoothed)", linewidth=2)
                ax.set_title(f"{key.capitalize()} over Steps – {self.scheme_id} (Epoch {self.epoch_idx})")
                ax.set_xlabel("Step")
                ax.set_ylabel(key.replace('_', ' ').capitalize())
                ax.grid(True)
                ax.legend()
                plt.tight_layout()
                plot_path = os.path.join(self.plot_dir, f"{self.scheme_id}_epoch_{self.epoch_idx}_{key}.png")
                plt.savefig(plot_path)
                plt.close(fig)

                print(f"[CustomLogger ✅] Saved {key} plot → {plot_path}", flush=True)

            print(f"[CustomLogger ✅] Logged rollout summary + plots for '{self.scheme_id}'", flush=True)

        except Exception as e:
            print(f"[CustomLogger ❌] Rollout summary logging error: {e}", flush=True)

    def _on_training_end(self) -> None:
        self.writer.flush()
        self.writer.close()
        print(f"[CustomLogger ✅] Training finished for '{self.scheme_id}'", flush=True)


def get_memory_usage():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2  # in MB

def monitor_memory(agent_id, stop_event, epoch_idx):
    log_file = f"memlog_{agent_id}.txt"
    while not stop_event.is_set():
        process = psutil.Process(os.getpid())
        used = process.memory_info().rss / 1024 ** 2  # in MB
        mem = psutil.virtual_memory()

        msg_lines = [
            f"[{agent_id}] Memory Report @ {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"  Current Epoch   : {epoch_idx}",
            f"  Process RAM     : {used:.2f} MB",
            f"  Total RAM       : {mem.total / (1024 ** 2):.2f} MB",
            f"  Available RAM   : {mem.available / (1024 ** 2):.2f} MB",
            f"  Used RAM        : {mem.used / (1024 ** 2):.2f} MB",
            f"  RAM Usage %     : {mem.percent:.2f}%\n"
        ]

        msg = "\n".join(msg_lines)
        print(msg, flush=True)

        with open(log_file, "a") as f:
            f.write(msg + "\n")

        time.sleep(90)

def train_reward_scheme(scheme_id, env_config, total_timesteps=100_000, save_dir="trained_models/sb3_agents"):
    """
    Trains a single profile (formerly 'scheme') named `scheme_id`.

    IMPORTANT:
      - `env_config` MUST include: env_config["agent_specs"] = {<profile_name>: {...}, ...}
      - This function will set env_config["active_agent_profile"] = scheme_id (non-destructively).
    """

    print(f"Training reward scheme: {scheme_id}", flush=True)

    # --- Validate new config surface ---
    if "agent_specs" not in env_config or not isinstance(env_config["agent_specs"], dict):
        raise ValueError(
            "env_config must include 'agent_specs' (dict of profiles). "
            "Example: env_config['agent_specs'] = {'time_under_slo': {...}, ...}"
        )
    if scheme_id not in env_config["agent_specs"]:
        raise KeyError(f"Profile '{scheme_id}' not found in env_config['agent_specs'] keys: "
                       f"{list(env_config['agent_specs'].keys())}")

    # --- Prepare env config (non-destructive copy) ---
    cfg = dict(env_config)
    cfg["active_agent_profile"] = scheme_id
    # DO NOT set 'reward_agent_type' anymore; the env ignores it

    raw_env = ResourceEnv(cfg)
    death_wrapped = black_death_v3(raw_env)
    vec_env = supersuit.pettingzoo_env_to_vec_env_v1(death_wrapped)
    venv = supersuit.concat_vec_envs_v1(vec_env, num_vec_envs=2, base_class="stable_baselines3")

    # --- Paths per profile ---
    model_path = os.path.join(save_dir, scheme_id)
    os.makedirs(model_path, exist_ok=True)
    final_model_file = os.path.join(model_path, "final_model.zip")
    latest_checkpoint = os.path.join(model_path, "ppo_agent_latest.zip")

    agent_list = raw_env.agents
    logger_callback = CustomLoggerCallback(scheme_id, cfg["epoch_idx"], model_path, agent_list)

    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=model_path,
        name_prefix="ppo_agent",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    callbacks = [checkpoint_callback, logger_callback]

    # --- (Re)load model if present ---
    if Path(final_model_file).exists():
        print(f"Resuming from existing model: {final_model_file}", flush=True)
        model = PPO.load(final_model_file, env=venv, verbose=1, n_steps=64, batch_size=64, device="cpu")
    elif Path(latest_checkpoint).exists():
        print(f"Resuming from latest checkpoint: {latest_checkpoint}", flush=True)
        model = PPO.load(latest_checkpoint, env=venv, verbose=1, n_steps=64, batch_size=64, device="cpu")
    else:
        print(f"Starting fresh for scheme: {scheme_id}", flush=True)
        model = PPO(MlpPolicy, env=venv, verbose=1, n_steps=64, batch_size=64, device="cpu")

    # --- Optional memory monitor (kept as in your code) ---
    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=monitor_memory, args=(scheme_id, stop_event, cfg["epoch_idx"]))
    monitor_thread.start()

    try:
        print(f"[{scheme_id}] Memory before training: {get_memory_usage():.2f} MB", flush=True)
        model.learn(total_timesteps=total_timesteps, callback=callbacks)
        print(f"[{scheme_id}] Memory after training: {get_memory_usage():.2f} MB", flush=True)
    except Exception as e:
        print(f"[{scheme_id}] Training failed: {e}", flush=True)
        traceback.print_exc()
    finally:
        model.save(final_model_file, exclude=["replay_buffer", "optimizer"])
        print(f"✅ Finished training {scheme_id} — saved to {final_model_file}", flush=True)

        stop_event.set()
        monitor_thread.join()

        del model, venv, vec_env, death_wrapped, raw_env
        gc.collect()
        print(f"[{scheme_id}] Post-cleanup memory: {get_memory_usage():.2f} MB", flush=True)

    return None



import multiprocessing

def train_reward_scheme_wrapper(agent_id, env_config, total_timesteps):
    # env_config is expected to already contain 'agent_specs'
    return train_reward_scheme(agent_id, env_config, total_timesteps=total_timesteps)

def train_all_schemes(epoch_df,
                      epoch_summary,
                      epoch_idx,
                      node_properties,
                      agent_specs,                # <-- NEW required arg (dict of profiles)
                      num_datacenters=12,
                      total_timesteps=100_000):
    """
    Trains all profiles listed in `agent_specs` (dict), keeping the original function name.

    Example `agent_specs`:
    {
      "time_under_slo": {
        "weights": {"ttft": 10, "carbon": 0, "water": 0, "cost": 0},
        "constraints": {
          "slo_violation_rate": {"budget": 0.25, "scope": "global", "window": "episode", "hard": False}
        },
        "lambda_init": {"slo_violation_rate": 0.0},
        "lambda_lr": {"slo_violation_rate": 0.005},
        "include_duals_in_obs": True
      },
      "carbon_capped": {
        "weights": {"ttft": 7, "carbon": 2, "water": 0, "cost": 1},
        "constraints": {
          "carbon": {"budget": 2.2e5, "scope": "global", "window": "episode", "hard": False, "budget_units": "raw"}
        }
      }
    }
    """

    reward_schemes = list(agent_specs.keys())

    base_env_config = {
        "epoch_df": epoch_df,
        "epoch_summary": epoch_summary,
        "epoch_idx": epoch_idx,
        "node_properties": node_properties,
        "num_datacenters": num_datacenters,
        "max_steps": 50,
        "agent_specs": agent_specs,  # <- pass through
        # 'active_agent_profile' gets set inside train_reward_scheme
    }

    for scheme_id in reward_schemes:
        print(f"\n==== Training reward scheme: {scheme_id} ====")
        p = multiprocessing.Process(
            target=train_reward_scheme_wrapper,
            args=(scheme_id, base_env_config, total_timesteps),
        )
        p.start()
        p.join()


from concurrent.futures import ProcessPoolExecutor, as_completed
import os


def timed(name):
    def wrapper(func):
        def inner(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            print(f"[⏱️ {name}] Took {end - start:.4f} seconds")
            return result
        return inner
    return wrapper

def run_agent_inference(scheme_id, epoch_df, epoch_summary, epoch_idx, node_properties, model_base_path):
    print(f"[{scheme_id}] Starting inference...")
    total_start = time.perf_counter()

    # === 1. Create env config ===
    t0 = time.perf_counter()
    env_config = {
        "epoch_df": epoch_df,
        "epoch_summary": epoch_summary,
        "epoch_idx": epoch_idx,
        "node_properties": node_properties,
        "num_datacenters": 12,
        "max_steps": 50,
        "reward_agent_type": scheme_id,
    }

    raw_env = ResourceEnv(env_config)
    wrapped_env = black_death_v3(raw_env)
    vec_env = pettingzoo_env_to_vec_env_v1(wrapped_env)
    vec_env = concat_vec_envs_v1(vec_env, num_vec_envs=1, base_class="stable_baselines3")
    print(f"[{scheme_id}] Environment ready in {time.perf_counter() - t0:.4f}s")

    # === 2. Load model ===
    t1 = time.perf_counter()
    model_path = os.path.join(model_base_path, scheme_id, "final_model.zip")
    if not os.path.exists(model_path):
        print(f"[{scheme_id}] ERROR: Model not found at {model_path}")
        return scheme_id, {"error": "Model file missing"}

    model = PPO.load(model_path, env=vec_env, device='cpu')
    print(f"[{scheme_id}] Model loaded in {time.perf_counter() - t1:.4f}s")

    # === 3. Run rollout ===
    rollout_start = time.perf_counter()
    obs = vec_env.reset()
    done = False
    step_count = 0
    last_action = None

    while not done and step_count < raw_env.max_steps:
        step_t = time.perf_counter()
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = vec_env.step(action)
        done = dones["__all__"] if isinstance(dones, dict) else all(dones)
        step_count += 1
        last_action = action

        if step_count in {1, 10, 20, 30, 40, 50} or done:
            print(f"[{scheme_id}] Step {step_count}/{raw_env.max_steps}...", flush=True)

    print(f"[{scheme_id}] Rollout completed in {time.perf_counter() - rollout_start:.4f}s")

    if last_action is None:
        print(f"[{scheme_id}] No actions executed.")
        return scheme_id, {"error": "No steps run"}

    # === 4. Extract plan ===
    plan_start = time.perf_counter()

    def _unwrap_to_resource_env(env):
        def recursive_find(env, max_depth=20):
            visited = set()
            stack = [(env, 0)]
            while stack:
                current, depth = stack.pop()
                if id(current) in visited or depth > max_depth:
                    continue
                visited.add(id(current))
                if isinstance(current, ResourceEnv):
                    return current
                for attr in dir(current):
                    if attr.startswith("__"):
                        continue
                    try:
                        sub = getattr(current, attr)
                        if isinstance(sub, (list, tuple)):
                            stack.extend((item, depth + 1) for item in sub)
                        elif hasattr(sub, "__class__"):
                            stack.append((sub, depth + 1))
                    except Exception:
                        continue
            return None
        return recursive_find(env)

    unwrapped_env = _unwrap_to_resource_env(vec_env)
    schedule_plan = unwrapped_env.schedule_plan
    power_plan = unwrapped_env.power_plan

    print(f"[{scheme_id}] Plan extraction took {time.perf_counter() - plan_start:.4f}s")

    if not schedule_plan:
        print(f"[{scheme_id}] WARNING: schedule_plan is empty")
    if not power_plan:
        print(f"[{scheme_id}] WARNING: power_plan is empty")

    # === 5. Run final simulator ===
    sim_start = time.perf_counter()
    print(f"[{scheme_id}] Running full simulator...")
    metrics, _ = LLM_Simulator(epoch_idx, epoch_df, schedule_plan, power_plan)
    print(f"[{scheme_id}] Simulator took {time.perf_counter() - sim_start:.4f}s")
    print(f"[{scheme_id}] Final metrics: {metrics}")

    print(f"[{scheme_id}] Total time: {time.perf_counter() - total_start:.4f}s")

    return scheme_id, metrics

import cloudpickle

def run_multiagent(epoch_df, epoch_summary, epoch_idx, node_properties, model_base_path="trained_models/sb3_agents"):
    agent_ids = list(ResourceEnv.agent_reward_weights.keys())
    results = {}

    print("[Multiagent] Running agents one by one (no multiprocessing)...")

    for agent_id in agent_ids:
        try:
            agent_id, metrics = run_agent_inference(agent_id, epoch_df, epoch_summary, epoch_idx, node_properties, model_base_path)
            results[agent_id] = metrics
        except Exception as e:
            print(f"[Multiagent] ERROR: Exception during inference for {agent_id}: {e}")
            results[agent_id] = {"error": str(e)}

    print("[Multiagent] Inference complete.")
    return results



# def train_marl_agents(env, total_timesteps=100000, save_path="trained_models/marl_agents"):
#     os.makedirs(save_path, exist_ok=True)
#     vec_env = supersuit.pettingzoo_env_to_vec_env_v1(env)
#     concat_env = supersuit.concat_vec_envs_v1(vec_env, num_vec_envs=1, num_cpus=6, base_class="stable_baselines3")
#
#     agent_models = {}
#     for agent in env.agents:
#         model = PPO(
#             policy="MlpPolicy",
#             env=concat_env,
#             learning_rate=3e-4,
#             n_steps=2048,
#             batch_size=64,
#             n_epochs=10,
#             gamma=0.99,
#             gae_lambda=0.95,
#             clip_range=0.2,
#         )
#
#         obs, info = env.reset()
#
#         # for _ in range(1):
#         #     action, _ = model.predict(obs, deterministic=True)
#         #
#         #     # Debug: Check Action Dtype Before Passing to Step Function
#         #     print(f"DEBUG: {agent} - SB3 Raw Action Before Step: {action}, dtype: {action.dtype}")
#         #
#         #     # Ensure Action is `int8` before sending to step
#         #     action = action.astype(np.int8)
#         #     print(f"DEBUG: {agent} - Action After Cast: {action}, dtype: {action.dtype}")
#         #
#         #     obs, rewards, dones, infos = env.step(action)
#         #
#         #     obs_dict = {agent: obs[i] for i, agent in enumerate(env.agents)}
#         #
#         #
#         #     for agent_id, obs_val in obs_dict.items():
#         #         print(f"DEBUG: {agent_id} - Observation dtype: {obs_val.dtype}, Shape: {obs_val.shape}")
#
#         model.learn(total_timesteps=total_timesteps)
#
#         model_path = os.path.join(save_path, f"{agent}.zip")
#         model.save(model_path)
#
#         agent_models[agent] = model
#
#     return agent_models

def main():
    # Test Configuration
    fake_func_invocations = np.array([
        np.linspace(10, 100, 901),
        np.linspace(5, 50, 901)
    ])
    config = {
        "number_of_nodes": 8,
        "max_steps": 10,
        "epoch_idx": 0,
        "func_distribution_arr": fake_func_invocations
    }

    env = ResourceEnv(config)

    observations = env.reset()
    print("\n Initial Observations:", observations)

    # Run a few test steps
    for step in range(config["max_steps"]):
        print(f"\n Step {step + 1}")

        # Generate a random action (MultiBinary action per agent)
        actions = {agent: env.action_spaces[agent].sample() for agent in env.agents}

        # Step the environment
        observations, rewards, dones, infos = env.step(actions)

        print("Actions:", actions)
        print("Observations:", observations)
        print("Rewards:", rewards)
        print("Dones:", dones)

        if all(dones.values()):
            print("\n Finished")
            break

if __name__ == "__main__":
    main()


# class TimeEnv(gym.Env):
#     def __init__(self, config=None):
#         super().__init__()
#         if config is None:
#             config = {}
#
#         self.rng = seed
#         self.num_nodes = config.get("number_of_nodes", 8)
#
#         self.action_space = spaces.MultiBinary(self.num_nodes)
#
#         self.observation_space = spaces.Box(
#             low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
#         )
#
#         self.epoch_idx = config.get("epoch_idx", 0)
#         # Internal counters / arrays
#         self.current_step = 0
#         self.prev_action = 0
#         self.prev_reward = 0.0
#         self.max_steps = config.get("max_steps", 20)
#
#         self.vm_distribution_arr = config.get("vm_distribution_arr", np.zeros((8, 901)))
#         self.new_vm_distribution_arr = copy.deepcopy(self.vm_distribution_arr)
#         self.func_distribution_arr = config.get("func_distribution_arr", np.zeros((3, 901)))
#         self.leftover_resource_time_arr = config.get("leftover_resource_time_arr", np.zeros((8, 901)))
#         self.func_priority_list = config.get("func_priority_list", [0,1])
#         self.lut_list = config.get("lut_list", [])
#
#     def seed(self, seed=None):
#         self.rng = seed
#
#     def reset(self, *, seed=None, options=None):
#         super().reset(seed=seed)
#         self.current_step = 0
#
#         self.leftover_resource_time_arr.fill(0.0)
#         obs = self._get_obs()
#         info = {}
#         return obs, info
#
#     def step(self, action):
#         self.current_step += 1
#
#         self.new_vm_distribution_arr[self.epoch_idx, :] = action
#
#         (
#             cumulative_resource_time_arr,
#             average_time_to_first_token,
#             total_carbon_emissions,
#             total_water_usage,
#             total_invocation_served,
#             updated_vm_distribution_arr
#         ) = simulation(
#             vm_distribution_arr=self.vm_distribution_arr,
#             new_vm_distribution_arr=self.new_vm_distribution_arr,
#             func_distribution_arr=self.func_distribution_arr,
#             leftover_resource_time_arr=self.leftover_resource_time_arr,
#             func_priority_list=self.func_priority_list,
#             lut_list=self.lut_list
#         )
#
#
#         self.vm_distribution_arr = updated_vm_distribution_arr
#         # leftover_resource_time_arr for next step
#         self.leftover_resource_time_arr = cumulative_resource_time_arr[:, epoch_length:]
#
#
#
#         reward = -float(average_time_to_first_token)
#         terminated = (self.current_step >= self.max_steps)
#         truncated = False
#
#         self.prev_action = float(self.multi_binary_to_int(action))
#         self.prev_reward = float(reward)
#
#         obs = self._get_obs()
#
#         info = {"time_val": average_time_to_first_token}
#         return obs, reward, terminated, truncated, info
#
#     def _get_obs(self):
#
#         obs = np.array([
#             self.prev_action,
#             self.prev_reward,
#             self.current_step
#         ], dtype=np.float32)
#         return obs
#
#     def multi_binary_to_int(self, action):
#         bit_string = ''.join(str(int(a)) for a in action)
#         return int(bit_string, 2)
#
#

# class CarbonEnv(gym.Env):
#     def __init__(self, config=None):
#         super().__init__()
#         if config is None:
#             config = {}
#
#         self.rng = seed
#         self.num_nodes = config.get("number_of_nodes", 8)
#
#         self.action_space = spaces.MultiBinary(self.num_nodes)
#
#         self.observation_space = spaces.Box(
#             low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
#         )
#
#         self.epoch_idx = config.get("epoch_idx", 0)
#
#         self.current_step = 0
#         self.prev_action = 0
#         self.prev_reward = 0.0
#         self.max_steps = config.get("max_steps", 20)
#
#         self.vm_distribution_arr = config.get("vm_distribution_arr", np.zeros((8, 901)))
#         self.new_vm_distribution_arr = copy.deepcopy(self.vm_distribution_arr)
#         self.func_distribution_arr = config.get("func_distribution_arr", np.zeros((3, 901)))
#         self.leftover_resource_time_arr = config.get("leftover_resource_time_arr", np.zeros((8, 901)))
#         self.func_priority_list = config.get("func_priority_list", [0,1])
#         self.lut_list = config.get("lut_list", [])
#
#     def seed(self, seed=None):
#         self.rng = seed
#
#     def reset(self, *, seed=None, options=None):
#         super().reset(seed=seed)
#         self.current_step = 0
#         self.leftover_resource_time_arr.fill(0.0)
#         obs = self._get_obs()
#         info = {}
#         return obs, info
#
#     def step(self, action):
#         self.current_step += 1
#         # print("epoch")
#         # print(self.epoch_idx)
#         # print("Action taken: ")
#         # print(action)
#         self.new_vm_distribution_arr[self.epoch_idx, :] = action
#
#         (
#             cumulative_resource_time_arr,
#             average_time_to_first_token,
#             total_carbon_emissions,
#             total_water_usage,
#             total_invocation_served,
#             updated_vm_distribution_arr
#         ) = simulation(
#             vm_distribution_arr=self.vm_distribution_arr,
#             new_vm_distribution_arr=self.new_vm_distribution_arr,
#             func_distribution_arr=self.func_distribution_arr,
#             leftover_resource_time_arr=self.leftover_resource_time_arr,
#             func_priority_list=self.func_priority_list,
#             lut_list=self.lut_list
#         )
#
#         self.vm_distribution_arr = updated_vm_distribution_arr
#         self.leftover_resource_time_arr = cumulative_resource_time_arr[:, epoch_length:]
#         # print("vm distribution: ")
#         # print(self.vm_distribution_arr)
#         # print("cumulative resource array: ")
#         # print(cumulative_resource_time_arr)
#         # print("leftover resource array")
#         # print(self.leftover_resource_time_arr)
#         # print("Carbon Emissions")
#         # print(total_carbon_emissions)
#         reward = -float(total_carbon_emissions)
#
#         terminated = (self.current_step >= self.max_steps)
#         truncated = False
#
#         self.prev_action = float(self.multi_binary_to_int(action))
#         self.prev_reward = float(reward)
#         obs = self._get_obs()
#         # print("observation space")
#         # print(obs)
#         info = {"carbon_val": total_carbon_emissions}
#         return obs, reward, terminated, truncated, info
#
#     def _get_obs(self):
#
#         obs = np.array([
#             self.prev_action,
#             self.prev_reward,
#             self.current_step
#         ], dtype=np.float32)
#         return obs
#
#     def multi_binary_to_int(self, action):
#         bit_string = ''.join(str(int(a)) for a in action)
#         return int(bit_string, 2)
#
#
#

# class WaterEnv(gym.Env):
#     def __init__(self, config=None):
#         super().__init__()
#         if config is None:
#             config = {}
#
#         self.rng = seed
#         self.num_nodes = config.get("number_of_nodes", 8)
#
#         self.action_space = spaces.MultiBinary(self.num_nodes)
#
#         self.observation_space = spaces.Box(
#             low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
#         )
#
#         self.epoch_idx = config.get("epoch_idx", 0)
#
#         self.current_step = 0
#         self.prev_action = 0
#         self.prev_reward = 0.0
#         self.max_steps = config.get("max_steps", 20)
#
#         self.vm_distribution_arr = config.get("vm_distribution_arr", np.zeros((901, 8)))
#         self.new_vm_distribution_arr = copy.deepcopy(self.vm_distribution_arr)
#         self.func_distribution_arr = config.get("func_distribution_arr", np.zeros((3, 901)))
#         self.leftover_resource_time_arr = config.get("leftover_resource_time_arr", np.zeros((8, 901)))
#         self.func_priority_list = config.get("func_priority_list", [0,1])
#         self.lut_list = config.get("lut_list", [])
#
#     def seed(self, seed=None):
#         self.rng = seed
#
#     def reset(self, *, seed=None, options=None):
#         super().reset(seed=seed)
#         self.current_step = 0
#         self.leftover_resource_time_arr.fill(0.0)
#         obs = self._get_obs()
#         info = {}
#         return obs, info
#
#     def step(self, action):
#         self.current_step += 1
#         self.new_vm_distribution_arr[self.epoch_idx, :] = action
#
#         (
#             cumulative_resource_time_arr,
#             average_time_to_first_token,
#             total_carbon_emissions,
#             total_water_usage,
#             total_invocation_served,
#             updated_vm_distribution_arr
#         ) = simulation(
#             vm_distribution_arr=self.vm_distribution_arr,
#             new_vm_distribution_arr=self.new_vm_distribution_arr,
#             func_distribution_arr=self.func_distribution_arr,
#             leftover_resource_time_arr=self.leftover_resource_time_arr,
#             func_priority_list=self.func_priority_list,
#             lut_list=self.lut_list
#         )
#
#         self.vm_distribution_arr = updated_vm_distribution_arr
#         self.leftover_resource_time_arr = cumulative_resource_time_arr[:, epoch_length:]
#
#         reward = float(total_water_usage)
#         terminated = (self.current_step >= self.max_steps)
#         truncated = False
#
#         self.prev_action = float(self.multi_binary_to_int(action))
#         self.prev_reward = float(reward)
#
#         obs = self._get_obs()
#         info = {"water_val": total_water_usage}
#         return obs, reward, terminated, truncated, info
#
#     def _get_obs(self):
#
#         obs = np.array([
#             self.prev_action,
#             self.prev_reward,
#             self.current_step
#         ], dtype=np.float32)
#         return obs
#
#     def multi_binary_to_int(self, action):
#         bit_string = ''.join(str(int(a)) for a in action)
#         return int(bit_string, 2)
#

# def train_marl_agents(config=None, total_timesteps=20000, n_envs=3):
#     save_dir = "trained_models"
#     if config is None:
#         config = {}
#
#     env_classes = {
#         "carbon_model": CarbonEnv,
#         "time_model": TimeEnv,
#         "water_model": WaterEnv
#     }
#
#     os.makedirs(save_dir, exist_ok=True)
#
#     trained_models = {}
#     for agent_name, env_class in env_classes.items():
#         print(f"Starting training for {agent_name}...")
#
#         # Create parallel environments
#         parallel_env = make_parallel_env(env_class, config, n_envs)
#         model_path = os.path.join(save_dir, f"{agent_name}.zip")
#         if os.path.exists(model_path):
#             # Load the previously trained model
#             print(f"Loading previously trained model from {model_path}")
#             model = PPO.load(model_path)
#             # Attach parallel environments to the loaded model
#             parallel_env = make_parallel_env(env_class, config, n_envs)
#             model.set_env(parallel_env)
#         else:
#             # Initialize a new model if none exists
#             print("No previously trained model found. Initializing a new model.")
#             parallel_env = make_parallel_env(env_class, config, n_envs)
#             model = PPO("MlpPolicy", parallel_env, verbose=1, device='cpu')
#
#         # Train with progress bar
#         progress_callback = TQDMProgressBarCallback(total_timesteps=total_timesteps)
#         step_tracker_callback = StepTrackingCallback(agent_max_steps=100)
#         model.learn(total_timesteps=total_timesteps, callback=[progress_callback, step_tracker_callback])
#
#         # Store trained model
#         trained_models[agent_name] = model
#
#         parallel_env.close()
#
#         model_path = os.path.join(save_dir, f"{agent_name}.zip")
#         model.save(model_path)
#
#         print(f"Finished training for {agent_name}.\n")
#
#     return trained_models
#
#
# def make_parallel_env(env_class, config, n_env=3):
#     def make_env(seed):
#         def _init():
#             env = env_class(config)
#             env.seed(seed)
#             return env
#         return _init
#     return SubprocVecEnv([make_env(i) for i in range(n_env)])
#
#
# class TQDMProgressBarCallback(BaseCallback):
#     def __init__(self, total_timesteps, verbose=0):
#         super().__init__(verbose)
#         self.total_timesteps = total_timesteps
#         self.pbar = None
#
#     def _on_training_start(self):
#         self.pbar = tqdm(total=self.total_timesteps, desc="Training Progress", unit="steps")
#
#     def _on_step(self):
#         self.pbar.update(self.model.n_envs)  # Update progress bar with the number of environments
#         return True
#
#     def _on_training_end(self):
#         self.pbar.close()
#
# class StepTrackingCallback(BaseCallback):
#     def __init__(self, agent_max_steps, verbose=0):
#         super().__init__(verbose)
#         self.agent_max_steps = agent_max_steps
#         self.steps_taken = 0
#
#     def _on_step(self) -> bool:
#         self.steps_taken += self.training_env.num_envs
#         if self.steps_taken >= self.agent_max_steps:
#             print(f"Stopping training: Agent reached {self.steps_taken} steps (max: {self.agent_max_steps}).")
#             return False  # Stops training
#         return True
#
#
# if __name__ == "__main__":
#     config = {
#         "max_steps": 20
#     }
#     models = train_marl_agents(config, total_timesteps=5000)
#     print("Trained 3 separate PPO models using real simulation() logic.")

