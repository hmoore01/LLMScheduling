import os
from typing import Dict, Any, List, Optional, Callable, Union, Literal

import numpy as np
import pandas as pd
from gymnasium import spaces

from pettingzoo.utils.env import ParallelEnv
from pettingzoo.utils import parallel_to_aec

from supersuit import black_death_v3, pettingzoo_env_to_vec_env_v1, concat_vec_envs_v1

from stable_baselines3 import PPO

from Rate_Flow_Sim import LLM_Simulator

import matplotlib.pyplot as plt

from stable_baselines3.common.callbacks import BaseCallback


class ResourceEnv(ParallelEnv):
    """
    Multi-datacenter PettingZoo ParallelEnv for constrained PPO.

    Each agent = one datacenter: "dc_0", "dc_1", ..., "dc_{N-1}".
    Action (per agent): [logit_7b, logit_70b, power_scalar] in [0,1].
      - First two are interpreted as logits and turned into a *global*
        distribution over DCs for routing Llama7b / Llama70b.
      - power_scalar selects a discrete node-type power pattern.

    The env runs a single epoch per episode by calling the rate-based
    LLM_Simulator from Rate_Flow_Sim and turns its metrics into rewards,
    with optional constraints encoded via a Lagrangian penalty.
    """

    metadata = {"render_modes": ["human"], "name": "ResourceEnv"}

    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        # --- Core config from simulator_LLM / caller ---
        self.epoch_df: pd.DataFrame = config["epoch_df"]
        self.node_properties: Dict[str, Any] = config["node_properties"]
        self.epoch_idx: int = int(config["epoch_idx"])
        self.NUM_DATACENTERS: int = int(config["num_datacenters"])
        self.epoch_summary: Dict[str, Any] = config.get("epoch_summary", {})

        self._epoch_pool: Dict[int, pd.DataFrame] = {}

        if "epoch" in self.epoch_df.columns:
            for e, df_e in self.epoch_df.groupby("epoch"):
                self._epoch_pool[int(e)] = df_e.copy()
        else:
            self._epoch_pool[int(self.epoch_idx)] = self.epoch_df.copy()

        self._available_epochs: List[int] = sorted(self._epoch_pool.keys())

        self.max_steps: int = int(config.get("max_steps", 1))

        # Per-profile spec (weights, constraints, dual settings)
        self.agent_specs: Dict[str, Dict[str, Any]] = config["agent_specs"]
        self.active_agent_profile: str = config["active_agent_profile"]
        if self.active_agent_profile not in self.agent_specs:
            raise KeyError(
                f"Active profile '{self.active_agent_profile}' not found in agent_specs keys "
                f"{list(self.agent_specs.keys())}"
            )
        self.profile: Dict[str, Any] = self.agent_specs[self.active_agent_profile]

        # Reward weights (metrics -> scalar reward)
        self.reward_weights: Dict[str, float] = self._normalize_weights(
            self.profile.get("weights", {"ttft": 1.0})
        )

        # Identify primary metric for this scheme (used for routing/power bias)
        if self.reward_weights:
            self.primary_metric: str = max(self.reward_weights.items(), key=lambda kv: kv[1])[0]
        else:
            self.primary_metric = "ttft"

        # Constraints / duals
        self.constraints: Dict[str, Dict[str, Any]] = self.profile.get("constraints", {})
        self.include_duals_in_obs: bool = bool(self.profile.get("include_duals_in_obs", True))
        self.lambda_lr: Dict[str, float] = self.profile.get("lambda_lr", {})
        self.duals: Dict[str, float] = {}
        lambda_init = self.profile.get("lambda_init", {})
        for cname in self.constraints.keys():
            self.duals[cname] = float(lambda_init.get(cname, 0.0))

        # Basic workload snapshot (used for observations)
        self.llama7b_total: float = float(self.epoch_summary.get("llama7b_total", 0.0))
        self.llama70b_total: float = float(self.epoch_summary.get("llama70b_total", 0.0))

        # PettingZoo agent ids
        self.agents: List[str] = [f"dc_{i}" for i in range(self.NUM_DATACENTERS)]
        self.possible_agents = list(self.agents)

        # Metric tracking for normalization
        self.metric_max_tracker: Dict[str, float] = {
            "ttft": 1e-6,
            "carbon": 1e-6,
            "water": 1e-6,
            "cost": 1e-6,
            "total_energy": 1e-6,
            "network_load": 1e-6,
        }

        # Moving reward scaling (adaptive normalization)
        # Exponential moving average of |reward| so we can rescale to O(1)
        self.reward_scale: float = 1.0
        # Smoothing factor for EMA; can be overridden per profile
        self.reward_scale_alpha: float = float(self.profile.get("reward_scale_alpha", 0.01))
        # Optional clip after scaling (0 disables clipping)
        self.reward_clip: float = float(self.profile.get("reward_clip", 10.0))

        # Internal simulator (lazy init)
        self._rate_sim = None

        # Runtime state
        self.current_step: int = 0
        self._last_metrics = None
        self._last_results = None
        self._last_leftovers = None

        # Action/observation spaces
        self._build_spaces()

        # Per-DC metric profiles and scheme-specific bias
        self._init_dc_metric_profiles()
        self._metric_bias: np.ndarray = self._build_metric_bias_vector()

        self.render_mode = "human"

    # ------------------------------------------------------------------
    # PettingZoo required attributes
    # ------------------------------------------------------------------
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    # ------------------------------------------------------------------
    # Helper: normalize weights
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
        total = float(sum(abs(v) for v in weights.values()))
        if total <= 0:
            return weights
        return {k: float(v) / total for k, v in weights.items()}

    # ------------------------------------------------------------------
    # Build action/observation spaces
    # ------------------------------------------------------------------
    def _build_spaces(self):
        """
        Build action and observation spaces.

        Action per agent:
          [logit_7b, logit_70b, power_scalar]

        - logit_7b, logit_70b: in [-5, 5], fed into a softmax over DCs.
          This gives enough dynamic range for sharp or flat distributions.
        - power_scalar: in [0, 1], used to pick a discrete power pattern.
        """
        act_space = spaces.Box(
            low=np.array([-5.0, -5.0, 0.0], dtype=np.float32),
            high=np.array([5.0, 5.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Observation: [workload features, last metrics, duals, agent one-hot]
        #   workload: llama7b share, llama70b share
        #   last metrics: 4 (ttft, carbon, water, cost) + total_energy + network_load
        base_obs_dim = 2 + 6

        dual_dim = len(self.constraints) if self.include_duals_in_obs else 0
        agent_id_dim = self.NUM_DATACENTERS

        obs_dim = base_obs_dim + dual_dim + agent_id_dim

        obs_space = spaces.Box(
            low=-np.inf * np.ones(obs_dim, dtype=np.float32),
            high=np.inf * np.ones(obs_dim, dtype=np.float32),
            dtype=np.float32,
        )

        self.action_spaces = {agent: act_space for agent in self.agents}
        self.observation_spaces = {agent: obs_space for agent in self.agents}

    # ------------------------------------------------------------------
    # Internal helpers: per-DC metric profiles and routing bias
    # ------------------------------------------------------------------
    def _init_dc_metric_profiles(self) -> None:
        """Initialize simple per-DC carbon / water / price profiles.

        We try to pull arrays from epoch_summary if present, otherwise fall
        back to a monotonic pattern over DC index so that schemes at least
        see some heterogeneity (which helps different reward weightings
        learn different policies).
        """
        num_dc = int(self.NUM_DATACENTERS)
        idxs = np.arange(num_dc, dtype=np.float32)

        # Carbon intensity per DC (gCO2/kWh or relative units)
        ci_raw = self.epoch_summary.get("dc_carbon_intensity", None)
        if isinstance(ci_raw, (list, tuple, np.ndarray)) and len(ci_raw) == num_dc:
            self._dc_carbon = np.asarray(ci_raw, dtype=np.float32)
        else:
            # Fallback: arbitrary but monotonic gradient (higher index -> "cleaner")
            self._dc_carbon = 1.0 + (idxs / max(1, num_dc - 1))

        # Water intensity per DC (L/kWh or relative units)
        water_raw = self.epoch_summary.get("dc_water_intensity", None)
        if isinstance(water_raw, (list, tuple, np.ndarray)) and len(water_raw) == num_dc:
            self._dc_water = np.asarray(water_raw, dtype=np.float32)
        else:
            # Fallback: slightly favor higher-index DCs for water efficiency
            self._dc_water = 1.0 + 0.5 * (idxs / max(1, num_dc - 1))

        # Energy price per DC ($/kWh or relative units)
        price_raw = self.epoch_summary.get("dc_energy_price", None)
        if isinstance(price_raw, (list, tuple, np.ndarray)) and len(price_raw) == num_dc:
            self._dc_price = np.asarray(price_raw, dtype=np.float32)
        else:
            # Fallback: simple increasing price with index
            self._dc_price = 1.0 + (idxs / max(1, num_dc - 1))

    def _build_metric_bias_vector(self) -> np.ndarray:
        """Return a per-DC multiplicative bias based on primary_metric.

        This bias is deliberately mild: it is *multiplicative* on top of the
        learned base distribution and the locality bias, so PPO can still
        override it, but schemes with different primary metrics naturally
        gravitate toward different DCs.
        """
        num_dc = int(self.NUM_DATACENTERS)
        ones = np.ones(num_dc, dtype=np.float32)

        metric = getattr(self, "primary_metric", "ttft")

        if metric == "carbon":
            # Prefer lower-carbon DCs: bias ∝ 1 / CI
            ci = np.maximum(self._dc_carbon, 1e-3)
            bias = 1.0 / ci
        elif metric == "water":
            # Prefer lower-water DCs
            w = np.maximum(self._dc_water, 1e-3)
            bias = 1.0 / w
        elif metric in ("cost", "energy_cost", "price"):
            # Prefer cheaper DCs
            p = np.maximum(self._dc_price, 1e-3)
            bias = 1.0 / p
        elif metric == "ttft":
            # Simple "fast vs slow" prior: prefer lower-index DCs
            idxs = np.arange(num_dc, dtype=np.float32)
            # Map idxs in [0, N-1] -> bias in [1.0, 2.0]
            bias = 2.0 - (idxs / max(1, num_dc - 1))
        else:
            bias = ones

        # Normalize so that average bias is ~1.0 (keeps behavior well-scaled)
        mean = float(bias.mean()) if bias.size > 0 else 1.0
        if mean <= 0.0 or not np.isfinite(mean):
            return ones
        return (bias / mean).astype(np.float32)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        # Keep the PettingZoo agent list alive
        self.agents = list(self.possible_agents)
        self.current_step = 0

        # Clear last metrics
        self._last_metrics = None
        self._last_results = None
        self._last_leftovers = None

        # Reset Lagrange multipliers and reward scale each episode
        for cname in self.duals.keys():
            self.duals[cname] = 0.0
        self.reward_scale = 1.0

        # ------------------------------------------------------
        # Randomly select an epoch from the pool (if >1 available)
        # ------------------------------------------------------
        if hasattr(self, "_epoch_pool") and self._epoch_pool:
            # Uniform random choice; you could bias this if desired
            chosen = int(np.random.choice(self._available_epochs))
            self.epoch_idx = chosen
            self.epoch_df = self._epoch_pool[chosen].copy()

            # Update basic workload snapshot for observations
            if (
                "model_type" in self.epoch_df.columns
                and "num_tokens" in self.epoch_df.columns
            ):
                self.llama7b_total = float(
                    self.epoch_df[self.epoch_df["model_type"] == "Llama7b"]["num_tokens"].sum()
                )
                self.llama70b_total = float(
                    self.epoch_df[self.epoch_df["model_type"] == "Llama70b"]["num_tokens"].sum()
                )
            # else: keep whatever was in epoch_summary as a fallback
        # else: single-epoch case; self.epoch_df / epoch_idx already set

        # (Optional) recompute metric bias in case epoch_summary changed upstream
        self._init_dc_metric_profiles()
        self._metric_bias = self._build_metric_bias_vector()

        # Initial observation after choosing the epoch
        obs = self._get_obs_dict()
        infos = {agent: {} for agent in self.agents}
        print(
            f"[ResourceEnv] Starting episode with epoch_idx={self.epoch_idx}, "
            f"total rows={len(self.epoch_df)}"
        )
        return obs, infos

    def _softmax_across_dcs(self, logits_per_agent: Dict[str, float]) -> np.ndarray:
        """Softmax over DC agents for a single model."""
        vals = np.array([logits_per_agent[a] for a in self.agents], dtype=np.float64)
        vals = np.clip(vals, -50.0, 50.0)
        vals -= np.max(vals)
        ex = np.exp(vals)
        s = ex.sum()
        if s <= 0 or not np.isfinite(s):
            return np.full(len(self.agents), 1.0 / len(self.agents), dtype=np.float64)
        return ex / s

    def _apply_safety_layer(self, normalized_actions: Dict[str, np.ndarray]):
        """
        Placeholder safety layer: currently a no-op that just returns the
        actions unchanged. You can extend this to project onto a feasible set
        if you want hard per-step guarantees.
        """
        flags = {agent: False for agent in normalized_actions.keys()}
        return normalized_actions, flags

    # ------------------------------------------------------------------
    # Moving reward scaling helper
    # ------------------------------------------------------------------
    def _scale_reward(self, raw_reward: float) -> float:
        """Apply exponential moving scaling + optional clipping."""
        alpha = self.reward_scale_alpha
        abs_r = abs(raw_reward)

        # Initialize scale sensibly if very small
        if self.reward_scale <= 1e-6:
            self.reward_scale = max(abs_r, 1.0)
        else:
            self.reward_scale = (1.0 - alpha) * self.reward_scale + alpha * max(abs_r, 1.0)

        scaled = raw_reward / max(self.reward_scale, 1e-6)

        if self.reward_clip > 0.0:
            scaled = float(np.clip(scaled, -self.reward_clip, self.reward_clip))

        return float(scaled)

    def step(self, actions: Dict[str, np.ndarray]):
        assert set(actions.keys()) == set(self.agents), (
            f"Action keys {list(actions.keys())} do not match agents {self.agents}"
        )
        self.current_step += 1

        # ------------------------------------------------------------------
        # 1) Decode actions: logits -> global routing dists, power scalars
        # ------------------------------------------------------------------
        raw = {a: np.asarray(actions[a], dtype=np.float32).copy() for a in self.agents}

        # First two dims: logits for 7B / 70B (in [-5,5])
        logits_7b = {a: float(raw[a][0]) for a in self.agents}
        logits_70b = {a: float(raw[a][1]) for a in self.agents}
        # Third dim: power scalar in [0,1]
        power_scalars = {a: float(np.clip(raw[a][2], 0.0, 1.0)) for a in self.agents}

        # Global distributions over DCs (one per model)
        dist_7b = self._softmax_across_dcs(logits_7b)  # shape [NUM_DATACENTERS]
        dist_70b = self._softmax_across_dcs(logits_70b)

        # Align dists with agents order, attach power scalar
        normalized_actions: Dict[str, np.ndarray] = {}
        for idx, agent in enumerate(self.agents):
            normalized_actions[agent] = np.array(
                [dist_7b[idx], dist_70b[idx], power_scalars[agent]], dtype=np.float32
            )

        # Optional safety layer (currently a no-op but keeps plumbing + flags)
        projected_actions, projection_flags = self._apply_safety_layer(normalized_actions)

        # Replace dist_7b / dist_70b with the possibly adjusted values
        dist_7b = np.array([projected_actions[a][0] for a in self.agents], dtype=np.float64)
        dist_70b = np.array([projected_actions[a][1] for a in self.agents], dtype=np.float64)
        power_scalars = {a: float(projected_actions[a][2]) for a in self.agents}

        # ------------------------------------------------------------------
        # 2) Build rate-based workload from epoch_df
        # ------------------------------------------------------------------
        df = self.epoch_df

        # Be robust to different column names
        if "source_dc_id" in df.columns:
            src_col = "source_dc_id"
        elif "source_dc" in df.columns:
            src_col = "source_dc"
        elif "src_dc" in df.columns:
            src_col = "src_dc"
        else:
            raise KeyError(
                "epoch_df must have one of ['source_dc_id', 'source_dc', 'src_dc']"
            )

        if "model_type" in df.columns:
            model_col = "model_type"
        elif "model" in df.columns:
            model_col = "model"
        else:
            raise KeyError(
                "epoch_df must have one of ['model_type', 'model']"
            )

        if "num_tokens" in df.columns:
            tok_col = "num_tokens"
        elif "total_tokens" in df.columns:
            tok_col = "total_tokens"
        elif "tokens" in df.columns:
            tok_col = "tokens"
        else:
            raise KeyError(
                "epoch_df must have one of ['num_tokens', 'total_tokens', 'tokens']"
            )

        # Aggregate tokens per (src_dc, model_type) for this epoch
        work_df = (
            df.groupby([src_col, model_col], as_index=False)[tok_col]
            .sum()
            .rename(
                columns={
                    src_col: "src_dc",
                    model_col: "model_type",
                    tok_col: "total_tokens",
                }
            )
        )

        # ------------------------------------------------------------------
        # 3) Build workload_df + schedule_plan for LLM_Simulator
        #    Fractional routing, biased toward nearest DC index + scheme bias.
        # ------------------------------------------------------------------
        req_rows: List[Dict[str, Any]] = []
        plan_map: Dict[int, int] = {}
        row_idx = 0

        num_dc = self.NUM_DATACENTERS
        max_dc_distance = max(1, num_dc - 1)

        for r in work_df.itertuples(index=False):
            src_dc = int(getattr(r, "src_dc"))
            model = str(getattr(r, "model_type"))
            tokens = float(getattr(r, "total_tokens"))

            if tokens <= 0.0:
                continue

            # Base global distribution for this model from the agents
            if model == "Llama7b":
                base_dist = np.asarray(dist_7b, dtype=np.float64).copy()
            else:
                base_dist = np.asarray(dist_70b, dtype=np.float64).copy()

            # Apply proximity bias + scheme-specific metric bias.
            weights = np.zeros_like(base_dist)
            for dc_id in range(num_dc):
                dist_idx = abs(dc_id - src_dc)
                # closeness in [0.5, 1.0]; tweak 0.5 for stronger/weaker bias
                closeness = 1.0 - 0.5 * (dist_idx / max_dc_distance)
                closeness = max(closeness, 0.0)
                weights[dc_id] = (
                    base_dist[dc_id]
                    * closeness
                    * float(self._metric_bias[dc_id])
                )

            total_w = float(weights.sum())
            if total_w <= 0.0 or not np.isfinite(total_w):
                # Fallback: uniform if something degenerates
                weights[:] = 1.0 / num_dc
            else:
                weights /= total_w

            # Split this (src_dc, model) workload across DCs using the biased weights
            for dc_id in range(num_dc):
                share = float(weights[dc_id])
                if share <= 0.0:
                    continue

                token_share = tokens * share
                if token_share <= 0.0:
                    continue

                req_rows.append(
                    {
                        "source_dc": src_dc,
                        "model": model,
                        "arrival_ms": 0,
                        "tokens": token_share,
                    }
                )
                plan_map[row_idx] = int(dc_id)
                row_idx += 1

        # Failsafe: if no rows were generated, fall back to src->src
        if not req_rows:
            for r in work_df.itertuples(index=False):
                src_dc = int(getattr(r, "src_dc"))
                model = str(getattr(r, "model_type"))
                tokens = float(getattr(r, "total_tokens"))
                if tokens <= 0.0:
                    continue
                req_rows.append(
                    {
                        "source_dc": src_dc,
                        "model": model,
                        "arrival_ms": 0,
                        "tokens": tokens,
                    }
                )
                plan_map[row_idx] = src_dc
                row_idx += 1

        workload_df = pd.DataFrame(req_rows)
        self.schedule_plan = {"map": plan_map}

        # ------------------------------------------------------------------
        # 4) Build power_plan from projected power scalars
        # ------------------------------------------------------------------
        # Patterns over node types (indexes must match DC node_ids)
        power_patterns = [
            [0, 0, 0, 0, 0, 0],  # all off
            [1, 0, 0, 0, 0, 0],  # 8_A100s
            [0, 1, 0, 0, 0, 0],  # 8_H100s
            [0, 0, 1, 0, 0, 0],  # 4_A100s
            [0, 0, 0, 1, 0, 0],  # 4_H100s
            [1, 1, 0, 0, 0, 0],  # both 8-GPU types
            [0, 0, 1, 1, 0, 0],  # both 4-GPU types
            [1, 1, 1, 1, 0, 0],  # all big GPU types
        ]
        num_patterns = len(power_patterns)

        # Translate pattern into per-DC plan slice compatible with Datacenter.apply_power_plan
        self.power_plan: Dict[int, Dict[str, Dict[int, str]]] = {}
        for agent in self.agents:
            dc_id = int(agent.split("_")[1])
            lever_val = float(np.clip(power_scalars[agent], 0.0, 1.0))

            # Scheme-specific skew on power lever:
            #  - ttft-like schemes push toward higher patterns
            #  - carbon/water/cost-like schemes push toward lower patterns
            metric = getattr(self, "primary_metric", "ttft")
            if metric == "ttft":
                # push toward higher patterns
                lever_val = 0.5 + 0.5 * lever_val
            elif metric in ("carbon", "water", "cost", "energy_cost", "price"):
                # push toward lower patterns
                lever_val = 0.5 * lever_val

            lever_val = float(np.clip(lever_val, 0.0, 1.0))
            idx = min(int(lever_val * num_patterns), num_patterns - 1)
            node_pattern = power_patterns[idx]

            self.power_plan[dc_id] = {
                "unit": {
                    node_type: ("ON" if on else "OFF")
                    for node_type, on in enumerate(node_pattern)
                }
            }

        # ------------------------------------------------------------------
        # 5) Call rate-based LLM_Simulator for this epoch
        # ------------------------------------------------------------------
        if self._rate_sim is None:
            # epoch_summary should carry these; fall back to sensible defaults
            spec_dir = self.epoch_summary.get("spec_dir", "sim_specs")
            epoch_len = int(self.epoch_summary.get("epoch_length", 900))
            self._rate_sim = LLM_Simulator(
                spec_dir=spec_dir,
                epoch_length=epoch_len,
                debug=False,
            )

        metrics, results, dc_usage = self._rate_sim.run_epoch(
            self.epoch_idx,
            workload_df,
            self.schedule_plan,
            self.power_plan,
        )
        self._last_metrics = dict(metrics)
        self._last_results = results
        self._last_leftovers = dc_usage

        # ------------------------------------------------------------------
        # 6) Turn metrics into normalized reward + constraint penalty
        # ------------------------------------------------------------------
        ttft = float(metrics.get("avg_ttft_sec", metrics.get("avg_ttft", 0.0)))
        carbon = float(metrics.get("carbon_emissions", 0.0))
        water = float(metrics.get("water_usage", 0.0))
        cost = float(metrics.get("energy_cost", 0.0))
        total_energy = float(metrics.get("total_energy", metrics.get("energy_kwh", 0.0)))
        network_load = float(metrics.get("avg_net_latency_ms", 0.0))

        # Update max trackers for moving normalization
        self.metric_max_tracker["ttft"] = max(self.metric_max_tracker["ttft"], ttft)
        self.metric_max_tracker["carbon"] = max(self.metric_max_tracker["carbon"], carbon)
        self.metric_max_tracker["water"] = max(self.metric_max_tracker["water"], water)
        self.metric_max_tracker["cost"] = max(self.metric_max_tracker["cost"], cost)
        self.metric_max_tracker["total_energy"] = max(
            self.metric_max_tracker["total_energy"], total_energy
        )
        self.metric_max_tracker["network_load"] = max(
            self.metric_max_tracker["network_load"], network_load
        )

        def inv_norm(x: float, key: str) -> float:
            denom = max(self.metric_max_tracker[key], 1e-12)
            # larger x -> smaller normalized score in [0,1]
            return float(np.clip(1.0 - (x / denom), 0.0, 1.0))

        global_norm = {
            "ttft": inv_norm(ttft, "ttft"),
            "carbon": inv_norm(carbon, "carbon"),
            "water": inv_norm(water, "water"),
            "cost": inv_norm(cost, "cost"),
            "total_energy": inv_norm(total_energy, "total_energy"),
            "network_load": inv_norm(network_load, "network_load"),
        }

        # Base reward from weights
        base_rw = 0.0
        for m, w in self.reward_weights.items():
            base_rw += float(w) * float(global_norm.get(m, 0.0))

        # Constraints: Lagrangian penalty
        constraint_costs = {
            "ttft": ttft,
            "carbon": carbon,
            "water": water,
            "cost": cost,
            "total_energy": total_energy,
            "network_load": network_load,
        }
        total_penalty = 0.0
        for cname, rule in self.constraints.items():
            val = float(constraint_costs.get(cname, 0.0))
            bound = float(rule.get("bound", rule.get("budget", 0.0)))
            ctype = rule.get("type", "upper_bound")
            if ctype == "upper_bound":
                viol = max(0.0, val - bound)
            elif ctype == "lower_bound":
                viol = max(0.0, bound - val)
            else:
                viol = 0.0
            lr = float(self.lambda_lr.get(cname, 0.0))
            self.duals[cname] = max(0.0, self.duals.get(cname, 0.0) + lr * viol)
            total_penalty += self.duals.get(cname, 0.0) * viol

        # Moving reward scaling
        raw_reward = float(base_rw - total_penalty)
        final_reward = self._scale_reward(raw_reward)

        # ------------------------------------------------------------------
        # 7) Build outputs for all agents (single-step episode)
        # ------------------------------------------------------------------
        rewards = {agent: final_reward for agent in self.agents}
        terminations = {agent: True for agent in self.agents}
        truncations = {agent: False for agent in self.agents}

        infos = {
            agent: {
                "raw_metrics": {
                    "ttft": ttft,
                    "carbon_emissions": carbon,
                    "water_usage": water,
                    "energy_cost": cost,
                    "total_energy": total_energy,
                    "network_load": network_load,
                },
                "metrics_normalized": global_norm,
                "schedule_plan": self.schedule_plan,
                "power_plan": self.power_plan,
                "projected": bool(projection_flags.get(agent, False)),
                "duals": dict(self.duals),
                "dc_usage": dc_usage,
                "reward_raw": raw_reward,
                "reward_scaled": final_reward,
                "reward_scale": self.reward_scale,
            }
            for agent in self.agents
        }

        obs = self._get_obs_dict()
        return obs, rewards, terminations, truncations, infos

    # ------------------------------------------------------------------
    # Observation helper
    # ------------------------------------------------------------------
    def _get_obs_dict(self) -> Dict[str, np.ndarray]:
        obs: Dict[str, np.ndarray] = {}
        # Basic workload snapshot, normalized to something reasonable
        total_tokens = self.llama7b_total + self.llama70b_total
        norm_7b = float(self.llama7b_total / (total_tokens + 1e-9))
        norm_70b = float(self.llama70b_total / (total_tokens + 1e-9))

        # Last metrics normalized (or zeros if none yet)
        if self._last_metrics is None:
            last_ttft = last_carbon = last_water = last_cost = 0.0
            last_total_energy = last_network_load = 0.0
        else:
            last_ttft = float(
                self._last_metrics.get("avg_ttft_sec", self._last_metrics.get("avg_ttft", 0.0))
            )
            last_carbon = float(self._last_metrics.get("carbon_emissions", 0.0))
            last_water = float(self._last_metrics.get("water_usage", 0.0))
            last_cost = float(self._last_metrics.get("energy_cost", 0.0))
            last_total_energy = float(
                self._last_metrics.get("total_energy", self._last_metrics.get("energy_kwh", 0.0))
            )
            last_network_load = float(self._last_metrics.get("avg_net_latency_ms", 0.0))

        last_features = np.array(
            [
                last_ttft,
                last_carbon,
                last_water,
                last_cost,
                last_total_energy,
                last_network_load,
            ],
            dtype=np.float32,
        )

        # Duals (if included)
        dual_vec = np.zeros(len(self.constraints), dtype=np.float32)
        if self.include_duals_in_obs and self.constraints:
            for i, cname in enumerate(self.constraints.keys()):
                dual_vec[i] = float(self.duals.get(cname, 0.0))

        # Build per-agent observation
        for dc_idx, agent in enumerate(self.agents):
            agent_one_hot = np.zeros(self.NUM_DATACENTERS, dtype=np.float32)
            agent_one_hot[dc_idx] = 1.0

            vec = np.concatenate(
                [
                    np.array([norm_7b, norm_70b], dtype=np.float32),
                    last_features,
                    dual_vec,
                    agent_one_hot,
                ],
                axis=0,
            )
            obs[agent] = vec

        return obs

    # ------------------------------------------------------------------
    # Accessor for last metrics (used after running an episode)
    # ------------------------------------------------------------------
    def get_last_metrics(self) -> Optional[Dict[str, Any]]:
        return self._last_metrics




# ======================================================================
# Inference helper used by simulator_LLM
# ======================================================================

def run_multiagent(
    epoch_df: pd.DataFrame,
    epoch_summary: Dict[str, Any],
    epoch_idx: int,
    node_properties: Dict[str, Any],
    model_base_path: str = "trained_models/sb3_agents",
) -> Dict[str, Dict[str, Any]]:
    """
    Run all trained MARL profiles for a single epoch and return
    profile_id -> metrics dict.

    Mirrors the training env construction:
        raw_env      = ResourceEnv(cfg)
        death_wrapped = black_death_v3(raw_env)
        vec_env      = pettingzoo_env_to_vec_env_v1(death_wrapped)
        venv         = concat_vec_envs_v1(vec_env, num_vec_envs=1, base_class="stable_baselines3")

    NOTE: Episodes are single-step, so we do exactly one predict+step.
    """

    # --- Determine which profiles to run ---
    try:
        from simulator_LLM import build_agent_specs  # defines profiles

        # Infer number of datacenters
        if isinstance(epoch_summary, dict):
            if "num_datacenters" in epoch_summary:
                num_dc = int(epoch_summary["num_datacenters"])
            elif "datacenters" in epoch_summary:
                num_dc = int(len(epoch_summary["datacenters"]))
            else:
                try:
                    num_dc = int(epoch_df["source_dc_id"].max()) + 1
                except Exception:
                    num_dc = 12
        else:
            try:
                num_dc = int(epoch_df["source_dc_id"].max()) + 1
            except Exception:
                num_dc = 12

        agent_specs = build_agent_specs(num_datacenters=num_dc)
        profile_ids = list(agent_specs.keys())
    except Exception:
        # Fallback if build_agent_specs is unavailable
        num_dc = 12
        agent_specs = {
            "time_agent":   {"weights": {"ttft": 10}},
            "carbon_agent": {"weights": {"carbon": 10}},
            "water_agent":  {"weights": {"water": 10}},
            "cost_agent":   {"weights": {"cost": 10}},
        }
        profile_ids = list(agent_specs.keys())

    results: Dict[str, Dict[str, Any]] = {}

    for profile_id in profile_ids:
        # Look for either <profile>.zip or <profile>/final_model.zip
        model_path = os.path.join(model_base_path, f"{profile_id}.zip")
        if not os.path.isfile(model_path):
            alt_path = os.path.join(model_base_path, profile_id, "final_model.zip")
            if os.path.isfile(alt_path):
                model_path = alt_path
            else:
                # Skip profiles without a trained model
                continue

        print(f"[MARL EVAL] Running profile '{profile_id}' on epoch {epoch_idx}")

        config = {
            "epoch_df": epoch_df,
            "node_properties": node_properties,
            "epoch_idx": epoch_idx,
            "num_datacenters": num_dc,
            "epoch_summary": epoch_summary,
            "agent_specs": agent_specs,
            "active_agent_profile": profile_id,
            "max_steps": 1,
        }

        # Build env same as in training
        raw_env = ResourceEnv(config)
        death_wrapped = black_death_v3(raw_env)
        vec_env = pettingzoo_env_to_vec_env_v1(death_wrapped)
        venv = concat_vec_envs_v1(
            vec_env,
            num_vec_envs=1,
            num_cpus=1,
            base_class="stable_baselines3",
        )

        # Force model to CPU for evaluation (simple & avoids CUDA issues)
        model = PPO.load(model_path, env=venv, device="cpu")

        # Single-step episode: reset -> predict -> step once
        obs = venv.reset()
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = venv.step(action)

        # After episode, ResourceEnv cached metrics on the underlying raw_env
        metrics = raw_env.get_last_metrics()
        if metrics is None or metrics == {}:
            # If that failed, fall back to parsing `info` from the VecEnv
            # (the env puts per-agent metrics under "raw_metrics")
            extracted = None

            # VecEnv usually returns list[dict] for info; handle both list/dict
            if isinstance(info, (list, tuple)) and len(info) > 0:
                info0 = info[0]
            else:
                info0 = info

            if isinstance(info0, dict):
                # Case A: flattened "raw_metrics" at top level
                if "raw_metrics" in info0:
                    extracted = info0["raw_metrics"]
                else:
                    # Case B: dict of per-agent infos
                    for v in info0.values():
                        if isinstance(v, dict) and "raw_metrics" in v:
                            extracted = v["raw_metrics"]
                            break

            if extracted is not None:
                rm = dict(extracted)
                # Normalize into the keys simulator_LLM expects
                metrics = {
                    "avg_ttft_sec": float(rm.get("ttft", 0.0)),
                    "avg_ttft": float(rm.get("ttft", 0.0)),
                    "carbon_emissions": float(rm.get("carbon_emissions", 0.0)),
                    "water_usage": float(rm.get("water_usage", 0.0)),
                    "energy_cost": float(rm.get("energy_cost", 0.0)),
                    "total_energy": float(rm.get("total_energy", rm.get("energy_kwh", 0.0))),
                    "avg_net_latency_ms": float(rm.get("network_load", 0.0)),
                }
            else:
                # Still nothing – last resort: an empty dict so caller doesn't see None
                print("[MARL EVAL] WARNING: Could not extract metrics from env or info for "
                      f"profile '{profile_id}' on epoch {epoch_idx}")
                metrics = {}

        results[profile_id] = metrics

        venv.close()
        print(f"[MARL EVAL] Finished profile '{profile_id}'")
        results[profile_id] = metrics

        venv.close()
        print(f"[MARL EVAL] Finished profile '{profile_id}'")

    return results





# ======================================================================
# Framework entrypoint for simulator_LLM
# ======================================================================

def milp_optimizer(
    epoch_data,
    epoch_idx: int,
    node_properties: Dict[str, Any],
    epoch_summary: Dict[str, Any],
):
    """
    Make MARL look like a traditional "framework" to simulator_LLM.

    This is called once per epoch, runs each trained profile via run_multiagent,
    picks a preferred profile's metrics to report, and returns the standard
    (stats, results, leftovers) triple expected by simulator_LLM.
    """
    if isinstance(epoch_data, pd.DataFrame):
        df = epoch_data
    else:
        df = pd.DataFrame(epoch_data)

    profile_metrics = run_multiagent(
        df,
        epoch_summary,
        epoch_idx,
        node_properties,
    )

    if not profile_metrics:
        empty = {
            "processed_tokens": 0.0,
            "avg_ttft_sec": 0.0,
            "avg_ttft": 0.0,
            "energy_kwh": 0.0,
            "total_energy": 0.0,
            "energy_cost": 0.0,
            "carbon_emissions": 0.0,
            "water_usage": 0.0,
        }
        # was: return empty, [], {}
        return empty, {}, {}

    # Preference order when reporting framework stats
    preference = [
        "green_perf",
        "cost_guard",
        "water_saver",
        "peak_power_guard",
        "time_agent",
        "carbon_agent",
        "water_agent",
        "cost_agent",
    ]
    chosen_id = None
    for name in preference:
        if name in profile_metrics:
            chosen_id = name
            break
    if chosen_id is None:
        chosen_id = next(iter(profile_metrics.keys()))

    stats = dict(profile_metrics.get(chosen_id, {}))

    # Normalize key names so simulator_LLM aggregation works
    if "avg_ttft_sec" not in stats and "avg_ttft" in stats:
        stats["avg_ttft_sec"] = float(stats["avg_ttft"])
    if "avg_ttft" not in stats and "avg_ttft_sec" in stats:
        stats["avg_ttft"] = float(stats["avg_ttft_sec"])

    if "energy_kwh" in stats and "total_energy" not in stats:
        stats["total_energy"] = float(stats["energy_kwh"])

    # was: return stats, [], {}
    return stats, profile_metrics, {}


# ======================================================================
# Simple training helper
# ======================================================================

def make_training_vec_env(
    profile_id: str,
    agent_specs: Dict[str, Dict[str, Any]],
    epoch_df: pd.DataFrame,
    epoch_summary: Dict[str, Any],
    epoch_idx: int,
    node_properties: Dict[str, Any],
    num_datacenters: int,
    num_envs: int = 1,
):
    """
    Build a vectorized SB3-compatible env for a single profile.

    Pattern requested by user:
        raw_env      = ResourceEnv(cfg)
        death_wrapped = black_death_v3(raw_env)
        vec_env      = pettingzoo_env_to_vec_env_v1(death_wrapped)
        venv         = concat_vec_envs_v1(vec_env, num_vec_envs=..., base_class='stable_baselines3')
    """
    if not isinstance(epoch_df, pd.DataFrame):
        epoch_df = pd.DataFrame(epoch_df)

    cfg = {
        "epoch_df": epoch_df,
        "node_properties": node_properties,
        "epoch_idx": epoch_idx,
        "num_datacenters": num_datacenters,
        "epoch_summary": epoch_summary,
        "agent_specs": agent_specs,
        "active_agent_profile": profile_id,
        "max_steps": 1,
    }

    # Base ParallelEnv
    raw_env = ResourceEnv(cfg)

    # Black-death wrapper so agents that “die” still produce valid obs/rewards
    death_wrapped = black_death_v3(raw_env)

    # PettingZoo -> SB3 VecEnv
    vec_env = pettingzoo_env_to_vec_env_v1(death_wrapped)

    # Optionally replicate to multiple vector envs
    if num_envs > 1:
        vec_env = concat_vec_envs_v1(
            vec_env,
            num_vec_envs=num_envs,
            num_cpus=num_envs,
            base_class="stable_baselines3",
        )
    else:
        # Keep interface symmetric: still return a VecEnv
        vec_env = concat_vec_envs_v1(
            vec_env,
            num_vec_envs=1,
            num_cpus=1,
            base_class="stable_baselines3",
        )

    return vec_env



# ======================================================================
# PPO training helpers (used by simulator_LLM.train_marl_constrained_profiles)
# ======================================================================

def train_reward_scheme(
    profile_id: str,
    *,
    epoch_df: pd.DataFrame,
    epoch_summary: Dict[str, Any],
    epoch_idx: int,
    node_properties: Dict[str, Any],
    agent_specs: Dict[str, Dict[str, Any]],
    num_datacenters: int,
    total_timesteps: int = 100_000,
    num_envs: int = 1,
    overwrite_existing: bool = False,
    model_dir: str = "trained_models/sb3_agents",
) -> None:
    """
    Train a single MARL profile (e.g. 'green_perf') on ResourceEnv using PPO.

    - Keeps ResourceEnv truly multi-agent (one agent per DC).
    - Uses Supersuit's PettingZoo -> VecEnv adapter.
    - Logs and plots reward/value curves via RewardPlotCallback.
    """
    os.makedirs(model_dir, exist_ok=True)

    # Paths for saving
    root_model_path = os.path.join(model_dir, f"{profile_id}.zip")
    profile_dir = os.path.join(model_dir, profile_id)
    final_model_path = os.path.join(profile_dir, "final_model.zip")

    if not overwrite_existing and (os.path.exists(root_model_path) or os.path.exists(final_model_path)):
        print(f"[MARL TRAIN] {profile_id}: model exists, skipping (overwrite_existing=False).")
        return

    os.makedirs(profile_dir, exist_ok=True)

    print(f"[MARL TRAIN] Training profile '{profile_id}' for {total_timesteps} timesteps")

    # Build vectorized multi-agent env
    vec_env = make_training_vec_env(
        profile_id=profile_id,
        agent_specs=agent_specs,
        epoch_df=epoch_df,
        epoch_summary=epoch_summary,
        epoch_idx=epoch_idx,
        node_properties=node_properties,
        num_datacenters=num_datacenters,
        num_envs=num_envs,
    )

    # Create PPO model
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
    )

    # Attach callback to log and plot reward/value
    plot_cb = RewardPlotCallback(
        profile_id=profile_id,
        out_dir=profile_dir,
        verbose=1,
    )

    # Train
    model.learn(total_timesteps=total_timesteps, callback=plot_cb)

    # Save models
    model.save(root_model_path)
    model.save(final_model_path)

    vec_env.close()
    print(f"[MARL TRAIN] Finished '{profile_id}'. Saved to:")
    print(f"  - {root_model_path}")
    print(f"  - {final_model_path}")
    print(f"  - reward/value plot: {os.path.join(profile_dir, profile_id + '_reward_plot.png')}")



def train_all_schemes(
    *,
    epoch_df: pd.DataFrame,
    epoch_summary: Dict[str, Any],
    epoch_idx: int,
    node_properties: Dict[str, Any],
    agent_specs: Dict[str, Dict[str, Any]],
    num_datacenters: int,
    total_timesteps: int = 100_000,
    num_envs: int = 1,
    overwrite_existing: bool = False,
    model_dir: str = "trained_models/sb3_agents",
) -> None:
    """
    Train PPO for *all* provided profiles in agent_specs, sequentially.

    This is what simulator_LLM.train_marl_constrained_profiles(...) calls.
    It keeps the training loop multi-agent by using ResourceEnv via Supersuit.
    """
    # Make sure epoch_df is a DataFrame
    if not isinstance(epoch_df, pd.DataFrame):
        epoch_df = pd.DataFrame(epoch_df)

    profile_ids = list(agent_specs.keys())
    print(f"[MARL TRAIN] Using epoch {epoch_idx} for training data")
    print(f"[MARL TRAIN] Profiles to train: {profile_ids}")

    for pid in profile_ids:
        spec = agent_specs[pid]
        # You could skip unconstrained or specific profiles here if desired
        train_reward_scheme(
            profile_id=pid,
            epoch_df=epoch_df,
            epoch_summary=epoch_summary,
            epoch_idx=epoch_idx,
            node_properties=node_properties,
            agent_specs=agent_specs,
            num_datacenters=num_datacenters,
            total_timesteps=total_timesteps,
            num_envs=num_envs,
            overwrite_existing=overwrite_existing,
            model_dir=model_dir,
        )

    print("[MARL TRAIN] Completed training for all profiles in agent_specs.")


class RewardPlotCallback(BaseCallback):
    """
    Collects mean reward, optional raw reward (from infos['reward_raw']),
    and critic value estimates over time, and saves a plot at the end
    of training.

    Works with SB3 + Supersuit PettingZoo wrapper.
    """

    def __init__(self, profile_id: str, out_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self.profile_id = profile_id
        self.out_dir = out_dir

        self.timesteps = []
        self.mean_rewards = []
        self.mean_raw_rewards = []
        self.mean_values = []

    def _on_step(self) -> bool:
        # Called after each environment step
        t = self.num_timesteps

        # rewards: shape (n_envs,)
        rewards = self.locals.get("rewards", None)
        if rewards is not None:
            rewards = np.array(rewards, dtype=np.float32)
            if rewards.size > 0:
                self.timesteps.append(t)
                self.mean_rewards.append(float(rewards.mean()))
                self.logger.record("train/mean_reward_scaled", self.mean_rewards[-1])

        # critic values: shape (n_envs, 1) or (n_envs,)
        values = self.locals.get("values", None)
        if values is not None:
            if hasattr(values, "detach"):  # it's a torch tensor
                values = values.detach().cpu().numpy()
            values = np.array(values, dtype=np.float32)
            if values.size > 0:
                # Flatten to (n_envs,)
                if values.ndim > 1:
                    values = values.squeeze(-1)
                self.mean_values.append(float(values.mean()))
                self.logger.record("train/mean_value", self.mean_values[-1])
        else:
            self.mean_values.append(np.nan)

        # raw reward (if env puts it in infos)
        infos = self.locals.get("infos", None)
        raw_vals = []
        if infos is not None and len(infos) > 0:
            # infos is usually a list of dicts, one per env
            for info in infos:
                if not isinstance(info, dict):
                    continue
                rr = info.get("reward_raw", None)
                if rr is None:
                    continue
                if isinstance(rr, (list, tuple, np.ndarray)):
                    raw_vals.append(float(np.mean(rr)))
                else:
                    raw_vals.append(float(rr))

        if raw_vals:
            mean_raw = float(np.mean(raw_vals))
            self.mean_raw_rewards.append(mean_raw)
            self.logger.record("train/mean_reward_raw", mean_raw)
        else:
            # keep array lengths aligned
            self.mean_raw_rewards.append(np.nan)

        return True

    def _on_training_end(self) -> None:
        """Save a plot of reward/value vs timesteps when training finishes."""
        if not self.timesteps:
            if self.verbose > 0:
                print(f"[{self.profile_id}] RewardPlotCallback: no timesteps logged, skipping plot.")
            return

        os.makedirs(self.out_dir, exist_ok=True)
        out_path = os.path.join(self.out_dir, f"{self.profile_id}_reward_plot.png")

        plt.figure(figsize=(8, 4))
        plt.plot(self.timesteps, self.mean_rewards, label="scaled reward")
        # Only plot raw reward if we actually got any finite values
        if np.any(np.isfinite(self.mean_raw_rewards)):
            plt.plot(self.timesteps, self.mean_raw_rewards, label="raw reward (env)", alpha=0.7)
        if np.any(np.isfinite(self.mean_values)):
            plt.plot(self.timesteps, self.mean_values, label="value estimate (critic)", alpha=0.7)

        plt.xlabel("timesteps")
        plt.ylabel("value")
        plt.title(f"Training: {self.profile_id}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()

        if self.verbose > 0:
            print(f"[{self.profile_id}] Saved reward/value plot to {out_path}")


