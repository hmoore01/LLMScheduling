import os
import time
from typing import Dict, Any, List, Optional, Callable, Union, Literal, Tuple

import numpy as np
import pandas as pd
import torch as th
from gymnasium import spaces

from pettingzoo.utils.env import ParallelEnv
from pettingzoo.utils import parallel_to_aec

from supersuit import black_death_v3, pettingzoo_env_to_vec_env_v1, concat_vec_envs_v1

from stable_baselines3 import PPO

from Rate_Flow_Sim import LLM_Simulator

import matplotlib.pyplot as plt

from stable_baselines3.common.callbacks import BaseCallback

import threading
from collections import deque

try:
    from training_dashboard import dashboard, run_dashboard_server

    DASHBOARD_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] Dashboard import failed: {e}")
    DASHBOARD_AVAILABLE = False
    dashboard = None


    def run_dashboard_server(*args, **kwargs):
        pass


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

        # --- DC Masking for Scalability Experiments ---
        # active_dcs: list of DC indices that are "active" (enabled)
        # If not provided, all DCs are active
        # This allows running with fewer logical DCs while keeping observation space fixed
        self.active_dcs: List[int] = config.get("active_dcs", list(range(self.NUM_DATACENTERS)))
        self.dc_mask: np.ndarray = np.zeros(self.NUM_DATACENTERS, dtype=bool)
        for dc_idx in self.active_dcs:
            if 0 <= dc_idx < self.NUM_DATACENTERS:
                self.dc_mask[dc_idx] = True

        # Penalty multiplier for routing to disabled DCs
        self.disabled_dc_penalty: float = float(config.get("disabled_dc_penalty", 1000.0))

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

        # ------------------------------------------------------------------
        # Debug flag: config["debug"]=True or MARL_DEBUG=1 in env
        # ------------------------------------------------------------------
        cfg_debug = bool(config.get("debug", False))
        env_debug = str(os.environ.get("MARL_DEBUG", "0")).strip().lower() in (
            "1",
            "true",
            "yes",
            "y",
        )
        self.debug: bool = bool(cfg_debug or env_debug)
        if self.debug:
            print(
                f"[ResourceEnv INIT] Debug enabled for profile '{self.active_agent_profile}', "
                f"NUM_DATACENTERS={self.NUM_DATACENTERS}, active_dcs={self.active_dcs}"
            )

        # Reward weights (metrics -> scalar reward)
        self.reward_weights: Dict[str, float] = self._normalize_weights(
            self.profile.get("weights", {"ttft": 1.0})
        )

        # Identify primary metric for this scheme (used for routing/power bias)
        if self.reward_weights:
            self.primary_metric: str = max(self.reward_weights.items(), key=lambda kv: kv[1])[0]
        else:
            self.primary_metric = "ttft"

        if self.debug:
            print(f"[ResourceEnv INIT] reward_weights={self.reward_weights}")
            print(f"[ResourceEnv INIT] primary_metric={self.primary_metric!r}")

        # Constraints / duals
        self.constraints: Dict[str, Dict[str, Any]] = self.profile.get("constraints", {})
        self.include_duals_in_obs: bool = bool(self.profile.get("include_duals_in_obs", True))
        self.lambda_lr: Dict[str, float] = self.profile.get("lambda_lr", {})
        self.duals: Dict[str, float] = {}
        lambda_init = self.profile.get("lambda_init", {})
        for cname in self.constraints.keys():
            self.duals[cname] = float(lambda_init.get(cname, 0.0))

        if self.debug and self.constraints:
            print(f"[ResourceEnv INIT] constraints={self.constraints}")
            print(f"[ResourceEnv INIT] lambda_lr={self.lambda_lr}")

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

        self.metric_scales: Dict[str, float] = {
            "ttft": 0.0,
            "carbon": 0.0,
            "water": 0.0,
            "cost": 0.0,
            "total_energy": 0.0,
            "network_load": 0.0,
        }
        # How fast we adapt the metric scales (like a running "typical" value)
        self.metric_scale_alpha: float = float(self.profile.get("metric_scale_alpha", 0.01))

        # Global weight on constraint penalties for this profile
        # (e.g. set penalty_weight ~ 1–5 in agent_specs for constrained profiles)
        self.penalty_weight: float = float(self.profile.get("penalty_weight", 1.0))

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

        self._rate_sim = None
        self._baseline_sim = None
        self._baseline_cache: Dict[int, Dict[str, float]] = {}
        self.baseline_metrics: Dict[str, float] = {}

        self._reward_computer = ImprovedRewardComputer(
            reward_weights=self.reward_weights,
            constraints=self.constraints,
            primary_metric=self.primary_metric,
        )

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

        # DO NOT NORMALIZE - let weight=10 mean 10x more important!

        # This is critical for scheme divergence

        return {k: float(v) for k, v in weights.items()}

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
        base_obs_dim = 2 + 6 + 3

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
        """Return a per-DC multiplicative bias based on reward_weights.

        For hybrid profiles (e.g., green_perf with ttft:6, carbon:3, cost:1),
        this blends biases from all metrics proportionally to their weights.
        This allows hybrid profiles to balance multiple objectives.
        """
        num_dc = int(self.NUM_DATACENTERS)
        ones = np.ones(num_dc, dtype=np.float32)

        # Get individual biases for each metric
        def get_bias_for_metric(metric: str) -> np.ndarray:
            if metric == "carbon":
                ci = np.maximum(self._dc_carbon, 1e-3)
                return 1.0 / ci
            elif metric == "water":
                w = np.maximum(self._dc_water, 1e-3)
                return 1.0 / w
            elif metric in ("cost", "energy_cost", "price"):
                p = np.maximum(self._dc_price, 1e-3)
                return 1.0 / p
            elif metric == "ttft":
                # For TTFT: Use UNIFORM bias - let proximity bias handle latency
                # Load balancing is handled dynamically in step() based on actual traffic
                # Previously this favored DC 0 which caused severe overloading
                return ones.copy()
            elif metric == "total_energy":
                p = np.maximum(self._dc_price, 1e-3)
                return 1.0 / p
            else:
                return ones.copy()

        # Blend biases based on reward weights
        weights = getattr(self, "reward_weights", {})
        if not weights:
            # Fallback to primary_metric if no weights
            metric = getattr(self, "primary_metric", "ttft")
            bias = get_bias_for_metric(metric)
        else:
            # Weighted blend of all metric biases
            total_weight = sum(weights.values())
            if total_weight <= 0:
                total_weight = 1.0

            blended_bias = np.zeros(num_dc, dtype=np.float32)
            for metric, weight in weights.items():
                if weight > 0:
                    metric_bias = get_bias_for_metric(metric)
                    # Normalize each bias to mean=1 before blending
                    metric_bias = metric_bias / max(metric_bias.mean(), 1e-6)
                    blended_bias += (weight / total_weight) * metric_bias

            bias = blended_bias if blended_bias.sum() > 0 else ones

        # Normalize so that average bias is ~1.0 (keeps behavior well-scaled)
        mean = float(bias.mean()) if bias.size > 0 else 1.0
        if mean <= 0.0 or not np.isfinite(mean):
            out = ones
        else:
            out = (bias / mean).astype(np.float32)

        if getattr(self, "debug", False):
            print(
                f"[ResourceEnv INIT] metric bias for weights={weights}: "
                f"{np.round(out, 3)}"
            )
        return out

    # ------------------------------------------------------------------
    # Baseline plan for each epoch: local-only routing + simple power plan
    # ------------------------------------------------------------------
    def _compute_baseline_for_current_epoch(self) -> None:
        """
        For the current epoch_idx / epoch_df, compute metrics for a very basic
        plan and cache them. The reward later is shaped as improvement vs these
        baseline metrics.

        Baseline plan:
          - Routing: send all tokens to their *source* datacenter (local-only).
          - Power: all node types ON in every DC (simple, overprovisioned).
        """
        eid = int(self.epoch_idx)

        # If we've already computed a baseline for this epoch, just reuse it
        if eid in self._baseline_cache:
            self.baseline_metrics = dict(self._baseline_cache[eid])
            if getattr(self, "debug", False):
                print(f"[ResourceEnv BASELINE] Using cached baseline for epoch {eid}: {self.baseline_metrics}")
            return

        df = self.epoch_df

        # --- Robust column detection (same as step) ---
        if "source_dc_id" in df.columns:
            src_col = "source_dc_id"
        elif "source_dc" in df.columns:
            src_col = "source_dc"
        elif "src_dc" in df.columns:
            src_col = "src_dc"
        else:
            raise KeyError(
                "epoch_df must have one of ['source_dc_id', 'source_dc', 'src_dc'] for baseline"
            )

        if "model_type" in df.columns:
            model_col = "model_type"
        elif "model" in df.columns:
            model_col = "model"
        else:
            raise KeyError(
                "epoch_df must have one of ['model_type', 'model'] for baseline"
            )

        if "num_tokens" in df.columns:
            tok_col = "num_tokens"
        elif "total_tokens" in df.columns:
            tok_col = "total_tokens"
        elif "tokens" in df.columns:
            tok_col = "tokens"
        else:
            raise KeyError(
                "epoch_df must have one of ['num_tokens', 'total_tokens', 'tokens'] for baseline"
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

        # Build a baseline workload: each (src_dc, model) stays local
        req_rows: List[Dict[str, Any]] = []
        plan_map: Dict[int, int] = {}
        row_idx = 0

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

        # Failsafe: if something went wrong and we have no rows
        if not req_rows:
            if getattr(self, "debug", False):
                print(f"[ResourceEnv BASELINE] No workload rows for epoch {eid}; using empty baseline.")
            self.baseline_metrics = {
                "ttft": 0.0,
                "carbon": 0.0,
                "water": 0.0,
                "cost": 0.0,
                "total_energy": 0.0,
                "network_load": 0.0,
            }
            self._baseline_cache[eid] = dict(self.baseline_metrics)
            return

        workload_df = pd.DataFrame(req_rows)
        schedule_plan = {"map": plan_map}

        # Power plan: simple "everything ON" per DC
        power_plan = {}
        for dc_id in range(self.NUM_DATACENTERS):
            power_plan[dc_id] = {
                "unit": {node_type: "ON" for node_type in range(6)}
            }

        # Instantiate a separate baseline simulator if needed
        if self._baseline_sim is None:
            spec_dir = self.epoch_summary.get("spec_dir", "sim_specs")
            epoch_len = int(self.epoch_summary.get("epoch_length", 900))
            self._baseline_sim = LLM_Simulator(
                spec_dir=spec_dir,
                epoch_length=epoch_len,
                debug=False,  # baseline usually doesn't need debug spam
            )

        if getattr(self, "debug", False):
            print(f"[ResourceEnv BASELINE] Running baseline for epoch {eid} with local-only routing.")

        metrics, _, _ = self._baseline_sim.run_epoch(
            eid,
            workload_df,
            schedule_plan,
            power_plan,
        )

        # Extract metrics into a simple dict
        ttft = float(metrics.get("avg_ttft_sec", metrics.get("avg_ttft", 0.0)))
        carbon = float(metrics.get("carbon_emissions", 0.0))
        water = float(metrics.get("water_usage", 0.0))
        cost = float(metrics.get("energy_cost", 0.0))
        total_energy = float(metrics.get("total_energy", metrics.get("energy_kwh", 0.0)))
        network_load = float(metrics.get("avg_net_latency_ms", 0.0))

        self.baseline_metrics = {
            "ttft": ttft,
            "carbon": carbon,
            "water": water,
            "cost": cost,
            "total_energy": total_energy,
            "network_load": network_load,
        }
        self._baseline_cache[eid] = dict(self.baseline_metrics)

        if getattr(self, "debug", False):
            print(f"[ResourceEnv BASELINE] baseline_metrics[{eid}] = {self.baseline_metrics}")

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

        self._compute_baseline_for_current_epoch()

        # Initial observation after choosing the epoch
        obs = self._get_obs_dict()
        infos = {agent: {} for agent in self.agents}
        print(
            f"[ResourceEnv] Starting episode with epoch_idx={self.epoch_idx}, "
            f"total rows={len(self.epoch_df)}"
        )
        if getattr(self, "debug", False):
            print(
                f"[ResourceEnv RESET] epoch_idx={self.epoch_idx}, "
                f"llama7b_total={self.llama7b_total:.1f}, "
                f"llama70b_total={self.llama70b_total:.1f}"
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
    def _update_metric_scale(self, key: str, value: float) -> None:
        """Exponential moving average of each metric's typical scale."""
        value = float(max(value, 0.0))
        if value <= 0.0:
            return

        cur = float(self.metric_scales.get(key, 0.0))
        if cur <= 0.0:
            # First observation: use it as initial scale
            self.metric_scales[key] = value
        else:
            alpha = self.metric_scale_alpha
            self.metric_scales[key] = (1.0 - alpha) * cur + alpha * value

    def _metric_score(self, key: str, value: float) -> float:
        """
        Convert a raw metric into a reward-style score in (0, 1], where
        smaller metric -> larger score.

        IMPROVED: Use exponential scaling for stronger gradient signal.
        This creates larger reward differences between good and bad actions,
        helping PPO learn more effectively.

            score = exp(-value / scale)

        This gives:
        - value = 0: score = 1.0 (perfect)
        - value = scale: score = 0.37 (mediocre)
        - value = 2*scale: score = 0.14 (bad)
        - value = 3*scale: score = 0.05 (very bad)
        """
        self._update_metric_scale(key, value)
        scale = max(self.metric_scales.get(key, 1.0), 1e-6)
        v = max(float(value), 0.0)

        # Exponential decay creates stronger gradients than soft inverse
        score = float(np.exp(-v / scale))

        # Ensure score is in valid range
        return float(np.clip(score, 1e-6, 1.0))

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

            # Determine if we should use proximity bias
            # For TTFT, proximity matters (network latency)
            # For carbon/water/cost, we want to route to best DC regardless of distance
            metric = getattr(self, "primary_metric", "ttft")
            use_proximity_bias = (metric == "ttft")

            for dc_id in range(num_dc):
                if use_proximity_bias:
                    dist_idx = abs(dc_id - src_dc)
                    # closeness in [0.5, 1.0]; tweak 0.5 for stronger/weaker bias
                    closeness = 1.0 - 0.5 * (dist_idx / max_dc_distance)
                    closeness = max(closeness, 0.0)
                else:
                    # No proximity bias for non-TTFT metrics
                    closeness = 1.0

                # Apply metric bias with exponent to amplify differences
                # For TTFT: metric_bias is uniform (1.0), so exponent doesn't matter much
                #           - proximity (closeness) is the main differentiator
                # For carbon/water/cost: need strong exponent to route to best DC
                if use_proximity_bias:
                    # TTFT: Lower exponent since proximity is the main factor
                    metric_bias_strength = 1.0
                else:
                    # Carbon/water/cost: Higher exponent for stronger differentiation
                    metric_bias_strength = 4.0

                amplified_bias = self._metric_bias[dc_id] ** metric_bias_strength

                weights[dc_id] = (
                        base_dist[dc_id]
                        * closeness
                        * amplified_bias
                )

            # --- Apply DC mask: zero out weights for disabled DCs ---
            if hasattr(self, 'dc_mask'):
                for dc_id in range(num_dc):
                    if not self.dc_mask[dc_id]:
                        weights[dc_id] = 0.0

            total_w = float(weights.sum())
            if total_w <= 0.0 or not np.isfinite(total_w):
                # Fallback: uniform across ACTIVE DCs only
                if hasattr(self, 'dc_mask') and self.dc_mask.any():
                    weights[:] = 0.0
                    for dc_id in range(num_dc):
                        if self.dc_mask[dc_id]:
                            weights[dc_id] = 1.0
                    weights /= weights.sum()
                else:
                    weights[:] = 1.0 / num_dc
            else:
                weights /= total_w

            # Routing strategy depends on metric:
            # - TTFT: Route to SINGLE best DC (argmax) to minimize latency
            #         Fractional routing adds overhead and can hurt TTFT
            # - Carbon/Water/Cost: Fractional routing is fine, focuses on best DC anyway

            if use_proximity_bias:
                # TTFT: Single-DC routing - pick the DC with highest weight
                best_dc = int(np.argmax(weights))

                # Add model variant suffix for proper simulator lookup
                model_suffix = "_FP16 (Base)_B16"
                full_model_str = f"{model}{model_suffix}"

                req_rows.append(
                    {
                        "source_dc": src_dc,
                        "model": full_model_str,
                        "arrival_ms": 0,
                        "tokens": tokens,
                    }
                )
                plan_map[row_idx] = best_dc
                row_idx += 1
            else:
                # Carbon/Water/Cost: Split workload across DCs based on weights
                for dc_id in range(num_dc):
                    share = float(weights[dc_id])
                    if share <= 0.0:
                        continue

                    token_share = tokens * share
                    if token_share <= 0.0:
                        continue

                    # Add model variant suffix for proper simulator lookup
                    model_suffix = "_FP16 (Base)_B1"
                    full_model_str = f"{model}{model_suffix}"

                    req_rows.append(
                        {
                            "source_dc": src_dc,
                            "model": full_model_str,
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
                # Add model variant suffix
                model_suffix = "_FP16 (Base)_B1"
                full_model_str = f"{model}{model_suffix}"
                req_rows.append(
                    {
                        "source_dc": src_dc,
                        "model": full_model_str,
                        "arrival_ms": 0,
                        "tokens": tokens,
                    }
                )
                plan_map[row_idx] = src_dc
                row_idx += 1

        workload_df = pd.DataFrame(req_rows)
        self.schedule_plan = {"map": plan_map}

        # ------------------------------------------------------------------
        # 3b) Calculate token distribution per DC for power planning
        #     This is critical: DCs receiving minimal tokens should be OFF
        #     to avoid idle power dominating the metrics.
        # ------------------------------------------------------------------
        dc_token_totals = np.zeros(self.NUM_DATACENTERS, dtype=np.float64)
        for row_idx_key, dc_id in self.schedule_plan["map"].items():
            try:
                tok = float(workload_df.iloc[row_idx_key]["tokens"])
            except Exception:
                tok = 0.0
            dc_token_totals[dc_id] += tok

        total_tokens_all = float(dc_token_totals.sum())
        if total_tokens_all > 0:
            dc_token_fractions = dc_token_totals / total_tokens_all
        else:
            dc_token_fractions = np.ones(self.NUM_DATACENTERS) / self.NUM_DATACENTERS

        # Store for debugging/analysis
        self._dc_token_fractions = dc_token_fractions.copy()

        # Optional debug: aggregate token routing per target DC
        if getattr(self, "debug", False):
            tgt_tokens = np.zeros(self.NUM_DATACENTERS, dtype=np.float64)
            for row, dc_id in self.schedule_plan["map"].items():
                try:
                    tok = float(workload_df.iloc[row]["tokens"])
                except Exception:
                    tok = 0.0
                tgt_tokens[dc_id] += tok
            total_tok = float(tgt_tokens.sum())
            frac = tgt_tokens / total_tok if total_tok > 0.0 else tgt_tokens
            print(
                f"[ResourceEnv STEP] epoch={self.epoch_idx}, step={self.current_step}, "
                f"token routing fractions per DC={np.round(frac, 3)}"
            )

        # ------------------------------------------------------------------
        # 4) Build power_plan from projected power scalars
        #    CRITICAL FIX: Turn OFF units in DCs receiving < 1% of tokens
        #    This prevents idle power from dominating metrics.
        # ------------------------------------------------------------------

        # Threshold: DCs receiving less than this fraction of tokens get turned OFF
        # This is essential for making routing decisions actually affect carbon/energy
        MIN_TOKEN_FRACTION_FOR_POWER = 0.01  # 1%

        # Power patterns for 7 node types (0-6)
        # Each position: 0=OFF, 1=ON for that node type
        # IMPORTANT: Pattern 0 is now minimal (one type ON), not all OFF
        # This ensures DCs that receive work can actually process it
        power_patterns = [
            [0, 1, 0, 0, 0, 0, 0],  # minimal: just 8_H100s (most efficient)
            [1, 0, 0, 0, 0, 0, 0],  # 8_A100s only
            [0, 1, 0, 0, 0, 0, 0],  # 8_H100s only
            [0, 0, 1, 0, 0, 0, 0],  # 4_A100s only
            [0, 0, 0, 1, 0, 0, 0],  # 4_H100s only
            [1, 1, 0, 0, 0, 0, 0],  # both 8-GPU types
            [0, 0, 1, 1, 0, 0, 0],  # both 4-GPU types
            [1, 1, 1, 1, 0, 0, 0],  # all big GPU types
        ]
        num_patterns = len(power_patterns)

        self.power_plan = {}
        for agent in self.agents:
            dc_id = int(agent.split("_")[1])

            # CRITICAL FIX: If this DC receives minimal tokens, turn everything OFF
            # This prevents idle power from dominating and allows routing to affect metrics
            # Use range(7) to include all node types 0-6
            if dc_token_fractions[dc_id] < MIN_TOKEN_FRACTION_FOR_POWER:
                self.power_plan[dc_id] = {
                    "unit": {node_type: "OFF" for node_type in range(7)}
                }
                continue

            lever_val = float(np.clip(power_scalars[agent], 0.0, 1.0))

            metric = getattr(self, "primary_metric", "ttft")
            if metric == "ttft":
                lever_val = 0.5 + 0.5 * lever_val  # push toward higher patterns
            elif metric in ("carbon", "water", "cost", "energy_cost", "price", "total_energy"):
                lever_val = 0.5 * lever_val  # push toward lower patterns

            lever_val = float(np.clip(lever_val, 0.0, 1.0))
            idx = min(int(lever_val * num_patterns), num_patterns - 1)
            node_pattern = power_patterns[idx]

            self.power_plan[dc_id] = {
                "unit": {
                    node_type: ("ON" if on else "OFF")
                    for node_type, on in enumerate(node_pattern)
                }
            }

        if getattr(self, "debug", False):
            print(f"[ResourceEnv STEP] epoch={self.epoch_idx}, step={self.current_step}, power patterns:")
            for dc_id in sorted(self.power_plan.keys()):
                units = self.power_plan[dc_id]["unit"]
                pattern_vec = [
                    1 if units.get(nt, "OFF") == "ON" else 0
                    for nt in sorted(units.keys())
                ]
                token_pct = dc_token_fractions[dc_id] * 100
                print(f"  DC {dc_id}: {pattern_vec} (tokens: {token_pct:.1f}%)")

        # ------------------------------------------------------------------
        # 5) Call rate-based LLM_Simulator for this epoch
        # ------------------------------------------------------------------
        if self._rate_sim is None:
            spec_dir = self.epoch_summary.get("spec_dir", "sim_specs")
            epoch_len = int(self.epoch_summary.get("epoch_length", 900))
            self._rate_sim = LLM_Simulator(
                spec_dir=spec_dir,
                epoch_length=epoch_len,
                debug=getattr(self, "debug", False),
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
        # 6) Turn metrics into shaped reward + constraint penalty
        # ------------------------------------------------------------------
        # Raw metrics from the simulator
        ttft = float(metrics.get("avg_ttft_sec", metrics.get("avg_ttft", 0.0)))
        carbon = float(metrics.get("carbon_emissions", 0.0))
        water = float(metrics.get("water_usage", 0.0))
        cost = float(metrics.get("energy_cost", 0.0))
        total_energy = float(metrics.get("total_energy", metrics.get("energy_kwh", 0.0)))
        network_load = float(metrics.get("avg_net_latency_ms", 0.0))

        total_tokens_step = work_df["total_tokens"].sum() if "total_tokens" in work_df.columns else 0.0

        ep_len = float(self.epoch_summary.get("epoch_length", 900))
        workload_tps = total_tokens_step / max(1.0, ep_len)

        if hasattr(self, '_dc_token_fractions') and hasattr(self, '_dc_carbon'):
            # _dc_token_fractions sums to 1.0 (calculated in step 3b)
            effective_ci = float(np.sum(self._dc_token_fractions * self._dc_carbon))
        else:
            effective_ci = float(np.mean(self._dc_carbon))

        avg_power_level = float(np.mean(list(power_scalars.values())))

        current_routing_dist = self._dc_token_fractions.tolist() if hasattr(self, '_dc_token_fractions') else []

        # Track global max for debugging if desired
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

        if getattr(self, "debug", False):
            print(
                f"[ResourceEnv STEP] metrics: ttft={ttft:.6f}, carbon={carbon:.6f}, "
                f"water={water:.6f}, cost={cost:.6f}, total_energy={total_energy:.6f}, "
                f"network_load={network_load:.6f}"
            )

        # Build raw metric dict
        metric_raw = {
            "ttft": ttft,
            "carbon": carbon,
            "water": water,
            "cost": cost,
            "total_energy": total_energy,
            "network_load": network_load,

            "workload": workload_tps,  # Tokens per second
            "carbon_intensity": effective_ci,  # Traffic-weighted carbon intensity
            "action_mean": avg_power_level,
            "routing_dist": current_routing_dist,
        }

        # 6a) Baseline-based improvements (relative to our simple baseline plan)
        # baseline_metrics was computed in reset() for this epoch
        baseline_vals: Dict[str, Optional[float]] = {}
        improvements: Dict[str, float] = {}
        for m, v in metric_raw.items():
            if m == "routing_dist": continue
            b_val = None
            if hasattr(self, "baseline_metrics"):
                b_val = self.baseline_metrics.get(m, None)
            baseline_vals[m] = b_val

            if b_val is not None and b_val > 0.0 and np.isfinite(b_val):
                # Positive if we beat baseline (lower is better), negative if worse
                improvements[m] = (b_val - v) / b_val
            else:
                improvements[m] = 0.0

        # 6b) Scale-based scores in (0,1], smaller metric -> higher score
        metric_scores: Dict[str, float] = {}
        for k, v in metric_raw.items():
            if k == "routing_dist": continue
            metric_scores[k] = self._metric_score(k, v)

        # 6c) Blend absolute quality with relative improvement
        alpha_improve = float(self.profile.get("alpha_improve", 0.5))
        alpha_improve = float(np.clip(alpha_improve, 0.0, 1.0))

        global_scores: Dict[str, float] = {}
        for k in metric_raw.keys():
            if k == "routing_dist": continue
            base_score = float(metric_scores[k])  # in (0,1]
            # improvements[k] is unbounded, so squash to (-1,1) then map to (0,1)
            imp = float(np.tanh(improvements[k]))  # (-1,1)
            imp_score = 0.5 * (1.0 + imp)  # (0,1)
            global_scores[k] = (1.0 - alpha_improve) * base_score + alpha_improve * imp_score

        if getattr(self, "debug", False):
            disp_scores = {k: f"{v:.3f}" for k, v in global_scores.items()}
            disp_scales = {k: f"{self.metric_scales.get(k, 0.0):.3f}" for k in metric_raw.keys()}
            print(f"[ResourceEnv STEP] metric_scales={disp_scales}")
            print(f"[ResourceEnv STEP] metric_scores_abs={ {k: f'{v:.3f}' for k, v in metric_scores.items()} }")
            print(f"[ResourceEnv STEP] metric_improvements={ {k: f'{v:.3f}' for k, v in improvements.items()} }")
            print(f"[ResourceEnv STEP] metric_scores_blend={disp_scores}")

        # 6d) Weighted scalar reward from metrics (all components ~[0,1])
        base_rw = 0.0
        contribs: Dict[str, Tuple[float, float, float]] = {}
        for m, w in self.reward_weights.items():
            val = float(global_scores.get(m, 0.0))
            contrib = float(w) * val
            base_rw += contrib
            contribs[m] = (w, val, contrib)

        if getattr(self, "debug", False):
            print("[ResourceEnv STEP] reward contributions per metric:")
            for m, (w, val, contrib) in contribs.items():
                print(f"  {m}: weight={w:.4f}, score={val:.4f}, contrib={contrib:.4f}")

        # 6e) Constraints: HARD constraint enforcement with severe penalties
        # Constraints are treated as budgets that should NOT be exceeded
        # Violations receive exponentially increasing penalties
        constraint_costs = metric_raw
        total_penalty = 0.0
        constraint_violations = {}  # Track violations for reporting

        if getattr(self, "debug", False) and self.constraints:
            print("[ResourceEnv STEP] constraint evaluation (HARD CONSTRAINTS):")

        for cname, rule in self.constraints.items():
            val = float(constraint_costs.get(cname, 0.0))
            bound = float(rule.get("bound", rule.get("budget", 0.0)))
            ctype = str(rule.get("type", "upper_bound"))

            # Get penalty multiplier (default very high for hard constraints)
            penalty_mult = float(rule.get("penalty", 10.0))

            # Check if this is a hard constraint (default True)
            is_hard = bool(rule.get("hard", True))

            if bound <= 0.0 or not np.isfinite(bound):
                viol_frac = 0.0
                violated = False
            else:
                if ctype == "upper_bound":
                    # val <= bound is OK; penalize relative excess
                    viol_frac = max(0.0, (val / bound) - 1.0)
                    violated = val > bound
                elif ctype == "lower_bound":
                    # val >= bound is OK; penalize relative shortfall
                    viol_frac = max(0.0, 1.0 - (val / bound))
                    violated = val < bound
                else:
                    viol_frac = 0.0
                    violated = False

            # Track violation status
            constraint_violations[cname] = {
                "violated": violated,
                "value": val,
                "budget": bound,
                "violation_pct": viol_frac * 100
            }

            # Update dual variable (Lagrangian multiplier)
            lr = float(self.lambda_lr.get(cname, 0.1))  # Default learning rate
            old_lambda = float(self.duals.get(cname, 0.0))
            new_lambda = max(0.0, old_lambda + lr * viol_frac)
            self.duals[cname] = new_lambda

            # HARD CONSTRAINT PENALTY STRUCTURE:
            # - No violation (viol_frac = 0): No penalty
            # - Small violation (<10%): Quadratic penalty (soft warning)
            # - Medium violation (10-50%): Cubic penalty (strong deterrent)
            # - Large violation (>50%): Exponential penalty (virtually impossible to choose)

            if is_hard and violated:
                if viol_frac <= 0.1:
                    # Small violation: quadratic
                    penalty_contrib = penalty_mult * (viol_frac ** 2) * 10
                elif viol_frac <= 0.5:
                    # Medium violation: cubic
                    penalty_contrib = penalty_mult * (viol_frac ** 3) * 100
                else:
                    # Large violation: exponential (capped to avoid NaN)
                    penalty_contrib = penalty_mult * min(np.exp(viol_frac * 3), 1000.0)
            else:
                # Soft constraint or no violation: quadratic penalty
                penalty_contrib = new_lambda * (viol_frac ** 2)

            total_penalty += penalty_contrib

            if getattr(self, "debug", False):
                status = "VIOLATED!" if violated else "OK"
                print(
                    f"  {cname}: val={val:.2f}, budget={bound:.2f}, {status}, "
                    f"viol={viol_frac * 100:.1f}%, penalty={penalty_contrib:.4f}"
                )

        # Store constraint violations for external access
        self._constraint_violations = constraint_violations

        # 6f) Action diversity bonus: encourage non-uniform routing distributions
        # This helps PPO learn that different actions lead to different outcomes
        action_diversity_bonus = 0.0
        if hasattr(self, '_dc_token_fractions') and self._dc_token_fractions is not None:
            fracs = np.array(self._dc_token_fractions)
            # Measure concentration: high when tokens go to few DCs
            # Use Gini coefficient or similar
            sorted_fracs = np.sort(fracs)[::-1]  # Descending
            top_dc_share = sorted_fracs[0] if len(sorted_fracs) > 0 else 0

            # Bonus for concentrated routing (helps single-objective agents)
            # Penalty for too-uniform routing (prevents lazy uniform policy)
            if top_dc_share > 0.5:
                # Good: tokens concentrated to optimal DC
                action_diversity_bonus = 0.1 * (top_dc_share - 0.5)
            elif top_dc_share < 0.15:
                # Bad: too uniform, not making meaningful decisions
                action_diversity_bonus = -0.05

        # 6g) Final reward: base minus penalties plus diversity bonus
        # For hard constraints, the penalty weight is increased significantly
        effective_penalty_weight = self.penalty_weight
        if any(v["violated"] for v in constraint_violations.values()):
            effective_penalty_weight = max(self.penalty_weight, 5.0)  # Minimum 5x penalty weight when violated

        raw_reward = float(base_rw - effective_penalty_weight * total_penalty + action_diversity_bonus)
        final_reward = self._scale_reward(raw_reward)

        if getattr(self, "debug", False):
            violations_str = ", ".join(
                f"{k}:{v['violation_pct']:.1f}%"
                for k, v in constraint_violations.items()
                if v["violated"]
            ) or "none"
            print(
                f"[ResourceEnv STEP] reward summary: base={base_rw:.4f}, "
                f"penalty={total_penalty:.4f} (weight={effective_penalty_weight:.1f}), "
                f"diversity_bonus={action_diversity_bonus:.4f}, "
                f"raw={raw_reward:.4f}, scaled={final_reward:.4f}, "
                f"violations=[{violations_str}]"
            )

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
                    "workload": workload_tps,
                    "carbon_intensity": effective_ci,
                    "action_mean": avg_power_level,
                    "routing_dist": current_routing_dist,
                },
                "metrics_normalized": global_scores,
                "metric_raw": metric_raw,
                "metric_improvements": improvements,
                "metric_baselines": baseline_vals,
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
        # Basic workload snapshot, normalized
        total_tokens = self.llama7b_total + self.llama70b_total
        norm_7b = float(self.llama7b_total / (total_tokens + 1e-9))
        norm_70b = float(self.llama70b_total / (total_tokens + 1e-9))

        # Last metrics
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

            # Per-DC profile features (carbon / water / price for THIS DC)
            dc_profile = np.array(
                [
                    float(self._dc_carbon[dc_idx]),
                    float(self._dc_water[dc_idx]),
                    float(self._dc_price[dc_idx]),
                ],
                dtype=np.float32,
            )

            vec = np.concatenate(
                [
                    np.array([norm_7b, norm_70b], dtype=np.float32),
                    last_features,
                    dc_profile,
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

    def get_last_leftovers(self) -> Optional[Any]:
        """Return leftover datacenter usage information from the last episode."""
        return self._last_leftovers


class ImprovedRewardComputer:
    """
    Compute rewards with better gradient signal for single-metric optimization.

    Key improvements:
    1. Logarithmic reward shaping for single-metric agents
    2. Percentile-based scoring using historical values
    3. NO reward scaling for single-metric agents
    """

    def __init__(
            self,
            reward_weights: Dict[str, float],
            constraints: Dict[str, Dict[str, Any]],
            primary_metric: str,
    ):
        self.reward_weights = reward_weights
        self.constraints = constraints
        self.primary_metric = primary_metric
        self.is_single_metric = (
                len(reward_weights) == 1 and
                not constraints
        )

        # Historical tracking for percentile scoring
        self.metric_history: Dict[str, List[float]] = {
            "ttft": [], "carbon": [], "water": [], "cost": [], "total_energy": []
        }
        self.history_max_len = 500

    def update_history(self, metrics: Dict[str, float]):
        for k, v in metrics.items():
            if k in self.metric_history and v is not None and np.isfinite(v):
                self.metric_history[k].append(float(v))
                if len(self.metric_history[k]) > self.history_max_len:
                    self.metric_history[k].pop(0)

    def get_percentile_score(self, metric: str, value: float) -> float:
        """Score in [0,1] where 1 = best (lowest value compared to history)."""
        hist = self.metric_history.get(metric, [])
        if len(hist) < 10:
            return 0.5  # Not enough data
        better_count = sum(1 for h in hist if h >= value)
        return better_count / len(hist)

    def compute_single_metric_reward(
            self,
            metrics: Dict[str, float],
            baseline: Dict[str, float],
    ) -> float:
        """
        Compute reward for single-metric agents.
        Uses logarithmic shaping and NO scaling.
        """
        metric = self.primary_metric
        value = max(metrics.get(metric, 0.0), 1e-9)
        baseline_val = max(baseline.get(metric, value), 1e-9)

        # Component 1: Logarithmic reward (negative, since lower is better)
        # log(baseline/value) is positive when value < baseline (improvement)
        log_reward = np.log(baseline_val / value)

        # Component 2: Relative improvement bonus
        improvement = (baseline_val - value) / baseline_val
        improvement = np.clip(improvement, -2.0, 2.0)
        improvement_bonus = improvement * 3.0

        # Component 3: Percentile bonus
        percentile = self.get_percentile_score(metric, value)
        percentile_bonus = (percentile - 0.5) * 2.0  # [-1, 1]

        # Combine (all components in similar scale now)
        reward = log_reward + improvement_bonus + percentile_bonus

        return float(reward)

    def compute_constrained_reward(
            self,
            metrics: Dict[str, float],
            baseline: Dict[str, float],
            duals: Dict[str, float],
            penalty_weight: float = 1.0,
    ) -> Tuple[float, float, Dict[str, Dict[str, Any]]]:
        """Compute reward for constrained multi-objective agents."""
        total_weight = sum(self.reward_weights.values()) or 1.0
        base_reward = 0.0

        for metric, weight in self.reward_weights.items():
            value = max(metrics.get(metric, 0.0), 1e-9)
            baseline_val = max(baseline.get(metric, value), 1e-9)

            # Normalized improvement score
            score = (baseline_val - value) / baseline_val
            score = np.clip(score, -1.0, 1.0)
            base_reward += (weight / total_weight) * score

        # Constraint penalties
        total_penalty = 0.0
        violations = {}

        for cname, rule in self.constraints.items():
            value = metrics.get(cname, 0.0)
            budget = float(rule.get("budget", rule.get("bound", float('inf'))))
            penalty_mult = float(rule.get("penalty", 1.0))

            if budget > 0 and value > budget:
                viol_frac = (value - budget) / budget
                violations[cname] = {
                    "violated": True,
                    "value": value,
                    "budget": budget,
                    "violation_pct": viol_frac * 100,
                }

                # Dual update
                lr = float(rule.get("lambda_lr", 0.1))
                old_dual = duals.get(cname, 0.0)
                duals[cname] = max(0.0, old_dual + lr * viol_frac)

                total_penalty += duals[cname] * (viol_frac ** 2) + penalty_mult * viol_frac
            else:
                violations[cname] = {
                    "violated": False, "value": value, "budget": budget, "violation_pct": 0.0
                }

        return base_reward, total_penalty * penalty_weight, violations

    def compute(
            self,
            metrics: Dict[str, float],
            baseline: Dict[str, float],
            duals: Dict[str, float],
            penalty_weight: float = 1.0,
    ) -> Tuple[float, Dict[str, Any]]:
        """Main entry point."""
        self.update_history(metrics)

        if self.is_single_metric:
            reward = self.compute_single_metric_reward(metrics, baseline)
            return reward, {
                "reward_type": "single_metric",
                "raw_reward": reward,
                "penalty": 0.0,
                "violations": {},
            }
        else:
            base, penalty, violations = self.compute_constrained_reward(
                metrics, baseline, duals, penalty_weight
            )
            return base - penalty, {
                "reward_type": "constrained",
                "base_reward": base,
                "raw_reward": base - penalty,
                "penalty": penalty,
                "violations": violations,
            }


# ======================================================================
# Heuristic fallback when MARL model doesn't match current DC count
# ======================================================================

def _run_heuristic_fallback(
        raw_env: ResourceEnv,
        epoch_df: pd.DataFrame,
        epoch_summary: Dict[str, Any],
        epoch_idx: int,
        num_dc: int,
        agent_specs: Dict[str, Dict[str, Any]],
        profile_id: str,
) -> Dict[str, Any]:
    """
    Run a heuristic policy when the trained MARL model can't be used
    (e.g., due to observation space mismatch from different DC count).

    The heuristic is based on the profile's reward weights:
    - time_agent: Route to DCs with lowest expected latency (closest)
    - carbon_agent: Route to DCs with lowest carbon intensity
    - water_agent: Route to DCs with lowest water usage
    - cost_agent: Route to DCs with lowest energy cost
    """
    from Rate_Flow_Sim import LLM_Simulator

    # Get profile weights to determine routing strategy
    profile = agent_specs.get(profile_id, {})
    weights = profile.get("weights", {"ttft": 1.0})

    # Determine primary objective
    if "carbon" in weights and weights.get("carbon", 0) > weights.get("ttft", 0):
        strategy = "carbon"
    elif "water" in weights and weights.get("water", 0) > weights.get("ttft", 0):
        strategy = "water"
    elif "cost" in weights and weights.get("cost", 0) > weights.get("ttft", 0):
        strategy = "cost"
    else:
        strategy = "ttft"

    print(f"[MARL HEURISTIC] Using '{strategy}' strategy for profile '{profile_id}'")

    # Initialize simulator
    spec_dir = epoch_summary.get("spec_dir", "sim_specs")
    epoch_len = epoch_summary.get("epoch_length", 900)

    try:
        sim = LLM_Simulator(spec_dir=spec_dir, epoch_length=epoch_len, debug=False)
        dcs = sorted(int(dc_id) for dc_id in sim.datacenters.keys())
    except Exception as e:
        print(f"[MARL HEURISTIC] Could not init simulator: {e}")
        return {}

    if not dcs:
        dcs = list(range(num_dc))

    # Get DC properties for routing decisions
    dc_carbon = epoch_summary.get("dc_carbon_intensity", [400.0] * len(dcs))
    dc_water = epoch_summary.get("dc_water_intensity", [1.0] * len(dcs))
    dc_price = epoch_summary.get("dc_energy_price", [0.1] * len(dcs))

    # Ensure lists are long enough
    while len(dc_carbon) < len(dcs):
        dc_carbon.append(400.0)
    while len(dc_water) < len(dcs):
        dc_water.append(1.0)
    while len(dc_price) < len(dcs):
        dc_price.append(0.1)

    # Build routing based on strategy
    if strategy == "carbon":
        # Route to lowest carbon intensity DC
        dc_scores = {dcs[i]: dc_carbon[i] for i in range(len(dcs))}
    elif strategy == "water":
        # Route to lowest water intensity DC
        dc_scores = {dcs[i]: dc_water[i] for i in range(len(dcs))}
    elif strategy == "cost":
        # Route to lowest cost DC
        dc_scores = {dcs[i]: dc_price[i] for i in range(len(dcs))}
    else:
        # Route to balance load (round-robin style)
        dc_scores = {dc: i for i, dc in enumerate(dcs)}

    # Sort DCs by score (lower is better)
    sorted_dcs = sorted(dc_scores.keys(), key=lambda x: dc_scores[x])

    # Build request dataframe and routing plan
    if not isinstance(epoch_df, pd.DataFrame):
        epoch_df = pd.DataFrame(epoch_df)

    # Normalize column names
    col_map = {
        "src_dc": "source_dc_id",
        "src": "source_dc_id",
        "model": "model_type",
        "tokens": "num_tokens",
    }
    for old, new in col_map.items():
        if old in epoch_df.columns and new not in epoch_df.columns:
            epoch_df = epoch_df.rename(columns={old: new})

    # Group by source_dc and model
    if "source_dc_id" in epoch_df.columns and "model_type" in epoch_df.columns:
        work_df = epoch_df.groupby(["source_dc_id", "model_type"], as_index=False).agg({
            "num_tokens": "sum"
        }).rename(columns={"source_dc_id": "src_dc", "num_tokens": "total_tokens"})
    else:
        # Fallback
        work_df = epoch_df.copy()
        if "total_tokens" not in work_df.columns and "num_tokens" in work_df.columns:
            work_df["total_tokens"] = work_df["num_tokens"]

    # Build routing plan
    req_rows = []
    plan_map = {}
    load_per_dc = {dc: 0.0 for dc in dcs}

    row_idx = 0
    for _, row in work_df.iterrows():
        src_dc = int(row.get("src_dc", row.get("source_dc_id", 0)))
        model = str(row.get("model_type", row.get("model", "Llama7b")))
        tokens = float(row.get("total_tokens", row.get("num_tokens", 1000)))

        # Choose target DC based on strategy
        if strategy in ["carbon", "water", "cost"]:
            # Route to best DC for this metric
            target_dc = sorted_dcs[0]
        else:
            # Load balance: route to DC with least load
            target_dc = min(load_per_dc.keys(), key=lambda x: load_per_dc[x])

        load_per_dc[target_dc] += tokens

        # Add model variant suffix for proper simulator lookup
        model_suffix = "_FP16 (Base)_B1"
        full_model_str = f"{model}{model_suffix}"

        req_rows.append({
            "source_dc": src_dc,
            "model": full_model_str,
            "arrival_ms": 0,
            "tokens": int(tokens),
        })
        plan_map[row_idx] = target_dc
        row_idx += 1

    # Build power plan (simple: all Idle)
    power_plan = {dc: {nt: "Idle" for nt in range(6)} for dc in dcs}

    # Run simulator
    requests_df = pd.DataFrame(req_rows)
    schedule_plan = {"map": plan_map}

    try:
        metrics, details, leftovers = sim.run_epoch(
            epoch_idx, requests_df, schedule_plan, power_plan
        )

        result = {
            "avg_ttft": float(metrics.get("avg_ttft", 0.0)),
            "avg_ttft_sec": float(metrics.get("avg_ttft", 0.0)),
            "carbon_emissions": float(metrics.get("carbon_emissions", 0.0)),
            "water_usage": float(metrics.get("water_usage", 0.0)),
            "energy_cost": float(metrics.get("energy_cost", 0.0)),
            "total_energy": float(metrics.get("total_energy", 0.0)),
        }
        return result
    except Exception as e:
        print(f"[MARL HEURISTIC] Simulation failed: {e}")
        return {}


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

    IMPORTANT: If the number of datacenters differs from training, the model
    cannot be used directly (observation space mismatch). In this case, we
    fall back to a heuristic policy based on the trained profile's weights.
    """

    # --- Determine which profiles to run ---
    # IMPORTANT: For scalability experiments, we always use 12 DCs internally
    # to match the trained model's observation space. We use `active_dcs` to
    # mask which DCs are actually enabled for routing.

    # Get the "logical" number of DCs requested (for masking)
    if isinstance(epoch_summary, dict):
        if "num_datacenters" in epoch_summary:
            logical_num_dc = int(epoch_summary["num_datacenters"])
        elif "datacenters" in epoch_summary:
            logical_num_dc = int(len(epoch_summary["datacenters"]))
        else:
            try:
                logical_num_dc = int(epoch_df["source_dc_id"].max()) + 1
            except Exception:
                logical_num_dc = 12
    else:
        try:
            logical_num_dc = int(epoch_df["source_dc_id"].max()) + 1
        except Exception:
            logical_num_dc = 12

    # Always use 12 DCs for MARL to match trained model observation space
    # Use active_dcs to mask which ones are actually enabled
    MARL_FIXED_NUM_DC = 12
    num_dc = MARL_FIXED_NUM_DC

    # Determine which DCs are active based on logical_num_dc
    # Strategy: Use first N DCs where N = logical_num_dc
    active_dcs = list(range(min(logical_num_dc, MARL_FIXED_NUM_DC)))

    if logical_num_dc != MARL_FIXED_NUM_DC:
        print(f"[MARL EVAL] Scalability mode: {logical_num_dc} logical DCs -> "
              f"using {MARL_FIXED_NUM_DC} DCs with mask, active_dcs={active_dcs}")

    try:
        from simulator_LLM import build_agent_specs  # defines profiles
        agent_specs = build_agent_specs(num_datacenters=num_dc)
        profile_ids = list(agent_specs.keys())
    except Exception:
        # Fallback if build_agent_specs is unavailable
        agent_specs = {
            "time_agent": {"weights": {"ttft": 10}},
            "carbon_agent": {"weights": {"carbon": 10}},
            "water_agent": {"weights": {"water": 10}},
            "cost_agent": {"weights": {"cost": 10}},
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
            "num_datacenters": num_dc,  # Always 12 for MARL
            "epoch_summary": epoch_summary,
            "agent_specs": agent_specs,
            "active_agent_profile": profile_id,
            "max_steps": 1,
            "active_dcs": active_dcs,  # DC mask for scalability experiments
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

        # Load model - observation space should now match since we always use 12 DCs
        try:
            model = PPO.load(model_path, env=None, device="cpu")
            model_obs_shape = model.observation_space.shape
            env_obs_shape = venv.observation_space.shape

            if model_obs_shape != env_obs_shape:
                print(f"[MARL EVAL] WARNING: Observation space mismatch for '{profile_id}':")
                print(f"  Model trained with: {model_obs_shape}")
                print(f"  Current env has: {env_obs_shape}")
                print(f"  Using heuristic fallback.")

                # Fall back to heuristic policy
                metrics = _run_heuristic_fallback(
                    raw_env, epoch_df, epoch_summary, epoch_idx,
                    logical_num_dc, agent_specs, profile_id
                )
                results[profile_id] = metrics
                venv.close()
                continue

            # Set the environment for the model
            model.set_env(venv)

        except Exception as e:
            print(f"[MARL EVAL] WARNING: Could not load model for '{profile_id}': {e}")
            print(f"  Using heuristic fallback.")
            metrics = _run_heuristic_fallback(
                raw_env, epoch_df, epoch_summary, epoch_idx,
                num_dc, agent_specs, profile_id
            )
            results[profile_id] = metrics
            venv.close()
            continue

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

    # Ensure epoch_summary has dc_carbon_intensity loaded from specs
    # This is CRITICAL for routing to work correctly!
    if "dc_carbon_intensity" not in epoch_summary:
        import os

        # Handle node_properties being either a dict or a list
        if isinstance(node_properties, dict):
            spec_dir = node_properties.get("spec_dir", "sim_specs")
        else:
            spec_dir = "sim_specs"

        dc_specs_path = os.path.join(spec_dir, "Datacenter_specs.csv")
        num_dc = epoch_summary.get("num_datacenters", len(epoch_summary.get("datacenters", range(12))))

        if os.path.exists(dc_specs_path):
            dc_df = pd.read_csv(dc_specs_path)
            if "DC_Num" in dc_df.columns:
                dc_df = dc_df.sort_values("DC_Num")

            # Load Carbon Intensity
            if "Carbon_Intensity" in dc_df.columns:
                epoch_summary["dc_carbon_intensity"] = dc_df["Carbon_Intensity"].astype(float).tolist()
            else:
                epoch_summary["dc_carbon_intensity"] = [400.0] * num_dc

            # Load Water Intensity (use Water_Static column if available)
            if "Water_Static" in dc_df.columns:
                epoch_summary["dc_water_intensity"] = dc_df["Water_Static"].astype(float).tolist()
            elif "Water_Intensity" in dc_df.columns:
                epoch_summary["dc_water_intensity"] = dc_df["Water_Intensity"].astype(float).tolist()

            # Load Energy Price (try multiple possible column names)
            # Parse Time_of_Use to get average price if no direct price column
            price_cols = ["Energy_Price", "Price", "Cost"]
            found_price = False
            for col in price_cols:
                if col in dc_df.columns:
                    epoch_summary["dc_energy_price"] = dc_df[col].astype(float).tolist()
                    found_price = True
                    break

            if not found_price and "Time_of_Use(24_Hours)" in dc_df.columns:
                # Parse the semicolon-separated ToU values and compute average
                prices = []
                for tou_str in dc_df["Time_of_Use(24_Hours)"]:
                    try:
                        tou_vals = [float(x) for x in str(tou_str).split(";")]
                        prices.append(sum(tou_vals) / len(tou_vals) if tou_vals else 0.1)
                    except:
                        prices.append(0.1)
                epoch_summary["dc_energy_price"] = prices
        else:
            epoch_summary["dc_carbon_intensity"] = [400.0] * num_dc

    # Wrap node_properties in a dict if it's a list
    if isinstance(node_properties, list):
        node_properties = {"spec_dir": "sim_specs", "nodes": node_properties}

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

    return stats, profile_metrics, {}


# ======================================================================
# Enhanced Dashboard Training Callback
# ======================================================================

class DashboardTrainingCallback(BaseCallback):
    """
    Send training metrics to dashboard for live visualization.

    Enhanced version that sends:
    - Power plan information (which nodes are ON/OFF per DC)
    - Schedule/routing plan
    - Raw reward (unscaled)
    - Agent configuration info
    """

    def __init__(self, profile_id: str, agent_config: dict = None, verbose: int = 0):
        super().__init__(verbose)
        self.profile_id = profile_id
        self.agent_config = agent_config or {}
        print(f"[DASHBOARD DEBUG] Initialized callback for profile: {self.profile_id}")

    def _on_training_start(self):
        if DASHBOARD_AVAILABLE and dashboard:
            # Register profile with config info for display
            total_timesteps = getattr(self, 'total_timesteps', 100000)
            if hasattr(self, 'locals') and self.locals:
                total_timesteps = self.locals.get('total_timesteps', total_timesteps)

            dashboard.register_profile(
                self.profile_id,
                total_timesteps
            )
            dashboard.start_training(self.profile_id)
            if self.verbose > 0:
                print(f"[DASHBOARD] Started tracking: {self.profile_id}")

    def _on_step(self) -> bool:
        # Check for early stopping from dashboard
        if DASHBOARD_AVAILABLE and dashboard and dashboard.should_stop(self.profile_id):
            print(f"[{self.profile_id}] Early stop requested via dashboard!")
            return False

        # Log metrics periodically
        self._log_metrics()
        return True

    def _log_metrics(self):
        if not DASHBOARD_AVAILABLE or not dashboard:
            return

        # Log every 50 steps for good responsiveness
        if self.num_timesteps > 1 and self.num_timesteps % 50 != 0:
            return

        # Basic stats
        rewards = self.locals.get("rewards", [0])
        mean_reward = float(np.mean(rewards))

        metrics = {}
        power_plan = None
        raw_reward = None

        infos = self.locals.get("infos", [])

        # Extract data from infos
        for info in (infos or []):
            if not isinstance(info, dict):
                continue

            # Get raw metrics
            raw = info.get("raw_metrics", info.get("metric_raw"))

            # Check nested structure (for multi-agent)
            if not raw:
                for val in info.values():
                    if isinstance(val, dict):
                        raw = val.get("raw_metrics", val.get("metric_raw"))
                        if raw:
                            break

            if raw:
                metrics = {
                    "ttft": raw.get("ttft", raw.get("avg_ttft_sec", 0)),
                    "carbon": raw.get("carbon", raw.get("carbon_emissions", 0)),
                    "water": raw.get("water", raw.get("water_usage", 0)),
                    "cost": raw.get("cost", raw.get("energy_cost", 0)),
                    "total_energy": raw.get("total_energy", raw.get("energy_kwh", 0)),
                    "workload": raw.get("workload", 0),
                    "carbon_intensity": raw.get("carbon_intensity", 0),
                    "action_mean": raw.get("action_mean", 0),
                    "routing_dist": raw.get("routing_dist", []),
                }

            # Get power plan
            if "power_plan" in info:
                power_plan = info["power_plan"]
            else:
                for val in info.values():
                    if isinstance(val, dict) and "power_plan" in val:
                        power_plan = val["power_plan"]
                        break

            # Get raw reward
            raw_reward = info.get("reward_raw", info.get("raw_reward"))
            if raw_reward is not None:
                raw_reward = float(raw_reward)

            if raw:
                break

        # Log to dashboard - this also writes to shared state for cross-process visibility
        dashboard.log_step(
            profile_id=self.profile_id,
            timestep=self.num_timesteps,
            reward=mean_reward,
            metrics=metrics,
            raw_reward=raw_reward,
            power_plan=power_plan,
        )

        # Also use cross-process logging function for parallel training
        try:
            from training_dashboard import log_to_dashboard
            log_to_dashboard(
                profile_id=self.profile_id,
                timestep=self.num_timesteps,
                reward=mean_reward,
                metrics=metrics,
                power_plan=power_plan,
            )
        except ImportError:
            pass

    def _on_training_end(self):
        if DASHBOARD_AVAILABLE and dashboard:
            dashboard.end_training(self.profile_id)
            if self.verbose > 0:
                print(f"[DASHBOARD] Completed: {self.profile_id}")


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

    # Black-death wrapper so agents that "die" still produce valid obs/rewards
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

    # IMPROVED PPO HYPERPARAMETERS for better learning
    # These settings help PPO learn stronger, more differentiated policies
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        # Learning rate: slightly higher for faster initial learning
        learning_rate=3e-4,
        # Number of steps to run per update (larger = more stable gradients)
        n_steps=2048,
        # Batch size for updates
        batch_size=64,
        # Number of epochs when optimizing the surrogate loss
        n_epochs=10,
        # Discount factor (slightly lower to focus on immediate rewards)
        gamma=0.95,
        # GAE lambda (generalized advantage estimation)
        gae_lambda=0.9,
        # Clipping parameter (slightly tighter for more conservative updates)
        clip_range=0.2,
        # Entropy coefficient (higher = more exploration)
        ent_coef=0.01,
        # Value function coefficient
        vf_coef=0.5,
        # Max gradient norm for clipping
        max_grad_norm=0.5,
        # Policy network architecture: deeper network for more capacity
        policy_kwargs={
            "net_arch": {
                "pi": [128, 128, 64],  # Policy network: 3 layers
                "vf": [128, 128, 64],  # Value network: 3 layers
            },
            # Use tanh activation for bounded outputs
            "activation_fn": th.nn.Tanh,
        },
    )

    # Get agent config for dashboard display
    agent_config = agent_specs.get(profile_id, {})

    # Attach callbacks
    callbacks = [
        RewardPlotCallback(profile_id=profile_id, out_dir=profile_dir, verbose=1),
    ]

    if DASHBOARD_AVAILABLE:
        callbacks.append(DashboardTrainingCallback(
            profile_id=profile_id,
            agent_config=agent_config,
            verbose=1
        ))

    # Train
    model.learn(total_timesteps=total_timesteps, callback=callbacks)

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
        parallel: bool = False,
        max_workers: int = None,
) -> None:
    """
    Train PPO for *all* provided profiles in agent_specs.

    Args:
        parallel: If True, train profiles in parallel using multiprocessing
        max_workers: Max parallel processes (default: min(num_profiles, cpu_count))
    """
    if not isinstance(epoch_df, pd.DataFrame):
        epoch_df = pd.DataFrame(epoch_df)

    profile_ids = list(agent_specs.keys())
    print(f"[MARL TRAIN] Using epoch {epoch_idx} for training data")
    print(f"[MARL TRAIN] Profiles to train: {profile_ids}")

    if parallel:
        train_all_schemes_parallel(
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
            max_workers=max_workers,
        )
    else:
        # Original sequential training
        for pid in profile_ids:
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


def _train_single_profile_worker(args: tuple) -> str:
    """
    Worker function for parallel training.
    Must be at module level for pickling.

    Returns profile_id on success, or error message on failure.
    """
    (
        profile_id,
        epoch_df_dict,  # Pass as dict for pickling
        epoch_summary,
        epoch_idx,
        node_properties,
        agent_specs,
        num_datacenters,
        total_timesteps,
        num_envs,
        overwrite_existing,
        model_dir,
    ) = args

    try:
        # Reconstruct DataFrame in worker process
        epoch_df = pd.DataFrame(epoch_df_dict)

        # Import here to avoid issues with multiprocessing
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU to avoid GPU contention

        train_reward_scheme(
            profile_id=profile_id,
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
        return f"SUCCESS: {profile_id}"
    except Exception as e:
        import traceback
        return f"FAILED: {profile_id} - {str(e)}\n{traceback.format_exc()}"


def train_all_schemes_parallel(
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
        max_workers: int = 8,
) -> Dict[str, str]:
    """
    Train all profiles in PARALLEL using multiprocessing.

    This can significantly speed up training when you have multiple profiles
    and sufficient CPU cores.

    Args:
        max_workers: Maximum number of parallel training processes.
                    Default: min(num_profiles, cpu_count - 1)

    Returns:
        Dict mapping profile_id to result status
    """
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed

    if not isinstance(epoch_df, pd.DataFrame):
        epoch_df = pd.DataFrame(epoch_df)

    profile_ids = list(agent_specs.keys())
    n_profiles = len(profile_ids)

    # Determine number of workers
    if max_workers is None:
        max_workers = min(n_profiles, max(1, mp.cpu_count() - 1))
    max_workers = min(max_workers, n_profiles)

    print(f"[MARL PARALLEL] Training {n_profiles} profiles with {max_workers} parallel workers")

    # Convert DataFrame to dict for pickling
    epoch_df_dict = epoch_df.to_dict('list')

    # Prepare arguments for each worker
    worker_args = [
        (
            pid,
            epoch_df_dict,
            epoch_summary,
            epoch_idx,
            node_properties,
            agent_specs,
            num_datacenters,
            total_timesteps,
            num_envs,
            overwrite_existing,
            model_dir,
        )
        for pid in profile_ids
    ]

    results = {}

    # Use ProcessPoolExecutor for true parallelism
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_profile = {
            executor.submit(_train_single_profile_worker, args): args[0]
            for args in worker_args
        }

        # Collect results as they complete
        for future in as_completed(future_to_profile):
            profile_id = future_to_profile[future]
            try:
                result = future.result()
                results[profile_id] = result
                print(f"[MARL PARALLEL] {result}")
            except Exception as e:
                results[profile_id] = f"EXCEPTION: {profile_id} - {str(e)}"
                print(f"[MARL PARALLEL] EXCEPTION for {profile_id}: {e}")

    # Summary
    successes = sum(1 for r in results.values() if r.startswith("SUCCESS"))
    failures = n_profiles - successes
    print(f"\n[MARL PARALLEL] Complete: {successes}/{n_profiles} succeeded, {failures} failed")

    return results


def train_with_multi_epoch_parallel(
        *,
        trace_path: str = None,
        num_datacenters: int = 12,
        agent_specs: Dict[str, Dict[str, Any]] = None,
        node_properties: Dict[str, Any] = None,
        total_timesteps: int = 100_000,
        sampling_strategy: str = "stratified",
        num_envs: int = 1,
        overwrite_existing: bool = True,
        model_dir: str = "trained_models/sb3_agents",
        spec_dir: str = "sim_specs",
        epoch_length: int = 900,
        max_workers: int = 8,
        **kwargs,
) -> None:
    """
    Multi-epoch training with PARALLEL profile training.

    Each profile trains in its own process, utilizing multiple CPU cores.

    Args:
        max_workers: Max parallel processes. Default: min(num_profiles, cpu_count-1)
    """
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import os

    # Start dashboard in main process
    if DASHBOARD_AVAILABLE:
        run_dashboard_server(port=5000)
        print("\n" + "=" * 60)
        print("🚀 Training Dashboard: http://localhost:5000")
        print("=" * 60 + "\n")

    # Load trace
    if trace_path is None:
        trace_path = "simulator_ready_trace.csv"

    print(f"[MARL PARALLEL] Loading trace from: {trace_path}")
    full_trace_df = pd.read_csv(trace_path)

    # Get epochs
    epoch_col = "epoch_id" if "epoch_id" in full_trace_df.columns else \
        "epoch" if "epoch" in full_trace_df.columns else None

    if epoch_col:
        unique_epochs = sorted(full_trace_df[epoch_col].unique())
    else:
        unique_epochs = [0]

    # Sample epochs
    n_samples = min(50, len(unique_epochs))
    if sampling_strategy == "stratified" and len(unique_epochs) > n_samples:
        indices = np.linspace(0, len(unique_epochs) - 1, n_samples, dtype=int)
        training_epochs = [unique_epochs[i] for i in indices]
    else:
        training_epochs = unique_epochs[:n_samples]

    print(f"[MARL PARALLEL] Using {len(training_epochs)} epochs")

    # Load DC specs
    dc_specs_path = os.path.join(spec_dir, "Datacenter_specs.csv")
    if os.path.exists(dc_specs_path):
        dc_df = pd.read_csv(dc_specs_path)
        if "DC_Num" in dc_df.columns:
            dc_df = dc_df.sort_values("DC_Num")
        real_ci = dc_df[
            "Carbon_Intensity"].tolist() if "Carbon_Intensity" in dc_df.columns else [400.0] * num_datacenters
    else:
        real_ci = [400.0] * num_datacenters

    if node_properties is None:
        node_properties = {"spec_dir": spec_dir}

    if agent_specs is None:
        agent_specs = {
            "time_agent": {"weights": {"ttft": 10}, "constraints": {}},
            "carbon_agent": {"weights": {"carbon": 10}, "constraints": {}},
            "water_agent": {"weights": {"water": 10}, "constraints": {}},
            "cost_agent": {"weights": {"cost": 10}, "constraints": {}},
        }

    profile_ids = list(agent_specs.keys())
    n_profiles = len(profile_ids)

    if max_workers is None:
        max_workers = min(n_profiles, max(1, mp.cpu_count() - 1))

    print(f"[MARL PARALLEL] Training {n_profiles} profiles with {max_workers} workers")

    # Build combined training data from sampled epochs
    combined_dfs = []
    for epoch_idx in training_epochs:
        if epoch_col:
            edf = full_trace_df[full_trace_df[epoch_col] == epoch_idx].copy()
        else:
            edf = full_trace_df.copy()
        if len(edf) > 0:
            edf["_sampled_epoch"] = epoch_idx
            combined_dfs.append(edf)

    if combined_dfs:
        training_df = pd.concat(combined_dfs, ignore_index=True)
    else:
        training_df = full_trace_df.copy()

    epoch_summary = {
        "num_datacenters": num_datacenters,
        "spec_dir": spec_dir,
        "epoch_length": epoch_length,
        "dc_carbon_intensity": real_ci,
    }

    # Register all profiles with dashboard before training
    if DASHBOARD_AVAILABLE and dashboard:
        for pid in profile_ids:
            dashboard.register_profile(pid, total_timesteps, agent_specs.get(pid, {}))

    # Prepare worker arguments
    training_df_dict = training_df.to_dict('list')

    worker_args = [
        (
            pid,
            training_df_dict,
            epoch_summary,
            training_epochs[0],  # Use first epoch as reference
            node_properties,
            agent_specs,
            num_datacenters,
            total_timesteps,
            num_envs,
            overwrite_existing,
            model_dir,
        )
        for pid in profile_ids
    ]

    # Train in parallel
    print(f"[MARL PARALLEL] Starting parallel training...")
    start_time = time.time() if 'time' in dir() else 0

    results = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_profile = {
            executor.submit(_train_single_profile_worker, args): args[0]
            for args in worker_args
        }

        for future in as_completed(future_to_profile):
            profile_id = future_to_profile[future]
            try:
                result = future.result()
                results[profile_id] = result
                print(f"[MARL PARALLEL] {result}")

                # Update dashboard
                if DASHBOARD_AVAILABLE and dashboard:
                    dashboard.end_training(profile_id)
            except Exception as e:
                results[profile_id] = f"EXCEPTION: {str(e)}"
                print(f"[MARL PARALLEL] EXCEPTION for {profile_id}: {e}")

    elapsed = time.time() - start_time if 'time' in dir() and start_time else 0
    successes = sum(1 for r in results.values() if "SUCCESS" in r)

    print(f"\n[MARL PARALLEL] ==========================================")
    print(f"[MARL PARALLEL] Training complete!")
    print(f"[MARL PARALLEL] Profiles: {successes}/{n_profiles} succeeded")
    print(f"[MARL PARALLEL] Time: {elapsed:.1f}s (vs ~{elapsed * max_workers:.1f}s sequential)")
    print(f"[MARL PARALLEL] Speedup: ~{max_workers}x")
    print(f"[MARL PARALLEL] ==========================================\n")


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


# Wrapper for backwards compatibility with different calling conventions
def train_with_multi_epoch(
        *,
        trace_path: str = None,
        num_datacenters: int = 12,
        agent_specs: Dict[str, Dict[str, Any]] = None,
        node_properties: Dict[str, Any] = None,
        total_timesteps: int = 100_000,
        sampling_strategy: str = "stratified",
        use_domain_randomization: bool = True,
        num_envs: int = 1,
        overwrite_existing: bool = True,
        model_dir: str = "trained_models/sb3_agents",
        spec_dir: str = "sim_specs",
        epoch_length: int = 900,
        parallel: bool = True,
        max_workers: int = 8,
        **kwargs,
) -> None:
    """
    Multi-epoch training wrapper function.

    Trains agents across MULTIPLE epochs to learn policies that generalize
    across different workload patterns, pricing, and conditions.

    Args:
        trace_path: Path to the CSV trace file
        num_datacenters: Number of datacenters in the simulation
        agent_specs: Dictionary of agent specifications
        node_properties: Node configuration properties
        total_timesteps: Total training timesteps per profile
        sampling_strategy: How to sample epochs ("stratified", "uniform", "curriculum")
        use_domain_randomization: Whether to use domain randomization
        num_envs: Number of parallel environments
        overwrite_existing: Whether to overwrite existing models
        model_dir: Directory to save trained models
        spec_dir: Directory containing simulation specs
        epoch_length: Length of each epoch in seconds
        parallel: If True, train all profiles in parallel (faster!)
        max_workers: Max parallel processes (default: cpu_count - 1)
    """
    # If parallel requested, use the parallel implementation
    if parallel:
        return train_with_multi_epoch_parallel(
            trace_path=trace_path,
            num_datacenters=num_datacenters,
            agent_specs=agent_specs,
            node_properties=node_properties,
            total_timesteps=total_timesteps,
            sampling_strategy=sampling_strategy,
            num_envs=num_envs,
            overwrite_existing=overwrite_existing,
            model_dir=model_dir,
            spec_dir=spec_dir,
            epoch_length=epoch_length,
            max_workers=max_workers,
            **kwargs,
        )

    import os

    if DASHBOARD_AVAILABLE:
        run_dashboard_server(port=5000)
        print("\\n" + "=" * 60)
        print("🚀 Training Dashboard: http://localhost:5000")
        print("=" * 60 + "\\n")

    # Load trace data
    if trace_path is None:
        trace_path = "simulator_ready_trace.csv"

    print(f"[MARL TRAIN] Loading trace from: {trace_path}")
    full_trace_df = pd.read_csv(trace_path)
    print(f"[MARL TRAIN] Loaded {len(full_trace_df)} rows")

    # Identify epoch column
    if "epoch_id" in full_trace_df.columns:
        epoch_col = "epoch_id"
    elif "epoch" in full_trace_df.columns:
        epoch_col = "epoch"
    else:
        # Assume each row is its own epoch or create synthetic epochs
        epoch_col = None

    # Get list of unique epochs
    if epoch_col:
        unique_epochs = sorted(full_trace_df[epoch_col].unique())
    else:
        unique_epochs = [0]

    print(f"[MARL TRAIN] Found {len(unique_epochs)} unique epochs")

    # Sample epochs based on strategy
    if sampling_strategy == "stratified":
        # Sample epochs that cover different workload levels
        n_samples = min(50, len(unique_epochs))  # Use up to 50 representative epochs
        if len(unique_epochs) > n_samples:
            # Sample evenly across the range
            indices = np.linspace(0, len(unique_epochs) - 1, n_samples, dtype=int)
            training_epochs = [unique_epochs[i] for i in indices]
        else:
            training_epochs = unique_epochs
    elif sampling_strategy == "uniform":
        # Random sample
        n_samples = min(50, len(unique_epochs))
        training_epochs = list(np.random.choice(unique_epochs, size=n_samples, replace=False))
    elif sampling_strategy == "curriculum":
        # Start with easier (smaller) epochs, progress to harder ones
        # Sort by workload size
        epoch_sizes = full_trace_df.groupby(epoch_col).size().sort_values()
        training_epochs = list(epoch_sizes.index[:50])  # Start with 50 smallest
    else:
        training_epochs = unique_epochs[:50]

    print(f"[MARL TRAIN] Using {len(training_epochs)} epochs for training: {training_epochs[:5]}...")

    # Load carbon intensity from specs
    dc_specs_path = os.path.join(spec_dir, "Datacenter_specs.csv")
    if os.path.exists(dc_specs_path):
        dc_df = pd.read_csv(dc_specs_path)
        if "DC_Num" in dc_df.columns:
            dc_df = dc_df.sort_values("DC_Num")
        if "Carbon_Intensity" in dc_df.columns:
            real_ci = dc_df["Carbon_Intensity"].astype(float).tolist()
        else:
            real_ci = [400.0] * num_datacenters
    else:
        real_ci = [400.0] * num_datacenters

    # Build default node_properties if not provided
    if node_properties is None:
        node_properties = {"spec_dir": spec_dir}

    # Build default agent_specs if not provided
    if agent_specs is None:
        agent_specs = {
            "time_agent": {"weights": {"ttft": 10}, "constraints": {}},
            "carbon_agent": {"weights": {"carbon": 10}, "constraints": {}},
            "water_agent": {"weights": {"water": 10}, "constraints": {}},
            "cost_agent": {"weights": {"cost": 10}, "constraints": {}},
        }

    print(f"[MARL TRAIN] Strategy: {sampling_strategy}, Domain Randomization: {use_domain_randomization}")
    print(f"[MARL TRAIN] Training profiles: {list(agent_specs.keys())}")

    # Train each profile
    for profile_id in agent_specs.keys():
        print(f"\n{'=' * 60}")
        print(f"[MARL TRAIN] Training profile: {profile_id}")
        print(f"{'=' * 60}")

        # Check if model exists and skip if not overwriting
        root_model_path = os.path.join(model_dir, f"{profile_id}.zip")
        profile_dir = os.path.join(model_dir, profile_id)
        final_model_path = os.path.join(profile_dir, "final_model.zip")

        if not overwrite_existing and (os.path.exists(root_model_path) or os.path.exists(final_model_path)):
            print(f"[MARL TRAIN] {profile_id}: model exists, skipping (overwrite_existing=False).")
            continue

        os.makedirs(profile_dir, exist_ok=True)

        # Calculate timesteps per epoch
        timesteps_per_epoch = max(1000, total_timesteps // len(training_epochs))

        model = None

        if DASHBOARD_AVAILABLE:
            # Calculate total steps so the progress bar is accurate
            total_estimated_steps = timesteps_per_epoch * len(training_epochs)
            print(f"[DASHBOARD] Registering {profile_id} with {total_estimated_steps} steps")
            dashboard.register_profile(profile_id, total_estimated_steps)

        for epoch_round, epoch_idx in enumerate(training_epochs):
            # Get data for this epoch
            if epoch_col:
                epoch_df = full_trace_df[full_trace_df[epoch_col] == epoch_idx].copy()
            else:
                epoch_df = full_trace_df.copy()

            if len(epoch_df) == 0:
                continue

            # Build epoch_summary for this epoch
            if "model_type" in epoch_df.columns:
                is_7b = epoch_df["model_type"].astype(str).str.lower().str.contains("7b")
                is_70b = epoch_df["model_type"].astype(str).str.lower().str.contains("70b")
                tok_col = "total_tokens" if "total_tokens" in epoch_df.columns else "num_tokens"
                if tok_col in epoch_df.columns:
                    llama7b_total = float(epoch_df[is_7b][tok_col].sum())
                    llama70b_total = float(epoch_df[is_70b][tok_col].sum())
                else:
                    llama7b_total = 0.0
                    llama70b_total = 0.0
            else:
                llama7b_total = 0.0
                llama70b_total = 0.0

            epoch_summary = {
                "llama7b_total": llama7b_total,
                "llama70b_total": llama70b_total,
                "num_datacenters": num_datacenters,
                "spec_dir": spec_dir,
                "epoch_length": epoch_length,
                "dc_carbon_intensity": real_ci,
            }

            # Build vectorized multi-agent env for this epoch
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

            if model is None:
                # First epoch: create new model
                model = PPO(
                    policy="MlpPolicy",
                    env=vec_env,
                    verbose=0,
                )
                print(f"[MARL TRAIN] Created new PPO model for {profile_id}")
            else:
                # Subsequent epochs: update environment
                model.set_env(vec_env)

            current_callbacks = []
            if DASHBOARD_AVAILABLE:
                # Create a new callback instance for this epoch's learn call
                # We use verbose=1 so you can see the [DASHBOARD DEBUG] prints
                current_callbacks.append(DashboardTrainingCallback(profile_id=profile_id, verbose=1))

            # Train on this epoch
            model.learn(
                total_timesteps=timesteps_per_epoch,
                reset_num_timesteps=False,
                callback=current_callbacks  # <--- CRITICAL ADDITION
            )

            if (epoch_round + 1) % 10 == 0 or epoch_round == len(training_epochs) - 1:
                print(f"[MARL TRAIN] {profile_id}: Completed epoch {epoch_round + 1}/{len(training_epochs)} "
                      f"(epoch_idx={epoch_idx}, rows={len(epoch_df)})")

            vec_env.close()

        # Save final model
        if model is not None:
            model.save(root_model_path)
            model.save(final_model_path)
            print(f"[MARL TRAIN] Saved {profile_id} to {root_model_path}")

    print(f"\n[MARL TRAIN] Completed training for all profiles.")