"""
utility_gnn.py — MARLIN Utility Grid Agent (Graph Attention Network)

Implements the Utility Agent from MARLIN §1.2:
  • Observation: physical state of the local distribution graph
    (line capacities, generator limits, P_res, P_dc).
  • Action:     continuous dispatch vector [0, 1] per dispatchable generator
               (Gas, Peaker) + optional LMP multiplier.
  • Reward:     minimise total generation cost while penalising blackouts.

Architecture:
  GridGATExtractor — torch_geometric GAT as a Stable-Baselines3
                     BaseFeaturesExtractor.  Falls back to a flat MLP if
                     torch_geometric is not installed.
  UtilityAgent     — thin wrapper around SB3 PPO that exposes the MARLIN
                     interface (observe → act → apply dispatch).

Why GAT?
  Standard MLPs fail to capture topological bottlenecks.  A GAT assigns
  dynamic attention weights to power-line edges.  When the DC node draws
  massive power during the evening ramp, the GAT focuses its attention on
  the congested feeder and outputs a targeted Demand Response signal
  (LMP spike) that forces the DC agent to throttle.

Reference:
  Veličković et al. (2018) "Graph Attention Networks." ICLR.
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch_geometric
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data
import torch.nn as nn
import torch.nn.functional as F

# ── Optional SB3 / PyG imports — fail gracefully ──────────────────────────
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.policies import ActorCriticPolicy
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    import gymnasium as gym
    _SB3_AVAILABLE = True
except ImportError:
    _SB3_AVAILABLE = False
    BaseFeaturesExtractor = nn.Module   # stub so class definition works
    print("[utility_gnn] stable-baselines3 not found — UtilityAgent will use standalone mode.")

try:
    from torch_geometric.nn import GATConv
    _PYG_AVAILABLE = True
except ImportError:
    _PYG_AVAILABLE = False
    print("[utility_gnn] torch_geometric not found — using edge-weighted MLP fallback.")

from grid_topology import (
    GENERATOR_SPECS, InteractiveGridNetwork,
    LINE_IMPEDANCE, LINE_CAPACITY_KW,
)

# ─────────────────────────────────────────────────────────────────────────────
# GRAPH STRUCTURE CONSTANTS
# Node indices used in the edge_index tensor (fixed topology).
# ─────────────────────────────────────────────────────────────────────────────
_NODE_NAMES = ["Solar", "Gas", "Peaker", "Bus_A", "Bus_B", "Residential", "Datacenter"]
_NODE_IDX   = {n: i for i, n in enumerate(_NODE_NAMES)}

_EDGE_LIST  = [
    ("Solar",   "Bus_A"),
    ("Gas",     "Bus_A"),
    ("Peaker",  "Bus_A"),
    ("Bus_A",   "Bus_B"),
    ("Bus_B",   "Residential"),
    ("Bus_B",   "Datacenter"),
]

_EDGE_INDEX = torch.tensor(
    [[_NODE_IDX[src], _NODE_IDX[dst]] for src, dst in _EDGE_LIST],
    dtype=torch.long,
).t().contiguous()   # shape [2, num_edges]

_EDGE_ATTR = torch.tensor(
    [LINE_IMPEDANCE.get((src, dst), 0.05) for src, dst in _EDGE_LIST],
    dtype=torch.float32,
).unsqueeze(1)       # shape [num_edges, 1]

NUM_NODES     = len(_NODE_NAMES)
NODE_FEAT_DIM = 4    # [gen_frac, load_frac, impedance_proxy, is_congested]
EDGE_FEAT_DIM = 1    # [line_impedance]
GAT_OUT_DIM   = 32   # GAT output per node
GAT_HEADS     = 4


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH STATE BUILDER
# Converts InteractiveGridNetwork state into node feature matrix.
# ─────────────────────────────────────────────────────────────────────────────

def build_node_features(grid: InteractiveGridNetwork, p_dc_kw: float) -> torch.Tensor:
    """
    Build a [NUM_NODES, NODE_FEAT_DIM] feature matrix from the current grid state.

    Features per node:
      0: generation fraction (gen_kw / capacity_kw), 0 for non-generator nodes
      1: load fraction (load_kw / peak_load_kw), 0 for non-load nodes
      2: impedance proxy (mean impedance of incident edges), 0 for bus nodes
      3: congestion flag (1.0 if any incident line is congested, else 0.0)
    """
    dispatch  = grid._dispatch
    p_res     = grid._p_res_kw
    congested = set(f"{s}-{d}" for s, d in grid._congested_lines)

    def _is_congested(node_name: str) -> float:
        for (s, d) in _EDGE_LIST:
            if (s == node_name or d == node_name) and f"{s}-{d}" in congested:
                return 1.0
        return 0.0

    rows = []
    for name in _NODE_NAMES:
        if name in GENERATOR_SPECS:
            spec = GENERATOR_SPECS[name]
            gen_frac  = dispatch.get(name, 0.0) / max(spec["capacity_kw"], 1.0)
            load_frac = 0.0
        elif name == "Residential":
            gen_frac  = 0.0
            load_frac = p_res / max(12_000.0, 1.0)
        elif name == "Datacenter":
            gen_frac  = 0.0
            load_frac = p_dc_kw / max(35_000.0, 1.0)
        else:
            gen_frac = load_frac = 0.0

        # Mean impedance of incident edges
        inc_imps = [v for (s, d), v in LINE_IMPEDANCE.items()
                    if s == name or d == name]
        imp_proxy = float(np.mean(inc_imps)) if inc_imps else 0.0

        rows.append([gen_frac, load_frac, imp_proxy, _is_congested(name)])

    return torch.tensor(rows, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# GAT FEATURE EXTRACTOR (Stable-Baselines3 compatible)
# ─────────────────────────────────────────────────────────────────────────────

class GridGATExtractor(BaseFeaturesExtractor):
    """
    Graph Attention Network used as a custom ``BaseFeaturesExtractor`` for
    Stable-Baselines3 PPO.

    The graph topology is fixed (see ``_EDGE_LIST``).  Node features change
    each step based on the grid state; the GAT attends over power-line edges
    and outputs a pooled feature vector for the PPO policy head.

    When PyTorch Geometric is unavailable, a plain MLP operating on the flat
    node-feature vector is substituted — this preserves training compatibility
    at the cost of losing topological attention.

    Args:
        observation_space: Gymnasium Box space of shape (16,) from
                           ``InteractiveGridNetwork.get_graph_feature_vector()``.
        features_dim:      Output feature dimension fed to the PPO policy head.
    """

    def __init__(self, observation_space, features_dim: int = 64):
        super().__init__(observation_space, features_dim)

        self._use_gat = _PYG_AVAILABLE

        if self._use_gat:
            # GAT layers: node features → attended node embeddings → global pool
            self.gat1 = GATConv(
                in_channels=NODE_FEAT_DIM,
                out_channels=GAT_OUT_DIM,
                heads=GAT_HEADS,
                concat=True,
                edge_dim=EDGE_FEAT_DIM,
                dropout=0.0,
            )
            self.gat2 = GATConv(
                in_channels=GAT_OUT_DIM * GAT_HEADS,
                out_channels=GAT_OUT_DIM,
                heads=1,
                concat=False,
                edge_dim=EDGE_FEAT_DIM,
                dropout=0.0,
            )
            # Global mean pool → linear projection to features_dim
            self.head = nn.Sequential(
                nn.Linear(GAT_OUT_DIM, features_dim),
                nn.ReLU(),
            )
            # Register fixed graph structure as buffers (non-trainable)
            self.register_buffer("_edge_index", _EDGE_INDEX)
            self.register_buffer("_edge_attr",  _EDGE_ATTR)
        else:
            # MLP fallback: flat 16-dim observation → features_dim
            obs_dim = int(np.prod(observation_space.shape))
            self.mlp = nn.Sequential(
                nn.Linear(obs_dim, 128), nn.ReLU(),
                nn.Linear(128, features_dim), nn.ReLU(),
            )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            observations: Batch of flat grid feature vectors [B, 16].

        Returns:
            Encoded features [B, features_dim].
        """
        if not self._use_gat:
            return self.mlp(observations)

        batch_size = observations.shape[0]
        outputs = []

        # Process each sample independently (graph has fixed topology)
        for i in range(batch_size):
            # Reconstruct node feature matrix from flat obs
            # Shape: [NUM_NODES, NODE_FEAT_DIM]
            x = self._flat_obs_to_node_features(observations[i])

            # GAT forward pass
            x = F.elu(self.gat1(x, self._edge_index, self._edge_attr))
            x = F.elu(self.gat2(x, self._edge_index, self._edge_attr))

            # Global mean pooling over nodes → [GAT_OUT_DIM]
            x_pool = x.mean(dim=0)
            outputs.append(x_pool)

        pooled = torch.stack(outputs, dim=0)  # [B, GAT_OUT_DIM]
        return self.head(pooled)

    @staticmethod
    def _flat_obs_to_node_features(obs_flat: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct a [NUM_NODES, NODE_FEAT_DIM] node feature matrix from a
        flat 16-dim observation vector.
        """
        dev = obs_flat.device  # [FIX] Extract the dynamic device context

        # Solar, Gas, Peaker
        x_solar = torch.stack([obs_flat[0], torch.tensor(0., device=dev), torch.tensor(0.02, device=dev),
                               (obs_flat[8] > 0.9).float()])
        x_gas = torch.stack([obs_flat[1], torch.tensor(0., device=dev), torch.tensor(0.03, device=dev),
                             (obs_flat[8] > 0.9).float()])
        x_peaker = torch.stack([obs_flat[2], torch.tensor(0., device=dev), torch.tensor(0.04, device=dev),
                                (obs_flat[8] > 0.9).float()])
        # Bus_A, Bus_B
        x_bus_a = torch.stack([obs_flat[3], obs_flat[4], torch.tensor(0.0, device=dev),
                               (obs_flat[8] > 0.9).float()])
        x_bus_b = torch.stack([obs_flat[5], obs_flat[6], torch.tensor(0.0, device=dev),
                               (obs_flat[9] > 0.9).float()])
        # Residential, Datacenter
        x_res = torch.stack([torch.tensor(0., device=dev), obs_flat[6], torch.tensor(0.01, device=dev),
                             (obs_flat[10] > 0.9).float()])
        x_dc = torch.stack([torch.tensor(0., device=dev), obs_flat[7], torch.tensor(0.02, device=dev),
                            (obs_flat[9] > 0.9).float()])

        return torch.stack([x_solar, x_gas, x_peaker, x_bus_a, x_bus_b, x_res, x_dc])

# ─────────────────────────────────────────────────────────────────────────────
# UTILITY AGENT
# ─────────────────────────────────────────────────────────────────────────────

class UtilityAgent:
    """
    MARLIN Utility Grid Agent (§1.2).

    Wraps a PPO policy (with GridGATExtractor) when SB3 is available.
    Falls back to a rule-based heuristic that ensures grid stability without
    learned parameters — useful for Phase 1 curriculum training and CI.

    Action space (continuous, clipped to [0, 1]):
        [Gas_dispatch_frac, Peaker_dispatch_frac]

    Observation space:
        16-dim flat vector from InteractiveGridNetwork.get_graph_feature_vector().

    Reward:
        InteractiveGridNetwork.compute_utility_reward(economics)
    """

    ACTION_DIM = 2   # Gas, Peaker dispatch fractions

    def __init__(self,
                 grid: InteractiveGridNetwork,
                 use_ppo: bool = True,
                 model_path: Optional[str] = None):
        """
        Args:
            grid:       InteractiveGridNetwork instance this agent controls.
            use_ppo:    If False or SB3 unavailable, use heuristic dispatch only.
            model_path: Optional path to a saved PPO checkpoint.
        """
        self.grid     = grid
        self._use_ppo = use_ppo and _SB3_AVAILABLE
        self._ppo: Optional["PPO"] = None

        if self._use_ppo:
            self._init_ppo(model_path)

        # Running stats for EMA-normalised reward baseline
        self._reward_ema: float = 0.0
        self._epoch_count: int  = 0

    def _init_ppo(self, model_path: Optional[str] = None) -> None:
        """Initialise or load the SB3 PPO agent."""
        obs_dim = 16  # matches get_graph_feature_vector() length
        act_dim = self.ACTION_DIM

        try:
            import gymnasium as gym
            from gymnasium import spaces
            obs_space = spaces.Box(
                low=-1.0, high=2.0, shape=(obs_dim,), dtype=np.float32)
            act_space = spaces.Box(
                low=0.0,  high=1.0, shape=(act_dim,), dtype=np.float32)

            policy_kwargs = dict(
                features_extractor_class=GridGATExtractor,
                features_extractor_kwargs=dict(features_dim=64),
                net_arch=[dict(pi=[64, 64], vf=[64, 64])],
            )

            # Minimal dummy env for SB3 initialisation
            dummy_env = _DummyGridEnv(obs_space, act_space)

            if model_path and os.path.exists(model_path):
                self._ppo = PPO.load(model_path, env=dummy_env)
                print(f"[UtilityAgent] Loaded PPO from {model_path}")
            else:
                self._ppo = PPO(
                    "MlpPolicy", dummy_env,
                    policy_kwargs=policy_kwargs,
                    learning_rate=3e-4,
                    n_steps=256,
                    batch_size=64,
                    gamma=0.99,
                    verbose=0,
                )
        except Exception as exc:
            print(f"[UtilityAgent] PPO init failed ({exc}) — using heuristic.")
            self._use_ppo = False

    # ── Heuristic dispatch ─────────────────────────────────────────────────

    def _heuristic_dispatch(self, obs: np.ndarray) -> np.ndarray:
        """
        Rule-based dispatch: fill load with Gas first, then Peaker as last resort.

        This mirrors the standard economic merit-order dispatch:
          1. Solar (free, non-dispatchable) is already running.
          2. Gas fills the remaining load gap.
          3. Peaker activates only when Gas is insufficient.

        The grid feature vector encodes:
          obs[6] = P_res normalised
          obs[7] = P_dc  normalised
          obs[0] = solar dispatch fraction
        """
        p_res_n   = float(obs[6])
        p_dc_n    = float(obs[7])
        p_solar_n = float(obs[0])

        total_load_n = p_res_n + p_dc_n
        net_load_n   = max(0.0, total_load_n - p_solar_n)

        # Gas capacity as fraction of total load
        gas_cap_n    = (GENERATOR_SPECS["Gas"]["capacity_kw"]    / 35_000.0)
        peaker_cap_n = (GENERATOR_SPECS["Peaker"]["capacity_kw"] / 35_000.0)

        gas_dispatch    = min(1.0, net_load_n / max(gas_cap_n, 1e-4))
        remaining       = max(0.0, net_load_n - gas_dispatch * gas_cap_n)
        peaker_dispatch = min(1.0, remaining / max(peaker_cap_n, 1e-4))

        # During evening ramp (17–21h), pre-activate Gas to avoid scramble
        hour = self.grid._hour
        if 17.0 <= hour <= 21.0:
            gas_dispatch = min(1.0, gas_dispatch * 1.15)

        return np.array([gas_dispatch, peaker_dispatch], dtype=np.float32)

    # ── Public interface ───────────────────────────────────────────────────

    def act(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """
        Select a dispatch action given the current grid observation.

        Args:
            obs:           16-dim flat observation from get_graph_feature_vector().
            deterministic: Use the policy mean (no exploration noise).

        Returns:
            action: [Gas_frac, Peaker_frac] in [0, 1].
        """
        if self._use_ppo and self._ppo is not None:
            try:
                action, _ = self._ppo.predict(obs, deterministic=deterministic)
                return np.clip(action, 0.0, 1.0).astype(np.float32)
            except Exception:
                pass   # fall through to heuristic

        return self._heuristic_dispatch(obs)

    def dispatch_to_fracs(self, action: np.ndarray) -> Dict[str, float]:
        """Convert [Gas_frac, Peaker_frac] action to named dispatch dict."""
        return {
            "Gas":    float(np.clip(action[0], 0.0, 1.0)),
            "Peaker": float(np.clip(action[1], 0.0, 1.0)),
        }

    def compute_lmp_signal(self, economics: Dict[str, float]) -> float:
        """
        Return the LMP ($/kWh) to signal to the Datacenter Agent.

        During congestion or scarcity, this is elevated above the base TOU
        rate to trigger load shifting or throttling by the DC agent.
        """
        return float(economics.get("lmp_dc", 0.10))

    def update_ema_reward(self, reward: float) -> None:
        """Track an EMA of utility rewards for logging/convergence detection."""
        alpha = 0.05
        self._reward_ema = (1 - alpha) * self._reward_ema + alpha * reward
        self._epoch_count += 1

    def save(self, path: str) -> None:
        """Persist the PPO policy to disk."""
        if self._use_ppo and self._ppo:
            self._ppo.save(path)
            print(f"[UtilityAgent] Saved PPO to {path}")

    def load(self, path: str) -> bool:
        """Load a previously saved PPO policy."""
        if not _SB3_AVAILABLE:
            return False
        try:
            self._ppo = PPO.load(path)
            self._use_ppo = True
            print(f"[UtilityAgent] Loaded PPO from {path}")
            return True
        except Exception as exc:
            print(f"[UtilityAgent] Load failed: {exc}")
            return False


# ─────────────────────────────────────────────────────────────────────────────
# DUMMY GYM ENVIRONMENT (SB3 initialisation only)
# ─────────────────────────────────────────────────────────────────────────────

if _SB3_AVAILABLE:
    import gymnasium as gym

    class _DummyGridEnv(gym.Env):
        """Minimal Gymnasium env so PPO can initialise its internal spaces."""
        def __init__(self, obs_space, act_space):
            super().__init__()
            self.observation_space = obs_space
            self.action_space      = act_space

        def reset(self, **kwargs):
            return self.observation_space.sample(), {}

        def step(self, action):
            obs  = self.observation_space.sample()
            return obs, 0.0, False, False, {}

        def render(self):
            pass
else:
    class _DummyGridEnv:
        pass