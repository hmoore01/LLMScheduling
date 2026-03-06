"""
LA_Hyper_DDQN.py  —  Offline-Online Hybrid PSL-MORL Framework
═══════════════════════════════════════════════════════════════════════════════

Architecture overview
─────────────────────
This framework separates learning into two distinct phases that run every epoch:

  OFFLINE BASE (OfflineBaseAgent)
  ─────────────────────────────────
  • Large Transformer actor + critic (hidden_dim=128, n_layers=3, n_heads=4).
  • Trained with GAMMA=0.9 and full target networks — temporal credit
    assignment is active and meaningful here because the offline buffer
    accumulates transitions from many epochs, giving the network a genuine
    multi-step value signal to bootstrap from.
  • Persistent cross-epoch replay buffer (OFFLINE_MEMORY_SIZE=100k).
    Transitions from both the online exploration and exploitation phases are
    pushed here, so the offline model continuously learns from a growing,
    diverse experience set.
  • Offline training runs concurrently with the exploitation parallel
    evaluation (ThreadPoolExecutor I/O), hiding most of its wall-clock cost.

  ONLINE ADAPTER (OnlineAdapterAgent)
  ────────────────────────────────────
  • Identical architecture but initialized from offline weights at the start
    of every epoch (a direct weight copy, not distillation).
  • Adapted with GAMMA=0.0 (contextual bandit) — no target networks needed —
    so each gradient step is fast and converges quickly to current conditions.
  • Small per-epoch buffer (ONLINE_MEMORY_SIZE=5k) is cleared each epoch;
    the adapter only sees the current epoch's data, giving it strong
    inductive bias toward the present state without being contaminated by
    stale history.
  • ONLINE_OPTIM_STEPS=200 sim calls × ONLINE_GRAD_STEPS_PER_ENV=4 =
    800 online gradient steps, keeping wall-clock cost firmly in budget.

  SEPARATION OF CONCERNS
  ──────────────────────
  The offline model answers: "What is the generally optimal policy for this
  DC topology, given the full history of conditions?"
  The online adapter answers: "Given the offline prior, how should I adjust
  for the specific carbon intensities, TOU prices, and traffic load right now?"

  This is analogous to a pre-trained foundation model (offline) being
  fine-tuned on a specific downstream task (online).

Wall-clock budget (15 min / epoch)
────────────────────────────────────
  Phase 1 – Online exploration   : 200 sim calls  + 800  grad steps
  Phase 2 – Exploitation eval    : ~49 sim calls (parallelised, 8 workers)
  Phase 3 – Offline training     : 0 sim calls   + 350  grad steps
             ↳ runs concurrently with Phase 2 (futures submitted, then train)
  Total sim calls: ~249  |  Total grad steps: ~1150
═══════════════════════════════════════════════════════════════════════════════
"""

import copy
import hashlib
import math
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import Rate_Flow_Sim

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# Sim / action space
NUM_NODE_TYPES   = 6
NUM_MODEL_CLASSES = 2
PREF_DIM         = 4
COND_DIM         = 8   # PREF_DIM + 4 constraint slack channels

# Online adapter (fast, contextual bandit)
ONLINE_OPTIM_STEPS        = 200   # sim calls per epoch
ONLINE_GRAD_STEPS_PER_ENV = 4     # 200 × 4 = 800 online gradient steps
ONLINE_BATCH_SIZE         = 64
ONLINE_MEMORY_SIZE        = 5_000
ONLINE_LR_ACTOR           = 3e-4  # slightly higher LR for fast adaptation
ONLINE_LR_CRITIC          = 6e-4
GAMMA_ONLINE              = 0.0   # contextual bandit; no target networks needed

# Offline base (thorough, temporal credit assignment)
OFFLINE_GRAD_STEPS        = 350   # gradient steps; runs during exploitation eval
OFFLINE_BATCH_SIZE        = 128   # larger batches for offline stability
OFFLINE_MEMORY_SIZE       = 100_000
OFFLINE_LR_ACTOR          = 5e-5  # conservative LR; offline model should be stable
OFFLINE_LR_CRITIC         = 1e-4
GAMMA_OFFLINE             = 0.9   # full Bellman bootstrapping
TAU_OFFLINE               = 0.005  # soft target-update coefficient

# Transformer dimensions
HIDDEN_DIM_OFFLINE = 128    # offline: 3-layer Transformer
N_LAYERS_OFFLINE   = 3
HIDDEN_DIM_ONLINE  = 128    # online adapter: 2-layer Transformer (same width, fewer layers)
N_LAYERS_ONLINE    = 2
N_HEADS            = 4      # shared; must divide hidden_dim evenly

# Exploration
NOISE_INIT  = 0.5
NOISE_FLOOR = 0.10
NOISE_DECAY = 0.9975   # per online gradient step; 800 steps → approaches floor near epoch end

REWARD_CLIP = 5.0    # clip raw (pre-0.01-scale) reward to ±5 before buffer push

# Per-DC-node-type request capacity.
# Calibrated to auto-scale prediction: 64.9% utilization at 250k req / 72 nodes
# → per-node throughput ≈ 250k/(72×0.649) ≈ 5 350.  Use 5 000 as a round floor.
# Routing and slider enforcement both use this constant so they stay consistent.
REQUESTS_PER_NODE_CAP = 5_000


# ── ADAPTIVE METRIC NORMALIZER ────────────────────────────────────────────────
class MetricNormalizer:
    """
    EMA tracker for observed metric magnitudes across all scored solutions.

    Why this is necessary
    ─────────────────────
    The reward function uses per-metric normalization denominators calibrated at
    a specific workload scale.  When the simulator applies a large traffic
    multiplier, metrics jump by orders of magnitude.  With a hardcoded denominator
    of 90 for carbon, a 700 000x scale gives norm_carbon ≈ 22 instead of ≈ 1,
    making weighted_penalty * 4 ≈ 88 instead of ≈ 4.  The gradient signal is
    then dominated by scale artefacts rather than preference differences.

    This class tracks EMA estimates of typical metric values.  On the first
    observation the EMA is seeded directly; thereafter it decays slowly (alpha=0.05)
    so denominators track epoch-level trends without chasing noise.

    sla_target tracks the EMA of actually-observed serve rates so the SLA penalty
    does not fire permanently when infrastructure is at capacity — the target
    converges to what is genuinely achievable under the current load.
    """
    EMA_ALPHA_METRIC = 0.05   # slow metric EMA — tracks epoch-level trends
    EMA_ALPHA_SLA    = 0.15   # faster SLA EMA — adapts to capacity changes quickly
    FLOOR            = 1e-6

    def __init__(self):
        self.ttft       = None
        self.carbon     = None
        self.water      = None
        self.cost       = None
        self.sla_target = 0.80   # adaptive: EMA of observed serve rates (soft floor 0.70)
        self.n_obs      = 0

    def update(self, metrics: dict):
        """Call with the raw metrics dict from run_epoch before scoring."""
        ttft   = float(metrics.get("avg_ttft",         0.0))
        carbon = float(metrics.get("carbon_emissions",  0.0)) / 1000.0
        water  = float(metrics.get("water_usage",       0.0)) / 100.0
        cost   = float(metrics.get("energy_cost",       0.0))

        req_done = float(metrics.get("requests_completed", metrics.get("served_requests", 0.0)))
        req_drop = float(metrics.get("requests_dropped", 0.0))
        req_tot  = max(0.0, req_done + req_drop)
        if req_tot > 0.0:
            sr = req_done / req_tot
            # sla_target tracks achievable serve rates; never let it drop below 0.70
            self.sla_target = max(0.70,
                (1 - self.EMA_ALPHA_SLA) * self.sla_target + self.EMA_ALPHA_SLA * sr)

        vals = (max(abs(ttft),   self.FLOOR),
                max(abs(carbon), self.FLOOR),
                max(abs(water),  self.FLOOR),
                max(abs(cost),   self.FLOOR))

        if self.n_obs == 0:
            self.ttft, self.carbon, self.water, self.cost = vals
        else:
            a = self.EMA_ALPHA_METRIC
            self.ttft   = (1 - a) * self.ttft   + a * vals[0]
            self.carbon = (1 - a) * self.carbon + a * vals[1]
            self.water  = (1 - a) * self.water  + a * vals[2]
            self.cost   = (1 - a) * self.cost   + a * vals[3]
        self.n_obs += 1

    @property
    def ready(self) -> bool:
        return self.n_obs > 0

    def denominators(self):
        """Return (d_ttft, d_carbon, d_water, d_cost) safe for division."""
        if not self.ready:
            return 100.0, 100.0, 100.0, 100.0   # pre-seed fallback (will be replaced at first update)
        return (max(self.ttft,   self.FLOOR),
                max(self.carbon, self.FLOOR),
                max(self.water,  self.FLOOR),
                max(self.cost,   self.FLOOR))


# ── FAST REPLAY BUFFER ────────────────────────────────────────────────────────
class ReplayBuffer:
    """Pre-allocated ring buffer. Uses np.random.randint (no Fisher-Yates shuffle)."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._buf     = [None] * capacity
        self.position = 0
        self._len     = 0

    def push(self, state, pref, action, reward, next_state, done):
        self._buf[self.position] = (state, pref, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
        self._len = min(self._len + 1, self.capacity)

    def sample(self, batch_size: int):
        idx   = np.random.randint(0, self._len, size=batch_size)
        batch = [self._buf[i] for i in idx]
        state, pref, action, reward, next_state, done = map(np.stack, zip(*batch))
        return (
            torch.FloatTensor(state),
            torch.FloatTensor(pref),
            torch.FloatTensor(action),
            torch.FloatTensor(reward),
            torch.FloatTensor(next_state),
            torch.FloatTensor(done),
        )

    def __len__(self):
        return self._len


# ── TRANSFORMER-BASED NETWORKS ────────────────────────────────────────────────
class AttentionPSLActor(nn.Module):
    """
    Transformer actor conditioned on preference/constraint vector.

    Per-DC token sequence: state features + preference broadcast → Transformer
    encoder → FiLM preference gating → three output heads (small routing,
    large routing, power sliders).

    hidden_dim and n_layers are parameterised so the offline (3-layer) and
    online (2-layer) variants share exactly this class.
    """

    def __init__(self, num_dcs, state_feat_per_dc,
                 pref_dim=COND_DIM, hidden_dim=128, n_heads=N_HEADS, n_layers=2):
        super().__init__()
        self.num_dcs   = num_dcs
        self.token_dim = state_feat_per_dc + pref_dim   # input dim per DC token

        self.state_embed = nn.Linear(self.token_dim, hidden_dim)
        encoder_layer    = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim * 2, batch_first=True, dropout=0.0,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # FiLM modulation — scales transformer output by a preference-derived gate
        self.pref_gate = nn.Sequential(nn.Linear(pref_dim, hidden_dim), nn.Sigmoid())

        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim + pref_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64),  nn.ReLU(),
        )
        self.out_small = nn.Linear(64, 1)
        self.out_large = nn.Linear(64, 1)
        self.out_power = nn.Linear(64, 1)

    def forward(self, state: torch.Tensor, pref: torch.Tensor) -> torch.Tensor:
        B              = state.size(0)
        state_seq      = state.view(B, self.num_dcs, -1)
        pref_expanded  = pref.unsqueeze(1).expand(-1, self.num_dcs, -1)
        tokens         = torch.cat([state_seq, pref_expanded], dim=-1)

        emb    = torch.relu(self.state_embed(tokens))
        enc    = self.transformer(emb)                               # (B, num_dcs, H)
        gate   = self.pref_gate(pref).unsqueeze(1).expand_as(enc)   # FiLM gate
        gated  = enc * (gate * 2.0)

        h      = self.action_head(torch.cat([gated, pref_expanded], dim=-1))
        return torch.cat([
            torch.softmax(self.out_small(h).squeeze(2), dim=1),
            torch.softmax(self.out_large(h).squeeze(2), dim=1),
            torch.sigmoid(self.out_power(h).squeeze(2)),
        ], dim=1)


class AttentionPSLCritic(nn.Module):
    """
    Transformer critic conditioned on preference/constraint vector.
    Maps (state, pref, action) → scalar Q-value.
    """

    def __init__(self, num_dcs, state_feat_per_dc, action_dim,
                 pref_dim=COND_DIM, hidden_dim=128, n_heads=N_HEADS, n_layers=2):
        super().__init__()
        self.num_dcs   = num_dcs
        self.token_dim = state_feat_per_dc + pref_dim

        self.state_embed = nn.Linear(self.token_dim, hidden_dim)
        encoder_layer    = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim * 2, batch_first=True, dropout=0.0,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.sa_net  = nn.Sequential(
            nn.Linear(num_dcs * hidden_dim + action_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
        )
        self.pref_net = nn.Sequential(nn.Linear(pref_dim, 256), nn.ReLU())
        self.out_net  = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, state: torch.Tensor, pref: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        B             = state.size(0)
        state_seq     = state.view(B, self.num_dcs, -1)
        pref_expanded = pref.unsqueeze(1).expand(-1, self.num_dcs, -1)
        tokens        = torch.cat([state_seq, pref_expanded], dim=-1)

        emb      = torch.relu(self.state_embed(tokens))
        enc      = self.transformer(emb).view(B, -1)                # (B, num_dcs*H)

        sa_feat  = self.sa_net(torch.cat([enc, action], dim=1))
        pf_feat  = self.pref_net(pref)
        return self.out_net(sa_feat * pf_feat)


# ── OFFLINE BASE AGENT ────────────────────────────────────────────────────────
class OfflineBaseAgent:
    """
    Larger Transformer (3 encoder layers) trained across epochs on accumulated
    experience with GAMMA=0.9 and soft target-network updates.

    The offline agent answers "what is the generally optimal policy given all
    historical transitions?".  It is never used for action selection — only
    as a weight source for the online adapter.
    """

    def __init__(self, num_dcs: int, state_feat_per_dc: int, cond_dim: int = COND_DIM):
        self.num_dcs        = num_dcs
        self.action_dim     = (num_dcs * NUM_MODEL_CLASSES) + num_dcs

        self.actor         = AttentionPSLActor(
            num_dcs, state_feat_per_dc, pref_dim=cond_dim,
            hidden_dim=HIDDEN_DIM_OFFLINE, n_layers=N_LAYERS_OFFLINE,
        )
        self.actor_target  = copy.deepcopy(self.actor)
        self.critic        = AttentionPSLCritic(
            num_dcs, state_feat_per_dc, self.action_dim, pref_dim=cond_dim,
            hidden_dim=HIDDEN_DIM_OFFLINE, n_layers=N_LAYERS_OFFLINE,
        )
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_opt  = optim.Adam(self.actor.parameters(),  lr=OFFLINE_LR_ACTOR)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=OFFLINE_LR_CRITIC)
        self.buffer     = ReplayBuffer(OFFLINE_MEMORY_SIZE)

    def train_step(self, batch_size: int = OFFLINE_BATCH_SIZE):
        if len(self.buffer) < batch_size:
            return None

        state, pref, action, reward, next_state, done = self.buffer.sample(batch_size)
        reward = reward.unsqueeze(1)
        done   = done.unsqueeze(1)

        with torch.no_grad():
            next_action  = self.actor_target(next_state, pref)
            target_value = reward + (1 - done) * GAMMA_OFFLINE * self.critic_target(next_state, pref, next_action)

        closs = nn.MSELoss()(self.critic(state, pref, action), target_value)
        self.critic_opt.zero_grad()
        closs.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        aloss = -self.critic(state, pref, self.actor(state, pref)).mean()
        self.actor_opt.zero_grad()
        aloss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        # Soft target update
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(TAU_OFFLINE * p.data + (1 - TAU_OFFLINE) * tp.data)
        for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
            tp.data.copy_(TAU_OFFLINE * p.data + (1 - TAU_OFFLINE) * tp.data)

        return closs.item()

    def run_offline_training(self, n_steps: int = OFFLINE_GRAD_STEPS):
        """Run n_steps gradient steps. Called while exploitation eval runs in parallel."""
        losses = []
        for _ in range(n_steps):
            l = self.train_step()
            if l is not None:
                losses.append(l)
        if losses:
            print(f"  [Offline] {len(losses)} grad steps — avg critic loss: {np.mean(losses):.4f}")


# ── ONLINE ADAPTER AGENT ──────────────────────────────────────────────────────
class OnlineAdapterAgent:
    """
    2-layer Transformer initialized fresh from offline weights every epoch.
    Trained with GAMMA=0.0 (contextual bandit) for fast, stable convergence
    to the current epoch's conditions without target networks.

    Holds its own small per-epoch buffer that is cleared at epoch start —
    the adapter only adapts on what it has seen this epoch.
    """

    def __init__(self, num_dcs: int, state_feat_per_dc: int, cond_dim: int = COND_DIM):
        self.num_dcs    = num_dcs
        self.cond_dim   = cond_dim
        self.action_dim = (num_dcs * NUM_MODEL_CLASSES) + num_dcs

        self.actor  = AttentionPSLActor(
            num_dcs, state_feat_per_dc, pref_dim=cond_dim,
            hidden_dim=HIDDEN_DIM_ONLINE, n_layers=N_LAYERS_ONLINE,
        )
        self.critic = AttentionPSLCritic(
            num_dcs, state_feat_per_dc, self.action_dim, pref_dim=cond_dim,
            hidden_dim=HIDDEN_DIM_ONLINE, n_layers=N_LAYERS_ONLINE,
        )
        self.actor_opt  = optim.Adam(self.actor.parameters(),  lr=ONLINE_LR_ACTOR)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=ONLINE_LR_CRITIC)
        self.buffer     = ReplayBuffer(ONLINE_MEMORY_SIZE)
        self.noise_std  = NOISE_INIT

    def load_from_offline(self, offline: OfflineBaseAgent):
        """
        Warm-start from offline base weights.

        The offline model has 3 encoder layers; the online adapter has 2.
        We copy the embedding layer, the first 2 encoder layers, all action
        heads, and the pref_gate exactly. This gives the adapter a strong
        prior without requiring identical architectures.
        """
        o_actor  = offline.actor.state_dict()
        o_critic = offline.critic.state_dict()

        def _transfer(src_sd: dict, dst: nn.Module):
            dst_sd   = dst.state_dict()
            to_load  = {}
            for k, v in src_sd.items():
                if k in dst_sd and dst_sd[k].shape == v.shape:
                    to_load[k] = v
            dst_sd.update(to_load)
            dst.load_state_dict(dst_sd)

        _transfer(o_actor,  self.actor)
        _transfer(o_critic, self.critic)

        # Reset optimisers so momentum from offline training doesn't distort
        # the first online gradient steps
        self.actor_opt  = optim.Adam(self.actor.parameters(),  lr=ONLINE_LR_ACTOR)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=ONLINE_LR_CRITIC)
        self.buffer     = ReplayBuffer(ONLINE_MEMORY_SIZE)
        self.noise_std  = NOISE_INIT

    def _bootstrap_action(self, state: np.ndarray, pref: np.ndarray) -> np.ndarray:
        """
        Quality-based routing for cold start (before the network has seen enough
        data to produce meaningful preference-conditioned outputs).

        Power sliders are now preference-aware:
        • Time preference (w_perf dominant): all needed DCs on, extras on too
          (latency trumps eco; better to have spare capacity).
        • Eco preference (w_carb/w_wat/w_cost dominant): only the minimum required
          DCs are powered on; the rest are set to 0.  This seeds the replay buffer
          with eco examples from step 1, breaking the cold-start catch-22 where the
          agent never observes positive eco reward because it never turns anything off.
        """
        n  = self.num_dcs
        s  = state.reshape(n, -1)             # (n, state_feat_per_dc)
        w_perf, w_carb, w_wat, w_cost = float(pref[0]), float(pref[1]), float(pref[2]), float(pref[3])
        eco_focus = w_carb + w_wat + w_cost

        # Invert features: higher score = better DC for this preference
        ci_inv  = 1.0 - np.clip(s[:, 0], 0.0, 1.0)
        tou_inv = 1.0 - np.clip(s[:, 1], 0.0, 1.0)
        pue_inv = 1.0 - np.clip(s[:, 2], 0.0, 1.0)
        uniform = np.ones(n, dtype=np.float64) / n

        quality = (w_carb * ci_inv + w_cost * tou_inv +
                   w_wat  * pue_inv + w_perf * uniform)
        quality = np.maximum(quality, 1e-8)
        quality /= quality.sum()

        action         = np.empty(3 * n, dtype=np.float32)
        action[0:n]    = quality.astype(np.float32)
        action[n:2*n]  = quality.astype(np.float32)

        # Power sliders: preference-aware capacity allocation
        req_intensity  = float(np.clip(s[:, 3].mean(), 0.0, 1.0))
        req_estimate   = req_intensity * 50_000    # req_intensity = req / 50k
        needed_nodes   = max(1, math.ceil(req_estimate / max(REQUESTS_PER_NODE_CAP, 1)))
        needed_dcs     = max(1, min(n, math.ceil(needed_nodes / max(NUM_NODE_TYPES, 1))))

        ranked = np.argsort(quality)[::-1]         # best DCs for this preference first

        sliders = np.zeros(n, dtype=np.float32)
        for i in range(needed_dcs):
            sliders[ranked[i]] = 1.0               # force-on best DCs for this preference

        if eco_focus < 0.4:
            # Time-dominant: turn on extras too for latency headroom
            for i in range(needed_dcs, min(n, needed_dcs + 2)):
                sliders[ranked[i]] = 0.8

        action[2*n:] = sliders
        return action

    def select_action(self, state: np.ndarray, pref: np.ndarray, exploration: bool = True) -> np.ndarray:
        pref = np.asarray(pref, dtype=np.float32)
        if pref.shape[-1] != self.cond_dim:
            raise ValueError(f"Expected cond dim {self.cond_dim}, got {pref.shape[-1]}")

        state_t = torch.FloatTensor(state).unsqueeze(0)
        pref_t  = torch.FloatTensor(pref).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state_t, pref_t).cpu().numpy()[0]

        # ── Cold-start bootstrap blend ─────────────────────────────────────
        # When the online buffer is small, the network output is near-uniform
        # (random init → uniform softmax).  Blend with a DC-quality heuristic
        # that uses state features to route toward DCs that suit the preference.
        # Bootstrap weight fades linearly to 0 as the buffer reaches 4× batch size.
        buf_fill     = len(self.buffer)
        blend_thresh = ONLINE_BATCH_SIZE * 4
        if buf_fill < blend_thresh:
            bootstrap_w = 1.0 - buf_fill / blend_thresh
            bootstrap   = self._bootstrap_action(state, pref)
            action      = (1.0 - bootstrap_w) * action + bootstrap_w * bootstrap
            # Re-normalise routing weight softmaxes after blending
            for k in range(NUM_MODEL_CLASSES):
                s_i, e_i = k * self.num_dcs, (k + 1) * self.num_dcs
                seg      = np.maximum(action[s_i:e_i], 0.0)
                seg_sum  = seg.sum()
                action[s_i:e_i] = seg / seg_sum if seg_sum > 0 else np.ones(self.num_dcs) / self.num_dcs
            action[2 * self.num_dcs:] = np.clip(action[2 * self.num_dcs:], 0.0, 1.0)

        if exploration:
            alpha        = random.choice([0.1, 0.3, 1.0])
            noise_weight = min(0.6, self.noise_std)

            # Routing noise: Dirichlet perturbation on allocation weights
            for k in range(NUM_MODEL_CLASSES):
                s, e        = k * self.num_dcs, (k + 1) * self.num_dcs
                noise       = np.random.dirichlet([alpha] * self.num_dcs)
                action[s:e] = (1 - noise_weight) * action[s:e] + noise_weight * noise

            # Power slider noise: Gaussian perturbation
            pw = 2 * self.num_dcs
            action[pw:] = np.clip(
                action[pw:] + np.random.normal(0, self.noise_std, self.num_dcs), 0.0, 1.0
            )

            # DC power shutdown exploration (30% of steps).
            # Explicitly zeros power sliders on 1–3 DCs chosen by LOWEST routing weight.
            # This seeds the replay buffer with eco examples even before the network
            # has learned to prefer them — breaking the cold-start catch-22.
            # We target LOW-weight DCs (not random ones) so we don't zero a DC that the
            # routing weights are sending traffic to, which would cause drops.
            if random.random() < 0.30 and self.num_dcs > 2:
                combined_w = (action[0:self.num_dcs] + action[self.num_dcs:2*self.num_dcs]) / 2.0
                combined_w = np.maximum(combined_w, 0.0)
                n_shutdown = random.randint(1, min(3, self.num_dcs - 2))
                # Pick lowest-weighted DCs to shut down
                shutdown_idxs = np.argsort(combined_w)[:n_shutdown]
                action[pw + shutdown_idxs] = 0.0

        return action

    def train_step(self, batch_size: int = ONLINE_BATCH_SIZE):
        if len(self.buffer) < batch_size:
            return None

        state, pref, action, reward, _, done = self.buffer.sample(batch_size)
        # GAMMA=0.0 → target is immediate reward; no target network needed
        target = reward.unsqueeze(1)

        closs = nn.MSELoss()(self.critic(state, pref, action), target)
        self.critic_opt.zero_grad()
        closs.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        aloss = -self.critic(state, pref, self.actor(state, pref)).mean()
        self.actor_opt.zero_grad()
        aloss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        self.noise_std = max(NOISE_FLOOR, self.noise_std * NOISE_DECAY)
        return closs.item()


# ── HYBRID AGENT (coordinates offline + online) ───────────────────────────────
class HybridPSLAgent:
    """
    Top-level agent. Owns the offline base and online adapter.
    Manages the offline→online weight transfer and dual-buffer experience routing.
    """

    def __init__(self, num_dcs: int, state_feat_per_dc: int, cond_dim: int = COND_DIM):
        self.num_dcs = num_dcs
        self.offline = OfflineBaseAgent(num_dcs, state_feat_per_dc, cond_dim)
        self.online  = OnlineAdapterAgent(num_dcs, state_feat_per_dc, cond_dim)

    def prepare_epoch(self):
        """Call at the start of every epoch: sync offline → online."""
        self.online.load_from_offline(self.offline)

    def select_action(self, state, pref, exploration=True):
        return self.online.select_action(state, pref, exploration)

    def push(self, state, pref, action, reward, next_state, done):
        """Push transition to BOTH buffers so offline learns from online experience."""
        self.online.buffer.push(state, pref, action, reward, next_state, done)
        self.offline.buffer.push(state, pref, action, reward, next_state, done)

    def train_online(self, n_steps: int):
        for _ in range(n_steps):
            self.online.train_step()

    def train_offline_concurrent(self, futures, fut_to_idx, candidate_solutions):
        """
        Run offline gradient steps while waiting on exploitation simulation futures.
        This hides most of the offline training cost behind I/O-bound simulator calls.
        Two nested bars are shown: one for candidates evaluated, one for offline steps.
        """
        steps_done   = 0
        steps_target = OFFLINE_GRAD_STEPS
        losses       = []
        batch_per_fut = max(1, steps_target // max(1, len(futures)))

        eval_bar    = tqdm(total=len(futures),  desc="  Phase 2 │ Exploit eval",
                           unit="cand", leave=False, dynamic_ncols=True)
        offline_bar = tqdm(total=steps_target, desc="  Phase 3 │ Offline train",
                           unit="step", leave=False, dynamic_ncols=True)
        offline_bar.set_postfix(loss=0.0)

        for fut in as_completed(futures):
            idx = fut_to_idx[fut]
            candidate_solutions[idx] = fut.result()
            eval_bar.update(1)

            for _ in range(batch_per_fut):
                if steps_done < steps_target:
                    l = self.offline.train_step()
                    if l is not None:
                        losses.append(l)
                        offline_bar.set_postfix(loss=round(float(np.mean(losses[-20:])), 4), refresh=False)
                    offline_bar.update(1)
                    steps_done += 1

        # Drain remaining offline steps
        while steps_done < steps_target:
            l = self.offline.train_step()
            if l is not None:
                losses.append(l)
                offline_bar.set_postfix(loss=round(float(np.mean(losses[-20:])), 4), refresh=False)
            offline_bar.update(1)
            steps_done += 1

        eval_bar.close()
        offline_bar.close()

        if losses:
            tqdm.write(f"  [Offline] {steps_done} grad steps — avg critic loss: {np.mean(losses):.4f}")

        return candidate_solutions


# ── PARETO TRACKER ────────────────────────────────────────────────────────────
class ParallelParetoTracker:
    def __init__(self):
        self.epoch_solutions = []
        self.epoch_count     = 0
        self.last_sim_ref    = None

    def clear_epoch(self):
        self.epoch_solutions = []

    def set_sim_ref(self, sim):
        self.last_sim_ref = sim

    def record_solution(self, metrics, weights, power_plan, mode_name="Scan"):
        active_nodes = 0
        for p in power_plan.values():
            if str(p.get("all", "")).upper() in ("IDLE", "ON"):
                active_nodes += NUM_NODE_TYPES
                continue
            active_nodes += sum(
                1 for st in p.get("unit", {}).values()
                if str(st).upper() in ("IDLE", "ON")
            )
        self.epoch_solutions.append({
            "mode":         mode_name,
            "ttft":         metrics.get("avg_ttft", 0.0),
            "carbon":       metrics.get("carbon_emissions", 0.0) / 1000.0,
            "water":        metrics.get("water_usage",      0.0) / 100.0,
            "cost":         metrics.get("energy_cost",      0.0),
            "total_energy": metrics.get("total_energy",     0.0),
            "served":       float(metrics.get("requests_completed", metrics.get("served_requests", 0.0))),
            "dropped":      float(metrics.get("requests_dropped", 0.0)),
            "active_nodes": active_nodes,
            "weights":      weights,
            "power_plan":   power_plan,
        })

    def increment_epoch(self):
        self.epoch_count += 1

    def get_report(self):
        if not self.epoch_solutions:
            return "No data."
        report = [
            f"\n=== EPOCH {self.epoch_count - 1} PARETO FRONT EVALUATION ===",
            f"{'Mode':<18} | {'TTFT(s)':<8} | {'Carb(kg)':<8} | {'Wat(L)':<8} | "
            f"{'Cost($)':<8} | {'Energy(kWh)':<11} | {'Served':<6} | {'Drop%':<6} | {'ActTypes'}",
            "-" * 122,
        ]
        for s in sorted(self.epoch_solutions, key=lambda x: x["carbon"]):
            total    = s["served"] + s["dropped"]
            drop_pct = (100.0 * s["dropped"] / total) if total > 0.0 else 0.0
            report.append(
                f"{s['mode']:<18} | {s['ttft']:.4f}   | {s['carbon']:.3f}     | "
                f"{s['water']:.3f}    | {s['cost']:.3f}    | {s['total_energy']:.3f}       | "
                f"{int(s['served']):<6} | {drop_pct:5.1f}% | {s['active_nodes']}"
            )
        return "\n".join(report)


# ── GLOBALS ───────────────────────────────────────────────────────────────────
_PARETO_TRACKER = ParallelParetoTracker()
_GLOBAL_AGENT: HybridPSLAgent = None
_NORM           = MetricNormalizer()    # persistent across epochs; seeded from first observations


# ── UTILITIES ─────────────────────────────────────────────────────────────────
def _stable_hash_int(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest()[:16], 16)


def _normalize_weights(w: np.ndarray) -> np.ndarray:
    out = np.maximum(0.0, np.asarray(w, dtype=np.float64))
    s   = float(out.sum())
    if s <= 0.0:
        out[:] = 0.0; out[0] = 1.0
        return out
    return out / s


def get_rich_state(sim, dc_ids, epoch_data, epoch_idx: int,
                   prev_utilisation: np.ndarray = None,
                   prev_power_sliders: np.ndarray = None) -> np.ndarray:
    """
    Build the per-DC state tensor.  Now 6 features per DC (was 4):

      [0] carbon_intensity_norm   ci / 1000
      [1] tou_price_norm          price × 5
      [2] pue_norm                pue / 2
      [3] req_intensity           min(total_req / 50k, 1.0)  — rescaled from /1000
      [4] active_nodes_frac       active_nodes / NUM_NODE_TYPES  (0 if unknown)
      [5] prev_utilisation        last epoch's DC utilisation (0 if first epoch)

    The req_intensity normalisation is corrected from /1000 to /50000 so that a
    250k-request epoch reads ~1.0 instead of the previous value of 250 (clipped to 1).
    The extra features give the Transformer direct visibility into per-DC capacity
    and utilisation, enabling it to learn load-aware routing.
    """
    num_dcs = len(dc_ids)
    state   = np.zeros((num_dcs, 6), dtype=np.float32)

    total_requests = len(epoch_data) if epoch_data is not None else 0
    epoch_hour     = int(epoch_idx % 24)

    # Active nodes per DC from previous power plan (0 if unavailable)
    if prev_power_sliders is not None:
        active_fracs = _active_nodes_per_dc(prev_power_sliders) / max(NUM_NODE_TYPES, 1)
    else:
        active_fracs = np.zeros(num_dcs, dtype=np.float32)

    for idx, dc_id in enumerate(dc_ids):
        ci, cost, water_val = 400.0, 0.10, 1.18
        if hasattr(sim, "datacenters") and dc_id in sim.datacenters:
            dc = sim.datacenters[dc_id]
            ci = float(getattr(dc, "carbon_intensity_g_per_kwh", 400.0))
            try:
                tou = getattr(dc, "tou_price", None)
                if isinstance(tou, (list, tuple)) and len(tou) == 24:
                    cost = float(tou[epoch_hour])
                else:
                    cost = float(getattr(dc, "tou_price", [0.10])[0])
            except Exception:
                pass
            water_val = float(getattr(dc, "pue_value", 1.18))

        util = float(prev_utilisation[idx]) if prev_utilisation is not None else 0.0

        # req_intensity: fraction of a comfortable per-epoch max (~50k req/epoch)
        req_intensity = min(total_requests / 50_000.0, 1.0)

        state[idx] = [
            ci / 1000.0,
            cost * 5.0,
            water_val / 2.0,
            req_intensity,
            float(active_fracs[idx]),
            np.clip(util, 0.0, 1.0),
        ]
    return state


def _budget_ratio(constraints: dict, key: str, default: float) -> float:
    cfg    = constraints.get(key, {})
    budget = float(cfg.get("budget", 0.0)) if isinstance(cfg, dict) else 0.0
    if budget <= 0.0:
        return 1.0
    return float(np.clip(budget / max(default, 1e-6), 0.0, 2.0))


def build_condition_vector(pref_vec: np.ndarray, constraints: dict) -> np.ndarray:
    pref = np.asarray(pref_vec, dtype=np.float32)
    if pref.shape[0] != PREF_DIM:
        raise ValueError(f"Expected pref dim {PREF_DIM}, got {pref.shape[0]}")
    slack = np.array([
        _budget_ratio(constraints, "ttft",   50.0),
        _budget_ratio(constraints, "carbon", 90.0),
        _budget_ratio(constraints, "water",  40.0),
        max(_budget_ratio(constraints, "cost", 30.0),
            _budget_ratio(constraints, "total_energy", 260.0)),
    ], dtype=np.float32)
    return np.concatenate([pref, slack])


def build_power_plan_sliding(dc_ids, slider_values) -> dict:
    plan = {}
    for idx, dc_id in enumerate(dc_ids):
        n = min(max(int(np.floor(float(slider_values[idx]) * (NUM_NODE_TYPES + 0.99))), 0), NUM_NODE_TYPES)
        plan[int(dc_id)] = ({"all": "OFF"} if n == 0
                            else {"unit": {str(t): "IDLE" if t < n else "OFF" for t in range(NUM_NODE_TYPES)}})
    return plan


def _active_nodes_per_dc(power_sliders) -> np.ndarray:
    """Convert power sliders → integer active node count per DC."""
    sliders = np.asarray(power_sliders, dtype=np.float64)
    return np.array(
        [min(max(int(np.floor(s * (NUM_NODE_TYPES + 0.99))), 0), NUM_NODE_TYPES)
         for s in sliders],
        dtype=np.int32,
    )


def build_schedule_map(small_indices, large_indices, dc_ids, w_small, w_large,
                       power_sliders, epoch_idx, token_counts: np.ndarray = None) -> dict:
    """
    Capacity-aware request routing with single-pass overflow redistribution.

    Previous sequential overflow loop bug
    ──────────────────────────────────────
    The previous version iterated over DCs one at a time.  When DC0's overflow
    was redistributed to DCs 1–11, their counts increased before they were
    checked against their own caps.  This caused a cascade:
      DC0 overflow → raised DC1 count above its cap → DC1 overflow → raised DC2
      count → ... → at the end all remaining overflow piled on argmax(cap_weights).
    A single DC ended up with 30k+ requests per node type → deep queues → 120s TTFT.

    Single-pass fix
    ───────────────
    1. Compute all caps and overflows simultaneously from the initial allocation.
    2. Compute remaining spare capacity across all DCs simultaneously.
    3. Redistribute overflow proportionally to spare capacity in one operation.
    4. If total load > total capacity, the excess is left UNROUTED — the simulator
       will handle these as queue-level drops.  Piling them on one DC is always
       worse: it converts "polite drops" into "everyone waits behind a huge queue".

    Token-weighted load estimation
    ──────────────────────────────
    If token_counts is provided, each request's weight is proportional to its
    token count relative to the median.  Heavy requests (large token count) count
    as more than 1 unit of load, preventing a DC from being overwhelmed by a few
    very expensive requests while appearing nominally under-count-cap.
    """
    if not small_indices and not large_indices:
        return {"map": {}}

    active_nodes = _active_nodes_per_dc(power_sliders)   # (n_dcs,) int array
    n_dcs        = len(dc_ids)

    # Use the module-level constant so routing and slider enforcement are consistent.
    dc_capacity = active_nodes.astype(np.float64) * REQUESTS_PER_NODE_CAP   # (n_dcs,)

    def _load_weight(indices: list) -> float:
        """Return total load weight for a set of request indices."""
        if token_counts is None or len(indices) == 0:
            return float(len(indices))
        tc = token_counts[np.asarray(indices, dtype=np.int64)]
        median_tc = float(np.median(tc)) if len(tc) > 0 else 1.0
        return float(np.sum(np.clip(tc / max(median_tc, 1.0), 0.2, 5.0)))

    def allocate(req_indices: list, pref_weights: np.ndarray, bucket: str) -> dict:
        if not req_indices:
            return {}

        n_req = len(req_indices)

        # ── Step 1: effective allocation weights (preference × capacity) ───
        # A DC with 0 active nodes cannot receive traffic.
        eff_w = np.asarray(pref_weights, dtype=np.float64) * active_nodes.astype(np.float64)
        total_eff = eff_w.sum()
        if total_eff <= 0.0:
            # All DCs offline or zero-weight — fall back to equal share of active DCs
            eff_w = (active_nodes > 0).astype(np.float64)
            total_eff = eff_w.sum()
            if total_eff <= 0.0:
                return {}           # nothing online, drop everything
        eff_w /= total_eff

        # ── Step 2: initial allocation ────────────────────────────────────
        raw_counts = eff_w * n_req
        counts     = np.floor(raw_counts).astype(np.int64)
        remainder  = n_req - int(counts.sum())
        frac_order = np.argsort(raw_counts - counts)[::-1]
        counts[frac_order[:remainder]] += 1

        # ── Step 3: single-pass overflow detection ────────────────────────
        # Compute all overflows from the initial allocation simultaneously.
        # DCs with active_nodes == 0 have capacity 0 → any count is overflow.
        overflow_vec = np.maximum(0, counts - dc_capacity.astype(np.int64))
        counts      -= overflow_vec
        total_overflow = int(overflow_vec.sum())

        if total_overflow > 0:
            # ── Step 4: single-pass proportional redistribution ───────────
            # spare[j] = how many MORE requests DC j can take after initial alloc
            spare = np.maximum(0.0, dc_capacity - counts.astype(np.float64))
            spare[active_nodes == 0] = 0.0           # offline DCs have no spare
            total_spare = spare.sum()

            if total_spare > 0.0:
                # Distribute overflow proportionally to available spare capacity.
                extra   = np.floor(spare / total_spare * total_overflow).astype(np.int64)
                leftover = total_overflow - int(extra.sum())
                # Give fractional leftovers to the DCs with the most remaining spare
                spare_order = np.argsort(spare)[::-1]
                extra[spare_order[:leftover]] += 1
                counts += extra
                # If any DC is now over cap due to discrete rounding, trim it
                overshoot = np.maximum(0, counts - dc_capacity.astype(np.int64))
                counts -= overshoot
                # The trimmed amount is genuinely unroutable — leave it unrouted.
                # (The simulator handles these as drops, spreading queue depth evenly.)
            # else: total system capacity exceeded — leave excess unrouted entirely.
            # DO NOT pile overflow on a single DC (the previous behaviour that caused
            # 30k+ requests per node type and 120s TTFT).

        # ── Step 5: deterministic request ordering + assignment ───────────
        ordered = sorted(req_indices, key=lambda r: _stable_hash_int(f"{epoch_idx}:{bucket}:{r}"))
        alloc, ptr = {}, 0
        for dc_idx, count in enumerate(counts):
            for _ in range(int(count)):
                if ptr < len(ordered):
                    alloc[int(ordered[ptr])] = int(dc_ids[dc_idx])
                    ptr += 1
        return alloc

    m = {}
    m.update(allocate(small_indices, w_small, "small"))
    m.update(allocate(large_indices, w_large, "large"))
    return {"map": m}


def _precompute_request_split(sim_data: pd.DataFrame):
    if len(sim_data) == 0:
        return [], []
    col    = "model" if "model" in sim_data.columns else "model_type"
    # Match on base model name — the variant suffix "_FP16 (Base)_B16" doesn't
    # change whether a model is small or large.
    models = sim_data[col].astype(str).str.lower()
    mask   = (models.str.contains("7b") | models.str.contains("8b") | models.str.contains("small")).to_numpy()
    row    = np.arange(len(sim_data), dtype=int)
    return row[mask].tolist(), row[~mask].tolist()


def _simplex_lattice_points(dim: int, levels: int):
    if dim == 1:
        return [[levels]]
    pts = []
    for i in range(levels + 1):
        for rest in _simplex_lattice_points(dim - 1, levels - i):
            pts.append([i] + rest)
    return pts


def _farthest_point_sample(points: np.ndarray, k: int, seed_points=None) -> np.ndarray:
    if len(points) <= k:
        return points
    selected  = list(seed_points) if seed_points is not None and len(seed_points) > 0 else [points[0]]
    remaining = points.tolist()
    for s in selected:
        try:
            remaining.remove(s if isinstance(s, list) else s.tolist())
        except ValueError:
            pass
    while len(selected) < k and remaining:
        rem = np.asarray(remaining, dtype=np.float32)
        sel = np.asarray(selected,  dtype=np.float32)
        pick = int(np.argmax(np.linalg.norm(rem[:, None, :] - sel[None, :, :], axis=2).min(axis=1)))
        selected.append(rem[pick])
        remaining.pop(pick)
    return np.asarray(selected, dtype=np.float32)


def build_preference_cloud(population, target_size: int = 40) -> list:
    base    = [np.array(c["pref"], dtype=np.float32) for c in population]
    corners = [np.eye(PREF_DIM, dtype=np.float32)[i] for i in range(PREF_DIM)]
    edges   = []
    for i in range(PREF_DIM):
        for j in range(i + 1, PREF_DIM):
            v = np.zeros(PREF_DIM, dtype=np.float32); v[i] = 0.5; v[j] = 0.5
            edges.append(v)

    lattice   = np.asarray(_simplex_lattice_points(PREF_DIM, 6), dtype=np.float32) / 6.0
    rand_pool = np.random.dirichlet(np.ones(PREF_DIM), size=300).astype(np.float32)
    all_pts   = np.vstack([lattice, rand_pool])

    seed    = np.vstack(corners + edges + base + [np.full(PREF_DIM, 0.25, dtype=np.float32)])
    sampled = _farthest_point_sample(all_pts, target_size, seed_points=seed)

    result, seen = list(corners), {tuple(c.tolist()) for c in corners}
    for p in sampled:
        key = tuple(np.round(p, 4).tolist())
        if key not in seen:
            result.append(p.astype(np.float32)); seen.add(key)
    return [p.astype(np.float32) for p in sorted(result, key=lambda v: tuple(v))]


def _is_dominated(candidate: dict, others: list) -> bool:
    keys = ["avg_ttft", "carbon_emissions", "water_usage", "energy_cost"]
    c = tuple(candidate["metrics"].get(k, 1e9) for k in keys)
    for other in others:
        if other is candidate:
            continue
        o = tuple(other["metrics"].get(k, 1e9) for k in keys)
        if all(oi <= ci for oi, ci in zip(o, c)) and any(oi < ci for oi, ci in zip(o, c)):
            return True
    return False


def _score_solution(metrics, power_sliders, dc_usage, pref_vec, constraints, dc_to_idx) -> float:
    """
    Compute a scalar reward for a (metrics, action) pair.

    All metric normalization uses _NORM's adaptive EMA denominators so the
    reward magnitude stays in a stable range regardless of workload scale.
    Each normalized metric is ≈ 1.0 for a typical observation, so the total
    weighted_metric term is ≈ 1.0 and the final reward is in roughly [-5, +1].

    The SLA penalty uses _NORM.sla_target (EMA of observed serve rates) rather
    than a hardcoded 85% target.  When infrastructure is at capacity and the
    best achievable serve rate is 50%, the target converges to ~50% so the
    penalty rewards improvement over baseline rather than firing a flat maximum
    penalty on every step.
    """
    ttft         = float(metrics.get("avg_ttft", metrics.get("avg_ttft_sec", 0.0)))
    carbon       = float(metrics.get("carbon_emissions", 0.0)) / 1000.0
    water        = float(metrics.get("water_usage",      0.0)) / 100.0
    cost         = float(metrics.get("energy_cost",      0.0))
    total_energy = float(metrics.get("total_energy",     0.0))

    # ── Adaptive denominators ──────────────────────────────────────────────
    d_ttft, d_carbon, d_water, d_cost = _NORM.denominators()

    # ── Constraint lagrangian (scale-independent fractions) ───────────────
    lagrangian = 0.0
    for key, val, denom in [("carbon",       carbon,       d_carbon),
                             ("water",        water,        d_water),
                             ("cost",         cost,         d_cost),
                             ("total_energy", total_energy, max(d_cost * 8, 1.0))]:
        if key in constraints and constraints[key].get("budget", 0) > 0:
            budget = constraints[key]["budget"]
            viol   = max(0.0, (val - budget) / max(budget, 1e-6))
            lagrangian += min(8.0, constraints[key]["penalty"] * viol * 15.0)

    # ── Preference weights ─────────────────────────────────────────────────
    w_perf, w_carb, w_wat, w_cost = (float(x) for x in pref_vec)
    eco_focus = w_carb + w_wat + w_cost

    # ── Normalised metric scores (each ≈ 1.0 for a typical observation) ───
    # Cap TTFT at 2× the running typical value to limit outlier influence
    capped_ttft = min(ttft, 2.0 * d_ttft)
    norm_ttft   = capped_ttft / d_ttft
    norm_carbon = carbon      / d_carbon
    norm_water  = water       / d_water
    norm_cost   = cost        / d_cost

    # weighted_metric ≈ 1.0 for typical; lower is better
    weighted_metric = (w_perf * norm_ttft  +
                       w_carb * norm_carbon +
                       w_wat  * norm_water  +
                       w_cost * norm_cost)

    # ── Eco bonus: reward actual node/DC shutdown when traffic is served ──────
    # The eco bonus must use ACTUAL active node count (not slider average) to
    # correctly reflect powered-off infrastructure.
    # When all 12 DCs are on (72 nodes), powered_off_frac = 0 → eco_bonus = 0.
    # When 3 DCs off (9 DCs, 54 nodes), powered_off_frac = 0.25 → meaningful bonus.
    # Gated by serve_ratio so turning things off to drop traffic earns nothing.
    req_done = float(metrics.get("requests_completed", metrics.get("served_requests", 0.0)))
    req_drop = float(metrics.get("requests_dropped", 0.0))
    req_tot  = max(0.0, req_done + req_drop)
    sr       = (req_done / req_tot) if req_tot > 0.0 else 0.0
    dr       = 1.0 - sr

    active_nodes_arr    = _active_nodes_per_dc(power_sliders)
    total_possible      = max(len(power_sliders) * NUM_NODE_TYPES, 1)
    active_frac         = float(active_nodes_arr.sum()) / total_possible
    powered_off_frac    = 1.0 - active_frac

    n_dcs_off     = int(np.sum(active_nodes_arr == 0))
    total_dcs     = max(len(power_sliders), 1)
    dcs_off_frac  = n_dcs_off / total_dcs

    # ── Performance bonus for time_agent: reward having ALL nodes available ──
    # Each additional active node reduces average queue depth.  A solution with
    # 72/72 nodes active gets a full +w_perf bonus; 54/72 nodes gets 75%.
    # This gives the time_agent a clear gradient toward "turn everything on".
    perf_capacity_bonus = active_frac * w_perf * 2.0

    # ── Eco bonus: node + DC level, gated by serve rate ────────────────────
    # Use a concave curve (sqrt) so the first few DCs shut down are worth more
    # than the last few, matching the diminishing returns on carbon savings.
    eco_bonus = (
        math.sqrt(max(powered_off_frac, 0.0)) * eco_focus * sr * 2.0
        + dcs_off_frac * eco_focus * sr * 2.5
    )

    # ── Utilisation penalties ─────────────────────────────────────────────────
    # Zombie: DC that is ON (slider > 0.3) but has < 10% utilisation.
    #         Threshold raised from 0.25 to 0.10 and requires slider > 0.3 so that
    #         legitimate "low-load receiving" DCs don't trigger it.
    # Crucially: zombie PENALISES the time agent (w_perf), NOT eco agents.
    #         Eco agents deliberately concentrate load → some DCs idle → not a zombie.
    # Overload: always bad for time agent (long queues → high TTFT).
    zombie_frac = overload_frac = 0.0
    for dc_id, usage in dc_usage.items():
        util = float(usage.get("utilization", 0.0))
        idx  = dc_to_idx.get(int(dc_id))
        if idx is not None and idx < len(power_sliders):
            s = float(power_sliders[idx])
            if s > 0.3 and util < 0.10:    # fully on but nearly idle
                zombie_frac  += 1.0 / max(len(power_sliders), 1)
            if util > 0.90:                # saturated
                overload_frac += 1.0 / max(len(power_sliders), 1)

    # ── Base reward ────────────────────────────────────────────────────────────
    base = (perf_capacity_bonus
            + eco_bonus
            - weighted_metric   * 4.0
            - zombie_frac       * w_perf    * 2.0
            - overload_frac     * w_perf    * 2.0)

    # Relative TTFT hard penalty (only for preference-weighted agents)
    if ttft > 1.5 * d_ttft:
        base -= 2.0 * max(w_perf, 0.15)
    base -= norm_ttft * 1.0 * w_perf

    # ── Adaptive SLA penalty ──────────────────────────────────────────────
    # Uses _NORM.sla_target (EMA of observed serve rates) instead of a fixed
    # 85% threshold.  When the infrastructure can genuinely only serve 50% of
    # traffic, sla_target converges to ~50% and the penalty rewards improvement
    # over baseline rather than permanently maxing out.
    if req_tot > 0.0:
        deficit = max(0.0, _NORM.sla_target - sr)
        base -= deficit * 5.0          # proportional to gap vs achievable baseline
        base -= dr * 1.5               # direct drop fraction penalty
        base -= (1.0 - sr) * w_perf   # latency agents penalised more for drops
        if req_done <= 0.0:
            base -= 8.0                # all-off / total failure

    return base - lagrangian


def _bias_power_sliders(power_sliders, w_small, w_large,
                        pref_vec, has_traffic: bool, num_requests: int):
    """
    Preference-conditioned power slider nudge.

    Replaces the old floor-enforcement function. Instead of forcing every agent
    toward the same capacity floor, this function pushes each agent toward its
    natural ideal:

      time_agent  (w_perf → 1.0) : wants ALL DCs on at full power for minimum
                                   queue depth. Sliders are biased upward.

      eco agents  (w_carb/wat/cost) : want as few nodes as possible while still
                                   serving traffic. Only a hard minimum of 1 DC
                                   is enforced. The agent is free to go lower
                                   than the old capacity-formula floor and must
                                   learn from drop penalties if it over-cuts.

    The minimum-1-DC safety floor is the only hard constraint kept — removing
    even this would cause immediate total failure with no learning signal.
    """
    sliders   = np.clip(np.asarray(power_sliders, dtype=np.float64), 0.0, 1.0)
    n         = len(sliders)
    w_total   = _normalize_weights(
        (np.asarray(w_small, dtype=np.float64) + np.asarray(w_large, dtype=np.float64)) / 2.0
    )
    w_perf    = float(pref_vec[0])
    eco_focus = float(pref_vec[1] + pref_vec[2] + pref_vec[3])

    if has_traffic and num_requests > 0:
        if w_perf > 0.5:
            # ── Time-dominant: push ALL sliders toward 1.0 proportional to w_perf
            # At w_perf=1.0, every slider is floored at 1.0 (all nodes on).
            # At w_perf=0.5, every slider is floored at 0.5.
            perf_floor = w_perf
            for i in range(n):
                sliders[i] = max(sliders[i], perf_floor)

        elif eco_focus > 0.5:
            # ── Eco-dominant: do NOT enforce capacity floor.
            # Only guarantee 1 DC stays on (absolute minimum for any requests).
            # Let the network discover the right cut-off through rewards.
            if float(np.max(sliders)) < 0.01:
                best = int(np.argmax(w_total))
                sliders[best] = 1.0
        else:
            # ── Balanced: soft floor at ~50% of needed capacity
            needed_nodes = max(1, math.ceil(num_requests / max(REQUESTS_PER_NODE_CAP, 1)))
            needed_dcs   = max(1, min(n, math.ceil(needed_nodes / max(NUM_NODE_TYPES, 1))))
            ranked = np.argsort(w_total)[::-1]
            for idx in ranked[:needed_dcs]:
                sliders[int(idx)] = max(sliders[int(idx)], 0.5)

    else:
        # No traffic: pass through unchanged, agent can turn everything off
        pass

    return sliders


def _build_training_schedule(population, n_steps: int) -> list:
    """
    Pre-build the full (pref_vec, cond_vec, constraints) list so
    build_condition_vector() is not called in the hot training loop.
    """
    corners  = [np.eye(PREF_DIM, dtype=np.float32)[i] for i in range(PREF_DIM)]
    edges    = []
    for i in range(PREF_DIM):
        for j in range(i + 1, PREF_DIM):
            v = np.zeros(PREF_DIM, dtype=np.float32); v[i] = 0.5; v[j] = 0.5
            edges.append(v)
    pop_pairs = [(np.array(c["pref"], dtype=np.float32), c.get("constraints", {})) for c in population]

    schedule           = []
    c_idx = e_idx = p_idx = 0
    for _ in range(n_steps):
        roll = random.random()
        if roll < 0.25:
            pv, cons = corners[c_idx % len(corners)].copy(), {}; c_idx += 1
        elif roll < 0.45:
            pv, cons = edges[e_idx % len(edges)].copy(), {}; e_idx += 1
        elif roll < 0.65:
            pv, cons = pop_pairs[p_idx % len(pop_pairs)]; pv = pv.copy(); p_idx += 1
        elif roll < 0.85:
            pv, cons = np.random.dirichlet(np.ones(PREF_DIM)).astype(np.float32), {}
        else:
            pv, cons = np.random.dirichlet(np.ones(PREF_DIM) * 0.3).astype(np.float32), {}
        schedule.append((pv, build_condition_vector(pv, cons), cons))
    return schedule


# ── MAIN ENTRY POINT ──────────────────────────────────────────────────────────
def milp_optimizer(epoch_data, epoch_idx, node_properties, epoch_summary):
    global _PARETO_TRACKER, _GLOBAL_AGENT

    spec_dir  = epoch_summary.get("spec_dir", "sim_specs")
    epoch_len = int(epoch_summary.get("epoch_length", 900))
    temp_sim  = Rate_Flow_Sim.LLM_Simulator(spec_dir=spec_dir, epoch_length=epoch_len, debug=False)
    _PARETO_TRACKER.set_sim_ref(temp_sim)

    dc_ids       = sorted(int(dc_id) for dc_id in temp_sim.datacenters.keys()) or [0]
    real_num_dcs = len(dc_ids)
    dc_to_idx    = {int(dc_id): idx for idx, dc_id in enumerate(dc_ids)}

    if _GLOBAL_AGENT is None or _GLOBAL_AGENT.num_dcs != real_num_dcs:
        print(f"[INIT] Booting HybridPSLAgent ({real_num_dcs} DCs) — offline 3-layer / online 2-layer Transformer")
        _GLOBAL_AGENT = HybridPSLAgent(real_num_dcs, 6, cond_dim=COND_DIM)

    agent = _GLOBAL_AGENT

    # Sync offline → online at epoch start (warm-start)
    agent.prepare_epoch()
    print(f"[EPOCH {epoch_idx}] Online adapter warm-started from offline base. "
          f"Offline buffer size: {len(agent.offline.buffer)}")

    # ── Clean epoch data ──────────────────────────────────────────────────────
    clean_data = epoch_data.copy() if isinstance(epoch_data, pd.DataFrame) else pd.DataFrame(epoch_data)
    clean_data = clean_data.rename(
        columns={"source_dc_id": "source_dc", "model_type": "model", "num_tokens": "tokens"}
    )
    for col, default in [("model", "Llama7b"), ("tokens", 1024), ("source_dc", 0), ("arrival_ms", 0.0)]:
        if col not in clean_data.columns:
            clean_data[col] = default
    clean_data["source_dc"]  = pd.to_numeric(clean_data["source_dc"],  errors="coerce").fillna(0).astype(int)
    clean_data["model"]      = clean_data["model"].astype(str)
    clean_data["tokens"]     = pd.to_numeric(clean_data["tokens"],     errors="coerce").fillna(0).astype(int).clip(lower=0)
    clean_data["arrival_ms"] = pd.to_numeric(clean_data["arrival_ms"], errors="coerce").fillna(0.0).clip(lower=0.0)

    # ── Apply model variant suffix ─────────────────────────────────────────
    # The simulator performance tables are keyed on the FULL model string
    # including variant, e.g. "Llama7b_FP16 (Base)_B16".
    # Without this suffix the sim cannot find throughput data and falls back
    # to worst-case estimates — the root cause of the 35-45× TTFT gap vs Helix.
    _MODEL_VARIANT = "_FP16 (Base)_B16"
    def _apply_variant(name: str) -> str:
        return name if _MODEL_VARIANT in name else name + _MODEL_VARIANT
    clean_data["model"] = clean_data["model"].map(_apply_variant)

    clean_data = clean_data.reset_index(drop=True)

    if len(clean_data) == 0:
        print(f"[EPOCH {epoch_idx}] Zero traffic — all DCs OFF.")
        power_plan = {int(dc_id): {"all": "OFF"} for dc_id in dc_ids}
        metrics    = {k: 0.0 for k in ["avg_ttft", "carbon_emissions", "water_usage",
                                        "energy_cost", "total_energy", "requests_completed", "requests_dropped"]}
        _PARETO_TRACKER.increment_epoch()
        _PARETO_TRACKER.clear_epoch()
        _PARETO_TRACKER.record_solution(metrics, np.zeros(real_num_dcs), power_plan, "Zero_Traffic")
        print(_PARETO_TRACKER.get_report())
        return metrics, [], []

    _PARETO_TRACKER.increment_epoch()
    _PARETO_TRACKER.clear_epoch()

    current_state = get_rich_state(temp_sim, dc_ids, clean_data, epoch_idx)
    num_requests  = len(clean_data)

    # Pre-compute request split and token weights once — avoids repeated pandas calls in the loop
    small_indices, large_indices = _precompute_request_split(clean_data)
    token_counts = (
        clean_data["tokens"].to_numpy(dtype=np.float32)
        if "tokens" in clean_data.columns
        else None
    )

    population = [
        {"mode": "time_agent",       "pref": [1.0, 0.0, 0.0, 0.0], "constraints": {}},
        {"mode": "carbon_agent",     "pref": [0.0, 1.0, 0.0, 0.0], "constraints": {}},
        {"mode": "water_agent",      "pref": [0.0, 0.0, 1.0, 0.0], "constraints": {}},
        {"mode": "cost_agent",       "pref": [0.0, 0.0, 0.0, 1.0], "constraints": {}},
        {"mode": "Balanced",         "pref": [0.25, 0.25, 0.25, 0.25], "constraints": {}},
        {"mode": "green_perf",       "pref": [0.6, 0.3, 0.0, 0.1],
         "constraints": {"carbon":       {"budget": 4000.0 / 96.0, "penalty": 0.5}}},
        {"mode": "cost_guard",       "pref": [0.7, 0.0, 0.0, 0.3],
         "constraints": {"cost":         {"budget": 2800.0 / 96.0, "penalty": 0.5}}},
        {"mode": "water_saver",      "pref": [0.7, 0.0, 0.3, 0.0],
         "constraints": {"water":        {"budget": 2500.0 / 96.0, "penalty": 0.5}}},
        {"mode": "peak_power_guard", "pref": [0.8, 0.0, 0.0, 0.2],
         "constraints": {"total_energy": {"budget": 25051.0 / 96.0, "penalty": 0.3}}},
    ]

    training_schedule = _build_training_schedule(population, ONLINE_OPTIM_STEPS)

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 1 — ONLINE EXPLORATION (fast adaptation on current epoch)
    # ══════════════════════════════════════════════════════════════════════════
    phase1_bar = tqdm(
        training_schedule,
        desc=f"  Phase 1 │ Online explore (ep {epoch_idx})",
        unit="sim",
        dynamic_ncols=True,
    )
    phase1_bar.set_postfix(reward=0.0, noise=NOISE_INIT, pref="?")

    for pref_vec, cond_vec, constraints in phase1_bar:
        full_action = agent.select_action(current_state, cond_vec, exploration=True)

        w_small       = _normalize_weights(full_action[0:real_num_dcs])
        w_large       = _normalize_weights(full_action[real_num_dcs:2 * real_num_dcs])
        power_sliders = _bias_power_sliders(
            full_action[2 * real_num_dcs:], w_small, w_large, pref_vec, True, num_requests
        )
        power_plan    = build_power_plan_sliding(dc_ids, power_sliders)
        schedule_plan = build_schedule_map(
            small_indices, large_indices, dc_ids, w_small, w_large, power_sliders, epoch_idx,
            token_counts=token_counts
        )

        metrics, _, dc_usage = temp_sim.run_epoch(epoch_idx, clean_data, schedule_plan, power_plan)

        # Seed the normalizer before scoring so denominators are correct from step 1
        _NORM.update(metrics)
        reward = np.clip(
            _score_solution(metrics, power_sliders, dc_usage, pref_vec, constraints, dc_to_idx),
            -REWARD_CLIP, REWARD_CLIP
        ) * 0.01

        # Push to BOTH buffers via HybridPSLAgent
        agent.push(current_state, cond_vec, full_action, reward, current_state, False)

        # Online gradient steps (fast adaptation, GAMMA=0)
        agent.train_online(ONLINE_GRAD_STEPS_PER_ENV)

        # Update postfix: dominant preference dimension label for readability
        dom   = int(np.argmax(pref_vec))
        label = ["time", "carbon", "water", "cost"][dom]
        phase1_bar.set_postfix(
            reward=reward, noise=agent.online.noise_std, pref=label, refresh=False
        )

    phase1_bar.close()

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 2 — EXPLOITATION (preference cloud evaluation, parallelised)
    #         + PHASE 3 — OFFLINE TRAINING (interleaved with Phase 2 futures)
    # ══════════════════════════════════════════════════════════════════════════
    eval_configs = [{"pref": p, "constraints": {}} for p in build_preference_cloud(population, target_size=40)]
    eval_configs.extend(
        {"pref": np.array(c["pref"], dtype=np.float32), "constraints": c.get("constraints", {}), "mode": c["mode"]}
        for c in population
    )

    # ── Per-mode ideal seeds ────────────────────────────────────────────────
    # For each named mode we inject one "ideal" hand-crafted action that
    # represents the theoretical best for that preference. This guarantees the
    # exploitation pool always contains the true extreme solutions even when the
    # network hasn't fully converged, and forces genuine Pareto diversity.
    state_2d      = current_state.reshape(real_num_dcs, -1)
    ci_inv        = 1.0 - np.clip(state_2d[:, 0], 0.0, 1.0)
    tou_inv       = 1.0 - np.clip(state_2d[:, 1], 0.0, 1.0)
    pue_inv       = 1.0 - np.clip(state_2d[:, 2], 0.0, 1.0)
    uniform_w     = np.ones(real_num_dcs, dtype=np.float32) / real_num_dcs

    def _ideal_action(mode_name: str, pref_vec: np.ndarray) -> np.ndarray:
        """Hand-crafted action representing the ideal for this named mode."""
        n  = real_num_dcs
        pw = 2 * n
        a  = np.empty(3 * n, dtype=np.float32)

        if mode_name == "time_agent":
            # Maximum throughput: all 12 DCs fully on, uniform routing
            a[0:n]  = uniform_w
            a[n:pw] = uniform_w
            a[pw:]  = 1.0   # all sliders at full

        elif mode_name == "carbon_agent":
            # Route to lowest-carbon DCs; shut down high-carbon ones
            w = np.maximum(ci_inv, 1e-8); w /= w.sum()
            a[0:n]  = w; a[n:pw] = w
            needed  = max(1, min(n, math.ceil(num_requests / max(REQUESTS_PER_NODE_CAP * NUM_NODE_TYPES, 1))))
            ranked  = np.argsort(w)[::-1]
            a[pw:]  = 0.0
            for i in ranked[:needed]: a[pw + i] = 1.0

        elif mode_name == "water_agent":
            w = np.maximum(pue_inv, 1e-8); w /= w.sum()
            a[0:n]  = w; a[n:pw] = w
            needed  = max(1, min(n, math.ceil(num_requests / max(REQUESTS_PER_NODE_CAP * NUM_NODE_TYPES, 1))))
            ranked  = np.argsort(w)[::-1]
            a[pw:]  = 0.0
            for i in ranked[:needed]: a[pw + i] = 1.0

        elif mode_name == "cost_agent":
            w = np.maximum(tou_inv, 1e-8); w /= w.sum()
            a[0:n]  = w; a[n:pw] = w
            needed  = max(1, min(n, math.ceil(num_requests / max(REQUESTS_PER_NODE_CAP * NUM_NODE_TYPES, 1))))
            ranked  = np.argsort(w)[::-1]
            a[pw:]  = 0.0
            for i in ranked[:needed]: a[pw + i] = 1.0

        else:
            # Balanced / constrained modes: return None → use network output
            return None

        return a

    candidate_inputs = []
    for cfg in eval_configs:
        pref_vec    = np.array(cfg["pref"], dtype=np.float32)
        constraints = cfg.get("constraints", {})
        cond_vec    = build_condition_vector(pref_vec, constraints)
        mode_name   = cfg.get("mode", None)

        # Variant 0: network output (with bias applied)
        # Variant 1 (named modes only): ideal seed for this mode
        # Variant 2 (named modes only): network output with heavier perturbation
        variants_to_run = [("network", None)]
        if mode_name is not None:
            ideal = _ideal_action(mode_name, pref_vec)
            if ideal is not None:
                variants_to_run.append(("ideal", ideal))
            variants_to_run.append(("perturb", None))

        for variant_tag, forced_action in variants_to_run:
            if forced_action is not None:
                full_action = forced_action.copy()
            else:
                full_action = agent.select_action(current_state, cond_vec, exploration=False)
                if variant_tag == "perturb":
                    for k in range(NUM_MODEL_CLASSES):
                        s_i, e_i = k * real_num_dcs, (k + 1) * real_num_dcs
                        noise    = np.random.dirichlet([0.3] * real_num_dcs)
                        full_action[s_i:e_i] = 0.6 * full_action[s_i:e_i] + 0.4 * noise
                        seg = np.maximum(full_action[s_i:e_i], 0.0)
                        full_action[s_i:e_i] = seg / seg.sum() if seg.sum() > 0 else seg
                    pw = 2 * real_num_dcs
                    full_action[pw:] = np.clip(
                        full_action[pw:] + np.random.normal(0, 0.2, real_num_dcs), 0.0, 1.0
                    )

            w_small       = _normalize_weights(full_action[0:real_num_dcs])
            w_large       = _normalize_weights(full_action[real_num_dcs:2 * real_num_dcs])
            power_sliders = _bias_power_sliders(
                full_action[2 * real_num_dcs:], w_small, w_large, pref_vec, True, num_requests
            )
            candidate_inputs.append({
                "pref_vec":      pref_vec,
                "constraints":   constraints,
                "power_plan":    build_power_plan_sliding(dc_ids, power_sliders),
                "schedule_plan": build_schedule_map(
                    small_indices, large_indices, dc_ids, w_small, w_large, power_sliders, epoch_idx,
                    token_counts=token_counts
                ),
                "w_total":       (w_small + w_large) / 2.0,
                "power_sliders": power_sliders,
            })

    def _evaluate_candidate(cand):
        local_sim = Rate_Flow_Sim.LLM_Simulator(spec_dir=spec_dir, epoch_length=epoch_len, debug=False)
        m, r, u   = local_sim.run_epoch(epoch_idx, clean_data, cand["schedule_plan"], cand["power_plan"])
        return {**cand, "metrics": m, "results": r, "dc_usage": u}

    candidate_solutions = [None] * len(candidate_inputs)
    max_workers = min(max(1, os.cpu_count() or 1), 8, len(candidate_inputs))

    if max_workers <= 1:
        # Sequential: run offline training between candidates
        seq_bar = tqdm(
            enumerate(candidate_inputs),
            total=len(candidate_inputs),
            desc="  Phase 2+3 │ Exploit+Offline (sequential)",
            unit="cand",
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        )
        offline_losses_seq = []
        for i, cand in seq_bar:
            candidate_solutions[i] = _evaluate_candidate(cand)
            steps_per_cand = max(1, OFFLINE_GRAD_STEPS // max(1, len(candidate_inputs)))
            for _ in range(steps_per_cand):
                l = agent.offline.train_step()
                if l is not None:
                    offline_losses_seq.append(l)
            if offline_losses_seq:
                seq_bar.set_postfix(off_loss=f"{np.mean(offline_losses_seq[-10:]):.4f}", refresh=False)
        seq_bar.close()
    else:
        # Parallel: interleave offline gradient steps with incoming futures
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            fut_to_idx = {ex.submit(_evaluate_candidate, c): i for i, c in enumerate(candidate_inputs)}
            candidate_solutions = agent.train_offline_concurrent(
                list(fut_to_idx.keys()), fut_to_idx, candidate_solutions
            )

    # Push exploitation solutions into both buffers
    for cand in candidate_solutions:
        _NORM.update(cand["metrics"])
        cond_vec = build_condition_vector(cand["pref_vec"], cand["constraints"])
        reward   = np.clip(
            _score_solution(
                cand["metrics"], cand["power_sliders"], cand["dc_usage"],
                cand["pref_vec"], cand["constraints"], dc_to_idx
            ),
            -REWARD_CLIP, REWARD_CLIP
        ) * 0.01
        approx_action = np.concatenate([cand["w_total"], cand["w_total"], cand["power_sliders"]])
        agent.push(current_state, cond_vec, approx_action, reward, current_state, False)

    # ══════════════════════════════════════════════════════════════════════════
    # ASSIGN BEST CANDIDATE PER NAMED MODE
    # ══════════════════════════════════════════════════════════════════════════
    # Single-objective modes (time/carbon/water/cost) pick by their RAW METRIC
    # directly — argmin(ttft), argmin(carbon_emissions), etc.
    # Using composite _score_solution for these modes caused them to pick
    # higher-carbon / higher-latency candidates because SLA/drop penalties
    # outweighed the metric improvement (e.g. a 106 kg carbon sample with 65%
    # drops scores worse than a 151 kg sample with 43% drops even for a pure
    # carbon preference).
    #
    # Multi-objective and constrained modes (Balanced, green_perf, etc.) still
    # use composite scoring because they genuinely need to balance objectives.
    best_balanced_metrics = best_balanced_results = None

    # Map mode name → (raw metric key, minimise=True)
    SINGLE_OBJ_METRIC = {
        "time_agent":  "avg_ttft",
        "carbon_agent": "carbon_emissions",
        "water_agent":  "water_usage",
        "cost_agent":   "energy_cost",
    }

    # Pre-score every candidate for multi-objective modes
    all_scores = {}
    for config in population:
        mode_name = config["mode"]
        if mode_name in SINGLE_OBJ_METRIC:
            continue
        pref_vec    = np.array(config["pref"], dtype=np.float32)
        constraints = config.get("constraints", {})
        for idx, cand in enumerate(candidate_solutions):
            all_scores[(mode_name, idx)] = _score_solution(
                cand["metrics"], cand["power_sliders"], cand["dc_usage"],
                pref_vec, constraints, dc_to_idx
            )

    for config in population:
        mode_name = config["mode"]

        if mode_name in SINGLE_OBJ_METRIC:
            # Pick the globally best candidate by raw metric value (minimise)
            metric_key = SINGLE_OBJ_METRIC[mode_name]
            best_idx   = min(range(len(candidate_solutions)),
                             key=lambda i, mk=metric_key: candidate_solutions[i]["metrics"].get(mk, float("inf")))
        else:
            best_idx = max(range(len(candidate_solutions)),
                           key=lambda i, mn=mode_name: all_scores[(mn, i)])

        best_cand = candidate_solutions[best_idx]
        _PARETO_TRACKER.record_solution(
            best_cand["metrics"], best_cand["w_total"], best_cand["power_plan"], mode_name
        )
        if mode_name == "Balanced":
            best_balanced_metrics = best_cand["metrics"]
            best_balanced_results = best_cand["results"]

    # ══════════════════════════════════════════════════════════════════════════
    # PARETO SAMPLE SELECTION — maximally spread non-dominated front
    # ══════════════════════════════════════════════════════════════════════════
    # Run non-domination filter over the ENTIRE candidate pool (not just
    # "unassigned" candidates — that filter was artificially shrinking the pool).
    keys     = ["avg_ttft", "carbon_emissions", "water_usage", "energy_cost"]
    non_dom  = [c for c in candidate_solutions if not _is_dominated(c, candidate_solutions)]
    pareto_pool = non_dom if len(non_dom) >= 3 else candidate_solutions

    if pareto_pool:
        obj_vecs = np.array([[c["metrics"].get(k, 0.0) for k in keys] for c in pareto_pool], dtype=np.float32)
        obj_rng  = np.where((obj_vecs.max(0) - obj_vecs.min(0)) > 0,
                             obj_vecs.max(0) - obj_vecs.min(0), 1.0)
        obj_norm = (obj_vecs - obj_vecs.min(0)) / obj_rng

        # Seed farthest-point sampling with the four objective-axis extremes so
        # the Pareto_Sample set always spans the full front.
        seed_idxs = set()
        for obj_col in range(obj_norm.shape[1]):
            seed_idxs.add(int(np.argmin(obj_vecs[:, obj_col])))   # best (min) per objective
        seed_pts = obj_norm[list(seed_idxs)]

        diverse = _farthest_point_sample(obj_norm, k=min(6, len(pareto_pool)),
                                          seed_points=seed_pts)
        reported = set()
        for nv in diverse:
            match = int(np.argmin(np.linalg.norm(obj_norm - nv, axis=1)))
            if match not in reported:
                _PARETO_TRACKER.record_solution(
                    pareto_pool[match]["metrics"], pareto_pool[match]["w_total"],
                    pareto_pool[match]["power_plan"], "Pareto_Sample"
                )
                reported.add(match)

    print(_PARETO_TRACKER.get_report())

    if best_balanced_metrics is None and candidate_solutions:
        balanced_pref = np.full(PREF_DIM, 0.25, dtype=np.float32)
        fallback = max(candidate_solutions,
                       key=lambda c: _score_solution(
                           c["metrics"], c["power_sliders"], c["dc_usage"],
                           balanced_pref, {}, dc_to_idx
                       ))
        best_balanced_metrics = fallback["metrics"]
        best_balanced_results = fallback["results"]

    if best_balanced_metrics is None:
        best_balanced_metrics = {k: 0.0 for k in ["avg_ttft", "carbon_emissions", "water_usage",
                                                    "energy_cost", "total_energy",
                                                    "requests_completed", "requests_dropped"]}
        best_balanced_results = []

    return best_balanced_metrics, best_balanced_results, []