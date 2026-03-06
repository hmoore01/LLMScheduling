import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import random
import math
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import Rate_Flow_Sim
import threading

# ─────────────────────────────────────────────────────────────────────────────
# METRIC NORMALIZER  — one instance per agent, fully isolated
# Prevents cross-contamination of EMA baselines between agents.
# ─────────────────────────────────────────────────────────────────────────────
class MetricNormalizer:
    EMA_ALPHA_METRIC = 0.05
    EMA_ALPHA_SLA    = 0.15
    FLOOR            = 1e-6

    def __init__(self):
        self.ttft = self.carbon = self.water = self.cost = None
        self.ratio_ema = [1.0, 1.0, 1.0, 1.0]
        self.ratio_sq_ema = [1.0, 1.0, 1.0, 1.0]
        self.sla_target = 0.80
        self.n_obs      = 0
        self.lock       = threading.Lock()  # Protects global state from concurrent threads

    def update(self, metrics: dict):
        with self.lock:
            ttft   = float(metrics.get("avg_ttft",         0.0))
            carbon = float(metrics.get("carbon_emissions", 0.0)) / 1000.0
            water  = float(metrics.get("water_usage",      0.0)) / 100.0
            cost   = float(metrics.get("energy_cost",      0.0))

            req_done = float(metrics.get("requests_completed",
                                         metrics.get("served_requests", 0.0)))
            req_drop = float(metrics.get("requests_dropped", 0.0))
            req_tot  = max(0.0, req_done + req_drop)
            if req_tot > 0.0:
                sr = req_done / req_tot
                self.sla_target = max(
                    0.70, (1 - self.EMA_ALPHA_SLA) * self.sla_target + self.EMA_ALPHA_SLA * sr)

            vals = (max(abs(ttft), self.FLOOR), max(abs(carbon), self.FLOOR),
                    max(abs(water), self.FLOOR), max(abs(cost),  self.FLOOR))
            if self.n_obs == 0:
                self.ttft, self.carbon, self.water, self.cost = vals
            else:
                a = self.EMA_ALPHA_METRIC
                self.ttft   = (1 - a) * self.ttft   + a * vals[0]
                self.carbon = (1 - a) * self.carbon + a * vals[1]
                self.water  = (1 - a) * self.water  + a * vals[2]
                self.cost   = (1 - a) * self.cost   + a * vals[3]

            if self.n_obs >= 1:
                denoms = self._denominators_unsafe()
                ratios = [v / d for v, d in zip(vals, denoms)]
                a2 = min(0.15, self.EMA_ALPHA_METRIC * 3)
                for i in range(4):
                    self.ratio_ema[i]    = (1 - a2) * self.ratio_ema[i]    + a2 * ratios[i]
                    self.ratio_sq_ema[i] = (1 - a2) * self.ratio_sq_ema[i] + a2 * ratios[i] ** 2

            self.n_obs += 1

    def _denominators_unsafe(self):
        """Internal helper strictly for when the lock is already acquired."""
        if self.n_obs == 0:
            return 100.0, 100.0, 100.0, 100.0
        # Enforcing realistic minimum floors to prevent punishment for optimal performance
        return (max(self.ttft, 0.5), max(self.carbon, 5.0),
                max(self.water, 5.0), max(self.cost, 0.5))

    def denominators(self):
        with self.lock:
            return self._denominators_unsafe()

    def ratio_stds(self):
        with self.lock:
            stds = []
            for i in range(4):
                var = max(0.0, self.ratio_sq_ema[i] - self.ratio_ema[i] ** 2)
                stds.append(max(math.sqrt(var), 0.01))
            return stds

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
BATCH_SIZE            = 64
OPTIM_STEPS           = 15
LR_ACTOR              = 0.0003
LR_CRITIC             = 0.001
LR_ALPHA              = 0.0003          # SAC entropy temperature learning rate
GAMMA                 = 0.95
TAU                   = 0.005
MEMORY_SIZE           = 20_000          # Intra-epoch PER buffer capacity
MEMORY_CROSS_SIZE     = 5_000           # Cross-epoch PER buffer capacity (smaller, higher value)
NUM_NODE_TYPES        = 6
NUM_MODEL_CLASSES     = 2
REQUESTS_PER_NODE_CAP = 5_000
MODEL_VARIANT         = "_FP16 (Base)_B16"
LOG_STD_MIN           = -5
LOG_STD_MAX           = 2
PARLIAMENT_GRAD_STEPS = 5               # Gradient ascent steps to refine consensus
PARLIAMENT_GRAD_LR    = 0.05
CAPITAL_DECAY         = 0.9             # EMA decay for persistent political capital
VETO_CAPITAL_THRESH   = 150.0           # Minimum capital to exercise veto power
VETO_Q_DEGRADATION    = 0.25            # Q-value degradation ratio that triggers veto
VETO_STRENGTH_CAP     = 0.5             # Max fraction veto can pull consensus
METRIC_REWARD_SCALE   = 8.0             # Amplified metric penalty in reward (was 4.0)
ECO_BONUS_SCALE       = 0.05            # Dampen shared eco bonus to avoid homogeneity
HER_CROSS_PRIORITY    = 0.4             # Priority discount for cross-agent HER samples

# ─────────────────────────────────────────────────────────────────────────────
# METRIC INDEX MAPPING  (weight vector position → simulation output key)
# Weight vectors are always ordered: [Time, Carbon, Water, Cost]
# ─────────────────────────────────────────────────────────────────────────────
METRIC_KEYS   = ["avg_ttft", "carbon_emissions", "water_usage", "energy_cost"]
METRIC_LABELS = ["TTFT(s)", "Carbon(kg)", "Water(L)", "Cost($)"]
METRIC_SCALE  = [1.0, 1/1000.0, 1/100.0, 1.0]  # Raw sim value → display unit

# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT SCHEME LIST
# Edit this list to change what runs when no configure_schemes() call is made.
# Each entry is (name, [Time_weight, Carbon_weight, Water_weight, Cost_weight]).
# Weights are auto-normalised — raw magnitudes are fine.
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_SCHEMES = [
    ("MinCost",    [0.00, 0.00, 0.00, 1.00]),   # Pure cost minimiser
    ("Balanced",   [0.25, 0.25, 0.25, 0.25]),   # Equal weight across all metrics
    # ── Add more schemes below ───────────────────────────────────────────
    ("MinLatency", [1.00, 0.00, 0.00, 0.00]),
    ("MinCarbon", [0.00, 1.00, 0.00, 0.00]),
    ("MinWater", [0.00, 0.00, 1.00, 0.00]),
    # ("GreenFirst", [0.05, 0.50, 0.35, 0.10]),
    # ("WaterSaver", [0.00, 0.10, 0.80, 0.10]),
    # ("SpeedBudget",[0.60, 0.00, 0.00, 0.40]),
]

# ─────────────────────────────────────────────────────────────────────────────
# ACTIVE SCHEME STATE  (populated from DEFAULT_SCHEMES at import, or by
# calling configure_schemes() with a custom list at any time)
# ─────────────────────────────────────────────────────────────────────────────
SCHEMES          = []
SCHEME_WEIGHTS   = {}
PRIMARY_METRIC_KEY = {}

# ─────────────────────────────────────────────────────────────────────────────
# GLOBALS
# ─────────────────────────────────────────────────────────────────────────────
_GLOBAL_AGENTS     = {}
_PREV_STATES       = {}                          # Cross-epoch state memory per scheme
_POLITICAL_CAPITAL = {}                          # Persistent across epochs
_EPOCH_HISTORY     = []                          # List of {epoch, metrics} dicts
_GLOBAL_NORMALIZER = MetricNormalizer()


def configure_schemes(scheme_list: list = None):
    """
    Configure the agent roster from a list of (name, weights) tuples.
    Weights are always ordered [Time, Carbon, Water, Cost] and will be normalised
    to sum to 1.0.  Agents, buffers, and capital are all reset.

    If scheme_list is None or omitted, uses DEFAULT_SCHEMES defined at the top
    of the file — edit that list to change the default roster.

    Examples
    --------
    >>> configure_schemes([
    ...     ("GreenFirst",  [0.05, 0.50, 0.35, 0.10]),
    ...     ("BudgetHawk",  [0.10, 0.05, 0.05, 0.80]),
    ...     ("SpeedKing",   [0.90, 0.03, 0.03, 0.04]),
    ...     ("Balanced",    [0.25, 0.25, 0.25, 0.25]),
    ... ])
    >>> configure_schemes()            # reloads DEFAULT_SCHEMES
    """
    global SCHEMES, SCHEME_WEIGHTS, PRIMARY_METRIC_KEY
    global _GLOBAL_AGENTS, _PREV_STATES, _POLITICAL_CAPITAL, _EPOCH_HISTORY
    global _GLOBAL_NORMALIZER

    if scheme_list is None:
        scheme_list = DEFAULT_SCHEMES

    if not scheme_list:
        raise ValueError("scheme_list must contain at least one (name, weights) tuple")

    SCHEMES        = []
    SCHEME_WEIGHTS = {}
    PRIMARY_METRIC_KEY = {}

    seen_names = set()
    for name, weights in scheme_list:
        if not isinstance(weights, (list, tuple, np.ndarray)) or len(weights) != 4:
            raise ValueError(f"Scheme '{name}' needs exactly 4 weights "
                             f"[Time, Carbon, Water, Cost], got {weights}")
        w = np.array(weights, dtype=np.float64)
        if w.sum() <= 0:
            raise ValueError(f"Scheme '{name}' weights must sum to > 0")
        w = w / w.sum()

        base_name = name
        counter = 2
        while name in seen_names:
            name = f"{base_name}_{counter}"
            counter += 1
        seen_names.add(name)

        SCHEMES.append(name)
        SCHEME_WEIGHTS[name] = w.tolist()
        PRIMARY_METRIC_KEY[name] = METRIC_KEYS[int(np.argmax(w))]

    _GLOBAL_AGENTS.clear()
    _PREV_STATES.clear()
    _POLITICAL_CAPITAL = {s: 100.0 for s in SCHEMES}
    _EPOCH_HISTORY     = []
    _GLOBAL_NORMALIZER = MetricNormalizer()

    nw = max(12, max(len(s) for s in SCHEMES))
    print(f"[CONFIG] {len(SCHEMES)} scheme(s) registered:")
    for s in SCHEMES:
        w = SCHEME_WEIGHTS[s]
        wstr = "  ".join(f"{ml.split('(')[0]}={v:.2f}" for ml, v in zip(METRIC_LABELS, w))
        print(f"  {s:>{nw}}  {wstr}")


def reset_simulation():
    """Clear all agent state and history for a clean restart with current schemes."""
    global _GLOBAL_AGENTS, _PREV_STATES, _POLITICAL_CAPITAL, _EPOCH_HISTORY
    _GLOBAL_AGENTS.clear()
    _PREV_STATES.clear()
    _POLITICAL_CAPITAL = {s: 100.0 for s in SCHEMES}
    _EPOCH_HISTORY     = []
    _GLOBAL_NORMALIZER = MetricNormalizer()
    print("[RESET] All agents, buffers, capital, and history cleared.")


def _print_epoch_table(epoch_idx: int, all_metrics: dict):
    """
    Print a compact per-epoch comparison table.  ★ marks the best (lowest)
    value in each metric column across schemes.  Parliament is shown last
    with a ◀ marker but excluded from ★ competition.
    """
    all_keys = SCHEMES + ["Parliament"]
    nw = max(12, max((len(s) for s in all_keys), default=12))

    # Gather display values: (ttft, carbon_kg, water_L, cost, served)
    rows = {}
    for s in all_keys:
        if s not in all_metrics:
            continue
        m = all_metrics[s]
        rows[s] = [
            float(m.get("avg_ttft", 0)),
            float(m.get("carbon_emissions", 0)) / 1000.0,
            float(m.get("water_usage", 0)) / 100.0,
            float(m.get("energy_cost", 0)),
            int(m.get("requests_completed", 0)),
        ]

    # Find best (lowest) per metric column among scheme agents only
    best_idx = [None, None, None, None]
    for ci in range(4):
        best_v = float("inf")
        for s in SCHEMES:
            if s in rows and rows[s][ci] < best_v:
                best_v = rows[s][ci]
                best_idx[ci] = s

    # Adaptive formatting based on value magnitudes
    def _fmt(v, ci):
        if   ci == 0:  return f"{v:>9.3f}"     # TTFT always small
        elif v >= 1000: return f"{v:>10.0f}"
        elif v >= 100:  return f"{v:>10.1f}"
        else:           return f"{v:>10.3f}"

    hdr = (f"  {'Scheme':<{nw}} {'TTFT(s)':>10} {'Carbon(kg)':>11} "
           f"{'Water(L)':>11} {'Cost($)':>11} {'Served':>8} {'Cap':>5}")
    bar = "─" * len(hdr)
    print(f"┌ EPOCH {epoch_idx} {bar[len(f'  EPOCH {epoch_idx} ') + 1:]}┐")
    print(f"│{hdr[1:]}│")
    print(f"│{bar[1:]}│")

    for s in all_keys:
        if s not in rows:
            continue
        v = rows[s]
        stars = ["★" if best_idx[ci] == s else " " for ci in range(4)]
        cap_str = f"{_POLITICAL_CAPITAL.get(s, 0):>5.0f}" if s != "Parliament" else "  ---"
        tag = " ◀" if s == "Parliament" else "  "
        cols = "".join(f"{_fmt(v[ci], ci)}{stars[ci]}" for ci in range(4))
        print(f"│ {s:<{nw}} {cols} {v[4]:>8}  {cap_str}{tag}│")

    print(f"└{bar[1:]}┘")


def print_run_summary():
    """
    Print a formatted summary table across all recorded epochs.
    TTFT is averaged (it's a latency); Carbon, Water, Cost are summed (cumulative).
    Each scheme + Parliament gets its own row.  ★ marks the best scheme per column.
    Zero-traffic epochs (where all schemes served 0 requests) are excluded.
    """
    if not _EPOCH_HISTORY:
        print("[SUMMARY] No epochs recorded yet.")
        return

    all_keys = SCHEMES + ["Parliament"]
    accum = {s: {k: [] for k in METRIC_KEYS + ["requests_completed"]}
             for s in all_keys}

    skipped = 0
    for record in _EPOCH_HISTORY:
        em = record["metrics"]
        # Check if ANY scheme served requests this epoch
        any_served = any(
            float(em.get(s, {}).get("requests_completed",
                  em.get(s, {}).get("served_requests", 0.0))) > 0
            for s in all_keys if s in em
        )
        if not any_served:
            skipped += 1
            continue
        for s in all_keys:
            if s in em:
                m = em[s]
                for k in METRIC_KEYS + ["requests_completed"]:
                    accum[s][k].append(float(m.get(k, 0.0)))

    n_total  = len(_EPOCH_HISTORY)
    n_active = n_total - skipped
    nw = max(14, max(len(s) for s in all_keys) + 2)

    if n_active == 0:
        print("[SUMMARY] All epochs had zero traffic — nothing to report.")
        return

    # Compute display values per scheme
    display = {}   # s → [ttft_avg, carbon_sum, water_sum, cost_sum, served_sum, n]
    for s in all_keys:
        vals = accum[s]
        if not vals[METRIC_KEYS[0]]:
            continue
        n = len(vals[METRIC_KEYS[0]])
        display[s] = [
            float(np.mean(vals["avg_ttft"])),
            float(np.sum(vals["carbon_emissions"])) * METRIC_SCALE[1],
            float(np.sum(vals["water_usage"]))       * METRIC_SCALE[2],
            float(np.sum(vals["energy_cost"]))       * METRIC_SCALE[3],
            int(np.sum(vals["requests_completed"])),
            n,
        ]

    # Best per metric column (schemes only, not Parliament)
    best_idx = [None, None, None, None]
    for ci in range(4):
        best_v = float("inf")
        for s in SCHEMES:
            if s in display and display[s][ci] < best_v:
                best_v = display[s][ci]
                best_idx[ci] = s

    # Adaptive column formatting
    def _sfmt(v, ci):
        if   ci == 0:  return f"{v:>11.3f}"
        elif v >= 10000: return f"{v:>14.0f}"
        elif v >= 100:   return f"{v:>14.1f}"
        else:            return f"{v:>14.3f}"

    hdr = (f"  {'Scheme':<{nw}} {'avgTTFT(s)':>12} {'sumCarbon(kg)':>15} "
           f"{'sumWater(L)':>15} {'sumCost($)':>15} {'Served':>10} {'Epochs':>7}")
    sep = "═" * len(hdr)
    thin = "─" * len(hdr)

    skip_note = f", {skipped} zero-traffic skipped" if skipped else ""
    print(f"\n{sep}")
    print(f"  RUN SUMMARY  ({n_active} active epoch{'s' if n_active != 1 else ''}"
          f" of {n_total}{skip_note})")
    print(sep)

    # Show what each scheme is optimising
    print("  Scheme weights:  [TTFT  Carbon  Water  Cost]")
    for s in SCHEMES:
        w = SCHEME_WEIGHTS[s]
        primary = METRIC_LABELS[METRIC_KEYS.index(PRIMARY_METRIC_KEY[s])]
        wstr = "  ".join(f"{v:.2f}" for v in w)
        print(f"    {s:>{nw}}  [{wstr}]  → {primary}")
    print(thin)

    # Main results table
    print(hdr)
    print(thin)
    for s in all_keys:
        if s not in display:
            continue
        d = display[s]
        stars = ["★" if best_idx[ci] == s else " " for ci in range(4)]
        tag = " ◀" if s == "Parliament" else ""
        cols = "".join(f"{_sfmt(d[ci], ci)}{stars[ci]}" for ci in range(4))
        print(f"  {s:<{nw}} {cols} {d[4]:>10}{d[5]:>7}{tag}")

    print(sep)

    # Per-metric best
    print("\n  Best per metric (schemes only):")
    labels_short = ["Avg TTFT", "Total Carbon", "Total Water", "Total Cost"]
    for ci in range(4):
        if best_idx[ci] and best_idx[ci] in display:
            print(f"    {labels_short[ci]:<14} → {best_idx[ci]} "
                  f"({display[best_idx[ci]][ci]:.3f})")

    print(sep + "\n")


# ── Auto-initialise from DEFAULT_SCHEMES on first import ──────────────────
configure_schemes()


# ─────────────────────────────────────────────────────────────────────────────
# PRIORITIZED REPLAY BUFFER  (Sum-Tree implementation)
# Transitions with higher TD error are sampled more frequently.
# Importance-sampling weights correct for the sampling bias.
# ─────────────────────────────────────────────────────────────────────────────
class SumTree:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree     = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data     = np.empty(capacity, dtype=object)
        self.size     = 0
        self.ptr      = 0

    def update(self, idx: int, priority: float):
        delta = priority - self.tree[idx]
        self.tree[idx] = priority
        while idx != 0:          # Iterative propagation — no recursion limit
            idx = (idx - 1) // 2
            self.tree[idx] += delta

    def add(self, priority: float, data):
        idx = self.ptr + self.capacity - 1
        self.data[self.ptr] = data
        self.update(idx, priority)
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get(self, s: float):
        idx = 0
        while True:
            left, right = 2 * idx + 1, 2 * idx + 2
            if left >= len(self.tree):
                break
            if s <= self.tree[left]:
                idx = left
            else:
                s  -= self.tree[left]
                idx = right
        return idx, self.tree[idx], self.data[idx - (self.capacity - 1)]

    @property
    def total(self) -> float:
        return float(self.tree[0])

    @property
    def max_priority(self) -> float:
        if self.size == 0:
            return 1.0
        return float(self.tree[self.capacity - 1: self.capacity - 1 + self.size].max())


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6,
                 beta_start: float = 0.4, beta_steps: int = 50_000):
        self.tree       = SumTree(capacity)
        self.capacity   = capacity
        self.alpha      = alpha             # Priority exponent
        self.beta       = beta_start        # IS weight exponent (annealed to 1)
        self.beta_delta = (1.0 - beta_start) / beta_steps
        self.eps        = 1e-6

    def push(self, state, action, reward, next_state, done,
             priority_boost: float = 1.0):
        """priority_boost > 1 artificially elevates a transition (e.g. cross-epoch)."""
        self.tree.add(self.tree.max_priority * priority_boost,
                      (state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        if self.tree.size < batch_size:
            return None
        self.beta = min(1.0, self.beta + self.beta_delta)
        seg       = self.tree.total / batch_size
        batch, idxs, prios = [], [], []
        for i in range(batch_size):
            s = random.uniform(seg * i, seg * (i + 1))
            idx, p, data = self.tree.get(s)
            if data is None:
                return None
            batch.append(data)
            idxs.append(idx)
            prios.append(max(float(p), self.eps))

        probs   = np.array(prios) / (self.tree.total + self.eps)
        weights = (self.tree.size * probs) ** (-self.beta)
        weights /= weights.max()

        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return (torch.FloatTensor(state),  torch.FloatTensor(action),
                torch.FloatTensor(reward), torch.FloatTensor(next_state),
                torch.FloatTensor(done),   torch.FloatTensor(weights), idxs)

    def update_priorities(self, idxs, td_errors):
        for idx, td in zip(idxs, td_errors):
            self.tree.update(idx, (abs(float(td)) + self.eps) ** self.alpha)

    def __len__(self) -> int:
        return self.tree.size

# ─────────────────────────────────────────────────────────────────────────────
# SAC ACTOR  (Gaussian policy with reparameterization)
# Replaces hand-crafted Dirichlet/Gaussian/dropout noise with a single
# learnable entropy temperature α that auto-tunes exploration per agent.
# ─────────────────────────────────────────────────────────────────────────────
class SACActorNetwork(nn.Module):
    def __init__(self, num_dcs: int, dc_state_dim: int, hidden_dim: int = 128):
        """
        dc_state_dim = num_dcs * state_feat_per_dc  (raw DC features, no appended weights)
        Scheme weights enter via a dedicated FiLM branch that modulates hidden activations.
        This guarantees distinct behavior per agent from epoch 0, independent of training.
        """
        super().__init__()
        self.num_dcs    = num_dcs
        self.action_dim = num_dcs * NUM_MODEL_CLASSES + num_dcs

        # DC feature trunk
        self.dc_net = nn.Sequential(
            nn.Linear(dc_state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),   nn.ReLU(),
        )
        # FiLM: scheme weights produce scale + shift for each hidden unit.
        # Initialized with Xavier (not zeros!) so the 4-dim weight identity
        # immediately produces distinct modulation per agent from epoch 0.
        # This is critical for early differentiation before training kicks in.
        self.film = nn.Linear(4, hidden_dim * 2)
        nn.init.xavier_uniform_(self.film.weight, gain=0.5)
        nn.init.zeros_(self.film.bias)

        self.mean_head    = nn.Linear(hidden_dim, self.action_dim)
        self.log_std_head = nn.Linear(hidden_dim, self.action_dim)

    def _film_modulate(self, h: torch.Tensor, scheme_w: torch.Tensor) -> torch.Tensor:
        film_out     = self.film(scheme_w)
        scale, shift = film_out.chunk(2, dim=-1)
        return h * (1.0 + scale) + shift   # FiLM: h ← h*(1+γ) + β

    def _project(self, raw: torch.Tensor) -> torch.Tensor:
        parts = [F.softmax(raw[:, k * self.num_dcs:(k + 1) * self.num_dcs], dim=1)
                 for k in range(NUM_MODEL_CLASSES)]
        parts.append(torch.sigmoid(raw[:, NUM_MODEL_CLASSES * self.num_dcs:]))
        return torch.cat(parts, dim=1)

    def _split_state(self, state: torch.Tensor):
        """Split augmented state into DC features and scheme weight identity."""
        return state[:, :-4], state[:, -4:]

    def sample(self, state: torch.Tensor):
        dc_feat, scheme_w = self._split_state(state)
        h       = self._film_modulate(self.dc_net(dc_feat), scheme_w)
        mean    = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std     = log_std.exp()
        raw     = mean + std * torch.randn_like(mean)

        dist        = torch.distributions.Normal(mean, std)
        lp_raw      = dist.log_prob(raw)
        pwr         = NUM_MODEL_CLASSES * self.num_dcs
        routing_lp  = lp_raw[:, :pwr].sum(dim=1, keepdim=True)
        a_power     = torch.sigmoid(raw[:, pwr:])
        power_lp    = (lp_raw[:, pwr:] -
                       torch.log(a_power * (1.0 - a_power) + 1e-8)).sum(dim=1, keepdim=True)
        return self._project(raw), routing_lp + power_lp

    def deterministic_action(self, state: torch.Tensor) -> torch.Tensor:
        dc_feat, scheme_w = self._split_state(state)
        h = self._film_modulate(self.dc_net(dc_feat), scheme_w)
        return self._project(self.mean_head(h))


# ─────────────────────────────────────────────────────────────────────────────
# SAC DUAL CRITIC  (clipped double-Q prevents overestimation)
# ─────────────────────────────────────────────────────────────────────────────
class SACCriticNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        def _q():
            return nn.Sequential(
                nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),             nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
        self.q1, self.q2 = _q(), _q()

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa), self.q2(sa)

    def q_min(self, state, action):
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)


# ─────────────────────────────────────────────────────────────────────────────
# SAC AGENT
# ─────────────────────────────────────────────────────────────────────────────
class SACAgent:
    def __init__(self, num_dcs: int, state_feat_per_dc: int, weights):
        self.num_dcs    = num_dcs
        self.weights    = np.asarray(weights, dtype=np.float32)
        # state_dim includes raw DC features + appended 4-dim scheme weight identity
        self.dc_state_dim = num_dcs * state_feat_per_dc
        self.state_dim  = self.dc_state_dim + 4
        self.action_dim = num_dcs * NUM_MODEL_CLASSES + num_dcs

        self.actor         = SACActorNetwork(num_dcs, self.dc_state_dim)
        self.critic        = SACCriticNetwork(self.state_dim, self.action_dim)
        self.critic_target = SACCriticNetwork(self.state_dim, self.action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        # Learnable entropy temperature — auto-tunes exploration vs exploitation
        self.target_entropy = float(-self.action_dim * 0.5)
        self.log_alpha      = torch.zeros(1, requires_grad=True)
        self.alpha          = float(self.log_alpha.exp())

        self.actor_optimizer  = optim.Adam(self.actor.parameters(),  lr=LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)
        self.alpha_optimizer  = optim.Adam([self.log_alpha],         lr=LR_ALPHA)

        # Per-agent normalizer: rewards are scaled against this agent's own history only
        # self.normalizer = MetricNormalizer()

        # Separate buffers: intra-epoch (same-state transitions) and
        # cross-epoch (true temporal transitions, sampled 70% of the time)
        self.intra_buffer = PrioritizedReplayBuffer(MEMORY_SIZE)
        self.cross_buffer = PrioritizedReplayBuffer(MEMORY_CROSS_SIZE)

        self.epoch_count = 0

    def _heuristic_action(self, state: np.ndarray) -> np.ndarray:
        """
        Compute a greedy DC preference action directly from observable state features.
        State shape: (num_dcs, 4) = [carbon_norm, cost_norm, water_norm, req_intensity]

        CRITICAL: includes a scheme-deterministic DC bias that breaks symmetry when
        datacenters have identical features.  Without this, homogeneous DCs → uniform
        softmax → identical routing for every scheme → no differentiation.

        The bias is derived from a hash of the weight vector, so each scheme gets a
        unique, reproducible preferred DC ordering.  When DCs ARE heterogeneous the
        real feature scores dominate; the bias only matters as a tiebreaker.
        """
        dc_state = np.asarray(state, dtype=np.float32).reshape(self.num_dcs, -1)[:, :4]
        w = self.weights  # [w_ttft, w_carbon, w_water, w_cost]
        n = self.num_dcs

        # ── Feature-based scoring (dominates when DCs differ) ────────────
        dc_score = -(w[1] * dc_state[:, 0] +  # carbon intensity
                     w[3] * dc_state[:, 1] +  # electricity price
                     w[2] * dc_state[:, 2])    # PUE / water

        load_col = dc_state[:, 3]
        resource_tiebreak = -(dc_state[:, 0] + dc_state[:, 1] + dc_state[:, 2]) / 3.0
        dc_score += w[0] * ((1.0 - load_col) + resource_tiebreak * 0.5)

        # ── Scheme-deterministic symmetry breaker ────────────────────────
        # When all DCs have identical features, dc_score is uniform and the
        # softmax yields 1/N for every DC.  This bias gives each scheme a
        # unique preferred DC ordering derived from its weight identity.
        # The primary-metric index selects a "home DC", and concentration
        # controls how strongly the agent is pulled toward it.
        primary_idx = int(np.argmax(w))
        concentration = float(np.max(w))

        # Each primary metric gets a deterministic DC ordering.
        # With 4 metrics and N DCs, metric i prefers DC (i % N) most.
        # The bias magnitude is proportional to concentration and to the
        # feature range, ensuring it only matters when features are tied.
        feature_range = float(dc_score.max() - dc_score.min())
        # When features are identical, feature_range ≈ 0.  In that case the
        # bias needs to be absolute (not relative) to have any effect.
        bias_scale = max(feature_range * 0.5, 0.3 * concentration)

        # Build a per-DC bias: home DC gets +bias_scale, others decay linearly
        home_dc = primary_idx % n
        # Create a deterministic permutation seeded from weights
        rng = np.random.RandomState(
            int(abs(hash(tuple(np.round(w, 4).tolist())))) % (2**31))
        perm = rng.permutation(n)
        # The home DC always gets rank 0 (strongest bias)
        rank = np.zeros(n)
        rank[home_dc] = 0
        other_idx = [j for j in perm if j != home_dc]
        for r, j in enumerate(other_idx, start=1):
            rank[j] = r

        dc_bias = bias_scale * (1.0 - rank / max(n - 1, 1))
        dc_score += dc_bias

        # ── Temperature & softmax ────────────────────────────────────────
        temp_scale = 1.0 - w[0] * 0.95
        temperature = (8.0 + 22.0 * concentration) * temp_scale
        exp_scores = np.exp((dc_score - dc_score.max()) * temperature)
        routing_pref = exp_scores / (exp_scores.sum() + 1e-8)

        # ── Power follows routing ────────────────────────────────────────
        pref_rank = routing_pref / (routing_pref.max() + 1e-8)
        latency_power_floor = w[0]
        power_pref = np.clip((pref_rank ** (1.0 + concentration)) + latency_power_floor, 0.0, 1.0)

        return np.concatenate([routing_pref, routing_pref, power_pref])

    def _blend_with_heuristic(self, sac_action: np.ndarray,
                              state: np.ndarray) -> np.ndarray:
        """
        Blend SAC action with heuristic, decaying blend over epochs.
        Ensures distinct behavior from epoch 0 while gracefully handing off to learned policy.
        Schedule: epoch 0 → 60% heuristic, epoch ~10 → 21%, epoch ~20 → 7%
        """
        blend = max(0.0, 0.6 * (0.90 ** self.epoch_count))
        heur = self._heuristic_action(state)
        blended = (1.0 - blend) * sac_action + blend * heur

        # Re-normalise routing segments after blend
        n = self.num_dcs
        for k in range(NUM_MODEL_CLASSES):
            seg = blended[k * n:(k + 1) * n]
            s = seg.sum()
            blended[k * n:(k + 1) * n] = seg / s if s > 0 else np.ones(n) / n
        blended[NUM_MODEL_CLASSES * n:] = np.clip(blended[NUM_MODEL_CLASSES * n:], 0.0, 1.0)
        return blended

    def _augment(self, state: np.ndarray) -> np.ndarray:
        """Append scheme weights to state so the policy is conditioned on its identity."""
        return np.concatenate([np.asarray(state, dtype=np.float32).flatten(), self.weights])

    def select_action(self, state, exploration: bool = True) -> np.ndarray:
        s_t = torch.FloatTensor(self._augment(state)).unsqueeze(0)
        with torch.no_grad():
            if exploration:
                action, _ = self.actor.sample(s_t)
            else:
                action = self.actor.deterministic_action(s_t)
        raw = action.cpu().numpy()[0]
        # Always blend with heuristic so that training data (exploration) is
        # scheme-differentiated from epoch 0 — not just the final proposal.
        return self._blend_with_heuristic(raw, state)

    def _combined_batch(self, batch_size: int):
        """
        Sample 70% from cross_buffer (temporal dynamics) and 30% from intra_buffer.
        Falls back gracefully when one buffer is undersized.
        """
        have_c = len(self.cross_buffer) >= 8
        have_i = len(self.intra_buffer) >= 8
        if not have_c and not have_i:
            return None

        if have_c and have_i:
            n_c, n_i = int(batch_size * 0.7), batch_size - int(batch_size * 0.7)
        elif have_c:
            n_c, n_i = min(batch_size, len(self.cross_buffer)), 0
        else:
            n_c, n_i = 0, min(batch_size, len(self.intra_buffer))

        parts, all_idxs, all_bufs = [], [], []
        if n_c > 0:
            r = self.cross_buffer.sample(n_c)
            if r is not None:
                parts.append(r)
                all_idxs += r[6]
                all_bufs += [self.cross_buffer] * n_c
        if n_i > 0:
            r = self.intra_buffer.sample(n_i)
            if r is not None:
                parts.append(r)
                all_idxs += r[6]
                all_bufs += [self.intra_buffer] * n_i
        if not parts:
            return None

        if len(parts) == 2:
            s  = torch.cat([parts[0][0], parts[1][0]])
            a  = torch.cat([parts[0][1], parts[1][1]])
            rw = torch.cat([parts[0][2], parts[1][2]])
            ns = torch.cat([parts[0][3], parts[1][3]])
            d  = torch.cat([parts[0][4], parts[1][4]])
            iw = torch.cat([parts[0][5], parts[1][5]])
            iw = iw / iw.max()    # Re-normalise combined IS weights
        else:
            s, a, rw, ns, d, iw, _ = parts[0]

        return s, a, rw, ns, d, iw, all_idxs, all_bufs

    def train(self, batch_size: int = BATCH_SIZE):
        res = self._combined_batch(batch_size)
        if res is None:
            return None
        state, action, reward, next_state, done, is_w, idxs, bufs = res

        bs    = state.size(0)
        w_b   = torch.FloatTensor(self.weights).unsqueeze(0).expand(bs, -1)
        s_aug = torch.cat([state.view(bs, -1),      w_b], dim=1)
        n_aug = torch.cat([next_state.view(bs, -1), w_b], dim=1)
        rw    = reward.unsqueeze(1)
        dn    = done.unsqueeze(1)
        is_w  = is_w.unsqueeze(1)

        # ── Critic update (PER-weighted MSE) ─────────────────────────────
        with torch.no_grad():
            na, nlp  = self.actor.sample(n_aug)
            q_next   = self.critic_target.q_min(n_aug, na)
            tgt_q    = rw + (1 - dn) * GAMMA * (q_next - self.alpha * nlp)

        q1, q2    = self.critic(s_aug, action)
        td_errors = (q1 - tgt_q).abs().detach().squeeze(-1).cpu().numpy()

        c_loss = (is_w * (F.mse_loss(q1, tgt_q, reduction='none') +
                          F.mse_loss(q2, tgt_q, reduction='none'))).mean()
        self.critic_optimizer.zero_grad()
        c_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # ── Actor update ─────────────────────────────────────────────────
        na2, lp2 = self.actor.sample(s_aug)
        a_loss   = (self.alpha * lp2 - self.critic.q_min(s_aug, na2)).mean()
        self.actor_optimizer.zero_grad()
        a_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # ── Entropy temperature update (auto-tune α) ──────────────────────
        al_loss = -(self.log_alpha * (lp2.detach() + self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad()
        al_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = float(self.log_alpha.exp())

        # ── Polyak update critic target ───────────────────────────────────
        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.mul_(1 - TAU).add_(TAU * p.data)

        # ── Update PER priorities ─────────────────────────────────────────
        td_arr = np.atleast_1d(td_errors)
        for i, (idx, buf) in enumerate(zip(idxs, bufs)):
            if i < len(td_arr):
                buf.update_priorities([idx], [td_arr[i]])

        return float(c_loss)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _stable_hash_int(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest()[:16], 16)


def _normalize_weights(w: np.ndarray) -> np.ndarray:
    out = np.maximum(0.0, np.asarray(w, dtype=np.float64))
    s   = float(out.sum())
    if s <= 0.0:
        out[:] = 0.0; out[0] = 1.0
        return out
    return out / s


def get_rich_state(sim, dc_ids, epoch_data, epoch_idx: int) -> np.ndarray:
    n_dcs = len(dc_ids)
    state = np.zeros((n_dcs, 4), dtype=np.float32)
    n_reqs = len(epoch_data) if epoch_data is not None else 0

    try:
        epoch_len = float(next(iter(sim.datacenters.values()))._epoch_length_s)
    except Exception:
        epoch_len = 900.0

    epoch_start_sec = epoch_idx * epoch_len
    hour = int((epoch_start_sec % 86400) // 3600)

    for i, dc_id in enumerate(dc_ids):
        ci, cost, true_water_intensity = 400.0, 0.10, 1.0

        if hasattr(sim, 'datacenters') and dc_id in sim.datacenters:
            dc = sim.datacenters[dc_id]
            ci = float(getattr(dc, 'carbon_intensity_g_per_kwh', 400.0))

            try:
                tou = getattr(dc, 'tou_price', None)
                cost = float(tou[hour]) if isinstance(tou, (list, tuple)) and len(tou) == 24 \
                    else float(getattr(dc, 'tou_price', [0.10])[0])
            except Exception:
                pass

            # Extract true water intensity based on the simulator's physical math
            static_factor = float(getattr(dc, 'water_static', 0.0))
            evap_factor = float(getattr(dc, 'water_cycling_density', 0.0))
            blowdown = max(1e-9, float(getattr(dc, 'blowdown_ratio', 0.30)))

            # Total m3 of water drawn per kWh of heat rejected
            true_water_intensity = static_factor + (evap_factor / blowdown)

        # Scale the water intensity by 10.0 to keep it roughly in the [0, 1] range for the neural network
        state[i] = [ci / 1000.0, cost * 5.0, true_water_intensity / 10.0, min(n_reqs / 50_000.0, 1.0)]

    return state


def build_power_plan_sliding(dc_ids, slider_values: np.ndarray) -> dict:
    plan = {}
    for i, dc_id in enumerate(dc_ids):
        n = min(max(int(np.floor(float(slider_values[i]) * (NUM_NODE_TYPES + 0.99))), 0), NUM_NODE_TYPES)
        plan[int(dc_id)] = ({"all": "OFF"} if n == 0 else
                            {"unit": {str(t): "IDLE" if t < n else "OFF"
                                      for t in range(NUM_NODE_TYPES)}})
    return plan


def _active_nodes_per_dc(power_sliders) -> np.ndarray:
    return np.array([
        min(max(int(np.floor(s * (NUM_NODE_TYPES + 0.99))), 0), NUM_NODE_TYPES)
        for s in np.asarray(power_sliders, dtype=np.float64)
    ], dtype=np.int32)


def build_schedule_map(sim_data: pd.DataFrame, dc_ids, w_small, w_large,
                       power_sliders, epoch_idx: int) -> dict:
    if len(sim_data) == 0:
        return {"map": {}}
    models = (sim_data["model"] if "model" in sim_data.columns
              else sim_data["model_type"]).astype(str).str.lower()
    msk   = (models.str.contains("7b") | models.str.contains("8b") |
             models.str.contains("small")).to_numpy()
    rows  = np.arange(len(sim_data), dtype=int)
    s_idx, l_idx = rows[msk].tolist(), rows[~msk].tolist()

    active = _active_nodes_per_dc(power_sliders)
    cap    = active.astype(np.float64) * REQUESTS_PER_NODE_CAP

    def allocate(req_idx, pref_w, bucket):
        if not req_idx:
            return {}
        n   = len(req_idx)
        ew  = np.asarray(pref_w, np.float64) * active.astype(np.float64)
        tot = ew.sum()
        if tot <= 0.0:
            ew  = (active > 0).astype(np.float64)
            tot = ew.sum()
            if tot <= 0.0:
                return {}
        ew /= tot
        raw    = ew * n
        counts = np.floor(raw).astype(np.int64)
        rem    = n - int(counts.sum())
        counts[np.argsort(raw - counts)[::-1][:rem]] += 1
        ov     = np.maximum(0, counts - cap.astype(np.int64))
        counts -= ov
        tov    = int(ov.sum())
        if tov > 0:
            spare = np.maximum(0.0, cap - counts.astype(np.float64))
            spare[active == 0] = 0.0
            ts = spare.sum()
            if ts > 0:
                ex   = np.floor(spare / ts * tov).astype(np.int64)
                lft  = tov - int(ex.sum())
                ex[np.argsort(spare)[::-1][:lft]] += 1
                counts += ex
                counts -= np.maximum(0, counts - cap.astype(np.int64))
        ordered = sorted(req_idx, key=lambda r: _stable_hash_int(f"{epoch_idx}:{bucket}:{r}"))
        alloc, ptr = {}, 0
        for di, cnt in enumerate(counts):
            for _ in range(int(cnt)):
                if ptr < len(ordered):
                    alloc[int(ordered[ptr])] = int(dc_ids[di])
                    ptr += 1
        return alloc

    m = {}
    m.update(allocate(s_idx, w_small, "small"))
    m.update(allocate(l_idx, w_large, "large"))
    return {"map": m}


def _enforce_route_power_coherence(action: np.ndarray, num_dcs: int,
                                   threshold: float = 0.05) -> np.ndarray:
    """
    Zero power for DCs that receive negligible routing weight.

    This is the critical bridge between routing intent and physical configuration.
    Without it, an agent can say 'route 95% to DC3' while keeping all 5 DCs
    powered on — which burns carbon, water, and cost on idle DCs, making every
    scheme's metrics near-identical.

    threshold: DCs receiving less than this fraction of combined routing
               weight get powered off (default 5%).
    """
    action = action.copy()
    n  = num_dcs
    ws = action[:n]
    wl = action[n:2 * n]
    ps = action[2 * n:]

    avg_route = (ws + wl) / 2.0
    route_total = avg_route.sum()
    if route_total > 0:
        route_share = avg_route / route_total
    else:
        route_share = np.ones(n) / n

    # Power off DCs with negligible routing share
    for i in range(n):
        if route_share[i] < threshold:
            ps[i] = 0.0

    # Boost top-routed DC power proportionally: if you route 80% to a DC,
    # make sure it has at least 80% power
    top_dc = int(np.argmax(route_share))
    ps[top_dc] = max(float(ps[top_dc]), float(route_share[top_dc]))

    # Safety: at least one DC must be powered
    if ps.max() < (1.0 / (NUM_NODE_TYPES + 0.99) + 1e-6):
        ps[top_dc] = 1.0

    action[2 * n:] = np.clip(ps, 0.0, 1.0)
    return action


def _ensure_feasible_power_sliders(power_sliders, w_small, w_large,
                                   has_traffic: bool, num_requests: int) -> np.ndarray:
    """
    Minimal feasibility: guarantee at least ONE DC can handle the traffic.
    Intentionally light-touch — the agent's power preferences are respected
    as much as possible to preserve scheme differentiation.
    """
    sl = np.clip(np.asarray(power_sliders, np.float64), 0.0, 1.0)
    if not has_traffic:
        return sl
    # Only ensure the single most-preferred DC has minimum power
    wt  = _normalize_weights(
        (np.asarray(w_small, np.float64) + np.asarray(w_large, np.float64)) / 2.0)
    best_dc = int(np.argmax(wt))
    sl[best_dc] = max(float(sl[best_dc]), 0.35)
    # If nothing is powered on at all, turn on the preferred DC fully
    if float(sl.max()) < (1.0 / (NUM_NODE_TYPES + 0.99) + 1e-6):
        sl[best_dc] = 1.0
    return sl


def _score_solution(metrics: dict, power_sliders, dc_usage: dict,
                    dc_to_idx: dict, weights, normalizer: MetricNormalizer,
                    update_norm: bool = True) -> float:
    """
    Score a simulation outcome against this agent's private normalizer and weights.

    Key design decisions:
    1. Metric ratios are variance-normalised so each metric contributes equally
       to the gradient signal for Balanced agents (prevents cost domination).
    2. Eco bonus is weighted by the agent's "eco-relevance" (1 − w_time),
       so MinLatency is not rewarded for powering off DCs.
    3. Dominance bonus is quadratic on the primary metric.

    update_norm=False during HER relabelling to avoid contaminating normalizers.
    """
    ttft   = float(metrics.get('avg_ttft',         metrics.get('avg_ttft_sec', 0.0)))
    carbon = float(metrics.get('carbon_emissions', 0.0)) / 1000.0
    water  = float(metrics.get('water_usage',      0.0)) / 100.0
    cost   = float(metrics.get('energy_cost',      0.0))

    if update_norm:
        normalizer.update(metrics)
    d_t, d_c, d_w, d_co = normalizer.denominators()

    # ── Raw metric ratios (lower is better, ~1.0 at baseline) ─────────────
    raw_ratios = [ttft / d_t, carbon / d_c, water / d_w, cost / d_co]

    # ── Inverse-variance effective weights ────────────────────────────────
    # Scale each metric weight inversely by its ratio std, then renormalise.
    # Effect: low-variance metrics (TTFT) get boosted weight, high-variance
    # metrics (cost) get reduced weight.  This prevents whichever metric has
    # the highest dynamic range from dominating the Balanced agent's gradient.
    # Pure agents (w=[0,0,0,1]) are unaffected since only one weight is nonzero.
    stds = normalizer.ratio_stds()
    raw_ew = [w / s for w, s in zip(weights, stds)]
    ew_sum = sum(raw_ew) + 1e-8
    eff_weights = [e / ew_sum for e in raw_ew]

    # Weighted metric penalty: uses variance-balanced effective weights
    wm = sum(ew * r for ew, r in zip(eff_weights, raw_ratios))

    # ── Dominance bonus: primary metric uses RAW ratio ────────────────────
    # (raw is more interpretable: "below EMA" means genuine improvement)
    primary_idx   = int(np.argmax(weights))
    primary_ratio = raw_ratios[primary_idx]
    dominance_bonus = max(0.0, 1.0 - primary_ratio) ** 2 * weights[primary_idx] * 3.0

    # ── Service rate ──────────────────────────────────────────────────────
    req_done  = float(metrics.get("requests_completed", metrics.get("served_requests", 0.0)))
    req_drop  = float(metrics.get("requests_dropped", 0.0))
    req_tot   = max(0.0, req_done + req_drop)
    sr        = (req_done / req_tot) if req_tot > 0.0 else 0.0

    # ── Eco bonus: weighted by eco-relevance ──────────────────────────────
    # Powering off DCs reduces carbon, water, and cost — but NOT latency.
    # MinLatency (w_time=1.0) gets zero eco bonus.
    # MinCarbon/MinWater/MinCost (w_time=0.0) get full eco bonus.
    # Balanced (w_time=0.25) gets 75% of the eco bonus.
    eco_relevance = 1.0 - float(weights[0])   # 0 for MinLatency, 1 for pure eco agents

    active       = _active_nodes_per_dc(power_sliders)
    pof          = 1.0 - float(active.sum()) / max(len(power_sliders) * NUM_NODE_TYPES, 1)
    dof          = int(np.sum(active == 0)) / max(len(power_sliders), 1)
    eco_bonus    = (math.sqrt(max(pof, 0.0)) * sr * 0.75 + dof * sr * 1.0) * ECO_BONUS_SCALE * eco_relevance

    # ── SLA penalty ───────────────────────────────────────────────────────
    sla_penalty  = (1.0 - sr) * 1.5 if req_tot > 0.0 else 0.0

    return dominance_bonus + eco_bonus - wm * METRIC_REWARD_SCALE - sla_penalty


# ─────────────────────────────────────────────────────────────────────────────
# MAIN OPTIMIZER
# ─────────────────────────────────────────────────────────────────────────────
def milp_optimizer(epoch_data, epoch_idx: int, node_properties: dict, epoch_summary: dict):
    global _GLOBAL_AGENTS, _PREV_STATES, _POLITICAL_CAPITAL, _EPOCH_HISTORY

    spec_dir  = epoch_summary.get('spec_dir',      'sim_specs')
    epoch_len = int(epoch_summary.get('epoch_length', 900))

    # ── Robust DC discovery ───────────────────────────────────────────────
    # Try multiple sources since the caller may not pass all DCs in node_properties.
    # Priority: node_properties → simulator.datacenters → epoch_data.source_dc
    dc_id_set = set()

    # Source 1: node_properties (the explicit argument)
    if node_properties:
        dc_id_set.update(int(d) for d in node_properties.keys())

    # Source 2: probe the simulator's datacenter registry
    try:
        _probe_sim = Rate_Flow_Sim.LLM_Simulator(spec_dir=spec_dir, epoch_length=epoch_len, debug=False)
        if hasattr(_probe_sim, 'datacenters') and _probe_sim.datacenters:
            dc_id_set.update(int(d) for d in _probe_sim.datacenters.keys())
        del _probe_sim
    except Exception:
        pass

    # Source 3: unique source_dc values in the request data
    try:
        _edf = epoch_data if isinstance(epoch_data, pd.DataFrame) else pd.DataFrame(epoch_data)
        _src_col = "source_dc_id" if "source_dc_id" in _edf.columns else "source_dc"
        if _src_col in _edf.columns:
            dc_id_set.update(int(v) for v in pd.to_numeric(_edf[_src_col], errors="coerce").dropna().unique())
    except Exception:
        pass

    # Fallback
    if not dc_id_set:
        dc_id_set.add(0)

    dc_ids       = sorted(dc_id_set)
    real_num_dcs = len(dc_ids)
    dc_to_idx    = {int(d): i for i, d in enumerate(dc_ids)}

    # First-run info
    if not _GLOBAL_AGENTS or list(_GLOBAL_AGENTS.values())[0].num_dcs != real_num_dcs:
        np_keys = sorted(node_properties.keys()) if node_properties else []
        print(f"[DC-DISCOVERY] node_properties keys={np_keys}  "
              f"final dc_ids={dc_ids}  ({real_num_dcs} DCs)")

    # ── Agent initialisation ──────────────────────────────────────────────
    # Re-init if agents don't exist, DC count changed, or scheme roster changed
    agents_stale = (not _GLOBAL_AGENTS
                    or list(_GLOBAL_AGENTS.values())[0].num_dcs != real_num_dcs
                    or set(_GLOBAL_AGENTS.keys()) != set(SCHEMES))
    if agents_stale:
        print(f"[INIT] Booting {len(SCHEMES)} Parliament Agents (SAC)...")
        _GLOBAL_AGENTS.clear()
        _PREV_STATES.clear()
        _POLITICAL_CAPITAL = {s: 100.0 for s in SCHEMES}
        for s in SCHEMES:
            _GLOBAL_AGENTS[s] = SACAgent(real_num_dcs, 4, SCHEME_WEIGHTS[s])

    # ── Data preparation ──────────────────────────────────────────────────
    clean_data = (epoch_data.copy() if isinstance(epoch_data, pd.DataFrame)
                  else pd.DataFrame(epoch_data))
    clean_data = clean_data.rename(
        columns={"source_dc_id": "source_dc", "model_type": "model", "num_tokens": "tokens"})
    for col, default in [("model", "Llama7b"), ("tokens", 1024),
                          ("source_dc", 0), ("arrival_ms", 0.0)]:
        if col not in clean_data.columns:
            clean_data[col] = default

    clean_data["source_dc"]  = pd.to_numeric(clean_data["source_dc"],  errors="coerce").fillna(0).astype(int)
    clean_data["model"]      = clean_data["model"].astype(str)
    clean_data["tokens"]     = pd.to_numeric(clean_data["tokens"],     errors="coerce").fillna(0).astype(int).clip(lower=0)
    clean_data["arrival_ms"] = pd.to_numeric(clean_data["arrival_ms"], errors="coerce").fillna(0.0).clip(lower=0.0)
    clean_data["model"]      = clean_data["model"].map(
        lambda n: n if MODEL_VARIANT in n else n + MODEL_VARIANT)
    clean_data = clean_data.reset_index(drop=True)
    has_traffic = len(clean_data) > 0

    if not has_traffic:
        print(f"[EPOCH {epoch_idx}] Zero traffic — shutting down datacenters.")
        zero = {k: 0. for k in ['avg_ttft', 'carbon_emissions', 'water_usage',
                                  'energy_cost', 'total_energy',
                                  'requests_completed', 'requests_dropped']}
        zero_metrics = {s: zero.copy() for s in SCHEMES + ["Parliament"]}
        _EPOCH_HISTORY.append({"epoch": epoch_idx, "metrics": zero_metrics})
        return (zero_metrics,
                {s: [] for s in SCHEMES + ["Parliament"]}, [])

    # ── Phase 1: Parallel SAC training + per-scheme proposal generation ───
    # First-epoch diagnostic: show what agents see and what they plan to do
    if epoch_idx == 0 or not _PREV_STATES:
        _diag_sim = Rate_Flow_Sim.LLM_Simulator(spec_dir=spec_dir, epoch_length=epoch_len, debug=False)
        _diag_state = get_rich_state(_diag_sim, dc_ids, clean_data, epoch_idx)
        _ds = _diag_state.reshape(real_num_dcs, -1)[:, :4]
        _homogeneous = bool(np.allclose(_ds, _ds[0:1], atol=1e-4))
        print(f"[DIAG] DC state ({'HOMOGENEOUS' if _homogeneous else 'heterogeneous'}):")
        print(f"       {'DC':>4}  {'Carbon':>8}  {'Cost':>8}  {'Water':>8}  {'Load':>8}")
        for di in range(real_num_dcs):
            print(f"       {dc_ids[di]:>4}  {_ds[di,0]:>8.4f}  {_ds[di,1]:>8.4f}  "
                  f"{_ds[di,2]:>8.4f}  {_ds[di,3]:>8.4f}")
        print(f"[DIAG] Heuristic actions (home_dc, top_route%, DCs_on):")
        for s in SCHEMES:
            ag = _GLOBAL_AGENTS[s]
            ha = ag._heuristic_action(_diag_state)
            ha = _enforce_route_power_coherence(ha, real_num_dcs)
            r_avg = (ha[:real_num_dcs] + ha[real_num_dcs:2*real_num_dcs]) / 2
            pw = ha[2*real_num_dcs:]
            nodes = _active_nodes_per_dc(pw)
            top_dc = int(np.argmax(r_avg))
            print(f"       {s:>12}: home=DC{int(np.argmax(ag.weights)) % real_num_dcs}  "
                  f"top_route=DC{top_dc}({r_avg[top_dc]:.0%})  "
                  f"nodes={list(nodes)}  DCs_on={int(np.sum(nodes > 0))}")
        del _diag_sim

    def process_scheme(scheme: str):
        agent      = _GLOBAL_AGENTS[scheme]
        temp_sim   = Rate_Flow_Sim.LLM_Simulator(spec_dir=spec_dir, epoch_length=epoch_len, debug=False)
        curr_state = get_rich_state(temp_sim, dc_ids, clean_data, epoch_idx)
        prev_state = _PREV_STATES.get(scheme)

        # her_log collects raw simulation data for cross-agent relabelling after threads join
        her_log: list = []
        last_action = last_reward = None

        for _ in range(OPTIM_STEPS):
            full_action   = agent.select_action(curr_state, exploration=True)
            full_action   = _enforce_route_power_coherence(full_action, real_num_dcs)
            w_s  = _normalize_weights(full_action[:real_num_dcs])
            w_l  = _normalize_weights(full_action[real_num_dcs:2 * real_num_dcs])
            ps   = _ensure_feasible_power_sliders(
                full_action[2 * real_num_dcs:], w_s, w_l,
                has_traffic=has_traffic, num_requests=len(clean_data))
            pp   = build_power_plan_sliding(dc_ids, ps)
            sp   = build_schedule_map(clean_data, dc_ids, w_s, w_l, ps, epoch_idx)

            metrics, _, dc_usage = temp_sim.run_epoch(epoch_idx, clean_data, sp, pp)
            reward = _score_solution(metrics, ps, dc_usage, dc_to_idx,
                                     agent.weights, _GLOBAL_NORMALIZER, update_norm=True)

            agent.intra_buffer.push(curr_state, full_action, reward, curr_state, False)
            her_log.append((curr_state.copy(), full_action.copy(), metrics, ps.copy(), dc_usage))
            last_action, last_reward = full_action, reward

            for _ in range(4):
                agent.train()

        # Cross-epoch transition: bridges temporal gap with 3× priority boost
        # so the critic learns to value state changes between epochs with GAMMA=0.95
        if prev_state is not None and last_action is not None:
            agent.cross_buffer.push(prev_state, last_action, last_reward,
                                    curr_state, False, priority_boost=3.0)
            for _ in range(4):
                agent.train()

        _PREV_STATES[scheme] = curr_state

        # Final deterministic exploitation run — unique solution per scheme
        best_action = agent.select_action(curr_state, exploration=False)
        best_action = _enforce_route_power_coherence(best_action, real_num_dcs)
        agent.epoch_count += 1
        w_s  = _normalize_weights(best_action[:real_num_dcs])
        w_l  = _normalize_weights(best_action[real_num_dcs:2 * real_num_dcs])
        ps   = _ensure_feasible_power_sliders(
            best_action[2 * real_num_dcs:], w_s, w_l,
            has_traffic=has_traffic, num_requests=len(clean_data))
        pp   = build_power_plan_sliding(dc_ids, ps)
        sp   = build_schedule_map(clean_data, dc_ids, w_s, w_l, ps, epoch_idx)

        final_sim = Rate_Flow_Sim.LLM_Simulator(spec_dir=spec_dir, epoch_length=epoch_len, debug=False)
        fm, fr, _ = final_sim.run_epoch(epoch_idx, clean_data, sp, pp)
        return scheme, curr_state, best_action, fm, fr, her_log

    all_metrics: dict = {}
    all_results: dict = {}
    proposals:   dict = {}
    states:      dict = {}
    her_pool:    list = []

    with ThreadPoolExecutor(max_workers=min(len(SCHEMES), 8)) as ex:
        futures = [ex.submit(process_scheme, s) for s in SCHEMES]
        for f in as_completed(futures):
            scheme, curr, action, fm, fr, her_log = f.result()
            states[scheme]      = curr
            proposals[scheme]   = action
            all_metrics[scheme] = fm
            all_results[scheme] = fr
            her_pool.extend(her_log)

    # ── HER: Cross-label every simulation across all agents ───────────────
    # Each of the OPTIM_STEPS * 4 simulation results is re-scored under every
    # agent's weights and pushed into their intra_buffer. Cross-agent samples
    # get reduced priority (HER_CROSS_PRIORITY) to avoid diluting the agent's
    # own policy signal while still providing diverse experience.
    # update_norm=False preserves each agent's normalizer integrity.
    for s_vec, a_vec, mets, ps, dc_use in her_pool:
        for scheme in SCHEMES:
            ag = _GLOBAL_AGENTS[scheme]
            r  = _score_solution(mets, ps, dc_use, dc_to_idx,
                                 ag.weights, _GLOBAL_NORMALIZER, update_norm=False)
            ag.intra_buffer.push(s_vec, a_vec, r, s_vec, False,
                                 priority_boost=HER_CROSS_PRIORITY)

    # ── Phase 2: Parliament negotiation ───────────────────────────────────
    # Each agent evaluates all proposals with its own critic and casts a
    # capital-weighted vote. Votes are squared to penalise extreme proposals.
    capital_votes = {s: 0.0 for s in SCHEMES}

    for eval_s in SCHEMES:
        evaluator = _GLOBAL_AGENTS[eval_s]
        q_vals    = {}
        s_t       = torch.FloatTensor(evaluator._augment(states[eval_s])).unsqueeze(0)
        for prop_s in SCHEMES:
            a_t = torch.FloatTensor(proposals[prop_s]).unsqueeze(0)
            with torch.no_grad():
                q_vals[prop_s] = evaluator.critic.q_min(s_t, a_t).item()

        mn, mx = min(q_vals.values()), max(q_vals.values())
        for ps in SCHEMES:
            ns = (q_vals[ps] - mn) / (mx - mn + 1e-8)
            capital_votes[ps] += (ns ** 2) * _POLITICAL_CAPITAL[eval_s]

    tot = sum(capital_votes.values())
    consensus_action = (
        sum((capital_votes[s] / tot) * proposals[s] for s in SCHEMES)
        if tot > 0 else proposals[SCHEMES[0]]
    )

    # ── Gradient ascent consensus refinement (capital-weighted) ─────────
    # The blended action is a geometric average that no critic necessarily endorses.
    # Gradient steps now weight each critic by its agent's political capital,
    # so high-performing agents steer refinement more aggressively.
    c_t   = torch.FloatTensor(consensus_action).unsqueeze(0).requires_grad_(True)
    g_opt = optim.SGD([c_t], lr=PARLIAMENT_GRAD_LR)

    # Pre-compute normalised capital weights for gradient weighting
    cap_total  = sum(_POLITICAL_CAPITAL[s] for s in SCHEMES)
    cap_weight = {s: _POLITICAL_CAPITAL[s] / (cap_total + 1e-8) for s in SCHEMES}

    for _ in range(PARLIAMENT_GRAD_STEPS):
        g_opt.zero_grad()
        total_q = sum(
            cap_weight[s] * _GLOBAL_AGENTS[s].critic.q_min(
                torch.FloatTensor(_GLOBAL_AGENTS[s]._augment(states[s])).unsqueeze(0), c_t)
            for s in SCHEMES
        )
        (-total_q).backward()
        g_opt.step()
        # Projected gradient: re-normalise back to valid action space after each step.
        # Use .data to bypass autograd on the projection itself.
        with torch.no_grad():
            for k in range(NUM_MODEL_CLASSES):
                st, en = k * real_num_dcs, (k + 1) * real_num_dcs
                c_t.data[:, st:en] = F.softmax(c_t.data[:, st:en], dim=1)
            c_t.data[:, NUM_MODEL_CLASSES * real_num_dcs:].clamp_(0.0, 1.0)

    consensus_action = c_t.detach().cpu().numpy()[0]

    # ── Veto phase: high-capital agents reject harmful consensus ──────
    # If an agent has enough capital and the consensus would significantly
    # degrade its Q-value relative to its own proposal, it vetoes by pulling
    # the consensus back toward its proposal proportional to the degradation.
    for scheme in SCHEMES:
        if _POLITICAL_CAPITAL[scheme] < VETO_CAPITAL_THRESH:
            continue
        ag  = _GLOBAL_AGENTS[scheme]
        s_t = torch.FloatTensor(ag._augment(states[scheme])).unsqueeze(0)
        with torch.no_grad():
            q_own  = ag.critic.q_min(s_t, torch.FloatTensor(proposals[scheme]).unsqueeze(0)).item()
            q_cons = ag.critic.q_min(s_t, torch.FloatTensor(consensus_action).unsqueeze(0)).item()

        degradation = (q_own - q_cons) / (abs(q_own) + 1e-6)
        if degradation > VETO_Q_DEGRADATION:
            # Veto strength scales with capital and degradation severity
            veto_str = min(VETO_STRENGTH_CAP,
                           degradation * _POLITICAL_CAPITAL[scheme] / 500.0)
            consensus_action = ((1.0 - veto_str) * consensus_action
                                + veto_str * proposals[scheme])
            # Re-normalise after veto blend
            for k in range(NUM_MODEL_CLASSES):
                seg = consensus_action[k * real_num_dcs:(k + 1) * real_num_dcs]
                seg_sum = seg.sum()
                if seg_sum > 0:
                    consensus_action[k * real_num_dcs:(k + 1) * real_num_dcs] = seg / seg_sum
                else:
                    consensus_action[k * real_num_dcs:(k + 1) * real_num_dcs] = 1.0 / real_num_dcs
            consensus_action[NUM_MODEL_CLASSES * real_num_dcs:] = np.clip(
                consensus_action[NUM_MODEL_CLASSES * real_num_dcs:], 0.0, 1.0)
            print(f"  [VETO] {scheme} agent exercised veto "
                  f"(capital={_POLITICAL_CAPITAL[scheme]:.0f}, "
                  f"degradation={degradation:.2f}, strength={veto_str:.2f})")

    # ── Phase 3: Parliament final execution ──────────────────────────────
    consensus_action = _enforce_route_power_coherence(consensus_action, real_num_dcs)
    w_p  = _normalize_weights(consensus_action[:real_num_dcs])
    wl_p = _normalize_weights(consensus_action[real_num_dcs:2 * real_num_dcs])
    ps_p = _ensure_feasible_power_sliders(
        consensus_action[2 * real_num_dcs:], w_p, wl_p,
        has_traffic=has_traffic, num_requests=len(clean_data))
    pp_p = build_power_plan_sliding(dc_ids, ps_p)
    sp_p = build_schedule_map(clean_data, dc_ids, w_p, wl_p, ps_p, epoch_idx)

    parl_sim = Rate_Flow_Sim.LLM_Simulator(spec_dir=spec_dir, epoch_length=epoch_len, debug=False)
    mp, rp, _ = parl_sim.run_epoch(epoch_idx, clean_data, sp_p, pp_p)
    all_metrics["Parliament"] = mp
    all_results["Parliament"] = rp

    # ── Evolving political capital (performance-based) ──────────────────
    # Agents earn capital based on how well the Parliament outcome performs
    # on their primary metric. Lower is better — agents whose metric is
    # well-served by the consensus gain influence; those whose metric suffers
    # lose it. This creates a genuine competitive reputation mechanism.
    primary_raw = {}
    for scheme in SCHEMES:
        mk = PRIMARY_METRIC_KEY[scheme]
        primary_raw[scheme] = float(mp.get(mk, 0.0))

    # Normalise primary metrics into [0, 1] where 0 = best, 1 = worst
    raw_vals = [primary_raw[s] for s in SCHEMES]
    mn_raw, mx_raw = min(raw_vals), max(raw_vals)
    for scheme in SCHEMES:
        if mx_raw > mn_raw:
            # Invert: lower raw metric → higher performance score
            perf_score = 1.0 - (primary_raw[scheme] - mn_raw) / (mx_raw - mn_raw)
        else:
            perf_score = 0.5   # All metrics identical — neutral

        # Also compare Parliament outcome vs this agent's own proposal outcome
        own_mk  = float(all_metrics[scheme].get(PRIMARY_METRIC_KEY[scheme], 0.0))
        parl_mk = primary_raw[scheme]
        # Bonus if Parliament outcome is at least as good as agent's own proposal
        # on the agent's metric (incentivises agents to propose good actions)
        proposal_bonus = max(0.0, (own_mk - parl_mk) / (abs(own_mk) + 1e-6))

        combined_performance = perf_score + proposal_bonus * 0.5

        _POLITICAL_CAPITAL[scheme] = max(
            10.0,  # Floor: no scheme goes permanently silent
            CAPITAL_DECAY * _POLITICAL_CAPITAL[scheme]
            + (1 - CAPITAL_DECAY) * combined_performance * 250.0)

    # ── Print unified epoch comparison table ─────────────────────────────
    _print_epoch_table(epoch_idx, all_metrics)

    # ── Record epoch for summary ──────────────────────────────────────────
    _EPOCH_HISTORY.append({"epoch": epoch_idx, "metrics": dict(all_metrics)})

    return all_metrics, all_results, []