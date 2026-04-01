import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import random
import math
import hashlib
import os
import gc
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
        # TTFT floor lowered (0.5 → 0.1) so normalizer penalises high latency harder
        return (max(self.ttft, 0.1), max(self.carbon, 5.0),
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
ONLINE_ADJUST_STEPS   = 10              # Gradient steps on replay buffer before proposing
OFFLINE_TRAIN_STEPS   = 40              # Gradient steps on replay buffer after epoch execution
OPTIM_STEPS           = 4               # Exploration sims per agent in Phase 1 (with heuristic blend)
OFFLINE_EXPLORE_SIMS  = 3               # Simulations per agent during offline training
OFFLINE_GRAD_STEPS    = 30              # Base gradient steps per agent after offline exploration
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
# METRIC AGENTS  — exactly 4, one per optimisation objective.
# These are the actual RL agents.  Schemes set their voting weights in Phase 2.
# ─────────────────────────────────────────────────────────────────────────────
METRIC_AGENTS = ["TTFT", "Carbon", "Water", "Cost"]
METRIC_AGENT_IDENTITY = {
    "TTFT":   np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    "Carbon": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
    "Water":  np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
    "Cost":   np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
}
METRIC_AGENT_INDEX = {"TTFT": 0, "Carbon": 1, "Water": 2, "Cost": 3}

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

# ─────────────────────────────────────────────────────────────────────────────
# GLOBALS  — agents are keyed by metric name, NOT by scheme name.
# ─────────────────────────────────────────────────────────────────────────────
_GLOBAL_AGENTS     = {}                          # metric_name → SACAgent (always 4)
_PREV_STATES       = {}                          # metric_name → prev state array
_POLITICAL_CAPITAL = {}                          # metric_name → float (personal to agent)
_EPOCH_HISTORY     = []                          # List of {epoch, metrics} dicts
_GLOBAL_NORMALIZER = MetricNormalizer()
ABLATION_MODE      = ""                          # Set externally: no-film, no-veto, etc.

CAPITAL_BASE  = 50.0    # Floor: every agent gets at least this much capital
CAPITAL_TOTAL = 400.0   # Total capital budget distributed across 4 agents


def _compute_initial_capital() -> dict:
    """
    Distribute initial capital across the 4 metric agents based on scheme weights.

    Aggregates how much voting weight each agent receives across all registered
    schemes.  Agents that appear more prominently in the scheme roster start
    with more capital — so if MinCarbon is the only scheme, the Carbon agent
    dominates from epoch 0.

    Every agent receives at least CAPITAL_BASE to ensure no agent is permanently
    silenced.  The remaining budget (CAPITAL_TOTAL − 4 × CAPITAL_BASE) is
    distributed proportionally to aggregate scheme weight.

    Example with schemes [MinCarbon, Balanced]:
        Aggregate weights: TTFT=0.25, Carbon=1.25, Water=0.25, Cost=0.25
        Carbon gets the largest share of the bonus capital.
    """
    if not SCHEME_WEIGHTS:
        return {ag: CAPITAL_BASE + (CAPITAL_TOTAL - 4 * CAPITAL_BASE) / 4
                for ag in METRIC_AGENTS}

    # Sum each agent's weight across all schemes
    agg = np.zeros(4, dtype=np.float64)
    for sw in SCHEME_WEIGHTS.values():
        agg += np.array(sw, dtype=np.float64)

    agg_total = agg.sum()
    if agg_total <= 0:
        return {ag: CAPITAL_TOTAL / 4 for ag in METRIC_AGENTS}

    bonus_pool = CAPITAL_TOTAL - 4 * CAPITAL_BASE
    capital = {}
    for i, ag_name in enumerate(METRIC_AGENTS):
        capital[ag_name] = CAPITAL_BASE + bonus_pool * (agg[i] / agg_total)

    return capital


def configure_schemes(scheme_list: list = None):
    """
    Configure the scheme roster from a list of (name, weights) tuples.
    Weights are always ordered [Time, Carbon, Water, Cost] and will be normalised
    to sum to 1.0.  These weights control how the 4 metric agents' proposals are
    blended in Phase 2 — they do NOT create agents.  Agents are always exactly 4,
    one per metric (TTFT, Carbon, Water, Cost).

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
    global SCHEMES, SCHEME_WEIGHTS
    global _GLOBAL_AGENTS, _PREV_STATES, _POLITICAL_CAPITAL, _EPOCH_HISTORY
    global _GLOBAL_NORMALIZER

    if scheme_list is None:
        scheme_list = DEFAULT_SCHEMES

    if not scheme_list:
        raise ValueError("scheme_list must contain at least one (name, weights) tuple")

    SCHEMES        = []
    SCHEME_WEIGHTS = {}

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

    # Reset agents (keyed by metric, not scheme) and capital
    _GLOBAL_AGENTS.clear()
    _PREV_STATES.clear()
    _POLITICAL_CAPITAL = _compute_initial_capital()
    _EPOCH_HISTORY     = []
    _GLOBAL_NORMALIZER = MetricNormalizer()

    nw = max(12, max(len(s) for s in SCHEMES))
    print(f"[CONFIG] {len(SCHEMES)} scheme(s) registered  "
          f"(4 metric agents: {', '.join(METRIC_AGENTS)}):")
    for s in SCHEMES:
        w = SCHEME_WEIGHTS[s]
        wstr = "  ".join(f"{ml.split('(')[0]}={v:.2f}" for ml, v in zip(METRIC_LABELS, w))
        print(f"  {s:>{nw}}  {wstr}")
    print(f"  Initial capital: "
          + "  ".join(f"{ag}={_POLITICAL_CAPITAL[ag]:.0f}" for ag in METRIC_AGENTS))


def reset_simulation():
    """Clear all agent state and history for a clean restart with current schemes."""
    global _GLOBAL_AGENTS, _PREV_STATES, _POLITICAL_CAPITAL, _EPOCH_HISTORY
    _GLOBAL_AGENTS.clear()
    _PREV_STATES.clear()
    _POLITICAL_CAPITAL = _compute_initial_capital()
    _EPOCH_HISTORY     = []
    _GLOBAL_NORMALIZER = MetricNormalizer()
    print("[RESET] All agents, buffers, capital, and history cleared.")


def _print_epoch_table(epoch_idx: int, all_metrics: dict):
    """
    Print a compact per-epoch comparison table.  ★ marks the best (lowest)
    value in each metric column across schemes.  Each scheme result is the
    full framework output (Phase 1 agents + Phase 2 consensus).
    """
    all_keys = [s for s in SCHEMES if s in all_metrics]
    if not all_keys:
        return
    nw = max(12, max((len(s) for s in all_keys), default=12))

    # Gather display values: (ttft, carbon_kg, water_L, cost, served)
    rows = {}
    for s in all_keys:
        m = all_metrics[s]
        rows[s] = [
            float(m.get("avg_ttft", 0)),
            float(m.get("carbon_emissions", 0)) / 1000.0,
            float(m.get("water_usage", 0)) / 100.0,
            float(m.get("energy_cost", 0)),
            int(m.get("requests_completed", 0)),
        ]

    # Find best (lowest) per metric column
    best_idx = [None, None, None, None]
    for ci in range(4):
        best_v = float("inf")
        for s in all_keys:
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
           f"{'Water(L)':>11} {'Cost($)':>11} {'Served':>8}")
    bar = "─" * len(hdr)
    print(f"┌ EPOCH {epoch_idx} {bar[len(f'  EPOCH {epoch_idx} ') + 1:]}┐")
    print(f"│{hdr[1:]}│")
    print(f"│{bar[1:]}│")

    for s in all_keys:
        if s not in rows:
            continue
        v = rows[s]
        stars = ["★" if best_idx[ci] == s else " " for ci in range(4)]
        cols = "".join(f"{_fmt(v[ci], ci)}{stars[ci]}" for ci in range(4))
        print(f"│ {s:<{nw}} {cols} {v[4]:>8}  │")

    # Show agent capital below the table
    cap_str = "  ".join(f"{ag}={_POLITICAL_CAPITAL.get(ag, 0):.0f}" for ag in METRIC_AGENTS)
    print(f"│ Agent Capital: {cap_str:<{len(bar) - 18}}│")
    print(f"└{bar[1:]}┘")


def print_run_summary():
    """
    Print a formatted summary table across all recorded epochs.
    TTFT is averaged (it's a latency); Carbon, Water, Cost are summed (cumulative).
    Each scheme gets its own row (all are full-framework results).
    ★ marks the best scheme per column.
    Zero-traffic epochs (where all schemes served 0 requests) are excluded.
    """
    if not _EPOCH_HISTORY:
        print("[SUMMARY] No epochs recorded yet.")
        return

    all_keys = list(SCHEMES)
    accum = {s: {k: [] for k in METRIC_KEYS + ["requests_completed"]}
             for s in all_keys}

    skipped = 0
    for record in _EPOCH_HISTORY:
        em = record["metrics"]
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

    display = {}
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

    best_idx = [None, None, None, None]
    for ci in range(4):
        best_v = float("inf")
        for s in all_keys:
            if s in display and display[s][ci] < best_v:
                best_v = display[s][ci]
                best_idx[ci] = s

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

    # Show what each scheme weights mean for agent voting
    print("  Scheme voting weights:  [TTFT  Carbon  Water  Cost]")
    for s in SCHEMES:
        w = SCHEME_WEIGHTS[s]
        dominant = METRIC_AGENTS[int(np.argmax(w))]
        wstr = "  ".join(f"{v:.2f}" for v in w)
        print(f"    {s:>{nw}}  [{wstr}]  → {dominant} agent dominates")
    print(thin)

    print(hdr)
    print(thin)
    for s in all_keys:
        if s not in display:
            continue
        d = display[s]
        stars = ["★" if best_idx[ci] == s else " " for ci in range(4)]
        cols = "".join(f"{_sfmt(d[ci], ci)}{stars[ci]}" for ci in range(4))
        print(f"  {s:<{nw}} {cols} {d[4]:>10}{d[5]:>7}")

    print(sep)

    print("\n  Best per metric:")
    labels_short = ["Avg TTFT", "Total Carbon", "Total Water", "Total Cost"]
    for ci in range(4):
        if best_idx[ci] and best_idx[ci] in display:
            print(f"    {labels_short[ci]:<14} → {best_idx[ci]} "
                  f"({display[best_idx[ci]][ci]:.3f})")

    # Show final agent capital
    print(f"\n  Agent Capital: ", end="")
    print("  ".join(f"{ag}={_POLITICAL_CAPITAL.get(ag, 0):.0f}" for ag in METRIC_AGENTS))
    print(sep + "\n")


# ── Auto-initialise from DEFAULT_SCHEMES on first import ──────────────────
configure_schemes()


def save_agents(path: str):
    """
    Save all 4 metric agents' learned parameters and state to disk.
    Includes actor/critic networks, entropy temperature, replay buffers,
    normalizer, capital, and epoch counts.
    """
    if not _GLOBAL_AGENTS:
        print("[SAVE] No agents to save — run at least one epoch first.")
        return

    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

    checkpoint = {
        "agents": {},
        "capital": dict(_POLITICAL_CAPITAL),
        "normalizer": {
            "ttft": _GLOBAL_NORMALIZER.ttft,
            "carbon": _GLOBAL_NORMALIZER.carbon,
            "water": _GLOBAL_NORMALIZER.water,
            "cost": _GLOBAL_NORMALIZER.cost,
            "n_obs": _GLOBAL_NORMALIZER.n_obs,
            "ratio_ema": list(_GLOBAL_NORMALIZER.ratio_ema),
            "ratio_sq_ema": list(_GLOBAL_NORMALIZER.ratio_sq_ema),
            "sla_target": _GLOBAL_NORMALIZER.sla_target,
        },
        "schemes": list(SCHEMES),
        "scheme_weights": dict(SCHEME_WEIGHTS),
    }

    for ag_name in METRIC_AGENTS:
        if ag_name not in _GLOBAL_AGENTS:
            continue
        ag = _GLOBAL_AGENTS[ag_name]
        checkpoint["agents"][ag_name] = {
            "actor_state": ag.actor.state_dict(),
            "critic_state": ag.critic.state_dict(),
            "critic_target_state": ag.critic_target.state_dict(),
            "log_alpha": ag.log_alpha.detach().clone(),
            "epoch_count": ag.epoch_count,
            "num_dcs": ag.num_dcs,
            "weights": ag.weights.tolist(),
        }

    torch.save(checkpoint, path)
    print(f"[SAVE] Saved 4 agents to {path}  "
          f"(epoch_counts: {[_GLOBAL_AGENTS[ag].epoch_count for ag in METRIC_AGENTS]})")


def load_agents(path: str, num_dcs: int = None):
    """
    Load pre-trained agents from disk.  If the checkpoint has a different DC
    count than requested, automatically falls back to transfer_agents for
    compatible weight transfer instead of failing.
    """
    global _GLOBAL_AGENTS, _POLITICAL_CAPITAL, _GLOBAL_NORMALIZER

    if not os.path.exists(path):
        print(f"[LOAD] File not found: {path}")
        return False

    checkpoint = torch.load(path, weights_only=False)

    # Check for DC count mismatch
    saved_agents = checkpoint.get("agents", {})
    if saved_agents and num_dcs is not None:
        first_ag = next(iter(saved_agents.values()))
        saved_num_dcs = first_ag.get("num_dcs", num_dcs)
        if saved_num_dcs != num_dcs:
            print(f"[LOAD] DC mismatch: checkpoint has {saved_num_dcs} DCs, "
                  f"need {num_dcs} — falling back to transfer learning")
            return transfer_agents(path, target_num_dcs=num_dcs)

    # Restore normalizer
    norm_state = checkpoint.get("normalizer", {})
    _GLOBAL_NORMALIZER.ttft = norm_state.get("ttft")
    _GLOBAL_NORMALIZER.carbon = norm_state.get("carbon")
    _GLOBAL_NORMALIZER.water = norm_state.get("water")
    _GLOBAL_NORMALIZER.cost = norm_state.get("cost")
    _GLOBAL_NORMALIZER.n_obs = norm_state.get("n_obs", 0)
    _GLOBAL_NORMALIZER.ratio_ema = norm_state.get("ratio_ema", [1.0]*4)
    _GLOBAL_NORMALIZER.ratio_sq_ema = norm_state.get("ratio_sq_ema", [1.0]*4)
    _GLOBAL_NORMALIZER.sla_target = norm_state.get("sla_target", 0.80)

    # Restore capital
    saved_capital = checkpoint.get("capital", {})
    for ag_name in METRIC_AGENTS:
        if ag_name in saved_capital:
            _POLITICAL_CAPITAL[ag_name] = saved_capital[ag_name]

    # Debug: print keys from first agent
    if saved_agents:
        first_name = next(iter(saved_agents))
        print(f"[LOAD] Checkpoint agent keys: {list(saved_agents[first_name].keys())}")

    def _find_state(src_dict, *candidates):
        for key in candidates:
            if key in src_dict:
                return src_dict[key]
        return None

    for ag_name in METRIC_AGENTS:
        if ag_name not in saved_agents:
            continue
        ag_state = saved_agents[ag_name]
        saved_num_dcs = ag_state.get("num_dcs", num_dcs)

        # Create agent if needed
        if (ag_name not in _GLOBAL_AGENTS
                or _GLOBAL_AGENTS[ag_name].num_dcs != saved_num_dcs):
            _GLOBAL_AGENTS[ag_name] = SACAgent(
                saved_num_dcs, 4, METRIC_AGENT_IDENTITY[ag_name])

        ag = _GLOBAL_AGENTS[ag_name]

        actor_sd = _find_state(ag_state, "actor_state", "actor", "actor_state_dict")
        if actor_sd is not None:
            ag.actor.load_state_dict(actor_sd)

        critic_sd = _find_state(ag_state, "critic_state", "critic", "critic_state_dict")
        if critic_sd is not None:
            ag.critic.load_state_dict(critic_sd)

        target_sd = _find_state(ag_state, "critic_target_state", "critic_target",
                                "target_critic_state", "target_state")
        if target_sd is not None:
            ag.critic_target.load_state_dict(target_sd)

        alpha_val = _find_state(ag_state, "log_alpha", "alpha")
        if alpha_val is not None and hasattr(alpha_val, 'clone'):
            ag.log_alpha = alpha_val.clone().requires_grad_(True)
            ag.alpha = float(ag.log_alpha.exp())
            ag.alpha_optimizer = optim.Adam([ag.log_alpha], lr=LR_ALPHA)

        ag.epoch_count = ag_state.get("epoch_count", 0)

    print(f"[LOAD] Loaded 4 agents from {path}  "
          f"(epoch_counts: {[_GLOBAL_AGENTS[ag].epoch_count for ag in METRIC_AGENTS if ag in _GLOBAL_AGENTS]})")
    return True


def transfer_agents(source_path: str, target_num_dcs: int):
    """
    Transfer learned weights from a checkpoint trained on a different DC count.

    Copies all layers whose shapes are DC-independent (hidden-to-hidden layers,
    FiLM conditioning, critic middle/output layers, entropy temperature, normalizer).
    Layers whose dimensions depend on DC count (input projections, output heads)
    keep their fresh random initialization.

    Transferred layers (DC-independent):
      Actor:  dc_net hidden layer (128→128), FiLM (4→256), all biases
      Critic: q1/q2 hidden layer (256→256), output layer (256→1)
      Other:  log_alpha, normalizer EMA state

    Reinitialized layers (DC-dependent):
      Actor:  dc_net input layer (dc_state_dim→128), mean_head, log_std_head
      Critic: q1/q2 input layer (state_dim+action_dim→256)

    This preserves the agent's learned internal representations (how to
    evaluate DC tradeoffs) while allowing adaptation to a new action space.
    """
    global _GLOBAL_AGENTS, _POLITICAL_CAPITAL, _GLOBAL_NORMALIZER

    if not os.path.exists(source_path):
        print(f"[TRANSFER] Source not found: {source_path}")
        return False

    checkpoint = torch.load(source_path, weights_only=False)
    source_agents = checkpoint.get("agents", {})

    if not source_agents:
        print(f"[TRANSFER] No agents found in checkpoint")
        return False

    # Check source DC count
    first_ag = next(iter(source_agents.values()))
    source_dcs = first_ag.get("num_dcs", 0)
    if source_dcs == target_num_dcs:
        print(f"[TRANSFER] Source has same DC count ({source_dcs}) — using full load instead")
        return load_agents(source_path, num_dcs=target_num_dcs)

    print(f"[TRANSFER] Transferring from {source_dcs}-DC model → {target_num_dcs}-DC agents")

    # Create fresh agents with target DC count
    for ag_name in METRIC_AGENTS:
        if (ag_name not in _GLOBAL_AGENTS
                or _GLOBAL_AGENTS[ag_name].num_dcs != target_num_dcs):
            _GLOBAL_AGENTS[ag_name] = SACAgent(
                target_num_dcs, 4, METRIC_AGENT_IDENTITY[ag_name])

    transferred = 0
    skipped = 0

    for ag_name in METRIC_AGENTS:
        if ag_name not in source_agents:
            continue

        ag = _GLOBAL_AGENTS[ag_name]
        src = source_agents[ag_name]

        # Debug: print actual keys in checkpoint to diagnose format mismatches
        if ag_name == METRIC_AGENTS[0]:
            print(f"[TRANSFER] Checkpoint keys for '{ag_name}': {list(src.keys())}")

        # ── Flexible key mapping (handles different checkpoint versions) ──
        # Try multiple key name conventions
        def _find_state(src_dict, *candidates):
            for key in candidates:
                if key in src_dict:
                    return src_dict[key]
            return None

        # ── Transfer actor layers ─────────────────────────────────────────
        src_actor = _find_state(src, "actor_state", "actor", "actor_state_dict")
        if src_actor is not None:
            tgt_actor = ag.actor.state_dict()
            for key in tgt_actor:
                if key in src_actor and src_actor[key].shape == tgt_actor[key].shape:
                    tgt_actor[key] = src_actor[key]
                    transferred += 1
                else:
                    skipped += 1
            ag.actor.load_state_dict(tgt_actor)
        else:
            print(f"[TRANSFER] WARNING: No actor state found for {ag_name}")

        # ── Transfer critic layers ────────────────────────────────────────
        for critic, keys in [(ag.critic, ("critic_state", "critic", "critic_state_dict")),
                              (ag.critic_target, ("critic_target_state", "critic_target",
                                                   "target_critic_state", "target_state"))]:
            src_critic = _find_state(src, *keys)
            if src_critic is not None:
                tgt_critic = critic.state_dict()
                for key in tgt_critic:
                    if key in src_critic and src_critic[key].shape == tgt_critic[key].shape:
                        tgt_critic[key] = src_critic[key]
                        transferred += 1
                    else:
                        skipped += 1
                critic.load_state_dict(tgt_critic)

        # ── Transfer entropy temperature ──────────────────────────────────
        src_alpha = _find_state(src, "log_alpha", "alpha")
        if src_alpha is not None and hasattr(src_alpha, 'clone'):
            ag.log_alpha = src_alpha.clone().requires_grad_(True)
            ag.alpha = float(ag.log_alpha.exp())
            ag.alpha_optimizer = optim.Adam([ag.log_alpha], lr=LR_ALPHA)
            transferred += 1

    # ── Transfer normalizer ───────────────────────────────────────────────
    norm_state = checkpoint.get("normalizer", {})
    if norm_state.get("n_obs", 0) > 0:
        _GLOBAL_NORMALIZER.ttft = norm_state.get("ttft")
        _GLOBAL_NORMALIZER.carbon = norm_state.get("carbon")
        _GLOBAL_NORMALIZER.water = norm_state.get("water")
        _GLOBAL_NORMALIZER.cost = norm_state.get("cost")
        _GLOBAL_NORMALIZER.n_obs = norm_state.get("n_obs", 0)
        _GLOBAL_NORMALIZER.ratio_ema = norm_state.get("ratio_ema", [1.0]*4)
        _GLOBAL_NORMALIZER.ratio_sq_ema = norm_state.get("ratio_sq_ema", [1.0]*4)
        _GLOBAL_NORMALIZER.sla_target = norm_state.get("sla_target", 0.80)

    print(f"[TRANSFER] Done: {transferred} tensors transferred, "
          f"{skipped} reinitialized (shape mismatch)")
    print(f"[TRANSFER] Transferred: hidden layers, FiLM, critic mid/out, "
          f"entropy temp, normalizer")
    print(f"[TRANSFER] Reinitialized: input projections, output heads "
          f"(DC-dependent dimensions)")
    return True


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
        Agent identity (one-hot metric vector) enters via a dedicated FiLM branch
        that modulates hidden activations.  This guarantees distinct behavior per
        metric agent from epoch 0, independent of training.
        """
        super().__init__()
        self.num_dcs    = num_dcs
        self.action_dim = num_dcs * NUM_MODEL_CLASSES + num_dcs

        # DC feature trunk
        self.dc_net = nn.Sequential(
            nn.Linear(dc_state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),   nn.ReLU(),
        )
        # FiLM: agent identity produces scale + shift for each hidden unit.
        # Initialized with Xavier (not zeros!) so the 4-dim one-hot identity
        # immediately produces distinct modulation per agent from epoch 0.
        # This is critical for early differentiation before training kicks in.
        self.film = nn.Linear(4, hidden_dim * 2)
        nn.init.xavier_uniform_(self.film.weight, gain=0.5)
        nn.init.zeros_(self.film.bias)

        self.mean_head    = nn.Linear(hidden_dim, self.action_dim)
        self.log_std_head = nn.Linear(hidden_dim, self.action_dim)

    def _film_modulate(self, h: torch.Tensor, scheme_w: torch.Tensor) -> torch.Tensor:
        if ABLATION_MODE == "no-film":
            return h  # Ablation: skip FiLM, return unmodulated hidden state
        film_out     = self.film(scheme_w)
        scale, shift = film_out.chunk(2, dim=-1)
        return h * (1.0 + scale) + shift   # FiLM: h ← h*(1+γ) + β

    def _project(self, raw: torch.Tensor) -> torch.Tensor:
        parts = [F.softmax(raw[:, k * self.num_dcs:(k + 1) * self.num_dcs], dim=1)
                 for k in range(NUM_MODEL_CLASSES)]
        parts.append(torch.sigmoid(raw[:, NUM_MODEL_CLASSES * self.num_dcs:]))
        return torch.cat(parts, dim=1)

    def _split_state(self, state: torch.Tensor):
        """Split augmented state into DC features and agent identity (one-hot metric vector)."""
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
        # state_dim includes raw DC features + appended 4-dim agent identity (one-hot)
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

    def _augment(self, state: np.ndarray) -> np.ndarray:
        """Append agent identity (one-hot metric vector) to state so the policy is conditioned on its role."""
        return np.concatenate([np.asarray(state, dtype=np.float32).flatten(), self.weights])

    def _heuristic_action(self, state: np.ndarray) -> np.ndarray:
        """
        Compute a greedy DC preference action from observable state features.
        State shape: (num_dcs, 4) = [carbon_norm, cost_norm, water_norm, req_intensity]

        Includes an agent-deterministic DC bias that breaks symmetry when
        datacenters have identical features.  The bias is derived from a hash
        of the weight vector, so each metric agent gets a unique preferred DC
        ordering.  When DCs ARE heterogeneous the real feature scores dominate.
        """
        dc_state = np.asarray(state, dtype=np.float32).reshape(self.num_dcs, -1)[:, :4]
        w = self.weights  # [w_ttft, w_carbon, w_water, w_cost]
        n = self.num_dcs

        # Feature-based scoring
        dc_score = -(w[1] * dc_state[:, 0] +   # carbon intensity
                     w[3] * dc_state[:, 1] +   # electricity price
                     w[2] * dc_state[:, 2])     # PUE / water

        load_col = dc_state[:, 3]
        resource_tiebreak = -(dc_state[:, 0] + dc_state[:, 1] + dc_state[:, 2]) / 3.0
        dc_score += w[0] * ((1.0 - load_col) + resource_tiebreak * 0.5)

        # Agent-deterministic symmetry breaker
        primary_idx = int(np.argmax(w))
        concentration = float(np.max(w))
        feature_range = float(dc_score.max() - dc_score.min())
        bias_scale = max(feature_range * 0.5, 0.3 * concentration)

        home_dc = primary_idx % n
        rng = np.random.RandomState(
            int(abs(hash(tuple(np.round(w, 4).tolist())))) % (2**31))
        perm = rng.permutation(n)
        rank = np.zeros(n)
        rank[home_dc] = 0
        other_idx = [j for j in perm if j != home_dc]
        for r, j in enumerate(other_idx, start=1):
            rank[j] = r
        dc_bias = bias_scale * (1.0 - rank / max(n - 1, 1))
        dc_score += dc_bias

        # Temperature & softmax
        temp_scale = 1.0 - w[0] * 0.95
        temperature = (8.0 + 22.0 * concentration) * temp_scale
        exp_scores = np.exp((dc_score - dc_score.max()) * temperature)
        routing_pref = exp_scores / (exp_scores.sum() + 1e-8)

        # Power follows routing
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
        if ABLATION_MODE == "no-heuristic":
            blend = 0.0  # Ablation: pure SAC, no heuristic warm-start
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

    def select_action(self, state, exploration: bool = True) -> np.ndarray:
        s_t = torch.FloatTensor(self._augment(state)).unsqueeze(0)
        with torch.no_grad():
            if exploration:
                action, _ = self.actor.sample(s_t)
            else:
                action = self.actor.deterministic_action(s_t)
        raw = action.cpu().numpy()[0]
        # Always blend with heuristic so behavior is differentiated from epoch 0
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

    # Per-DC request count from source_dc column — gives the TTFT agent a
    # differentiated load signal per DC rather than one shared global value.
    dc_request_counts = np.zeros(n_dcs, dtype=np.float32)
    if epoch_data is not None and len(epoch_data) > 0:
        src_col = ("source_dc" if "source_dc" in epoch_data.columns
                   else "source_dc_id" if "source_dc_id" in epoch_data.columns
                   else None)
        if src_col:
            for i, dc_id in enumerate(dc_ids):
                dc_request_counts[i] = float((epoch_data[src_col] == dc_id).sum())
    total_reqs = max(float(dc_request_counts.sum()), 1.0)

    for i, dc_id in enumerate(dc_ids):
        ci, cost, true_water_intensity = 400.0, 0.10, 1.0

        if hasattr(sim, 'datacenters') and dc_id in sim.datacenters:
            dc = sim.datacenters[dc_id]
            ci = float(getattr(dc, 'carbon_intensity_g_per_kwh', 400.0))

            try:
                tou = getattr(dc, 'tou_price', None)
                # Handle numpy arrays, lists, tuples (all indexable with len),
                # and scalar fallback.  The old isinstance(tou, (list, tuple))
                # check silently failed for numpy arrays — the most common type
                # in simulation code — falling back to tou[0] (midnight price)
                # every epoch regardless of the actual hour.
                if tou is not None and hasattr(tou, '__len__') and len(tou) == 24:
                    cost = float(tou[hour])
                elif tou is not None:
                    cost = float(tou) if np.isscalar(tou) else float(np.asarray(tou).flat[0])
            except Exception:
                pass

            # Extract true water intensity based on the simulator's physical math
            static_factor = float(getattr(dc, 'water_static', 0.0))
            evap_factor = float(getattr(dc, 'water_cycling_density', 0.0))
            blowdown = max(1e-9, float(getattr(dc, 'blowdown_ratio', 0.30)))

            # Total m3 of water drawn per kWh of heat rejected
            true_water_intensity = static_factor + (evap_factor / blowdown)

            # Effective cost rate = tou_price × PUE.
            # Without PUE, MinCost routes to the cheapest $/kWh DC even if its
            # PUE is so high that actual energy cost (tokens × energy × PUE × price)
            # is more expensive than a slightly pricier DC with lower PUE.
            pue  = float(getattr(dc, 'pue', getattr(dc, 'power_usage_effectiveness', 1.0)))
            cost = cost * max(pue, 1.0)   # multiply in-place; floor PUE at 1.0

        # Per-DC load: fraction of total epoch requests originating from this DC,
        # normalised so a DC with 50% of traffic scores ~1.0.
        per_dc_load = min(dc_request_counts[i] / max(total_reqs * 0.5, 1.0), 1.0)
        state[i] = [ci / 1000.0, cost * 5.0, true_water_intensity / 10.0, per_dc_load]

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
    """
    Map each request to a datacenter.

    Routing priority:
    1. If the request's source DC is powered on and has capacity → route there
       (eliminates network transmission latency, the dominant TTFT component).
    2. Overflow requests are distributed across powered DCs using agent weights.

    This locality-first policy is applied uniformly across all schemes — agents
    control which DCs are powered and their relative weights for overflow, but
    cannot override the physical constraint that local routing is faster.
    """
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

    # Build source_dc lookup: row index → DC index (or -1 if unknown/off)
    dc_id_to_idx = {int(d): i for i, d in enumerate(dc_ids)}
    source_dcs = sim_data["source_dc"].values if "source_dc" in sim_data.columns else None

    def allocate(req_idx, pref_w, bucket):
        if not req_idx:
            return {}

        n = len(req_idx)
        remaining_cap = cap.copy()
        alloc = {}
        overflow_idx = []

        # ── Phase A: Route to source DC if powered on and has capacity ────
        if source_dcs is not None and ABLATION_MODE != "no-source-routing":
            for r in req_idx:
                src_dc = int(source_dcs[r])
                di = dc_id_to_idx.get(src_dc, -1)
                if di >= 0 and active[di] > 0 and remaining_cap[di] > 0:
                    alloc[int(r)] = int(dc_ids[di])
                    remaining_cap[di] -= 1
                else:
                    overflow_idx.append(r)
        else:
            overflow_idx = list(req_idx)

        # ── Phase B: Distribute overflow by agent weights ─────────────────
        if not overflow_idx:
            return alloc

        n_ov = len(overflow_idx)
        ew   = np.asarray(pref_w, np.float64) * active.astype(np.float64)
        # Zero out DCs with no remaining capacity
        ew   = ew * (remaining_cap > 0).astype(np.float64)
        tot  = ew.sum()
        if tot <= 0.0:
            ew  = (remaining_cap > 0).astype(np.float64)
            tot = ew.sum()
            if tot <= 0.0:
                return alloc  # All DCs full
        ew /= tot

        # Route overflow toward DCs with most remaining capacity to minimise
        # queuing latency.  Agent weights bias the initial distribution but
        # remaining_cap re-weights so requests flow to least-loaded DCs first.
        ew_latency = remaining_cap.copy()
        ew_latency[active == 0] = 0.0
        lt = ew_latency.sum()
        if lt > 0:
            ew_latency /= lt
            raw    = ew_latency * n_ov
            counts = np.floor(raw).astype(np.int64)
            rem    = n_ov - int(counts.sum())
            counts[np.argsort(raw - counts)[::-1][:rem]] += 1
            counts -= np.maximum(0, counts - remaining_cap.astype(np.int64))

        ordered = sorted(overflow_idx,
                         key=lambda r: _stable_hash_int(f"{epoch_idx}:{bucket}:{r}"))
        ptr = 0
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

    # Boost top-routed DC power with headroom: routing 80% of traffic needs
    # more than 80% power to avoid becoming a queuing bottleneck.
    top_dc = int(np.argmax(route_share))
    ps[top_dc] = max(float(ps[top_dc]), min(float(route_share[top_dc]) * 1.4, 1.0))

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
    # Raise floor to 0.70 so the top DC has enough active nodes to avoid queuing
    sl[best_dc] = max(float(sl[best_dc]), 0.70)
    # If nothing is powered on at all, turn on the preferred DC fully
    if float(sl.max()) < (1.0 / (NUM_NODE_TYPES + 0.99) + 1e-6):
        sl[best_dc] = 1.0
    return sl



def _score_solution(metrics: dict, power_sliders, dc_usage: dict,
                    dc_to_idx: dict, weights, normalizer: MetricNormalizer,
                    update_norm: bool = True) -> float:
    """
    Score a simulation outcome using the original reward structure (Eq. 7):
        r = EMA + Eco + metricSAC − penalty

    Since agents now have one-hot identity weights (e.g. Carbon = [0,1,0,0]),
    the weighted sum naturally selects only that agent's metric.  The full
    reward structure is preserved so the code matches the paper exactly.

    Components:
    - EMA (dominance_bonus):  quadratic bonus for beating the agent's primary
      metric EMA baseline.
    - Eco (eco_bonus):  consolidation reward, weighted by eco-relevance
      (1 − w_time), so the TTFT agent gets zero eco bonus.
    - metricSAC (wm):  inverse-variance weighted metric penalty.
    - penalty (sla_penalty):  punishment for dropping requests.

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
    # metrics (cost) get reduced weight.  With one-hot weights only the
    # non-zero entry survives, so this is effectively a no-op for pure
    # metric agents — but the structure is preserved for paper fidelity.
    stds = normalizer.ratio_stds()
    raw_ew = [w / s for w, s in zip(weights, stds)]
    ew_sum = sum(raw_ew) + 1e-8
    eff_weights = [e / ew_sum for e in raw_ew]

    # Weighted metric penalty (metricSAC in Eq. 7)
    wm = sum(ew * r for ew, r in zip(eff_weights, raw_ratios))

    # ── Dominance bonus (EMA in Eq. 7): primary metric uses RAW ratio ─────
    # (raw is more interpretable: "below EMA" means genuine improvement)
    primary_idx   = int(np.argmax(weights))
    primary_ratio = raw_ratios[primary_idx]
    dominance_coeff = 4.5 if primary_idx == 0 else 3.0  # TTFT agent (idx 0): 3.0 → 4.5
    dominance_bonus = max(0.0, 1.0 - primary_ratio) ** 2 * weights[primary_idx] * dominance_coeff

    # ── Service rate ──────────────────────────────────────────────────────
    req_done  = float(metrics.get("requests_completed", metrics.get("served_requests", 0.0)))
    req_drop  = float(metrics.get("requests_dropped", 0.0))
    req_tot   = max(0.0, req_done + req_drop)
    sr        = (req_done / req_tot) if req_tot > 0.0 else 0.0

    # ── Eco bonus (Eco in Eq. 7): weighted by eco-relevance ───────────────
    # Powering off DCs reduces carbon, water, and cost — but NOT latency.
    # TTFT agent (w_time=1.0) gets zero eco bonus.
    # Carbon/Water/Cost agents (w_time=0.0) get full eco bonus.
    eco_relevance = 1.0 - float(weights[0])   # 0 for TTFT, 1 for sustainability agents

    active       = _active_nodes_per_dc(power_sliders)
    pof          = 1.0 - float(active.sum()) / max(len(power_sliders) * NUM_NODE_TYPES, 1)
    dof          = int(np.sum(active == 0)) / max(len(power_sliders), 1)
    eco_bonus    = (math.sqrt(max(pof, 0.0)) * sr * 0.75 + dof * sr * 1.0) * ECO_BONUS_SCALE * eco_relevance

    # ── SLA penalty (penalty in Eq. 7) ────────────────────────────────────
    sla_penalty  = (1.0 - sr) * 1.5 if req_tot > 0.0 else 0.0

    effective_scale = METRIC_REWARD_SCALE * (1.5 if primary_idx == 0 else 1.0)  # TTFT: 1.5× penalty
    return dominance_bonus + eco_bonus - wm * effective_scale - sla_penalty


# ─────────────────────────────────────────────────────────────────────────────
# OFFLINE TRAINING  — trains agents only (no Phase 2, no capital, no schemes)
# ─────────────────────────────────────────────────────────────────────────────
def offline_train_epoch(epoch_data, epoch_idx: int, node_properties: dict, epoch_summary: dict):
    """
    Offline training: each agent explores via simulation, fills replay buffers,
    and trains its actor/critic networks.  No game-theoretic negotiation (Phase 2),
    no capital evolution, no scheme execution.

    This builds the agents' individual policies so they produce good proposals
    when milp_optimizer is called later for inference.
    """
    global _GLOBAL_AGENTS, _PREV_STATES, _GLOBAL_NORMALIZER

    spec_dir  = epoch_summary.get('spec_dir',      'sim_specs')
    epoch_len = int(epoch_summary.get('epoch_length', 900))

    # ── DC discovery ─────────────────────────────────────────────────────
    # Use epoch_summary["datacenters"] as authoritative when provided (set
    # by the simulator based on --num-dcs).  Only fall back to probing if
    # not set.  This prevents the simulator's spec files (which define all
    # 12 DCs) from overriding the configured DC count.
    configured_dcs = epoch_summary.get('datacenters', None)
    if configured_dcs and len(configured_dcs) > 0:
        dc_ids = sorted(int(d) for d in configured_dcs)
    else:
        dc_id_set = set()
        if node_properties:
            dc_id_set.update(int(d) for d in node_properties.keys())
        try:
            _probe_sim = Rate_Flow_Sim.LLM_Simulator(spec_dir=spec_dir, epoch_length=epoch_len, debug=False)
            if hasattr(_probe_sim, 'datacenters') and _probe_sim.datacenters:
                dc_id_set.update(int(d) for d in _probe_sim.datacenters.keys())
            del _probe_sim
        except Exception:
            pass
        try:
            _edf = epoch_data if isinstance(epoch_data, pd.DataFrame) else pd.DataFrame(epoch_data)
            _src_col = "source_dc_id" if "source_dc_id" in _edf.columns else "source_dc"
            if _src_col in _edf.columns:
                dc_id_set.update(int(v) for v in pd.to_numeric(_edf[_src_col], errors="coerce").dropna().unique())
        except Exception:
            pass
        if not dc_id_set:
            dc_id_set.add(0)
        dc_ids = sorted(dc_id_set)
    real_num_dcs = len(dc_ids)
    dc_to_idx    = {int(d): i for i, d in enumerate(dc_ids)}

    # ── Agent initialisation ──────────────────────────────────────────────
    # Only reinitialise if no agents exist or metric set changed.
    # Do NOT reinitialise on DC count mismatch — transfer_agents may have
    # created agents with a different DC count that will be adapted via
    # offline training.  The sim will handle the DC mapping.
    agents_stale = (not _GLOBAL_AGENTS
                    or set(_GLOBAL_AGENTS.keys()) != set(METRIC_AGENTS))
    if agents_stale:
        print(f"[INIT] Booting 4 Metric Agents (SAC): {', '.join(METRIC_AGENTS)}")
        _GLOBAL_AGENTS.clear()
        _PREV_STATES.clear()
        for ag_name in METRIC_AGENTS:
            _GLOBAL_AGENTS[ag_name] = SACAgent(
                real_num_dcs, 4, METRIC_AGENT_IDENTITY[ag_name])
    elif list(_GLOBAL_AGENTS.values())[0].num_dcs != real_num_dcs:
        # Agents exist but DC count differs (e.g. transferred from 12-DC)
        # Reinitialise with correct DC count — transfer was already applied
        print(f"[INIT] Resizing agents: {list(_GLOBAL_AGENTS.values())[0].num_dcs} DCs → "
              f"{real_num_dcs} DCs")
        _PREV_STATES.clear()
        for ag_name in METRIC_AGENTS:
            _GLOBAL_AGENTS[ag_name] = SACAgent(
                real_num_dcs, 4, METRIC_AGENT_IDENTITY[ag_name])

    # ── Data preparation (same as milp_optimizer) ─────────────────────────
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
        return {}

    # ── Exploration: agents share one simulator, run sequentially ───────
    # Memory-critical: autoscaled workloads can be 500K+ rows per epoch.
    # We share one simulator, don't store heavy objects in her_pool, and
    # force GC between agents.
    her_pool: list = []   # Stores only (action, metrics_dict, power_sliders) — no dc_usage
    agent_rewards: dict = {ag: [] for ag in METRIC_AGENTS}
    shared_state = None

    for ag_name in METRIC_AGENTS:
        agent = _GLOBAL_AGENTS[ag_name]
        prev_state = _PREV_STATES.get(ag_name)
        rewards = []
        last_action = last_reward = None

        for sim_i in range(OFFLINE_EXPLORE_SIMS):
            # Fresh simulator each call — prevents internal state accumulation
            # from run_epoch that survives del of returned results.
            sim = Rate_Flow_Sim.LLM_Simulator(spec_dir=spec_dir, epoch_length=epoch_len, debug=False)
            if shared_state is None:
                shared_state = get_rich_state(sim, dc_ids, clean_data, epoch_idx)

            full_action = agent.select_action(shared_state, exploration=True)
            full_action = _enforce_route_power_coherence(full_action, real_num_dcs)
            w_s = _normalize_weights(full_action[:real_num_dcs])
            w_l = _normalize_weights(full_action[real_num_dcs:2 * real_num_dcs])
            ps  = _ensure_feasible_power_sliders(
                full_action[2 * real_num_dcs:], w_s, w_l,
                has_traffic=True, num_requests=len(clean_data))
            pp  = build_power_plan_sliding(dc_ids, ps)
            sp  = build_schedule_map(clean_data, dc_ids, w_s, w_l, ps, epoch_idx)

            metrics, results_list, dc_usage = sim.run_epoch(epoch_idx, clean_data, sp, pp)
            reward = _score_solution(metrics, ps, dc_usage, dc_to_idx,
                                     agent.weights, _GLOBAL_NORMALIZER, update_norm=True)
            del results_list, dc_usage, sp, pp, sim
            gc.collect()

            agent.intra_buffer.push(shared_state, full_action, reward, shared_state, False)
            her_pool.append((full_action.copy(), metrics, ps.copy()))
            rewards.append(reward)
            last_action, last_reward = full_action, reward

        # Cross-epoch transition
        if prev_state is not None and last_action is not None:
            agent.cross_buffer.push(prev_state, last_action, last_reward,
                                    shared_state, False, priority_boost=3.0)

        _PREV_STATES[ag_name] = shared_state.copy()
        agent.epoch_count += 1
        agent_rewards[ag_name] = rewards
        gc.collect()

    del clean_data
    gc.collect()

    # ── HER cross-labeling ────────────────────────────────────────────────
    # _score_solution doesn't actually use dc_usage, so we pass {} safely.
    for a_vec, mets, ps in her_pool:
        for ag_name in METRIC_AGENTS:
            ag = _GLOBAL_AGENTS[ag_name]
            r  = _score_solution(mets, ps, {}, dc_to_idx,
                                 ag.weights, _GLOBAL_NORMALIZER, update_norm=False)
            ag.intra_buffer.push(shared_state, a_vec, r, shared_state, False,
                                 priority_boost=HER_CROSS_PRIORITY)

    del her_pool
    gc.collect()

    # ── Offline gradient steps (adaptive to prevent overfitting) ────────
    # Scale gradient steps by buffer-to-batch ratio: when the buffer is
    # small relative to BATCH_SIZE, fewer steps prevent overfitting.
    # When full, cap at OFFLINE_GRAD_STEPS to avoid repeated passes.
    agent_losses: dict = {ag: [] for ag in METRIC_AGENTS}
    for ag_name in METRIC_AGENTS:
        ag = _GLOBAL_AGENTS[ag_name]
        buf_size = len(ag.intra_buffer) + len(ag.cross_buffer)
        # Ratio: how many unique batches fit in the buffer
        # If buffer=640 and batch=64, ratio=10 → max 10 useful steps
        # If buffer=20000 and batch=64, ratio=312 → cap at OFFLINE_GRAD_STEPS
        buf_ratio = max(1, buf_size // max(BATCH_SIZE, 1))
        n_steps = min(OFFLINE_GRAD_STEPS, buf_ratio)
        for _ in range(n_steps):
            loss = ag.train()
            if loss is not None:
                agent_losses[ag_name].append(loss)

    # ── Return training stats ─────────────────────────────────────────────
    stats = {}
    for ag_name in METRIC_AGENTS:
        rews = agent_rewards[ag_name]
        losses = agent_losses[ag_name]
        ag = _GLOBAL_AGENTS[ag_name]
        stats[ag_name] = {
            "avg_reward":  float(np.mean(rews)) if rews else 0.0,
            "max_reward":  float(np.max(rews)) if rews else 0.0,
            "avg_loss":    float(np.mean(losses)) if losses else 0.0,
            "buffer_size": len(ag.intra_buffer) + len(ag.cross_buffer),
            "alpha":       float(ag.alpha),
        }
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# MAIN OPTIMIZER (inference — Phase 1 online proposal + Phase 2 game-theoretic)
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
    # Always exactly 4 agents, one per metric.  Re-init if DC count changed.
    agents_stale = (not _GLOBAL_AGENTS
                    or list(_GLOBAL_AGENTS.values())[0].num_dcs != real_num_dcs
                    or set(_GLOBAL_AGENTS.keys()) != set(METRIC_AGENTS))
    if agents_stale:
        print(f"[INIT] Booting 4 Metric Agents (SAC): {', '.join(METRIC_AGENTS)}")
        _GLOBAL_AGENTS.clear()
        _PREV_STATES.clear()
        _POLITICAL_CAPITAL = _compute_initial_capital()
        for ag_name in METRIC_AGENTS:
            _GLOBAL_AGENTS[ag_name] = SACAgent(
                real_num_dcs, 4, METRIC_AGENT_IDENTITY[ag_name])


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
        zero_metrics = {s: zero.copy() for s in SCHEMES}
        _EPOCH_HISTORY.append({"epoch": epoch_idx, "metrics": zero_metrics})
        return (zero_metrics,
                {s: [] for s in SCHEMES}, [])

    # ── Phase 1: Simulation-based exploration + proposal generation ─────
    # Each agent runs OPTIM_STEPS exploration simulations (with heuristic
    # blend for warm-start), trains on the results, then produces a
    # deterministic proposal.  This is the key to good cold-start behavior:
    # the heuristic gives sensible routing from epoch 0 while SAC learns.

    # First-epoch diagnostic
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
        print(f"[DIAG] Heuristic actions per metric agent:")
        for ag_name in METRIC_AGENTS:
            ag = _GLOBAL_AGENTS[ag_name]
            ha = ag._heuristic_action(_diag_state)
            ha = _enforce_route_power_coherence(ha, real_num_dcs)
            r_avg = (ha[:real_num_dcs] + ha[real_num_dcs:2*real_num_dcs]) / 2
            pw = ha[2*real_num_dcs:]
            top_dc = int(np.argmax(r_avg))
            on_count = int(np.sum(pw > 0.1))
            print(f"       {ag_name:>8}: top_route=DC{top_dc}({r_avg[top_dc]:.0%})  "
                  f"DCs_on={on_count}")
        del _diag_sim

    def explore_and_propose(ag_name: str):
        """Phase 1: run OPTIM_STEPS sims with heuristic blend, train, propose."""
        agent      = _GLOBAL_AGENTS[ag_name]
        temp_sim   = Rate_Flow_Sim.LLM_Simulator(spec_dir=spec_dir, epoch_length=epoch_len, debug=False)
        curr_state = get_rich_state(temp_sim, dc_ids, clean_data, epoch_idx)
        prev_state = _PREV_STATES.get(ag_name)

        her_log: list = []
        last_action = last_reward = None

        if ABLATION_MODE == "no-exploration":
            # Ablation: skip sims, just do gradient steps on stale buffer
            for _ in range(ONLINE_ADJUST_STEPS):
                agent.train()
        else:
            for _ in range(OPTIM_STEPS):
                # select_action includes heuristic blend (decays with epoch_count)
                full_action = agent.select_action(curr_state, exploration=True)
                full_action = _enforce_route_power_coherence(full_action, real_num_dcs)
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

                # Train on fresh data immediately
                for _ in range(4):
                    agent.train()

        # Cross-epoch transition
        if prev_state is not None and last_action is not None:
            agent.cross_buffer.push(prev_state, last_action, last_reward,
                                    curr_state, False, priority_boost=3.0)
            for _ in range(4):
                agent.train()

        _PREV_STATES[ag_name] = curr_state

        # Final deterministic proposal
        proposal = agent.select_action(curr_state, exploration=False)
        proposal = _enforce_route_power_coherence(proposal, real_num_dcs)
        agent.epoch_count += 1

        return ag_name, curr_state, proposal, her_log

    # Run all 4 metric agents in parallel
    proposals:     dict = {}
    agent_states:  dict = {}
    her_pool:      list = []

    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = [ex.submit(explore_and_propose, ag) for ag in METRIC_AGENTS]
        for f in as_completed(futures):
            ag_name, curr, action, her_log = f.result()
            agent_states[ag_name] = curr
            proposals[ag_name]    = action
            her_pool.extend(her_log)

    # ── HER: Cross-label Phase 1 simulations across all 4 agents ──────
    if ABLATION_MODE != "no-her":
        for s_vec, a_vec, mets, ps, dc_use in her_pool:
            for ag_name in METRIC_AGENTS:
                ag = _GLOBAL_AGENTS[ag_name]
                r  = _score_solution(mets, ps, dc_use, dc_to_idx,
                                     ag.weights, _GLOBAL_NORMALIZER, update_norm=False)
                ag.intra_buffer.push(s_vec, a_vec, r, s_vec, False,
                                     priority_boost=HER_CROSS_PRIORITY)

    # ── Phase 2: Run consensus once per scheme ────────────────────────────
    # Each scheme's weight vector sets the voting power of the 4 agents.
    # Capital modulates voting power: effective_weight = scheme_w * capital.
    all_metrics: dict = {}
    all_results: dict = {}
    _execution_log: list = []  # Collects (state, action, metrics, ps, dc_usage) per scheme

    def _renormalise_action(action):
        """Project action back into valid space after blending."""
        action = action.copy()
        for k in range(NUM_MODEL_CLASSES):
            seg = action[k * real_num_dcs:(k + 1) * real_num_dcs]
            seg_sum = seg.sum()
            if seg_sum > 0:
                action[k * real_num_dcs:(k + 1) * real_num_dcs] = seg / seg_sum
            else:
                action[k * real_num_dcs:(k + 1) * real_num_dcs] = 1.0 / real_num_dcs
        action[NUM_MODEL_CLASSES * real_num_dcs:] = np.clip(
            action[NUM_MODEL_CLASSES * real_num_dcs:], 0.0, 1.0)
        return action

    for scheme in SCHEMES:
        sw = np.array(SCHEME_WEIGHTS[scheme], dtype=np.float64)

        # ── no-phase2 ablation: skip voting/SGD/veto, use dominant agent ──
        if ABLATION_MODE == "no-phase2":
            dominant_ag = METRIC_AGENTS[int(np.argmax(sw))]
            consensus_action = proposals[dominant_ag].copy()
            # Skip straight to execution (no voting, no SGD, no veto)
        else:
            # ── Compute effective weights: scheme_weight × agent_capital ──────
            eff_w = {}
            for i, ag_name in enumerate(METRIC_AGENTS):
                cap = 1.0 if ABLATION_MODE == "no-capital" else _POLITICAL_CAPITAL[ag_name]
                eff_w[ag_name] = sw[i] * cap
            ew_total = sum(eff_w.values()) + 1e-8
            eff_w_norm = {ag: eff_w[ag] / ew_total for ag in METRIC_AGENTS}

            # ── Voting: each agent evaluates all proposals via its critic ─────
            proposal_scores = {ag: 0.0 for ag in METRIC_AGENTS}
            for eval_ag in METRIC_AGENTS:
                evaluator = _GLOBAL_AGENTS[eval_ag]
                q_vals = {}
                s_t = torch.FloatTensor(evaluator._augment(agent_states[eval_ag])).unsqueeze(0)
                for prop_ag in METRIC_AGENTS:
                    a_t = torch.FloatTensor(proposals[prop_ag]).unsqueeze(0)
                    with torch.no_grad():
                        q_vals[prop_ag] = evaluator.critic.q_min(s_t, a_t).item()

                mn, mx = min(q_vals.values()), max(q_vals.values())
                for prop_ag in METRIC_AGENTS:
                    ns = (q_vals[prop_ag] - mn) / (mx - mn + 1e-8)
                    proposal_scores[prop_ag] += (ns ** 2) * eff_w[eval_ag]

            # ── Blend proposals using voting scores ───────────────────────────
            vote_total = sum(proposal_scores.values())
            if vote_total > 0:
                consensus_action = sum(
                    (proposal_scores[ag] / vote_total) * proposals[ag]
                    for ag in METRIC_AGENTS)
            else:
                consensus_action = proposals[METRIC_AGENTS[0]].copy()

            # ── SGD consensus refinement ──────────────────────────────────────
            if ABLATION_MODE != "no-sgd":
                c_t   = torch.FloatTensor(consensus_action).unsqueeze(0).requires_grad_(True)
                g_opt = optim.SGD([c_t], lr=PARLIAMENT_GRAD_LR)

                for _ in range(PARLIAMENT_GRAD_STEPS):
                    g_opt.zero_grad()
                    total_q = sum(
                        eff_w_norm[ag] * _GLOBAL_AGENTS[ag].critic.q_min(
                            torch.FloatTensor(_GLOBAL_AGENTS[ag]._augment(
                                agent_states[ag])).unsqueeze(0), c_t)
                        for ag in METRIC_AGENTS
                    )
                    (-total_q).backward()
                    g_opt.step()
                    with torch.no_grad():
                        for k in range(NUM_MODEL_CLASSES):
                            st, en = k * real_num_dcs, (k + 1) * real_num_dcs
                            c_t.data[:, st:en] = F.softmax(c_t.data[:, st:en], dim=1)
                        c_t.data[:, NUM_MODEL_CLASSES * real_num_dcs:].clamp_(0.0, 1.0)

                consensus_action = c_t.detach().cpu().numpy()[0]

            # ── Veto phase ────────────────────────────────────────────────────
            if ABLATION_MODE != "no-veto":
                for ag_name in METRIC_AGENTS:
                    if _POLITICAL_CAPITAL[ag_name] < VETO_CAPITAL_THRESH:
                        continue
                    ag_idx = METRIC_AGENT_INDEX[ag_name]
                    if sw[ag_idx] < 0.05:
                        continue
                    ag  = _GLOBAL_AGENTS[ag_name]
                    s_t = torch.FloatTensor(ag._augment(agent_states[ag_name])).unsqueeze(0)
                    with torch.no_grad():
                        q_own  = ag.critic.q_min(s_t, torch.FloatTensor(proposals[ag_name]).unsqueeze(0)).item()
                        q_cons = ag.critic.q_min(s_t, torch.FloatTensor(consensus_action).unsqueeze(0)).item()

                    degradation = (q_own - q_cons) / (abs(q_own) + 1e-6)
                    if degradation > VETO_Q_DEGRADATION:
                        veto_str = min(VETO_STRENGTH_CAP,
                                       degradation * _POLITICAL_CAPITAL[ag_name] / 500.0)
                        consensus_action = ((1.0 - veto_str) * consensus_action
                                            + veto_str * proposals[ag_name])
                        consensus_action = _renormalise_action(consensus_action)
                        print(f"  [VETO] {ag_name} agent vetoed in {scheme} "
                              f"(capital={_POLITICAL_CAPITAL[ag_name]:.0f}, "
                              f"degradation={degradation:.2f}, strength={veto_str:.2f})")

        # ── Execute consensus for this scheme ─────────────────────────────
        consensus_action = _enforce_route_power_coherence(consensus_action, real_num_dcs)
        w_p  = _normalize_weights(consensus_action[:real_num_dcs])
        wl_p = _normalize_weights(consensus_action[real_num_dcs:2 * real_num_dcs])
        ps_p = _ensure_feasible_power_sliders(
            consensus_action[2 * real_num_dcs:], w_p, wl_p,
            has_traffic=has_traffic, num_requests=len(clean_data))
        pp_p = build_power_plan_sliding(dc_ids, ps_p)
        sp_p = build_schedule_map(clean_data, dc_ids, w_p, wl_p, ps_p, epoch_idx)

        scheme_sim = Rate_Flow_Sim.LLM_Simulator(spec_dir=spec_dir, epoch_length=epoch_len, debug=False)
        mp, rp, dc_usage = scheme_sim.run_epoch(epoch_idx, clean_data, sp_p, pp_p)
        all_metrics[scheme] = mp
        all_results[scheme] = rp
        # Collect execution data for offline training
        _execution_log.append((
            agent_states[METRIC_AGENTS[0]].copy(),  # state (same for all agents)
            consensus_action.copy(),                  # action taken
            mp,                                       # metrics result
            ps_p.copy(),                              # power sliders
            dc_usage                                  # DC usage
        ))

    # ── Evolve agent capital (performance-based, personal to each agent) ──
    # Each agent's capital is updated based on how well its metric was served
    # across the scheme outcomes.  We normalise each agent's metric against
    # the RANGE OF THAT SAME METRIC across schemes — never compare different
    # metrics against each other (TTFT seconds vs carbon kg is meaningless).
    # No extra simulations needed — we use the existing scheme execution results.
    for ag_name in METRIC_AGENTS:
        mk = METRIC_KEYS[METRIC_AGENT_INDEX[ag_name]]
        # Gather this agent's metric from all scheme outcomes
        metric_vals = [float(all_metrics[s].get(mk, 0.0)) for s in SCHEMES]
        avg_metric = np.mean(metric_vals) if metric_vals else 0.0
        mn_metric  = min(metric_vals) if metric_vals else 0.0
        mx_metric  = max(metric_vals) if metric_vals else 0.0

        # Performance score: how well did the consensus serve THIS metric?
        if mx_metric > mn_metric:
            perf_score = 1.0 - (avg_metric - mn_metric) / (mx_metric - mn_metric)
        else:
            perf_score = 0.5

        # Proposal bonus: use Q-value comparison instead of extra simulation.
        # If the agent's critic values its own proposal higher than the average
        # consensus outcome, the agent earns a bonus.
        ag = _GLOBAL_AGENTS[ag_name]
        s_t = torch.FloatTensor(ag._augment(agent_states[ag_name])).unsqueeze(0)
        with torch.no_grad():
            q_own = ag.critic.q_min(
                s_t, torch.FloatTensor(proposals[ag_name]).unsqueeze(0)).item()
        # Average Q across all executed consensus actions
        q_consensus_vals = []
        for _, a_vec, _, _, _ in _execution_log:
            with torch.no_grad():
                q_c = ag.critic.q_min(
                    s_t, torch.FloatTensor(a_vec).unsqueeze(0)).item()
            q_consensus_vals.append(q_c)
        q_consensus_avg = np.mean(q_consensus_vals) if q_consensus_vals else q_own
        proposal_bonus = max(0.0, (q_own - q_consensus_avg) / (abs(q_own) + 1e-6))

        combined_performance = perf_score + proposal_bonus * 0.5

        _POLITICAL_CAPITAL[ag_name] = max(
            10.0,
            CAPITAL_DECAY * _POLITICAL_CAPITAL[ag_name]
            + (1 - CAPITAL_DECAY) * combined_performance * 250.0)

    # ── Online learning: store execution results in buffers ───────────────
    # Phase 2 execution results feed back into agent replay buffers so that
    # online adjustment improves over time even without offline training.
    curr_state = agent_states[METRIC_AGENTS[0]]

    for s_vec, a_vec, mets, ps, dc_use in _execution_log:
        for ag_name in METRIC_AGENTS:
            ag = _GLOBAL_AGENTS[ag_name]
            r  = _score_solution(mets, ps, dc_use, dc_to_idx,
                                 ag.weights, _GLOBAL_NORMALIZER, update_norm=True)
            ag.intra_buffer.push(s_vec, a_vec, r, s_vec, False)

    # HER: cross-label execution results under each agent's reward
    if ABLATION_MODE != "no-her":
        for s_vec, a_vec, mets, ps, dc_use in _execution_log:
            for ag_name in METRIC_AGENTS:
                ag = _GLOBAL_AGENTS[ag_name]
                r  = _score_solution(mets, ps, dc_use, dc_to_idx,
                                     ag.weights, _GLOBAL_NORMALIZER, update_norm=False)
                ag.intra_buffer.push(s_vec, a_vec, r, s_vec, False,
                                     priority_boost=HER_CROSS_PRIORITY)

    # Cross-epoch transition
    if ABLATION_MODE != "no-dual-buffer":
        for ag_name in METRIC_AGENTS:
            ag = _GLOBAL_AGENTS[ag_name]
            prev_state = _PREV_STATES.get(ag_name)
            if prev_state is not None and _execution_log:
                best_r = None
                for _, a_vec, mets, ps, dc_use in _execution_log:
                    r = _score_solution(mets, ps, dc_use, dc_to_idx,
                                        ag.weights, _GLOBAL_NORMALIZER, update_norm=False)
                    if best_r is None or r > best_r:
                        best_r = r
                        best_a = a_vec
                if best_r is not None:
                    ag.cross_buffer.push(prev_state, best_a, best_r,
                                         curr_state, False, priority_boost=3.0)
    for ag_name in METRIC_AGENTS:
        _PREV_STATES[ag_name] = curr_state

    # Quick online training on fresh data
    for ag_name in METRIC_AGENTS:
        ag = _GLOBAL_AGENTS[ag_name]
        for _ in range(OFFLINE_TRAIN_STEPS):
            ag.train()
        ag.epoch_count += 1

    # ── Print unified epoch comparison table ─────────────────────────────
    _print_epoch_table(epoch_idx, all_metrics)

    # ── Record epoch for summary ──────────────────────────────────────────
    _EPOCH_HISTORY.append({"epoch": epoch_idx, "metrics": dict(all_metrics)})

    return all_metrics, all_results, []


# ─────────────────────────────────────────────────────────────────────────────
# AGENT PERSISTENCE  — save / load trained agents to disk
# ─────────────────────────────────────────────────────────────────────────────
def save_agents(path: str = "gtarl_agents.pt"):
    """
    Save all 4 metric agents, their replay buffers, capital, normalizer,
    and epoch history to a single file.  Call after offline training.
    """
    if not _GLOBAL_AGENTS:
        print("[SAVE] No agents to save.")
        return

    state = {
        "agents": {},
        "capital": dict(_POLITICAL_CAPITAL),
        "prev_states": {k: v.tolist() if hasattr(v, 'tolist') else v
                        for k, v in _PREV_STATES.items()},
        "normalizer": {
            "ttft": _GLOBAL_NORMALIZER.ttft,
            "carbon": _GLOBAL_NORMALIZER.carbon,
            "water": _GLOBAL_NORMALIZER.water,
            "cost": _GLOBAL_NORMALIZER.cost,
            "n_obs": _GLOBAL_NORMALIZER.n_obs,
            "ratio_ema": list(_GLOBAL_NORMALIZER.ratio_ema),
            "ratio_sq_ema": list(_GLOBAL_NORMALIZER.ratio_sq_ema),
            "sla_target": _GLOBAL_NORMALIZER.sla_target,
        },
        "schemes": list(SCHEMES),
        "scheme_weights": dict(SCHEME_WEIGHTS),
        "epoch_count": {ag: _GLOBAL_AGENTS[ag].epoch_count for ag in METRIC_AGENTS},
    }

    for ag_name in METRIC_AGENTS:
        ag = _GLOBAL_AGENTS[ag_name]
        state["agents"][ag_name] = {
            "actor": ag.actor.state_dict(),
            "critic": ag.critic.state_dict(),
            "critic_target": ag.critic_target.state_dict(),
            "log_alpha": ag.log_alpha.detach().clone(),
            "alpha": ag.alpha,
            "num_dcs": ag.num_dcs,
            "weights": ag.weights.tolist(),
        }

    torch.save(state, path)
    total_epochs = sum(state["epoch_count"].values()) // 4
    print(f"[SAVE] Saved 4 agents to '{path}' "
          f"(trained for ~{total_epochs} epochs, "
          f"capital: {', '.join(f'{ag}={_POLITICAL_CAPITAL[ag]:.0f}' for ag in METRIC_AGENTS)})")


def load_agents(path: str = "gtarl_agents.pt", num_dcs: int = None):
    """
    Load trained agents from disk.  If num_dcs differs from the saved agents,
    raises an error (agent architecture is tied to DC count).

    Call before running milp_optimizer to use pre-trained agents.
    """
    global _GLOBAL_AGENTS, _POLITICAL_CAPITAL, _PREV_STATES
    global _GLOBAL_NORMALIZER

    if not os.path.exists(path):
        print(f"[LOAD] File '{path}' not found — starting with fresh agents.")
        return False

    state = torch.load(path, weights_only=False)

    # Restore capital
    _POLITICAL_CAPITAL = state["capital"]

    # Restore normalizer
    ns = state["normalizer"]
    _GLOBAL_NORMALIZER.ttft = ns["ttft"]
    _GLOBAL_NORMALIZER.carbon = ns["carbon"]
    _GLOBAL_NORMALIZER.water = ns["water"]
    _GLOBAL_NORMALIZER.cost = ns["cost"]
    _GLOBAL_NORMALIZER.n_obs = ns["n_obs"]
    _GLOBAL_NORMALIZER.ratio_ema = list(ns["ratio_ema"])
    _GLOBAL_NORMALIZER.ratio_sq_ema = list(ns["ratio_sq_ema"])
    _GLOBAL_NORMALIZER.sla_target = ns["sla_target"]

    # Restore prev_states
    _PREV_STATES = {k: np.array(v, dtype=np.float32) if v is not None else None
                    for k, v in state.get("prev_states", {}).items()}

    # Restore agents
    _GLOBAL_AGENTS.clear()
    for ag_name in METRIC_AGENTS:
        ag_state = state["agents"][ag_name]
        saved_num_dcs = ag_state["num_dcs"]
        if num_dcs is not None and num_dcs != saved_num_dcs:
            raise ValueError(
                f"[LOAD] DC count mismatch: saved agents have {saved_num_dcs} DCs, "
                f"but current config has {num_dcs}. Retrain with matching DC count.")

        ag = SACAgent(saved_num_dcs, 4, METRIC_AGENT_IDENTITY[ag_name])
        ag.actor.load_state_dict(ag_state["actor"])
        ag.critic.load_state_dict(ag_state["critic"])
        ag.critic_target.load_state_dict(ag_state["critic_target"])
        ag.log_alpha = ag_state["log_alpha"]
        ag.alpha = ag_state["alpha"]
        ag.epoch_count = state.get("epoch_count", {}).get(ag_name, 0)
        _GLOBAL_AGENTS[ag_name] = ag

    total_epochs = sum(state.get("epoch_count", {}).values()) // max(1, len(METRIC_AGENTS))
    print(f"[LOAD] Loaded 4 agents from '{path}' "
          f"(trained for ~{total_epochs} epochs, {_GLOBAL_AGENTS[METRIC_AGENTS[0]].num_dcs} DCs, "
          f"capital: {', '.join(f'{ag}={_POLITICAL_CAPITAL[ag]:.0f}' for ag in METRIC_AGENTS)})")
    return True