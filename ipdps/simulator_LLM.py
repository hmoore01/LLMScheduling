import random
import copy
import numpy as np
import math
import time
import pickle
import argparse
import os
import csv
from sklearn.cluster import KMeans
import pandas as pd
import hashlib
from typing import Dict, Any, List, Optional, Callable, Union, Literal, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class DatacenterSurrogate(nn.Module):
    def __init__(self):
        super(DatacenterSurrogate, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)


class CondorMPCAgent:
    def __init__(self, surrogate_model):
        self.model = surrogate_model
        self.model.eval()

    def select_action(self, req_7b, req_70b, num_candidates=1000, alpha=0.5):
        c_logits = np.random.uniform(-5, 5, (num_candidates, 2))
        c_power = np.random.uniform(0.1, 1.0, (num_candidates, 1))
        actions = np.hstack([c_logits, c_power])

        w_tensor = np.array([[req_7b, req_70b]] * num_candidates)
        inputs = np.hstack([w_tensor, actions])
        inputs_t = torch.FloatTensor(inputs)

        with torch.no_grad():
            preds = self.model(inputs_t).numpy()

        costs = (alpha * preds[:, 1]) + ((1 - alpha) * preds[:, 0])
        best_idx = np.argmin(costs)
        best_act = actions[best_idx]

        return {
            "logit_7b": best_act[0],
            "logit_70b": best_act[1],
            "power_scalar": best_act[2]
        }


def train_marl_constrained_profiles(*args, **kwargs):
    print("[MARL TRAIN] Placeholder function.")


def train_condor_profile(epoch_data: pd.DataFrame, node_properties: List[Dict]):
    print("\n=== Starting CONDOR Model-Based Training ===")
    pass


_CONDOR_MODEL_CACHE = None


def get_cached_condor_model(model_path="models/condor_physics_model.pth"):
    global _CONDOR_MODEL_CACHE
    if _CONDOR_MODEL_CACHE is None:
        if not os.path.exists(model_path):
            print(f"[CONDOR] WARNING: {model_path} not found. Using random weights.")
            _CONDOR_MODEL_CACHE = DatacenterSurrogate()
        else:
            surrogate = DatacenterSurrogate()
            surrogate.load_state_dict(torch.load(model_path))
            surrogate.eval()
            _CONDOR_MODEL_CACHE = surrogate
    return _CONDOR_MODEL_CACHE


def condor_optimizer(epoch_data, epoch_idx: int, node_properties: Dict[str, Any], epoch_summary: Dict[str, Any]):
    return {}, {}, {}


def _map_model_to_llama(m: str) -> str:
    """
    Normalize a raw model-type string to a v2 simulator base model name.

    If the string already starts with a recognized v2 base model name it is
    returned unchanged, preserving the richer vocabulary from BurstGPT_process v2.

    Legacy BurstGPT raw names (ChatGPT, GPT-3.5, GPT-4) are mapped to large-class
    models — Llama7b is no longer used since the pipeline focuses on 70B+ models.
    """
    s       = str(m).strip()
    s_lower = s.lower()

    # Already a recognized v2 base model — preserve as-is (covers all 6 sim models)
    _V2_BASES = ("Llama7b", "Llama70b", "Llama2_70B", "Llama31_405B",
                 "Mixtral_8x7B", "DeepSeek_R1")
    if s.startswith(_V2_BASES):
        return s

    # Frontier GPT-4 variants → large frontier tier
    if "gpt-4o" in s_lower or "gpt-4-turbo" in s_lower or "gpt-4-32k" in s_lower:
        return "Llama31_405B"
    if ("gpt-4" in s_lower) or ("gpt4" in s_lower):
        return "Llama70b"

    # GPT-3.5 / ChatGPT — map to large tier (Llama70b) since 7B class is removed
    if ("chatgpt" in s_lower) or ("gpt-3.5" in s_lower) or ("gpt3.5" in s_lower):
        return "Llama70b"

    # Mistral / Mixtral
    if "mixtral" in s_lower or "mistral" in s_lower:
        return "Mixtral_8x7B"

    # DeepSeek
    if "deepseek" in s_lower:
        return "DeepSeek_R1"

    # Explicit large-class size patterns (must come before generic digit checks)
    if "405b" in s_lower:
        return "Llama31_405B"
    if "70b" in s_lower or "llama-2-70b" in s_lower or "llama2-70b" in s_lower:
        return "Llama70b"

    # Explicit small-class size pattern — kept for backward compatibility with
    # traces that still have 7B labels; rare after the pipeline migration
    if "llama-2-7b" in s_lower or "llama2-7b" in s_lower:
        return "Llama7b"

    # Unknown — pass through; the v2 sim will attempt substring fallback
    return str(m)


DEFAULT_POPULATION_WEIGHTS = {0: 0.12, 1: 0.10, 2: 0.08, 3: 0.15, 4: 0.10, 5: 0.05, 6: 0.18, 7: 0.08, 8: 0.06, 9: 0.04,
                              10: 0.02, 11: 0.02}
DEFAULT_TIMEZONE_OFFSETS = {0: -5, 1: -8, 2: -6, 3: 0, 4: 1, 5: 2, 6: 8, 7: 9, 8: 7, 9: -3, 10: 3, 11: 2}
DEFAULT_BASE_POPULATION = DEFAULT_POPULATION_WEIGHTS
AUTOSCALE_MAX_MULTIPLIER = 700000.0
# ── Autoscale load split: row replication x token scaling ─────────────────────
# A total autoscale multiplier m is split into (count_mult, remainder_scale)
# such that count_mult * remainder_scale == m EXACTLY.  Because the product is
# preserved, the split changes only HOW the load is delivered (more rows vs
# fatter requests) — it never changes the achieved utilisation.
#
# AUTOSCALE_TARGET_TOKEN_SCALE is the token multiplier the split aims for: row
# replication is chosen as round(m / target), so the leftover token scale
# (m / count_mult) lands at ~target.  A HIGHER target means FEWER replicated
# rows, which is the whole point — fewer rows means less memory and fewer
# per-request scheduling operations.  At 5x this cuts per-epoch row counts
# roughly 3-4x versus the old row-dominated split (which produced only
# ~1.1-1.6x token scale and millions of rows) with no change to target util.
AUTOSCALE_TARGET_TOKEN_SCALE = 50.0
AUTOSCALE_MAX_COUNT_MULT = 1000000
AUTOSCALE_MAX_DROP_FRAC = 0.05
AUTOSCALE_SEARCH_STEPS = 7
AUTOSCALE_MAX_EXPANDED_ROWS = 250000
# Safety ceiling on token inflation.  The split TARGETS AUTOSCALE_TARGET_TOKEN_
# SCALE (5x); this ceiling only ever trips when the row-replication budget
# (count_cap / --autoscale-max-rows) is exhausted, so token scaling alone would
# otherwise have to carry an unreasonable share of the load.  When it trips the
# run honestly reaches a lower utilisation rather than fabricating mega-prompts.
# Kept comfortably above the 5x target so normal operation never clamps.
AUTOSCALE_MAX_TOKEN_SCALE = 50.0

# Runtime-tunable copy of AUTOSCALE_TARGET_TOKEN_SCALE.  One static value lands
# at very different per-request latencies depending on the trace's token
# distribution, so this is now a mutable target: it is overwritten either by
# the --autoscale-target-token-scale flag (explicit pin, used by the sweep
# runner to keep every framework of a config on the SAME scaling) or by the
# --ttft-calibrate search, which bisects it until the helix baseline's average
# TTFT lands inside the requested band (default 2-5 s).  Changing this value
# only re-balances the (rows x tokens) split — the total multiplier, and hence
# the achieved utilisation, is untouched.
_TARGET_TOKEN_SCALE = AUTOSCALE_TARGET_TOKEN_SCALE


def _set_target_token_scale(value: float) -> float:
    """Clamp + install a new token-scale target for the autoscale split."""
    global _TARGET_TOKEN_SCALE
    _TARGET_TOKEN_SCALE = max(1.0, min(float(value), AUTOSCALE_MAX_TOKEN_SCALE))
    return _TARGET_TOKEN_SCALE


def _even_src_dc(df: pd.DataFrame, num_dcs: int) -> pd.Series:
    out = np.zeros(len(df), dtype=int)
    if "epoch" not in df.columns:
        out = np.arange(len(df)) % max(1, num_dcs)
        return pd.Series(out, index=df.index, dtype=int)
    for ep, idx in df.groupby("epoch").indices.items():
        n = len(idx)
        out[idx] = np.arange(n) % max(1, num_dcs)
    return pd.Series(out, index=df.index, dtype=int)


def _population_weighted_src_dc(df: pd.DataFrame, num_dcs: int, weights: dict = None) -> pd.Series:
    if weights is None: weights = DEFAULT_POPULATION_WEIGHTS
    available_weights = {k: v for k, v in weights.items() if k < num_dcs}
    total = sum(available_weights.values())
    available_weights = {k: v / total for k, v in available_weights.items()} if total > 0 else {i: 1.0 / num_dcs for i
                                                                                                in range(num_dcs)}
    dc_ids = list(available_weights.keys())
    dc_probs = list(available_weights.values())
    out = np.zeros(len(df), dtype=int)
    if "epoch" not in df.columns:
        out = np.random.choice(dc_ids, size=len(df), p=dc_probs)
        return pd.Series(out, index=df.index, dtype=int)
    for ep, idx in df.groupby("epoch").indices.items():
        np.random.seed(int(ep) * 42)
        out[idx] = np.random.choice(dc_ids, size=len(idx), p=dc_probs)
    return pd.Series(out, index=df.index, dtype=int)


def _time_based_src_dc(df: pd.DataFrame, num_dcs: int, timezone_offsets: dict = None, base_population: dict = None,
                       epoch_length_sec: int = 900, simulation_start_hour: int = 0) -> pd.Series:
    return _even_src_dc(df, num_dcs)  # simplified for brevity


def _select_diverse_dcs(num_dcs: int, spec_dir: str = "sim_specs") -> list:
    """
    Select `num_dcs` datacenters from the v2 simulator's full roster, keeping
    the most extreme (best) DC on each metric dimension to maximise diversity.

    Uses Rate_Flow_Sim_v2.  Falls back to range(num_dcs) if the simulator
    cannot be probed.
    """
    try:
        from Rate_Flow_Sim_v2 import LLM_Simulator
        sim = LLM_Simulator(debug=False, spec_dir=spec_dir)
        all_dc_ids = sorted(int(d) for d in sim.datacenters.keys())
        n_total = len(all_dc_ids)
    except Exception:
        return list(range(num_dcs))

    if num_dcs >= n_total:
        return all_dc_ids[:num_dcs]

    # Feature vectors: [carbon_intensity, effective_cost, wue_l_per_kwh]
    features = {}
    for dc_id in all_dc_ids:
        dc = sim.datacenters[dc_id]
        ci = float(getattr(dc, 'carbon_intensity_g_per_kwh', 400.0))

        tou = getattr(dc, 'tou_price', None)
        if tou is not None and hasattr(tou, '__len__') and len(tou) > 0:
            mean_tou = float(np.mean(tou))
        elif tou is not None:
            mean_tou = float(np.asarray(tou).flat[0])
        else:
            mean_tou = 0.10

        # v2 sim exposes pue_value directly
        pue = float(getattr(dc, 'pue_value', 1.18))

        # v2 sim's Water Usage Effectiveness (L/kWh IT) — lower is better.
        # Falls back to a PUE-derived proxy if the attribute is absent.
        wue = float(getattr(dc, 'wue_l_per_kwh', pue * 0.8))

        # Cost = tou * PUE so cooling overhead is baked in
        effective_cost = mean_tou * max(pue, 1.0)

        features[dc_id] = np.array([ci, effective_cost, wue])

    # Must-keep: one DC with the best value on each metric dimension
    selected = set()
    metric_names = ["carbon", "cost", "water"]
    for dim in range(3):
        selected.add(min(all_dc_ids, key=lambda d: features[d][dim]))

    # Farthest-first diversity fill
    feat_matrix  = np.array([features[d] for d in all_dc_ids])
    mins         = feat_matrix.min(axis=0)
    maxs         = feat_matrix.max(axis=0)
    ranges       = np.where(maxs - mins > 1e-9, maxs - mins, 1.0)
    norm_features = {d: (features[d] - mins) / ranges for d in all_dc_ids}

    while len(selected) < num_dcs:
        best_candidate, best_min_dist = None, -1.0
        for d in all_dc_ids:
            if d in selected:
                continue
            min_dist = min(
                float(np.linalg.norm(norm_features[d] - norm_features[s]))
                for s in selected
            )
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_candidate = d
        if best_candidate is None:
            break
        selected.add(best_candidate)

    result = sorted(selected)

    print(f"[DC-SELECT] Chose {len(result)} of {n_total} DCs for max diversity:")
    print(f"  {'DC':>4}  {'Carbon':>8}  {'EffCost':>8}  {'WUE(L/kWh)':>10}  {'Reason'}")
    extremes = {}
    for dim, mname in enumerate(metric_names):
        best = min(result, key=lambda d: features[d][dim])
        extremes[best] = extremes.get(best, [])
        extremes[best].append(f"lowest {mname}")
    for d in result:
        f = features[d]
        reason = ", ".join(extremes.get(d, ["diversity fill"]))
        print(f"  {d:>4}  {f[0]:>8.1f}  {f[1]:>8.4f}  {f[2]:>10.3f}  {reason}")

    del sim
    return result


def _remap_source_dc(df: pd.DataFrame, active_dc_ids: list,
                     distribution: str = "population",
                     weights: dict = None) -> pd.Series:
    """
    Remap the trace's source_dc_id column so all requests originate from
    DCs in `active_dc_ids`.  Requests from DCs not in the active set are
    redistributed using the chosen distribution strategy.

    This is the key function that makes --num-dcs work dynamically without
    re-running trace_process.py.
    """
    num_dcs = len(active_dc_ids)
    dc_set = set(active_dc_ids)

    if "source_dc_id" not in df.columns:
        # No existing assignments — assign fresh
        return _even_src_dc(df, num_dcs)

    existing = pd.to_numeric(df["source_dc_id"], errors="coerce").fillna(0).astype(int)
    trace_dcs = set(existing.unique())

    # If trace already matches the active set exactly, keep it
    if trace_dcs.issubset(dc_set):
        return existing

    # Build a deterministic remap: map old DC IDs to new DC IDs
    # Requests that were on a DC still in the active set keep their assignment.
    # Requests on removed DCs get redistributed to the closest active DC
    # (by index proximity) or round-robin if no proximity metric.
    result = existing.copy()

    # Create a mapping from old DC id -> new DC id
    remap = {}
    for old_dc in sorted(trace_dcs):
        if old_dc in dc_set:
            remap[old_dc] = old_dc
        else:
            # Map to the nearest active DC by ID
            nearest = min(active_dc_ids, key=lambda d: abs(d - old_dc))
            remap[old_dc] = nearest

    result = existing.map(remap).fillna(active_dc_ids[0]).astype(int)

    # Now reassign using population weights within the active set
    if distribution == "population":
        if weights is None:
            weights = DEFAULT_POPULATION_WEIGHTS
        active_weights = {d: weights.get(d, 1.0 / num_dcs) for d in active_dc_ids}
        total = sum(active_weights.values())
        active_weights = {d: v / total for d, v in active_weights.items()}
        dc_ids_list = list(active_weights.keys())
        dc_probs = list(active_weights.values())

        out = np.zeros(len(df), dtype=int)
        if "epoch" in df.columns:
            for ep, idx in df.groupby("epoch").indices.items():
                np.random.seed(int(ep) * 42 + num_dcs)  # seed includes num_dcs for reproducibility
                out[idx] = np.random.choice(dc_ids_list, size=len(idx), p=dc_probs)
        else:
            out = np.random.choice(dc_ids_list, size=len(df), p=dc_probs)
        return pd.Series(out, index=df.index, dtype=int)

    # Even distribution across active DCs
    out = np.zeros(len(df), dtype=int)
    if "epoch" in df.columns:
        for ep, idx in df.groupby("epoch").indices.items():
            n = len(idx)
            out[idx] = np.array([active_dc_ids[i % num_dcs] for i in range(n)])
    else:
        out = np.array([active_dc_ids[i % num_dcs] for i in range(len(df))])
    return pd.Series(out, index=df.index, dtype=int)


def _assign_src_dc(df: pd.DataFrame, num_dcs: int, distribution: str, weights: dict = None,
                   timezone_offsets: dict = None, active_dc_ids: list = None) -> pd.Series:
    """
    Assign source datacenter IDs to each request.

    If active_dc_ids is provided (from _select_diverse_dcs), remaps the trace
    to use only those DCs.  Otherwise falls back to range(num_dcs).
    """
    if active_dc_ids is not None:
        return _remap_source_dc(df, active_dc_ids, distribution=distribution, weights=weights)

    # Legacy path: no intelligent selection, just use 0..num_dcs-1
    if "source_dc_id" in df.columns:
        existing = pd.to_numeric(df["source_dc_id"], errors="coerce").fillna(0).astype(int)
        if existing.max() > 0 and existing.max() < num_dcs:
            return existing
    if distribution == "population":
        return _population_weighted_src_dc(df, num_dcs, weights)
    elif distribution == "time":
        return _time_based_src_dc(df, num_dcs, timezone_offsets, weights)
    else:
        return _even_src_dc(df, num_dcs)


def _ensure_num_tokens(df: pd.DataFrame, default_tokens: int = 400) -> pd.Series:
    if "num_tokens" in df.columns: return pd.to_numeric(df["num_tokens"], errors="coerce").fillna(0).astype(int)
    return pd.Series(default_tokens, index=df.index, dtype=int)


def _derive_arrival_ms(df: pd.DataFrame, epoch_length_s: int = 900) -> pd.Series:
    epoch_max_ms = float(max(1, int(epoch_length_s))) * 1000.0
    if "arrival_ms" in df.columns:
        arr = pd.to_numeric(df["arrival_ms"], errors="coerce")
        if not arr.isna().all():
            return arr.fillna(0.0).clip(lower=0.0, upper=epoch_max_ms).astype(float)

    if "time_index" in df.columns:
        t = pd.to_numeric(df["time_index"], errors="coerce").fillna(0.0)
        # Existing traces typically keep time_index in seconds [0, epoch_length).
        if float(t.max()) <= float(epoch_length_s) + 1e-9:
            arr = t * 1000.0
        else:
            arr = t
        return arr.clip(lower=0.0, upper=epoch_max_ms).astype(float)

    return pd.Series(0.0, index=df.index, dtype=float)


def _apply_prediction_noise(df: pd.DataFrame, noise_level: float, epoch_idx: int) -> pd.DataFrame:
    """
    Return a perturbed copy of epoch_data to simulate workload forecast inaccuracy.

    The framework receives this noisy forecast and makes scheduling decisions based
    on it; the caller retains the original clean epoch_data for any ground-truth
    bookkeeping.  Three axes of noise are applied jointly:

      token_noise   — each request's token count is scaled by
                      clip(1 + noise_level * N(0,1), 0.1, ∞)
      volume_noise  — total request count is scaled by
                      clip(1 + noise_level * N(0,1), 0.1, ∞) via sample/replicate
      origin_noise  — fraction `noise_level` of source_dc_id values are
                      randomly shuffled among the rows (misrouted origins)

    The RNG is seeded from epoch_idx and noise_level for reproducibility.
    """
    # Zero-traffic epochs (epoch index absent from the trace) reach this
    # function as an EMPTY frame — the [ZERO TRAFFIC] path builds one and the
    # rest of the loop handles it fine.  Without this guard the volume-noise
    # step computes target_n = max(1, round(0 * vol)) = 1 and then tries to
    # sample 1 row from a 0-row frame, which raises
    # "ValueError: a must be greater than 0 unless no samples are taken" and
    # killed every noise-sweep run at its first empty epoch (noise=0 runs were
    # unaffected because of the early return below).  An empty forecast has
    # nothing to perturb, so the identity is the only sane result.
    if noise_level <= 0.0 or len(df) == 0:
        return df
    rng = np.random.default_rng(seed=int(epoch_idx) * 997 + int(noise_level * 10_000))
    out = df.copy()

    # ── Token noise ───────────────────────────────────────────────────────────
    if "num_tokens" in out.columns:
        n             = len(out)
        token_factors = np.maximum(0.1, 1.0 + noise_level * rng.standard_normal(n))
        out["num_tokens"] = (
            pd.to_numeric(out["num_tokens"], errors="coerce").fillna(0).to_numpy() * token_factors
        ).round().astype(int).clip(1)

    # ── Volume noise (request count) ─────────────────────────────────────────
    n             = len(out)
    vol_factor    = max(0.1, 1.0 + noise_level * float(rng.standard_normal()))
    target_n      = max(1, int(round(n * vol_factor)))
    if target_n < n:
        out = out.sample(n=target_n, random_state=int(epoch_idx)).reset_index(drop=True)
    elif target_n > n and n > 0:
        extra = out.sample(n=target_n - n, replace=True, random_state=int(epoch_idx) + 1)
        out   = pd.concat([out, extra], ignore_index=True)

    # ── Origin noise (source DC misassignment) ────────────────────────────────
    if "source_dc_id" in out.columns and len(out) > 1:
        n_noisy = max(0, int(len(out) * noise_level))
        if n_noisy > 0:
            noisy_idx        = rng.choice(len(out), size=n_noisy, replace=False)
            dc_vals          = out["source_dc_id"].to_numpy().copy()
            src_pool         = dc_vals[rng.permutation(len(dc_vals))][:n_noisy]
            dc_vals[noisy_idx] = src_pool
            out["source_dc_id"] = dc_vals

    return out


def summarize_epoch_rate(df: pd.DataFrame):
    if len(df) == 0:
        return pd.DataFrame(columns=["source_dc_id", "model_type", "tokens"])
    grp = df.groupby(["source_dc_id", "model_type"], as_index=False)["num_tokens"].sum()
    grp.rename(columns={"num_tokens": "tokens"}, inplace=True)
    return grp


def _split_autoscale_multiplier(multiplier: float, count_cap: Optional[int] = None) -> Tuple[int, float]:
    """Split a total autoscale multiplier into (row-replication, token-scale).

    The split targets a fixed token scale (AUTOSCALE_TARGET_TOKEN_SCALE): row
    replication is chosen as round(m / target) so the leftover token multiplier
    lands at ~target.  Because remainder_scale is then defined as m / count_mult,
    the product count_mult * remainder_scale equals m exactly — the split never
    changes the achieved utilisation, only the row/token mix.  Targeting a
    larger token scale yields fewer replicated rows (less memory, faster sim).

    If row replication is capped (by count_cap) the leftover token scale rises
    above the target to keep the product equal to m.  Only if it would exceed
    AUTOSCALE_MAX_TOKEN_SCALE is it clamped — in which case the effective
    multiplier (count_mult * remainder_scale) is LESS than requested and the run
    honestly reaches a lower utilisation rather than fabricating mega-prompts.
    """
    m = max(0.0, float(multiplier))
    if m <= 1.0:
        return 1, m
    cap = AUTOSCALE_MAX_COUNT_MULT if count_cap is None else max(1, int(count_cap))
    # Pick row replication so the leftover token scale lands at ~the target.
    # remainder_scale = m / count_mult below, so count_mult * remainder_scale
    # == m exactly: utilisation is unaffected, only the row/token split moves.
    count_mult = max(1, int(round(m / _TARGET_TOKEN_SCALE)))
    count_mult = min(count_mult, cap)
    remainder_scale = m / float(count_mult)
    # Cap token inflation.  Realistic LLM requests are hundreds-to-thousands of
    # tokens; multiplying that by 100x produces a workload no real system sees.
    if remainder_scale > AUTOSCALE_MAX_TOKEN_SCALE:
        achieved = count_mult * AUTOSCALE_MAX_TOKEN_SCALE
        print(f"[Auto-Scale] WARNING: token scale clamped "
              f"{remainder_scale:.2f}x -> {AUTOSCALE_MAX_TOKEN_SCALE:.2f}x. "
              f"Requested multiplier {m:.1f}x, achievable only {achieved:.1f}x "
              f"(rows x{count_mult}, tokens x{AUTOSCALE_MAX_TOKEN_SCALE:.2f}). "
              f"Target utilisation may not be reached — raise --autoscale-max-rows "
              f"to allow more row replication if higher load is needed.")
        remainder_scale = AUTOSCALE_MAX_TOKEN_SCALE
    return count_mult, remainder_scale


def _apply_autoscale_multiplier(
        epoch_data: pd.DataFrame,
        multiplier: float,
        epoch_idx: int,
        epoch_length_s: int = 900,
        count_cap: Optional[int] = None,
) -> Tuple[pd.DataFrame, int, float]:
    """Scale an epoch's load by (row replication x token scaling).

    Memory-light builder: the previous implementation held `count_mult`
    separate DataFrame copies in a Python list AND the concatenated result
    simultaneously (~2x peak memory), which OOM-killed runs at large
    count_mult.  This version tiles the underlying numpy arrays once and
    builds the result frame in a single allocation.  The output is
    row-equivalent to the old builder (same rows, same token counts, same
    arrival distribution); row ORDER differs (block-tiled vs interleaved),
    which does not affect the simulation — Rate_Flow_Sim schedules by each
    request's arrival_ms against a node-availability heap, not by row order,
    and post-jitter arrival times are unique.
    """
    out = epoch_data.copy()
    count_mult, remainder_scale = _split_autoscale_multiplier(multiplier, count_cap=count_cap)

    if count_mult > 1:
        n_base = len(out)
        base_arrival_ms = _derive_arrival_ms(out, epoch_length_s=epoch_length_s).to_numpy(copy=True)

        # Tile every column count_mult times in one pass — no per-copy list.
        # np.tile on each column array, then build the frame once.
        tiled = {}
        for col in out.columns:
            tiled[col] = np.tile(out[col].to_numpy(), count_mult)
        out = pd.DataFrame(tiled)

        # dup_idx: which replica each row belongs to.  Block-tiled order means
        # rows [0:n_base] are replica 0, [n_base:2*n_base] replica 1, etc.
        dup_idx = np.repeat(np.arange(count_mult, dtype=float), n_base)
        base_arr = np.tile(base_arrival_ms, count_mult)

        epoch_window_ms = float(max(1, int(epoch_length_s))) * 1000.0
        rng = np.random.default_rng(100000 + int(epoch_idx))
        slot_ms = epoch_window_ms / float(count_mult)
        jitter_ms = rng.uniform(0.0, slot_ms, size=len(out))
        new_arrival_ms = (base_arr + (dup_idx * slot_ms) + jitter_ms) % epoch_window_ms
        out["arrival_ms"] = new_arrival_ms
        if "time_index" in out.columns:
            out["time_index"] = np.floor(new_arrival_ms / 1000.0).astype(int)

    out["num_tokens"] = (
        pd.to_numeric(out["num_tokens"], errors="coerce").fillna(0.0) * max(0.0, remainder_scale)
    ).round().astype(int)
    return out, count_mult, remainder_scale


def _evaluate_autoscale_candidate(
        dry_sim,
        epoch_idx: int,
        base_epoch_data: pd.DataFrame,
        multiplier: float,
        count_cap: Optional[int] = None,
) -> Dict[str, float]:
    scaled_df, count_mult, remainder_scale = _apply_autoscale_multiplier(
        base_epoch_data,
        multiplier,
        epoch_idx,
        epoch_length_s=int(getattr(dry_sim, "epoch_length", 900)),
        count_cap=count_cap,
    )
    dry_df = scaled_df.copy()
    # Rename to the canonical column names the v2 sim accepts.
    # source_dc_id/model_type/num_tokens are also accepted natively by the v2 sim,
    # but renaming keeps this dry-run path consistent with the framework's own
    # data-cleaning step and avoids any ambiguity in the fallback lookup chain.
    dry_df.rename(
        columns={
            "source_dc_id":  "source_dc",
            "model_type":    "model",
            "num_tokens":    "tokens",
            "prompt_tokens": "prefill_tokens",   # informational; sim ignores unknown cols
            "gen_tokens":    "decode_tokens",    # informational; sim ignores unknown cols
        },
        inplace=True,
    )
    dry_stats, dry_details, _ = dry_sim.run_epoch(epoch_idx, dry_df, schedule_plan={}, power_plan={"all": "ON"})

    epoch_ms = float(getattr(dry_sim, "epoch_length", 900)) * 1000.0
    total_capacity_ms = sum([len(dc.units) * epoch_ms for dc in dry_sim.datacenters.values()])
    total_used_ms = sum([float(req.get("exec_ms", 0.0)) for req in dry_details if "exec_ms" in req])
    util = (total_used_ms / total_capacity_ms) if total_capacity_ms > 0 else 0.0

    completed = int(dry_stats.get("requests_completed", 0))
    dropped = int(dry_stats.get("requests_dropped", 0))
    total = max(1, completed + dropped)
    drop_frac = float(dropped) / float(total)
    return {
        "multiplier": float(multiplier),
        "count_mult": int(count_mult),
        "remainder_scale": float(remainder_scale),
        "util": float(util),
        "drop_frac": float(drop_frac),
    }


def _build_global_peak_plan(
        dry_sim,
        grouped_trace,
        number_of_epoch: int,
        target_util: float,
        max_multiplier: float,
        max_rows: int,
        max_drop: float,
        search_steps: int,
) -> Dict[str, Any]:
    candidate_epochs = [int(e) for e in range(int(number_of_epoch)) if int(e) in grouped_trace.groups]
    if not candidate_epochs:
        return {"enabled": False, "reason": "no_traffic_epochs"}

    # ── Peak-epoch selection: token-sum proxy + top-K verification ───────────
    # Dry-running EVERY epoch at 1x just to locate the peak costs ~N_epochs of
    # simulation per run — and it is recomputed identically for every framework
    # and every run of the same (trace, config).  Total token volume is a very
    # strong proxy for baseline utilisation, so rank epochs by token sum
    # (vectorised, free) and dry-run only the top PEAK_PROBE_CANDIDATES to pick
    # the true utilisation peak among them.
    PEAK_PROBE_CANDIDATES = 5
    cand_set = set(candidate_epochs)
    try:
        tok_sums = grouped_trace["num_tokens"].sum()
        ranked = [int(e) for e in tok_sums.sort_values(ascending=False).index
                  if int(e) in cand_set]
    except Exception:
        ranked = []
    probe_epochs = ranked[:PEAK_PROBE_CANDIDATES] if ranked else list(candidate_epochs)
    if len(probe_epochs) < len(candidate_epochs):
        print(f"[Auto-Scale] Peak search: probing top {len(probe_epochs)} "
              f"epochs by token volume (of {len(candidate_epochs)} candidates) "
              f"instead of dry-running all.", flush=True)

    # Row maximum still spans ALL candidate epochs (it sizes the count cap).
    try:
        _sizes = grouped_trace.size()
        max_epoch_rows = int(max(int(_sizes.get(e, 0)) for e in candidate_epochs))
    except Exception:
        max_epoch_rows = max(len(grouped_trace.get_group(e)) for e in candidate_epochs)

    peak_epoch = None
    peak_util = 0.0
    peak_df = None

    for epoch_idx in probe_epochs:
        ep_df = grouped_trace.get_group(epoch_idx).copy()
        base_eval = _evaluate_autoscale_candidate(dry_sim, epoch_idx, ep_df, 1.0, count_cap=1)
        if float(base_eval.get("util", 0.0)) >= peak_util:
            peak_util = float(base_eval.get("util", 0.0))
            peak_epoch = int(epoch_idx)
            peak_df = ep_df

    if peak_df is None or peak_util <= 0.0:
        return {"enabled": False, "reason": "zero_baseline_util"}

    raw_multiplier = float(target_util) / max(peak_util, 1e-9)
    desired_multiplier = min(float(max_multiplier), float(raw_multiplier))

    count_cap_global = max(1, int(max_rows // max(1, int(max_epoch_rows))))

    # ── Memory-safe util estimate via linear extrapolation ───────────────────
    # The peak epoch needs count_mult in the thousands to reach target_util.
    # Building that full ~14M-row frame just to MEASURE util OOM-kills the run
    # (the autoscaler co-resides with the framework agent in the same process).
    # STEP-3 diagnostics showed util scales ~linearly with the load multiplier
    # in the regime where count_mult is free to grow, so: measure util at a
    # small, cheap multiplier (PROBE_MULT below -> ~PROBE_MULT x base rows,
    # well within memory), then extrapolate linearly.  The REAL epoch loop
    # still builds the true full frame once with the chosen multiplier.
    PROBE_MULT = 100.0   # ~100x base epoch rows for the probe — cheap, safe
    probe_eval = _evaluate_autoscale_candidate(
        dry_sim, int(peak_epoch), peak_df, PROBE_MULT, count_cap=count_cap_global
    )
    probe_util = float(probe_eval.get("util", 0.0))
    probe_drop = float(probe_eval.get("drop_frac", 0.0))
    # util-per-unit-multiplier from the probe; fall back to the 1x baseline
    # ratio if the probe somehow returned zero.
    if probe_util > 0.0:
        util_per_mult = probe_util / PROBE_MULT
    else:
        util_per_mult = peak_util  # baseline ratio (util at multiplier 1)
    # Multiplier predicted to hit target_util under the linear model.
    extrapolated_multiplier = float(target_util) / max(util_per_mult, 1e-12)
    desired_multiplier = min(float(max_multiplier), float(extrapolated_multiplier))

    # Predict util/drop at the chosen multiplier WITHOUT building the full
    # frame: util extrapolates linearly; drop is taken from the probe (a
    # conservative proxy — if the small probe already drops, the full run will
    # too, and the drop-search below will pull the multiplier back).
    predicted_util = min(1.0, util_per_mult * desired_multiplier)
    high_eval = {
        "multiplier": desired_multiplier,
        "count_mult": _split_autoscale_multiplier(desired_multiplier,
                                                  count_cap=count_cap_global)[0],
        "remainder_scale": _split_autoscale_multiplier(desired_multiplier,
                                                       count_cap=count_cap_global)[1],
        "util": predicted_util,
        "drop_frac": probe_drop,
    }
    chosen_eval = high_eval
    drop_limited = False

    # If the probe already shows drops above the limit, the workload is
    # over-subscribed even at PROBE_MULT — binary-search DOWN for a multiplier
    # whose probe-scaled drop is acceptable.  Each search eval uses the small
    # PROBE_MULT-scaled frame, so the search itself stays memory-safe.
    if probe_drop > float(max_drop) and desired_multiplier > 1.0:
        low = 1.0
        high = float(desired_multiplier)
        best = high_eval
        for _ in range(max(1, int(search_steps))):
            mid = (low + high) / 2.0
            # Probe at a multiplier proportional to mid but capped small for
            # memory: scale the probe to mid only if mid is itself small,
            # otherwise probe at PROBE_MULT and extrapolate the drop estimate.
            probe_at = min(mid, PROBE_MULT)
            mid_probe = _evaluate_autoscale_candidate(
                dry_sim, int(peak_epoch), peak_df, probe_at, count_cap=count_cap_global
            )
            mid_drop = float(mid_probe.get("drop_frac", 1.0))
            if mid_drop <= float(max_drop):
                best = {
                    "multiplier": mid,
                    "count_mult": _split_autoscale_multiplier(mid, count_cap=count_cap_global)[0],
                    "remainder_scale": _split_autoscale_multiplier(mid, count_cap=count_cap_global)[1],
                    "util": min(1.0, util_per_mult * mid),
                    "drop_frac": mid_drop,
                }
                low = mid
            else:
                high = mid
        chosen_eval = best
        drop_limited = True

    return {
        "enabled": True,
        "peak_epoch": int(peak_epoch),
        "peak_util_baseline": float(peak_util),
        "raw_multiplier": float(raw_multiplier),
        "extrapolated_multiplier": float(extrapolated_multiplier),
        "probe_util": float(probe_util),
        "desired_multiplier": float(desired_multiplier),
        "chosen_multiplier": float(chosen_eval.get("multiplier", desired_multiplier)),
        "chosen_count_mult": int(chosen_eval.get("count_mult", 1)),
        "chosen_remainder_scale": float(chosen_eval.get("remainder_scale", 1.0)),
        "predicted_peak_util": float(chosen_eval.get("util", 0.0)),
        "predicted_peak_drop": float(chosen_eval.get("drop_frac", 0.0)),
        "count_cap": int(count_cap_global),
        "max_epoch_rows": int(max_epoch_rows),
        "drop_limited": bool(drop_limited),
        "cap_limited": bool(desired_multiplier < raw_multiplier),
    }


# ── TTFT-band token-scale calibration ─────────────────────────────────────────
# Static token scaling produces wildly different per-request latencies across
# traces (BurstGPT vs the Azure traces have very different token distributions).
# These helpers replace the static choice with a measured one: bisect the
# token-scale target until the HELIX baseline's request-weighted average TTFT
# lands inside a band (default 2-5 s).  The total autoscale multiplier — and
# therefore the achieved utilisation — is held fixed throughout; only the
# (rows x tokens) split moves.

def _measure_calib_ttft(FW_calib, grouped_trace, calib_epochs, plan_multiplier,
                        token_scale, calib_count_cap, node_properties,
                        active_dc_ids, spec_dir):
    """Run the calibration framework (helix) over the chosen epochs at the
    candidate token scale; return the request-weighted average TTFT (s).

    The real run replicates rows count_mult = round(m / token_scale) times.
    Building that full frame for every probe would be far too slow, so the
    probe frame caps count_mult at calib_count_cap while holding the TOKEN
    SCALE EXACTLY at the candidate value (eff_mult = capped_count x scale, so
    the split resolves to precisely that token scale).  Per-request service
    time — the part of TTFT the token scale drives — is therefore measured
    faithfully; queueing at the reduced replication is slightly optimistic,
    but the full run prints its true average TTFT so the landing point is
    always verifiable in the results."""
    global _TARGET_TOKEN_SCALE
    prev = _TARGET_TOKEN_SCALE
    _TARGET_TOKEN_SCALE = max(1.0, float(token_scale))
    try:
        ttft_w_sum = 0.0
        weight_sum = 0.0
        plain = []
        for epoch_idx in calib_epochs:
            ep_df = grouped_trace.get_group(int(epoch_idx)).copy()
            cm_full = max(1, int(round(plan_multiplier / max(1e-9, float(token_scale)))))
            cm_cal = min(cm_full, max(1, int(calib_count_cap)))
            eff_mult = float(cm_cal) * float(token_scale)
            scaled, _cm, _rs = _apply_autoscale_multiplier(
                ep_df, eff_mult, int(epoch_idx),
                epoch_length_s=900, count_cap=cm_cal)
            scaled["arrival_ms"] = _derive_arrival_ms(scaled, epoch_length_s=900)
            stats, _results, _leftovers = FW_calib.milp_optimizer(
                epoch_data=scaled,
                epoch_idx=int(epoch_idx),
                node_properties=node_properties,
                epoch_summary={
                    "node_types": [0, 1, 2, 3, 4, 5],
                    "datacenters": active_dc_ids,
                    "avg_input_tokens": 100,
                    "avg_output_tokens": 100,
                    "spec_dir": spec_dir,
                    "epoch_length": 900,
                },
            )
            flat = stats
            if (isinstance(stats, dict) and stats
                    and all(isinstance(v, dict) for v in stats.values())):
                flat = next(iter(stats.values()))
            ttft = float(flat.get("avg_ttft", flat.get("avg_ttft_sec", 0.0)))
            served = float(flat.get(
                "requests_completed",
                flat.get("served_requests", flat.get("requests", 0.0))))
            plain.append(ttft)
            if served > 0.0:
                ttft_w_sum += ttft * served
                weight_sum += served
        if weight_sum > 0.0:
            return ttft_w_sum / weight_sum
        return sum(plain) / max(1, len(plain))
    finally:
        _TARGET_TOKEN_SCALE = prev


def _calibrate_token_scale(FW_calib, grouped_trace, plan, args,
                           node_properties, active_dc_ids) -> Dict[str, Any]:
    """Bisect the token-scale target so helix's avg TTFT lands in the band.

    TTFT is monotonically increasing in the token scale (fatter requests run
    longer), so a bisection on a geometric midpoint converges quickly.  The
    search stops as soon as a probe lands inside [band_lo, band_hi]; if the
    band is unreachable at either end of [1, AUTOSCALE_MAX_TOKEN_SCALE] the
    nearest endpoint is returned with an explicit clamped status."""
    band_lo = float(getattr(args, "ttft_band_low", 2.0))
    band_hi = float(getattr(args, "ttft_band_high", 5.0))
    if band_hi < band_lo:
        band_lo, band_hi = band_hi, band_lo
    steps = max(1, int(getattr(args, "ttft_calib_steps", 6)))
    n_epochs = max(1, int(getattr(args, "ttft_calib_epochs", 2)))
    max_rows = max(1, int(getattr(args, "ttft_calib_max_rows", 250_000)))
    plan_mult = float(plan.get("chosen_multiplier", 1.0))
    spec_dir = getattr(args, "spec_dir", "sim_specs")

    avail = sorted(int(e) for e in grouped_trace.groups
                   if len(grouped_trace.get_group(int(e))) > 0)
    if not avail:
        return {"token_scale": _TARGET_TOKEN_SCALE,
                "measured_ttft": float("nan"), "status": "no-epochs"}

    # Representative epochs: the plan's peak epoch first (the load the
    # multiplier was sized for), then epochs at the median / quartiles of
    # per-epoch request counts so the measured TTFT reflects typical load.
    by_load = sorted(avail, key=lambda e: len(grouped_trace.get_group(e)))
    picks = []
    peak_ep = int(plan.get("peak_epoch", by_load[-1]))
    if peak_ep in avail:
        picks.append(peak_ep)
    for q in (0.50, 0.75, 0.25, 0.90, 0.10):
        if len(picks) >= n_epochs:
            break
        cand = by_load[min(len(by_load) - 1, int(round(q * (len(by_load) - 1))))]
        if cand not in picks:
            picks.append(cand)
    picks = picks[:n_epochs]

    base_rows = max(len(grouped_trace.get_group(e)) for e in picks)
    calib_count_cap = max(1, max_rows // max(1, base_rows))

    def measure(ts):
        return _measure_calib_ttft(
            FW_calib, grouped_trace, picks, plan_mult, ts,
            calib_count_cap, node_properties, active_dc_ids, spec_dir)

    lo_s, hi_s = 1.0, float(AUTOSCALE_MAX_TOKEN_SCALE)
    print(f"[TTFT-Calib] Calibrating token scale for band "
          f"[{band_lo:.2f}, {band_hi:.2f}] s on epochs {picks} "
          f"(probe count_cap={calib_count_cap}, framework=helix). "
          f"Preferring the HIGHEST in-band scale — fewer replicated rows, "
          f"faster runs.", flush=True)

    ttft_lo = measure(lo_s)
    print(f"[TTFT-Calib]   probe scale={lo_s:.3f}x -> avg TTFT {ttft_lo:.3f} s", flush=True)
    if ttft_lo > band_hi:
        # Even unscaled requests exceed the band — nothing lower exists.
        print(f"[TTFT-Calib]   WARNING: avg TTFT at 1.0x already above "
              f"{band_hi:.2f} s; band unreachable, using 1.0x.", flush=True)
        return {"token_scale": lo_s, "measured_ttft": ttft_lo, "status": "clamped-low"}

    ttft_hi = measure(hi_s)
    print(f"[TTFT-Calib]   probe scale={hi_s:.3f}x -> avg TTFT {ttft_hi:.3f} s", flush=True)
    if ttft_hi < band_lo:
        print(f"[TTFT-Calib]   WARNING: avg TTFT at the {hi_s:.0f}x ceiling is "
              f"still below {band_lo:.2f} s; band unreachable, using {hi_s:.0f}x.",
              flush=True)
        return {"token_scale": hi_s, "measured_ttft": ttft_hi, "status": "clamped-high"}
    if band_lo <= ttft_hi <= band_hi:
        # The ceiling itself is in band — it is also the cheapest-to-simulate
        # scale possible (maximum tokens per row, minimum rows).  Take it.
        return {"token_scale": hi_s, "measured_ttft": ttft_hi, "status": "in-band"}

    # Here ttft(lo) < band_hi and ttft(hi) > band_hi: the band edge lies
    # between them.  TTFT is monotonically increasing in the token scale, so
    # bisect (geometric midpoint — the scale axis is multiplicative) for the
    # LARGEST scale whose TTFT stays inside the band.  Among all in-band
    # scales the largest is the cheapest to run: count_mult = m / scale, so
    # landing at 4.5 s instead of 2.2 s roughly halves every epoch's row
    # count for every framework and run of this config.
    aim_hi = band_lo + 0.85 * (band_hi - band_lo)   # stop early near the ceiling
    best = (lo_s, ttft_lo) if band_lo <= ttft_lo <= band_hi else None
    for i in range(steps):
        mid = math.sqrt(lo_s * hi_s)
        ttft_mid = measure(mid)
        print(f"[TTFT-Calib]   step {i + 1}/{steps}: scale={mid:.3f}x -> "
              f"avg TTFT {ttft_mid:.3f} s", flush=True)
        if ttft_mid <= band_hi:
            if ttft_mid >= band_lo and (best is None or mid > best[0]):
                best = (mid, ttft_mid)
            lo_s = mid
            if ttft_mid >= aim_hi:
                break   # within 15% of the ceiling — close enough, stop probing
        else:
            hi_s = mid
    if best is not None:
        return {"token_scale": best[0], "measured_ttft": best[1], "status": "in-band"}
    # No probe landed in band (band narrower than the search resolved):
    # return the endpoint measurement closest to the band midpoint.
    target_mid = 0.5 * (band_lo + band_hi)
    if abs(ttft_lo - target_mid) <= abs(ttft_hi - target_mid):
        return {"token_scale": lo_s, "measured_ttft": ttft_lo, "status": "nearest"}
    return {"token_scale": hi_s, "measured_ttft": ttft_hi, "status": "nearest"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epoch', type=int, default=96)
    parser.add_argument('-f', '--framework', type=str, default='lahyper',
                        help='Framework to run: lahyper, marl, qlearning, condor, ddqn, actorcritic, helix, nsga2, perllm, splitwise, hybrid, parliament')
    parser.add_argument('--freq-scale', type=float, default=1.0)
    parser.add_argument('--token-scale', type=float, default=1.0)
    parser.add_argument('--count-scale', type=int, default=1)
    parser.add_argument('--target-util', type=float, default=0.0,
                        help="Target epoch utilization (e.g. 0.95). Overrides static scaling.")
    parser.add_argument('--autoscale-mode', type=str, default='global_peak', choices=['global_peak', 'per_epoch'],
                        help="Autoscaling strategy: one-time global scaling by peak epoch, or per-epoch scaling.")
    parser.add_argument('--autoscale-max-mult', type=float, default=AUTOSCALE_MAX_MULTIPLIER,
                        help="Upper bound for autoscale multiplier search.")
    parser.add_argument('--autoscale-max-drop', type=float, default=AUTOSCALE_MAX_DROP_FRAC,
                        help="Max tolerated drop fraction in autoscale dry-run search.")
    parser.add_argument('--autoscale-search-steps', type=int, default=AUTOSCALE_SEARCH_STEPS,
                        help="Binary-search steps for autoscale drop-constrained tuning.")
    parser.add_argument('--autoscale-max-rows', type=int, default=AUTOSCALE_MAX_EXPANDED_ROWS,
                        help="Max expanded row count per epoch during autoscaling.")
    parser.add_argument('--num-dcs', type=int, default=12)
    parser.add_argument('--distribution', type=str, default='even')
    parser.add_argument('--spec-dir', type=str, default='sim_specs')
    parser.add_argument('--ql-theta', type=float, default=0.87)
    parser.add_argument('--ql-alpha', type=float, default=0.1)
    parser.add_argument('--ql-gamma', type=float, default=0.9)
    parser.add_argument('--ql-epsilon', type=float, default=0.1)
    parser.add_argument('--offline-train', type=int, default=0,
                        help="Offline training: randomly sample this many epochs from the "
                             "workload and train agents without running inference. "
                             "Saves model to --model-dir when done. 0 = disabled (default).")
    parser.add_argument('--model-dir', type=str, default='models/gtarl',
                        help="Directory for saving/loading GTARL agent checkpoints.")
    parser.add_argument('--load-model', action='store_true',
                        help="Load pre-trained agents from --model-dir before running. "
                             "Agents will be used for inference (online adjustment still applies).")
    parser.add_argument('--transfer-from', type=str, default='',
                        help="Path to a source checkpoint for transfer learning. "
                             "Transfers compatible weights (hidden layers, FiLM, normalizer) "
                             "from a model trained on a different DC count. "
                             "Combine with --offline-train for fine-tuning after transfer.")
    parser.add_argument('--ablation', type=str, default='',
                        choices=['', 'no-film', 'no-veto', 'no-dual-buffer',
                                 'no-her', 'no-capital', 'no-source-routing',
                                 'no-heuristic', 'no-phase2', 'no-sgd',
                                 'no-exploration'],
                        help="Ablation mode: disable one GTARL component for study.")
    parser.add_argument('--prediction-noise', type=float, default=0.0,
                        help="Workload forecast inaccuracy level [0.0–1.0]. "
                             "When > 0, the epoch data passed to the framework is "
                             "perturbed (token counts, request volume, source DC "
                             "assignments) while the structure seen by internal "
                             "simulator calls is unchanged at the OS level. "
                             "Measures how framework performance degrades when "
                             "its workload forecast is inaccurate.")
    parser.add_argument('--autoscale-target-token-scale', type=float, default=0.0,
                        help="Explicit token-scale target for the autoscale split, "
                             "replacing the static AUTOSCALE_TARGET_TOKEN_SCALE. "
                             "The sweep runner passes the helix-calibrated value here "
                             "so every framework of a config uses identical scaling. "
                             "0 = use the static default (or --ttft-calibrate).")
    parser.add_argument('--ttft-calibrate', action='store_true',
                        help="Before the run, bisect the token-scale target until the "
                             "helix baseline's average TTFT lands inside "
                             "[--ttft-band-low, --ttft-band-high] seconds, then run "
                             "with the calibrated value.")
    parser.add_argument('--ttft-calibrate-only', action='store_true',
                        help="Run the TTFT calibration, print the machine-readable "
                             "'[TTFT-Calib] RESULT ...' line, and exit without running "
                             "the framework. Used by the sweep runner to compute one "
                             "shared value per (trace, config).")
    parser.add_argument('--ttft-band-low', type=float, default=2.0,
                        help="Lower edge of the helix avg-TTFT target band (s).")
    parser.add_argument('--ttft-band-high', type=float, default=5.0,
                        help="Upper edge of the helix avg-TTFT target band (s).")
    parser.add_argument('--ttft-calib-epochs', type=int, default=2,
                        help="Representative epochs evaluated per calibration probe.")
    parser.add_argument('--ttft-calib-steps', type=int, default=6,
                        help="Max bisection probes after the two endpoint probes.")
    parser.add_argument('--ttft-calib-max-rows', type=int, default=250000,
                        help="Row budget per calibration probe epoch (keeps probes fast; "
                             "the token scale is still measured at its exact candidate value).")
    parser.add_argument('--autoscale-plan-cache', type=str, default='',
                        help="Path to a JSON cache of global autoscale plans. The plan "
                             "for a (trace, config) is identical across frameworks and "
                             "runs, so caching it skips the dry-run peak search and "
                             "probe on every run after the first.")
    parser.add_argument('--autoscale-plan-key', type=str, default='',
                        help="Cache key identifying this (trace, config) in "
                             "--autoscale-plan-cache. Both flags must be set for "
                             "caching to activate.")
    args = parser.parse_args()

    workload_path = "simulator_ready_trace.csv"
    if not os.path.exists(workload_path):
        raise FileNotFoundError(f"Could not find workload CSV: {workload_path}")
    trace = pd.read_csv(workload_path)

    if "epoch" not in trace.columns: trace["epoch"] = 0
    trace["epoch"] = pd.to_numeric(trace["epoch"], errors="coerce").fillna(0).astype(int)

    if "src_dc" in trace.columns and "source_dc_id" not in trace.columns:
        trace = trace.rename(columns={"src_dc": "source_dc_id"})

    trace["source_dc_id"] = pd.to_numeric(
        trace.get("source_dc_id", pd.Series(0, index=trace.index)),
        errors="coerce").fillna(0).astype(int)

    # ── Intelligent DC selection ──────────────────────────────────────────
    # Probe the simulator to find which DCs exist, then select the --num-dcs
    # most diverse ones (keeping metric extremes).  Remap the trace so all
    # requests originate from the selected subset.
    active_dc_ids = _select_diverse_dcs(args.num_dcs, spec_dir=args.spec_dir)
    trace["source_dc_id"] = _assign_src_dc(
        trace, args.num_dcs,
        distribution=args.distribution,
        active_dc_ids=active_dc_ids)

    if "model_type" not in trace.columns: trace["model_type"] = "Llama70b"
    trace["model_type"] = trace["model_type"].astype(str).map(_map_model_to_llama)
    # Preserve scenario column if present (output by BurstGPT_process v2)
    if "scenario" not in trace.columns:
        trace["scenario"] = "Chat"   # safe default for legacy traces
    trace["num_tokens"] = _ensure_num_tokens(trace, default_tokens=400)
    if "time_index" not in trace.columns: trace["time_index"] = 0
    trace["arrival_ms"] = _derive_arrival_ms(trace, epoch_length_s=900)

    trace["source_dc_id"] = pd.to_numeric(trace["source_dc_id"], errors="coerce").fillna(0).astype(int)
    trace["num_tokens"] = pd.to_numeric(trace["num_tokens"], errors="coerce").fillna(0).astype(int)

    grouped_trace = trace.groupby("epoch")
    max_epoch = int(trace["epoch"].max())
    print(f"[INIT] Loaded workload with {len(trace)} entries across {max_epoch + 1} epochs")

    framework = args.framework
    number_of_epoch = args.epoch
    # Build node_properties keyed by actual active DC IDs so the framework
    # can discover all DCs from this dictionary alone.
    node_properties: dict = {dc_id: {"id": dc_id} for dc_id in active_dc_ids}

    cumulative_ttft = 0.0
    cumulative_carbon = 0.0
    cumulative_water = 0.0
    cumulative_energy = 0.0
    cumulative_total_energy = 0.0
    cumulative_ttft_weighted = 0.0
    cumulative_ttft_weight = 0.0
    epoch_counter = 0

    lahyper_scheme_sums: Dict[str, Dict[str, float]] = {}


    def get_framework(framework_name):
        fw = framework_name.lower()
        if fw == 'lahyper':
            import LA_Hyper_DDQN
            return LA_Hyper_DDQN
        elif fw == 'marl':
            import MARL
            return MARL
        elif fw == 'qlearning':
            import QLearning
            return QLearning
        elif fw == 'condor':
            import CONDOR
            return CONDOR
        elif fw == 'ddqn':
            import DDQN_Consolidator
            return DDQN_Consolidator
        elif fw == 'actorcritic':
            import ActorCritic_Consolidator
            return ActorCritic_Consolidator
        elif fw == 'helix':
            import Helix
            return Helix.Helix
        elif fw == 'nsga2':
            import NSGA2
            return NSGA2.NSGA2
        elif fw == 'perllm':
            import PerLLM
            return PerLLM.PerLLM
        elif fw == 'splitwise':
            import Splitwise
            return Splitwise.Splitwise
        elif fw == 'hybrid':
            import Hybrid_Scheduler_LLM
            return Hybrid_Scheduler_LLM
        elif fw == 'parliament':
            import Game_Theoretic_RL
            return Game_Theoretic_RL
        else:
            raise ValueError(f"Framework '{framework_name}' not supported. Please check your spelling and available module imports.")


    FW = get_framework(framework)

    # Pass ablation mode to GTARL if applicable
    ablation = getattr(args, 'ablation', '')
    if ablation and framework.lower() == "parliament":
        FW.ABLATION_MODE = ablation
        print(f"[ABLATION] Mode: {ablation}")
    elif framework.lower() == "parliament":
        FW.ABLATION_MODE = ""
    autoscale_dry_sim = None
    autoscale_mode = str(getattr(args, "autoscale_mode", "global_peak")).strip().lower()
    global_autoscale_plan: Optional[Dict[str, Any]] = None
    if getattr(args, "target_util", 0.0) > 0.0:
        print(
            f"[Auto-Scale] Config: target={float(args.target_util):.4f}, "
            f"mode={autoscale_mode}, "
            f"max_mult={float(getattr(args, 'autoscale_max_mult', AUTOSCALE_MAX_MULTIPLIER)):.1f}, "
            f"max_drop={float(getattr(args, 'autoscale_max_drop', AUTOSCALE_MAX_DROP_FRAC)):.3f}, "
            f"search_steps={int(getattr(args, 'autoscale_search_steps', AUTOSCALE_SEARCH_STEPS))}, "
            f"max_rows={int(getattr(args, 'autoscale_max_rows', AUTOSCALE_MAX_EXPANDED_ROWS))}"
        )
        from Rate_Flow_Sim_v2 import LLM_Simulator
        autoscale_dry_sim = LLM_Simulator(debug=False, spec_dir=args.spec_dir)
        if autoscale_mode == "global_peak":
            # ── Plan cache ────────────────────────────────────────────────
            # The global plan depends only on (trace, config) — not on the
            # framework or run number — so the dry-run peak search and probe
            # are pure recomputation on every run after the first.  When the
            # runner supplies a cache path + key, reuse a stored plan.
            _plan_cache_path = str(getattr(args, "autoscale_plan_cache", "") or "")
            _plan_cache_key  = str(getattr(args, "autoscale_plan_key", "") or "")
            _plan_from_cache = False
            if _plan_cache_path and _plan_cache_key:
                try:
                    import json as _json
                    with open(_plan_cache_path) as _fh:
                        _plan_data = _json.load(_fh)
                    if _plan_cache_key in _plan_data:
                        global_autoscale_plan = _plan_data[_plan_cache_key]
                        _plan_from_cache = True
                        print(f"[Auto-Scale] Plan cache HIT for '{_plan_cache_key}' "
                              f"({_plan_cache_path}) — skipping dry-run peak search.",
                              flush=True)
                except (OSError, ValueError):
                    pass
            if not _plan_from_cache:
                print(f"[DEBUG] calling _build_global_peak_plan — "
                      f"number_of_epoch={number_of_epoch} "
                      f"grouped_trace type={type(grouped_trace).__name__} "
                      f"n_groups={len(grouped_trace.groups)} "
                      f"key sample={list(grouped_trace.groups)[:5]}", flush=True)
                global_autoscale_plan = _build_global_peak_plan(
                    dry_sim=autoscale_dry_sim,
                    grouped_trace=grouped_trace,
                    number_of_epoch=number_of_epoch,
                    target_util=float(args.target_util),
                    max_multiplier=max(1.0, float(getattr(args, "autoscale_max_mult", AUTOSCALE_MAX_MULTIPLIER))),
                    max_rows=max(1, int(getattr(args, "autoscale_max_rows", AUTOSCALE_MAX_EXPANDED_ROWS))),
                    max_drop=min(1.0, max(0.0, float(getattr(args, "autoscale_max_drop", AUTOSCALE_MAX_DROP_FRAC)))),
                    search_steps=max(1, int(getattr(args, "autoscale_search_steps", AUTOSCALE_SEARCH_STEPS))),
                )
                print(f"[DEBUG] _build_global_peak_plan RETURNED — "
                      f"enabled={global_autoscale_plan.get('enabled')} "
                      f"reason={global_autoscale_plan.get('reason', 'n/a')}", flush=True)
                if _plan_cache_path and _plan_cache_key:
                    try:
                        import json as _json
                        try:
                            with open(_plan_cache_path) as _fh:
                                _plan_data = _json.load(_fh)
                        except (OSError, ValueError):
                            _plan_data = {}
                        _plan_data[_plan_cache_key] = global_autoscale_plan
                        _dirn = os.path.dirname(_plan_cache_path)
                        if _dirn:
                            os.makedirs(_dirn, exist_ok=True)
                        _tmp = _plan_cache_path + ".tmp"
                        with open(_tmp, "w") as _fh:
                            _json.dump(_plan_data, _fh, indent=2, sort_keys=True)
                        os.replace(_tmp, _plan_cache_path)
                        print(f"[Auto-Scale] Plan cached as '{_plan_cache_key}' "
                              f"in {_plan_cache_path}.", flush=True)
                    except Exception as _exc:
                        print(f"[Auto-Scale] (plan cache write failed: {_exc})", flush=True)
            if bool(global_autoscale_plan.get("enabled", False)):
                print(
                    f"[Auto-Scale] Global peak epoch {int(global_autoscale_plan['peak_epoch'])} baseline "
                    f"{float(global_autoscale_plan['peak_util_baseline']) * 100:.6f}%."
                )
                if bool(global_autoscale_plan.get("cap_limited", False)):
                    print(
                        f"[Auto-Scale] Multiplier capped at "
                        f"{float(global_autoscale_plan['desired_multiplier']):.1f}x "
                        f"(raw {float(global_autoscale_plan['raw_multiplier']):.1f}x)."
                    )
                if bool(global_autoscale_plan.get("drop_limited", False)):
                    print(
                        f"[Auto-Scale] Drop-constrained global multiplier selected: "
                        f"{float(global_autoscale_plan['chosen_multiplier']):.2f}x."
                    )
                print(
                    f"[Auto-Scale] Global plan -> multiplier {float(global_autoscale_plan['chosen_multiplier']):.3f}x, "
                    f"Requests x{int(global_autoscale_plan['chosen_count_mult'])}, "
                    f"Tokens x{float(global_autoscale_plan['chosen_remainder_scale']):.3f}, "
                    f"Pred peak util {float(global_autoscale_plan['predicted_peak_util']) * 100:.4f}%, "
                    f"Pred peak drop {float(global_autoscale_plan['predicted_peak_drop']) * 100:.2f}%."
                )
            else:
                print(f"[Auto-Scale] Global plan unavailable: {str(global_autoscale_plan.get('reason', 'unknown'))}.")

    # ── Token-scale selection: explicit flag > TTFT calibration > static ──
    # Order matters: an explicit --autoscale-target-token-scale (what the sweep
    # runner passes after calibrating once with helix) always wins, so every
    # framework of a config runs with byte-identical scaling.  Otherwise, if
    # calibration was requested, bisect against the helix baseline now.
    _explicit_ts = float(getattr(args, "autoscale_target_token_scale", 0.0) or 0.0)
    if _explicit_ts > 0.0:
        _set_target_token_scale(_explicit_ts)
        print(f"[TTFT-Calib] Using explicit token-scale target "
              f"{_TARGET_TOKEN_SCALE:.4f}x (calibration skipped).", flush=True)
    elif getattr(args, "ttft_calibrate", False) or getattr(args, "ttft_calibrate_only", False):
        if (getattr(args, "target_util", 0.0) > 0.0
                and autoscale_mode == "global_peak"
                and global_autoscale_plan
                and bool(global_autoscale_plan.get("enabled", False))):
            FW_calib = get_framework("helix")
            _calib = _calibrate_token_scale(
                FW_calib, grouped_trace, global_autoscale_plan, args,
                node_properties, active_dc_ids)
            _set_target_token_scale(float(_calib["token_scale"]))
            print(f"[TTFT-Calib] RESULT "
                  f"token_scale_target={_TARGET_TOKEN_SCALE:.4f} "
                  f"measured_ttft={float(_calib['measured_ttft']):.4f} "
                  f"band=[{float(args.ttft_band_low):.2f},{float(args.ttft_band_high):.2f}] "
                  f"status={_calib['status']}", flush=True)
        else:
            print(f"[TTFT-Calib] RESULT "
                  f"token_scale_target={_TARGET_TOKEN_SCALE:.4f} "
                  f"measured_ttft=nan "
                  f"band=[{float(getattr(args, 'ttft_band_low', 2.0)):.2f},"
                  f"{float(getattr(args, 'ttft_band_high', 5.0)):.2f}] "
                  f"status=plan-unavailable", flush=True)
    # The plan's informational split was computed under the static target; the
    # per-epoch application re-splits with the live target anyway, but re-derive
    # and reprint here so the log shows what will actually run.
    if (global_autoscale_plan and bool(global_autoscale_plan.get("enabled", False))
            and abs(_TARGET_TOKEN_SCALE - AUTOSCALE_TARGET_TOKEN_SCALE) > 1e-9):
        _cm, _rs = _split_autoscale_multiplier(
            float(global_autoscale_plan["chosen_multiplier"]),
            count_cap=int(global_autoscale_plan.get("count_cap", AUTOSCALE_MAX_COUNT_MULT)))
        global_autoscale_plan["chosen_count_mult"] = int(_cm)
        global_autoscale_plan["chosen_remainder_scale"] = float(_rs)
        print(f"[Auto-Scale] Split re-derived for token-scale target "
              f"{_TARGET_TOKEN_SCALE:.4f}x -> Requests x{_cm}, Tokens x{_rs:.3f} "
              f"(total multiplier, and thus utilisation, unchanged).", flush=True)
    if getattr(args, "ttft_calibrate_only", False):
        print("[DONE]")
        exit(0)

    # ── Model loading (before offline training or inference) ──────────────
    if getattr(args, 'load_model', False) and framework.lower() == "parliament":
        model_path = os.path.join(args.model_dir, "gtarl_agents.pt")
        if os.path.exists(model_path):
            FW.load_agents(model_path, num_dcs=args.num_dcs)
            print(f"[MODEL] Loaded pre-trained agents from {model_path}")
        else:
            print(f"[MODEL] No checkpoint found at {model_path} — starting fresh.")

    # ── Transfer learning (from a model trained on different DC count) ────
    transfer_from = getattr(args, 'transfer_from', '')
    if transfer_from and framework.lower() == "parliament":
        if os.path.exists(transfer_from):
            FW.transfer_agents(transfer_from, target_num_dcs=args.num_dcs)
        else:
            print(f"[TRANSFER] Source checkpoint not found: {transfer_from}")

    # ── Offline training mode ─────────────────────────────────────────────
    # Randomly sample epochs from the workload and train agents offline.
    # No inference results are reported — this is purely for pre-training.
    # After training, agents are saved to --model-dir for later inference.
    offline_train_epochs = int(getattr(args, 'offline_train', 0))
    if offline_train_epochs > 0 and framework.lower() == "parliament":
        # Free the autoscale dry simulator — we only need the plan dict
        if autoscale_dry_sim is not None:
            del autoscale_dry_sim
            autoscale_dry_sim = None
            import gc; gc.collect()

        print(f"[OFFLINE] Baseline memory freed. Starting training...")

        available_epochs = sorted([
            int(e) for e in grouped_trace.groups.keys()
            if len(grouped_trace.get_group(int(e))) > 0
        ])
        if not available_epochs:
            print("[OFFLINE] No non-empty epochs in workload — cannot train.")
        else:
            # Sample with replacement if requesting more epochs than available
            rng = np.random.default_rng(seed=42)
            sampled_epochs = rng.choice(
                available_epochs,
                size=min(offline_train_epochs, len(available_epochs)),
                replace=False
            ).tolist()
            # If user wants more epochs than unique ones, add repeated passes
            if offline_train_epochs > len(available_epochs):
                extra = rng.choice(
                    available_epochs,
                    size=offline_train_epochs - len(available_epochs),
                    replace=True
                ).tolist()
                sampled_epochs += extra
            rng.shuffle(sampled_epochs)

            print(f"[OFFLINE] Training on {len(sampled_epochs)} randomly sampled epochs "
                  f"({len(available_epochs)} unique epochs available)")

            # ── Set up log files ──────────────────────────────────────────────
            os.makedirs(args.model_dir, exist_ok=True)
            log_path = os.path.join(args.model_dir, "training_log.txt")
            csv_path = os.path.join(args.model_dir, "training_metrics.csv")
            _log_file = open(log_path, "w")
            _csv_file = open(csv_path, "w")
            _csv_file.write("step,epoch,elapsed_s,step_time_s,"
                            "ttft_reward,carbon_reward,water_reward,cost_reward,"
                            "ttft_loss,carbon_loss,water_loss,cost_loss,"
                            "ttft_buf,carbon_buf,water_buf,cost_buf,"
                            "ttft_alpha,carbon_alpha,water_alpha,cost_alpha\n")

            def _log(msg: str):
                """Print to console and write to log file."""
                print(msg)
                _log_file.write(msg + "\n")
                _log_file.flush()

            _log(f"[OFFLINE] Training on {len(sampled_epochs)} randomly sampled epochs "
                 f"({len(available_epochs)} unique epochs available)")
            _log(f"[OFFLINE] Log file: {log_path}")
            _log(f"[OFFLINE] CSV metrics: {csv_path}")

            # Rolling averages for convergence detection
            _reward_history = {ag: [] for ag in ["TTFT", "Carbon", "Water", "Cost"]}
            _loss_history   = {ag: [] for ag in ["TTFT", "Carbon", "Water", "Cost"]}
            _window = 20  # Rolling window size for convergence check
            _train_start = time.time()

            for train_step, epoch_idx in enumerate(sampled_epochs):
                step_start = time.time()

                epoch_data = grouped_trace.get_group(epoch_idx).copy()

                # Apply autoscaling if configured
                if getattr(args, "target_util", 0.0) > 0.0:
                    if autoscale_mode == "global_peak" and global_autoscale_plan \
                            and bool(global_autoscale_plan.get("enabled", False)):
                        epoch_data, count_mult, remainder_scale = _apply_autoscale_multiplier(
                            epoch_data,
                            float(global_autoscale_plan.get("chosen_multiplier", 1.0)),
                            epoch_idx,
                            epoch_length_s=int(getattr(autoscale_dry_sim, "epoch_length", 900))
                                if autoscale_dry_sim else 900,
                            count_cap=int(global_autoscale_plan.get("count_cap",
                                          AUTOSCALE_MAX_COUNT_MULT)),
                        )

                epoch_data["arrival_ms"] = _derive_arrival_ms(epoch_data, epoch_length_s=900)

                stats = FW.offline_train_epoch(
                    epoch_data=epoch_data,
                    epoch_idx=train_step,
                    node_properties=node_properties,
                    epoch_summary={
                        "node_types": [0, 1, 2, 3, 4, 5],
                        "datacenters": active_dc_ids,
                        "avg_input_tokens": 100,
                        "avg_output_tokens": 100,
                        "spec_dir": args.spec_dir,
                        "epoch_length": 900,
                    }
                )

                # Free autoscaled epoch data immediately (can be 500K+ rows)
                del epoch_data
                import gc; gc.collect()

                step_time = time.time() - step_start
                elapsed = time.time() - _train_start

                # Track per-agent reward and loss history
                if stats:
                    for ag in _reward_history:
                        if ag in stats:
                            _reward_history[ag].append(stats[ag]["avg_reward"])
                            _loss_history[ag].append(stats[ag]["avg_loss"])

                # Write CSV row every step (lightweight, enables plotting)
                if stats:
                    csv_vals = [
                        str(train_step + 1), str(epoch_idx),
                        f"{elapsed:.1f}", f"{step_time:.1f}",
                    ]
                    for ag in ["TTFT", "Carbon", "Water", "Cost"]:
                        csv_vals.append(f"{stats.get(ag, {}).get('avg_reward', 0):.4f}")
                    for ag in ["TTFT", "Carbon", "Water", "Cost"]:
                        csv_vals.append(f"{stats.get(ag, {}).get('avg_loss', 0):.6f}")
                    for ag in ["TTFT", "Carbon", "Water", "Cost"]:
                        csv_vals.append(str(stats.get(ag, {}).get('buffer_size', 0)))
                    for ag in ["TTFT", "Carbon", "Water", "Cost"]:
                        csv_vals.append(f"{stats.get(ag, {}).get('alpha', 0):.6f}")
                    _csv_file.write(",".join(csv_vals) + "\n")
                    _csv_file.flush()

                # Log every 10 steps or on the last step
                if (train_step + 1) % 10 == 0 or train_step == len(sampled_epochs) - 1:
                    eta = (elapsed / (train_step + 1)) * (len(sampled_epochs) - train_step - 1)

                    # Build per-agent reward summary
                    rew_parts = []
                    for ag in ["TTFT", "Carbon", "Water", "Cost"]:
                        recent = _reward_history[ag][-_window:]
                        if recent:
                            rew_parts.append(f"{ag}={np.mean(recent):+.3f}")
                    rew_str = "  ".join(rew_parts) if rew_parts else "n/a"

                    # Build per-agent loss summary
                    loss_parts = []
                    for ag in ["TTFT", "Carbon", "Water", "Cost"]:
                        recent = _loss_history[ag][-_window:]
                        if recent:
                            loss_parts.append(f"{ag}={np.mean(recent):.4f}")
                    loss_str = "  ".join(loss_parts) if loss_parts else "n/a"

                    # Buffer sizes
                    buf_str = ""
                    if stats:
                        buf_sizes = [f"{ag}={stats[ag]['buffer_size']}" for ag in ["TTFT", "Carbon", "Water", "Cost"] if ag in stats]
                        buf_str = f"  buf=[{', '.join(buf_sizes)}]"

                    _log(f"[OFFLINE] Step {train_step + 1:>4}/{len(sampled_epochs)}  "
                         f"({step_time:.1f}s/step  elapsed={elapsed:.0f}s  eta={eta:.0f}s)  "
                         f"wkld_epoch={epoch_idx}")
                    _log(f"         avg_reward: {rew_str}")
                    _log(f"         avg_loss:   {loss_str}{buf_str}")

                    # Convergence check: compare last window to previous window
                    if train_step + 1 >= 2 * _window:
                        converged_agents = 0
                        for ag in ["TTFT", "Carbon", "Water", "Cost"]:
                            hist = _reward_history[ag]
                            if len(hist) >= 2 * _window:
                                prev_mean = np.mean(hist[-2*_window:-_window])
                                curr_mean = np.mean(hist[-_window:])
                                pct_change = abs(curr_mean - prev_mean) / (abs(prev_mean) + 1e-8) * 100
                                if pct_change < 5.0:
                                    converged_agents += 1
                        if converged_agents == 4:
                            _log(f"[OFFLINE] ★ All 4 agents converged "
                                 f"(<5% reward change over last {_window} steps)")
                        elif converged_agents >= 2:
                            _log(f"[OFFLINE]   {converged_agents}/4 agents converging")

            # Final summary
            total_time = time.time() - _train_start
            _log(f"\n[OFFLINE] ═══ Training Summary ═══")
            _log(f"  Steps: {len(sampled_epochs)}  |  Total time: {total_time:.0f}s  "
                 f"|  Avg: {total_time/max(1,len(sampled_epochs)):.1f}s/step")
            for ag in ["TTFT", "Carbon", "Water", "Cost"]:
                hist = _reward_history[ag]
                if len(hist) >= _window:
                    first_w = np.mean(hist[:_window])
                    last_w  = np.mean(hist[-_window:])
                    delta   = last_w - first_w
                    _log(f"  {ag:>8}: reward {first_w:+.3f} → {last_w:+.3f}  "
                         f"(Δ={delta:+.3f}{'  ✓ improved' if delta > 0 else ''})")
                elif hist:
                    _log(f"  {ag:>8}: reward {np.mean(hist):+.3f} (too few steps for trend)")

            _csv_file.close()
            _log_file.close()

            # Save trained agents
            os.makedirs(args.model_dir, exist_ok=True)
            model_path = os.path.join(args.model_dir, "gtarl_agents.pt")
            FW.save_agents(model_path)
            print(f"[OFFLINE] Training complete. Agents saved to {model_path}")
            print(f"[OFFLINE] Logs saved to {log_path}")
            print(f"[OFFLINE] Metrics CSV saved to {csv_path}")
            print("[OFFLINE] Exiting after offline training. "
                  "Use --load-model to run inference with trained agents.")
            print("[DONE]")
            exit(0)

    print(f"[DEBUG] entering main epoch loop — number_of_epoch={number_of_epoch} "
          f"framework={framework} grouped n_groups={len(grouped_trace.groups)}", flush=True)
    for epoch_idx in range(number_of_epoch):
        print(f"[DEBUG] epoch loop iter epoch_idx={epoch_idx} "
              f"in_groups={epoch_idx in grouped_trace.groups}", flush=True)
        if epoch_idx not in grouped_trace.groups:
            print(f"\n--- Epoch {epoch_idx} ({framework}) [ZERO TRAFFIC] ---")
            epoch_data = pd.DataFrame(columns=trace.columns)
        else:
            epoch_data = grouped_trace.get_group(epoch_idx).copy()

            # --- AUTO-SCALING INJECTION START ---
            if getattr(args, "target_util", 0.0) > 0.0:
                if autoscale_mode == "global_peak":
                    if global_autoscale_plan and bool(global_autoscale_plan.get("enabled", False)):
                        epoch_data, count_mult, remainder_scale = _apply_autoscale_multiplier(
                            epoch_data,
                            float(global_autoscale_plan.get("chosen_multiplier", 1.0)),
                            epoch_idx,
                            epoch_length_s=int(getattr(autoscale_dry_sim, "epoch_length", 900)) if autoscale_dry_sim else 900,
                            count_cap=int(global_autoscale_plan.get("count_cap", AUTOSCALE_MAX_COUNT_MULT)),
                        )
                        print(
                            f"  [Auto-Scale] Global x{float(global_autoscale_plan.get('chosen_multiplier', 1.0)):.3f} -> "
                            f"Requests x{count_mult}, Tokens x{remainder_scale:.3f}"
                        )
                    elif epoch_idx == 0:
                        print("  [Auto-Scale] Global plan unavailable; skipping autoscale for this run.")
                else:
                    print(f"  [Auto-Scale] Dry-running Epoch {epoch_idx} to calculate target multiplier...")
                    dry_sim = autoscale_dry_sim
                    if dry_sim is None:
                        from Rate_Flow_Sim_v2 import LLM_Simulator
                        dry_sim = LLM_Simulator(debug=False, spec_dir=args.spec_dir)
                        autoscale_dry_sim = dry_sim

                    total_nodes = sum([len(dc.units) for dc in dry_sim.datacenters.values()])
                    base_req_count = max(1, len(epoch_data))
                    max_rows = max(1, int(getattr(args, "autoscale_max_rows", AUTOSCALE_MAX_EXPANDED_ROWS)))
                    count_cap_by_rows = max(1, int(max_rows // base_req_count))
                    min_count_for_target = max(1, int(math.ceil((float(args.target_util) * float(total_nodes)) / float(base_req_count))))

                    # ── Size the row-replication cap for the split ───────────
                    # The autoscale split targets AUTOSCALE_TARGET_TOKEN_SCALE for
                    # the token multiplier; row replication supplies m / target.
                    # Do a quick probe to
                    # estimate the multiplier the peak epoch needs, then ensure
                    # count_cap is large enough that replication (not token
                    # scaling) can deliver it.
                    _probe = _evaluate_autoscale_candidate(
                        dry_sim, epoch_idx, epoch_data, 1.0, count_cap=1
                    )
                    _probe_util = float(_probe.get("util", 0.0))
                    if _probe_util > 0.0:
                        _needed_mult = float(args.target_util) / max(_probe_util, 1e-9)
                        # count_mult must reach ~ needed_mult / target_token_scale
                        # so the split (which targets that token scale) is not
                        # starved of row-replication headroom by the count cap.
                        _needed_count = int(math.ceil(_needed_mult / max(1.0, _TARGET_TOKEN_SCALE)))
                    else:
                        _needed_count = AUTOSCALE_MAX_COUNT_MULT
                    # count_cap is the largest of: the row-budget cap, the static
                    # floor, the target-coverage estimate, and the replication
                    # needed for the workload.  The row budget (max_rows) still
                    # bounds memory — if it is too small to reach target_util via
                    # replication, _split_autoscale_multiplier will warn and the
                    # run honestly reaches a lower utilisation.
                    count_cap_dynamic = min(
                        count_cap_by_rows,
                        max(AUTOSCALE_MAX_COUNT_MULT, min_count_for_target, _needed_count),
                    )

                    base_eval = _evaluate_autoscale_candidate(
                        dry_sim, epoch_idx, epoch_data, 1.0, count_cap=count_cap_dynamic
                    )
                    current_util = float(base_eval.get("util", 0.0))
                    if current_util > 0.0:
                        raw_multiplier = float(args.target_util) / max(current_util, 1e-9)
                        max_multiplier = max(1.0, float(getattr(args, "autoscale_max_mult", AUTOSCALE_MAX_MULTIPLIER)))
                        max_drop = min(1.0, max(0.0, float(getattr(args, "autoscale_max_drop", AUTOSCALE_MAX_DROP_FRAC))))
                        search_steps = max(1, int(getattr(args, "autoscale_search_steps", AUTOSCALE_SEARCH_STEPS)))

                        capped_multiplier = min(raw_multiplier, max_multiplier)
                        chosen_eval = base_eval

                        if capped_multiplier > 1.0:
                            high_eval = _evaluate_autoscale_candidate(
                                dry_sim, epoch_idx, epoch_data, capped_multiplier, count_cap=count_cap_dynamic
                            )
                            chosen_eval = high_eval

                            if high_eval["drop_frac"] > max_drop:
                                low = 1.0
                                high = capped_multiplier
                                best = base_eval if base_eval["drop_frac"] <= max_drop else high_eval
                                for _ in range(search_steps):
                                    mid = (low + high) / 2.0
                                    mid_eval = _evaluate_autoscale_candidate(
                                        dry_sim, epoch_idx, epoch_data, mid, count_cap=count_cap_dynamic
                                    )
                                    if mid_eval["drop_frac"] <= max_drop:
                                        best = mid_eval
                                        low = mid
                                    else:
                                        high = mid
                                chosen_eval = best
                                print(
                                    f"  [Auto-Scale] Drop-constrained multiplier selected: "
                                    f"{chosen_eval['multiplier']:.2f}x (drop limit {max_drop * 100:.1f}%)."
                                )

                            if capped_multiplier < raw_multiplier:
                                print(
                                    f"  [Auto-Scale] Multiplier capped at {max_multiplier:.1f}x "
                                    f"(raw {raw_multiplier:.1f}x)."
                                )
                                if chosen_eval["util"] + 1e-9 < float(args.target_util):
                                    print(
                                        f"  [Auto-Scale] Cap-limited: predicted util {chosen_eval['util'] * 100:.3f}% "
                                        f"below target {float(args.target_util) * 100:.1f}%."
                                    )
                        if count_cap_dynamic < min_count_for_target:
                            print(
                                f"  [Auto-Scale] Row-budget-limited: count cap {count_cap_dynamic} < required "
                                f"{min_count_for_target} for {float(args.target_util) * 100:.1f}% fleet occupancy."
                            )

                        epoch_data, count_mult, remainder_scale = _apply_autoscale_multiplier(
                            epoch_data,
                            float(chosen_eval["multiplier"]),
                            epoch_idx,
                            epoch_length_s=int(getattr(dry_sim, "epoch_length", 900)),
                            count_cap=count_cap_dynamic,
                        )
                        print(
                            f"  [Auto-Scale] Baseline: {current_util * 100:.6f}% | "
                            f"Applied -> Requests x{count_mult}, Tokens x{remainder_scale:.3f} | "
                            f"Predicted Util {chosen_eval['util'] * 100:.4f}% | "
                            f"Predicted Drop {chosen_eval['drop_frac'] * 100:.2f}%"
                        )
                    else:
                        print("  [Auto-Scale] Baseline utilization is zero; skipping dynamic scaling for this epoch.")
            else:
                # Fallback to your original static scaling logic
                epoch_data["time_index"] = (epoch_data["time_index"] * args.freq_scale).clip(upper=899).astype(int)
                if args.token_scale != 1.0:
                    epoch_data["num_tokens"] = (epoch_data["num_tokens"] * args.token_scale).round().astype(int)
                if args.count_scale > 1:
                    epoch_data = pd.concat([epoch_data] * args.count_scale, ignore_index=True)
            # --- AUTO-SCALING INJECTION END ---

            epoch_data["arrival_ms"] = _derive_arrival_ms(epoch_data, epoch_length_s=900)
            epoch_summary = summarize_epoch_rate(epoch_data)

        epoch_counter += 1

        # ── Prediction noise: give the framework a perturbed forecast ─────────
        _pred_noise = float(getattr(args, "prediction_noise", 0.0))
        if _pred_noise > 0.0:
            # Safety net: a noise-injection failure on ONE epoch must never
            # abort the whole run (that's how the noise sweep produced blank
            # rows).  Fall back to the clean forecast for that epoch, loudly —
            # a warned epoch is recoverable, a dead run is not.
            try:
                framework_epoch_data = _apply_prediction_noise(epoch_data, _pred_noise, epoch_idx)
            except Exception as _noise_exc:
                print(f"  [PredNoise] WARNING: noise injection failed on epoch "
                      f"{epoch_idx} ({type(_noise_exc).__name__}: {_noise_exc}); "
                      f"using the clean forecast for this epoch.", flush=True)
                framework_epoch_data = epoch_data
            print(f"  [PredNoise] noise={_pred_noise:.2f} | "
                  f"real_reqs={len(epoch_data)} → forecast_reqs={len(framework_epoch_data)}")
        else:
            framework_epoch_data = epoch_data

        stats, results, leftovers = FW.milp_optimizer(
            epoch_data=framework_epoch_data,
            epoch_idx=epoch_idx,
            node_properties=node_properties,
            epoch_summary={
                "node_types": [0, 1, 2, 3, 4, 5],
                "datacenters": active_dc_ids,
                "avg_input_tokens": 100,
                "avg_output_tokens": 100,
                "spec_dir": args.spec_dir,
                "epoch_length": 900,
            }
        )

        # GTARL (parliament) returns dict-of-dicts {scheme_name: metrics_dict}.
        # All other frameworks return a flat metrics dict that may contain nested
        # dicts like 'by_datacenter' — so we must guard on framework name, not
        # just on the presence of nested dict values.
        if framework.lower() == "parliament" and isinstance(stats, dict) and any(isinstance(v, dict) for v in stats.values()):
            # For parliament: use Balanced if available, else first scheme
            if "Balanced" in stats:
                flat_stats = stats["Balanced"]
            else:
                flat_stats = next((v for v in stats.values() if isinstance(v, dict)), stats)
        else:
            flat_stats = stats

        if framework.lower() == "lahyper":
            tracker = getattr(FW, "_PARETO_TRACKER", None)
            if tracker and hasattr(tracker, "epoch_solutions"):
                for sol in tracker.epoch_solutions:
                    mode = sol["mode"]
                    if mode not in lahyper_scheme_sums:
                        lahyper_scheme_sums[mode] = {"ttft_sum": 0.0, "carbon_sum": 0.0, "water_sum": 0.0,
                                                     "energy_sum": 0.0, "total_energy_sum": 0.0, "epochs": 0,
                                                     "ttft_weighted_sum": 0.0, "ttft_weight": 0.0}
                    agg = lahyper_scheme_sums[mode]
                    # TTFT is request-weighted (weight = requests served) to
                    # match simulator_LLM's final_avg_ttft.  ttft_sum/epochs is
                    # kept only as a divide-by-zero fallback for all-zero-served
                    # modes (e.g. Zero_Traffic).
                    _sol_served = float(sol.get("served", 0.0))
                    agg["ttft_sum"]          += float(sol.get("ttft", 0.0))
                    agg["ttft_weighted_sum"] += float(sol.get("ttft", 0.0)) * _sol_served
                    agg["ttft_weight"]       += _sol_served
                    agg["carbon_sum"] += float(sol.get("carbon", 0.0))
                    agg["water_sum"] += float(sol.get("water", 0.0))
                    agg["energy_sum"] += float(sol.get("cost", 0.0))
                    agg["total_energy_sum"] += float(sol.get("total_energy", 0.0))
                    agg["epochs"] += 1

        epoch_avg_ttft = float(flat_stats.get("avg_ttft", flat_stats.get("avg_ttft_sec", 0.0)))
        cumulative_ttft += epoch_avg_ttft
        req_weight = float(
            flat_stats.get(
                "requests_completed",
                flat_stats.get("served_requests", flat_stats.get("requests", 0.0))
            )
        )
        if req_weight > 0.0:
            cumulative_ttft_weighted += epoch_avg_ttft * req_weight
            cumulative_ttft_weight += req_weight

        cumulative_carbon += float(flat_stats.get("carbon_emissions", 0.0)) / 1000.0
        cumulative_water  += float(flat_stats.get("water_usage", 0.0))   # v2 sim: native m³
        cumulative_energy += float(flat_stats.get("energy_cost", 0.0))
        cumulative_total_energy += float(flat_stats.get('total_energy', 0.0))

    final_avg_ttft = (
        cumulative_ttft_weighted / cumulative_ttft_weight
        if cumulative_ttft_weight > 0.0
        else cumulative_ttft / max(1, epoch_counter)
    )

    # ── Parliament per-scheme run summary ─────────────────────────────────
    if framework.lower() == "parliament" and hasattr(FW, "print_run_summary"):
        FW.print_run_summary()
        # Auto-save agents after inference run (enables resume / warm-start)
        os.makedirs(args.model_dir, exist_ok=True)
        model_path = os.path.join(args.model_dir, "gtarl_agents.pt")
        FW.save_agents(model_path)
        print(f"[MODEL] Agents auto-saved to {model_path}")

    print("\n=== Final Report ===")
    print(f"Epochs: {epoch_counter}")
    print(f"Average TTFT (s):    {final_avg_ttft:.6f}")
    print(f"Total Carbon (kg):   {cumulative_carbon:.3f}")
    print(f"Total Water (m³):    {cumulative_water:.4f}")
    print(f"Total Energy ($):    {cumulative_energy:.3f}")
    print(f"Total Energy (kWh):  {cumulative_total_energy:.3f}")

    if framework.lower() == "lahyper" and lahyper_scheme_sums:
        print("\n=== LA_HYPER MULTI-AGENT SUMMARY (Run Totals) ===")
        # [FIX] Table formatted to indicate Summation
        header = f"{'Mode':<18} | {'Avg TTFT(s)':<11} | {'Total Carb(kg)':<14} | {'Total Wat(m³)':<13} | {'Total Cost($)':<13} | {'Total Energy(kWh)'}"
        print(header)
        print("-" * len(header))

        for mode in sorted(lahyper_scheme_sums.keys()):
            agg = lahyper_scheme_sums[mode]
            ep = max(1, agg["epochs"])

            # TTFT is request-weighted across epochs — the SAME aggregation
            # as the Final Report's final_avg_ttft, so a scheme's TTFT here
            # matches the headline number.  Plain epoch-mean only as a
            # divide-by-zero fallback when nothing was served.
            if agg["ttft_weight"] > 0.0:
                avg_ttft = agg["ttft_weighted_sum"] / agg["ttft_weight"]
            else:
                avg_ttft = agg["ttft_sum"] / ep
            total_carb = agg["carbon_sum"]
            total_wat = agg["water_sum"]
            total_cost = agg["energy_sum"]
            total_kwh = agg["total_energy_sum"]

            print(
                f"{mode:<18} | {avg_ttft:.4f}      | {total_carb:.3f}         | {total_wat:.3f}       | {total_cost:.3f}       | {total_kwh:.3f}")

    print("[DONE]")