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
    s = str(m).strip().lower()
    if ("chatgpt" in s) or ("gpt-3.5" in s) or ("gpt3.5" in s):
        return "Llama7b"
    if ("gpt-4" in s) or ("gpt4" in s):
        return "Llama70b"
    if "70" in s or "70b" in s or "llama-2-70b" in s or "llama2-70b" in s:
        return "Llama70b"
    if "7" in s or "7b" in s or "llama-2-7b" in s or "llama2-7b" in s:
        return "Llama7b"
    return str(m)


DEFAULT_POPULATION_WEIGHTS = {0: 0.12, 1: 0.10, 2: 0.08, 3: 0.15, 4: 0.10, 5: 0.05, 6: 0.18, 7: 0.08, 8: 0.06, 9: 0.04,
                              10: 0.02, 11: 0.02}
DEFAULT_TIMEZONE_OFFSETS = {0: -5, 1: -8, 2: -6, 3: 0, 4: 1, 5: 2, 6: 8, 7: 9, 8: 7, 9: -3, 10: 3, 11: 2}
DEFAULT_BASE_POPULATION = DEFAULT_POPULATION_WEIGHTS
AUTOSCALE_MAX_MULTIPLIER = 700000.0
AUTOSCALE_COUNT_SHARE = 0.95
AUTOSCALE_MAX_COUNT_MULT = 1000
AUTOSCALE_MAX_DROP_FRAC = 0.05
AUTOSCALE_SEARCH_STEPS = 7
AUTOSCALE_MAX_EXPANDED_ROWS = 250000


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
    Select `num_dcs` datacenters from the simulator's full roster, keeping the
    most extreme (best) DC on each metric dimension to maximise differentiation.

    Strategy:
      1. Probe the simulator for all DC characteristics (carbon, cost, PUE).
      2. For each metric, mark the DC with the best (lowest) value as "must-keep".
      3. Fill remaining slots by maximising diversity (farthest-first from kept set).
      4. Return sorted list of selected DC IDs.

    If the simulator cannot be probed (e.g. Rate_Flow_Sim not available), falls
    back to range(num_dcs).
    """
    try:
        from Rate_Flow_Sim import LLM_Simulator
        sim = LLM_Simulator(debug=False, spec_dir=spec_dir)
        all_dc_ids = sorted(int(d) for d in sim.datacenters.keys())
        n_total = len(all_dc_ids)
    except Exception:
        return list(range(num_dcs))

    if num_dcs >= n_total:
        return all_dc_ids[:num_dcs]

    # Extract per-DC feature vectors [carbon, cost, pue]
    features = {}
    for dc_id in all_dc_ids:
        dc = sim.datacenters[dc_id]
        ci = float(getattr(dc, 'carbon_intensity_g_per_kwh', 400.0))
        # Average TOU price across hours as a single cost proxy
        tou = getattr(dc, 'tou_price', None)
        if isinstance(tou, (list, tuple)) and len(tou) > 0:
            cost_val = float(np.mean(tou))
        else:
            cost_val = float(tou) if tou is not None else 0.10
        pue = float(getattr(dc, 'pue_value', 1.18))
        features[dc_id] = np.array([ci, cost_val, pue])

    # Step 1: find the extreme (best = lowest) DC for each metric
    selected = set()
    metric_names = ["carbon", "cost", "PUE/water"]
    for dim in range(3):
        best_dc = min(all_dc_ids, key=lambda d: features[d][dim])
        selected.add(best_dc)

    # Step 2: farthest-first fill to maximise diversity
    # Normalise features to [0,1] so all dimensions contribute equally
    feat_matrix = np.array([features[d] for d in all_dc_ids])
    mins = feat_matrix.min(axis=0)
    maxs = feat_matrix.max(axis=0)
    ranges = np.where(maxs - mins > 1e-9, maxs - mins, 1.0)
    norm_features = {d: (features[d] - mins) / ranges for d in all_dc_ids}

    while len(selected) < num_dcs:
        best_candidate = None
        best_min_dist = -1.0
        for d in all_dc_ids:
            if d in selected:
                continue
            # Minimum distance to any already-selected DC
            min_dist = min(
                float(np.linalg.norm(norm_features[d] - norm_features[s]))
                for s in selected
            )
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_candidate = d
        if best_candidate is not None:
            selected.add(best_candidate)
        else:
            break

    result = sorted(selected)

    # Print the selection with characteristics
    print(f"[DC-SELECT] Chose {len(result)} of {n_total} DCs for max diversity:")
    print(f"  {'DC':>4}  {'Carbon':>8}  {'AvgCost':>8}  {'PUE':>6}  {'Reason'}")
    extremes = {}
    for dim, mname in enumerate(metric_names):
        best = min(result, key=lambda d: features[d][dim])
        extremes[best] = extremes.get(best, [])
        extremes[best].append(f"lowest {mname}")
    for d in result:
        f = features[d]
        reason = ", ".join(extremes.get(d, ["diversity fill"]))
        print(f"  {d:>4}  {f[0]:>8.1f}  {f[1]:>8.4f}  {f[2]:>6.3f}  {reason}")

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
    re-running BurstGPT_process.py.
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


def summarize_epoch_rate(df: pd.DataFrame):
    if len(df) == 0:
        return pd.DataFrame(columns=["source_dc_id", "model_type", "tokens"])
    grp = df.groupby(["source_dc_id", "model_type"], as_index=False)["num_tokens"].sum()
    grp.rename(columns={"num_tokens": "tokens"}, inplace=True)
    return grp


def _split_autoscale_multiplier(multiplier: float, count_cap: Optional[int] = None) -> Tuple[int, float]:
    m = max(0.0, float(multiplier))
    if m <= 1.0:
        return 1, m
    cap = AUTOSCALE_MAX_COUNT_MULT if count_cap is None else max(1, int(count_cap))
    count_mult = int(math.floor(m * AUTOSCALE_COUNT_SHARE))
    count_mult = max(1, min(count_mult, cap))
    remainder_scale = m / float(count_mult)
    return count_mult, remainder_scale


def _apply_autoscale_multiplier(
        epoch_data: pd.DataFrame,
        multiplier: float,
        epoch_idx: int,
        epoch_length_s: int = 900,
        count_cap: Optional[int] = None,
) -> Tuple[pd.DataFrame, int, float]:
    out = epoch_data.copy()
    count_mult, remainder_scale = _split_autoscale_multiplier(multiplier, count_cap=count_cap)

    if count_mult > 1:
        base_arrival_ms = _derive_arrival_ms(out, epoch_length_s=epoch_length_s).to_numpy(copy=True)
        repeated_parts = []
        for rep in range(count_mult):
            part = out.copy()
            part["_dup_id"] = rep
            part["_base_arrival_ms"] = base_arrival_ms
            repeated_parts.append(part)
        out = pd.concat(repeated_parts, ignore_index=True)

        epoch_window_ms = float(max(1, int(epoch_length_s))) * 1000.0
        rng = np.random.default_rng(100000 + int(epoch_idx))
        slot_ms = epoch_window_ms / float(count_mult)
        dup_idx = pd.to_numeric(out["_dup_id"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        base_arr = pd.to_numeric(out["_base_arrival_ms"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        jitter_ms = rng.uniform(0.0, slot_ms, size=len(out))
        new_arrival_ms = (base_arr + (dup_idx * slot_ms) + jitter_ms) % epoch_window_ms
        out["arrival_ms"] = new_arrival_ms
        if "time_index" in out.columns:
            out["time_index"] = np.floor(new_arrival_ms / 1000.0).astype(int)
        out.drop(columns=["_dup_id", "_base_arrival_ms"], inplace=True, errors="ignore")

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
    dry_df.rename(columns={"source_dc_id": "source_dc", "model_type": "model", "num_tokens": "tokens"}, inplace=True)
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

    peak_epoch = None
    peak_util = 0.0
    peak_df = None
    max_epoch_rows = 0

    for epoch_idx in candidate_epochs:
        ep_df = grouped_trace.get_group(epoch_idx).copy()
        max_epoch_rows = max(max_epoch_rows, len(ep_df))
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
    high_eval = _evaluate_autoscale_candidate(
        dry_sim, int(peak_epoch), peak_df, float(desired_multiplier), count_cap=count_cap_global
    )
    chosen_eval = high_eval
    drop_limited = False

    if float(high_eval.get("drop_frac", 0.0)) > float(max_drop) and float(desired_multiplier) > 1.0:
        low = 1.0
        high = float(desired_multiplier)
        best = _evaluate_autoscale_candidate(dry_sim, int(peak_epoch), peak_df, 1.0, count_cap=count_cap_global)
        if float(best.get("drop_frac", 1.0)) > float(max_drop):
            best = high_eval
        for _ in range(max(1, int(search_steps))):
            mid = (low + high) / 2.0
            mid_eval = _evaluate_autoscale_candidate(dry_sim, int(peak_epoch), peak_df, mid, count_cap=count_cap_global)
            if float(mid_eval.get("drop_frac", 1.0)) <= float(max_drop):
                best = mid_eval
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

    if "model_type" not in trace.columns: trace["model_type"] = "Llama7b"
    trace["model_type"] = trace["model_type"].astype(str).map(_map_model_to_llama)
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
            import Hybrid
            return Hybrid
        elif fw == 'parliament':
            import Game_Theoretic_RL
            return Game_Theoretic_RL
        else:
            raise ValueError(f"Framework '{framework_name}' not supported. Please check your spelling and available module imports.")


    FW = get_framework(framework)
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
        from Rate_Flow_Sim import LLM_Simulator
        autoscale_dry_sim = LLM_Simulator(debug=False, spec_dir=args.spec_dir)
        if autoscale_mode == "global_peak":
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

    for epoch_idx in range(number_of_epoch):
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
                        from Rate_Flow_Sim import LLM_Simulator
                        dry_sim = LLM_Simulator(debug=False, spec_dir=args.spec_dir)
                        autoscale_dry_sim = dry_sim

                    total_nodes = sum([len(dc.units) for dc in dry_sim.datacenters.values()])
                    base_req_count = max(1, len(epoch_data))
                    max_rows = max(1, int(getattr(args, "autoscale_max_rows", AUTOSCALE_MAX_EXPANDED_ROWS)))
                    count_cap_by_rows = max(1, int(max_rows // base_req_count))
                    min_count_for_target = max(1, int(math.ceil((float(args.target_util) * float(total_nodes)) / float(base_req_count))))
                    count_cap_dynamic = min(count_cap_by_rows, max(AUTOSCALE_MAX_COUNT_MULT, min_count_for_target))

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

        stats, results, leftovers = FW.milp_optimizer(
            epoch_data=epoch_data,
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

        # Parliament returns dict-of-dicts {scheme_name: metrics_dict}.
        # Extract the Parliament (consensus) metrics for the simulator's
        # cumulative tracking; other frameworks return a flat metrics dict.
        if isinstance(stats, dict) and "Parliament" in stats:
            flat_stats = stats["Parliament"]
        elif isinstance(stats, dict) and any(isinstance(v, dict) for v in stats.values()):
            # Fallback: grab the first scheme's metrics
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
                                                     "energy_sum": 0.0, "total_energy_sum": 0.0, "epochs": 0}
                    agg = lahyper_scheme_sums[mode]
                    agg["ttft_sum"] += float(sol.get("ttft", 0.0))
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
        cumulative_water += float(flat_stats.get("water_usage", 0.0)) / 100
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

    print("\n=== Final Report ===")
    print(f"Epochs: {epoch_counter}")
    print(f"Average TTFT (s): {final_avg_ttft:.6f}")
    print(f"Total Carbon (kg): {cumulative_carbon:.3f}")
    print(f"Total Water (L): {cumulative_water:.3f}")
    print(f"Total Energy ($): {cumulative_energy:.3f}")
    print(f"Total Energy (kWh): {cumulative_total_energy:.3f}")

    if framework.lower() == "lahyper" and lahyper_scheme_sums:
        print("\n=== LA_HYPER MULTI-AGENT SUMMARY (Run Totals) ===")
        # [FIX] Table formatted to indicate Summation
        header = f"{'Mode':<18} | {'Avg TTFT(s)':<11} | {'Total Carb(kg)':<14} | {'Total Wat(L)':<12} | {'Total Cost($)':<13} | {'Total Energy(kWh)'}"
        print(header)
        print("-" * len(header))

        for mode in sorted(lahyper_scheme_sums.keys()):
            agg = lahyper_scheme_sums[mode]
            ep = max(1, agg["epochs"])

            # TTFT is Averaged, everything else is strictly Summed
            avg_ttft = agg["ttft_sum"] / ep
            total_carb = agg["carbon_sum"]
            total_wat = agg["water_sum"]
            total_cost = agg["energy_sum"]
            total_kwh = agg["total_energy_sum"]

            print(
                f"{mode:<18} | {avg_ttft:.4f}      | {total_carb:.3f}         | {total_wat:.3f}       | {total_cost:.3f}       | {total_kwh:.3f}")

    print("[DONE]")