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
from typing import Dict, Any, List, Optional, Callable, Union, Literal

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


def _even_src_dc(df: pd.DataFrame, num_dcs: int) -> pd.Series:
    if "source_dc_id" in df.columns:
        return pd.to_numeric(df["source_dc_id"], errors="coerce").fillna(0).astype(int)
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


def _assign_src_dc(df: pd.DataFrame, num_dcs: int, distribution: str, weights: dict = None,
                   timezone_offsets: dict = None) -> pd.Series:
    if "source_dc_id" in df.columns:
        existing = pd.to_numeric(df["source_dc_id"], errors="coerce").fillna(0).astype(int)
        if existing.max() > 0: return existing
    if distribution == "population":
        return _population_weighted_src_dc(df, num_dcs, weights)
    elif distribution == "time":
        return _time_based_src_dc(df, num_dcs, timezone_offsets, weights)
    else:
        return _even_src_dc(df, num_dcs)


def _ensure_num_tokens(df: pd.DataFrame, default_tokens: int = 400) -> pd.Series:
    if "num_tokens" in df.columns: return pd.to_numeric(df["num_tokens"], errors="coerce").fillna(0).astype(int)
    return pd.Series(default_tokens, index=df.index, dtype=int)


def summarize_epoch_rate(df: pd.DataFrame):
    if len(df) == 0:
        return pd.DataFrame(columns=["source_dc_id", "model_type", "tokens"])
    grp = df.groupby(["source_dc_id", "model_type"], as_index=False)["num_tokens"].sum()
    grp.rename(columns={"num_tokens": "tokens"}, inplace=True)
    return grp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epoch', type=int, default=96)
    parser.add_argument('-f', '--framework', type=str, default='lahyper')
    parser.add_argument('--freq-scale', type=float, default=1.0)
    parser.add_argument('--token-scale', type=float, default=1.0)
    parser.add_argument('--count-scale', type=int, default=1)
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

    trace["source_dc_id"] = _assign_src_dc(trace, args.num_dcs, distribution=args.distribution)

    if "model_type" not in trace.columns: trace["model_type"] = "Llama7b"
    trace["model_type"] = trace["model_type"].astype(str).map(_map_model_to_llama)
    trace["num_tokens"] = _ensure_num_tokens(trace, default_tokens=400)
    trace["arrival_ms"] = 0
    if "time_index" not in trace.columns: trace["time_index"] = 0

    trace["source_dc_id"] = pd.to_numeric(trace["source_dc_id"], errors="coerce").fillna(0).astype(int)
    trace["num_tokens"] = pd.to_numeric(trace["num_tokens"], errors="coerce").fillna(0).astype(int)

    grouped_trace = trace.groupby("epoch")
    max_epoch = int(trace["epoch"].max())
    print(f"[INIT] Loaded workload with {len(trace)} entries across {max_epoch + 1} epochs")

    framework = args.framework
    number_of_epoch = args.epoch
    node_properties: List[dict] = []

    cumulative_ttft = 0.0
    cumulative_carbon = 0.0
    cumulative_water = 0.0
    cumulative_energy = 0.0
    cumulative_total_energy = 0.0
    epoch_counter = 0

    lahyper_scheme_sums: Dict[str, Dict[str, float]] = {}


    def get_framework(framework):
        if framework.lower() == 'lahyper':
            import LA_Hyper_DDQN
            return LA_Hyper_DDQN
        else:
            raise ValueError(f"Framework '{framework}' not supported in this simplified snippet")


    FW = get_framework(framework)

    for epoch_idx in range(number_of_epoch):
        if epoch_idx not in grouped_trace.groups:
            print(f"\n--- Epoch {epoch_idx} ({framework}) [ZERO TRAFFIC] ---")
            epoch_data = pd.DataFrame(columns=trace.columns)
        else:
            epoch_data = grouped_trace.get_group(epoch_idx).copy()
            epoch_data["time_index"] = (epoch_data["time_index"] * args.freq_scale).clip(upper=899).astype(int)
            if args.token_scale != 1.0:
                epoch_data["num_tokens"] = (epoch_data["num_tokens"] * args.token_scale).round().astype(int)
            if args.count_scale > 1:
                epoch_data = pd.concat([epoch_data] * args.count_scale, ignore_index=True)
            epoch_data["arrival_ms"] = 0

            epoch_summary = summarize_epoch_rate(epoch_data)
            print(f"\n--- Epoch {epoch_idx} ({framework}) ---")
            print(epoch_summary.head())

        epoch_counter += 1

        stats, results, leftovers = FW.milp_optimizer(
            epoch_data=epoch_data,
            epoch_idx=epoch_idx,
            node_properties=node_properties,
            epoch_summary={
                "node_types": [0, 1, 2, 3, 4, 5],
                "datacenters": list(range(args.num_dcs)),
                "avg_input_tokens": 100,
                "avg_output_tokens": 100,
                "spec_dir": args.spec_dir,
                "epoch_length": 900,
            }
        )

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

        cumulative_ttft += float(stats.get("avg_ttft", 0.0))
        cumulative_carbon += float(stats.get("carbon_emissions", 0.0)) / 1000.0
        cumulative_water += float(stats.get("water_usage", 0.0)) / 100
        cumulative_energy += float(stats.get("energy_cost", 0.0))
        cumulative_total_energy += float(stats.get('total_energy', 0.0))

    print("\n=== Final Report ===")
    print(f"Epochs: {epoch_counter}")
    print(f"Average TTFT (s): {cumulative_ttft / max(1, epoch_counter):.6f}")
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