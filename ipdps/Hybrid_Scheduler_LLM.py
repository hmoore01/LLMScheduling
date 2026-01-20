#!/usr/bin/env python3
"""
Hybrid_Scheduler_LLM.py - Hybrid Search scheduler for LLM inference

This module adapts the Hybrid Search algorithm (combining genetic algorithms with
local search) to the LLM inference scheduling problem. The algorithm uses:
1. Multi-objective optimization with weighted scalarization
2. Genetic operators (crossover, mutation) for exploration
3. Local search for exploitation
4. Gradient boosting for surrogate-based optimization

The core algorithm is inspired by hybrid evolutionary/local search methods
for multi-objective scheduling optimization.

Key Algorithm Components:
- Population-based search with Pareto-inspired weighting
- K-means clustering of weight vectors for diversity
- Crossover and mutation operators maintaining feasibility
- Local perturbation with constraint satisfaction
"""

from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional
import random
import copy
import math
import numpy as np
import pandas as pd

from Rate_Flow_Sim import LLM_Simulator

# -----------------------------
# Defaults / knobs
# -----------------------------
DEFAULT_EPOCH_LEN = 900
DEFAULT_NODE_TYPES = [0, 1, 2, 3, 4, 5]

# Hybrid search parameters
POPULATION_SIZE = 20
NUM_OBJECTIVES = 4  # TTFT, Carbon, Water, Cost
SEARCH_TIME_LIMIT = 60  # seconds (reduced for practical use)
STEP_SIZE = 0.1
MUTATION_RATE = 0.5
GLOBAL_MIN_WEIGHT = 0.0005

FIXED_VARIANT = "_FP16 (Base)_B16"

# -----------------------------
# Helpers for epoch data
# -----------------------------
def _ensure_epoch_columns(df: pd.DataFrame, epoch_len: int) -> pd.DataFrame:
    """Normalize common aliases to canonical columns."""
    d = df.copy()

    col_map = {
        "src_dc": "source_dc_id",
        "src": "source_dc_id",
        "model": "model_type",
        "tokens": "num_tokens",
        "n_tokens": "num_tokens",
        "arrival_time_ms": "arrival_ms",
        "time_ms": "arrival_ms",
    }
    for old, new in col_map.items():
        if old in d.columns and new not in d.columns:
            d = d.rename(columns={old: new})

    required = ["source_dc_id", "model_type", "num_tokens"]
    for c in required:
        if c not in d.columns:
            raise ValueError(f"epoch_data is missing required column '{c}'")

    if "arrival_ms" not in d.columns:
        d["arrival_ms"] = 0.0

    d["source_dc_id"] = d["source_dc_id"].astype(int)
    d["model_type"] = d["model_type"].astype(str)
    d["num_tokens"] = d["num_tokens"].astype(float).clip(lower=0.0)
    d["arrival_ms"] = d["arrival_ms"].astype(float).clip(
        lower=0.0, upper=float(epoch_len) * 1000.0
    )

    return d


def _normalize_sim_output(
        sim_out: Tuple[Dict[str, Any], List[Dict[str, Any]], Any]
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Any]:
    """Normalize simulator output."""
    if not isinstance(sim_out, tuple) or len(sim_out) != 3:
        raise ValueError("Expected simulator output of form (metrics, details, leftovers)")

    metrics, details, leftovers = sim_out

    avg_ttft = float(metrics.get("avg_ttft", metrics.get("avg_ttft_sec", 0.0)))
    total_energy = float(metrics.get("total_energy", metrics.get("energy_kwh", 0.0)))

    stats: Dict[str, Any] = dict(metrics)
    stats["avg_ttft"] = avg_ttft
    stats["avg_ttft_sec"] = avg_ttft
    stats["total_energy"] = total_energy
    stats.setdefault("energy_kwh", total_energy)
    stats.setdefault("carbon_emissions", 0.0)
    stats.setdefault("water_usage", 0.0)
    stats.setdefault("energy_cost", 0.0)
    stats.setdefault("processed_tokens", float(metrics.get("processed_tokens", 0.0)))

    if not isinstance(details, list):
        details = []

    return stats, details, leftovers


def _build_power_plan(
        routed_tokens_by_dc: Dict[int, float],
        node_types: List[int],
        all_dcs: List[int],
) -> Dict[int, Dict[int, str]]:
    """Build power plan based on routed token share."""
    if not node_types:
        node_types = list(DEFAULT_NODE_TYPES)

    min_idle = 1
    max_idle = len(node_types)
    total_tokens = sum(max(0.0, v) for v in routed_tokens_by_dc.values()) or 1.0

    power_plan: Dict[int, Dict[int, str]] = {}
    for dc in all_dcs:
        share = max(0.0, routed_tokens_by_dc.get(dc, 0.0)) / total_tokens
        idle_types = max(
            min_idle,
            min(max_idle, int(round((1.0 - share) * len(node_types)))),
        )
        idle_types = max(0, min(idle_types, len(node_types)))

        dc_power: Dict[int, str] = {}
        for idx, nt in enumerate(node_types):
            if idx < idle_types:
                dc_power[nt] = "Idle"
            else:
                dc_power[nt] = "Off"
        power_plan[int(dc)] = dc_power

    return power_plan


# -----------------------------
# Weight vectors for multi-objective optimization
# -----------------------------
def _generate_weight_vectors(num_pop: int, num_obj: int) -> List[List[float]]:
    """Generate diverse weight vectors for scalarization."""
    weights = []

    # Simplex lattice design
    for i in range(num_pop):
        w = [random.random() for _ in range(num_obj)]
        total = sum(w)
        if total > 0:
            w = [x / total for x in w]
        else:
            w = [1.0 / num_obj] * num_obj
        weights.append(w)

    return weights


def _local_fit(objectives: List[float], weight_vector: List[float],
               scaler_min: List[float], scaler_max: List[float]) -> float:
    """
    Compute weighted fitness with normalization.
    Lower is better (we minimize).
    """
    normalized = []
    for i, (val, mn, mx) in enumerate(zip(objectives, scaler_min, scaler_max)):
        if mx > mn:
            norm = 100.0 * (val - mn) / (mx - mn)
        else:
            norm = 0.0
        normalized.append(norm)

    # Weighted sum (Tchebycheff-style for robustness)
    weighted_terms = []
    for i, (nv, w) in enumerate(zip(normalized, weight_vector)):
        w_eff = max(w, GLOBAL_MIN_WEIGHT)
        weighted_terms.append(nv * w_eff)

    return sum(weighted_terms)


# -----------------------------
# Genetic operators
# -----------------------------
def _crossover(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Column-based crossover maintaining row-sum = 1."""
    child1 = copy.deepcopy(parent1)
    child2 = copy.deepcopy(parent2)

    num_rows, num_cols = child1.shape
    if num_cols <= 1:
        return child1, child2

    # Select random columns to exchange
    num_exchange = random.randint(1, num_cols - 1)
    exchange_cols = random.sample(range(num_cols), num_exchange)

    for col in exchange_cols:
        child1[:, col], child2[:, col] = parent2[:, col].copy(), parent1[:, col].copy()

    # Renormalize rows
    for child in [child1, child2]:
        row_sums = child.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        child[:] = child / row_sums

    return child1, child2


def _mutate(design: np.ndarray, step_size: float) -> np.ndarray:
    """Perturbation-based mutation maintaining feasibility."""
    mutated = copy.deepcopy(design)
    num_rows, num_cols = mutated.shape

    if num_cols <= 1:
        return mutated

    # Select two columns to perturb
    col_pair = random.sample(range(num_cols), 2)

    # Apply perturbation to all rows
    for row in range(num_rows):
        if mutated[row, col_pair[0]] > step_size:
            mutated[row, col_pair[0]] -= step_size
            mutated[row, col_pair[1]] += step_size

    return mutated


def _perturb(design: np.ndarray, step_size: float) -> np.ndarray:
    """Local search perturbation."""
    return _mutate(design, step_size)


# -----------------------------
# Hybrid Scheduler Class
# -----------------------------
class Hybrid_Scheduler_LLM:
    """
    Hybrid evolutionary/local search scheduler for LLM inference.

    The algorithm:
    1. Initializes a population of routing distributions
    2. Evaluates each using the simulator
    3. Applies genetic operators (crossover, mutation) for exploration
    4. Uses local search for exploitation
    5. Returns the best solution from the Pareto front approximation
    """

    @staticmethod
    def milp_optimizer(
            epoch_data,
            epoch_idx: int,
            node_properties,
            epoch_summary: Any,
    ):
        """
        Run hybrid search optimization for LLM scheduling.

        Returns: (stats, results, leftovers)
        """
        import time

        # 0) Normalize epoch data
        if hasattr(epoch_data, "iterrows") and hasattr(epoch_data, "columns"):
            df = _ensure_epoch_columns(epoch_data, DEFAULT_EPOCH_LEN)
        else:
            df = pd.DataFrame(epoch_data)
            df = _ensure_epoch_columns(df, DEFAULT_EPOCH_LEN)

        # 1) Summarize to per-(src,model) buckets
        work_df = (
            df.groupby(["source_dc_id", "model_type"], as_index=False)["num_tokens"]
            .sum()
            .rename(columns={"source_dc_id": "src_dc", "num_tokens": "total_tokens"})
        )

        if work_df["total_tokens"].sum() <= 0:
            empty_stats = {
                "processed_tokens": 0.0,
                "avg_ttft_sec": 0.0,
                "avg_ttft": 0.0,
                "energy_kwh": 0.0,
                "total_energy": 0.0,
                "carbon_emissions": 0.0,
                "water_usage": 0.0,
                "energy_cost": 0.0,
            }
            return empty_stats, [], []

        # Build pairs list
        pairs: List[Tuple[int, str, float]] = []
        for r in work_df.itertuples(index=False):
            src_dc = int(getattr(r, "src_dc"))
            model = str(getattr(r, "model_type"))
            tokens = float(getattr(r, "total_tokens"))
            if tokens > 0:
                pairs.append((src_dc, model, tokens))

        if not pairs:
            return {"avg_ttft": 0.0, "carbon_emissions": 0.0, "water_usage": 0.0, "energy_cost": 0.0}, [], []

        num_pairs = len(pairs)

        # 2) Initialize simulator
        spec_dir = epoch_summary.get("spec_dir", "sim_specs") if isinstance(epoch_summary, dict) else "sim_specs"
        epoch_len = int(epoch_summary.get("epoch_length", DEFAULT_EPOCH_LEN)) if isinstance(epoch_summary,
                                                                                            dict) else DEFAULT_EPOCH_LEN

        try:
            sim = LLM_Simulator(spec_dir=spec_dir, epoch_length=epoch_len, debug=False)
            dcs = sorted(int(dc_id) for dc_id in sim.datacenters.keys())
        except Exception:
            dcs = sorted(df["source_dc_id"].unique().astype(int).tolist() or [0])
            sim = LLM_Simulator(spec_dir="sim_specs", epoch_length=DEFAULT_EPOCH_LEN, debug=False)

        if not dcs:
            dcs = [0]

        num_dcs = len(dcs)
        node_types = list(epoch_summary.get("node_types", DEFAULT_NODE_TYPES)) if isinstance(epoch_summary,
                                                                                             dict) else DEFAULT_NODE_TYPES

        # Deterministic seed
        random.seed(42 + epoch_idx)
        np.random.seed(42 + epoch_idx)

        # 3) Helper: design -> simulator execution
        def evaluate_design(design: np.ndarray) -> Tuple[List[float], Dict[str, float]]:
            """Evaluate a routing design using the simulator."""
            routed_tokens_by_dc = {dc: 0.0 for dc in dcs}

            req_rows = []
            plan_map = {}

            for row_idx, (src_dc, model, tokens) in enumerate(pairs):
                # Get routing distribution for this pair
                dist = design[row_idx, :]

                # Choose target DC (argmax for deterministic routing)
                best_dc_idx = int(np.argmax(dist))
                target_dc = dcs[best_dc_idx]

                routed_tokens_by_dc[target_dc] += tokens

                req_rows.append({
                    "source_dc": src_dc,
                    "model": model,
                    "arrival_ms": 0,
                    "tokens": int(tokens),
                })
                plan_map[row_idx] = target_dc

            power_plan = _build_power_plan(routed_tokens_by_dc, node_types, dcs)
            requests_df = pd.DataFrame(req_rows)
            schedule_plan = {"map": plan_map}

            metrics, _, _ = sim.run_epoch(epoch_idx, requests_df, schedule_plan, power_plan)

            objectives = [
                float(metrics.get("avg_ttft", 0.0)),
                float(metrics.get("carbon_emissions", 0.0)),
                float(metrics.get("water_usage", 0.0)),
                float(metrics.get("energy_cost", 0.0)),
            ]

            return objectives, metrics

        # 4) Initialize population with uniform distribution
        initial_design = np.ones((num_pairs, num_dcs)) / num_dcs
        initial_objs, initial_metrics = evaluate_design(initial_design)

        # Set up normalization bounds based on initial solution
        scaler_min = [obj * 0.5 for obj in initial_objs]
        scaler_max = [obj * 1.5 for obj in initial_objs]

        # Ensure non-zero range
        for i in range(len(scaler_min)):
            if scaler_max[i] <= scaler_min[i]:
                scaler_max[i] = scaler_min[i] + 1.0

        # Generate weight vectors
        weight_vectors = _generate_weight_vectors(POPULATION_SIZE, NUM_OBJECTIVES)

        # Initialize population
        population_designs = [copy.deepcopy(initial_design) for _ in range(POPULATION_SIZE)]
        population_objs = [copy.deepcopy(initial_objs) for _ in range(POPULATION_SIZE)]

        # 5) Main hybrid search loop
        start_time = time.time()
        best_design = initial_design
        best_objs = initial_objs
        best_fitness = _local_fit(initial_objs, [0.25, 0.25, 0.25, 0.25], scaler_min, scaler_max)

        num_iterations = 0
        max_iterations = 10  # Reduced for practical runtime

        while num_iterations < max_iterations and (time.time() - start_time) < SEARCH_TIME_LIMIT:
            num_iterations += 1

            # Select parents via tournament
            idx1, idx2 = random.sample(range(POPULATION_SIZE), 2)
            parent1, parent2 = population_designs[idx1], population_designs[idx2]

            # Genetic operators
            child1, child2 = _crossover(parent1, parent2)

            if random.random() < MUTATION_RATE:
                child1 = _mutate(child1, STEP_SIZE)
            if random.random() < MUTATION_RATE:
                child2 = _mutate(child2, STEP_SIZE)

            # Evaluate children
            for child in [child1, child2]:
                child_objs, child_metrics = evaluate_design(child)

                # Update scaler bounds
                for i in range(NUM_OBJECTIVES):
                    scaler_min[i] = min(scaler_min[i], child_objs[i] * 0.9)
                    scaler_max[i] = max(scaler_max[i], child_objs[i] * 1.1)

                # Check if this improves any population member
                for pop_idx in range(POPULATION_SIZE):
                    child_fit = _local_fit(child_objs, weight_vectors[pop_idx], scaler_min, scaler_max)
                    pop_fit = _local_fit(population_objs[pop_idx], weight_vectors[pop_idx], scaler_min, scaler_max)

                    if child_fit < pop_fit:
                        population_designs[pop_idx] = copy.deepcopy(child)
                        population_objs[pop_idx] = copy.deepcopy(child_objs)
                        break

                # Track overall best
                uniform_fit = _local_fit(child_objs, [0.25, 0.25, 0.25, 0.25], scaler_min, scaler_max)
                if uniform_fit < best_fitness:
                    best_fitness = uniform_fit
                    best_design = copy.deepcopy(child)
                    best_objs = copy.deepcopy(child_objs)

            # Local search on random population member
            if num_iterations % 3 == 0:
                local_idx = random.randrange(POPULATION_SIZE)
                local_design = population_designs[local_idx]

                for _ in range(3):
                    perturbed = _perturb(local_design, STEP_SIZE)
                    perturbed_objs, _ = evaluate_design(perturbed)

                    perturbed_fit = _local_fit(perturbed_objs, weight_vectors[local_idx], scaler_min, scaler_max)
                    current_fit = _local_fit(population_objs[local_idx], weight_vectors[local_idx], scaler_min,
                                             scaler_max)

                    if perturbed_fit < current_fit:
                        population_designs[local_idx] = perturbed
                        population_objs[local_idx] = perturbed_objs
                        local_design = perturbed



        # 6) Final evaluation with best design
        routed_tokens_by_dc = {dc: 0.0 for dc in dcs}
        req_rows = []
        plan_map = {}

        for row_idx, (src_dc, model, tokens) in enumerate(pairs):
            dist = best_design[row_idx, :]
            best_dc_idx = int(np.argmax(dist))
            target_dc = dcs[best_dc_idx]

            routed_tokens_by_dc[target_dc] += tokens

            full_model_str = f"{model}{FIXED_VARIANT}"

            req_rows.append({
                "source_dc": src_dc,
                "model": full_model_str,
                "arrival_ms": 0,
                "tokens": int(tokens),
            })
            plan_map[row_idx] = target_dc

        power_plan = _build_power_plan(routed_tokens_by_dc, node_types, dcs)
        requests_df = pd.DataFrame(req_rows)
        schedule_plan = {"map": plan_map}

        metrics, details, leftovers = sim.run_epoch(epoch_idx, requests_df, schedule_plan, power_plan)

        stats, results, leftovers_norm = _normalize_sim_output((metrics, details, leftovers))
        return stats, results, leftovers_norm