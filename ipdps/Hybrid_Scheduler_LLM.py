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

from Rate_Flow_Sim_v2 import LLM_Simulator

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

# ── Objective order throughout this file ─────────────────────────────────────
# Index 0: TTFT   (avg time-to-first-token, seconds)
# Index 1: Carbon (total carbon emissions, g)
# Index 2: Water  (total water usage, m³)
# Index 3: Cost   (total energy cost, USD)
# ─────────────────────────────────────────────────────────────────────────────

# OUTPUT_WEIGHT — selects which Pareto solution is reported each epoch.
# Lower weight on an objective means the reported solution is allowed to be
# worse on it in exchange for gains on higher-weight objectives.
# Weights are normalised internally so only their ratios matter.
#
# Current setting: TTFT at 10%, sustainability objectives at 30% each.
# To restore fully balanced reporting: set all four values to 0.25.
OUTPUT_WEIGHT: List[float] = [0.25, 0.25, 0.25, 0.25]  # [TTFT, Carbon, Water, Cost]

# DIRICHLET_CONCENTRATION — shapes the population weight-vector distribution.
# Each population member draws its tradeoff weight from Dirichlet(concentration).
# Higher concentration on an objective biases the search toward solutions that
# are better on that objective.
#
# Current setting: Carbon, Water, Cost each get 3× the concentration of TTFT,
# so the search explores sustainability-favourable Pareto regions more densely.
# To restore uniform exploration: set all four values to 1.0.
DIRICHLET_CONCENTRATION: List[float] = [1.0, 1.0, 1.0, 1.0]  # [TTFT, Carbon, Water, Cost]

# CONSOLIDATION_FACTOR — fraction of DCs to consolidate traffic onto.
#
# Controls the core TTFT vs sustainability tradeoff by concentrating requests
# onto fewer, greener DCs and powering the rest fully OFF.
#
# Mechanism:
#   active_dcs = max(1, round(CONSOLIDATION_FACTOR * num_dcs))
#   All traffic → top-k lowest-carbon DCs (ranked by carbon intensity)
#   Remaining DCs → fully OFF (zero idle energy, carbon, water)
#
#   Fewer active DCs → each DC handles more requests than it would spread
#   → request queue builds up → wait_ms increases → TTFT rises
#   → but idle energy on powered-off DCs drops to zero → carbon/water/cost fall
#
# 1.0 = all DCs active, traffic spread thin, minimal queuing  (best TTFT)
# 0.5 = half the DCs active, moderate queuing                 (balanced)
# 0.25 = quarter of DCs active, heavy queuing                 (worst TTFT, best sustainability)
CONSOLIDATION_FACTOR: float = 0.33  # ~4 of 12 DCs active

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


def _rank_dcs_balanced(sim, all_dcs: List[int]) -> List[int]:
    """
    Rank DCs by a balanced composite score across all four sustainability
    dimensions: carbon intensity, electricity cost, water usage, and PUE.

    Each metric is min-max normalised across all DCs so no single dimension
    dominates.  Lower composite score = better overall DC to consolidate onto.
    """
    if len(all_dcs) <= 1:
        return list(all_dcs)

    def _dc_features(dc_id):
        dc = sim.datacenters.get(dc_id)
        if dc is None:
            return [999.0, 999.0, 999.0, 999.0]
        carbon = float(getattr(dc, "carbon_intensity_g_per_kwh", 400.0))
        tou    = getattr(dc, "tou_price", None)
        cost   = float(np.mean(tou)) if tou and len(tou) > 0 else 0.10
        water  = (float(getattr(dc, "water_static", 5.0)) +
                  float(getattr(dc, "water_cycling_density", 0.10)))
        pue    = float(getattr(dc, "pue_value", 1.18))
        return [carbon, cost, water, pue]

    features = {dc: _dc_features(dc) for dc in all_dcs}
    arr = np.array([features[dc] for dc in all_dcs], dtype=float)

    # Min-max normalise each column to [0, 1]
    mins = arr.min(axis=0)
    maxs = arr.max(axis=0)
    ranges = np.where(maxs - mins > 1e-9, maxs - mins, 1.0)
    norm = (arr - mins) / ranges          # lower = better for all four metrics

    # Equal-weight composite score
    scores = norm.mean(axis=1)
    return [dc for _, dc in sorted(zip(scores, all_dcs))]


def _build_power_plan(
        routed_tokens_by_dc: Dict[int, float],
        node_types: List[int],
        all_dcs: List[int],
) -> Dict[int, Dict[str, Any]]:
    """
    Build a simple two-state power plan driven by actual routed traffic.

    DCs that receive traffic:   all node types → ON  (fully active, accept requests)
    DCs that receive no traffic: all node types → OFF (zero idle energy)

    Keeping active DCs fully ON maximises their queue depth — when
    CONSOLIDATION_FACTOR routes traffic to fewer DCs those DCs must handle
    more load, creating intentional queuing that trades TTFT for the energy
    savings from the powered-off DCs.
    """
    if not node_types:
        node_types = list(DEFAULT_NODE_TYPES)

    power_plan: Dict[int, Dict[str, Any]] = {}
    for dc in all_dcs:
        tokens = max(0.0, routed_tokens_by_dc.get(dc, 0.0))
        state  = "ON" if tokens > 0.0 else "OFF"
        power_plan[int(dc)] = {"unit": {nt: state for nt in node_types}}

    return power_plan


# -----------------------------
# Weight vectors for multi-objective optimization
# -----------------------------
def _generate_weight_vectors(num_pop: int, num_obj: int) -> List[List[float]]:
    """
    Generate population weight vectors using a Dirichlet distribution shaped by
    DIRICHLET_CONCENTRATION.  Higher concentration on an objective increases the
    probability that population members have a high weight there, biasing the
    search toward solutions that are better on those objectives.

    The first vector is always OUTPUT_WEIGHT so the reported-best candidate is
    guaranteed a seat in the population.
    """
    alpha = list(DIRICHLET_CONCENTRATION) if len(DIRICHLET_CONCENTRATION) == num_obj             else [1.0] * num_obj
    weights = [list(OUTPUT_WEIGHT)]  # reserve slot 0 for the output weight
    rng = np.random.default_rng(seed=None)  # non-deterministic across epochs
    samples = rng.dirichlet(alpha, size=num_pop - 1)
    for row in samples:
        weights.append(row.tolist())
    return weights


def _log_objectives(objectives: List[float]) -> List[float]:
    """
    Apply log1p transform to raw objective values before fitness computation.

    Raw objectives span wildly different scales:
      TTFT    ~  100 – 600 s
      Carbon  ~  300,000 – 1,200,000 g
      Water   ~   15,000 –    80,000 m³
      Cost    ~       80 –       450 $

    After log1p the ranges compress to within ~1.5× of each other, so
    min-max normalisation treats every objective fairly regardless of units.
    """
    return [math.log1p(max(0.0, v)) for v in objectives]


def _local_fit(objectives: List[float], weight_vector: List[float],
               scaler_min: List[float], scaler_max: List[float]) -> float:
    """
    Compute weighted fitness with log-scale normalisation.
    scaler_min / scaler_max must already be in log space (use _log_objectives).
    Lower is better (we minimise).
    """
    log_objs = _log_objectives(objectives)
    normalized = []
    for log_val, mn, mx in zip(log_objs, scaler_min, scaler_max):
        if mx > mn:
            norm = 100.0 * (log_val - mn) / (mx - mn)
        else:
            norm = 0.0
        normalized.append(norm)

    weighted_terms = []
    for nv, w in zip(normalized, weight_vector):
        weighted_terms.append(nv * max(w, GLOBAL_MIN_WEIGHT))

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
        except Exception as e:
            print(f"[Hybrid] WARNING: LLM_Simulator init failed ({e}), retrying with defaults")
            dcs = sorted(df["source_dc_id"].unique().astype(int).tolist() or [0])
            sim = LLM_Simulator(spec_dir="sim_specs", epoch_length=DEFAULT_EPOCH_LEN, debug=False)

        if not dcs:
            dcs = [0]

        # Rank DCs by carbon intensity (greenest first) for consolidation routing.
        dc_balanced_rank = _rank_dcs_balanced(sim, dcs)

        num_dcs = len(dcs)
        node_types = list(epoch_summary.get("node_types", DEFAULT_NODE_TYPES)) if isinstance(epoch_summary,
                                                                                             dict) else DEFAULT_NODE_TYPES

        # Deterministic seed
        random.seed(42 + epoch_idx)
        np.random.seed(42 + epoch_idx)

        # 3) Helper: design -> simulator execution
        def evaluate_design(design: np.ndarray) -> Tuple[List[float], Dict[str, float]]:
            """Evaluate a routing design using the simulator.

            Each (src_dc, model) pair's tokens are split proportionally across DCs
            according to the design weight vector — one simulator row per (pair, DC)
            with a non-zero weight.
            """
            routed_tokens_by_dc = {dc: 0.0 for dc in dcs}
            req_rows = []
            plan_map = {}

            for pair_idx, (src_dc, model, tokens) in enumerate(pairs):
                dist = design[pair_idx, :]
                for dc_idx, (dc, weight) in enumerate(zip(dcs, dist)):
                    if weight <= 0.0:
                        continue
                    dc_tokens = tokens * float(weight)
                    routed_tokens_by_dc[dc] += dc_tokens
                    row_key = len(req_rows)
                    req_rows.append({
                        "source_dc": src_dc,
                        "model": model,
                        "arrival_ms": 0,
                        "tokens": max(1, int(dc_tokens)),
                    })
                    plan_map[row_key] = dc

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

        # 4) Initialize population with diverse designs.
        #    Member 0: uniform (equal weight to every DC).
        #    Members 1..num_dcs: DC-specialist (full weight on one DC each).
        #    Remaining members cycle through specialists again.
        initial_design = np.ones((num_pairs, num_dcs)) / num_dcs
        initial_objs, initial_metrics = evaluate_design(initial_design)

        population_designs: List[np.ndarray] = []
        population_objs: List[List[float]] = []

        population_designs.append(copy.deepcopy(initial_design))
        population_objs.append(copy.deepcopy(initial_objs))

        for i in range(1, POPULATION_SIZE):
            specialist = np.zeros((num_pairs, num_dcs))
            specialist[:, i % num_dcs] = 1.0
            spec_objs, _ = evaluate_design(specialist)
            population_designs.append(specialist)
            population_objs.append(spec_objs)

                # Set up normalisation bounds in log space from the initial population.
        # Using log-transformed values means the bounds reflect proportional
        # variation rather than absolute scale, so a 2× change in TTFT has the
        # same fitness impact as a 2× change in Carbon regardless of their units.
        all_log_objs = [_log_objectives(o) for o in population_objs]
        scaler_min = [min(lo[i] for lo in all_log_objs) - 0.1 for i in range(NUM_OBJECTIVES)]
        scaler_max = [max(lo[i] for lo in all_log_objs) + 0.1 for i in range(NUM_OBJECTIVES)]
        for i in range(NUM_OBJECTIVES):
            if scaler_max[i] <= scaler_min[i]:
                scaler_max[i] = scaler_min[i] + 1.0

        # Generate diverse weight vectors for Pareto exploration
        weight_vectors = _generate_weight_vectors(POPULATION_SIZE, NUM_OBJECTIVES)
        start_time = time.time()
        best_idx = min(range(POPULATION_SIZE),
                       key=lambda i: _local_fit(population_objs[i], OUTPUT_WEIGHT, scaler_min, scaler_max))
        best_design  = copy.deepcopy(population_designs[best_idx])
        best_objs    = copy.deepcopy(population_objs[best_idx])
        best_fitness = _local_fit(best_objs, OUTPUT_WEIGHT, scaler_min, scaler_max)

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

                # Update scaler bounds in log space
                log_child = _log_objectives(child_objs)
                for i in range(NUM_OBJECTIVES):
                    scaler_min[i] = min(scaler_min[i], log_child[i] - 0.1)
                    scaler_max[i] = max(scaler_max[i], log_child[i] + 0.1)

                # Check if this improves any population member
                for pop_idx in range(POPULATION_SIZE):
                    child_fit = _local_fit(child_objs, weight_vectors[pop_idx], scaler_min, scaler_max)
                    pop_fit = _local_fit(population_objs[pop_idx], weight_vectors[pop_idx], scaler_min, scaler_max)

                    if child_fit < pop_fit:
                        population_designs[pop_idx] = copy.deepcopy(child)
                        population_objs[pop_idx] = copy.deepcopy(child_objs)
                        break

                # Track overall best
                uniform_fit = _local_fit(child_objs, OUTPUT_WEIGHT, scaler_min, scaler_max)
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



        # 6) Final evaluation using CONSOLIDATION_FACTOR routing.
        #
        # Route all traffic to the top-k greenest DCs (ranked by carbon intensity)
        # and power all other DCs fully OFF.  Concentrating load on fewer DCs:
        #   • Intentionally creates request queues → wait_ms rises → TTFT worsens
        #   • Powered-off DCs charge zero idle energy → carbon/water/cost fall
        #
        # k = max(1, round(CONSOLIDATION_FACTOR × num_dcs))
        k = max(1, round(float(CONSOLIDATION_FACTOR) * num_dcs))
        active_dcs = set(dc_balanced_rank[:k])

        # Build uniform distribution over active DCs only
        active_list = [dc for dc in dcs if dc in active_dcs]
        active_probs = np.ones(len(active_list)) / len(active_list)
        active_idx   = {dc: i for i, dc in enumerate(active_list)}

        rng = np.random.default_rng(42 + epoch_idx)

        final_df = df.copy()
        final_plan_map: Dict[int, int] = {}
        routed_tokens_by_dc: Dict[int, float] = {dc: 0.0 for dc in dcs}

        src_ids = final_df["source_dc_id"].to_numpy()
        models  = final_df["model_type"].to_numpy()
        toks    = final_df["num_tokens"].to_numpy()

        for row_idx in range(len(final_df)):
            # Sample uniformly from the active (green) DCs
            chosen_active_idx = int(rng.choice(len(active_list), p=active_probs))
            target_dc = active_list[chosen_active_idx]
            final_plan_map[row_idx] = target_dc
            routed_tokens_by_dc[target_dc] += float(toks[row_idx])

        final_df = final_df.copy()
        final_df["model_type"] = final_df["model_type"].apply(
            lambda m: f"{m}{FIXED_VARIANT}"
        )

        power_plan = _build_power_plan(routed_tokens_by_dc, node_types, dcs)
        schedule_plan = {"map": final_plan_map}

        metrics, details, leftovers = sim.run_epoch(
            epoch_idx, final_df, schedule_plan, power_plan
        )

        stats, results, leftovers_norm = _normalize_sim_output((metrics, details, leftovers))

        # Emit a tagged per-epoch Pareto point for PHV calculation.
        # Parsed by hybrid_experiments.py _parse_epoch_phv().
        # Self-contained on one line so it survives interleaving with
        # simulator_LLM.py output and is unambiguous across all epochs.
        print(
            f"[HYBRID-FRONT] epoch={epoch_idx}"
            f" ttft={stats.get('avg_ttft', 0.0):.6f}"
            f" carbon={stats.get('carbon_emissions', 0.0):.4f}"
            f" water={stats.get('water_usage', 0.0):.4f}"
            f" cost={stats.get('energy_cost', 0.0):.6f}"
        )

        return stats, results, leftovers_norm

# ---------------------------------------------------------------------------
# Module-level entry point
# ---------------------------------------------------------------------------
# simulator_LLM.py calls FW.milp_optimizer(...) on the imported module object,
# so we expose the static method at module scope.
def milp_optimizer(
    epoch_data,
    epoch_idx: int,
    node_properties,
    epoch_summary: Any,
):
    """Module-level wrapper — delegates to Hybrid_Scheduler_LLM.milp_optimizer."""
    return Hybrid_Scheduler_LLM.milp_optimizer(
        epoch_data=epoch_data,
        epoch_idx=epoch_idx,
        node_properties=node_properties,
        epoch_summary=epoch_summary,
    )