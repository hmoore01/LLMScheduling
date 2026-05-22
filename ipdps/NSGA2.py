#!/usr/bin/env python3
# NSGA2.py — NSGA-II scheduling wrapper around Rate_Flow_Sim.LLM_Simulator

from __future__ import annotations
from typing import Any, Dict, List, Tuple
import random
import math
import gc

import pandas as pd

from Rate_Flow_Sim_v2 import LLM_Simulator

# -----------------------------
# Defaults / knobs
# -----------------------------
DEFAULT_EPOCH_LEN = 900
DEFAULT_NODE_TYPES = [0, 1, 2, 3, 4, 5]  # for simple power-plan heuristic


# -----------------------------
# Helpers for epoch data
# -----------------------------
def _ensure_epoch_columns(df: pd.DataFrame, epoch_len: int) -> pd.DataFrame:
    """
    Normalize common aliases to canonical columns used downstream.

    Expected logical fields:
      - source_dc_id : int
      - model_type   : str
      - num_tokens   : float/int
      - arrival_ms   : float/int (optional; default 0)
    """
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

    # Arrival time is optional – default everything to 0
    if "arrival_ms" not in d.columns:
        d["arrival_ms"] = 0.0

    # Clamp/normalize obvious things
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
    """
    Convert whatever the simulator returns into a stable, simpler shape for the frameworks.

    sim_out = (metrics, details, leftovers)
      metrics: dict with avg_ttft (s), total_energy (kWh),
               carbon_emissions (g), water_usage (m^3), energy_cost ($)
      details: per-request info
      leftovers: any structure – passed through
    """
    if not isinstance(sim_out, tuple) or len(sim_out) != 3:
        raise ValueError("Expected simulator output of form (metrics, details, leftovers)")

    metrics, details, leftovers = sim_out

    avg_ttft = float(metrics.get("avg_ttft", metrics.get("avg_ttft_sec", 0.0)))
    total_energy = float(metrics.get("total_energy", metrics.get("energy_kwh", 0.0)))

    stats: Dict[str, Any] = dict(metrics)  # start with whatever the simulator gave us
    stats["avg_ttft"] = avg_ttft
    stats["avg_ttft_sec"] = avg_ttft
    stats["total_energy"] = total_energy
    stats.setdefault("energy_kwh", total_energy)
    stats.setdefault("carbon_emissions", 0.0)
    stats.setdefault("water_usage", 0.0)
    stats.setdefault("energy_cost", 0.0)
    stats.setdefault("processed_tokens", float(metrics.get("processed_tokens", 0.0)))
    stats.setdefault(
        "requests_completed",
        float(metrics.get("requests_completed", metrics.get("served_requests", 0.0))),
    )
    stats.setdefault("requests_dropped", float(metrics.get("requests_dropped", 0.0)))

    # Remove any nested-dict fields (e.g. by_datacenter) from the returned stats.
    # simulator_LLM.py detects Parliament-style multi-scheme output by checking
    # whether any value in the stats dict is itself a dict. If by_datacenter is
    # present it triggers that check, causing the caller to treat per-DC utilization
    # dicts as the metrics and read all scalar fields as zero.
    stats = {k: v for k, v in stats.items() if not isinstance(v, dict)}

    if not isinstance(details, list):
        details = []

    return stats, details, leftovers


# -----------------------------
# Power plan heuristic
# -----------------------------
def _build_power_plan(
    routed_tokens_by_dc: Dict[int, float],
    node_types: List[int],     # kept in signature for backward compat; unused
    all_dcs: List[int],
) -> Dict[int, Dict[str, str]]:
    """
    Build a power plan using the {"all": ...} shorthand so every unit in each
    DC is addressed regardless of its node type ID.

    The v2 simulator assigns a unique node-type-ID range to each DC
    (e.g. DC0: 0-5, DC1: 6-11, DC8: 8-13).  A plan keyed on a hardcoded
    [0-5] list silently has no effect on the majority of DCs.

    DCs with routed traffic  →  IDLE  (ready to accept requests; ~13% idle power)
    DCs with no traffic      →  OFF   (zero power draw)
    """
    power_plan: Dict[int, Dict[str, str]] = {}
    for dc in all_dcs:
        tokens = max(0.0, routed_tokens_by_dc.get(dc, 0.0))
        power_plan[int(dc)] = {"all": "IDLE" if tokens > 0.0 else "OFF"}
    return power_plan


# -----------------------------
# Tiny NSGA-II implementation
# -----------------------------
class _Individual:
    __slots__ = ("gene", "objs", "rank", "crowding")

    def __init__(self, gene: List[float]):
        self.gene: List[float] = gene
        self.objs: Tuple[float, float, float, float] | None = None
        self.rank: int | None = None
        self.crowding: float = 0.0


def _dominates(a: _Individual, b: _Individual) -> bool:
    """Return True if individual a Pareto-dominates b (all objectives <= and one <)."""
    assert a.objs is not None and b.objs is not None
    better_or_equal = True
    strictly_better = False
    for av, bv in zip(a.objs, b.objs):
        if av > bv:
            better_or_equal = False
            break
        if av < bv:
            strictly_better = True
    return better_or_equal and strictly_better


def _non_dominated_sort(pop: List[_Individual]) -> List[List[int]]:
    n = len(pop)
    S: List[List[int]] = [[] for _ in range(n)]
    n_dom = [0] * n
    fronts: List[List[int]] = [[]]

    for i in range(n):
        pop[i].rank = None
        S[i] = []
        n_dom[i] = 0
        for j in range(n):
            if i == j:
                continue
            if _dominates(pop[i], pop[j]):
                S[i].append(j)
            elif _dominates(pop[j], pop[i]):
                n_dom[i] += 1
        if n_dom[i] == 0:
            pop[i].rank = 0
            fronts[0].append(i)

    f = 0
    while f < len(fronts) and fronts[f]:
        next_front: List[int] = []
        for i in fronts[f]:
            for j in S[i]:
                n_dom[j] -= 1
                if n_dom[j] == 0:
                    pop[j].rank = f + 1
                    next_front.append(j)
        if not next_front:
            break
        fronts.append(next_front)
        f += 1

    return fronts


def _assign_crowding(pop: List[_Individual], front: List[int]) -> None:
    if not front:
        return
    m = len(pop[front[0]].objs or [])
    for idx in front:
        pop[idx].crowding = 0.0

    for obj_idx in range(m):
        front_sorted = sorted(front, key=lambda i: pop[i].objs[obj_idx])  # type: ignore[index]
        min_val = pop[front_sorted[0]].objs[obj_idx]  # type: ignore[index]
        max_val = pop[front_sorted[-1]].objs[obj_idx]  # type: ignore[index]
        pop[front_sorted[0]].crowding = float("inf")
        pop[front_sorted[-1]].crowding = float("inf")
        if max_val == min_val:
            continue
        denom = max_val - min_val
        for k in range(1, len(front_sorted) - 1):
            prev_v = pop[front_sorted[k - 1]].objs[obj_idx]  # type: ignore[index]
            next_v = pop[front_sorted[k + 1]].objs[obj_idx]  # type: ignore[index]
            pop[front_sorted[k]].crowding += (next_v - prev_v) / denom


def _tournament_select(pop: List[_Individual]) -> _Individual:
    i = random.randrange(len(pop))
    j = random.randrange(len(pop))
    a, b = pop[i], pop[j]
    if a.rank is None or b.rank is None:
        return a
    if a.rank < b.rank:
        return a
    if b.rank < a.rank:
        return b
    if a.crowding > b.crowding:
        return a
    if b.crowding > a.crowding:
        return b
    return a if random.random() < 0.5 else b


def _crossover_and_mutate(
    g1: List[float],
    g2: List[float],
    num_pairs: int,
    num_dcs: int,
    crossover_prob: float,
    mutation_prob: float,
    mutation_sigma: float,
) -> Tuple[List[float], List[float]]:
    L = len(g1)
    assert L == len(g2)
    c1 = g1[:]
    c2 = g2[:]

    # Blend crossover
    if random.random() < crossover_prob:
        alpha = random.random()
        for i in range(L):
            c1[i] = alpha * g1[i] + (1.0 - alpha) * g2[i]
            c2[i] = alpha * g2[i] + (1.0 - alpha) * g1[i]

    # Gaussian mutation
    def _mutate(c: List[float]) -> None:
        for i in range(L):
            if random.random() < mutation_prob:
                c[i] += random.gauss(0.0, mutation_sigma)

    _mutate(c1)
    _mutate(c2)

    # Enforce non-negative and per-(src,model) normalization
    for p in range(num_pairs):
        start = p * num_dcs
        end = start + num_dcs
        for child in (c1, c2):
            block = [max(0.0, x) for x in child[start:end]]
            s = sum(block)
            if s <= 0.0:
                block = [1.0 / float(num_dcs)] * num_dcs
                s = 1.0
            else:
                block = [x / s for x in block]
            child[start:end] = block

    return c1, c2


# -----------------------------
# NSGA2 class wrapper
# -----------------------------
class NSGA2:
    @staticmethod
    def milp_optimizer(
        epoch_data,
        epoch_idx: int,
        node_properties,
        epoch_summary: Any,
    ):
        """
        Build a schedule + power plan and run the LLM_Simulator on a per-request path
        using a small NSGA-II multi-objective search over per-(src_dc, model) routing
        fractions.

        Returns: (stats, results, leftovers)
          - stats: dict with keys avg_ttft (s), carbon_emissions (g),
                   water_usage (m^3), total_energy (kWh), energy_cost ($)
          - results: per-request details
          - leftovers: simulator leftovers (e.g., per-DC utilization)
        """

        # 0) Normalize epoch rows to ensure required columns exist
        print(f"[NSGA2] milp_optimizer called: epoch={epoch_idx}, rows={len(epoch_data) if hasattr(epoch_data, '__len__') else '?'}")
        if hasattr(epoch_data, "iterrows") and hasattr(epoch_data, "columns"):
            df = _ensure_epoch_columns(epoch_data, DEFAULT_EPOCH_LEN)
        else:
            df = pd.DataFrame(epoch_data)
            df = _ensure_epoch_columns(df, DEFAULT_EPOCH_LEN)

        # 1) Summarize to per-(src,model) buckets
        work_df = (
            df.groupby(["source_dc_id", "model_type"], as_index=False)["num_tokens"]
            .sum()
            .rename(
                columns={"source_dc_id": "src_dc", "num_tokens": "total_tokens"}
            )
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
                "requests_completed": 0.0,
                "requests_dropped": 0.0,
            }
            return empty_stats, [], []

        # Pairs we actually route
        pairs: List[Tuple[int, str, float]] = []
        for r in work_df.itertuples(index=False):
            src_dc = int(getattr(r, "src_dc"))
            model = str(getattr(r, "model_type"))
            tokens = float(getattr(r, "total_tokens"))
            if tokens <= 0.0:
                continue
            pairs.append((src_dc, model, tokens))

        if not pairs:
            empty_stats = {
                "processed_tokens": 0.0,
                "avg_ttft_sec": 0.0,
                "avg_ttft": 0.0,
                "energy_kwh": 0.0,
                "total_energy": 0.0,
                "carbon_emissions": 0.0,
                "water_usage": 0.0,
                "energy_cost": 0.0,
                "requests_completed": 0.0,
                "requests_dropped": 0.0,
            }
            return empty_stats, [], []

        num_pairs = len(pairs)

        # 2) Build simulator and discover DCs
        try:
            if isinstance(epoch_summary, dict):
                spec_dir = epoch_summary.get("spec_dir", "sim_specs")
                epoch_len = int(epoch_summary.get("epoch_length", DEFAULT_EPOCH_LEN))
            else:
                spec_dir = "sim_specs"
                epoch_len = DEFAULT_EPOCH_LEN

            _discovery_sim = LLM_Simulator(
                spec_dir=spec_dir, epoch_length=epoch_len, debug=False)
            dcs = sorted(int(dc_id) for dc_id in _discovery_sim.datacenters.keys())
            del _discovery_sim
        except Exception:
            spec_dir = "sim_specs"
            epoch_len = DEFAULT_EPOCH_LEN
            dcs = sorted(df["source_dc_id"].unique().astype(int).tolist() or [0])

        if not dcs:
            dcs = [0]
        dcs_set = set(dcs)

        num_dcs = len(dcs)
        gene_length = num_pairs * num_dcs

        # 3) NSGA-II hyperparameters
        # pop_size=4, generations=1 → 8 search evals + 1 final = 9 sim calls
        pop_size = 4
        generations = 1
        crossover_prob = 0.9
        mutation_prob = 0.15
        mutation_sigma = 0.15

        # Deterministic seed per epoch
        random.seed(12345 + int(epoch_idx))

        # Node types for power plan
        node_types = (
            list(epoch_summary.get("node_types", DEFAULT_NODE_TYPES))
            if isinstance(epoch_summary, dict)
            else list(DEFAULT_NODE_TYPES)
        )

        # ----- Helper: gene -> (power_plan, requests_df, schedule_plan) -----
        # Pre-build the requests DataFrame ONCE — the request data (source_dc,
        # model, tokens, arrival_ms) never changes between evaluations.  Only
        # the routing assignment (plan_map) changes based on the gene.
        import numpy as np

        FIXED_VARIANT = "_FP16 (Base)_B16"
        _src_dc_arr = df["source_dc_id"].values.astype(int)
        _model_arr  = df["model_type"].values.astype(str)
        _tokens_arr = df["num_tokens"].values.astype(float).clip(min=0)
        _arrival_arr = df["arrival_ms"].values.astype(float)

        # Build the shared requests DataFrame once
        _requests_df = pd.DataFrame({
            "source_dc": _src_dc_arr,
            "model": [f"{m}{FIXED_VARIANT}" for m in _model_arr],
            "arrival_ms": _arrival_arr,
            "tokens": _tokens_arr.astype(int),
        })

        # Pre-compute per-row pair index for vectorized plan_map construction
        # Map each (src_dc, model) pair to its index in `pairs`
        _pair_to_idx = {(src_dc, model): idx for idx, (src_dc, model, _) in enumerate(pairs)}
        _row_pair_idx = np.full(len(df), -1, dtype=int)
        for row_i in range(len(df)):
            key = (int(_src_dc_arr[row_i]), str(_model_arr[row_i]))
            _row_pair_idx[row_i] = _pair_to_idx.get(key, -1)

        # Default DC for rows with no routing info
        _default_dc = int(dcs[0])
        _dcs_arr = np.array(dcs, dtype=int)

        def build_plans_from_gene(gene: List[float]):
            routed_tokens_by_dc: Dict[int, float] = {int(dc): 0.0 for dc in dcs}

            # Decode gene into per-pair fractions and find argmax DC per pair
            pair_best_dc = np.empty(num_pairs, dtype=int)
            for p_idx, (src_dc, model, tokens) in enumerate(pairs):
                start = p_idx * num_dcs
                end = start + num_dcs
                block = [max(0.0, x) for x in gene[start:end]]
                s = sum(block)
                if s <= 0.0:
                    block = [1.0 / float(num_dcs)] * num_dcs
                    s = 1.0
                else:
                    block = [x / s for x in block]

                best_j = max(range(num_dcs), key=lambda j: (block[j], -dcs[j]))
                pair_best_dc[p_idx] = int(dcs[best_j])

                for j, dc in enumerate(dcs):
                    routed_tokens_by_dc[int(dc)] += tokens * block[j]

            power_plan = _build_power_plan(
                routed_tokens_by_dc=routed_tokens_by_dc,
                node_types=node_types,
                all_dcs=[int(dc) for dc in dcs],
            )

            # Vectorized plan_map: look up each row's pair → argmax DC
            plan_map: Dict[int, int] = {}
            for row_i in range(len(df)):
                p_idx = _row_pair_idx[row_i]
                if p_idx >= 0:
                    plan_map[row_i] = int(pair_best_dc[p_idx])
                else:
                    src = int(_src_dc_arr[row_i])
                    plan_map[row_i] = src if src in dcs_set else _default_dc

            schedule_plan = {"map": plan_map}
            return power_plan, _requests_df, schedule_plan

        # ----- Helper: evaluate individual (fills objs) -----
        def evaluate(ind: _Individual) -> None:
            if ind.objs is not None:
                return
            power_plan, requests_df, schedule_plan = build_plans_from_gene(ind.gene)
            # Fresh simulator per evaluation — reusing a single instance causes
            # progressive slowdown as internal state accumulates across run_epoch calls
            eval_sim = LLM_Simulator(spec_dir=spec_dir, epoch_length=epoch_len, debug=False)
            metrics, _, _ = eval_sim.run_epoch(
                epoch_idx, requests_df, schedule_plan, power_plan
            )
            del eval_sim
            avg_ttft = float(metrics.get("avg_ttft", metrics.get("avg_ttft_sec", 0.0)))
            carbon = float(metrics.get("carbon_emissions", 0.0))
            water = float(metrics.get("water_usage", 0.0))
            cost = float(metrics.get("energy_cost", 0.0))
            ind.objs = (avg_ttft, carbon, water, cost)

        # ----- Initialize population with smart seeds -----
        population: List[_Individual] = []

        # Seed 1: route to source DC (locality-first)
        src_gene: List[float] = []
        for (src_dc, model, tokens) in pairs:
            block = [0.0] * num_dcs
            if src_dc in dcs_set:
                block[dcs.index(src_dc)] = 1.0
            else:
                block = [1.0 / float(num_dcs)] * num_dcs
            src_gene.extend(block)
        population.append(_Individual(src_gene))

        # Seed 2: uniform distribution
        uni_gene: List[float] = [1.0 / float(num_dcs)] * gene_length
        population.append(_Individual(uni_gene))

        # Seeds 3+: random feasible genes
        while len(population) < pop_size:
            gene: List[float] = []
            for _ in range(num_pairs):
                raw = [random.random() for _ in range(num_dcs)]
                s = sum(raw)
                if s <= 0.0:
                    raw = [1.0 / float(num_dcs)] * num_dcs
                    s = 1.0
                gene.extend([x / s for x in raw])
            population.append(_Individual(gene))

        # Evaluate initial population
        for ind in population:
            evaluate(ind)

        # Compute initial ranks/crowding
        fronts = _non_dominated_sort(population)
        for front in fronts:
            _assign_crowding(population, front)

        # ----- Main NSGA-II loop -----
        for _gen in range(generations):
            # Mating pool via tournament
            mating_pool: List[_Individual] = [
                _tournament_select(population) for _ in range(pop_size)
            ]

            # Variation
            children: List[_Individual] = []
            for i in range(0, pop_size, 2):
                p1 = mating_pool[i]
                p2 = mating_pool[(i + 1) % pop_size]
                c1_gene, c2_gene = _crossover_and_mutate(
                    p1.gene,
                    p2.gene,
                    num_pairs,
                    num_dcs,
                    crossover_prob,
                    mutation_prob,
                    mutation_sigma,
                )
                children.append(_Individual(c1_gene))
                children.append(_Individual(c2_gene))

            # Evaluate children
            for ind in children:
                evaluate(ind)

            # Combine and select next generation
            combined = population + children
            fronts = _non_dominated_sort(combined)
            for front in fronts:
                _assign_crowding(combined, front)

            new_pop: List[_Individual] = []
            for front in fronts:
                # Sort this front by descending crowding distance
                front_sorted = sorted(
                    front, key=lambda idx: combined[idx].crowding, reverse=True
                )
                for idx in front_sorted:
                    if len(new_pop) >= pop_size:
                        break
                    new_pop.append(combined[idx])
                if len(new_pop) >= pop_size:
                    break
            population = new_pop
            gc.collect()  # Free sim instances and discarded individuals

        # ----- Final selection from first front -----
        fronts = _non_dominated_sort(population)
        for front in fronts:
            _assign_crowding(population, front)
        first_front = fronts[0] if fronts else list(range(len(population)))

        # Equal weights for now; adjust in file if you want different trade-offs
        weights = [1.0, 1.0, 1.0, 1.0]

        best_idx = first_front[0]
        if len(first_front) > 1:
            obj_matrix = [population[i].objs for i in first_front]  # type: ignore[index]
            mins = [min(col) for col in zip(*obj_matrix)]  # type: ignore[arg-type]
            maxs = [max(col) for col in zip(*obj_matrix)]  # type: ignore[arg-type]

            def score(ind_idx: int) -> float:
                objs = population[ind_idx].objs  # type: ignore[index]
                total = 0.0
                for v, mn, mx, w in zip(objs, mins, maxs, weights):
                    if mx == mn:
                        continue
                    total += w * (v - mn) / (mx - mn)
                return total

            best_idx = min(first_front, key=score)

        best = population[best_idx]

        # ----- Execute ALL Pareto front members on fresh simulators -----
        # The search-phase objectives are from a reused sim (potentially stale).
        # Re-execute each front member to get clean metrics for PHV computation.
        front_results = []
        for fi, front_idx in enumerate(first_front):
            ind = population[front_idx]
            front_sim = LLM_Simulator(spec_dir=spec_dir, epoch_length=epoch_len, debug=False)
            pp, rdf, sp = build_plans_from_gene(ind.gene)
            fm, _, _ = front_sim.run_epoch(epoch_idx, rdf, sp, pp)
            front_metrics = {
                "avg_ttft": float(fm.get("avg_ttft", fm.get("avg_ttft_sec", 0.0))),
                "carbon_emissions": float(fm.get("carbon_emissions", 0.0)),
                "water_usage": float(fm.get("water_usage", 0.0)),
                "energy_cost": float(fm.get("energy_cost", 0.0)),
            }
            front_results.append(front_metrics)
            del front_sim
            # Parseable line for PHV extraction from logs
            print(f"[NSGA2-FRONT] epoch={epoch_idx} member={fi} "
                  f"ttft={front_metrics['avg_ttft']:.6f} "
                  f"carbon={front_metrics['carbon_emissions']:.4f} "
                  f"water={front_metrics['water_usage']:.4f} "
                  f"cost={front_metrics['energy_cost']:.4f}")

        print(f"[NSGA2] Epoch {epoch_idx}: Pareto front has {len(front_results)} "
              f"members (all re-executed on fresh sims)")

        # Use the "best" individual's results as the primary return
        best_fm = front_results[first_front.index(best_idx)] if best_idx in first_front \
                  else front_results[0]

        # But also run the best on a fully fresh sim for the detailed results
        reporting_sim = LLM_Simulator(spec_dir=spec_dir, epoch_length=epoch_len, debug=False)
        power_plan, requests_df, schedule_plan = build_plans_from_gene(best.gene)
        metrics, details, leftovers = reporting_sim.run_epoch(
            epoch_idx, requests_df, schedule_plan, power_plan
        )
        print(f"[NSGA2] Epoch {epoch_idx}: best -> ttft={metrics.get('avg_ttft',0):.4f}s  "
              f"carbon={metrics.get('carbon_emissions',0):.1f}  "
              f"cost={metrics.get('energy_cost',0):.2f}  "
              f"completed={metrics.get('requests_completed',0)}")

        stats, results, leftovers_norm = _normalize_sim_output(
            (metrics, details, leftovers)
        )

        # Attach front results for downstream PHV computation
        stats["pareto_front"] = [
            (fr["avg_ttft"], fr["carbon_emissions"], fr["water_usage"], fr["energy_cost"])
            for fr in front_results
        ]

        # Aggressive cleanup — prevent cross-epoch memory accumulation
        del reporting_sim, population, best, _requests_df
        del _src_dc_arr, _model_arr, _tokens_arr, _arrival_arr, _row_pair_idx
        gc.collect()

        return stats, results, leftovers_norm