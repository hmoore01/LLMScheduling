#!/usr/bin/env python3
# PerLLM.py — PerLLM-style scheduling wrapper around Rate_Flow_Sim.LLM_Simulator
#
# This module is a drop-in framework for `simulator_LLM.py`:
#   from PerLLM import PerLLM
#   stats, results, leftovers = PerLLM.milp_optimizer(...)
#
# It mirrors Helix.py’s interface but uses a constraint-satisfaction,
# energy-aware routing heuristic inspired by the PerLLM CS-UCB algorithm:
#   - treat each datacenter as an arm
#   - estimate processing delay (queue + network) for each (src, model) bucket
#   - enforce a delay budget (constraint satisfaction)
#   - among feasible DCs, choose the lowest estimated energy cost
#
# To keep runtime practical, this implementation uses static surrogate
# estimates (capacity, TOU price, and latency) rather than a full online
# multi-armed bandit loop.

from __future__ import annotations
from typing import Any, Dict, List, Tuple
import math

import pandas as pd

from Rate_Flow_Sim_v2 import LLM_Simulator

# -----------------------------
# Defaults / knobs
# -----------------------------
DEFAULT_EPOCH_LEN = 900
DEFAULT_NODE_TYPES = [0, 1, 2, 3, 4, 5]  # for simple power-plan heuristic

# PerLLM-style knobs (can be overridden via epoch_summary if desired)
DEFAULT_DELAY_BUDGET_SEC = 2.0       # processing time constraint (edge-cloud delay budget)
DEFAULT_ENERGY_WEIGHT = 1.0          # weight on energy cost
DEFAULT_DELAY_PENALTY_WEIGHT = 0.0   # optional blended objective; kept 0 since we hard-constrain delay


# -----------------------------
# Helpers for epoch data
# -----------------------------
def _ensure_epoch_columns(df: pd.DataFrame, epoch_len: int) -> pd.DataFrame:
    """Normalize common aliases to canonical columns used downstream."""
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
      metrics: dict with avg_ttft (s), energy_kwh, carbon_emissions (g),
               water_usage (m^3), energy_cost ($)
      details: per-request info
      leftovers: any structure – passed through
    """
    if not isinstance(sim_out, tuple) or len(sim_out) != 3:
        raise ValueError("Expected simulator output of form (metrics, details, leftovers)")

    metrics, details, leftovers = sim_out

    avg_ttft = float(metrics.get("avg_ttft", metrics.get("avg_ttft_sec", 0.0)))
    energy_kwh = float(
        metrics.get("total_energy", metrics.get("energy_kwh", 0.0))
    )

    stats = {
        "avg_ttft_sec": avg_ttft,
        "avg_ttft": avg_ttft,
        "energy_kwh": energy_kwh,
        "total_energy": energy_kwh,
        "carbon_emissions": float(metrics.get("carbon_emissions", 0.0)),
        "water_usage": float(metrics.get("water_usage", 0.0)),
        "energy_cost": float(metrics.get("energy_cost", 0.0)),
        "processed_tokens": float(metrics.get("processed_tokens", 0.0)),
        "requests_completed": float(
            metrics.get("requests_completed", metrics.get("served_requests", 0.0))
        ),
        "requests_dropped": float(metrics.get("requests_dropped", 0.0)),
    }

    if not isinstance(details, list):
        details = []

    return stats, details, leftovers


# -----------------------------
# DC discovery & capacity helpers
# -----------------------------
def _discover_dcs_from_node_props(node_properties) -> List[int]:
    """
    Attempt to infer the set of DC ids from node_properties.

    Accepts:
      - dict[node_id] -> {"dc_id": int, ...}
      - iterable of dicts with "dc_id" or "dc" key
      - anything else returns an empty list
    """
    dcs = set()

    if isinstance(node_properties, dict):
        iterable = node_properties.values()
    elif isinstance(node_properties, (list, tuple)):
        iterable = node_properties
    else:
        try:
            iterable = dict(node_properties).values()
        except Exception:
            iterable = []

    for rec in iterable:
        if not isinstance(rec, dict):
            continue
        for key in ("dc_id", "dc", "datacenter_id", "datacenter"):
            if key in rec:
                try:
                    dcs.add(int(rec[key]))
                except Exception:
                    pass
                break

    return sorted(dcs)


def _capacity_per_dc(node_properties, dcs: List[int]) -> Dict[int, float]:
    """
    Estimate relative capacity from node_properties by counting entries per DC.
    If unavailable, use equal capacities across discovered DCs.
    """
    caps = {dc: 0.0 for dc in dcs}
    count_any = False

    if isinstance(node_properties, dict):
        iterable = node_properties.values()
    elif isinstance(node_properties, (list, tuple)):
        iterable = node_properties
    else:
        try:
            iterable = dict(node_properties).values()
        except Exception:
            iterable = []

    for rec in iterable:
        if not isinstance(rec, dict):
            continue
        try:
            dc_id = int(rec.get("dc_id", rec.get("dc")))
        except Exception:
            continue

        if dc_id in caps:
            caps[dc_id] += 1.0
            count_any = True

    if not count_any:
        for dc in caps:
            caps[dc] = 1.0

    for dc in caps:
        caps[dc] = max(1e-6, caps[dc])
    return caps


def _capacity_per_dc_from_sim(sim: LLM_Simulator) -> Dict[int, float]:
    """
    Estimate per-DC capacity from the simulator's GPU perf tables.

    We aggregate tokens/epoch over all processors in each datacenter using
    ms_per_token (or ms_per_request / avg_tokens_per_request if needed).
    This gives us a Helix/PerLLM-style "capacity" for the max-flow style heuristic.
    """
    caps: Dict[int, float] = {}
    epoch_len_s = float(getattr(sim, "epoch_length", DEFAULT_EPOCH_LEN))
    epoch_ms = epoch_len_s * 1000.0

    for dc_id, dc in getattr(sim, "datacenters", {}).items():
        total_tokens = 0.0
        units = getattr(dc, "units", [])
        for u in units:
            perf = getattr(u, "model_perf", {})
            if not isinstance(perf, dict):
                continue

            # Aggregate capacity across all models this unit can serve.
            for rec in perf.values():
                ms_per_tok = 0.0

                # Prefer ms_per_token if it exists
                try:
                    ms_per_tok = float(rec.get("ms_per_token", 0.0))
                except Exception:
                    ms_per_tok = 0.0

                if ms_per_tok <= 0.0:
                    # Fallback: derive from ms_per_request / avg_tokens_per_request if available
                    try:
                        ms_req = float(rec.get("ms_per_request", 0.0))
                        avg_tok = float(
                            rec.get(
                                "avg_tokens_per_request",
                                rec.get("avg_tokens_per_req", 0.0),
                            )
                        )
                    except Exception:
                        ms_req, avg_tok = 0.0, 0.0

                    if ms_req > 0.0 and avg_tok > 0.0:
                        ms_per_tok = ms_req / avg_tok

                if ms_per_tok > 0.0:
                    tokens_per_ms = 1.0 / ms_per_tok
                    total_tokens += tokens_per_ms * epoch_ms

        caps[int(dc_id)] = total_tokens

    # If everything somehow came out zero, fall back to equal capacities.
    if not any(v > 0.0 for v in caps.values()):
        for dc_id in caps:
            caps[dc_id] = 1.0

    # Avoid divide-by-zero downstream
    for dc_id in caps:
        caps[dc_id] = max(1e-6, caps[dc_id])

    return caps


# -----------------------------
# Power plan (Idle/Off per node type) scaled by routed share
# -----------------------------
def _build_power_plan(
    routed_token_share_by_dc: Dict[int, float],
    epoch_summary: Any,
) -> Dict[int, Dict[str, Dict[int, str]]]:
    """
    Heuristic Idle/Off power plan scaled by per-DC share.
    epoch_summary may contain:
      - node_types (default DEFAULT_NODE_TYPES)
      - min_idle_types (default 1)
      - max_idle_types (default len(node_types))
    """
    if isinstance(epoch_summary, dict):
        node_types = list(epoch_summary.get("node_types", DEFAULT_NODE_TYPES))
        min_idle = int(epoch_summary.get("min_idle_types", 1))
        max_idle = int(epoch_summary.get("max_idle_types", len(node_types)))
    else:
        node_types = list(DEFAULT_NODE_TYPES)
        min_idle = 1
        max_idle = len(node_types)

    max_idle = max(1, min(max_idle, len(node_types)))

    total = sum(max(0.0, v) for v in routed_token_share_by_dc.values()) or 1.0
    shares = {
        dc: max(0.0, routed_token_share_by_dc.get(dc, 0.0)) / total
        for dc in routed_token_share_by_dc
    }

    power_plan: Dict[int, Dict[str, Dict[int, str]]] = {}
    for dc, share in shares.items():
        # More share = fewer Idle node types (high share -> more "On")
        idle_types = max(
            min_idle,
            min(max_idle, int(round((1.0 - share) * len(node_types)))),
        )
        idle_types = max(0, min(idle_types, len(node_types)))

        dc_power: Dict[int, str] = {}
        for idx, nt in enumerate(node_types):
            if idx < idle_types:
                dc_power[nt] = "IDLE"
            else:
                dc_power[nt] = "OFF"
        power_plan[int(dc)] = {"unit": dc_power}

    return power_plan


# -----------------------------
# PerLLM class wrapper
# -----------------------------
class PerLLM:
    @staticmethod
    def milp_optimizer(
        epoch_data,
        epoch_idx: int,
        node_properties,
        epoch_summary: Any,
    ):
        """
        Build a schedule + power plan and run the LLM_Simulator on a per-request path.

        This follows the *spirit* of PerLLM's CS-UCB algorithm:

          1) Aggregate work by (source_dc, model_type).
          2) Use LLM_Simulator to discover datacenters, capacities, TOU prices, and latency.
          3) For each (src, model) bucket, treat each DC as an arm:
               - estimate processing delay: queueing + network
               - enforce a delay budget (constraint satisfaction)
               - among feasible DCs, choose the lowest estimated energy cost
          4) Build a fraction plan and then a deterministic per-request map.
          5) Derive a power plan based on routed token share.
          6) Run a single epoch of the simulator and normalize the results.

        Returns: (stats, results, leftovers)
          - stats: dict with keys avg_ttft (s), carbon_emissions (g),
                   water_usage (m^3), total_energy (kWh), energy_cost ($)
          - results: per-request details
          - leftovers: simulator leftovers (e.g., per-DC utilization)
        """

        # 0) Normalize epoch rows to ensure required columns exist
        if hasattr(epoch_data, "iterrows") and hasattr(epoch_data, "columns"):
            df = _ensure_epoch_columns(epoch_data, DEFAULT_EPOCH_LEN)
        else:
            df = pd.DataFrame(epoch_data)
            df = _ensure_epoch_columns(df, DEFAULT_EPOCH_LEN)

        # 1) Summarize to per-(src,model) buckets (work units)
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

        # 2) Discover DCs, capacities, TOU prices, and latency based on the simulator
        try:
            if isinstance(epoch_summary, dict):
                spec_dir = epoch_summary.get("spec_dir", "sim_specs")
                epoch_len = int(
                    epoch_summary.get("epoch_length", DEFAULT_EPOCH_LEN)
                )
                delay_budget_sec = float(
                    epoch_summary.get("delay_budget_sec", DEFAULT_DELAY_BUDGET_SEC)
                )
            else:
                spec_dir = "sim_specs"
                epoch_len = DEFAULT_EPOCH_LEN
                delay_budget_sec = DEFAULT_DELAY_BUDGET_SEC

            sim = LLM_Simulator(
                spec_dir=spec_dir, epoch_length=epoch_len, debug=False
            )

            dcs = sorted(int(dc_id) for dc_id in sim.datacenters.keys())
            cap_tps = _capacity_per_dc_from_sim(sim)

            # Latency matrix (ms)
            latency_mat = getattr(sim.network, "latency_ms", None)

            # Time-of-use prices per DC for this epoch hour (simplified PerLLM-style energy modeling)
            epoch_hour = int(epoch_idx % 24)
            dc_price: Dict[int, float] = {}
            for dc_id, dc in sim.datacenters.items():
                try:
                    tou = dc.time_of_use_24h
                    if isinstance(tou, (list, tuple)) and len(tou) == 24:
                        dc_price[int(dc_id)] = float(tou[epoch_hour])
                    else:
                        # fallback: flat price
                        dc_price[int(dc_id)] = float(getattr(dc, "time_of_use_24h", 0.1))
                except Exception:
                    dc_price[int(dc_id)] = 0.1
        except Exception:
            # Fallback: keep the old behavior based on node_properties only
            dcs = _discover_dcs_from_node_props(node_properties)
            if not dcs:
                dcs = (
                    sorted(
                        df["source_dc_id"].unique().astype(int).tolist()
                        or [0]
                    )
                )
            cap_tps = _capacity_per_dc(node_properties, dcs)
            latency_mat = None
            dc_price = {int(dc): 0.1 for dc in dcs}

            # And build a default simulator so we can still run the epoch
            sim = LLM_Simulator(
                spec_dir="sim_specs",
                epoch_length=DEFAULT_EPOCH_LEN,
                debug=False,
            )
            epoch_len = DEFAULT_EPOCH_LEN
            delay_budget_sec = DEFAULT_DELAY_BUDGET_SEC

        if not dcs:
            dcs = sorted(df["source_dc_id"].unique().astype(int).tolist() or [0])
        for dc in dcs:
            cap_tps.setdefault(int(dc), 1.0)
            dc_price.setdefault(int(dc), 0.1)

        epoch_len_s = float(getattr(sim, "epoch_length", epoch_len))
        dcs_set = set(dcs)

        # 3) PerLLM-style routing: constraint satisfaction on delay, then minimize energy
        pending_tokens_by_dc: Dict[int, float] = {dc: 0.0 for dc in dcs}
        routed_tokens_by_dc: Dict[int, float] = {dc: 0.0 for dc in dcs}
        frac_plan: Dict[Tuple[int, str], Dict[int, float]] = {}

        for row in work_df.itertuples(index=False):
            src_dc = int(getattr(row, "src_dc"))
            model = str(getattr(row, "model_type"))
            tokens = float(getattr(row, "total_tokens"))

            # For each DC (arm), estimate:
            #   - queueing time based on current pending tokens and capacity
            #   - network latency
            #   - total delay; enforce delay_budget_sec
            #   - energy cost surrogate: TOU price * (service time)
            feasible_arms: List[Tuple[int, float, float]] = []  # (dc, delay_s, energy_score)
            all_arms: List[Tuple[int, float, float]] = []

            for dc in dcs:
                cap_tokens_epoch = cap_tps.get(dc, 1e-6)
                cap_tokens_per_s = cap_tokens_epoch / max(epoch_len_s, 1e-6)

                # queue + service time (very coarse M/M/1 style surrogate)
                queued_tokens = pending_tokens_by_dc[dc]
                service_tokens = queued_tokens + tokens
                service_time_s = service_tokens / max(cap_tokens_per_s, 1e-6)

                # network latency component (edge-cloud)
                if latency_mat is not None:
                    try:
                        lat_ms = float(latency_mat[src_dc][dc])
                    except Exception:
                        lat_ms = 0.0
                else:
                    lat_ms = 0.0
                latency_s = max(0.0, lat_ms / 1000.0)

                total_delay_s = service_time_s + latency_s

                # energy cost surrogate: price * service_time (PerLLM: minimize energy cost)
                price = float(dc_price.get(dc, 0.1))
                energy_score = price * service_time_s * DEFAULT_ENERGY_WEIGHT

                all_arms.append((dc, total_delay_s, energy_score))

                # Constraint satisfaction: keep only arms within delay budget
                if total_delay_s <= delay_budget_sec:
                    feasible_arms.append((dc, total_delay_s, energy_score))

            # If no arm satisfies the delay constraint, fall back to the smallest-delay DC
            if feasible_arms:
                # Among feasible arms, choose the one with minimal energy cost
                chosen_dc, _, _ = min(feasible_arms, key=lambda tup: tup[2])
            else:
                # Constraint-relaxation: pick minimal delay arm, even if it violates
                chosen_dc, _, _ = min(all_arms, key=lambda tup: tup[1])

            # Update pending and routed tokens
            chosen_dc = int(chosen_dc)
            pending_tokens_by_dc[chosen_dc] += tokens
            routed_tokens_by_dc[chosen_dc] += tokens

            key = (src_dc, model)
            d = frac_plan.setdefault(key, {})
            d[chosen_dc] = d.get(chosen_dc, 0.0) + tokens  # proportional to actual work

        # Normalize weights to fractions (per (src, model))
        for key, dist in frac_plan.items():
            s = sum(dist.values())
            if s > 0:
                for dc in list(dist.keys()):
                    dist[dc] = dist[dc] / s

        # 4) Build a simple power plan from routed shares (energy-aware)
        total_tokens = sum(routed_tokens_by_dc.values()) or 1.0
        routed_share = {
            dc: (routed_tokens_by_dc[dc] / total_tokens) for dc in dcs
        }
        power_plan = _build_power_plan(
            routed_share,
            epoch_summary if isinstance(epoch_summary, dict) else {},
        )

        # 5) Convert the fraction plan into a per-request 'map' (deterministic argmax)
        FIXED_VARIANT = "_FP16 (Base)_B16"
        req_rows = []
        plan_map: Dict[int, int] = {}
        for row_idx, r in enumerate(df.itertuples(index=False)):
            src_dc = int(getattr(r, "source_dc_id"))
            model = str(getattr(r, "model_type"))
            tokens = int(max(0.0, round(float(getattr(r, "num_tokens")))))
            arrival_ms = float(getattr(r, "arrival_ms"))

            full_model = f"{model}{FIXED_VARIANT}"
            req_rows.append(
                {
                    "source_dc": src_dc,
                    "model": full_model,
                    "arrival_ms": arrival_ms,
                    "tokens": tokens,
                }
            )

            choices = frac_plan.get((src_dc, model), {})
            if choices:
                # pick highest fraction; break ties by smallest dc_id
                tgt = sorted(
                    choices.items(), key=lambda kv: (-kv[1], kv[0])
                )[0][0]
            else:
                tgt = src_dc if src_dc in dcs_set else dcs[0]
            plan_map[row_idx] = int(tgt)

        requests_df = pd.DataFrame(req_rows)
        schedule_plan = {"map": plan_map}

        # 6) Run the simulator for this epoch (per-request path)
        metrics, details, leftovers = sim.run_epoch(
            epoch_idx, requests_df, schedule_plan, power_plan
        )

        # 7) Normalize & return
        stats, results, leftovers_norm = _normalize_sim_output(
            (metrics, details, leftovers)
        )
        return stats, results, leftovers_norm
