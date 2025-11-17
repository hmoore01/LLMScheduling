#!/usr/bin/env python3
# Helix.py — Helix-style scheduling wrapper around Rate_Flow_Sim.LLM_Simulator

from __future__ import annotations
from typing import Any, Dict, List, Tuple
import pandas as pd

from Rate_Flow_Sim import LLM_Simulator

# -----------------------------
# Defaults / knobs
# -----------------------------
DEFAULT_EPOCH_LEN = 900
DEFAULT_NODE_TYPES = [0, 1, 2, 3, 4, 5]  # for simple power-plan heuristic


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
    }

    if not isinstance(details, list):
        details = []

    return stats, details, leftovers


# -----------------------------
# DC discovery from node_properties
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
    This gives us a Helix-style "capacity" for the max-flow style heuristic.
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
) -> Dict[int, Dict[int, str]]:
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

    power_plan: Dict[int, Dict[int, str]] = {}
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
                dc_power[nt] = "Idle"
            else:
                dc_power[nt] = "Off"
        power_plan[int(dc)] = dc_power

    return power_plan


# -----------------------------
# Helix class wrapper
# -----------------------------
class Helix:
    @staticmethod
    def milp_optimizer(
        epoch_data,
        epoch_idx: int,
        node_properties,
        epoch_summary: Any,
    ):
        """
        Build a schedule + power plan and run the LLM_Simulator on a per-request path.

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
            }
            return empty_stats, [], []

        # 2) Discover DCs and capacities using the simulator's GPU perf tables
        try:
            if isinstance(epoch_summary, dict):
                spec_dir = epoch_summary.get("spec_dir", "sim_specs")
                epoch_len = int(
                    epoch_summary.get("epoch_length", DEFAULT_EPOCH_LEN)
                )
            else:
                spec_dir = "sim_specs"
                epoch_len = DEFAULT_EPOCH_LEN

            sim = LLM_Simulator(
                spec_dir=spec_dir, epoch_length=epoch_len, debug=False
            )

            dcs = sorted(int(dc_id) for dc_id in sim.datacenters.keys())
            cap_tps = _capacity_per_dc_from_sim(sim)
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

            # And build a default simulator so we can still run the epoch
            sim = LLM_Simulator(
                spec_dir="sim_specs",
                epoch_length=DEFAULT_EPOCH_LEN,
                debug=False,
            )

        # 3) Greedy routing "fraction plan" per (src, model) -> {tgt_dc: frac}
        pending: Dict[int, float] = {dc: 0.0 for dc in dcs}
        routed_tokens_by_dc: Dict[int, float] = {dc: 0.0 for dc in dcs}
        frac_plan: Dict[Tuple[int, str], Dict[int, float]] = {}

        for row in work_df.itertuples(index=False):
            src_dc = int(getattr(row, "src_dc"))
            model = str(getattr(row, "model_type"))
            tokens = float(getattr(row, "total_tokens"))

            # choose DC with minimal load ratio
            best_dc, best_score = None, None
            for dc in dcs:
                score = pending[dc] / cap_tps[dc]
                if (best_score is None) or (score < best_score):
                    best_score, best_dc = score, dc

            tgt = int(best_dc)
            pending[tgt] += tokens
            routed_tokens_by_dc[tgt] += tokens

            key = (src_dc, model)
            d = frac_plan.setdefault(key, {})
            d[tgt] = d.get(tgt, 0.0) + 1.0

            # light smoothing so one huge bucket doesn't dominate
            pending[tgt] = max(0.0, pending[tgt] - cap_tps[tgt])

        # Normalize weights to fractions
        for key, dist in frac_plan.items():
            s = sum(dist.values())
            if s > 0:
                for dc in list(dist.keys()):
                    dist[dc] = dist[dc] / s

        # 4) Build a simple power plan from routed shares
        total_tokens = sum(routed_tokens_by_dc.values()) or 1.0
        routed_share = {
            dc: (routed_tokens_by_dc[dc] / total_tokens) for dc in dcs
        }
        power_plan = _build_power_plan(
            routed_share,
            epoch_summary if isinstance(epoch_summary, dict) else {},
        )

        # 5) Convert the fraction plan into a per-request 'map' (deterministic argmax)
        req_rows = []
        plan_map: Dict[int, int] = {}
        row_idx = 0
        for r in work_df.itertuples(index=False):
            src_dc = int(getattr(r, "src_dc"))
            model = str(getattr(r, "model_type"))
            tokens = int(getattr(r, "total_tokens"))
            req_rows.append(
                {
                    "source_dc": src_dc,
                    "model": model,
                    "arrival_ms": 0,
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
                tgt = src_dc
            plan_map[row_idx] = int(tgt)
            row_idx += 1

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




