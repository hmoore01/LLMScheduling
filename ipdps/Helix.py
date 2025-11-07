#!/usr/bin/env python3
# Helix.py — updated to use Rate_Flow_Sim.LLM_Simulator.run_epoch (per-request path)

from __future__ import annotations
from typing import Any, Dict, List, Tuple
import pandas as pd

# Import the new simulator (which loads CSV specs internally and exposes run_epoch)
from Rate_Flow_Sim import LLM_Simulator

# -----------------------------
# Defaults / knobs
# -----------------------------
DEFAULT_EPOCH_LEN = 900
DEFAULT_NODE_TYPES = [0, 1, 2, 3, 4, 5]  # for simple power-plan heuristic


# -----------------------------
# Column normalization
# -----------------------------
def _ensure_epoch_columns(df: pd.DataFrame, epoch_len: int) -> pd.DataFrame:
    """Normalize common aliases to canonical columns used downstream."""
    d = df.copy()
    col_map = {
        "src_dc": "source_dc_id",
        "src": "source_dc_id",
        "model": "model_type",
        "tokens": "num_tokens",
        "total_tokens": "num_tokens",
        "epoch_idx": "epoch",
        "epoch_id": "epoch",
    }
    for k, v in col_map.items():
        if k in d.columns and v not in d.columns:
            d = d.rename(columns={k: v})

    # Ensure presence and types
    if "time_index" not in d.columns:
        d["time_index"] = 0
    d["epoch"] = pd.to_numeric(d["epoch"], errors="coerce").fillna(0).astype(int)
    d["source_dc_id"] = pd.to_numeric(d["source_dc_id"], errors="coerce").fillna(0).astype(int)
    d["num_tokens"] = pd.to_numeric(d["num_tokens"], errors="coerce").fillna(0).astype(int)
    d["model_type"] = d["model_type"].astype(str)
    return d


# -----------------------------
# DC discovery / capacity helpers
# -----------------------------
def _discover_dcs_from_node_props(node_properties) -> List[int]:
    """Try to derive available DC ids from node_properties entries."""
    dcs = set()
    try:
        iterable = node_properties.values() if isinstance(node_properties, dict) else node_properties
        for p in iterable:
            try:
                dcs.add(int(p.get("datacenter_id")))
            except Exception:
                pass
    except Exception:
        pass
    return sorted(dcs)


def _discover_dcs_from_epoch(df: pd.DataFrame) -> List[int]:
    try:
        return sorted(df["source_dc_id"].dropna().astype(int).unique().tolist())
    except Exception:
        return []


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

    for p in iterable:
        try:
            dc = int(p.get("datacenter_id"))
            if dc in caps:
                caps[dc] += 1.0
                count_any = True
        except Exception:
            pass

    if not count_any:
        for dc in caps:
            caps[dc] = 1.0

    for dc in caps:
        caps[dc] = max(1e-6, caps[dc])
    return caps


# -----------------------------
# Power plan (Idle/Off per node type) scaled by routed share
# -----------------------------
def _build_power_plan(routed_token_share_by_dc: Dict[int, float], epoch_summary: Any) -> Dict[int, Dict[int, str]]:
    """
    Heuristic Idle/Off power plan scaled by per-DC share.
    epoch_summary may contain:
      - node_types (default DEFAULT_NODE_TYPES)
      - min_idle_types (default 1), max_idle_types (default len(node_types))
    """
    node_types = list(epoch_summary.get("node_types", DEFAULT_NODE_TYPES)) if isinstance(epoch_summary, dict) else list(DEFAULT_NODE_TYPES)
    min_idle = int(epoch_summary.get("min_idle_types", 1)) if isinstance(epoch_summary, dict) else 1
    max_idle = int(epoch_summary.get("max_idle_types", len(node_types))) if isinstance(epoch_summary, dict) else len(node_types)
    max_idle = max(1, min(max_idle, len(node_types)))

    total = sum(max(0.0, v) for v in routed_token_share_by_dc.values()) or 1.0
    shares = {dc: max(0.0, v) / total for dc, v in routed_token_share_by_dc.items()}

    power_plan: Dict[int, Dict[int, str]] = {}
    for dc_id, share in shares.items():
        if share <= 0.0:
            power_plan[dc_id] = {nt: "Off" for nt in node_types}
            continue
        k = min_idle + int(round((max_idle - min_idle) * share))
        k = max(min_idle, min(max_idle, k))
        plan = {nt: ("Idle" if idx < k else "Off") for idx, nt in enumerate(node_types)}
        power_plan[dc_id] = plan
    return power_plan


# -----------------------------
# Output normalization
# -----------------------------
def _normalize_sim_output(sim_out):
    """Return (stats, results, leftovers) with required keys present."""
    stats, results, leftovers = {}, [], []
    if isinstance(sim_out, tuple):
        if len(sim_out) > 0 and sim_out[0] is not None: stats = dict(sim_out[0])
        if len(sim_out) > 1 and sim_out[1] is not None: results = list(sim_out[1])
        if len(sim_out) > 2 and sim_out[2] is not None: leftovers = sim_out[2]
    elif isinstance(sim_out, dict):
        stats = dict(sim_out.get("metrics", {}))
        results = list(sim_out.get("results", []))
        leftovers = sim_out.get("leftover_requests", [])
    # Common keys
    stats.setdefault("avg_ttft", stats.get("avg_ttft", stats.get("avg_ttft_sec", 0.0)))
    stats.setdefault("energy_cost", stats.get("energy_cost", 0.0))
    stats.setdefault("carbon_emissions", stats.get("carbon_emissions", 0.0))
    stats.setdefault("water_usage", stats.get("water_usage", 0.0))
    return stats, results, leftovers


# ----------------- Public API -----------------
class Helix:
    @staticmethod
    def milp_optimizer(epoch_data, epoch_idx: int, node_properties, epoch_summary: Any):
        """
        Build a schedule + power plan and run the new LLM_Simulator on a **per-request** path.

        Returns: (stats, results, leftovers)
          - stats: dict with keys avg_ttft (s), carbon_emissions (g), water_usage (m^3), total_energy (kWh), energy_cost ($)
          - results: per-request details
          - leftovers: per-DC utilization or leftover map if provided by the simulator
        """

        # 0) Normalize epoch rows to ensure required columns exist
        if hasattr(epoch_data, "iterrows") and hasattr(epoch_data, "columns"):
            df = _ensure_epoch_columns(epoch_data, DEFAULT_EPOCH_LEN)
        else:
            df = pd.DataFrame(epoch_data)
            df = _ensure_epoch_columns(df, DEFAULT_EPOCH_LEN)

        # 1) Summarize to per-(src,model) buckets (Helix uses this for routing + power sizing)
        work_df = (
            df.groupby(["source_dc_id", "model_type"], as_index=False)["num_tokens"]
              .sum()
              .rename(columns={"source_dc_id": "src_dc", "num_tokens": "total_tokens"})
        )
        if work_df["total_tokens"].sum() <= 0:
            empty_stats = {
                "processed_tokens": 0.0, "avg_ttft_sec": 0.0, "energy_kwh": 0.0,
                "carbon_emissions": 0.0, "water_usage": 0.0
            }
            return empty_stats, [], []

        # 2) Discover DCs and capacities
        dcs = _discover_dcs_from_node_props(node_properties)
        if not dcs:
            dcs = sorted(df["source_dc_id"].unique().astype(int).tolist() or [0])
        cap_tps = _capacity_per_dc(node_properties, dcs)

        # 3) Build a greedy routing "fraction plan" per (src, model) -> {tgt_dc: frac}
        pending: Dict[int, float] = {dc: 0.0 for dc in dcs}
        routed_tokens_by_dc: Dict[int, float] = {dc: 0.0 for dc in dcs}
        frac_plan: Dict[Tuple[int, str], Dict[int, float]] = {}

        for row in work_df.itertuples(index=False):
            src_dc  = int(getattr(row, "src_dc"))
            model   = str(getattr(row, "model_type"))
            tokens  = float(getattr(row, "total_tokens"))

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
        shares = {dc: routed_tokens_by_dc[dc] / total_tokens for dc in routed_tokens_by_dc}
        power_plan = _build_power_plan(shares, epoch_summary if isinstance(epoch_summary, dict) else {})

        # 5) Convert the fraction plan into a per-request 'map' (deterministic argmax)
        #    Build one request per (src_dc, model) bucket, arrival at t=0
        req_rows = []
        plan_map: Dict[int, int] = {}
        row_idx = 0
        for r in work_df.itertuples(index=False):
            src_dc = int(getattr(r, "src_dc"))
            model  = str(getattr(r, "model_type"))
            tokens = int(getattr(r, "total_tokens"))
            req_rows.append({"source_dc": src_dc, "model": model, "arrival_ms": 0, "tokens": tokens})

            choices = frac_plan.get((src_dc, model), {})
            if choices:
                # pick highest fraction; break ties by smallest dc_id
                tgt = sorted(choices.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
            else:
                tgt = src_dc
            plan_map[row_idx] = int(tgt)
            row_idx += 1

        requests_df = pd.DataFrame(req_rows)  # columns: source_dc, model, arrival_ms
        schedule_plan = {"map": plan_map}

        # 6) Run the simulator for this epoch (per-request path)
        sim = LLM_Simulator(spec_dir="sim_specs", epoch_length=DEFAULT_EPOCH_LEN, debug=False)
        metrics, details, leftovers = sim.run_epoch(epoch_idx, requests_df, schedule_plan, power_plan)

        # 7) Normalize & return
        stats, results, leftovers_norm = _normalize_sim_output((metrics, details, leftovers))
        return stats, results, leftovers_norm


