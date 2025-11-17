#!/usr/bin/env python3
"""
Splitwise.py — Splitwise-style scheduler for Rate_Flow_Sim.LLM_Simulator

API expected by simulator_LLM.py:
    from Splitwise import Splitwise
    stats, results, leftovers = Splitwise.milp_optimizer(
        epoch_data, epoch_idx, node_properties, epoch_summary
    )

- epoch_data: pd.DataFrame or iterable of dicts with at least:
    source_dc_id (or source_dc), model_type (or model), num_tokens (or tokens)
- node_properties: ignored here (kept for interface compatibility)
- epoch_summary: optional hints dict; ignored except for spec_dir / epoch_length
"""

from __future__ import annotations
from typing import Any, Dict, List, Tuple

import pandas as pd

from Rate_Flow_Sim import LLM_Simulator

# -----------------------------
# Defaults / knobs
# -----------------------------
DEFAULT_EPOCH_LEN = 900
# Weights for Splitwise objective
W_TOKEN = 1.5   # weight on generation tokens vs prompt tokens
W_LAT   = 0.001 # weight on ring latency (ms) in cost


# -----------------------------
# Helpers for epoch data
# -----------------------------
def _prepare_epoch_df(epoch_data: Any) -> pd.DataFrame:
    """Normalize epoch_data into a DataFrame with canonical columns."""
    if isinstance(epoch_data, pd.DataFrame):
        df = epoch_data.copy()
    else:
        df = pd.DataFrame(epoch_data)

    # Rename common aliases
    col_renames = {}
    if "source_dc_id" in df.columns and "source_dc" not in df.columns:
        col_renames["source_dc_id"] = "source_dc"
    if "model_type" in df.columns and "model" not in df.columns:
        col_renames["model_type"] = "model"
    if "num_tokens" in df.columns and "tokens" not in df.columns:
        col_renames["num_tokens"] = "tokens"
    if col_renames:
        df = df.rename(columns=col_renames)

    required = ["source_dc", "model", "tokens"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Splitwise.milp_optimizer: epoch_data missing required columns: {missing}")

    # Arrival time: optional, default to 0
    if "arrival_ms" not in df.columns:
        df["arrival_ms"] = 0

    # Type normalization
    df["source_dc"] = pd.to_numeric(df["source_dc"], errors="coerce").fillna(0).astype(int)
    df["model"] = df["model"].astype(str)
    df["tokens"] = pd.to_numeric(df["tokens"], errors="coerce").fillna(0).astype(int)
    df["arrival_ms"] = pd.to_numeric(df["arrival_ms"], errors="coerce").fillna(0).astype(int)

    return df


# -----------------------------
# Capacity estimation from simulator
# -----------------------------
def _capacity_per_dc_from_sim(sim: LLM_Simulator) -> Dict[int, float]:
    """
    Estimate per-DC capacity in tokens/epoch from GPU perf tables.

    For each ProcNode in each Datacenter we:
      - read ms_per_token if present,
      - else derive from ms_per_request / avg_tokens_per_request,
      - convert to tokens/ms, multiply by epoch_ms, sum across nodes.
    """
    caps: Dict[int, float] = {}
    epoch_ms = float(getattr(sim, "epoch_length", DEFAULT_EPOCH_LEN)) * 1000.0

    for dc_id, dc in getattr(sim, "datacenters", {}).items():
        total_tokens = 0.0
        units = getattr(dc, "units", [])
        for u in units:
            perf = getattr(u, "model_perf", {})
            if not isinstance(perf, dict):
                continue
            for rec in perf.values():
                ms_per_tok = None
                try:
                    ms_per_tok = float(rec.get("ms_per_token", None))
                except Exception:
                    ms_per_tok = None

                if ms_per_tok is None or ms_per_tok <= 0.0:
                    # Fallback: derive from request-level data if available
                    try:
                        ms_req = float(rec.get("ms_per_request", 0.0))
                        avg_tok = float(rec.get("avg_tokens_per_request", 0.0))
                    except Exception:
                        ms_req, avg_tok = 0.0, 0.0
                    if ms_req > 0.0 and avg_tok > 0.0:
                        ms_per_tok = ms_req / avg_tok

                if ms_per_tok and ms_per_tok > 0.0:
                    tokens_per_ms = 1.0 / ms_per_tok
                    total_tokens += tokens_per_ms * epoch_ms

        caps[int(dc_id)] = total_tokens

    # Fallback if all zeros
    if not caps:
        return {}
    if not any(v > 0.0 for v in caps.values()):
        for dc_id in caps:
            caps[dc_id] = 1.0

    # Avoid degenerate zeros
    for dc_id in caps:
        caps[dc_id] = max(1e-6, float(caps[dc_id]))

    return caps


# -----------------------------
# Power plan construction
# -----------------------------
def _build_power_plan_from_share(routed_token_share_by_dc: Dict[int, float]) -> Dict[int, Dict[str, Any]]:
    """
    Simple heuristic power plan:
      - DCs with non-zero share: keep all units IDLE (they will be used on demand).
      - DCs with zero share   : turn all units OFF.
    This matches Datacenter.apply_power_plan(plan_slice) expecting keys:
      - 'all': "ON" | "IDLE" | "OFF"
    """
    power_plan: Dict[int, Dict[str, Any]] = {}
    for dc, share in routed_token_share_by_dc.items():
        if share > 0.0:
            power_plan[int(dc)] = {"all": "IDLE"}
        else:
            power_plan[int(dc)] = {"all": "OFF"}
    return power_plan


# -----------------------------
# Splitwise class
# -----------------------------
class Splitwise:
    @staticmethod
    def milp_optimizer(
        epoch_data: Any,
        epoch_idx: int,
        node_properties: Any,
        epoch_summary: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Any]:
        """
        Co-located Splitwise scheduler:

        For each row in epoch_data:
          - treat it as an aggregate "request" with `tokens` total tokens,
          - split into prompt and generation tokens,
          - route to a single target DC that minimizes:

                cost(dc) = in_tok / prompt_cap[dc]
                           + W_TOKEN * out_tok / token_cap[dc]
                           + W_LAT   * ring_latency_ms(src_dc, dc)

        Then:
          - build schedule_plan = {'map': {row_idx -> target_dc}},
          - build a simple per-DC power_plan from routed token share,
          - call Rate_Flow_Sim.LLM_Simulator.run_epoch(...) and normalize outputs.
        """
        # 1) Normalize epoch data
        df = _prepare_epoch_df(epoch_data)

        if df.empty or df["tokens"].sum() <= 0:
            empty_stats: Dict[str, Any] = {
                "avg_ttft": 0.0,
                "avg_ttft_sec": 0.0,
                "energy_cost": 0.0,
                "carbon_emissions": 0.0,
                "water_usage": 0.0,
                "total_energy": 0.0,
                "energy_kwh": 0.0,
                "processed_tokens": 0.0,
            }
            return empty_stats, [], []

        # 2) Instantiate simulator & discover datacenters/capacities
        spec_dir = "sim_specs"
        epoch_len = DEFAULT_EPOCH_LEN
        if isinstance(epoch_summary, dict):
            spec_dir = epoch_summary.get("spec_dir", spec_dir)
            epoch_len = int(epoch_summary.get("epoch_length", epoch_len))

        sim = LLM_Simulator(
            spec_dir=spec_dir,
            epoch_length=epoch_len,
            debug=False,
        )

        dcs = sorted(int(dc_id) for dc_id in sim.datacenters.keys())
        if not dcs:
            # Fallback: infer from source_dc present in workload
            dcs = sorted(int(x) for x in df["source_dc"].unique().tolist())

        cap_tokens = _capacity_per_dc_from_sim(sim)
        if not cap_tokens:
            # Fallback to equal capacities if estimation failed
            cap_tokens = {dc: 1.0 for dc in dcs}
        else:
            # Ensure every DC in dcs has a capacity entry
            for dc in dcs:
                cap_tokens.setdefault(dc, 1.0)

        # Prompt capacity is a fraction of total; generation uses full capacity
        prompt_cap: Dict[int, float] = {}
        token_cap: Dict[int, float] = {}
        for dc in dcs:
            base = float(cap_tokens.get(dc, 1.0))
            base = max(1e-6, base)
            prompt_cap[dc] = 0.5 * base
            token_cap[dc] = base

        # Optional ring latency accessor
        ring_latency_fn = None
        try:
            net = getattr(sim, "network", None)
            if net is not None and hasattr(net, "_ring_path_latency_ms"):
                ring_latency_fn = net._ring_path_latency_ms  # type: ignore[attr-defined]
        except Exception:
            ring_latency_fn = None

        # 3) Per-request Splitwise routing (co-located)
        plan_map: Dict[int, int] = {}
        routed_tokens_by_dc: Dict[int, float] = {dc: 0.0 for dc in dcs}

        # Use rough split 70% prompt / 30% generation unless overridden
        default_in_frac = float(epoch_summary.get("in_frac", 0.7)) if isinstance(epoch_summary, dict) else 0.7
        default_out_frac = float(epoch_summary.get("out_frac", 0.3)) if isinstance(epoch_summary, dict) else 0.3
        if default_in_frac <= 0.0 and default_out_frac <= 0.0:
            default_in_frac, default_out_frac = 0.7, 0.3

        for row_idx, row in enumerate(df.itertuples(index=False), start=0):
            src_dc = int(getattr(row, "source_dc"))
            tokens = int(getattr(row, "tokens"))
            arrival_ms = int(getattr(row, "arrival_ms"))

            if tokens <= 0:
                tokens = int(epoch_summary.get("default_tokens", 950)) if isinstance(epoch_summary, dict) else 950

            in_tok = max(1.0, tokens * default_in_frac)
            out_tok = max(1.0, tokens * default_out_frac)

            best_dc = None
            best_cost = None

            for dc in dcs:
                # Core Splitwise objective: prompt term + weighted generation term
                cost = (in_tok / prompt_cap[dc]) + W_TOKEN * (out_tok / token_cap[dc])

                # Optional ring-latency penalty
                if ring_latency_fn is not None and W_LAT != 0.0:
                    try:
                        lat_ms = float(ring_latency_fn(src_dc, dc))
                    except Exception:
                        lat_ms = 0.0
                    cost += W_LAT * lat_ms

                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_dc = dc

            tgt_dc = int(best_dc if best_dc is not None else src_dc)
            plan_map[row_idx] = tgt_dc
            routed_tokens_by_dc[tgt_dc] += float(out_tok)

            # Update DataFrame row in-place for clarity (not strictly required)
            df.at[df.index[row_idx], "source_dc"] = src_dc
            df.at[df.index[row_idx], "tokens"] = tokens
            df.at[df.index[row_idx], "arrival_ms"] = arrival_ms

        # 4) Build power plan from routed token share
        total_out = sum(max(0.0, v) for v in routed_tokens_by_dc.values()) or 1.0
        share_by_dc = {dc: (max(0.0, routed_tokens_by_dc.get(dc, 0.0)) / total_out) for dc in dcs}
        power_plan = _build_power_plan_from_share(share_by_dc)

        # 5) Build schedule_plan & run simulator
        schedule_plan: Dict[str, Any] = {"map": plan_map}
        workload_df = df[["source_dc", "model", "arrival_ms", "tokens"]].copy()

        metrics, details, leftovers = sim.run_epoch(
            epoch_idx=epoch_idx,
            workload_df=workload_df,
            schedule_plan=schedule_plan,
            power_plan=power_plan,
        )

        # 6) Normalize outputs
        stats: Dict[str, Any] = {}
        if isinstance(metrics, dict):
            avg_ttft = float(metrics.get("avg_ttft", metrics.get("avg_ttft_sec", 0.0)))
            total_energy = float(metrics.get("total_energy", metrics.get("energy_kwh", 0.0)))

            stats = {
                "avg_ttft": avg_ttft,
                "avg_ttft_sec": avg_ttft,
                "energy_cost": float(metrics.get("energy_cost", 0.0)),
                "carbon_emissions": float(metrics.get("carbon_emissions", 0.0)),
                "water_usage": float(metrics.get("water_usage", 0.0)),
                "total_energy": total_energy,
                "energy_kwh": total_energy,
                "processed_tokens": float(df["tokens"].sum()),
            }
        else:
            # Fallback safe defaults
            stats = {
                "avg_ttft": 0.0,
                "avg_ttft_sec": 0.0,
                "energy_cost": 0.0,
                "carbon_emissions": 0.0,
                "water_usage": 0.0,
                "total_energy": 0.0,
                "energy_kwh": 0.0,
                "processed_tokens": float(df["tokens"].sum()),
            }

        if not isinstance(details, list):
            details = []

        return stats, details, leftovers

