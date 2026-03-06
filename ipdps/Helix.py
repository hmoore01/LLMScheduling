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
DEFAULT_NODE_TYPES = [0, 1, 2, 3, 4, 5]

# -----------------------------
# Configuration
# -----------------------------
# Fixed Variant: Full Model (FP16), Reasonable Batch Size (32)
FIXED_VARIANT = "_FP16 (Base)_B16"


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
        "requests_completed": float(metrics.get("requests_completed", 0.0)),
        "requests_dropped": float(metrics.get("requests_dropped", 0.0)),
    }

    if not isinstance(details, list):
        details = []

    return stats, details, leftovers


# -----------------------------
# DC discovery from node_properties
# -----------------------------
def _discover_dcs_from_node_props(node_properties) -> List[int]:
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
            for rec in perf.values():
                ms_per_tok = 0.0
                try:
                    ms_per_tok = float(rec.get("ms_per_token", 0.0))
                except Exception:
                    ms_per_tok = 0.0
                if ms_per_tok <= 0.0:
                    try:
                        ms_req = float(rec.get("ms_per_request", 0.0))
                        avg_tok = float(rec.get("avg_tokens_per_request", rec.get("avg_tokens_per_req", 0.0)))
                    except Exception:
                        ms_req, avg_tok = 0.0, 0.0
                    if ms_req > 0.0 and avg_tok > 0.0:
                        ms_per_tok = ms_req / avg_tok
                if ms_per_tok > 0.0:
                    tokens_per_ms = 1.0 / ms_per_tok
                    total_tokens += tokens_per_ms * epoch_ms
        caps[int(dc_id)] = total_tokens

    if not any(v > 0.0 for v in caps.values()):
        for dc_id in caps:
            caps[dc_id] = 1.0
    for dc_id in caps:
        caps[dc_id] = max(1e-6, caps[dc_id])

    return caps


# -----------------------------
# Power plan (Idle/Off per node type)
# -----------------------------
def _build_power_plan(
    routed_token_share_by_dc: Dict[int, float],
    epoch_summary: Any,
) -> Dict[int, Dict[str, Dict[int, str]]]:
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
        # 0) Normalize epoch rows
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
            return {}, [], []

        # 2) Initialize Simulator to discover Capacities
        spec_dir = "sim_specs"
        epoch_len = DEFAULT_EPOCH_LEN
        a100_csv = None
        h100_csv = None

        if isinstance(epoch_summary, dict):
            spec_dir = epoch_summary.get("spec_dir", "sim_specs")
            epoch_len = int(epoch_summary.get("epoch_length", DEFAULT_EPOCH_LEN))
            a100_csv = epoch_summary.get("a100_csv")
            h100_csv = epoch_summary.get("h100_csv")

        try:
            sim = LLM_Simulator(
                spec_dir=spec_dir,
                epoch_length=epoch_len,
                debug=False,
                a100_csv=a100_csv,
                h100_csv=h100_csv
            )
            dcs = sorted(int(dc_id) for dc_id in sim.datacenters.keys())
            cap_tps = _capacity_per_dc_from_sim(sim)

        except Exception as e:
            print(f"[Helix] Init warning: {e}. Using node property fallbacks.")
            dcs = _discover_dcs_from_node_props(node_properties)
            cap_tps = _capacity_per_dc(node_properties, dcs)
            sim = LLM_Simulator(
                spec_dir=spec_dir,
                epoch_length=epoch_len,
                debug=False,
                a100_csv=a100_csv,
                h100_csv=h100_csv
            )

        # 3) Greedy Routing (Load Balancing)
        pending = {dc: 0.0 for dc in dcs}
        frac_plan = {}

        for row in work_df.itertuples(index=False):
            best_dc = None
            best_score = float('inf')

            # Safe capacity division
            for dc in dcs:
                cap = cap_tps.get(dc, 1e-6)
                score = pending.get(dc, 0.0) / cap
                if score < best_score:
                    best_score, best_dc = score, dc

            tgt = int(best_dc if best_dc is not None else dcs[0])
            pending[tgt] += row.total_tokens

            key = (row.src_dc, row.model_type)
            d = frac_plan.setdefault(key, {})
            d[tgt] = d.get(tgt, 0.0) + 1.0

        # 4) Power Plan
        total_tokens = work_df["total_tokens"].sum() or 1.0
        routed_share = {dc: (pending.get(dc, 0.0) / total_tokens) for dc in dcs}
        power_plan = _build_power_plan(routed_share, epoch_summary if isinstance(epoch_summary, dict) else {})

        # 5) Build Request List with FIXED VARIANT (preserve per-request arrivals/tokens)
        req_rows = []
        plan_map = {}
        row_idx = 0

        for r in df.itertuples(index=False):
            # Deterministic Routing
            src_dc = int(r.source_dc_id)
            model = str(r.model_type)
            choices = frac_plan.get((src_dc, model), {})
            if choices:
                tgt = sorted(choices.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
            else:
                tgt = src_dc

            # --- FORCE FIXED VARIANT (No dynamic optimization) ---
            full_model_str = f"{model}{FIXED_VARIANT}"

            req_rows.append({
                "source_dc": src_dc,
                "model": full_model_str,
                "arrival_ms": float(r.arrival_ms),
                "tokens": float(r.num_tokens),
            })
            plan_map[row_idx] = int(tgt)
            row_idx += 1

        requests_df = pd.DataFrame(req_rows)
        schedule_plan = {"map": plan_map}

        # 6) Run Simulator
        metrics, details, leftovers = sim.run_epoch(
            epoch_idx, requests_df, schedule_plan, power_plan
        )

        stats, results, leftovers_norm = _normalize_sim_output((metrics, details, leftovers))
        return stats, results, leftovers_norm
