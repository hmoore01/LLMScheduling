#!/usr/bin/env python3

from __future__ import annotations
from typing import Any, Dict, List, Tuple

import pandas as pd

from Rate_Flow_Sim import LLM_Simulator

# -----------------------------
# Defaults / knobs
# -----------------------------
DEFAULT_EPOCH_LEN = 900
W_TOKEN = 1.5
W_LAT = 0.001

# [FIX 1] Disable the suffix so model names match the trace (e.g., "Llama7b_Chat")
FIXED_VARIANT = ""

def _prepare_epoch_df(epoch_data: Any) -> pd.DataFrame:
    if isinstance(epoch_data, pd.DataFrame):
        df = epoch_data.copy()
    else:
        df = pd.DataFrame(epoch_data)

    col_renames = {}
    if "source_dc_id" in df.columns and "source_dc" not in df.columns: col_renames["source_dc_id"] = "source_dc"
    if "model_type" in df.columns and "model" not in df.columns: col_renames["model_type"] = "model"
    if "num_tokens" in df.columns and "tokens" not in df.columns: col_renames["num_tokens"] = "tokens"
    if col_renames: df = df.rename(columns=col_renames)

    required = ["source_dc", "model", "tokens"]
    missing = [c for c in required if c not in df.columns]
    if missing: raise ValueError(f"Splitwise: missing columns {missing}")

    if "arrival_ms" not in df.columns: df["arrival_ms"] = 0
    df["source_dc"] = pd.to_numeric(df["source_dc"], errors="coerce").fillna(0).astype(int)
    df["model"] = df["model"].astype(str)
    df["tokens"] = pd.to_numeric(df["tokens"], errors="coerce").fillna(0).astype(int)
    df["arrival_ms"] = pd.to_numeric(df["arrival_ms"], errors="coerce").fillna(0).astype(int)
    return df


def _capacity_per_dc_from_sim(sim: LLM_Simulator) -> Dict[int, float]:
    caps: Dict[int, float] = {}
    epoch_ms = float(getattr(sim, "epoch_length", DEFAULT_EPOCH_LEN)) * 1000.0

    for dc_id, dc in getattr(sim, "datacenters", {}).items():
        total_tokens = 0.0
        units = getattr(dc, "units", [])
        for u in units:
            perf = getattr(u, "model_perf", {})
            if not isinstance(perf, dict): continue

            node_peak_tpm = 0.0
            for rec in perf.values():
                ms_per_tok = float(rec.get("ms_per_token", 0.0))
                if ms_per_tok <= 0.0:
                    try:
                        ms_req = float(rec.get("ms_per_request", 0.0))
                        batch = int(rec.get("batch_size", 1))
                        if ms_req > 0: ms_per_tok = ms_req / (batch * 1000.0)
                    except:
                        pass

                if ms_per_tok > 0.0:
                    tpm = 1.0 / ms_per_tok
                    if tpm > node_peak_tpm: node_peak_tpm = tpm

            total_tokens += node_peak_tpm * epoch_ms

        caps[int(dc_id)] = total_tokens

    if not caps: return {}
    if not any(v > 0.0 for v in caps.values()):
        for dc_id in caps: caps[dc_id] = 1.0
    for dc_id in caps: caps[dc_id] = max(1e-6, float(caps[dc_id]))
    return caps


def _build_power_plan_from_share(routed_token_share_by_dc: Dict[int, float]) -> Dict[int, Dict[str, Any]]:
    power_plan: Dict[int, Dict[str, Any]] = {}
    for dc, share in routed_token_share_by_dc.items():
        if share > 0.0:
            # [FIX 2] Active Data Centers must be "ON", not "IDLE"
            power_plan[int(dc)] = {"all": "ON"}
        else:
            power_plan[int(dc)] = {"all": "OFF"}
    return power_plan


class Splitwise:
    @staticmethod
    def milp_optimizer(epoch_data, epoch_idx, node_properties, epoch_summary):
        # 1) Normalize
        df = _prepare_epoch_df(epoch_data)
        if df.empty or df["tokens"].sum() <= 0:
            return {
                "avg_ttft": 0.0, "avg_e2e_latency": 0.0, "total_energy": 0.0,
                "carbon_emissions": 0.0, "water_usage": 0.0, "energy_cost": 0.0
            }, [], []

        # 2) Init Simulator
        spec_dir = epoch_summary.get("spec_dir", "sim_specs")
        epoch_len = int(epoch_summary.get("epoch_length", DEFAULT_EPOCH_LEN))

        sim = LLM_Simulator(
            spec_dir=spec_dir, epoch_length=epoch_len, debug=False,
            a100_csv=epoch_summary.get("a100_csv"), h100_csv=epoch_summary.get("h100_csv")
        )

        dcs = sorted(int(d) for d in sim.datacenters.keys()) or sorted(df["source_dc"].unique())
        cap_tokens = _capacity_per_dc_from_sim(sim)
        if not cap_tokens: cap_tokens = {d: 1.0 for d in dcs}
        for d in dcs: cap_tokens.setdefault(d, 1.0)

        prompt_cap = {d: 0.5 * v for d, v in cap_tokens.items()}
        token_cap = {d: v for d, v in cap_tokens.items()}

        ring_fn = None
        try:
            ring_fn = sim.network._ring_path_latency_ms
        except:
            pass

        # 3) Routing
        plan_map = {}
        routed_load = {d: 0.0 for d in dcs}
        req_rows = []

        default_in = float(epoch_summary.get("in_frac", 0.7))
        default_out = float(epoch_summary.get("out_frac", 0.3))

        for idx, row in enumerate(df.itertuples(index=False)):
            tokens = int(row.tokens) if row.tokens > 0 else 950
            in_tok = max(1.0, tokens * default_in)
            out_tok = max(1.0, tokens * default_out)

            best_dc, best_cost = None, None
            for dc in dcs:
                cost = (in_tok / prompt_cap[dc]) + W_TOKEN * (out_tok / token_cap[dc])
                if ring_fn and W_LAT != 0: cost += W_LAT * float(ring_fn(row.source_dc, dc))
                if best_cost is None or cost < best_cost: best_cost, best_dc = cost, dc

            tgt_dc = int(best_dc if best_dc is not None else row.source_dc)

            # --- FIXED VARIANT SELECTION ---
            # [FIX 1 Redux] Use row.model directly without appending garbage string
            full_model = f"{row.model}{FIXED_VARIANT}"

            req_rows.append({
                "source_dc": row.source_dc,
                "model": full_model,
                "arrival_ms": row.arrival_ms,
                "tokens": tokens
            })
            plan_map[idx] = tgt_dc
            routed_load[tgt_dc] += out_tok

        # 4) Run
        total = sum(routed_load.values()) or 1.0
        power_plan = _build_power_plan_from_share({k: v / total for k, v in routed_load.items()})

        metrics, details, leftovers = sim.run_epoch(
            epoch_idx, pd.DataFrame(req_rows), {"map": plan_map}, power_plan
        )

        stats = {
            "avg_ttft": float(metrics.get("avg_ttft", 0.0)),
            "avg_e2e_latency": float(metrics.get("avg_e2e_latency", metrics.get("avg_ttft", 0.0))),
            "energy_cost": float(metrics.get("energy_cost", 0.0)),
            "carbon_emissions": float(metrics.get("carbon_emissions", 0.0)),
            "water_usage": float(metrics.get("water_usage", 0.0)),
            "total_energy": float(metrics.get("total_energy", 0.0)),
            "processed_tokens": float(df["tokens"].sum())
        }
        return stats, details or [], leftovers