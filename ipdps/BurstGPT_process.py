#!/usr/bin/env python3

from __future__ import annotations
import argparse
import sys
import os
from typing import Optional

import pandas as pd
import numpy as np

# --------- Defaults ---------
DEFAULT_INPUT = "BurstGPT_without_fails_2.csv"
DEFAULT_OUTPUT = "simulator_ready_trace.csv"
DEFAULT_EPOCH_LENGTH = 900  # seconds
DEFAULT_NUM_DCS = 3

# --- Scenario Injection Configuration ---
# Probability distribution for assigning scenarios to raw trace requests
SCENARIO_PROBS = {
    "Chat": 0.60,  # High volume, latency sensitive
    "Summarization": 0.30,  # Medium volume, prefill heavy
    "Novel": 0.10  # Low volume, generation heavy
}


# --------- Helpers ---------
def _pick_col(df: pd.DataFrame, *candidates: str) -> Optional[str]:
    """Return the first matching column name from candidates, else None."""
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def _timestamp_to_seconds(series: pd.Series) -> pd.Series:
    """Convert a timestamp column to seconds (float)."""
    try:
        return series.astype(float)
    except Exception:
        pass

    ts = pd.to_datetime(series, utc=True, errors="coerce")
    if ts.isna().all():
        raise ValueError("All timestamps failed to parse.")
    return ts.view("int64") / 1e9  # ns -> s


def _map_model_to_base(m: str) -> str:
    """
    Normalize raw model names to simulator's Base Classes (7B or 70B).
    Crucial for hardware constraint enforcement.
    """
    if not isinstance(m, str):
        return "Llama7b"  # Default fallback

    s = m.strip().lower()

    # Explicit OpenAI mappings (Approximations for simulation)
    if "gpt-4" in s or "gpt4" in s:
        return "Llama70b"
    if "chatgpt" in s or "gpt-3.5" in s or "gpt3.5" in s:
        return "Llama7b"

    # Llama mappings
    if "70b" in s:
        return "Llama70b"
    if "7b" in s or "llama-2-7b" in s:
        return "Llama7b"

    # Size based heuristics if model name contains param count
    if "13b" in s or "30b" in s or "34b" in s:
        # Map mid-sized models to 70B class for conservative resource estimation
        # or 7B if you want to be optimistic. Let's map >13B to 70B class.
        return "Llama70b"

    # Default fallback
    return "Llama7b"


# Default population weights for major regions (normalized)
DEFAULT_POPULATION_WEIGHTS = {
    0: 0.12,  # US East
    1: 0.10,  # US West
    2: 0.08,  # US Central
    3: 0.15,  # Europe West
    4: 0.10,  # Europe Central
    5: 0.05,  # Europe North
    6: 0.18,  # Asia Pacific (China region)
    7: 0.08,  # Asia Pacific (Japan/Korea)
    8: 0.06,  # Asia Pacific (Southeast)
    9: 0.04,  # South America
    10: 0.02,  # Middle East
    11: 0.02,  # Africa
}


# --------- Core processing ---------
def process_trace_per_request(
        trace: pd.DataFrame,
        epoch_length: int,
        num_dcs: int,
        prefer_ms: bool = False,
        distribution: str = "even",
        population_weights: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Convert raw trace to per-request format with Intent Injection.

    Args:
        trace: Input DataFrame with raw trace data
        epoch_length: Length of each epoch in seconds
        num_dcs: Number of datacenters
        prefer_ms: If True, treat timestamps as milliseconds
        distribution: 'even' for round-robin, 'population' for weighted
        population_weights: Custom weights for population distribution
    """
    # ---- 1. Identify Columns ----
    model_col = _pick_col(trace, "Model", "model", "Model_Name", "model_name", "model_type")
    # If no model column, we assume Llama7b for all
    has_model_col = model_col is not None

    ts_col = _pick_col(trace, "Timestamp", "time", "arrival_s", "arrival", "arrival_time", "arrival_ms")
    if ts_col is None:
        raise KeyError(f"Could not find timestamp column in {list(trace.columns)}")

    total_tok_col = _pick_col(trace, "Total tokens", "total_tokens", "tokens", "toks", "num_tokens")
    p_col = _pick_col(trace, "Prompt tokens", "prompt_tokens", "input_tokens", "in_tokens", "prompt")
    o_col = _pick_col(trace, "Output tokens", "output_tokens", "gen_tokens", "out_tokens", "completion")

    # ---- 2. Timestamps -> Seconds ----
    t = trace[ts_col]
    t = _timestamp_to_seconds(t)
    if prefer_ms:
        t = t / 1000.0

    # ---- 3. Tokens per request ----
    if total_tok_col is not None:
        toks = pd.to_numeric(trace[total_tok_col], errors="coerce").fillna(0.0)
    else:
        p = pd.to_numeric(trace[p_col], errors="coerce").fillna(0.0) if p_col else 0.0
        o = pd.to_numeric(trace[o_col], errors="coerce").fillna(0.0) if o_col else 0.0
        toks = p + o

    # ---- 4. Epoch index ----
    min_time = float(np.nanmin(t.values))
    epoch = ((t - min_time) // epoch_length).astype(int)

    # ---- 5. Model Intent Generation (The "New Space" Logic) ----
    # A. Map to Base Class (7B vs 70B)
    if has_model_col:
        base_models = trace[model_col].astype(str).map(_map_model_to_base)
    else:
        base_models = pd.Series(["Llama7b"] * len(trace))

    # B. Inject Scenarios (Probabilistic)
    # We assign a scenario to every request based on defined probabilities
    scenarios = np.random.choice(
        list(SCENARIO_PROBS.keys()),
        size=len(trace),
        p=list(SCENARIO_PROBS.values())
    )

    # C. Create Compound Intent String (e.g. "Llama7b_Chat")
    # This matches what Helix.find_best_implementation() expects
    model_intents = base_models.astype(str) + "_" + scenarios

    # ---- 6. Source DC Assignment ----
    src_dc_col = _pick_col(trace, "Source_DC", "source_dc_id", "src_dc", "Src_DC")
    if src_dc_col is not None:
        src_dc = pd.to_numeric(trace[src_dc_col], errors="coerce").fillna(0).astype(int) % num_dcs
    elif distribution == "population":
        # Population-weighted distribution
        weights = population_weights if population_weights else DEFAULT_POPULATION_WEIGHTS
        # Normalize weights to available DCs
        available_weights = {k: v for k, v in weights.items() if k < num_dcs}
        total = sum(available_weights.values())
        if total > 0:
            available_weights = {k: v / total for k, v in available_weights.items()}
        else:
            available_weights = {i: 1.0 / num_dcs for i in range(num_dcs)}

        dc_ids = list(available_weights.keys())
        dc_probs = list(available_weights.values())

        # Per-epoch assignment for reproducibility
        src_dc = np.zeros(len(trace), dtype=int)
        for ep_val in epoch.unique():
            mask = (epoch == ep_val)
            np.random.seed(int(ep_val) * 42)
            src_dc[mask] = np.random.choice(dc_ids, size=mask.sum(), p=dc_probs)
        src_dc = pd.Series(src_dc)
    else:
        # Even (round-robin) distribution
        n = len(trace)
        src_dc = (np.arange(n, dtype=np.int64) % int(num_dcs)).astype(int)

    # ---- 7. Build Output ----
    out = pd.DataFrame({
        "epoch": epoch.astype(int),
        "source_dc_id": src_dc.astype(int),
        "model_type": model_intents,  # The new Intent Key
        "arrival_ms": 0,  # Force epoch alignment
        "num_tokens": pd.to_numeric(toks, errors="coerce").fillna(0.0).astype(float),
    }).sort_values(["epoch", "source_dc_id"]).reset_index(drop=True)

    return out


# --------- CLI ---------
def main():
    ap = argparse.ArgumentParser(description="Convert raw trace to per-request CSV for the simulator.")
    ap.add_argument("--input", default=DEFAULT_INPUT, help="Input CSV (raw trace)")
    ap.add_argument("--output", default=DEFAULT_OUTPUT, help="Output CSV (per-request)")
    ap.add_argument("--epoch-length", type=int, default=DEFAULT_EPOCH_LENGTH, help="Epoch length in seconds")
    ap.add_argument("--num-dcs", type=int, default=DEFAULT_NUM_DCS, help="Number of datacenters")
    ap.add_argument("--timestamps-in-ms", action="store_true", help="Treat input timestamps as milliseconds")
    ap.add_argument("--distribution", type=str, default="even", choices=["even", "population"],
                    help="Request origin distribution: even (round-robin) or population (weighted)")
    ap.add_argument("--population-weights", type=str, default=None,
                    help="JSON string of population weights per DC")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"[load] {args.input}")
    try:
        trace = pd.read_csv(args.input)
    except Exception as e:
        print(f"ERROR: failed to read input CSV: {e}", file=sys.stderr)
        sys.exit(1)

    # Parse population weights if provided
    population_weights = None
    if args.population_weights:
        import json
        try:
            population_weights = {int(k): float(v) for k, v in json.loads(args.population_weights).items()}
        except Exception as e:
            print(f"[WARNING] Failed to parse population weights: {e}. Using defaults.")
            population_weights = None

    print(f"[process] Intent Injection Enabled | epoch_length={args.epoch_length}s | distribution={args.distribution}")
    per_req = process_trace_per_request(
        trace=trace,
        epoch_length=args.epoch_length,
        num_dcs=args.num_dcs,
        prefer_ms=args.timestamps_in_ms,
        distribution=args.distribution,
        population_weights=population_weights,
    )

    print(f"[save] {args.output}  (rows={len(per_req)})")
    try:
        per_req.to_csv(args.output, index=False)
    except Exception as e:
        print(f"ERROR: failed to write output: {e}", file=sys.stderr)
        sys.exit(1)

    print("\n[preview] Generated Intents:")
    print(per_req["model_type"].value_counts().head(10))

    print("\n[preview] Source DC Distribution:")
    print(per_req["source_dc_id"].value_counts().sort_index())


if __name__ == "__main__":
    main()