#!/usr/bin/env python3

from __future__ import annotations
import argparse
import sys
import os
from typing import Optional

import pandas as pd
import numpy as np

# --------- Defaults ---------
DEFAULT_INPUT = "BurstGPT_1.csv"
DEFAULT_OUTPUT = "simulator_ready_trace.csv"
DEFAULT_EPOCH_LENGTH = 900  # seconds (15 minutes)
DEFAULT_NUM_DCS = 12

# --- Scenario Injection Configuration ---
SCENARIO_PROBS = {
    "Chat": 0.60,
    "Summarization": 0.30,
    "Novel": 0.10
}

DEFAULT_POPULATION_WEIGHTS = {
    0: 0.12, 1: 0.10, 2: 0.08, 3: 0.15, 4: 0.10, 5: 0.05,
    6: 0.18, 7: 0.08, 8: 0.06, 9: 0.04, 10: 0.02, 11: 0.02
}


# --------- Helpers ---------
def _map_model_to_base(m: str) -> str:
    """
    Maps the raw 'Model' column ('ChatGPT' or 'GPT-4')
    to the simulator's hardware classes.
    """
    if not isinstance(m, str): return "Llama7b"
    s = m.strip().lower()
    if "gpt-4" in s or "gpt4" in s:
        return "Llama70b"
    if "chatgpt" in s or "gpt-3.5" in s or "gpt3.5" in s:
        return "Llama7b"
    return "Llama7b"  # Default fallback


# --------- Core processing ---------
def process_trace_per_request(
        trace: pd.DataFrame,
        epoch_length: int,
        num_dcs: int,
        distribution: str = "even",
        population_weights: Optional[dict] = None,
        max_epochs: Optional[int] = None
) -> pd.DataFrame:
    # ---- 0. DATA CLEANING (Filter out Failures) ----
    # Drop rows where 'Response tokens' is 0
    if "Response tokens" in trace.columns:
        initial_len = len(trace)
        # Coerce to numeric in case of corrupted lines, then filter > 0
        trace = trace[pd.to_numeric(trace["Response tokens"], errors='coerce') > 0].copy()
        dropped = initial_len - len(trace)
        print(f"[clean] Dropped {dropped} failed requests (Response tokens == 0)")
    else:
        print("[WARNING] 'Response tokens' column not found. Skipping failure filter.")

    # ---- 1. Map Explicit Columns ----
    ts_col = "Timestamp"
    model_col = "Model"
    req_tok_col = "Request tokens"
    resp_tok_col = "Response tokens"
    tot_tok_col = "Total tokens"

    # ---- 2. Timestamps -> Epoch Index ----
    # Timestamp: request submission time, seconds from 0:00:00 on the first day
    t = pd.to_numeric(trace[ts_col], errors="coerce").fillna(0.0)
    min_time = float(np.nanmin(t.values))
    epoch = ((t - min_time) // epoch_length).astype(int)

    # ---- 3. Token Extraction ----
    # We grab total, prompt, and gen tokens so advanced algorithms (like Splitwise)
    # can use the exact split rather than estimating.
    tot_toks = pd.to_numeric(trace[tot_tok_col], errors="coerce").fillna(0.0)
    prompt_toks = pd.to_numeric(trace[req_tok_col], errors="coerce").fillna(0.0)
    gen_toks = pd.to_numeric(trace[resp_tok_col], errors="coerce").fillna(0.0)

    # ---- 4. Model Intent Generation ----
    # Map ChatGPT/GPT-4 to Llama7b/Llama70b
    base_models = trace[model_col].astype(str).map(_map_model_to_base)

    # Inject Scenarios (Chat, Summarization, Novel)
    scenarios = np.random.choice(
        list(SCENARIO_PROBS.keys()),
        size=len(trace),
        p=list(SCENARIO_PROBS.values())
    )

    # Compound Intent String (e.g. "Llama7b_Chat")
    model_intents = base_models.astype(str) + "_" + scenarios

    # ---- 5. Source DC Assignment ----
    if distribution == "population":
        weights = population_weights if population_weights else DEFAULT_POPULATION_WEIGHTS
        available_weights = {k: v for k, v in weights.items() if k < num_dcs}
        total = sum(available_weights.values())
        available_weights = {k: v / total for k, v in available_weights.items()} if total > 0 else {i: 1.0 / num_dcs for
                                                                                                    i in range(num_dcs)}
        dc_ids, dc_probs = list(available_weights.keys()), list(available_weights.values())

        src_dc = np.zeros(len(trace), dtype=int)
        for ep_val in epoch.unique():
            mask = (epoch == ep_val)
            np.random.seed(int(ep_val) * 42)
            src_dc[mask] = np.random.choice(dc_ids, size=mask.sum(), p=dc_probs)
        src_dc = pd.Series(src_dc)
    else:
        # Even (Round-Robin) distribution
        src_dc = (np.arange(len(trace), dtype=np.int64) % int(num_dcs)).astype(int)

    # ---- 6. Build Output ----
    out = pd.DataFrame({
        "epoch": epoch.astype(int),
        "source_dc_id": src_dc.astype(int),
        "model_type": model_intents,
        "arrival_ms": 0,  # Rate-flow simulator evaluates entire epochs at once
        "num_tokens": tot_toks.astype(int),
        "prompt_tokens": prompt_toks.astype(int),
        "gen_tokens": gen_toks.astype(int),
    }).sort_values(["epoch", "source_dc_id"]).reset_index(drop=True)

    # ---- 7. Truncation for RL Training ----
    if max_epochs is not None and max_epochs > 0:
        print(f"[clean] Truncating trace to first {max_epochs} epochs...")
        out = out[out["epoch"] < max_epochs].copy()

    return out


# --------- CLI ---------
def main():
    ap = argparse.ArgumentParser(description="Convert BurstGPT trace to per-request CSV for the simulator.")
    ap.add_argument("--input", default=DEFAULT_INPUT, help="Input CSV (raw trace)")
    ap.add_argument("--output", default=DEFAULT_OUTPUT, help="Output CSV (per-request)")
    ap.add_argument("--epoch-length", type=int, default=DEFAULT_EPOCH_LENGTH, help="Epoch length in seconds")
    ap.add_argument("--num-dcs", type=int, default=DEFAULT_NUM_DCS, help="Number of datacenters")
    ap.add_argument("--distribution", type=str, default="even", choices=["even", "population"])
    ap.add_argument("--population-weights", type=str, default=None)
    ap.add_argument("--max-epochs", type=int, default=None,
                    help="Truncate the dataset to this many epochs (96 = 24 hours)")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"[load] Loading {args.input} (This may take a moment for large files...)")
    trace = pd.read_csv(args.input)

    population_weights = None
    if args.population_weights:
        import json
        population_weights = {int(k): float(v) for k, v in json.loads(args.population_weights).items()}

    print(f"[process] Intent Injection Enabled | epoch_length={args.epoch_length}s")
    per_req = process_trace_per_request(
        trace=trace,
        epoch_length=args.epoch_length,
        num_dcs=args.num_dcs,
        distribution=args.distribution,
        population_weights=population_weights,
        max_epochs=args.max_epochs
    )

    print(f"[save] Writing to {args.output} (rows={len(per_req)})")
    per_req.to_csv(args.output, index=False)

    print("\n[preview] Top 5 Model Intents Generated:")
    print(per_req["model_type"].value_counts().head(5))

    print("\n[preview] Average Token Split:")
    print(f"  Prompt: {per_req['prompt_tokens'].mean():.1f}")
    print(f"  Generation: {per_req['gen_tokens'].mean():.1f}")


if __name__ == "__main__":
    main()