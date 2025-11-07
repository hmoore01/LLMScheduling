#!/usr/bin/env python3
"""
BurstGPT_process.py — request-based only (even DC distribution)

Converts a raw trace CSV into a per-request CSV the simulator can consume.
All requests are forced to arrive at the start of their epoch (arrival_ms = 0).

If the raw trace does not include a source DC column, source_dc_id is assigned
evenly in round-robin order across [0..num_dcs-1].

Output columns:
  - epoch (int)
  - source_dc_id (int)
  - model_type (str)
  - arrival_ms (int)  # always 0
  - num_tokens (float)

CLI:
  python BurstGPT_process.py \
    --input RAW.csv \
    --output per_request.csv \
    --epoch-length 900 \
    --num-dcs 12
"""

from __future__ import annotations
import argparse
import sys
import os
from typing import Optional

import pandas as pd
import numpy as np


# --------- Defaults ---------
DEFAULT_INPUT = "BurstGPT_without_fails_2.csv"
DEFAULT_OUTPUT = "simulator_per_request.csv"
DEFAULT_EPOCH_LENGTH = 900          # seconds
DEFAULT_NUM_DCS = 12


# --------- Helpers ---------
def _pick_col(df: pd.DataFrame, *candidates: str) -> Optional[str]:
    """Return the first matching column name from candidates, else None."""
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def _timestamp_to_seconds(series: pd.Series) -> pd.Series:
    """Convert a timestamp column to seconds (float).

    Accepts:
      - numeric seconds already (returned as-is)
      - ISO8601/string timestamps (parsed via pandas.to_datetime, converted to epoch seconds)
    """
    # If it's numeric-ish, try returning float directly
    try:
        return series.astype(float)
    except Exception:
        pass

    # Otherwise, parse as datetime and convert to seconds (UTC-naive)
    ts = pd.to_datetime(series, utc=True, errors="coerce")
    if ts.isna().all():
        raise ValueError("All timestamps failed to parse.")
    return ts.view("int64") / 1e9  # ns -> s


def _map_model_to_llama(m: str) -> str:
    """Normalize model names to simulator's Llama variants.

    ChatGPT → Llama7b (light)
    GPT-4 → Llama70b (heavy)
    All other strings fall back to their closest Llama mapping.
    """
    if not isinstance(m, str):
        return str(m)

    s = m.strip().lower()

    # Explicit OpenAI model names
    if "chatgpt" in s or "gpt-3.5" in s or "gpt3.5" in s:
        return "Llama7b"
    if "gpt-4" in s or "gpt4" in s:
        return "Llama70b"

    # Generic Llama mappings
    if "70" in s or "70b" in s:
        return "Llama70b"
    if "7" in s and "70" not in s:
        return "Llama7b"
    if "llama-2-7b" in s or "llama2-7b" in s:
        return "Llama7b"
    if "llama-2-70b" in s or "llama2-70b" in s:
        return "Llama70b"

    return m.strip()


# --------- Core processing (request-based only) ---------
def process_trace_per_request(
    trace: pd.DataFrame,
    epoch_length: int,
    num_dcs: int,
    prefer_ms: bool = False,
) -> pd.DataFrame:
    """
    Convert raw trace to per-request format for the simulator.
    All requests arrive at the beginning of their epoch (arrival_ms = 0).

    Output columns:
      ['epoch', 'source_dc_id', 'model_type', 'arrival_ms', 'num_tokens']
    """
    # ---- Required-ish columns (with flexible names) ----
    model_col = _pick_col(trace, "Model", "model", "Model_Name", "model_name", "model_type")
    if model_col is None:
        raise KeyError(f"Could not find model column in {list(trace.columns)}")

    ts_col = _pick_col(trace, "Timestamp", "time", "arrival_s", "arrival", "arrival_time", "arrival_ms")
    if ts_col is None:
        raise KeyError(f"Could not find timestamp column in {list(trace.columns)}")

    total_tok_col = _pick_col(trace, "Total tokens", "total_tokens", "tokens", "toks", "num_tokens")
    p_col = _pick_col(trace, "Prompt tokens", "prompt_tokens", "input_tokens", "in_tokens", "prompt")
    o_col = _pick_col(trace, "Output tokens", "output_tokens", "gen_tokens", "out_tokens", "completion")

    # ---- Timestamps -> seconds ----
    t = trace[ts_col]
    t = _timestamp_to_seconds(t)
    if prefer_ms:
        t = t / 1000.0

    # ---- Tokens per request ----
    if total_tok_col is not None:
        toks = pd.to_numeric(trace[total_tok_col], errors="coerce").fillna(0.0)
    else:
        p = pd.to_numeric(trace[p_col], errors="coerce").fillna(0.0) if p_col else 0.0
        o = pd.to_numeric(trace[o_col], errors="coerce").fillna(0.0) if o_col else 0.0
        toks = p + o

    # ---- Epoch index (relative to min timestamp) ----
    min_time = float(np.nanmin(t.values))
    epoch = ((t - min_time) // epoch_length).astype(int)

    # ---- Model ----
    model_type = trace[model_col].astype(str).map(_map_model_to_llama)

    # ---- Source DC: use existing column if present, else even round-robin ----
    src_dc_col = _pick_col(trace, "Source_DC", "source_dc_id", "src_dc", "Src_DC")
    if src_dc_col is not None:
        src_dc = pd.to_numeric(trace[src_dc_col], errors="coerce").fillna(0).astype(int) % num_dcs
    else:
        # Even distribution by row order: 0,1,2,...,num_dcs-1, 0,1,2,... (round-robin)
        n = len(trace)
        src_dc = (np.arange(n, dtype=np.int64) % int(num_dcs)).astype(int)

    # ---- Build per-request output (arrival_ms forced to 0) ----
    out = pd.DataFrame({
        "epoch": epoch.astype(int),
        "source_dc_id": src_dc.astype(int),
        "model_type": model_type.astype(str),
        "arrival_ms": 0,                                # all requests arrive at epoch start
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
    ap.add_argument("--timestamps-in-ms", action="store_true",
                    help="Treat input timestamps as milliseconds explicitly (divide by 1000)")
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

    print(f"[process] request-based | epoch_length={args.epoch_length}s | num_dcs={args.num_dcs}")
    per_req = process_trace_per_request(
        trace=trace,
        epoch_length=args.epoch_length,
        num_dcs=args.num_dcs,
        prefer_ms=args.timestamps_in_ms,
    )

    print(f"[save] {args.output}  (rows={len(per_req)})")
    try:
        per_req.to_csv(args.output, index=False)
    except Exception as e:
        print(f"ERROR: failed to write output: {e}", file=sys.stderr)
        sys.exit(1)

    print("\n[preview] first 10 rows:")
    with pd.option_context("display.max_rows", 10, "display.max_columns", None, "display.width", 120):
        print(per_req.head(10))


if __name__ == "__main__":
    main()






