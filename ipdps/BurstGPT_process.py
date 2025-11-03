#!/usr/bin/env python3
import argparse
import hashlib
import math
import sys
from typing import Dict, Optional

import pandas as pd

# -----------------------------
# Defaults (can be overridden by CLI)
# -----------------------------
DEFAULT_EPOCH_LENGTH = 900
DEFAULT_NUM_DCS = 12
DEFAULT_INPUT = "BurstGPT_without_fails_2.csv"
DEFAULT_OUTPUT = "simulator_ready_trace.csv"
DEFAULT_DEBUG_DETAILED = "simulator_ready_detailed.csv"

# -----------------------------
# Helpers
# -----------------------------
def _norm(s: str) -> str:
    return "".join(ch for ch in str(s).lower() if ch not in " _\t")

def _pick_col(df: pd.DataFrame, *cands: str) -> Optional[str]:
    cols = { _norm(c): c for c in df.columns }
    for c in cands:
        if _norm(c) in cols:
            return cols[_norm(c)]
    return None

def _timestamp_to_seconds(ts: pd.Series) -> pd.Series:
    """Heuristic: if the median timestamp is > 1e10, assume ms and convert to s."""
    med = float(ts.median())
    if med > 1e10:  # ~Sat Nov 20 2286 if seconds; so this flags ms
        return ts.astype(float) / 1000.0
    return ts.astype(float)

def _stable_dc_from_keys(row: pd.Series, num_dcs: int) -> int:
    """Consistent DC assignment using any stable keys if present; else row index."""
    for k in ["request_id", "id", "user", "uid", "session", "trace_id"]:
        if k in {c.lower(): c for c in row.index} and pd.notna(row[k]):
            h = hashlib.blake2b(str(row[k]).encode("utf-8"), digest_size=4).hexdigest()
            return int(h, 16) % num_dcs
    # fallback: round-robin by positional index if available
    # (the caller can pass the dataframe index as a column if desired)
    return int(row.get("_row_idx", 0)) % num_dcs

def _map_model_to_llama(model_str: str) -> str:
    s = str(model_str).lower()
    # Anything clearly "big" → Llama70b
    if "gpt-4" in s or "gpt4" in s or "70b" in s or "70 b" in s or "llama-3.1-70b" in s:
        return "Llama70b"
    # Otherwise treat as 7b-class
    return "Llama7b"

# -----------------------------
# Core processing
# -----------------------------
def process_trace_for_simulator(
    trace: pd.DataFrame,
    epoch_length: int,
    num_dcs: int,
    prefer_ms: bool = False,
) -> pd.DataFrame:
    """
    Convert a raw request trace to simulator rate-flow format.

    Returns aggregated DataFrame with columns:
      ['epoch', 'src_dc', 'model_type', 'total_tokens']
    """

    # ---- Identify columns (case- and underscore-insensitive) ----
    model_col = _pick_col(trace, "Model", "model", "Model_Name", "model_name", "model_type")
    if model_col is None:
        raise KeyError(f"Could not find model column in {list(trace.columns)}")

    ts_col = _pick_col(trace, "Timestamp", "time", "arrival_s", "arrival", "arrival_time", "arrival_ms")
    if ts_col is None:
        raise KeyError(f"Could not find timestamp column in {list(trace.columns)}")

    # Tokens: prefer a total, else sum prompt+output or input+output
    total_tok_col = _pick_col(trace, "Total tokens", "total_tokens", "tokens", "toks", "num_tokens")
    if total_tok_col is None:
        p_col = _pick_col(trace, "Prompt tokens", "prompt_tokens", "input_tokens", "in_tokens", "prompt")
        o_col = _pick_col(trace, "Output tokens", "output_tokens", "gen_tokens", "out_tokens", "completion")
        if p_col is None and o_col is None:
            raise KeyError(
                "Could not find tokens column(s). "
                "Expected 'Total tokens' or 'Prompt/Output tokens' in the input CSV."
            )

    src_dc_col = _pick_col(trace, "Source_DC", "source_dc_id", "src_dc", "Src_DC")

    # ---- Normalize timestamps ----
    t = trace[ts_col].astype(float)
    if prefer_ms:
        t = t / 1000.0
    else:
        t = _timestamp_to_seconds(t)

    # ---- Normalize tokens ----
    if total_tok_col is not None:
        toks = trace[total_tok_col].astype(float)
    else:
        # safe get; missing side becomes 0
        p_col = _pick_col(trace, "Prompt tokens", "prompt_tokens", "input_tokens", "in_tokens", "prompt")
        o_col = _pick_col(trace, "Output tokens", "output_tokens", "gen_tokens", "out_tokens", "completion")
        p = trace[p_col].astype(float) if p_col else 0.0
        o = trace[o_col].astype(float) if o_col else 0.0
        toks = p + o

    # ---- Compute epoch and time_index ----
    min_time = float(t.min())
    epoch = ((t - min_time) // epoch_length).astype(int)
    time_index = ((t - min_time) % epoch_length).astype(float)

    # ---- Model mapping ----
    model_type = trace[model_col].astype(str).map(_map_model_to_llama)

    # ---- Source DC mapping ----
    if src_dc_col is not None:
        src_dc = trace[src_dc_col].astype(int).mod(num_dcs)
    else:
        # Stable but data-driven: try hashing a stable id if present; else round-robin
        tmp = trace.copy()
        tmp["_row_idx"] = range(len(tmp))
        src_dc = tmp.apply(lambda r: _stable_dc_from_keys(r, num_dcs), axis=1).astype(int)

    # ---- Build detailed frame (optional to save for debugging) ----
    detailed = pd.DataFrame({
        "epoch": epoch,
        "model_type": model_type,
        "num_tokens": toks.astype(float),
        "time_index": time_index,
        "src_dc": src_dc,
        "batch_size": 1,  # downstream doesn’t need this in rate-flow; keep for audit
    })

    # ---- Aggregate to rate-flow format ----
    agg = (detailed
           .groupby(["epoch", "src_dc", "model_type"], as_index=False)["num_tokens"]
           .sum()
           .rename(columns={"num_tokens": "total_tokens"}))

    # Ensure proper dtypes
    agg["epoch"] = agg["epoch"].astype(int)
    agg["src_dc"] = agg["src_dc"].astype(int)
    agg["model_type"] = agg["model_type"].astype(str)
    agg["total_tokens"] = agg["total_tokens"].astype(float)

    return agg, detailed

# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Convert raw trace to simulator rate-flow format.")
    ap.add_argument("--input", default=DEFAULT_INPUT, help="Input CSV (raw trace)")
    ap.add_argument("--output", default=DEFAULT_OUTPUT, help="Output CSV (aggregated rate-flow)")
    ap.add_argument("--debug-detailed", default=DEFAULT_DEBUG_DETAILED,
                    help="Optional detailed per-request CSV for auditing")
    ap.add_argument("--epoch-length", type=int, default=DEFAULT_EPOCH_LENGTH, help="Epoch length in seconds")
    ap.add_argument("--num-dcs", type=int, default=DEFAULT_NUM_DCS, help="Number of datacenters")
    ap.add_argument("--timestamps-in-ms", action="store_true",
                    help="Treat input timestamps as milliseconds explicitly")
    ap.add_argument("--no-debug", action="store_true", help="Skip writing detailed debug CSV")
    args = ap.parse_args()

    print(f"[load] {args.input}")
    try:
        trace = pd.read_csv(args.input)
    except Exception as e:
        print(f"ERROR: failed to read input CSV: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"[process] epoch_length={args.epoch_length}s, num_dcs={args.num_dcs}")
    agg, detailed = process_trace_for_simulator(
        trace=trace,
        epoch_length=args.epoch_length,
        num_dcs=args.num_dcs,
        prefer_ms=args.timestamps_in_ms,
    )

    print(f"[save] aggregated → {args.output}  (rows={len(agg)})")
    agg.to_csv(args.output, index=False)

    if not args.no_debug:
        print(f"[save] detailed   → {args.debug_detailed}  (rows={len(detailed)})")
        detailed.to_csv(args.debug_detailed, index=False)

    # quick sanity print
    print("\n[preview] first 10 aggregated rows:")
    with pd.option_context("display.max_rows", 10, "display.max_columns", None, "display.width", 120):
        print(agg.head(10))

if __name__ == "__main__":
    main()





