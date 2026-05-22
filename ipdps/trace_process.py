#!/usr/bin/env python3
"""
trace_process.py — Unified trace processor for Rate_Flow_Sim_v2.

Accepts BurstGPT and Azure LLM Inference 2024 traces through a single CLI.
Pass one or two CSVs via --input; the format is detected automatically from
column names.

Supported formats
─────────────────
  BurstGPT (any single CSV with these columns):
    Timestamp       — seconds since midnight, day 1
    Model           — raw model label  (ChatGPT, GPT-4, …)
    Request tokens  — prompt token count
    Response tokens — generation token count
    Total tokens    — total token count

  Azure LLM Inference 2024
  (https://github.com/Azure/AzurePublicDataset):
    TIMESTAMP       — invocation time (Unix sec, Unix ms, or datetime string)
    ContextTokens   — prompt token count
    GeneratedTokens — generation token count

  Two Azure files may be passed together (code + conversation); they are
  merged and each assigned its workload label from the filename.

Format detection
────────────────
  Presence of a "Model" column  → BurstGPT
  Presence of "ContextTokens"   → Azure
  If neither is found the script exits with a clear error.

Model assignment
────────────────
  BurstGPT: raw "Model" value is matched as a case-insensitive substring
  against the DEFAULT_MODEL_MAP keys (e.g. "chatgpt", "gpt-4o", "mistral").

  Azure: each file is assigned a workload label ("code", "conv", or "azure")
  based on its filename, then that label is matched against the model map.
  "code" maps exclusively to large-class frontier models; "conv" maps across
  both tiers reflecting the mixed-size nature of conversational deployments.

  --large-frac overrides the tier split after mapping for both formats.

Output columns (identical for both formats)
───────────────────────────────────────────
  epoch          — 15-minute window index, 0-based
  source_dc_id   — originating datacenter ID
  model_type     — simulator base model key
  scenario       — "Chat" | "Summarization" | "Novel"
  arrival_ms     — intra-epoch arrival offset in milliseconds
  num_tokens     — total tokens (prompt + generation)
  prompt_tokens  — prompt token count
  gen_tokens     — generation token count

Usage examples
──────────────
  # BurstGPT, first 24 hours (96 epochs), default model map
  python trace_process.py --input BurstGPT_1.csv --max-epochs 96

  # Azure code trace, 24-hour window starting on day 2
  python trace_process.py --input AzureLLMInferenceTrace_code_1week.csv \\
         --day-offset 2 --max-epochs 96

  # Azure code + conversation merged, large-model-heavy workload
  python trace_process.py \\
         --input AzureLLMInferenceTrace_code_1week.csv \\
                 AzureLLMInferenceTrace_conv_1week.csv \\
         --large-frac 0.80 --max-epochs 96

  # Show active model map and exit
  python trace_process.py --show-map

  # Override a specific model-map entry
  python trace_process.py --input BurstGPT_1.csv \\
         --model-map '{"chatgpt": {"Mixtral_8x7B": 1.0}}'
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

# ── Shared constants ──────────────────────────────────────────────────────────
DEFAULT_OUTPUT        = "simulator_ready_trace.csv"
DEFAULT_EPOCH_LENGTH  = 900      # seconds (15 minutes)
DEFAULT_NUM_DCS       = 12

SCENARIO_PROBS: Dict[str, float] = {
    "Chat":          0.60,
    "Summarization": 0.30,
    "Novel":         0.10,
}

DEFAULT_POPULATION_WEIGHTS: Dict[int, float] = {
    0: 0.12, 1: 0.10, 2: 0.08, 3: 0.15, 4: 0.10, 5: 0.05,
    6: 0.18, 7: 0.08, 8: 0.06, 9: 0.04, 10: 0.02, 11: 0.02,
}

# ── Simulator model vocabulary ────────────────────────────────────────────────
# Rate_Flow_Sim_v2  ALL_MODEL_COLUMNS — do not add names the simulator doesn't know.
V2_SMALL_MODELS: List[str] = [
    "Llama7b",       # Meta Llama 2 7B
    "Mixtral_8x7B",  # Mistral AI Mixtral 8×7B MoE  (7B active per token)
]
V2_LARGE_MODELS: List[str] = [
    "Llama70b",      # Meta Llama 2 70B
    "Llama2_70B",    # Meta Llama 2 70B  (explicit variant key)
    "Llama31_405B",  # Meta Llama 3.1 405B
    "DeepSeek_R1",   # DeepSeek R1 671B MoE reasoning
]
V2_ALL_MODELS: List[str] = V2_SMALL_MODELS + V2_LARGE_MODELS

DEFAULT_MODEL: str = "Llama70b"   # fallback when no pattern matches

# ── Azure workload labels (inferred from filename) ────────────────────────────
LABEL_CODE  = "code"    # code completion / generation — long context, frontier models
LABEL_CONV  = "conv"    # conversational — shorter context, mixed model sizes
LABEL_AZURE = "azure"   # generic Azure (filename has neither "code" nor "conv")

# ── Combined DEFAULT_MODEL_MAP ────────────────────────────────────────────────
# Used for both formats via the same substring-matching logic.
#
# ▸ BurstGPT keys: matched as case-insensitive substrings of the raw "Model" column.
#   BurstGPT only contains "ChatGPT" and "GPT-4" in practice, but we cover the full
#   vocabulary for forward-compatibility with richer traces.
#
# ▸ Azure keys: exact short labels ("code", "conv", "azure").  Substring matching
#   still works because "code" ⊆ "code", "conv" ⊆ "conv", and none of the BurstGPT
#   model strings contain these substrings, so there is no cross-contamination.
#
# Longest-pattern-first evaluation means more-specific entries always win.
#
DEFAULT_MODEL_MAP: Dict[str, Dict[str, float]] = {

    # ── BurstGPT — frontier / reasoning ──────────────────────────────────────
    "gpt-4o":       {"Llama31_405B": 0.50, "DeepSeek_R1":  0.50},
    "gpt-4-turbo":  {"Llama31_405B": 0.55, "DeepSeek_R1":  0.45},
    "gpt-4-32k":    {"Llama31_405B": 0.60, "DeepSeek_R1":  0.40},

    # ── BurstGPT — standard large ─────────────────────────────────────────────
    "gpt-4":        {"Llama70b": 0.45, "Llama31_405B": 0.30,
                     "DeepSeek_R1": 0.15, "Mixtral_8x7B": 0.10},

    # ── BurstGPT — commodity (GPT-3.5 / ChatGPT) ────────────────────────────
    "gpt-3.5":      {"Llama70b": 0.45, "Llama2_70B": 0.30, "Mixtral_8x7B": 0.25},
    "chatgpt":      {"Llama70b": 0.45, "Llama2_70B": 0.30, "Mixtral_8x7B": 0.25},

    # ── BurstGPT — Mistral / Mixtral explicit ────────────────────────────────
    "mixtral":      {"Mixtral_8x7B": 1.0},
    "mistral":      {"Mixtral_8x7B": 1.0},

    # ── BurstGPT — DeepSeek explicit ─────────────────────────────────────────
    "deepseek-r1":  {"DeepSeek_R1": 1.0},
    "deepseek":     {"DeepSeek_R1": 1.0},

    # ── BurstGPT — Meta Llama explicit ───────────────────────────────────────
    "llama-3":      {"Llama31_405B": 0.60, "Llama70b":  0.40},
    "llama-70":     {"Llama70b":     0.60, "Llama2_70B": 0.40},
    "llama-2":      {"Llama2_70B":   0.70, "Llama70b":  0.30},
    "llama2":       {"Llama2_70B":   0.70, "Llama70b":  0.30},
    "llama-7":      {"Llama7b":      1.0},

    # ── BurstGPT — generic tier patterns ─────────────────────────────────────
    "405b":         {"Llama31_405B": 1.0},
    "70b":          {"Llama70b":     0.60, "Llama2_70B": 0.40},
    "7b":           {"Llama7b":      1.0},

    # ── Azure — code completions (long context, frontier models) ─────────────
    # Empirically: p50 context ≈ 1500 tok, p99 ≈ 10 000+ tok (DynamoLLM, HPCA 2025)
    LABEL_CODE:     {"Llama31_405B": 0.40, "DeepSeek_R1": 0.35, "Llama70b": 0.25},

    # ── Azure — conversations (shorter context, mixed tier) ───────────────────
    LABEL_CONV:     {"Llama70b": 0.35, "Llama2_70B": 0.25,
                     "Mixtral_8x7B": 0.25, "Llama7b": 0.15},

    # ── Azure — generic fallback (filename has neither "code" nor "conv") ─────
    LABEL_AZURE:    {"Llama70b": 0.40, "Llama2_70B": 0.20,
                     "Mixtral_8x7B": 0.20, "Llama31_405B": 0.10, "DeepSeek_R1": 0.10},
}


# ── Model map helpers ─────────────────────────────────────────────────────────

def _normalise_map(raw: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Validate targets and normalise weights to sum to 1.0."""
    out: Dict[str, Dict[str, float]] = {}
    for pattern, targets in raw.items():
        if not targets:
            raise ValueError(
                f"Model map entry '{pattern}' has no targets.  "
                f"Valid models: {V2_ALL_MODELS}"
            )
        unknown = [t for t in targets if t not in V2_ALL_MODELS]
        if unknown:
            raise ValueError(
                f"Entry '{pattern}' references unknown model(s): {unknown}\n"
                f"  Small-class: {V2_SMALL_MODELS}\n"
                f"  Large-class: {V2_LARGE_MODELS}"
            )
        total = sum(targets.values())
        if total <= 0:
            raise ValueError(f"Entry '{pattern}' has zero total weight.")
        out[pattern] = {t: w / total for t, w in targets.items()}
    return out


def _compile_map(
    model_map: Dict[str, Dict[str, float]],
) -> List[Tuple[str, List[str], List[float]]]:
    """Sort patterns longest-first so more-specific entries always win."""
    return sorted(
        [(p, list(d.keys()), list(d.values())) for p, d in model_map.items()],
        key=lambda x: len(x[0]),
        reverse=True,
    )


def _load_model_map(
    base:          Dict[str, Dict[str, float]],
    override_json: Optional[str] = None,
    override_file: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Build the active model map by merging CLI overrides into `base`.

    Semantics:
      • --model-map-file is applied first; --model-map JSON wins over it.
      • Each key in the override replaces the corresponding base entry.
      • Pass {} for a pattern to remove it entirely from the map.
    """
    active = dict(base)
    for label, source in [("--model-map-file", override_file),
                           ("--model-map",      override_json)]:
        if not source:
            continue
        try:
            data = json.load(open(source)) if label == "--model-map-file" else json.loads(source)
        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError(f"Could not parse {label}: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"{label} must be a JSON object.")
        for k, v in data.items():
            if isinstance(v, dict) and not v:
                active.pop(k, None)
            else:
                active[k] = v
    return _normalise_map(active)


def _apply_model_map(
    key_series: pd.Series,
    compiled:   List[Tuple[str, List[str], List[float]]],
    rng:        np.random.Generator,
) -> pd.Series:
    """
    Assign a simulator model to each row via case-insensitive substring matching.

    Patterns are evaluated longest-first; the first match wins.  Rows that
    match no pattern receive DEFAULT_MODEL.  Works identically for BurstGPT
    raw model names and Azure workload labels.
    """
    lower    = key_series.astype(str).str.strip().str.lower().to_numpy()
    out      = np.full(len(lower), DEFAULT_MODEL, dtype=object)
    assigned = np.zeros(len(lower), dtype=bool)

    for pattern, targets, probs in compiled:
        mask = (~assigned) & np.array([pattern in s for s in lower], dtype=bool)
        n    = int(mask.sum())
        if n > 0:
            out[mask]  = rng.choice(targets, size=n, p=probs)
            assigned  |= mask

    return pd.Series(out, index=key_series.index, dtype=str)


def _apply_tier_split(
    model_series: pd.Series,
    large_frac:   float,
    rng:          np.random.Generator,
) -> pd.Series:
    """
    Post-process model assignments so exactly `large_frac` fraction of rows
    are large-class models.

    Rows already in the correct tier keep their specific model (preserving
    intra-tier diversity from the map).  Rows that must change tier are
    reassigned by sampling uniformly from the destination tier's model list.
    """
    large_frac     = float(np.clip(large_frac, 0.0, 1.0))
    n              = len(model_series)
    if n == 0:
        return model_series

    arr            = model_series.to_numpy(dtype=object)
    is_large       = np.array([m in V2_LARGE_MODELS for m in arr], dtype=bool)
    large_idx      = np.where(is_large)[0]
    small_idx      = np.where(~is_large)[0]
    n_target_large = int(round(n * large_frac))

    if len(large_idx) > n_target_large:
        to_flip        = rng.choice(large_idx, size=len(large_idx) - n_target_large, replace=False)
        arr[to_flip]   = rng.choice(V2_SMALL_MODELS, size=len(to_flip))
    elif len(large_idx) < n_target_large:
        n_flip         = min(n_target_large - len(large_idx), len(small_idx))
        to_flip        = rng.choice(small_idx, size=n_flip, replace=False)
        arr[to_flip]   = rng.choice(V2_LARGE_MODELS, size=n_flip)

    return pd.Series(arr, index=model_series.index, dtype=str)


def print_model_map(model_map: Dict[str, Dict[str, float]]) -> None:
    """Pretty-print the active mapping table with source and tier annotations."""
    compiled = _compile_map(model_map)
    W = 66
    print(f"\n╔══ Simulator model vocabulary {'═'*(W-31)}╗")
    print(f"  Small-class (7B nodes):   {', '.join(V2_SMALL_MODELS)}")
    print(f"  Large-class (70B+ nodes): {', '.join(V2_LARGE_MODELS)}")
    print(f"╠══ Active model-mapping table (longest-first) {'═'*(W-47)}╣")
    print(f"  {'Pattern':<15}  {'Target model':<22}  {'Weight':>7}  {'Tier':<6}  Source")
    print(f"  {'─'*15}  {'─'*22}  {'─'*7}  {'─'*6}  {'─'*10}")
    azure_labels = {LABEL_CODE, LABEL_CONV, LABEL_AZURE}
    for pattern, targets, probs in compiled:
        source = "Azure" if pattern in azure_labels else "BurstGPT"
        for i, (t, p) in enumerate(zip(targets, probs)):
            tier  = "small" if t in V2_SMALL_MODELS else "large"
            lbl   = f'"{pattern}"' if i == 0 else ""
            src   = source if i == 0 else ""
            print(f"  {lbl:<15}  {t:<22}  {p:>6.1%}  {tier:<6}  {src}")
    tier = "small" if DEFAULT_MODEL in V2_SMALL_MODELS else "large"
    print(f"  {'(no match)':<15}  {DEFAULT_MODEL:<22}  {'100.0%':>7}  {tier:<6}  fallback")
    print(f"╚{'═'*W}╝")


# ── Format detection ──────────────────────────────────────────────────────────

def _detect_format(df: pd.DataFrame) -> Literal["burstgpt", "azure"]:
    """
    Infer trace format from column names.

    BurstGPT: has "Model" column
    Azure    : has "ContextTokens" column
    """
    cols = set(df.columns)
    if "Model" in cols:
        return "burstgpt"
    if "ContextTokens" in cols:
        return "azure"
    # Case-insensitive fallback
    cols_lower = {c.lower() for c in cols}
    if "model" in cols_lower:
        return "burstgpt"
    if "contexttokens" in cols_lower:
        return "azure"
    raise ValueError(
        f"Cannot detect trace format from columns: {sorted(cols)}\n"
        f"  BurstGPT requires: Timestamp, Model, Request tokens, Response tokens, Total tokens\n"
        f"  Azure requires: TIMESTAMP, ContextTokens, GeneratedTokens"
    )


# ── Format-specific loaders ───────────────────────────────────────────────────

def _load_burstgpt(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and normalise a BurstGPT DataFrame into the internal format:
      seconds   — float, elapsed from trace start
      model_key — raw model string for map matching
      prompt_tokens, gen_tokens, num_tokens — int
    """
    required = {"Timestamp", "Model", "Request tokens", "Response tokens", "Total tokens"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"BurstGPT CSV missing column(s): {missing}\n"
            f"Expected: {sorted(required)}"
        )
    # Drop failed requests (Response tokens ≤ 0)
    before = len(df)
    df = df[pd.to_numeric(df["Response tokens"], errors="coerce") > 0].copy()
    print(f"[clean]   Dropped {before - len(df):,} BurstGPT rows "
          f"(Response tokens ≤ 0) — {len(df):,} remaining.")

    t = pd.to_numeric(df["Timestamp"], errors="coerce").fillna(0.0)
    return pd.DataFrame({
        "seconds":      (t - float(t.min())).to_numpy(dtype=float),
        "model_key":    df["Model"].astype(str).to_numpy(),
        "prompt_tokens": pd.to_numeric(df["Request tokens"],  errors="coerce").fillna(0).astype(int).to_numpy(),
        "gen_tokens":   pd.to_numeric(df["Response tokens"], errors="coerce").fillna(0).astype(int).to_numpy(),
        "num_tokens":   pd.to_numeric(df["Total tokens"],    errors="coerce").fillna(0).astype(int).to_numpy(),
    })


def _parse_timestamps_to_seconds(series: pd.Series) -> pd.Series:
    """
    Convert TIMESTAMP to float seconds elapsed from the first record.

    Auto-detects three formats:
      • Unix seconds    — median ~1.715e9 for May 2024
      • Unix milliseconds — median ~1.715e12 for May 2024
      • Datetime string — parsed with pd.to_datetime
    """
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() >= 0.90:
        sec = numeric / 1000.0 if float(numeric.median()) > 1e11 else numeric
        return (sec - sec.min()).fillna(0.0)
    dt = pd.to_datetime(series, errors="coerce")
    if dt.notna().mean() >= 0.90:
        return (dt - dt.min()).dt.total_seconds().fillna(0.0)
    raise ValueError(
        f"Cannot parse TIMESTAMP.  First 5 values: {series.head().tolist()}"
    )


def _azure_label_from_path(filepath: str) -> str:
    """Infer workload label from filename."""
    name = os.path.basename(filepath).lower()
    if "code" in name:
        return LABEL_CODE
    if "conv" in name:
        return LABEL_CONV
    return LABEL_AZURE


def _load_azure(df: pd.DataFrame, label: str, filepath: str) -> pd.DataFrame:
    """
    Validate and normalise an Azure DataFrame into the internal format.
    arrival_offset_ms carries the sub-epoch timing (ms within epoch);
    set to NaN here and computed later after the epoch index is known.
    """
    required = {"TIMESTAMP", "ContextTokens", "GeneratedTokens"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Azure CSV '{os.path.basename(filepath)}' missing column(s): {missing}\n"
            f"Expected: TIMESTAMP, ContextTokens, GeneratedTokens"
        )
    # Drop failed requests
    before = len(df)
    df = df[pd.to_numeric(df["GeneratedTokens"], errors="coerce") > 0].copy()
    print(f"[clean]   Dropped {before - len(df):,} Azure rows "
          f"(GeneratedTokens ≤ 0) — {len(df):,} remaining.")

    sec   = _parse_timestamps_to_seconds(df["TIMESTAMP"]).to_numpy(dtype=float)
    ptok  = pd.to_numeric(df["ContextTokens"],   errors="coerce").fillna(0).clip(lower=0).astype(int).to_numpy()
    gtok  = pd.to_numeric(df["GeneratedTokens"], errors="coerce").fillna(0).clip(lower=0).astype(int).to_numpy()
    return pd.DataFrame({
        "seconds":       sec,
        "model_key":     np.full(len(df), label, dtype=object),
        "prompt_tokens": ptok,
        "gen_tokens":    gtok,
        "num_tokens":    ptok + gtok,
    })


# ── Unified pipeline ──────────────────────────────────────────────────────────

def process_trace(
        inputs:             List[str],
        output:             str                            = DEFAULT_OUTPUT,
        epoch_length:       int                            = DEFAULT_EPOCH_LENGTH,
        num_dcs:            int                            = DEFAULT_NUM_DCS,
        distribution:       str                            = "even",
        population_weights: Optional[Dict[int, float]]    = None,
        day_offset:         int                            = 0,
        max_epochs:         Optional[int]                  = None,
        model_map:          Optional[Dict[str, Dict[str, float]]] = None,
        large_frac:         Optional[float]               = None,
        seed:               int                            = 42,
) -> pd.DataFrame:
    """
    Load one or more trace CSVs, auto-detect their format, and produce a
    simulator-ready per-request DataFrame.

    Parameters
    ----------
    inputs      : List of 1–2 file paths.  Multiple Azure files are merged;
                  multiple BurstGPT files are concatenated.
    day_offset  : Skip the first N × 86400 s from the trace (Azure only; BurstGPT
                  timestamps already span a single collection window).
    max_epochs  : Keep only the first N epochs after day_offset (96 = 24 h).
    model_map   : Normalised {pattern: {model: prob}} mapping table.
                  None = use DEFAULT_MODEL_MAP.
    large_frac  : Force this fraction of rows to large-class models after mapping.
                  None = use map as-is.
    """
    if not inputs:
        raise ValueError("At least one --input file is required.")

    rng = np.random.default_rng(seed)

    # ── 1. Load all input files ───────────────────────────────────────────────
    parts: List[pd.DataFrame] = []
    fmt_set: set = set()
    for path in inputs:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Input file not found: {path}")
        print(f"[load]    {path}")
        raw = pd.read_csv(path, low_memory=False)
        print(f"          {len(raw):,} raw rows")
        fmt = _detect_format(raw)
        fmt_set.add(fmt)
        if len(fmt_set) > 1:
            raise ValueError(
                "Cannot mix BurstGPT and Azure files in a single call.  "
                "Pass only BurstGPT files or only Azure files."
            )
        if fmt == "burstgpt":
            parts.append(_load_burstgpt(raw))
        else:
            label = _azure_label_from_path(path)
            print(f"          Azure workload label: '{label}'")
            parts.append(_load_azure(raw, label, path))

    trace_fmt = fmt_set.pop()
    df = pd.concat(parts, ignore_index=True)

    # Re-anchor seconds when multiple files are merged
    df["seconds"] = df["seconds"] - float(df["seconds"].min())

    total_before = len(df)
    if total_before == 0:
        raise ValueError("No valid requests remain after loading and filtering.")
    print(f"[merged]  {total_before:,} total valid requests ({trace_fmt} format)")

    # ── 2. Day offset (Azure) ─────────────────────────────────────────────────
    if day_offset > 0:
        if trace_fmt == "burstgpt":
            print(f"[warn]    --day-offset ignored for BurstGPT format.")
        else:
            offset_s = float(day_offset) * 86400.0
            df = df[df["seconds"] >= offset_s].copy()
            df["seconds"] = df["seconds"] - offset_s
            if len(df) == 0:
                raise ValueError(
                    f"No requests remain after day_offset={day_offset}. "
                    f"The trace may span fewer than {day_offset + 1} day(s)."
                )
            print(f"[offset]  Day {day_offset}: skipped {offset_s/3600:.0f}h — "
                  f"{len(df):,} rows remain.")

    # ── 3. Epoch index ────────────────────────────────────────────────────────
    epoch_arr = (df["seconds"].to_numpy() // epoch_length).astype(int)

    # ── 4. Arrival_ms (intra-epoch offset) ───────────────────────────────────
    # BurstGPT: set to 0 (no sub-epoch timing available in the dataset)
    # Azure:    preserve real intra-epoch offset for realistic arrival patterns
    if trace_fmt == "azure":
        arrival_ms = np.clip(
            (df["seconds"].to_numpy() - epoch_arr * epoch_length) * 1000.0,
            0.0, epoch_length * 1000.0 - 1.0
        )
    else:
        arrival_ms = np.zeros(len(df), dtype=float)

    # ── 5. Model map application ──────────────────────────────────────────────
    active_map = model_map if model_map is not None else _normalise_map(DEFAULT_MODEL_MAP)
    compiled   = _compile_map(active_map)
    models     = _apply_model_map(df["model_key"], compiled, rng)

    n_fallback = int((models == DEFAULT_MODEL).sum())
    if n_fallback > 0:
        print(f"[map]     {n_fallback:,} rows ({n_fallback/len(df)*100:.1f}%) "
              f"matched no pattern → fallback '{DEFAULT_MODEL}'.")

    # ── 5b. Tier split adjustment ─────────────────────────────────────────────
    if large_frac is not None:
        lf     = float(np.clip(large_frac, 0.0, 1.0))
        models = _apply_tier_split(models, lf, rng)
        actual = models.isin(V2_LARGE_MODELS).mean() * 100
        print(f"[tier]    large_frac={lf:.2f} applied → "
              f"large={actual:.1f}%  small={100-actual:.1f}%")

    # ── 6. Scenario injection ─────────────────────────────────────────────────
    scenarios = rng.choice(
        list(SCENARIO_PROBS.keys()),
        size=len(df),
        p=list(SCENARIO_PROBS.values()),
    )

    # ── 7. Source DC assignment ───────────────────────────────────────────────
    if distribution == "population":
        weights = population_weights or DEFAULT_POPULATION_WEIGHTS
        avail   = {k: v for k, v in weights.items() if k < num_dcs}
        total   = sum(avail.values()) or 1.0
        avail   = {k: v / total for k, v in avail.items()} or {
            i: 1.0 / num_dcs for i in range(num_dcs)
        }
        dc_ids_l = list(avail.keys())
        dc_probs = list(avail.values())
        src_dc   = np.zeros(len(df), dtype=int)
        for ep in np.unique(epoch_arr):
            mask        = epoch_arr == ep
            ep_rng      = np.random.default_rng(seed + int(ep) * 1000)
            src_dc[mask] = ep_rng.choice(dc_ids_l, size=int(mask.sum()), p=dc_probs)
    else:
        src_dc = np.arange(len(df), dtype=np.int64) % int(num_dcs)

    # ── 8. Assemble output ────────────────────────────────────────────────────
    out = pd.DataFrame({
        "epoch":         epoch_arr.astype(int),
        "source_dc_id":  src_dc.astype(int),
        "model_type":    models.to_numpy(),
        "scenario":      scenarios,
        "arrival_ms":    arrival_ms.round(1),
        "num_tokens":    df["num_tokens"].to_numpy().astype(int),
        "prompt_tokens": df["prompt_tokens"].to_numpy().astype(int),
        "gen_tokens":    df["gen_tokens"].to_numpy().astype(int),
    }).sort_values(["epoch", "arrival_ms"]).reset_index(drop=True)

    # ── 9. Optional epoch truncation ─────────────────────────────────────────
    if max_epochs is not None and max_epochs > 0:
        out = out[out["epoch"] < max_epochs].copy()
        # Re-anchor to epoch 0 if day_offset left a gap at the start.
        # Important: clamp the shift so we never reduce the epoch range below
        # what the data actually covers — subtract only a positive min_ep.
        min_ep = int(out["epoch"].min()) if len(out) > 0 else 0
        if min_ep > 0:
            out["epoch"] = out["epoch"] - min_ep
        if len(out) > 0:
            print(f"[trunc]   Kept {max_epochs} epochs ({len(out):,} rows, "
                  f"~{max_epochs * epoch_length / 3600:.1f}h simulated time).")

    # ── 10. Guarantee full epoch coverage ────────────────────────────────────
    # The simulator determines its loop bound from trace["epoch"].max().  If
    # the raw data is sparse or shorter than max_epochs, the run would stop at
    # the last real epoch.  We add one minimal sentinel row at epoch
    # max_epochs-1 so the simulator always iterates the full range.
    # Epochs with no real rows between 0 and max_epochs-1 are handled by the
    # simulator's own zero-traffic path — no synthetic requests are injected
    # for those; only the final anchor row is added when needed.
    if max_epochs is not None and max_epochs > 0 and len(out) > 0:
        target_last_ep = max_epochs - 1
        actual_last_ep = int(out["epoch"].max())
        if actual_last_ep < target_last_ep:
            missing = target_last_ep - actual_last_ep
            print(f"[pad]     Trace spans epochs 0–{actual_last_ep} "
                  f"({actual_last_ep + 1} epochs).  Adding sentinel row at "
                  f"epoch {target_last_ep} so the simulator runs all "
                  f"{max_epochs} epochs ({missing} zero-traffic epoch(s) "
                  f"will be handled natively by the simulator).")
            sentinel = pd.DataFrame([{
                "epoch":         target_last_ep,
                "source_dc_id":  int(out["source_dc_id"].iloc[0]),
                "model_type":    str(out["model_type"].iloc[0]),
                "scenario":      "Chat",
                "arrival_ms":    0.0,
                "num_tokens":    1,
                "prompt_tokens": 1,
                "gen_tokens":    1,
            }])
            out = pd.concat([out, sentinel], ignore_index=True) \
                    .sort_values(["epoch", "arrival_ms"]) \
                    .reset_index(drop=True)

    return out


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Unified trace processor for Rate_Flow_Sim_v2.\n"
            "Supports BurstGPT and Azure LLM Inference 2024 traces."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Input / output ─────────────────────────────────────────────────────
    io = ap.add_argument_group("input / output")
    io.add_argument(
        "--input", nargs="+", metavar="PATH", default=None,
        help=(
            "One or two input CSV files.  Format is auto-detected: BurstGPT if "
            "a 'Model' column exists; Azure if 'ContextTokens' exists.  "
            "Two Azure files (code + conv) may be merged in a single call."
        ),
    )
    io.add_argument("--output", default=DEFAULT_OUTPUT,
                    help=f"Output CSV  [default: {DEFAULT_OUTPUT}]")

    # ── Epoch / windowing ──────────────────────────────────────────────────
    ew = ap.add_argument_group("epoch / windowing")
    ew.add_argument("--epoch-length", type=int, default=DEFAULT_EPOCH_LENGTH,
                    help=f"Epoch window in seconds  [default: {DEFAULT_EPOCH_LENGTH}]")
    ew.add_argument(
        "--max-epochs", type=int, default=None,
        help="Keep only the first N epochs (96 = 24 h).  [default: keep all]",
    )
    ew.add_argument(
        "--day-offset", type=int, default=0,
        help=(
            "Azure only: skip the first N × 24 h from the trace start.  "
            "Use this to select a specific day within the one-week Azure dataset.  "
            "[default: 0]"
        ),
    )

    # ── DC assignment ──────────────────────────────────────────────────────
    dc = ap.add_argument_group("datacenter assignment")
    dc.add_argument("--num-dcs", type=int, default=DEFAULT_NUM_DCS,
                    help=f"Number of datacenters  [default: {DEFAULT_NUM_DCS}]")
    dc.add_argument("--distribution", default="even",
                    choices=["even", "population"],
                    help="DC assignment strategy  [default: even]")
    dc.add_argument("--population-weights", default=None, metavar="JSON",
                    help='DC weight overrides as JSON e.g. \'{"0":0.3,"1":0.7}\'')

    # ── Model mapping ──────────────────────────────────────────────────────
    mm = ap.add_argument_group("model mapping")
    mm.add_argument(
        "--model-map", default=None, metavar="JSON",
        help=(
            "JSON object overriding entries in the default model map.  "
            'Format: {"pattern": {"ModelName": weight, ...}, ...}  '
            f"Valid targets: {', '.join(V2_ALL_MODELS)}"
        ),
    )
    mm.add_argument(
        "--model-map-file", default=None, metavar="PATH",
        help="JSON file with model-map overrides (same format as --model-map).",
    )
    mm.add_argument(
        "--large-frac", type=float, default=None, metavar="FLOAT",
        help=(
            "Target fraction of large-class model requests (0.0–1.0).  "
            "Applied after the model map as a post-processing tier split.  "
            "0.0 = all small-class  1.0 = all large-class  [default: use map as-is]"
        ),
    )
    mm.add_argument("--show-map", action="store_true",
                    help="Print the active model-mapping table and exit.")

    ap.add_argument("--seed", type=int, default=42,
                    help="Master random seed  [default: 42]")

    args = ap.parse_args()

    # ── Validate ──────────────────────────────────────────────────────────
    if not args.show_map and not args.input:
        ap.error("--input is required (unless --show-map is used).")

    # ── Build model map ───────────────────────────────────────────────────
    try:
        active_map = _load_model_map(
            DEFAULT_MODEL_MAP,
            override_json=args.model_map,
            override_file=args.model_map_file,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.show_map:
        print_model_map(active_map)
        sys.exit(0)

    # ── Population weights ────────────────────────────────────────────────
    population_weights = None
    if args.population_weights:
        try:
            population_weights = {
                int(k): float(v)
                for k, v in json.loads(args.population_weights).items()
            }
        except (json.JSONDecodeError, ValueError) as exc:
            print(f"ERROR: could not parse --population-weights: {exc}", file=sys.stderr)
            sys.exit(1)

    print_model_map(active_map)
    print(f"\n[config]  epoch_length={args.epoch_length}s  num_dcs={args.num_dcs}  "
          f"distribution={args.distribution}  day_offset={args.day_offset}  seed={args.seed}"
          + (f"  large_frac={args.large_frac:.2f}" if args.large_frac is not None else ""))

    # ── Process ───────────────────────────────────────────────────────────
    try:
        out = process_trace(
            inputs             = args.input,
            output             = args.output,
            epoch_length       = args.epoch_length,
            num_dcs            = args.num_dcs,
            distribution       = args.distribution,
            population_weights = population_weights,
            day_offset         = args.day_offset,
            max_epochs         = args.max_epochs,
            model_map          = active_map,
            large_frac         = args.large_frac,
            seed               = args.seed,
        )
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"\n[save]    {args.output}  ({len(out):,} rows)")
    out.to_csv(args.output, index=False)

    # ── Summary report ────────────────────────────────────────────────────
    n_epochs = out["epoch"].nunique()
    sep = "═" * 66
    print(f"\n{sep}")
    print(f"  Epochs:            {n_epochs:>9,}  "
          f"({n_epochs * args.epoch_length / 3600:.1f}h simulated time)")
    print(f"  Total requests:    {len(out):>9,}")
    print(f"  Requests / epoch:  {len(out) / max(n_epochs, 1):>9,.1f}")

    print(f"\n  Model type distribution:")
    for model, count in out["model_type"].value_counts().items():
        pct  = count / len(out) * 100
        tier = "small" if model in V2_SMALL_MODELS else "large"
        bar  = "█" * int(pct / 2)
        print(f"    {model:<22}  {count:>9,}  ({pct:5.1f}%)  [{tier}]  {bar}")
    small_pct = out["model_type"].isin(V2_SMALL_MODELS).mean() * 100
    print(f"\n  Tier split:  small={small_pct:.1f}%  large={100-small_pct:.1f}%")

    print(f"\n  Scenario distribution:")
    for sc, count in out["scenario"].value_counts().items():
        print(f"    {sc:<18}  {count:>9,}  ({count/len(out)*100:5.1f}%)")

    print(f"\n  Token statistics per request:")
    for col, lbl in [("prompt_tokens","Prompt"),("gen_tokens","Generated"),("num_tokens","Total")]:
        s = out[col]
        print(f"    {lbl:<12}  mean={s.mean():>8.1f}  p50={s.median():>8.1f}  "
              f"p95={s.quantile(0.95):>8.1f}  max={s.max():>8,}")
    print(sep)


if __name__ == "__main__":
    main()