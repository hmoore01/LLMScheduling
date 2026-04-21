#!/usr/bin/env python3
"""
BurstGPT_process.py — Convert a raw BurstGPT trace CSV to a simulator-ready
per-request workload for Rate_Flow_Sim_v2.

Simulator model vocabulary
──────────────────────────
Rate_Flow_Sim_v2 supports exactly six base models (ALL_MODEL_COLUMNS).
They are split into two hardware tiers that determine which node type handles
each request:

  Small-class  (7B-equivalent per-expert, lighter GPU nodes)
    • Llama7b          — Meta Llama 2 7B
    • Mixtral_8x7B     — Mistral AI Mixtral 8×7B  (MoE; 7B active per token)

  Large-class  (70B+ dense parameters, high-memory GPU nodes)
    • Llama70b         — Meta Llama 2 70B
    • Llama2_70B       — Meta Llama 2 70B  (alias / explicit variant key)
    • Llama31_405B     — Meta Llama 3.1 405B
    • DeepSeek_R1      — DeepSeek R1 671B MoE reasoning model

Model mapping is table-driven: each raw BurstGPT source pattern maps to a
weighted distribution over any subset of the six targets.  Any entry can be
overridden at the CLI — see --model-map, --model-map-file, --show-map.

Output columns
──────────────
  epoch          : 15-minute window index (0-based)
  source_dc_id   : originating datacenter ID
  model_type     : simulator base model key  (one of the six above)
  scenario       : semantic intent label     ("Chat", "Summarization", "Novel")
  arrival_ms     : arrival offset within epoch  (0 = epoch-batch / rate-flow)
  num_tokens     : total tokens  (prompt + generation)
  prompt_tokens  : prompt token count
  gen_tokens     : generation token count

Usage examples
──────────────
  # Standard run using default mapping
  python BurstGPT_process.py --input BurstGPT_1.csv

  # Print the active model-mapping table and exit
  python BurstGPT_process.py --show-map

  # Route all Mistral traffic to Mixtral_8x7B exclusively
  python BurstGPT_process.py --model-map '{"mistral": {"Mixtral_8x7B": 1.0}}'

  # Route GPT-3.5 to the small tier instead (Mixtral MoE)
  python BurstGPT_process.py --model-map '{"gpt-3.5": {"Mixtral_8x7B": 1.0}}'

  # Override from a file
  python BurstGPT_process.py --model-map-file my_overrides.json

  # Delete a pattern entirely (pass empty dict for that key)
  python BurstGPT_process.py --model-map '{"chatgpt": {}}'
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_INPUT        = "BurstGPT_1.csv"
DEFAULT_OUTPUT       = "simulator_ready_trace.csv"
DEFAULT_EPOCH_LENGTH = 900   # seconds (15 minutes)
DEFAULT_NUM_DCS      = 12

# ── Scenario injection ────────────────────────────────────────────────────────
SCENARIO_PROBS: Dict[str, float] = {
    "Chat":          0.60,
    "Summarization": 0.30,
    "Novel":         0.10,
}

# ── DC population weights ─────────────────────────────────────────────────────
DEFAULT_POPULATION_WEIGHTS: Dict[int, float] = {
    0: 0.12, 1: 0.10, 2: 0.08, 3: 0.15, 4: 0.10, 5: 0.05,
    6: 0.18, 7: 0.08, 8: 0.06, 9: 0.04, 10: 0.02, 11: 0.02,
}

# ── Simulator model vocabulary ────────────────────────────────────────────────
# These six names are the ONLY base models Rate_Flow_Sim_v2 recognises.
# They correspond exactly to the ALL_MODEL_COLUMNS list in build_world_from_csvs_exact
# and to the {model}_Process column headers in the GPU spec CSVs.
# Update this list only when new models are added to the simulator spec CSVs.

# Small-class: 7B-equivalent hardware tier
# (Mixtral is 7B *per expert*; the routing framework sends it to 7B nodes)
V2_SMALL_MODELS: List[str] = [
    "Llama7b",       # Meta Llama 2 7B
    "Mixtral_8x7B",  # Mistral AI Mixtral 8×7B MoE
]

# Large-class: 70B+ dense-parameter hardware tier
V2_LARGE_MODELS: List[str] = [
    "Llama70b",       # Meta Llama 2 70B
    "Llama2_70B",     # Meta Llama 2 70B  (explicit variant key used by some spec rows)
    "Llama31_405B",   # Meta Llama 3.1 405B
    "DeepSeek_R1",    # DeepSeek R1 671B MoE reasoning model
]

# Full vocabulary — the union of both tiers
V2_ALL_MODELS: List[str] = V2_SMALL_MODELS + V2_LARGE_MODELS


# ── Default model-mapping table ───────────────────────────────────────────────
# Format:  { "source_pattern": { "TargetModel": weight, ... }, ... }
#
# Matching rules
#   • source_pattern is matched as a case-insensitive substring of the raw
#     BurstGPT "Model" field.
#   • Patterns are evaluated longest-first so more-specific entries always
#     take priority (e.g. "gpt-4o" beats "gpt-4", "gpt-4-turbo" beats "gpt-4").
#   • Weights are relative and are normalised to sum to 1.0 automatically.
#   • When no pattern matches, DEFAULT_MODEL is used.
#   • All target names must be members of V2_ALL_MODELS (validated at load time).
#
# Design rationale
# ────────────────
# BurstGPT only contains two raw model names: "ChatGPT" and "GPT-4".
# The mapping distributes this binary signal across the full six-model vocabulary:
#
#   GPT-4o / GPT-4-turbo   → frontier reasoning tier   (405B / DeepSeek-R1)
#   GPT-4                  → standard large tier        (70B majority, frontier minority)
#   GPT-3.5 / ChatGPT      → commodity tier             (70B / Mixtral MoE split)
#   mistral / mixtral      → explicit Mixtral_8x7B      (MoE / small-class routing)
#   llama-2 / llama2       → Llama2_70B direct          (trace already has this label)
#   70b                    → generic 70B-class split
#   7b / llama-7           → Llama7b direct             (if trace labels small models)
#
DEFAULT_MODEL_MAP: Dict[str, Dict[str, float]] = {
    # ── Frontier / reasoning tier ─────────────────────────────────────────────
    "gpt-4o":          {"Llama31_405B": 0.50, "DeepSeek_R1":  0.50},
    "gpt-4-turbo":     {"Llama31_405B": 0.55, "DeepSeek_R1":  0.45},
    "gpt-4-32k":       {"Llama31_405B": 0.60, "DeepSeek_R1":  0.40},

    # ── Standard large tier ───────────────────────────────────────────────────
    "gpt-4":           {"Llama70b": 0.45, "Llama31_405B": 0.30, "DeepSeek_R1": 0.15,
                        "Mixtral_8x7B": 0.10},

    # ── Commodity large tier (GPT-3.5 / ChatGPT) ─────────────────────────────
    # Mixtral is included here to reflect real-world deployments where Mixtral MoE
    # is often chosen as a cost-efficient alternative to dense 70B models.
    "gpt-3.5":         {"Llama70b": 0.45, "Llama2_70B": 0.30, "Mixtral_8x7B": 0.25},
    "chatgpt":         {"Llama70b": 0.45, "Llama2_70B": 0.30, "Mixtral_8x7B": 0.25},

    # ── Mistral AI / Mixtral explicit patterns ────────────────────────────────
    "mixtral":         {"Mixtral_8x7B": 1.0},
    "mistral":         {"Mixtral_8x7B": 1.0},

    # ── DeepSeek explicit patterns ────────────────────────────────────────────
    "deepseek":        {"DeepSeek_R1": 1.0},
    "deepseek-r1":     {"DeepSeek_R1": 1.0},

    # ── Meta Llama explicit patterns ──────────────────────────────────────────
    "llama-3":         {"Llama31_405B": 0.60, "Llama70b": 0.40},
    "llama-2":         {"Llama2_70B":   0.70, "Llama70b": 0.30},
    "llama2":          {"Llama2_70B":   0.70, "Llama70b": 0.30},
    "llama-7":         {"Llama7b":      1.0},
    "llama-70":        {"Llama70b":     0.60, "Llama2_70B": 0.40},

    # ── Generic tier-based patterns ───────────────────────────────────────────
    "405b":            {"Llama31_405B": 1.0},
    "70b":             {"Llama70b":     0.60, "Llama2_70B": 0.40},
    "7b":              {"Llama7b":      1.0},
}

DEFAULT_MODEL: str = "Llama70b"   # fallback when no pattern matches


# ── Model map helpers ─────────────────────────────────────────────────────────

def _normalise_map(raw: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Validate and normalise each pattern entry so probabilities sum to 1.0.

    Raises ValueError for:
      • target models not in V2_ALL_MODELS
      • entries with no targets or zero total weight
    """
    normalised: Dict[str, Dict[str, float]] = {}
    for pattern, targets in raw.items():
        if not targets:
            raise ValueError(
                f"Model map entry '{pattern}' has no targets.  "
                f"Valid models: {V2_ALL_MODELS}"
            )
        unknown = [t for t in targets if t not in V2_ALL_MODELS]
        if unknown:
            raise ValueError(
                f"Model map entry '{pattern}' references unknown model(s): {unknown}.\n"
                f"  Small-class: {V2_SMALL_MODELS}\n"
                f"  Large-class: {V2_LARGE_MODELS}"
            )
        total = sum(targets.values())
        if total <= 0:
            raise ValueError(
                f"Model map entry '{pattern}' has zero total weight."
            )
        normalised[pattern] = {t: w / total for t, w in targets.items()}
    return normalised


def _compile_map(
    model_map: Dict[str, Dict[str, float]],
) -> List[Tuple[str, List[str], List[float]]]:
    """
    Compile the normalised map into a list of
        (pattern, [target_models], [probabilities])
    sorted by pattern length descending so more-specific patterns win.
    """
    return sorted(
        [(p, list(d.keys()), list(d.values())) for p, d in model_map.items()],
        key=lambda x: len(x[0]),
        reverse=True,
    )


def _load_model_map(
    override_json: Optional[str] = None,
    override_file: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Build the active model map by merging overrides into the default table.

    Override semantics:
      • --model-map-file is applied first; --model-map inline JSON wins over it.
      • Each key in the override replaces the corresponding default entry.
      • Pass an empty dict {} for a pattern to remove it entirely from the table.
    """
    active: Dict[str, Dict[str, float]] = dict(DEFAULT_MODEL_MAP)

    for label, source in [("--model-map-file", override_file),
                           ("--model-map",      override_json)]:
        if not source:
            continue
        try:
            data = (json.load(open(source))
                    if label == "--model-map-file"
                    else json.loads(source))
        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError(f"Could not parse {label}: {exc}") from exc

        if not isinstance(data, dict):
            raise ValueError(f"{label} must be a JSON object (dict).")

        for k, v in data.items():
            if isinstance(v, dict) and len(v) == 0:
                active.pop(k, None)    # explicit deletion
            else:
                active[k] = v

    return _normalise_map(active)


def _apply_model_map(
    raw_series: pd.Series,
    compiled: List[Tuple[str, List[str], List[float]]],
    default_model: str,
    rng: np.random.Generator,
) -> pd.Series:
    """
    Vectorised model mapping.

    For each raw BurstGPT model string, finds the first matching pattern
    (longest-first) and samples from that pattern's weighted distribution.
    Rows that match no pattern receive default_model.
    """
    raw_lower = raw_series.astype(str).str.strip().str.lower().to_numpy()
    out       = np.full(len(raw_lower), default_model, dtype=object)
    assigned  = np.zeros(len(raw_lower), dtype=bool)

    for pattern, targets, probs in compiled:
        mask = (~assigned) & np.array(
            [pattern in s for s in raw_lower], dtype=bool
        )
        n = int(mask.sum())
        if n > 0:
            out[mask]  = rng.choice(targets, size=n, p=probs)
            assigned  |= mask

    return pd.Series(out, index=raw_series.index, dtype=str)


def print_model_map(model_map: Dict[str, Dict[str, float]]) -> None:
    """Pretty-print the active mapping table with tier annotations."""
    compiled = _compile_map(model_map)
    width = 62

    print(f"\n╔══ Simulator model vocabulary {'═'*(width-31)}╗")
    print(f"  Small-class (7B nodes): {', '.join(V2_SMALL_MODELS)}")
    print(f"  Large-class (70B+ nodes): {', '.join(V2_LARGE_MODELS)}")
    print(f"╠══ Active model-mapping table (longest pattern first) {'═'*(width-53)}╣")
    print(f"  {'Pattern':<20}  {'Target model':<22}  {'Weight':>7}  {'Tier'}")
    print(f"  {'─'*20}  {'─'*22}  {'─'*7}  {'─'*5}")

    for pattern, targets, probs in compiled:
        for i, (t, p) in enumerate(zip(targets, probs)):
            tier  = "small" if t in V2_SMALL_MODELS else "large"
            label = f'"{pattern}"' if i == 0 else ""
            print(f"  {label:<20}  {t:<22}  {p:>6.1%}  {tier}")

    tier = "small" if DEFAULT_MODEL in V2_SMALL_MODELS else "large"
    print(f"  {'(no match)':<20}  {DEFAULT_MODEL:<22}  {'100.0%':>7}  {tier}  ← fallback")
    print(f"╚{'═'*width}╝")


# ── Core processing ───────────────────────────────────────────────────────────

def process_trace_per_request(
        trace: pd.DataFrame,
        epoch_length: int,
        num_dcs: int,
        distribution: str = "even",
        population_weights: Optional[Dict[int, float]] = None,
        max_epochs: Optional[int] = None,
        model_map: Optional[Dict[str, Dict[str, float]]] = None,
        seed: int = 42,
) -> pd.DataFrame:
    """
    Transform a raw BurstGPT trace into a Rate_Flow_Sim_v2-ready per-request
    DataFrame.

    Parameters
    ----------
    trace             : raw BurstGPT DataFrame
    epoch_length      : epoch window in seconds  (default 900 = 15 min)
    num_dcs           : number of datacenters
    distribution      : "even" (round-robin) | "population" (weighted)
    population_weights: DC weight dict; falls back to DEFAULT_POPULATION_WEIGHTS
    max_epochs        : truncate output to this many epochs  (None = keep all)
    model_map         : normalised mapping table; falls back to DEFAULT_MODEL_MAP
    seed              : master random seed
    """
    rng = np.random.default_rng(seed)

    # ── 0. Validate required columns ─────────────────────────────────────────
    required_cols = {
        "Timestamp":       "request submission time (seconds from midnight day 1)",
        "Model":           "model identifier  (ChatGPT / GPT-4 / …)",
        "Request tokens":  "prompt token count",
        "Response tokens": "generation token count",
        "Total tokens":    "total token count",
    }
    missing = [c for c in required_cols if c not in trace.columns]
    if missing:
        raise KeyError(
            f"Input CSV is missing required column(s): {missing}\n"
            f"Expected: {list(required_cols.keys())}"
        )

    # ── 1. Drop failed requests ───────────────────────────────────────────────
    before = len(trace)
    trace  = trace[pd.to_numeric(trace["Response tokens"], errors="coerce") > 0].copy()
    print(f"[clean] Dropped {before - len(trace):,} failed requests "
          f"(Response tokens ≤ 0) — {len(trace):,} remaining.")
    if len(trace) == 0:
        raise ValueError("No valid requests remain after filtering.")

    # ── 2. Timestamps → epoch index ───────────────────────────────────────────
    t        = pd.to_numeric(trace["Timestamp"], errors="coerce").fillna(0.0)
    min_time = float(np.nanmin(t.to_numpy()))
    epoch    = ((t - min_time) // epoch_length).astype(int)

    # ── 3. Token extraction ───────────────────────────────────────────────────
    tot_toks    = pd.to_numeric(trace["Total tokens"],    errors="coerce").fillna(0.0)
    prompt_toks = pd.to_numeric(trace["Request tokens"],  errors="coerce").fillna(0.0)
    gen_toks    = pd.to_numeric(trace["Response tokens"], errors="coerce").fillna(0.0)

    # ── 4. Model mapping ──────────────────────────────────────────────────────
    active_map  = (model_map if model_map is not None
                   else _normalise_map(DEFAULT_MODEL_MAP))
    compiled    = _compile_map(active_map)
    base_models = _apply_model_map(trace["Model"], compiled, DEFAULT_MODEL, rng)

    n_fallback = int((base_models == DEFAULT_MODEL).sum())
    if n_fallback > 0:
        print(f"[map]   {n_fallback:,} requests ({n_fallback/len(trace)*100:.1f}%) "
              f"matched no pattern → fallback '{DEFAULT_MODEL}'.")

    # ── 5. Scenario injection ─────────────────────────────────────────────────
    scenarios = rng.choice(
        list(SCENARIO_PROBS.keys()),
        size=len(trace),
        p=list(SCENARIO_PROBS.values()),
    )

    # ── 6. Source DC assignment ───────────────────────────────────────────────
    if distribution == "population":
        weights = population_weights or DEFAULT_POPULATION_WEIGHTS
        avail   = {k: v for k, v in weights.items() if k < num_dcs}
        total   = sum(avail.values()) or 1.0
        avail   = {k: v / total for k, v in avail.items()} or {
            i: 1.0 / num_dcs for i in range(num_dcs)
        }
        dc_ids_list   = list(avail.keys())
        dc_probs_list = list(avail.values())
        src_dc = np.zeros(len(trace), dtype=int)
        for ep_val in np.unique(epoch.to_numpy()):
            mask   = (epoch == ep_val).to_numpy()
            ep_rng = np.random.default_rng(seed + int(ep_val) * 1000)
            src_dc[mask] = ep_rng.choice(
                dc_ids_list, size=int(mask.sum()), p=dc_probs_list
            )
        src_dc = pd.Series(src_dc, index=trace.index)
    else:
        src_dc = pd.Series(
            np.arange(len(trace), dtype=np.int64) % int(num_dcs),
            index=trace.index,
            dtype=int,
        )

    # ── 7. Assemble output ────────────────────────────────────────────────────
    out = pd.DataFrame({
        "epoch":         epoch.to_numpy().astype(int),
        "source_dc_id":  src_dc.to_numpy().astype(int),
        "model_type":    base_models.to_numpy(),
        "scenario":      scenarios,
        "arrival_ms":    np.zeros(len(trace), dtype=int),
        "num_tokens":    tot_toks.to_numpy().astype(int),
        "prompt_tokens": prompt_toks.to_numpy().astype(int),
        "gen_tokens":    gen_toks.to_numpy().astype(int),
    }).sort_values(["epoch", "source_dc_id"]).reset_index(drop=True)

    # ── 8. Optional epoch truncation ─────────────────────────────────────────
    if max_epochs is not None and max_epochs > 0:
        out = out[out["epoch"] < max_epochs].copy()
        print(f"[clean] Truncated to first {max_epochs} epochs "
              f"({len(out):,} rows remaining).")

    return out


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Convert BurstGPT trace to per-request CSV for Rate_Flow_Sim_v2.\n"
            "All six simulator models are supported as routing targets."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Core
    ap.add_argument("--input",  default=DEFAULT_INPUT,
                    help=f"Input CSV (raw BurstGPT trace)  [default: {DEFAULT_INPUT}]")
    ap.add_argument("--output", default=DEFAULT_OUTPUT,
                    help=f"Output CSV (per-request workload)  [default: {DEFAULT_OUTPUT}]")
    ap.add_argument("--epoch-length", type=int, default=DEFAULT_EPOCH_LENGTH,
                    help=f"Epoch window in seconds  [default: {DEFAULT_EPOCH_LENGTH}]")
    ap.add_argument("--num-dcs", type=int, default=DEFAULT_NUM_DCS,
                    help=f"Number of datacenters  [default: {DEFAULT_NUM_DCS}]")
    ap.add_argument("--distribution", default="even",
                    choices=["even", "population"],
                    help="Source-DC assignment: 'even' (round-robin) or 'population' (weighted)")
    ap.add_argument("--population-weights", default=None, metavar="JSON",
                    help='DC weight overrides as JSON, e.g. \'{"0": 0.3, "1": 0.7}\'')
    ap.add_argument("--max-epochs", type=int, default=None,
                    help="Truncate output to this many epochs  (96 = 24 hours)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Master random seed  [default: 42]")

    # Model mapping
    mg = ap.add_argument_group("model mapping")
    mg.add_argument(
        "--model-map", default=None, metavar="JSON",
        help=(
            "JSON object that overrides / extends the default mapping table.  "
            'Format: {"source_pattern": {"TargetModel": weight, ...}, ...}  '
            'Pass {} for a pattern to remove it.  '
            "Valid targets: " + ", ".join(V2_ALL_MODELS)
        ),
    )
    mg.add_argument(
        "--model-map-file", default=None, metavar="PATH",
        help="Path to a JSON file with model-map overrides (same format as --model-map).",
    )
    mg.add_argument(
        "--default-model", default=DEFAULT_MODEL,
        choices=V2_ALL_MODELS,
        help=f"Fallback model when no pattern matches  [default: {DEFAULT_MODEL}]",
    )
    mg.add_argument(
        "--show-map", action="store_true",
        help="Print the active model-mapping table and exit.",
    )

    args = ap.parse_args()

    # Build and validate the active model map
    try:
        active_map = _load_model_map(
            override_json=args.model_map,
            override_file=args.model_map_file,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    # --show-map
    if args.show_map:
        print_model_map(active_map)
        sys.exit(0)

    # Input file
    if not os.path.exists(args.input):
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Population weights
    population_weights = None
    if args.population_weights:
        try:
            population_weights = {
                int(k): float(v)
                for k, v in json.loads(args.population_weights).items()
            }
        except (json.JSONDecodeError, ValueError) as exc:
            print(f"ERROR: could not parse --population-weights: {exc}",
                  file=sys.stderr)
            sys.exit(1)

    print(f"[load]    {args.input}")
    trace = pd.read_csv(args.input)
    print(f"          {len(trace):,} raw rows.")

    print_model_map(active_map)

    print(f"\n[process] epoch_length={args.epoch_length}s | "
          f"num_dcs={args.num_dcs} | "
          f"distribution={args.distribution} | "
          f"seed={args.seed}")

    per_req = process_trace_per_request(
        trace=trace,
        epoch_length=args.epoch_length,
        num_dcs=args.num_dcs,
        distribution=args.distribution,
        population_weights=population_weights,
        max_epochs=args.max_epochs,
        model_map=active_map,
        seed=args.seed,
    )

    print(f"\n[save]    {args.output}  ({len(per_req):,} rows)")
    per_req.to_csv(args.output, index=False)

    # ── Summary report ────────────────────────────────────────────────────────
    n_epochs = per_req["epoch"].nunique()
    sep = "═" * 62

    print(f"\n{sep}")
    print(f"  Epochs:             {n_epochs:>8,}")
    print(f"  Total requests:     {len(per_req):>8,}")
    print(f"  Requests / epoch:   {len(per_req) / max(n_epochs, 1):>8,.1f}")

    print(f"\n  Model type distribution:")
    model_counts = per_req["model_type"].value_counts()
    for model, count in model_counts.items():
        pct  = count / len(per_req) * 100
        tier = "small" if model in V2_SMALL_MODELS else "large"
        bar  = "█" * int(pct / 2)
        print(f"    {model:<22}  {count:>8,}  ({pct:5.1f}%)  [{tier}]  {bar}")

    small_pct = (per_req["model_type"].isin(V2_SMALL_MODELS).sum()
                 / len(per_req) * 100)
    large_pct = 100.0 - small_pct
    print(f"\n  Tier split:  small={small_pct:.1f}%  large={large_pct:.1f}%")

    print(f"\n  Scenario distribution:")
    for sc, count in per_req["scenario"].value_counts().items():
        pct = count / len(per_req) * 100
        print(f"    {sc:<18}  {count:>8,}  ({pct:5.1f}%)")

    print(f"\n  Token statistics (per request):")
    for col, label in [("prompt_tokens", "Prompt"),
                       ("gen_tokens",    "Generation"),
                       ("num_tokens",    "Total")]:
        s = per_req[col]
        print(f"    {label:<12}  mean={s.mean():>8.1f}  "
              f"median={s.median():>8.1f}  "
              f"p95={s.quantile(0.95):>8.1f}  "
              f"max={s.max():>8,}")
    print(sep)


if __name__ == "__main__":
    main()