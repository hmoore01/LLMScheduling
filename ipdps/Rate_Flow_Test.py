#!/usr/bin/env python3
"""
run_rate_sim.py — End-to-end runner for the Rate-Flow LLM Scheduling Simulator

What this script does
- Option A: Ingest a RAW trace (--input-trace), convert to rate-flow
- Option B: Ingest an ALREADY-AGGREGATED workload (--workload) with columns:
    epoch,src_dc,model_type,total_tokens
- Builds Datacenter → Node → Processor graph via rfs.build_datacenters_from_csv(...)
- Constructs a default schedule_plan (local routing unless overridden)
- Runs a single epoch with the new rate-based simulator entrypoint
- Prints global metrics + per-DC energy/carbon/water breakdown
- (Optional) Loops over 24 hours to include hourly solar profiles
- Includes a synthetic sanity test

Examples:
  # From RAW per-request trace:
  python run_rate_sim.py \
    --input-trace BurstGPT_without_fails_2.csv \
    --node-specs /mnt/data/Node_Specs.csv \
    --dc-specs   /mnt/data/Datacenter_specs.csv \
    --a100       /mnt/data/A100_GPU.csv \
    --h100       /mnt/data/H100_GPU.csv \
    --latencies  /mnt/data/Geo_Latencies.csv \
    --epoch-length 900 --epoch-index 0 --hour 12

  # From ALREADY-AGGREGATED rate-flow CSV:
  python run_rate_sim.py \
    --workload simulator_ready_trace.csv \
    --node-specs /mnt/data/Node_Specs.csv \
    --dc-specs   /mnt/data/Datacenter_specs.csv \
    --a100       /mnt/data/A100_GPU.csv \
    --h100       /mnt/data/H100_GPU.csv \
    --latencies  /mnt/data/Geo_Latencies.csv \
    --epoch-length 900 --epoch-index 0 --hour 12
"""

from __future__ import annotations
import argparse
import csv
import hashlib
import json
import sys
from typing import Dict, List, Optional, Tuple

try:
    import pandas as pd
    PANDAS = True
except Exception:
    PANDAS = False

# Import your simulator module
try:
    import Rate_Flow_Sim as rfs
except Exception as e:
    print(f"ERROR: Failed to import Rate_Flow_Sim: {e}", file=sys.stderr)
    sys.exit(1)

# =========================
# Latency loader
# =========================
def load_latency_matrix(path: str) -> List[List[float]]:
    """Load a square latency matrix (ms). Supports with/without header row."""
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    def is_float(s: str) -> bool:
        try:
            float(s); return True
        except Exception:
            return False

    if rows and rows[0] and not is_float(rows[0][0]):
        data = []
        for r in rows[1:]:
            vals = r[1:] if (r and not is_float(r[0])) else r
            if vals:
                data.append([float(x) for x in vals])
        return data
    else:
        return [[float(x) for x in r] for r in rows]

# =========================
# RAW Trace → Rate-Flow
# =========================
def _norm(s: str) -> str:
    return "".join(ch for ch in str(s).lower() if ch not in " _\t")

def _pick_col(df: pd.DataFrame, *cands: str) -> Optional[str]:
    cols = { _norm(c): c for c in df.columns }
    for c in cands:
        if _norm(c) in cols:
            return cols[_norm(c)]
    return None

def _timestamp_to_seconds(ts: pd.Series) -> pd.Series:
    med = float(ts.median())
    # Heuristic: if very large, likely ms
    if med > 1e10:
        return ts.astype(float) / 1000.0
    return ts.astype(float)

def _stable_dc_from_keys(row: pd.Series, num_dcs: int) -> int:
    for k in ["request_id","id","user","uid","session","trace_id"]:
        if k in {c.lower(): c for c in row.index} and pd.notna(row[k]):
            h = hashlib.blake2b(str(row[k]).encode("utf-8"), digest_size=4).hexdigest()
            return int(h, 16) % num_dcs
    return int(row.get("_row_idx", 0)) % num_dcs

def _map_model_to_llama(model_str: str) -> str:
    s = str(model_str).lower()
    if "gpt-4" in s or "gpt4" in s or "70b" in s or "70 b" in s or "llama-3.1-70b" in s:
        return "Llama70b"
    return "Llama7b"

def process_trace_for_simulator(
    trace: pd.DataFrame,
    epoch_length: int,
    num_dcs: int,
    timestamps_in_ms: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert a raw request trace → (aggregated_rate_flow, detailed_requests).
    Aggregated columns: ['epoch','src_dc','model_type','total_tokens']
    """
    model_col = _pick_col(trace, "Model","model","Model_Name","model_name","model_type")
    if model_col is None:
        raise KeyError(f"Missing model column in {list(trace.columns)}")

    ts_col = _pick_col(trace, "Timestamp","time","arrival_s","arrival","arrival_time","arrival_ms")
    if ts_col is None:
        raise KeyError(f"Missing timestamp column in {list(trace.columns)}")

    total_tok_col = _pick_col(trace, "Total tokens","total_tokens","tokens","toks","num_tokens")
    if total_tok_col is None:
        p_col = _pick_col(trace, "Prompt tokens","prompt_tokens","input_tokens","in_tokens","prompt")
        o_col = _pick_col(trace, "Output tokens","output_tokens","gen_tokens","out_tokens","completion")
        if p_col is None and o_col is None:
            raise KeyError("No token columns found (need 'Total tokens' or Prompt/Output tokens).")

    src_dc_col = _pick_col(trace, "Source_DC","source_dc_id","src_dc","Src_DC")

    # Normalize timestamps to seconds
    t = trace[ts_col].astype(float)
    if timestamps_in_ms:
        t = t / 1000.0
    else:
        t = _timestamp_to_seconds(t)

    # Tokens
    if total_tok_col is not None:
        toks = trace[total_tok_col].astype(float)
    else:
        p_col = _pick_col(trace, "Prompt tokens","prompt_tokens","input_tokens","in_tokens","prompt")
        o_col = _pick_col(trace, "Output tokens","output_tokens","gen_tokens","out_tokens","completion")
        p = trace[p_col].astype(float) if p_col else 0.0
        o = trace[o_col].astype(float) if o_col else 0.0
        toks = p + o

    # Epoch & time within epoch (relative to min time)
    min_time = float(t.min())
    epoch = ((t - min_time) // epoch_length).astype(int)
    time_index = ((t - min_time) % epoch_length).astype(float)

    # Model map
    model_type = trace[model_col].astype(str).map(_map_model_to_llama)

    # Source DC
    if src_dc_col is not None:
        src_dc = trace[src_dc_col].astype(int).mod(num_dcs)
    else:
        tmp = trace.copy()
        tmp["_row_idx"] = range(len(tmp))
        src_dc = tmp.apply(lambda r: _stable_dc_from_keys(r, num_dcs), axis=1).astype(int)

    detailed = pd.DataFrame({
        "epoch": epoch,
        "model_type": model_type,
        "num_tokens": toks.astype(float),
        "time_index": time_index,
        "src_dc": src_dc,
        "batch_size": 1,
    })

    agg = (detailed.groupby(["epoch","src_dc","model_type"], as_index=False)["num_tokens"]
           .sum()
           .rename(columns={"num_tokens":"total_tokens"}))

    # Dtypes
    agg["epoch"] = agg["epoch"].astype(int)
    agg["src_dc"] = agg["src_dc"].astype(int)
    agg["model_type"] = agg["model_type"].astype(str)
    agg["total_tokens"] = agg["total_tokens"].astype(float)

    return agg, detailed

# =========================
# Aggregated workload helpers
# =========================
def load_processed_epoch(agg_csv: str, epoch_idx: int) -> Dict[int, Dict[str, float]]:
    """Load agg CSV (epoch,src_dc,model_type,total_tokens) → {src_dc:{model:tokens}} for one epoch."""
    df = pd.read_csv(agg_csv)
    required = {"epoch","src_dc","model_type","total_tokens"}
    if not required.issubset(set(df.columns)):
        raise KeyError(f"{agg_csv} missing columns {required} (found {list(df.columns)})")
    df_e = df[df["epoch"].astype(int) == int(epoch_idx)].copy()
    out: Dict[int, Dict[str, float]] = {}
    for r in df_e.itertuples(index=False):
        dc = int(r.src_dc); model = str(r.model_type); tok = float(r.total_tokens)
        out.setdefault(dc, {}).setdefault(model, 0.0)
        out[dc][model] += tok
    return out

# =========================
# Schedule & Power plans
# =========================
def build_local_schedule_plan(work: Dict[int, Dict[str, float]]) -> Dict[Tuple[int, str], Dict[int, float]]:
    """
    Local-only routing: for each (src_dc, model) key in the workload mapping,
    route 100% to the same DC.
    Input: {src_dc: {model: total_tokens, ...}, ...}
    Output: {(src_dc, model): {src_dc: 1.0}}
    """
    plan: Dict[Tuple[int, str], Dict[int, float]] = {}
    for src_dc, models in work.items():
        for model in models:
            plan[(int(src_dc), str(model))] = {int(src_dc): 1.0}
    return plan

def build_all_on_power_plan(datacenters: List["rfs.Datacenter"]) -> Dict[int, Dict]:
    return {dc.dc_id: {"nodes": {n.node_id: "on" for n in dc.nodes}} for dc in datacenters}

# =========================
# Simulator entry dispatcher
# =========================
def run_simulator_epoch(datacenters, workload_by_dc: Dict[int, Dict[str, float]], epoch_len: int, epoch_hour: int, args) -> Tuple[dict, dict, dict]:
    """
    Calls the first available entry point:
      - simulate_epoch_rate_flow(dcs, workload_by_dc, epoch_length, epoch_hour)
      - run_epoch(dcs, workload_by_dc, epoch_length, epoch_hour)
      - LLM_Simulator(epoch_idx, workload_df_like, schedule_plan, power_plan)  [shim]
    Returns (stats, per_dc, leftovers)-like tuple.
    """
    # Dedicated rate-flow
    if hasattr(rfs, "simulate_epoch_rate_flow"):
        return rfs.simulate_epoch_rate_flow(datacenters, workload_by_dc, epoch_len, epoch_hour)

    # Generic
    if hasattr(rfs, "run_epoch"):
        return rfs.run_epoch(datacenters, workload_by_dc, epoch_len, epoch_hour)

    # Shim over legacy LLM_Simulator
    if hasattr(rfs, "LLM_Simulator"):
        rows = []
        for dc, models in workload_by_dc.items():
            for m, tok in models.items():
                rows.append({"src_dc": dc, "model_type": m, "total_tokens": tok})
        wl_df = pd.DataFrame(rows)
        dummy_sched = build_local_schedule_plan(workload_by_dc)
        power_plan = build_all_on_power_plan(datacenters)
        return rfs.LLM_Simulator(
            epoch_idx=args.epoch_index,
            epoch_work_df=wl_df,
            schedule_plan=dummy_sched,
            power_plan=power_plan,
            node_properties={},
            epoch_length=epoch_len,
            dc_latency_ms=args._lat_ms,   # injected below
            datacenters=datacenters,
            mode="rate",
            leftover_carry_in=None,
            epoch_hour=epoch_hour,
        )

    raise RuntimeError("No known simulator entry point found in Rate_Flow_Sim.")

# =========================
# Pretty printers
# =========================
def print_global(stats: Dict):
    print("\n=== Global Epoch Metrics ===")
    print(f"Epoch length (s):   {stats.get('epoch_length')}")
    print(f"Processed tokens:   {stats.get('processed_tokens', 0.0):.2f}")
    print(f"Avg TTFT (sec):     {stats.get('avg_ttft_sec', 0.0):.4f}")
    print(f"Energy (kWh):       {stats.get('energy_kwh', 0.0):.3f}")
    print(f"Carbon (kgCO2e):    {stats.get('carbon_emissions', 0.0):.3f}")
    print(f"Water (m^3):        {stats.get('water_usage', 0.0):.6f}")

def print_per_dc(per_dc: Dict[int, Dict]):
    print("\n=== Per-DC Breakdown ===")
    for dc_id in sorted(per_dc.keys()):
        m = per_dc[dc_id]
        eb = m.get("energy_breakdown", {})
        print(f"\n[DC {dc_id}]")
        print(f"  Utilization:          {m.get('dc_utilization', 0.0):.4f}")
        print(f"  Demand total (kWh):   {m.get('energy_kwh', 0.0):.3f}")
        print(f"  Grid import (kWh):    {eb.get('grid_import_kwh', float('nan')):.3f}")
        print(f"  Solar used (kWh):     {eb.get('solar_used_kwh', float('nan')):.3f}")
        print(f"  Batt discharge (kWh): {eb.get('battery_discharge_kwh', float('nan')):.3f}")
        print(f"  Carbon (kg):          {m.get('carbon_emissions', 0.0):.3f}")
        print(f"  Water (m^3):          {m.get('water_usage', 0.0):.6f}")
        pm = m.get("per_model", {})
        for model in sorted(pm.keys()):
            mm = pm[model]
            print(f"    - {model}: assigned={mm.get('assigned_tokens', 0.0):.1f}, "
                  f"processed={mm.get('processed_tokens', 0.0):.1f}, "
                  f"leftover={mm.get('leftover_tokens', 0.0):.1f}, "
                  f"util={mm.get('utilization', 0.0):.3f}")

# =========================
# Synthetic test
# =========================
def test_single_epoch_synthetic():
    print("\n[TEST] Synthetic single-epoch sanity check...")
    # One DC, one node, one processor
    proc = rfs.Processor(
        proc_id=0,
        node_id=0,
        epoch_length=10,             # 10 seconds
        power_state="on",
        tdp_kw=3.0,                  # active kW
        idle_kw=0.3,                 # idle kW
        model_perf={"Llama7b": {"ms_per_token": 1.0}},  # 1000 tokens/sec
    )
    node = rfs.Node(node_id=0, type_id=0, processors=[proc])
    dc = rfs.Datacenter(
        dc_id=0,
        nodes=[node],
        cop=3.0,
        other_hw_overhead=0.13,
        cooling_overhead_multiplier=3.0,
        carbon_intensity_kg_per_kwh=0.4,
        solar_kw_capacity=0.0,
        battery_kwh_capacity=0.0,
    )
    net = rfs.GeoNetwork([dc], [[0.0]])

    rows = [(0, "Llama7b", 12000.0)]  # demand
    schedule = { (0, "Llama7b"): {0: 1.0} }
    power = { 0: {"nodes": {0: "on"}} }

    stats, per_dc, leftovers = net.apply_schedule_plan_rate(rows, schedule, power, epoch_length=10)

    processed_expected = 10000.0
    leftover_expected  = 2000.0
    pm0 = per_dc[0]["per_model"]["Llama7b"]
    assert abs(pm0["processed_tokens"] - processed_expected) < 1e-6, "Processed tokens mismatch"
    assert abs(pm0["leftover_tokens"]  - leftover_expected)  < 1e-6, "Leftover tokens mismatch"
    assert per_dc[0]["dc_utilization"] > 0.99, "Utilization should be ~1.0"

    hours = 10.0 / 3600.0
    processor_kwh  = 3.0 * hours
    other_hw_kwh   = 0.13 * processor_kwh
    cooling_kwh    = (processor_kwh / dc.cop) * dc.cooling_overhead_multiplier
    demand_expected = processor_kwh + other_hw_kwh + cooling_kwh
    assert abs(per_dc[0]["energy_kwh"] - demand_expected) < 1e-9, "Demand energy mismatch"

    print("[TEST] OK: processed, leftovers, utilization, and energy match expectations.")

# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser(description="Run rate-flow simulator from raw or aggregated workload.")
    # Infra CSVs
    ap.add_argument("--node-specs", default="/mnt/c/Users/hmoor/Documents/LLMScheduling/ipdps/sim_specs/Node_Specs.csv")
    ap.add_argument("--dc-specs",
                    default="/mnt/c/Users/hmoor/Documents/LLMScheduling/ipdps/sim_specs/Datacenter_specs.csv")
    ap.add_argument("--a100", default="/mnt/c/Users/hmoor/Documents/LLMScheduling/ipdps/sim_specs/A100_GPU.csv")
    ap.add_argument("--h100", default="/mnt/c/Users/hmoor/Documents/LLMScheduling/ipdps/sim_specs/H100_GPU.csv")
    ap.add_argument("--latencies",
                    default="/mnt/c/Users/hmoor/Documents/LLMScheduling/ipdps/sim_specs/Geo_Latencies.csv")
    # Workload inputs (choose one)
    ap.add_argument("--workload",
                    default="/mnt/c/Users/hmoor/Documents/LLMScheduling/ipdps/simulator_ready_trace.csv",
                    help="Aggregated CSV with columns epoch,src_dc,model_type,total_tokens")
    ap.add_argument("--input-trace",
                    default="/mnt/c/Users/hmoor/Documents/LLMScheduling/ipdps/BurstGPT_without_fails_2.csv",
                    help="RAW trace CSV (will be processed to rate-flow if workload file is absent)")
    # Runtime
    ap.add_argument("--epoch-index", type=int, default=0, help="Epoch index to simulate")
    ap.add_argument("--epoch-length", type=int, default=900, help="Epoch length (seconds)")
    ap.add_argument("--num-dcs",     type=int, default=12, help="Number of datacenters (for raw trace processing)")
    ap.add_argument("--timestamps-in-ms", action="store_true", help="Treat raw timestamps as milliseconds explicitly")
    ap.add_argument("--hour", type=int, default=12, help="Hour-of-day to use for COP/solar (0-23)")
    ap.add_argument("--loop-24h", action="store_true", help="Simulate a 24hr loop (sum metrics)")
    # Debug outputs if processing raw trace
    ap.add_argument("--agg-out", default="simulator_ready_trace.csv", help="(When using --input-trace) write aggregated CSV here")
    ap.add_argument("--detailed-out", default="simulator_ready_detailed.csv", help="(When using --input-trace) write detailed CSV here")
    args = ap.parse_args()

    # --- Build DCs & latencies ---
    gpu_csvs = {"A100": args.a100, "H100": args.h100}
    datacenters = rfs.build_datacenters_from_csv(
        node_specs_csv=args.node_specs,
        gpu_perf_csvs=gpu_csvs,
        dc_specs_csv=args.dc_specs,
        epoch_length=args.epoch_length,
    )
    lat_ms = load_latency_matrix(args.latencies)
    # Stash latencies for the shim
    setattr(args, "_lat_ms", lat_ms)

    # --- Workload: either process RAW, or read aggregated ---
    if args.input_trace:
        trace = pd.read_csv(args.input_trace)
        agg, detailed = process_trace_for_simulator(
            trace=trace,
            epoch_length=args.epoch_length,
            num_dcs=args.num_dcs,
            timestamps_in_ms=args.timestamps_in_ms,
        )
        # write for audit
        agg.to_csv(args.agg_out, index=False)
        detailed.to_csv(args.detailed_out, index=False)
        workload_by_dc = load_processed_epoch(args.agg_out, args.epoch_index)
    elif args.workload:
        workload_by_dc = load_processed_epoch(args.workload, args.epoch_index)
    else:
        raise ValueError("Provide either --input-trace (raw) or --workload (aggregated).")

    # --- Plans ---
    schedule_plan = build_local_schedule_plan(workload_by_dc)
    power_plan = build_all_on_power_plan(datacenters)

    # --- Run ---
    if args.loop_24h:
        cumulative = {"energy_kwh": 0.0, "carbon_emissions": 0.0,
                      "water_usage": 0.0, "processed_tokens": 0.0}
        leftovers = None
        for h in range(24):
            stats, per_dc, leftovers = run_simulator_epoch(datacenters, workload_by_dc, args.epoch_length, h, args)
            print(f"\n===== Hour {h} =====")
            print_global(stats)
            print_per_dc(per_dc)
            cumulative["energy_kwh"]       += stats.get("energy_kwh", 0.0)
            cumulative["carbon_emissions"] += stats.get("carbon_emissions", 0.0)
            cumulative["water_usage"]      += stats.get("water_usage", 0.0)
            cumulative["processed_tokens"] += stats.get("processed_tokens", 0.0)
        print("\n===== 24h Cumulative =====")
        print(f"Energy (kWh):    {cumulative['energy_kwh']:.3f}")
        print(f"Carbon (kg):     {cumulative['carbon_emissions']:.3f}")
        print(f"Water (m^3):     {cumulative['water_usage']:.6f}")
        print(f"Tokens processed:{cumulative['processed_tokens']:.1f}")
    else:
        stats, per_dc, leftovers = run_simulator_epoch(datacenters, workload_by_dc, args.epoch_length, args.hour, args)
        print_global(stats)
        print_per_dc(per_dc)

    # --- Synthetic sanity test ---
    test_single_epoch_synthetic()

if __name__ == "__main__":
    main()

