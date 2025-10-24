#!/usr/bin/env python3
"""
run_rate_sim.py — End-to-end runner for the Rate-Flow LLM Scheduling Simulator

What this script does
- Loads CSV inputs (Node_Specs, A100/H100 perf, Datacenter specs, Geo latencies, Workload)
- Builds Datacenter → Node → Processor graph via build_datacenters_from_csv(...)
- Constructs a default schedule_plan (local routing unless overridden)
- Runs a single epoch with the new rate-based LLM_Simulator(..., mode="rate")
- Prints global metrics + per-DC energy/carbon/water breakdown
- (Optional) Loops over 24 hours to include hourly solar profiles
- Includes a unit-style "testing" section with a synthetic, fake-data sanity check

Usage (defaults point to the example paths you uploaded):
  python run_rate_sim.py \
    --node-specs /mnt/data/Node_Specs.csv \
    --dc-specs /mnt/data/Datacenter_specs.csv \
    --a100 /mnt/data/A100_GPU.csv \
    --h100 /mnt/data/H100_GPU.csv \
    --latencies /mnt/data/Geo_Latencies.csv \
    --workload /mnt/data/Workload_Granularity.csv \
    --epoch-length 900 \
    --hour 12

Notes
- Requires rate_flow_simulator.py (the module we built) in the same directory.
- The GPU perf CSVs are passed as a dict: {"A100": path, "H100": path}. If you have more types, add keys.
- Workload CSV is expected to have: src_dc,model_type,total_tokens
- Latencies CSV: square matrix in ms, rows=src_dc, cols=tgt_dc (with header row or not; parser will try both)
"""

from __future__ import annotations
import argparse
import csv
import math
import os
from typing import Dict, Tuple, List

try:
    import pandas as pd  # optional but convenient
    PANDAS = True
except Exception:
    PANDAS = False

# Import the module created earlier on the canvas
import Rate_Flow_Sim as rfs


# -----------------------------
# Helpers to load CSV inputs
# -----------------------------

def load_latency_matrix(path: str) -> List[List[float]]:
    """Load a square latency matrix (ms). Supports with/without header row.
    Returns a list of lists [src][tgt] -> ms.
    """
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    # Try to detect header (non-numeric first cell)
    def is_float(s: str) -> bool:
        try:
            float(s)
            return True
        except Exception:
            return False
    if rows and rows[0] and not is_float(rows[0][0]):
        # assume first row is header, drop first col too if header-like
        data = []
        for r in rows[1:]:
            # skip label in col0 if present
            vals = r[1:] if (r and not is_float(r[0])) else r
            data.append([float(x) for x in vals])
        return data
    else:
        return [[float(x) for x in r] for r in rows]


def load_workload(path: str):
    """Return iterable of (src_dc, model_type, total_tokens)."""
    if PANDAS:
        df = pd.read_csv(path)
        # normalize column names
        cols = {c.lower(): c for c in df.columns}
        src = cols.get("src_dc", "src_dc")
        model = cols.get("model_type", "model_type")
        toks = cols.get("total_tokens", "total_tokens")
        return df[[src, model, toks]].rename(columns={src: "src_dc", model: "model_type", toks: "total_tokens"})
    # csv fallback
    out = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append((int(row["src_dc"]), str(row["model_type"]), float(row["total_tokens"])) )
    return out


# -----------------------------
# Default schedule plan builder
# -----------------------------

def build_local_schedule_plan(work_rows) -> Dict[Tuple[int, str], Dict[int, float]]:
    """Routes all work to its origin DC. Work rows can be a pandas DF or iterable.
    Returns {(src_dc, model): {src_dc: 1.0}}
    """
    plan: Dict[Tuple[int, str], Dict[int, float]] = {}
    # Pandas path
    if PANDAS and hasattr(work_rows, "itertuples"):
        for r in work_rows.itertuples(index=False):
            key = (int(getattr(r, "src_dc")), str(getattr(r, "model_type")))
            plan.setdefault(key, {key[0]: 1.0})
        return plan
    # Iterable path
    for src, model, _toks in work_rows:
        key = (int(src), str(model))
        plan.setdefault(key, {int(src): 1.0})
    return plan


# -----------------------------
# Pretty printers
# -----------------------------

def print_global(stats: Dict):
    print("\n=== Global Epoch Metrics ===")
    print(f"Epoch length (s):   {stats['epoch_length']}")
    print(f"Processed tokens:   {stats['processed_tokens']:.2f}")
    print(f"Avg TTFT (sec):     {stats['avg_ttft_sec']:.4f}")
    print(f"Energy (kWh):       {stats['energy_kwh']:.3f}")
    print(f"Carbon (kgCO2e):    {stats['carbon_emissions']:.3f}")
    print(f"Water (m^3):        {stats['water_usage']:.6f}")


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
        # Per-model quick view
        pm = m.get("per_model", {})
        for model in sorted(pm.keys()):
            mm = pm[model]
            print(f"    - {model}: assigned={mm['assigned_tokens']:.1f}, processed={mm['processed_tokens']:.1f}, leftover={mm['leftover_tokens']:.1f}, util={mm['utilization']:.3f}")


# -----------------------------
# Testing (synthetic sanity check)
# -----------------------------

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

    # Assign 12,000 tokens; capacity = 1000 tok/s * 10 s = 10,000 tokens
    rows = [(0, "Llama7b", 12000.0)]
    schedule = { (0, "Llama7b"): {0: 1.0} }
    power = { 0: {"nodes": {0: "on"}} }

    stats, per_dc, leftovers = net.apply_schedule_plan_rate(rows, schedule, power, epoch_length=10)

    # Expectations
    processed_expected = 10000.0
    leftover_expected = 2000.0
    util_expected = processed_expected / processed_expected  # 1.0

    pm0 = per_dc[0]["per_model"]["Llama7b"]
    assert abs(pm0["processed_tokens"] - processed_expected) < 1e-6, "Processed tokens mismatch"
    assert abs(pm0["leftover_tokens"] - leftover_expected) < 1e-6, "Leftover tokens mismatch"
    assert per_dc[0]["dc_utilization"] > 0.99, "Utilization should be ~1.0"

    # Energy demand check (approx):
    # duty=1.0, hours=10/3600
    hours = 10.0 / 3600.0
    processor_kwh = 3.0 * hours + 0.0 * hours  # active 3kW, no idle since duty=1
    other_hw_kwh = 0.13 * processor_kwh
    cooling_kwh = (processor_kwh / dc.cop) * dc.cooling_overhead_multiplier
    demand_expected = processor_kwh + other_hw_kwh + cooling_kwh
    assert abs(per_dc[0]["energy_kwh"] - demand_expected) < 1e-9, "Demand energy mismatch"

    print("[TEST] OK: processed, leftovers, utilization, and energy match expectations.")


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node-specs", default="/mnt/c/Users/hmoor/Documents/LLMScheduling/ipdps/sim_specs/Node_Specs.csv")
    ap.add_argument("--dc-specs", default="/mnt/c/Users/hmoor/Documents/LLMScheduling/ipdps/sim_specs/Datacenter_specs.csv")
    ap.add_argument("--a100", default="/mnt/c/Users/hmoor/Documents/LLMScheduling/ipdps/sim_specs/A100_GPU.csv")
    ap.add_argument("--h100", default="/mnt/c/Users/hmoor/Documents/LLMScheduling/ipdps/sim_specs/H100_GPU.csv")
    ap.add_argument("--latencies", default="/mnt/c/Users/hmoor/Documents/LLMScheduling/ipdps/sim_specs/Geo_Latencies.csv")
    ap.add_argument("--workload", default="/mnt/c/Users/hmoor/Documents/LLMScheduling/ipdps/sim_specs/Workload_Granularity.csv")
    ap.add_argument("--epoch-length", type=int, default=900)
    ap.add_argument("--hour", type=int, default=12, help="0..23; used for solar profile selection")
    ap.add_argument("--loop-24h", action="store_true", help="simulate a 24hr loop, summing metrics")
    args = ap.parse_args()

    # Build datacenters from CSVs
    gpu_csvs = {"A100": args.a100, "H100": args.h100}
    datacenters = rfs.build_datacenters_from_csv(
        node_specs_csv=args.node_specs,
        gpu_perf_csvs=gpu_csvs,
        dc_specs_csv=args.dc_specs,
        epoch_length=args.epoch_length,
    )

    # Load latency matrix
    lat_ms = load_latency_matrix(args.latencies)

    # Load workload
    workload = load_workload(args.workload)

    # Construct default schedule (local-only)
    schedule_plan = build_local_schedule_plan(workload)

    # Construct default power plan: turn on all nodes
    power_plan: Dict[int, Dict] = {}
    for dc in datacenters:
        power_plan[dc.dc_id] = {"nodes": {n.node_id: "on" for n in dc.nodes}}

    if args.loop_24h:
        # 24-hour loop with solar variation
        cumulative = {"energy_kwh": 0.0, "carbon_emissions": 0.0, "water_usage": 0.0, "processed_tokens": 0.0}
        net = rfs.GeoNetwork(datacenters, lat_ms)
        leftovers = None
        for h in range(24):
            stats, per_dc, leftovers = rfs.LLM_Simulator(
                epoch_idx=h,
                epoch_work_df=workload,
                schedule_plan=schedule_plan,
                power_plan=power_plan,
                node_properties={},
                epoch_length=args.epoch_length,
                dc_latency_ms=lat_ms,
                datacenters=datacenters,
                mode="rate",
                leftover_carry_in=leftovers,
                epoch_hour=h,
            )
            print(f"\n===== Hour {h} =====")
            print_global(stats)
            print_per_dc(per_dc)
            cumulative["energy_kwh"] += stats["energy_kwh"]
            cumulative["carbon_emissions"] += stats["carbon_emissions"]
            cumulative["water_usage"] += stats["water_usage"]
            cumulative["processed_tokens"] += stats["processed_tokens"]
        print("\n===== 24h Cumulative =====")
        print(f"Energy (kWh):    {cumulative['energy_kwh']:.3f}")
        print(f"Carbon (kg):     {cumulative['carbon_emissions']:.3f}")
        print(f"Water (m^3):     {cumulative['water_usage']:.6f}")
        print(f"Tokens processed:{cumulative['processed_tokens']:.1f}")
        # Also run synthetic test at the end
        test_single_epoch_synthetic()
        return

    # Single-epoch run
    stats, per_dc, leftovers = rfs.LLM_Simulator(
        epoch_idx=0,
        epoch_work_df=workload,
        schedule_plan=schedule_plan,
        power_plan=power_plan,
        node_properties={},
        epoch_length=args.epoch_length,
        dc_latency_ms=lat_ms,
        datacenters=datacenters,
        mode="rate",
        leftover_carry_in=None,
        epoch_hour=args.hour,
    )

    print_global(stats)
    print_per_dc(per_dc)

    # Run synthetic test after the real run
    test_single_epoch_synthetic()


if __name__ == "__main__":
    main()


# -----------------------------
# Testing & CLI (optional)
# -----------------------------
if __name__ == "__main__":
    import argparse, csv
    try:
        import pandas as pd
        PANDAS = True
    except Exception:
        PANDAS = False

    def _load_latency_matrix(path: str):
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
                data.append([float(x) for x in vals])
            return data
        else:
            return [[float(x) for x in r] for r in rows]

    def _load_workload(path: str):
        if PANDAS:
            df = pd.read_csv(path)
            cols = {c.lower(): c for c in df.columns}
            src = cols.get("src_dc", "src_dc")
            model = cols.get("model_type", "model_type")
            toks = cols.get("total_tokens", "total_tokens")
            return df[[src, model, toks]].rename(columns={src: "src_dc", model: "model_type", toks: "total_tokens"})
        out = []
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                out.append((int(row["src_dc"]), str(row["model_type"]), float(row["total_tokens"])) )
        return out

    def _build_local_schedule_plan(work_rows):
        plan = {}
        if PANDAS and hasattr(work_rows, "itertuples"):
            for r in work_rows.itertuples(index=False):
                key = (int(getattr(r, "src_dc")), str(getattr(r, "model_type")))
                plan.setdefault(key, {key[0]: 1.0})
            return plan
        for src, model, _toks in work_rows:
            key = (int(src), str(model))
            plan.setdefault(key, {int(src): 1.0})
        return plan

    def _print_global(stats: Dict):
        print("=== Global Epoch Metrics ===")
        print(f"Epoch length (s):   {stats['epoch_length']}")
        print(f"Processed tokens:   {stats['processed_tokens']:.2f}")
        print(f"Avg TTFT (sec):     {stats['avg_ttft_sec']:.4f}")
        print(f"Energy (kWh):       {stats['energy_kwh']:.3f}")
        print(f"Carbon (kgCO2e):    {stats['carbon_emissions']:.3f}")
        print(f"Water (m^3):        {stats['water_usage']:.6f}")

    def _print_per_dc(per_dc: Dict[int, Dict]):
        print("=== Per-DC Breakdown ===")
        for dc_id in sorted(per_dc.keys()):
            m = per_dc[dc_id]
            eb = m.get("energy_breakdown", {})
            print(f"[DC {dc_id}]")
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
                print(f"    - {model}: assigned={mm['assigned_tokens']:.1f}, processed={mm['processed_tokens']:.1f}, leftover={mm['leftover_tokens']:.1f}, util={mm['utilization']:.3f}")

    def _test_single_epoch_synthetic():
        print("[TEST] Synthetic single-epoch sanity check...")
        proc = Processor(
            proc_id=0,
            node_id=0,
            epoch_length=10,
            power_state="on",
            tdp_kw=3.0,
            idle_kw=0.3,
            model_perf={"Llama7b": {"ms_per_token": 1.0}},
        )
        node = Node(node_id=0, type_id=0, processors=[proc])
        dc = Datacenter(
            dc_id=0,
            nodes=[node],
            cop=3.0,
            other_hw_overhead=0.13,
            cooling_overhead_multiplier=3.0,
            carbon_intensity_kg_per_kwh=0.4,
            solar_kw_capacity=0.0,
            battery_kwh_capacity=0.0,
        )
        net = GeoNetwork([dc], [[0.0]])
        rows = [(0, "Llama7b", 12000.0)]
        schedule = { (0, "Llama7b"): {0: 1.0} }
        power = { 0: {"nodes": {0: "on"}} }
        stats, per_dc, leftovers = net.apply_schedule_plan_rate(rows, schedule, power, epoch_length=10)
        processed_expected = 10000.0
        leftover_expected = 2000.0
        pm0 = per_dc[0]["per_model"]["Llama7b"]
        assert abs(pm0["processed_tokens"] - processed_expected) < 1e-6, "Processed tokens mismatch"
        assert abs(pm0["leftover_tokens"] - leftover_expected) < 1e-6, "Leftover tokens mismatch"
        assert per_dc[0]["dc_utilization"] > 0.99, "Utilization should be ~1.0"
        hours = 10.0 / 3600.0
        processor_kwh = 3.0 * hours
        other_hw_kwh = 0.13 * processor_kwh
        cooling_kwh = (processor_kwh / dc.cop) * dc.cooling_overhead_multiplier
        demand_expected = processor_kwh + other_hw_kwh + cooling_kwh
        assert abs(per_dc[0]["energy_kwh"] - demand_expected) < 1e-9, "Demand energy mismatch"
        print("[TEST] OK: processed, leftovers, utilization, and energy match expectations.")

    ap = argparse.ArgumentParser(description="Run rate-flow simulator and built-in tests")
    ap.add_argument("--node-specs", default="/mnt/c/Users/hmoor/Documents/LLMScheduling/ipdps/sim_specs/Node_Specs.csv")
    ap.add_argument("--dc-specs", default="/mnt/c/Users/hmoor/Documents/LLMScheduling/ipdps/sim_specs/Datacenter_specs.csv")
    ap.add_argument("--a100", default="/mnt/c/Users/hmoor/Documents/LLMScheduling/ipdps/sim_specs/A100_GPU.csv")
    ap.add_argument("--h100", default="/mnt/c/Users/hmoor/Documents/LLMScheduling/ipdps/sim_specs/H100_GPU.csv")
    ap.add_argument("--latencies", default="/mnt/c/Users/hmoor/Documents/LLMScheduling/ipdps/sim_specs/Geo_Latencies.csv")
    ap.add_argument("--workload", default="/mnt/c/Users/hmoor/Documents/LLMScheduling/ipdps/sim_specs/Workload_Granularity.csv")
    ap.add_argument("--epoch-length", type=int, default=900)
    ap.add_argument("--hour", type=int, default=12)
    ap.add_argument("--loop-24h", action="store_true")
    ap.add_argument("--run-tests", action="store_true")
    args = ap.parse_args()

    # Build from CSVs
    gpu_csvs = {"A100": args.a100, "H100": args.h100}
    dcs = build_datacenters_from_csv(
        node_specs_csv=args.node_specs,
        gpu_perf_csvs=gpu_csvs,
        dc_specs_csv=args.dc_specs,
        epoch_length=args.epoch_length,
    )

    lat_ms = _load_latency_matrix(args.latencies)
    work = _load_workload(args.workload)
    plan = _build_local_schedule_plan(work)
    power_plan = {dc.dc_id: {"nodes": {n.node_id: "on" for n in dc.nodes}} for dc in dcs}

    # Run single epoch or 24h
    if args.loop_24h:
        net = GeoNetwork(dcs, lat_ms)
        leftovers = None
        cumulative = {"energy_kwh": 0.0, "carbon_emissions": 0.0, "water_usage": 0.0, "processed_tokens": 0.0}
        for h in range(24):
            stats, per_dc, leftovers = LLM_Simulator(
                epoch_idx=h,
                epoch_work_df=work,
                schedule_plan=plan,
                power_plan=power_plan,
                node_properties={},
                epoch_length=args.epoch_length,
                dc_latency_ms=lat_ms,
                datacenters=dcs,
                mode="rate",
                leftover_carry_in=leftovers,
                epoch_hour=h,
            )
            print(f"===== Hour {h} =====")
            _print_global(stats)
            _print_per_dc(per_dc)
            cumulative["energy_kwh"] += stats["energy_kwh"]
            cumulative["carbon_emissions"] += stats["carbon_emissions"]
            cumulative["water_usage"] += stats["water_usage"]
            cumulative["processed_tokens"] += stats["processed_tokens"]
        print("=== 24h Cumulative =====")
        print(f"Energy (kWh):    {cumulative['energy_kwh']:.3f}")
        print(f"Carbon (kg):     {cumulative['carbon_emissions']:.3f}")
        print(f"Water (m^3):     {cumulative['water_usage']:.6f}")
        print(f"Tokens processed:{cumulative['processed_tokens']:.1f}")
    else:
        stats, per_dc, leftovers = LLM_Simulator(
            epoch_idx=0,
            epoch_work_df=work,
            schedule_plan=plan,
            power_plan=power_plan,
            node_properties={},
            epoch_length=args.epoch_length,
            dc_latency_ms=lat_ms,
            datacenters=dcs,
            mode="rate",
            leftover_carry_in=None,
            epoch_hour=args.hour,
        )
        _print_global(stats)
        _print_per_dc(per_dc)

    if args.run_tests:
        _test_single_epoch_synthetic()


# -----------------------------
# Testing & CLI (optional)
# -----------------------------
if __name__ == "__main__":
    import argparse, csv
    try:
        import pandas as pd
        PANDAS = True
    except Exception:
        PANDAS = False

    def _load_latency_matrix(path: str):
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
                data.append([float(x) for x in vals])
            return data
        else:
            return [[float(x) for x in r] for r in rows]

    def _load_workload(path: str):
        if PANDAS:
            df = pd.read_csv(path)
            cols = {c.lower(): c for c in df.columns}
            src = cols.get("src_dc", "src_dc")
            model = cols.get("model_type", "model_type")
            toks = cols.get("total_tokens", "total_tokens")
            return df[[src, model, toks]].rename(columns={src: "src_dc", model: "model_type", toks: "total_tokens"})
        out = []
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                out.append((int(row["src_dc"]), str(row["model_type"]), float(row["total_tokens"])) )
        return out

    def _build_local_schedule_plan(work_rows):
        plan = {}
        if PANDAS and hasattr(work_rows, "itertuples"):
            for r in work_rows.itertuples(index=False):
                key = (int(getattr(r, "src_dc")), str(getattr(r, "model_type")))
                plan.setdefault(key, {key[0]: 1.0})
            return plan
        for src, model, _toks in work_rows:
            key = (int(src), str(model))
            plan.setdefault(key, {int(src): 1.0})
        return plan

    def _print_global(stats: Dict):
        print("=== Global Epoch Metrics ===")
        print(f"Epoch length (s):   {stats['epoch_length']}")
        print(f"Processed tokens:   {stats['processed_tokens']:.2f}")
        print(f"Avg TTFT (sec):     {stats['avg_ttft_sec']:.4f}")
        print(f"Energy (kWh):       {stats['energy_kwh']:.3f}")
        print(f"Carbon (kgCO2e):    {stats['carbon_emissions']:.3f}")
        print(f"Water (m^3):        {stats['water_usage']:.6f}")

    def _print_per_dc(per_dc: Dict[int, Dict]):
        print("=== Per-DC Breakdown ===")
        for dc_id in sorted(per_dc.keys()):
            m = per_dc[dc_id]
            eb = m.get("energy_breakdown", {})
            print(f"[DC {dc_id}]")
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
                print(f"    - {model}: assigned={mm['assigned_tokens']:.1f}, processed={mm['processed_tokens']:.1f}, leftover={mm['leftover_tokens']:.1f}, util={mm['utilization']:.3f}")

    def _test_single_epoch_synthetic():
        print("[TEST] Synthetic single-epoch sanity check...")
        proc = Processor(
            proc_id=0,
            node_id=0,
            epoch_length=10,
            power_state="on",
            tdp_kw=3.0,
            idle_kw=0.3,
            model_perf={"Llama7b": {"ms_per_token": 1.0}},
        )
        node = Node(node_id=0, type_id=0, processors=[proc])
        dc = Datacenter(
            dc_id=0,
            nodes=[node],
            cop=3.0,
            other_hw_overhead=0.13,
            cooling_overhead_multiplier=3.0,
            carbon_intensity_kg_per_kwh=0.4,
            solar_kw_capacity=0.0,
            battery_kwh_capacity=0.0,
        )
        net = GeoNetwork([dc], [[0.0]])
        rows = [(0, "Llama7b", 12000.0)]
        schedule = { (0, "Llama7b"): {0: 1.0} }
        power = { 0: {"nodes": {0: "on"}} }
        stats, per_dc, leftovers = net.apply_schedule_plan_rate(rows, schedule, power, epoch_length=10)
        processed_expected = 10000.0
        leftover_expected = 2000.0
        pm0 = per_dc[0]["per_model"]["Llama7b"]
        assert abs(pm0["processed_tokens"] - processed_expected) < 1e-6, "Processed tokens mismatch"
        assert abs(pm0["leftover_tokens"] - leftover_expected) < 1e-6, "Leftover tokens mismatch"
        assert per_dc[0]["dc_utilization"] > 0.99, "Utilization should be ~1.0"
        hours = 10.0 / 3600.0
        processor_kwh = 3.0 * hours
        other_hw_kwh = 0.13 * processor_kwh
        cooling_kwh = (processor_kwh / dc.cop) * dc.cooling_overhead_multiplier
        demand_expected = processor_kwh + other_hw_kwh + cooling_kwh
        assert abs(per_dc[0]["energy_kwh"] - demand_expected) < 1e-9, "Demand energy mismatch"
        print("[TEST] OK: processed, leftovers, utilization, and energy match expectations.")

    ap = argparse.ArgumentParser(description="Rate-flow simulator with built-in runner & tests")
    ap.add_argument("--node-specs", default="/mnt/c/Users/hmoor/Documents/LLMScheduling/ipdps/sim_specs/Node_Specs.csv")
    ap.add_argument("--dc-specs", default="/mnt/c/Users/hmoor/Documents/LLMScheduling/ipdps/sim_specs/Datacenter_specs.csv")
    ap.add_argument("--a100", default="/mnt/c/Users/hmoor/Documents/LLMScheduling/ipdps/sim_specs/A100_GPU.csv")
    ap.add_argument("--h100", default="/mnt/c/Users/hmoor/Documents/LLMScheduling/ipdps/sim_specs/H100_GPU.csv")
    ap.add_argument("--latencies", default="/mnt/c/Users/hmoor/Documents/LLMScheduling/ipdps/sim_specs/Geo_Latencies.csv")
    ap.add_argument("--workload", default="/mnt/c/Users/hmoor/Documents/LLMScheduling/ipdps/sim_specs/Workload_Granularity.csv")
    ap.add_argument("--epoch-length", type=int, default=900)
    ap.add_argument("--hour", type=int, default=12)
    ap.add_argument("--loop-24h", action="store_true")
    ap.add_argument("--run-tests", action="store_true")
    args = ap.parse_args()

    # Build from CSVs
    gpu_csvs = {"A100": args.a100, "H100": args.h100}
    dcs = build_datacenters_from_csv(
        node_specs_csv=args.node_specs,
        gpu_perf_csvs=gpu_csvs,
        dc_specs_csv=args.dc_specs,
        epoch_length=args.epoch_length,
    )

    lat_ms = _load_latency_matrix(args.latencies)
    work = _load_workload(args.workload)
    plan = _build_local_schedule_plan(work)
    power_plan = {dc.dc_id: {"nodes": {n.node_id: "on" for n in dc.nodes}} for dc in dcs}

    # Run single epoch or 24h
    if args.loop_24h:
        net = GeoNetwork(dcs, lat_ms)
        leftovers = None
        cumulative = {"energy_kwh": 0.0, "carbon_emissions": 0.0, "water_usage": 0.0, "processed_tokens": 0.0}
        for h in range(24):
            stats, per_dc, leftovers = LLM_Simulator(
                epoch_idx=h,
                epoch_work_df=work,
                schedule_plan=plan,
                power_plan=power_plan,
                node_properties={},
                epoch_length=args.epoch_length,
                dc_latency_ms=lat_ms,
                datacenters=dcs,
                mode="rate",
                leftover_carry_in=leftovers,
                epoch_hour=h,
            )
            print(f"===== Hour {h} =====")
            _print_global(stats)
            _print_per_dc(per_dc)
            cumulative["energy_kwh"] += stats["energy_kwh"]
            cumulative["carbon_emissions"] += stats["carbon_emissions"]
            cumulative["water_usage"] += stats["water_usage"]
            cumulative["processed_tokens"] += stats["processed_tokens"]
        print("===== 24h Cumulative =====")
        print(f"Energy (kWh):    {cumulative['energy_kwh']:.3f}")
        print(f"Carbon (kg):     {cumulative['carbon_emissions']:.3f}")
        print(f"Water (m^3):     {cumulative['water_usage']:.6f}")
        print(f"Tokens processed:{cumulative['processed_tokens']:.1f}")
    else:
        stats, per_dc, leftovers = LLM_Simulator(
            epoch_idx=0,
            epoch_work_df=work,
            schedule_plan=plan,
            power_plan=power_plan,
            node_properties={},
            epoch_length=args.epoch_length,
            dc_latency_ms=lat_ms,
            datacenters=dcs,
            mode="rate",
            leftover_carry_in=None,
            epoch_hour=args.hour,
        )
        _print_global(stats)
        _print_per_dc(per_dc)

    if args.run_tests:
        _test_single_epoch_synthetic()
