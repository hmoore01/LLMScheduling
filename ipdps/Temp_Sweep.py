#!/usr/bin/env python3
"""
Temp_Sweep.py

Temperature sweep on a single datacenter using Rate_Flow_Sim and
simulator_ready_trace.csv.

Assumptions about the workload file:
  simulator_ready_trace.csv has EXACTLY these columns:

      epoch, src_dc, model_type, total_tokens

We convert them to what Rate_Flow_Sim expects internally:

      source_dc  <- src_dc
      arrival_ms <- 0 for ALL rows (all at start of epoch)
      model      <- model_type
      tokens     <- total_tokens

For each temperature setpoint in [temp_min, temp_max] °C, we:
  - set that DC's temp_c_setpoint
  - run the same epoch with the same workload rows
  - record total_energy and energy_cost

Example:
    python Temp_Sweep.py \
        --spec_dir sim_specs \
        --workload_csv simulator_ready_trace.csv \
        --epoch 0 \
        --dc 0 \
        --filter_source_dc \
        --temp_min 20 --temp_max 60 --temp_step 1 \
        --output_csv temp_sweep_dc0.csv
"""

import argparse
import pandas as pd
from typing import Optional

from Rate_Flow_Sim import LLM_Simulator


def build_epoch_workload(
    sim: LLM_Simulator,
    workload_csv: str,
    epoch_idx: int,
    focus_dc: Optional[int] = None,   # <-- fixed typing
) -> pd.DataFrame:
    """
    Load simulator_ready_trace.csv for a single epoch (and optionally a single DC)
    and convert it to the exact columns Rate_Flow_Sim expects:
        source_dc, arrival_ms, model, tokens
    Here, ALL arrivals are at the start of the epoch (arrival_ms = 0).
    """
    df = pd.read_csv(workload_csv)
    expected_cols = {"epoch", "src_dc", "model_type", "total_tokens"}
    if set(df.columns) != expected_cols:
        raise ValueError(
            f"Expected columns {sorted(expected_cols)}, "
            f"but found {sorted(df.columns)} in {workload_csv}."
        )

    df = df[df["epoch"] == epoch_idx].copy()
    if df.empty:
        raise ValueError(f"No rows found for epoch == {epoch_idx}")

    if focus_dc is not None:
        df = df[df["src_dc"] == focus_dc].copy()
        if df.empty:
            raise ValueError(f"No rows found for epoch == {epoch_idx} and src_dc == {focus_dc}")

    df = df.reset_index(drop=True)
    df = df.rename(columns={"src_dc": "source_dc", "model_type": "model", "total_tokens": "tokens"})
    df["arrival_ms"] = 0  # all at start of epoch
    df["source_dc"] = df["source_dc"].astype(int)
    df["tokens"] = df["tokens"].astype(int)
    return df[["source_dc", "arrival_ms", "model", "tokens"]]


def build_local_power_plan(sim: LLM_Simulator, focus_dc: int) -> dict:
    """
    Build a simple power plan:
      - focus_dc: all units ON
      - all other DCs: OFF (so they don't add idle energy/cooling)
    """
    power_plan: dict = {}
    for dc_id in sim.datacenters.keys():
        if dc_id == focus_dc:
            power_plan[dc_id] = {"all": "ON"}
        else:
            power_plan[dc_id] = {"all": "OFF"}
    return power_plan


def main():
    parser = argparse.ArgumentParser(
        description="Single-DC temperature sweep using simulator_ready_trace.csv"
    )
    parser.add_argument(
        "--spec_dir",
        default="sim_specs",
        help="Directory with Datacenter_specs.csv, Node_Specs.csv, etc.",
    )
    parser.add_argument(
        "--workload_csv",
        required=True,
        help="Path to simulator_ready_trace.csv",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=0,
        help="Epoch index to reuse for the sweep.",
    )
    parser.add_argument(
        "--dc",
        type=int,
        default=0,
        help="Datacenter ID to focus on.",
    )
    parser.add_argument(
        "--filter_source_dc",
        action="store_true",
        help="If set, only use rows with src_dc == --dc from the trace.",
    )
    parser.add_argument(
        "--temp_min",
        type=float,
        default=20.0,
        help="Minimum setpoint temperature (°C).",
    )
    parser.add_argument(
        "--temp_max",
        type=float,
        default=40.0,
        help="Maximum setpoint temperature (°C).",
    )
    parser.add_argument(
        "--temp_step",
        type=float,
        default=1.0,
        help="Step size for temperature sweep (°C).",
    )
    parser.add_argument(
        "--output_csv",
        default=None,
        help="Optional path to save results as CSV.",
    )

    args = parser.parse_args()

    # ---- 1) Build simulator so we know epoch_length, DCs, etc. ----
    sim = LLM_Simulator(spec_dir=args.spec_dir, debug=False)

    if args.dc not in sim.datacenters:
        raise ValueError(
            f"Requested DC {args.dc} not found in simulator DC IDs: "
            f"{list(sim.datacenters.keys())}"
        )

    # ---- 2) Build the epoch workload for this DC ----
    focus_dc = args.dc if args.filter_source_dc else None
    workload_df = build_epoch_workload(
        sim=sim,
        workload_csv=args.workload_csv,
        epoch_idx=args.epoch,
        focus_dc=focus_dc,
    )

    # Schedule plan: send everything to the focus DC
    schedule_plan = {"default_target_dc": args.dc}

    # Power plan: focus DC ON, others OFF
    power_plan = build_local_power_plan(sim, args.dc)

    # ---- 3) Sweep temperature setpoints ----
    print("Temp_C,Total_Energy_KWh,Energy_Cost_USD")

    results = []
    temp = args.temp_min
    while temp <= args.temp_max + 1e-9:
        # Set the setpoint for the focus DC only
        dc_obj = sim.datacenters[args.dc]
        dc_obj.temp_c_setpoint = float(temp)

        # Run the epoch. run_epoch() resets per-epoch stats internally.
        metrics, details, dc_usage = sim.run_epoch(
            epoch_idx=args.epoch,
            workload_df=workload_df,
            schedule_plan=schedule_plan,
            power_plan=power_plan,
        )

        total_energy = float(metrics.get("total_energy", 0.0))
        energy_cost = float(metrics.get("energy_cost", 0.0))

        print(f"{temp:.2f},{total_energy:.6f},{energy_cost:.6f}")

        results.append(
            {
                "temp_c": temp,
                "total_energy_kwh": total_energy,
                "energy_cost_usd": energy_cost,
            }
        )

        temp += args.temp_step

    # ---- 4) Optional: write to CSV ----
    if args.output_csv is not None:
        out_df = pd.DataFrame(results)
        out_df.to_csv(args.output_csv, index=False)
        print(f"\nSaved sweep results to {args.output_csv}")


if __name__ == "__main__":
    main()


