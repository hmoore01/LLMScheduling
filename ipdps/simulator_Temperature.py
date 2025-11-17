# simulator_Temperature.py
#
# Standalone CPU + temperature sweep driver built on Rate_Flow_Sim.LLM_Simulator.
# - Forces all-CPU node allocation via Rate_Flow_Sim.MANUAL_NODE_TYPE_COUNTS.
# - Sets datacenter temp_c_setpoint manually per sweep step.
# - Assumes ProcNode has workload_class and a non-linear IT power vs T curve.
# - Can generate a synthetic workload: 1 povray_r request per node in ONE DC,
#   across multiple epochs (default 4).
#
# Example (synthetic workload, 4 epochs, DC 0):
#   python simulator_Temperature.py \
#       --synthetic-workload \
#       --cpu-type-id 6 \
#       --cpu-nodes-per-dc 4 \
#       --temp-min 20 --temp-max 40 --temp-step 2 \
#       --epoch-index 0 --debug
#
# Example (also save synthetic workload CSV):
#   python simulator_Temperature.py \
#       --synthetic-workload \
#       --workload-csv povray_synth.csv \
#       --save-synthetic-workload \
#       --cpu-type-id 6 \
#       --cpu-nodes-per-dc 4
#

import argparse
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd

import Rate_Flow_Sim as rfs


def generate_synthetic_povray_workload(
    num_epochs: int,
    cpu_nodes_per_dc: int,
    dc_id: int = 0,
    workload_type: str = "povray_r",
) -> pd.DataFrame:
    """
    Generate a synthetic workload with:
      - one <workload_type> instance per CPU node
      - in a single datacenter (dc_id)
      - across num_epochs epochs

    Columns:
      epoch, request_id, source_dc, model, tokens, arrival_ms
    """
    rows: List[Dict[str, Any]] = []
    req_id = 0
    for epoch in range(num_epochs):
        for node_idx in range(cpu_nodes_per_dc):
            rows.append(
                {
                    "epoch": epoch,
                    "request_id": req_id,
                    "source_dc": dc_id,
                    "model": workload_type,
                    "tokens": 1,
                    "arrival_ms": 0,
                }
            )
            req_id += 1

    return pd.DataFrame(rows)


def build_local_schedule(epoch_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Build a simple local-only schedule plan:
      - Each row is processed in its source_dc.
    The plan uses the 'map' format expected by Geo_Network.apply_schedule_plan:
      schedule_plan = {"map": {row_index: target_dc_id}}.
    """
    # Normalize column name for source dc
    if "source_dc" not in epoch_df.columns:
        if "source_dc_id" in epoch_df.columns:
            epoch_df = epoch_df.rename(columns={"source_dc_id": "source_dc"})
        elif "src_dc" in epoch_df.columns:
            epoch_df = epoch_df.rename(columns={"src_dc": "source_dc"})
        else:
            raise ValueError(
                "build_local_schedule: workload must contain 'source_dc', "
                "'source_dc_id', or 'src_dc' column"
            )

    idx_to_dc = {
        int(idx): int(src)
        for idx, src in zip(epoch_df.index, epoch_df["source_dc"])
    }
    return {"map": idx_to_dc}


def prepare_workload(
    df: pd.DataFrame,
    num_dcs: int,
    default_model: str = "povray_r",
    default_tokens: int = 1,
) -> pd.DataFrame:
    """
    Normalize workload columns so they match Rate_Flow_Sim expectations:

      Required final columns:
        - epoch      : int
        - source_dc  : int
        - model      : str
        - tokens     : float
        - arrival_ms : int

      This helper is tolerant of several common variants:
        - epoch / (missing -> 0)
        - source_dc_id / src_dc / (missing -> 0)
        - model_type / model / (missing -> default_model)
        - num_tokens / tokens / (missing -> default_tokens)
        - arrival_ms / arrival / (missing -> 0)
    """
    # Epoch index
    if "epoch" not in df.columns:
        df["epoch"] = 0
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce").fillna(0).astype(int)

    # Source DC
    if "source_dc" not in df.columns:
        if "source_dc_id" in df.columns:
            df = df.rename(columns={"source_dc_id": "source_dc"})
        elif "src_dc" in df.columns:
            df = df.rename(columns={"src_dc": "source_dc"})
        else:
            df["source_dc"] = 0  # default all to DC 0 if nothing provided

    df["source_dc"] = pd.to_numeric(df["source_dc"], errors="coerce").fillna(0).astype(int)
    df["source_dc"] = df["source_dc"].clip(lower=0, upper=max(0, num_dcs - 1))

    # Model / workload type
    if "model" not in df.columns:
        if "model_type" in df.columns:
            df = df.rename(columns={"model_type": "model"})
        else:
            df["model"] = default_model
    df["model"] = df["model"].astype(str)

    # Tokens (work unit)
    if "tokens" not in df.columns:
        if "num_tokens" in df.columns:
            df = df.rename(columns={"num_tokens": "tokens"})
        else:
            df["tokens"] = float(default_tokens)
    df["tokens"] = pd.to_numeric(df["tokens"], errors="coerce").fillna(default_tokens).astype(float)

    # Arrival times in ms (rate-based; all at start is fine)
    if "arrival_ms" not in df.columns:
        if "arrival" in df.columns:
            df["arrival_ms"] = pd.to_numeric(df["arrival"], errors="coerce").fillna(0).astype(int)
        else:
            df["arrival_ms"] = 0

    return df


def sweep_temperatures(
    workload_df: pd.DataFrame,
    temps_c: List[float],
    epoch_idx: int,
    epoch_length: int,
    num_dcs: int,
    cpu_type_id: int,
    cpu_nodes_per_dc: int,
    spec_dir: str,
    workload_type: str,
    debug: bool = False,
) -> pd.DataFrame:
    """
    For each temperature setpoint, configure an all-CPU node allocation via
    Rate_Flow_Sim.MANUAL_NODE_TYPE_COUNTS, build an LLM_Simulator, override
    each datacenter's temp_c_setpoint, and run a single epoch.

    Returns a DataFrame with one row per temperature.
    """
    results: List[Dict[str, Any]] = []

    # Filter workload to the chosen epoch
    epoch_df = workload_df[workload_df["epoch"] == epoch_idx].copy()
    if epoch_df.empty:
        raise ValueError(f"No workload rows found for epoch {epoch_idx}")

    # Build a local-only schedule (src_dc -> same dc)
    schedule_plan = build_local_schedule(epoch_df)
    power_plan: Dict[str, Any] = {}  # no explicit power control

    # Configure manual node allocation for all DCs
    counts_per_dc = {
        dc_id: {cpu_type_id: cpu_nodes_per_dc}
        for dc_id in range(num_dcs)
    }

    print(counts_per_dc)

    # Make sure MANUAL_NODE_TYPE_COUNTS exists
    if not hasattr(rfs, "MANUAL_NODE_TYPE_COUNTS"):
        raise RuntimeError(
            "Rate_Flow_Sim.MANUAL_NODE_TYPE_COUNTS is not defined.\n"
            "You must add the global override to Rate_Flow_Sim.py and wire it into "
            "build_world_from_csvs_exact(...) before using simulator_Temperature.py."
        )

    # Do the sweep
    for temp_c in temps_c:
        # Set manual node counts (CPU-only)
        rfs.MANUAL_NODE_TYPE_COUNTS = counts_per_dc  # type: ignore[attr-defined]

        # Build simulator
        sim = rfs.LLM_Simulator(
            spec_dir=spec_dir,
            epoch_length=epoch_length,
            debug=debug,
        )

        for dc in sim.datacenters.values():
            for u in dc.units:
                # ProcNode has accel_type field
                if getattr(u, "accel_type", "").upper() == "CPU":
                    u.workload_class = workload_type


        # Override datacenter temperature setpoint for this sweep step
        for dc in sim.datacenters.values():
            dc.temp_c_setpoint = float(temp_c)

        # Run a single epoch
        metrics, details, dc_usage = sim.run_epoch(
            epoch_idx=epoch_idx,
            workload_df=epoch_df,
            schedule_plan=schedule_plan,
            power_plan=power_plan,
        )

        # Collect key metrics (will reflect your new IT power profiles)
        results.append(
            {
                "Temp_C": float(temp_c),
                "Avg_TTFT_s": float(metrics.get("avg_ttft", 0.0)),
                "Carbon_kg": float(metrics.get("carbon_emissions", 0.0)),
                "Water_L": float(metrics.get("water_usage", 0.0)),
                "Energy_Cost_USD": float(metrics.get("energy_cost", 0.0)),
                "Total_Energy_kWh": float(metrics.get("total_energy", 0.0)),
            }
        )

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description="Standalone CPU + temperature sweep driver for Rate_Flow_Sim."
    )
    parser.add_argument(
        "--workload-csv",
        type=str,
        default="povray_synthetic_workload.csv",
        help="Path to workload CSV. With --synthetic-workload and "
             "--save-synthetic-workload, this file will be created.",
    )
    parser.add_argument(
        "--spec-dir",
        type=str,
        default="sim_specs",
        help="Directory containing Datacenter_specs.csv, Node_Specs.csv, etc.",
    )
    parser.add_argument(
        "--epoch-index",
        type=int,
        default=0,
        help="Which epoch index from the workload to simulate.",
    )
    parser.add_argument(
        "--epoch-length",
        type=int,
        default=3600,
        help="Epoch length in seconds (should match Workload_Granularity.csv).",
    )
    parser.add_argument(
        "--num-dcs",
        type=int,
        default=1,
        help="Number of datacenters (IDs assumed 0..num_dcs-1).",
    )
    parser.add_argument(
        "--cpu-type-id",
        type=int,
        default=6,
        help="Node type_id in Node_Specs.csv that corresponds to a CPU node.",
    )
    parser.add_argument(
        "--cpu-nodes-per-dc",
        type=int,
        default=1000,
        help="Number of CPU nodes to allocate per datacenter.",
    )
    parser.add_argument(
        "--temp-min",
        type=float,
        default=20.0,
        help="Minimum temperature setpoint (°C) for sweep.",
    )
    parser.add_argument(
        "--temp-max",
        type=float,
        default=40.0,
        help="Maximum temperature setpoint (°C) for sweep.",
    )
    parser.add_argument(
        "--temp-step",
        type=float,
        default=2.0,
        help="Temperature step (°C) for sweep.",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default="Temperature_Sweep_Results.csv",
        help="Output CSV file for sweep results.",
    )
    parser.add_argument(
        "--synthetic-workload",
        action="store_true",
        help="Generate a synthetic workload: 1 povray_r request per CPU node "
             "in one datacenter, across multiple epochs.",
    )
    parser.add_argument(
        "--synthetic-epochs",
        type=int,
        default=1,
        help="Number of epochs for the synthetic workload.",
    )
    parser.add_argument(
        "--synthetic-dc-id",
        type=int,
        default=0,
        help="Datacenter ID to use for the synthetic workload.",
    )
    parser.add_argument(
        "--save-synthetic-workload",
        action="store_true",
        help="If set with --synthetic-workload, save the generated workload "
             "to --workload-csv.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose simulator logging.",
    )
    parser.add_argument(
        "--workload-type",
        type=str,
        default="povray_r",
        choices=[
            "bwaves_r",
            "namd_r",
            "povray_r",
            "cactusBSSN_r",
            "parest_r",
            "fotonik3d_r",
        ],
        help="SPEC workload label to use for synthetic generation and as the default model.",
    )

    args = parser.parse_args()

    # Prepare workload: either synthetic or from CSV
    if args.synthetic_workload:
        df = generate_synthetic_povray_workload(
            num_epochs=args.synthetic_epochs,
            cpu_nodes_per_dc=args.cpu_nodes_per_dc,
            dc_id=0,
            workload_type=args.workload_type,
        )
        if args.save_synthetic_workload:
            df.to_csv(args.workload_csv, index=False)
    else:
        df = pd.read_csv(args.workload_csv)

    # Normalize columns for the simulator
    df = prepare_workload(
        df,
        num_dcs=args.num_dcs,
        default_model="povray_r",
        default_tokens=1,
    )

    # Build temperature grid
    if args.temp_step <= 0:
        raise ValueError("--temp-step must be > 0")
    temps = list(np.arange(args.temp_min, args.temp_max + 1e-9, args.temp_step))

    # Run sweep
    results_df = sweep_temperatures(
        workload_df=df,
        temps_c=temps,
        epoch_idx=args.epoch_index,
        epoch_length=args.epoch_length,
        num_dcs=args.num_dcs,
        cpu_type_id=args.cpu_type_id,
        cpu_nodes_per_dc=args.cpu_nodes_per_dc,
        spec_dir=args.spec_dir,
        workload_type=args.workload_type,
        debug=args.debug,
    )

    # Save & print
    results_df.to_csv(args.out_csv, index=False)
    print(f"[DONE] Wrote temperature sweep results to {args.out_csv}\n")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()


