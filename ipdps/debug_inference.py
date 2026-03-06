import pandas as pd
import math
from Rate_Flow_Sim import LLM_Simulator


def find_scaling_factors(trace_csv="simulator_ready_trace.csv", target_utilization=0.95):
    print("Initializing Simulator...")
    # Load the simulator. It will automatically load Node_Specs.csv, Datacenter_specs.csv, etc.
    sim = LLM_Simulator(debug=False)

    print(f"Loading Trace: {trace_csv}\n")
    df = pd.read_csv(trace_csv)

    # Map your CSV columns to what the Simulator explicitly expects
    if 'source_dc_id' in df.columns:
        df.rename(columns={'source_dc_id': 'source_dc'}, inplace=True)
    if 'model_type' in df.columns:
        df.rename(columns={'model_type': 'model'}, inplace=True)
    if 'num_tokens' in df.columns:
        df.rename(columns={'num_tokens': 'tokens'}, inplace=True)

    epochs = sorted(df['epoch'].unique())

    # 1. Calculate the maximum raw capacity per Datacenter
    dc_capacity_ms = {}
    epoch_length_ms = sim.epoch_length * 1000.0

    for dc_id, dc in sim.datacenters.items():
        # Treat all units as ON to find the theoretical max capacity
        total_nodes = len(dc.units)
        dc_capacity_ms[dc_id] = total_nodes * epoch_length_ms

    # 2. Dry run each epoch to measure baseline compute demand
    for ep in epochs:
        print(f"====================================")
        print(f"       ANALYZING EPOCH {ep}        ")
        print(f"====================================")

        ep_df = df[df['epoch'] == ep].copy()

        # Turn all servers ON and use default routing (requests stay in source_dc)
        power_plan = {"all": "ON"}
        schedule_plan = {}

        # Run Simulator to calculate the raw `exec_ms` for every request
        metrics, details, dc_usage = sim.run_epoch(
            epoch_idx=ep,
            workload_df=ep_df,
            schedule_plan=schedule_plan,
            power_plan=power_plan
        )

        # 3. Sum up the actual Execution MS per datacenter
        dc_exec_sums = {dc_id: 0.0 for dc_id in dc_capacity_ms.keys()}

        for req in details:
            if 'tag' in req and req['tag'] == 'epoch_finalize_idle':
                continue  # Skip idle accumulation records

            if 'dc_id' in req and 'exec_ms' in req:
                dc_id = int(req['dc_id'])
                if dc_id in dc_exec_sums:
                    dc_exec_sums[dc_id] += float(req['exec_ms'])

        # 4. Calculate the Multipliers needed to reach Target Utilization
        for dc_id, capacity in dc_capacity_ms.items():
            if capacity == 0:
                continue

            used_ms = dc_exec_sums[dc_id]
            current_util = used_ms / capacity

            if used_ms == 0:
                print(f"DC {dc_id}: 0.00% Utilized (No traffic routed here).")
                continue

            # The Math: Target Compute Time / Current Compute Time
            target_ms = capacity * target_utilization
            multiplier = target_ms / used_ms

            print(f"DC {dc_id} | Baseline Util: {current_util * 100:.2f}% | Target: {target_utilization * 100:.0f}%")

            if multiplier < 1.0:
                print(f"  -> OVERLOADED. Scale volume DOWN by a factor of {multiplier:.2f}x\n")
            else:
                print(f"  -> UNDERUTILIZED. Scale volume UP by a factor of {multiplier:.2f}x")
                print(f"     Option A: Multiply 'num_tokens' column by {multiplier:.2f}")
                print(f"     Option B: Duplicate the request rows {math.ceil(multiplier)} times\n")


if __name__ == "__main__":
    # You can change 0.95 to 1.00 if you want to push it to the absolute breaking point
    find_scaling_factors("simulator_ready_trace.csv", target_utilization=0.95)