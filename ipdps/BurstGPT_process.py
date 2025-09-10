import pandas as pd
import math

# Constants
EPOCH_LENGTH = 900
NUM_DATACENTERS = 12
INPUT_FILE = "BurstGPT_without_fails_2.csv"
OUTPUT_FILE = "simulator_ready_trace.csv"

def process_trace_for_simulator(trace):
    trace['model_type'] = trace['Model'].apply(
        lambda x: 'Llama70b' if 'GPT-4' in x else 'Llama7b'
    )

    trace['epoch'] = trace['Timestamp'].apply(lambda x: math.floor(x / EPOCH_LENGTH))
    min_epoch = trace['epoch'].min()
    trace['epoch'] -= min_epoch
    trace['time_index'] = trace['Timestamp'] % EPOCH_LENGTH

    trace = trace.sort_values(by="Timestamp").reset_index(drop=True)
    trace['source_dc_id'] = trace.index % NUM_DATACENTERS

    trace['batch_size'] = 1

    processed = trace[[
        'epoch', 'model_type', 'Total tokens', 'time_index', 'source_dc_id', 'batch_size'
    ]].rename(columns={
        'Total tokens': 'num_tokens'
    })

    return processed

def main():
    print(f"Loading trace from {INPUT_FILE}...")
    trace = pd.read_csv(INPUT_FILE)

    print("Processing trace...")
    simulator_ready = process_trace_for_simulator(trace)

    print(f"Saving to {OUTPUT_FILE}...")
    simulator_ready.to_csv(OUTPUT_FILE, index=False)
    print("Done.")

if __name__ == "__main__":
    main()




