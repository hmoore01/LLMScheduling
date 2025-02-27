import pandas as pd
import math

# Configuration: Set epoch length (15 minutes = 900 seconds)
EPOCH_LENGTH = 900  # in seconds

# Load the BurstGPT trace dataset
trace = pd.read_csv('BurstGPT_without_fails_2.csv')

# Map BurstGPT models to Llama models (GPT-4 -> Llama_70B, GPT-3.5 -> Llama_8B)
trace['Mapped Model'] = trace['Model'].apply(lambda x: 'Llama_70B' if 'GPT-4' in x else 'Llama_8B')

# Calculate which epoch each request belongs to
trace['Epoch'] = trace['Timestamp'].apply(lambda x: math.floor(x / EPOCH_LENGTH))

# Normalize epoch numbers to start at 0
min_epoch = trace['Epoch'].min()
trace['Normalized Epoch'] = trace['Epoch'] - min_epoch

# Get the full range of epochs
max_epoch = trace['Normalized Epoch'].max()
all_epochs = range(0, max_epoch + 1)  # List of all possible epochs

# Group the trace data by normalized epoch
grouped_data = trace.groupby('Normalized Epoch')

# Initialize the epoch summary list
epochs_summary = []

# Iterate over the full range of epochs to ensure no missing epochs
previous_entry = None  # Track the last valid entry for forward filling
for epoch in all_epochs:
    if epoch in grouped_data.groups:
        # Get the group for the current epoch
        group = grouped_data.get_group(epoch)

        # Calculate statistics for the current epoch
        llama_8b_group = group[group['Mapped Model'] == 'Llama_8B']
        llama_70b_group = group[group['Mapped Model'] == 'Llama_70B']

        epoch_entry = {
            'Epoch': epoch,
            'Total Requests': len(group),
            'Llama_8B Requests': len(llama_8b_group),
            'Llama_70B Requests': len(llama_70b_group),
            'Total tokens': group['Total tokens'].sum(),
            'Llama_8B tokens': llama_8b_group['Total tokens'].sum(),
            'Llama_70B tokens': llama_70b_group['Total tokens'].sum()
        }

        # Store the current entry as the previous entry for future use
        previous_entry = epoch_entry
    else:
        # If the epoch is missing, use the previous entry or fill with zeros
        if previous_entry is not None:
            epoch_entry = previous_entry.copy()
            epoch_entry['Epoch'] = epoch  # Update the epoch number
        else:
            # If there is no previous entry, initialize with zeros
            epoch_entry = {
                'Epoch': epoch,
                'Total Requests': 0,
                'Llama_8B Requests': 0,
                'Llama_70B Requests': 0,
                'Total tokens': 0,
                'Llama_8B Tokens': 0,
                'Llama_70B Tokens': 0
            }

    # Add the epoch entry to the summary list
    epochs_summary.append(epoch_entry)

# Convert the summary list into a DataFrame and save it to a file
epoch_summary_df = pd.DataFrame(epochs_summary)

missing_epochs = []
for e in range(0, max_epoch + 1):
    if e not in epoch_summary_df['Epoch'].values:
        missing_epochs.append(e)

if missing_epochs:
    raise ValueError(f"These epochs are missing in epoch_summary_df: {missing_epochs}")

# 2) Check that no epoch has zero requests
bad_epochs = epoch_summary_df[epoch_summary_df['Total Requests'] == 0]['Epoch'].tolist()
if bad_epochs:
    raise ValueError(f"These epochs have zero requests (which shouldn't happen): {bad_epochs}")

print("All epochs exist and have non-zero requests.")

filled_output_path = "BurstGPT_processed.csv"
epoch_summary_df.to_csv(filled_output_path, index=False)

print(f"Filled epoch summary saved to: {filled_output_path}")

