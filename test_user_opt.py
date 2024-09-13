import pandas as pd
import numpy as np

# Load your data
mir_file = './MIR_test_sroccs.txt'
ici_file = './ICI_test_sroccs.txt'
mir_data = pd.read_csv(mir_file, delim_whitespace=True)
ici_data = pd.read_csv(ici_file, delim_whitespace=True)

# Ensure both datasets have the same users
users = list(set(mir_data['User_ID']).intersection(set(ici_data['User_ID'])))

# Create dictionaries for quick lookup of SROCC values
mir_srocc = mir_data.set_index('User_ID')['SROCC'].to_dict()
ici_srocc = ici_data.set_index('User_ID')['SROCC'].to_dict()

# Target SROCC values
target_mir_srocc = 0.716
target_ici_srocc = 0.739

# Initialize empty set to store the best sample of users
best_sample = set()

# Start with zero average SROCCs
current_avg_mir = 0
current_avg_ici = 0

# Function to compute the average SROCC for a given set of users
def compute_avg_srocc(selected_users):
    if len(selected_users) == 0:
        return 0, 0
    avg_mir_srocc = np.mean([mir_srocc[user] for user in selected_users])
    avg_ici_srocc = np.mean([ici_srocc[user] for user in selected_users])
    return avg_mir_srocc, avg_ici_srocc

# Greedy algorithm to find a subset of 40 users that gets closest to the target
while len(best_sample) < 40:
    best_user = None
    best_diff = float('inf')

    # Try adding each remaining user and calculate the resulting difference
    for user in users:
        if user in best_sample:
            continue  # Skip if the user is already in the sample

        new_sample = best_sample | {user}
        avg_mir_srocc, avg_ici_srocc = compute_avg_srocc(new_sample)

        # Calculate the absolute differences from the target
        mir_diff = abs(avg_mir_srocc - target_mir_srocc)
        ici_diff = abs(avg_ici_srocc - target_ici_srocc)
        total_diff = mir_diff + ici_diff

        # Keep track of the user that minimizes the difference
        if total_diff < best_diff:
            best_diff = total_diff
            best_user = user

    # Add the best user to the sample
    if best_user:
        best_sample.add(best_user)
        current_avg_mir, current_avg_ici = compute_avg_srocc(best_sample)

# Final selected users and their average SROCC values
best_sample = list(best_sample)
avg_mir_srocc, avg_ici_srocc = compute_avg_srocc(best_sample)

# Output the results
print("Best sample of users:", best_sample)
print(f"Mean SROCC for MIR: {avg_mir_srocc}")
print(f"Mean SROCC for ICI: {avg_ici_srocc}")
