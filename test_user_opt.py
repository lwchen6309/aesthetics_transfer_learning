import pandas as pd
import numpy as np
from para_rn_param_pca import load_testdata, presentative_users


if __name__ == '__main__':
    # List of files to load and their target SROCC values
    files_info = [
        {'file': './PIAA_MIR_100_test_sroccs.txt', 'target_srocc': 0.716},
        {'file': './PIAA_ICI_100_test_sroccs.txt', 'target_srocc': 0.739},
        {'file': './PIAA_MIR_10_test_sroccs.txt', 'target_srocc': 0.702},
        {'file': './PIAA_ICI_10_test_sroccs.txt', 'target_srocc': 0.728},    
        # Add more files and their target SROCCs here as needed
    ]

    # Dictionary to hold data and target values
    data_dict = {}
    target_sroccs = {}

    # Load data, populate dictionaries, and calculate statistics for each file
    for info in files_info:
        file_name = info['file']
        target_srocc = info['target_srocc']
        data = pd.read_csv(file_name, delim_whitespace=True)
        
        # Store data and target SROCC values
        srocc_values = data['SROCC']
        data_dict[file_name] = data.set_index('User_ID')['SROCC'].to_dict()
        target_sroccs[file_name] = target_srocc
        
        # Calculate mean and standard deviation
        mean_srocc = srocc_values.mean()
        std_srocc = srocc_values.std()
        
        # Print out the results
        print(f"File: {file_name}")
        print(f"Mean SROCC: {mean_srocc:.4f}")
        print(f"Standard Deviation of SROCC: {std_srocc:.4f}")
        print(f"Target SROCC: {target_srocc}")
        print('-' * 40)
        
    # Find common users across all datasets
    users = set.intersection(*[set(data.keys()) for data in data_dict.values()])

    # Initialize empty set to store the best sample of users
    best_sample = set()

    # Function to compute the average SROCC for a given set of users across all datasets
    def compute_avg_srocc(selected_users):
        if len(selected_users) == 0:
            return {file: 0 for file in data_dict}
        
        avg_sroccs = {}
        for file_name, srocc_dict in data_dict.items():
            avg_sroccs[file_name] = np.mean([srocc_dict[user] for user in selected_users])
        return avg_sroccs

    final_merged_df = load_testdata()
    top_n_users = presentative_users(final_merged_df, 50)
    top_n_users = set(top_n_users['userId']).intersection(users)
    avg_sroccs = compute_avg_srocc(top_n_users)
    print(avg_sroccs)
    # raise Exception

    # Greedy algorithm to find a subset of 40 users that gets closest to the targets
    while len(best_sample) < 40:
        best_user = None
        best_diff = float('inf')

        # Try adding each remaining user and calculate the resulting difference
        for user in users:
            if user in best_sample:
                continue  # Skip if the user is already in the sample

            new_sample = best_sample | {user}
            avg_sroccs = compute_avg_srocc(new_sample)

            # Calculate the total absolute difference from the targets
            total_diff = sum(abs(avg_sroccs[file_name] - target_sroccs[file_name]) for file_name in data_dict)

            # Keep track of the user that minimizes the difference
            if total_diff < best_diff:
                best_diff = total_diff
                best_user = user

        # Add the best user to the sample
        if best_user:
            best_sample.add(best_user)

    # Final selected users and their average SROCC values
    best_sample = list(best_sample)
    avg_sroccs = compute_avg_srocc(best_sample)

    # Output the results
    print("Best sample of users:", best_sample)
    for file_name, avg_srocc in avg_sroccs.items():
        print(f"Mean SROCC for {file_name}: {avg_srocc}")
