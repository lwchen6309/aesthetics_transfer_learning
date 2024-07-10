import os
import pickle
import numpy as np
from sklearn.manifold import MDS
import argparse


def load_emd_scores(directory, trait_value_pairs):
    emd_scores = {}
    pair_to_index = {pair: index for index, pair in enumerate(trait_value_pairs)}
    files = os.listdir(directory)
    for file in files:
        if file.endswith(".pkl"):
            parts = file.replace("score_distributions_", "").replace(".pkl", "").split("_")
            trait1, value1, trait2, value2 = parts[0], parts[1], parts[2], parts[3]
            with open(os.path.join(directory, file), 'rb') as f:
                score_dict = pickle.load(f)
            mean_emd = np.mean([np.mean(emd) for emd in score_dict.values()])
            emd_scores[(trait1, value1, trait2, value2)] = mean_emd
    return emd_scores

def create_distance_matrix(trait_value_pairs, emd_scores):
    n = len(trait_value_pairs)
    distance_matrix = np.zeros((n, n))
    pair_to_index = {pair: index for index, pair in enumerate(trait_value_pairs)}

    for (trait1, value1, trait2, value2), emd in emd_scores.items():
        index1 = pair_to_index[(trait1, value1)]
        index2 = pair_to_index[(trait2, value2)]
        distance_matrix[index1, index2] = emd
        distance_matrix[index2, index1] = emd  # Ensure the matrix is symmetric

    return distance_matrix

def compute_mds(distance_matrix):
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
    embedding = mds.fit_transform(distance_matrix)
    return embedding


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute MDS from EMD scores')
    parser.add_argument('--output_dir', type=str, default='score_dict', help='Directory where the EMD score files are stored')
    parser.add_argument('--pairs', type=str, nargs='+', help='List of trait:value pairs')

    args = parser.parse_args()

    # Parse the pairs into a list of tuples
    trait_value_pairs = [tuple(pair.split(":")) for pair in args.pairs]

    # Load EMD scores
    emd_scores = load_emd_scores(args.output_dir, trait_value_pairs)

    # Create distance matrix
    distance_matrix = create_distance_matrix(trait_value_pairs, emd_scores)

    # Compute MDS
    embedding = compute_mds(distance_matrix)

    # Print the resulting embedding
    print("MDS Embedding:")
    for pair, coord in zip(trait_value_pairs, embedding):
        print(f"{pair}: {coord}")
