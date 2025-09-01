import numpy as np
import os
from tqdm import tqdm
import pickle
from LAPIS_histogram_dataloader import load_testdata
from compute_gini_index import compute_mean_gini


if __name__ == '__main__':
    from utils.argflags import parse_arguments
    parser = parse_arguments(False)
    parser.add_argument('--output_dir', type=str, default='score_dict')
    parser.add_argument('--values', type=str, nargs='+', required=True, help="List of demographic values for the trait (e.g., 'male', 'female', '18-21', etc.)")
    args = parser.parse_args()

    # Load datasets for each value in the specified demographic trait
    datasets = []
    print(args.values)
    for value in args.values:
        args.value = value
        dataset, _ = load_testdata(args)
        datasets.append(dataset)

    # Compute and print the mean Gini index for the current trait
    mean_gini = compute_mean_gini(datasets, args)
    print(f'{args.trait} Gini Index:', mean_gini)

