import numpy as np
import os
from tqdm import tqdm
import pickle
from PARA_histogram_dataloader import load_testdata

def gini_index_calculator(score_distributions):
    """
    Compute the Gini index for a list of score distributions across different demographic groups.

    Args:
        score_distributions (list of np.ndarray): List of arrays where each array represents 
                                                  the score distribution for a specific demographic group.
    
    Returns:
        float: The Gini index for the combined distributions.
    """

    gini_index = np.stack([(1 - np.sum(p ** 2)) for p in score_distributions]).mean()
    
    return gini_index


def compute_mean_gini(datasets, args):
    # Dictionary to store score distributions and common images
    score_dict = {}
    
    # Step 1: Extract unique images across all datasets (ensuring the same images are considered)
    common_images = set(datasets[0].unique_images)
    for dataset in datasets[1:]:
        common_images &= set(dataset.unique_images)

    common_images = list(common_images)
    if not common_images:
        raise ValueError("No common images found across the datasets")

    # Step 2: Create image index mappings for each dataset
    img_2index_list = []
    for dataset in datasets:
        img_2index = {img: index for index, img in enumerate(dataset.unique_images)}
        img_2index_list.append(img_2index)

    # Step 3: Compute Gini index for each common image and save the distributions
    gini_scores = []
    for image in tqdm(common_images, desc="Computing Gini Index"):
        score_distributions = []
        
        # Collect aesthetic score distributions for each demographic class
        for dataset, img_2index in zip(datasets, img_2index_list):
            score_dist = dataset[img_2index[image]]['aestheticScore']
            score_distributions.append(score_dist.cpu().numpy())

        # Compute Gini index for the combined score distributions
        gini = gini_index_calculator(score_distributions)
        gini_scores.append(gini)
        
        # Save the distributions for later use
        score_dict[image] = score_distributions

    # Save the score distributions to a file using pickle
    filename = f"gini_score_distributions_{args.trait}.pkl"
    with open(os.path.join(args.output_dir, filename), 'wb') as f:
        pickle.dump(score_dict, f)
    
    # Return the mean Gini index
    mean_gini = np.mean(gini_scores)
    return mean_gini


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

