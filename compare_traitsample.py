import numpy as np
import os
from tqdm import tqdm
import pickle
from torch.utils.data import DataLoader
from PARA_histogram_dataloader import load_data, load_data_testpair, load_testdata
from utils.losses import EarthMoverDistance
earth_mover_distance = EarthMoverDistance()


def compute_mean_emd(giaa_dataset1, giaa_dataset2, args):
    # Step 1: Extract common unique images
    unique_images_1 = set(giaa_dataset1.unique_images)
    unique_images_2 = set(giaa_dataset2.unique_images)
    common_images = list(unique_images_1.intersection(unique_images_2))
    
    img_2index1 = {img: index for index, img in enumerate(giaa_dataset1.unique_images)}
    img_2index2 = {img: index for index, img in enumerate(giaa_dataset2.unique_images)}
    
    if not common_images:
        raise ValueError("No common images found between the two datasets")

    # Dictionary to store score distributions and common images
    score_dict = {}

    # Step 2: Compute EMD for each common image and save the distributions
    emd_scores = []
    for image in tqdm(common_images):
        # Get aesthetic score distributions
        score_dist1 = giaa_dataset1[img_2index1[image]]['aestheticScore']
        score_dist2 = giaa_dataset2[img_2index2[image]]['aestheticScore']
        
        # Compute EMD
        emd = earth_mover_distance(score_dist1, score_dist2)
        emd_scores.append(emd)
        
        # Save the distributions
        score_dict[image] = [score_dist1.cpu().numpy(), score_dist2.cpu().numpy()]
    
    # Save the score distributions to a file using pickle
    if args.trait1 and args.trait2 is not None:
        filename = f"score_distributions_{args.trait1}_{args.value1}_{args.trait2}_{args.value2}.pkl"
    else:
        filename = f"score_distributions_{args.trait}_{args.value}.pkl"
    with open(os.path.join(args.output_dir, filename), 'wb') as f:
        pickle.dump(score_dict, f)
    # Return the mean EMD
    mean_emd = np.mean(emd_scores)
    return mean_emd


if __name__ == '__main__':
    from utils.argflags import parse_arguments
    parser = parse_arguments(False)
    parser.add_argument('--output_dir', type=str, default='score_dict')
    parser.add_argument('--trait1', type=str, default=None)
    parser.add_argument('--value1', type=str, default=None)
    parser.add_argument('--trait2', type=str, default=None)
    parser.add_argument('--value2', type=str, default=None)
    args = parser.parse_args()
    
    if args.trait1 and args.trait2 is not None:
        args.trait, args.value = args.trait1, args.value1
        test_giaa_dataset, _ = load_testdata(args)
        args.trait, args.value = args.trait2, args.value2
        testc_giaa_dataset, _ = load_testdata(args)
    else:
        train_dataset, val_giaa_dataset, val_piaa_imgsort_dataset, test_giaa_dataset, test_piaa_imgsort_dataset, testc_giaa_dataset = load_data_testpair(args)
    mean_EMD = compute_mean_emd(test_giaa_dataset, testc_giaa_dataset, args)
    
    print(f'{args.trait} == {args.value}', mean_EMD)
    