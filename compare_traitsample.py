import numpy as np
import os
from tqdm import tqdm
import pickle
from torch.utils.data import DataLoader
from PARA_histogram_dataloader import load_data, load_data_testpair
import argparse
from utils.losses import EarthMoverDistance
earth_mover_distance = EarthMoverDistance()


def compute_mean_emd(giaa_dataset1, giaa_dataset2, args):
    # Step 1: Extract common unique images
    unique_images_1 = set(giaa_dataset1.unique_images)
    unique_images_2 = set(giaa_dataset2.unique_images)
    common_images = list(unique_images_1.intersection(unique_images_2))
    
    img_2index1 = {img: index for index, img in enumerate(unique_images_1)}
    img_2index2 = {img: index for index, img in enumerate(unique_images_2)}
    
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
    filename = f"score_distributions_{args.trait}_{args.value}.pkl"
    with open(os.path.join(args.output_dir, filename), 'wb') as f:
        pickle.dump(score_dict, f)
    # Return the mean EMD
    mean_emd = np.mean(emd_scores)
    return mean_emd


# Example usage
# Assuming giaa_dataset1 and giaa_dataset2 are two instances of PARA_GIAA_HistogramDataset
# mean_emd = compute_mean_emd(giaa_dataset1, giaa_dataset2)
# print("Mean EMD between the two dataset instances:", mean_emd)

if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description='Training and Testing the Combined Model for data spliting')
    parser.add_argument('--trainset', type=str, default='GIAA', choices=["GIAA", "sGIAA", "PIAA"])
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--is_eval', action='store_true', help='Enable evaluation mode')
    parser.add_argument('--eval_on_piaa', action='store_true', help='Evaluation metric on PIAA')
    parser.add_argument('--no_log', action='store_false', dest='is_log', help='Disable logging')
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--max_patience_epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--lr_schedule_epochs', type=int, default=5)
    parser.add_argument('--lr_decay_factor', type=float, default=0.5)
    parser.add_argument('--trait', type=str, default=None)
    parser.add_argument('--value', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='score_dict')

    args = parser.parse_args()

    batch_size = args.batch_size
    
    random_seed = 42
    n_workers = 8
    
    train_dataset, val_giaa_dataset, val_piaa_imgsort_dataset, test_giaa_dataset, test_piaa_imgsort_dataset, testc_giaa_dataset = load_data_testpair(args)
    mean_EMD = compute_mean_emd(test_giaa_dataset, testc_giaa_dataset, args)
    print(f'{args.trait} == {args.value}', mean_EMD)
