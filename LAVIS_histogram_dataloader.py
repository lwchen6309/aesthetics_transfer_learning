import os
import torch
from torchvision import transforms
import torch.nn.functional as F
from LAVIS_PIAA_dataloader import LAVIS_PIAADataset, create_image_split_dataset
from torch.utils.data import DataLoader, Dataset
import random
import pickle
from tqdm import tqdm
import copy
from time import time



class LAVIS_GIAA_HistogramDataset(LAVIS_PIAADataset):
    def __init__(self, root_dir, transform=None, data=None, map_file=None, precompute_file=None):
        super().__init__(root_dir, transform)
        if data is not None:
            self.data = data
        
        if map_file and os.path.exists(map_file):
            print('Loading image to indices map from file...')
            self.image_to_indices_map = self._load_map(map_file)
            self.unique_images = [img for img in self.image_to_indices_map.keys()]
        else:
            self.image_to_indices_map = dict()
            for image in tqdm(self.data['imageName'].unique(), desc='Processing images'):
                indices_for_image = [i for i, img in enumerate(self.data['imageName']) if img == image]
                if any(not idx < len(self.data) for idx in indices_for_image):
                    print(indices_for_image)
                    raise Exception('Index out of bounds for the data.')
                if len(indices_for_image) > 0:  # Only add if there are indices for the image
                    self.image_to_indices_map[image] = indices_for_image
            
            self.unique_images = [img for img in self.image_to_indices_map.keys()]  # Filtered unique_images

            if map_file:
                print(f"Saving image to indices map to {map_file}")
                self._save_map(map_file)
        
        # If precompute_file is given and exists, load it, otherwise recompute the data
        if precompute_file and os.path.exists(precompute_file):
            print(f'Loading precomputed data from {precompute_file}...')
            self.load(precompute_file)
        else:
            self.precompute_data()
            if precompute_file:
                print(f"Saving precomputed data to {precompute_file}")
                self.save(precompute_file)

    def update_image_to_indices_map(self):
        self.unique_images = self.data['imageName'].unique()
        self.image_to_indices_map = {img: self.image_to_indices_map[img] for img in self.unique_images if img in self.image_to_indices_map}

    def precompute_data(self):
        self.precomputed_data = []
        for idx in tqdm(range(len(self)), desc='Precompute images'):
            self.precomputed_data.append(self._compute_item(idx))

    def _compute_item(self, idx):
        associated_indices = self.image_to_indices_map[self.unique_images[idx]]
        
        # Assuming maximum scores for response and VAIAK/2VAIAK ratings for proper one-hot encoding
        max_response_score = 10  # Adjust based on your data
        max_vaia_score = 7
        
        # Initialize accumulators for one-hot encoded vectors
        accumulated_response = torch.zeros(max_response_score, dtype=torch.float16)
        accumulated_vaia = torch.zeros(7 * max_vaia_score, dtype=torch.float16)  # For VAIAK1 to VAIAK7
        accumulated_2vaia = torch.zeros(4 * max_vaia_score, dtype=torch.float16)  # For 2VAIAK1 to 2VAIAK4
        
        accumulated_trait = {triat:torch.zeros(len(enc), dtype=torch.float16) for triat, enc in zip(self.encoded_trait_columns, self.trait_encoders)}
        for ai in associated_indices:
            sample = super().__getitem__(ai, use_image=False)

            # One-hot encode 'response' and accumulate
            round_score = min(int(sample['response'])//10, 9)
            response_one_hot = F.one_hot(torch.tensor(round_score), num_classes=max_response_score)
            accumulated_response += response_one_hot
            
            # Encode and accumulate VAIAK1 to VAIAK7
            for i in range(1, 8):  # VAIAK1 to VAIAK7
                vaia_score = sample[f'VAIAK{i}']
                offset = (i-1) * max_vaia_score
                vaia_one_hot = F.one_hot(torch.tensor(int(vaia_score)), num_classes=max_vaia_score)
                accumulated_vaia[offset:offset+max_vaia_score] += vaia_one_hot
                
            # Encode and accumulate 2VAIAK1 to 2VAIAK4
            for i in range(1, 5):  # 2VAIAK1 to 2VAIAK4
                vaia_score = sample[f'2VAIAK{i}']
                offset = (i-1) * max_vaia_score
                vaia_one_hot = F.one_hot(torch.tensor(int(vaia_score)), num_classes=max_vaia_score)
                accumulated_2vaia[offset:offset+max_vaia_score] += vaia_one_hot

            for trait in self.encoded_trait_columns:
                vaia_one_hot = F.one_hot(torch.tensor(int(sample[trait])), num_classes=len(accumulated_trait[trait]))
                accumulated_trait[trait] += vaia_one_hot

        # Prepare the final accumulated histogram for the image
        accumulated_histogram = {
            'aestheticScore': accumulated_response,
            'onehot_traits': accumulated_trait,
            'VAIAK': accumulated_vaia,
            '2VAIAK': accumulated_2vaia,
        }

        # Average out histograms over the number of samples
        total_samples = len(associated_indices)
        accumulated_histogram['aestheticScore'] /= total_samples
        accumulated_histogram['VAIAK'] /= total_samples
        accumulated_histogram['2VAIAK'] /= total_samples
        for trait in accumulated_histogram['onehot_traits'].keys():
            accumulated_histogram['onehot_traits'][trait] /= total_samples
            # print(trait, sum(accumulated_histogram['onehot_traits'][trait]))
        onehot_traits = list(accumulated_histogram['onehot_traits'].values())
        accumulated_histogram['art_type'] = onehot_traits[0]
        accumulated_histogram['onehot_traits'] = torch.cat(onehot_traits[1:])

        accumulated_histogram['n_samples'] = total_samples
        
        # print(accumulated_histogram['art_type'])
        # print(accumulated_histogram['onehot_traits'])
        # print('score', sum(accumulated_histogram['aestheticScore']))
        # for i in range(1, 8):  # VAIAK1 to VAIAK7
        #     offset = (i-1) * max_vaia_score
        #     print(sum(accumulated_histogram['VAIAK'][offset:offset+max_vaia_score]))
        # for i in range(1, 5):  # VAIAK1 to VAIAK7
        #     offset = (i-1) * max_vaia_score
        #     print(sum(accumulated_histogram['2VAIAK'][offset:offset+max_vaia_score]))
        
        return accumulated_histogram

    def __getitem__(self, idx):
        item_data = copy.deepcopy(self.precomputed_data[idx])
        img_sample = super().__getitem__(self.image_to_indices_map[self.unique_images[idx]][0], use_image=True)
        item_data['image'] = img_sample['image']
        return item_data

    def _save_map(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self.image_to_indices_map, f)

    def _load_map(self, file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def __len__(self):
        return len(self.image_to_indices_map)

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self.precomputed_data, f)

    def load(self, file_path):
        with open(file_path, 'rb') as f:
            self.precomputed_data = pickle.load(f)




if __name__ == '__main__':
    # Usage example:
    root_dir = '/home/lwchen/datasets/LAVIS'
    
    # Define transformations for training set and test set
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    # Create datasets with the appropriate transformations
    piaa_dataset = LAVIS_PIAADataset(root_dir, transform=train_transform)
    train_piaa_dataset, test_piaa_dataset = create_image_split_dataset(piaa_dataset)
    """Precompute"""
    pkl_dir = './LAVIS_dataset_pkl'
    train_dataset = LAVIS_GIAA_HistogramDataset(root_dir, transform=train_transform, data=train_piaa_dataset.data, map_file=os.path.join(pkl_dir,'trainset_image_dct.pkl'), precompute_file=os.path.join(pkl_dir,'trainset_GIAA_dct.pkl'))
    test_dataset = LAVIS_GIAA_HistogramDataset(root_dir, transform=train_transform, data=train_piaa_dataset.data, map_file=os.path.join(pkl_dir,'testset_image_dct.pkl'), precompute_file=os.path.join(pkl_dir,'testset_GIAA_dct.pkl'))