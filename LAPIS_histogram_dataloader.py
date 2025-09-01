import os
import torch
from torchvision import transforms
import torch.nn.functional as F
from LAPIS_PIAA_dataloader import LAPIS_PIAADataset, create_image_split_dataset, create_user_split_dataset_kfold, datapath
from torch.utils.data import DataLoader
from utils.custom_transform import ResizeToNearestDivisible

import random
import pickle
from tqdm import tqdm
import copy
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from PIL import Image
import textwrap


def ensure_dir_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


class LAPIS_GIAA_HistogramDataset(LAPIS_PIAADataset):
    def __init__(self, root_dir, transform=None, data=None, map_file=None, precompute_file=None, disable_onehot=False):
        super().__init__(root_dir, transform)
        if data is not None:
            self.data = data
        self.disable_onehot = disable_onehot
        
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

    def traits_len(self):
        units_len = [len(enc) for enc in self.trait_encoders[1:]]  # Encoders for various traits
        units_len.extend([7] * 11)  # Extending for VAIAK1 to VAIAK7 and 2VAIAK1 to 2VAIAK4, 11 in total
        return units_len

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
        accumulated_response = torch.zeros(max_response_score, dtype=torch.float32)
        if self.disable_onehot:
            accumulated_vaia = torch.zeros(7, dtype=torch.float32)  # For VAIAK1 to VAIAK7
            accumulated_2vaia = torch.zeros(4, dtype=torch.float32)  # For 2VAIAK1 to 2VAIAK4
        else:
            accumulated_vaia = torch.zeros(7 * max_vaia_score, dtype=torch.float32)  # For VAIAK1 to VAIAK7
            accumulated_2vaia = torch.zeros(4 * max_vaia_score, dtype=torch.float32)  # For 2VAIAK1 to 2VAIAK4
        accumulated_trait = {triat:torch.zeros(len(enc), dtype=torch.float32) for triat, enc in zip(self.encoded_trait_columns, self.trait_encoders)}

        for ai in associated_indices:
            sample = super().__getitem__(ai, use_image=False)

            # One-hot encode 'response' and accumulate
            round_score = min(int(sample['response'])//10, 9)
            response_one_hot = F.one_hot(torch.tensor(round_score), num_classes=max_response_score)
            accumulated_response += response_one_hot
            if self.disable_onehot:
                accumulated_vaia = torch.stack([sample[f'VAIAK{i}'] for i in range(1, 8)], dim=0).view(-1)
                accumulated_2vaia = torch.stack([sample[f'2VAIAK{i}'] for i in range(1, 5)], dim=0).view(-1)
            else:
                accumulated_vaia += torch.stack([sample[f'VAIAK{i}_onehot'] for i in range(1, 8)], dim=0).view(-1)
                accumulated_2vaia += torch.stack([sample[f'2VAIAK{i}_onehot'] for i in range(1, 5)], dim=0).view(-1)
            for trait in self.encoded_trait_columns:
                accumulated_trait[trait] += sample[f'{trait}_onehot']

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
        # accumulated_histogram['art_type'] = onehot_traits[0]
        accumulated_histogram['onehot_traits'] = torch.cat(onehot_traits[2:])

        accumulated_histogram['n_samples'] = total_samples
        accumulated_histogram['imgName'] = sample['imgName']

        return accumulated_histogram

    def __getitem__(self, idx, print_detail=False):
        item_data = copy.deepcopy(self.precomputed_data[idx])
        img_sample = super().__getitem__(self.image_to_indices_map[self.unique_images[idx]][0], use_image=True)
        inherit_list = ['image', 'image_path', 'QIP', 'genre_onehot', 'style_onehot']
        for item in inherit_list:
            item_data[item] = img_sample[item]

        if print_detail:
            item_data['p_id'] = self.image_to_indices_map[self.unique_images[idx]]
            item_data['p_uid'] = []
            item_data['p_score'] = []
            for idx in item_data['p_id']:
                p_sample = super().__getitem__(idx, use_image=False)
                item_data['p_uid'].append(p_sample['participant_id'])
                item_data['p_score'].append(min(int(p_sample['response'])//10, 9))

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


class LAPIS_sGIAA_HistogramDataset(LAPIS_GIAA_HistogramDataset):
    def __init__(self, root_dir, transform=None, data=None, map_file=None, precompute_file=None,
                importance_sampling=False, num_samples=40, disable_onehot=False):
        # super(LAPIS_sGIAA_HistogramDataset, self).__init__()
        LAPIS_PIAADataset.__init__(self, root_dir, transform)
        self.disable_onehot = disable_onehot

        if data is not None:
            self.data = data
        self.importance_sampling = importance_sampling
        self.num_samples = num_samples

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
    
    def traits_len(self):
        units_len = [len(enc) for enc in self.trait_encoders[1:]]  # Encoders for various traits
        units_len.extend([7] * 11)  # Extending for VAIAK1 to VAIAK7 and 2VAIAK1 to 2VAIAK4, 11 in total
        return units_len

    def precompute_data(self):
        self.precomputed_data = []
        for idx in tqdm(range(len(self))):
            self.precomputed_data.append(self._compute_sgiaa_item(idx, num_samples=self.num_samples, importance_sampling=self.importance_sampling))

    def compute_score_hist(self, associated_indices):
        bin_indecies = []
        for random_idx in associated_indices:
            sample = LAPIS_PIAADataset.__getitem__(self, random_idx, use_image=False)
            round_score = min(int(sample['response'])//10, 9)
            bin_indecies.append(round_score)
        scores = np.array(bin_indecies)
        
        unique_scores = np.unique(scores)
        hist_values = np.zeros_like(unique_scores, dtype=np.float32)
        for i, s in enumerate(unique_scores):
            hist_values[i] = sum(scores == s)
        return scores, unique_scores, hist_values

    def _compute_importance_prob(self, associated_indices):
        scores, unique_scores, hist_values = self.compute_score_hist(associated_indices)
        inv_prob = 1 / hist_values
        sample_prob = np.zeros_like(scores, dtype=np.float32)
        for s, p in zip(unique_scores, inv_prob):
            sample_prob[scores == s] = p
        sample_prob = sample_prob / sample_prob.sum()
        return sample_prob

    def _compute_sgiaa_item(self, idx, num_samples=40, importance_sampling=False):
        accumulated_histograms = []
        associated_indices = self.image_to_indices_map[self.unique_images[idx]]
        if importance_sampling:
            sample_prob = self._compute_importance_prob(associated_indices)
        else:
            sample_prob = None

        for i in range(num_samples):
            if i > 0:
                accumulated_histograms.append(self._compute_item(idx, is_giaa=False, sample_prob=sample_prob))
            else:
                accumulated_histograms.append(self._compute_item(idx, is_giaa=True, sample_prob=sample_prob))
        return accumulated_histograms

    def _compute_item(self, idx, is_giaa=True, sample_prob=None):
        associated_indices = self.image_to_indices_map[self.unique_images[idx]]
        upper_bound = len(associated_indices) if is_giaa else len(associated_indices)-1
        n_sample = random.randint(2, upper_bound)
        if sample_prob is not None:
            associated_indices = random.choices(associated_indices, weights=sample_prob, k=n_sample)
        else:
            associated_indices = random.sample(associated_indices, n_sample)
            # associated_indices = random.choices(associated_indices, k=n_sample)
        
        # Assuming maximum scores for response and VAIAK/2VAIAK ratings for proper one-hot encoding
        max_response_score = 10  # Adjust based on your data
        max_vaia_score = 7
        
        # Initialize accumulators for one-hot encoded vectors
        accumulated_response = torch.zeros(max_response_score, dtype=torch.float32)
        if self.disable_onehot:
            accumulated_vaia = torch.zeros(7, dtype=torch.float32)  # For VAIAK1 to VAIAK7
            accumulated_2vaia = torch.zeros(4, dtype=torch.float32)  # For 2VAIAK1 to 2VAIAK4
        else:
            accumulated_vaia = torch.zeros(7 * max_vaia_score, dtype=torch.float32)  # For VAIAK1 to VAIAK7
            accumulated_2vaia = torch.zeros(4 * max_vaia_score, dtype=torch.float32)  # For 2VAIAK1 to 2VAIAK4
        
        accumulated_trait = {triat:torch.zeros(len(enc), dtype=torch.float32) for triat, enc in zip(self.encoded_trait_columns, self.trait_encoders)}
        for ai in associated_indices:
            # sample = super(LAPIS_sGIAA_HistogramDataset, self).__getitem__(ai, use_image=False)
            sample = LAPIS_PIAADataset.__getitem__(self, ai, use_image=False)

            # One-hot encode 'response' and accumulate
            round_score = min(int(sample['response'])//10, 9)
            response_one_hot = F.one_hot(torch.tensor(round_score), num_classes=max_response_score)
            accumulated_response += response_one_hot
            if self.disable_onehot:
                accumulated_vaia += torch.stack([sample[f'VAIAK{i}'] for i in range(1, 8)], dim=0).view(-1)
                accumulated_2vaia += torch.stack([sample[f'2VAIAK{i}'] for i in range(1, 5)], dim=0).view(-1)
            else:
                accumulated_vaia += torch.stack([sample[f'VAIAK{i}_onehot'] for i in range(1, 8)], dim=0).view(-1)
                accumulated_2vaia += torch.stack([sample[f'2VAIAK{i}_onehot'] for i in range(1, 5)], dim=0).view(-1)
            for trait in self.encoded_trait_columns:
                accumulated_trait[trait] += sample[f'{trait}_onehot']

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
        # accumulated_histogram['art_type'] = onehot_traits[0]
        accumulated_histogram['onehot_traits'] = torch.cat(onehot_traits[2:])

        accumulated_histogram['n_samples'] = total_samples
        accumulated_histogram['imgName'] = sample['imgName']

        return accumulated_histogram
    
    def __getitem__(self, idx):
        histograms = self.precomputed_data[idx]
        item_data = copy.deepcopy(random.choice(histograms))
        img_sample = LAPIS_PIAADataset.__getitem__(self, self.image_to_indices_map[self.unique_images[idx]][0], use_image=True)
        inherit_list = ['image', 'QIP', 'genre_onehot', 'style_onehot']
        for item in inherit_list:
            item_data[item] = img_sample[item]
        return item_data


class LAPIS_PIAA_HistogramDataset(LAPIS_PIAADataset):
    def __init__(self, root_dir, transform=None, data=None, map_file=None, precompute_file=None, disable_onehot=False):
        super().__init__(root_dir, transform)
        if data is not None:
            self.data = data
        self.disable_onehot = disable_onehot

    def traits_len(self):
        units_len = [len(enc) for enc in self.trait_encoders[1:]]  # Encoders for various traits
        units_len.extend([7] * 11)  # Extending for VAIAK1 to VAIAK7 and 2VAIAK1 to 2VAIAK4, 11 in total
        return units_len

    def __getitem__(self, idx):
        # Assuming maximum scores for response and VAIAK/2VAIAK ratings for proper one-hot encoding
        max_response_score = 10  # Adjust based on your data
        max_vaia_score = 7
        
        # Initialize accumulators for one-hot encoded vectors
        accumulated_response = torch.zeros(max_response_score, dtype=torch.float32)
        # accumulated_vaia = torch.zeros(7 * max_vaia_score, dtype=torch.float32)  # For VAIAK1 to VAIAK7
        # accumulated_2vaia = torch.zeros(4 * max_vaia_score, dtype=torch.float32)  # For 2VAIAK1 to 2VAIAK4
        accumulated_trait = {triat:torch.zeros(len(enc), dtype=torch.float32) for triat, enc in zip(self.encoded_trait_columns, self.trait_encoders)}
        sample = LAPIS_PIAADataset.__getitem__(self, idx, use_image=True)
        
        # One-hot encode 'response' and accumulate
        round_score = min(int(sample['response'])//10, 9)
        accumulated_response = F.one_hot(torch.tensor(round_score), num_classes=max_response_score)
        if self.disable_onehot:
            accumulated_vaia = torch.stack([sample[f'VAIAK{i}'] for i in range(1, 8)], dim=0).view(-1)
            accumulated_2vaia = torch.stack([sample[f'2VAIAK{i}'] for i in range(1, 5)], dim=0).view(-1)
        else:
            accumulated_vaia = torch.stack([sample[f'VAIAK{i}_onehot'] for i in range(1, 8)], dim=0).view(-1)
            accumulated_2vaia = torch.stack([sample[f'2VAIAK{i}_onehot'] for i in range(1, 5)], dim=0).view(-1)
        for trait in self.encoded_trait_columns:
            accumulated_trait[trait] += sample[f'{trait}_onehot']

        # Prepare the final accumulated histogram for the image
        accumulated_histogram = {
            'aestheticScore': accumulated_response,
            'onehot_traits': accumulated_trait,
            'VAIAK': accumulated_vaia,
            '2VAIAK': accumulated_2vaia,
        }

        # Average out histograms over the number of samples
        onehot_traits = list(accumulated_histogram['onehot_traits'].values())
        # accumulated_histogram['art_type'] = onehot_traits[0]
        accumulated_histogram['onehot_traits'] = torch.cat(onehot_traits[2:])

        accumulated_histogram['n_samples'] = 1
        inherit_list = ['imgName', 'image', 'QIP', 'genre_onehot', 'style_onehot']
        for item in inherit_list:
            accumulated_histogram[item] = sample[item]

        return accumulated_histogram

class LAPIS_PIAA_HistogramDataset_imgsort(LAPIS_GIAA_HistogramDataset):
    def __init__(self, root_dir, transform=None, data=None, map_file=None, disable_onehot=False):
        LAPIS_PIAADataset.__init__(self, root_dir, transform)
        if data is not None:
            self.data = data
        self.disable_onehot = disable_onehot
        
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

    def traits_len(self):
        units_len = [len(enc) for enc in self.trait_encoders[1:]]  # Encoders for various traits
        units_len.extend([7] * 11)  # Extending for VAIAK1 to VAIAK7 and 2VAIAK1 to 2VAIAK4, 11 in total
        return units_len

    def __getitem__(self, idx):
        associated_indices = self.image_to_indices_map[self.unique_images[idx]]
        sample = LAPIS_PIAADataset.__getitem__(self, associated_indices[0], use_image=True)
        image = sample['image']

        # Assuming maximum scores for response and VAIAK/2VAIAK ratings for proper one-hot encoding
        max_response_score = 10  # Adjust based on your data
        # max_vaia_score = 7

        # List to hold histograms for each sample
        histograms_list = []
        for ai in associated_indices:
            # Initialize accumulators for one-hot encoded vectors
            accumulated_trait = {triat:torch.zeros(len(enc), dtype=torch.float32) for triat, enc in zip(self.encoded_trait_columns, self.trait_encoders)}
            sample = LAPIS_PIAADataset.__getitem__(self, ai, use_image=False)

            # One-hot encode 'response' and accumulate
            round_score = min(int(sample['response'])//10, 9)
            accumulated_response = F.one_hot(torch.tensor(round_score), num_classes=max_response_score)

            if self.disable_onehot:
                accumulated_vaia = torch.stack([sample[f'VAIAK{i}'] for i in range(1, 8)], dim=0).view(-1)
                accumulated_2vaia = torch.stack([sample[f'2VAIAK{i}'] for i in range(1, 5)], dim=0).view(-1)
            else:
                accumulated_vaia = torch.stack([sample[f'VAIAK{i}_onehot'] for i in range(1, 8)], dim=0).view(-1)
                accumulated_2vaia = torch.stack([sample[f'2VAIAK{i}_onehot'] for i in range(1, 5)], dim=0).view(-1)
            for trait in self.encoded_trait_columns:
                accumulated_trait[trait] = sample[f'{trait}_onehot']

            # Prepare the final accumulated histogram for the image
            accumulated_histogram = {
                'aestheticScore': accumulated_response,
                'onehot_traits': accumulated_trait,
                'VAIAK': accumulated_vaia,
                '2VAIAK': accumulated_2vaia,
                'participant_id': sample['participant_id'],
            }
            histograms_list.append(accumulated_histogram)

        # Now, stack the histograms after the loop
        accumulated_histogram = {
            'image': image,
            'userId': [h['participant_id'] for h in histograms_list],
            'aestheticScore': torch.stack([h['aestheticScore'] for h in histograms_list]),
            'VAIAK': torch.stack([h['VAIAK'] for h in histograms_list]),
            '2VAIAK': torch.stack([h['2VAIAK'] for h in histograms_list]),
        }
        
        h_onehot = []
        for h in histograms_list:
            onehot_traits = list(h['onehot_traits'].values())
            h_onehot.append(torch.cat(onehot_traits[2:]))
        accumulated_histogram['onehot_traits'] = torch.stack(h_onehot)

        accumulated_histogram['n_samples'] = len(associated_indices)
        inherit_list = ['imgName', 'QIP', 'genre_onehot', 'style_onehot']
        for item in inherit_list:
            accumulated_histogram[item] = sample[item]

        return accumulated_histogram

    def decode_batch_to_dataframe(self, batch_features):
        encoded_trait_columns = self.encoded_trait_columns[2:]
        trait_encoders = self.trait_encoders[2:]
        # ['age', 'art_type', 'nationality', 'demo_gender', 'demo_edu', 'demo_colorblind']
        # Invert the encoders to create decoders
        trait_decoders = [{idx: value for value, idx in encoder.items()} for encoder in trait_encoders]
        
        # Calculate the splits for each segment based on the number of features
        num_features = [len(encoder) for encoder in trait_encoders]
        splits = np.cumsum(num_features)[:-1]

        decoded_batch = []
        for onehot_encoded_vector in batch_features:
            # Split the one-hot encoded vector into segments
            segments = np.split(onehot_encoded_vector, splits)
            
            # Decode each segment
            decoded_features = {
                trait: decoder[np.argmax(segment)] 
                for trait, decoder, segment in zip(encoded_trait_columns, trait_decoders, segments)
            }
            decoded_batch.append(decoded_features)

        # Convert the list of dictionaries to a pandas DataFrame
        decoded_df = pd.DataFrame(decoded_batch)
        return decoded_df


def collate_fn(batch):
    # Extracting individual components
    traits_histograms_concatenated = torch.cat([
        torch.stack([item['onehot_traits'] for item in batch]),
        torch.stack([item['VAIAK'] for item in batch]),
        torch.stack([item['2VAIAK'] for item in batch]),
        ], dim=1)
    
    QIP_collated = {}
    for item in batch:
        for key, value in item['QIP'].items():
            if key not in QIP_collated:
                QIP_collated[key] = []
            QIP_collated[key].append(value)
    qip = torch.tensor(list(QIP_collated.values())).permute(1, 0)

    return {
        'imgName':[item['imgName'] for item in batch],
        'image': torch.stack([item['image'] for item in batch]),
        'aestheticScore': torch.stack([item['aestheticScore'] for item in batch]),
        'traits': traits_histograms_concatenated,
        'style_onehot': torch.stack([item['style_onehot'] for item in batch]),
        'genre_onehot': torch.stack([item['genre_onehot'] for item in batch]),
        'QIP': qip
    }

def collate_fn_imgsort(batch):
    images = [item['image'].unsqueeze(0).repeat(item['aestheticScore'].shape[0], 1, 1, 1) for item in batch]
    images_stacked = torch.cat(images)
    
    # Extracting individual components
    traits_histograms_concatenated = torch.cat([
        torch.cat([item['onehot_traits'] for item in batch]),
        torch.cat([item['VAIAK'] for item in batch]),
        torch.cat([item['2VAIAK'] for item in batch]),
        ], dim=1)
    
    userID = []
    for item in batch:
        if isinstance(item['userId'], int):
            userID.append(item['userId'])
        elif isinstance(item['userId'], list):
            userID.extend(item['userId'])
    
    QIP_collated = {}
    style_onehot_collated, genre_onehot_collated = [], []
    for item in batch:
        n_repeat = item['aestheticScore'].shape[0]
        for key, value in item['QIP'].items():
            if key not in QIP_collated:
                QIP_collated[key] = []
            for i in range(n_repeat):
                QIP_collated[key].append(value)
        for i in range(n_repeat):
            style_onehot_collated.append(item['style_onehot'])
            genre_onehot_collated.append(item['genre_onehot'])
    qip = torch.tensor(list(QIP_collated.values())).permute(1, 0)
    
    return {
        'userId': torch.stack(userID),
        'imgName':[item['imgName'] for item in batch],
        'image': images_stacked,
        'aestheticScore': torch.cat([item['aestheticScore'] for item in batch]),
        'traits': traits_histograms_concatenated,
        'style_onehot': torch.stack(style_onehot_collated),
        'genre_onehot': torch.stack(genre_onehot_collated),        
        'QIP': qip
    }

def collate_fn_pair(batch):
    # Flatten the batch: convert [(item1, item2), (item3, item4), ...] to [item1, item2, item3, item4, ...]
    flattened_batch = [item for pair in batch for item in pair]
    
    # Call the original collate function with the flattened batch
    return collate_fn(flattened_batch)


def load_data(args, root_dir = datapath['LAPIS_datapath']):
    # Dataset transformations
    fold_id = getattr(args, 'fold_id', None)
    n_fold = getattr(args, 'n_fold', None)
    disable_onehot = getattr(args, 'disable_onehot', False)
    disable_resize = getattr(args, 'disable_resize', False)

    if disable_resize:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            ResizeToNearestDivisible(n=args.patch_size),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            ResizeToNearestDivisible(n=args.patch_size),
            transforms.ToTensor(),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

    # Create datasets with the appropriate transformations
    piaa_dataset = LAPIS_PIAADataset(root_dir, transform=train_transform)
    if getattr(args, 'binarized_vaiak', True):
        # Define your column names
        vaiaks = [f'VAIAK{i}' for i in range(1, 8)] + [f'2VAIAK{i}' for i in range(1, 5)]
        piaa_dataset.data[vaiaks] = piaa_dataset.data[vaiaks].gt(3).astype(float)
    
    train_dataset, val_dataset, test_dataset = create_image_split_dataset(piaa_dataset)
    # print(len(train_dataset), len(val_dataset), len(test_dataset))
    if getattr(args, 'use_cv', False):
        train_dataset, val_dataset, test_dataset = create_user_split_dataset_kfold(piaa_dataset, train_dataset, val_dataset, test_dataset, fold_id, n_fold=n_fold)

    is_trait_specific = getattr(args, 'trait', False) and getattr(args, 'value', False)
    is_disjoint_trait = getattr(args, 'trait_disjoint', True)    
    if is_trait_specific:
        args.value = float(args.value) if 'VAIAK' in args.trait else args.value
        if is_disjoint_trait:
            print(f'Split trait according to {args.trait} == {args.value} with disjoint user')
            train_dataset.data = train_dataset.data[train_dataset.data[args.trait] != args.value]
            val_dataset.data = val_dataset.data[val_dataset.data[args.trait] != args.value]    
        else:
            print(f'Split trait according to {args.trait} == {args.value} with joint user')
            train_dataset.data = train_dataset.data[train_dataset.data[args.trait] == args.value]
            val_dataset.data = val_dataset.data[val_dataset.data[args.trait] == args.value]
        test_dataset.data = test_dataset.data[test_dataset.data[args.trait] == args.value]
    
    print(len(train_dataset), len(val_dataset), len(test_dataset))

    """Precompute"""
    pkl_dir = './LAPIS_dataset_pkl'
    if getattr(args, 'use_cv', False):
        pkl_dir = os.path.join(pkl_dir, 'user_cv')
        ensure_dir_exists(pkl_dir)
        map_file = os.path.join(pkl_dir,'trainset_image_dct_%dfold.pkl'%fold_id)
        if args.trainset == 'GIAA':
            train_dataset = LAPIS_GIAA_HistogramDataset(root_dir, transform=train_transform, 
                data=train_dataset.data, map_file=map_file, disable_onehot=disable_onehot,
                precompute_file=os.path.join(pkl_dir,'trainset_GIAA_dct_%dfold.pkl'%fold_id))
        else:
            train_dataset = LAPIS_PIAA_HistogramDataset(root_dir, transform=train_transform, data=train_dataset.data, disable_onehot=disable_onehot)
        
        val_mapfile = os.path.join(pkl_dir,'valset_image_dct_%dfold.pkl'%fold_id)
        val_precompute_file = os.path.join(pkl_dir,'valset_GIAA_dct_%dfold.pkl'%fold_id)
        test_mapfile = os.path.join(pkl_dir,'testset_image_dct_%dfold.pkl'%fold_id)
        test_precompute_file = os.path.join(pkl_dir,'testset_GIAA_dct_%dfold.pkl'%fold_id)
    
    elif is_trait_specific:
        if is_disjoint_trait:
            pkl_dir = os.path.join(pkl_dir, 'trait_split')
        else:
            pkl_dir = os.path.join(pkl_dir, 'trait_specific')
        ensure_dir_exists(pkl_dir)
        suffix = '%s_%s'%(args.trait, args.value)
        map_file = os.path.join(pkl_dir,'trainset_image_dct_%s.pkl'%suffix)
        if args.trainset == 'GIAA':
            precompute_file = os.path.join(pkl_dir,'trainset_GIAA_dct_%s.pkl'%suffix)
            train_dataset = LAPIS_GIAA_HistogramDataset(root_dir, transform=train_transform, data=train_dataset.data, map_file=map_file, precompute_file=precompute_file, disable_onehot=disable_onehot)
        else:
            train_dataset = LAPIS_PIAA_HistogramDataset(root_dir, transform=train_transform, data=train_dataset.data, disable_onehot=disable_onehot)
        
        val_mapfile=os.path.join(pkl_dir,'valset_image_dct_%s.pkl'%suffix)
        val_precompute_file=os.path.join(pkl_dir,'valset_GIAA_dct_%s.pkl'%suffix)
        test_mapfile=os.path.join(pkl_dir,'testset_image_dct_%s.pkl'%suffix)
        test_precompute_file=os.path.join(pkl_dir,'testset_GIAA_dct_%s.pkl'%suffix)
    
    else:
        ensure_dir_exists(pkl_dir)
        map_file = os.path.join(pkl_dir,'trainset_image_dct.pkl')
        if args.trainset == 'GIAA':
            precompute_file = os.path.join(pkl_dir, 'trainset_GIAA_dct.pkl')
            train_dataset = LAPIS_GIAA_HistogramDataset(root_dir, transform=train_transform, data=train_dataset.data, map_file=map_file, precompute_file=precompute_file, disable_onehot=disable_onehot)
        else:
            train_dataset = LAPIS_PIAA_HistogramDataset(root_dir, transform=train_transform, data=train_dataset.data, disable_onehot=disable_onehot)

        val_mapfile=os.path.join(pkl_dir,'valset_image_dct.pkl')
        val_precompute_file=os.path.join(pkl_dir,'valset_GIAA_dct.pkl')
        test_mapfile=os.path.join(pkl_dir,'testset_image_dct.pkl')
        test_precompute_file=os.path.join(pkl_dir,'testset_GIAA_dct.pkl')
    
    val_giaa_dataset = LAPIS_GIAA_HistogramDataset(root_dir, transform=test_transform, data=val_dataset.data, map_file=val_mapfile, precompute_file=val_precompute_file, disable_onehot=disable_onehot)
    val_piaa_imgsort_dataset = LAPIS_PIAA_HistogramDataset_imgsort(root_dir, transform=test_transform, data=val_dataset.data, map_file=val_mapfile, disable_onehot=disable_onehot)
    test_giaa_dataset = LAPIS_GIAA_HistogramDataset(root_dir, transform=test_transform, data=test_dataset.data, map_file=test_mapfile, precompute_file=test_precompute_file, disable_onehot=disable_onehot)
    test_piaa_imgsort_dataset = LAPIS_PIAA_HistogramDataset_imgsort(root_dir, transform=test_transform, data=test_dataset.data, map_file=test_mapfile, disable_onehot=disable_onehot)
    return train_dataset, val_giaa_dataset, val_piaa_imgsort_dataset, test_giaa_dataset, test_piaa_imgsort_dataset


def load_data_testpair(args, root_dir = datapath['LAPIS_datapath']):
    # Dataset transformations
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    fold_id = getattr(args, 'fold_id', None)
    n_fold = getattr(args, 'n_fold', None)

    # Create datasets with the appropriate transformations
    piaa_dataset = LAPIS_PIAADataset(root_dir, transform=train_transform)
    if getattr(args, 'binarized_vaiak', True):
        # Define your column names
        vaiaks = [f'VAIAK{i}' for i in range(1, 8)] + [f'2VAIAK{i}' for i in range(1, 5)]
        piaa_dataset.data[vaiaks] = piaa_dataset.data[vaiaks].gt(3).astype(float)

    train_dataset, val_dataset, test_dataset = create_image_split_dataset(piaa_dataset)
    # print(len(train_dataset), len(val_dataset), len(test_dataset))
    if getattr(args, 'use_cv', False):
        train_dataset, val_dataset, test_dataset = create_user_split_dataset_kfold(piaa_dataset, train_dataset, val_dataset, test_dataset, fold_id, n_fold=n_fold)

    is_trait_disjoint = getattr(args, 'trait', False) and getattr(args, 'value', False)
    if is_trait_disjoint:
        args.value = float(args.value) if 'VAIAK' in args.trait else args.value
        print(f'Split trait according to {args.trait} == {args.value}')
        train_dataset.data = train_dataset.data[train_dataset.data[args.trait] != args.value]
        val_dataset.data = val_dataset.data[val_dataset.data[args.trait] != args.value]
        # test_dataset.data = test_dataset.data[test_dataset.data[args.trait] == args.value]
        testc_dataset = copy.deepcopy(test_dataset)
        test_dataset.data = test_dataset.data[test_dataset.data[args.trait] == args.value]
        testc_dataset.data = testc_dataset.data[testc_dataset.data[args.trait] != args.value]
    
    print(len(train_dataset), len(val_dataset), len(test_dataset), len(testc_dataset))

    """Precompute"""
    pkl_dir = './LAPIS_dataset_pkl'
    if getattr(args, 'use_cv', False):
        pkl_dir = os.path.join(pkl_dir, 'user_cv')
        if args.trainset == 'GIAA':
            train_dataset = LAPIS_GIAA_HistogramDataset(root_dir, transform=train_transform, 
                data=train_dataset.data, map_file=os.path.join(pkl_dir,'trainset_image_dct_%dfold.pkl'%fold_id), 
                precompute_file=os.path.join(pkl_dir,'trainset_GIAA_dct_%dfold.pkl'%fold_id))
        elif args.trainset == 'sGIAA':
            precompute_file = 'trainset_MIAA_dct_%dfold_IS.pkl'%fold_id if args.importance_sampling else 'trainset_MIAA_dct_%dfold.pkl'%fold_id
            train_dataset = LAPIS_sGIAA_HistogramDataset(root_dir, transform=train_transform, 
                data=train_dataset.data, map_file=os.path.join(pkl_dir,'trainset_image_dct_%dfold.pkl'%fold_id), 
                precompute_file=os.path.join(pkl_dir,precompute_file))
        else:
            train_dataset = LAPIS_PIAA_HistogramDataset(root_dir, transform=train_transform, data=train_dataset.data)
        
        val_mapfile = os.path.join(pkl_dir,'valset_image_dct_%dfold.pkl'%fold_id)
        val_precompute_file = os.path.join(pkl_dir,'valset_GIAA_dct_%dfold.pkl'%fold_id)
        test_mapfile = os.path.join(pkl_dir,'testset_image_dct_%dfold.pkl'%fold_id)
        test_precompute_file = os.path.join(pkl_dir,'testset_GIAA_dct_%dfold.pkl'%fold_id)
    
    elif is_trait_disjoint:
        pkl_dir = os.path.join(pkl_dir, 'trait_split')
        suffix = '%s_%s'%(args.trait, args.value)
        train_mapfile = os.path.join(pkl_dir,'trainset_image_dct_%s.pkl'%suffix)
        if args.trainset == 'GIAA':
            precompute_file = os.path.join(pkl_dir,'trainset_GIAA_dct_%s.pkl'%suffix)
            train_dataset = LAPIS_GIAA_HistogramDataset(root_dir, transform=train_transform, data=train_dataset.data, map_file=os.path.join(pkl_dir,'trainset_image_dct.pkl'), precompute_file=precompute_file)
        elif args.trainset == 'sGIAA':
            precompute_file = 'trainset_MIAA_dct_IS_%s.pkl'%suffix if args.importance_sampling else 'trainset_MIAA_dct_%s.pkl'%suffix
            precompute_file = os.path.join(pkl_dir,precompute_file)
            train_dataset = LAPIS_sGIAA_HistogramDataset(root_dir, transform=train_transform, data=train_dataset.data, map_file=os.path.join(pkl_dir,'trainset_image_dct.pkl'), precompute_file=precompute_file)    
        else:
            train_dataset = LAPIS_PIAA_HistogramDataset(root_dir, transform=train_transform, data=train_dataset.data)
        # test_sgiaa_dataset = LAPIS_sGIAA_HistogramDataset(root_dir, transform=test_transform, data=test_dataset.data, map_file=os.path.join(pkl_dir,'testset_image_dct.pkl'), precompute_file=os.path.join(pkl_dir,'testset_MIAA_dct.pkl'))

        val_mapfile=os.path.join(pkl_dir,'valset_image_dct_%s.pkl'%suffix)
        val_precompute_file=os.path.join(pkl_dir,'valset_GIAA_dct_%s.pkl'%suffix)
        test_mapfile=os.path.join(pkl_dir,'testset_image_dct_%s.pkl'%suffix)
        test_precompute_file=os.path.join(pkl_dir,'testset_GIAA_dct_%s.pkl'%suffix)
        testc_mapfile=os.path.join(pkl_dir,'testset_c_image_dct_%s.pkl'%suffix)
        testc_precompute_file=os.path.join(pkl_dir,'testset_c_GIAA_dct_%s.pkl'%suffix)
    else:
        train_mapfile = os.path.join(pkl_dir,'trainset_image_dct.pkl')
        if args.trainset == 'GIAA':
            precompute_file = os.path.join(pkl_dir, 'trainset_GIAA_dct.pkl')
            train_dataset = LAPIS_GIAA_HistogramDataset(root_dir, transform=train_transform, data=train_dataset.data, map_file=train_mapfile, precompute_file=precompute_file)
        elif args.trainset == 'sGIAA':
            precompute_file = 'trainset_MIAA_dct_IS.pkl' if args.importance_sampling else 'trainset_MIAA_dct.pkl'
            precompute_file = os.path.join(pkl_dir,precompute_file)
            train_dataset = LAPIS_sGIAA_HistogramDataset(root_dir, transform=train_transform, data=train_dataset.data, map_file=train_mapfile, precompute_file=precompute_file)
        else:
            train_dataset = LAPIS_PIAA_HistogramDataset(root_dir, transform=train_transform, data=train_dataset.data)
        # test_sgiaa_dataset = LAPIS_sGIAA_HistogramDataset(root_dir, transform=test_transform, data=test_dataset.data, map_file=os.path.join(pkl_dir,'testset_image_dct.pkl'), precompute_file=os.path.join(pkl_dir,'testset_MIAA_dct.pkl'))

        val_mapfile=os.path.join(pkl_dir,'valset_image_dct.pkl')
        val_precompute_file=os.path.join(pkl_dir,'valset_GIAA_dct.pkl')
        test_mapfile=os.path.join(pkl_dir,'testset_image_dct.pkl')
        test_precompute_file=os.path.join(pkl_dir,'testset_GIAA_dct.pkl')
    
    val_giaa_dataset = LAPIS_GIAA_HistogramDataset(root_dir, transform=test_transform, data=val_dataset.data, map_file=val_mapfile, precompute_file=val_precompute_file)
    val_piaa_imgsort_dataset = LAPIS_PIAA_HistogramDataset_imgsort(root_dir, transform=test_transform, data=val_dataset.data, map_file=val_mapfile)
    test_giaa_dataset = LAPIS_GIAA_HistogramDataset(root_dir, transform=test_transform, data=test_dataset.data, map_file=test_mapfile, precompute_file=test_precompute_file)
    testc_giaa_dataset = LAPIS_GIAA_HistogramDataset(root_dir, transform=test_transform, data=testc_dataset.data, map_file=testc_mapfile, precompute_file=testc_precompute_file)
    test_piaa_imgsort_dataset = LAPIS_PIAA_HistogramDataset_imgsort(root_dir, transform=test_transform, data=test_dataset.data, map_file=test_mapfile)
    
    return train_dataset, val_giaa_dataset, val_piaa_imgsort_dataset, test_giaa_dataset, test_piaa_imgsort_dataset, testc_giaa_dataset


def load_testdata(args, root_dir = datapath['LAPIS_datapath']):
    # Dataset transformations
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    fold_id = getattr(args, 'fold_id', None)
    n_fold = getattr(args, 'n_fold', None)

    # Create datasets with the appropriate transformations
    piaa_dataset = LAPIS_PIAADataset(root_dir, transform=train_transform)
    if getattr(args, 'binarized_vaiak', True):
        # Define your column names
        vaiaks = [f'VAIAK{i}' for i in range(1, 8)] + [f'2VAIAK{i}' for i in range(1, 5)]
        piaa_dataset.data[vaiaks] = piaa_dataset.data[vaiaks].gt(3).astype(float)
    train_dataset, val_dataset, test_dataset = create_image_split_dataset(piaa_dataset)
    # print(len(train_dataset), len(val_dataset), len(test_dataset))
    if getattr(args, 'use_cv', False):
        train_dataset, val_dataset, test_dataset = create_user_split_dataset_kfold(piaa_dataset, train_dataset, val_dataset, test_dataset, fold_id, n_fold=n_fold)
    
    is_trait_specific = getattr(args, 'trait', False) and getattr(args, 'value', False)
    is_disjoint_trait = getattr(args, 'trait_disjoint', True)    
    
    if is_trait_specific:
        args.value = float(args.value) if 'VAIAK' in args.trait else args.value
        if is_disjoint_trait:
            print(f'Split trait according to {args.trait} == {args.value} with disjoint user')
            train_dataset.data = train_dataset.data[train_dataset.data[args.trait] != args.value]
            val_dataset.data = val_dataset.data[val_dataset.data[args.trait] != args.value]    
        else:
            print(f'Split trait according to {args.trait} == {args.value} with joint user')
            train_dataset.data = train_dataset.data[train_dataset.data[args.trait] == args.value]
            val_dataset.data = val_dataset.data[val_dataset.data[args.trait] == args.value]
        test_dataset.data = test_dataset.data[test_dataset.data[args.trait] == args.value]
    
    print(len(train_dataset), len(val_dataset), len(test_dataset))

    """Precompute"""
    pkl_dir = './LAPIS_dataset_pkl'
    if getattr(args, 'use_cv', False):
        pkl_dir = os.path.join(pkl_dir, 'user_cv')
        ensure_dir_exists(pkl_dir)
        test_mapfile = os.path.join(pkl_dir,'testset_image_dct_%dfold.pkl'%fold_id)
        test_precompute_file = os.path.join(pkl_dir,'testset_GIAA_dct_%dfold.pkl'%fold_id)
    
    elif is_trait_specific:
        if is_disjoint_trait:
            pkl_dir = os.path.join(pkl_dir, 'trait_split')
        else:
            pkl_dir = os.path.join(pkl_dir, 'trait_specific')
        ensure_dir_exists(pkl_dir)
        suffix = '%s_%s'%(args.trait, args.value)
        test_mapfile=os.path.join(pkl_dir,'testset_image_dct_%s.pkl'%suffix)
        test_precompute_file=os.path.join(pkl_dir,'testset_GIAA_dct_%s.pkl'%suffix)

    else:
        ensure_dir_exists(pkl_dir)
        test_mapfile=os.path.join(pkl_dir,'testset_image_dct.pkl')
        test_precompute_file=os.path.join(pkl_dir,'testset_GIAA_dct.pkl')
    
    test_giaa_dataset = LAPIS_GIAA_HistogramDataset(root_dir, transform=test_transform, data=test_dataset.data, map_file=test_mapfile, precompute_file=test_precompute_file)
    test_piaa_imgsort_dataset = LAPIS_PIAA_HistogramDataset_imgsort(root_dir, transform=test_transform, data=test_dataset.data, map_file=test_mapfile)
    return test_giaa_dataset, test_piaa_imgsort_dataset


def extract_features(model, dataloader, device):
    model.eval()  # Set the model to evaluation mode
    feature_dict = {}

    with torch.no_grad():
        for sample in tqdm(dataloader, leave=False):
            images = sample['image'].to(device)
            img_name = sample['imgName']  # Assuming the dataset provides image names
            
            # Pass the images through the model up to the avgpool layer
            features = model(images)
            features = torch.flatten(features, start_dim=1)  # Flatten the output of avgpool
            
            # Store the features in the dictionary with image name as the key
            for idx, name in enumerate(img_name):
                feature_dict[name] = features[idx].cpu().numpy()

    return feature_dict


def print_LAPIS_demodata(giaa_dataset, idx=0, font_size=20, wrap_width=40):
    # Load sample data from the GIAA dataset
    sample = giaa_dataset.__getitem__(idx, print_detail=True)
    
    # Define the scale and calculate the target mean
    scale = torch.arange(0, 10)
    aesthetic_score_histogram = sample['aestheticScore']
    target_mean = torch.sum(aesthetic_score_histogram * scale)
    img_path = sample['image_path']
    
    # Calculate the mean score for PIAA component
    mscore = sum(sample['p_score']) / len(sample['p_score'])
    print(target_mean)

    # Plotting with only two subplots (image and histogram)
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [1, 1.5]})
    
    # Plot the image on axs[0]
    image = Image.open(img_path)
    axs[0].imshow(image)
    axs[0].axis('off')
    axs[0].set_title("Image", fontsize=font_size)

    # Plot the aesthetic score histogram on axs[1]
    axs[1].bar(scale.numpy(), aesthetic_score_histogram.numpy(), width=0.8, align='center')
    axs[1].set_title("Aesthetic Score Histogram", fontsize=font_size)
    axs[1].set_xlabel("Aesthetic Score", fontsize=font_size)
    axs[1].set_ylabel("Probability", fontsize=font_size)
    axs[1].tick_params(axis='both', labelsize=font_size)  # Set tick label size

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig('LAPIS_histogram_component.pdf')


if __name__ == '__main__':
    from utils.argflags import parse_arguments, parse_arguments_piaa
    import torch.nn as nn
    from torchvision.models import resnet50

    args = parse_arguments()
    print(args)

    train_dataset, val_giaa_dataset, val_piaa_imgsort_dataset, test_giaa_dataset, test_piaa_imgsort_dataset = load_data(args, datapath['LAPIS_datapath'])
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_giaa_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_giaa_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    test_piaa_imgsort_dataloader = DataLoader(test_piaa_imgsort_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, timeout=300, collate_fn=collate_fn_imgsort)
    
    print_LAPIS_demodata(test_giaa_dataset)
    raise Exception
    for sample in tqdm(train_dataloader):
        print(sample['traits'].shape)
        print(sample['genre_onehot'].shape)
        print(sample['style_onehot'].shape)
        raise Exception


    # Initialize the pretrained ResNet50 model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resnet = resnet50(pretrained=True).to(device)
    model = nn.Sequential(*list(resnet.children())[:-1])  # Remove the last fc layer
    model = model.to(device)
    
    # Extract features
    # feature_dict = extract_features(model, test_piaa_imgsort_dataloader, device)
    train_feature_dict = extract_features(model, train_dataloader, device)
    val_feature_dict = extract_features(model, val_dataloader, device)
    test_feature_dict = extract_features(model, test_dataloader, device)
    
    # Save the features to a file
    with open('LAPIS_train_extracted_features.pkl', 'wb') as f:
        pickle.dump(train_feature_dict, f)
    with open('LAPIS_val_extracted_features.pkl', 'wb') as f:
        pickle.dump(val_feature_dict, f)
    with open('LAPIS_test_extracted_features.pkl', 'wb') as f:
        pickle.dump(test_feature_dict, f)



