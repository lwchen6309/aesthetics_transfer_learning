import os
import torch
from torchvision import transforms
# import torch.nn.functional as F
from PARA_PIAA_dataloader import PARA_PIAADataset, create_user_split_dataset_kfold, split_dataset_by_images, split_data_by_user,datapath
import random
import pickle
from tqdm import tqdm
import copy
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


def ensure_dir_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


class PARA_GIAA_HistogramDataset(PARA_PIAADataset):
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

        # Initialize accumulators
        accumulated_histogram = {
            'aestheticScore': torch.zeros(len([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])),
            'attributes': {
                'qualityScore': torch.zeros(5),
                'compositionScore': torch.zeros(5),
                'colorScore': torch.zeros(5),
                'dofScore': torch.zeros(5),
                'contentScore': torch.zeros(5),
                'lightScore': torch.zeros(5),
                'contentPreference': torch.zeros(5),
                'willingnessToShare': torch.zeros(5)
            },
            'big5': {
                'personality-E': torch.zeros(10),
                'personality-A': torch.zeros(10),
                'personality-N': torch.zeros(10),
                'personality-O': torch.zeros(10),
                'personality-C': torch.zeros(10)
            },
            'traits': {
                'age': torch.zeros(len(self.age_encoder)),
                'gender': torch.zeros(len(self.gender_encoder)),
                'EducationalLevel': torch.zeros(len(self.education_encoder)),
                'artExperience': torch.zeros(len(self.art_experience_encoder)),
                'photographyExperience': torch.zeros(len(self.photo_experience_encoder))
            }
        }
        
        for random_idx in associated_indices:
            sample = super().__getitem__(random_idx, use_image=False)
            
            # Compute aesthetic score histogram
            bin_idx = self._discretize(sample['aestheticScores']['aestheticScore'], [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
            accumulated_histogram['aestheticScore'][bin_idx] += 1

            # Round the quality score to the nearest integer and update histogram
            rounded_quality_score = round(sample['aestheticScores']['qualityScore'])
            accumulated_histogram['attributes']['qualityScore'][rounded_quality_score - 1] += 1

            # Compute histograms for other attributes
            for attribute in ['compositionScore', 'colorScore', 'dofScore', 'contentScore', 'lightScore', 'contentPreference', 'willingnessToShare']:
                bin_idx = int(sample['aestheticScores'][attribute] - 1)
                accumulated_histogram['attributes'][attribute][bin_idx] += 1

            # Compute histograms for traits
            for trait, _ in accumulated_histogram['big5'].items():
                bin_idx = int(sample['userTraits'][trait] - 1)
                accumulated_histogram['big5'][trait][bin_idx] += 1
            
            # Update onehot traits
            for trait in accumulated_histogram['traits'].keys():
                accumulated_histogram['traits'][trait] += sample['userTraits'][trait]

        # Average out histograms over the number of samples
        total_samples = len(associated_indices)
        accumulated_histogram['aestheticScore'] /= total_samples
        for trait in accumulated_histogram['traits'].keys():
            accumulated_histogram['traits'][trait] /= total_samples
        for key in accumulated_histogram['attributes'].keys():
            accumulated_histogram['attributes'][key] /= total_samples
        for key in accumulated_histogram['big5'].keys():
            accumulated_histogram['big5'][key] /= total_samples

        accumulated_histogram['n_samples'] = total_samples
        # accumulated_histogram['image'] = img_sample['image']

        # Convert histograms to stacked tensors
        stacked_attributes = torch.cat(list(accumulated_histogram['attributes'].values()))
        stacked_traits = torch.cat(list(accumulated_histogram['big5'].values()))
        stacked_onehot_traits = torch.cat(list(accumulated_histogram['traits'].values()))

        # Replace dictionaries with stacked tensors
        accumulated_histogram['attributes'] = stacked_attributes
        accumulated_histogram['big5'] = stacked_traits
        accumulated_histogram['traits'] = stacked_onehot_traits

        return accumulated_histogram

    def decode_onhot_traits(self):        
        trait_encoders = [self.age_encoder, self.gender_encoder, self.education_encoder, self.art_experience_encoder, self.photo_experience_encoder]
        traits = []
        for encoders in trait_encoders:
            decoder = {v: k for k, v in encoders.items()}
            traits.extend([decoder[i] for i in range(len(decoder))])
        return traits
    
    def __getitem__(self, idx):
        item_data = copy.deepcopy(self.precomputed_data[idx])
        img_sample = super().__getitem__(self.image_to_indices_map[self.unique_images[idx]][0], use_image=True)
        item_data['image_path'] = img_sample['image_path']
        item_data['image'] = img_sample['image']
        item_data['semantic'] = img_sample['imageAttributes']['semantic']
        return item_data

    def _discretize(self, value, bins):
        for i, bin_value in enumerate(bins):
            if value <= bin_value:
                return i
        return len(bins) - 1

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


class PARA_sGIAA_HistogramDataset(PARA_GIAA_HistogramDataset):
    def __init__(self, root_dir, transform=None, data=None, map_file=None, precompute_file=None, 
                importance_sampling=False, num_samples=40):

        self.importance_sampling = importance_sampling
        self.num_samples = num_samples
        super().__init__(root_dir, transform, data, map_file, precompute_file)

    def precompute_data(self):
        self.precomputed_data = []
        for idx in tqdm(range(len(self))):
            self.precomputed_data.append(self._compute_sgiaa_item(idx, num_samples=self.num_samples, importance_sampling=self.importance_sampling))

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

    def compute_score_hist(self, associated_indices):
        bin_indecies = []
        for random_idx in associated_indices:
            sample = PARA_PIAADataset.__getitem__(random_idx, use_image=False)
            bin_idx = self._discretize(sample['aestheticScores']['aestheticScore'], [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
            bin_indecies.append(bin_idx)
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

    def _compute_item(self, idx, is_giaa=True, sample_prob=None):
        associated_indices = self.image_to_indices_map[self.unique_images[idx]]
        upper_bound = len(associated_indices) if is_giaa else len(associated_indices)-1
        n_sample = random.randint(2, upper_bound)
        if sample_prob is not None:
            associated_indices = random.choices(associated_indices, weights=sample_prob, k=n_sample)
        else:
            associated_indices = random.sample(associated_indices, n_sample)
            # associated_indices = random.choices(associated_indices, k=n_sample)
        
        # scores, unique_scores, hist_values = self.compute_score_hist(associated_indices)
        # prob = hist_values / hist_values.sum()
        # entropy = sum(prob * np.log(prob))
        # print(entropy)
        # ent = []
        # for j in range(100):
        #     if sample_prob is not None:
        #         indices = random.choices(associated_indices, weights=sample_prob, k=n_sample)
        #     else:
        #         indices = random.sample(associated_indices, n_sample)
        #         # associated_indices = random.choices(associated_indices, k=n_sample)
        #     scores, unique_scores, hist_values = self.compute_score_hist(indices)
        #     prob = hist_values / hist_values.sum()
        #     entropy = sum(prob * np.log(prob))
        #     ent.append(entropy)
        # ent = np.array(ent)
        # print(ent.mean(), ent.std())

        # Initialize accumulators
        accumulated_histogram = {
            'aestheticScore': torch.zeros(len([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])),
            'attributes': {
                'qualityScore': torch.zeros(5),
                'compositionScore': torch.zeros(5),
                'colorScore': torch.zeros(5),
                'dofScore': torch.zeros(5),
                'contentScore': torch.zeros(5),
                'lightScore': torch.zeros(5),
                'contentPreference': torch.zeros(5),
                'willingnessToShare': torch.zeros(5)
            },
            'big5': {
                'personality-E': torch.zeros(10),
                'personality-A': torch.zeros(10),
                'personality-N': torch.zeros(10),
                'personality-O': torch.zeros(10),
                'personality-C': torch.zeros(10)
            },
            'traits': {
                'age': torch.zeros(len(self.age_encoder)),
                'gender': torch.zeros(len(self.gender_encoder)),
                'EducationalLevel': torch.zeros(len(self.education_encoder)),
                'artExperience': torch.zeros(len(self.art_experience_encoder)),
                'photographyExperience': torch.zeros(len(self.photo_experience_encoder))
            }
        }
        
        for random_idx in associated_indices:
            sample = PARA_PIAADataset.__getitem__(self, random_idx, use_image=False)
            
            # Compute aesthetic score histogram
            bin_idx = self._discretize(sample['aestheticScores']['aestheticScore'], [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
            accumulated_histogram['aestheticScore'][bin_idx] += 1

            # Round the quality score to the nearest integer and update histogram
            rounded_quality_score = round(sample['aestheticScores']['qualityScore'])
            accumulated_histogram['attributes']['qualityScore'][rounded_quality_score - 1] += 1

            # Compute histograms for other attributes
            for attribute in ['compositionScore', 'colorScore', 'dofScore', 'contentScore', 'lightScore', 'contentPreference', 'willingnessToShare']:
                bin_idx = int(sample['aestheticScores'][attribute] - 1)
                accumulated_histogram['attributes'][attribute][bin_idx] += 1

            # Compute histograms for traits
            for trait, _ in accumulated_histogram['big5'].items():
                bin_idx = int(sample['userTraits'][trait] - 1)
                accumulated_histogram['big5'][trait][bin_idx] += 1
            
            # Update onehot traits
            for trait in accumulated_histogram['traits'].keys():
                accumulated_histogram['traits'][trait] += sample['userTraits'][trait]

        # Average out histograms over the number of samples
        total_samples = len(associated_indices)
        accumulated_histogram['aestheticScore'] /= total_samples
        for trait in accumulated_histogram['traits'].keys():
            accumulated_histogram['traits'][trait] /= total_samples
        for key in accumulated_histogram['attributes'].keys():
            accumulated_histogram['attributes'][key] /= total_samples
        for key in accumulated_histogram['big5'].keys():
            accumulated_histogram['big5'][key] /= total_samples

        accumulated_histogram['n_samples'] = total_samples
        # accumulated_histogram['image'] = img_sample['image']

        # Convert histograms to stacked tensors
        stacked_attributes = torch.cat(list(accumulated_histogram['attributes'].values()))
        stacked_traits = torch.cat(list(accumulated_histogram['big5'].values()))
        stacked_onehot_traits = torch.cat(list(accumulated_histogram['traits'].values()))

        # Replace dictionaries with stacked tensors
        accumulated_histogram['attributes'] = stacked_attributes
        accumulated_histogram['big5'] = stacked_traits
        accumulated_histogram['traits'] = stacked_onehot_traits

        return accumulated_histogram

    def __getitem__(self, idx):
        histograms = self.precomputed_data[idx]
        selected_histogram = copy.deepcopy(random.choice(histograms))
        img_sample = PARA_PIAADataset.__getitem__(self, self.image_to_indices_map[self.unique_images[idx]][0], use_image=True)
            
        selected_histogram['image'] = img_sample['image']
        selected_histogram['semantic'] = img_sample['imageAttributes']['semantic']
        return selected_histogram


class PARA_PIAA_HistogramDataset(PARA_PIAADataset):
    def __init__(self, root_dir, transform=None, data=None):
        super().__init__(root_dir, transform)
        if data is not None:
            self.data = data

        # Initialize accumulators
        self.accumulated_histogram_template = {
            'aestheticScore': torch.zeros(len([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])),
            'attributes': {
                'qualityScore': torch.zeros(5),
                'compositionScore': torch.zeros(5),
                'colorScore': torch.zeros(5),
                'dofScore': torch.zeros(5),
                'contentScore': torch.zeros(5),
                'lightScore': torch.zeros(5),
                'contentPreference': torch.zeros(5),
                'willingnessToShare': torch.zeros(5)
            },
            'big5': {
                'personality-E': torch.zeros(10),
                'personality-A': torch.zeros(10),
                'personality-N': torch.zeros(10),
                'personality-O': torch.zeros(10),
                'personality-C': torch.zeros(10)
            },
            'traits': {
                'age': torch.zeros(len(self.age_encoder)),
                'gender': torch.zeros(len(self.gender_encoder)),
                'EducationalLevel': torch.zeros(len(self.education_encoder)),
                'artExperience': torch.zeros(len(self.art_experience_encoder)),
                'photographyExperience': torch.zeros(len(self.photo_experience_encoder))
            }
        }

    def _discretize(self, value, bins):
        """Helper function to discretize a value given specific bins"""
        for i, bin_value in enumerate(bins):
            if value <= bin_value:
                return i
        return len(bins) - 1

    def __getitem__(self, idx):
        accumulated_histogram = copy.deepcopy(self.accumulated_histogram_template)
        sample = super().__getitem__(idx)
        # Compute aesthetic score histogram
        bin_idx = self._discretize(sample['aestheticScores']['aestheticScore'], [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
        accumulated_histogram['aestheticScore'][bin_idx] += 1

        # Round the quality score to the nearest integer and update histogram
        rounded_quality_score = round(sample['aestheticScores']['qualityScore'])
        accumulated_histogram['attributes']['qualityScore'][rounded_quality_score - 1] += 1

        # Compute histograms for other attributes
        for attribute in ['compositionScore', 'colorScore', 'dofScore', 'contentScore', 'lightScore', 'contentPreference', 'willingnessToShare']:
            bin_idx = int(sample['aestheticScores'][attribute] - 1)
            accumulated_histogram['attributes'][attribute][bin_idx] += 1

        # Compute histograms for traits
        for trait, _ in accumulated_histogram['big5'].items():
            bin_idx = int(sample['userTraits'][trait] - 1)
            accumulated_histogram['big5'][trait][bin_idx] += 1
        
        # Update onehot traits
        for trait in accumulated_histogram['traits'].keys():
            accumulated_histogram['traits'][trait] += sample['userTraits'][trait]

        accumulated_histogram['n_samples'] = 1
        accumulated_histogram['image'] = sample['image']

        # Convert histograms to stacked tensors
        stacked_attributes = torch.cat(list(accumulated_histogram['attributes'].values()))
        stacked_traits = torch.cat(list(accumulated_histogram['big5'].values()))
        stacked_onehot_traits = torch.cat(list(accumulated_histogram['traits'].values()))

        # Replace dictionaries with stacked tensors
        accumulated_histogram['attributes'] = stacked_attributes
        accumulated_histogram['big5'] = stacked_traits
        accumulated_histogram['traits'] = stacked_onehot_traits

        return accumulated_histogram


class PARA_PIAA_HistogramDataset_imgsort(PARA_PIAADataset):
    def __init__(self, root_dir, transform=None, data=None, map_file=None):
        super().__init__(root_dir, transform)
        if data is not None:
            self.data = data
        
        # Create a dictionary mapping each unique image name to a list of indices corresponding to that image
        if map_file and os.path.exists(map_file):
            # Load precomputed map from file
            print('Loading image to indices map from file...')
            self.image_to_indices_map = self._load_map(map_file)
            self.unique_images = [img for img in self.image_to_indices_map.keys()]
        else:
            self.unique_images = self.data['imageName'].unique()
            # Compute the mapping from images to indices
            self.image_to_indices_map = dict()
            for image in tqdm(self.unique_images, desc='Processing images'):
                indices_for_image = [i for i, img in enumerate(self.data['imageName']) if img == image]
                # Check if any index is out of bounds
                if any(not idx < len(self.data) for idx in indices_for_image):
                    print(indices_for_image)
                    raise Exception('Index out of bounds for the data.')
                self.image_to_indices_map[image] = indices_for_image
                
            if map_file:
                print(f"Saving image to indices map to {map_file}")
                self._save_map(map_file)

    def _discretize(self, value, bins):
        """Helper function to discretize a value given specific bins"""
        for i, bin_value in enumerate(bins):
            if value <= bin_value:
                return i
        return len(bins) - 1

    def _save_map(self, file_path):
        """Save image to indices map to a file."""
        with open(file_path, 'wb') as f:
            pickle.dump(self.image_to_indices_map, f)

    def _load_map(self, file_path):
        """Load image to indices map from a file."""
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def __len__(self):
        return len(self.unique_images)
    
    def __getitem__(self, idx):
        # Determine the number of samples to draw (d)
        associated_indices = self.image_to_indices_map[self.unique_images[idx]]
        
        # List to hold histograms for each sample
        histograms_list = []

        sample = super().__getitem__(associated_indices[0], use_image=True)
        image = sample['image']
        img_path = sample['image_path']

        histogram_tmp = {
            'aestheticScore': torch.zeros(len([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])),
            'attributes': {
                'qualityScore': torch.zeros(5),
                'compositionScore': torch.zeros(5),
                'colorScore': torch.zeros(5),
                'dofScore': torch.zeros(5),
                'contentScore': torch.zeros(5),
                'lightScore': torch.zeros(5),
                'contentPreference': torch.zeros(5),
                'willingnessToShare': torch.zeros(5)
            },
            'big5': {
                'personality-E': torch.zeros(10),
                'personality-A': torch.zeros(10),
                'personality-N': torch.zeros(10),
                'personality-O': torch.zeros(10),
                'personality-C': torch.zeros(10)
            },
            'traits': {
                'age': torch.zeros(len(self.age_encoder)),
                'gender': torch.zeros(len(self.gender_encoder)),
                'EducationalLevel': torch.zeros(len(self.education_encoder)),
                'artExperience': torch.zeros(len(self.art_experience_encoder)),
                'photographyExperience': torch.zeros(len(self.photo_experience_encoder))
            }
        }
        
        for _idx, idx in enumerate(associated_indices):
            # Initialize histogram for the current sample
            histogram = copy.deepcopy(histogram_tmp)
            sample = super().__getitem__(idx, use_image=False)
            # Compute aesthetic score histogram
            bin_idx = self._discretize(sample['aestheticScores']['aestheticScore'], [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
            histogram['aestheticScore'][bin_idx] += 1

            # Round the quality score to the nearest integer and update histogram
            rounded_quality_score = round(sample['aestheticScores']['qualityScore'])
            histogram['attributes']['qualityScore'][rounded_quality_score - 1] += 1

            # Compute histograms for other attributes
            for attribute in ['compositionScore', 'colorScore', 'dofScore', 'contentScore', 'lightScore', 'contentPreference', 'willingnessToShare']:
                bin_idx = int(sample['aestheticScores'][attribute] - 1)
                histogram['attributes'][attribute][bin_idx] += 1

            # Compute histograms for traits
            for trait, _ in histogram['big5'].items():
                bin_idx = int(sample['userTraits'][trait] - 1)
                histogram['big5'][trait][bin_idx] += 1

            # Update onehot traits
            for trait in histogram['traits'].keys():
                histogram['traits'][trait] += sample['userTraits'][trait]

            # Convert histograms to stacked tensors
            stacked_attributes = torch.cat(list(histogram['attributes'].values()))
            stacked_traits = torch.cat(list(histogram['big5'].values()))
            stacked_onehot_traits = torch.cat(list(histogram['traits'].values()))
            
            # Replace dictionaries with stacked tensors
            histogram['attributes'] = stacked_attributes
            histogram['big5'] = stacked_traits
            histogram['traits'] = stacked_onehot_traits
            histogram['userId'] = sample['userId']
            
            # Add the histogram for the current sample to the list
            histograms_list.append(histogram)

        # Now, stack the histograms after the loop
        stacked_histogram = {
            'image':image,
            'image_path':img_path,
            'userId': [h['userId'] for h in histograms_list],
            'aestheticScore': torch.stack([h['aestheticScore'] for h in histograms_list]),
            'attributes': torch.stack([h['attributes'] for h in histograms_list]),
            'big5': torch.stack([h['big5'] for h in histograms_list]),
            'traits': torch.stack([h['traits'] for h in histograms_list]),
        }
        
        return stacked_histogram

    def decode_batch_to_dataframe(self, batch_features):
        # Invert the encoders to create decoders
        age_decoder = {idx: group for group, idx in self.age_encoder.items()}
        gender_decoder = {idx: gender for gender, idx in self.gender_encoder.items()}
        education_decoder = {idx: level for level, idx in self.education_encoder.items()}
        art_experience_decoder = {idx: experience for experience, idx in self.art_experience_encoder.items()}
        photo_experience_decoder = {idx: experience for experience, idx in self.photo_experience_encoder.items()}

        # Calculate the splits for each segment based on the number of features
        num_features = [len(self.age_encoder), len(self.gender_encoder), len(self.education_encoder),
                        len(self.art_experience_encoder), len(self.photo_experience_encoder)]

        splits = np.cumsum(num_features)[:-1]

        decoded_batch = []
        for onehot_encoded_vector in batch_features:
            # Split the one-hot encoded vector into segments
            segments = np.split(onehot_encoded_vector, splits)
            
            # Decode each segment
            decoded_features = {
                "age": age_decoder[np.argmax(segments[0])],
                "gender": gender_decoder[np.argmax(segments[1])],
                "education": education_decoder[np.argmax(segments[2])],
                "art_experience": art_experience_decoder[np.argmax(segments[3])],
                "photo_experience": photo_experience_decoder[np.argmax(segments[4])],
            }
            decoded_batch.append(decoded_features)

        # Convert the list of dictionaries to a pandas DataFrame
        decoded_df = pd.DataFrame(decoded_batch)
        return decoded_df


def collate_fn_imgsort(batch):
    # Extracting individual components
    aesthetic_scores = [item['aestheticScore'] for item in batch]
    attributes = [item['attributes'] for item in batch]
    userId = []
    for item in batch:
        userId += item['userId'] 
    traits_histograms = [item['big5'] for item in batch]
    onehot_big5s = [item['traits'] for item in batch]

    # Stacking images
    images = [item['image'].unsqueeze(0).repeat(item['big5'].shape[0], 1, 1, 1) for item in batch]
    images_stacked = torch.cat(images)
    img_path = [[item['image_path']] * item['big5'].shape[0] for item in batch]
    
    # Concatenating other data
    aesthetic_scores_concatenated = torch.cat(aesthetic_scores)
    attribute_concatenated = torch.cat(attributes)
    traits_histograms_concatenated = torch.cat(traits_histograms)
    onehot_big5s_concatenated = torch.cat(onehot_big5s)
    
    return {
        'image': images_stacked,
        'image_path':img_path,
        'userId':userId,
        'aestheticScore': aesthetic_scores_concatenated,
        'attributes': attribute_concatenated,
        'big5': traits_histograms_concatenated,
        'traits': onehot_big5s_concatenated
    }


def load_data(args, root_dir = datapath['PARA_datapath']):
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
    trainset = getattr(args, 'trainset', 'GIAA')

    # Load datasets
    # Create datasets with the appropriate transformations
    dataset = PARA_PIAADataset(root_dir, transform=train_transform)
    piaa_dataset = PARA_PIAADataset(root_dir, transform=train_transform)
    train_dataset, val_dataset, test_dataset = split_dataset_by_images(piaa_dataset, root_dir)
    # Assuming shell_users_df contains the shell user DataFrame
    if getattr(args, 'use_cv', False) and (fold_id is not None) and (n_fold is not None):
        train_dataset, val_dataset, test_dataset = create_user_split_dataset_kfold(dataset, train_dataset, val_dataset, test_dataset, fold_id=fold_id, n_fold=n_fold)
    
    is_trait_specific = getattr(args, 'trait', False) and getattr(args, 'value', False)
    is_disjoint_trait = getattr(args, 'trait_disjoint', True)
    if is_trait_specific:
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

    pkl_dir = './dataset_pkl'
    if getattr(args, 'use_cv', False):
        pkl_dir = os.path.join(pkl_dir, 'user_cv')
        ensure_dir_exists(pkl_dir)
        train_mapfile = os.path.join(pkl_dir,'trainset_image_dct_%dfold.pkl'%fold_id)
        if trainset == 'GIAA':
            precompute_file = os.path.join(pkl_dir,'trainset_GIAA_dct_%dfold.pkl'%fold_id)
            train_dataset = PARA_GIAA_HistogramDataset(root_dir, transform=train_transform, 
                data=train_dataset.data, map_file=train_mapfile, 
                precompute_file=precompute_file)
        elif trainset == 'sGIAA':
            importance_sampling = args.importance_sampling
            precompute_file = 'trainset_MIAA_dct_%dfold_IS.pkl'%fold_id if importance_sampling else 'trainset_MIAA_dct_%dfold.pkl'%fold_id
            train_dataset = PARA_sGIAA_HistogramDataset(root_dir, transform=train_transform, importance_sampling=importance_sampling,
                data=train_dataset.data, map_file=train_mapfile, 
                precompute_file=os.path.join(pkl_dir,precompute_file))
        else:
            train_dataset = PARA_PIAA_HistogramDataset(root_dir, transform=train_transform, data=train_dataset.data)

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
        train_mapfile = os.path.join(pkl_dir,'trainset_image_dct_%s.pkl'%suffix)
        if trainset == 'GIAA':
            precompute_file = os.path.join(pkl_dir,'trainset_GIAA_dct_%s.pkl'%suffix)
            train_dataset = PARA_GIAA_HistogramDataset(root_dir, transform=train_transform, data=train_dataset.data, map_file=train_mapfile, precompute_file=precompute_file)
        elif trainset == 'sGIAA':
            importance_sampling = args.importance_sampling
            precompute_file = 'trainset_MIAA_nopiaa_dct_IS_%s.pkl'%suffix if importance_sampling else 'trainset_MIAA_nopiaa_dct_%s.pkl'%suffix
            train_dataset = PARA_sGIAA_HistogramDataset(root_dir, transform=train_transform, data=train_dataset.data, map_file=train_mapfile, precompute_file=os.path.join(pkl_dir,precompute_file))
        else:
            train_dataset = PARA_PIAA_HistogramDataset(root_dir, transform=train_transform, data=train_dataset.data)
        
        val_mapfile=os.path.join(pkl_dir,'valset_image_dct_%s.pkl'%suffix)
        val_precompute_file=os.path.join(pkl_dir,'valset_GIAA_dct_%s.pkl'%suffix)
        test_mapfile=os.path.join(pkl_dir,'testset_image_dct_%s.pkl'%suffix)
        test_precompute_file=os.path.join(pkl_dir,'testset_GIAA_dct_%s.pkl'%suffix)

    else:
        ensure_dir_exists(pkl_dir)
        train_mapfile = os.path.join(pkl_dir,'trainset_image_dct.pkl')
        if trainset == 'GIAA':
            precompute_file = os.path.join(pkl_dir,'trainset_GIAA_dct.pkl')
            train_dataset = PARA_GIAA_HistogramDataset(root_dir, transform=train_transform, data=train_dataset.data, map_file=train_mapfile, precompute_file=precompute_file)
        elif trainset == 'sGIAA':
            importance_sampling = args.importance_sampling
            precompute_file = 'trainset_MIAA_nopiaa_dct_IS.pkl' if importance_sampling else 'trainset_MIAA_nopiaa_dct.pkl'
            train_dataset = PARA_sGIAA_HistogramDataset(root_dir, transform=train_transform, data=train_dataset.data, map_file=train_mapfile, precompute_file=os.path.join(pkl_dir,precompute_file))
        else:
            train_dataset = PARA_PIAA_HistogramDataset(root_dir, transform=train_transform, data=train_dataset.data)
        
        val_mapfile=os.path.join(pkl_dir,'valset_image_dct.pkl')
        val_precompute_file=os.path.join(pkl_dir,'valset_GIAA_dct.pkl')
        test_mapfile=os.path.join(pkl_dir,'testset_image_dct.pkl')
        test_precompute_file=os.path.join(pkl_dir,'testset_GIAA_dct.pkl')

    # test_user_piaa_dataset = PARA_PIAA_HistogramDataset(root_dir, transform=test_transform, data=test_user_piaa_dataset.data)
    val_giaa_dataset = PARA_GIAA_HistogramDataset(root_dir, transform=test_transform, data=val_dataset.data, map_file=val_mapfile, precompute_file=val_precompute_file)
    val_piaa_imgsort_dataset = PARA_PIAA_HistogramDataset_imgsort(root_dir, transform=test_transform, data=val_dataset.data, map_file=val_mapfile)
    test_giaa_dataset = PARA_GIAA_HistogramDataset(root_dir, transform=test_transform, data=test_dataset.data, map_file=test_mapfile, precompute_file=test_precompute_file)
    test_piaa_imgsort_dataset = PARA_PIAA_HistogramDataset_imgsort(root_dir, transform=test_transform, data=test_dataset.data, map_file=test_mapfile)
    
    return train_dataset, val_giaa_dataset, val_piaa_imgsort_dataset, test_giaa_dataset, test_piaa_imgsort_dataset


def load_data_testpair(args, root_dir = datapath['PARA_datapath']):
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
    trainset = getattr(args, 'trainset', 'GIAA')

    # Load datasets
    # Create datasets with the appropriate transformations
    dataset = PARA_PIAADataset(root_dir, transform=train_transform)
    piaa_dataset = PARA_PIAADataset(root_dir, transform=train_transform)
    train_dataset, val_dataset, test_dataset = split_dataset_by_images(piaa_dataset, root_dir)
    # Assuming shell_users_df contains the shell user DataFrame
    if getattr(args, 'use_cv', False) and (fold_id is not None) and (n_fold is not None):
        train_dataset, val_dataset, test_dataset = create_user_split_dataset_kfold(dataset, train_dataset, val_dataset, test_dataset, fold_id=fold_id, n_fold=n_fold)
    
    is_trait_disjoint = getattr(args, 'trait', False) and getattr(args, 'value', False)
    if is_trait_disjoint:
        print(f'Split trait according to {args.trait} == {args.value}')
        train_dataset.data = train_dataset.data[train_dataset.data[args.trait] != args.value]
        val_dataset.data = val_dataset.data[val_dataset.data[args.trait] != args.value]
        testc_dataset = copy.deepcopy(test_dataset)
        test_dataset.data = test_dataset.data[test_dataset.data[args.trait] == args.value]
        testc_dataset.data = testc_dataset.data[testc_dataset.data[args.trait] != args.value]

    print(len(train_dataset), len(val_dataset), len(test_dataset), len(testc_dataset))

    pkl_dir = './dataset_pkl'
    if getattr(args, 'use_cv', False):
        pkl_dir = os.path.join(pkl_dir, 'user_cv')
        train_mapfile = os.path.join(pkl_dir,'trainset_image_dct_%dfold.pkl'%fold_id)
        if trainset == 'GIAA':
            precompute_file = os.path.join(pkl_dir,'trainset_GIAA_dct_%dfold.pkl'%fold_id)
            train_dataset = PARA_GIAA_HistogramDataset(root_dir, transform=train_transform, 
                data=train_dataset.data, map_file=train_mapfile, 
                precompute_file=precompute_file)
        elif trainset == 'sGIAA':
            importance_sampling = args.importance_sampling
            precompute_file = 'trainset_MIAA_dct_%dfold_IS.pkl'%fold_id if importance_sampling else 'trainset_MIAA_dct_%dfold.pkl'%fold_id
            train_dataset = PARA_sGIAA_HistogramDataset(root_dir, transform=train_transform, importance_sampling=importance_sampling,
                data=train_dataset.data, map_file=train_mapfile, 
                precompute_file=os.path.join(pkl_dir,precompute_file))
        else:
            train_dataset = PARA_PIAA_HistogramDataset(root_dir, transform=train_transform, data=train_dataset.data)

        val_mapfile = os.path.join(pkl_dir,'valset_image_dct_%dfold.pkl'%fold_id)
        val_precompute_file = os.path.join(pkl_dir,'valset_GIAA_dct_%dfold.pkl'%fold_id)
        test_mapfile = os.path.join(pkl_dir,'testset_image_dct_%dfold.pkl'%fold_id) 
        test_precompute_file = os.path.join(pkl_dir,'testset_GIAA_dct_%dfold.pkl'%fold_id)

    elif is_trait_disjoint:
        pkl_dir = os.path.join(pkl_dir, 'trait_split')
        suffix = '%s_%s'%(args.trait, args.value)
        train_mapfile = os.path.join(pkl_dir,'trainset_image_dct_%s.pkl'%suffix)
        if trainset == 'GIAA':
            precompute_file = os.path.join(pkl_dir,'trainset_GIAA_dct_%s.pkl'%suffix)
            train_dataset = PARA_GIAA_HistogramDataset(root_dir, transform=train_transform, data=train_dataset.data, map_file=train_mapfile, precompute_file=precompute_file)
        elif trainset == 'sGIAA':
            importance_sampling = args.importance_sampling
            precompute_file = 'trainset_MIAA_nopiaa_dct_IS_%s.pkl'%suffix if importance_sampling else 'trainset_MIAA_nopiaa_dct_%s.pkl'%suffix
            train_dataset = PARA_sGIAA_HistogramDataset(root_dir, transform=train_transform, data=train_dataset.data, map_file=train_mapfile, precompute_file=os.path.join(pkl_dir,precompute_file))
        else:
            train_dataset = PARA_PIAA_HistogramDataset(root_dir, transform=train_transform, data=train_dataset.data)
        
        val_mapfile=os.path.join(pkl_dir,'valset_image_dct_%s.pkl'%suffix)
        val_precompute_file=os.path.join(pkl_dir,'valset_GIAA_dct_%s.pkl'%suffix)
        test_mapfile=os.path.join(pkl_dir,'testset_image_dct_%s.pkl'%suffix)
        test_precompute_file=os.path.join(pkl_dir,'testset_GIAA_dct_%s.pkl'%suffix)

        testc_mapfile=os.path.join(pkl_dir,'testset_c_image_dct_%s.pkl'%suffix)
        testc_precompute_file=os.path.join(pkl_dir,'testset_c_GIAA_dct_%s.pkl'%suffix)

    else:
        train_mapfile = os.path.join(pkl_dir,'trainset_image_dct.pkl')
        if trainset == 'GIAA':
            precompute_file = os.path.join(pkl_dir,'trainset_GIAA_dct.pkl')
            train_dataset = PARA_GIAA_HistogramDataset(root_dir, transform=train_transform, data=train_dataset.data, map_file=train_mapfile, precompute_file=precompute_file)
        elif trainset == 'sGIAA':
            importance_sampling = args.importance_sampling
            precompute_file = 'trainset_MIAA_nopiaa_dct_IS.pkl' if importance_sampling else 'trainset_MIAA_nopiaa_dct.pkl'
            train_dataset = PARA_sGIAA_HistogramDataset(root_dir, transform=train_transform, data=train_dataset.data, map_file=train_mapfile, precompute_file=os.path.join(pkl_dir,precompute_file))
        else:
            train_dataset = PARA_PIAA_HistogramDataset(root_dir, transform=train_transform, data=train_dataset.data)
        
        val_mapfile=os.path.join(pkl_dir,'valset_image_dct.pkl')
        val_precompute_file=os.path.join(pkl_dir,'valset_GIAA_dct.pkl')
        test_mapfile=os.path.join(pkl_dir,'testset_image_dct.pkl')
        test_precompute_file=os.path.join(pkl_dir,'testset_GIAA_dct.pkl')

    # test_user_piaa_dataset = PARA_PIAA_HistogramDataset(root_dir, transform=test_transform, data=test_user_piaa_dataset.data)
    val_giaa_dataset = PARA_GIAA_HistogramDataset(root_dir, transform=test_transform, data=val_dataset.data, map_file=val_mapfile, precompute_file=val_precompute_file)
    val_piaa_imgsort_dataset = PARA_PIAA_HistogramDataset_imgsort(root_dir, transform=test_transform, data=val_dataset.data, map_file=val_mapfile)
    test_giaa_dataset = PARA_GIAA_HistogramDataset(root_dir, transform=test_transform, data=test_dataset.data, map_file=test_mapfile, precompute_file=test_precompute_file)
    testc_giaa_dataset = PARA_GIAA_HistogramDataset(root_dir, transform=test_transform, data=testc_dataset.data, map_file=testc_mapfile, precompute_file=testc_precompute_file)
    
    test_piaa_imgsort_dataset = PARA_PIAA_HistogramDataset_imgsort(root_dir, transform=test_transform, data=test_dataset.data, map_file=test_mapfile)
    return train_dataset, val_giaa_dataset, val_piaa_imgsort_dataset, test_giaa_dataset, test_piaa_imgsort_dataset, testc_giaa_dataset


def load_testdata(args, root_dir = datapath['PARA_datapath']):
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
    trainset = getattr(args, 'trainset', 'GIAA')

    # Load datasets
    # Create datasets with the appropriate transformations
    dataset = PARA_PIAADataset(root_dir, transform=train_transform)
    piaa_dataset = PARA_PIAADataset(root_dir, transform=train_transform)
    train_dataset, val_dataset, test_dataset = split_dataset_by_images(piaa_dataset, root_dir)
    # Assuming shell_users_df contains the shell user DataFrame
    if getattr(args, 'use_cv', False) and (fold_id is not None) and (n_fold is not None):
        train_dataset, val_dataset, test_dataset = create_user_split_dataset_kfold(dataset, train_dataset, val_dataset, test_dataset, fold_id=fold_id, n_fold=n_fold)
    
    is_trait_specific = getattr(args, 'trait', False) and getattr(args, 'value', False)
    is_disjoint_trait = getattr(args, 'trait_disjoint', True)
    if is_trait_specific:
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
    
    pkl_dir = './dataset_pkl'
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

    test_giaa_dataset = PARA_GIAA_HistogramDataset(root_dir, transform=test_transform, data=test_dataset.data, map_file=test_mapfile, precompute_file=test_precompute_file)
    test_piaa_imgsort_dataset = PARA_PIAA_HistogramDataset_imgsort(root_dir, transform=test_transform, data=test_dataset.data, map_file=test_mapfile)
    
    return test_giaa_dataset, test_piaa_imgsort_dataset


def extract_features(model, dataloader, device):
    model.eval()  # Set the model to evaluation mode
    feature_dict = {}

    with torch.no_grad():
        for sample in tqdm(dataloader, leave=False):
            images = sample['image'].to(device)
            img_paths = sample['image_path']  # Assuming the dataset provides image paths
            
            # Pass the images through the model up to the avgpool layer
            features = model(images)
            features = torch.flatten(features, start_dim=1)  # Flatten the output of avgpool
            
            # Store the features in the dictionary with image path as the key
            for idx, img_path in enumerate(img_paths):
                # feature_dict[img_path[0]] = features[idx].cpu().numpy()
                feature_dict[img_path] = features[idx].cpu().numpy()

    return feature_dict


if __name__ == '__main__':
    from utils.argflags import parse_arguments, parse_arguments_piaa
    import torch.nn as nn
    from torchvision.models import resnet50


    args = parse_arguments()
    train_dataset, val_giaa_dataset, val_piaa_imgsort_dataset, test_giaa_dataset, test_piaa_imgsort_dataset = load_data(args, datapath['PARA_datapath'])
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataloader = DataLoader(val_giaa_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_dataloader = DataLoader(test_giaa_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_piaa_imgsort_dataloader = DataLoader(test_piaa_imgsort_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, timeout=300, collate_fn=collate_fn_imgsort)
    # Iterate over the training dataloader
    # for sample in tqdm(test_piaa_imgsort_dataloader):
        # Perform training operations here
        # [print(k, v.shape) for k, v in sample.items()]
        # raise Exception
    
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
    with open('train_extracted_features.pkl', 'wb') as f:
        pickle.dump(train_feature_dict, f)
    with open('val_extracted_features.pkl', 'wb') as f:
        pickle.dump(val_feature_dict, f)
    with open('test_extracted_features.pkl', 'wb') as f:
        pickle.dump(test_feature_dict, f)

