import os
import torch
from torchvision import transforms
import torch.nn.functional as F
from PARA_PIAA_dataloader import PARA_PIAADataset, split_dataset_by_user, split_dataset_by_images, split_data_by_user
from torch.utils.data import DataLoader
import random
import pickle
from tqdm import tqdm
import copy
from scipy.optimize import minimize, Bounds
from time import time


class PARA_HistogramDataset(PARA_PIAADataset):
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
    
    def __getitem__(self, idx, max_sample=30):
        # Determine the number of samples to draw (d)
        associated_indices = self.image_to_indices_map[self.unique_images[idx]]
        n_sample = random.randint(1, min(max_sample, len(associated_indices)))
        sampled_indices = random.sample(associated_indices, n_sample)

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
            'traits': {
                'personality-E': torch.zeros(10),
                'personality-A': torch.zeros(10),
                'personality-N': torch.zeros(10),
                'personality-O': torch.zeros(10),
                'personality-C': torch.zeros(10)
            },
            'onehot_traits': {
                'age': torch.zeros(len(self.age_encoder)),
                'gender': torch.zeros(len(self.gender_encoder)),
                'EducationalLevel': torch.zeros(len(self.education_encoder)),
                'artExperience': torch.zeros(len(self.art_experience_encoder)),
                'photographyExperience': torch.zeros(len(self.photo_experience_encoder))
            }
        }

        img_sample = super().__getitem__(sampled_indices[0], use_image=True)
        for random_idx in sampled_indices:
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
            for trait, _ in accumulated_histogram['traits'].items():
                bin_idx = int(sample['userTraits'][trait] - 1)
                accumulated_histogram['traits'][trait][bin_idx] += 1
            
            # Update onehot traits
            for trait in accumulated_histogram['onehot_traits'].keys():
                accumulated_histogram['onehot_traits'][trait] += sample['userTraits'][trait]

        # Average out histograms over the number of samples
        accumulated_histogram['aestheticScore'] /= n_sample
        for trait in accumulated_histogram['onehot_traits'].keys():
            accumulated_histogram['onehot_traits'][trait] /= n_sample
        for key in accumulated_histogram['attributes'].keys():
            accumulated_histogram['attributes'][key] /= n_sample
        for key in accumulated_histogram['traits'].keys():
            accumulated_histogram['traits'][key] /= n_sample

        accumulated_histogram['n_samples'] = n_sample
        accumulated_histogram['image'] = img_sample['image']

        # Convert histograms to stacked tensors
        stacked_attributes = torch.cat(list(accumulated_histogram['attributes'].values()))
        stacked_traits = torch.cat(list(accumulated_histogram['traits'].values()))
        stacked_onehot_traits = torch.cat(list(accumulated_histogram['onehot_traits'].values()))

        # Replace dictionaries with stacked tensors
        accumulated_histogram['attributes'] = stacked_attributes
        accumulated_histogram['traits'] = stacked_traits
        accumulated_histogram['onehot_traits'] = stacked_onehot_traits

        return accumulated_histogram


class PARA_MIAA_HistogramDataset(PARA_PIAADataset):
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
        for idx in tqdm(range(len(self))):
            self.precomputed_data.append(self._compute_miaa_item(idx))

    def _compute_miaa_item(self, idx, num_samples=40):
        accumulated_histograms = []
        for i in range(num_samples):
            if i > 0:
                accumulated_histograms.append(self._compute_item(idx, is_giaa=False))
            else:
                accumulated_histograms.append(self._compute_item(idx, is_giaa=True))
        return accumulated_histograms

    def augment_entire_dataset(self, num_trial=400):
        tot_count = 0
        for idx, histograms in enumerate(tqdm(self.precomputed_data, desc="Augmenting Dataset")):
            count = 0
            augment_list = []
            for _ in range(num_trial):
                augmented_histogram = self.augment_data(histograms)
                if augmented_histogram:
                    # Extend the original histograms with the augmented one
                    augment_list.append(augmented_histogram)
                    count += 1
            if count > 0:
                self.precomputed_data[idx].extend(augment_list)        
                tot_count += count
        print('augment %d data'%tot_count)

    def _compute_item(self, idx, is_giaa=True):
        associated_indices = self.image_to_indices_map[self.unique_images[idx]]
        if not is_giaa:
            # n_sample = random.randint(1, len(associated_indices)-1)
            n_sample = random.randint(2, len(associated_indices)-1)
            associated_indices = random.sample(associated_indices, n_sample)
        
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
            'traits': {
                'personality-E': torch.zeros(10),
                'personality-A': torch.zeros(10),
                'personality-N': torch.zeros(10),
                'personality-O': torch.zeros(10),
                'personality-C': torch.zeros(10)
            },
            'onehot_traits': {
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
            for trait, _ in accumulated_histogram['traits'].items():
                bin_idx = int(sample['userTraits'][trait] - 1)
                accumulated_histogram['traits'][trait][bin_idx] += 1
            
            # Update onehot traits
            for trait in accumulated_histogram['onehot_traits'].keys():
                accumulated_histogram['onehot_traits'][trait] += sample['userTraits'][trait]

        # Average out histograms over the number of samples
        total_samples = len(associated_indices)
        accumulated_histogram['aestheticScore'] /= total_samples
        for trait in accumulated_histogram['onehot_traits'].keys():
            accumulated_histogram['onehot_traits'][trait] /= total_samples
        for key in accumulated_histogram['attributes'].keys():
            accumulated_histogram['attributes'][key] /= total_samples
        for key in accumulated_histogram['traits'].keys():
            accumulated_histogram['traits'][key] /= total_samples

        accumulated_histogram['n_samples'] = total_samples
        # accumulated_histogram['image'] = img_sample['image']

        # Convert histograms to stacked tensors
        stacked_attributes = torch.cat(list(accumulated_histogram['attributes'].values()))
        stacked_traits = torch.cat(list(accumulated_histogram['traits'].values()))
        stacked_onehot_traits = torch.cat(list(accumulated_histogram['onehot_traits'].values()))

        # Replace dictionaries with stacked tensors
        accumulated_histogram['attributes'] = stacked_attributes
        accumulated_histogram['traits'] = stacked_traits
        accumulated_histogram['onehot_traits'] = stacked_onehot_traits

        return accumulated_histogram

    def __getitem__(self, idx):
        histograms = self.precomputed_data[idx]
        selected_histogram = copy.deepcopy(random.choice(histograms))
        img_sample = super().__getitem__(self.image_to_indices_map[self.unique_images[idx]][0], use_image=True)
        selected_histogram['image'] = img_sample['image']
        selected_histogram['semantic'] = img_sample['imageAttributes']['semantic']
        return selected_histogram
    
    def augment_data(self, histograms):
        # Randomly select two histograms from the provided list
        x0, x1 = random.sample(histograms, 2)

        # Define lengths of each component in traits and onehot_traits
        len_traits = {'personality-E': 10, 'personality-A': 10, 'personality-N': 10, 'personality-O': 10, 'personality-C': 10}
        len_onehot_traits = {'age': len(self.age_encoder), 'gender': len(self.gender_encoder), 'EducationalLevel': len(self.education_encoder), 
                             'artExperience': len(self.art_experience_encoder), 'photographyExperience': len(self.photo_experience_encoder)}

        # Function to split concatenated tensor into its components
        def split_tensor(tensor, lengths):
            split_tensors = {}
            start = 0
            for key, length in lengths.items():
                split_tensors[key] = tensor[start:start + length]
                start += length
            return split_tensors

        # Define the objective function for minimization
        def objective_function(c1):
            c0 = 1
            c1_tensor = torch.tensor(c1, dtype=torch.float32)  # Convert c1 to a PyTorch tensor

            d = {key: (c0 * x0[key] - c1_tensor * x1[key]) / (c0 - c1_tensor) for key in ['aestheticScore', 'traits', 'onehot_traits', 'attributes']}
            std_components = []
            std_components.append(torch.std(d['aestheticScore']))

            d_traits = split_tensor(d['traits'], len_traits)
            d_onehot_traits = split_tensor(d['onehot_traits'], len_onehot_traits)
            
            for component in d_traits.values():
                std_components.append(torch.std(component))
            for component in d_onehot_traits.values():
                std_components.append(torch.std(component))
            return sum(std_components).item()  # Ensure to return a Python scalar

        # Initial standard deviation without augmentation
        initial_std = objective_function(0)

        # Find coefficient c1 that minimizes the objective function
        result = minimize(objective_function, [random.random()-0.5], bounds=Bounds(-1, 0.9))

        # Check if the optimization was successful and actually reduces the standard deviation
        if result.success and result.fun < initial_std:
            c1 = result.x[0]
            selected_histogram = {key: (x0[key] - c1 * x1[key]) / (1 - c1) for key in ['aestheticScore', 'traits', 'onehot_traits', 'attributes']}
            for key in selected_histogram:
                selected_histogram[key] = torch.clamp(selected_histogram[key], min=0)

            # Split traits and onehot_traits back into their components
            selected_histogram['traits'] = split_tensor(selected_histogram['traits'], len_traits)
            selected_histogram['onehot_traits'] = split_tensor(selected_histogram['onehot_traits'], len_onehot_traits)
            return selected_histogram
        else:
            return None

    def augment_and_save_dataset(self, save_file):
        self.augment_entire_dataset()
        self.save(save_file)
        print(f"Augmented data saved to {save_file}")

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
        for idx in tqdm(range(len(self))):
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
            'traits': {
                'personality-E': torch.zeros(10),
                'personality-A': torch.zeros(10),
                'personality-N': torch.zeros(10),
                'personality-O': torch.zeros(10),
                'personality-C': torch.zeros(10)
            },
            'onehot_traits': {
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
            for trait, _ in accumulated_histogram['traits'].items():
                bin_idx = int(sample['userTraits'][trait] - 1)
                accumulated_histogram['traits'][trait][bin_idx] += 1
            
            # Update onehot traits
            for trait in accumulated_histogram['onehot_traits'].keys():
                accumulated_histogram['onehot_traits'][trait] += sample['userTraits'][trait]

        # Average out histograms over the number of samples
        total_samples = len(associated_indices)
        accumulated_histogram['aestheticScore'] /= total_samples
        for trait in accumulated_histogram['onehot_traits'].keys():
            accumulated_histogram['onehot_traits'][trait] /= total_samples
        for key in accumulated_histogram['attributes'].keys():
            accumulated_histogram['attributes'][key] /= total_samples
        for key in accumulated_histogram['traits'].keys():
            accumulated_histogram['traits'][key] /= total_samples

        accumulated_histogram['n_samples'] = total_samples
        # accumulated_histogram['image'] = img_sample['image']

        # Convert histograms to stacked tensors
        stacked_attributes = torch.cat(list(accumulated_histogram['attributes'].values()))
        stacked_traits = torch.cat(list(accumulated_histogram['traits'].values()))
        stacked_onehot_traits = torch.cat(list(accumulated_histogram['onehot_traits'].values()))

        # Replace dictionaries with stacked tensors
        accumulated_histogram['attributes'] = stacked_attributes
        accumulated_histogram['traits'] = stacked_traits
        accumulated_histogram['onehot_traits'] = stacked_onehot_traits

        return accumulated_histogram

    def __getitem__(self, idx):
        item_data = copy.deepcopy(self.precomputed_data[idx])
        img_sample = super().__getitem__(self.image_to_indices_map[self.unique_images[idx]][0], use_image=True)
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
            'traits': {
                'personality-E': torch.zeros(10),
                'personality-A': torch.zeros(10),
                'personality-N': torch.zeros(10),
                'personality-O': torch.zeros(10),
                'personality-C': torch.zeros(10)
            },
            'onehot_traits': {
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
        for trait, _ in accumulated_histogram['traits'].items():
            bin_idx = int(sample['userTraits'][trait] - 1)
            accumulated_histogram['traits'][trait][bin_idx] += 1
        
        # Update onehot traits
        for trait in accumulated_histogram['onehot_traits'].keys():
            accumulated_histogram['onehot_traits'][trait] += sample['userTraits'][trait]

        accumulated_histogram['n_samples'] = 1
        accumulated_histogram['image'] = sample['image']

        # Convert histograms to stacked tensors
        stacked_attributes = torch.cat(list(accumulated_histogram['attributes'].values()))
        stacked_traits = torch.cat(list(accumulated_histogram['traits'].values()))
        stacked_onehot_traits = torch.cat(list(accumulated_histogram['onehot_traits'].values()))

        # Replace dictionaries with stacked tensors
        accumulated_histogram['attributes'] = stacked_attributes
        accumulated_histogram['traits'] = stacked_traits
        accumulated_histogram['onehot_traits'] = stacked_onehot_traits

        return accumulated_histogram


class PARA_GSP_HistogramDataset(PARA_PIAA_HistogramDataset):
    def __init__(self, root_dir, transform=None, piaa_data=None, 
                 giaa_data=None, map_file=None, precompute_file=None):
        super().__init__(root_dir, transform, piaa_data)
        
        # Store the GIAA dataset as a member attribute
        self.giaa_dataset = PARA_GIAA_HistogramDataset(
            root_dir, transform, giaa_data, map_file, precompute_file)

        # Build a mapping from PIAA indices to GIAA indices
        self.piaa_to_giaa_index_map = self._build_index_map()
    
    def _build_index_map(self):
        piaa_to_giaa_index_map = {}
        print(len(self.data['imageName']))
        for idx, piaa_image_name in enumerate(self.data['imageName']):
            # Find the corresponding index in GIAA dataset
            giaa_indices = [i for i, img in enumerate(self.giaa_dataset.unique_images) if img == piaa_image_name]
            if giaa_indices:
                piaa_to_giaa_index_map[idx] = giaa_indices[0]  # Assuming one-to-one mapping for simplicity
        return piaa_to_giaa_index_map

    def __getitem__(self, idx):
        # First get the PIAA histogram data
        piaa_sample = super().__getitem__(idx)

        # Now, get the corresponding GIAA data
        giaa_idx = self.piaa_to_giaa_index_map.get(idx)
        giaa_sample = self.giaa_dataset[giaa_idx]
        # Return the combined sample data
        
        gsp_sample = copy.deepcopy(giaa_sample)
        n_samples = giaa_sample['n_samples']
        gsp_sample['n_samples'] -= 1
        for key in ['traits', 'onehot_traits', 'attributes', 'aestheticScore']:
            gsp_sample[key] = (n_samples * gsp_sample[key] - piaa_sample[key]) / (n_samples - 1)
        return piaa_sample, gsp_sample, giaa_sample

    def __len__(self):
        # The length is the same as that of the PIAA dataset
        return len(self.data)



class PARA_PIAA_HistogramDataset_Precomputed(PARA_PIAADataset):
    def __init__(self, root_dir, transform=None, data=None, save_file=None):
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
            'traits': {
                'personality-E': torch.zeros(10),
                'personality-A': torch.zeros(10),
                'personality-N': torch.zeros(10),
                'personality-O': torch.zeros(10),
                'personality-C': torch.zeros(10)
            },
            'onehot_traits': {
                'age': torch.zeros(len(self.age_encoder)),
                'gender': torch.zeros(len(self.gender_encoder)),
                'EducationalLevel': torch.zeros(len(self.education_encoder)),
                'artExperience': torch.zeros(len(self.art_experience_encoder)),
                'photographyExperience': torch.zeros(len(self.photo_experience_encoder))
            }
        }

        self.save_file = save_file
        if self.save_file:
            if os.path.exists(self.save_file):
                # Load precomputed histograms if the file exists
                with open(self.save_file, 'rb') as f:
                    self.precomputed_histograms = pickle.load(f)
            else:
                # Precompute and save the histograms
                self.precompute_histograms()
        else:
            # Precompute histograms without saving
            self.precomputed_histograms = [self.__getitem__(i) for i in tqdm(range(len(self)))]

    def _discretize(self, value, bins):
        """Helper function to discretize a value given specific bins"""
        for i, bin_value in enumerate(bins):
            if value <= bin_value:
                return i
        return len(bins) - 1

    def __getitem__(self, idx):
        if hasattr(self, 'precomputed_histograms'):
            return self.precomputed_histograms[idx]
        
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
        for trait, _ in accumulated_histogram['traits'].items():
            bin_idx = int(sample['userTraits'][trait] - 1)
            accumulated_histogram['traits'][trait][bin_idx] += 1

        # Update onehot traits
        for trait in accumulated_histogram['onehot_traits'].keys():
            accumulated_histogram['onehot_traits'][trait] += sample['userTraits'][trait]

        accumulated_histogram['n_samples'] = 1
        accumulated_histogram['image'] = sample['image']

        # Convert histograms to stacked tensors
        stacked_attributes = torch.cat(list(accumulated_histogram['attributes'].values()))
        stacked_traits = torch.cat(list(accumulated_histogram['traits'].values()))
        stacked_onehot_traits = torch.cat(list(accumulated_histogram['onehot_traits'].values()))

        # Replace dictionaries with stacked tensors
        accumulated_histogram['attributes'] = stacked_attributes
        accumulated_histogram['traits'] = stacked_traits
        accumulated_histogram['onehot_traits'] = stacked_onehot_traits

        return accumulated_histogram

    def precompute_histograms(self):
        self.precomputed_histograms = [self.__getitem__(i) for i in tqdm(range(len(self)))]
        if self.save_file:
            with open(self.save_file, 'wb') as f:
                pickle.dump(self.precomputed_histograms, f)



def load_usersplit_data(root_dir = '/home/lwchen/datasets/PARA/', miaa=True):
    # Define transformations for training set and test set
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
    ])

    # Set the random seed for reproducibility in the test set
    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    # Create datasets with the appropriate transformations
    train_piaa_dataset = PARA_PIAADataset(root_dir, transform=train_transform)
    test_piaa_dataset = PARA_PIAADataset(root_dir, transform=train_transform)
    train_users, test_users = split_data_by_user(train_piaa_dataset.data, test_count=40, seed=42)
    # Filter data by user IDs
    train_piaa_dataset.data = train_piaa_dataset.data[train_piaa_dataset.data['userId'].isin(train_users)]
    test_piaa_dataset.data = test_piaa_dataset.data[test_piaa_dataset.data['userId'].isin(test_users)]
    train_piaa_dataset, test_piaa_dataset = split_dataset_by_images(train_piaa_dataset, test_piaa_dataset, root_dir)
    print(len(train_piaa_dataset), len(test_piaa_dataset))

    """Precompute"""
    pkl_dir = './dataset_pkl'
    # train_dataset = PARA_HistogramDataset(root_dir, transform=train_transform, data=train_piaa_dataset.data, map_file='trainset_image_dct.pkl')
    # train_dataset = PARA_MIAA_HistogramDataset(root_dir, transform=train_transform, data=train_piaa_dataset.data, map_file=os.path.join(pkl_dir,'trainset_image_dct.pkl'), precompute_file=os.path.join(pkl_dir,'trainset_MIAA_nopiaa_dct.pkl'))
    # train_dataset = PARA_GIAA_HistogramDataset(root_dir, transform=train_transform, data=train_piaa_dataset.data, map_file=os.path.join(pkl_dir,'trainset_image_dct.pkl'), precompute_file=os.path.join(pkl_dir,'trainset_GIAA_dct.pkl'))
    test_dataset = PARA_GIAA_HistogramDataset(root_dir, transform=test_transform, data=test_piaa_dataset.data, map_file=os.path.join(pkl_dir,'testset_image_testuser_dct.pkl'), precompute_file=os.path.join(pkl_dir,'testset_GIAA_testuser_dct.pkl'))
    
    train_user_piaa_dataset = PARA_PIAA_HistogramDataset(root_dir, transform=test_transform, data=train_piaa_dataset.data)
    test_user_piaa_dataset = PARA_PIAA_HistogramDataset(root_dir, transform=test_transform, data=test_piaa_dataset.data)
    if miaa:
        train_dataset = PARA_MIAA_HistogramDataset(root_dir, transform=train_transform, data=train_piaa_dataset.data, map_file=os.path.join(pkl_dir,'trainset_image_trainuser_dct.pkl'), precompute_file=os.path.join(pkl_dir,'trainset_MIAA_nopiaa_trainuser_dct.pkl'))
    else:
        train_dataset = PARA_GIAA_HistogramDataset(root_dir, transform=train_transform, data=train_piaa_dataset.data, map_file=os.path.join(pkl_dir,'trainset_image_trainuser_dct.pkl'), precompute_file=os.path.join(pkl_dir,'trainset_GIAA_trainuser_dct.pkl'))
    return train_dataset, test_dataset, train_user_piaa_dataset, test_user_piaa_dataset


if __name__ == '__main__':
    # Usage example:
    root_dir = '/home/lwchen/datasets/PARA/'
    
    # Define transformations for training set and test set
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
    ])

    # Set the random seed for reproducibility in the test set
    random_seed = 42
    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    
    # Create datasets with the appropriate transformations
    train_piaa_dataset = PARA_PIAADataset(root_dir, transform=train_transform)
    test_piaa_dataset = PARA_PIAADataset(root_dir, transform=train_transform)
    # train_piaa_dataset, test_piaa_dataset = split_dataset_by_user(train_piaa_dataset, test_piaa_dataset, test_count=40, max_annotations_per_user=10, seed=random_seed)

    train_users, test_users = split_data_by_user(train_piaa_dataset.data, test_count=40, seed=42)
    # Filter data by user IDs
    train_piaa_dataset.data = train_piaa_dataset.data[train_piaa_dataset.data['userId'].isin(train_users)]
    test_piaa_dataset.data = test_piaa_dataset.data[test_piaa_dataset.data['userId'].isin(test_users)]
    train_piaa_dataset, test_piaa_dataset = split_dataset_by_images(train_piaa_dataset, test_piaa_dataset, root_dir)
    print(len(train_piaa_dataset), len(test_piaa_dataset))

    """Precompute"""
    pkl_dir = './dataset_pkl'
    # train_dataset = PARA_HistogramDataset(root_dir, transform=train_transform, data=train_piaa_dataset.data, map_file='trainset_image_dct.pkl')
    # train_dataset = PARA_MIAA_HistogramDataset(root_dir, transform=train_transform, data=train_piaa_dataset.data, map_file=os.path.join(pkl_dir,'trainset_image_dct.pkl'), precompute_file=os.path.join(pkl_dir,'trainset_MIAA_nopiaa_dct.pkl'))
    # train_dataset = PARA_GIAA_HistogramDataset(root_dir, transform=train_transform, data=train_piaa_dataset.data, map_file=os.path.join(pkl_dir,'trainset_image_dct.pkl'), precompute_file=os.path.join(pkl_dir,'trainset_GIAA_dct.pkl'))
    train_dataset = PARA_MIAA_HistogramDataset(root_dir, transform=train_transform, data=train_piaa_dataset.data, map_file=os.path.join(pkl_dir,'trainset_image_trainuser_dct.pkl'), precompute_file=os.path.join(pkl_dir,'trainset_MIAA_nopiaa_trainuser_dct.pkl'))
    # train_dataset.augment_and_save_dataset(os.path.join(pkl_dir,'trainset_MIAA_nopiaa_trainuser_dct_augment.pkl'))
    raise Exception
    train_dataset = PARA_GIAA_HistogramDataset(root_dir, transform=train_transform, data=train_piaa_dataset.data, map_file=os.path.join(pkl_dir,'trainset_image_trainuser_dct.pkl'), precompute_file=os.path.join(pkl_dir,'trainset_GIAA_trainuser_dct.pkl'))

    # test_dataset = PARA_GSP_HistogramDataset(root_dir, transform=test_transform, piaa_data=test_piaa_dataset.data, 
    #                 giaa_data=test_piaa_dataset.data, map_file=os.path.join(pkl_dir,'testset_image_dct.pkl'), precompute_file=os.path.join(pkl_dir,'testset_GIAA_dct.pkl'))
    # piaa_sample, giaa_sample = test_dataset[0]
    # test_dataset = PARA_HistogramDataset(root_dir, transform=test_transform, data=test_piaa_dataset.data, map_file='testset_image_dct.pkl')
    # test_dataset = PARA_GIAA_HistogramDataset(root_dir, transform=test_transform, data=test_piaa_dataset.data, map_file=os.path.join(pkl_dir,'testset_image_dct.pkl'), precompute_file=os.path.join(pkl_dir,'testset_GIAA_dct.pkl'))
    test_dataset = PARA_GIAA_HistogramDataset(root_dir, transform=test_transform, data=test_piaa_dataset.data, map_file=os.path.join(pkl_dir,'testset_image_testuser_dct.pkl'), precompute_file=os.path.join(pkl_dir,'testset_GIAA_testuser_dct.pkl'))
    raise Exception
    # test_dataset = PARA_PIAA_HistogramDataset(root_dir, transform=test_transform, data=test_piaa_dataset.data)
    # raise Exception
    nworkers = 20
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=nworkers)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=nworkers)
    # Iterate over the training dataloader
    for sample in tqdm(test_dataloader):
        # Perform training operations here
        piaa_sample, gsp_sample, giaa_sample = sample
        

        raise Exception
