import os
import torch
from torchvision import transforms
import torch.nn.functional as F
from PARA_PIAA_dataloader import PARA_PIAADataset, split_dataset_by_user
from torch.utils.data import DataLoader
import random
import pickle
from tqdm import tqdm


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

        for random_idx in sampled_indices:
            sample = super().__getitem__(random_idx)
            
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


class PARA_GIAA_HistogramDataset(PARA_PIAADataset):    
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
            sample = super().__getitem__(random_idx)
            
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


class PARA_PIAA_HistogramDataset(PARA_PIAADataset):
    def __init__(self, root_dir, transform=None, data=None, map_file=None):
        super().__init__(root_dir, transform)
        if data is not None:
            self.data = data

    def _discretize(self, value, bins):
        """Helper function to discretize a value given specific bins"""
        for i, bin_value in enumerate(bins):
            if value <= bin_value:
                return i
        return len(bins) - 1
    
    def __getitem__(self, idx):
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
    train_piaa_dataset, test_piaa_dataset = split_dataset_by_user(train_piaa_dataset, test_piaa_dataset, test_count=40, max_annotations_per_user=10, seed=random_seed)
    
    # Create datasets with the appropriate transformations
    train_dataset = PARA_HistogramDataset(root_dir, transform=train_transform, data=train_piaa_dataset.data, map_file='trainset_image_dct_%d.pkl'%random_seed)
    test_dataset = PARA_HistogramDataset(root_dir, transform=test_transform, data=test_piaa_dataset.data, map_file='testset_image_dct.pkl_%d'%random_seed)    
    train_giaa_histo_dataset = PARA_GIAA_HistogramDataset(root_dir, transform=train_transform, data=train_piaa_dataset.data, map_file='trainset_image_dct_%d.pkl'%random_seed)
    train_piaa_histo_dataset = PARA_PIAA_HistogramDataset(root_dir, transform=train_transform, data=train_piaa_dataset.data)
    
    nworkers = 20
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=nworkers)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=nworkers)
    # Iterate over the training dataloader
    for sample in tqdm(train_dataloader):
        # Perform training operations here
        sample
        # if i ==3:
        raise Exception
