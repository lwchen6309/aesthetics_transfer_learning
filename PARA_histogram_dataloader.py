import os
import torch
from torchvision import transforms
import torch.nn.functional as F
from PARA_PIAA_dataloader import PARA_PIAADataset, split_dataset_by_user
from torch.utils.data import DataLoader
import random
import pickle
from tqdm import tqdm


class HistogramDataloaderModified(PARA_PIAADataset):
    def __init__(self, root_dir, transform=None, data=None, map_file=None):
        super().__init__(root_dir, transform)
        if data is not None:
            self.data = data
        # self.data = self.data[:1000] 
        
        # Create a dictionary mapping each unique image name to a list of indices corresponding to that image
        self.unique_images = self.data['imageName'].unique()
        # self.image_to_indices_map = {image: self.data[self.data['imageName'] == image].index.tolist() for image in self.unique_images}
        if map_file and os.path.exists(map_file):
            # Load precomputed map from file
            print("Loading image to indices map from file...")
            self.image_to_indices_map = self._load_map(map_file)
        else:
            # Compute the image to indices map
            self.image_to_indices_map = {image: self.data[self.data['imageName'] == image].index.tolist() for image in self.unique_images}
            if map_file:
                print(f"Saving image to indices map to {map_file}")
                self._save_map(map_file)


    def _discretize(self, value, bins):
        """Helper function to discretize a value given specific bins"""
        for i, bin_value in enumerate(bins):
            if value <= bin_value:
                return i
        return len(bins) - 1

    def _compute_histogram(self, idx):
        sample = super().__getitem__(idx)
        aesthetic_score_histogram = torch.zeros(len([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]))
        attributes_histogram = {
            'qualityScore': torch.zeros(5),
            'compositionScore': torch.zeros(5),
            'colorScore': torch.zeros(5),
            'dofScore': torch.zeros(5),
            'contentScore': torch.zeros(5),
            'lightScore': torch.zeros(5),
            'contentPreference': torch.zeros(5),
            'willingnessToShare': torch.zeros(5)
        }
        traits_histogram = {
            'personality-E': torch.zeros(10),
            'personality-A': torch.zeros(10),
            'personality-N': torch.zeros(10),
            'personality-O': torch.zeros(10),
            'personality-C': torch.zeros(10)
        }

        # Compute histograms for the given sample
        bin_idx = self._discretize(sample['aestheticScores']['aestheticScore'], [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
        aesthetic_score_histogram[bin_idx] += 1

        # Round the quality score to the nearest integer
        rounded_quality_score = round(sample['aestheticScores']['qualityScore'])
        attributes_histogram['qualityScore'][rounded_quality_score - 1] += 1  # Assuming scores are 1-based

        # Compute histograms for other attributes
        for attribute in ['compositionScore', 'colorScore', 'dofScore', 'contentScore', 'lightScore', 'contentPreference', 'willingnessToShare']:
            bin_idx = int(sample['aestheticScores'][attribute] - 1)  # Assuming scores are 1-based
            attributes_histogram[attribute][bin_idx] += 1

        for trait, _ in traits_histogram.items():
            bin_idx = int(sample['userTraits'][trait] - 1)  # Assuming scores are 1-based
            traits_histogram[trait][bin_idx] += 1

        return {
            'userId': sample['userId'],
            'aestheticScore': aesthetic_score_histogram,
            'attributes': attributes_histogram,
            'traits': traits_histogram
        }


    def _save_map(self, file_path):
        """Save image to indices map to a file."""
        with open(file_path, 'wb') as f:
            pickle.dump(self.image_to_indices_map, f)

    def _load_map(self, file_path):
        """Load image to indices map from a file."""
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def __getitem__(self, idx, max_sample=20):
        # Determine the number of samples to draw (d)
        image_name = self.unique_images[idx]
        associated_indices = self.image_to_indices_map[image_name]
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
            histogram_data = self._compute_histogram(random_idx)
            sample = super().__getitem__(random_idx)  # Only needed for one-hot accumulators
            
            accumulated_histogram['aestheticScore'] += histogram_data['aestheticScore']
            
            for trait in accumulated_histogram['onehot_traits'].keys():
                accumulated_histogram['onehot_traits'][trait] += sample['userTraits'][trait]

            for key in accumulated_histogram['attributes'].keys():
                accumulated_histogram['attributes'][key] += histogram_data['attributes'][key]
            
            for key in accumulated_histogram['traits'].keys():
                accumulated_histogram['traits'][key] += histogram_data['traits'][key]
        
        accumulated_histogram['aestheticScore'] /= n_sample
        for trait in accumulated_histogram['onehot_traits'].keys():
            accumulated_histogram['onehot_traits'][trait] /= n_sample

        for key in accumulated_histogram['attributes'].keys():
            accumulated_histogram['attributes'][key] /= n_sample

        for key in accumulated_histogram['traits'].keys():
            accumulated_histogram['traits'][key] /= n_sample

        accumulated_histogram['n_samples'] = n_sample
        accumulated_histogram['image'] = sample['image']
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
    train_dataset = HistogramDataloaderModified(root_dir, transform=train_transform, data=train_piaa_dataset.data, map_file='trainset_image_dct.pkl')
    test_dataset = HistogramDataloaderModified(root_dir, transform=test_transform, data=test_piaa_dataset.data, map_file='testset_image_dct.pkl')

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    # print(train_piaa_dataset[0])
    # Iterate over the training dataloader
    for i, sample in enumerate(train_dataloader):
        # Perform training operations here
        print(sample)
        raise Exception
