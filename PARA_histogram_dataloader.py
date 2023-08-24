import os
import torch
from torchvision import transforms
import torch.nn.functional as F
from PARA_PIAA_dataloader import PARA_PIAADataset, split_dataset_by_user
import random
import pickle
from tqdm import tqdm


class HistogramDataloaderModified(PARA_PIAADataset):
    def __init__(self, root_dir, transform=None, histograms_file=None, data=None):
        super().__init__(root_dir, transform)
        if data is not None:
            self.data = data
        # self.data = self.data[:100]

        # Try to load precomputed histograms if a file path is provided
        if histograms_file and os.path.exists(histograms_file):
            print("Loading precomputed histograms from file...")
            self.precomputed_histograms = self.load_histograms(histograms_file)
        else:
            # Precompute histograms for the entire dataset
            print("Precomputing histograms...")
            self.precomputed_histograms = [self._compute_histogram(i) 
                for i in tqdm(range(len(self.data)), desc="Precomputing histograms")]
            if histograms_file:
                print(f"Saving histograms to {histograms_file}")
                self.save_histograms(histograms_file)

        # Accumulators for one-hot encoded traits
        self.onehot_accumulators = {
            'age': torch.zeros(len(self.age_encoder)),
            'gender': torch.zeros(len(self.gender_encoder)),
            'EducationalLevel': torch.zeros(len(self.education_encoder)),
            'artExperience': torch.zeros(len(self.art_experience_encoder)),
            'photographyExperience': torch.zeros(len(self.photo_experience_encoder))
        }

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

    def __getitem__(self, idx, max_sample=50):
        # Determine the number of samples to draw (d)
        n_sample = random.randint(1, max_sample)
        sampled_indices = random.sample(range(0, len(self.data)), n_sample)
        
        # Initialize accumulators
        accumulated_histogram = {
            'aestheticScore': torch.zeros_like(self.precomputed_histograms[0]['aestheticScore']),
            'attributes': {key: torch.zeros_like(value) for key, value in self.precomputed_histograms[0]['attributes'].items()},
            'traits': {key: torch.zeros_like(value) for key, value in self.precomputed_histograms[0]['traits'].items()},
            'onehot_traits': self.onehot_accumulators.copy()
        }

        for random_idx in sampled_indices:
            accumulated_histogram['aestheticScore'] += self.precomputed_histograms[random_idx]['aestheticScore']

            sample = super().__getitem__(random_idx)  # Only needed for one-hot accumulators
            for trait, _ in self.onehot_accumulators.items():
                accumulated_histogram['onehot_traits'][trait] += sample['userTraits'][trait]

            for key in accumulated_histogram['attributes'].keys():
                accumulated_histogram['attributes'][key] += self.precomputed_histograms[random_idx]['attributes'][key]

            for key in accumulated_histogram['traits'].keys():
                accumulated_histogram['traits'][key] += self.precomputed_histograms[random_idx]['traits'][key]
        
        accumulated_histogram['n_samples'] = n_sample
        return accumulated_histogram

    def save_histograms(self, file_path):
        """Save precomputed histograms to a file."""
        with open(file_path, 'wb') as f:
            pickle.dump(self.precomputed_histograms, f)

    def load_histograms(self, file_path):
        """Load precomputed histograms from a file."""
        with open(file_path, 'rb') as f:
            return pickle.load(f)


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
    train_dataset = HistogramDataloaderModified(root_dir, transform=train_transform, data=train_piaa_dataset.data, histograms_file='trainset_histograms.pkl')
    test_dataset = HistogramDataloaderModified(root_dir, transform=test_transform, data=test_piaa_dataset.data, histograms_file='testset_histograms.pkl')

    print(train_dataset.precomputed_histograms[0])
    print(train_piaa_dataset[0])
    # Iterate over the training dataloader
    for sample in train_dataset:
        # Perform training operations here
        print(sample)
        raise Exception
