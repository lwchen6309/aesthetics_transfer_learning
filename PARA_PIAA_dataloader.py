import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F


class PARA_PIAADataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images and CSVs.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.images_df = pd.read_csv(os.path.join(root_dir, 'annotation', 'PARA-Images.csv'))
        self.user_info_df = pd.read_csv(os.path.join(root_dir, 'annotation', 'PARA-UserInfo.csv'))
        self.data = pd.merge(self.images_df, self.user_info_df, on='userId', how='inner')
        
        # Encoding personal traits
        self.age_encoder = {group: idx for idx, group in enumerate(self.user_info_df['age'].unique())}
        self.gender_encoder = {gender: idx for idx, gender in enumerate(self.user_info_df['gender'].unique())}
        self.education_encoder = {level: idx for idx, level in enumerate(self.user_info_df['EducationalLevel'].unique())}
        self.art_experience_encoder = {experience: idx for idx, experience in enumerate(self.user_info_df['artExperience'].unique())}
        self.photo_experience_encoder = {experience: idx for idx, experience in enumerate(self.user_info_df['photographyExperience'].unique())}
        
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        session_dir = self.data.iloc[idx]['sessionId']
        img_path = os.path.join(self.root_dir, 'imgs', session_dir, self.data.iloc[idx]['imageName'])
        image = Image.open(img_path).convert('RGB')
        
        # One-hot encoding for personal traits
        age_onehot = F.one_hot(torch.tensor(self.age_encoder[self.data.iloc[idx]['age']]), num_classes=len(self.age_encoder))
        gender_onehot = F.one_hot(torch.tensor(self.gender_encoder[self.data.iloc[idx]['gender']]), num_classes=len(self.gender_encoder))
        education_onehot = F.one_hot(torch.tensor(self.education_encoder[self.data.iloc[idx]['EducationalLevel']]), num_classes=len(self.education_encoder))
        art_experience_onehot = F.one_hot(torch.tensor(self.art_experience_encoder[self.data.iloc[idx]['artExperience']]), num_classes=len(self.art_experience_encoder))
        photo_experience_onehot = F.one_hot(torch.tensor(self.photo_experience_encoder[self.data.iloc[idx]['photographyExperience']]), num_classes=len(self.photo_experience_encoder))
        
        sample = {
            'userId': self.data.iloc[idx]['userId'],
            'image': image,
            'aestheticScores': {
                'aestheticScore': self.data.iloc[idx]['aestheticScore'],
                'qualityScore': self.data.iloc[idx]['qualityScore'],
                'compositionScore': self.data.iloc[idx]['compositionScore'],
                'colorScore': self.data.iloc[idx]['colorScore'],
                'dofScore': self.data.iloc[idx]['dofScore'],
                'contentScore': self.data.iloc[idx]['contentScore'],
                'lightScore': self.data.iloc[idx]['lightScore'],
                'contentPreference': self.data.iloc[idx]['contentPreference'],
                'willingnessToShare': self.data.iloc[idx]['willingnessToShare']
            },
            'userTraits': {
                'age': age_onehot,
                'gender': gender_onehot,
                'EducationalLevel': education_onehot,
                'artExperience': art_experience_onehot,
                'photographyExperience': photo_experience_onehot,
                'personality-E': self.data.iloc[idx]['personality-E'],
                'personality-A': self.data.iloc[idx]['personality-A'],
                'personality-N': self.data.iloc[idx]['personality-N'],
                'personality-O': self.data.iloc[idx]['personality-O'],
                'personality-C': self.data.iloc[idx]['personality-C']
            }
        }

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample

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
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
    ])

    # Create datasets with the appropriate transformations
    train_dataset = PARA_PIAADataset(root_dir, transform=train_transform)
    test_dataset = PARA_PIAADataset(root_dir, transform=test_transform)
    
    # Create dataloaders for training and test sets
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Iterate over the training dataloader
    for sample in train_dataloader:
        # Perform training operations here
        raise Exception
