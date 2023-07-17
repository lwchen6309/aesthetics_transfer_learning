import os
import random
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np


class PARADataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True, random_seed=None, 
                 use_attr=True, use_hist=True):
        self.root_dir = root_dir
        self.transform = transform
        self.use_attr = use_attr
        self.use_hist = use_hist
        self.train = train
        if self.train:
            self.annotations = pd.read_csv(os.path.join(root_dir, 'annotation', 'PARA-GiaaTrain.csv'))
        else:
            self.annotations = pd.read_csv(os.path.join(root_dir, 'annotation', 'PARA-GiaaTest.csv'))
            if random_seed is not None:
                random.seed(random_seed)
                torch.manual_seed(random_seed)
        
        self.attributes = [
            'aestheticScore',
            'qualityScore',
            'compositionScore',
            'colorScore',
            'dofScore',
            'lightScore',
            'contentScore',
            'contentPreference',
            'willingnessToShare'
        ]
        if self.use_hist:
            self.attributes = [self.attributes[0],*self.attributes[2:]]
        if not self.use_attr:
            self.attributes = [self.attributes[0]]

        self.mean_attributes = ['%s_mean'%x for x in self.attributes]
        self.std_attributes = ['%s_std'%x for x in self.attributes]
        scale = np.arange(1, 5.5, 0.5)
        self.score_hist = []
        self.score_hist.append(['%s_%2.1f'%(self.attributes[0], i) for i in scale])
        if self.use_attr:
            for attr in self.attributes[1:]:
                scale = np.arange(1, 5.5, 1.0)
                self.score_hist.append(['%s_%d'%(attr, i) for i in scale])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotations = self.annotations
        session_dir = annotations.iloc[idx]['sessionId']
        img_path = os.path.join(self.root_dir, 'imgs', session_dir, annotations.iloc[idx]['imageName'])
        if self.use_hist:
            aesthetic_score_hist = [np.array(annotations.iloc[idx][attr], dtype=np.float32) for attr in self.score_hist]
        else:
            aesthetic_score_hist = None
        aesthetic_score_mean = np.array(annotations.iloc[idx][self.mean_attributes], dtype=np.float32)
        aesthetic_score_std = np.array(annotations.iloc[idx][self.std_attributes], dtype=np.float32)
        
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, aesthetic_score_mean, aesthetic_score_std, aesthetic_score_hist


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
    train_dataset = PARADataset(root_dir, transform=train_transform, train=True, use_attr=True)
    test_dataset = PARADataset(root_dir, transform=test_transform, train=False, use_attr=True, random_seed=random_seed)

    # Create dataloaders for training and test sets
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Iterate over the training dataloader
    for images, mean_scores, std_scores, score_hist in train_dataloader:
        # Perform training operations here
        print(mean_scores)
        print(std_scores)
        print(score_hist)
        raise Exception

    # Iterate over the test dataloader
    for images, mean_scores, std_scores in test_dataloader:
        # Perform test operations here
        pass
