import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import random
from tqdm import tqdm
import copy
from time import time
from sklearn.model_selection import KFold
import argparse


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
        
        # Encoding image attributes
        self.img_emotion_encoder = {emotion: idx for idx, emotion in enumerate(self.images_df['imgEmotion'].unique())}
        self.difficulty_of_judgment_encoder = {difficulty: idx for idx, difficulty in enumerate(self.images_df['difficultyOfJudgment'].unique())}
        self.semantic_encoder = {content: idx for idx, content in enumerate(self.images_df['semantic'].unique())}

        self.transform = transform

    def one_hot_personality(self, trait_value):
        """
        One-hot encode the personality trait values.

        Args:
            trait_value (int): Personality trait value.

        Returns:
            torch.Tensor: One-hot encoded tensor for the trait.
        """
        if trait_value < 1 or trait_value > 10:
            raise ValueError("Personality trait value must be between 1 and 10")
        return F.one_hot(torch.tensor(trait_value - 1), num_classes=10)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx, use_image=True):
        session_dir = self.data.iloc[idx]['sessionId']
        img_path = os.path.join(self.root_dir, 'imgs', session_dir, self.data.iloc[idx]['imageName'])
        if use_image:
            image = Image.open(img_path).convert('RGB')
        
        # One-hot encoding for personal traits and image attributes
        age_onehot = F.one_hot(torch.tensor(self.age_encoder[self.data.iloc[idx]['age']]), num_classes=len(self.age_encoder))
        gender_onehot = F.one_hot(torch.tensor(self.gender_encoder[self.data.iloc[idx]['gender']]), num_classes=len(self.gender_encoder))
        education_onehot = F.one_hot(torch.tensor(self.education_encoder[self.data.iloc[idx]['EducationalLevel']]), num_classes=len(self.education_encoder))
        art_experience_onehot = F.one_hot(torch.tensor(self.art_experience_encoder[self.data.iloc[idx]['artExperience']]), num_classes=len(self.art_experience_encoder))
        photo_experience_onehot = F.one_hot(torch.tensor(self.photo_experience_encoder[self.data.iloc[idx]['photographyExperience']]), num_classes=len(self.photo_experience_encoder))
        img_emotion_onehot = F.one_hot(torch.tensor(self.img_emotion_encoder[self.data.iloc[idx]['imgEmotion']]), num_classes=len(self.img_emotion_encoder))
        difficulty_of_judgment_onehot = F.one_hot(torch.tensor(self.difficulty_of_judgment_encoder[self.data.iloc[idx]['difficultyOfJudgment']]), num_classes=len(self.difficulty_of_judgment_encoder))
        semantic_onehot = F.one_hot(torch.tensor(self.semantic_encoder[self.data.iloc[idx]['semantic']]), num_classes=len(self.semantic_encoder))
        
        user_traits = {
            'age': age_onehot,
            'gender': gender_onehot,
            'EducationalLevel': education_onehot,
            'artExperience': art_experience_onehot,
            'photographyExperience': photo_experience_onehot,
            'personality-E': self.data.iloc[idx]['personality-E'],
            'personality-A': self.data.iloc[idx]['personality-A'],
            'personality-N': self.data.iloc[idx]['personality-N'],
            'personality-O': self.data.iloc[idx]['personality-O'],
            'personality-C': self.data.iloc[idx]['personality-C'],
            # 'personality-E-onehot': self.one_hot_personality(self.data.iloc[idx]['personality-E']),
            # 'personality-A-onehot': self.one_hot_personality(self.data.iloc[idx]['personality-A']),
            # 'personality-N-onehot': self.one_hot_personality(self.data.iloc[idx]['personality-N']),
            # 'personality-O-onehot': self.one_hot_personality(self.data.iloc[idx]['personality-O']),
            # 'personality-C-onehot': self.one_hot_personality(self.data.iloc[idx]['personality-C'])
        }

        sample = {
            'userId': self.data.iloc[idx]['userId'],
            'image_path': img_path,
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
            'userTraits': user_traits,
            'imageAttributes': {
                'imgEmotion': img_emotion_onehot,
                'difficultyOfJudgment': difficulty_of_judgment_onehot,
                'semantic': semantic_onehot
            }
        }
        
        if use_image:
            sample['image'] = image
            if self.transform:
                sample['image'] = self.transform(sample['image'])
        
        return sample

    def print_trait_encoders(self):
        # Encoding personal traits
        print(self.age_encoder)
        print(self.gender_encoder)
        print(self.education_encoder)
        print(self.art_experience_encoder)
        print(self.photo_experience_encoder)


def collect_batch_personal_trait(sample,
        personal_traits = ['age', 'gender', 'EducationalLevel', 'artExperience', 'photographyExperience'],
        big5 = ['personality-E', 'personality-A', 'personality-N', 'personality-O', 'personality-C']):
    pt = torch.cat([sample['userTraits'][_pt] for _pt in personal_traits], dim=1)   
    pt_big5 = torch.stack([sample['userTraits'][pt] for pt in big5], dim=1)
    total_pt = torch.cat([pt, pt_big5], dim=1)
    return total_pt

def collect_batch_attribute(sample,     
        attr = ['aestheticScore','qualityScore','compositionScore','colorScore','dofScore',
        'contentScore','lightScore','contentPreference','willingnessToShare']):
    batch_attr = torch.stack([sample['aestheticScores'][_attr] for _attr in attr], dim=1)
    return batch_attr[:,0][:,None], batch_attr[:,1:]


def limit_annotations_per_user(data, max_annotations_per_user=100):
    # Group by userId
    grouped = data.groupby('userId', group_keys=False)

    def sample_disjoint(group, n1, n2):
        # Sample min(len(group), n1 + n2) items randomly
        total_sample_size = min(len(group), n1 + n2)
        total_sample = random.sample(range(len(group)), total_sample_size)
        
        # Create boolean masks for two disjoint sets
        mask1 = [True if i in total_sample[:n1] else False for i in range(len(group))]
        mask2 = [True if i in total_sample[n1:n1 + n2] else False for i in range(len(group))]

        return group[mask1], group[mask2]

    # Initialize lists to store the two sets
    train_set = []
    test_set = []

    # Apply the function to each group and append results to the lists
    for name, group in grouped:
        n_test = len(group) - max_annotations_per_user
        set1, set2 = sample_disjoint(group.reset_index(drop=True), max_annotations_per_user, n_test)
        train_set.append(set1)
        test_set.append(set2)

    # Concatenate the lists into DataFrames
    train_set_df = pd.concat(train_set)
    test_set_df = pd.concat(test_set)
    
    return train_set_df, test_set_df

def generate_data_per_user(dataset, max_annotations_per_user=100, num_users=40, num_image_threshold=500, train_transform=None, test_transform=None):
    data = dataset.data
    # users = data['userId'].unique()
    n_users = data['userId'].value_counts()
    
    # Filter users who have annotated at least num_image_threshold images
    filtered_users = n_users[n_users >= num_image_threshold].index.to_list()
    users = np.random.choice(filtered_users, min(num_users, len(filtered_users)), replace=False)
    print(f'Sample {len(users)} users') 
    # users = np.random.choice(users, num_users, replace=False)
    
    for user_id in users:
        user_data = data[data['userId'] == user_id]
        train_data, test_data = limit_annotations_per_user(user_data, max_annotations_per_user=max_annotations_per_user)
        # Create a deep copy of the dataset and replace its data with the limited user data
        train_dataset = copy.deepcopy(dataset)
        test_dataset = copy.deepcopy(dataset)

        train_dataset.data = train_data
        test_dataset.data = test_data
        train_dataset.transforms = train_transform
        test_dataset.transforms = test_transform

        # Yield the dataset for the user
        yield (user_id, train_dataset, test_dataset)

def split_data_by_user(data, test_count, user_id_list=None, seed=None):
    if seed is not None:
        random.seed(seed)  # Setting the random seed

    if user_id_list is not None:
        # Filter data for user IDs that are in the provided user_id_list if it is not None
        data = data[data['userId'].isin(user_id_list)]
    
    # Get the unique user IDs
    user_ids = data['userId'].unique()
    user_ids = list(user_ids)  # Ensure user_ids is a list for shuffling
    
    # Shuffle the list of user IDs
    random.shuffle(user_ids)
    
    # Calculate the number of users for training based on the total minus test_count
    # total_users = len(user_ids)
    train_users = user_ids[:-test_count]
    test_users = user_ids[-test_count:]
    
    return train_users, test_users

def split_dataset_by_user(dataset, test_count=40, max_annotations_per_user=100, n_samples=10, user_id_list=None, seed=None):
    # Split data by user
    _, test_users = split_data_by_user(dataset.data, test_count=test_count, user_id_list=user_id_list, seed=seed)
    # Filter data by user IDs
    
    dataset.data = dataset.data[dataset.data['userId'].isin(test_users)]
    train_dataset = dataset
    test_dataset = copy.deepcopy(dataset)
    
    # Limit the number of annotations per user
    databank = [limit_annotations_per_user(train_dataset.data, max_annotations_per_user=max_annotations_per_user) for _ in range(n_samples)]
    train_dataset.databank = [data[0] for data in databank]
    test_dataset.databank = [data[1] for data in databank]
    train_dataset.data = train_dataset.databank[0]
    test_dataset.data = test_dataset.databank[0]
    return train_dataset, test_dataset

def split_dataset_by_images(dataset, root_dir, validation_split=0.1):
    # Read train and test CSV files
    trainset = pd.read_csv(os.path.join(root_dir, 'annotation', 'PARA-GiaaTrain.csv'))
    testset = pd.read_csv(os.path.join(root_dir, 'annotation', 'PARA-GiaaTest.csv'))
    
    # Create copies of the original dataset for training, validation, and testing
    train_dataset = copy.deepcopy(dataset)
    val_dataset = copy.deepcopy(dataset)
    test_dataset = copy.deepcopy(dataset)

    # Shuffle trainset['imageName'] and split into train and validation sets
    train_image_names = trainset['imageName'].sample(frac=1, random_state=42)
    num_train_samples = int(len(train_image_names) * (1 - validation_split))
    train_image_names_train = train_image_names[:num_train_samples]
    train_image_names_val = train_image_names[num_train_samples:]
    
    # Filter datasets based on image names
    train_dataset.data = train_dataset.data[train_dataset.data['imageName'].isin(train_image_names_train)]
    val_dataset.data = val_dataset.data[val_dataset.data['imageName'].isin(train_image_names_val)]
    test_dataset.data = test_dataset.data[test_dataset.data['imageName'].isin(testset['imageName'])]
    
    # Path to the validation images file
    validation_images_file = os.path.join(root_dir, 'validation_images.txt')

    # Check if validation images file already exists
    if os.path.exists(validation_images_file):
        # Read the existing validation image names
        with open(validation_images_file, 'r') as file:
            existing_validation_images = {line.strip() for line in file.readlines()}
        
        # Compare the existing names with the newly generated ones
        new_validation_images = set(train_image_names_val.tolist())
        if existing_validation_images != new_validation_images:
            raise ValueError("Validation set inconsistency detected. Existing validation images do not match newly generated validation images.")
        else:
            print('Assert fixed validation images')
    else:
        # Save validation image names to a txt file if not exist
        with open(validation_images_file, 'w') as file:
            for image_name in train_image_names_val:
                file.write(image_name + '\n')

    return train_dataset, val_dataset, test_dataset


def split_dataset_by_trait(dataset, trait, value):
    """
    Split the dataset based on a specific trait and its value.

    Args:
        dataset (Dataset): The dataset to filter.
        trait (str): The trait to filter by, e.g., 'gender', 'EducationalLevel'.
        value (str): The value of the trait to filter by, e.g., 'male', 'Bachelor'.

    Returns:
        filtered_dataset (Dataset): A new dataset containing only the entries with the specified trait value.
    """
    filtered_data = dataset.data[dataset.data[trait] == value]
    filtered_dataset = PARA_PIAADataset(dataset.root_dir, transform=dataset.transform)
    filtered_dataset.data = filtered_data
    return filtered_dataset

def create_user_split_kfold(dataset, k=4):
    root_dir = dataset.root_dir
    
    # Assuming 'userId' is a column in your dataset
    user_ids = dataset.data['userId'].unique()
    random.shuffle(user_ids)  # Shuffle the user IDs to randomize the distribution

    kf = KFold(n_splits=k, shuffle=True, random_state=42)  # Prepare the KFold object
    
    all_files_exist = True
    for fold in range(1, k+1):
        train_ids_path = os.path.join(root_dir, f'TrainUserIDs_Fold{fold}.txt')
        test_ids_path = os.path.join(root_dir, f'TestUserIDs_Fold{fold}.txt')
        
        # Check if both files for this fold exist
        if not (os.path.exists(train_ids_path) and os.path.exists(test_ids_path)):
            all_files_exist = False
            break
    
    if all_files_exist:
        print("All fold files already exist, skipping computation.")
        return

    for fold, (train_index, test_index) in enumerate(kf.split(user_ids), start=1):
        train_ids_path = os.path.join(root_dir, f'TrainUserIDs_Fold{fold}.txt')
        test_ids_path = os.path.join(root_dir, f'TestUserIDs_Fold{fold}.txt')
        
        print(f"Processing Fold {fold}")
        
        # Get train and test user IDs for the current fold
        train_user_ids = user_ids[train_index]
        test_user_ids = user_ids[test_index]
        
        # Save train user IDs to file
        with open(train_ids_path, "w") as train_ids_file:
            for user_id in train_user_ids:
                train_ids_file.write(str(user_id) + "\n")
        
        # Save test user IDs to file
        with open(test_ids_path, "w") as test_ids_file:
            for user_id in test_user_ids:
                test_ids_file.write(str(user_id) + "\n")
        
        print(f"Fold {fold}: Train User IDs: {len(train_user_ids)}, Test User IDs: {len(test_user_ids)}")

def create_user_split_dataset_kfold(dataset, train_dataset, val_dataset, test_dataset, fold_id, n_fold = 4):
    
    create_user_split_kfold(dataset, k=n_fold)
    
    # File paths for saving the user IDs
    root_dir = dataset.root_dir
    train_ids_path = os.path.join(root_dir, f'TrainUserIDs_Fold{fold_id}.txt')
    test_ids_path = os.path.join(root_dir, f'TestUserIDs_Fold{fold_id}.txt')
    print('Read Image Set')
    with open(train_ids_path, "r") as train_file:
        train_user_id = train_file.read().splitlines()
    with open(test_ids_path, "r") as test_file:
        test_user_id = test_file.read().splitlines()

    # train_dataset = copy.deepcopy(train_dataset)
    # val_dataset = copy.deepcopy(val_dataset)
    # test_dataset = copy.deepcopy(test_dataset)
    train_dataset.data = train_dataset.data[train_dataset.data['userId'].isin(train_user_id)]
    val_dataset.data = val_dataset.data[val_dataset.data['userId'].isin(train_user_id)]
    test_dataset.data = test_dataset.data[test_dataset.data['userId'].isin(test_user_id)]
    return train_dataset, val_dataset, test_dataset


def load_data(args, root_dir = '/home/lwchen/datasets/PARA/'):
# def load_data(args, root_dir = '/data/leuven/362/vsc36208/datasets/PARA/'):
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

    # Load datasets
    # Create datasets with the appropriate transformations
    dataset = PARA_PIAADataset(root_dir, transform=train_transform)
    piaa_dataset = PARA_PIAADataset(root_dir, transform=train_transform)
    train_dataset, val_dataset, test_dataset = split_dataset_by_images(piaa_dataset, root_dir)
    # Assuming shell_users_df contains the shell user DataFrame
    if getattr(args, 'use_cv', False):
        train_dataset, val_dataset, test_dataset = create_user_split_dataset_kfold(dataset, train_dataset, val_dataset, test_dataset, fold_id=fold_id, n_fold=n_fold)
    
    is_trait_disjoint = getattr(args, 'trait', False) and getattr(args, 'value', False)
    if is_trait_disjoint:
        print(f'Split trait according to {args.trait} == {args.value}')
        train_dataset.data = train_dataset.data[train_dataset.data[args.trait] != args.value]
        val_dataset.data = val_dataset.data[val_dataset.data[args.trait] != args.value]
        test_dataset.data = test_dataset.data[test_dataset.data[args.trait] == args.value]
    
    test_dataset.transform = test_transform
    val_dataset.transform = test_transform
    print(len(train_dataset), len(val_dataset), len(test_dataset))

    return train_dataset, val_dataset, test_dataset


def load_user_sample_data(args, root_dir = '/home/lwchen/datasets/PARA/'):
# def load_data(args, root_dir = '/data/leuven/362/vsc36208/datasets/PARA/'):
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
    
    max_annotations_per_user = getattr(args, 'max_annotations_per_user', 100)
    num_image_threshold = getattr(args, 'num_image_threshold', 500)
    num_users = getattr(args, 'num_users', 10)

    # Load datasets
    # Create datasets with the appropriate transformations
    dataset = PARA_PIAADataset(root_dir, transform=train_transform)
    piaa_data_gen = generate_data_per_user(dataset, max_annotations_per_user=max_annotations_per_user, num_users=num_users, num_image_threshold=num_image_threshold, train_transform=train_transform, test_transform=test_transform)

    return piaa_data_gen


if __name__ == '__main__':
    # Usage example:
    root_dir = '/home/lwchen/datasets/PARA/'

    parser = argparse.ArgumentParser()
    parser.add_argument('--fold_id', type=int, default=1)
    parser.add_argument('--n_fold', type=int, default=4)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--use_cv', action='store_true', help='Enable cross validation')
    parser.add_argument('--is_eval', action='store_true', help='Enable evaluation mode')
    parser.add_argument('--no_log', action='store_false', dest='is_log', help='Disable logging')
    args = parser.parse_args()

    train_dataset, val_dataset, test_dataset = load_data(args)
    raise Exception
    # Create dataloaders for training and test sets
    n_workers = 8
    batch_size = 100
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers)

    # Iterate over the training dataloader
    t0 = time()
    for sample in tqdm(test_dataloader):
        sample_score, sample_attr = collect_batch_attribute(sample)
        sample_pt = collect_batch_personal_trait(sample)
    print(time() - t0)

    # Iterate over the training dataloader
    t0 = time()
    for sample in tqdm(test_bak_dataloader):
        sample_score, sample_attr = collect_batch_attribute(sample)
        sample_pt = collect_batch_personal_trait(sample)        
    print(time() - t0)
