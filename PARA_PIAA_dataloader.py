import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import random
from tqdm import tqdm
import copy


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
        
        sample = {
            'userId': self.data.iloc[idx]['userId'],
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
            },
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


def limit_annotations_per_user(data, max_annotations_per_user):
    grouped = data.groupby('userId', group_keys=False)
    filtered_data = grouped.apply(lambda x: x.sample(min(len(x), max_annotations_per_user)))
    return filtered_data

def generate_data_per_user(dataset, max_annotations_per_user=50):
    data = dataset.data  # Assuming this is a pandas DataFrame
    for user_id in data['userId'].unique():
        # Filter data for the current user
        user_data = data[data['userId'] == user_id]
        # Limit the number of annotations per user
        user_data_limited = limit_annotations_per_user(user_data, max_annotations_per_user=max_annotations_per_user)
        # Create a deep copy of the dataset and replace its data with the limited user data
        user_dataset = copy.deepcopy(dataset)
        user_dataset.data = user_data_limited
        # Yield the dataset for the user
        yield (user_id, user_dataset)

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

def split_dataset_by_user(train_dataset, test_dataset, test_count=40, max_annotations_per_user=[10, 50], n_samples=10, user_id_list=None, seed=None):
    # Split data by user
    train_users, test_users = split_data_by_user(train_dataset.data, test_count=test_count, user_id_list=user_id_list, seed=seed)
    # Filter data by user IDs
    train_dataset.data = train_dataset.data[train_dataset.data['userId'].isin(train_users)]
    test_dataset.data = test_dataset.data[test_dataset.data['userId'].isin(test_users)]
    
    # Limit the number of annotations per user
    train_dataset.databank = [limit_annotations_per_user(train_dataset.data, max_annotations_per_user=max_annotations_per_user[0]) for _ in range(n_samples)]
    train_dataset.data = train_dataset.databank[0]
    test_dataset.databank = [limit_annotations_per_user(test_dataset.data, max_annotations_per_user=max_annotations_per_user[1]) for _ in range(n_samples)]
    test_dataset.data = test_dataset.databank[0]
    return train_dataset, test_dataset

def split_dataset_by_images(train_dataset, test_dataset, root_dir):
    trainset = pd.read_csv(os.path.join(root_dir, 'annotation', 'PARA-GiaaTrain.csv'))
    testset = pd.read_csv(os.path.join(root_dir, 'annotation', 'PARA-GiaaTest.csv'))
    train_dataset.data = train_dataset.data[train_dataset.data['imageName'].isin(trainset['imageName'])]
    test_dataset.data = test_dataset.data[test_dataset.data['imageName'].isin(testset['imageName'])]
    return train_dataset, test_dataset


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
    train_dataset = PARA_PIAADataset(root_dir, transform=train_transform)
    test_dataset = PARA_PIAADataset(root_dir, transform=train_transform)
    # train_dataset, test_dataset = split_dataset_by_user(train_dataset, test_dataset, test_count=40, max_annotations_per_user=[10,50])
    
    train_dataset = split_dataset_by_trait(train_dataset, 'gender', 'male')
    test_dataset = split_dataset_by_trait(test_dataset, 'gender', 'male')
    train_dataset, test_dataset = split_dataset_by_images(train_dataset, test_dataset, root_dir)
    print(len(train_dataset), len(test_dataset))
    
    # Create dataloaders for training and test sets
    n_workers = 10
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=n_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=n_workers)

    # Iterate over the training dataloader
    for sample in tqdm(train_dataloader):
        # Perform training operations here
        # print(sample['image'])
        sample_score, sample_attr = collect_batch_attribute(sample)
        sample_pt = collect_batch_personal_trait(sample)
        # print(sample_score.shape)
        # print(sample_attr.shape)
        # print(sample_pt.shape)
        # raise Exception
