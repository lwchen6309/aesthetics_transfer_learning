import os
import random
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from tqdm import tqdm 
from scipy.stats import pearsonr, spearmanr
import copy
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import warnings
import argparse

# Ignore all warnings
warnings.filterwarnings('ignore')


class LAPIS_PIAADataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images and CSVs.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        annot_dir = os.path.join(root_dir, 'annotation')
        piaa_path = os.path.join(annot_dir, 'Dataset_individualratings_metadata.xlsx')
        self.image_dir = os.path.join(root_dir, 'datasetImages_originalSize')
        
        # Load and preprocess data
        data = pd.read_excel(piaa_path)
        split_paths = [os.path.split(path) for path in data['image_filename'].values]
        data['art_type'], data['imageName'] = zip(*split_paths)
        # Filter non-exsiting file, or the non-detected file caused by error code of filenames
        image_names = list(data['imageName'].unique())
        existing_image_names = [img for img in image_names if os.path.exists(os.path.join(root_dir, 'datasetImages_originalSize', img))]
        data = data[data['imageName'].isin(existing_image_names)]

        # Drop empty entries
        required_fields = ['image_id', 'response', 'participant_id', 'age', 'nationality', 'demo_gender', 'demo_edu', 'demo_colorblind']
        self.art_interest_fields = [f'VAIAK{i}' for i in range(1, 8)] + [f'2VAIAK{i}' for i in range(1, 5)]
        required_fields += self.art_interest_fields
        self.data = data.dropna(subset=required_fields)
        # Encode text columna
        self.trait_columns = ['image_id', 'response', 'participant_id', *self.art_interest_fields]
        self.encoded_trait_columns = ['art_type', 'nationality', 'demo_gender', 'demo_edu', 'demo_colorblind', 'age']

        # Categorize ages into 5 bins
        min_age, max_age = data['age'].min(), data['age'].max()
        interval_edges = np.linspace(min_age, max_age, num=6)
        interval_labels = [f'{int(interval_edges[i])}-{int(interval_edges[i+1])-1}' for i in range(len(interval_edges)-1)]
        age_intervals = pd.cut(data['age'], bins=interval_edges, labels=interval_labels, include_lowest=True)
        self.data['age'] = age_intervals
        
        self.trait_encoders = [{group: idx for idx, group in enumerate(self.data[attribute].unique())} for attribute in self.encoded_trait_columns]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx, use_image=True):
        # Encoding attributes on-the-fly
        sample = {attribute:encoder[self.data.iloc[idx][attribute]] 
            for attribute, encoder in zip(self.encoded_trait_columns, self.trait_encoders)}
        sample.update({attribute:self.data.iloc[idx][attribute] for attribute in self.trait_columns})
        
        img_path = os.path.join(self.root_dir, 'datasetImages_originalSize', self.data.iloc[idx]['imageName'])
        sample['imgName'] = self.data.iloc[idx]['imageName']
        if use_image:
            sample['image'] = Image.open(img_path).convert('RGB')
            if self.transform:
                sample['image'] = self.transform(sample['image'])

        return sample


def assert_score_to_GIAA(root_dir = '/home/lwchen/datasets/LAPIS'):
    annot_dir = os.path.join(root_dir, 'annotation')
    piaa_path = os.path.join(annot_dir, 'Dataset_individualratings_metadata.xlsx')
    piaa_table = pd.read_excel(piaa_path)
    giaa_path = os.path.join(annot_dir, 'AADB_dataset_bak.csv')
    giaa_table = pd.read_csv(giaa_path)
    
    piaa_table['art_type'], piaa_table['imgName'] = zip(*[os.path.split(path) for path in piaa_table['image_filename'].values])
    v1_col = ['VAIAK%d'%i for i in range(1,8)]
    v2_col = ['2VAIAK%d'%i for i in range(1,5)]
    art_interest = [*v1_col, *v2_col]
    rating = ['response']
    
    # Assrt the same score
    giaa_score_map = {img_path: mean_score for img_path, mean_score in zip(giaa_table['imgName'], giaa_table['meanScore'])}
    
    all_data_collection = []
    matched_score = []
    for i, (imgname, group) in enumerate(piaa_table.groupby('imgName')):
        all_data_collection.append(group[rating].mean())
        matched_score.append(giaa_score_map[imgname])
        if i > 200:
            break
    all_data_collection = np.array(all_data_collection).T
    matched_score = np.array(matched_score)

    for data in all_data_collection:
        filtered_idx = np.logical_not(np.isnan(data))
        plcc, p_value = pearsonr(matched_score, data)
        print(plcc)


def limit_annotations_per_user(data, max_annotations_per_user=[100, 50]):
    # Ensure the input is a list of two elements
    if len(max_annotations_per_user) != 2:
        raise ValueError("max_annotations must be a list of two elements")

    # Group by userId
    grouped = data.groupby('participant_id', group_keys=False)

    def sample_disjoint(group, n1, n2):
        # Sample min(len(group), n1 + n2) items randomly
        total_sample_size = min(len(group), n1 + n2)
        total_sample = random.sample(range(len(group)), total_sample_size)
        
        # Create boolean masks for two disjoint sets
        mask1 = [True if i in total_sample[:n1] else False for i in range(len(group))]
        mask2 = [True if i in total_sample[n1:n1 + n2] else False for i in range(len(group))]

        return group[mask1], group[mask2]

    # Initialize lists to store the two sets
    first_set = []
    second_set = []

    # Apply the function to each group and append results to the lists
    for name, group in grouped:
        set1, set2 = sample_disjoint(group.reset_index(drop=True), max_annotations_per_user[0], max_annotations_per_user[1])
        first_set.append(set1)
        second_set.append(set2)

    # Concatenate the lists into DataFrames
    first_set_df = pd.concat(first_set)
    second_set_df = pd.concat(second_set)
    
    return first_set_df, second_set_df

def generate_data_per_user(dataset, max_annotations_per_user=[100, 50]):
    data = dataset.data  # Assuming this is a pandas DataFrame
    for user_id, user_data in data.groupby('participant_id'):
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
        data = data[data['participant_id'].isin(user_id_list)]
    
    # Get the unique user IDs
    user_ids = data['participant_id'].unique()
    user_ids = list(user_ids)  # Ensure user_ids is a list for shuffling
    
    # Shuffle the list of user IDs
    random.shuffle(user_ids)
    
    # Calculate the number of users for training based on the total minus test_count
    # total_users = len(user_ids)
    train_users = user_ids[:-test_count]
    test_users = user_ids[-test_count:]
    
    return train_users, test_users

def split_dataset_by_user(dataset, test_count=40, max_annotations_per_user=[10, 50], n_samples=10, user_id_list=None, seed=None):
    # Split data by user
    _, test_users = split_data_by_user(dataset.data, test_count=test_count, user_id_list=user_id_list, seed=seed)
    # Filter data by user IDs
    
    dataset.data = dataset.data[dataset.data['participant_id'].isin(test_users)]
    train_dataset = dataset
    test_dataset = copy.deepcopy(dataset)
    
    # Limit the number of annotations per user
    databank = [limit_annotations_per_user(train_dataset.data, max_annotations_per_user=max_annotations_per_user) for _ in range(n_samples)]
    train_dataset.databank = [data[0] for data in databank]
    test_dataset.databank = [data[1] for data in databank]
    train_dataset.data = train_dataset.databank[0]
    test_dataset.data = test_dataset.databank[0]
    return train_dataset, test_dataset

def create_user_split_kfold(dataset, k=4):
    root_dir = dataset.root_dir
    
    # Assuming 'participant_id' is a column in your dataset
    user_ids = dataset.data['participant_id'].unique()
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
    train_user_id = [int(x) for x in train_user_id]
    test_user_id = [int(x) for x in test_user_id]

    train_dataset.data = train_dataset.data[train_dataset.data['participant_id'].isin(train_user_id)]
    val_dataset.data = val_dataset.data[val_dataset.data['participant_id'].isin(train_user_id)]
    test_dataset.data = test_dataset.data[test_dataset.data['participant_id'].isin(test_user_id)]
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
    filtered_dataset = LAPIS_PIAADataset(dataset.root_dir, transform=dataset.transform)
    filtered_dataset.data = filtered_data
    return filtered_dataset

def create_image_split_dataset(dataset):
    # Saving the lists to TrainImageSet.txt and TestImageSet.txt
    root_dir = dataset.root_dir
    train_file_path = os.path.join(root_dir, 'TrainImageSet.txt')
    val_file_path = os.path.join(root_dir, 'ValImageSet.txt')
    test_file_path = os.path.join(root_dir, 'TestImageSet.txt')
    
    print('Read Image Set')
    with open(train_file_path, "r") as train_file:
        train_image_names = train_file.read().splitlines()
    with open(val_file_path, "r") as val_file:
        val_image_names = val_file.read().splitlines()            
    with open(test_file_path, "r") as test_file:
        test_image_names = test_file.read().splitlines()          

    train_dataset, val_dataset, test_dataset = copy.deepcopy(dataset), copy.deepcopy(dataset), copy.deepcopy(dataset)
    train_dataset.data = train_dataset.data[train_dataset.data['imageName'].isin(train_image_names)]
    val_dataset.data = val_dataset.data[val_dataset.data['imageName'].isin(val_image_names)]
    test_dataset.data = test_dataset.data[test_dataset.data['imageName'].isin(test_image_names)]
    return train_dataset, val_dataset, test_dataset

def plot_histogram_comparison(dataset):
    print(len(dataset))
    number_users = len(dataset.data['participant_id'].unique())
    number_image = len(dataset.data['imageName'].unique())
    len('Number users: %d'%number_users)
    len('Number images: %d'%number_image)
    number_image_per_user = [len(group) for _, group in dataset.data.groupby('participant_id')]
    number_user_per_image = [len(group) for _, group in dataset.data.groupby('imageName')]
    
    fig, axs = plt.subplots(2)
    fig.suptitle('LAPIS Dataset')  # This adds a main title to the figure
    axs[0].hist(number_image_per_user, bins=20)
    axs[0].set_xlabel('Number of Annotated Image per user')
    axs[0].set_ylabel('Frequency')
    axs[0].set_title('Histogram of Annotated image (min %d, max %d)'%(min(number_image_per_user), max(number_image_per_user)))
    axs[1].hist(number_user_per_image, bins=20)
    axs[1].set_xlabel('Number user per image')
    axs[1].set_ylabel('Frequency')
    axs[1].set_title('Histogram of User')
    axs[1].set_title('Histogram of User (min %d, max %d)'%(min(number_user_per_image), max(number_user_per_image)))
    plt.tight_layout()
    plt.savefig('LAPIS_histogram.jpg', dpi=300)

    root_dir = '/home/lwchen/datasets/PARA/'
    para_df = pd.read_csv(os.path.join(root_dir, 'annotation', 'PARA-Images.csv'))
    number_image_per_user = [len(group) for _, group in para_df.groupby('userId')]
    number_user_per_image = [len(group) for _, group in para_df.groupby('imageName')]
    
    fig, axs = plt.subplots(2)
    fig.suptitle('PARA dataset')
    axs[0].hist(number_image_per_user, bins=20)
    axs[0].set_xlabel('Number of Annotated Image per user')
    axs[0].set_ylabel('Frequency')
    axs[0].set_title('Histogram of Annotated image (min %d, max %d)'%(min(number_image_per_user), max(number_image_per_user)))
    axs[1].hist(number_user_per_image, bins=20)
    axs[1].set_xlabel('Number user per image')
    axs[1].set_ylabel('Frequency')
    axs[1].set_title('Histogram of User')
    axs[1].set_title('Histogram of User (min %d, max %d)'%(min(number_user_per_image), max(number_user_per_image)))
    plt.tight_layout()
    plt.savefig('PARA_histogram.jpg', dpi=300)


def load_data(args, root_dir = '/home/lwchen/datasets/LAPIS'):
# def load_data(args, root_dir = '/data/leuven/362/vsc36208/datasets/LAPIS'):
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
        test_dataset.data = test_dataset.data[test_dataset.data[args.trait] == args.value]
    
    test_dataset.transform = test_transform
    val_dataset.transform = test_transform
    print(len(train_dataset), len(val_dataset), len(test_dataset))
    
    return train_dataset, val_dataset, test_dataset


def collate_fn(batch):

    max_response_score = 10
    max_vaia_score = 7 
    traits_columns = ['nationality','demo_gender','demo_edu', 'demo_colorblind', 'age']
    
    batch_concatenated_traits = []
    batch_art_type = []
    batch_round_score = []
    for item in batch:
        torch.tensor(np.stack([item[f'VAIAK{i}'] for i in range(1, 8)]))

        vaiak1 = torch.from_numpy(np.stack([item[f'VAIAK{i}'] for i in range(1, 8)]))
        vaiak2 = torch.from_numpy(np.stack([item[f'2VAIAK{i}'] for i in range(1, 5)]))
        traits = torch.from_numpy(np.stack([item[trait] for trait in traits_columns]))
        concatenated_traits = torch.cat([traits,vaiak1,vaiak2])
        batch_concatenated_traits.append(concatenated_traits)
        batch_art_type.append(item['art_type'])
        batch_round_score.append(item['response'])

    return {
        'imgName':[item['imgName'] for item in batch],
        'image_id':[item['image_id'] for item in batch],
        'participant_id':[item['participant_id'] for item in batch],
        'imgName':[item['imgName'] for item in batch],
        'image': torch.stack([item['image'] for item in batch]),
        'aestheticScore': torch.from_numpy(np.stack(batch_round_score)),
        'traits': torch.stack(batch_concatenated_traits),
        'art_type':torch.from_numpy(np.stack(batch_art_type))
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training and Testing the Combined Model for data spliting')
    parser.add_argument('--trainset', type=str, default='GIAA', choices=["GIAA", "sGIAA", "PIAA"])
    # parser.add_argument('--fold_id', type=int, default=1)
    # parser.add_argument('--n_fold', type=int, default=4)
    parser.add_argument('--resume', type=str, default=None)
    # parser.add_argument('--use_cv', action='store_true', help='Enable cross validation')
    parser.add_argument('--is_eval', action='store_true', help='Enable evaluation mode')
    parser.add_argument('--eval_on_piaa', action='store_true', help='Evaluation metric on PIAA')
    parser.add_argument('--no_log', action='store_false', dest='is_log', help='Disable logging')
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--max_patience_epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--lr_schedule_epochs', type=int, default=5)
    parser.add_argument('--lr_decay_factor', type=float, default=0.5)
    parser.add_argument('--trait', type=str, default=None)
    parser.add_argument('--value', type=str, default=None)    
    args = parser.parse_args()
    
    n_workers = 4
    train_dataset, val_dataset, test_dataset = load_data(args)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=n_workers, timeout=300, collate_fn=collate_fn)
    for sample in test_dataloader:
        print(sample)
        raise Exception