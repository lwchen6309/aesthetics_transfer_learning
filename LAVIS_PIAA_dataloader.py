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


class LAVIS_PIAADataset(Dataset):
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
        required_fields = ['image_id', 'response', 'participant_id', 'nationality', 'demo_gender', 'demo_edu', 'demo_colorblind']
        self.art_interest_fields = [f'VAIAK{i}' for i in range(1, 8)] + [f'2VAIAK{i}' for i in range(1, 5)]
        required_fields += self.art_interest_fields
        self.data = data.dropna(subset=required_fields)
        # Encode text columna
        self.trait_columns = ['image_id', 'response', 'participant_id', *self.art_interest_fields]
        self.encoded_trait_columns = ['art_type', 'nationality', 'demo_gender', 'demo_edu', 'demo_colorblind']
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


def assert_score_to_GIAA(root_dir = '/home/lwchen/datasets/LAVIS'):
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

def split_dataset_by_images(dataset, train_img, testset_img):
    train_dataset, test_dataset = copy.deepcopy(dataset), copy.deepcopy(dataset)
    train_dataset.data = train_dataset.data[train_dataset.data['imageName'].isin(train_img)]
    test_dataset.data = test_dataset.data[test_dataset.data['imageName'].isin(testset_img)]
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
    filtered_dataset = LAVIS_PIAADataset(dataset.root_dir, transform=dataset.transform)
    filtered_dataset.data = filtered_data
    return filtered_dataset

def create_image_split_dataset(lavis_dataset):
    # Saving the lists to TrainImageSet.txt and TestImageSet.txt
    root_dir = lavis_dataset.root_dir
    train_file_path = os.path.join(root_dir, 'TrainImageSet.txt')
    test_file_path = os.path.join(root_dir, 'TestImageSet.txt')
    # Check if train file exists
    if os.path.exists(train_file_path) and os.path.exists(test_file_path):
        print('Read Image Set')
        with open(train_file_path, "r") as train_file:
            train_image_names = train_file.read().splitlines()
        with open(test_file_path, "r") as test_file:
            test_image_names = test_file.read().splitlines()          
    else:
        print('Compute Image Set')
        image_names = list(lavis_dataset.data['imageName'].unique())
        existing_image_names = [img for img in image_names if os.path.exists(os.path.join(lavis_dataset.root_dir, 'datasetImages_originalSize', img))]
        # Printing out the number of dropped files
        print('Total missing files: %d' % (len(image_names) - len(existing_image_names)))
        random.shuffle(image_names)
        # Splitting the list using a ratio of 10:1
        train_size = int(len(image_names) * 10 / 11)  # Calculate train size based on the 10:1 ratio
        train_image_names = image_names[:train_size]
        test_image_names = image_names[train_size:]
        with open(train_file_path, "w") as train_file:
            for name in train_image_names:
                train_file.write(name + "\n")
        with open(test_file_path, "w") as test_file:
            for name in test_image_names:
                test_file.write(name + "\n")

    train_dataset, test_dataset = split_dataset_by_images(lavis_dataset, train_image_names, test_image_names)   
    return train_dataset, test_dataset

def plot_histogram_comparison(lavis_dataset):
    print(len(lavis_dataset))
    number_users = len(lavis_dataset.data['participant_id'].unique())
    number_image = len(lavis_dataset.data['imageName'].unique())
    len('Number users: %d'%number_users)
    len('Number images: %d'%number_image)
    number_image_per_user = [len(group) for _, group in lavis_dataset.data.groupby('participant_id')]
    number_user_per_image = [len(group) for _, group in lavis_dataset.data.groupby('imageName')]
    
    fig, axs = plt.subplots(2)
    fig.suptitle('LAVIS Dataset')  # This adds a main title to the figure
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
    plt.savefig('LAVIS_histogram.jpg', dpi=300)

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



if __name__ == '__main__':
    root_dir = '/home/lwchen/datasets/LAVIS'
    # Define transformations for training set and test set
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
    ])
    
    from glob import glob
    lavis_dataset = LAVIS_PIAADataset(root_dir, transform=train_transform)
    print(len(lavis_dataset))
    # train_dataset, test_dataset = limit_annotations_per_user(lavis_dataset)
    image_names = set(lavis_dataset.data['imageName'].unique())
    image_exist_list = set([os.path.basename(file) for file in glob(os.path.join(lavis_dataset.root_dir, 'datasetImages_originalSize','*.jpg'))])
    print(image_exist_list - image_names)
    print(image_names - image_exist_list)
    
    train_dataset, test_dataset = create_image_split_dataset(lavis_dataset)
    
    # train_dataset, test_dataset = split_dataset_by_user(copy.deepcopy(lavis_dataset), max_annotations_per_user=[400, 50])
    print(len(train_dataset), len(test_dataset))
    train_dataloader = DataLoader(train_dataset, batch_size=32, num_workers=16, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=32, num_workers=16, shuffle=False)
    
    # Iterate over the training dataloader
    for sample in tqdm(train_dataloader):
        # Perform training operations here
        # print(sample)
        sample

    # Iterate over the training dataloader
    for sample in tqdm(test_dataloader):
        # Perform training operations here
        # print(sample)
        sample
