import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet50
import numpy as np
from tqdm import tqdm
import wandb
from itertools import chain
from scipy.stats import spearmanr
from PARA_histogram_dataloader import PARA_HistogramDataset, PARA_GIAA_HistogramDataset, PARA_PIAA_HistogramDataset
from PARA_PIAA_dataloader import PARA_PIAADataset, split_dataset_by_user, split_dataset_by_images, generate_data_per_user
import matplotlib.pyplot as plt
from copy import deepcopy
import pandas as pd


# Evaluation Function
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_emd_loss = 0.0
    running_mse_loss = 0.0
    running_srocc = 0.0
    scale = torch.arange(1, 5.5, 0.5).to(device)
    eval_srocc = True
    criterion_mse = nn.MSELoss()

    progress_bar = tqdm(dataloader, leave=False)
    mean_pred = []
    mean_target = []
    with torch.no_grad():
        for sample in progress_bar:
            images = sample['image'].to(device)
            aesthetic_score_histogram = sample['aestheticScore'].to(device)
            attributes_histogram = sample['attributes'].to(device)
            traits_histogram = sample['traits'].to(device)
            onehot_traits_histogram = sample['onehot_traits'].to(device)
            traits_histogram = torch.cat([traits_histogram, onehot_traits_histogram], dim=1)
            
            logits = model(images)
            prob = F.softmax(logits, dim=-1)
            
            if eval_srocc:
                outputs_mean = torch.sum(prob * scale, dim=1, keepdim=True)
                target_mean = torch.sum(aesthetic_score_histogram * scale, dim=1, keepdim=True)
                mean_pred.append(outputs_mean.view(-1).cpu().numpy())
                mean_target.append(target_mean.view(-1).cpu().numpy())
                # MSE
                mse = criterion_mse(outputs_mean, target_mean)
                running_mse_loss += mse.item()

            loss = criterion(prob, aesthetic_score_histogram).mean()
            running_emd_loss += loss.item()
            progress_bar.set_postfix({
                'Test EMD Loss': loss.item(),
            })

    # Calculate SROCC
    predicted_scores = np.concatenate(mean_pred, axis=0)
    true_scores = np.concatenate(mean_target, axis=0)
    srocc, _ = spearmanr(predicted_scores, true_scores)

    emd_loss = running_emd_loss / len(dataloader)
    mse_loss = running_mse_loss / len(dataloader)
    return emd_loss, srocc, mse_loss

def earth_mover_distance(x, y, dim=-1):
    """
    Compute Earth Mover's Distance (EMD) between two 1D tensors x and y using 2-norm.
    """
    cdf_x = torch.cumsum(x, dim=dim)
    cdf_y = torch.cumsum(y, dim=dim)
    emd = torch.norm(cdf_x - cdf_y, p=2, dim=dim)
    return emd


# Model Definition
class CombinedModel(nn.Module):
    def __init__(self, num_bins, num_attr, num_pt, resume = None):
        super(CombinedModel, self).__init__()

        self.resnet = resnet50(pretrained=True)
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 512),
            nn.ReLU(),
            nn.Linear(512, num_bins),
        )
        if resume is not None:
            self.resnet.load_state_dict(torch.load(resume))
    
    def forward(self, images):
        x = self.resnet(images)
        return x


def load_data():
    root_dir = '/home/lwchen/datasets/PARA/'
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

    # Load datasets
    # Create datasets with the appropriate transformations
    train_piaa_dataset = PARA_PIAADataset(root_dir, transform=train_transform)
    test_piaa_dataset = PARA_PIAADataset(root_dir, transform=train_transform)
    train_dataset, test_dataset = split_dataset_by_images(train_piaa_dataset, test_piaa_dataset, root_dir)
    
    user_ids_from = None
    user_ids_from = pd.read_csv('top25_user_ids.csv')['User ID'].tolist()
    train_user_piaa_dataset, test_user_piaa_dataset = split_dataset_by_user(
        PARA_PIAADataset(root_dir, transform=train_transform),
        test_count=40, max_annotations_per_user=[100, 50], seed=None, user_id_list=user_ids_from)
    
    all_each_user_piaa_dataset = generate_data_per_user(
        PARA_PIAADataset(root_dir, transform=train_transform), 
        max_annotations_per_user=50)

    testimg_each_user_piaa_dataset = generate_data_per_user(
        deepcopy(test_dataset), 
        max_annotations_per_user=3000)

    # Create datasets with the appropriate transformations
    # train_dataset = PARA_HistogramDataset(root_dir, transform=train_transform, data=train_dataset.data, map_file='trainset_image_dct.pkl')
    # test_dataset = PARA_HistogramDataset(root_dir, transform=test_transform, data=test_dataset.data, map_file='testset_image_dct.pkl')
    print(len(train_dataset), len(test_dataset))
    pkl_dir = './dataset_pkl'
    train_dataset = PARA_GIAA_HistogramDataset(root_dir, transform=train_transform, data=train_piaa_dataset.data, map_file=os.path.join(pkl_dir,'trainset_image_dct.pkl'), precompute_file=os.path.join(pkl_dir,'trainset_GIAA_dct.pkl'))
    test_giaa_dataset = PARA_GIAA_HistogramDataset(root_dir, transform=test_transform, data=test_piaa_dataset.data, map_file=os.path.join(pkl_dir,'testset_image_dct.pkl'), precompute_file=os.path.join(pkl_dir,'testset_GIAA_dct.pkl'))
    test_piaa_dataset = PARA_PIAA_HistogramDataset(root_dir, transform=test_transform, data=test_dataset.data)
    train_user_piaa_dataset = [PARA_PIAA_HistogramDataset(root_dir, transform=test_transform, data=data) for data in train_user_piaa_dataset.databank]
    test_user_piaa_dataset = [PARA_PIAA_HistogramDataset(root_dir, transform=test_transform, data=data) for data in test_user_piaa_dataset.databank]
    # train_user_piaa_dataset = PARA_PIAA_HistogramDataset(root_dir, transform=test_transform, data=train_user_piaa_dataset.data)
    # test_user_piaa_dataset = PARA_PIAA_HistogramDataset(root_dir, transform=test_transform, data=train_user_piaa_dataset.data)
    return train_dataset, test_giaa_dataset, test_piaa_dataset, train_user_piaa_dataset, test_user_piaa_dataset, all_each_user_piaa_dataset, testimg_each_user_piaa_dataset

def evaluate_emd(model, dataloader, criterion, device):
    model.eval()
    running_mse_loss = 0.0
    running_mse_emd_loss = 0.0
    scale = torch.arange(1, 5.5, 0.5).to(device)
    criterion_mse = nn.MSELoss()
    progress_bar = tqdm(dataloader, leave=False)
    mean_pred = []
    mean_target = []
    emd_losses = []  # List to collect emd_loss for all data
    semantic_values = []

    for sample in progress_bar:
        images = sample['image'].to(device)
        aesthetic_score_histogram = sample['aestheticScore'].to(device)
        attributes_histogram = sample['attributes'].to(device)
        traits_histogram = sample['traits'].to(device)
        onehot_traits_histogram = sample['onehot_traits'].to(device)
        traits_histogram = torch.cat([traits_histogram, onehot_traits_histogram], dim=1)
        semantic_value = torch.argmax(sample['semantic'], dim=1).cpu().numpy()
        semantic_values.append(semantic_value)

        with torch.no_grad():
            logits = model(images)
            prob = F.softmax(logits, dim=-1)

            outputs_mean = torch.sum(prob * scale, dim=1, keepdim=True)
            target_mean = torch.sum(aesthetic_score_histogram * scale, dim=1, keepdim=True)
            mean_pred.append(outputs_mean.view(-1).cpu().numpy())
            mean_target.append(target_mean.view(-1).cpu().numpy())
            
            # MSE
            mse = criterion_mse(outputs_mean, target_mean)
            running_mse_loss += mse.item()

            # EMD Loss
            emd_loss = criterion(prob, aesthetic_score_histogram)
            emd_losses.append(emd_loss.view(-1).cpu().numpy())  # Append the emd_loss for this sample to the list
            running_mse_emd_loss += torch.mean(emd_loss)
    
    emd_losses = np.concatenate(emd_losses, axis=0)
    semantic_values = np.concatenate(semantic_values, axis=0)
    avg_semantic_emd_losses = compute_mean_emd_for_semantics(emd_losses, semantic_values)

    # Calculate SROCC
    predicted_scores = np.concatenate(mean_pred, axis=0)
    true_scores = np.concatenate(mean_target, axis=0)
    srocc, _ = spearmanr(predicted_scores, true_scores)

    mse_loss = running_mse_loss / len(dataloader)
    mse_emd_loss = running_mse_emd_loss / len(dataloader)
    
    return srocc, mse_loss, mse_emd_loss, emd_losses, avg_semantic_emd_losses

def compute_mean_emd_for_semantics(emd_losses, semantic_values):
    # Ensure they have the same length
    assert len(emd_losses) == len(semantic_values), "Mismatched lengths!"

    # Dictionary to collect emd_loss for each unique semantic value
    semantic_emd_losses = {}

    for semantic, emd_loss in zip(semantic_values, emd_losses):
        if semantic not in semantic_emd_losses:
            semantic_emd_losses[semantic] = []
        semantic_emd_losses[semantic].append(emd_loss)

    # Calculate average EMD loss for each semantic value
    avg_semantic_emd_losses = {k: np.mean(v) for k, v in semantic_emd_losses.items()}

    return avg_semantic_emd_losses

def analyze_histograms(dataset, emd_losses, ratio=0.3):
    # Get the indices of the top n% of the emd_losses, in descending order.
    topn_len = int(ratio * len(emd_losses))
    sorted_indices = sorted(range(len(emd_losses)), key=lambda k: emd_losses[k], reverse=True)
    topn_indices = sorted_indices[:topn_len]

    # Initialize histograms to zeros
    total_aesthetic_score_histogram = torch.zeros_like(dataset[0]['aestheticScore'])
    total_attributes_histogram = torch.zeros_like(dataset[0]['attributes'])
    total_traits_histogram = torch.zeros_like(dataset[0]['traits'])
    total_onehot_traits_histogram = torch.zeros_like(dataset[0]['onehot_traits'])
    total_semantic_histogram = torch.zeros_like(dataset[0]['semantic'])

    topn_aesthetic_score_histogram = torch.zeros_like(dataset[0]['aestheticScore'])
    topn_attributes_histogram = torch.zeros_like(dataset[0]['attributes'])
    topn_traits_histogram = torch.zeros_like(dataset[0]['traits'])
    topn_onehot_traits_histogram = torch.zeros_like(dataset[0]['onehot_traits'])
    topn_semantic_histogram = torch.zeros_like(dataset[0]['semantic'])

    # Iterate over the dataset to accumulate histograms
    for i, sample in enumerate(dataset):
        total_aesthetic_score_histogram += sample['aestheticScore']
        total_attributes_histogram += sample['attributes']
        total_traits_histogram += sample['traits']
        total_onehot_traits_histogram += sample['onehot_traits']
        total_semantic_histogram += sample['semantic']
    
        # Check if the current batch index is in the top 30% indices
        if i in topn_indices:
            topn_aesthetic_score_histogram += sample['aestheticScore']
            topn_attributes_histogram += sample['attributes']
            topn_traits_histogram += sample['traits']
            topn_onehot_traits_histogram += sample['onehot_traits']
            topn_semantic_histogram += sample['semantic']

    results = {
        "All Data": {
            "Aesthetic Score Histogram": total_aesthetic_score_histogram / len(dataset),
            "Attributes Histogram": total_attributes_histogram / len(dataset),
            "Traits Histogram": total_traits_histogram / len(dataset),
            "Onehot Traits Histogram": total_onehot_traits_histogram / len(dataset),
            "Semantic Histogram": total_semantic_histogram / len(dataset),
        },
        "Top %d%% EMD"%(int(ratio*100)): {
            "Aesthetic Score Histogram": topn_aesthetic_score_histogram / topn_len,
            "Attributes Histogram": topn_attributes_histogram / topn_len,
            "Traits Histogram": topn_traits_histogram / topn_len,
            "Onehot Traits Histogram": topn_onehot_traits_histogram / topn_len,
            "Semantic Histogram": topn_semantic_histogram / topn_len,
        }
    }

    return results

def visualize_difference_trait(results, savefig=True):
    def parse_subset(subset):
        data = results[subset]
        score = data['Aesthetic Score Histogram']
        # attr = data['Attributes Histogram'].reshape(8,5) # assuming reshaping is needed
        trait = data['Onehot Traits Histogram'] 
        # onehot_big5 = data['Traits Histogram'].reshape(5,10) # assuming reshaping is needed
        age = trait[:5]
        gender = trait[5:7]
        EducationalLevel = trait[7:12]
        artExperience = trait[12:16]
        photographyExperience = trait[16:20]
        return score, age, gender, EducationalLevel, artExperience, photographyExperience

    subsets = list(results.keys())
    fig, axes = plt.subplots(3, 6, figsize=(18, 10))
    
    set1 = parse_subset(subsets[0])
    set2 = parse_subset(subsets[1])
    subsets_diff = [a - b for a, b in zip(set2, set1)]
    
    titles = ["score", "age", "gender", "EducationalLevel", "artExperience", "photographyExperience"]

    for axis, item, title in zip(axes[0], set1, titles):
        axis.bar(range(len(item)), item)
        axis.set_title(title)
        axis.set_xticks([])  # Hide x-axis tick labels
    for axis, item in zip(axes[1], set2):
        axis.bar(range(len(item)), item)
        axis.set_xticks([])  # Hide x-axis tick labels
    for axis, item in zip(axes[2], subsets_diff):
        axis.bar(range(len(item)), item)
        axis.set_xticks([])  # Hide x-axis tick labels

    # Setting the row labels (titles) for the left-most column
    axes[0][0].set_ylabel(subsets[0], fontsize=14, fontweight='bold')
    axes[1][0].set_ylabel(subsets[1], fontsize=14, fontweight='bold')
    axes[2][0].set_ylabel("Diff", fontsize=14, fontweight='bold')

    plt.tight_layout()
    if savefig:
        plt.savefig('trait_diff.jpg', dpi=300)

def visualize_difference_attr(results, savefig=True):
    def parse_subset_attr(subset):
        data = results[subset]
        attr = data['Attributes Histogram'].reshape(8,5)  # assuming reshaping is needed
        return attr

    subsets = list(results.keys())
    fig, axes = plt.subplots(3, 8, figsize=(20, 10))  # 8 subplots for attr
    titles = ['quality','composition','color','dof','content','light','contentPreference','willingnessToShare']
    
    set1 = parse_subset_attr(subsets[0])
    set2 = parse_subset_attr(subsets[1])
    subsets_diff = [a - b for a, b in zip(set2, set1)]

    for axis, item, title in zip(axes[0], set1, titles):
        axis.bar(range(len(item)), item)
        axis.set_title(title)
        axis.set_xticks([])  # Hide x-axis tick labels
    for axis, item in zip(axes[1], set2):
        axis.bar(range(len(item)), item)
        axis.set_xticks([])  # Hide x-axis tick labels
    for axis, item in zip(axes[2], subsets_diff):
        axis.bar(range(len(item)), item)
        axis.set_xticks([])  # Hide x-axis tick labels

    # Setting the row labels (titles) for the left-most column
    axes[0][0].set_ylabel(subsets[0], fontsize=14, fontweight='bold')
    axes[1][0].set_ylabel(subsets[1], fontsize=14, fontweight='bold')
    axes[2][0].set_ylabel("Diff", fontsize=14, fontweight='bold')

    plt.tight_layout()
    if savefig:
        plt.savefig('attr_diff.jpg', dpi=300)

def visualize_difference_big5(results, savefig=True):
    def parse_subset_big5(subset):
        data = results[subset]
        onehot_big5 = data['Traits Histogram'].reshape(5,10)  # assuming reshaping is needed
        return onehot_big5

    subsets = list(results.keys())
    fig, axes = plt.subplots(3, 5, figsize=(15, 10))  # 5 subplots for onehot_big5
    titles = ['E','A','N','O','C']
    
    set1 = parse_subset_big5(subsets[0])
    set2 = parse_subset_big5(subsets[1])
    subsets_diff = [a - b for a, b in zip(set2, set1)]

    for axis, item, title in zip(axes[0], set1, titles):
        axis.bar(range(len(item)), item)
        axis.set_xticks([])  # Hide x-axis tick labels
        axis.set_title(title)
    for axis, item in zip(axes[1], set2):
        axis.bar(range(len(item)), item)
        axis.set_xticks([])  # Hide x-axis tick labels
    for axis, item in zip(axes[2], subsets_diff):
        axis.bar(range(len(item)), item)
        axis.set_xticks([])  # Hide x-axis tick labels

    # Setting the row labels (titles) for the left-most column
    axes[0][0].set_ylabel(subsets[0], fontsize=14, fontweight='bold')
    axes[1][0].set_ylabel(subsets[1], fontsize=14, fontweight='bold')
    axes[2][0].set_ylabel("Diff", fontsize=14, fontweight='bold')

    plt.tight_layout()
    if savefig:
        plt.savefig('big5_diff.jpg', dpi=300)

def visualize_difference_semantic(results, savefig=True):
    def parse_subset_semantic(subset):
        data = results[subset]
        semantic = data['Semantic Histogram']
        return semantic

    subsets = list(results.keys())
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))  # 3 rows, 1 column
    xlabel = ['animal', 'stilllife', 'portrait', 'plant', 'scene',
              'indoor', 'others', 'nightScene', 'food', 'building']

    set1 = parse_subset_semantic(subsets[0])
    set2 = parse_subset_semantic(subsets[1])
    subsets_diff = set2 - set1

    axes[0].bar(range(len(set1)), set1)
    axes[0].set_xticks([])  # Hide x-axis tick labels
    axes[0].set_ylabel(subsets[0], fontsize=14, fontweight='bold')

    axes[1].bar(range(len(set2)), set2)
    axes[1].set_xticks([])  # Hide x-axis tick labels
    axes[1].set_ylabel(subsets[1], fontsize=14, fontweight='bold')

    axes[2].bar(range(len(subsets_diff)), subsets_diff)
    axes[2].set_xticks(range(len(subsets_diff)))  # Set x-ticks
    axes[2].set_xticklabels(xlabel, rotation=45, ha='right')  # Set x-labels
    axes[2].set_ylabel("Diff", fontsize=14, fontweight='bold')

    plt.tight_layout()
    if savefig:
        plt.savefig('semantic_diff.jpg', dpi=300)

def evaluate_user_datasets(user_datasets, model, earth_mover_distance, device, batch_size, n_workers, pin_memory):
    results = []
    for user_id, user_dataset in user_datasets:
        test_user_piaa_dataset = PARA_PIAA_HistogramDataset(root_dir, transform=test_transform, data=user_dataset.data)
        user_dataloader = DataLoader(test_user_piaa_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers, pin_memory=pin_memory, timeout=300)
        user_emd_loss, user_srocc, user_mse = evaluate(model, user_dataloader, earth_mover_distance, device)
        results.append((user_id, user_srocc))
    return results


is_eval = True
is_log = False
num_bins = 9
num_attr = 40
num_pt = 50 + 20
resume = None
resume = 'best_model_resnet50_hidden512_cls_lr5e-05_decay_20epoch_noattr_distinctive-glade-243.pth'
root_dir = '/home/lwchen/datasets/PARA/'
test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])


def eval_user_piaa():
    batch_size = 100
    train_dataset, test_giaa_dataset, test_piaa_dataset, train_user_piaa_dataset, test_user_piaa_dataset, all_each_user_piaa_dataset, testimg_each_user_piaa_dataset = load_data()
    n_workers = 4
    
    pin_memory = False
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers, pin_memory=pin_memory, timeout=300)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Initialize the combined model
    model = CombinedModel(num_bins, num_attr, num_pt, resume = resume).to(device)
    # Loss and optimizer

    # Testing   
    scores = []
    for dataset in test_user_piaa_dataset:
        test_user_piaa_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers, pin_memory=pin_memory, timeout=300)
        _, test_user_piaa_srocc, _ = evaluate(model, test_user_piaa_dataloader, earth_mover_distance, device)
        scores.append(test_user_piaa_srocc)
    return scores


if __name__ == '__main__':
    scroccs_users = []
    for i in range(10):
        scroccs = eval_user_piaa()
        mean_scrocc = np.mean(np.array(scroccs))
        scroccs_users.append(mean_scrocc)
    scroccs_users = np.array(scroccs_users)
    print(scroccs_users.mean(), scroccs_users.std())
    raise Exception

    # results = np.array([eval_user_piaa() for _ in range(10)])
    # tag = 'top20'
    # np.savez('testuser_piaa_%s.npz'%tag, results=results)
    # with open('results_%s.csv'%tag, 'w') as csvfile:
    #     # Write the header
    #     csvfile.write('SROCC\n')
    #     # Write each result on a new line
    #     for result in results:
    #         csvfile.write(f'{result}\n')

    random_seed = 42
    lr = 5e-5
    batch_size = 100
    num_epochs = 1
    lr_schedule_epochs = 5
    lr_decay_factor = 0.5
    max_patience_epochs = 10
    
    if is_log:
        wandb.init(project="resnet_PARA_PIAA")
        wandb.config = {
            "learning_rate": lr,
            "batch_size": batch_size,
            "num_epochs": num_epochs
        }
        experiment_name = wandb.run.name
    else:
        experiment_name = ''
    
    train_dataset, test_giaa_dataset, test_piaa_dataset, train_user_piaa_dataset, test_user_piaa_dataset, all_each_user_piaa_dataset, testimg_each_user_piaa_dataset = load_data()
    n_workers = 8
    
    pin_memory = False
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers, pin_memory=pin_memory, timeout=300)
    test_giaa_dataloader = DataLoader(test_giaa_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers, pin_memory=pin_memory, timeout=300)
    test_piaa_dataloader = DataLoader(test_piaa_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers, pin_memory=pin_memory, timeout=300)
    train_user_piaa_dataloader = DataLoader(train_user_piaa_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers, pin_memory=pin_memory, timeout=300)
    test_user_piaa_dataloader = DataLoader(test_user_piaa_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers, pin_memory=pin_memory, timeout=300)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the combined model
    model = CombinedModel(num_bins, num_attr, num_pt, resume = resume).to(device)
    # Loss and optimizer
    # criterion_mse = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Initialize the best test loss and the best model
    best_model = None
    best_modelname = 'best_model_resnet50_histo_lr%1.0e_decay_%depoch' % (lr, num_epochs)
    best_modelname += '_%s'%experiment_name
    best_modelname += '.pth'

    srocc, mse_loss, mse_emd_loss, emd_losses, avg_semantic_emd_losses = evaluate_emd(model, test_giaa_dataloader, earth_mover_distance, device)
    # Sorting the data for clarity
    sorted_keys = sorted(avg_semantic_emd_losses.keys())
    sorted_values = [avg_semantic_emd_losses[key] for key in sorted_keys]
    
    plot_diff = False
    if plot_diff:
        savefig = True
        results = analyze_histograms(test_giaa_dataset, emd_losses, ratio=0.3)
        # Plot image
        plt.bar(sorted_keys, sorted_values)
        plt.xlabel('Semantic Value')
        plt.ylabel('Average EMD Loss')
        plt.title('Average EMD Loss per Semantic Value')
        plt.xticks(list(range(10)))  # x-axis ticks from 0 to 9
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        xlabel = ['animal', 'stilllife', 'portrait', 'plant', 'scene',
                'indoor', 'others', 'nightScene', 'food', 'building']
        plt.gca().set_xticklabels(xlabel, rotation=45, ha='right')  # Set x-labels

        # Plot trait
        visualize_difference_trait(results, savefig=savefig)
        visualize_difference_attr(results, savefig=savefig)
        visualize_difference_big5(results, savefig=savefig)
        visualize_difference_semantic(results, savefig=savefig)
        fig = plt.figure()
        plt.hist(emd_losses, bins=40)
        plt.xlabel('EMD loss')
        plt.show()
    
    # raise Exception
    # _, test_user_piaa_srocc, _ = evaluate(model, test_user_piaa_dataloader, earth_mover_distance, device)
    # all_user_results = evaluate_user_datasets(all_each_user_piaa_dataset, model, earth_mover_distance, device, batch_size, n_workers, pin_memory)
    # testimg_user_results = evaluate_user_datasets(testimg_each_user_piaa_dataset, model, earth_mover_distance, device, batch_size, n_workers, pin_memory)    

    # Testing
    _, test_giaa_srocc, _ = evaluate(model, test_giaa_dataloader, earth_mover_distance, device)
    _, test_piaa_srocc, _ = evaluate(model, test_piaa_dataloader, earth_mover_distance, device)
    _, train_user_piaa_srocc, _ = evaluate(model, train_user_piaa_dataloader, earth_mover_distance, device)
    _, test_user_piaa_srocc, _ = evaluate(model, test_user_piaa_dataloader, earth_mover_distance, device)

    # Print the epoch loss
    print(
        f"Test GIAA SROCC Loss: {test_giaa_srocc:.4f}, "
        f"Test PIAA SROCC Loss: {test_piaa_srocc:.4f}, "
        f"Test user PIAA SROCC Loss: {test_user_piaa_srocc:.4f}, "
        f"Train user PIAA SROCC Loss: {train_user_piaa_srocc:.4f}, "
        )

    