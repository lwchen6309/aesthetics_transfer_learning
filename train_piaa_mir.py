import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
# from torchvision.models import resnet50
import numpy as np
from tqdm import tqdm
from PARA_PIAA_dataloader import PARA_PIAADataset, collect_batch_attribute, collect_batch_personal_trait, split_dataset_by_user, split_dataset_by_images, limit_annotations_per_user
import wandb
from itertools import chain
from scipy.stats import spearmanr
from train_resnet_giaa import CombinedModel
import pandas as pd


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train(model, mlp1, mlp2, dataloader, criterion_mse, optimizer, device):
    model.train()
    running_mse_loss = 0.0

    progress_bar = tqdm(dataloader, leave=False)
    scale = torch.arange(1, 5.5, 0.5).to(device)    
    for sample in progress_bar:
        if is_eval:
            break
        
        images = sample['image']
        sample_score, sample_attr = collect_batch_attribute(sample)
        sample_pt = collect_batch_personal_trait(sample)
        images = images.to(device)
        sample_score = sample_score.to(device).float()
        sample_attr = sample_attr.to(device)
        sample_pt = sample_pt.to(device)
        batch_size = len(images)
        optimizer.zero_grad()

        logit, attr_mean_pred = model(images)
        prob = F.softmax(logit, dim=1)

        # Interation_map
        A_ij = attr_mean_pred.unsqueeze(2) * sample_pt.unsqueeze(1)
        I_ij = A_ij.view(batch_size,-1)
        y_ij = mlp1(I_ij) + mlp2(prob * scale)
        # y_ij = mlp1(I_ij) + torch.sum(prob * scale, dim=1, keepdim=True)
        
        # MSE loss
        loss = criterion_mse(y_ij, sample_score)

        loss.backward()
        optimizer.step()

        running_mse_loss += loss.item()

        progress_bar.set_postfix({
            'Train MSE Mean Loss': loss.item(),
        })

    epoch_mse_loss = running_mse_loss / len(dataloader)
    return epoch_mse_loss


def evaluate(model, mlp1, mlp2, dataloader, criterion_mse, device):
    model.eval()  # Set the model to evaluation mode
    running_mse_loss = 0.0

    all_predicted_scores = []  # List to store all predictions
    all_true_scores = []  # List to store all true scores

    progress_bar = tqdm(dataloader, leave=False)
    scale = torch.arange(1, 5.5, 0.5).to(device)
    for sample in progress_bar:
        with torch.no_grad():
            images = sample['image']
            sample_score, sample_attr = collect_batch_attribute(sample)
            sample_pt = collect_batch_personal_trait(sample)
            images = images.to(device)
            sample_score = sample_score.to(device).float()
            sample_attr = sample_attr.to(device)
            sample_pt = sample_pt.to(device)
            batch_size = len(images)

            logit, attr_mean_pred = model(images)
            prob = F.softmax(logit, dim=1)

            # Interaction_map
            A_ij = attr_mean_pred.unsqueeze(2) * sample_pt.unsqueeze(1)
            I_ij = A_ij.view(batch_size, -1)
            y_ij = mlp1(I_ij) + mlp2(prob * scale)

            # MSE loss
            loss = criterion_mse(y_ij, sample_score)
            running_mse_loss += loss.item()

            # Store predicted and true scores for SROCC calculation
            predicted_scores = y_ij.view(-1).cpu().numpy()
            true_scores = sample_score.view(-1).cpu().numpy()
            all_predicted_scores.extend(predicted_scores)
            all_true_scores.extend(true_scores)

            progress_bar.set_postfix({'Test MSE Mean Loss': loss.item()})

    epoch_mse_loss = running_mse_loss / len(dataloader)

    # Calculate SROCC for all predictions
    srocc, _ = spearmanr(all_predicted_scores, all_true_scores)

    return epoch_mse_loss, srocc
    

def load_data(root_dir = '/home/lwchen/datasets/PARA/'):
    # Define transformations for training set and test set
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

    # Create datasets with the appropriate transformations
    # train_dataset, test_dataset = split_dataset_by_images(
    #     PARA_PIAADataset(root_dir, transform=train_transform), 
    #     PARA_PIAADataset(root_dir, transform=test_transform), root_dir)

    # Create datasets with the appropriate transformations
    # train_dataset, _ = split_dataset_by_user(
    #     train_dataset, PARA_PIAADataset(root_dir, transform=test_transform),
    #     test_count=40, max_annotations_per_user=[10, 100])
    
    # train_dataset.data = limit_annotations_per_user(train_dataset.data, max_annotations_per_user=100)
    
    # user_ids_from = pd.read_csv('top30_user_ids.csv')['User ID'].tolist()
    user_ids_from = None
    train_dataset, test_dataset = split_dataset_by_user(
        PARA_PIAADataset(root_dir, transform=train_transform),
        test_count=40, max_annotations_per_user=[100, 50], seed=None, user_id_list=user_ids_from)

    return train_dataset, test_dataset


is_eval = False
is_log = True
num_bins = 9
num_attr = 8
num_pt = 25 # number of personal trait
pretrained_rn = 'best_model_resnet50_giaa_hidden512_lr5e-05_decay_20epoch_twilight-dream-273.pth'


if __name__ == '__main__':
    lr = 5e-5
    batch_size = 100
    num_epochs = 5
    if is_log:
        wandb.init(project="resnet_PARA_GIAA",
                notes="PIAA-MIR",
                tags=['Train user sample'])
        wandb.config = {
            "learning_rate": lr,
            "batch_size": batch_size,
            "num_epochs": num_epochs
        }
        experiment_name = wandb.run.name
    else:
        experiment_name = ''
    
    # Create dataloaders for training and test sets
    train_dataset, test_dataset = load_data()
    n_workers = 4
    sroccs = []
    # for idx in range(10):
    for idx in range(10):
        train_dataset.data = train_dataset.databank[idx]
        test_dataset.data = test_dataset.databank[idx]

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers)

        # Define the number of classes in your dataset
        num_classes = num_attr + num_bins
        # Define the device for training
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = CombinedModel(num_bins, num_attr, 512).to(device)
        model.load_state_dict(torch.load(pretrained_rn))

        # Define two MLPs
        d_interactio = num_attr * num_pt
        mlp1 = MLP(d_interactio, 1024, 1)
        mlp2 = MLP(num_bins, 1024, 1)
        
        # Move the model to the device
        model = model.to(device)
        mlp1 = mlp1.to(device)
        mlp2 = mlp2.to(device)
        
        # Define the loss functions
        criterion_mse = nn.MSELoss()

        # Define the optimizer
        optimizer = optim.Adam(chain([*model.parameters(), *mlp1.parameters(), *mlp2.parameters()]), lr=lr)
        # optimizer = optim.Adam(chain([*mlp1.parameters(), *mlp2.parameters()]), lr=lr)

        # Initialize the best test loss and the best model
        best_model = None
        best_modelname = 'best_model_resnet50_piaamir_lr%1.0e_decay_%depoch' % (lr, num_epochs)
        best_modelname += '_%s'%experiment_name
        best_modelname1 = best_modelname + '_mlp1.pth'
        best_modelname2 = best_modelname + '_mlp1.pth'
        # best_modelname += '.pth'

        # Training loop
        lr_schedule_epochs = 5
        lr_decay_factor = 0.5
        max_patience_epochs = 10
        num_patience_epochs = 0
        best_test_loss = float('inf')
        for epoch in range(num_epochs):
            # Learning rate schedule
            if (epoch + 1) % lr_schedule_epochs == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= lr_decay_factor
            # Training
            train_mse_loss = train(model, mlp1, mlp2, train_dataloader, criterion_mse, optimizer, device)
            if is_log:
                wandb.log({"Train PIAA MSE Loss": train_mse_loss,}, commit=True)

        # Testing
        test_mse_loss, test_srocc = evaluate(model, mlp1, mlp2, test_dataloader, criterion_mse, device)
        if is_log:
            wandb.log({"Test user PIAA MSE Loss": test_mse_loss,
                    "Test user PIAA SROCC": test_srocc}, commit=True)
        

        # Print the epoch loss
        print(f"Epoch [{epoch + 1}/{num_epochs}], "
            # f"Train PIAA MSE Loss: {train_mse_loss:.4f}, "
            f"Test user PIAA MSE Loss: {test_mse_loss:.4f}, "
            f"Test user PIAA SROCC Loss: {test_srocc:.4f}, ")
            
        sroccs.append(test_srocc)
    sroccs = np.array(sroccs)
    print(sroccs.mean(), sroccs.std())
