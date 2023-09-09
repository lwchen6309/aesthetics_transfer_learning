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
from PARA_PIAA_dataloader import PARA_PIAADataset, collect_batch_attribute, collect_batch_personal_trait, split_dataset_by_user
import wandb
from itertools import chain
from scipy.stats import spearmanr


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

        outputs = model(images)
        logit = outputs[:,:num_bins]
        attr_mean_pred = outputs[:,num_bins:]
        prob = F.softmax(logit, dim=1)

        # Interation_map
        A_ij = attr_mean_pred.unsqueeze(2) * sample_pt.unsqueeze(1)
        I_ij = A_ij.view(batch_size,-1)
        y_ij = mlp1(I_ij) + mlp2(prob * scale)
        # y_ij = y_ij + torch.sum(prob * scale, dim=1, keepdim=True)
        
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
    model.train()
    running_mse_loss = 0.0
    running_srocc_loss = 0.0

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

            outputs = model(images)
            logit = outputs[:,:num_bins]
            attr_mean_pred = outputs[:,num_bins:]
            prob = F.softmax(logit, dim=1)

            # Interation_map
            A_ij = attr_mean_pred.unsqueeze(2) * sample_pt.unsqueeze(1)
            I_ij = A_ij.view(batch_size,-1)
            y_ij = mlp1(I_ij) + mlp2(prob * scale)
            # y_ij = y_ij + torch.sum(prob * scale, dim=1, keepdim=True)

            # MSE loss
            loss = criterion_mse(y_ij, sample_score)          

            running_mse_loss += loss.item()

            # Calculate SROCC
            predicted_scores = y_ij.view(-1).cpu().numpy()
            true_scores = sample_score.view(-1).cpu().numpy()
            srocc, _ = spearmanr(predicted_scores, true_scores)
            running_srocc_loss += srocc

            progress_bar.set_postfix({
                'Test MSE Mean Loss': loss.item(),
                'Test SROCC': srocc,
            })

    epoch_mse_loss = running_mse_loss / len(dataloader)
    epoch_srocc = running_srocc_loss / len(dataloader)
    return epoch_mse_loss, epoch_srocc


is_eval = False
is_log = True
num_bins = 9
num_attr = 8
num_pt = 25 # number of personal trait
pretrained_rn = 'best_model_resnet50_giaa_lr5e-05_decay_20epoch.pth'


if __name__ == '__main__':
    # Set random seed for reproducibility
    # random_seed = 42
    # torch.manual_seed(random_seed)
    # torch.cuda.manual_seed(random_seed)
    # np.random.seed(random_seed)
    # random.seed(random_seed)

    lr = 1e-5
    batch_size = 100
    num_epochs = 5
    if is_log:
        wandb.init(project="resnet_PARA_GIAA")
        wandb.config = {
            "learning_rate": lr,
            "batch_size": batch_size,
            "num_epochs": num_epochs
        }
        experiment_name = wandb.run.name
    else:
        experiment_name = ''
    
    # Define the root directory of the PARA dataset
    root_dir = '/home/lwchen/datasets/PARA/'

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
    train_dataset = PARA_PIAADataset(root_dir, transform=train_transform)
    test_dataset = PARA_PIAADataset(root_dir, transform=test_transform)
    train_dataset, test_dataset = split_dataset_by_user(train_dataset, test_dataset, 
        test_count=40, max_annotations_per_user=100)

    # Create dataloaders for training and test sets
    n_workers = 4
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers)

    # Define the number of classes in your dataset
    num_classes = num_attr + num_bins
    # Define the device for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the pre-trained ResNet model
    model_resnet50 = resnet50(pretrained=True)

    # Modify the last fully connected layer to match the number of classes
    num_features = model_resnet50.fc.in_features
    model_resnet50.fc = nn.Sequential(
        nn.Linear(num_features, 1024),
        nn.ReLU(),
        nn.Linear(1024, num_classes),  # Output for the first task (num_classes)
    )
    if pretrained_rn is not None:
        model_resnet50.load_state_dict(torch.load(pretrained_rn))
    
    # Define two MLPs
    d_interactio = num_attr * num_pt
    mlp1 = MLP(d_interactio, 1024, 1)
    mlp2 = MLP(num_bins, 1024, 1)

    # Move the model to the device
    model_resnet50 = model_resnet50.to(device)
    mlp1 = mlp1.to(device)
    mlp2 = mlp2.to(device)
    
    # Define the loss functions
    criterion_mse = nn.MSELoss()

    # Define the optimizer
    optimizer = optim.Adam(chain([*model_resnet50.parameters(), *mlp1.parameters(), *mlp2.parameters()]), lr=lr)
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
        train_mse_loss = train(model_resnet50, mlp1, mlp2, train_dataloader, criterion_mse, optimizer, device)
        if is_log:
            wandb.log({"Train MSE Loss": train_mse_loss,}, commit=False)

        # Testing
        test_mse_loss, test_srocc = evaluate(model_resnet50, mlp1, mlp2, test_dataloader, criterion_mse, device)
        if is_log:
            wandb.log({"Test MSE Loss": train_mse_loss,
                       "Test SROCC": test_srocc}, commit=True)

        # Print the epoch loss
        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train MSE Loss: {train_mse_loss:.4f}, "
              f"Test MSE Loss: {test_mse_loss:.4f}, "
              f"Test SROCC Loss: {test_srocc:.4f}, ")
        if is_eval:
            raise Exception

        # Early stopping check
        if test_mse_loss < best_test_loss:
            best_test_loss = test_mse_loss
            num_patience_epochs = 0
            # torch.save(model_resnet50.state_dict(), best_modelname)
            torch.save(mlp1.state_dict(), best_modelname1)
            torch.save(mlp2.state_dict(), best_modelname2)
        else:
            num_patience_epochs += 1
            if num_patience_epochs >= max_patience_epochs:
                print("Validation loss has not decreased for {} epochs. Stopping training.".format(max_patience_epochs))
                break