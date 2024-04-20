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
from PARA_PIAA_dataloader import PARA_PIAADataset, collect_batch_attribute, collect_batch_personal_trait, split_dataset_by_user, split_dataset_by_images, create_user_split_dataset_kfold
import wandb
from itertools import chain
from scipy.stats import spearmanr
# from train_resnet_giaa import CombinedModel
from train_nima_attr import NIMA_attr
import pandas as pd
import argparse


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CombinedModel(nn.Module):
    def __init__(self, num_bins, num_attr, num_pt, hidden_size=1024):
        super(CombinedModel, self).__init__()
        self.num_bins = num_bins
        self.num_attr = num_attr
        self.num_pt = num_pt
        self.scale = torch.arange(1, 5.5, 0.5)  # This will be moved to the correct device in forward()
        
        # Placeholder for the NIMA_attr model
        self.nima_attr = NIMA_attr(num_bins, num_attr)  # Make sure to define or import NIMA_attr
        
        # Interaction MLPs
        self.mlp1 = MLP(num_attr * num_pt, hidden_size, 1)
        self.mlp2 = MLP(num_bins, hidden_size, 1)

    def forward(self, images, personal_traits):
        logit, attr_mean_pred = self.nima_attr(images)
        prob = F.softmax(logit, dim=1)
        
        # Interaction map calculation
        A_ij = attr_mean_pred.unsqueeze(2) * personal_traits.unsqueeze(1)
        I_ij = A_ij.view(images.size(0), -1)
        interaction_outputs = self.mlp1(I_ij)
        direct_outputs = self.mlp2(prob * self.scale.to(images.device))
        
        return interaction_outputs + direct_outputs

def train(model, dataloader, criterion_mse, optimizer, device):
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

        y_ij = model(images, sample_pt)
        # logit, attr_mean_pred = model(images)
        # prob = F.softmax(logit, dim=1)

        # # Interation_map
        # A_ij = attr_mean_pred.unsqueeze(2) * sample_pt.unsqueeze(1)
        # I_ij = A_ij.view(batch_size,-1)
        # y_ij = mlp1(I_ij) + mlp2(prob * scale)
        
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


def evaluate(model, dataloader, criterion_mse, device):
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

            y_ij = model(images, sample_pt)
            # logit, attr_mean_pred = model(images)
            # prob = F.softmax(logit, dim=1)
            # # Interaction_map
            # A_ij = attr_mean_pred.unsqueeze(2) * sample_pt.unsqueeze(1)
            # I_ij = A_ij.view(batch_size, -1)
            # y_ij = mlp1(I_ij) + mlp2(prob * scale)

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
    

def load_data(args, root_dir = '/home/lwchen/datasets/PARA/'):
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
    trait=args.trait
    value=args.value

    # Create datasets with the appropriate transformations
    dataset = PARA_PIAADataset(root_dir, transform=train_transform)
    train_piaa_dataset = PARA_PIAADataset(root_dir, transform=train_transform)
    test_piaa_dataset = PARA_PIAADataset(root_dir, transform=test_transform)
    train_dataset, test_dataset = split_dataset_by_images(train_piaa_dataset, test_piaa_dataset, root_dir)
    # Assuming shell_users_df contains the shell user DataFrame
    if args.use_cv:
        train_dataset, test_dataset = create_user_split_dataset_kfold(dataset, train_dataset, test_dataset, fold_id=args.fold_id, n_fold=args.n_fold)
    if trait is not None and value is not None:
        orig_len = len(train_dataset)
        train_dataset.data = train_dataset.data[train_dataset.data[trait] == value]
        test_dataset.data = test_dataset.data[test_dataset.data[trait] != value]
        print('Original data len: %d, Train: %d, Test: %d'%(orig_len, len(train_dataset), len(test_dataset)))

    # user_ids_from = pd.read_csv('top30_user_ids.csv')['User ID'].tolist()
    # user_ids_from = None

    # train_dataset, test_dataset = split_dataset_by_user(
    #     PARA_PIAADataset(root_dir, transform=train_transform),
    #     test_count=40, max_annotations_per_user=[100, 50], seed=None, user_id_list=user_ids_from)

    return train_dataset, test_dataset


num_bins = 9
num_attr = 8
num_pt = 25 # number of personal trait
# pretrained_rn = os.path.join('models_pth','best_model_resnet50_giaa_hidden512_lr5e-05_decay_20epoch_twilight-dream-273.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training and Testing the Combined Model for data spliting')
    parser.add_argument('--trait', type=str, default=None)
    parser.add_argument('--value', type=str, default=None)
    parser.add_argument('--fold_id', type=int, default=1)
    parser.add_argument('--n_fold', type=int, default=4)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--pretrained_model', type=str, required=True)
    parser.add_argument('--use_cv', action='store_true', help='Enable cross validation')
    parser.add_argument('--is_eval', action='store_true', help='Enable evaluation mode')
    parser.add_argument('--no_log', action='store_false', dest='is_log', help='Disable logging')
    args = parser.parse_args()

    is_eval = args.is_eval
    is_log = args.is_log
    resume = args.resume
    pretrained_model = args.pretrained_model
    
    lr = 5e-5
    batch_size = 100
    num_epochs = 5
    if is_log:
        tags = ["no_attr","PIAA"]
        if args.use_cv:
            tags += ["CV%d/%d"%(args.fold_id, args.n_fold)]
        wandb.init(project="resnet_PARA_PIAA",
                notes="PIAA-MIR",
                tags = tags)
        wandb.config = {
            "learning_rate": lr,
            "batch_size": batch_size,
            "num_epochs": num_epochs
        }
        experiment_name = wandb.run.name
    else:
        experiment_name = ''
    
    # Create dataloaders for training and test sets
    train_dataset, test_dataset = load_data(args)
    n_workers = 8
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers, timeout=300)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers, timeout=300)

    # Define the number of classes in your dataset
    num_classes = num_attr + num_bins
    # Define the device for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NIMA_attr(num_bins, num_attr)
    model = CombinedModel(num_bins, num_attr, num_pt).to(device)
    model.nima_attr.load_state_dict(torch.load(pretrained_model))
    model = model.to(device)
    # # Define two MLPs
    # d_interaction = num_attr * num_pt
    # mlp1 = MLP(d_interaction, 1024, 1)
    # mlp2 = MLP(num_bins, 1024, 1)
    
    # # Move the model to the device
    # model = model.to(device)
    # mlp1 = mlp1.to(device)
    # mlp2 = mlp2.to(device)
    
    # Define the loss functions
    criterion_mse = nn.MSELoss()

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # optimizer = optim.Adam(chain([*model.parameters(), *mlp1.parameters(), *mlp2.parameters()]), lr=lr)
    
    # Initialize the best test loss and the best model
    best_model = None
    best_modelname = 'best_model_resnet50_piaamir_lr%1.0e_decay_%depoch' % (lr, num_epochs)
    best_modelname += '_%s'%experiment_name
    best_modelname += '.pth'
    dirname = 'models_pth'
    if args.use_cv:
        dirname = os.path.join(dirname, 'random_cvs')
    best_modelname = os.path.join(dirname, best_modelname)

    # Training loop
    lr_schedule_epochs = 5
    lr_decay_factor = 0.5
    max_patience_epochs = 10
    num_patience_epochs = 0
    best_test_srocc = 0
    for epoch in range(num_epochs):
        if is_eval:
            break
        # Learning rate schedule
        if (epoch + 1) % lr_schedule_epochs == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_decay_factor
        # Training
        train_mse_loss = train(model, train_dataloader, criterion_mse, optimizer, device)
        # Testing
        test_mse_loss, test_srocc = evaluate(model, test_dataloader, criterion_mse, device)

        if is_log:
            wandb.log({"Train PIAA MSE Loss": train_mse_loss,
                       "Test PIAA MSE Loss": test_mse_loss,
                    "Test PIAA SROCC": test_srocc,
                    }, commit=True)

        # Print the epoch loss
        print(f"Epoch [{epoch + 1}/{num_epochs}], "
            f"Test PIAA MSE Loss: {test_mse_loss:.4f}, "
            f"Test PIAA SROCC Loss: {test_srocc:.4f}, ")

        # Early stopping check
        if test_srocc > best_test_srocc:
            best_test_srocc = test_srocc
            num_patience_epochs = 0
            torch.save(model.state_dict(), best_modelname)
        else:
            num_patience_epochs += 1
            if num_patience_epochs >= max_patience_epochs:
                print("Validation loss has not decreased for {} epochs. Stopping training.".format(max_patience_epochs))
                break

    if not is_eval:
        model.load_state_dict(torch.load(best_modelname))
    # Testing
    test_mse_loss, test_srocc = evaluate(model, test_dataloader, criterion_mse, device)
    if is_log:
        wandb.log({"Test PIAA MSE Loss": test_mse_loss,
                   "Test PIAA SROCC": test_srocc}, commit=True)
    print(
        # f"Train PIAA MSE Loss: {train_mse_loss:.4f}, "
        f"Test PIAA MSE Loss: {test_mse_loss:.4f}, "
        f"Test PIAA SROCC Loss: {test_srocc:.4f}, ")
    
    # sroccs = []
    # # for idx in range(10):
    # for idx in range(1):
    #     train_dataset.data = train_dataset.databank[idx]
    #     test_dataset.data = test_dataset.databank[idx]
    
    #     train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers)
    #     test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers)
    
    #     # Define the number of classes in your dataset
    #     num_classes = num_attr + num_bins
    #     # Define the device for training
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     model = CombinedModel(num_bins, num_attr, 512).to(device)
    #     model.load_state_dict(torch.load(pretrained_rn))
    
    #     # Define two MLPs
    #     d_interactio = num_attr * num_pt
    #     mlp1 = MLP(d_interactio, 1024, 1)
    #     mlp2 = MLP(num_bins, 1024, 1)
    
    #     # Move the model to the device
    #     model = model.to(device)
    #     mlp1 = mlp1.to(device)
    #     mlp2 = mlp2.to(device)
    
    #     # Define the loss functions
    #     criterion_mse = nn.MSELoss()
    
    #     # Define the optimizer
    #     optimizer = optim.Adam(chain([*model.parameters(), *mlp1.parameters(), *mlp2.parameters()]), lr=lr)
    #     # optimizer = optim.Adam(chain([*mlp1.parameters(), *mlp2.parameters()]), lr=lr)
    
    #     # Initialize the best test loss and the best model
    #     best_model = None
    #     best_modelname = 'best_model_resnet50_piaamir_lr%1.0e_decay_%depoch' % (lr, num_epochs)
    #     best_modelname += '_%s'%experiment_name
    #     best_modelname1 = best_modelname + '_mlp1.pth'
    #     best_modelname2 = best_modelname + '_mlp1.pth'
    #     # best_modelname += '.pth'
    
    #     # Training loop
    #     lr_schedule_epochs = 5
    #     lr_decay_factor = 0.5
    #     max_patience_epochs = 10
    #     num_patience_epochs = 0
    #     best_test_loss = float('inf')
    #     for epoch in range(num_epochs):
    #         # Learning rate schedule
    #         if (epoch + 1) % lr_schedule_epochs == 0:
    #             for param_group in optimizer.param_groups:
    #                 param_group['lr'] *= lr_decay_factor
    #         # Training
    #         train_mse_loss = train(model, mlp1, mlp2, train_dataloader, criterion_mse, optimizer, device)
    #         if is_log:
    #             wandb.log({"Train PIAA MSE Loss": train_mse_loss,}, commit=True)
    
    #     # Testing
    #     test_mse_loss, test_srocc = evaluate(model, mlp1, mlp2, test_dataloader, criterion_mse, device)
    #     if is_log:
    #         wandb.log({"Test user PIAA MSE Loss": test_mse_loss,
    #                 "Test user PIAA SROCC": test_srocc}, commit=True)
    
    
    #     # Print the epoch loss
    #     print(f"Epoch [{epoch + 1}/{num_epochs}], "
    #         # f"Train PIAA MSE Loss: {train_mse_loss:.4f}, "
    #         f"Test user PIAA MSE Loss: {test_mse_loss:.4f}, "
    #         f"Test user PIAA SROCC Loss: {test_srocc:.4f}, ")
    
    #     sroccs.append(test_srocc)
    # sroccs = np.array(sroccs)
    # print(sroccs.mean(), sroccs.std())
