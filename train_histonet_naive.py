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
from PARA_PIAA_dataloader import PARA_PIAADataset, split_dataset_by_user, split_dataset_by_images
from train_resnet_cls import earth_mover_distance
from train_histonet import train, evaluate


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
    
    def forward(self, images, attributes_histogram, traits_histogram):
        x = self.resnet(images)
        return x


is_eval = True
is_log = False
num_bins = 9
num_attr = 40
num_pt = 50 + 20
resume = None


if __name__ == '__main__':
    random_seed = 42
    lr = 5e-5
    batch_size = 100
    num_epochs = 10
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
    _, test_user_piaa_dataset = split_dataset_by_user(
        PARA_PIAADataset(root_dir, transform=train_transform), 
        PARA_PIAADataset(root_dir, transform=train_transform), 
        test_count=40, max_annotations_per_user=50, seed=random_seed)
    
    # Create datasets with the appropriate transformations
    # train_dataset = PARA_HistogramDataset(root_dir, transform=train_transform, data=train_dataset.data, map_file='trainset_image_dct.pkl')
    train_dataset = PARA_GIAA_HistogramDataset(root_dir, transform=train_transform, data=train_dataset.data, map_file='trainset_image_dct.pkl')
    
    # test_dataset = PARA_HistogramDataset(root_dir, transform=test_transform, data=test_dataset.data, map_file='testset_image_dct.pkl')
    test_giaa_dataset = PARA_GIAA_HistogramDataset(root_dir, transform=test_transform, data=test_dataset.data, map_file='testset_image_dct.pkl')
    test_piaa_dataset = PARA_PIAA_HistogramDataset(root_dir, transform=test_transform, data=test_dataset.data)
    test_user_piaa_dataset = PARA_PIAA_HistogramDataset(root_dir, transform=test_transform, data=test_user_piaa_dataset.data)
    
    # Create dataloaders
    n_workers = 20
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers)
    test_giaa_dataloader = DataLoader(test_giaa_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers)
    test_piaa_dataloader = DataLoader(test_piaa_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers)
    test_user_piaa_dataloader = DataLoader(test_user_piaa_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the combined model
    model = CombinedModel(num_bins, num_attr, num_pt, 
        resume = 'best_model_resnet50_hidden512_cls_lr5e-05_decay_20epoch_noattr_distinctive-glade-243.pth').to(device)
    if resume is not None:
        model.load_state_dict(torch.load(resume))
    # Loss and optimizer
    # criterion_mse = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Initialize the best test loss and the best model
    best_model = None
    best_modelname = 'best_model_resnet50_histo_lr%1.0e_decay_%depoch' % (lr, num_epochs)
    best_modelname += '_%s'%experiment_name
    best_modelname += '.pth'

    # Training loop
    best_test_loss = float('inf')
    for epoch in range(num_epochs):
        # Learning rate schedule
        if (epoch + 1) % lr_schedule_epochs == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_decay_factor

        # Training
        # train_emd_loss = train(model, train_dataloader, earth_mover_distance, optimizer, device)
        train_emd_loss = 0
        if is_log:
            wandb.log({"Train EMD Loss": train_emd_loss,}, commit=False)
        
        # Testing
        test_piaa_emd_loss, test_piaa_srocc, test_piaa_mse = evaluate(model, test_piaa_dataloader, earth_mover_distance, device)
        test_giaa_emd_loss, test_giaa_srocc, test_giaa_mse = evaluate(model, test_giaa_dataloader, earth_mover_distance, device)
        test_user_piaa_emd_loss, test_user_piaa_srocc, test_user_piaa_mse = evaluate(model, test_user_piaa_dataloader, earth_mover_distance, device)
        if is_log:
            wandb.log({"Test PIAA EMD Loss": test_piaa_emd_loss,
                       "Test PIAA SROCC": test_piaa_srocc,
                       "Test GIAA EMD Loss": test_giaa_emd_loss,
                       "Test GIAA SROCC": test_giaa_srocc,
                       "Test user PIAA EMD Loss": test_user_piaa_emd_loss,
                       "Test user PIAA SROCC": test_user_piaa_srocc,
                       }, commit=True)

        # Print the epoch loss
        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train EMD Loss: {train_emd_loss:.4f}, "
              f"Test PIAA EMD Loss: {test_piaa_emd_loss:.4f}, "
              f"Test PIAA SROCC Loss: {test_piaa_srocc:.4f}, "
              f"Test PIAA MSE Loss: {test_piaa_mse:.4f}, "
              f"Test user PIAA EMD Loss: {test_user_piaa_emd_loss:.4f}, "
              f"Test user PIAA SROCC Loss: {test_user_piaa_srocc:.4f}, "
              f"Test user PIAA MSE Loss: {test_user_piaa_mse:.4f}, "
              f"Test GIAA EMD Loss: {test_giaa_emd_loss:.4f}, "
              f"Test GIAA SROCC Loss: {test_giaa_srocc:.4f}, "
              f"Test GIAA MSE Loss: {test_giaa_mse:.4f}, "
              )
        if is_eval:
            break

        # Early stopping check
        if test_piaa_emd_loss < best_test_loss:
            best_test_loss = test_piaa_emd_loss
            num_patience_epochs = 0
            torch.save(model.state_dict(), best_modelname)
        else:
            num_patience_epochs += 1
            if num_patience_epochs >= max_patience_epochs:
                print("Validation loss has not decreased for {} epochs. Stopping training.".format(max_patience_epochs))
                break