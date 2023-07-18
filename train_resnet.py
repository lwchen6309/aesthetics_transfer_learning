import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50
import numpy as np
from tqdm import tqdm
from PARA_dataloader import PARADataset
import wandb


def train(model, train_dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    progress_bar = tqdm(train_dataloader, leave=False)
    for images, mean_scores, std_scores, score_hist in progress_bar:
        images = images.to(device)
        mean_scores = mean_scores.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, mean_scores)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        progress_bar.set_postfix({'Train Loss': loss.item()})
        
    epoch_loss = running_loss / len(train_dataloader)
    return epoch_loss


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0

    progress_bar = tqdm(dataloader, leave=False)
    with torch.no_grad():
        for images, mean_scores, std_scores, score_hist in progress_bar:
            images = images.to(device)
            mean_scores = mean_scores.to(device)

            outputs = model(images)
            loss = criterion(outputs, mean_scores)

            running_loss += loss.item()
            progress_bar.set_postfix({'Eval Loss': loss.item()})

    epoch_loss = running_loss / len(dataloader)
    return epoch_loss


if __name__ == '__main__':
    # Set random seed for reproducibility
    random_seed = 42
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    is_log = True
    use_attr = False
    lr = 1e-3
    batch_size = 32
    num_epochs = 30
    if is_log:
        wandb.init(project="resnet_PARA_GIAA")
        wandb.config = {
        "learning_rate": lr,
        "batch_size": batch_size,
        "num_epochs": num_epochs
        }

    # Define the root directory of the PARA dataset
    root_dir = '/home/lwchen/datasets/PARA/'

    # Define transformations for training set and test set
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    # Create datasets with the appropriate transformations
    train_dataset = PARADataset(root_dir, transform=train_transform, train=True, use_attr=use_attr, random_seed=random_seed)
    test_dataset = PARADataset(root_dir, transform=test_transform, train=False, use_attr=use_attr, random_seed=random_seed)

    # Create dataloaders for training and test sets
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define the number of classes in your dataset
    if use_attr:
        num_classes = 9
    else:
        num_classes = 1

    # Define the device for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the pre-trained ResNet model
    model_resnet50 = resnet50(pretrained=True)

    # Modify the last fully connected layer to match the number of classes
    num_features = model_resnet50.fc.in_features
    model_resnet50.fc = nn.Linear(num_features, num_classes)
    # model_resnet50.load_state_dict(torch.load('best_model_resnet50.pth'))

    # Move the model to the device
    model_resnet50 = model_resnet50.to(device)

    # Define the loss function
    criterion = nn.MSELoss()

    # Define the optimizer
    optimizer_resnet50 = optim.SGD(model_resnet50.parameters(), lr=lr, momentum=0.9)

    # Define a list to record the training and test losses
    train_loss_list = []
    test_loss_list = []

    # Initialize the best test loss and the best model
    best_test_loss = float('inf')
    best_model = None
    
    # Training loop for ResNet-50
    for epoch in range(num_epochs):
        # Training
        train_loss = train(model_resnet50, train_dataloader, criterion, optimizer_resnet50, device)
        train_loss_list.append(train_loss)
        if is_log:
            wandb.log({"Train Loss": train_loss}, commit=False)

        # Testing
        test_loss = evaluate(model_resnet50, test_dataloader, criterion, device)
        test_loss_list.append(test_loss)
        if is_log:
            wandb.log({"Test Loss": test_loss})

        # Print the epoch loss
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss}, Test Loss: {test_loss}")

        # Check if the current model has the best test loss so far
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_model = model_resnet50.state_dict()
    
    # Save the best model
    best_modelname = 'best_model_resnet50'
    if not use_attr:
        best_modelname += '_noattr'
    best_modelname += '.pth'
    torch.save(best_model, best_modelname)
