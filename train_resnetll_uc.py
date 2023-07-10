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
    running_mse_loss = 0.0
    running_custom_loss = 0.0

    progress_bar = tqdm(train_dataloader, leave=False)
    for images, mean_scores, std_scores in progress_bar:
        images = images.to(device)
        mean_scores = mean_scores.to(device)
        std_scores = std_scores.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        mse_loss = criterion(outputs, mean_scores)
        custom_loss = custom_criterion(outputs, mean_scores, std_scores)

        loss = custom_loss

        loss.backward()
        optimizer.step()

        running_mse_loss += mse_loss.item()
        running_custom_loss += custom_loss.item()
        progress_bar.set_postfix({'MSE Loss': mse_loss.item(), 'Custom Loss': custom_loss.item()})

    epoch_mse_loss = running_mse_loss / len(train_dataloader)
    epoch_custom_loss = running_custom_loss / len(train_dataloader)
    return epoch_mse_loss, epoch_custom_loss


def evaluate(model, dataloader, criterion, device, num_samples=50):
    model.train()
    running_mse_loss = 0.0
    running_custom_loss = 0.0
    dropout_outputs = []

    progress_bar = tqdm(dataloader, leave=False)
    with torch.no_grad():
        for images, mean_scores, std_scores in progress_bar:
            images = images.to(device)
            mean_scores = mean_scores.to(device)

            batch_outputs = []
            for _ in range(num_samples):
                model.train()
                outputs = model(images)
                batch_outputs.append(outputs)

            batch_outputs = torch.stack(batch_outputs)
            batch_outputs_mean = torch.mean(batch_outputs, dim=0)
            batch_outputs_std = torch.std(batch_outputs, dim=0)
            batch_outputs = torch.stack([batch_outputs_mean, batch_outputs_std], dim=-1)

            dropout_outputs.append(batch_outputs)

            mse_loss = criterion(batch_outputs_mean, mean_scores)
            custom_loss = custom_criterion(batch_outputs_mean, mean_scores, batch_outputs_std)

            running_mse_loss += mse_loss.item()
            running_custom_loss += custom_loss.item()
            progress_bar.set_postfix({'MSE Loss': mse_loss.item(), 'Custom Loss': custom_loss.item()})

    epoch_mse_loss = running_mse_loss / len(dataloader)
    epoch_custom_loss = running_custom_loss / len(dataloader)
    dropout_outputs = torch.cat(dropout_outputs, dim=0)
    dropout_mean = dropout_outputs[..., 0]
    dropout_std = dropout_outputs[..., 1]
    return epoch_mse_loss, epoch_custom_loss, dropout_mean, dropout_std



if __name__ == '__main__':
    # Set random seed for reproducibility
    random_seed = 42
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    is_log = True
    lr = 1e-3
    batch_size = 32
    num_epochs = 30
    if is_log:
        wandb.init(project="resnet_PARA")
        wandb.config.update({
            "learning_rate": lr,
            "batch_size": batch_size,
            "num_epochs": num_epochs
        })

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
    train_dataset = PARADataset(root_dir, transform=train_transform, train=True, random_seed=random_seed)
    test_dataset = PARADataset(root_dir, transform=test_transform, train=False, random_seed=random_seed)

    # Create dataloaders for training and test sets
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define the number of classes in your dataset
    num_classes = 9  # Replace with the actual number of classes

    # Define the device for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the pre-trained ResNet model
    model_resnet50 = resnet50(pretrained=True)

    # Modify the last fully connected layer to match the number of classes
    num_features = model_resnet50.fc.in_features
    model_resnet50.fc = nn.Sequential(
        # nn.Dropout(p=0.1),
        nn.Linear(num_features, num_classes)
    )

    # Move the model to the device
    model_resnet50 = model_resnet50.to(device)

    # Define the custom loss function
    def custom_criterion(outputs, mean_scores, std_scores):
        return 0.5 * torch.mean(((torch.abs(outputs - mean_scores) + 1e-6) / (std_scores + 1e-6)) ** 2)

    # Define the MSE loss function
    criterion = nn.MSELoss()

    # Define the optimizer
    optimizer_resnet50 = optim.SGD(model_resnet50.fc.parameters(), lr=lr, momentum=0.9)

    # Define a list to record the training and test losses
    train_mse_loss_list = []
    train_custom_loss_list = []
    test_mse_loss_list = []
    test_custom_loss_list = []

    # Initialize the best test loss and the best model
    best_test_loss = float('inf')
    best_model = None

    # Training loop for ResNet-50
    for epoch in range(num_epochs):
        # Training
        train_mse_loss, train_custom_loss = train(model_resnet50, train_dataloader, criterion, optimizer_resnet50,
                                                  device)
        train_mse_loss_list.append(train_mse_loss)
        train_custom_loss_list.append(train_custom_loss)
        if is_log:
            wandb.log({"Train MSE Loss": train_mse_loss, "Train Custom Loss": train_custom_loss})

        # Testing
        test_mse_loss, test_custom_loss, dropout_mean, dropout_std = evaluate(model_resnet50, test_dataloader, criterion, device,
                                                        num_samples=5)
        test_mse_loss_list.append(test_mse_loss)
        test_custom_loss_list.append(test_custom_loss)
        if is_log:
            wandb.log({"Test MSE Loss": test_mse_loss, "Test Custom Loss": test_custom_loss}, commit=False)

        # Print the epoch loss
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Train MSE Loss: {train_mse_loss}, Train Custom Loss: {train_custom_loss}, Test Loss: {test_mse_loss}")

        # Check if the current model has the best test loss so far
        if test_mse_loss < best_test_loss:
            best_test_loss = test_mse_loss
            best_model = model_resnet50.state_dict()

    # Save the best model
    torch.save(best_model, 'best_model_resnet50_ll.pth')

    # Record the training and test losses into a textfile
    lossrecords = {
        'Train MSE Loss': train_mse_loss_list,
        'Train Custom Loss': train_custom_loss_list,
        'Test MSE Loss': test_mse_loss_list,
        'Test Custom Loss': test_custom_loss_list
    }
