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
        if is_eval:
            break
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


is_log = True
use_attr = False
# resume = 'best_model_resnet50_adam_lr1e-04_200epoch_noattr.pth'
resume = None
is_eval = False

# Define sweep config
sweep_configuration = {
    'method': 'bayes',
    'name': 'sweep',
    'metric': {'goal': 'minimize', 'name': 'Test MSE Mean Loss'},
    'parameters': 
    {
        'batch_size': {'values': [16, 32, 64]},
        'epochs': {'values': [200]},
        'lr': {'max': 1e-2, 'min': 1e-5},
        'lr_decay':{'value':0.9},
        'adam_weight_decay':{'max':1e-2, 'min':0.},
     }
}

# Initialize sweep by passing in config. 
# (Optional) Provide a name of the project.
sweep_id = wandb.sweep(
  sweep=sweep_configuration, 
  project='resnet_PARA_GIAA_sweep'
  )

# config = {
#     'lr' : 1e-4,
#     'bs' : 32,
#     'epochs': 200,
#     'lr_decay':0.9,
# }

def main():
    # Set random seed for reproducibility
    random_seed = 42
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    # random_seed = None

    run = wandb.init()
    # note that we define values from `wandb.config`  
    # instead of defining hard values
    lr  =  wandb.config.lr
    batch_size = wandb.config.batch_size
    num_epochs = wandb.config.epochs
    lr_decay_factor = wandb.config.lr_decay
    adam_weight_decay = wandb.config.adam_weight_decay

    # if is_log:
    #     wandb.init(project="resnet_PARA_GIAA_sweep")
    #     wandb.config = {
    #     "learning_rate": lr,
    #     "batch_size": batch_size,
    #     "num_epochs": num_epochs
    #     }

    # Define the root directory of the PARA dataset
    root_dir = '/home/lwchen/datasets/PARA/'

    # Define transformations for training set and test set
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(224),
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
    model_resnet50.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.Linear(256, num_classes),
        # nn.Linear(num_features, num_classes)
    )
    # model_resnet50.load_state_dict(torch.load('best_model_resnet50.pth'))
    if resume is not None:
        model_resnet50.load_state_dict(torch.load(resume))
    
    # Move the model to the device
    model_resnet50 = model_resnet50.to(device)
    
    # Define the loss function
    criterion = nn.MSELoss()

    # Define the optimizer
    optimizer_resnet50 = optim.Adam(model_resnet50.parameters(), lr=lr, weight_decay=adam_weight_decay)

    # Define a list to record the training and test losses
    train_loss_list = []
    test_loss_list = []

    # Initialize the best test loss and the best model
    best_test_loss = float('inf')
    best_model = None
    best_modelname = 'best_model_resnet50_sweep_lr%1.0e_%depoch' % (lr, num_epochs)
    if not use_attr:
        best_modelname += '_noattr'
    best_modelname += '.pth'

    # Training loop
    lr_schedule_epochs = 1
    # lr_decay_factor = 0.9
    max_patience_epochs = 10
    num_patience_epochs = 0
    best_test_loss = float('inf') 
    for epoch in range(num_epochs):
        # Learning rate schedule
        if (epoch + 1) % lr_schedule_epochs == 0:
            for param_group in optimizer_resnet50.param_groups:
                param_group['lr'] *= lr_decay_factor
        
        # Training
        train_loss = train(model_resnet50, train_dataloader, criterion, optimizer_resnet50, device)
        train_loss_list.append(train_loss)
        if is_log:
            wandb.log({"Train MSE Mean Loss": train_loss}, commit=False)

        # Testing
        test_loss = evaluate(model_resnet50, test_dataloader, criterion, device)
        test_loss_list.append(test_loss)
        if is_log:
            wandb.log({"Test MSE Mean Loss": test_loss})

        # Print the epoch loss
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss}, Test Loss: {test_loss}")

        if is_eval:
            break
        # Check if the current model has the best test loss so far
        # Early stopping check
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            num_patience_epochs = 0
            torch.save(model_resnet50.state_dict(), best_modelname)
        else:
            num_patience_epochs += 1
            if num_patience_epochs >= max_patience_epochs:
                print("Validation loss has not decreased for {} epochs. Stopping training.".format(max_patience_epochs))
                break


if __name__ == '__main__':    
    # Start sweep job.
    wandb.agent(sweep_id, function=main, count=4)