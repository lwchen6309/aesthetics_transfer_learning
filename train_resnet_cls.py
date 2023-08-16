import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50
import numpy as np
from tqdm import tqdm
from PARA_dataloader import PARADataset
import wandb


def earth_mover_distance(x, y):
    """
    Compute Earth Mover's Distance (EMD) between two 1D tensors x and y using 2-norm.
    """
    cdf_x = torch.cumsum(x, dim=1)
    cdf_y = torch.cumsum(y, dim=1)
    emd = torch.norm(cdf_x - cdf_y, p=2, dim=1)
    return torch.mean(emd)


def train(model, train_dataloader, criterion_weight_ce, criterion_raw_ce, criterion_mse, criterion_emd, optimizer, device):
    model.train()
    running_ce_loss = 0.0
    running_raw_ce_loss = 0.0
    running_mse_mean_loss = 0.0
    running_mse_std_loss = 0.0
    running_emd_loss = 0.0

    progress_bar = tqdm(train_dataloader, leave=False)
    scale = torch.arange(1, 5.5, 0.5).to(device)
    for images, mean_scores, std_scores, score_prob in progress_bar:
        if is_eval:
            break
        images = images.to(device)
        mean_scores = mean_scores.to(device)
        std_scores = std_scores.to(device)
        score_prob = score_prob.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        prob = F.softmax(outputs, dim=1)

        # Cross-entropy loss
        ce_loss = criterion_weight_ce(outputs, score_prob)
        raw_ce_loss = criterion_raw_ce(outputs, score_prob)
        
        # Earth Mover's Distance (EMD) loss
        emd_loss = criterion_emd(prob, score_prob)

        with torch.no_grad():
            # MSE loss for mean
            outputs_mean = torch.sum(prob * scale, dim=1, keepdim=True)
            mse_mean_loss = criterion_mse(outputs_mean, mean_scores)

            # MSE loss for std
            outputs_std = torch.sqrt(torch.sum(prob * (scale - outputs_mean) ** 2, dim=1, keepdim=True))
            mse_std_loss = criterion_mse(outputs_std, std_scores)

        if use_ce:
            loss = ce_loss
        else:
            loss = emd_loss

        loss.backward()
        optimizer.step()

        running_ce_loss += ce_loss.item()
        running_raw_ce_loss += raw_ce_loss.item()
        running_mse_mean_loss += mse_mean_loss.item()
        running_mse_std_loss += mse_std_loss.item()
        running_emd_loss += emd_loss.item()

        progress_bar.set_postfix({
            'Train CE Loss': ce_loss.item(),
            'Train Raw CE Loss': raw_ce_loss.item(),
            # 'Train MSE Mean Loss': mse_mean_loss.item(),
            # 'Train MSE Std Loss': mse_std_loss.item(),
            'Train EMD Loss': emd_loss.item()
        })

    epoch_ce_loss = running_ce_loss / len(train_dataloader)
    epoch_raw_ce_loss = running_raw_ce_loss / len(train_dataloader)
    epoch_mse_mean_loss = running_mse_mean_loss / len(train_dataloader)
    epoch_mse_std_loss = running_mse_std_loss / len(train_dataloader)
    epoch_emd_loss = running_emd_loss / len(train_dataloader)

    return epoch_ce_loss, epoch_raw_ce_loss, epoch_mse_mean_loss, epoch_mse_std_loss, epoch_emd_loss


def evaluate(model, dataloader, criterion_weight_ce, criterion_raw_ce, criterion_mse, criterion_emd, device):
    model.eval()
    running_ce_loss = 0.0
    running_raw_ce_loss = 0.0
    running_mse_mean_loss = 0.0
    running_mse_std_loss = 0.0
    running_emd_loss = 0.0
    running_brier_score = 0.0

    scale = torch.arange(1, 5.5, 0.5).to(device)
    progress_bar = tqdm(dataloader, leave=False)
    with torch.no_grad():
        for images, mean_scores, std_scores, score_prob in progress_bar:
            images = images.to(device)
            mean_scores = mean_scores.to(device)
            std_scores = std_scores.to(device)
            score_prob = score_prob.to(device)

            outputs = model(images)
            prob = F.softmax(outputs, dim=1)

            # Cross-entropy loss
            ce_loss = criterion_weight_ce(outputs, score_prob)
            raw_ce_loss = criterion_raw_ce(outputs, score_prob)

            # MSE loss for mean
            outputs_mean = torch.sum(prob * scale, dim=1, keepdim=True)
            mse_mean_loss = criterion_mse(outputs_mean, mean_scores)

            # MSE loss for std
            outputs_std = torch.sqrt(torch.sum(prob * (scale - outputs_mean) ** 2, dim=1, keepdim=True))
            mse_std_loss = criterion_mse(outputs_std, std_scores)

            # Earth Mover's Distance (EMD) loss
            emd_loss = criterion_emd(prob, score_prob)
            brier_score = criterion_mse(prob, score_prob)

            running_ce_loss += ce_loss.item()
            running_raw_ce_loss += raw_ce_loss.item()
            running_mse_mean_loss += mse_mean_loss.item()
            running_mse_std_loss += mse_std_loss.item()
            running_emd_loss += emd_loss.item()
            running_brier_score += brier_score.item()
            
            progress_bar.set_postfix({
                'Eval CE Loss': ce_loss.item(),
                'Eval Raw CE Loss': raw_ce_loss.item(),
                # 'Eval MSE Mean Loss': mse_mean_loss.item(),
                # 'Eval MSE Std Loss': mse_std_loss.item(),
                'Eval EMD Loss': emd_loss.item(),
                'Eval Brier Score': brier_score.item()
            })

    epoch_ce_loss = running_ce_loss / len(dataloader)
    epoch_raw_ce_loss = running_raw_ce_loss / len(dataloader)
    epoch_mse_mean_loss = running_mse_mean_loss / len(dataloader)
    epoch_mse_std_loss = running_mse_std_loss / len(dataloader)
    epoch_emd_loss = running_emd_loss / len(dataloader)
    epoch_brier_score = running_brier_score / len(dataloader)
    
    return epoch_ce_loss, epoch_raw_ce_loss, epoch_mse_mean_loss, epoch_mse_std_loss, epoch_emd_loss, epoch_brier_score


is_eval = False
is_log = True
use_attr = False
use_hist = True
use_ce = False

if __name__ == '__main__':
    # Set random seed for reproducibility
    random_seed = 42
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    lr = 1e-4
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
    train_dataset = PARADataset(root_dir, transform=train_transform, train=True, use_attr=use_attr,
                                use_hist=use_hist, random_seed=random_seed)
    test_dataset = PARADataset(root_dir, transform=test_transform, train=False, use_attr=use_attr,
                               use_hist=use_hist, random_seed=random_seed)

    # Create dataloaders for training and test sets
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define the number of classes in your dataset
    if use_attr:
        num_classes = 9 + 5 * 7
    else:
        num_classes = 9

    # Define the device for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the pre-trained ResNet model
    model_resnet50 = resnet50(pretrained=True)

    # Modify the last fully connected layer to match the number of classes
    num_features = model_resnet50.fc.in_features
    model_resnet50.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.Linear(256, num_classes)
    )
    model_resnet50.load_state_dict(torch.load("best_model_resnet50_noattr.pth"), strict=False)
    # model_resnet50.load_state_dict(torch.load("best_model_resnet50_cls_lr1e-03_30epoch_noattr.pth"))
    
    # Move the model to the device
    model_resnet50 = model_resnet50.to(device)
    

    # Define the loss functions
    ce_weight = 1 / train_dataset.aesthetic_score_hist_prob
    ce_weight = ce_weight / np.sum(ce_weight) * len(ce_weight)
    criterion_weight_ce = nn.CrossEntropyLoss(weight=torch.tensor(ce_weight, device=device))
    criterion_raw_ce = nn.CrossEntropyLoss()
    criterion_mse = nn.MSELoss()
    criterion_emd = earth_mover_distance

    # Define the optimizer
    # optimizer_resnet50 = optim.SGD(model_resnet50.parameters(), lr=lr, momentum=0.9)
    optimizer_resnet50 = optim.Adam(model_resnet50.parameters(), lr=lr)

    # Initialize the best test loss and the best model
    best_model = None
    best_modelname = 'best_model_resnet50_cls_lr%1.0e_decay_%depoch' % (lr, num_epochs)
    if not use_attr:
        best_modelname += '_noattr'
    if use_ce:
        best_modelname += '_ce'
    best_modelname += '.pth'

    # Training loop
    lr_schedule_epochs = 1
    lr_decay_factor = 0.9
    max_patience_epochs = 10
    num_patience_epochs = 0
    best_test_loss = float('inf')    
    for epoch in range(num_epochs):
        # Learning rate schedule
        if (epoch + 1) % lr_schedule_epochs == 0:
            for param_group in optimizer_resnet50.param_groups:
                param_group['lr'] *= lr_decay_factor

        # Training
        train_ce_loss, train_raw_ce_loss, train_mse_mean_loss, train_mse_std_loss, train_emd_loss = train(
            model_resnet50, train_dataloader, criterion_weight_ce, criterion_raw_ce, criterion_mse, criterion_emd,
            optimizer_resnet50, device)
        if is_log:
            wandb.log({"Train CE Loss": train_ce_loss,
                       "Train Raw CE Loss": train_raw_ce_loss,
                       "Train MSE Mean Loss": train_mse_mean_loss,
                       "Train MSE Std Loss": train_mse_std_loss,
                       "Train EMD Loss": train_emd_loss}, commit=False)

        # Testing
        test_ce_loss, test_raw_ce_loss, test_mse_mean_loss, test_mse_std_loss, test_emd_loss, test_brier_score = evaluate(
            model_resnet50, test_dataloader, criterion_weight_ce, criterion_raw_ce, criterion_mse, criterion_emd,
            device)
        if is_log:
            wandb.log({"Test CE Loss": test_ce_loss,
                       "Test Raw CE Loss": test_raw_ce_loss,
                       "Test MSE Mean Loss": test_mse_mean_loss,
                       "Test MSE Std Loss": test_mse_std_loss,
                       "Test EMD Loss": test_emd_loss,
                       "Test Brier Score": test_brier_score})

        # Print the epoch loss
        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train CE Loss: {train_ce_loss:.4f}, "
              f"Train Raw CE Loss: {train_raw_ce_loss:.4f}, "
              f"Train MSE Mean Loss: {train_mse_mean_loss:.4f}, "
              f"Train MSE Std Loss: {train_mse_std_loss:.4f}, "
              f"Train EMD Loss: {train_emd_loss:.4f}, "
              f"Test CE Loss: {test_ce_loss:.4f}, "
              f"Test Raw CE Loss: {test_raw_ce_loss:.4f}, "
              f"Test MSE Mean Loss: {test_mse_mean_loss:.4f}, "
              f"Test MSE Std Loss: {test_mse_std_loss:.4f}, "
              f"Test EMD Loss: {test_emd_loss:.4f}, "
              f"Test Brier Score: {test_brier_score:.4f}")
        if is_eval:
            raise Exception

        # Check if the current model has the best test loss so far
        test_loss = test_ce_loss if use_ce else test_emd_loss
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