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
from train_resnet_cls import earth_mover_distance
from scipy.stats import norm



def metric_to_cdf(scale, mu, sigma, py, ce_weight=None, debug=False):
    """
    Compute Earth Mover's Distance (EMD) between two 1D tensors x and y using 2-norm.
    """
    half_res = ((scale[1] - scale[0]) * 0.5)
    x1 = (scale + half_res).cpu().numpy()
    x0 = (scale - half_res).cpu().numpy()
    mu = mu.cpu().numpy()
    sigma = sigma.cpu().numpy()
    px = torch.tensor(np.array([norm.cdf(x1, loc=_mu, scale=_sigma) - norm.cdf(x0, loc=_mu, scale=_sigma) for _mu, _sigma in zip(mu, sigma)])).to(py.device)
    cdf_px = torch.cumsum(px, dim=1)
    cdf_py = torch.cumsum(py, dim=1)
    emd = torch.mean(torch.norm(cdf_px - cdf_py, p=2, dim=1))
    brier_score = torch.mean((px - py)**2)

    if debug:
        scale = scale.cpu().numpy()
        sqrt_2pi = np.sqrt(2*np.pi)
        prob = np.exp(-0.5*((mu - scale) / sigma)**2) / sigma / sqrt_2pi * 0.5
        print(prob / px.cpu().numpy())
    
    logprob = torch.log(px + 1e-6)
    raw_ce = - torch.mean(torch.sum(logprob * py, dim=1))
    if ce_weight is not None:
        ce = - torch.mean(torch.sum(logprob * py * ce_weight, dim=1))
    else:
        ce = raw_ce

    return emd, brier_score, ce, raw_ce


def train(model, train_dataloader, optimizer, device):
    model.train()
    running_loss = 0.0
    running_loss_mean = 0.0
    running_loss_std = 0.0
    running_loss_ce = 0.0
    running_loss_raw_ce = 0.0
    running_loss_emd = 0.0

    progress_bar = tqdm(train_dataloader, leave=False)
    scale = torch.arange(1, 5.5, 0.5).to(device)
    sqrt_2pi = np.sqrt(2*np.pi)
    for images, mean_scores, std_scores, score_prob in progress_bar:
        if is_eval:
            break
        images = images.to(device)
        mean_scores = mean_scores.to(device)
        std_scores = std_scores.to(device)
        score_prob = score_prob.to(device)
        optimizer.zero_grad()
        
        outputs = model(images)
        num_classes = outputs.shape[1] // 2
        output_mean_score = outputs[:, :num_classes]
        output_std_score = outputs[:, num_classes:]
        loss_mean = criterion(output_mean_score, mean_scores)
        loss_std = criterion(output_std_score, std_scores)
        prob = torch.exp(-0.5*((output_mean_score - scale) / output_std_score)**2) / output_std_score / sqrt_2pi
        prob = prob / torch.sum(prob, dim=1, keepdim=True)
        logprob = torch.log(prob + 1e-6)
        ce_loss = -torch.mean(torch.sum(logprob * score_prob * ce_weight, dim=1))
        raw_ce_loss = -torch.mean(torch.sum(logprob * score_prob, dim=1))
        emd_loss = criterion_emd(prob, score_prob)

        if loss_func == 'ce':
            loss = ce_loss
        elif loss_func == 'emd':
            loss = emd_loss
        else:
            loss = loss_mean + loss_std
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_loss_mean += loss_mean.item()
        running_loss_std += loss_std.item()
        running_loss_ce += ce_loss.item()
        running_loss_raw_ce += raw_ce_loss.item()
        running_loss_emd += emd_loss.item()

        progress_bar.set_postfix({'Train Loss': loss.item(), 'Train Loss Mean': loss_mean.item(),
                                  'Train Loss Std': loss_std.item(), 'Train Raw CE Loss': raw_ce_loss.item(),
                                  'Train EMD Loss': emd_loss.item()})

    epoch_loss = running_loss / len(train_dataloader)
    epoch_loss_mean = running_loss_mean / len(train_dataloader)
    epoch_loss_std = running_loss_std / len(train_dataloader)
    epoch_loss_ce = running_loss_ce / len(train_dataloader)
    epoch_loss_raw_ce = running_loss_raw_ce / len(train_dataloader)
    epoch_loss_emd = running_loss_emd / len(train_dataloader)
    return epoch_loss, epoch_loss_mean, epoch_loss_std, epoch_loss_ce, epoch_loss_raw_ce, epoch_loss_emd


def evaluate(model, dataloader, device):
    model.eval()
    running_loss = 0.0
    running_loss_mean = 0.0
    running_loss_std = 0.0
    running_loss_raw_ce = 0.0
    running_loss_ce = 0.0
    running_loss_emd = 0.0
    running_brier_score = 0.0

    scale = torch.arange(1, 5.5, 0.5).to(device)
    sqrt_2pi = np.sqrt(2*np.pi)
    progress_bar = tqdm(dataloader, leave=False)
    with torch.no_grad():
        for images, mean_scores, std_scores, score_prob in progress_bar:
            images = images.to(device)
            mean_scores = mean_scores.to(device)
            std_scores = std_scores.to(device)
            score_prob = score_prob.to(device)

            outputs = model(images)
            num_classes = outputs.shape[1] // 2
            output_mean_score = outputs[:, :num_classes]
            output_std_score = outputs[:, num_classes:]
            
            emd_loss, brier_score, ce_loss, raw_ce_loss = metric_to_cdf(scale, output_mean_score, output_std_score, score_prob, ce_weight=ce_weight)

            loss_mean = criterion(output_mean_score, mean_scores)
            loss_std = criterion(output_std_score, std_scores)
            
            if loss_func == 'ce':
                loss = ce_loss
            elif loss_func == 'emd':
                loss = emd_loss
            else:
                loss = loss_mean + loss_std

            running_loss += loss.item()
            running_loss_mean += loss_mean.item()
            running_loss_std += loss_std.item()
            running_loss_raw_ce += raw_ce_loss.item()
            running_loss_ce += ce_loss.item()
            running_loss_emd += emd_loss.item()
            running_brier_score += brier_score.item()

            progress_bar.set_postfix({'Eval Loss': loss.item(), 'Eval Loss Mean': loss_mean.item(),
                                      'Eval Loss Std': loss_std.item(), 'Eval Raw CE Loss': raw_ce_loss.item(),
                                      'Eval CE Loss': ce_loss.item(), 'Eval EMD Loss': emd_loss.item()})

    epoch_loss = running_loss / len(dataloader)
    epoch_loss_mean = running_loss_mean / len(dataloader)
    epoch_loss_std = running_loss_std / len(dataloader)
    epoch_loss_raw_ce = running_loss_raw_ce / len(dataloader)
    epoch_loss_ce = running_loss_ce / len(dataloader)
    epoch_loss_emd = running_loss_emd / len(dataloader)
    epoch_brier_score = running_brier_score / len(dataloader)
    return epoch_loss, epoch_loss_mean, epoch_loss_std, epoch_loss_ce, epoch_loss_raw_ce, epoch_loss_emd, epoch_brier_score


is_eval = True
is_log = False
use_attr = False
use_hist = True
loss_func = 'ce' # mse / emd / ce


if __name__ == '__main__':
    # Set random seed for reproducibility
    random_seed = 42
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
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
    train_dataset = PARADataset(root_dir, transform=train_transform, train=True, use_attr=use_attr,
                               use_hist=use_hist, random_seed=random_seed)
    test_dataset = PARADataset(root_dir, transform=test_transform, train=False, use_attr=use_attr,
                              use_hist=use_hist, random_seed=random_seed)

    # Create dataloaders for training and test sets
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define the number of classes in your dataset
    if use_attr:
        num_classes = 9
    else:
        num_classes = 1
    # Predict mean_scores and std_scores
    num_classes *= 2

    # Define the device for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the pre-trained ResNet model
    model_resnet50 = resnet50(pretrained=True)

    # Modify the last fully connected layer to match the number of classes
    num_features = model_resnet50.fc.in_features
    model_resnet50.fc = nn.Linear(num_features, num_classes)
    if is_eval:
        model_resnet50.load_state_dict(torch.load('best_model_resnet50_dup_mse_lr1e-03_30epoch_noattr.pth'))
    else:
        model_resnet50.load_state_dict(torch.load('best_model_resnet50_noattr.pth'), strict=False)
    
    # Move the model to the device
    model_resnet50 = model_resnet50.to(device)

    # Define the loss function
    criterion = nn.MSELoss()
    ce_weight = 1 / train_dataset.aesthetic_score_hist_prob
    ce_weight = torch.tensor(ce_weight / np.sum(ce_weight) * len(ce_weight), device=device)
    criterion_weight_ce = nn.CrossEntropyLoss(weight=ce_weight)
    criterion_raw_ce = nn.CrossEntropyLoss()
    criterion_emd = earth_mover_distance

    # Define the optimizer
    optimizer_resnet50 = optim.SGD(model_resnet50.parameters(), lr=lr, momentum=0.9)

    # Initialize the best test loss and the best model
    best_test_loss = float('inf')
    best_model = None
    best_modelname = 'best_model_resnet50_dup_%s_lr%1.0e_%depoch' % (loss_func, lr, num_epochs)
    if not use_attr:
        best_modelname += '_noattr'
    best_modelname += '.pth'

    # Training loop for ResNet-50
    for epoch in range(num_epochs):
        # Training
        train_loss, train_loss_mean, train_loss_std, train_loss_ce, train_loss_raw_ce, train_loss_emd = train(model_resnet50,
                                                                                                train_dataloader,
                                                                                                optimizer_resnet50,
                                                                                                device)
        if is_log:
            wandb.log({"Train MSE Loss": train_loss, "Train MSE Mean Loss": train_loss_mean,
                       "Train MSE Std Loss": train_loss_std, "Train Raw CE Loss": train_loss_raw_ce,
                       "Train CE Loss": train_loss_ce, "Train EMD Loss": train_loss_emd}, commit=False)

        # Testing
        test_loss, test_loss_mean, test_loss_std, test_loss_ce, test_loss_raw_ce, test_loss_emd, test_brier_score = evaluate(model_resnet50,
                                                                                            test_dataloader,
                                                                                            device)
        if is_log:
            wandb.log({"Test MSE Loss": test_loss, "Test MSE Mean Loss": test_loss_mean,
                       "Test MSE Std Loss": test_loss_std, "Test Raw CE Loss": test_loss_raw_ce,
                       "Test CE Loss": test_loss_ce, "Test EMD Loss": test_loss_emd, "Test Brier Score": test_brier_score})

        # Print the epoch loss
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss}, Train Loss Mean: {train_loss_mean}, "
              f"Train Loss Std: {train_loss_std}, Train Raw CE Loss: {train_loss_raw_ce}, Train CE Loss: {train_loss_ce}"
              f"Train EMD Loss: {train_loss_emd}, Test Loss: {test_loss}, Test Loss Mean: {test_loss_mean}, "
              f"Test Loss Std: {test_loss_std}, Test Raw CE Loss: {test_loss_raw_ce}, Test CE Loss: {test_loss_ce}, "
              f"Test EMD Loss: {test_loss_emd}, Test Brier Score: {test_brier_score}")
        if is_eval:
            raise Exception
        # Check if the current model has the best test loss so far
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model_resnet50.state_dict(), best_modelname)
