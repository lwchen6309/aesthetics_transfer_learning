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
from train_resnet_dup import earth_mover_distance_to_cdf


def train(model, train_dataloader, criterion, optimizer, device):
    model.train()
    running_mse_loss = 0.0
    running_custom_loss = 0.0

    progress_bar = tqdm(train_dataloader, leave=False)
    for images, mean_scores, std_scores, score_prob in progress_bar:
        images = images.to(device)
        mean_scores = mean_scores.to(device)
        std_scores = std_scores.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        mse_loss = criterion(outputs, mean_scores)
        custom_loss = custom_criterion(outputs, mean_scores, std_scores)

        if use_uc:
            loss = custom_loss
        else:
            loss = mse_loss

        loss.backward()
        optimizer.step()

        running_mse_loss += mse_loss.item()
        running_custom_loss += custom_loss.item()
        progress_bar.set_postfix({'MSE Loss': mse_loss.item(), 'Custom Loss': custom_loss.item()})

    epoch_mse_loss = running_mse_loss / len(train_dataloader)
    epoch_custom_loss = running_custom_loss / len(train_dataloader)
    return epoch_mse_loss, epoch_custom_loss



def evaluate(model, dataloader, criterion, ce_weight, device, num_samples=50):
    model.train()
    running_mse_loss = 0.0
    running_custom_loss = 0.0
    running_mse_std_loss = 0.0
    running_ce_loss = 0.0
    running_raw_ce_loss = 0.0
    running_emd_loss = 0.0
    dropout_outputs = []

    scale = torch.arange(1, 5.5, 0.5).to(device)
    sqrt_2pi = np.sqrt(2 * np.pi)
    progress_bar = tqdm(dataloader, leave=False)
    with torch.no_grad():
        for images, mean_scores, std_scores, score_prob in progress_bar:
            images = images.to(device)
            mean_scores = mean_scores.to(device)
            std_scores = std_scores.to(device)
            score_prob = score_prob.to(device)

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

            # prob = torch.exp(-0.5 * ((batch_outputs_mean - scale) / batch_outputs_std) ** 2) / batch_outputs_std / sqrt_2pi
            # prob = prob / torch.sum(prob, dim=1, keepdim=True)
            # logit = torch.log(prob + 1e-6)
            # ce_loss = -torch.mean(logit * score_prob * ce_weight)
            # raw_ce_loss = -torch.mean(logit * score_prob)
            # emd_loss = criterion_emd(prob, score_prob)

            # Continuous
            ce_offset = -(torch.log(batch_outputs_std) + 0.5 * np.log(2*np.pi)) * 4.5
            logit = - 0.5 * ((scale + 0.25 - batch_outputs_mean)**3 - (scale - 0.25 - batch_outputs_mean)**3) / 3 / batch_outputs_std**2 + ce_offset
            ce_loss = -torch.mean(logit * score_prob * ce_weight)
            raw_ce_loss = -torch.mean(logit * score_prob)
            emd_loss = earth_mover_distance_to_cdf(scale, batch_outputs_mean, batch_outputs_std, score_prob)

            mse_mean_loss = criterion(batch_outputs_mean, mean_scores)
            mse_std_loss = criterion(batch_outputs_std, std_scores)
            custom_loss = custom_criterion(batch_outputs_mean, mean_scores, batch_outputs_std)
            
            running_mse_loss += mse_mean_loss.item()
            running_mse_std_loss += mse_std_loss.item()
            running_custom_loss += custom_loss.item()
            running_ce_loss += ce_loss.item()
            running_raw_ce_loss += raw_ce_loss.item()
            running_emd_loss += emd_loss.item()
            progress_bar.set_postfix({'MSE Loss': mse_mean_loss.item(), 'Custom Loss': custom_loss.item()})

    epoch_mse_loss = running_mse_loss / len(dataloader)
    epoch_mse_std_loss = running_mse_std_loss / len(dataloader)
    epoch_custom_loss = running_custom_loss / len(dataloader)
    epoch_ce_loss = running_ce_loss / len(dataloader)
    epoch_raw_ce_loss = running_raw_ce_loss / len(dataloader)
    epoch_emd_loss = running_emd_loss / len(dataloader)
    dropout_outputs = torch.cat(dropout_outputs, dim=0)
    dropout_mean = dropout_outputs[..., 0]
    dropout_std = dropout_outputs[..., 1]
    return epoch_mse_loss, epoch_mse_std_loss, epoch_custom_loss, epoch_ce_loss, epoch_raw_ce_loss, epoch_emd_loss, dropout_mean, dropout_std

def custom_criterion(outputs, mean_scores, std_scores):
    return 0.5 * torch.mean(((torch.abs(outputs - mean_scores) + 1e-6) / (std_scores + 1e-6)) ** 2)



is_log = True
use_attr = False
use_hist = True
use_uc = True


if __name__ == '__main__':
    random_seed = 42
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    # random_seed = None

    lr = 5e-4
    batch_size = 32
    num_epochs = 10
    if is_log:
        wandb.init(project="resnet_PARA_GIAA")
        wandb.config.update({
            "learning_rate": lr,
            "batch_size": batch_size,
            "num_epochs": num_epochs
        })

    root_dir = '/home/lwchen/datasets/PARA/'

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    train_dataset = PARADataset(root_dir, transform=train_transform, train=True, use_attr=use_attr, use_hist=use_hist, random_seed=random_seed)
    test_dataset = PARADataset(root_dir, transform=test_transform, train=False, use_attr=use_attr, use_hist=use_hist, random_seed=random_seed)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if use_attr:
        num_classes = 9
    else:
        num_classes = 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_resnet50 = resnet50(pretrained=True)
    num_features = model_resnet50.fc.in_features
    ll_dims = 128
    model_resnet50.fc = nn.Sequential(
        nn.Dropout(0.1),
        nn.Linear(num_features, ll_dims),
        nn.LeakyReLU(),
        nn.Dropout(0.1),
        nn.Linear(ll_dims, ll_dims),
        nn.LeakyReLU(),
        nn.Dropout(0.1),
        nn.Linear(ll_dims, num_classes)
    )
    model_resnet50.load_state_dict(torch.load("best_model_resnet50_noattr.pth"), strict=False)
    # model_resnet50.load_state_dict(torch.load("best_model_resnet50_lluc0_lr5e-04_10epoch_noattr.pth"))
    model_resnet50 = model_resnet50.to(device)


    criterion = nn.MSELoss()
    ce_weight = 1 / train_dataset.aesthetic_score_hist_prob
    ce_weight = ce_weight / np.sum(ce_weight) * len(ce_weight)
    ce_weight = torch.tensor(ce_weight, device=device)
    criterion_weight_ce = nn.CrossEntropyLoss(weight=ce_weight)
    criterion_raw_ce = nn.CrossEntropyLoss()
    criterion_emd = earth_mover_distance

    optimizer_resnet50 = optim.SGD(model_resnet50.fc.parameters(), lr=lr, momentum=0.9)

    best_test_loss = float('inf')
    best_model = None
    best_modelname = 'best_model_resnet50_lluc_nlayer_lr%1.0e_%depoch' % (lr, num_epochs)
    if not use_attr:
        best_modelname += '_noattr'
    if not use_uc:
        best_modelname += '_nouc'
    best_modelname += '.pth'

    for epoch in range(num_epochs):
        train_mse_loss, train_custom_loss = train(model_resnet50, train_dataloader, criterion, optimizer_resnet50, device)
        if is_log:
            wandb.log({"Train MSE Loss": train_mse_loss, "Train Custom Loss": train_custom_loss}, commit=False)

        test_mse_loss, test_mse_std_loss, test_custom_loss, test_ce_loss, test_raw_ce_loss, test_emd_loss, dropout_mean, dropout_std = evaluate(
            model_resnet50, test_dataloader, criterion, ce_weight, device, num_samples=50)
        if is_log:
            wandb.log({
                "Test MSE Mean Loss": test_mse_loss,
                "Test MSE Std Loss": test_mse_std_loss,
                "Test Custom Loss": test_custom_loss,
                "Test CE Loss": test_ce_loss,
                "Test Raw CE Loss": test_raw_ce_loss,
                "Test EMD Loss": test_emd_loss,
            })

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Train MSE Loss: {train_mse_loss}, Train Custom Loss: {train_custom_loss}, "
            f"Test MSE Loss: {test_mse_loss}, Test MSE Std Loss: {test_mse_std_loss}, "
            f"Test CE Loss: {test_ce_loss}, Test Raw CE Loss: {test_raw_ce_loss}, Test EMD Loss: {test_emd_loss}"
        )
             
        if test_mse_loss < best_test_loss:
            test_custom_loss = test_mse_loss
            torch.save(model_resnet50.state_dict(), best_modelname)
    
