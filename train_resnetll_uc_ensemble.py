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
from train_resnet_dup import metric_to_cdf
from train_resnetll_uc import evaluate, custom_criterion


def evaluate_ensemble(dataloader, criterion, criterion_raw_ce, criterion_emd, device, ensemble_mean, ensemble_std):
    running_mse_loss = 0.0
    running_custom_loss = 0.0
    running_mse_std_loss = 0.0
    running_ce_loss = 0.0
    running_raw_ce_loss = 0.0
    running_emd_loss = 0.0
    running_brier_score = 0.0
    dropout_outputs = []

    scale = torch.arange(1, 5.5, 0.5).to(device)
    sqrt_2pi = np.sqrt(2 * np.pi)
    progress_bar = tqdm(dataloader, leave=False)
    with torch.no_grad():
        for i, (images, mean_scores, std_scores, score_prob) in enumerate(progress_bar):
            images = images.to(device)
            mean_scores = mean_scores.to(device)
            std_scores = std_scores.to(device)
            score_prob = score_prob.to(device)

            # batch_outputs_mean = torch.mean(batch_outputs, dim=0)
            # batch_outputs_std = torch.std(batch_outputs, dim=0)
            batch_outputs_mean = ensemble_mean[batch_size*i:batch_size*(i+1)]
            batch_outputs_std = ensemble_std[batch_size*i:batch_size*(i+1)]
            batch_outputs = torch.stack([batch_outputs_mean, batch_outputs_std], dim=-1)
            dropout_outputs.append(batch_outputs)

            emd_loss, brier_score, ce_loss, raw_ce_loss = metric_to_cdf(scale, batch_outputs_mean, batch_outputs_std, score_prob, ce_weight=ce_weight)

            mse_mean_loss = criterion(batch_outputs_mean, mean_scores)
            mse_std_loss = criterion(batch_outputs_std, std_scores)
            custom_loss = custom_criterion(batch_outputs_mean, mean_scores, batch_outputs_std)

            running_mse_loss += mse_mean_loss.item()
            running_mse_std_loss += mse_std_loss.item()
            running_custom_loss += custom_loss.item()
            running_ce_loss += ce_loss.item()
            running_raw_ce_loss += raw_ce_loss.item()
            running_emd_loss += emd_loss.item()
            running_brier_score += brier_score.item()
            progress_bar.set_postfix({'MSE Loss': mse_mean_loss.item(), 'Custom Loss': custom_loss.item()})

    epoch_mse_loss = running_mse_loss / len(dataloader)
    epoch_mse_std_loss = running_mse_std_loss / len(dataloader)
    epoch_custom_loss = running_custom_loss / len(dataloader)
    epoch_ce_loss = running_ce_loss / len(dataloader)
    epoch_raw_ce_loss = running_raw_ce_loss / len(dataloader)
    epoch_emd_loss = running_emd_loss / len(dataloader)
    epoch_brier_score = running_brier_score / len(dataloader)

    dropout_outputs = torch.cat(dropout_outputs, dim=0)
    dropout_mean = dropout_outputs[..., 0]
    dropout_std = dropout_outputs[..., 1]
    return epoch_mse_loss, epoch_mse_std_loss, epoch_custom_loss, epoch_ce_loss, epoch_raw_ce_loss, epoch_emd_loss, epoch_brier_score, dropout_mean, dropout_std


is_log = False
use_attr = False
use_hist = True
use_uc = False

if __name__ == '__main__':
    # random_seed = 42
    # torch.manual_seed(random_seed)
    # torch.cuda.manual_seed(random_seed)
    # np.random.seed(random_seed)
    # random.seed(random_seed)
    random_seed = None

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
    mcdo_dir = 'mcdo_exp'
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

    # Define the number of classes in your dataset
    if use_attr:
        num_classes = 9 + 5 * 7
    else:
        num_classes = 9

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_resnet50 = resnet50(pretrained=True)
    num_features = model_resnet50.fc.in_features
    model_resnet50.fc = nn.Sequential(
        nn.Dropout(0.1),
        nn.Linear(num_features, num_classes)
    )
    model_resnet50.load_state_dict(torch.load("best_model_resnet50_noattr.pth"), strict=False)
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
    best_modelname = 'best_model_resnet50_lluc_lr%1.0e_%depoch' % (lr, num_epochs)
    if not use_attr:
        best_modelname += '_noattr'
    if not use_uc:
        best_modelname += '_nouc'        
    best_modelname += '.pth'

    # Load pretrained models and compute the ensemble mean and variance
    num_models = 5
    ensemble_mean = 0.0
    ensemble_var = 0.0

    for i in range(num_models):
        model_resnet50 = resnet50(pretrained=True)
        num_features = model_resnet50.fc.in_features
        model_resnet50.fc = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(num_features, num_classes)
        )

        best_modelname = 'best_model_resnet50_lluc%d_lr%1.0e_%depoch' % (i, lr, num_epochs)
        if not use_attr:
            best_modelname += '_noattr'
        if not use_uc:
            best_modelname += '_nouc'
        best_modelname += '.pth'
        best_modelname = os.path.join(mcdo_dir, best_modelname)
        model_resnet50.load_state_dict(torch.load(best_modelname), strict=False)
        model_resnet50 = model_resnet50.to(device)
        
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        test_mse_loss, test_mse_std_loss, test_custom_loss, test_ce_loss, test_raw_ce_loss, test_emd_loss, test_brier_score, dropout_mean, dropout_std = evaluate(
            model_resnet50, test_dataloader, criterion, ce_weight, device, num_samples=50)
        
        print(f"Model {i}: Test MSE Mean Loss: {test_mse_loss}, Test MSE Std Loss: {test_mse_std_loss}, "
              f"Test Custom Loss: {test_custom_loss}, Test CE Loss: {test_ce_loss}, "
              f"Test Raw CE Loss: {test_raw_ce_loss}, Test EMD Loss: {test_emd_loss}, Test Brier Score: {test_brier_score}")

        ensemble_mean += dropout_mean
        ensemble_var += (dropout_std ** 2 + dropout_mean**2)

    ensemble_mean /= num_models
    ensemble_var /= num_models
    ensemble_var -= ensemble_mean**2
    ensemble_std = torch.sqrt(ensemble_var)
    
    test_mse_loss, test_mse_std_loss, test_custom_loss, test_ce_loss, test_raw_ce_loss, test_emd_loss, test_brier_score, dropout_mean, dropout_std = evaluate_ensemble(
        test_dataloader, criterion, criterion_raw_ce, criterion_emd, device, ensemble_mean, ensemble_std)
    
    print(f"Ensemble Model: Test MSE Mean Loss: {test_mse_loss}, Test MSE Std Loss: {test_mse_std_loss}, "
            f"Test Custom Loss: {test_custom_loss}, Test CE Loss: {test_ce_loss}, Test Raw CE Loss: {test_raw_ce_loss}, "
            f"Test EMD Loss: {test_emd_loss}, Test Brier Score: {test_brier_score}")
    
