import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50
from torchvision.models.feature_extraction import create_feature_extractor
import numpy as np
from tqdm import tqdm
from PARA_dataloader import PARADataset
import wandb
from train_resnet_cls import earth_mover_distance
import normflows as nf


def train_with_flow(model_resnet, model_flow, train_dataloader, optimizer_resnet, optimizer_flow, device):
    model_resnet.eval()
    model_flow.train()
    running_kld_loss = 0.0
    running_emd_loss = 0.0

    progress_bar = tqdm(train_dataloader, leave=False)
    scale = torch.arange(1, 5.5, 0.5).to(device)
    sqrt_2pi = np.sqrt(2 * np.pi)    
    feature_extractor = create_feature_extractor(model_resnet, return_nodes={'layer4': 'layer4'})

    for images, mean_scores, std_scores, score_prob in progress_bar:
        images = images.to(device)
        mean_scores = mean_scores.to(device)
        std_scores = std_scores.to(device)
        score_prob = score_prob.to(device)

        # Extract features from ResNet for conditional flow
        with torch.no_grad():
            # context = model_resnet(images)
            context = feature_extractor(images)['layer4']
            context = F.adaptive_avg_pool2d(context, (1,1))[:,:,0,0]

        # Train Normalizing Flow model for score_prob
        optimizer_flow.zero_grad()
        # x = scale[torch.multinomial(score_prob, 1)]
        x = scale.repeat(len(images), 1).view(-1,1)
        context = context.repeat(1,len(scale)).view(-1,context.shape[-1])
        # Compute loss
        kld_loss = model_flow.forward_kld(x, context)
        log_prob_score_prob = model_flow.log_prob(x, context) # Use features as context for score_prob prediction
        log_prob_score_prob = log_prob_score_prob.view(-1,len(scale))
        prob = torch.exp(log_prob_score_prob)
        emd_loss = criterion_emd(prob, score_prob)
        
        # kld_loss.backward()
        emd_loss.backward()
        optimizer_flow.step()

        running_kld_loss += kld_loss.item()
        running_emd_loss += emd_loss.item()

        progress_bar.set_postfix(
            {
                "Train KLD Loss": kld_loss.item(),
                "Train EMD Loss": emd_loss.item(),
            }
        )

    epoch_kld_loss = running_kld_loss / len(train_dataloader)
    epoch_emd_loss = running_emd_loss / len(train_dataloader)
    return epoch_kld_loss, epoch_emd_loss


def evaluate_with_flow(model_resnet, model_flow, dataloader, device):
    model_resnet.eval()
    model_flow.eval()
    running_kld_loss = 0.0
    running_emd_loss = 0.0
    running_mse_mean_loss = 0.0
    running_mse_std_loss = 0.0
    running_ce_loss = 0.0
    running_raw_ce_loss = 0.0
    running_brier_score = 0.0

    scale = torch.arange(1, 5.5, 0.5).to(device)
    sqrt_2pi = np.sqrt(2 * np.pi)
    progress_bar = tqdm(dataloader, leave=False)
    feature_extractor = create_feature_extractor(model_resnet, return_nodes={'layer4': 'layer4'})
    with torch.no_grad():
        for images, mean_scores, std_scores, score_prob in progress_bar:
            images = images.to(device)
            mean_scores = mean_scores.to(device)
            std_scores = std_scores.to(device)
            score_prob = score_prob.to(device)

            # Extract features from ResNet for conditional flow
            with torch.no_grad():
                # context = model_resnet(images)
                context = feature_extractor(images)['layer4']
                context = F.adaptive_avg_pool2d(context, (1,1))[:,:,0,0]

            # Evaluate Normalizing Flow model for score_prob
            batch_scale = scale.repeat(len(images), 1).view(-1,1)
            batch_context = context.repeat(1,len(scale)).view(-1,context.shape[-1])
            log_prob_score_prob = model_flow.log_prob(batch_scale, batch_context) # Use features as context for score_prob prediction
            log_prob_score_prob = log_prob_score_prob.view(-1,len(scale))
            prob = torch.exp(log_prob_score_prob)
            kld_loss = model_flow.forward_kld(batch_scale, batch_context)
            
            # MSE loss for mean
            outputs_mean = torch.sum(prob * scale, dim=1, keepdim=True)
            mse_mean_loss = criterion(outputs_mean, mean_scores)

            # MSE loss for std
            outputs_std = torch.sqrt(torch.sum(prob * (scale - outputs_mean) ** 2, dim=1, keepdim=True))
            mse_std_loss = criterion(outputs_std, std_scores)

            # Brier Score
            brier_score = criterion(prob, score_prob)
            # EMD
            emd_loss = criterion_emd(prob, score_prob)

            # Cross-entropy loss
            # ce_loss = criterion_weight_ce(log_prob_score_prob, score_prob)
            # raw_ce_loss = criterion_raw_ce(log_prob_score_prob, score_prob)
            ce_loss = -torch.mean(log_prob_score_prob * score_prob * ce_weight)
            raw_ce_loss = -torch.mean(log_prob_score_prob * score_prob)

            running_kld_loss += kld_loss.item()
            running_emd_loss += emd_loss.item()
            running_mse_mean_loss += mse_mean_loss.item()
            running_mse_std_loss += mse_std_loss.item()
            running_ce_loss += ce_loss.item()
            running_raw_ce_loss += raw_ce_loss.item()
            running_brier_score += brier_score.item()

            progress_bar.set_postfix(
                {
                    "Eval EMD Loss": emd_loss.item(),
                    "MSE Mean Loss": mse_mean_loss.item(),
                    "MSE Std Loss": mse_std_loss.item(),
                }
            )

    epoch_kld_loss = running_kld_loss / len(dataloader)
    epoch_emd_loss = running_emd_loss / len(dataloader)
    epoch_mse_mean_loss = running_mse_mean_loss / len(dataloader)
    epoch_mse_std_loss = running_mse_std_loss / len(dataloader)
    epoch_ce_loss = running_ce_loss / len(dataloader)
    epoch_raw_ce_loss = running_raw_ce_loss / len(dataloader)
    epoch_brier_score = running_brier_score / len(dataloader)

    return epoch_kld_loss, epoch_mse_mean_loss, epoch_mse_std_loss, epoch_ce_loss, epoch_raw_ce_loss, epoch_emd_loss, epoch_brier_score


is_log = True
use_attr = False
use_hist = True
is_resume = False

if __name__ == '__main__':
    # Set random seed for reproducibility
    random_seed = 42
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    lr = 5e-4
    batch_size = 32
    num_epochs = 50
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
    model_resnet50.fc = nn.Linear(num_features, num_classes)
    model_resnet50.load_state_dict(torch.load('best_model_resnet50_cls_lr1e-03_100epoch_noattr_ce.pth'))
    # Move the model to the device
    model_resnet50 = model_resnet50.to(device)

    # Define flows
    K = 4
    latent_size = 1
    hidden_units = 128
    hidden_layers = 2
    context_size = 2048

    flows = []
    for i in range(K):
        flows += [nf.flows.AutoregressiveRationalQuadraticSpline(latent_size, hidden_layers, hidden_units,
                                                                num_context_channels=context_size)]
        flows += [nf.flows.LULinearPermute(latent_size)]
    # Set base distribution
    q0 = nf.distributions.DiagGaussian(1, trainable=False)
    # Construct flow model
    model_flow = nf.ConditionalNormalizingFlow(q0, flows)
    # FLAG
    if is_resume:
        model_flow.load_state_dict(torch.load('best_model_resnet50_flow_lr5e-04_50epoch_noattr.pth'))
    model_flow = model_flow.to(device)
    
    # Define the loss function
    criterion = nn.MSELoss()
    ce_weight = 1 / train_dataset.aesthetic_score_hist_prob
    ce_weight = torch.tensor(ce_weight / np.sum(ce_weight) * len(ce_weight), device=device)
    criterion_weight_ce = nn.CrossEntropyLoss(weight=ce_weight)
    criterion_raw_ce = nn.CrossEntropyLoss()
    criterion_emd = earth_mover_distance

    # Define the optimizer
    optimizer_resnet50 = optim.SGD(model_resnet50.parameters(), lr=lr, momentum=0.9)
    optimizer_flow = optim.SGD(model_flow.parameters(), lr=lr, momentum=0.9)

    # Initialize the best test loss and the best model
    best_test_loss = float('inf')
    best_model = None
    best_modelname = 'best_model_resnet50_flow_emd_lr%1.0e_%depoch' % (lr, num_epochs)
    if not use_attr:
        best_modelname += '_noattr'
    if is_resume:
        best_modelname += '_resume'
    best_modelname += '.pth'

    # Training loop for ResNet-50
    for epoch in range(num_epochs):
        # Training
        train_loss, train_emd_loss = train_with_flow(model_resnet50, model_flow, train_dataloader, optimizer_resnet50, optimizer_flow, device)
        if is_log:
            wandb.log({"Train KLD Loss": train_loss,
                       "Train EMD Loss": train_emd_loss}, commit=False)

        # Testing
        test_loss, test_loss_mean, test_loss_std, test_loss_ce, test_loss_raw_ce, test_loss_emd, test_brier_score = evaluate_with_flow(model_resnet50, model_flow, test_dataloader, device)
        if is_log:
            wandb.log({"Test KLD Loss": test_loss, "Test MSE Mean Loss": test_loss_mean,
                       "Test MSE Std Loss": test_loss_std, "Test Raw CE Loss": test_loss_raw_ce,
                       "Test CE Loss": test_loss_ce, "Test EMD Loss": test_loss_emd, "Test Brier Score": test_brier_score})

        # Print the epoch loss
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss}, Test Loss: {test_loss}, Test Loss Mean: {test_loss_mean}"
              f"Test Loss Std: {test_loss_std}, Test Raw CE Loss: {test_loss_raw_ce}, Test CE Loss: {test_loss_ce}, Test EMD Loss: {test_loss_emd}, Test Brier Score: {test_brier_score}")
        
        # Check if the current model has the best test loss so far
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model_flow.state_dict(), best_modelname)

