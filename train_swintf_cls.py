import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from PARA_dataloader import PARADataset
import wandb
from transformers import AutoImageProcessor, Swinv2Model
from train_resnet_cls import earth_mover_distance


class SwinTFAdapter(nn.Module):
    def __init__(self, num_classes):
        super(SwinTFAdapter, self).__init__()
        self.image_processor = AutoImageProcessor.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
        self.model = Swinv2Model.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
        self.adapter = nn.Linear(self.model.num_features, num_classes)

    def forward(self, images):
        inputs = self.image_processor(images, return_tensors="pt").to(images.device)
        outputs = self.model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        logits = self.adapter(torch.mean(last_hidden_states, dim=1))
        return logits


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

        # Forward pass
        logits = model(images)
        prob = F.softmax(logits, dim=1)

        # Cross-entropy loss
        ce_loss = criterion_weight_ce(logits, score_prob)
        raw_ce_loss = criterion_raw_ce(logits, score_prob)

        # MSE loss for mean
        outputs_mean = torch.sum(prob * scale, dim=1, keepdim=True)
        mse_mean_loss = criterion_mse(outputs_mean, mean_scores)

        # MSE loss for std
        outputs_std = torch.sqrt(torch.sum(prob * (scale - outputs_mean) ** 2, dim=1, keepdim=True))
        mse_std_loss = criterion_mse(outputs_std, std_scores)

        # Earth Mover's Distance (EMD) loss
        emd_loss = criterion_emd(prob, score_prob)

        if use_ce:
            loss = ce_loss
        else:
            loss = emd_loss

        # Backward pass and optimization
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


is_log = True
use_attr = False
use_hist = True
use_ce = False
# resume = 'best_model_vae_cls_lr1e-01_100epoch_noattr.pth'
resume = None
is_eval = False


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
        num_classes = 9 + 5 * 7
    else:
        num_classes = 9

    # Define the device for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create the model with CLIP as the backbone and an adapter linear layer
    model_vae = SwinTFAdapter(num_classes)
    if resume is not None:
        model_vae.load_state_dict(torch.load(resume))
    # Move the model to the device
    model_vae = model_vae.to(device)
    
    # Define the loss functions
    ce_weight = 1 / train_dataset.aesthetic_score_hist_prob
    ce_weight = ce_weight / np.sum(ce_weight) * len(ce_weight)
    criterion_weight_ce = nn.CrossEntropyLoss(weight=torch.tensor(ce_weight, device=device))
    criterion_raw_ce = nn.CrossEntropyLoss()
    criterion_mse = nn.MSELoss()
    criterion_emd = earth_mover_distance

    # Define the optimizer
    optimizer_clip = optim.SGD(model_vae.parameters(), lr=lr, momentum=0.9)

    # Initialize the best test loss and the best model
    best_test_loss = float('inf')
    best_model = None
    best_modelname = 'best_model_swintf_cls_lr%1.0e_%depoch' % (lr, num_epochs)
    if not use_attr:
        best_modelname += '_noattr'
    if use_ce:
        best_modelname += '_ce'
    if resume is not None:
        best_modelname += '_ft'
    best_modelname += '.pth'

    # Training loop
    for epoch in range(num_epochs):
        # Training
        train_ce_loss, train_raw_ce_loss, train_mse_mean_loss, train_mse_std_loss, train_emd_loss = train(
            model_vae, train_dataloader, criterion_weight_ce, criterion_raw_ce, criterion_mse, criterion_emd,
            optimizer_clip, device)
        if is_log:
            wandb.log({"Train CE Loss": train_ce_loss,
                       "Train Raw CE Loss": train_raw_ce_loss,
                       "Train MSE Mean Loss": train_mse_mean_loss,
                       "Train MSE Std Loss": train_mse_std_loss,
                       "Train EMD Loss": train_emd_loss}, commit=False)

        # Testing
        test_ce_loss, test_raw_ce_loss, test_mse_mean_loss, test_mse_std_loss, test_emd_loss, test_brier_score = evaluate(
            model_vae, test_dataloader, criterion_weight_ce, criterion_raw_ce, criterion_mse, criterion_emd,
            device)
        if is_log:
            wandb.log({"Test CE Loss": test_ce_loss,
                       "Test Raw CE Loss": test_raw_ce_loss,
                       "Test MSE Mean Loss": test_mse_mean_loss,
                       "Test MSE Std Loss": test_mse_std_loss,
                       "Test EMD Loss": test_emd_loss})

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
              f"Test Brier Score: {test_brier_score}")
        
        if is_eval:
            raise Exception

        # Check if the current model has the best test loss so far
        test_loss = test_ce_loss if use_ce else test_emd_loss
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model_vae.state_dict(), best_modelname)
