import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
# from torchvision.models import resnet50
# import numpy as np
from tqdm import tqdm
from PARA_PIAA_dataloader import load_data, PARA_PIAADataset, collect_batch_attribute, collect_batch_personal_trait, split_dataset_by_user, split_dataset_by_images, create_user_split_dataset_kfold
import wandb
from scipy.stats import spearmanr
from train_paiaa_giaa import PAIAA_GIAA_Model
import pandas as pd
import argparse
import math


class PerModelCorrection(nn.Module):
    def __init__(self, keep_probability, inputsize):

        super(PerModelCorrection, self).__init__()
        self.fc1 = nn.Linear(inputsize, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.drop_prob = (1 - keep_probability)
        self.relu1 = nn.PReLU()
        self.drop1 = nn.Dropout(self.drop_prob)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.relu2 = nn.PReLU()
        self.drop2 = nn.Dropout(p=self.drop_prob)
        self.fc3 = nn.Linear(512, 5)
        self.tanh = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Weight initialization reference: https://arxiv.org/abs/1502.01852
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()

    def forward(self, x):
        """
        Feed-forward pass.
        :param x: Input tensor
        : return: Output tensor
        """
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.drop1(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.drop2(out)
        out = self.fc3(out)
        out = self.tanh(out)
        return out


class PAIAA(nn.Module):
    def __init__(self, num_bins, keep_probability, giaa_score_scale):
        super(PAIAA, self).__init__()
        self.multi_tasks_model = PAIAA_GIAA_Model(num_bins, keep_probability)
        self.percorr = PerModelCorrection(keep_probability, self.multi_tasks_model.num_ftrs)
        self.giaa_score_scale = giaa_score_scale
    
    def forward(self, x):
        x = self.multi_tasks_model.backbone(x)
        giaa_logits = self.multi_tasks_model.AesNet(x)
        giaa_prob = F.softmax(giaa_logits, dim=1)
        giaa_mean_score = (giaa_prob * self.giaa_score_scale).sum(1)
        big5 = self.multi_tasks_model.PerNet(x)
        
        big5_weight = self.percorr(x)
        piaa_corr = torch.sum(big5_weight * big5, 1)
        x = giaa_mean_score + piaa_corr
        return x


def train(model, dataloader, optimizer, device):
    model.train()
    running_mse_loss = 0.0

    progress_bar = tqdm(dataloader, leave=False)
    # scale = torch.arange(1, 5.5, 0.5).to(device)    
    for sample in progress_bar:
        images = sample['image']
        sample_score, _ = collect_batch_attribute(sample)
        images = images.to(device)
        sample_score = sample_score.to(device).float()
        piaa_score = model(images)
        
        # MSE loss
        optimizer.zero_grad()
        loss = criterion_mse(piaa_score, sample_score)
        loss.backward()
        optimizer.step()

        running_mse_loss += loss.item()

        progress_bar.set_postfix({
            'Train MSE Mean Loss': loss.item(),
        })

    epoch_mse_loss = running_mse_loss / len(dataloader)
    return epoch_mse_loss


def evaluate(model, dataloader, device):
    model.eval()  # Set the model to evaluation mode
    running_mse_loss = 0.0

    all_predicted_scores = []  # List to store all predictions
    all_true_scores = []  # List to store all true scores

    progress_bar = tqdm(dataloader, leave=False)
    # scale = torch.arange(1, 5.5, 0.5).to(device)
    for sample in progress_bar:
        with torch.no_grad():
            images = sample['image']
            sample_score, _ = collect_batch_attribute(sample)
            images = images.to(device)
            sample_score = sample_score.to(device).float()
            piaa_score = model(images)

            # MSE loss
            loss = criterion_mse(piaa_score, sample_score)
            running_mse_loss += loss.item()

            # Store predicted and true scores for SROCC calculation
            predicted_scores = piaa_score.view(-1).cpu().numpy()
            true_scores = sample_score.view(-1).cpu().numpy()
            all_predicted_scores.extend(predicted_scores)
            all_true_scores.extend(true_scores)

            progress_bar.set_postfix({'Test MSE Mean Loss': loss.item()})

    epoch_mse_loss = running_mse_loss / len(dataloader)

    # Calculate SROCC for all predictions
    srocc, _ = spearmanr(all_predicted_scores, all_true_scores)

    return epoch_mse_loss, srocc


def trainer(dataloaders, model, optimizer, args, train_fn, evaluate_fn, device, best_modelname):
    
    train_dataloader, val_dataloader, test_dataloader = dataloaders

    num_patience_epochs = 0
    best_test_srocc = 0
    for epoch in range(args.num_epochs):
        if args.is_eval:
            break
        # Learning rate schedule
        if (epoch + 1) % args.lr_schedule_epochs == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= args.lr_decay_factor
        # Training
        train_mse_loss = train_fn(model, train_dataloader, optimizer, device)
        # Testing
        val_mse_loss, val_srocc = evaluate_fn(model, val_dataloader, device)
        
        if args.is_log:
            wandb.log({"Train PIAA MSE Loss": train_mse_loss,
                       "Val PIAA MSE Loss": val_mse_loss,
                        "Val PIAA SROCC": val_srocc,
                    }, commit=True)

        # Early stopping check
        if val_srocc > best_test_srocc:
            best_test_srocc = val_srocc
            num_patience_epochs = 0
            torch.save(model.state_dict(), best_modelname)
        else:
            num_patience_epochs += 1
            if num_patience_epochs >= args.max_patience_epochs:
                print("Validation loss has not decreased for {} epochs. Stopping training.".format(args.max_patience_epochs))
                break
    
    if not args.is_eval:
        model.load_state_dict(torch.load(best_modelname))
    # Testing
    test_mse_loss, test_srocc = evaluate_fn(model, test_dataloader, device)
    if args.is_log:
        wandb.log({"Test PIAA MSE Loss": test_mse_loss,
                   "Test PIAA SROCC": test_srocc}, commit=True)
    print(
        # f"Train PIAA MSE Loss: {train_mse_loss:.4f}, "
        f"Test PIAA MSE Loss: {test_mse_loss:.4f}, "
        f"Test PIAA SROCC Loss: {test_srocc:.4f}, ")
        

criterion_mse = nn.MSELoss()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training and Testing the Combined Model for data spliting')
    parser.add_argument('--trainset', type=str, default='GIAA', choices=["GIAA", "sGIAA", "PIAA"])
    parser.add_argument('--fold_id', type=int, default=1)
    parser.add_argument('--n_fold', type=int, default=4)
    parser.add_argument('--trait', type=str, default=None)
    parser.add_argument('--value', type=str, default=None)    
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--train_piaa_augment', type=bool, default=True)
    parser.add_argument('--pretrained_model', type=str, required=True)
    parser.add_argument('--use_cv', action='store_true', help='Enable cross validation')
    parser.add_argument('--is_eval', action='store_true', help='Enable evaluation mode')
    parser.add_argument('--no_log', action='store_false', dest='is_log', help='Disable logging')
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--max_patience_epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--lr_schedule_epochs', type=int, default=5)
    parser.add_argument('--lr_decay_factor', type=float, default=0.5)

    args = parser.parse_args()
    resume = args.resume
    batch_size = args.batch_size
    random_seed = 42
    n_workers = 8
    num_bins = 9
    # num_attr = 8
    # num_pt = 25 # number of personal trait
    
    if args.is_log:
        tags = ["no_attr","PIAA"]
        if args.use_cv:
            tags += ["CV%d/%d"%(args.fold_id, args.n_fold)]
        wandb.init(project="resnet_PARA_PIAA",
                notes="PAIAA",
                tags = tags)
        wandb.config = {
            "learning_rate": args.lr,
            "batch_size": batch_size,
            "num_epochs": args.num_epochs
        }
        experiment_name = wandb.run.name
    else:
        experiment_name = ''
    
    # Create dataloaders for training and test sets
    train_dataset, val_dataset, test_dataset = load_data(args)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers, timeout=300)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers, timeout=300)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers, timeout=300)
    dataloaders = (train_dataloader, val_dataloader, test_dataloader)
    # Define the device for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    giaa_score_scale = torch.arange(1, 5.5, 0.5).to(device).to(device)
    model = PAIAA(num_bins, keep_probability=0.9, giaa_score_scale=giaa_score_scale)
    model.multi_tasks_model.load_state_dict(torch.load(args.pretrained_model))
    model = model.to(device)

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Initialize the best test loss and the best model
    best_model = None
    best_modelname = 'best_model_resnet50_paiaa_lr%1.0e_decay_%depoch' % (args.lr, args.num_epochs)
    best_modelname += '_%s'%experiment_name
    best_modelname += '.pth'
    dirname = 'models_pth'
    if args.use_cv:
        dirname = os.path.join(dirname, 'random_cvs')
    best_modelname = os.path.join(dirname, best_modelname)

    # Training loop
    trainer(dataloaders, model, optimizer, args, train, evaluate, device, best_modelname)