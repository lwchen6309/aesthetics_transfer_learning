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
from scipy.stats import spearmanr
import argparse


# Model Definition
class CombinedModel(nn.Module):
    def __init__(self, num_bins, num_attrs, h_dims=512, resume=None):
        super(CombinedModel, self).__init__()
        tot_num_bins = num_bins + num_attrs
        self.num_bins = num_bins
        self.num_attrs = num_attrs
        self.resnet = resnet50(pretrained=True)
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, h_dims),
            nn.ReLU(),
            nn.Linear(h_dims, tot_num_bins),
        )
        if resume is not None:
            self.resnet.load_state_dict(torch.load(resume))
    
    def forward(self, images):
        outputs = self.resnet(images)
        logit = outputs[:,:self.num_bins]
        attr_mean_pred = outputs[:,self.num_bins:]
        return logit, attr_mean_pred


def earth_mover_distance(x, y):
    """
    Compute Earth Mover's Distance (EMD) between two 1D tensors x and y using 2-norm.
    """
    cdf_x = torch.cumsum(x, dim=1)
    cdf_y = torch.cumsum(y, dim=1)
    emd = torch.norm(cdf_x - cdf_y, p=2, dim=1)
    return torch.mean(emd)


def train(model, train_dataloader, optimizer, device):
    model.train()
    running_mse_mean_loss = 0.0
    running_mse_std_loss = 0.0
    running_emd_loss = 0.0
    running_attr_mse_loss = 0.0
    criterion_mse = nn.MSELoss()
    criterion_emd = earth_mover_distance

    progress_bar = tqdm(train_dataloader, leave=False)
    scale = torch.arange(1, 5.5, 0.5).to(device)
    for images, mean_scores, std_scores, score_prob in progress_bar:
        if is_eval:
            break
        images = images.to(device)
        mean_scores = mean_scores.to(device)
        std_scores = std_scores.to(device)
        score_prob = score_prob.to(device)
        
        attr_mean_scores = mean_scores[:,1:] # Remove aesthetic score
        aesthetic_mean_scores = mean_scores[:,0][:,None]
        aesthetic_std_scores = std_scores[:,0][:,None]
        score_prob = score_prob[:,:num_bins] # Take only score distribution

        optimizer.zero_grad()

        logit, attr_mean_pred = model(images)
        prob = F.softmax(logit, dim=1)
        
        # Earth Mover's Distance (EMD) loss
        emd_loss = criterion_emd(prob, score_prob)
        mse_attr_loss = criterion_mse(attr_mean_scores, attr_mean_pred)

        with torch.no_grad():
            # MSE loss for mean
            outputs_mean = torch.sum(prob * scale, dim=1, keepdim=True)
            mse_mean_loss = criterion_mse(outputs_mean, aesthetic_mean_scores)

            # MSE loss for std
            outputs_std = torch.sqrt(torch.sum(prob * (scale - outputs_mean) ** 2, dim=1, keepdim=True))            
            mse_std_loss = criterion_mse(outputs_std, aesthetic_std_scores)

        loss = emd_loss + mse_attr_loss
        loss.backward()
        optimizer.step()

        running_mse_mean_loss += mse_mean_loss.item()
        running_mse_std_loss += mse_std_loss.item()
        running_emd_loss += emd_loss.item()
        running_attr_mse_loss += mse_attr_loss.item()

        progress_bar.set_postfix({
            'Train MSE MeanAttr Loss': mse_attr_loss.item(),
            'Train EMD Loss': emd_loss.item()
        })

    epoch_mse_mean_loss = running_mse_mean_loss / len(train_dataloader)
    epoch_mse_std_loss = running_mse_std_loss / len(train_dataloader)
    epoch_emd_loss = running_emd_loss / len(train_dataloader)
    epoch_attr_mse_loss = running_attr_mse_loss / len(train_dataloader)

    return epoch_mse_mean_loss, epoch_mse_std_loss, epoch_attr_mse_loss, epoch_emd_loss


def evaluate(model, dataloader, num_bins, num_attr, device, eval_srocc=True):
    model.eval()
    running_emd_loss = 0.0
    running_mse_loss = 0.0
    running_attr_mse_loss = 0.0
    criterion_mse = nn.MSELoss()
    criterion_emd = earth_mover_distance

    scale = torch.arange(1, 5.5, 0.5).to(device)

    progress_bar = tqdm(dataloader, leave=False)
    mean_pred = []
    mean_target = []
    with torch.no_grad():
        for images, mean_scores, std_scores, score_prob in progress_bar:
            images = images.to(device)
            score_prob = score_prob.to(device)[:,:num_bins] # Take only score distribution
            mean_scores = mean_scores.to(device)
            attr_mean_scores = mean_scores[:,1:] # Remove aesthetic score

            logit, attr_mean_pred = model(images)
            prob = F.softmax(logit, dim=1)

            emd_loss = criterion_emd(prob, score_prob)
            mse_attr_loss = criterion_mse(attr_mean_scores, attr_mean_pred)
            running_emd_loss += emd_loss.item()
            running_attr_mse_loss += mse_attr_loss.item()
            
            if eval_srocc:
                # MSE loss for mean
                outputs_mean = torch.sum(prob * scale, dim=1, keepdim=True)
                mse_loss = criterion_mse(outputs_mean, mean_scores[:,0].unsqueeze(1))
                running_mse_loss += mse_loss.item()
                mean_pred.append(outputs_mean.view(-1).cpu().numpy())
                mean_target.append(mean_scores[:,0].cpu().numpy())

            progress_bar.set_postfix({
                'Eval EMD Loss': emd_loss.item(),
                'Eval MSE Loss': mse_loss.item() if eval_srocc else 0,
            })

    # Calculate SROCC
    srocc = None
    if eval_srocc:
        predicted_scores = np.concatenate(mean_pred, axis=0)
        true_scores = np.concatenate(mean_target, axis=0)
        srocc, _ = spearmanr(predicted_scores, true_scores)
    
    attr_mse_loss = running_attr_mse_loss / len(dataloader)
    emd_loss = running_emd_loss / len(dataloader)
    mse_loss = running_mse_loss / len(dataloader) if eval_srocc else 0
    return emd_loss, attr_mse_loss, mse_loss, srocc


def load_data(args, root_dir = '/home/lwchen/datasets/PARA/'):
    # Define the root directory of the PARA dataset

    # Define transformations for training set and test set
    train_transform = transforms.Compose([
        # transforms.Resize((256,256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    # Create datasets with the appropriate transformations
    train_dataset = PARADataset(root_dir, transform=train_transform, train=True, use_attr=use_attr,
                                use_hist=use_hist, random_seed=random_seed)
    test_dataset = PARADataset(root_dir, transform=test_transform, train=False, use_attr=use_attr,
                               use_hist=use_hist, random_seed=random_seed)
    return train_dataset, test_dataset

num_bins = 9
num_attr = 8
use_attr = True
use_hist = True


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train and Evaluate the Combined Model')
    parser.add_argument('--fold_id', type=int, default=1)
    parser.add_argument('--n_fold', type=int, default=4)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--use_cv', action='store_true', help='Enable cross validation')
    parser.add_argument('--is_eval', action='store_true', help='Enable evaluation mode')
    parser.add_argument('--no_log', action='store_false', dest='is_log', help='Disable logging')

    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr_schedule_epochs', type=int, default=5, help='Epochs after which to apply learning rate decay')
    parser.add_argument('--lr_decay_factor', type=float, default=0.1, help='Factor by which to decay the learning rate')
    parser.add_argument('--max_patience_epochs', type=int, default=10, help='Max patience epochs for early stopping')
    args = parser.parse_args()

    is_eval = args.is_eval
    is_log = args.is_log
    resume = args.resume

    lr = args.lr
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    lr_schedule_epochs = args.lr_schedule_epochs
    lr_decay_factor = args.lr_decay_factor
    max_patience_epochs = args.max_patience_epochs
    random_seed = None

    if is_log:
        wandb.init(project="resnet_PARA_GIAA",
                   notes='NIMA for MIR',
                   tags=["lr_decay_factor%.2f"%lr_decay_factor])
        wandb.config = {
            "learning_rate": lr,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
        }
        experiment_name = wandb.run.name
    else:
        experiment_name = ''
    
    train_dataset, test_dataset = load_data(args)
    # Create dataloaders for training and test sets
    n_workers = 4
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers, timeout=300, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers, timeout=300, pin_memory=True)
    
    # Define the number of classes in your dataset
    num_classes = num_attr + num_bins
    hidden_unit = 512
    # Define the device for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CombinedModel(num_bins, num_attr, hidden_unit, resume = resume).to(device)

    # Define the loss functions
    criterion_mse = nn.MSELoss()
    criterion_emd = earth_mover_distance

    # Define the optimizer
    optimizer_resnet50 = optim.Adam(model.parameters(), lr=lr)

    # Initialize the best test loss and the best model
    best_model = None
    best_modelname = 'best_model_resnet50_giaa_hidden%d_lr%1.0e_decay_%depoch' % (hidden_unit, lr, num_epochs)
    if not use_attr:
        best_modelname += '_noattr'
    best_modelname += '_%s'%experiment_name
    best_modelname += '.pth'

    # Training loop
    best_test_loss = float('inf')
    num_patience_epochs = 0
    for epoch in range(num_epochs):
        # Learning rate schedule
        if (epoch + 1) % lr_schedule_epochs == 0:
            for param_group in optimizer_resnet50.param_groups:
                param_group['lr'] *= lr_decay_factor

        # Training
        train_mse_mean_loss, train_mse_std_loss, train_attr_mse_loss, train_emd_loss = train(model, train_dataloader, optimizer_resnet50, device)
        if is_log:
            wandb.log({"Train MSE Mean Loss": train_mse_mean_loss,
                       "Train MSE Std Loss": train_mse_std_loss,
                       "Train EMD Loss": train_emd_loss,
                       "Train Attr MSE Loss": train_attr_mse_loss}, commit=False)

        # Testing
        test_emd_loss, test_attr_mse_loss, test_mse_mean_loss, test_srocc = evaluate(model, test_dataloader, num_bins, num_attr, device, eval_srocc=True)
        if is_log:
            wandb.log({"Test MSE Mean Loss": test_mse_mean_loss,
                       "Test EMD Loss": test_emd_loss,
                       "Test MSE Mean Attr Loss": test_attr_mse_loss,
                       "Test SROCC": test_srocc})

        # Print the epoch loss
        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train EMD Loss: {train_emd_loss:.4f}, "
              f"Test EMD Loss: {test_emd_loss:.4f}, "
              f"Test MSE Mean Attr Loss: {test_attr_mse_loss:.4f}, "
              f"Test SROCC: {test_srocc:.4f}")
        if is_eval:
            raise Exception

        # Check if the current model has the best test loss so far
        test_loss = test_emd_loss
        # Early stopping check
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            num_patience_epochs = 0
            torch.save(model.state_dict(), best_modelname)
        else:
            num_patience_epochs += 1
            if num_patience_epochs >= max_patience_epochs:
                print("Validation loss has not decreased for {} epochs. Stopping training.".format(max_patience_epochs))
                break


    if not is_eval:
        model.load_state_dict(torch.load(best_modelname))
    
    # Testing
    test_emd_loss, test_attr_mse_loss, test_mse_mean_loss, test_srocc = evaluate(model, test_dataloader, num_bins, num_attr, device, eval_srocc=True)
    if is_log:
        wandb.log({"Test MSE Mean Loss": test_mse_mean_loss,
                    "Test EMD Loss": test_emd_loss,
                    "Test MSE Mean Attr Loss": test_attr_mse_loss,
                    "Test SROCC": test_srocc})

    # Print the epoch loss
    print(f"Epoch [{epoch + 1}/{num_epochs}], "
            f"Train EMD Loss: {train_emd_loss:.4f}, "
            f"Test EMD Loss: {test_emd_loss:.4f}, "
            f"Test MSE Mean Attr Loss: {test_attr_mse_loss:.4f}, "
            f"Test SROCC: {test_srocc:.4f}")