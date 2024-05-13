import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
# from torchvision import transforms
from torchvision.models import resnet50
import numpy as np
# import pandas as pd
from tqdm import tqdm
from scipy.stats import spearmanr
from PARA_histogram_dataloader import load_data, collate_fn_imgsort
from train_nima import earth_mover_distance
import wandb


class NIMA_attr(nn.Module):
    def __init__(self, num_bins_aesthetic, num_attr):
        super(NIMA_attr, self).__init__()
        self.resnet = resnet50(pretrained=True)
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 512),
            nn.ReLU(),
        )
        self.num_bins_aesthetic = num_bins_aesthetic

        self.fc_aesthetic = nn.Sequential(
            nn.Linear(512, num_bins_aesthetic)
        )
        self.fc_attributes = nn.Sequential(
            nn.Linear(512, num_attr)
        )

    def forward(self, images):
        x = self.resnet(images)
        aesthetic_logits = self.fc_aesthetic(x)
        attribute_logits = self.fc_attributes(x)
        return aesthetic_logits, attribute_logits


# Training Function
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_aesthetic_emd_loss = 0.0
    running_total_emd_loss = 0.0
    progress_bar = tqdm(dataloader, leave=False)
    scale = torch.arange(1, 5.5, 0.5).to(device)
    attr_scale = torch.arange(1, 5.5, 1).to(device)
    
    for sample in progress_bar:
        if is_eval:
            break
        
        images = sample['image'].to(device)
        aesthetic_score_histogram = sample['aestheticScore'].to(device)
        # traits_histogram = sample['traits'].to(device)
        # onehot_big5 = sample['big5'].to(device)
        # total_traits_histogram = torch.cat([traits_histogram, onehot_big5], dim=1)
        attributes_target_histogram = sample['attributes'].to(device).view(-1, num_attr, num_bins_attr) # Reshape to match our logits shape
        sum_prob = torch.sum(attributes_target_histogram, dim=-1)
        target_attr_mean = torch.sum(attributes_target_histogram * attr_scale, dim=-1, keepdim=False)

        optimizer.zero_grad()
        aesthetic_logits, attr_mean = model(images)
        prob_aesthetic = F.softmax(aesthetic_logits, dim=1)
        # prob_attribute = F.softmax(attribute_logits, dim=-1) # Softmax along the bins dimension
        loss_aesthetic = torch.mean(criterion(prob_aesthetic, aesthetic_score_histogram))
        
        loss_attribute = criterion_mse(attr_mean, target_attr_mean)
        total_loss = loss_aesthetic + loss_attribute # Combining losses
        
        total_loss.backward()
        optimizer.step()
        running_total_emd_loss += total_loss.item()
        running_aesthetic_emd_loss += loss_aesthetic.item()

        progress_bar.set_postfix({
            'Train EMD Loss': total_loss.item(),
        })
    
    epoch_emd_loss = running_aesthetic_emd_loss / len(dataloader)
    epoch_total_emd_loss = running_total_emd_loss / len(dataloader)
    return epoch_emd_loss, epoch_total_emd_loss


# Evaluation Function
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_emd_loss = 0.0
    running_attr_emd_loss = 0.0
    running_mse_loss = 0.0
    scale = torch.arange(1, 5.5, 0.5).to(device)
    eval_srocc = True

    progress_bar = tqdm(dataloader, leave=False)
    mean_pred = []
    mean_target = []
    with torch.no_grad():
        for sample in progress_bar:
            images = sample['image'].to(device)
            aesthetic_score_histogram = sample['aestheticScore'].to(device)

            aesthetic_logits, attr_mean = model(images)
            prob_aesthetic = F.softmax(aesthetic_logits, dim=1)
            loss = criterion(prob_aesthetic, aesthetic_score_histogram).mean()
            
            if eval_srocc:
                outputs_mean = torch.sum(prob_aesthetic * scale, dim=1, keepdim=True)
                target_mean = torch.sum(aesthetic_score_histogram * scale, dim=1, keepdim=True)
                mean_pred.append(outputs_mean.view(-1).cpu().numpy())
                mean_target.append(target_mean.view(-1).cpu().numpy())
                # MSE
                mse = criterion_mse(outputs_mean, target_mean)
                running_mse_loss += mse.item()
            
            running_emd_loss += loss.item()
            progress_bar.set_postfix({
                'Test EMD Loss': loss.item(),
            })
    
    # Calculate SROCC
    predicted_scores = np.concatenate(mean_pred, axis=0)
    true_scores = np.concatenate(mean_target, axis=0)
    srocc, _ = spearmanr(predicted_scores, true_scores)
    
    emd_loss = running_emd_loss / len(dataloader)
    emd_attr_loss = running_attr_emd_loss / len(dataloader)
    mse_loss = running_mse_loss / len(dataloader)
    return emd_loss, emd_attr_loss, srocc, mse_loss


num_bins = 9
num_attr = 8
num_bins_attr = 5
num_pt = 50 + 20
criterion_mse = nn.MSELoss()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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
    n_workers = 8
    n_workers = 8
    eval_on_giaa = True
    
    if is_log:
        tags = ["no_attr","GIAA"]
        if args.use_cv:
            tags += ["CV%d/%d"%(args.fold_id, args.n_fold)]
        wandb.init(project="resnet_PARA_PIAA", 
                   notes="NIMA_attr",
                   tags = tags)
        wandb.config = {
            "learning_rate": lr,
            "batch_size": batch_size,
            "num_epochs": num_epochs
        }
        experiment_name = wandb.run.name
    else:
        experiment_name = ''
    
    train_dataset, val_giaa_dataset, val_piaa_imgsort_dataset, test_giaa_dataset, test_piaa_imgsort_dataset = load_data(args)

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers, timeout=300)
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers, timeout=300, collate_fn=collate_fn_imgsort)
    val_giaa_dataloader = DataLoader(val_giaa_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers, timeout=300)
    val_piaa_imgsort_dataloader = DataLoader(val_piaa_imgsort_dataset, batch_size=5, shuffle=False, num_workers=n_workers, timeout=300, collate_fn=collate_fn_imgsort)
    test_giaa_dataloader = DataLoader(test_giaa_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers, timeout=300)
    test_piaa_imgsort_dataloader = DataLoader(test_piaa_imgsort_dataset, batch_size=5, shuffle=False, num_workers=n_workers, timeout=300, collate_fn=collate_fn_imgsort)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the combined model
    model = NIMA_attr(num_bins, num_attr).to(device)

    if resume is not None:
        model.load_state_dict(torch.load(resume))
    # Loss and optimizer
    # criterion_mse = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Initialize the best test loss and the best model
    best_model = None
    best_modelname = 'best_model_resnet50_nima_attr_lr%1.0e_decay_%depoch' % (lr, num_epochs)
    best_modelname += '_%s'%experiment_name
    best_modelname += '.pth'
    dirname = 'models_pth'
    if args.use_cv:
        dirname = os.path.join(dirname, 'random_cvs')
    best_modelname = os.path.join(dirname, best_modelname)
    
    # Training loop
    best_test_srocc = 0
    num_patience_epochs = 0
    for epoch in range(num_epochs):
        if is_eval:
            break
        # Learning rate schedule
        if (epoch + 1) % lr_schedule_epochs == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_decay_factor

        # Training
        train_emd_loss, train_total_emd_loss = train(model, train_dataloader, earth_mover_distance, optimizer, device)
        if is_log:
            wandb.log({"Train EMD Loss": train_emd_loss,
                       "Train Total EMD Loss": train_total_emd_loss,
                       }, commit=False)
        
        # Testing
        test_giaa_emd_loss, test_giaa_attr_emd_loss, test_giaa_srocc, test_giaa_mse = evaluate(model, test_giaa_dataloader, earth_mover_distance, device)
        if is_log:
            wandb.log({
                "Test GIAA EMD Loss": test_giaa_emd_loss,
                "Test GIAA Attr EMD Loss": test_giaa_attr_emd_loss,
                "Test GIAA SROCC": test_giaa_srocc,
                "Test GIAA MSE": test_giaa_mse,
                       }, commit=True)        
        test_piaa_emd_loss, test_piaa_attr_emd_loss, test_piaa_srocc, test_piaa_mse = evaluate(model, test_piaa_imgsort_dataloader, earth_mover_distance, device)
        if is_log:
            wandb.log({
                "Test PIAA EMD Loss": test_piaa_emd_loss,
                "Test PIAA SROCC": test_piaa_srocc,
                       }, commit=False)
        
        eval_srocc = test_giaa_srocc if eval_on_giaa else test_piaa_srocc
        
        # Early stopping check
        if eval_srocc > best_test_srocc:
            best_test_srocc = eval_srocc
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
    test_piaa_emd_loss, test_piaa_attr_emd_loss, test_piaa_srocc, test_piaa_mse = evaluate(model, test_piaa_imgsort_dataloader, earth_mover_distance, device)
    test_giaa_emd_loss, test_giaa_attr_emd_loss, test_giaa_srocc, test_giaa_mse = evaluate(model, test_giaa_dataloader, earth_mover_distance, device)
    
    if is_log:
        wandb.log({
            "Test GIAA EMD Loss": test_giaa_emd_loss,
            "Test GIAA Attr EMD Loss": test_giaa_attr_emd_loss,
            "Test GIAA SROCC": test_giaa_srocc,
            "Test GIAA MSE": test_giaa_mse,
            "Test PIAA EMD Loss": test_piaa_emd_loss,
            "Test PIAA Attr EMD Loss": test_piaa_attr_emd_loss,
            "Test PIAA SROCC": test_piaa_srocc,
            "Test PIAA MSE": test_piaa_mse,
        }, commit=True)

    # Print the epoch loss
    print(f"Epoch [{epoch + 1}/{num_epochs}], "
            f"Test GIAA EMD Loss: {test_giaa_emd_loss:.4f}, "
            f"Test GIAA Attr EMD Loss: {test_giaa_attr_emd_loss:.4f}, "
            f"Test GIAA SROCC Loss: {test_giaa_srocc:.4f}, "
            f"Test GIAA MSE Loss: {test_giaa_mse:.4f}, "
            f"Test PIAA EMD Loss: {test_piaa_emd_loss:.4f}, "
            f"Test PIAA EMD Loss: {test_piaa_attr_emd_loss:.4f}, "
            f"Test PIAA SROCC Loss: {test_piaa_srocc:.4f}, "
            f"Test PIAA MSE Loss: {test_piaa_mse:.4f}, "
            )

