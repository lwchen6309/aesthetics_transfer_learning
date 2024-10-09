import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet50
import numpy as np
from tqdm import tqdm
import wandb
from itertools import chain
from scipy.stats import spearmanr
from PARA_histogram_dataloader import load_usersplit_data, PARA_sGIAA_HistogramDataset, PARA_GIAA_HistogramDataset, PARA_PIAA_HistogramDataset, PARA_GSP_HistogramDataset
from PARA_PIAA_dataloader import PARA_PIAADataset, split_dataset_by_user, split_dataset_by_images


def earth_mover_distance(x, y, dim=-1):
    """
    Compute Earth Mover's Distance (EMD) between two 1D tensors x and y using 2-norm.
    """
    
    cdf_x = torch.cumsum(x, dim=dim)
    cdf_y = torch.cumsum(y, dim=dim)
    emd = torch.norm(cdf_x - cdf_y, p=2, dim=dim)
    return emd


class CombinedModel(nn.Module):
    def __init__(self, num_bins_aesthetic, num_attr, num_bins_attr, num_pt):
        super(CombinedModel, self).__init__()
        self.resnet = resnet50(pretrained=True)
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 512),
            nn.ReLU(),
            # nn.Linear(512, num_bins_aesthetic),
        )
        self.num_bins_aesthetic = num_bins_aesthetic
        self.num_attr = num_attr
        self.num_bins_attr = num_bins_attr
        self.num_pt = num_pt
        # For predicting attribute histograms for each attribute
        self.pt_encoder = nn.Sequential(
            nn.Linear(num_bins_aesthetic, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512)
        )
        
        # For predicting aesthetic score histogram
        self.fc_aesthetic = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_pt)
        )

    def forward(self, images, score_histogram):
        x = self.resnet(images)
        pt_code = self.pt_encoder(score_histogram)
        xz = x + pt_code
        trait_logits = self.fc_aesthetic(xz)
        return trait_logits

trait_lengths = {
    'age': 5, 'gender': 2, 'EducationalLevel': 5, 'artExperience': 4,
    'photographyExperience': 4, 'personality-E': 10, 'personality-A': 10,
    'personality-N': 10, 'personality-O': 10, 'personality-C': 10
}

def split_trait_histogram(trait_histogram):
    split_histograms = {}
    start = 0
    for trait, length in trait_lengths.items():
        split_histograms[trait] = trait_histogram[:, start:start + length]
        start += length
    return split_histograms

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(dataloader, leave=False)
    for sample in progress_bar:
        if is_eval:
            break
        
        images = sample['image'].to(device)
        aesthetic_score_histogram = sample['aestheticScore'].to(device)
        target_traits = sample['traits'].to(device)
        target_onehot_big5 = sample['big5'].to(device)
        target_traits_combined = torch.cat([target_traits, target_onehot_big5], dim=1)

        optimizer.zero_grad()
        traits_logits = model(images, aesthetic_score_histogram)
        split_logits = split_trait_histogram(traits_logits)
        trait_target = split_trait_histogram(target_traits_combined)
        
        total_loss = 0
        for trait in trait_lengths.keys():
            softmaxed = F.softmax(split_logits[trait], dim=1)
            loss = torch.mean(criterion(softmaxed, trait_target[trait]))
            total_loss += loss
        total_loss /= len(trait_lengths)

        total_loss.backward()
        optimizer.step()
        running_loss += total_loss.item()

        progress_bar.set_postfix({
            'Train Loss': total_loss.item(),
        })

    epoch_loss = running_loss / len(dataloader)
    return epoch_loss

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    progress_bar = tqdm(dataloader, leave=False)
    with torch.no_grad():
        for sample in progress_bar:
            images = sample['image'].to(device)
            aesthetic_score_histogram = sample['aestheticScore'].to(device)
            target_traits = sample['traits'].to(device)
            target_onehot_big5 = sample['big5'].to(device)
            target_traits_combined = torch.cat([target_traits, target_onehot_big5], dim=1)

            traits_logits = model(images, aesthetic_score_histogram)
            split_logits = split_trait_histogram(traits_logits)
            trait_target = split_trait_histogram(target_traits_combined)

            total_loss = 0
            for trait in trait_lengths.keys():
                softmaxed = F.softmax(split_logits[trait], dim=1)
                loss = criterion(softmaxed, trait_target[trait]).mean()
                total_loss += loss
            total_loss /= len(trait_lengths)
            
            running_loss += total_loss.item()
            progress_bar.set_postfix({'Test Loss': total_loss.item()})


    emd_loss = running_loss / len(dataloader)
    return emd_loss


def load_data(root_dir = '/home/lwchen/datasets/PARA/'):
    # Dataset transformations
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    # Load datasets
    # Create datasets with the appropriate transformations
    train_piaa_dataset = PARA_PIAADataset(root_dir, transform=train_transform)
    test_piaa_dataset = PARA_PIAADataset(root_dir, transform=train_transform)
    train_dataset, test_dataset = split_dataset_by_images(train_piaa_dataset, test_piaa_dataset, root_dir)
    _, test_user_piaa_dataset = split_dataset_by_user(
        PARA_PIAADataset(root_dir, transform=train_transform),  
        test_count=40, max_annotations_per_user=[100,50], seed=random_seed)
    
    # Create datasets with the appropriate transformations
    # train_dataset = PARA_HistogramDataset(root_dir, transform=train_transform, data=train_dataset.data, map_file='trainset_image_dct.pkl')
    # test_dataset = PARA_HistogramDataset(root_dir, transform=test_transform, data=test_dataset.data, map_file='testset_image_dct.pkl')
    print(len(train_dataset), len(test_dataset))
    pkl_dir = './dataset_pkl'
    # train_dataset = PARA_sGIAA_HistogramDataset(root_dir, transform=train_transform, data=train_piaa_dataset.data, map_file=os.path.join(pkl_dir,'trainset_image_dct.pkl'), precompute_file=os.path.join(pkl_dir,'trainset_MIAA_dct.pkl'))
    # train_dataset = PARA_sGIAA_HistogramDataset(root_dir, transform=train_transform, data=train_piaa_dataset.data, map_file=os.path.join(pkl_dir,'trainset_image_dct.pkl'), precompute_file=os.path.join(pkl_dir,'trainset_MIAA_nopiaa_dct.pkl'))
    train_dataset = PARA_GIAA_HistogramDataset(root_dir, transform=train_transform, data=train_piaa_dataset.data, map_file=os.path.join(pkl_dir,'trainset_image_dct.pkl'), precompute_file=os.path.join(pkl_dir,'trainset_GIAA_dct.pkl'))
    # train_dataset = PARA_PIAA_HistogramDataset(root_dir, transform=train_transform, data=train_dataset.data)

    test_giaa_dataset = PARA_GIAA_HistogramDataset(root_dir, transform=test_transform, data=test_piaa_dataset.data, map_file=os.path.join(pkl_dir,'testset_image_dct.pkl'), precompute_file=os.path.join(pkl_dir,'testset_GIAA_dct.pkl'))
    test_piaa_dataset = PARA_PIAA_HistogramDataset(root_dir, transform=test_transform, data=test_dataset.data)
    # test_piaa_dataset = PARA_GSP_HistogramDataset(root_dir, transform=test_transform, piaa_data=test_piaa_dataset.data, 
    #                 giaa_data=test_piaa_dataset.data, map_file=os.path.join(pkl_dir,'testset_image_dct.pkl'), precompute_file=os.path.join(pkl_dir,'testset_GIAA_dct.pkl'))
    test_user_piaa_dataset = PARA_PIAA_HistogramDataset(root_dir, transform=test_transform, data=test_user_piaa_dataset.data)

    # train_dataset, test_giaa_dataset, _, test_piaa_dataset = load_usersplit_data(root_dir = '/home/lwchen/datasets/PARA/', miaa=False)

    return train_dataset, test_giaa_dataset, test_piaa_dataset, test_user_piaa_dataset


is_eval = False
is_log = True
num_bins = 9
num_attr = 8
num_bins_attr = 5
num_pt = 50 + 20
resume = None
# resume = "best_model_resnet50_histo_latefusion_lr5e-05_decay_20epoch_deep-paper-2.pth"
criterion_mse = nn.MSELoss()


if __name__ == '__main__':
    random_seed = 42
    lr = 5e-6
    batch_size = 100
    num_epochs = 20
    lr_schedule_epochs = 5
    lr_decay_factor = 0.5
    max_patience_epochs = 10
    n_workers = 8
    
    if is_log:
        wandb.init(project="resnet_PARA_PIAA", 
                   notes="inverse-latefusion",
                   tags = ["no_attr"])
        wandb.config = {
            "learning_rate": lr,
            "batch_size": batch_size,
            "num_epochs": num_epochs
        }
        experiment_name = wandb.run.name
    else:
        experiment_name = ''
    
    train_dataset, test_giaa_dataset, test_piaa_dataset, test_user_piaa_dataset = load_data()
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers, pin_memory=True, timeout=300)
    test_giaa_dataloader = DataLoader(test_giaa_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers, pin_memory=True, timeout=300)
    test_piaa_dataloader = DataLoader(test_piaa_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers, pin_memory=True, timeout=300)
    test_user_piaa_dataloader = DataLoader(test_user_piaa_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers, pin_memory=True, timeout=300)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the combined model
    model = CombinedModel(num_bins, num_attr, num_bins_attr, num_pt).to(device)
    
    if resume is not None:
        model.load_state_dict(torch.load(resume))
    # Loss and optimizer
    # criterion_mse = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Initialize the best test loss and the best model
    best_model = None
    best_modelname = 'best_model_resnet50_histo_latefusion_lr%1.0e_decay_%depoch' % (lr, num_epochs)
    best_modelname += '_%s'%experiment_name
    best_modelname += '.pth'
    
    # Training loop
    best_test_loss = float('inf')
    for epoch in range(num_epochs):
        if is_eval:
            break
        # Learning rate schedule
        if (epoch + 1) % lr_schedule_epochs == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_decay_factor

        # Training
        train_emd_loss = train(model, train_dataloader, earth_mover_distance, optimizer, device)
        if is_log:
            wandb.log({"Train Trait EMD Loss": train_emd_loss,
                       }, commit=False)
        
        # # Testing
        # test_piaa_emd_loss, test_piaa_attr_emd_loss, test_piaa_srocc, test_piaa_mse = evaluate(model, test_piaa_dataloader, earth_mover_distance, device)
        # if is_log:
        #     wandb.log({
        #         "Test PIAA EMD Loss": test_piaa_emd_loss,
        #         "Test PIAA SROCC": test_piaa_srocc,
        #                }, commit=True)
        test_giaa_emd_loss = evaluate(model, test_giaa_dataloader, earth_mover_distance, device)
        if is_log:
            wandb.log({
                "Test Trait EMD Loss": test_giaa_emd_loss,
                       }, commit=True)
        
        # Print the epoch loss
        print(  f"Epoch [{epoch + 1}/{num_epochs}], "
                f"Train EMD Loss: {train_emd_loss:.4f}, "
                f"Test GIAA EMD Loss: {test_giaa_emd_loss:.4f}, "
              )

        # Early stopping check
        if test_giaa_emd_loss < best_test_loss:
            best_test_loss = test_giaa_emd_loss
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
    test_piaa_emd_loss = evaluate(model, test_piaa_dataloader, earth_mover_distance, device)
    test_user_piaa_emd_loss = evaluate(model, test_user_piaa_dataloader, earth_mover_distance, device)
    test_giaa_emd_loss = evaluate(model, test_giaa_dataloader, earth_mover_distance, device)
    
    if is_log:
        wandb.log({
            "Test GIAA EMD Loss": test_giaa_emd_loss,
            "Test PIAA EMD Loss": test_piaa_emd_loss,
            "Test user PIAA EMD Loss": test_user_piaa_emd_loss,
        }, commit=True)

    # Print the epoch loss
    print(f"Epoch [{epoch + 1}/{num_epochs}], "
            f"Test GIAA EMD Loss: {test_giaa_emd_loss:.4f}, "
            f"Test PIAA EMD Loss: {test_piaa_emd_loss:.4f}, "
            f"Test user PIAA EMD Loss: {test_user_piaa_emd_loss:.4f}, "
            )
