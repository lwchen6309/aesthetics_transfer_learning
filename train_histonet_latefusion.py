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
from PARA_histogram_dataloader import load_usersplit_data, PARA_MIAA_HistogramDataset, PARA_GIAA_HistogramDataset, PARA_PIAA_HistogramDataset, PARA_GSP_HistogramDataset
from PARA_PIAA_dataloader import PARA_PIAADataset, split_dataset_by_user, split_dataset_by_images
import pandas as pd


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
            nn.Linear(num_pt, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512)
        )
        
        # For predicting aesthetic score histogram
        self.fc_aesthetic = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_bins_aesthetic)
        )

    def forward(self, images, traits_histogram):
        x = self.resnet(images)
        pt_code = self.pt_encoder(traits_histogram)
        xz = x + pt_code
        aesthetic_logits = self.fc_aesthetic(xz)
        return aesthetic_logits


# Training Function
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_aesthetic_emd_loss = 0.0
    running_total_emd_loss = 0.0
    progress_bar = tqdm(dataloader, leave=False)
    # scale_aesthetic = torch.arange(1, 5.5, 0.5).to(device)
    running_ce_improvement = 0
    running_ce_init = 0
    for sample in progress_bar:
        if is_eval:
            break
        
        images = sample['image'].to(device)
        aesthetic_score_histogram = sample['aestheticScore'].to(device)
        traits_histogram = sample['traits'].to(device)
        onehot_big5 = sample['onehot_traits'].to(device)
        attributes_target_histogram = sample['attributes'].to(device).view(-1, num_attr, num_bins_attr) # Reshape to match our logits shape
        total_traits_histogram = torch.cat([traits_histogram, onehot_big5], dim=1)
        # traits_histogram = traits_histogram[:,:5]

        # Optimize the traits_histogram for maximal entropy
        # coef, ce_improvement, ce_init = optimize_entropy_for_batch(traits_histogram)
        # running_ce_improvement += ce_improvement
        # running_ce_init += ce_init
        # coef = coef * len(images)

        optimizer.zero_grad()
        aesthetic_logits = model(images, total_traits_histogram)
        prob_aesthetic = F.softmax(aesthetic_logits, dim=1)
        # prob_attribute = F.softmax(attribute_logits, dim=-1) # Softmax along the bins dimension
        loss_aesthetic = torch.mean(criterion(prob_aesthetic, aesthetic_score_histogram))
        # loss_attribute = torch.mean(coef[:,None] * criterion(prob_attribute, attributes_target_histogram))
        # total_loss = loss_aesthetic + loss_attribute # Combining losses
        total_loss = loss_aesthetic # Combining losses
        
        total_loss.backward()
        optimizer.step()
        running_total_emd_loss += total_loss.item()
        running_aesthetic_emd_loss += loss_aesthetic.item()

        progress_bar.set_postfix({
            'Train EMD Loss': total_loss.item(),
        })
    epoch_ce_improvement = running_ce_improvement / len(dataloader)
    epoch_ce_init = running_ce_init / len(dataloader)
    
    epoch_emd_loss = running_aesthetic_emd_loss / len(dataloader)
    epoch_total_emd_loss = running_total_emd_loss / len(dataloader)
    # print(epoch_ce_init)
    # print(epoch_ce_improvement)
    # raise Exception
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
            traits_histogram = sample['traits'].to(device)
            onehot_big5 = sample['onehot_traits'].to(device)
            attributes_target_histogram = sample['attributes'].to(device).view(-1, num_attr, num_bins_attr) # Reshape to match our logits shape
            traits_histogram = torch.cat([traits_histogram, onehot_big5], dim=1)
            
            aesthetic_logits = model(images, traits_histogram)
            prob_aesthetic = F.softmax(aesthetic_logits, dim=1)
            # prob_attribute = F.softmax(attribute_logits, dim=-1) # Softmax along the bins dimension

            loss = criterion(prob_aesthetic, aesthetic_score_histogram).mean()
            # loss_attribute = criterion(prob_attribute, attributes_target_histogram).mean()
            
            if eval_srocc:
                outputs_mean = torch.sum(prob_aesthetic * scale, dim=1, keepdim=True)
                target_mean = torch.sum(aesthetic_score_histogram * scale, dim=1, keepdim=True)
                mean_pred.append(outputs_mean.view(-1).cpu().numpy())
                mean_target.append(target_mean.view(-1).cpu().numpy())
                # MSE
                mse = criterion_mse(outputs_mean, target_mean)
                running_mse_loss += mse.item()

            running_emd_loss += loss.item()
            # running_attr_emd_loss += loss_attribute.item()
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


# Evaluation Function
def evaluate_subPIAA(model, dataloader, criterion, device):
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
        for piaa_sample, gsp_sample, giaa_sample in progress_bar:
            images = piaa_sample['image'].to(device)
            aesthetic_score_histogram = piaa_sample['aestheticScore'].to(device)
            traits_histogram = piaa_sample['traits'].to(device)
            onehot_big5 = piaa_sample['onehot_traits'].to(device)
            attributes_target_histogram = piaa_sample['attributes'].to(device).view(-1, num_attr, num_bins_attr) # Reshape to match our logits shape
            traits_histogram = torch.cat([traits_histogram, onehot_big5], dim=1)
            
            aesthetic_logits = model(images, traits_histogram)
            prob_aesthetic = F.softmax(aesthetic_logits, dim=1)

            # GIAA
            traits_histogram = giaa_sample['traits'].to(device)
            onehot_big5 = giaa_sample['onehot_traits'].to(device)
            traits_histogram = torch.cat([traits_histogram, onehot_big5], dim=1)
            aesthetic_logits = model(images, traits_histogram)
            giaa_prob_aesthetic = F.softmax(aesthetic_logits, dim=1)

            # GSP
            traits_histogram = gsp_sample['traits'].to(device)
            onehot_big5 = gsp_sample['onehot_traits'].to(device)
            traits_histogram = torch.cat([traits_histogram, onehot_big5], dim=1)
            aesthetic_logits = model(images, traits_histogram)
            gsp_prob_aesthetic = F.softmax(aesthetic_logits, dim=1)

            n_giaa_samples = giaa_sample['n_samples'][...,None].to(device)
            n_gsp_samples = gsp_sample['n_samples'][...,None].to(device)
            spiaa_prob_aesthetic = (giaa_prob_aesthetic * n_giaa_samples - gsp_prob_aesthetic * n_gsp_samples).clamp(min=0, max=1)
            prob_aesthetic = (prob_aesthetic + spiaa_prob_aesthetic) / 2
            prob_aesthetic = spiaa_prob_aesthetic

            loss = criterion(prob_aesthetic, aesthetic_score_histogram).mean()
            # loss_attribute = criterion(prob_attribute, attributes_target_histogram).mean()
            
            if eval_srocc:
                outputs_mean = torch.sum(prob_aesthetic * scale, dim=1, keepdim=True)
                target_mean = torch.sum(aesthetic_score_histogram * scale, dim=1, keepdim=True)
                mean_pred.append(outputs_mean.view(-1).cpu().numpy())
                mean_target.append(target_mean.view(-1).cpu().numpy())
                # MSE
                mse = criterion_mse(outputs_mean, target_mean)
                running_mse_loss += mse.item()

            running_emd_loss += loss.item()
            # running_attr_emd_loss += loss_attribute.item()
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


def evaluate_trait(model, dataloader, device, num_iterations=1000, learning_rate=1e-4):
    model.eval()
    optimized_trait_histograms = []

    trait_lengths = {
        'age': 5, 'gender': 2, 'EducationalLevel': 5, 'artExperience': 4,
        'photographyExperience': 4, 'personality-E': 10, 'personality-A': 10,
        'personality-N': 10, 'personality-O': 10, 'personality-C': 10
    }
    # Initialize a dictionary to accumulate EMD results for each trait
    accumulated_emd_results = {trait: [] for trait in trait_lengths.keys()}    

    def split_trait_histogram(trait_histogram):
        split_histograms = {}
        start = 0
        for trait, length in trait_lengths.items():
            split_histograms[trait] = trait_histogram[:, start:start + length]
            start += length
        return split_histograms

    for sample in tqdm(dataloader, leave=False):
        images = sample['image'].to(device)
        target_aesthetic_histogram = sample['aestheticScore'].to(device)
        target_traits = sample['traits'].to(device)
        target_onehot_big5 = sample['onehot_traits'].to(device)
        target_traits_combined = torch.cat([target_traits, target_onehot_big5], dim=1)

        # Initialize trait histogram as a zero vector for each image in the batch
        trait_histogram = torch.rand_like(target_traits_combined, requires_grad=True, device=device)
        # trait_histogram = torch.tensor(target_traits_combined, requires_grad=True, device=device)
        optimizer = optim.Adam([trait_histogram], lr=learning_rate)

        # Split, normalize, and concatenate trait_histogram
        split_histograms = split_trait_histogram(trait_histogram)
        for trait in trait_lengths.keys():
            split_histograms[trait] = F.softmax(split_histograms[trait], dim=1)
        trait_histogram = torch.cat(list(split_histograms.values()), dim=1)
        
        for _ in range(num_iterations):
            optimizer.zero_grad()
            predicted_aesthetic_histogram = model(images, trait_histogram)
            predicted_aesthetic_histogram = F.softmax(predicted_aesthetic_histogram, dim=1)
            loss = earth_mover_distance(target_aesthetic_histogram, predicted_aesthetic_histogram).mean()
            loss.backward(retain_graph=True)
            optimizer.step()

        # Compute EMD for each trait subset
        optimized_histogram = trait_histogram.detach()
        split_optimized_histograms = split_trait_histogram(optimized_histogram)
        split_target_histograms = split_trait_histogram(target_traits_combined)

        for trait in trait_lengths.keys():
            emd_values = earth_mover_distance(split_optimized_histograms[trait], split_target_histograms[trait]).cpu().numpy()
            accumulated_emd_results[trait].extend(emd_values)
        optimized_trait_histograms.append(optimized_histogram.cpu().numpy())
    optimized_trait_histograms = np.concatenate(optimized_trait_histograms, axis=0)
    return optimized_trait_histograms, accumulated_emd_results


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
    # Assuming shell_users_df contains the shell user DataFrame
    shell_users_df = pd.read_csv('shell_user_ids.csv')
    filtered_data = train_dataset.data[train_dataset.data['userId'].isin(shell_users_df['userId'])]
    train_dataset.data = filtered_data

    _, test_user_piaa_dataset = split_dataset_by_user(
        PARA_PIAADataset(root_dir, transform=train_transform),  
        test_count=40, max_annotations_per_user=[100,50], seed=random_seed)
    
    # Create datasets with the appropriate transformations
    # train_dataset = PARA_HistogramDataset(root_dir, transform=train_transform, data=train_dataset.data, map_file='trainset_image_dct.pkl')
    # test_dataset = PARA_HistogramDataset(root_dir, transform=test_transform, data=test_dataset.data, map_file='testset_image_dct.pkl')
    print(len(train_dataset), len(test_dataset))
    pkl_dir = './dataset_pkl'
    # train_dataset = PARA_MIAA_HistogramDataset(root_dir, transform=train_transform, data=train_piaa_dataset.data, map_file=os.path.join(pkl_dir,'trainset_image_dct.pkl'), precompute_file=os.path.join(pkl_dir,'trainset_MIAA_dct.pkl'))
    # train_dataset = PARA_MIAA_HistogramDataset(root_dir, transform=train_transform, data=train_piaa_dataset.data, map_file=os.path.join(pkl_dir,'trainset_image_dct.pkl'), precompute_file=os.path.join(pkl_dir,'trainset_MIAA_nopiaa_dct.pkl'))
    # train_dataset = PARA_GIAA_HistogramDataset(root_dir, transform=train_transform, data=train_piaa_dataset.data, map_file=os.path.join(pkl_dir,'trainset_image_dct.pkl'), precompute_file=os.path.join(pkl_dir,'trainset_GIAA_dct.pkl'))
    train_dataset = PARA_PIAA_HistogramDataset(root_dir, transform=train_transform, data=train_dataset.data)

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
    lr = 5e-5
    batch_size = 100
    num_epochs = 20
    lr_schedule_epochs = 5
    lr_decay_factor = 0.5
    max_patience_epochs = 10
    n_workers = 8
    
    if is_log:
        wandb.init(project="resnet_PARA_PIAA", 
                   notes="latefusion",
                   tags = ["no_attr","PIAA","OuterShell"])
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
        train_emd_loss, train_total_emd_loss = train(model, train_dataloader, earth_mover_distance, optimizer, device)
        if is_log:
            wandb.log({"Train EMD Loss": train_emd_loss,
                       "Train Total EMD Loss": train_total_emd_loss,
                       }, commit=False)
        
        # # Testing
        test_piaa_emd_loss, test_piaa_attr_emd_loss, test_piaa_srocc, test_piaa_mse = evaluate(model, test_piaa_dataloader, earth_mover_distance, device)
        if is_log:
            wandb.log({
                "Test PIAA EMD Loss": test_piaa_emd_loss,
                "Test PIAA SROCC": test_piaa_srocc,
                       }, commit=True)
        # test_giaa_emd_loss, test_giaa_attr_emd_loss, test_giaa_srocc, test_giaa_mse = evaluate(model, test_giaa_dataloader, earth_mover_distance, device)
        # if is_log:
        #     wandb.log({
        #         "Test GIAA EMD Loss": test_giaa_emd_loss,
        #         "Test GIAA Attr EMD Loss": test_giaa_attr_emd_loss,
        #         "Test GIAA SROCC": test_giaa_srocc,
        #         "Test GIAA MSE": test_giaa_mse,
        #                }, commit=True)
        
        # # Print the epoch loss
        # print(  f"Epoch [{epoch + 1}/{num_epochs}], "
        #         f"Train EMD Loss: {train_emd_loss:.4f}, "
        #         f"Test GIAA EMD Loss: {test_giaa_emd_loss:.4f}, "
        #         f"Test GIAA Attr EMD Loss: {test_giaa_attr_emd_loss:.4f}, "
        #         f"Test GIAA SROCC Loss: {test_giaa_srocc:.4f}, "
        #         f"Test GIAA MSE Loss: {test_giaa_mse:.4f}, "
        #       )

        # Early stopping check
        if test_piaa_emd_loss < best_test_loss:
            best_test_loss = test_piaa_emd_loss
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
    # test_piaa_emd_loss, test_piaa_attr_emd_loss, test_piaa_srocc, test_piaa_mse = evaluate_subPIAA(model, test_piaa_dataloader, earth_mover_distance, device)
    test_piaa_emd_loss, test_piaa_attr_emd_loss, test_piaa_srocc, test_piaa_mse = evaluate(model, test_piaa_dataloader, earth_mover_distance, device)
    test_user_piaa_emd_loss, test_user_piaa_attr_emd_loss, test_user_piaa_srocc, test_user_piaa_mse = evaluate(model, test_user_piaa_dataloader, earth_mover_distance, device)
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
            "Test user PIAA EMD Loss": test_user_piaa_emd_loss,
            "Test user PIAA Attr EMD Loss": test_user_piaa_attr_emd_loss,
            "Test user PIAA SROCC": test_user_piaa_srocc,
            "Test user PIAA MSE": test_user_piaa_mse
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
            f"Test user PIAA EMD Loss: {test_user_piaa_emd_loss:.4f}, "
            f"Test user PIAA Attr EMD Loss: {test_user_piaa_attr_emd_loss:.4f}, "
            f"Test user PIAA SROCC Loss: {test_user_piaa_srocc:.4f}, "
            f"Test user PIAA MSE Loss: {test_user_piaa_mse:.4f}"
            )
