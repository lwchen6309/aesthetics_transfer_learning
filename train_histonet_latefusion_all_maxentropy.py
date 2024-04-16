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
from PARA_histogram_dataloader import PARA_sGIAA_HistogramDataset, PARA_GIAA_HistogramDataset, PARA_PIAA_HistogramDataset
from PARA_PIAA_dataloader import PARA_PIAADataset, split_dataset_by_user, split_dataset_by_images


class Extended_PARA_GIAA_HistogramDataset(PARA_GIAA_HistogramDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.coef = None
        self.compute_coef()

    def compute_coef(self):
        # Aggregating all traits_histogram tensors from the training dataset
        all_traits_histogram = torch.stack([self.process_traits(sample) for sample in self], dim=0)
        
        # Computing coef for all training data
        coef, _, _ = optimize_entropy_for_batch(all_traits_histogram)
        
        # Saving coef into the dataset
        self.set_coef(coef)

    def set_coef(self, coef):
        self.coef = coef  # New method to set coef values

    def process_traits(self, sample):
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        traits_histogram = sample['traits']
        onehot_traits_histogram = sample['big5']
        traits_histogram = torch.cat([traits_histogram, onehot_traits_histogram], dim=0)
        return traits_histogram  # Return the processed traits_histogram without modifying the sample

    def __getitem__(self, index):
        sample = super().__getitem__(index)  # Call parent class's __getitem__
        if self.coef is not None:
            sample['coef'] = self.coef[index]  # Add coef value to the sample
        return sample


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
        self.temperature = 0.01
        self.z_dims = 128
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

def earth_mover_distance(x, y, dim=-1):
    """
    Compute Earth Mover's Distance (EMD) between two 1D tensors x and y using 2-norm.
    """
    
    cdf_x = torch.cumsum(x, dim=dim)
    cdf_y = torch.cumsum(y, dim=dim)
    emd = torch.norm(cdf_x - cdf_y, p=2, dim=dim)
    return emd

def negative_entropy(c, distributions):
    P = torch.matmul(c, distributions)
    H = torch.sum(P * torch.log(P + 1e-10))
    return H

def project_simplex(x):
    sorted_x, _ = torch.sort(x, descending=True)
    cumulative_sum = torch.cumsum(sorted_x, dim=0)
    j = torch.arange(1, x.size()[0] + 1, dtype=torch.float32, device=x.device)
    check = 1 + j * sorted_x > cumulative_sum
    rho = j.masked_select(check)[-1]
    theta = (cumulative_sum[rho.long() - 1] - 1) / rho
    return torch.nn.functional.relu(x - theta)

def optimize_entropy_for_batch(distributions, lr=0.1, num_sample=100, num_epochs=1):
    distributions_torch = distributions
    num_dists = distributions_torch.shape[0]
    
    # Initial uniform guess
    c_uniform = torch.ones(num_dists, dtype=torch.float32, device=distributions_torch.device) / num_dists
    init_ce = -negative_entropy(c_uniform, distributions_torch)
    best_ce_improvement = 0.
    best_c = c_uniform
    
    for _ in range(num_sample):  # drawing 10 samples from Dirichlet
        c = torch.distributions.dirichlet.Dirichlet(torch.ones(num_dists, device=distributions_torch.device)).sample()
        c.requires_grad = True

        optimizer = optim.Adam([c], lr=lr)
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            loss = negative_entropy(c, distributions_torch)
            loss.backward()
            
            # Project c onto simplex to ensure it sums to 1 and remains non-negative
            with torch.no_grad():
                c.copy_(project_simplex(c))
        
        end_ce = -negative_entropy(c, distributions_torch)
        ce_improvement = end_ce - init_ce
        
        if ce_improvement > best_ce_improvement:
            best_ce_improvement = ce_improvement
            best_c = c.detach()
    print(init_ce)
    print(best_ce_improvement)
    return best_c, best_ce_improvement, init_ce


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
        onehot_traits_histogram = sample['big5'].to(device)
        attributes_target_histogram = sample['attributes'].to(device).view(-1, num_attr, num_bins_attr) # Reshape to match our logits shape
        traits_histogram = torch.cat([traits_histogram, onehot_traits_histogram], dim=1)
        coef = sample['coef'].to(device)

        # Optimize the traits_histogram for maximal entropy
        # coef = coef * len(images)

        optimizer.zero_grad()
        aesthetic_logits = model(images, traits_histogram)
        prob_aesthetic = F.softmax(aesthetic_logits, dim=1)
        # prob_attribute = F.softmax(attribute_logits, dim=-1) # Softmax along the bins dimension
        loss_aesthetic = torch.mean(coef * criterion(prob_aesthetic, aesthetic_score_histogram))
        # loss_attribute = torch.mean(coef[:,None] * criterion(prob_attribute, attributes_target_histogram))
        # total_loss = loss_aesthetic + loss_attribute # Combining losses
        total_loss = loss_aesthetic

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
            onehot_traits_histogram = sample['big5'].to(device)
            attributes_target_histogram = sample['attributes'].to(device).view(-1, num_attr, num_bins_attr) # Reshape to match our logits shape
            traits_histogram = torch.cat([traits_histogram, onehot_traits_histogram], dim=1)
            
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
        PARA_PIAADataset(root_dir, transform=train_transform), 
        test_count=40, max_annotations_per_user=50, seed=random_seed)
    
    # Create datasets with the appropriate transformations
    # train_dataset = PARA_HistogramDataset(root_dir, transform=train_transform, data=train_dataset.data, map_file='trainset_image_dct.pkl')
    # test_dataset = PARA_HistogramDataset(root_dir, transform=test_transform, data=test_dataset.data, map_file='testset_image_dct.pkl')
    print(len(train_dataset), len(test_dataset))
    pkl_dir = './dataset_pkl'
    # train_dataset = PARA_sGIAA_HistogramDataset(root_dir, transform=train_transform, data=train_piaa_dataset.data, map_file=os.path.join(pkl_dir,'trainset_image_dct.pkl'), precompute_file=os.path.join(pkl_dir,'trainset_MIAA_dct.pkl'))
    train_dataset = Extended_PARA_GIAA_HistogramDataset(root_dir, transform=train_transform, data=train_piaa_dataset.data, map_file=os.path.join(pkl_dir,'trainset_image_dct.pkl'), precompute_file=os.path.join(pkl_dir,'trainset_GIAA_dct.pkl'))
    test_giaa_dataset = PARA_GIAA_HistogramDataset(root_dir, transform=test_transform, data=test_piaa_dataset.data, map_file=os.path.join(pkl_dir,'testset_image_dct.pkl'), precompute_file=os.path.join(pkl_dir,'testset_GIAA_dct.pkl'))
    test_piaa_dataset = PARA_PIAA_HistogramDataset(root_dir, transform=test_transform, data=test_dataset.data)
    test_user_piaa_dataset = PARA_PIAA_HistogramDataset(root_dir, transform=test_transform, data=test_user_piaa_dataset.data)
    return train_dataset, test_giaa_dataset, test_piaa_dataset, test_user_piaa_dataset


is_eval = False
is_log = True
num_bins = 9
num_attr = 8
num_bins_attr = 5
num_pt = 50 + 20
resume = None
# resume = 'best_model_resnet50_histo_attr_latefusion_lr5e-05_decay_20epoch_tough-bush-60.pth'
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
        wandb.init(project="resnet_PARA_PIAA")
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
    best_modelname = 'best_model_resnet50_histo_latefusion_maxentropy_lr%1.0e_decay_%depoch' % (lr, num_epochs)
    best_modelname += '_%s'%experiment_name
    best_modelname += '.pth'

    # Training loop
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
        train_emd_loss, train_total_emd_loss = train(model, train_dataloader, earth_mover_distance, optimizer, device)  # Updated this line

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

        # Print the epoch loss
        print(f"Epoch [{epoch + 1}/{num_epochs}], "
                f"Test GIAA EMD Loss: {test_giaa_emd_loss:.4f}, "
                f"Test GIAA Attr EMD Loss: {test_giaa_attr_emd_loss:.4f}, "
                f"Test GIAA SROCC Loss: {test_giaa_srocc:.4f}, "
                f"Test GIAA MSE Loss: {test_giaa_mse:.4f}, "
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
