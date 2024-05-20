import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet50
from torchvision.models.feature_extraction import create_feature_extractor
import numpy as np
from tqdm import tqdm
import wandb
from itertools import chain
from scipy.stats import spearmanr
from PARA_histogram_dataloader import PARA_HistogramDataset, PARA_GIAA_HistogramDataset, PARA_PIAA_HistogramDataset
from PARA_PIAA_dataloader import PARA_PIAADataset, split_dataset_by_user, split_dataset_by_images
from train_resnet_cls import earth_mover_distance


class Attention(nn.Module):
    def __init__(self, image_feat_dim, trait_dim, attention_dim):
        super(Attention, self).__init__()
        self.key_layer = nn.Linear(image_feat_dim, attention_dim)
        self.value_layer = nn.Linear(image_feat_dim, attention_dim)
        # self.query_layer = nn.Linear(trait_dim, attention_dim)
        self.query_layer = nn.Sequential(
            nn.Linear(trait_dim, 512),
            nn.ReLU(),
            nn.Linear(512, attention_dim),
        )
        # For predicting attribute histograms for each attribute
        self.pt_encoder = nn.Sequential(
            nn.Linear(trait_dim, 512),
            nn.ReLU(),
            nn.Linear(512, attention_dim),
            nn.BatchNorm1d(attention_dim)
        )
    
    def forward(self, images_feat, traits_histogram):
        keys = self.key_layer(images_feat)
        values = self.value_layer(images_feat)
        queries = self.query_layer(traits_histogram)
        attn_weights = F.softmax(torch.bmm(keys, queries.unsqueeze(2)), dim=1)
        attn_output = torch.mean(attn_weights * values, dim=1)
        attn_output += self.pt_encoder(traits_histogram)
        return attn_output


class CombinedModel(nn.Module):
    def __init__(self, num_bins_aesthetic, num_attr, num_bins_attr, num_pt, attention_dim=512):
        super(CombinedModel, self).__init__()
        
        # Use create_feature_extractor to extract features before pooling
        self.resnet = create_feature_extractor(resnet50(pretrained=True),
                                               {'layer4':'layer4'})
        self.attention = Attention(2048, num_pt, attention_dim)
        
        # For predicting attribute histograms for each attribute
        # self.fc_attribute = nn.Sequential(
        #     nn.Linear(attention_dim, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, num_attr * num_bins_attr)
        # )

        # For predicting aesthetic score histogram
        self.fc_aesthetic = nn.Sequential(
            nn.Linear(attention_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_bins_aesthetic)
        )
    
    def forward(self, images, traits_histogram):
        # Extract feature map
        x = self.resnet(images)['layer4']
        b, c, h, w = x.shape
        # Flatten spatial dimensions
        x = x.view(b, c, h*w).permute(0, 2, 1)
        
        attention_output = self.attention(x, traits_histogram)
        # attribute_logits = self.fc_attribute(attention_output)
        # aesthetic_logits = self.fc_aesthetic(attribute_logits)
        aesthetic_logits = self.fc_aesthetic(attention_output)
        return aesthetic_logits


# Training Function
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_aesthetic_emd_loss = 0.0
    running_total_emd_loss = 0.0
    progress_bar = tqdm(dataloader, leave=False)
    # scale_aesthetic = torch.arange(1, 5.5, 0.5).to(device)

    for sample in progress_bar:
        if is_eval:
            break
        
        images = sample['image'].to(device)
        aesthetic_score_histogram = sample['aestheticScore'].to(device)
        traits_histogram = sample['traits'].to(device)
        onehot_traits_histogram = sample['big5'].to(device)
        attributes_target_histogram = sample['attributes'].to(device).view(-1, num_attr, num_bins_attr) # Reshape to match our logits shape
        traits_histogram = torch.cat([traits_histogram, onehot_traits_histogram], dim=1)
        
        optimizer.zero_grad()
        aesthetic_logits = model(images, traits_histogram)
        prob_aesthetic = F.softmax(aesthetic_logits, dim=1)
        # prob_attribute = F.softmax(attribute_logits, dim=-1) # Softmax along the bins dimension
        
        loss_aesthetic = criterion(prob_aesthetic, aesthetic_score_histogram)
        # loss_attribute = criterion(prob_attribute, attributes_target_histogram) # This will compute the loss for each attribute's histogram
        total_loss = loss_aesthetic #+ loss_attribute.sum() # Combining losses

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
            traits_histogram = sample['traits'].to(device)
            onehot_traits_histogram = sample['big5'].to(device)
            attributes_target_histogram = sample['attributes'].to(device).view(-1, num_attr, num_bins_attr) # Reshape to match our logits shape
            traits_histogram = torch.cat([traits_histogram, onehot_traits_histogram], dim=1)
            
            aesthetic_logits = model(images, traits_histogram)
            prob_aesthetic = F.softmax(aesthetic_logits, dim=1)
            # prob_attribute = F.softmax(attribute_logits, dim=-1) # Softmax along the bins dimension

            loss = criterion(prob_aesthetic, aesthetic_score_histogram)
            # loss_attribute = criterion(prob_attribute, attributes_target_histogram) # This will compute the loss for each attribute's histogram
            
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


is_eval = False
is_log = True
num_bins = 9
num_attr = 8
num_bins_attr = 5
num_pt = 50 + 20
resume = None
criterion_mse = nn.MSELoss()


def load_data():
    root_dir = '/home/lwchen/datasets/PARA/'
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
    train_dataset = PARA_GIAA_HistogramDataset(root_dir, transform=train_transform, data=train_piaa_dataset.data, map_file=os.path.join(pkl_dir,'trainset_image_dct.pkl'), precompute_file=os.path.join(pkl_dir,'trainset_GIAA_dct.pkl'))
    test_giaa_dataset = PARA_GIAA_HistogramDataset(root_dir, transform=test_transform, data=test_piaa_dataset.data, map_file=os.path.join(pkl_dir,'testset_image_dct.pkl'), precompute_file=os.path.join(pkl_dir,'testset_GIAA_dct.pkl'))
    test_piaa_dataset = PARA_PIAA_HistogramDataset(root_dir, transform=test_transform, data=test_dataset.data)
    test_user_piaa_dataset = PARA_PIAA_HistogramDataset(root_dir, transform=test_transform, data=test_user_piaa_dataset.data)
    return train_dataset, test_giaa_dataset, test_piaa_dataset, test_user_piaa_dataset


if __name__ == '__main__':
    random_seed = 42
    lr = 5e-5
    batch_size = 100
    num_epochs = 20
    lr_schedule_epochs = 5
    lr_decay_factor = 0.5
    max_patience_epochs = 10
    exp_tag = "crossattn_residual"
    if is_log:
        hyperparam_tags = [
            f"LR: {lr}",
            f"LR Decay Factor: {lr_decay_factor}",
            f"LR Decay Step: {lr_schedule_epochs}",
            "Trait BN"
        ]
        wandb.init(project="resnet_PARA_PIAA", tags=hyperparam_tags, notes=exp_tag)
        experiment_name = wandb.run.name
    else:
        experiment_name = ''
    
    train_dataset, test_giaa_dataset, test_piaa_dataset, test_user_piaa_dataset = load_data()

    # Create dataloaders
    n_workers = 8
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
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Initialize the best test loss and the best model
    best_model = None
    best_modelname = 'best_model_resnet50_histo_%s_lr%1.0e_decay_%depoch' % (exp_tag, lr, num_epochs)
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
