import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet50
import numpy as np
import pandas as pd
from tqdm import tqdm
import wandb
from scipy.stats import spearmanr
from LAPIS_histogram_dataloader import LAPIS_GIAA_HistogramDataset, LAPIS_PIAA_HistogramDataset, LAPIS_MIAA_HistogramDataset, LAPIS_PIAA_HistogramDataset_imgsort, collate_fn_imgsort, collate_fn
from LAPIS_PIAA_dataloader import LAPIS_PIAADataset, create_image_split_dataset, create_user_split_dataset_kfold
# from time import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random


def earth_mover_distance(x, y, dim=-1):
    """
    Compute Earth Mover's Distance (EMD) between two 1D tensors x and y using 2-norm.
    """
    
    cdf_x = torch.cumsum(x, dim=dim)
    cdf_y = torch.cumsum(y, dim=dim)
    emd = torch.norm(cdf_x - cdf_y, p=2, dim=dim)
    return emd


class CombinedModel_earlyfusion(nn.Module):
    def __init__(self, num_bins_aesthetic, num_attr, num_bins_attr, num_pt):
        super(CombinedModel_earlyfusion, self).__init__()
        self.resnet = resnet50(pretrained=True)
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 512),
            nn.ReLU(),
            # nn.Linear(512, num_bins_aesthetic),
        )
        self.num_bins_aesthetic = num_bins_aesthetic
        self.num_attr = num_attr
        # self.num_bins_attr = num_bins_attr
        self.num_pt = num_pt
        # For predicting attribute histograms for each attribute
        # self.inv_coef_decoder = nn.Sequential(
        #     nn.Linear(num_pt + num_bins_aesthetic, 512),
        #     nn.ReLU(),   
        #     nn.Linear(512, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 20)
        # )
        
        # For predicting aesthetic score histogram
        self.fc_aesthetic = nn.Sequential(
            nn.Linear(512 + num_pt, 512),
            nn.ReLU(),
            nn.Linear(512, num_bins_aesthetic)
        )
    
    def forward(self, images, traits_histogram):
        x = self.resnet(images)
        aesthetic_logits = self.fc_aesthetic(torch.cat([x, traits_histogram], dim=1))
        # coef = self.inv_coef_decoder(torch.cat([aesthetic_logits, traits_histogram], dim=1))
        return aesthetic_logits #, coef


class CombinedModel_bak(nn.Module):
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
            nn.BatchNorm1d(512),
        )
        
        # For predicting aesthetic score histogram
        self.fc_aesthetic = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_bins_aesthetic)
        )

        # For predicting attribute histograms for each attribute
        # self.inv_coef_decoder = nn.Sequential(
        #     nn.Linear(num_pt + num_bins_aesthetic, 512),
        #     nn.ReLU(),   
        #     nn.Linear(512, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 20)
        # )

    
    def forward(self, images, traits_histogram):
        x = self.resnet(images)
        pt_code = self.pt_encoder(traits_histogram)
        xz = x + pt_code
        aesthetic_logits = self.fc_aesthetic(xz)
        # coef = self.inv_coef_decoder(torch.cat([aesthetic_logits, traits_histogram], dim=1))
        return aesthetic_logits
        # return aesthetic_logits, coef


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
    

class NIMA(nn.Module):
    def __init__(self, num_bins_aesthetic):
        super(NIMA, self).__init__()
        self.resnet = resnet50(pretrained=True)
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 512),
            nn.ReLU(),
        )
        self.num_bins_aesthetic = num_bins_aesthetic
        
        # For predicting aesthetic score histogram
        self.fc_aesthetic = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_bins_aesthetic)
        )

    def forward(self, images, traits_histogram):
        # traits_histogram is dummy variable
        x = self.resnet(images)
        aesthetic_logits = self.fc_aesthetic(x)
        return aesthetic_logits


# Training Function
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_aesthetic_emd_loss = 0.0
    running_total_emd_loss = 0.0
    progress_bar = tqdm(dataloader, leave=False)
    units_len = dataloader.dataset.traits_len()
    proj_simplex = False
    for sample in progress_bar:
        if is_eval:
            break
        
        images = sample['image'].to(device)
        aesthetic_score_histogram = sample['aestheticScore'].to(device)
        traits_histogram = sample['traits'].to(device)
        art_type = sample['art_type'].to(device)
        # traits_histogram = torch.cat([traits_histogram, art_type], dim=1)
        
        optimizer.zero_grad()
        if proj_simplex:
            aesthetic_logits, inv_coef = model(images, traits_histogram)
        else:
            aesthetic_logits = model(images, traits_histogram)

        prob_aesthetic = F.softmax(aesthetic_logits, dim=1)
        loss_aesthetic = torch.mean(criterion(prob_aesthetic, aesthetic_score_histogram))

        if proj_simplex:
            inv_coef_transposed = inv_coef.transpose(0, 1)  # Transpose inv_coef to shape [20, B]
            piaa_score = torch.matmul(inv_coef_transposed, aesthetic_logits).clamp(min=0, max=1)  # Resulting shape [20, S]
            piaa_traits = torch.matmul(inv_coef_transposed, traits_histogram).clamp(min=0, max=1)  # Resulting shape [20, T]
            
            # Compute self-entropy
            piaa_traits = torch.split(piaa_traits, units_len, dim=1)
            mean_entropy_piaa = 0
            for traits in piaa_traits:
                entropy_piaa_traits = -torch.sum(traits * torch.log(traits + 1e-2), dim=-1)
                mean_entropy_piaa += torch.mean(entropy_piaa_traits)
            entropy_piaa_logits = -torch.sum(piaa_score * torch.log(piaa_score + 1e-2), dim=-1)
            mean_entropy_piaa += torch.mean(entropy_piaa_logits)
            total_loss = loss_aesthetic + 0.1 * mean_entropy_piaa
        else:
            total_loss = loss_aesthetic

        total_loss.backward()
        optimizer.step()
        running_total_emd_loss += total_loss.item()
        running_aesthetic_emd_loss += loss_aesthetic.item()

        progress_bar.set_postfix({
            'Train EMD Loss': loss_aesthetic.item(),
            # 'Train sCE Loss': mean_entropy_piaa.item(),
        })
    
    epoch_emd_loss = running_aesthetic_emd_loss / len(dataloader)
    epoch_total_emd_loss = running_total_emd_loss / len(dataloader)
    return epoch_emd_loss, epoch_total_emd_loss

def save_results(userIds, traits_histograms, emd_loss_data):
    
    # Convert traits_histograms into a DataFrame
    traits_df = pd.DataFrame(traits_histograms, columns=[f'Trait_{i+1}' for i in range(traits_histograms.shape[-1])])

    # Add userIds and emd_loss_data to the DataFrame
    traits_df['UserId'] = userIds
    traits_df['EMD_Loss_Data'] = emd_loss_data

    # Reorder DataFrame if you want 'UserId' and 'EMD_Loss_Data' at the beginning
    cols = traits_df.columns.tolist()
    cols = cols[-2:] + cols[:-2]
    traits_df = traits_df[cols]

    # Save to CSV
    traits_df.to_csv('LAPIS_evaluation_results.csv', index=False)


# Evaluation Function
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_emd_loss = 0.0
    running_attr_emd_loss = 0.0
    running_mse_loss = 0.0
    # scale = torch.arange(1, 5.5, 0.5).to(device)
    scale = torch.arange(0, 10).to(device)
    eval_srocc = True
    progress_bar = tqdm(dataloader, leave=False)
    mean_pred = []
    mean_target = []
    with torch.no_grad():
        for sample in progress_bar:
            images = sample['image'].to(device)
            aesthetic_score_histogram = sample['aestheticScore'].to(device)
            traits_histogram = sample['traits'].to(device)
            art_type = sample['art_type'].to(device)
            # traits_histogram = torch.cat([traits_histogram, art_type], dim=1)

            # aesthetic_logits, _ = model(images, traits_histogram)
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


def evaluate_each_datum(model, dataloader, criterion, device):
    model.eval()
    running_emd_loss = 0.0
    running_attr_emd_loss = 0.0
    running_mse_loss = 0.0
    # scale = torch.arange(1, 5.5, 0.5).to(device)
    scale = torch.arange(0, 10).to(device)
    eval_srocc = True
    progress_bar = tqdm(dataloader, leave=False)
    mean_pred = []
    mean_target = []
    emd_loss_data = []
    with torch.no_grad():
        for sample in progress_bar:
            images = sample['image'].to(device)
            aesthetic_score_histogram = sample['aestheticScore'].to(device)
            traits_histogram = sample['traits'].to(device)
            art_type = sample['art_type'].to(device)
            # traits_histogram = torch.cat([traits_histogram, art_type], dim=1)

            # aesthetic_logits, _ = model(images, traits_histogram)
            aesthetic_logits = model(images, traits_histogram)
            prob_aesthetic = F.softmax(aesthetic_logits, dim=1)
            # prob_attribute = F.softmax(attribute_logits, dim=-1) # Softmax along the bins dimension
            loss = criterion(prob_aesthetic, aesthetic_score_histogram).mean()
            # loss_attribute = criterion(prob_attribute, attributes_target_histogram).mean()
            emd_loss_datum = criterion(prob_aesthetic, aesthetic_score_histogram)
            emd_loss_data.append(emd_loss_datum.view(-1).cpu().numpy())

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
    emd_loss_data = np.concatenate(emd_loss_data)

    # Create a jointplot
    # g = sns.JointGrid(x=true_scores, y=emd_loss_data)
    # g.plot_joint(sns.kdeplot, bw_adjust=0.5)
    # g.plot_marginals(sns.histplot, kde=False)
    # g.set_axis_labels('PIAA Scores', 'EMD Loss Data', fontsize=12)
    # plt.subplots_adjust(top=0.9)
    # g.fig.suptitle('PIAA Scores vs EMD Loss Data', fontsize=16)
    # plt.tight_layout()
    # random_int = random.randint(1, 100000)
    # filename = 'LAPIS_KDE_plot_%d.jpg'%random_int
    # print('Save figure to %s'%filename)
    # plt.savefig(filename)

    # Assuming true_scores and emd_loss_data are numpy arrays
    unique_scores = np.unique(true_scores)
    mean_emd_loss = []
    std_emd_loss = []
    
    # Group data and compute mean and std for each group
    for score in unique_scores:
        group_emd_loss = emd_loss_data[true_scores == score]
        mean_emd_loss.append(np.mean(group_emd_loss))
        std_emd_loss.append(np.std(group_emd_loss))

    # Convert to numpy arrays for plotting
    mean_emd_loss = np.array(mean_emd_loss)
    std_emd_loss = np.array(std_emd_loss)

    # Creating a figure to hold both subplots
    plt.figure(figsize=(10, 12))

    # First subplot for the histogram of true_scores
    plt.subplot(2, 1, 1)  # 2 rows, 1 column, first plot
    plt.hist(true_scores, bins=20, color='skyblue', edgecolor='black')
    plt.title('Histogram of PIAA Scores', fontsize=16)
    plt.xlabel('PIAA Scores', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)

    # Second subplot for the mean and std of EMD Loss Data
    plt.subplot(2, 1, 2)  # 2 rows, 1 column, second plot
    plt.errorbar(unique_scores, mean_emd_loss, yerr=std_emd_loss, fmt='-o', ecolor='red', capsize=5, capthick=2, label='Mean EMD Loss with STD')
    plt.xlabel('PIAA Scores', fontsize=12)
    plt.ylabel('EMD Loss Data', fontsize=12)
    plt.title('PIAA Scores vs EMD Loss Data (Mean and STD)', fontsize=16)
    plt.legend()

    # Adjusting layout
    plt.tight_layout()
    
    i = 1  # Starting index
    filename = f'LAPIS_KDE_plot_cv{i}.jpg'
    # Loop until a filename is found that does not exist
    while os.path.exists(filename):
        i += 1  # Increment the counter if the file exists
        filename = f'LAPIS_KDE_plot_cv{i}.jpg'  # Update the filename with the new counter
    # Once a unique filename is determined, proceed with saving the plot
    print(f'Save figure to {filename}')
    plt.savefig(filename)


    emd_loss = running_emd_loss / len(dataloader)
    emd_attr_loss = running_attr_emd_loss / len(dataloader)
    mse_loss = running_mse_loss / len(dataloader)
    return emd_loss, emd_attr_loss, srocc, mse_loss


def load_data(root_dir = '/home/lwchen/datasets/LAPIS', fold_id=1, n_fold=4):
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

    # Create datasets with the appropriate transformations
    piaa_dataset = LAPIS_PIAADataset(root_dir, transform=train_transform)
    train_lavis_piaa_dataset, test_lavis_piaa_dataset = create_image_split_dataset(piaa_dataset)
    print(len(train_lavis_piaa_dataset), len(test_lavis_piaa_dataset))
    train_lavis_piaa_dataset, test_lavis_piaa_dataset = create_user_split_dataset_kfold(piaa_dataset, train_lavis_piaa_dataset, test_lavis_piaa_dataset, fold_id, n_fold=n_fold)
    print(len(train_lavis_piaa_dataset), len(test_lavis_piaa_dataset))
    
    """Precompute"""
    pkl_dir = './LAPIS_dataset_pkl'
    # train_dataset = LAPIS_GIAA_HistogramDataset(root_dir, transform=train_transform, 
    #     data=train_lavis_piaa_dataset.data, map_file=os.path.join(pkl_dir,'trainset_image_dct_%dfold.pkl'%fold_id), 
    #     precompute_file=os.path.join(pkl_dir,'trainset_GIAA_dct_%dfold.pkl'%fold_id))
    train_dataset = LAPIS_MIAA_HistogramDataset(root_dir, transform=train_transform, 
        data=train_lavis_piaa_dataset.data, map_file=os.path.join(pkl_dir,'trainset_image_dct_%dfold.pkl'%fold_id), 
        precompute_file=os.path.join(pkl_dir,'trainset_MIAA_dct_%dfold.pkl'%fold_id))
    
    # train_dataset = LAPIS_GIAA_HistogramDataset(root_dir, transform=train_transform, data=train_lavis_piaa_dataset.data, map_file=os.path.join(pkl_dir,'trainset_image_dct.pkl'), precompute_file=os.path.join(pkl_dir,'trainset_GIAA_dct.pkl'))
    # train_dataset = LAPIS_MIAA_HistogramDataset(root_dir, transform=train_transform, data=train_lavis_piaa_dataset.data, map_file=os.path.join(pkl_dir,'trainset_image_dct.pkl'), precompute_file=os.path.join(pkl_dir,'trainset_MIAA_dct.pkl'))
    # train_dataset = LAPIS_PIAA_HistogramDataset(root_dir, transform=train_transform, data=train_lavis_piaa_dataset.data)
    
    # test_sgiaa_dataset = LAPIS_MIAA_HistogramDataset(root_dir, transform=test_transform, data=test_lavis_piaa_dataset.data, map_file=os.path.join(pkl_dir,'testset_image_dct.pkl'), precompute_file=os.path.join(pkl_dir,'testset_MIAA_dct.pkl'))
    # test_giaa_dataset = LAPIS_GIAA_HistogramDataset(root_dir, transform=test_transform, data=test_lavis_piaa_dataset.data, map_file=os.path.join(pkl_dir,'testset_image_dct.pkl'), precompute_file=os.path.join(pkl_dir,'testset_GIAA_dct.pkl'))
    # test_piaa_imgsort_dataset = LAPIS_PIAA_HistogramDataset_imgsort(root_dir, transform=test_transform, data=test_lavis_piaa_dataset.data, map_file=os.path.join(pkl_dir,'testset_image_dct.pkl'), precompute_file=os.path.join(pkl_dir,'testset_GIAA_dct.pkl'))
    test_giaa_dataset = LAPIS_GIAA_HistogramDataset(root_dir, transform=test_transform, data=test_lavis_piaa_dataset.data, map_file=os.path.join(pkl_dir,'testset_image_dct_%dfold.pkl'%fold_id), precompute_file=os.path.join(pkl_dir,'testset_GIAA_dct_%dfold.pkl'%fold_id))
    test_piaa_imgsort_dataset = LAPIS_PIAA_HistogramDataset_imgsort(root_dir, transform=test_transform, data=test_lavis_piaa_dataset.data, map_file=os.path.join(pkl_dir,'testset_image_dct_%dfold.pkl'%fold_id))
    test_piaa_dataset = LAPIS_PIAA_HistogramDataset(root_dir, transform=test_transform, data=test_lavis_piaa_dataset.data)
    
    return train_dataset, test_giaa_dataset, test_piaa_dataset, test_piaa_imgsort_dataset


is_eval = False
is_log = True
num_bins = 10
num_attr = 8
num_bins_attr = 5
num_pt = 132
# num_pt = 152
# num_pt = 20
resume = None

criterion_mse = nn.MSELoss()


if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description='Training and Testing the Combined Model for data spliting')
    parser.add_argument('--fold_id', type=int, default=1)
    parser.add_argument('--n_fold', type=int, default=4)
    args = parser.parse_args()
    
    random_seed = 42
    lr = 5e-5
    batch_size = 100
    num_epochs = 20
    lr_schedule_epochs = 5
    lr_decay_factor = 0.5
    max_patience_epochs = 10
    n_workers = 8
    eval_on_giaa = True
    
    if is_log:
        wandb.init(project="resnet_LAVIS_PIAA", 
                   notes="latefusion",
                   tags = ["no_arttype", "sGIAA", "CV%d/%d"%(args.fold_id, args.n_fold)])
        wandb.config = {
            "learning_rate": lr,
            "batch_size": batch_size,
            "num_epochs": num_epochs
        }
        experiment_name = wandb.run.name
    else:
        experiment_name = ''
    
    train_dataset, test_giaa_dataset, test_piaa_dataset, test_piaa_imgsort_dataset = load_data(fold_id=args.fold_id, n_fold=args.n_fold)
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers, timeout=300, collate_fn=collate_fn)
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers, timeout=300, collate_fn=collate_fn_imgsort)
    test_giaa_dataloader = DataLoader(test_giaa_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers, timeout=300, collate_fn=collate_fn)
    test_piaa_dataloader = DataLoader(test_piaa_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers, timeout=300, collate_fn=collate_fn)
    test_piaa_imgsort_dataloader = DataLoader(test_piaa_imgsort_dataset, batch_size=2, shuffle=False, num_workers=n_workers, timeout=300, collate_fn=collate_fn_imgsort)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the combined model
    model = CombinedModel(num_bins, num_attr, num_bins_attr, num_pt).to(device)
    # model = NIMA(num_bins).to(device)
    
    if resume is not None:
        model.load_state_dict(torch.load(resume))
    # Loss and optimizer
    # criterion_mse = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Initialize the best test loss and the best model
    best_model = None
    best_modelname = 'lapis_best_model_resnet50_histo_lr%1.0e_decay_%depoch' % (lr, num_epochs)
    best_modelname += '_%s'%experiment_name
    best_modelname += '.pth'
    best_modelname = os.path.join('models_pth', best_modelname)
    
    # Training loop
    best_test_srocc = 0
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
        test_piaa_emd_loss, test_piaa_attr_emd_loss, test_piaa_srocc, test_piaa_mse = evaluate(model, test_piaa_imgsort_dataloader, earth_mover_distance, device)
        if is_log:
            wandb.log({
                "Test PIAA EMD Loss": test_piaa_emd_loss,
                "Test PIAA SROCC": test_piaa_srocc,
                       }, commit=False)
        test_giaa_emd_loss, test_giaa_attr_emd_loss, test_giaa_srocc, test_giaa_mse = evaluate(model, test_giaa_dataloader, earth_mover_distance, device)
        if is_log:
            wandb.log({
                "Test GIAA EMD Loss": test_giaa_emd_loss,
                "Test GIAA SROCC": test_giaa_srocc,
                       }, commit=True)
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
    # test_piaa_emd_loss, test_piaa_attr_emd_loss, test_piaa_srocc, test_piaa_mse = evaluate_subPIAA(model, test_piaa_dataloader, earth_mover_distance, device)
    test_piaa_emd_loss, test_piaa_attr_emd_loss, test_piaa_srocc, test_piaa_mse = evaluate(model, test_piaa_imgsort_dataloader, earth_mover_distance, device)
    # test_piaa_emd_loss, test_piaa_attr_emd_loss, test_piaa_srocc, test_piaa_mse = evaluate(model, test_piaa_dataloader, earth_mover_distance, device)
    # test_user_piaa_emd_loss, test_user_piaa_attr_emd_loss, test_user_piaa_srocc, test_user_piaa_mse = evaluate(model, test_user_piaa_dataloader, earth_mover_distance, device)
    test_giaa_emd_loss, test_giaa_attr_emd_loss, test_giaa_srocc, test_giaa_mse = evaluate(model, test_giaa_dataloader, earth_mover_distance, device)
    # test_sgiaa_emd_loss, test_sgiaa_attr_emd_loss, test_sgiaa_srocc, test_sgiaa_mse = evaluate(model, test_sgiaa_dataloader, earth_mover_distance, device)
    
    if is_log:
        wandb.log({
            "Test GIAA EMD Loss": test_giaa_emd_loss,
            "Test GIAA SROCC": test_giaa_srocc,
            "Test GIAA MSE": test_giaa_mse,
            "Test PIAA EMD Loss": test_piaa_emd_loss,
            "Test PIAA SROCC": test_piaa_srocc,
            "Test PIAA MSE": test_piaa_mse,
        }, commit=True)
