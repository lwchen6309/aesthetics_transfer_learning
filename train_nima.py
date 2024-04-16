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
from PARA_histogram_dataloader import PARA_GIAA_HistogramDataset, PARA_PIAA_HistogramDataset, PARA_sGIAA_HistogramDataset, PARA_PIAA_HistogramDataset_imgsort, collate_fn_imgsort
from PARA_PIAA_dataloader import PARA_PIAADataset, create_user_split_dataset_kfold, split_dataset_by_images
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
    # scale_aesthetic = torch.arange(1, 5.5, 0.5).to(device)
    running_ce_improvement = 0
    running_ce_init = 0
    for sample in progress_bar:
        if is_eval:
            break
        
        images = sample['image'].to(device)
        aesthetic_score_histogram = sample['aestheticScore'].to(device)
        traits_histogram = sample['traits'].to(device)
        onehot_big5 = sample['big5'].to(device)
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
    return epoch_emd_loss, epoch_total_emd_loss

def save_results(dataset, userIds, traits_histograms, emd_loss_data, predicted_scores, true_scores):
    
    df = dataset.decode_batch_to_dataframe(traits_histograms[:,:20])
    df['userID'] = userIds
    df['EMD_Loss_Data'] = emd_loss_data
    df['PIAA_Score'] = true_scores
    df['PIAA_Pred'] = predicted_scores
    
    i = 1  # Starting index
    filename = f'evaluation_results{i}.csv'
    # Loop until a filename is found that does not exist
    while os.path.exists(filename):
        i += 1  # Increment the counter if the file exists
        filename = f'evaluation_results{i}.csv'  # Update the filename with the new counter
    # Once a unique filename is determined, proceed with saving the plot
    print(f'Save EMD loss to {filename}')
    # Save to CSV
    df.to_csv(filename, index=False)


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
            onehot_big5 = sample['big5'].to(device)
            # attributes_target_histogram = sample['attributes'].to(device).view(-1, num_attr, num_bins_attr) # Reshape to match our logits shape
            traits_histogram = torch.cat([traits_histogram, onehot_big5], dim=1)
            
            aesthetic_logits = model(images, traits_histogram)
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


# Evaluation Function
def evaluate_each_datum(model, dataloader, criterion, device):
    model.eval()
    running_emd_loss = 0.0
    running_attr_emd_loss = 0.0
    running_mse_loss = 0.0
    scale = torch.arange(1, 5.5, 0.5).to(device)
    eval_srocc = True

    progress_bar = tqdm(dataloader, leave=False)
    mean_pred = []
    mean_target = []
    userIds = []
    emd_loss_data = []
    traits_histograms = []
    with torch.no_grad():
        for sample in progress_bar:
            images = sample['image'].to(device)
            userId = sample['userId']
            userIds.extend(userId)
            aesthetic_score_histogram = sample['aestheticScore'].to(device)
            traits_histogram = sample['traits'].to(device)
            onehot_big5 = sample['big5'].to(device)
            attributes_target_histogram = sample['attributes'].to(device).view(-1, num_attr, num_bins_attr) # Reshape to match our logits shape
            traits_histogram = torch.cat([traits_histogram, onehot_big5], dim=1)
            traits_histograms.append(traits_histogram.cpu().numpy())

            aesthetic_logits = model(images, traits_histogram)
            prob_aesthetic = F.softmax(aesthetic_logits, dim=1)

            emd_loss_datum = criterion(prob_aesthetic, aesthetic_score_histogram)
            emd_loss_data.append(emd_loss_datum.view(-1).cpu().numpy())
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

    traits_histograms = np.concatenate(traits_histograms)
    emd_loss_data = np.concatenate(emd_loss_data)
    save_results(dataloader.dataset, userIds, traits_histograms, emd_loss_data, predicted_scores, true_scores)
    
    emd_loss = running_emd_loss / len(dataloader)
    emd_attr_loss = running_attr_emd_loss / len(dataloader)
    mse_loss = running_mse_loss / len(dataloader)
    return emd_loss, emd_attr_loss, srocc, mse_loss


def load_data(args, root_dir = '/home/lwchen/datasets/PARA/'):
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

    fold_id = args.fold_id
    n_fold = args.n_fold
    # Load datasets
    # Create datasets with the appropriate transformations
    dataset = PARA_PIAADataset(root_dir, transform=train_transform)
    train_piaa_dataset = PARA_PIAADataset(root_dir, transform=train_transform)
    test_piaa_dataset = PARA_PIAADataset(root_dir, transform=train_transform)
    train_dataset, test_dataset = split_dataset_by_images(train_piaa_dataset, test_piaa_dataset, root_dir)
    # Assuming shell_users_df contains the shell user DataFrame
    if args.use_cv:
        train_dataset, test_dataset = create_user_split_dataset_kfold(dataset, train_dataset, test_dataset, fold_id=fold_id, n_fold=n_fold)

    # _, test_user_piaa_dataset = split_dataset_by_user(
    #     PARA_PIAADataset(root_dir, transform=train_transform),  
    #     test_count=40, max_annotations_per_user=[100,50], seed=random_seed)
    
    # Create datasets with the appropriate transformations
    # train_dataset = PARA_HistogramDataset(root_dir, transform=train_transform, data=train_dataset.data, map_file='trainset_image_dct.pkl')
    # test_dataset = PARA_HistogramDataset(root_dir, transform=test_transform, data=test_dataset.data, map_file='testset_image_dct.pkl')
    print(len(train_dataset), len(test_dataset))
    pkl_dir = './dataset_pkl'
    if args.use_cv:
        pkl_dir = os.path.join(pkl_dir, 'user_cv')
        train_dataset = PARA_GIAA_HistogramDataset(root_dir, transform=train_transform, data=train_piaa_dataset.data, map_file=os.path.join(pkl_dir,'trainset_image_dct_%dfold.pkl'%fold_id), precompute_file=os.path.join(pkl_dir,'trainset_GIAA_dct_%dfold.pkl'%fold_id))
        test_giaa_dataset = PARA_GIAA_HistogramDataset(root_dir, transform=test_transform, data=test_dataset.data, map_file=os.path.join(pkl_dir,'testset_image_dct_%dfold.pkl'%fold_id), precompute_file=os.path.join(pkl_dir,'testset_GIAA_dct_%dfold.pkl'%fold_id))
        test_piaa_imgsort_dataset = PARA_PIAA_HistogramDataset_imgsort(root_dir, transform=test_transform, data=test_dataset.data, map_file=os.path.join(pkl_dir,'testset_image_dct_%dfold.pkl'%fold_id))
        # train_dataset, test_giaa_dataset, _, test_piaa_dataset = load_usersplit_data(root_dir = '/home/lwchen/datasets/PARA/', miaa=False)
    else:
        train_dataset = PARA_GIAA_HistogramDataset(root_dir, transform=train_transform, data=train_piaa_dataset.data, map_file=os.path.join(pkl_dir,'trainset_image_dct.pkl'), precompute_file=os.path.join(pkl_dir,'trainset_GIAA_dct.pkl'))
        # train_dataset = PARA_sGIAA_HistogramDataset(root_dir, transform=train_transform, data=train_piaa_dataset.data, map_file=os.path.join(pkl_dir,'trainset_image_dct.pkl'), precompute_file=os.path.join(pkl_dir,'trainset_MIAA_nopiaa_dct.pkl'))
        # train_dataset = PARA_PIAA_HistogramDataset(root_dir, transform=train_transform, data=train_dataset.data)
        # train_dataset = PARA_PIAA_HistogramDataset_imgsort(root_dir, transform=train_transform, data=train_piaa_dataset.data, map_file=os.path.join(pkl_dir,'trainset_image_dct.pkl'))
        test_giaa_dataset = PARA_GIAA_HistogramDataset(root_dir, transform=test_transform, data=test_dataset.data, map_file=os.path.join(pkl_dir,'testset_image_dct.pkl'), precompute_file=os.path.join(pkl_dir,'testset_GIAA_dct.pkl'))
        test_piaa_imgsort_dataset = PARA_PIAA_HistogramDataset_imgsort(root_dir, transform=test_transform, data=test_dataset.data, map_file=os.path.join(pkl_dir,'testset_image_dct.pkl'))
    test_piaa_dataset = PARA_PIAA_HistogramDataset(root_dir, transform=test_transform, data=test_dataset.data)
    # test_user_piaa_dataset = PARA_PIAA_HistogramDataset(root_dir, transform=test_transform, data=test_user_piaa_dataset.data)

    return train_dataset, test_giaa_dataset, test_piaa_dataset, test_piaa_imgsort_dataset


num_bins = 9
num_attr = 8
num_bins_attr = 5
num_pt = 50 + 20
criterion_mse = nn.MSELoss()


if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description='Training and Testing the Combined Model for data spliting')
    parser.add_argument('--fold_id', type=int, default=1)
    parser.add_argument('--n_fold', type=int, default=4)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--use_cv', action='store_true', help='Enable cross validation')
    parser.add_argument('--is_eval', action='store_true', help='Enable evaluation mode')
    parser.add_argument('--no_log', action='store_false', dest='is_log', help='Disable logging')
    args = parser.parse_args()
    
    resume = args.resume
    is_eval = args.is_eval
    is_log = args.is_log

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
        tags = ["no_attr","GIAA"]
        if args.use_cv:
            tags += ["CV%d/%d"%(args.fold_id, args.n_fold)]
        wandb.init(project="resnet_PARA_PIAA", 
                   notes="NIMA",
                   tags = tags)
        wandb.config = {
            "learning_rate": lr,
            "batch_size": batch_size,
            "num_epochs": num_epochs
        }
        experiment_name = wandb.run.name
    else:
        experiment_name = ''
    
    train_dataset, test_giaa_dataset, test_piaa_dataset, test_piaa_imgsort_dataset = load_data(args)
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers, timeout=300)
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers, timeout=300, collate_fn=collate_fn_imgsort)
    test_giaa_dataloader = DataLoader(test_giaa_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers, timeout=300)
    test_piaa_dataloader = DataLoader(test_piaa_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers, timeout=300)
    test_piaa_imgsort_dataloader = DataLoader(test_piaa_imgsort_dataset, batch_size=5, shuffle=False, num_workers=n_workers, timeout=300, collate_fn=collate_fn_imgsort)
    # test_user_piaa_dataloader = DataLoader(test_user_piaa_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers, timeout=300)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the combined model
    model = NIMA(num_bins).to(device)

    if resume is not None:
        model.load_state_dict(torch.load(resume))
    # Loss and optimizer
    # criterion_mse = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Initialize the best test loss and the best model
    best_model = None
    best_modelname = 'best_model_resnet50_nima_lr%1.0e_decay_%depoch' % (lr, num_epochs)
    best_modelname += '_%s'%experiment_name
    best_modelname += '.pth'
    dirname = 'models_pth'
    if args.use_cv:
        dirname = os.path.join(dirname, 'random_cvs')
    best_modelname = os.path.join(dirname, best_modelname)
    
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
    test_piaa_emd_loss, test_piaa_attr_emd_loss, test_piaa_srocc, test_piaa_mse = evaluate_each_datum(model, test_piaa_imgsort_dataloader, earth_mover_distance, device)
    # raise Exception
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
