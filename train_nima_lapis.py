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
from train_histonet_latefusion_lapis import train, evaluate, earth_mover_distance
# from time import time


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

def load_data(args, root_dir = '/home/lwchen/datasets/LAPIS'):
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
    # Create datasets with the appropriate transformations
    piaa_dataset = LAPIS_PIAADataset(root_dir, transform=train_transform)
    train_lavis_piaa_dataset, test_lavis_piaa_dataset = create_image_split_dataset(piaa_dataset)
    if args.use_cv:
        train_lavis_piaa_dataset, test_lavis_piaa_dataset = create_user_split_dataset_kfold(piaa_dataset, train_lavis_piaa_dataset, test_lavis_piaa_dataset, fold_id, n_fold=n_fold)
    
    """Precompute"""
    pkl_dir = './LAPIS_dataset_pkl'
    if args.use_cv:
        train_dataset = LAPIS_GIAA_HistogramDataset(root_dir, transform=train_transform, 
            data=train_lavis_piaa_dataset.data, map_file=os.path.join(pkl_dir,'trainset_image_dct_%dfold.pkl'%fold_id), 
            precompute_file=os.path.join(pkl_dir,'trainset_GIAA_dct_%dfold.pkl'%fold_id))        
        # train_dataset = LAPIS_MIAA_HistogramDataset(root_dir, transform=train_transform, 
        #     data=train_lavis_piaa_dataset.data, map_file=os.path.join(pkl_dir,'trainset_image_dct_%dfold.pkl'%fold_id), 
        #     precompute_file=os.path.join(pkl_dir,'trainset_MIAA_dct_%dfold.pkl'%fold_id))
        test_giaa_dataset = LAPIS_GIAA_HistogramDataset(root_dir, transform=test_transform, data=test_lavis_piaa_dataset.data, map_file=os.path.join(pkl_dir,'testset_image_dct_%dfold.pkl'%fold_id), precompute_file=os.path.join(pkl_dir,'testset_GIAA_dct_%dfold.pkl'%fold_id))
        test_piaa_imgsort_dataset = LAPIS_PIAA_HistogramDataset_imgsort(root_dir, transform=test_transform, data=test_lavis_piaa_dataset.data, map_file=os.path.join(pkl_dir,'testset_image_dct_%dfold.pkl'%fold_id))
    else:
        train_dataset = LAPIS_GIAA_HistogramDataset(root_dir, transform=train_transform, data=train_lavis_piaa_dataset.data, map_file=os.path.join(pkl_dir,'trainset_image_dct.pkl'), precompute_file=os.path.join(pkl_dir,'trainset_GIAA_dct.pkl'))
        # train_dataset = LAPIS_MIAA_HistogramDataset(root_dir, transform=train_transform, data=train_lavis_piaa_dataset.data, map_file=os.path.join(pkl_dir,'trainset_image_dct.pkl'), precompute_file=os.path.join(pkl_dir,'trainset_MIAA_dct.pkl'))
        # train_dataset = LAPIS_PIAA_HistogramDataset(root_dir, transform=train_transform, data=train_lavis_piaa_dataset.data)
        test_giaa_dataset = LAPIS_GIAA_HistogramDataset(root_dir, transform=test_transform, data=test_lavis_piaa_dataset.data, map_file=os.path.join(pkl_dir,'testset_image_dct.pkl'), precompute_file=os.path.join(pkl_dir,'testset_GIAA_dct.pkl'))
        test_piaa_imgsort_dataset = LAPIS_PIAA_HistogramDataset_imgsort(root_dir, transform=test_transform, data=test_lavis_piaa_dataset.data, map_file=os.path.join(pkl_dir,'testset_image_dct.pkl'))
    test_piaa_dataset = LAPIS_PIAA_HistogramDataset(root_dir, transform=test_transform, data=test_lavis_piaa_dataset.data)

    return train_dataset, test_giaa_dataset, test_piaa_dataset, test_piaa_imgsort_dataset


def save_results(dataset, userIds, traits_histograms, emd_loss_data, predicted_scores, true_scores):
    
    df = dataset.decode_batch_to_dataframe(traits_histograms[:,:60])
    df['userID'] = userIds
    df['EMD_Loss_Data'] = emd_loss_data
    df['PIAA_Score'] = true_scores
    df['PIAA_Pred'] = predicted_scores
    
    i = 1  # Starting index
    filename = f'LAPIS_evaluation_results{i}.csv'
    # Loop until a filename is found that does not exist
    while os.path.exists(filename):
        i += 1  # Increment the counter if the file exists
        filename = f'LAPIS_evaluation_results{i}.csv'  # Update the filename with the new counter
    # Once a unique filename is determined, proceed with saving the plot
    print(f'Save EMD loss to {filename}')
    # Save to CSV
    df.to_csv(filename, index=False)


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
    userIds = []
    emd_loss_data = []
    traits_histograms = []
    with torch.no_grad():
        for sample in progress_bar:
            images = sample['image'].to(device)
            aesthetic_score_histogram = sample['aestheticScore'].to(device)
            traits_histogram = sample['traits'].to(device)
            art_type = sample['art_type'].to(device)
            userIds.extend(sample['userId'])
            traits_histogram = torch.cat([traits_histogram, art_type], dim=1)
            traits_histograms.append(traits_histogram.cpu().numpy())

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
    traits_histograms = np.concatenate(traits_histograms)
    
    save_results(dataloader.dataset, userIds, traits_histograms, emd_loss_data, predicted_scores, true_scores)

    emd_loss = running_emd_loss / len(dataloader)
    emd_attr_loss = running_attr_emd_loss / len(dataloader)
    mse_loss = running_mse_loss / len(dataloader)
    return emd_loss, emd_attr_loss, srocc, mse_loss

num_bins = 10
num_attr = 8
num_bins_attr = 5
num_pt = 137
criterion_mse = nn.MSELoss()


if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description='Training and Testing the Combined Model for data spliting')
    parser.add_argument('--fold_id', type=int, default=1)
    parser.add_argument('--n_fold', type=int, default=4)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--is_eval', action='store_true', help='Enable evaluation mode')
    parser.add_argument('--use_cv', action='store_true', help='Enable cross validation')
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
        tags = ["no_arttype", "GIAA"]
        if args.use_cv:
            tags += ["CV%d/%d"%(args.fold_id, args.n_fold)]
        wandb.init(project="resnet_LAVIS_PIAA", 
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
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers, timeout=300, collate_fn=collate_fn)
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers, timeout=300, collate_fn=collate_fn_imgsort)
    test_giaa_dataloader = DataLoader(test_giaa_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers, timeout=300, collate_fn=collate_fn)
    test_piaa_dataloader = DataLoader(test_piaa_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers, timeout=300, collate_fn=collate_fn)
    test_piaa_imgsort_dataloader = DataLoader(test_piaa_imgsort_dataset, batch_size=2, shuffle=False, num_workers=n_workers, timeout=300, collate_fn=collate_fn_imgsort)
    
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
    best_modelname = 'lapis_best_model_resnet50_nima_lr%1.0e_decay_%depoch' % (lr, num_epochs)
    best_modelname += '_%s'%experiment_name
    best_modelname += '.pth'
    dirname = 'models_pth'
    if args.use_cv:
        dirname = os.path.join(dirname, 'random_cvs')
    best_modelname = os.path.join(dirname, best_modelname)
    
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
        eval_loss = test_giaa_emd_loss if eval_on_giaa else test_piaa_emd_loss

        # Early stopping check
        if eval_loss < best_test_loss:
            best_test_loss = eval_loss
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
            # "Test sGIAA EMD Loss": test_sgiaa_emd_loss,
            # "Test sGIAA SROCC": test_sgiaa_srocc,
            # "Test sGIAA MSE": test_sgiaa_mse, 
            "Test PIAA EMD Loss": test_piaa_emd_loss,
            "Test PIAA SROCC": test_piaa_srocc,
            "Test PIAA MSE": test_piaa_mse,
            # "Test user PIAA EMD Loss": test_user_piaa_emd_loss,
            # "Test user PIAA Attr EMD Loss": test_user_piaa_attr_emd_loss,
            # "Test user PIAA SROCC": test_user_piaa_srocc,
            # "Test user PIAA MSE": test_user_piaa_mse
        }, commit=True)
