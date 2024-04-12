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
from PARA_histogram_dataloader import PARA_GIAA_HistogramDataset, PARA_PIAA_HistogramDataset, PARA_sGIAA_HistogramDataset, PARA_PIAA_HistogramDataset_imgsort, collate_fn_imgsort
from PARA_PIAA_dataloader import PARA_PIAADataset, split_dataset_by_user, split_dataset_by_images, split_dataset_by_trait
from train_nima import NIMA, earth_mover_distance, train, evaluate
import argparse
import copy


def load_model(model_path, device):
    model = NIMA(num_bins).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


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
    return emd_loss, emd_attr_loss, srocc, mse_loss, predicted_scores, true_scores


def ensemble_predictions(model_paths, dataloader, device):
    all_predictions = []
    for path in model_paths:
        model = load_model(path, device)
        _, _, _, _, predicted_scores, true_scores = evaluate(model, dataloader, earth_mover_distance, device)
        all_predictions.append(predicted_scores)
    # Average the predictions across all models
    ensemble_pred = np.stack(all_predictions)
    print('---'*3)
    for pred in ensemble_pred:
        srocc, _ = spearmanr(pred, true_scores)
        print(srocc)
    srocc, _ = spearmanr(ensemble_pred.mean(0), true_scores)
    print('Ensemble', srocc)
    return ensemble_pred, srocc


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

    # Load datasets
    # Create datasets with the appropriate transformations
    # dataset = PARA_PIAADataset(root_dir, transform=train_transform)
    train_piaa_dataset = PARA_PIAADataset(root_dir, transform=train_transform)
    test_piaa_dataset = PARA_PIAADataset(root_dir, transform=train_transform)
    train_dataset, test_dataset = split_dataset_by_images(train_piaa_dataset, test_piaa_dataset, root_dir)
    orig_train, orig_test = len(train_dataset), len(test_dataset)
    
    trait_disjoint = False
    if trait_disjoint:
        train_dataset.data = train_dataset.data[train_dataset.data[args.trait] != args.value]
    else:
        train_dataset.data = train_dataset.data[train_dataset.data[args.trait] == args.value]
    test_dataset.data = test_dataset.data[test_dataset.data[args.trait] == args.value]
    print('trainset %d->%d, testset %d->%d'%(orig_train, len(train_dataset), orig_test, len(test_dataset)))
    
    # Create datasets with the appropriate transformations
    if trait_disjoint:
        pkl_dir = './dataset_pkl/trait_split'
    else:
        pkl_dir = './dataset_pkl/trait_specific'

    suffix = '%s_%s'%(args.trait, args.value)
    train_dataset = PARA_GIAA_HistogramDataset(root_dir, transform=train_transform, data=train_dataset.data, map_file=os.path.join(pkl_dir,'trainset_image_dct_%s.pkl'%suffix), precompute_file=os.path.join(pkl_dir,'trainset_GIAA_dct_%s.pkl'%suffix))
    test_giaa_dataset = PARA_GIAA_HistogramDataset(root_dir, transform=test_transform, data=test_dataset.data, map_file=os.path.join(pkl_dir,'testset_image_dct_%s.pkl'%suffix), precompute_file=os.path.join(pkl_dir,'testset_GIAA_dct_%s.pkl'%suffix))
    test_piaa_dataset = PARA_PIAA_HistogramDataset(root_dir, transform=test_transform, data=test_dataset.data)
    test_piaa_imgsort_dataset = PARA_PIAA_HistogramDataset_imgsort(root_dir, transform=test_transform, data=test_dataset.data, map_file=os.path.join(pkl_dir,'testset_image_dct_%s.pkl'%suffix))
    # test_user_piaa_dataset = PARA_PIAA_HistogramDataset(root_dir, transform=test_transform, data=test_user_piaa_dataset.data)
    
    return train_dataset, test_giaa_dataset, test_piaa_dataset, test_piaa_imgsort_dataset



is_eval = False
is_log = False
num_bins = 9
num_attr = 8
num_bins_attr = 5
num_pt = 50 + 20
resume = None
criterion_mse = nn.MSELoss()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training and Testing the Combined Model for data spliting')
    parser.add_argument('--trait', type=str, default=None)
    parser.add_argument('--value', type=str, default=None)
    # Adding argument to receive a list of model paths
    parser.add_argument('--model_paths', nargs='+', required=True, help='Paths to pretrained NIMA model files')
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
        wandb.init(project="resnet_PARA_PIAA", 
                   notes="NIMA",
                   tags = ["no_attr","GIAA", "Test trait: %s_%s"%(args.trait, args.value)])
        wandb.config = {
            "learning_rate": lr,
            "batch_size": batch_size,
            "num_epochs": num_epochs
        }
        experiment_name = wandb.run.name
    else:
        experiment_name = ''
    

    # Load datasets
    train_dataset, test_giaa_dataset, test_piaa_dataset, test_piaa_imgsort_dataset = load_data(args=args)
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers, timeout=300)
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers, timeout=300, collate_fn=collate_fn_imgsort)
    test_giaa_dataloader = DataLoader(test_giaa_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers, timeout=300)
    test_piaa_dataloader = DataLoader(test_piaa_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers, timeout=300)
    test_piaa_imgsort_dataloader = DataLoader(test_piaa_imgsort_dataset, batch_size=5, shuffle=False, num_workers=n_workers, timeout=300, collate_fn=collate_fn_imgsort)
    # test_user_piaa_dataloader = DataLoader(test_user_piaa_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers, timeout=300)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ensemble_pred, srocc = ensemble_predictions(args.model_paths, test_giaa_dataloader, device)
    raise Exception
   

    # Initialize the combined model
    # model = CombinedModel(num_bins, num_attr, num_bins_attr, num_pt).to(device)
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
    best_modelname += '_%s_%s'%(args.trait, args.value)
    best_modelname += '.pth'
    
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
