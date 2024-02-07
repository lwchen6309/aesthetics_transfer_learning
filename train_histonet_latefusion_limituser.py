import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import wandb
from PARA_histogram_dataloader import PARA_GIAA_HistogramDataset, PARA_PIAA_HistogramDataset, PARA_MIAA_HistogramDataset, PARA_PIAA_HistogramDataset_imgsort, collate_fn_imgsort
from PARA_PIAA_dataloader import PARA_PIAADataset, split_dataset_by_user, split_dataset_by_images, limit_annotations_per_user
import pandas as pd
from train_histonet_latefusion import CombinedModel, earth_mover_distance, train, evaluate, evaluate_PIAA_imgsort


def load_data(root_dir = '/home/lwchen/datasets/PARA/', method = 'pacmap', dims = 2, num_user = 200, is_reverse=False):
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
    filename = '%dD_shell_%duser_ids_%s.csv'%(dims, num_user, method)
    if is_reverse:
        filename = filename.replace('.csv', '_rev.csv')
    filename = os.path.join('shell_users', '500imgs', filename)
    shell_users_df = pd.read_csv(filename)
    print('Read user from %s'%filename)
    
    filtered_data = train_dataset.data[train_dataset.data['userId'].isin(shell_users_df['userId'])]
    filtered_data, _ = limit_annotations_per_user(filtered_data, max_annotations_per_user=[500,1])
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
    
    test_giaa_dataset = PARA_GIAA_HistogramDataset(root_dir, transform=test_transform, data=test_dataset.data, map_file=os.path.join(pkl_dir,'testset_image_dct.pkl'), precompute_file=os.path.join(pkl_dir,'testset_GIAA_dct.pkl'))
    test_piaa_imgsort_dataset = PARA_PIAA_HistogramDataset_imgsort(root_dir, transform=test_transform, data=test_dataset.data, map_file=os.path.join(pkl_dir,'testset_image_dct.pkl'))
    test_piaa_dataset = PARA_PIAA_HistogramDataset(root_dir, transform=test_transform, data=test_dataset.data)
    # test_piaa_dataset = PARA_GSP_HistogramDataset(root_dir, transform=test_transform, piaa_data=test_piaa_dataset.data, 
    #                 giaa_data=test_piaa_dataset.data, map_file=os.path.join(pkl_dir,'testset_image_dct.pkl'), precompute_file=os.path.join(pkl_dir,'testset_GIAA_dct.pkl'))
    test_user_piaa_dataset = PARA_PIAA_HistogramDataset(root_dir, transform=test_transform, data=test_user_piaa_dataset.data)

    # train_dataset, test_giaa_dataset, _, test_piaa_dataset = load_usersplit_data(root_dir = '/home/lwchen/datasets/PARA/', miaa=False)

    return train_dataset, test_giaa_dataset, test_piaa_dataset, test_user_piaa_dataset, test_piaa_imgsort_dataset


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
    parser = argparse.ArgumentParser(description='Process dimensions and number of users.')
    parser.add_argument('--dims', type=int, default=2, help='Number of dimensions')
    parser.add_argument('--num_users', type=int, default=50, help='Number of users')
    parser.add_argument('--is_reverse', action='store_true', help='Reverse the order of processing')
    parser.add_argument('--method', type=str, default='pacmap', choices=['pacmap', 'pca'],
                        help='Method for dimensionality reduction: pacmap or pca (default: pacmap)')
    
    args = parser.parse_args()
    
    random_seed = 42
    lr = 5e-5
    batch_size = 100
    num_epochs = 20
    lr_schedule_epochs = 5
    lr_decay_factor = 0.5
    max_patience_epochs = 10
    n_workers = 4
    
    if is_log:
        wandb.init(project="resnet_PARA_PIAA_usersample", 
                   notes="latefusion",
                   tags = ["no_attr","PIAA","OuterShell_%ddim_%duser"%(args.dims, args.num_users),'reverse=%d'%args.is_reverse, args.method, '500imgs'])
        wandb.config = {
            "learning_rate": lr,
            "batch_size": batch_size,
            "num_epochs": num_epochs
        }
        experiment_name = wandb.run.name
    else:
        experiment_name = ''
    
    train_dataset, test_giaa_dataset, test_piaa_dataset, test_user_piaa_dataset, test_piaa_imgsort_dataset = load_data(method=args.method, dims=args.dims, num_user=args.num_users, is_reverse=args.is_reverse)
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers, timeout=300)
    test_giaa_dataloader = DataLoader(test_giaa_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers, timeout=300)
    test_piaa_dataloader = DataLoader(test_piaa_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers, timeout=300)
    test_piaa_imgsort_dataloader = DataLoader(test_piaa_imgsort_dataset, batch_size=5, shuffle=False, num_workers=n_workers, timeout=300, collate_fn=collate_fn_imgsort)
    test_user_piaa_dataloader = DataLoader(test_user_piaa_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers, timeout=300)

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
    experiment_name = 'tmp'
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
        
        if epoch % 4 > 0:
            continue
        
        # # Testing
        # test_piaa_emd_loss, test_piaa_attr_emd_loss, test_piaa_srocc, test_piaa_mse = evaluate(model, test_piaa_dataloader, earth_mover_distance, device)
        test_piaa_emd_loss, test_piaa_attr_emd_loss, test_piaa_srocc, test_piaa_mse = evaluate_PIAA_imgsort(model, test_piaa_imgsort_dataloader, earth_mover_distance, device)
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
    # test_piaa_emd_loss, test_piaa_attr_emd_loss, test_piaa_srocc, test_piaa_mse = evaluate(model, test_piaa_dataloader, earth_mover_distance, device)
    test_piaa_emd_loss, test_piaa_attr_emd_loss, test_piaa_srocc, test_piaa_mse = evaluate_PIAA_imgsort(model, test_piaa_imgsort_dataloader, earth_mover_distance, device)
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
