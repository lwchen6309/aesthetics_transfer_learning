import os
import random
import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
# from torchvision import transforms
# from torchvision.models import resnet50
import wandb
# from scipy.stats import spearmanr
from PARA_histogram_dataloader import load_data, collate_fn_imgsort
from train_histonet_latefusion import CombinedModel, trainer, train, evaluate, evaluate_with_prior
# import argparse
from utils.argflags import parse_arguments


num_bins = 9
num_attr = 8
num_bins_attr = 5
num_pt = 50 + 20

if __name__ == '__main__': 
    args = parse_arguments()
    
    batch_size = args.batch_size
    
    random_seed = 42
    n_workers = 8
    
    if args.is_log:
        tags = ["no_attr", args.trainset, "Trait specific", "Test trait: %s_%s"%(args.trait, args.value)]
        if args.dropout is not None:
            tags += [f"dropout={args.dropout}"]
        wandb.init(project="resnet_PARA_PIAA", 
                   notes="latefusion",
                   tags = tags)
        wandb.config = {
            "learning_rate": args.lr,
            "batch_size": batch_size,
            "num_epochs": args.num_epochs
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
    dataloaders = (train_dataloader, val_giaa_dataloader, val_piaa_imgsort_dataloader, test_giaa_dataloader, test_piaa_imgsort_dataloader)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the combined model
    model = CombinedModel(num_bins, num_attr, num_bins_attr, num_pt, args.dropout).to(device)
    
    if args.resume is not None:
        model.load_state_dict(torch.load(args.resume))
    # Loss and optimizer
    # criterion_mse = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Initialize the best test loss and the best model
    best_model = None
    best_modelname = 'best_model_resnet50_latefusion_lr%1.0e_decay_%depoch' % (args.lr, args.num_epochs)
    best_modelname += '_%s'%experiment_name
    best_modelname += '.pth'
    dirname = 'models_pth'
    dirname = os.path.join(dirname, 'trait_disjoint_exp')
    best_modelname = os.path.join(dirname, best_modelname)
    
    trainer(dataloaders, model, optimizer, args, train, (evaluate, evaluate_with_prior), device, best_modelname)
    # emd_loss, emd_attr_loss, srocc, mse_loss = evaluate_each_datum(model, test_piaa_imgsort_dataloader, device)
