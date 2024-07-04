import argparse
import os
import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
# from torchvision import transforms
from torchvision.models import resnet50
# import numpy as np
# import pandas as pd
# from tqdm import tqdm
import wandb
# from scipy.stats import spearmanr
from LAPIS_histogram_dataloader import load_data, collate_fn_imgsort, collate_fn
from train_histonet_latefusion_lapis import train, evaluate
from train_nima import trainer, NIMA

from utils.argflags import parse_arguments, wandb_tags, model_dir


num_bins = 10
num_attr = 8
num_bins_attr = 5
num_pt = 137
criterion_mse = nn.MSELoss()


if __name__ == '__main__':    
    args = parse_arguments()
    batch_size = args.batch_size

    if args.is_log:
        tags = ["no_attr","GIAA"]
        tags += wandb_tags(args)
        wandb.init(project="resnet_LAVIS_PIAA", 
                   notes="NIMA",
                   tags = tags)
        wandb.config = {
            "learning_rate": args.lr,
            "batch_size": batch_size,
            "num_epochs": args.num_epochs
        }
        experiment_name = wandb.run.name
    else:
        experiment_name = ''
    
    # Create dataloaders
    train_dataset, val_giaa_dataset, val_piaa_imgsort_dataset, test_giaa_dataset, test_piaa_imgsort_dataset = load_data(args)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, timeout=300, collate_fn=collate_fn)
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, timeout=300, collate_fn=collate_fn_imgsort)
    val_giaa_dataloader = DataLoader(val_giaa_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, timeout=300, collate_fn=collate_fn)
    val_piaa_imgsort_dataloader = DataLoader(val_piaa_imgsort_dataset, batch_size=5, shuffle=False, num_workers=args.num_workers, timeout=300, collate_fn=collate_fn_imgsort)
    test_giaa_dataloader = DataLoader(test_giaa_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, timeout=300, collate_fn=collate_fn)
    test_piaa_imgsort_dataloader = DataLoader(test_piaa_imgsort_dataset, batch_size=5, shuffle=False, num_workers=args.num_workers, timeout=300, collate_fn=collate_fn_imgsort)
    dataloaders = (train_dataloader, val_giaa_dataloader, val_piaa_imgsort_dataloader, test_giaa_dataloader, test_piaa_imgsort_dataloader)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the combined model
    model = NIMA(num_bins).to(device)

    if args.resume is not None:
        model.load_state_dict(torch.load(args.resume))
    # Loss and optimizer
    # criterion_mse = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Initialize the best test loss and the best model
    best_modelname = f'lapis_best_model_resnet50_nima_{experiment_name}.pth'
    dirname = model_dir(args)
    best_modelname = os.path.join(dirname, best_modelname)
    
    trainer(dataloaders, model, optimizer, args, train, evaluate, device, best_modelname)