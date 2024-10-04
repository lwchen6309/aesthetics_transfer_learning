# import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
# from torchvision import transforms
from torchvision.models import resnet50
# import numpy as np
# from tqdm import tqdm
import wandb
# from scipy.stats import spearmanr
from PARA_histogram_dataloader import load_data, collate_fn_imgsort
# from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA

from utils.argflags import parse_arguments, wandb_tags, model_dir
from utils.losses import EarthMoverDistance
earth_mover_distance = EarthMoverDistance()
from train_histonet_latefusion import num_bins, num_attr, num_bins_attr, num_pt
from train_histonet_latefusion import trainer, train, evaluate, evaluate_with_prior, evaluate_each_datum


class CrossAttn(nn.Module):
    def __init__(self, num_bins_aesthetic, num_attr, num_bins_attr, num_pt, dropout=None, num_heads=8):
        super(CrossAttn, self).__init__()
        self.resnet = resnet50(pretrained=True)
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 512),
            nn.ReLU(),
        )
        self.num_bins_aesthetic = num_bins_aesthetic
        self.num_attr = num_attr
        self.num_bins_attr = num_bins_attr
        self.num_pt = num_pt
        
        # For predicting attribute histograms for each attribute
        pt_encoder_layers = []
        if dropout is not None:
            pt_encoder_layers.append(nn.Dropout(dropout))
        pt_encoder_layers.extend([
            nn.Linear(num_pt, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512)
        ])
        self.pt_encoder = nn.Sequential(*pt_encoder_layers)
        
        # Cross attention layer
        self.cross_attention = nn.MultiheadAttention(embed_dim=512, num_heads=num_heads)
        
        # For predicting aesthetic score histogram
        self.fc_aesthetic = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_bins_aesthetic)
        )

    def forward(self, images, traits_histogram):
        x = self.resnet(images)  # ResNet output, shape [batch_size, 512]
        pt_code = self.pt_encoder(traits_histogram)  # Encoded traits, shape [batch_size, 512]
        
        # Reshape x and pt_code for multi-head attention
        x = x.unsqueeze(0)  # Shape [1, batch_size, 512]
        pt_code = pt_code.unsqueeze(0)  # Shape [1, batch_size, 512]
        
        # Apply cross-attention: pt_code as query, x as key and value
        attn_output, _ = self.cross_attention(pt_code, x, x)
        
        attn_output = attn_output.squeeze(0)  # Shape [batch_size, 512]
        
        aesthetic_logits = self.fc_aesthetic(attn_output)
        return aesthetic_logits


if __name__ == '__main__':    
    args = parse_arguments()
    batch_size = args.batch_size
    
    if args.is_log:
        tags = wandb_tags(args)

        wandb.init(project="resnet_PARA_PIAA", 
                   notes="crossattn",
                   tags=tags)
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
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, timeout=300)
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, timeout=300, collate_fn=collate_fn_imgsort)
    val_giaa_dataloader = DataLoader(val_giaa_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, timeout=300)
    val_piaa_imgsort_dataloader = DataLoader(val_piaa_imgsort_dataset, batch_size=5, shuffle=False, num_workers=args.num_workers, timeout=300, collate_fn=collate_fn_imgsort)
    test_giaa_dataloader = DataLoader(test_giaa_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, timeout=300)
    test_piaa_imgsort_dataloader = DataLoader(test_piaa_imgsort_dataset, batch_size=5, shuffle=False, num_workers=args.num_workers, timeout=300, collate_fn=collate_fn_imgsort)
    dataloaders = (train_dataloader, val_giaa_dataloader, val_piaa_imgsort_dataloader, test_giaa_dataloader, test_piaa_imgsort_dataloader)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize the combined model
    model = CrossAttn(num_bins, num_attr, num_bins_attr, num_pt, args.dropout).to(device)

    if args.resume is not None:
        model.load_state_dict(torch.load(args.resume))
    # Loss and optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Initialize the best test loss and the best model
    best_modelname = f'best_model_resnet50_histo_latefusion_{experiment_name}.pth'
    dirname = model_dir(args)
    best_modelname = os.path.join(dirname, best_modelname)

    trainer(dataloaders, model, optimizer, args, train, (evaluate, evaluate_with_prior), device, best_modelname)
    emd_loss, emd_attr_loss, srocc, mse_loss = evaluate_each_datum(model, test_piaa_imgsort_dataloader, device)