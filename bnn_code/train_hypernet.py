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
from tqdm import tqdm
import wandb
from scipy.stats import spearmanr
from PARA_histogram_dataloader import load_data, collate_fn_imgsort
from train_histonet_latefusion import earth_mover_distance, train, evaluate, evaluate_with_prior, evaluate_trait, save_results, evaluate_each_datum, trainer


class HyperNetCombinedModel(nn.Module):
    def __init__(self, num_bins_aesthetic, num_attr, num_bins_attr, num_pt):
        super(HyperNetCombinedModel, self).__init__()
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
        self.pt_encoder = nn.Sequential(
            nn.Linear(num_pt, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512)
        )
        
        # Hypernetwork to generate parameters for fc_aesthetic
        self.hidden_size = 512
        self.hypernet_fc_aesthetic_w1 = nn.Linear(512, 512 * self.hidden_size)
        self.hypernet_fc_aesthetic_b1 = nn.Linear(512, self.hidden_size)
        self.hypernet_fc_aesthetic_w2 = nn.Linear(512, self.hidden_size * num_bins_aesthetic)
        self.hypernet_fc_aesthetic_b2 = nn.Linear(512, num_bins_aesthetic)
        
    def forward(self, images, traits_histogram):
        # Pass images through ResNet
        x = self.resnet(images)
        
        # Encode traits_histogram
        pt_code = self.pt_encoder(traits_histogram)
        # Generate fc_aesthetic weights and biases
        batch_size = pt_code.size(0)
        w1 = self.hypernet_fc_aesthetic_w1(pt_code).view(batch_size, self.hidden_size, 512)
        b1 = self.hypernet_fc_aesthetic_b1(pt_code).view(batch_size, self.hidden_size)
        w2 = self.hypernet_fc_aesthetic_w2(pt_code).view(batch_size, self.num_bins_aesthetic, self.hidden_size)
        b2 = self.hypernet_fc_aesthetic_b2(pt_code).view(batch_size, self.num_bins_aesthetic)
        
        # Pass x through dynamically generated fc_aesthetic
        xz = torch.stack([F.relu(F.linear(x[i], w1[i], b1[i])) for i in range(batch_size)])
        aesthetic_logits = torch.stack([F.linear(xz[i], w2[i], b2[i]) for i in range(batch_size)])

        return aesthetic_logits


num_bins = 9
num_attr = 8
num_bins_attr = 5
num_pt = 50 + 20
criterion_mse = nn.MSELoss()


if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description='Training and Testing the Combined Model for data spliting')
    parser.add_argument('--fold_id', type=int, default=1)
    parser.add_argument('--n_fold', type=int, default=4)
    parser.add_argument('--trainset', type=str, default='GIAA', choices=["GIAA", "sGIAA", "PIAA"])
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--importance_sampling', action='store_true', help='Enable importance sampling for uniform score distribution')
    parser.add_argument('--use_cv', action='store_true', help='Enable cross validation')
    parser.add_argument('--is_eval', action='store_true', help='Enable evaluation mode')
    parser.add_argument('--eval_on_piaa', action='store_true', help='Evaluation metric on PIAA')
    parser.add_argument('--no_log', action='store_false', dest='is_log', help='Disable logging')
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--max_patience_epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--lr_schedule_epochs', type=int, default=5)
    parser.add_argument('--lr_decay_factor', type=float, default=0.5)    
    args = parser.parse_args()
    
    batch_size = args.batch_size
    random_seed = 42
    n_workers = 8
    args.eval_on_piaa = True if args.trainset == 'PIAA' else False

    if args.is_log:
        tags = ["no_attr", args.trainset]
        if args.use_cv:
            tags += ["CV%d/%d"%(args.fold_id, args.n_fold)]
        wandb.init(project="resnet_PARA_PIAA", 
                   notes="hypernet",
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
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers, timeout=300)
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers, timeout=300, collate_fn=collate_fn_imgsort)
    val_giaa_dataloader = DataLoader(val_giaa_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers, timeout=300)
    val_piaa_imgsort_dataloader = DataLoader(val_piaa_imgsort_dataset, batch_size=5, shuffle=False, num_workers=n_workers, timeout=300, collate_fn=collate_fn_imgsort)
    test_giaa_dataloader = DataLoader(test_giaa_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers, timeout=300)
    test_piaa_imgsort_dataloader = DataLoader(test_piaa_imgsort_dataset, batch_size=5, shuffle=False, num_workers=n_workers, timeout=300, collate_fn=collate_fn_imgsort)
    dataloaders = (train_dataloader, val_giaa_dataloader, val_piaa_imgsort_dataloader, test_giaa_dataloader, test_piaa_imgsort_dataloader)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the combined model
    model = HyperNetCombinedModel(num_bins, num_attr, num_bins_attr, num_pt).to(device)

    if args.resume is not None:
        model.load_state_dict(torch.load(args.resume))
    # Loss and optimizer
    # criterion_mse = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Initialize the best test loss and the best model
    best_model = None
    best_modelname = 'best_model_resnet50_histo_latefusion_lr%1.0e_decay_%depoch' % (args.lr, args.num_epochs)
    best_modelname += '_%s'%experiment_name
    best_modelname += '.pth'
    dirname = 'models_pth'
    if args.use_cv:
        dirname = os.path.join(dirname, 'random_cvs')
    best_modelname = os.path.join(dirname, best_modelname)
    
    trainer(dataloaders, model, optimizer, args, train, (evaluate, evaluate_with_prior), device, best_modelname)
