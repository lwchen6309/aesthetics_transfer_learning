import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
# from torchvision import transforms
from torchvision.models import resnet50, mobilenet_v2, resnet18, swin_v2_t, swin_v2_s
import numpy as np
# import pandas as pd
from tqdm import tqdm
import wandb
from scipy.stats import spearmanr
from LAPIS_histogram_dataloader import load_data, collate_fn_imgsort, collate_fn
from utils.argflags import parse_arguments, wandb_tags, model_dir
from utils.losses import EarthMoverDistance
earth_mover_distance = EarthMoverDistance()


class NIMA(nn.Module):
    def __init__(self, num_bins_aesthetic, model_name='resnet50'):
        super(NIMA, self).__init__()
        
        # Choose model backbone based on the input argument
        if model_name == 'resnet50':
            self.backbone = resnet50(pretrained=True)
            backbone_out_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove the last fully connected layer
        
        elif model_name == 'mobilenet_v2':
            self.backbone = mobilenet_v2(pretrained=True)
            backbone_out_features = self.backbone.last_channel
            self.backbone.classifier = nn.Identity()  # Remove the classifier

        elif model_name == 'resnet18':
            self.backbone = resnet18(pretrained=True)
            backbone_out_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove the last fully connected layer

        elif model_name == 'swin_v2_t':
            self.backbone = swin_v2_t(pretrained=True)
            backbone_out_features = self.backbone.head.in_features
            self.backbone.head = nn.Identity()  # Remove the head

        elif model_name == 'swin_v2_s':
            self.backbone = swin_v2_s(pretrained=True)
            backbone_out_features = self.backbone.head.in_features
            self.backbone.head = nn.Identity()  # Remove the head

        else:
            raise ValueError(f"Model {model_name} is not supported.")
        print(model_name)
        # Final fully connected layers for aesthetic score prediction
        self.fc_aesthetic = nn.Sequential(
            nn.Linear(backbone_out_features, 512),
            nn.ReLU(),
            nn.Linear(512, num_bins_aesthetic)
        )

    def forward(self, images):
        x = self.backbone(images)  # Pass images through the selected backbone
        aesthetic_logits = self.fc_aesthetic(x)
        return aesthetic_logits
    
def train(model, dataloader, optimizer, device):
    model.train()
    running_aesthetic_emd_loss = 0.0
    running_total_emd_loss = 0.0
    progress_bar = tqdm(dataloader, leave=False)
    units_len = dataloader.dataset.traits_len()
    for sample in progress_bar:        
        images = sample['image'].to(device)
        aesthetic_score_histogram = sample['aestheticScore'].to(device)
        traits_histogram = sample['traits'].to(device)
        art_type = sample['art_type'].to(device)
        # traits_histogram = torch.cat([traits_histogram, art_type], dim=1)
        
        optimizer.zero_grad()
        aesthetic_logits = model(images)

        prob_aesthetic = F.softmax(aesthetic_logits, dim=1)
        loss_aesthetic = earth_mover_distance(prob_aesthetic, aesthetic_score_histogram).mean()
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

# Evaluation Function
def evaluate(model, dataloader, device):
    model.eval()
    running_emd_loss = 0.0
    running_attr_emd_loss = 0.0
    running_mse_loss = 0.0
    # scale = torch.arange(1, 5.5, 0.5).to(device)
    scale = torch.arange(0, 10).to(device)
    
    progress_bar = tqdm(dataloader, leave=False)
    mean_pred = []
    mean_target = []
    with torch.no_grad():
        for sample in progress_bar:
            images = sample['image'].to(device)
            aesthetic_score_histogram = sample['aestheticScore'].to(device)
            traits_histogram = sample['traits'].to(device)
            art_type = sample['art_type'].to(device)

            aesthetic_logits = model(images)
            prob_aesthetic = F.softmax(aesthetic_logits, dim=1)
            loss = earth_mover_distance(prob_aesthetic, aesthetic_score_histogram).mean()
            
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


def trainer(dataloaders, model, optimizer, args, train_fn, evaluate_fn, device, best_modelname):

    train_dataloader, val_giaa_dataloader, val_piaa_imgsort_dataloader, test_giaa_dataloader, test_piaa_imgsort_dataloader = dataloaders

    # Training loop
    best_test_srocc = 0
    num_patience_epochs = 0
    for epoch in range(args.num_epochs):
        if args.is_eval:
            break
        # Learning rate schedule
        if (epoch + 1) % args.lr_schedule_epochs == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= args.lr_decay_factor

        # Training
        train_emd_loss, train_total_emd_loss = train_fn(model, train_dataloader, optimizer, device)
        if args.is_log:
            wandb.log({"Train EMD Loss": train_emd_loss,
                       "Train Total EMD Loss": train_total_emd_loss,
                       }, commit=False)
        
        # Testing
        val_giaa_emd_loss, _, val_giaa_srocc, _ = evaluate_fn(model, val_giaa_dataloader, device)
        val_piaa_emd_loss, _, val_piaa_srocc, _ = evaluate_fn(model, val_piaa_imgsort_dataloader, device)
        if args.is_log:
            wandb.log({
                "Val GIAA EMD Loss": val_giaa_emd_loss,
                "Val GIAA SROCC": val_giaa_srocc,
                "Val PIAA EMD Loss": val_piaa_emd_loss,
                "Val PIAA SROCC": val_piaa_srocc,
            }, commit=True)
        
        eval_srocc = val_piaa_srocc if args.eval_on_piaa else val_giaa_srocc
        
        # Early stopping check
        if eval_srocc > best_test_srocc:
            best_test_srocc = eval_srocc
            num_patience_epochs = 0
            torch.save(model.state_dict(), best_modelname)
        else:
            num_patience_epochs += 1
            if num_patience_epochs >= args.max_patience_epochs:
                print("Validation loss has not decreased for {} epochs. Stopping training.".format(args.max_patience_epochs))
                break
    
    if not args.is_eval:
        model.load_state_dict(torch.load(best_modelname))   
    
    # Testing
    test_giaa_emd_loss, _, test_giaa_srocc, test_giaa_mse = evaluate_fn(model, test_giaa_dataloader, device)
    test_piaa_emd_loss, _, test_piaa_srocc, test_piaa_mse = evaluate_fn(model, test_piaa_imgsort_dataloader, device)
    
    if args.is_log:
        wandb.log({
            "Test GIAA EMD Loss": test_giaa_emd_loss,
            "Test GIAA SROCC": test_giaa_srocc,
            "Test GIAA MSE": test_giaa_mse,
            "Test PIAA EMD Loss": test_piaa_emd_loss,
            "Test PIAA SROCC": test_piaa_srocc,
            "Test PIAA MSE": test_piaa_mse,
        }, commit=True)

    # Print the epoch loss
    print(f"Test GIAA SROCC Loss: {test_giaa_srocc:.4f}, "
        f"Test PIAA SROCC Loss: {test_piaa_srocc:.4f}, "
        )


num_bins = 10
num_attr = 8
num_bins_attr = 5
num_pt = 137
criterion_mse = nn.MSELoss()


if __name__ == '__main__':    
    parser = parse_arguments(False)
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['resnet50', 'mobilenet_v2', 'inception_v3', 'resnet18', 'swin_v2_t', 'swin_v2_s'], 
                    help="Choose the model backbone from: 'resnet50', 'mobilenet_v2', 'resnet18', 'swin_v2_t', 'swin_v2_s'")
    args = parser.parse_args()
    batch_size = args.batch_size
    
    if args.is_log:
        tags = ["no_attr","GIAA"]
        tags += wandb_tags(args)
        wandb.init(project="LAPIS_IAA", 
                   notes='NIMA-'+args.backbone,
                   tags = tags,
                   entity='KULeuven-GRAPPA',
                   )
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
    model = NIMA(num_bins, model_name=args.backbone).to(device)

    if args.resume is not None:
        model.load_state_dict(torch.load(args.resume))
    # Loss and optimizer
    # criterion_mse = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # Initialize the best test loss and the best model
    best_modelname = f'lapis_{args.backbone}_nima_{experiment_name}.pth'
    dirname = model_dir(args)
    best_modelname = os.path.join(dirname, best_modelname)
    
    trainer(dataloaders, model, optimizer, args, train, evaluate, device, best_modelname)