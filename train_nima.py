import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
# from torchvision import transforms
from torchvision.models import resnet50
import numpy as np
from tqdm import tqdm
import wandb
from scipy.stats import spearmanr, pearsonr
from PARA_histogram_dataloader import load_data, collate_fn_imgsort
# import matplotlib.pyplot as plt
# import pandas as pd
from utils.losses import EarthMoverDistance
earth_mover_distance = EarthMoverDistance()
from utils.argflags import parse_arguments, wandb_tags, model_dir
from timm import create_model
from torchvision.models import resnet50


class NIMA(nn.Module):
    def __init__(self, num_bins_aesthetic, backbone="resnet50", pretrained=True):
        super(NIMA, self).__init__()

        # Load the specified backbone (ResNet-50, ViT-Small, or Swin-Tiny)
        if backbone == "resnet50":
            self.backbone = resnet50(pretrained=pretrained)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove the classification head
        elif backbone == "vit_small_patch16_224":
            self.backbone = create_model("vit_small_patch16_224", pretrained=pretrained, num_classes=0)
            feature_dim = self.backbone.embed_dim
        elif backbone == "swin_tiny_patch4_window7_224":
            self.backbone = create_model("swin_tiny_patch4_window7_224", pretrained=pretrained, num_classes=0)
            feature_dim = self.backbone.num_features
        elif backbone == "swin_base_patch4_window7_224":
            self.backbone = create_model("swin_base_patch4_window7_224", pretrained=pretrained, num_classes=0)
            feature_dim = self.backbone.num_features
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        print('backbone: ', backbone)

        self.num_bins_aesthetic = num_bins_aesthetic

        # Fully connected layers for predicting the aesthetic score histogram
        self.fc_aesthetic = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_bins_aesthetic)
        )

    def forward(self, images, traits_histogram=None):
        # Extract features using the backbone
        x = self.backbone(images)
        
        # Predict aesthetic logits
        aesthetic_logits = self.fc_aesthetic(x)
        return aesthetic_logits


# class NIMA(nn.Module):
#     def __init__(self, num_bins_aesthetic):
#         super(NIMA, self).__init__()
#         self.resnet = resnet50(pretrained=True)
#         self.resnet.fc = nn.Sequential(
#             nn.Linear(self.resnet.fc.in_features, 512),
#             nn.ReLU(),
#         )
#         self.num_bins_aesthetic = num_bins_aesthetic
        
#         # For predicting aesthetic score histogram
#         self.fc_aesthetic = nn.Sequential(
#             nn.Linear(512, 512),
#             nn.ReLU(),
#             nn.Linear(512, num_bins_aesthetic)
#         )

#     def forward(self, images, traits_histogram):
#         # traits_histogram is dummy variable
#         x = self.resnet(images)
#         aesthetic_logits = self.fc_aesthetic(x)
#         return aesthetic_logits


# Training Function
def train(model, dataloader, optimizer, device):
    model.train()
    running_aesthetic_emd_loss = 0.0
    running_total_emd_loss = 0.0
    progress_bar = tqdm(dataloader, leave=False)
    # scale_aesthetic = torch.arange(1, 5.5, 0.5).to(device)
    for sample in progress_bar:
        images = sample['image'].to(device)
        aesthetic_score_histogram = sample['aestheticScore'].to(device)
        traits_histogram = sample['traits'].to(device)
        onehot_big5 = sample['big5'].to(device)
        # attributes_target_histogram = sample['attributes'].to(device).view(-1, num_attr, num_bins_attr) # Reshape to match our logits shape
        total_traits_histogram = torch.cat([traits_histogram, onehot_big5], dim=1)
        # traits_histogram = traits_histogram[:,:5]

        optimizer.zero_grad()
        aesthetic_logits = model(images, total_traits_histogram)
        prob_aesthetic = F.softmax(aesthetic_logits, dim=1)
        # prob_attribute = F.softmax(attribute_logits, dim=-1) # Softmax along the bins dimension
        loss_aesthetic = earth_mover_distance(prob_aesthetic, aesthetic_score_histogram).mean()
        # loss_attribute = torch.mean(coef[:,None] * earth_mover_distance(prob_attribute, attributes_target_histogram))
        # total_loss = loss_aesthetic + loss_attribute # Combining losses
        total_loss = loss_aesthetic # Combining losses
        
        total_loss.backward()
        optimizer.step()
        running_total_emd_loss += total_loss.item()
        running_aesthetic_emd_loss += loss_aesthetic.item()

        progress_bar.set_postfix({
            'Train EMD Loss': total_loss.item(),
        })
    
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


def evaluate_save_rn(model, dataloader, device):
    model.eval()
    running_emd_loss = 0.0
    running_mse_loss = 0.0
    scale = torch.arange(1, 5.5, 0.5).to(device)
    eval_srocc = True

    progress_bar = tqdm(dataloader, leave=False)
    mean_pred = []
    mean_target = []
    data_storage = {}

    with torch.no_grad():
        for sample in progress_bar:
            images = sample['image'].to(device)
            aesthetic_score_histogram = sample['aestheticScore'].to(device)
            traits_histogram = sample['traits'].to(device)
            onehot_big5 = sample['big5'].to(device)
            traits_histogram = torch.cat([traits_histogram, onehot_big5], dim=1)
            
            aesthetic_logits = model(images, traits_histogram)
            prob_aesthetic = F.softmax(aesthetic_logits, dim=1)
            loss = earth_mover_distance(prob_aesthetic, aesthetic_score_histogram).mean()
            rn_feature = model.resnet(images).cpu().numpy().astype(np.float16)
            img_path = [os.path.split(pth)[-1] for pth in sample['image_path']]

            # Store features and paths in a dictionary
            for path, feature in zip(img_path, rn_feature):
                data_storage[path] = feature

            if eval_srocc:
                outputs_mean = torch.sum(prob_aesthetic * scale, dim=1, keepdim=True)
                target_mean = torch.sum(aesthetic_score_histogram * scale, dim=1, keepdim=True)
                mean_pred.append(outputs_mean.view(-1).cpu().numpy())
                mean_target.append(target_mean.view(-1).cpu().numpy())
                mse = criterion_mse(outputs_mean, target_mean)
                running_mse_loss += mse.item()
            
            running_emd_loss += loss.item()
            progress_bar.set_postfix({
                'Test EMD Loss': loss.item(),
            })

    if eval_srocc:
        predicted_scores = np.concatenate(mean_pred)
        true_scores = np.concatenate(mean_target)
        srocc, _ = spearmanr(predicted_scores, true_scores)
        mse_loss = running_mse_loss / len(dataloader)
    else:
        srocc = None
        mse_loss = None

    emd_loss = running_emd_loss / len(dataloader)
    
    return emd_loss, srocc, mse_loss, data_storage



# Evaluation Function
def evaluate(model, dataloader, device):
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
            loss = earth_mover_distance(prob_aesthetic, aesthetic_score_histogram).mean()
            
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
    plcc, _ = pearsonr(predicted_scores, true_scores)

    emd_loss = running_emd_loss / len(dataloader)
    # emd_attr_loss = running_attr_emd_loss / len(dataloader)
    mse_loss = running_mse_loss / len(dataloader)
    return emd_loss, srocc, plcc, mse_loss

# Evaluation Function
def evaluate_each_datum(model, dataloader, device):
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
            # attributes_target_histogram = sample['attributes'].to(device).view(-1, num_attr, num_bins_attr) # Reshape to match our logits shape
            traits_histogram = torch.cat([traits_histogram, onehot_big5], dim=1)
            traits_histograms.append(traits_histogram.cpu().numpy())

            aesthetic_logits = model(images, traits_histogram)
            prob_aesthetic = F.softmax(aesthetic_logits, dim=1)
            rn_feature = model.resnet(images).cpu().numpy()
            
            emd_loss_datum = earth_mover_distance(prob_aesthetic, aesthetic_score_histogram)
            emd_loss_data.append(emd_loss_datum.view(-1).cpu().numpy())
            loss = earth_mover_distance(prob_aesthetic, aesthetic_score_histogram).mean()
            
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
    plcc, _ = pearsonr(predicted_scores, true_scores)

    traits_histograms = np.concatenate(traits_histograms)
    emd_loss_data = np.concatenate(emd_loss_data)
    save_results(dataloader.dataset, userIds, traits_histograms, emd_loss_data, predicted_scores, true_scores)
    
    emd_loss = running_emd_loss / len(dataloader)
    # emd_attr_loss = running_attr_emd_loss / len(dataloader)
    mse_loss = running_mse_loss / len(dataloader)
    return emd_loss, srocc, plcc, mse_loss


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
        val_giaa_emd_loss, val_giaa_srocc, val_giaa_plcc, _ = evaluate_fn(model, val_giaa_dataloader, device)
        val_piaa_emd_loss, val_piaa_srocc, val_piaa_plcc, _ = evaluate_fn(model, val_piaa_imgsort_dataloader, device)
        if args.is_log:
            wandb.log({
                "Val GIAA EMD Loss": val_giaa_emd_loss,
                "Val GIAA SROCC": val_giaa_srocc,
                "Val GIAA PLCC": val_giaa_plcc,
                "Val PIAA EMD Loss": val_piaa_emd_loss,
                "Val PIAA SROCC": val_piaa_srocc,
                "Val PIAA PLCC": val_piaa_plcc,
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
    test_giaa_emd_loss, test_giaa_srocc, test_giaa_plcc, test_giaa_mse = evaluate_fn(model, test_giaa_dataloader, device)
    test_piaa_emd_loss, test_piaa_srocc, test_piaa_plcc, test_piaa_mse = evaluate_fn(model, test_piaa_imgsort_dataloader, device)
    
    if args.is_log:
        wandb.log({
            "Test GIAA EMD Loss": test_giaa_emd_loss,
            "Test GIAA SROCC": test_giaa_srocc,
            "Test GIAA PLCC": test_giaa_plcc,
            "Test GIAA MSE": test_giaa_mse,
            "Test PIAA EMD Loss": test_piaa_emd_loss,
            "Test PIAA SROCC": test_piaa_srocc,
            "Test PIAA PLCC": test_piaa_plcc,
            "Test PIAA MSE": test_piaa_mse,
        }, commit=True)

    # Print the epoch loss
    print(f"Test GIAA SROCC Loss: {test_giaa_srocc:.4f}, "
        f"Test PIAA SROCC Loss: {test_piaa_srocc:.4f}, "
        )


num_bins = 9
num_attr = 8
num_bins_attr = 5
num_pt = 50 + 20
criterion_mse = nn.MSELoss()


if __name__ == '__main__':    
    args = parse_arguments()
    batch_size = args.batch_size
    num_workers = args.num_workers
    
    if args.is_log:
        tags = ["no_attr", args.trainset]
        tags += wandb_tags(args)
        wandb.init(project="resnet_PARA_PIAA", 
                   notes=f"NIMA-{args.backbone}",
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
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, timeout=300)
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers, timeout=300, collate_fn=collate_fn_imgsort)
    val_giaa_dataloader = DataLoader(val_giaa_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, timeout=300)
    val_piaa_imgsort_dataloader = DataLoader(val_piaa_imgsort_dataset, batch_size=5, shuffle=False, num_workers=num_workers, timeout=300, collate_fn=collate_fn_imgsort)
    test_giaa_dataloader = DataLoader(test_giaa_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, timeout=300)
    test_piaa_imgsort_dataloader = DataLoader(test_piaa_imgsort_dataset, batch_size=5, shuffle=False, num_workers=num_workers, timeout=300, collate_fn=collate_fn_imgsort)
    dataloaders = (train_dataloader, val_giaa_dataloader, val_piaa_imgsort_dataloader, test_giaa_dataloader, test_piaa_imgsort_dataloader)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the combined model
    model = NIMA(num_bins, backbone=args.backbone).to(device)
    
    if args.resume is not None:
        model.load_state_dict(torch.load(args.resume))

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Initialize the best test loss and the best model
    best_modelname = f'best_model_{args.backbone}_nima_{experiment_name}.pth'
    dirname = model_dir(args)
    best_modelname = os.path.join(dirname, best_modelname)
    
    trainer(dataloaders, model, optimizer, args, train, evaluate, device, best_modelname)