import os
import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from LAPIS_PIAA_dataloader import load_data, collate_fn
import wandb
from scipy.stats import spearmanr
from train_piaa_mir import CombinedModel, SimplePerModel, trainer
from utils.argflags import parse_arguments_piaa


def train(model, dataloader, criterion_mse, optimizer, device):
    model.train()
    running_mse_loss = 0.0

    progress_bar = tqdm(dataloader, leave=False)
    for sample in progress_bar:
        images = sample['image'].to(device)
        sample_pt = sample['traits'].float().to(device)
        sample_score = sample['response'].float().to(device) / 20.
        
        score_pred = model(images, sample_pt)
        # loss = criterion_mse(score_pred, sample_score)
        loss = criterion_mse(score_pred / 5., sample_score / 5.)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_mse_loss += loss.item()

        progress_bar.set_postfix({
            'Train MSE Mean Loss': loss.item(),
        })

    epoch_mse_loss = running_mse_loss / len(dataloader)
    return epoch_mse_loss


def evaluate(model, dataloader, criterion_mse, device):
    model.eval()  # Set the model to evaluation mode
    running_mse_loss = 0.0

    all_predicted_scores = []  # List to store all predictions
    all_true_scores = []  # List to store all true scores

    progress_bar = tqdm(dataloader, leave=False)
    for sample in progress_bar:
        with torch.no_grad():
            images = sample['image'].to(device)
            sample_pt = sample['traits'].float().to(device)
            sample_score = sample['response'].float().to(device) / 20.
            
            # MSE loss
            score_pred = model(images, sample_pt)
            # loss = criterion_mse(score_pred, sample_score)
            loss = criterion_mse(score_pred / 5., sample_score / 5.)
            running_mse_loss += loss.item()

            # Store predicted and true scores for SROCC calculation
            predicted_scores = score_pred.view(-1).cpu().numpy()
            true_scores = sample_score.view(-1).cpu().numpy()
            all_predicted_scores.extend(predicted_scores)
            all_true_scores.extend(true_scores)

            progress_bar.set_postfix({'Test MSE Mean Loss': loss.item()})

    epoch_mse_loss = running_mse_loss / len(dataloader)

    # Calculate SROCC for all predictions
    srocc, _ = spearmanr(all_predicted_scores, all_true_scores)

    return epoch_mse_loss, srocc


criterion_mse = nn.MSELoss()


if __name__ == '__main__':
    args = parse_arguments_piaa()
    
    batch_size = args.batch_size
    n_workers = 8
    num_bins = 9
    num_attr = 8
    num_pt = 71
    
    if args.is_log:
        tags = ["no_attr","PIAA",
                f"learning_rate: {args.lr}",
                f"batch_size: {args.batch_size}",
                f"num_epochs: {args.num_epochs}"]
        if args.use_cv:
            tags += ["CV%d/%d"%(args.fold_id, args.n_fold)]
        if args.dropout is not None:
            tags += [f"dropout={args.dropout}"]            
        wandb.init(project="resnet_LAVIS_PIAA",
                notes="PIAA-MIR",
                tags = tags)
        
        experiment_name = wandb.run.name
    else:
        experiment_name = ''
    
    # Create dataloaders for training and test sets
    train_dataset, val_dataset, test_dataset = load_data(args)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers, timeout=300, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers, timeout=300, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers, timeout=300, collate_fn=collate_fn)
    dataloaders = (train_dataloader, val_dataloader, test_dataloader)
    
    # Define the number of classes in your dataset
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CombinedModel(num_bins, num_attr, num_pt, dropout=args.dropout)
    # model = SimplePerModel(num_bins, num_attr, num_pt)
    
    model.nima_attr.load_state_dict(torch.load(args.pretrained_model))
    if args.resume:
        model.load_state_dict(torch.load(args.resume))
    model = model.to(device)
    
    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Initialize the best test loss and the best model
    best_model = None
    best_modelname = 'lapis_best_model_resnet50_piaamir_lr%1.0e_decay_%depoch' % (args.lr, args.num_epochs)
    best_modelname += '_%s'%experiment_name
    best_modelname += '.pth'
    dirname = 'models_pth'
    if args.use_cv:
        dirname = os.path.join(dirname, 'random_cvs')
    best_modelname = os.path.join(dirname, best_modelname)

    # Training loop
    trainer(dataloaders, model, optimizer, args, train, evaluate, device, best_modelname)
