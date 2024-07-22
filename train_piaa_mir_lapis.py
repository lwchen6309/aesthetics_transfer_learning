import os
import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
# from LAPIS_PIAA_dataloader import load_data #, collate_fn
from LAPIS_histogram_dataloader import load_data, collate_fn, collate_fn_imgsort
import wandb
from scipy.stats import spearmanr
from train_piaa_mir import PIAA_MIR, PIAA_MIR_Exp, CrossAttn_MIR, CrossAttn_MIR_Exp, SelfAttn_MIR, trainer
from utils.argflags import parse_arguments_piaa, wandb_tags, model_dir


def train_piaa(model, dataloader, criterion_mse, optimizer, device):
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


def evaluate_piaa(model, dataloader, criterion_mse, device):
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


def train(model, dataloader, criterion_mse, optimizer, device):
    model.train()
    running_mse_loss = 0.0
    scale = torch.arange(0, 10).to(device)

    progress_bar = tqdm(dataloader, leave=False)
    for sample in progress_bar:
        images = sample['image'].to(device)
        sample_pt = sample['traits'].float().to(device)
        # sample_score = sample['response'].float().to(device) / 20.
        aesthetic_score_histogram = sample['aestheticScore'].to(device)
        sample_score = torch.sum(aesthetic_score_histogram * scale, dim=1, keepdim=True) / 2.
        score_pred = model(images, sample_pt)
        loss = criterion_mse(score_pred, sample_score)
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
    scale = torch.arange(0, 10).to(device)
    all_predicted_scores = []  # List to store all predictions
    all_true_scores = []  # List to store all true scores

    progress_bar = tqdm(dataloader, leave=False)
    for sample in progress_bar:
        with torch.no_grad():
            images = sample['image'].to(device)
            sample_pt = sample['traits'].float().to(device)
            # sample_score = sample['response'].float().to(device) / 20.
            aesthetic_score_histogram = sample['aestheticScore'].to(device)
            sample_score = torch.sum(aesthetic_score_histogram * scale, dim=1, keepdim=True) / 2.

            # MSE loss
            score_pred = model(images, sample_pt)
            loss = criterion_mse(score_pred, sample_score)
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


def evaluate_with_prior(model, dataloader, prior_dataloader, criterion_mse, device):
    model.eval()  # Set the model to evaluation mode
    running_mse_loss = 0.0
    scale = torch.arange(0, 10).to(device)
    all_predicted_scores = []  # List to store all predictions
    all_true_scores = []  # List to store all true scores

    progress_bar = tqdm(dataloader, leave=False)
    with torch.no_grad():
        traits_histograms = []
        for sample in tqdm(prior_dataloader, leave=False):
            traits_histogram = sample['traits'].to(device)
            traits_histograms.append(traits_histogram)
        mean_traits_histogram = torch.mean(torch.cat(traits_histograms, dim=0), dim=0).unsqueeze(0)

        for sample in progress_bar:
            images = sample['image'].to(device)
            # sample_pt = sample['traits'].float().to(device)
            sample_pt = mean_traits_histogram.repeat(images.shape[0], 1)
            # sample_score = sample['response'].float().to(device) / 20.
            aesthetic_score_histogram = sample['aestheticScore'].to(device)
            sample_score = torch.sum(aesthetic_score_histogram * scale, dim=1, keepdim=True) / 2.

            # MSE loss
            score_pred = model(images, sample_pt)
            loss = criterion_mse(score_pred, sample_score)
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
    parser = parse_arguments_piaa(False)
    parser.add_argument('--model', type=str, default='PIAA-MIR')
    args = parser.parse_args()
    print(args)
    
    num_bins = 9
    num_attr = 8
    # num_pt = 71
    num_pt = 137
    
    if args.is_log:
        tags = ["no_attr"]
        tags += wandb_tags(args)
        wandb.init(project="resnet_LAVIS_PIAA",
                notes=args.model,
                tags = tags)
        experiment_name = wandb.run.name
    else:
        experiment_name = ''
    
    # Create dataloaders for training and test sets
    train_dataset, val_giaa_dataset, val_piaa_imgsort_dataset, test_giaa_dataset, test_piaa_imgsort_dataset = load_data(args)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, timeout=300, collate_fn=collate_fn)
    val_giaa_dataloader = DataLoader(val_giaa_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, timeout=300, collate_fn=collate_fn)
    val_piaa_imgsort_dataloader = DataLoader(val_piaa_imgsort_dataset, batch_size=5, shuffle=False, num_workers=args.num_workers, timeout=300, collate_fn=collate_fn_imgsort)
    test_giaa_dataloader = DataLoader(test_giaa_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, timeout=300, collate_fn=collate_fn)
    test_piaa_imgsort_dataloader = DataLoader(test_piaa_imgsort_dataset, batch_size=5, shuffle=False, num_workers=args.num_workers, timeout=300, collate_fn=collate_fn_imgsort)
    dataloaders = (train_dataloader, val_giaa_dataloader, val_piaa_imgsort_dataloader, test_giaa_dataloader, test_piaa_imgsort_dataloader)

    # Define the number of classes in your dataset
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.model == 'CrossAttn':
        model = CrossAttn_MIR(num_bins, num_attr, num_pt, dropout=args.dropout, num_heads=num_pt).to(device)
        best_modelname = f'best_model_resnet50_crossattn_mir_{experiment_name}.pth'
    if args.model == 'CrossAttnExp':
        model = CrossAttn_MIR_Exp(num_bins, num_attr, num_pt, dropout=args.dropout).to(device)
        best_modelname = f'best_model_resnet50_crossattn_mir_exp_{experiment_name}.pth'
    elif args.model == 'SelfAttn':
        model = SelfAttn_MIR(num_bins, num_attr, num_pt, dropout=args.dropout, num_heads=num_pt).to(device)
        best_modelname = f'best_model_resnet50_selfattn_mir_{experiment_name}.pth'
    elif args.model == 'MIRExp':
        model = PIAA_MIR_Exp(num_bins, num_attr, num_pt, dropout=args.dropout).to(device)
        best_modelname = f'best_model_resnet50_selfattn_mir_{experiment_name}.pth'
    else:
        model = PIAA_MIR(num_bins, num_attr, num_pt, dropout=args.dropout).to(device)
        best_modelname = f'best_model_resnet50_piaamir_{experiment_name}.pth'
    
    if args.pretrained_model:
        model.nima_attr.load_state_dict(torch.load(args.pretrained_model))
    if args.resume:
        model.load_state_dict(torch.load(args.resume))
    model = model.to(device)
    
    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Initialize the best test loss and the best model
    dirname = model_dir(args)
    best_modelname = os.path.join(dirname, best_modelname)

    # Training loop
    # trainer(dataloaders, model, optimizer, args, train, evaluate, device, best_modelname)
    trainer(dataloaders, model, optimizer, args, train, (evaluate, evaluate_with_prior), device, best_modelname)
