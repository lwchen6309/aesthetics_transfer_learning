import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
# from LAPIS_PIAA_dataloader import load_data as piaa_load_data
# from LAPIS_PIAA_dataloader import collate_fn as piaa_collate_fn
from LAPIS_histogram_dataloader import load_data, collate_fn, collate_fn_imgsort
import wandb
from scipy.stats import spearmanr
from train_piaa_mir import PIAA_MIR, CrossAttn_MIR, PIAA_MIR_Embed, PIAA_MIR_1layer, PIAA_MIR_avgp, PIAA_MIR_SelfAttn, PIAA_MIR_Conv
from train_piaa_mir import trainer, trainer_piaa
from utils.argflags import parse_arguments_piaa, wandb_tags, model_dir


def apply_1d_gaussian_blur(tensor, kernel_size, sigma):
    # tensor shape is [batch_size, num_features], e.g., [100, 137]
    batch_size, num_features = tensor.shape
    
    # Create a 1D Gaussian kernel
    kernel = torch.exp(-0.5 * (torch.arange(kernel_size, dtype=torch.float32).to(tensor.device) - (kernel_size - 1) / 2) ** 2 / sigma ** 2)
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, -1)  # Shape [1, 1, kernel_size]
    
    # Reshape tensor to [batch_size, 1, num_features] to apply 1D conv
    tensor = tensor.unsqueeze(1)  # Shape [batch_size, 1, num_features]
    
    # Apply 1D convolution
    smoothed_tensor = F.conv1d(tensor, kernel, padding=kernel_size // 2, groups=1)
    
    # Remove the added dimension
    smoothed_tensor = smoothed_tensor.squeeze(1)  # Shape [batch_size, num_features]
    
    return smoothed_tensor


def train_piaa(model, dataloader, criterion_mse, optimizer, device, args):
    model.train()
    running_mse_loss = 0.0

    progress_bar = tqdm(dataloader, leave=False)
    for sample in progress_bar:
        images = sample['image'].to(device)
        sample_pt = sample['traits'].float().to(device)
        sample_score = sample['response'].float().to(device) / 2.

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


def evaluate_piaa(model, dataloader, criterion_mse, device, args):
    model.eval()  # Set the model to evaluation mode
    running_mse_loss = 0.0

    all_predicted_scores = []  # List to store all predictions
    all_true_scores = []  # List to store all true scores

    progress_bar = tqdm(dataloader, leave=False)
    for sample in progress_bar:
        with torch.no_grad():
            images = sample['image'].to(device)
            sample_pt = sample['traits'].float().to(device)
            sample_score = sample['response'].float().to(device) / 2.
            
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


def train(model, dataloader, criterion_mse, optimizer, device, args):
    model.train()
    running_mse_loss = 0.0
    scale = torch.arange(0, 10).to(device)

    # Initialize GaussianBlur transform
    if args.blur_pt:
        kernel_size = getattr(args, 'kernel_size', 3)
        sigma = getattr(args, 'sigma', 1.0)

    progress_bar = tqdm(dataloader, leave=False)
    for sample in progress_bar:
        images = sample['image'].to(device)
        sample_pt = sample['traits'].float().to(device)

        # Conditionally apply Gaussian blur if args.blur_pt is True
        if args.blur_pt:
            sample_pt = apply_1d_gaussian_blur(sample_pt, kernel_size, sigma)
        
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


def evaluate(model, dataloader, criterion_mse, device, args):
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


def evaluate_with_prior(model, dataloader, prior_dataloader, criterion_mse, device, args):
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
    parser.add_argument('--freeze_nima', action='store_true', help='Enable evaluation mode')
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--sigma', type=float, default=2e-1)
    args = parser.parse_args()
    print(args)
    
    num_bins = 9
    num_attr = 8
    if args.disable_onehot:
        num_pt = 71
    else:
        num_pt = 137
    
    if args.is_log:
        tags = ["no_attr"]
        tags += wandb_tags(args)
        if not args.disable_onehot:
            tags += ['onehot enc']
        if args.blur_pt:
            tags += ['blur pt']        
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
    if args.disable_onehot:
        # train_piaa_dataset, val_piaa_dataset, test_piaa_dataset = piaa_load_data(args)
        dataloaders = (train_dataloader, val_piaa_imgsort_dataloader, test_piaa_imgsort_dataloader)
    else:
        dataloaders = (train_dataloader, val_giaa_dataloader, val_piaa_imgsort_dataloader, test_giaa_dataloader, test_piaa_imgsort_dataloader)
    
    # Define the number of classes in your dataset
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.model == 'CrossAttn':
        model = CrossAttn_MIR(num_bins, num_attr, num_pt, dropout=args.dropout, num_heads=num_pt).to(device)
        best_modelname = f'best_model_resnet50_crossattn_mir_{experiment_name}.pth'
    elif args.model == 'MIR_Embed':
        model = PIAA_MIR_Embed(num_bins, num_attr, num_pt, dropout=args.dropout).to(device)
        best_modelname = f'best_model_resnet50_piaamir_embed_{experiment_name}.pth'
    elif args.model == 'PIAA_MIR_1layer':
        model = PIAA_MIR_1layer(num_bins, num_attr, num_pt, dropout=args.dropout).to(device)
        best_modelname = f'best_model_resnet50_piaamir_1layer_{experiment_name}.pth'
    elif args.model == 'PIAA_MIR_avgp':
        model = PIAA_MIR_avgp(num_bins, num_attr, num_pt, dropout=args.dropout).to(device)
        best_modelname = f'best_model_resnet50_piaamir_4layer_{experiment_name}.pth'
    elif args.model == 'PIAA_MIR_SelfAttn':
        model = PIAA_MIR_SelfAttn(num_bins, num_attr, num_pt, dropout=args.dropout).to(device)
        best_modelname = f'best_model_resnet50_piaamir_selfattn_{experiment_name}.pth'
    elif args.model == 'PIAA_MIR_Conv':
        model = PIAA_MIR_Conv(num_bins, num_attr, num_pt, dropout=args.dropout).to(device)
        best_modelname = f'best_model_resnet50_piaamir_selfattn_{experiment_name}.pth'        
    else:
        model = PIAA_MIR(num_bins, num_attr, num_pt, dropout=args.dropout).to(device)
        best_modelname = f'best_model_resnet50_piaamir_{experiment_name}.pth'
    
    if args.pretrained_model:
        model.nima_attr.load_state_dict(torch.load(args.pretrained_model))
    if args.resume:
        model.load_state_dict(torch.load(args.resume))
    model = model.to(device)
    
    # Define the optimizer
    if args.freeze_nima:
        nima_attr_params = list(model.nima_attr.parameters())
        other_params = [param for param in model.parameters() if param not in nima_attr_params]
        # Define the optimizer excluding nima_attr parameters
        optimizer = optim.Adam(other_params, lr=args.lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Initialize the best test loss and the best model
    dirname = model_dir(args)
    best_modelname = os.path.join(dirname, best_modelname)
    
    # Training loop
    if args.disable_onehot:
        trainer_piaa(dataloaders, model, optimizer, args, train, evaluate, device, best_modelname)
        # trainer_piaa(dataloaders, model, optimizer, args, train_piaa, evaluate_piaa, device, best_modelname)
    else:
        trainer(dataloaders, model, optimizer, args, train, (evaluate, evaluate_with_prior), device, best_modelname)
