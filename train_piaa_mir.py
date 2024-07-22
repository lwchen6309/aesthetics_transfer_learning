import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from PARA_PIAA_dataloader import load_data, collect_batch_attribute, collect_batch_personal_trait
from PARA_histogram_dataloader import load_data, collate_fn_imgsort
import wandb
from scipy.stats import spearmanr
from train_nima_attr import NIMA_attr
from utils.argflags import parse_arguments_piaa, wandb_tags, model_dir


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class PIAA_MIR(nn.Module):
    def __init__(self, num_bins, num_attr, num_pt, hidden_size=1024, dropout=None):
        super(PIAA_MIR, self).__init__()
        self.num_bins = num_bins
        self.num_attr = num_attr
        self.num_pt = num_pt
        self.scale = torch.arange(1, 5.5, 0.5)  # This will be moved to the correct device in forward()
        
        # Placeholder for the NIMA_attr model
        self.nima_attr = NIMA_attr(num_bins, num_attr)  # Make sure to define or import NIMA_attr
        
        # Interaction MLPs
        self.dropout = dropout
        if dropout is not None:
            self.dropout_layer = nn.Dropout(self.dropout)
        self.mlp1 = MLP(num_attr * num_pt, hidden_size, 1)
        self.mlp2 = MLP(num_bins, hidden_size, 1)

    def forward(self, images, personal_traits):
        logit, attr_mean_pred = self.nima_attr(images)
        prob = F.softmax(logit, dim=1)
        
        # Interaction map calculation
        A_ij = attr_mean_pred.unsqueeze(2) * personal_traits.unsqueeze(1)
        I_ij = A_ij.view(images.size(0), -1)
        if self.dropout is not None:
            I_ij = self.dropout_layer(I_ij)
        interaction_outputs = self.mlp1(I_ij)
        direct_outputs = self.mlp2(prob * self.scale.to(images.device))
        
        return interaction_outputs + direct_outputs


class PIAA_MIR_Embed(nn.Module):
    def __init__(self, num_bins, num_attr, num_pt, hidden_size=1024, dropout=None):
        super(PIAA_MIR_Embed, self).__init__()
        self.num_bins = num_bins
        self.num_attr = num_attr
        self.num_pt = num_pt
        self.scale = torch.arange(1, 5.5, 0.5)  # This will be moved to the correct device in forward()
        
        # Placeholder for the NIMA_attr model
        self.nima_attr = NIMA_attr(num_bins, num_attr)  # Make sure to define or import NIMA_attr
        
        # Embedding layers
        self.attr_embedding = nn.Embedding(num_attr, 512)
        self.pt_embedding = nn.Embedding(num_pt, 512)
        
        # Interaction MLPs
        self.dropout = dropout
        if dropout is not None:
            self.dropout_layer = nn.Dropout(self.dropout)
        self.mlp1 = MLP(num_attr * num_pt, hidden_size, 1)
        self.mlp2 = MLP(num_bins, hidden_size, 1)

    def forward(self, images, personal_traits):
        logit, attr_mean_pred = self.nima_attr(images)
        prob = F.softmax(logit, dim=1)
        
        # Compute embeddings
        attr_embed = self.attr_embedding(torch.arange(self.num_attr).to(images.device)).unsqueeze(0).expand(images.size(0), -1, -1)
        pt_embed = self.pt_embedding(torch.arange(self.num_pt).to(images.device)).unsqueeze(0).expand(images.size(0), -1, -1)
        
        attr_embed = attr_embed * attr_mean_pred.unsqueeze(2)  # [B, 10, 512]
        pt_embed = pt_embed * personal_traits.unsqueeze(2)     # [B, 70, 512]
        
        # Compute inner product over the last channel to get [B, 10, 70]
        A_ij = torch.matmul(attr_embed, pt_embed.transpose(1, 2))
        
        I_ij = A_ij.view(images.size(0), -1)  # Flatten to [B, 10 * 70]
        if self.dropout is not None:
            I_ij = self.dropout_layer(I_ij)
        interaction_outputs = self.mlp1(I_ij)
        direct_outputs = self.mlp2(prob * self.scale.to(images.device))
        
        return interaction_outputs + direct_outputs


class PIAA_MIR_Exp(nn.Module):
    def __init__(self, num_bins, num_attr, num_pt, hidden_size=1024, dropout=None):
        super(PIAA_MIR_Exp, self).__init__()
        self.num_bins = num_bins
        self.num_attr = num_attr
        self.num_pt = num_pt
        self.scale = torch.arange(1, 5.5, 0.5)  # This will be moved to the correct device in forward()
        
        # Placeholder for the NIMA_attr model
        self.nima_attr = NIMA_attr(num_bins, num_attr)  # Make sure to define or import NIMA_attr
        
        # Interaction MLPs
        self.dropout = dropout
        if dropout is not None:
            self.dropout_layer = nn.Dropout(self.dropout)
        pt_encoder_dims = 512
        self.pt_encoder = MLP(num_pt, hidden_size, pt_encoder_dims)
        self.mlp1 = MLP(num_attr * pt_encoder_dims, hidden_size, 1)
        self.mlp2 = MLP(num_bins, hidden_size, 1)

    def forward(self, images, personal_traits):
        logit, attr_mean_pred = self.nima_attr(images)
        prob = F.softmax(logit, dim=1)
        
        # Interaction map calculation
        pt_code = self.pt_encoder(personal_traits)
        A_ij = attr_mean_pred.unsqueeze(2) * pt_code.unsqueeze(1)
        I_ij = A_ij.view(images.size(0), -1)
        if self.dropout is not None:
            I_ij = self.dropout_layer(I_ij)
        interaction_outputs = self.mlp1(I_ij)
        direct_outputs = self.mlp2(prob * self.scale.to(images.device))
        
        return interaction_outputs + direct_outputs


class CrossAttn_MIR(nn.Module):
    def __init__(self, num_bins, num_attr, num_pt, hidden_size=1024, dropout=None, num_heads=7):
        super(CrossAttn_MIR, self).__init__()
        self.num_bins = num_bins
        self.num_attr = num_attr
        self.num_pt = num_pt
        self.scale = torch.arange(1, 5.5, 0.5)  # This will be moved to the correct device in forward()
        
        # Placeholder for the NIMA_attr model
        self.nima_attr = NIMA_attr(num_bins, num_attr)  # Make sure to define or import NIMA_attr
        
        # Interaction MLPs
        self.dropout = dropout
        if dropout is not None:
            self.dropout_layer = nn.Dropout(self.dropout)
        self.mlp1 = MLP(num_attr * num_pt, hidden_size, 1)
        self.mlp2 = MLP(num_bins, hidden_size, 1)
        
        # Cross-attention mechanism
        self.cross_attention = nn.MultiheadAttention(embed_dim=num_pt, num_heads=num_heads, dropout=dropout if dropout else 0)

        # Linear projections to embedding dimension
        self.query_proj = nn.Linear(num_attr, num_pt)
        self.key_proj = nn.Linear(num_pt, num_pt)

    def forward(self, images, personal_traits):
        logit, attr_mean_pred = self.nima_attr(images)
        prob = F.softmax(logit, dim=1)

        key = self.key_proj(personal_traits) # Project to 512
        query = self.query_proj(attr_mean_pred) # Project to 512
        # Cross-attention calculation
        personal_traits, _ = self.cross_attention(query, key, personal_traits)
        
        # Preparing input for cross-attention
        A_ij = attr_mean_pred.unsqueeze(2) * personal_traits.unsqueeze(1)
        I_ij = A_ij.view(images.size(0), -1)
        interaction_outputs = self.mlp1(I_ij)
        direct_outputs = self.mlp2(prob * self.scale.to(images.device))
        
        return interaction_outputs + direct_outputs


class CrossAttn_MIR_Exp(nn.Module):
    def __init__(self, num_bins, num_attr, num_pt, hidden_size=1024, dropout=None, num_heads=8):
        super(CrossAttn_MIR_Exp, self).__init__()
        self.num_bins = num_bins
        self.num_attr = num_attr
        self.num_pt = num_pt
        self.scale = torch.arange(1, 5.5, 0.5)  # This will be moved to the correct device in forward()
        
        # Placeholder for the NIMA_attr model
        self.nima_attr = NIMA_attr(num_bins, num_attr)  # Make sure to define or import NIMA_attr
        
        # Interaction MLPs
        self.dropout = dropout
        if dropout is not None:
            self.dropout_layer = nn.Dropout(self.dropout)
        pt_enc_dims = 512
        # self.mlp1 = MLP(num_attr * num_pt, hidden_size, 1)
        self.mlp1 = MLP(num_attr * pt_enc_dims, hidden_size, 1)
        self.mlp2 = MLP(num_bins, hidden_size, 1)
        
        # Cross-attention mechanism
        self.cross_attention = nn.MultiheadAttention(embed_dim=pt_enc_dims, num_heads=num_heads, dropout=dropout if dropout else 0)

        # Linear projections to embedding dimension
        self.query_proj = nn.Linear(num_attr, pt_enc_dims)
        self.key_proj = nn.Linear(num_pt, pt_enc_dims)
        self.value_proj = nn.Linear(num_pt, pt_enc_dims)
        # self.value_proj = MLP(num_pt, hidden_size, pt_enc_dims)
    
    def forward(self, images, personal_traits):
        logit, attr_mean_pred = self.nima_attr(images)
        prob = F.softmax(logit, dim=1)

        query = self.query_proj(attr_mean_pred) # Project to 512
        key = self.key_proj(personal_traits) # Project to 512
        value = self.value_proj(personal_traits) # Project to 512
        # Cross-attention calculation
        personal_traits, _ = self.cross_attention(query, key, value)
        
        # Preparing input for cross-attention
        A_ij = attr_mean_pred.unsqueeze(2) * personal_traits.unsqueeze(1)
        I_ij = A_ij.view(images.size(0), -1)
        interaction_outputs = self.mlp1(I_ij)
        direct_outputs = self.mlp2(prob * self.scale.to(images.device))
        
        return interaction_outputs + direct_outputs

class SelfAttn_MIR(nn.Module):
    def __init__(self, num_bins, num_attr, num_pt, hidden_size=1024, dropout=None, num_heads=7):
        super(SelfAttn_MIR, self).__init__()
        self.num_bins = num_bins
        self.num_attr = num_attr
        self.num_pt = num_pt
        self.scale = torch.arange(1, 5.5, 0.5)  # This will be moved to the correct device in forward()
        
        # Placeholder for the NIMA_attr model
        self.nima_attr = NIMA_attr(num_bins, num_attr)  # Make sure to define or import NIMA_attr
        
        # Interaction MLPs
        self.dropout = dropout
        if dropout is not None:
            self.dropout_layer = nn.Dropout(self.dropout)
        self.mlp1 = MLP(num_attr * num_pt, hidden_size, 1)
        self.mlp2 = MLP(num_bins, hidden_size, 1)
        
        # Cross-attention mechanism
        self.cross_attention = nn.MultiheadAttention(embed_dim=num_pt, num_heads=num_heads, dropout=dropout if dropout else 0)

        # Linear projections to embedding dimension
        self.query_proj = nn.Linear(num_pt, num_pt)
        self.key_proj = nn.Linear(num_attr, num_pt)
        # self.value_proj = nn.Linear(num_attr, 512)

    def forward(self, images, personal_traits):
        logit, attr_mean_pred = self.nima_attr(images)
        prob = F.softmax(logit, dim=1)

        # if self.dropout is not None:
        #     attn_output = self.dropout_layer(attn_output)        
        attr_mean_pred_proj = self.key_proj(attr_mean_pred)  # Project to 512
        personal_traits_proj = self.query_proj(personal_traits)  # Project to 512
        # Cross-attention calculation
        attn_output, _ = self.cross_attention(attr_mean_pred_proj, personal_traits_proj, personal_traits)
        
        # Preparing input for cross-attention
        A_ij = attn_output.unsqueeze(2) * attr_mean_pred.unsqueeze(1)
        I_ij = A_ij.view(images.size(0), -1)
        interaction_outputs = self.mlp1(I_ij)
        direct_outputs = self.mlp2(prob * self.scale.to(images.device))
        
        return interaction_outputs + direct_outputs


def train_piaa(model, dataloader, criterion_mse, optimizer, device):
    model.train()
    running_mse_loss = 0.0

    progress_bar = tqdm(dataloader, leave=False)
    scale = torch.arange(1, 5.5, 0.5).to(device)    
    for sample in progress_bar:
        images = sample['image']
        sample_score, sample_attr = collect_batch_attribute(sample)
        sample_pt = collect_batch_personal_trait(sample)
        images = images.to(device)
        sample_score = sample_score.to(device).float()
        sample_attr = sample_attr.to(device)
        sample_pt = sample_pt.to(device)
        optimizer.zero_grad()

        y_ij = model(images, sample_pt)
        
        # MSE loss
        loss = criterion_mse(y_ij, sample_score)

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
    scale = torch.arange(1, 5.5, 0.5).to(device)
    for sample in progress_bar:
        with torch.no_grad():
            images = sample['image']
            sample_score, sample_attr = collect_batch_attribute(sample)
            sample_pt = collect_batch_personal_trait(sample)
            images = images.to(device)
            sample_score = sample_score.to(device).float()
            sample_attr = sample_attr.to(device)
            sample_pt = sample_pt.to(device)
            batch_size = len(images)

            y_ij = model(images, sample_pt)

            # MSE loss
            loss = criterion_mse(y_ij, sample_score)
            running_mse_loss += loss.item()

            # Store predicted and true scores for SROCC calculation
            predicted_scores = y_ij.view(-1).cpu().numpy()
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
    scale = torch.arange(1, 5.5, 0.5).to(device)

    progress_bar = tqdm(dataloader, leave=False)
    for sample in progress_bar:
        images = sample['image'].to(device)
        # sample_pt = sample['traits'].float().to(device)
        traits_histogram = sample['traits'].to(device)
        onehot_big5 = sample['big5'].to(device)
        sample_pt = torch.cat([traits_histogram, onehot_big5], dim=1)

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
    scale = torch.arange(1, 5.5, 0.5).to(device)
    all_predicted_scores = []  # List to store all predictions
    all_true_scores = []  # List to store all true scores

    progress_bar = tqdm(dataloader, leave=False)
    for sample in progress_bar:
        with torch.no_grad():
            images = sample['image'].to(device)
            # sample_pt = sample['traits'].float().to(device)
            traits_histogram = sample['traits'].to(device)
            onehot_big5 = sample['big5'].to(device)
            sample_pt = torch.cat([traits_histogram, onehot_big5], dim=1)            
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
    scale = torch.arange(1, 5.5, 0.5).to(device)
    all_predicted_scores = []  # List to store all predictions
    all_true_scores = []  # List to store all true scores
    
    with torch.no_grad():
        traits_histograms = []
        for sample in tqdm(prior_dataloader, leave=False):
            traits_histogram = sample['traits'].to(device)
            onehot_big5 = sample['big5'].to(device)
            traits_histogram = torch.cat([traits_histogram, onehot_big5], dim=1)
            traits_histograms.append(traits_histogram)
        mean_traits_histogram = torch.mean(torch.cat(traits_histograms, dim=0), dim=0).unsqueeze(0)
        
        progress_bar = tqdm(dataloader, leave=False)
        for sample in progress_bar:
            images = sample['image'].to(device)
            # sample_pt = sample['traits'].float().to(device)
            # traits_histogram = sample['traits'].to(device)
            # onehot_big5 = sample['big5'].to(device)
            # sample_pt = torch.cat([traits_histogram, onehot_big5], dim=1)
            sample_pt = mean_traits_histogram.repeat(images.shape[0], 1)
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


def trainer(dataloaders, model, optimizer, args, train_fn, evaluate_fns, device, best_modelname):

    train_dataloader, val_giaa_dataloader, val_piaa_imgsort_dataloader, test_giaa_dataloader, test_piaa_imgsort_dataloader = dataloaders
    evaluate_fn, evaluate_fn_with_prior = evaluate_fns

    if args.resume is not None:
        test_piaa_loss, test_piaa_srocc = evaluate_fn(model, test_piaa_imgsort_dataloader, criterion_mse, device)
        test_giaa_loss, test_giaa_srocc = evaluate_fn(model, test_giaa_dataloader, criterion_mse, device)
        test_giaa_loss_wprior, test_giaa_srocc_wprior = evaluate_fn_with_prior(model, test_giaa_dataloader, val_giaa_dataloader, criterion_mse, device)
        if args.is_log:
            wandb.log({"Test PIAA MSE Loss (Pretrained)": test_piaa_loss,
                    "Test PIAA SROCC (Pretrained)": test_piaa_srocc,
                    "Test GIAA MSE Loss (Pretrained)": test_giaa_loss,
                    "Test GIAA SROCC (Pretrained)": test_giaa_srocc,
                    "Test GIAA MSE Loss (Prior)(Pretrained)": test_giaa_loss_wprior,
                    "Test GIAA SROCC (Prior)(Pretrained)": test_giaa_srocc_wprior,                   
                    }, commit=True)
    
    num_patience_epochs = 0
    best_test_srocc = 0
    for epoch in range(args.num_epochs):
        if args.is_eval:
            break
        # Learning rate schedule
        if (epoch + 1) % args.lr_schedule_epochs == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= args.lr_decay_factor
        # Training
        train_mse_loss = train_fn(model, train_dataloader, criterion_mse, optimizer, device)
        # Testing
        val_mse_loss, val_srocc = evaluate_fn(model, val_piaa_imgsort_dataloader, criterion_mse, device)

        if args.is_log:
            wandb.log({"Train PIAA MSE Loss": train_mse_loss,
                        "Val PIAA MSE Loss": val_mse_loss,
                        "Val PIAA SROCC": val_srocc,
                    }, commit=True)

        # Early stopping check
        if val_srocc > best_test_srocc:
            best_test_srocc = val_srocc
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
    test_piaa_loss, test_piaa_srocc = evaluate_fn(model, test_piaa_imgsort_dataloader, criterion_mse, device)
    test_giaa_loss, test_giaa_srocc = evaluate_fn(model, test_giaa_dataloader, criterion_mse, device)
    test_giaa_loss_wprior, test_giaa_srocc_wprior = evaluate_fn_with_prior(model, test_giaa_dataloader, val_giaa_dataloader, criterion_mse, device)
    if args.is_log:
        wandb.log({"Test PIAA MSE Loss": test_piaa_loss,
                   "Test PIAA SROCC": test_piaa_srocc,
                   "Test GIAA MSE Loss": test_giaa_loss,
                   "Test GIAA SROCC": test_giaa_srocc,
                   "Test GIAA MSE Loss (Prior)": test_giaa_loss_wprior,
                   "Test GIAA SROCC (Prior)": test_giaa_srocc_wprior,                   
                   }, commit=True)
    print(
        f"Test GIAA SROCC: {test_giaa_srocc:.4f}, "
        f"Test GIAA SROCC (Prior): {test_giaa_srocc_wprior:.4f}, "
        f"Test PIAA SROCC: {test_piaa_srocc:.4f}, ")
    
    return test_piaa_srocc  

criterion_mse = nn.MSELoss()


if __name__ == '__main__':
    parser = parse_arguments_piaa(False)
    parser.add_argument('--model', type=str, default='PIAA-MIR')
    args = parser.parse_args()
    print(args)
    
    n_workers = 8
    num_bins = 9
    num_attr = 8
    # num_pt = 25 # number of personal trait
    num_pt = 50 + 20
    
    if args.is_log:
        tags = wandb_tags(args)
        tags += ['onehot enc']
        wandb.init(project="resnet_PARA_PIAA",
                notes=args.model,
                tags=tags)
        wandb.config = {
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "num_epochs": args.num_epochs
        }
        experiment_name = wandb.run.name
    else:
        experiment_name = ''
    
    # Create dataloaders for training and test sets
    # train_dataset, val_dataset, test_dataset = load_data(args)
    # train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=n_workers, timeout=300)
    # val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=n_workers, timeout=300)
    # test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=n_workers, timeout=300)
    # dataloaders = (train_dataloader, val_dataloader, test_dataloader)
    train_dataset, val_giaa_dataset, val_piaa_imgsort_dataset, test_giaa_dataset, test_piaa_imgsort_dataset = load_data(args)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, timeout=300)
    val_giaa_dataloader = DataLoader(val_giaa_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, timeout=300)
    val_piaa_imgsort_dataloader = DataLoader(val_piaa_imgsort_dataset, batch_size=5, shuffle=False, num_workers=args.num_workers, timeout=300, collate_fn=collate_fn_imgsort)
    test_giaa_dataloader = DataLoader(test_giaa_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, timeout=300)
    test_piaa_imgsort_dataloader = DataLoader(test_piaa_imgsort_dataset, batch_size=5, shuffle=False, num_workers=args.num_workers, timeout=300, collate_fn=collate_fn_imgsort)
    dataloaders = (train_dataloader, val_giaa_dataloader, val_piaa_imgsort_dataloader, test_giaa_dataloader, test_piaa_imgsort_dataloader)

    # Define the number of classes in your dataset
    num_classes = num_attr + num_bins
    # Define the device for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.model == 'CrossAttn':
        model = CrossAttn_MIR(num_bins, num_attr, num_pt, dropout=args.dropout).to(device)
        best_modelname = f'best_model_resnet50_crossattn_mir_{experiment_name}.pth'
    if args.model == 'CrossAttnExp':
        model = CrossAttn_MIR_Exp(num_bins, num_attr, num_pt, dropout=args.dropout).to(device)
        best_modelname = f'best_model_resnet50_crossattn_mir_exp_{experiment_name}.pth'
    elif args.model == 'SelfAttn':
        model = SelfAttn_MIR(num_bins, num_attr, num_pt, dropout=args.dropout).to(device)
        best_modelname = f'best_model_resnet50_selfattn_mir_{experiment_name}.pth'
    else:
        model = PIAA_MIR(num_bins, num_attr, num_pt, dropout=args.dropout).to(device)
        best_modelname = f'best_model_resnet50_piaamir_{experiment_name}.pth'

    if args.pretrained_model:
        model.nima_attr.load_state_dict(torch.load(args.pretrained_model))
    if args.resume:
        model.load_state_dict(torch.load(args.resume))
    model = model.to(device)
    
    # Define the loss functions
    criterion_mse = nn.MSELoss()

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Initialize the best test loss and the best model
    
    dirname = model_dir(args)
    best_modelname = os.path.join(dirname, best_modelname)
    
    # Training loop
    # trainer(dataloaders, model, optimizer, args, train_piaa, evaluate_piaa, device, best_modelname)
    trainer(dataloaders, model, optimizer, args, train, (evaluate, evaluate_with_prior), device, best_modelname)
    