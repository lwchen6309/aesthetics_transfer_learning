import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import resnet50
from torch.utils.data import DataLoader
from tqdm import tqdm
from PARA_PIAA_dataloader import collect_batch_attribute, collect_batch_personal_trait
from PARA_PIAA_dataloader import load_data as piaa_load_data
from PARA_histogram_dataloader import load_data, collate_fn_imgsort
import wandb
from scipy.stats import spearmanr
from train_nima_attr import NIMA_attr
from utils.argflags import parse_arguments_piaa, wandb_tags, model_dir


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class InternalInteraction(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(InternalInteraction, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.input_dim = input_dim
    
    def forward(self, attribute_embeddings):
        num_attributes = attribute_embeddings.size(1)
        batch_size = attribute_embeddings.size(0)

        # Initialize the output matrix
        interaction_matrix = torch.zeros(batch_size, num_attributes, num_attributes, self.input_dim).to(attribute_embeddings.device)

        for i in range(num_attributes):
            for j in range(num_attributes):
                # Concatenate the embeddings of the i-th and j-th attributes
                # combined_features = torch.cat((attribute_embeddings[:, i], attribute_embeddings[:, j]), dim=-1)  # Shape: [batch_size, 2 * input_dim]
                combined_features = attribute_embeddings[:, i] * attribute_embeddings[:, j]
                # Apply MLP to the concatenated features and store in the interaction matrix
                interaction_matrix[:, i, j] = self.mlp(combined_features)
        # Sum over the second dimension (dim=1)
        aggregated_interactions = torch.sum(interaction_matrix, dim=1)  # Shape: [batch_size, num_attributes, input_dim]
        return aggregated_interactions

class ExternalInteraction(nn.Module):
    def __init__(self):
        super(ExternalInteraction, self).__init__()
    
    def forward(self, user_attributes, image_attributes):
        # user_attributes and image_attributes have shapes [B, num_attr]
        # Compute the outer product: the result will have shape [B, num_attr_user, num_attr_img]
        interaction_results = user_attributes.unsqueeze(2) * image_attributes.unsqueeze(1)
        
        aggregated_interactions_user = torch.sum(interaction_results, dim=2)
        aggregated_interactions_img = torch.sum(interaction_results, dim=1)
        return aggregated_interactions_user, aggregated_interactions_img

class Interfusion_GRU(nn.Module):
    def __init__(self, input_dim=64):
        super(Interfusion_GRU, self).__init__()
        self.gru = nn.GRUCell(input_dim, input_dim)

    def forward(self, initial_node, internal_interaction, external_interaction):
        num_attr = initial_node.shape[1]
        results = []
        for i in range(num_attr):
            fused_node = self.gru(initial_node[:, i], None)
            fused_node = self.gru(internal_interaction[:, i], fused_node)
            fused_node = self.gru(external_interaction[:, i], fused_node)
            results.append(fused_node)
        results = torch.stack(results, dim=1)
        return results

class Interfusion_MLP(nn.Module):
    def __init__(self, input_dim=64, hidden_size=256):
        super(Interfusion_MLP, self).__init__()
        self.mlp = MLP(input_dim*3, hidden_size, input_dim)

    def forward(self, initial_node, internal_interaction, external_interaction):
        results = self.mlp(torch.cat([initial_node, internal_interaction, external_interaction], dim=1))
        return results
   

class PIAA_ICI(nn.Module):
    def __init__(self, num_bins, num_attr, num_pt, input_dim = 64, hidden_size=256, dropout=None):
        super(PIAA_ICI, self).__init__()
        self.num_bins = num_bins
        self.num_attr = num_attr
        self.num_pt = num_pt
        self.scale = torch.arange(1, 5.5, 0.5)
        
        # Placeholder for the NIMA_attr model
        self.nima_attr = NIMA_attr(num_bins, num_attr)
        self.input_dim = input_dim
        # Internal and External Interaction Modules
        
        self.internal_interaction_img = InternalInteraction(input_dim=input_dim, hidden_dim=hidden_size)
        self.internal_interaction_user = InternalInteraction(input_dim=input_dim, hidden_dim=hidden_size)
        self.external_interaction = ExternalInteraction()
        
        # Interfusion Module
        self.interfusion_img = Interfusion_GRU(input_dim=input_dim)
        self.interfusion_user = Interfusion_GRU(input_dim=input_dim)
        
        # MLPs for final prediction
        self.node_attr_user = MLP(num_pt, hidden_size, num_attr*input_dim)
        self.node_attr_img = MLP(num_attr, hidden_size, num_attr*input_dim)
        self.mlp_dist = MLP(num_bins, hidden_size, 1)
        self.attr_corr = nn.Linear(input_dim, 1)

        # Dropout
        self.dropout = dropout
        if self.dropout:
            self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, images, personal_traits):
        logit, attr_img = self.nima_attr(images)
        if self.dropout:
            personal_traits = self.dropout_layer(personal_traits)
        n_attr = attr_img.shape[1]
        attr_img = self.node_attr_img(attr_img).view(-1,n_attr,self.input_dim)
        attr_user = self.node_attr_user(personal_traits).view(-1,n_attr,self.input_dim)
        prob = F.softmax(logit, dim=1)
        
        # Internal Interaction (among image attributes)
        internal_img = self.internal_interaction_img(attr_img)
        internal_user = self.internal_interaction_user(attr_user)
        # External Interaction (between user and image attributes)
        aggregated_interactions_user, aggregated_interactions_img = self.external_interaction(attr_user, attr_img)
        
        # Interfusion to combine interactions and initial attributes
        fused_features_img = self.interfusion_img(attr_img, internal_img, aggregated_interactions_img)
        fused_features_user = self.interfusion_user(attr_user, internal_user, aggregated_interactions_user)
        
        # Final prediction
        interaction_outputs = torch.sum(fused_features_img, dim=1, keepdim=False) + torch.sum(fused_features_user, dim=1, keepdim=False)
        interaction_outputs = self.attr_corr(interaction_outputs)
        direct_outputs = self.mlp_dist(prob * self.scale.to(images.device))
        output = interaction_outputs + direct_outputs
        return output


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
        sample_pt = sample_pt.to(device).float()
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
            sample_pt = sample_pt.to(device).float()
            # batch_size = len(images)

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


def train(model, dataloader, criterion_mse, optimizer, device, args):
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


def evaluate(model, dataloader, criterion_mse, device, args):
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


def evaluate_with_prior(model, dataloader, prior_dataloader, criterion_mse, device, args):
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
        test_piaa_loss, test_piaa_srocc = evaluate_fn(model, test_piaa_imgsort_dataloader, criterion_mse, device, args)
        test_giaa_loss, test_giaa_srocc = evaluate_fn(model, test_giaa_dataloader, criterion_mse, device, args)
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
        train_mse_loss = train_fn(model, train_dataloader, criterion_mse, optimizer, device, args)
        # Testing
        val_mse_loss, val_srocc = evaluate_fn(model, val_piaa_imgsort_dataloader, criterion_mse, device, args)

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
    test_piaa_loss, test_piaa_srocc = evaluate_fn(model, test_piaa_imgsort_dataloader, criterion_mse, device, args)
    print(test_piaa_srocc)
    test_giaa_loss, test_giaa_srocc = evaluate_fn(model, test_giaa_dataloader, criterion_mse, device, args)
    test_giaa_loss_wprior, test_giaa_srocc_wprior = evaluate_fn_with_prior(model, test_giaa_dataloader, val_giaa_dataloader, criterion_mse, device, args)
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


def trainer_piaa(dataloaders, model, optimizer, args, train_fn, evaluate_fn, device, best_modelname):

    train_dataloader, val_dataloader, test_dataloader = dataloaders

    if args.resume is not None:    
        test_mse_loss, test_srocc = evaluate_fn(model, test_dataloader, criterion_mse, device)
        if args.is_log:
            wandb.log({"Test PIAA MSE Loss (Pretrained)": test_mse_loss,
                    "Test PIAA SROCC (Pretrained)": test_srocc}, commit=True)
    
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
        val_mse_loss, val_srocc = evaluate_fn(model, val_dataloader, criterion_mse, device)

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
    test_mse_loss, test_srocc = evaluate_fn(model, test_dataloader, criterion_mse, device)
    if args.is_log:
        wandb.log({"Test PIAA MSE Loss": test_mse_loss,
                   "Test PIAA SROCC": test_srocc}, commit=True)
    print(
        # f"Train PIAA MSE Loss: {train_mse_loss:.4f}, "
        f"Test PIAA MSE Loss: {test_mse_loss:.4f}, "
        f"Test PIAA SROCC Loss: {test_srocc:.4f}, ")
    
    return test_srocc  


criterion_mse = nn.MSELoss()


if __name__ == '__main__':
    parser = parse_arguments_piaa(False)
    parser.add_argument('--model', type=str, default='PIAA-ICI')
    args = parser.parse_args()
    print(args)
    
    n_workers = 8
    num_bins = 9
    num_attr = 8
    
    if args.disable_onehot:
        num_pt = 25 # number of personal trait
    else:
        num_pt = 50 + 20
    
    if args.is_log:
        tags = wandb_tags(args)
        if not args.disable_onehot:
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
    if args.disable_onehot:
        train_dataset, val_dataset, test_dataset = piaa_load_data(args)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=n_workers, timeout=300)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=n_workers, timeout=300)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=n_workers, timeout=300)
        dataloaders = (train_dataloader, val_dataloader, test_dataloader)
    else:
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
    model = PIAA_ICI(num_bins, num_attr, num_pt, dropout=args.dropout)
    best_modelname = f'best_model_resnet50_piaaici_{experiment_name}.pth'


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
    if args.disable_onehot:
        trainer_piaa(dataloaders, model, optimizer, args, train_piaa, evaluate_piaa, device, best_modelname)
    else:
        trainer(dataloaders, model, optimizer, args, train, (evaluate, evaluate_with_prior), device, best_modelname)
    