import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from PARA_PIAA_dataloader import load_data, collect_batch_attribute, collect_batch_personal_trait
import wandb
from scipy.stats import spearmanr
from train_nima_attr import NIMA_attr
import argparse


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CombinedModel(nn.Module):
    def __init__(self, num_bins, num_attr, num_pt, hidden_size=1024):
        super(CombinedModel, self).__init__()
        self.num_bins = num_bins
        self.num_attr = num_attr
        self.num_pt = num_pt
        self.scale = torch.arange(1, 5.5, 0.5)  # This will be moved to the correct device in forward()
        
        # Placeholder for the NIMA_attr model
        self.nima_attr = NIMA_attr(num_bins, num_attr)  # Make sure to define or import NIMA_attr
        
        # Interaction MLPs
        self.mlp1 = MLP(num_attr * num_pt, hidden_size, 1)
        self.mlp2 = MLP(num_bins, hidden_size, 1)

    def forward(self, images, personal_traits):
        logit, attr_mean_pred = self.nima_attr(images)
        prob = F.softmax(logit, dim=1)
        
        # Interaction map calculation
        A_ij = attr_mean_pred.unsqueeze(2) * personal_traits.unsqueeze(1)
        I_ij = A_ij.view(images.size(0), -1)
        interaction_outputs = self.mlp1(I_ij)
        direct_outputs = self.mlp2(prob * self.scale.to(images.device))
        
        return interaction_outputs + direct_outputs


class SimplePerModel(nn.Module):
    def __init__(self, num_bins, num_attr, num_pt, hidden_size=1024):
        super(SimplePerModel, self).__init__()
        self.num_bins = num_bins
        self.num_attr = num_attr
        self.num_pt = num_pt
        self.scale = torch.arange(1, 5.5, 0.5)  # This will be moved to the correct device in forward()
        
        # Placeholder for the NIMA_attr model
        self.nima_attr = NIMA_attr(num_bins, num_attr)  # Make sure to define or import NIMA_attr
        
        # Interaction MLPs
        self.per_mlp = MLP(num_pt, hidden_size, 1)

    def forward(self, images, personal_traits):
        logit, attr_mean_pred = self.nima_attr(images)
        prob = F.softmax(logit, dim=1)        
        # Interaction map calculation
        direct_outputs = torch.sum(prob * self.scale.to(images.device), dim=1, keepdim=True)
        interaction = self.per_mlp(personal_traits)
        return direct_outputs + interaction
    

def train(model, dataloader, criterion_mse, optimizer, device):
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


def evaluate(model, dataloader, criterion_mse, device):
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


def trainer(dataloaders, model, optimizer, args, train_fn, evaluate_fn, device, best_modelname):

    train_dataloader, val_dataloader, test_dataloader = dataloaders
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
    parser = argparse.ArgumentParser(description='Training and Testing the Combined Model for data spliting')
    parser.add_argument('--trait', type=str, default=None)
    parser.add_argument('--value', type=str, default=None)
    parser.add_argument('--fold_id', type=int, default=1)
    parser.add_argument('--n_fold', type=int, default=4)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--pretrained_model', type=str, required=True)
    parser.add_argument('--use_cv', action='store_true', help='Enable cross validation')
    parser.add_argument('--is_eval', action='store_true', help='Enable evaluation mode')
    parser.add_argument('--no_log', action='store_false', dest='is_log', help='Disable logging')
    
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr_schedule_epochs', type=int, default=5, help='Epochs after which to apply learning rate decay')
    parser.add_argument('--lr_decay_factor', type=float, default=0.1, help='Factor by which to decay the learning rate')
    parser.add_argument('--max_patience_epochs', type=int, default=10, help='Max patience epochs for early stopping')    
    args = parser.parse_args()
    
    batch_size = args.batch_size
    n_workers = 8
    num_bins = 9
    num_attr = 8
    num_pt = 25 # number of personal trait

    if args.is_log:
        tags = ["no_attr","PIAA"]
        if args.use_cv:
            tags += ["CV%d/%d"%(args.fold_id, args.n_fold)]
        wandb.init(project="resnet_PARA_PIAA",
                notes="PIAA-MIR",
                tags = tags)
        wandb.config = {
            "learning_rate": args.lr,
            "batch_size": batch_size,
            "num_epochs": args.num_epochs
        }
        experiment_name = wandb.run.name
    else:
        experiment_name = ''
    
    # Create dataloaders for training and test sets
    train_dataset, val_dataset, test_dataset = load_data(args)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers, timeout=300)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers, timeout=300)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers, timeout=300)
    dataloaders = (train_dataloader, val_dataloader, test_dataloader)
    
    # Define the number of classes in your dataset
    num_classes = num_attr + num_bins
    # Define the device for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CombinedModel(num_bins, num_attr, num_pt).to(device)
    model.nima_attr.load_state_dict(torch.load(args.pretrained_model))
    model = model.to(device)
    if args.resume:
        model.load_state_dict(torch.load(args.resume))

    # Define the loss functions
    criterion_mse = nn.MSELoss()

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Initialize the best test loss and the best model
    best_model = None
    best_modelname = 'best_model_resnet50_piaamir_lr%1.0e_decay_%depoch' % (args.lr, args.num_epochs)
    best_modelname += '_%s'%experiment_name
    best_modelname += '.pth'
    dirname = 'models_pth'
    if args.use_cv:
        dirname = os.path.join(dirname, 'random_cvs')
    best_modelname = os.path.join(dirname, best_modelname)

    # Training loop
    trainer(dataloaders, model, optimizer, args, train, evaluate, device, best_modelname)
    