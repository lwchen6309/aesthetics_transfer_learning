import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
# from LAPIS_PIAA_dataloader import load_data as piaa_load_data
# from LAPIS_PIAA_dataloader import collate_fn as piaa_collate_fn
from torchvision.models import resnet50, mobilenet_v2, resnet18, swin_v2_t, swin_v2_s
from LAPIS_histogram_dataloader import load_data, collate_fn, collate_fn_imgsort
import wandb
from scipy.stats import spearmanr
# from train_piaa_mir import PIAA_MIR
# from train_piaa_mir import trainer, trainer_piaa
from utils.argflags import parse_arguments_piaa, wandb_tags, model_dir


class NIMA_attr(nn.Module):
    def __init__(self, num_bins_aesthetic, num_attr, backbone='resnet50'):
        super(NIMA_attr, self).__init__()
        
        # Choose model backbone based on the input argument
        if backbone == 'resnet50':
            self.backbone = resnet50(pretrained=True)
            backbone_out_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove the last fully connected layer
        
        elif backbone == 'mobilenet_v2':
            self.backbone = mobilenet_v2(pretrained=True)
            backbone_out_features = self.backbone.last_channel
            self.backbone.classifier = nn.Identity()  # Remove the classifier

        elif backbone == 'resnet18':
            self.backbone = resnet18(pretrained=True)
            backbone_out_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove the last fully connected layer

        elif backbone == 'swin_v2_t':
            self.backbone = swin_v2_t(pretrained=True)
            backbone_out_features = self.backbone.head.in_features
            self.backbone.head = nn.Identity()  # Remove the head

        elif backbone == 'swin_v2_s':
            self.backbone = swin_v2_s(pretrained=True)
            backbone_out_features = self.backbone.head.in_features
            self.backbone.head = nn.Identity()  # Remove the head

        else:
            raise ValueError(f"Model {backbone} is not supported.")

        # Aesthetic score prediction head
        self.fc_aesthetic = nn.Sequential(
            nn.Linear(backbone_out_features, 512),
            nn.ReLU(),
            nn.Linear(512, num_bins_aesthetic)
        )
        
        # Attribute prediction head
        self.fc_attributes = nn.Sequential(
            nn.Linear(backbone_out_features, 512),
            nn.ReLU(),
            nn.Linear(512, num_attr)
        )

    def forward(self, images):
        # Pass images through the backbone
        x = self.backbone(images)
        
        # Aesthetic and attribute predictions
        aesthetic_logits = self.fc_aesthetic(x)
        attribute_logits = self.fc_attributes(x)
        
        return aesthetic_logits, attribute_logits

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
    def __init__(self, num_bins, num_attr, num_pt, hidden_size=1024, dropout=None, backbone='resnet-50'):
        super(PIAA_MIR, self).__init__()
        self.num_bins = num_bins
        self.num_attr = num_attr
        self.num_pt = num_pt
        self.scale = torch.arange(1, 5.5, 0.5)  # This will be moved to the correct device in forward()
        
        # Placeholder for the NIMA_attr model
        self.nima_attr = NIMA_attr(num_bins, num_attr, backbone=backbone)  # Make sure to define or import NIMA_attr
        
        # Interaction MLPs
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        self.mlp1 = MLP(num_attr * num_pt, hidden_size, 1)
        self.mlp2 = MLP(num_bins, hidden_size, 1)

    def forward(self, images, personal_traits):
        logit, attr_mean_pred = self.nima_attr(images)
        prob = F.softmax(logit, dim=1)
        
        # Interaction map calculation
        A_ij = attr_mean_pred.unsqueeze(2) * personal_traits.unsqueeze(1)
        I_ij = A_ij.view(images.size(0), -1)
        if self.dropout > 0:
            I_ij = self.dropout_layer(I_ij)
        interaction_outputs = self.mlp1(I_ij)
        direct_outputs = self.mlp2(prob * self.scale.to(images.device))
        
        return interaction_outputs + direct_outputs

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

def trainer_piaa(dataloaders, model, optimizer, args, train_fn, evaluate_fn, device, best_modelname):

    train_dataloader, val_dataloader, test_dataloader = dataloaders
    
    if args.resume is not None:
        test_mse_loss, test_srocc = evaluate_fn(model, test_dataloader, criterion_mse, device, args)
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
        train_mse_loss = train_fn(model, train_dataloader, criterion_mse, optimizer, device, args)
        # Testing
        val_mse_loss, val_srocc = evaluate_fn(model, val_dataloader, criterion_mse, device, args)

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
    test_mse_loss, test_srocc = evaluate_fn(model, test_dataloader, criterion_mse, device, args)
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
    parser.add_argument('--backbone', type=str, default='resnet50')
    args = parser.parse_args()
    args.disable_onehot = True
    print(args)
    
    num_bins = 9
    num_attr = 8
    num_pt = 71
    
    if args.is_log:
        tags = ["no_attr"]
        tags += wandb_tags(args)
        wandb.init(project="LAPIS_IAA", 
                   notes='PIAA-MIR-'+args.backbone,
                   tags = tags,
                   entity='KULeuven-GRAPPA',
                   )
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
    dataloaders = (train_dataloader, val_piaa_imgsort_dataloader, test_piaa_imgsort_dataloader)
    
    # Define the number of classes in your dataset
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PIAA_MIR(num_bins, num_attr, num_pt, dropout=args.dropout, backbone=args.backbone).to(device)
    best_modelname = f'lapis_{args.backbone}_piaamir_{experiment_name}.pth'
    
    if args.pretrained_model:
        model.nima_attr.load_state_dict(torch.load(args.pretrained_model))
    if args.resume:
        model.load_state_dict(torch.load(args.resume))
    model = model.to(device)
    
    # Define the optimizer
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # Initialize the best test loss and the best model
    dirname = model_dir(args)
    best_modelname = os.path.join(dirname, best_modelname)
    
    trainer_piaa(dataloaders, model, optimizer, args, train, evaluate, device, best_modelname)
