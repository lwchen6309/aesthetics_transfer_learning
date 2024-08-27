import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import resnet50
from torch.utils.data import DataLoader
# from tqdm import tqdm
# from PARA_PIAA_dataloader import collect_batch_attribute, collect_batch_personal_trait
from PARA_PIAA_dataloader import load_data as piaa_load_data
from PARA_histogram_dataloader import load_data, collate_fn_imgsort
import wandb
# from scipy.stats import spearmanr
from train_nima_attr import NIMA_attr
from utils.argflags import parse_arguments_piaa, wandb_tags, model_dir
from train_piaa_mir import train, evaluate, evaluate_with_prior, train_piaa, evaluate_piaa, trainer, trainer_piaa


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
    