import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50
import numpy as np
from tqdm import tqdm
from PARA_dataloader import PARADataset
from PARA_histogram_dataloader import PARA_GIAA_HistogramDataset
import wandb
from scipy.stats import spearmanr
from task_metric_learning_withcls import load_triplet_data
import argparse
from torch.optim.lr_scheduler import StepLR


class MLP(nn.Module):
    # layer_sizes[0] is the dimension of the input
    # layer_sizes[-1] is the dimension of the output
    def __init__(self, layer_sizes, image_feature_dim, num_pt, num_bins, dropout_p=0.5, final_relu=False):
        super().__init__()
        
        layer_list = []
        layer_sizes = [int(x) for x in layer_sizes]
        assert(layer_sizes[0] == image_feature_dim+256)
        num_layers = len(layer_sizes) - 1
        final_relu_layer = num_layers if final_relu else num_layers - 1

        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            curr_size = layer_sizes[i + 1]
            layer_list.append(nn.Linear(input_size, curr_size))
            if i < final_relu_layer:
                layer_list.append(nn.ReLU(inplace=False))
            layer_list.append(nn.Dropout(p=dropout_p))

        # If you don't want dropout after the last layer, you can remove the last dropout layer
        layer_list = layer_list[:-1]

        self.net = nn.Sequential(*layer_list)
        self.last_linear = self.net[-1]
        
        self.image_feature_dim = image_feature_dim
        self.num_pt = num_pt
        self.num_bins = num_bins
        self.index_split = [self.image_feature_dim, 
            self.image_feature_dim+self.num_pt, 
            self.image_feature_dim+self.num_pt+self.num_bins]

        self.task_encode = nn.Sequential(
            nn.Linear(self.num_bins, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.pt_encode = nn.Sequential(
            nn.Linear(self.num_pt, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

    def forward(self, x):
        image_feature = x[:,:self.index_split[0]]
        pt = x[:,self.index_split[0]:self.index_split[1]]
        task = x[:,self.index_split[1]:self.index_split[2]]
        x = torch.cat([image_feature, self.pt_encode(pt), self.task_encode(task)], dim=1)
        return self.net(x)


def earth_mover_distance(x, y, dim=-1):
    """
    Compute Earth Mover's Distance (EMD) between two 1D tensors x and y using 2-norm.
    """
    cdf_x = torch.cumsum(x, dim=dim)
    cdf_y = torch.cumsum(y, dim=dim)
    emd = torch.norm(cdf_x - cdf_y, p=2, dim=dim)
    return torch.mean(emd)


def train(model, train_dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(train_dataloader, leave=False)
    for images, label in progress_bar:
        if is_eval:
            raise RuntimeError("Evaluation mode activated. Exiting training.")
        
        images = images.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        
        # Cross Entropy loss
        loss = criterion(outputs, label)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        progress_bar.set_postfix({
            'Train Loss': loss.item()
        })

    epoch_loss = running_loss / len(train_dataloader)
    return epoch_loss

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    progress_bar = tqdm(dataloader, leave=False)
    with torch.no_grad():
        for images, label in progress_bar:
            images = images.to(device)
            label = label.to(device)
            
            outputs = model(images)
            
            # Predicted class labels
            _, predicted_labels = torch.max(outputs, 1)
            correct_predictions += (predicted_labels == label).sum().item()
            total_samples += images.size(0)

            # Cross Entropy loss
            loss = criterion(outputs, label)

            running_loss += loss.item()

            progress_bar.set_postfix({
                'Eval Loss': loss.item()
            })

    accuracy = correct_predictions / total_samples
    epoch_loss = running_loss / len(dataloader)
    return epoch_loss, accuracy


is_eval = False
is_log = True
resume = None
num_bins = 5
num_pt = 50 + 20


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training parameters')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate decay factor')
    parser.add_argument('--lr_decay_step', type=int, default=10, help='Number of epochs before lr decay')
    args = parser.parse_args()
    
    lr = args.lr
    lr_decay_factor = args.lr_decay
    lr_decay_step = args.lr_decay_step
    max_patience_epochs = 30

    random_seed = None
    batch_size = 256
    num_epochs = 500
    
    if is_log:
        wandb.init(project="resnet_PARA_Task")
        wandb.config = {
            "learning_rate": lr,
            "batch_size": batch_size,
            "num_epochs": num_epochs
        }
        experiment_name = wandb.run.name
    else:
        experiment_name = ''
    best_modelname = 'best_model_taskcls_%s.pth' % experiment_name

    root_dir = '/home/lwchen/datasets/PARA/'
    train_dataset, test_dataset = load_triplet_data(root_dir)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    num_classes = 9
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = 128 + 128 + 2048
    model = torch.nn.DataParallel(MLP(
        [input_dim, 1024, 1024, 512, 512, num_classes], 
        image_feature_dim=2048, num_pt=num_pt, num_bins=num_bins, dropout_p=0.5))

    if resume is not None:
        model.load_state_dict(torch.load(resume))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay_factor)

    best_test_loss = float('inf')
    num_patience_epochs = 0
    for epoch in range(num_epochs):
        train_loss = train(model, train_dataloader, criterion, optimizer, device)
        test_loss, accuracy = evaluate(model, test_dataloader, criterion, device)

        if is_log:
            wandb.log({
                "Train Loss": train_loss,
                "Accuracy": accuracy
            })

        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, "
              f"Test Loss: {test_loss:.4f}, "
              f"Accuracy: {accuracy:.4f}")

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            num_patience_epochs = 0
            # torch.save(model.state_dict(), best_modelname)
        else:
            num_patience_epochs += 1
            if num_patience_epochs >= max_patience_epochs:
                print(f"Validation loss has not decreased for {max_patience_epochs} epochs. Stopping training.")
                break

        # Adjust learning rate
        scheduler.step()