import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet50
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm
import wandb
from itertools import chain
from scipy.stats import spearmanr
from PARA_histogram_dataloader import PARA_HistogramDataset, PARA_GIAA_HistogramDataset, PARA_PIAA_HistogramDataset
from PARA_PIAA_dataloader import PARA_PIAADataset, split_dataset_by_user, split_dataset_by_images
from train_resnet_cls import earth_mover_distance


class TaskEmbeddingModel(nn.Module):
    def __init__(self, num_bins, num_pt, embedding_dim=128):
        super(TaskEmbeddingModel, self).__init__()
        
        self.resnet = resnet50(pretrained=True)
        self.feature_extractor = create_feature_extractor(self.resnet, return_nodes={'layer4': 'layer4', 'fc': 'fc'})
        self.num_bins = num_bins
        self.num_pt = num_pt
        self.embedding_layer = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features + self.num_bins + self.num_pt, 512),  # Assuming the trait and target histograms are 512-dimensional each
            nn.ReLU(),
            nn.Linear(512, embedding_dim)
        )
    
    def forward(self, image, traits_histogram, target_histogram):
        with torch.no_grad():
            # context = model_resnet(images)
            output_dict = self.feature_extractor(image)
            resnet_feature = F.adaptive_avg_pool2d(output_dict['layer4'], (1,1))[:,:,0,0]
        # resnet_feature = self.resnet(image)
        x = torch.cat((resnet_feature, traits_histogram, target_histogram), dim=1)
        x = self.embedding_layer(x)
        return F.normalize(x, p=2, dim=1)  # Normalize the embedding

class TripletLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)
        loss = torch.mean(torch.clamp(pos_dist - neg_dist + self.margin, min=0.0))
        return loss

class TripletDataset(PARA_HistogramDataset):
    def __init__(self, root_dir, transform=None, data=None, map_file=None):
        super().__init__(root_dir, transform, data, map_file)

    def __len__(self):
        return super().__len__()

    def get_task_data(self, index):
        sample = super().__getitem__(index)
        
        images = sample['image']
        aesthetic_score_histogram = sample['aestheticScore']
        aesthetic_score_histogram = torch.cat([aesthetic_score_histogram[0].unsqueeze(0), 
                                              aesthetic_score_histogram[1:].reshape(-1,2).sum(dim=1)], dim=0)
        attributes_histogram = sample['attributes']
        total_task = torch.cat([aesthetic_score_histogram.unsqueeze(0), attributes_histogram.reshape(-1,5)])

        traits_histogram = sample['traits']
        onehot_traits_histogram = sample['onehot_traits']
        traits_histogram = torch.cat([traits_histogram, onehot_traits_histogram], dim=0)
        
        return images, traits_histogram, total_task

    def __getitem__(self, index):
        # Get the task data for the provided index
        anchor_image, anchor_traits, anchor_task = self.get_task_data(index)
        
        # Randomly sample another index different from the current index
        random_index = index
        while random_index == index:
            random_index = random.choice(range(0, len(self)))
        
        # Get the task data for the randomly sampled index
        random_image, random_traits, random_task = self.get_task_data(random_index)
        
        # Randomly sample two different indices out of the length of `positive_task`
        i, j = random.sample(range(len(anchor_task)), 2)
        
        # Set anchor and positive based on index i
        anchor = (anchor_image, anchor_traits, anchor_task[i])
        positive = (random_image, random_traits, random_task[i])
        
        # Choose negative either from random_task[j] or anchor_task[j]
        if random.choice([True, False]):
            negative = (random_image, random_traits, random_task[j])
        else:
            negative = (anchor_image, anchor_traits, anchor_task[j])
        
        return anchor, positive, negative


def train_triplet(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(dataloader, leave=False)
    for anchor, positive, negative in progress_bar:
        anchor = [item.to(device) for item in anchor]
        positive = [item.to(device) for item in positive]
        negative = [item.to(device) for item in negative]
        
        optimizer.zero_grad()
        anchor_embed = model(*anchor)
        positive_embed = model(*positive)
        negative_embed = model(*negative)
        loss = criterion(anchor_embed, positive_embed, negative_embed)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        progress_bar.set_postfix({"Triplet Loss": loss.item()})
    return running_loss / len(dataloader)


def evaluate_triplet(model, dataloader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    with torch.no_grad():  # No gradient computation during evaluation
        progress_bar = tqdm(dataloader, leave=False)
        for anchor, positive, negative in progress_bar:
            anchor = [item.to(device) for item in anchor]
            positive = [item.to(device) for item in positive]
            negative = [item.to(device) for item in negative]
            
            anchor_embed = model(*anchor)
            positive_embed = model(*positive)
            negative_embed = model(*negative)
            loss = criterion(anchor_embed, positive_embed, negative_embed)
            running_loss += loss.item()
            progress_bar.set_postfix({"Eval Triplet Loss": loss.item()})
    return running_loss / len(dataloader)



attr_mask = 0
is_eval = False
is_log = False
num_bins = 5
num_attr = 8
num_bins_attr = 5
num_pt = 50 + 20
resume = None
criterion_mse = nn.MSELoss()


if __name__ == '__main__':
    random_seed = 42
    lr = 5e-5
    batch_size = 10
    num_epochs = 20
    lr_schedule_epochs = 5
    lr_decay_factor = 0.5
    max_patience_epochs = 10
    
    if is_log:
        wandb.init(project="resnet_PARA_PIAA")
        wandb.config = {
            "learning_rate": lr,
            "batch_size": batch_size,
            "num_epochs": num_epochs
        }
        experiment_name = wandb.run.name
    else:
        experiment_name = ''
    
    root_dir = '/home/lwchen/datasets/PARA/'
    
    # Dataset transformations
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    # Load datasets
    # Create datasets with the appropriate transformations
    train_piaa_dataset = PARA_PIAADataset(root_dir, transform=train_transform)
    test_piaa_dataset = PARA_PIAADataset(root_dir, transform=train_transform)
    train_dataset, test_dataset = split_dataset_by_images(train_piaa_dataset, test_piaa_dataset, root_dir)
    _, test_user_piaa_dataset = split_dataset_by_user(
        PARA_PIAADataset(root_dir, transform=train_transform), 
        PARA_PIAADataset(root_dir, transform=train_transform), 
        test_count=40, max_annotations_per_user=50, seed=random_seed)
    
    # Create datasets with the appropriate transformations
    train_piaa_dataset = PARA_PIAADataset(root_dir, transform=train_transform)
    test_piaa_dataset = PARA_PIAADataset(root_dir, transform=train_transform)
    train_dataset, test_dataset = split_dataset_by_images(train_piaa_dataset, test_piaa_dataset, root_dir)
    
    # Turn the datasets into triplet datasets for contrastive learning
    train_dataset = TripletDataset(root_dir, transform=train_transform, data=train_dataset.data, map_file='trainset_image_dct.pkl')
    test_dataset = TripletDataset(root_dir, transform=test_transform, data=test_dataset.data, map_file='testset_image_dct.pkl')
    
    # Create dataloaders
    n_workers = 4
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the task embedding model
    model = TaskEmbeddingModel(num_bins, num_pt).to(device)
    if resume is not None:
        model.load_state_dict(torch.load(resume))
    
    # Loss and optimizer
    criterion = TripletLoss(margin=0.5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    best_test_loss = float('inf')
    for epoch in range(num_epochs):
        if is_eval:
            break
        
        # Learning rate schedule
        if (epoch + 1) % lr_schedule_epochs == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_decay_factor
        
        # Training
        train_loss = train_triplet(model, train_dataloader, criterion, optimizer, device)
        if is_log:
            wandb.log({"Train Triplet Loss": train_loss,}, commit=False)
        
        # Testing
        test_loss = evaluate_triplet(model, test_dataloader, criterion, device)
        if is_log:
            wandb.log({"Test Triplet Loss": test_loss,}, commit=True)

        # Print the epoch loss
        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, "
              f"Test Loss: {test_loss:.4f}")

        # Early stopping check
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            num_patience_epochs = 0
            best_modelname = 'best_model_triplet_%s.pth' % experiment_name
            torch.save(model.state_dict(), best_modelname)
        else:
            num_patience_epochs += 1
            if num_patience_epochs >= max_patience_epochs:
                print("Validation loss has not decreased for {} epochs. Stopping training.".format(max_patience_epochs))
                break
