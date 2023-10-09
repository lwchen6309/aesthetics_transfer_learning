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
from PARA_histogram_dataloader import PARA_HistogramDataset, PARA_GIAA_HistogramDataset, PARA_PIAA_HistogramDataset
from PARA_PIAA_dataloader import PARA_PIAADataset, split_dataset_by_user, split_dataset_by_images
from scipy.stats import spearmanr
import argparse
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.distances import CosineSimilarity, SNRDistance
from pytorch_metric_learning.reducers import ThresholdReducer
from pytorch_metric_learning.regularizers import LpRegularizer

from task_metric import TripletTestOneImageDataset


class TaskEmbeddingModel(nn.Module):
    def __init__(self, num_bins, num_pt, embedding_dim=128):
        super(TaskEmbeddingModel, self).__init__()
        
        self.resnet = resnet50(pretrained=True)
        self.feature_extractor = create_feature_extractor(self.resnet, return_nodes={'layer4': 'layer4', 'fc': 'fc'})
        self.num_bins = num_bins
        self.num_pt = num_pt
        self.embedding_layer = nn.Sequential(
            # nn.Linear(self.resnet.fc.in_features + self.num_bins, 512),  # Assuming the trait and target histograms are 512-dimensional each
            nn.Linear(self.resnet.fc.in_features + self.num_bins + self.num_pt, 512),  # Assuming the trait and target histograms are 512-dimensional each
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim),
        )
    
    def forward(self, image, traits_histogram, target_histogram):
        with torch.no_grad():
            output_dict = self.feature_extractor(image)
            resnet_feature = F.adaptive_avg_pool2d(output_dict['layer4'], (1,1))[:,:,0,0]

        # resnet_feature = self.resnet(image)
        x = torch.cat((resnet_feature, traits_histogram, target_histogram), dim=1)
        x = self.embedding_layer(x)
        return F.normalize(x, p=2, dim=1)  # Normalize the embedding



class TripletDataset(PARA_GIAA_HistogramDataset):
    def __init__(self, root_dir, transform=None, data=None, map_file=None, precompute_file=None):
        super().__init__(root_dir, transform, data, map_file, precompute_file)
        self.num_tasks = 9
    
    def __len__(self):
        return super().__len__() * self.num_tasks

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
        img_index, task_index = index//self.num_tasks, index%self.num_tasks
        image, traits, tasks = self.get_task_data(img_index)
        data = (image, traits, tasks[task_index])
        return data, task_index


def train_triplet(model, dataloader, criterion, miner, optimizer, device):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(dataloader, leave=False)
    for data in progress_bar:
        data_list, labels = data  # assuming your dataloader yields both data and labels
        labels = labels.to(device)
        data_list = [item.to(device) for item in data_list]

        optimizer.zero_grad()
        embeddings = model(*data_list)
        hard_pairs = miner(embeddings, labels)
        loss = criterion(embeddings, labels, hard_pairs)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        progress_bar.set_postfix({"Triplet Loss": loss.item()})
    return running_loss / len(dataloader)


def evaluate_triplet(model, dataloader, criterion, miner, device):
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0

    with torch.no_grad():  # No gradient computation during evaluation
        progress_bar = tqdm(dataloader, leave=False)
        for data in progress_bar:
            data_list, labels = data  # assuming your dataloader yields both data and labels
            labels = labels.to(device)
            data_list = [item.to(device) for item in data_list]

            embeddings = model(*data_list)
            hard_pairs = miner(embeddings, labels)
            loss = criterion(embeddings, labels, hard_pairs)
            running_loss += loss.item()
            progress_bar.set_postfix({"Eval Triplet Loss": loss.item()})
    return running_loss / len(dataloader)


def evaluate_oneimg_triplet(model, dataloader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    neg_pair_losses = [0.0 for _ in range(8)]  # Assuming 8 negative pairs
    num_batches = 0

    with torch.no_grad():  # No gradient computation during evaluation
        progress_bar = tqdm(dataloader, leave=False)
        for anchor, positive, negatives in progress_bar:
            anchor = [item.to(device) for item in anchor]
            positive = [item.to(device) for item in positive]
            
            anchor_embed = model(*anchor)
            positive_embed = model(*positive)
            
            for idx, negative in enumerate(negatives):
                negative = [item.to(device) for item in negative]
                negative_embed = model(*negative)
                metric = torch.mean(criterion(anchor_embed, negative_embed))
                neg_pair_losses[idx] += metric.item()
            num_batches += 1

    avg_neg_pair_losses = [loss / num_batches for loss in neg_pair_losses]
    return avg_neg_pair_losses


def post_evaluation(avg_neg_pair_losses_oneimg, test_loss, epoch, num_epochs, is_log=True):
    attr2score_SCORR = [0.9693, 0.9144, 0.9068, 0.9147, 0.9255, 0.9257, 0.9266, 0.9273]
    attr2score_EMD = [0.1372, 0.2125, 0.2481, 0.2387, 0.2220, 0.2028, 0.2283, 0.2485]
    srocc_oneimg_SCORR = spearmanr(avg_neg_pair_losses_oneimg, attr2score_SCORR).correlation
    srocc_oneimg_EMD = spearmanr(avg_neg_pair_losses_oneimg, attr2score_EMD).correlation

    if is_log:
        wandb.log({"Test Triplet Loss": test_loss}, commit=False)
        for idx, pair_loss in enumerate(avg_neg_pair_losses_oneimg, 1):
            log_dict[f"Neg Pair OneImg {idx} Loss"] = pair_loss
        wandb.log(log_dict, commit=False)
        wandb.log({
            "SROCC OneImg SCORR": srocc_oneimg_SCORR,
            "SROCC OneImg EMD": srocc_oneimg_EMD,
        })
    print(avg_neg_pair_losses_oneimg)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"SROCC between avg_neg_pair_losses_oneimg and attr2score_SCORR: {srocc_oneimg_SCORR:.4f}")


def load_triplet_dataset(root_dir, pkl_dir):
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
    test_piaa_dataset = PARA_PIAADataset(root_dir, transform=test_transform)
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
    train_dataset = TripletDataset(root_dir, transform=train_transform, data=train_dataset.data, 
            map_file=os.path.join(pkl_dir,'trainset_image_dct.pkl'), precompute_file=os.path.join(pkl_dir,'trainset_GIAA_dct.pkl'))
    test_dataset = TripletDataset(root_dir, transform=test_transform, data=test_dataset.data,
            map_file=os.path.join(pkl_dir,'testset_image_dct.pkl'), precompute_file=os.path.join(pkl_dir,'testset_GIAA_dct.pkl'))
    test_oneimg_dataset = TripletTestOneImageDataset(root_dir, transform=test_transform, data=test_dataset.data,
            map_file=os.path.join(pkl_dir,'testset_image_dct.pkl'), precompute_file=os.path.join(pkl_dir,'testset_GIAA_dct.pkl'))
    return train_dataset, test_dataset, test_oneimg_dataset



is_eval = False
is_log = False
num_bins = 5
num_attr = 8
num_bins_attr = 5
num_pt = 50 + 20
resume = None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model with custom margin and learning rate')
    parser.add_argument('--margin', type=float, default=0.2, help='Margin for the Triplet Loss')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate for optimizer')
    args = parser.parse_args()
    margin = args.margin
    lr = args.lr
    
    random_seed = None
    batch_size = 64
    num_epochs = 5
    lr_schedule_epochs = 1
    lr_decay_factor = 0.5
    max_patience_epochs = 10
    n_workers = 20
    
    if is_log:
        wandb.init(project="resnet_PARA_PIAA_metric")
        wandb.config = {
            "learning_rate": lr,
            "batch_size": batch_size,
            "num_epochs": num_epochs
        }
        experiment_name = wandb.run.name
    else:
        experiment_name = 'local'
    
    
    # Initialize the task embedding model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TaskEmbeddingModel(num_bins, num_pt).to(device)
    # model = TaskAttnEmbeddingModel(num_bins, num_pt, embedding_dim=512).to(device)

    train_dataset, test_dataset, test_oneimg_dataset = load_triplet_dataset(root_dir = '/home/lwchen/datasets/PARA/', pkl_dir = './dataset_pkl')
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers, pin_memory=False, timeout=300)
    test_dataloader = DataLoader(test_dataset, batch_size=20, shuffle=False, num_workers=n_workers, pin_memory=False, timeout=300)
    test_oneimg_dataloader = DataLoader(test_oneimg_dataset, batch_size=20, shuffle=False, num_workers=n_workers, pin_memory=False, timeout=300)

    if resume is not None:
        model.load_state_dict(torch.load(resume))
    
    # Loss and optimizer
    distance = CosineSimilarity()
    reducer = ThresholdReducer(low=0)
    criterion = losses.TripletMarginLoss(margin=margin, distance=distance, reducer=reducer)
    miner = miners.TripletMarginMiner(
        margin=margin, distance=distance, type_of_triplets="semihard"
    )
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Testing Init performance
    test_loss = evaluate_triplet(model, test_dataloader, criterion, miner, device)
    avg_neg_pair_losses_oneimg = evaluate_oneimg_triplet(model, test_oneimg_dataloader, distance, device)
    post_evaluation(avg_neg_pair_losses_oneimg, test_loss, -1, num_epochs, is_log)

    # Training loop
    best_test_loss = float('inf')
    best_modelname = 'best_model_triplet_%s.pth' % experiment_name
    for epoch in range(num_epochs):
        if is_eval:
            break
        
        # Learning rate schedule
        if (epoch + 1) % lr_schedule_epochs == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_decay_factor
        
        # Training
        train_loss = train_triplet(model, train_dataloader, criterion, miner, optimizer, device)
        if is_log:
            wandb.log({"Train Triplet Loss": train_loss,}, commit=False)

        # Testing
        test_loss = evaluate_triplet(model, test_dataloader, criterion, miner, device)
        avg_neg_pair_losses_oneimg = evaluate_oneimg_triplet(model, test_oneimg_dataloader, distance, device)
        post_evaluation(avg_neg_pair_losses_oneimg, test_loss, epoch, num_epochs, is_log)

        # Early stopping check
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            num_patience_epochs = 0
            torch.save(model.state_dict(), best_modelname)
        else:
            num_patience_epochs += 1
            if num_patience_epochs >= max_patience_epochs:
                print("Validation loss has not decreased for {} epochs. Stopping training.".format(max_patience_epochs))
                break
    
    if not eval:
        model.load_state_dict(torch.load(best_modelname))
    test_loss = evaluate_triplet(model, test_dataloader, criterion, miner, device)