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
        # x = torch.cat((resnet_feature, target_histogram), dim=1)
        x = torch.cat((resnet_feature, traits_histogram, target_histogram), dim=1)
        # x = target_histogram
        x = self.embedding_layer(x)
        return F.normalize(x, p=2, dim=1)  # Normalize the embedding


class CrossAttention(nn.Module):
    def __init__(self, query_dim, key_value_dim, embedding_dim):
        super(CrossAttention, self).__init__()
        self.query_proj = nn.Linear(query_dim, embedding_dim)
        self.key_proj = nn.Linear(key_value_dim, embedding_dim)
        self.value_proj = nn.Linear(key_value_dim, embedding_dim)

    def forward(self, query, key, value):
        query = self.query_proj(query)
        key = self.key_proj(key)
        value = self.value_proj(value)

        attn_weights = F.softmax(query @ key.transpose(-2, -1), dim=-1)  # [B, 1, num_traits+num_hist]
        attn_output = attn_weights @ value  # [B, 1, embedding_dim]

        return attn_output.squeeze(1)  # [B, embedding_dim]


class TaskAttnEmbeddingModel(nn.Module):
    def __init__(self, num_bins, num_pt, embedding_dim=128):
        super(TaskAttnEmbeddingModel, self).__init__()

        self.resnet = resnet50(pretrained=True)
        self.feature_extractor = create_feature_extractor(self.resnet, return_nodes={'layer4': 'layer4', 'fc': 'fc'})

        self.cross_attention = CrossAttention(self.resnet.fc.in_features, num_bins + num_pt, embedding_dim)

    def forward(self, image, traits_histogram, target_histogram):
        with torch.no_grad():
            output_dict = self.feature_extractor(image)
            resnet_feature = F.adaptive_avg_pool2d(output_dict['layer4'], (1, 1))[:, :, 0, 0]

        combined_key_value = torch.cat((traits_histogram, target_histogram), dim=1)
        embedding = self.cross_attention(resnet_feature.unsqueeze(1), combined_key_value, combined_key_value)

        return F.normalize(embedding, p=2, dim=1)  # Normalize the embedding


class TripletLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)
        loss = torch.mean(torch.clamp(pos_dist - neg_dist + self.margin, min=0.0))
        return loss


class TripletDataset(PARA_GIAA_HistogramDataset):
    def __init__(self, root_dir, transform=None, data=None, map_file=None, precompute_file=None):
        super().__init__(root_dir, transform, data, map_file, precompute_file)
    
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
        anchor_image, anchor_traits, anchor_tasks = self.get_task_data(index)
        
        all_indices = list(range(len(self)))
        all_indices.remove(index)
        sampled_indices = random.sample(all_indices, 2)
        pos_i, neg_i = sampled_indices
        positive_image, positive_traits, positive_tasks = self.get_task_data(pos_i)
        negative_image, negative_traits, negative_tasks = self.get_task_data(neg_i)

        i, j = random.sample(range(len(anchor_task)), 2)
        anchor = (anchor_image, anchor_traits, anchor_tasks[i])
        positive = (positive_image, positive_traits, positive_tasks[i])
        negative = (negative_image, negative_traits, negative_tasks[j])
        
        return anchor, positive, negative


class TripletTestDataset(TripletDataset):
    def __getitem__(self, index):
        anchor_image, anchor_traits, anchor_tasks = self.get_task_data(index)
        
        all_indices = list(range(len(self)))
        all_indices.remove(index)
        sampled_indices = random.sample(all_indices, 2)
        pos_i, neg_i = sampled_indices
        positive_image, positive_traits, positive_tasks = self.get_task_data(pos_i)
        negative_image, negative_traits, negative_tasks = self.get_task_data(neg_i)

        anchor = (anchor_image, anchor_traits, anchor_tasks[0])
        positive = (positive_image, positive_traits, positive_tasks[0])
        negative = [(negative_image, negative_traits, negative_task) for negative_task in negative_tasks[1:]]

        return anchor, positive, negative


class TripletTrainOneImageDataset(TripletDataset):
    def __getitem__(self, index):
        # Get the task data for the provided index
        anchor_image, anchor_traits, anchor_task = self.get_task_data(index)
        i, j = random.sample(range(len(anchor_task)), 2)
        anchor = (anchor_image, anchor_traits, anchor_task[i])
        positive = (anchor_image, anchor_traits, anchor_task[i])
        negative = (anchor_image, anchor_traits, anchor_task[j])
        return anchor, positive, negative


class TupletDataset(TripletDataset):
    def __getitem__(self, index):
        anchor_image, anchor_traits, anchor_task = self.get_task_data(index)
        
        all_indices = list(range(len(self)))
        all_indices.remove(index)
        sampled_indices = random.sample(all_indices, 2)
        pos_i, neg_i = sampled_indices

        positive_image, positive_traits, positive_task = self.get_task_data(pos_i)
        negative_image, negative_traits, negative_task = self.get_task_data(neg_i)
        
        i = random.sample(range(len(anchor_task)), 1)[0]
        js =[j for j in range(len(anchor_task)) if j != i]
        anchor = (anchor_image, anchor_traits, anchor_task[i])
        positive = (positive_image, positive_traits, positive_task[i])
        negatives = [(negative_image, negative_traits, negative_task[j]) for j in js]
        return anchor, positive, negatives


class TripletTestOneImageDataset(TripletDataset):
    def __getitem__(self, index):
        # Get the task data for the provided index
        anchor_image, anchor_traits, anchor_task = self.get_task_data(index)

        anchor = (anchor_image, anchor_traits, anchor_task[0])
        positive = (anchor_image, anchor_traits, anchor_task[0])
        negative = [(anchor_image, anchor_traits, anchor) for anchor in anchor_task[1:]]
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


def train_tuplet(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(dataloader, leave=False)
    
    for anchor, positive, negatives in progress_bar:
        anchor = [item.to(device) for item in anchor]
        positive = [item.to(device) for item in positive]
        
        optimizer.zero_grad()
        
        anchor_embed = model(*anchor)
        positive_embed = model(*positive)
        
        loss = 0
        for negative in negatives:
            negative = [item.to(device) for item in negative]
            negative_embed = model(*negative)
            loss += criterion(anchor_embed, positive_embed, negative_embed)
        loss /= len(negatives)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        progress_bar.set_postfix({"Tuplet Loss": loss.item()})
        
    return running_loss / len(dataloader)


def train_triplet_negative_hardmining(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(dataloader, leave=False)
    
    for anchor, positive, negatives in progress_bar:
        anchor = [item.to(device) for item in anchor]
        positive = [item.to(device) for item in positive]
        
        optimizer.zero_grad()
        
        anchor_embed = model(*anchor)
        positive_embed = model(*positive)
        
        # Find the hardest negative sample based on pairwise distance
        min_distance = float('inf')  # Initialize with a very high value
        hardest_negative_embed = None
        
        for negative in negatives:
            negative = [item.to(device) for item in negative]
            negative_embed = model(*negative)
            
            distance = torch.mean(F.pairwise_distance(anchor_embed, negative_embed)).item()
            
            if distance < min_distance:
                min_distance = distance
                hardest_negative_embed = negative_embed
        
        # Backpropagate based on the hardest negative sample
        loss = criterion(anchor_embed, positive_embed, hardest_negative_embed)
        loss.backward()
        
        optimizer.step()
        running_loss += loss.item()
        progress_bar.set_postfix({"Triplet Loss": loss.item()})
        
    return running_loss / len(dataloader)


def evaluate_triplet(model, dataloader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    neg_pair_losses = [0.0 for _ in range(8)]  # Assuming 8 negative pairs
    num_batches = 0

    with torch.no_grad():  # No gradient computation during evaluation
        progress_bar = tqdm(dataloader, leave=False)
        for anchor, positive, negatives in progress_bar:
            anchor = [item.to(device) for item in anchor]
            positive = [item.to(device) for item in positive]
            
            anchor_embed = model(*anchor)
            positive_embed = model(*positive)
            
            batch_losses = [] # Store the batch losses for each negative
            for idx, negative in enumerate(negatives):
                negative = [item.to(device) for item in negative]
                negative_embed = model(*negative)
                loss = criterion(anchor_embed, positive_embed, negative_embed)

                metric = torch.mean(F.pairwise_distance(anchor_embed, negative_embed))
                batch_losses.append(loss.item())
                neg_pair_losses[idx] += metric.item()
            
            # Average the losses from all the negatives
            avg_loss = sum(batch_losses) / len(batch_losses)
            running_loss += avg_loss
            progress_bar.set_postfix({"Eval Triplet Loss": avg_loss})
            num_batches += 1

    avg_neg_pair_losses = [loss / num_batches for loss in neg_pair_losses]
    return avg_neg_pair_losses, running_loss / len(dataloader)


def post_evaluation(avg_neg_pair_losses, avg_neg_pair_losses_oneimg, test_loss, test_loss_oneimg, epoch, num_epochs, is_log=True):
    attr2score_SCORR = [0.9693, 0.9144, 0.9068, 0.9147, 0.9255, 0.9257, 0.9266, 0.9273]
    attr2score_EMD = [0.1372, 0.2125, 0.2481, 0.2387, 0.2220, 0.2028, 0.2283, 0.2485]
    srocc_SCORR = spearmanr(avg_neg_pair_losses, attr2score_SCORR).correlation
    srocc_EMD = spearmanr(avg_neg_pair_losses, attr2score_EMD).correlation
    srocc_oneimg_SCORR = spearmanr(avg_neg_pair_losses_oneimg, attr2score_SCORR).correlation
    srocc_oneimg_EMD = spearmanr(avg_neg_pair_losses_oneimg, attr2score_EMD).correlation

    if is_log:
        log_dict = {"Test Triplet Loss": test_loss}
        for idx, pair_loss in enumerate(avg_neg_pair_losses, 1):
            log_dict[f"Neg Pair {idx} Loss"] = pair_loss
        wandb.log(log_dict, commit=False)        
        
        log_dict = {"Test OneImg Triplet Loss": test_loss_oneimg}
        for idx, pair_loss in enumerate(avg_neg_pair_losses_oneimg, 1):
            log_dict[f"Neg Pair OneImg {idx} Loss"] = pair_loss
        wandb.log(log_dict, commit=False)

        wandb.log({
            "SROCC OneImg SCORR": srocc_oneimg_SCORR,
            "SROCC OneImg EMD": srocc_oneimg_EMD,
            "SROCC SCORR": srocc_SCORR,
            "SROCC EMD": srocc_EMD
        })

    print(f"Epoch [{epoch + 1}/{num_epochs}], "
          f"Test OneImg Loss: {test_loss_oneimg:.4f}")
    print(avg_neg_pair_losses_oneimg)
    print(f"Test Loss: {test_loss:.4f}")
    print(avg_neg_pair_losses)
    print(f"SROCC between avg_neg_pair_losses_oneimg and attr2score_SCORR: {srocc_oneimg_SCORR:.4f}")
    print(f"SROCC between avg_neg_pair_losses and attr2score_SCORR: {srocc_SCORR:.4f}")


is_eval = False
is_log = True
num_bins = 5
num_attr = 8
num_bins_attr = 5
num_pt = 50 + 20
resume = None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model with custom margin and learning rate')
    parser.add_argument('--margin', type=float, default=0.5, help='Margin for the Triplet Loss')
    parser.add_argument('--lr', type=float, default=5e-2, help='Learning rate for optimizer')
    args = parser.parse_args()
    margin = args.margin
    lr = args.lr
    
    random_seed = None
    # lr = 5e-2
    batch_size = 64
    num_epochs = 5
    lr_schedule_epochs = 10
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
    
    # Initialize the task embedding model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TaskEmbeddingModel(num_bins, num_pt).to(device)
    # model = TaskAttnEmbeddingModel(num_bins, num_pt, embedding_dim=512).to(device)
    
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
    pkl_dir = './dataset_pkl'
    train_dataset = TupletDataset(root_dir, transform=train_transform, data=train_dataset.data, 
            map_file=os.path.join(pkl_dir,'trainset_image_dct.pkl'), precompute_file=os.path.join(pkl_dir,'trainset_GIAA_dct.pkl'))
    test_dataset = TripletTestDataset(root_dir, transform=test_transform, data=test_dataset.data,
            map_file=os.path.join(pkl_dir,'testset_image_dct.pkl'), precompute_file=os.path.join(pkl_dir,'testset_GIAA_dct.pkl'))
    test_oneimg_dataset = TripletTestOneImageDataset(root_dir, transform=test_transform, data=test_dataset.data,
            map_file=os.path.join(pkl_dir,'testset_image_dct.pkl'), precompute_file=os.path.join(pkl_dir,'testset_GIAA_dct.pkl'))

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers, pin_memory=False, timeout=300)
    test_dataloader = DataLoader(test_dataset, batch_size=20, shuffle=False, num_workers=n_workers, pin_memory=False, timeout=300)
    test_oneimg_dataloader = DataLoader(test_oneimg_dataset, batch_size=20, shuffle=False, num_workers=n_workers, pin_memory=False, timeout=300)

    if resume is not None:
        model.load_state_dict(torch.load(resume))
    
    # Loss and optimizer
    criterion = TripletLoss(margin=margin).to(device)
    mse_criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Testing Init performance
    # avg_neg_pair_losses, test_loss = evaluate_triplet(model, test_dataloader, criterion, device)
    # avg_neg_pair_losses_oneimg, test_loss_oneimg = evaluate_triplet(model, test_oneimg_dataloader, criterion, device)
    # post_evaluation(avg_neg_pair_losses, avg_neg_pair_losses_oneimg, test_loss, test_loss_oneimg, -1, num_epochs,is_log)
    
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
        # train_loss = train_triplet(model, train_dataloader, criterion, optimizer, device)
        # train_loss = train_triplet_negative_hardmining(model, train_dataloader, criterion, optimizer, device)
        train_loss = train_tuplet(model, train_dataloader, criterion, optimizer, device)
        if is_log:
            wandb.log({"Train Triplet Loss": train_loss,}, commit=False)

        # Testing
        avg_neg_pair_losses, test_loss = evaluate_triplet(model, test_dataloader, criterion, device)
        avg_neg_pair_losses_oneimg, test_loss_oneimg = evaluate_triplet(model, test_oneimg_dataloader, criterion, device)
        post_evaluation(avg_neg_pair_losses, avg_neg_pair_losses_oneimg, test_loss, test_loss_oneimg, epoch, num_epochs, is_log)

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

    # Evaluating on test_dataloader
    avg_neg_pair_losses, test_loss = evaluate_triplet(model, test_dataloader, criterion, device)
    avg_neg_pair_losses_oneimg, test_loss_oneimg = evaluate_triplet(model, test_oneimg_dataloader, criterion, device)
    post_evaluation(avg_neg_pair_losses, avg_neg_pair_losses_oneimg, test_loss, test_loss_oneimg, epoch, num_epochs, is_log)
