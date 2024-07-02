import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet50
import numpy as np
from tqdm import tqdm
import wandb
from scipy.stats import spearmanr
from LAPIS_histogram_dataloader import load_data, collate_fn_imgsort, collate_fn
from train_histonet_latefusion import trainer, CombinedModel
from utils.losses import EarthMoverDistance
earth_mover_distance = EarthMoverDistance()
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from utils.argflags import parse_arguments


class CombinedModel_earlyfusion(nn.Module):
    def __init__(self, num_bins_aesthetic, num_attr, num_bins_attr, num_pt):
        super(CombinedModel_earlyfusion, self).__init__()
        self.resnet = resnet50(pretrained=True)
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 512),
            nn.ReLU(),
            # nn.Linear(512, num_bins_aesthetic),
        )
        self.num_bins_aesthetic = num_bins_aesthetic
        self.num_attr = num_attr
        # self.num_bins_attr = num_bins_attr
        self.num_pt = num_pt
        # For predicting attribute histograms for each attribute
        # self.inv_coef_decoder = nn.Sequential(
        #     nn.Linear(num_pt + num_bins_aesthetic, 512),
        #     nn.ReLU(),   
        #     nn.Linear(512, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 20)
        # )
        
        # For predicting aesthetic score histogram
        self.fc_aesthetic = nn.Sequential(
            nn.Linear(512 + num_pt, 512),
            nn.ReLU(),
            nn.Linear(512, num_bins_aesthetic)
        )
    
    def forward(self, images, traits_histogram):
        x = self.resnet(images)
        aesthetic_logits = self.fc_aesthetic(torch.cat([x, traits_histogram], dim=1))
        # coef = self.inv_coef_decoder(torch.cat([aesthetic_logits, traits_histogram], dim=1))
        return aesthetic_logits #, coef



# Training Function
def train(model, dataloader, optimizer, device):
    model.train()
    running_aesthetic_emd_loss = 0.0
    running_total_emd_loss = 0.0
    progress_bar = tqdm(dataloader, leave=False)
    units_len = dataloader.dataset.traits_len()
    for sample in progress_bar:        
        images = sample['image'].to(device)
        aesthetic_score_histogram = sample['aestheticScore'].to(device)
        traits_histogram = sample['traits'].to(device)
        art_type = sample['art_type'].to(device)
        # traits_histogram = torch.cat([traits_histogram, art_type], dim=1)
        
        optimizer.zero_grad()
        aesthetic_logits = model(images, traits_histogram)

        prob_aesthetic = F.softmax(aesthetic_logits, dim=1)
        loss_aesthetic = earth_mover_distance(prob_aesthetic, aesthetic_score_histogram).mean()
        total_loss = loss_aesthetic

        total_loss.backward()
        optimizer.step()
        running_total_emd_loss += total_loss.item()
        running_aesthetic_emd_loss += loss_aesthetic.item()

        progress_bar.set_postfix({
            'Train EMD Loss': loss_aesthetic.item(),
            # 'Train sCE Loss': mean_entropy_piaa.item(),
        })
    
    epoch_emd_loss = running_aesthetic_emd_loss / len(dataloader)
    epoch_total_emd_loss = running_total_emd_loss / len(dataloader)
    return epoch_emd_loss, epoch_total_emd_loss

# Evaluation Function
def evaluate(model, dataloader, device):
    model.eval()
    running_emd_loss = 0.0
    running_attr_emd_loss = 0.0
    running_mse_loss = 0.0
    # scale = torch.arange(1, 5.5, 0.5).to(device)
    scale = torch.arange(0, 10).to(device)
    
    progress_bar = tqdm(dataloader, leave=False)
    mean_pred = []
    mean_target = []
    with torch.no_grad():
        for sample in progress_bar:
            images = sample['image'].to(device)
            aesthetic_score_histogram = sample['aestheticScore'].to(device)
            traits_histogram = sample['traits'].to(device)
            art_type = sample['art_type'].to(device)
            # traits_histogram = torch.cat([traits_histogram, art_type], dim=1)

            # aesthetic_logits, _ = model(images, traits_histogram)
            aesthetic_logits = model(images, traits_histogram)
            prob_aesthetic = F.softmax(aesthetic_logits, dim=1)
            # prob_attribute = F.softmax(attribute_logits, dim=-1) # Softmax along the bins dimension
            loss = earth_mover_distance(prob_aesthetic, aesthetic_score_histogram).mean()
            # loss_attribute = earth_mover_distance(prob_attribute, attributes_target_histogram).mean()
            
            outputs_mean = torch.sum(prob_aesthetic * scale, dim=1, keepdim=True)
            target_mean = torch.sum(aesthetic_score_histogram * scale, dim=1, keepdim=True)
            mean_pred.append(outputs_mean.view(-1).cpu().numpy())
            mean_target.append(target_mean.view(-1).cpu().numpy())
            # MSE
            mse = criterion_mse(outputs_mean, target_mean)
            running_mse_loss += mse.item()
            
            running_emd_loss += loss.item()
            # running_attr_emd_loss += loss_attribute.item()
            progress_bar.set_postfix({
                'Test EMD Loss': loss.item(),
            })
    
    # Calculate SROCC
    predicted_scores = np.concatenate(mean_pred, axis=0)
    true_scores = np.concatenate(mean_target, axis=0)
    srocc, _ = spearmanr(predicted_scores, true_scores)
    
    emd_loss = running_emd_loss / len(dataloader)
    emd_attr_loss = running_attr_emd_loss / len(dataloader)
    mse_loss = running_mse_loss / len(dataloader)
    return emd_loss, emd_attr_loss, srocc, mse_loss


def evaluate_with_prior(model, dataloader, prior_dataloader, device):
    model.eval()
    running_emd_loss = 0.0
    running_attr_emd_loss = 0.0
    running_mse_loss = 0.0
    # scale = torch.arange(1, 5.5, 0.5).to(device)
    scale = torch.arange(0, 10).to(device)
    eval_srocc = True
    progress_bar = tqdm(dataloader, leave=False)
    mean_pred = []
    mean_target = []
    traits_histograms = []
    with torch.no_grad():
        for sample in tqdm(prior_dataloader, leave=False):
            # images = sample['image'].to(device)
            # aesthetic_score_histogram = sample['aestheticScore'].to(device)
            traits_histogram = sample['traits'].to(device)
            traits_histograms.append(traits_histogram)
        mean_traits_histogram = torch.mean(torch.cat(traits_histograms, dim=0), dim=0)

        for sample in progress_bar:
            images = sample['image'].to(device)
            aesthetic_score_histogram = sample['aestheticScore'].to(device)
            # traits_histogram = sample['traits'].to(device)
            # art_type = sample['art_type'].to(device)
            # traits_histogram = torch.cat([traits_histogram, art_type], dim=1)
            batch_size = images.shape[0]
            traits_histogram = mean_traits_histogram.repeat(batch_size, 1)
            
            aesthetic_logits = model(images, traits_histogram)
            prob_aesthetic = F.softmax(aesthetic_logits, dim=1)
            loss = earth_mover_distance(prob_aesthetic, aesthetic_score_histogram).mean()
            
            if eval_srocc:
                outputs_mean = torch.sum(prob_aesthetic * scale, dim=1, keepdim=True)
                target_mean = torch.sum(aesthetic_score_histogram * scale, dim=1, keepdim=True)
                mean_pred.append(outputs_mean.view(-1).cpu().numpy())
                mean_target.append(target_mean.view(-1).cpu().numpy())
                # MSE
                mse = criterion_mse(outputs_mean, target_mean)
                running_mse_loss += mse.item()
            
            running_emd_loss += loss.item()
            # running_attr_emd_loss += loss_attribute.item()
            progress_bar.set_postfix({
                'Test EMD Loss': loss.item(),
            })
    
    # Calculate SROCC
    predicted_scores = np.concatenate(mean_pred, axis=0)
    true_scores = np.concatenate(mean_target, axis=0)
    srocc, _ = spearmanr(predicted_scores, true_scores)
    
    emd_loss = running_emd_loss / len(dataloader)
    emd_attr_loss = running_attr_emd_loss / len(dataloader)
    mse_loss = running_mse_loss / len(dataloader)
    return emd_loss, emd_attr_loss, srocc, mse_loss


def save_results(dataset, userIds, traits_histograms, traits_codes, emd_loss_data, predicted_scores, true_scores, **kwargs):
    df = dataset.decode_batch_to_dataframe(traits_histograms[:,:60])
    df['userID'] = userIds
    df['EMD_Loss_Data'] = emd_loss_data
    df['PIAA_Score'] = true_scores
    df['PIAA_Pred'] = predicted_scores

    # Compute 2D PCA for traits_codes
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(traits_codes)
    df['PCA1'] = pca_results[:, 0]
    df['PCA2'] = pca_results[:, 1]
    
    # Print explained variance
    explained_variance = pca.explained_variance_ratio_
    print(f'Explained variance by PCA components: {explained_variance}')

    # Compute 2D t-SNE for traits_codes
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(traits_codes)
    df['TSNE1'] = tsne_results[:, 0]
    df['TSNE2'] = tsne_results[:, 1]

    prefix = kwargs['prefix'] if 'prefix' in kwargs else ''
    i = 1  # Initialize the index for file naming
    filename = f'LAPIS_{prefix}_evaluation_results{i}.csv'  # Include the prefix in the filename
    # Check for existing files and increment filename index to avoid overwriting
    while os.path.exists(filename):
        i += 1  # Increment the counter if the file exists
        filename = f'LAPIS_{prefix}_evaluation_results{i}.csv'  # Update the filename with the new counter
    
    print(f'Save EMD loss to {filename}')
    df.to_csv(filename, index=False)


def evaluate_each_datum(model, dataloader, device, **kwargs):
    model.eval()
    running_emd_loss = 0.0
    running_attr_emd_loss = 0.0
    running_mse_loss = 0.0
    # scale = torch.arange(1, 5.5, 0.5).to(device)
    scale = torch.arange(0, 10).to(device)
    progress_bar = tqdm(dataloader, leave=False)
    mean_pred = []
    mean_target = []
    userIds = []
    emd_loss_data = []
    traits_histograms = []
    traits_codes = []
    with torch.no_grad():
        for sample in progress_bar:
            images = sample['image'].to(device)
            aesthetic_score_histogram = sample['aestheticScore'].to(device)
            traits_histogram = sample['traits'].to(device)
            # art_type = sample['art_type'].to(device)
            userIds.extend(sample['userId'])
            # traits_histogram = torch.cat([traits_histogram, art_type], dim=1)
            traits_histograms.append(traits_histogram.cpu().numpy())

            # aesthetic_logits, _ = model(images, traits_histogram)
            aesthetic_logits = model(images, traits_histogram)
            prob_aesthetic = F.softmax(aesthetic_logits, dim=1)
            # prob_attribute = F.softmax(attribute_logits, dim=-1) # Softmax along the bins dimension
            loss = earth_mover_distance(prob_aesthetic, aesthetic_score_histogram).mean()
            # loss_attribute = criterion(prob_attribute, attributes_target_histogram).mean()
            emd_loss_datum = earth_mover_distance(prob_aesthetic, aesthetic_score_histogram)
            emd_loss_data.append(emd_loss_datum.view(-1).cpu().numpy())

            pt_code = model.pt_encoder(traits_histogram).detach().cpu().numpy()
            traits_codes.append(pt_code)

            outputs_mean = torch.sum(prob_aesthetic * scale, dim=1, keepdim=True)
            target_mean = torch.sum(aesthetic_score_histogram * scale, dim=1, keepdim=True)
            mean_pred.append(outputs_mean.view(-1).cpu().numpy())
            mean_target.append(target_mean.view(-1).cpu().numpy())
            # MSE
            mse = criterion_mse(outputs_mean, target_mean)
            running_mse_loss += mse.item()
            
            running_emd_loss += loss.item()
            # running_attr_emd_loss += loss_attribute.item()
            progress_bar.set_postfix({
                'Test EMD Loss': loss.item(),
            })
    
    # Calculate SROCC
    predicted_scores = np.concatenate(mean_pred, axis=0)
    true_scores = np.concatenate(mean_target, axis=0)
    srocc, _ = spearmanr(predicted_scores, true_scores)
    emd_loss_data = np.concatenate(emd_loss_data)
    traits_histograms = np.concatenate(traits_histograms)
    traits_codes = np.concatenate(traits_codes)
    
    save_results(dataloader.dataset, userIds, traits_histograms, traits_codes, emd_loss_data, predicted_scores, true_scores, **kwargs)
    
    emd_loss = running_emd_loss / len(dataloader)
    emd_attr_loss = running_attr_emd_loss / len(dataloader)
    mse_loss = running_mse_loss / len(dataloader)
    return emd_loss, emd_attr_loss, srocc, mse_loss


num_bins = 10
num_attr = 8
num_bins_attr = 5
num_pt = 137
criterion_mse = nn.MSELoss()


if __name__ == '__main__':
    args = parse_arguments()
    
    batch_size = args.batch_size
    random_seed = 42
    n_workers = 8
    args.eval_on_piaa = True if args.trainset == 'PIAA' else False

    if args.is_log:
        tags = ["no_attr", args.trainset]
        if args.use_cv:
            tags += ["CV%d/%d"%(args.fold_id, args.n_fold)]
        if args.dropout is not None:
            tags += [f"dropout={args.dropout}"]            
        wandb.init(project="resnet_LAVIS_PIAA", 
                   notes="latefusion",
                   tags=tags)
        wandb.config = {
            "learning_rate": args.lr,
            "batch_size": batch_size,
            "num_epochs": args.num_epochs
        }
        experiment_name = wandb.run.name
    else:
        experiment_name = ''

    train_dataset, val_giaa_dataset, val_piaa_imgsort_dataset, test_giaa_dataset, test_piaa_imgsort_dataset = load_data(args)
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers, timeout=300, collate_fn=collate_fn)
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers, timeout=300, collate_fn=collate_fn_imgsort)
    val_giaa_dataloader = DataLoader(val_giaa_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers, timeout=300, collate_fn=collate_fn)
    val_piaa_imgsort_dataloader = DataLoader(val_piaa_imgsort_dataset, batch_size=5, shuffle=False, num_workers=n_workers, timeout=300, collate_fn=collate_fn_imgsort)
    test_giaa_dataloader = DataLoader(test_giaa_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers, timeout=300, collate_fn=collate_fn)
    test_piaa_imgsort_dataloader = DataLoader(test_piaa_imgsort_dataset, batch_size=5, shuffle=False, num_workers=n_workers, timeout=300, collate_fn=collate_fn_imgsort)
    dataloaders = (train_dataloader, val_giaa_dataloader, val_piaa_imgsort_dataloader, test_giaa_dataloader, test_piaa_imgsort_dataloader)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize the combined model
    model = CombinedModel(num_bins, num_attr, num_bins_attr, num_pt, args.dropout).to(device)
    
    if args.resume is not None:
        model.load_state_dict(torch.load(args.resume))
    # Loss and optimizer
    # criterion_mse = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Initialize the best test loss and the best model
    best_model = None
    best_modelname = 'lapis_best_model_resnet50_histo_lr%1.0e_decay_%depoch' % (args.lr, args.num_epochs)
    best_modelname += '_%s'%experiment_name
    best_modelname += '.pth'
    dirname = 'models_pth'
    if args.use_cv:
        dirname = os.path.join(dirname, 'random_cvs')
    best_modelname = os.path.join(dirname, best_modelname)
    
    trainer(dataloaders, model, optimizer, args, train, (evaluate, evaluate_with_prior), device, best_modelname)
    emd_loss, emd_attr_loss, srocc, mse_loss = evaluate_each_datum(model, test_piaa_imgsort_dataloader, device)
