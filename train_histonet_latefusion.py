import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet50
import numpy as np
from tqdm import tqdm
import wandb
from scipy.stats import spearmanr
from PARA_histogram_dataloader import load_data, collate_fn_imgsort
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


from utils.losses import EarthMoverDistance
earth_mover_distance = EarthMoverDistance()


class CombinedModel(nn.Module):
    def __init__(self, num_bins_aesthetic, num_attr, num_bins_attr, num_pt):
        super(CombinedModel, self).__init__()
        self.resnet = resnet50(pretrained=True)
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 512),
            nn.ReLU(),
            # nn.Linear(512, num_bins_aesthetic),
        )
        self.num_bins_aesthetic = num_bins_aesthetic
        self.num_attr = num_attr
        self.num_bins_attr = num_bins_attr
        self.num_pt = num_pt
        # For predicting attribute histograms for each attribute
        self.pt_encoder = nn.Sequential(
            nn.Linear(num_pt, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512)
        )
        
        # For predicting aesthetic score histogram
        self.fc_aesthetic = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_bins_aesthetic)
        )

    def forward(self, images, traits_histogram):
        x = self.resnet(images)
        pt_code = self.pt_encoder(traits_histogram)
        xz = x + pt_code
        aesthetic_logits = self.fc_aesthetic(xz)
        return aesthetic_logits


# Training Function
def train(model, dataloader, optimizer, device):
    model.train()
    running_aesthetic_emd_loss = 0.0
    running_total_emd_loss = 0.0
    progress_bar = tqdm(dataloader, leave=False)
    # scale_aesthetic = torch.arange(1, 5.5, 0.5).to(device)
    running_ce_improvement = 0
    running_ce_init = 0
    for sample in progress_bar:        
        images = sample['image'].to(device)
        aesthetic_score_histogram = sample['aestheticScore'].to(device)
        traits_histogram = sample['traits'].to(device)
        onehot_big5 = sample['big5'].to(device)
        attributes_target_histogram = sample['attributes'].to(device).view(-1, num_attr, num_bins_attr) # Reshape to match our logits shape
        total_traits_histogram = torch.cat([traits_histogram, onehot_big5], dim=1)
        # traits_histogram = traits_histogram[:,:5]
        
        optimizer.zero_grad()
        aesthetic_logits = model(images, total_traits_histogram)
        prob_aesthetic = F.softmax(aesthetic_logits, dim=1)
        # prob_attribute = F.softmax(attribute_logits, dim=-1) # Softmax along the bins dimension
        loss_aesthetic = earth_mover_distance(prob_aesthetic, aesthetic_score_histogram).mean()
        total_loss = loss_aesthetic # Combining losses
        
        total_loss.backward()
        optimizer.step()
        running_total_emd_loss += total_loss.item()
        running_aesthetic_emd_loss += loss_aesthetic.item()

        progress_bar.set_postfix({
            'Train EMD Loss': total_loss.item(),
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
    scale = torch.arange(1, 5.5, 0.5).to(device)
    eval_srocc = True

    progress_bar = tqdm(dataloader, leave=False)
    mean_pred = []
    mean_target = []
    userIds = []
    emd_loss_data = []
    traits_histograms = []
    with torch.no_grad():
        for sample in progress_bar:
            images = sample['image'].to(device)
            # userId = sample['userId']
            # userIds.extend(userId)
            aesthetic_score_histogram = sample['aestheticScore'].to(device)
            traits_histogram = sample['traits'].to(device)
            try:
                onehot_big5 = sample['big5'].to(device)
            except KeyError:
                print(sample.keys())
            attributes_target_histogram = sample['attributes'].to(device).view(-1, num_attr, num_bins_attr) # Reshape to match our logits shape
            traits_histogram = torch.cat([traits_histogram, onehot_big5], dim=1)
            # traits_histograms.append(traits_histogram.cpu().numpy())

            aesthetic_logits = model(images, traits_histogram)
            prob_aesthetic = F.softmax(aesthetic_logits, dim=1)
            # prob_attribute = F.softmax(attribute_logits, dim=-1) # Softmax along the bins dimension

            # emd_loss_datum = criterion(prob_aesthetic, aesthetic_score_histogram)
            # emd_loss_data.append(emd_loss_datum.view(-1).cpu().numpy())
            loss = earth_mover_distance(prob_aesthetic, aesthetic_score_histogram).mean()
            # loss_attribute = criterion(prob_attribute, attributes_target_histogram).mean()
            
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


def evaluate_with_prior(model, dataloader, prior_dataloader, device):
    model.eval()
    running_emd_loss = 0.0
    running_attr_emd_loss = 0.0
    running_mse_loss = 0.0
    scale = torch.arange(1, 5.5, 0.5).to(device)
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
            try:
                onehot_big5 = sample['big5'].to(device)
            except KeyError:
                print(sample.keys())
            traits_histogram = torch.cat([traits_histogram, onehot_big5], dim=1)
            traits_histograms.append(traits_histogram)
        mean_traits_histogram = torch.mean(torch.cat(traits_histograms, dim=0), dim=0)
        
        for sample in progress_bar:
            images = sample['image'].to(device)
            aesthetic_score_histogram = sample['aestheticScore'].to(device)
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


def evaluate_trait(model, dataloader, device, num_iterations=1000, learning_rate=1e-4):
    model.eval()
    optimized_trait_histograms = []

    trait_lengths = {
        'age': 5, 'gender': 2, 'EducationalLevel': 5, 'artExperience': 4,
        'photographyExperience': 4, 'personality-E': 10, 'personality-A': 10,
        'personality-N': 10, 'personality-O': 10, 'personality-C': 10
    }
    # Initialize a dictionary to accumulate EMD results for each trait
    accumulated_emd_results = {trait: [] for trait in trait_lengths.keys()}    

    def split_trait_histogram(trait_histogram):
        split_histograms = {}
        start = 0
        for trait, length in trait_lengths.items():
            split_histograms[trait] = trait_histogram[:, start:start + length]
            start += length
        return split_histograms

    for sample in tqdm(dataloader, leave=False):
        images = sample['image'].to(device)
        target_aesthetic_histogram = sample['aestheticScore'].to(device)
        target_traits = sample['traits'].to(device)
        target_onehot_big5 = sample['big5'].to(device)
        target_traits_combined = torch.cat([target_traits, target_onehot_big5], dim=1)

        # Initialize trait histogram as a zero vector for each image in the batch
        trait_histogram = torch.rand_like(target_traits_combined, requires_grad=True, device=device)
        # trait_histogram = torch.tensor(target_traits_combined, requires_grad=True, device=device)
        optimizer = optim.Adam([trait_histogram], lr=learning_rate)

        # Split, normalize, and concatenate trait_histogram
        split_histograms = split_trait_histogram(trait_histogram)
        for trait in trait_lengths.keys():
            split_histograms[trait] = F.softmax(split_histograms[trait], dim=1)
        trait_histogram = torch.cat(list(split_histograms.values()), dim=1)
        
        for _ in range(num_iterations):
            optimizer.zero_grad()
            predicted_aesthetic_histogram = model(images, trait_histogram)
            predicted_aesthetic_histogram = F.softmax(predicted_aesthetic_histogram, dim=1)
            loss = earth_mover_distance(target_aesthetic_histogram, predicted_aesthetic_histogram).mean()
            loss.backward(retain_graph=True)
            optimizer.step()

        # Compute EMD for each trait subset
        optimized_histogram = trait_histogram.detach()
        split_optimized_histograms = split_trait_histogram(optimized_histogram)
        split_target_histograms = split_trait_histogram(target_traits_combined)

        for trait in trait_lengths.keys():
            emd_values = earth_mover_distance(split_optimized_histograms[trait], split_target_histograms[trait]).cpu().numpy()
            accumulated_emd_results[trait].extend(emd_values)
        optimized_trait_histograms.append(optimized_histogram.cpu().numpy())
    optimized_trait_histograms = np.concatenate(optimized_trait_histograms, axis=0)
    return optimized_trait_histograms, accumulated_emd_results

def save_results(dataset, userIds, traits_histograms, traits_codes, emd_loss_data, predicted_scores, true_scores):
    # Decode batch to DataFrame
    df = dataset.decode_batch_to_dataframe(traits_histograms[:, :20])
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

    # Determine a unique filename
    i = 1  # Starting index
    filename = f'PARA_LFsGIAA_evaluation_results{i}.csv'
    while os.path.exists(filename):
        i += 1  # Increment the counter if the file exists
        filename = f'PARA_LFsGIAA_evaluation_results{i}.csv'  # Update the filename with the new counter

    # Save to CSV
    print(f'Save EMD loss to {filename}')
    df.to_csv(filename, index=False)



def evaluate_each_datum(model, dataloader, device):
    model.eval()
    running_emd_loss = 0.0
    running_attr_emd_loss = 0.0
    running_mse_loss = 0.0
    scale = torch.arange(1, 5.5, 0.5).to(device)
    eval_srocc = True

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
            userId = sample['userId']
            userIds.extend(userId)
            aesthetic_score_histogram = sample['aestheticScore'].to(device)
            traits_histogram = sample['traits'].to(device)
            onehot_big5 = sample['big5'].to(device)
            # attributes_target_histogram = sample['attributes'].to(device).view(-1, num_attr, num_bins_attr) # Reshape to match our logits shape
            traits_histogram = torch.cat([traits_histogram, onehot_big5], dim=1)
            traits_histograms.append(traits_histogram.cpu().numpy())

            pt_code = model.pt_encoder(traits_histogram).detach().cpu().numpy()
            traits_codes.append(pt_code)

            aesthetic_logits = model(images, traits_histogram)
            prob_aesthetic = F.softmax(aesthetic_logits, dim=1)
            
            emd_loss_datum = earth_mover_distance(prob_aesthetic, aesthetic_score_histogram)
            emd_loss_data.append(emd_loss_datum.view(-1).cpu().numpy())
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
            progress_bar.set_postfix({
                'Test EMD Loss': loss.item(),
            })
    
    # Calculate SROCC
    predicted_scores = np.concatenate(mean_pred, axis=0)
    true_scores = np.concatenate(mean_target, axis=0)
    srocc, _ = spearmanr(predicted_scores, true_scores)
    
    traits_histograms = np.concatenate(traits_histograms)
    traits_codes = np.concatenate(traits_codes)
    emd_loss_data = np.concatenate(emd_loss_data)
    save_results(dataloader.dataset, userIds, traits_histograms, traits_codes, emd_loss_data, predicted_scores, true_scores)
    
    emd_loss = running_emd_loss / len(dataloader)
    emd_attr_loss = running_attr_emd_loss / len(dataloader)
    mse_loss = running_mse_loss / len(dataloader)
    return emd_loss, emd_attr_loss, srocc, mse_loss


def trainer(dataloaders, model, optimizer, args, train_fn, evaluate_fns, device, best_modelname):
    train_dataloader, val_giaa_dataloader, val_piaa_imgsort_dataloader, test_giaa_dataloader, test_piaa_imgsort_dataloader = dataloaders
    evaluate_fn, evaluate_fn_with_prior = evaluate_fns
    
    # Training loop
    best_test_srocc = 0
    num_patience_epochs = 0
    for epoch in range(args.num_epochs):
        if args.is_eval:
            break
        # Learning rate schedule
        if (epoch + 1) % args.lr_schedule_epochs == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= args.lr_decay_factor

        # Training
        train_emd_loss, train_total_emd_loss = train_fn(model, train_dataloader, optimizer, device)
        if args.is_log:
            wandb.log({"Train EMD Loss": train_emd_loss,
                       "Train Total EMD Loss": train_total_emd_loss,
                       }, commit=False)
        
        # Testing
        val_giaa_emd_loss, _, val_giaa_srocc, _ = evaluate_fn(model, val_giaa_dataloader, device)
        val_piaa_emd_loss, _, val_piaa_srocc, _ = evaluate_fn(model, val_piaa_imgsort_dataloader, device)
        if args.is_log:
            wandb.log({
                "Val GIAA EMD Loss": val_giaa_emd_loss,
                "Val GIAA SROCC": val_giaa_srocc,
                "Val PIAA EMD Loss": val_piaa_emd_loss,
                "Val PIAA SROCC": val_piaa_srocc,                
            }, commit=True)

        eval_srocc = val_piaa_srocc if args.eval_on_piaa else val_giaa_srocc
        
        # Early stopping check
        if eval_srocc > best_test_srocc:
            best_test_srocc = eval_srocc
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
    test_giaa_emd_loss_wprior, _, test_giaa_srocc_wprior, _ = evaluate_fn_with_prior(model, test_giaa_dataloader, val_giaa_dataloader, device)
    test_piaa_emd_loss, _, test_piaa_srocc, _ = evaluate_fn(model, test_piaa_imgsort_dataloader, device)
    test_giaa_emd_loss, _, test_giaa_srocc, _ = evaluate_fn(model, test_giaa_dataloader, device)
    
    if args.is_log:
        wandb.log({
            "Test GIAA EMD Loss": test_giaa_emd_loss,
            "Test GIAA SROCC": test_giaa_srocc,
            "Test GIAA EMD Loss (Prior)": test_giaa_emd_loss_wprior,
            "Test GIAA SROCC (Prior)": test_giaa_srocc_wprior,
            "Test PIAA EMD Loss": test_piaa_emd_loss,
            "Test PIAA SROCC": test_piaa_srocc,
        }, commit=True)

    # Print the epoch loss
    print(f"Test GIAA SROCC Loss: {test_giaa_srocc:.4f}, "
            f"Test GIAA SROCC Loss with Prior: {test_giaa_srocc_wprior:.4f}, "
            f"Test PIAA SROCC Loss: {test_piaa_srocc:.4f}, "
            )


num_bins = 9
num_attr = 8
num_bins_attr = 5
num_pt = 50 + 20
criterion_mse = nn.MSELoss()


if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description='Training and Testing the Combined Model for data spliting')
    parser.add_argument('--fold_id', type=int, default=1)
    parser.add_argument('--n_fold', type=int, default=4)
    parser.add_argument('--trainset', type=str, default='GIAA', choices=["GIAA", "sGIAA", "PIAA"])
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--importance_sampling', action='store_true', help='Enable importance sampling for uniform score distribution')
    parser.add_argument('--use_cv', action='store_true', help='Enable cross validation')
    parser.add_argument('--is_eval', action='store_true', help='Enable evaluation mode')
    parser.add_argument('--eval_on_piaa', action='store_true', help='Evaluation metric on PIAA')
    parser.add_argument('--no_log', action='store_false', dest='is_log', help='Disable logging')
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--max_patience_epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--lr_schedule_epochs', type=int, default=5)
    parser.add_argument('--lr_decay_factor', type=float, default=0.5)    
    args = parser.parse_args()
    
    batch_size = args.batch_size
    random_seed = 42
    n_workers = 8
    args.eval_on_piaa = True if args.trainset == 'PIAA' else False

    if args.is_log:
        tags = ["no_attr", args.trainset]
        if args.use_cv:
            tags += ["CV%d/%d"%(args.fold_id, args.n_fold)]
        wandb.init(project="resnet_PARA_PIAA", 
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
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers, timeout=300)
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers, timeout=300, collate_fn=collate_fn_imgsort)
    val_giaa_dataloader = DataLoader(val_giaa_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers, timeout=300)
    val_piaa_imgsort_dataloader = DataLoader(val_piaa_imgsort_dataset, batch_size=5, shuffle=False, num_workers=n_workers, timeout=300, collate_fn=collate_fn_imgsort)
    test_giaa_dataloader = DataLoader(test_giaa_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers, timeout=300)
    test_piaa_imgsort_dataloader = DataLoader(test_piaa_imgsort_dataset, batch_size=5, shuffle=False, num_workers=n_workers, timeout=300, collate_fn=collate_fn_imgsort)
    dataloaders = (train_dataloader, val_giaa_dataloader, val_piaa_imgsort_dataloader, test_giaa_dataloader, test_piaa_imgsort_dataloader)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the combined model
    model = CombinedModel(num_bins, num_attr, num_bins_attr, num_pt).to(device)

    if args.resume is not None:
        model.load_state_dict(torch.load(args.resume))
    # Loss and optimizer
    # criterion_mse = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Initialize the best test loss and the best model
    best_model = None
    best_modelname = 'best_model_resnet50_histo_latefusion_lr%1.0e_decay_%depoch' % (args.lr, args.num_epochs)
    best_modelname += '_%s'%experiment_name
    best_modelname += '.pth'
    dirname = 'models_pth'
    if args.use_cv:
        dirname = os.path.join(dirname, 'random_cvs')
    best_modelname = os.path.join(dirname, best_modelname)
    
    trainer(dataloaders, model, optimizer, args, train, (evaluate, evaluate_with_prior), device, best_modelname)
    emd_loss, emd_attr_loss, srocc, mse_loss = evaluate_each_datum(model, test_piaa_imgsort_dataloader, device)