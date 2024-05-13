import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
# from torchvision import transforms
from torchvision.models import resnet50
import numpy as np
from tqdm import tqdm
import wandb
from scipy.stats import spearmanr
from PARA_histogram_dataloader import load_data_paiaa, collate_fn_imgsort
from train_nima import earth_mover_distance
import math


class AesModel(nn.Module):
    def __init__(self, num_classes, keep_probability, inputsize):

        super(AesModel, self).__init__()

        self.fc1_2 = nn.Linear(inputsize, 2048)
        self.bn1_2 = nn.BatchNorm1d(2048)
        self.drop_prob = (1 - keep_probability)
        self.relu1_2 = nn.PReLU()
        self.drop1_2 = nn.Dropout(self.drop_prob)
        self.fc2_2 = nn.Linear(2048, 1024)
        self.bn2_2 = nn.BatchNorm1d(1024)
        self.relu2_2 = nn.PReLU()
        self.drop2_2 = nn.Dropout(p=self.drop_prob)
        self.fc3_2 = nn.Linear(1024, num_classes)
        # self.soft = nn.Softmax()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Weight initialization reference: https://arxiv.org/abs/1502.01852
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            # elif isinstance(m, nn.Linear):
            #     m.weight.data.normal_(0, 0.02)
            #     m.bias.data.zero_()

    def forward(self, x):
        """
        Feed-forward pass.
        :param x: Input tensor
        : return: Output tensor
        """

        out_a = self.fc1_2(x)
        out_a = self.bn1_2(out_a)
        out_a = self.relu1_2(out_a)
        out_a = self.drop1_2(out_a)
        out_a = self.fc2_2(out_a)
        out_a = self.bn2_2(out_a)
        out_a = self.relu2_2(out_a)
        out_a = self.drop2_2(out_a)
        out_a = self.fc3_2(out_a)
        # out_a = self.soft(out_a)
        # out_s = self.soft(out_a)
        # out_a = torch.cat((out_a, out_p), 1)


        # out_a = self.sig(out)
        return out_a

class PerModel(nn.Module):
    def __init__(self, keep_probability, inputsize):

        super(PerModel, self).__init__()

        self.fc1_1 = nn.Linear(inputsize, 2048)
        self.bn1_1 = nn.BatchNorm1d(2048)
        self.drop_prob = (1 - keep_probability)
        self.relu1_1 = nn.PReLU()
        self.drop1_1 = nn.Dropout(self.drop_prob)
        self.fc2_1 = nn.Linear(2048, 1024)
        self.bn2_1 = nn.BatchNorm1d(1024)
        self.relu2_1 = nn.PReLU()
        self.drop2_1 = nn.Dropout(p=self.drop_prob)
        self.fc3_1 = nn.Linear(1024, 5)
        self.bn3_1 = nn.BatchNorm1d(5)
        self.tanh = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Weight initialization reference: https://arxiv.org/abs/1502.01852
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            # elif isinstance(m, nn.Linear):
            #     m.weight.data.normal_(0, 0.02)
            #     m.bias.data.zero_()

    def forward(self, x):
        """
        Feed-forward pass.
        :param x: Input tensor
        : return: Output tensor
        """
        out_p = self.fc1_1(x)
        out_p = self.bn1_1(out_p)
        out_p = self.relu1_1(out_p)
        out_p = self.drop1_1(out_p)
        out_p = self.fc2_1(out_p)
        out_p = self.bn2_1(out_p)
        out_p = self.relu2_1(out_p)
        out_p = self.drop2_1(out_p)
        out_p = self.fc3_1(out_p)
        out_p = self.bn3_1(out_p)
        out_p = self.tanh(out_p)

        return out_p

class convNet(nn.Module):
    #constructor
    def __init__(self,resnet,aesnet,pernet):
        super(convNet, self).__init__()
        #defining layers in convnet
        self.resnet=resnet
        self.AesNet=aesnet
        self.PerNet=pernet
    def forward(self, x):
        x=self.resnet(x)
        x1=self.AesNet(x)
        x2=self.PerNet(x)
        return x1, x2


# Training Function
def train(model, dataloader, piaa_dataloader, optimizer, device):
    model.train()
    running_aesthetic_dist_mse_loss = 0.0
    running_big5_mse_loss = 0.0
    progress_bar = tqdm(dataloader, leave=False)
    # scale_aesthetic = torch.arange(1, 5.5, 0.5).to(device)    

    # Train aesthetic score
    for sample in progress_bar:
        images = sample['image'].to(device)
        aesthetic_score_histogram = sample['aestheticScore'].to(device)

        optimizer.zero_grad()
        aesthetic_logits, big5_pred = model(images)
        prob_aesthetic = F.softmax(aesthetic_logits, dim=1)
        loss_aesthetic = earth_mover_distance(prob_aesthetic, aesthetic_score_histogram).mean()
        
        loss_aesthetic.backward()
        optimizer.step()
        running_aesthetic_dist_mse_loss += loss_aesthetic.item()

        progress_bar.set_postfix({
            'Train MSE Loss for ScoreDistribution': loss_aesthetic.item(),
        })

    # Train big5
    progress_bar = tqdm(piaa_dataloader, leave=False)
    for sample in progress_bar:
        images = sample['image'].to(device)
        onehot_big5 = sample['big5'].to(device)
        batch_size = onehot_big5.shape[0]
        big5 = (torch.argmax(onehot_big5.view(batch_size, 5, 10), dim=2) + 1).float() # 1 - 10
        
        optimizer.zero_grad()
        aesthetic_logits, big5_pred = model(images)
        loss_big5 = criterion_mse(big5_pred, big5)
        
        loss_big5.backward()
        optimizer.step()
        running_big5_mse_loss += loss_big5.item()

        progress_bar.set_postfix({
            'Train MSE Loss for Big5': loss_big5.item(),
        })

    epoch_mse_loss_aesthetic = running_aesthetic_dist_mse_loss / len(dataloader)
    epoch_mse_loss_big5 = running_big5_mse_loss / len(piaa_dataloader)
    return epoch_mse_loss_aesthetic, epoch_mse_loss_big5


# Evaluation Function
def evaluate(model, dataloader, piaa_dataloader, criterion, device):
    model.eval()
    running_mean_aesthetic_mse_loss = 0.0
    running_aesthetic_dist_mse_loss = 0.0
    running_big5_mse_loss = 0.0
    scale = torch.arange(1, 5.5, 0.5).to(device)

    mean_pred = []
    mean_target = []

    with torch.no_grad():
        progress_bar = tqdm(dataloader, leave=False)
        for sample in progress_bar:
            images = sample['image'].to(device)
            aesthetic_score_histogram = sample['aestheticScore'].to(device)

            aesthetic_logits, big5_pred = model(images)
            prob_aesthetic = F.softmax(aesthetic_logits, dim=1)
            loss_aesthetic = earth_mover_distance(prob_aesthetic, aesthetic_score_histogram).mean()

            outputs_mean = torch.sum(prob_aesthetic * scale, dim=1)
            target_mean = torch.sum(aesthetic_score_histogram * scale, dim=1)
            mean_pred.append(outputs_mean.cpu().numpy())
            mean_target.append(target_mean.cpu().numpy())

            mse = ((outputs_mean - target_mean) ** 2).mean()
            running_mean_aesthetic_mse_loss += mse.item()
            running_aesthetic_dist_mse_loss += loss_aesthetic.item()
            progress_bar.set_postfix({
                'Test MSE Loss for Aesthetic': loss_aesthetic.item(),
            })
        
        # Evaluate big5
        progress_bar = tqdm(piaa_dataloader, leave=False)
        for sample in progress_bar:
            images = sample['image'].to(device)
            onehot_big5 = sample['big5'].to(device)
            batch_size = images.shape[0]
            big5 = (torch.argmax(onehot_big5.view(batch_size, 5, 10), dim=2) + 1).float() # 1 - 10

            aesthetic_logits, big5_pred = model(images)
            loss_aesthetic = criterion_mse(big5_pred, big5).mean()

            running_aesthetic_dist_mse_loss += loss_aesthetic.item()
            progress_bar.set_postfix({
                'Test MSE Loss for Big5': loss_aesthetic.item(),
            })
            break

    # Calculate SROCC
    predicted_scores = np.concatenate(mean_pred)
    true_scores = np.concatenate(mean_target)
    srocc, _ = spearmanr(predicted_scores, true_scores)

    mse_loss = running_mean_aesthetic_mse_loss / len(dataloader)
    epoch_mse_loss_aesthetic = running_aesthetic_dist_mse_loss / len(dataloader)
    epoch_mse_loss_big5 = running_big5_mse_loss / len(piaa_dataloader)
    return epoch_mse_loss_aesthetic, epoch_mse_loss_big5, srocc, mse_loss


num_bins = 9
# num_attr = 8
# num_bins_attr = 5
# num_pt = 50 + 20
criterion_mse = nn.MSELoss()


if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description='Training and Testing the Combined Model for data spliting')
    parser.add_argument('--trainset', type=str, default='GIAA', choices=["GIAA", "sGIAA", "PIAA"])
    parser.add_argument('--fold_id', type=int, default=1)
    parser.add_argument('--n_fold', type=int, default=4)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--use_cv', action='store_true', help='Enable cross validation')
    parser.add_argument('--is_eval', action='store_true', help='Enable evaluation mode')
    parser.add_argument('--no_log', action='store_false', dest='is_log', help='Disable logging')
    args = parser.parse_args()
    
    resume = args.resume
    is_eval = args.is_eval
    is_log = args.is_log
    
    random_seed = 42
    lr = 5e-5
    batch_size = 100
    num_epochs = 20
    lr_schedule_epochs = 5
    lr_decay_factor = 0.5
    max_patience_epochs = 10
    n_workers = 8
    eval_on_giaa = True

    if is_log:
        tags = ["no_attr","GIAA"]
        if args.use_cv:
            tags += ["CV%d/%d"%(args.fold_id, args.n_fold)]
        wandb.init(project="resnet_PARA_PIAA", 
                   notes="PAIAA-GIAA",
                   tags = tags)
        wandb.config = {
            "learning_rate": lr,
            "batch_size": batch_size,
            "num_epochs": num_epochs
        }
        experiment_name = wandb.run.name
    else:
        experiment_name = ''
    
    train_dataset, train_piaa_imgsort_dataset, val_giaa_dataset, val_piaa_imgsort_dataset, test_giaa_dataset, test_piaa_dataset, test_piaa_imgsort_dataset = load_data_paiaa(args)
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers, timeout=300)
    train_piaa_dataloader = DataLoader(train_piaa_imgsort_dataset, batch_size=5, shuffle=True, num_workers=n_workers, timeout=300, collate_fn=collate_fn_imgsort)

    val_giaa_dataloader = DataLoader(val_giaa_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers, timeout=300)
    val_piaa_imgsort_dataloader = DataLoader(val_piaa_imgsort_dataset, batch_size=5, shuffle=False, num_workers=n_workers, timeout=300, collate_fn=collate_fn_imgsort)
    test_giaa_dataloader = DataLoader(test_giaa_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers, timeout=300)
    test_piaa_imgsort_dataloader = DataLoader(test_piaa_imgsort_dataset, batch_size=5, shuffle=False, num_workers=n_workers, timeout=300, collate_fn=collate_fn_imgsort)
    # test_user_piaa_dataloader = DataLoader(test_user_piaa_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers, timeout=300)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the combined model
    model_ft = resnet50(pretrained=True)
    model_ft.aux_logits = False
    num_ftrs = model_ft.fc.out_features
    net1 = AesModel(num_bins, 0.5, num_ftrs)
    net2 = PerModel(0.5, num_ftrs)
    model = convNet(resnet=model_ft, aesnet=net1, pernet=net2).to(device)
    
    if resume is not None:
        model.load_state_dict(torch.load(resume))
    # Loss and optimizer
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = optim.Adam([
        {'params': model.AesNet.parameters(), 'lr': 1e-3},
        {'params': model.PerNet.parameters(), 'lr': 1e-3},
        {'params': model.resnet.parameters(), 'lr': lr}
    ], lr=lr)  # Default LR, applies to parameters not explicitly set above
    
    # Initialize the best test loss and the best model
    best_model = None
    best_modelname = 'best_model_resnet50_paiaa_giaa_lr%1.0e_decay_%depoch' % (lr, num_epochs)
    best_modelname += '_%s'%experiment_name
    best_modelname += '.pth'
    dirname = 'models_pth'
    if args.use_cv:
        dirname = os.path.join(dirname, 'random_cvs')
    best_modelname = os.path.join(dirname, best_modelname)
    
    # Training loop
    best_test_srocc = 0
    for epoch in range(num_epochs):
        if is_eval:
            break
        # Learning rate schedule
        if (epoch + 1) % lr_schedule_epochs == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_decay_factor

        # Training
        train_mse_loss_aesthetic, train_mse_loss_big5 = train(model, train_dataloader, train_piaa_dataloader, optimizer, device)
        if is_log:
            wandb.log({"Train MSE Loss for aesthetic distribution": train_mse_loss_aesthetic,
                       "Train MSE Loss for Big5": train_mse_loss_big5,
                       }, commit=False)
        
        # Testing
        val_mse_loss_aesthetic, val_mse_loss_big5, val_giaa_srocc, val_giaa_mse = evaluate(model, val_giaa_dataloader, device)
        if is_log:
            wandb.log({
                "Val MSE Loss for aesthetic distribution": val_mse_loss_aesthetic,
                "Val GIAA MSE Loss for Big5": val_mse_loss_big5,
                "Val GIAA SROCC": val_giaa_srocc,
                # "Val PIAA EMD Loss": val_piaa_emd_loss,
                # "Val PIAA SROCC": val_piaa_srocc,                
            }, commit=True)
        
        eval_srocc = val_giaa_srocc
        # Early stopping check
        if eval_srocc > best_test_srocc:
            best_test_srocc = eval_srocc
            num_patience_epochs = 0
            torch.save(model.state_dict(), best_modelname)
        else:
            num_patience_epochs += 1
            if num_patience_epochs >= max_patience_epochs:
                print("Validation loss has not decreased for {} epochs. Stopping training.".format(max_patience_epochs))
                break
    
    if not is_eval:
        model.load_state_dict(torch.load(best_modelname))   
    
    # Testing
    test_mse_loss_aesthetic, test_mse_loss_big5, test_giaa_srocc, test_giaa_mse = evaluate(model, test_giaa_dataloader, device)
    if is_log:
        wandb.log({
            "Test MSE Loss for aesthetic distribution": test_mse_loss_aesthetic,
            "Test GIAA MSE Loss for Big5": test_mse_loss_big5,
            "Test GIAA SROCC": test_giaa_srocc,
        }, commit=True)
    
    # Print the epoch loss
    print(f"Epoch [{epoch + 1}/{num_epochs}], "
            f"Test GIAA SROCC Loss: {test_giaa_srocc:.4f}, "
            f"Test GIAA MSE Loss: {test_giaa_mse:.4f}, "
            )