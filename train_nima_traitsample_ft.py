import os
import random
import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
# from torchvision import transforms
# from torchvision.models import resnet50
import wandb
# from scipy.stats import spearmanr
from PARA_histogram_dataloader import load_data, collate_fn_imgsort
from train_nima import NIMA, train, evaluate #, trainer
from utils.argflags import parse_arguments, wandb_tags, model_dir


def trainer(dataloaders, model, optimizer, args, train_fn, evaluate_fn, device, best_modelname):
    train_dataloader, val_giaa_dataloader, val_piaa_imgsort_dataloader, test_giaa_dataloader, test_piaa_imgsort_dataloader = dataloaders

    # Testing
    test_giaa_emd_loss, _, test_giaa_srocc, test_giaa_mse = evaluate_fn(model, test_giaa_dataloader, device)
    test_piaa_emd_loss, _, test_piaa_srocc, test_piaa_mse = evaluate_fn(model, test_piaa_imgsort_dataloader, device)
    
    if args.is_log:
        wandb.log({
            "Test GIAA EMD Loss (Pretrained)": test_giaa_emd_loss,
            "Test GIAA SROCC (Pretrained)": test_giaa_srocc,
            "Test GIAA MSE (Pretrained)": test_giaa_mse,
            "Test PIAA EMD Loss (Pretrained)": test_piaa_emd_loss,
            "Test PIAA SROCC (Pretrained)": test_piaa_srocc,
            "Test PIAA MSE (Pretrained)": test_piaa_mse,
        }, commit=True)
    
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
    test_giaa_emd_loss, _, test_giaa_srocc, test_giaa_mse = evaluate_fn(model, test_giaa_dataloader, device)
    test_piaa_emd_loss, _, test_piaa_srocc, test_piaa_mse = evaluate_fn(model, test_piaa_imgsort_dataloader, device)
    
    if args.is_log:
        wandb.log({
            "Test GIAA EMD Loss": test_giaa_emd_loss,
            "Test GIAA SROCC": test_giaa_srocc,
            "Test GIAA MSE": test_giaa_mse,
            "Test PIAA EMD Loss": test_piaa_emd_loss,
            "Test PIAA SROCC": test_piaa_srocc,
            "Test PIAA MSE": test_piaa_mse,
        }, commit=True)

    # Print the epoch loss
    print(f"Test GIAA SROCC Loss: {test_giaa_srocc:.4f}, "
        f"Test PIAA SROCC Loss: {test_piaa_srocc:.4f}, "
        )


num_bins = 9
num_attr = 8
num_bins_attr = 5
num_pt = 50 + 20


if __name__ == '__main__':    
    args = parse_arguments()
    batch_size = args.batch_size
    num_workers = args.num_workers
    
    if args.is_log:
        tags = ["no_attr", args.trainset]
        tags += wandb_tags(args)
        wandb.init(project="resnet_PARA_PIAA", 
                   notes="NIMA",
                   tags = tags)
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
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, timeout=300)
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers, timeout=300, collate_fn=collate_fn_imgsort)
    val_giaa_dataloader = DataLoader(val_giaa_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, timeout=300)
    val_piaa_imgsort_dataloader = DataLoader(val_piaa_imgsort_dataset, batch_size=5, shuffle=False, num_workers=num_workers, timeout=300, collate_fn=collate_fn_imgsort)
    test_giaa_dataloader = DataLoader(test_giaa_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, timeout=300)
    test_piaa_imgsort_dataloader = DataLoader(test_piaa_imgsort_dataset, batch_size=5, shuffle=False, num_workers=num_workers, timeout=300, collate_fn=collate_fn_imgsort)
    dataloaders = (train_dataloader, val_giaa_dataloader, val_piaa_imgsort_dataloader, test_giaa_dataloader, test_piaa_imgsort_dataloader)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the combined model
    model = NIMA(num_bins).to(device)
    
    if args.resume is not None:
        model.load_state_dict(torch.load(args.resume))

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
        
    # Initialize the best test loss and the best model
    best_modelname = f'best_model_resnet50_nima_{experiment_name}.pth'
    dirname = model_dir(args)
    best_modelname = os.path.join(dirname, best_modelname)
    
    trainer(dataloaders, model, optimizer, args, train, evaluate, device, best_modelname)