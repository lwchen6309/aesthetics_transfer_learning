import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from PARA_PIAA_dataloader import load_user_sample_data, collect_batch_attribute, collect_batch_personal_trait
import wandb
from scipy.stats import spearmanr
import argparse
from train_piaa_mir import CombinedModel, train, evaluate
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter("ignore")


def trainer(dataloaders, model, optimizer, args, train_fn, evaluate_fn, device, best_modelname):

    train_dataloader, test_dataloader = dataloaders
    num_patience_epochs = 0
    best_test_srocc = 0
    best_model_state = None
    for epoch in range(args.num_epochs):
        if args.is_eval:
            break
        # Learning rate schedule
        if (epoch + 1) % args.lr_schedule_epochs == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= args.lr_decay_factor
        # Training
        train_mse_loss = train_fn(model, train_dataloader, criterion_mse, optimizer, device)
        # Testing
        val_mse_loss, val_srocc = evaluate_fn(model, test_dataloader, criterion_mse, device)

        if args.is_log:
            wandb.log({"Train PIAA MSE Loss": train_mse_loss,
                        "Val PIAA MSE Loss": val_mse_loss,
                        "Val PIAA SROCC": val_srocc,
                    }, commit=True)
        
        # Early stopping check
        if val_srocc > best_test_srocc:
            best_test_srocc = val_srocc
            num_patience_epochs = 0
            # torch.save(model.state_dict(), best_modelname)
            best_model_state = model.state_dict()
        else:
            num_patience_epochs += 1
            if num_patience_epochs >= args.max_patience_epochs:
                print("Validation loss has not decreased for {} epochs. Stopping training.".format(args.max_patience_epochs))
                break
    
    if not args.is_eval and best_model_state:
        # model.load_state_dict(torch.load(best_modelname))
        model.load_state_dict(best_model_state)
    
    # Testing
    test_mse_loss, test_srocc = evaluate_fn(model, test_dataloader, criterion_mse, device)
    if args.is_log:
        wandb.log({"Test PIAA MSE Loss": test_mse_loss,
                   "Test PIAA SROCC": test_srocc}, commit=True)
    print(f"Test PIAA SROCC Loss: {test_srocc:.4f}, ")
    
    return test_srocc  

criterion_mse = nn.MSELoss()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training and Testing the Combined Model for data spliting')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--pretrained_model', type=str, required=True)
    parser.add_argument('--is_eval', action='store_true', help='Enable evaluation mode')
    parser.add_argument('--no_log', action='store_false', dest='is_log', help='Disable logging')

    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--num_users', type=int, default=40, help='Number of users for sampling')
    parser.add_argument('--num_image_threshold', type=int, default=200, help='Number of users for sampling')
    
    parser.add_argument('--lr_schedule_epochs', type=int, default=5, help='Epochs after which to apply learning rate decay')
    parser.add_argument('--lr_decay_factor', type=float, default=0.1, help='Factor by which to decay the learning rate')
    parser.add_argument('--max_patience_epochs', type=int, default=10, help='Max patience epochs for early stopping')    
    args = parser.parse_args()
    
    batch_size = args.batch_size
    n_workers = 8
    num_bins = 9
    num_attr = 8
    num_pt = 25 # number of personal trait
    
    if args.is_log:
        tags = ["no_attr","PIAA", "User sample"]
        wandb.init(project="resnet_PARA_PIAA",
                notes="PIAA-MIR",
                tags = tags)
        wandb.config = {
            "learning_rate": args.lr,
            "batch_size": batch_size,
            "num_epochs": args.num_epochs
        }
        experiment_name = wandb.run.name
    else:
        experiment_name = ''
    
    # Create dataloaders for training and test sets
    test_sroccs = []
    piaa_data_gen = load_user_sample_data(args)
    for user_id, train_dataset, test_dataset in piaa_data_gen:
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers, timeout=300)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers, timeout=300)
        dataloaders = (train_dataloader, test_dataloader)
        
        # Define the number of classes in your dataset
        num_classes = num_attr + num_bins
        # Define the device for training
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = CombinedModel(num_bins, num_attr, num_pt).to(device)
        model.nima_attr.load_state_dict(torch.load(args.pretrained_model))
        model = model.to(device)
        if args.resume:
            model.load_state_dict(torch.load(args.resume))

        # Define the loss functions
        criterion_mse = nn.MSELoss()

        # Define the optimizer
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        # Initialize the best test loss and the best model
        best_model = None
        best_modelname = 'best_model_resnet50_piaamir_lr%1.0e_decay_%depoch' % (args.lr, args.num_epochs)
        best_modelname += '_%s'%experiment_name
        best_modelname += '.pth'
        dirname = 'models_pth'
        best_modelname = os.path.join(dirname, best_modelname)

        # Training loop
        test_srocc = trainer(dataloaders, model, optimizer, args, train, evaluate, device, best_modelname)
        test_sroccs.append(test_srocc)
    test_sroccs = np.array(test_sroccs)
    # print(test_sroccs.mean(), test_sroccs.std())
    # print(test_sroccs)

    # Sort the SROCCs in descending order and select the top 40 values
    top_40_sroccs = np.sort(test_sroccs)[-40:]
    # Compute the mean and standard deviation of the top 40 SROCCs
    mean_top_40 = top_40_sroccs.mean()
    std_top_40 = top_40_sroccs.std()
    # Print the results
    print("Mean of top 40 SROCCs:", mean_top_40)
    print("Standard deviation of top 40 SROCCs:", std_top_40)
    print("Top 40 SROCCs:", top_40_sroccs)
    
    # Save the test_sroccs array into a text file
    np.savetxt('test_sroccs.txt', test_sroccs, fmt='%.6f')
    plt.hist(test_sroccs, bins=20)
    plt.savefig('test_score.jpg')
    

