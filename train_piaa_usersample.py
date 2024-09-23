import os
import torch
import numpy as np
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from PARA_PIAA_dataloader import load_user_sample_data
import wandb
from train_piaa_mir import PIAA_MIR, train_piaa, evaluate_piaa, train, evaluate, evaluate_with_prior
from train_piaa_ici import PIAA_ICI
import matplotlib.pyplot as plt
import warnings
from utils.argflags import parse_arguments_piaa, wandb_tags, model_dir

warnings.simplefilter("ignore")


def trainer_piaa(dataloaders, model, optimizer, args, train_fn, evaluate_fn, device, best_modelname):

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
        train_mse_loss = train_fn(model, train_dataloader, criterion_mse, optimizer, device, args)
        # Testing
        val_mse_loss, val_srocc = evaluate_fn(model, test_dataloader, criterion_mse, device, args)

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
    test_mse_loss, test_srocc = evaluate_fn(model, test_dataloader, criterion_mse, device, args)
    if args.is_log:
        wandb.log({"Test PIAA MSE Loss": test_mse_loss,
                   "Test PIAA SROCC": test_srocc}, commit=True)
    print(f"Test PIAA SROCC Loss: {test_srocc:.4f}, ")
    
    return test_srocc  


def trainer(dataloaders, model, optimizer, args, train_fn, evaluate_fns, device, best_modelname):

    train_dataloader, val_giaa_dataloader, val_piaa_imgsort_dataloader, test_giaa_dataloader, test_piaa_imgsort_dataloader = dataloaders
    evaluate_fn, evaluate_fn_with_prior = evaluate_fns

    if args.resume is not None:
        test_piaa_loss, test_piaa_srocc = evaluate_fn(model, test_piaa_imgsort_dataloader, criterion_mse, device, args)
        test_giaa_loss, test_giaa_srocc = evaluate_fn(model, test_giaa_dataloader, criterion_mse, device, args)
        test_giaa_loss_wprior, test_giaa_srocc_wprior = evaluate_fn_with_prior(model, test_giaa_dataloader, val_giaa_dataloader, criterion_mse, device, args)
        if args.is_log:
            wandb.log({"Test PIAA MSE Loss (Pretrained)": test_piaa_loss,
                    "Test PIAA SROCC (Pretrained)": test_piaa_srocc,
                    "Test GIAA MSE Loss (Pretrained)": test_giaa_loss,
                    "Test GIAA SROCC (Pretrained)": test_giaa_srocc,
                    "Test GIAA MSE Loss (Prior)(Pretrained)": test_giaa_loss_wprior,
                    "Test GIAA SROCC (Prior)(Pretrained)": test_giaa_srocc_wprior,                   
                    }, commit=True)
    
    num_patience_epochs = 0
    best_test_srocc = 0
    for epoch in range(args.num_epochs):
        if args.is_eval:
            break
        # Learning rate schedule
        if (epoch + 1) % args.lr_schedule_epochs == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= args.lr_decay_factor
        # Training
        train_mse_loss = train_fn(model, train_dataloader, criterion_mse, optimizer, device, args)
        # Testing
        val_mse_loss, val_srocc = evaluate_fn(model, val_piaa_imgsort_dataloader, criterion_mse, device, args)

        if args.is_log:
            wandb.log({"Train PIAA MSE Loss": train_mse_loss,
                        "Val PIAA MSE Loss": val_mse_loss,
                        "Val PIAA SROCC": val_srocc,
                    }, commit=True)

        # Early stopping check
        if val_srocc > best_test_srocc:
            best_test_srocc = val_srocc
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
    test_piaa_loss, test_piaa_srocc = evaluate_fn(model, test_piaa_imgsort_dataloader, criterion_mse, device, args)
    print(test_piaa_srocc)
    test_giaa_loss, test_giaa_srocc = evaluate_fn(model, test_giaa_dataloader, criterion_mse, device, args)
    test_giaa_loss_wprior, test_giaa_srocc_wprior = evaluate_fn_with_prior(model, test_giaa_dataloader, val_giaa_dataloader, criterion_mse, device, args)
    if args.is_log:
        wandb.log({"Test PIAA MSE Loss": test_piaa_loss,
                   "Test PIAA SROCC": test_piaa_srocc,
                   "Test GIAA MSE Loss": test_giaa_loss,
                   "Test GIAA SROCC": test_giaa_srocc,
                   "Test GIAA MSE Loss (Prior)": test_giaa_loss_wprior,
                   "Test GIAA SROCC (Prior)": test_giaa_srocc_wprior,                   
                   }, commit=True)
    print(
        f"Test GIAA SROCC: {test_giaa_srocc:.4f}, "
        f"Test GIAA SROCC (Prior): {test_giaa_srocc_wprior:.4f}, "
        f"Test PIAA SROCC: {test_piaa_srocc:.4f}, ")
    
    return test_piaa_srocc  

criterion_mse = nn.MSELoss()

if __name__ == '__main__':
    parser = parse_arguments_piaa(False)
    parser.add_argument('--model', type=str, default='PIAA-MIR')
    parser.add_argument('--num_users', type=int, default=40, help='Number of users for sampling')
    parser.add_argument('--num_image_threshold', type=int, default=200, help='Number of users for sampling')
    parser.add_argument('--max_annotations_per_user', type=int, default=100, help='Number of users for sampling')
    args = parser.parse_args()
    print(args)    
    # model_dir
    args.trainset = 'PIAA'
    batch_size = args.batch_size
    n_workers = args.num_workers
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
        tags += ['User sample']
        wandb.init(project="resnet_PARA_PIAA",
                notes=args.model,
                tags=tags)
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
        if args.disable_onehot:
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers, timeout=300)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers, timeout=300)
            dataloaders = (train_dataloader, test_dataloader)
        else:
            raise NotImplementedError()
        
        # Define the number of classes in your dataset
        num_classes = num_attr + num_bins
        # Define the device for training
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if args.model == 'PIAA_MIR':
            model = PIAA_MIR(num_bins, num_attr, num_pt, dropout=args.dropout).to(device)
            best_modelname = 'best_model_resnet50_piaamir_lr%1.0e_decay_%depoch' % (args.lr, args.num_epochs)
        elif args.model == 'PIAA_ICI':
            model = PIAA_ICI(num_bins, num_attr, num_pt, dropout=args.dropout).to(device)
            best_modelname = 'best_model_resnet50_piaaici_lr%1.0e_decay_%depoch' % (args.lr, args.num_epochs)
        else:
            raise NotImplementedError()

        if args.pretrained_model is not None:
            model.nima_attr.load_state_dict(torch.load(args.pretrained_model))
        if args.resume:
            model.load_state_dict(torch.load(args.resume))
        model = model.to(device)

        # Define the optimizer
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        # Initialize the best test loss and the best model
        best_modelname += '_%s'%experiment_name
        best_modelname += '.pth'
        dirname = model_dir(args)
        best_modelname = os.path.join(dirname, best_modelname)

        # Training loop
        if args.disable_onehot:
            test_srocc = trainer_piaa(dataloaders, model, optimizer, args, train_piaa, evaluate_piaa, device, best_modelname)
        else:
            test_srocc = trainer(dataloaders, model, optimizer, args, train, (evaluate, evaluate_with_prior), device, best_modelname)
                
        # Append the user_id and test_srocc as a tuple
        test_sroccs.append((user_id, test_srocc))

    # Convert the list of tuples into a numpy array
    test_sroccs_array = np.array(test_sroccs, dtype=[('user_id', 'U20'), ('srocc', 'f4')])

    # Sort the SROCCs in descending order and select the top 40 values
    sorted_sroccs = np.sort(test_sroccs_array, order='srocc')[-40:]

    # Compute the mean and standard deviation of the top 40 SROCCs
    mean_top_40 = sorted_sroccs['srocc'].mean()
    std_top_40 = sorted_sroccs['srocc'].std()
    
    # Print the results
    print("Mean of top 40 SROCCs:", mean_top_40)
    print("Standard deviation of top 40 SROCCs:", std_top_40)
    print("Top 40 SROCCs with user IDs:", sorted_sroccs)

    # Save the test_sroccs array into a text file, including user IDs
    # np.savetxt(f'{args.model}_{args.max_annotations_per_user}_test_sroccs.txt', test_sroccs_array, fmt='%s %.6f', header='User_ID SROCC')
    base_filename = f'{args.model}_{args.max_annotations_per_user}_test_sroccs'
    file_extension = '.txt'
    counter = 1
    filename = f'{base_filename}{file_extension}'
    while os.path.exists(filename):
        filename = f'{base_filename}_{counter}{file_extension}'
        counter += 1
    np.savetxt(filename, test_sroccs_array, fmt='%s %.6f', header='User_ID SROCC')

    # Plot histogram of the SROCC scores
    plt.hist(test_sroccs_array['srocc'], bins=20)
    plt.savefig('test_score.jpg')
    

