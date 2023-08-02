import time
from absl import app
import jax.numpy as jnp
import neural_tangents as nt
from neural_tangents import stax
# from examples import datasets
from examples import util

from torchvision import transforms
from PARA_dataloader import PARADataset
from transformers import ViTFeatureExtractor, ViTForImageClassification
from transformers import AutoProcessor, CLIPVisionModel
from torchvision.models import resnet50
import torch.nn as nn


from PIL import Image
import numpy as np
import torchvision
import torch
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from act_nngp import make_dataset_imbalance
from nngpll import extract_vit_feature, extract_clip_feature, extract_resnet_feature


device = "cuda" if torch.cuda.is_available() else "cpu"
plt.rcParams.update({'font.size': 18})




learning_rate = 1
# train_size = 50000
# test_size = 10000
_BATCH_SIZE = 0
train_time = 2000
root_dir = '/home/lwchen/datasets/PARA/'
data_dir = './img224_noattr'


def plot_nngp():
    modelnames = ['resnet', 'resnet_ft', 'clip', 'vit']
    fig, axis = plt.subplots(1)
    for modelname in modelnames:
        nngp_dct = jnp.load(os.path.join(data_dir, 'nngp_%s.npz'%modelname))
        nngp_diag_dct = jnp.load(os.path.join(data_dir, 'nngp_%s_uncertainty.npz'%modelname))
        train_sizes = nngp_dct['train_sizes']
        if modelname == 'vit':
            legend_name = 'ViT' 
        elif modelname == 'clip':
            legend_name = 'CLIP'
        elif 'resnet' in modelname:
            legend_name = modelname.replace('resnet', 'ResNet50')
        else:
            legend_name = modelname
        
        axis.plot(train_sizes, nngp_dct['mse_mean'], label='%s'%legend_name)
        # axis.plot(train_sizes, nngp_diag_dct['mse_mean'], label='%s_uncertainty'%legend_name)
        axis.set_title('MSE of mean score')
        # print(nngp_dct['mse_mean'].min())
        print(nngp_dct['mse_mean'].min())
        print(nngp_diag_dct['mse_mean'].min())
        
    axis.legend()
    axis.set_xlabel('Number training sample')
    axis.set_ylabel('MSE')
    # plt.legend()
    plt.show()


def plot_nngp_uncertainty():
    modelnames = ['resnet', 'resnet_ft']
    fig, axis = plt.subplots(1)
    for modelname in modelnames:
        nngp_dct = jnp.load(os.path.join(data_dir, 'nngp_%s.npz'%modelname))
        nngp_diag_dct = jnp.load(os.path.join(data_dir, 'nngp_%s_uncertainty.npz'%modelname))
        train_sizes = nngp_dct['train_sizes']
        if modelname == 'vit':
            legend_name = 'ViT' 
        elif modelname == 'clip':
            legend_name = 'CLIP'
        elif 'resnet' in modelname:
            legend_name = modelname.replace('resnet', 'ResNet')
        else:
            legend_name = modelname
        
        axis.plot(train_sizes, nngp_dct['mse_mean'], label='%s'%legend_name)
        axis.plot(train_sizes, nngp_diag_dct['mse_mean'], label='%s_uncertainty'%legend_name)
        axis.set_title('MSE of mean score')
    
    axis.legend()
    axis.set_xlabel('Number training sample')
    axis.set_ylabel('MSE')
    # plt.legend()
    plt.show()


use_uncertainty = True
use_attr = False


if __name__ == '__main__':
    # Build data pipelines.
    # plot_nngp()
    # plot_nngp_uncertainty()
    # plt.show()
    # raise Exception

    print('Loading data.')
    random_seed = 42
    # Define transformations for training set and test set
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    # Create datasets with the appropriate transformations
    train_dataset = PARADataset(root_dir, transform=test_transform, train=True, use_attr=use_attr, use_hist=True, random_seed=random_seed)
    test_dataset = PARADataset(root_dir, transform=test_transform, train=False, use_attr=use_attr, use_hist=True, random_seed=random_seed)

    modelname = 'resnet_ft'
    vit_feature_file = 'para_%s.npz' % modelname
    vit_feature_file = os.path.join(data_dir, vit_feature_file)
    if not os.path.isfile(vit_feature_file):
        if modelname == 'vit':
            # Model flag for ViT
            model_flag = 'google/vit-base-patch16-224'
            feature_extractor = ViTFeatureExtractor.from_pretrained(model_flag)
            model = ViTForImageClassification.from_pretrained(model_flag).to(device)
            x_test, y_test, y_test_std, y_test_prob = extract_vit_feature(test_dataset, model)
            x_train_full, y_train_full, y_train_std_full, y_train_prob_full = extract_vit_feature(train_dataset, model)
        elif modelname == 'resnet':
            model = resnet50(pretrained=True)
            model = model.to(device)
            model.eval()
            x_test, y_test, y_test_std, y_test_prob = extract_resnet_feature(test_dataset, model)
            x_train_full, y_train_full, y_train_std_full, y_train_prob_full = extract_resnet_feature(train_dataset, model)
        elif modelname == 'resnet_ft':
            model = resnet50(pretrained=False)
            # Modify the last fully connected layer to match the number of classes
            # num_classes = 9
            num_classes = 1
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, num_classes)
            model.load_state_dict(torch.load('best_model_resnet50_noattr.pth'))
            model = model.to(device)
            model.eval()
            # Remove the last fully connected layer to get the feature extractor
            # model = torch.nn.Sequential(*(list(model.children())[:-1]))
            x_test, y_test, y_test_std, y_test_prob = extract_resnet_feature(test_dataset, model)
            x_train_full, y_train_full, y_train_std_full, y_train_prob_full = extract_resnet_feature(train_dataset, model)
        else:
            # Model flag for CLIP
            model_flag = 'openai/clip-vit-base-patch32'
            processor = AutoProcessor.from_pretrained(model_flag)
            model = CLIPVisionModel.from_pretrained(model_flag).to(device)
            x_test, y_test, y_test_std, y_test_prob = extract_clip_feature(test_dataset, model)
            x_train_full, y_train_full, y_train_std_full, y_train_prob_full = extract_clip_feature(train_dataset, model)

        jnp.savez(vit_feature_file, 
                x_train=x_train_full, y_train=y_train_full, y_train_std=y_train_std_full, y_train_prob=y_train_prob_full,
                x_test=x_test, y_test=y_test, y_test_std=y_test_std, y_test_prob=y_test_prob)
    else:
        vit_dct = jnp.load(vit_feature_file)
        x_train_full = vit_dct['x_train']
        y_train_full = vit_dct['y_train']
        y_train_std_full = vit_dct['y_train_std']
        y_train_prob_full = vit_dct['y_train_prob']
        x_test = vit_dct['x_test']
        y_test = vit_dct['y_test']
        y_test_std = vit_dct['y_test_std']
        y_test_prob = vit_dct['y_test_prob']


    # Preprocess
    mu, sigma = x_train_full.mean(axis=0,keepdims=True), x_train_full.std(axis=0,keepdims=True)
    x_train_full = (x_train_full - mu) / sigma
    x_test = (x_test - mu) / sigma
    y_train_full = y_train_full / 5.0
    y_train_std_full = y_train_std_full / 5.0
    y_test = y_test / 5.0
    y_test_std = y_test_std / 5.0

    # Training
    # train_sizes = jnp.arange(100,10000,500)
    train_sizes = [9600]
    mse_mean = []
    mse_std = []
    x_test = jnp.array(x_test)
    # Print out accuracy and loss for infinite network predictions.
    loss = lambda fx, y_hat: jnp.mean((fx - y_hat) ** 2)

    ce_weight = 1 / train_dataset.aesthetic_score_hist_prob
    ce_weight = ce_weight / np.sum(ce_weight) * len(ce_weight)
    score_prob = y_test_prob
    scale = np.arange(1, 5.5, 0.5)
    sqrt_2pi = np.sqrt(2 * np.pi)

    for train_size in train_sizes:
        # train_size = 1024
        x_train = jnp.array(x_train_full[:train_size])
        # y_train = jnp.array(y_train_full[:train_size])
        y_train = jnp.array(y_train_prob_full[:train_size])
        
        y_train_std = y_train_std_full[:train_size]

        x_train_rest = jnp.array(x_train_full[train_size:])
        # y_train_rest = y_train_full[train_size:]

        # Build the infinite network.
        _, _, kernel_fn = stax.serial(
            stax.Dense(1, 1., 0.05),
            stax.Relu(),
            stax.Dense(1, 1., 0.05)
        )
        # Optionally, compute the kernel in batches, in parallel.
        kernel_fn = nt.batch(kernel_fn,
                            device_count=0,
                            batch_size=_BATCH_SIZE)

        start = time.time()
        # Bayesian and infinite-time gradient descent inference with infinite network.
        # predict_fn = nt.predict.gradient_descent_mse_ensemble(kernel_fn, x_train,
        #                                                     y_train, diag_reg=1e-3)
        # fx_test_nngp, fx_test_ntk = predict_fn(x_test=x_test)
        # fx_test_nngp.block_until_ready()
        # fx_test_ntk.block_until_ready()

        kdd = kernel_fn(x_train,x_train).nngp
        ktd = kernel_fn(x_train,x_test).nngp
        ktt = kernel_fn(x_test,x_test).nngp
        reg = np.sqrt((y_train_std**2).mean(-1)) if use_uncertainty else 1e-1
        kdd_inv = jnp.linalg.inv(kdd + jnp.eye(len(kdd)) * reg)
        fx_test_ntk = ktd.T @ kdd_inv @ y_train
        sigma_test_ntk = jnp.diag(ktt) - jnp.diag(ktd.T @ kdd_inv @ ktd)

        duration = time.time() - start
        print('Kernel construction and inference done in %s seconds.' % duration)

        prob = fx_test_ntk.clip(min=0)
        prob = prob / prob.sum(axis=1)[...,None]
        scale = np.arange(1, 5.5, 0.5)   
        # MSE loss for mean
        outputs_mean = np.sum(prob * scale, axis=1)[...,None]
        mse_mean_loss = loss(outputs_mean, 5.0 * y_test)

        # MSE loss for std
        outputs_std = np.sqrt(np.sum(prob * (scale - outputs_mean) ** 2, axis=1))
        mse_std_loss = loss(outputs_std, 5.0 * y_test_std)

        logit = np.log(prob + 1e-6)
        ce_loss = -np.mean(np.ravel(logit * score_prob * ce_weight))
        raw_ce_loss = -np.mean(np.ravel(logit * score_prob))
        emd_loss = np.sqrt(np.mean(np.ravel((np.cumsum(prob,axis=1) - np.cumsum(score_prob,axis=1))**2)))
        brier_score = np.mean(np.ravel((prob - score_prob)**2))

        print(f"Test MSE Mean loss: {mse_mean_loss}, Test MSE Std loss: {mse_std_loss}")
        print(f"Test CE loss: {ce_loss}, Test Raw CE loss: {raw_ce_loss}")
        print(f"Test EMD loss: {emd_loss}, Test Brier Score: {brier_score}")
    
    # filename = 'nngp_%s'%modelname
    # filename = os.path.join(data_dir,filename)
    # if use_uncertainty:
    #     filename += '_uncertainty'
    # jnp.savez(filename, train_sizes=train_sizes, mse_mean=mse_mean)
    