import time
from absl import app
import jax.numpy as np
import neural_tangents as nt
from neural_tangents import stax
# from examples import datasets
from examples import util

from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import numpy
import torchvision
import torch
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from glob import glob

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def extract_feature(dataset, feature_extractor, model, nclasses=10):
    xvit_train = []
    y_train = []
    for datum in tqdm(dataset):
        x, y = datum
        inputs = feature_extractor(images=x, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits
        xvit_train.append(logits[0].detach().cpu().numpy())
        y_train.append(y)
    xvit_train = np.stack(xvit_train)
    y_train = numpy.stack(y_train)
    onehot_ytrain = numpy.zeros((len(y_train), nclasses), dtype=int)
    onehot_ytrain[np.arange(len(y_train)),y_train] = 1
    # y_train = torchvision.transforms.functional.one_hot(torch.tensor(y_train), num_classes=10)
    return xvit_train, onehot_ytrain

def extract_raw(dataset, nclasses=10):
    xvit_train = []
    y_train = []   
    for datum in tqdm(dataset):
        x, y = datum
        x = torchvision.transforms.functional.pil_to_tensor(x).numpy().ravel()
        xvit_train.append(x)
        y_train.append(y)
    xvit_train = np.stack(xvit_train)
    xvit_train = (xvit_train - np.mean(xvit_train, axis=0)) / np.std(xvit_train, axis=0)
    y_train = numpy.stack(y_train)
    onehot_ytrain = numpy.zeros((len(y_train), nclasses), dtype=int)
    onehot_ytrain[np.arange(len(y_train)),y_train] = 1
    # y_train = torchvision.transforms.functional.one_hot(torch.tensor(y_train), num_classes=10)
    return xvit_train, onehot_ytrain

def plot_nngp_performance():
    # expfiles = ['nngp.npz', 'act_nngp_entropy.npz', 'act_nngp_entropy_unsample.npz']
    # labels = ['NTK w/ ViT', 'Active NTK w/ ViT entropy: Greedy', 'Active NTK w/ ViT entropy: Sample']
    # plt.figure()
    # for filename, label in zip(expfiles, labels):
    #     plt.plot(np.load(filename)['train_sizes'], np.load(filename)['accuracies'], label=label)
    # plt.legend()   
    # plt.title('CIFAR-10')
    # plt.xlabel('Number training sample')
    # plt.ylabel('Accuracy')
    # plt.xlim(XLIM)

    fig, axes = plt.subplots(1,2, figsize=(12, 5))
    expfiles = ['nngp.npz', 'act_nngp.npz',]
    labels = ['NTK w/ ViT', 'Active NTK w/ ViT',]
    # plt.figure()
    for filename, label in zip(expfiles, labels):
        axes[0].plot(np.load(filename)['train_sizes'], np.load(filename)['accuracies'], label=label)
    axes[0].set(title='CIFAR-10', xlim=XLIM, ylim=[0.6,1.0],
        xlabel='Number training sample', ylabel='Accuracy') 

    expfiles = ['nngp_imbalance.npz', 'act_nngp_imbalance.npz',]
    labels = ['NTK w/ ViT', 'Active NTK w/ ViT',]
    # plt.figure()
    for filename, label in zip(expfiles, labels):
        axes[1].plot(np.load(filename)['train_sizes'], np.load(filename)['accuracies'], label=label)
    axes[1].set(title='Imbalanced CIFAR-10', xlim=XLIM, ylim=[0.6,1.0],
        xlabel='Number training sample', ylabel='Accuracy')
    for axis in axes:
        axis.yaxis.set_visible(True)
        axis.legend()
    

def plot_nngp_nn_performance():
    plt.figure()
    active_nn = glob('/home/liwei/badge/*.npz')
    active_nn = [
        '/home/liwei/badge/rand_resnet.npz',
        # '/home/liwei/badge/entropy_resnet.npz',
        '/home/liwei/badge/badge_resnet.npz',]
    for filename in active_nn:
        label = os.path.basename(filename).split('.')[0].replace('_', ' w/ ').replace('resnet', 'ResNet')
        plt.plot(np.load(filename)['train_sizes'], np.load(filename)['accuracies'], label=label)
    expfiles = ['raw_nngp.npz', 'nngp.npz']
    labels = ['NTK', 'NTK w/ ViT']
    
    for filename, label in zip(expfiles, labels):
        plt.plot(np.load(filename)['train_sizes'], np.load(filename)['accuracies'], label=label)
    plt.legend()   
    plt.xlim(XLIM)
    plt.title('CIFAR-10')
    plt.xlabel('Number training sample')
    plt.ylabel('Accuracy')


def make_dataset_imbalance(x, y, supressed_cls, remove_ratio=0.9):
    x = x
    class_number = np.sum(y, axis=0)
    print(class_number)
    y_label = np.argmax(y, axis=-1)
    remain_idx = np.arange(len(x))
    remove_idx = []
    # Remove 90% of data in class
    for cls in supressed_cls:
        data_idx = np.nonzero(y_label == cls)[0]
        cls_number = len(data_idx)
        remove_idx.append(data_idx[:int(cls_number*remove_ratio)])
    remove_idx = np.concatenate(remove_idx)
    remain_idx = np.delete(remain_idx, remove_idx)
    x = x[remain_idx,:]
    y = y[remain_idx,:]
    class_number = np.sum(y, axis=0)
    print(class_number)
    return x, y



learning_rate = 1
train_size = 50000
test_size = 10000
_BATCH_SIZE = 0


XLIM=[0, 16000]

if __name__ == '__main__':
    # Build data pipelines.
    plt.close()
    plot_nngp_nn_performance()
    plot_nngp_performance()
    plt.show()
    raise Exception

    print('Loading data.')
    train_dataset = torchvision.datasets.CIFAR10(root='~/dataset/cifar10/', train=True, transform=None, download=True)
    test_dataset = torchvision.datasets.CIFAR10(root='~/dataset/cifar10/', train=False, transform=None, download=True)
    
    vit_feature_file = 'cifar_vit.npz'
    if not os.path.isfile(vit_feature_file):
        model_flag = 'google/vit-base-patch16-224'
        feature_extractor = ViTFeatureExtractor.from_pretrained(model_flag)
        model = ViTForImageClassification.from_pretrained(model_flag)
        train_dataset, _ = torch.utils.data.random_split(train_dataset, [train_size, len(train_dataset)-train_size])
        test_dataset, _ = torch.utils.data.random_split(test_dataset, [test_size, len(test_dataset)-test_size])
        x_train, y_train = extract_feature(train_dataset, feature_extractor, model)
        x_test, y_test = extract_feature(test_dataset, feature_extractor, model)
        np.savez(vit_feature_file, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    else:
        vit_dct = np.load(vit_feature_file)
        x_train_full = vit_dct['x_train']
        y_train_full = vit_dct['y_train']
        x_test = vit_dct['x_test']
        y_test = vit_dct['y_test']
    use_raw_data = False
    if use_raw_data:
        x_train_full, y_train_full = extract_raw(train_dataset)
        x_test, y_test = extract_raw(test_dataset)

    x_train_full, y_train_full = make_dataset_imbalance(
        x_train_full, y_train_full,
        supressed_cls = [3, 5, 9], remove_ratio=0.9)

    train_sizes = (2**np.arange(4,15)).astype(int)
    # train_sizes = np.arange(100,17000,1000)
    accuracies = []
    remained_idx = np.arange(len(x_train_full))
    # Iterate over the array and pick data
    use_sample = False
    for i, train_size in enumerate(train_sizes):
        # Pick the first index in the remained_idx list
        if i > 0:
            newdata_len = train_size - train_sizes[i-1]
            ## Choose sampling strategry
            var = numpy.array(var)
            p = var / var.sum()

            if not use_sample:
                # Greedy Algorithm
                picked_idx_idx = np.flip(np.argsort(var))[:newdata_len]
            else:
                # Uncertianty Sampling
                remained_idx_idx = np.arange(len(remained_idx))
                picked_idx_idx = numpy.random.choice(remained_idx_idx, size=newdata_len, p=p)

            data_idx = np.concatenate([data_idx, remained_idx[picked_idx_idx]])
            remained_idx = np.delete(remained_idx, picked_idx_idx)
            var = np.delete(var, picked_idx_idx)
        else:
            data_idx = remained_idx[:train_size]
            remained_idx = remained_idx[train_size:]
        assert(len(data_idx) == train_size)
        x_train = x_train_full[data_idx,:]
        y_train = y_train_full[data_idx,:]
        x_train_rest = x_train_full[remained_idx,:]

        # Build the infinite network.
        _, _, kernel_fn = stax.serial(
            stax.Dense(1, 2., 0.05),
            stax.Relu(),
            stax.Dense(1, 2., 0.05)
        )

        # Optionally, compute the kernel in batches, in parallel.
        kernel_fn = nt.batch(kernel_fn,
                            device_count=0,
                            batch_size=_BATCH_SIZE)

        start = time.time()
        # Bayesian and infinite-time gradient descent inference with infinite network.
        predict_fn = nt.predict.gradient_descent_mse_ensemble(kernel_fn, x_train,
                                                            y_train, diag_reg=1e-3)
        fx_test_nngp, fx_test_ntk = predict_fn(x_test=x_test)
        fx_test_nngp.block_until_ready()
        fx_test_ntk.block_until_ready()

        duration = time.time() - start
        print('Kernel construction and inference done in %s seconds.' % duration)

        # Print out accuracy and loss for infinite network predictions.
        loss = lambda fx, y_hat: 0.5 * np.mean((fx - y_hat) ** 2)
        # util.print_summary('NNGP test', y_test, fx_test_nngp, None, loss)
        util.print_summary('NTK test', y_test, fx_test_ntk, None, loss)
        accuracies.append(util._accuracy(fx_test_ntk, y_test))

        # _, fx_rest_ntk = predict_fn(x_test=x_train_rest)
        # exp_prob = np.exp(fx_rest_ntk)
        # exp_prob = exp_prob / exp_prob.sum(axis=1,keepdims=True)
        # entropy = (exp_prob * np.log(exp_prob + 1e-6)).sum(axis=1)
        # var = entropy
        if train_size < 2000:
            var = []
            for x_rest in tqdm(x_train_rest):
                _, fx_rest_ntk = predict_fn(x_test=x_rest[None], compute_cov=True)
                var.append(fx_rest_ntk.covariance)
            var = np.stack(var).ravel()

    filename = 'act_nngp_imbalance.npz'
    if use_raw_data:
        filename = 'raw_' + filename
    np.savez(filename, train_sizes=train_sizes, accuracies=np.array(accuracies))
    



