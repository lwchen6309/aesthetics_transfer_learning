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


def extract_feature(dataset, model, nclasses=10):
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


learning_rate = 1
train_size = 50000
test_size = 10000
_BATCH_SIZE = 0
train_time = 2000


def plot_raw_vit_nngp_performance():
    raw_dct = np.load('raw_nngp.npz')
    vit_dct = np.load('nngp.npz')
    train_sizes = vit_dct['train_sizes']
    raw_nngp_accuracies = raw_dct['accuracies']
    vit_nngp_accuracies = vit_dct['accuracies']
    plt.plot(train_sizes, raw_nngp_accuracies, label='Image')
    plt.plot(train_sizes, vit_nngp_accuracies, label='ViT feature')
    plt.title('CIFAR-10')
    plt.xlabel('Number training sample')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # Build data pipelines.
    # plot_raw_vit_nngp_performance()
    # raise Exception

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
        x_train, y_train = extract_feature(train_dataset, model)
        x_test, y_test = extract_feature(test_dataset, model)
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

    train_sizes = (2**np.arange(4,15)).astype(int)
    # train_sizes = np.arange(100,17000,1000)
    accuracies = []
    for train_size in train_sizes:
        # train_size = 1024
        x_train = x_train_full[:train_size]
        y_train = y_train_full[:train_size]
        x_train_rest = x_train_full[train_size:]
        # y_train_rest = y_train_full[train_size:]

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
    filename = 'nngp.npz'
    if use_raw_data:
        filename = 'raw_' + filename
    np.savez(filename, train_sizes=train_sizes, accuracies=np.array(accuracies))
    plt.plot(train_sizes, accuracies)
    plt.title('CIFAR-10')
    plt.xlabel('Number training sample')
    plt.ylabel('Accuracy')
    plt.show()