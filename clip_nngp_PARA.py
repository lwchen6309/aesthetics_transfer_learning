import time
from absl import app
import jax.numpy as np
import neural_tangents as nt
from neural_tangents import stax
# from examples import datasets
from examples import util

from torchvision import transforms
from PARA_dataloader import PARADataset
from transformers import ViTFeatureExtractor, ViTForImageClassification
from transformers import AutoProcessor, CLIPVisionModel

from PIL import Image
import numpy
import torchvision
import torch
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from act_nngp import make_dataset_imbalance

device = "cuda" if torch.cuda.is_available() else "cpu"

def extract_feature(dataset, model):
    xclip = []
    mean_y = []
    std_y = []
    for datum in tqdm(dataset):
        x, mean_score, std_score = datum
        inputs = processor(images=x, return_tensors="pt").to(device)
        outputs = model(**inputs)
        pooled_output = outputs.pooler_output  # pooled CLS states
        xclip.append(pooled_output[0].detach().cpu().numpy())
        mean_y.append(mean_score)
        std_y.append(std_score) 
    xclip = np.stack(xclip)
    mean_y = numpy.stack(mean_y)
    std_y = numpy.stack(std_y)
    return xclip, mean_y, std_y


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
    # Usage example:
    root_dir = '/home/lwchen/datasets/PARA/'

    # Define transformations for training set and test set
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
    ])

    # Set the random seed for reproducibility in the test set
    random_seed = 42
    test_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
    ])

    # Create datasets with the appropriate transformations
    train_dataset = PARADataset(root_dir, transform=train_transform, train=True)
    test_dataset = PARADataset(root_dir, transform=test_transform, train=False, random_seed=random_seed)

    use_raw_data = False
    is_imbalanced = False

    # vit_feature_file = 'para_vit.npz'
    vit_feature_file = 'para_clip.npz'
    if not os.path.isfile(vit_feature_file):
        # # Model flag for ViT
        # model_flag = 'google/vit-base-patch16-224'
        # feature_extractor = ViTFeatureExtractor.from_pretrained(model_flag)
        # model = ViTForImageClassification.from_pretrained(model_flag).to(device)
        # Model flag for CLIP
        model_flag = 'openai/clip-vit-base-patch32'
        processor = AutoProcessor.from_pretrained(model_flag)
        model = CLIPVisionModel.from_pretrained(model_flag).to(device)

        # train_dataset, _ = torch.utils.data.random_split(train_dataset, [train_size, len(train_dataset)-train_size])
        # test_dataset, _ = torch.utils.data.random_split(test_dataset, [test_size, len(test_dataset)-test_size])
        x_train, y_train, y_train_std = extract_feature(train_dataset, model)
        x_test, y_test, y_test_std = extract_feature(test_dataset, model)
        np.savez(vit_feature_file, 
                 x_train=x_train, y_train=y_train, y_train_std=y_train_std, 
                 x_test=x_test, y_test=y_test, y_test_std=y_test_std)
    else:
        vit_dct = np.load(vit_feature_file)
        x_train_full = vit_dct['x_train']
        y_train_full = vit_dct['y_train']
        x_test = vit_dct['x_test']
        y_test = vit_dct['y_test']
        x_test_std = vit_dct['x_test_std']
        y_test_std = vit_dct['y_test_std']

    # train_sizes = (2**np.arange(4,15)).astype(int)
    # train_sizes = np.arange(100,17000,1000)
    train_sizes = [1000]
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
        # util.print_summary('NTK test', y_test, fx_test_ntk, None, loss)
        accuracies.append(loss)
        
    filename = 'nngp'
    filename += '.npz'
    if use_raw_data:
        filename = 'raw_' + filename
    np.savez(filename, train_sizes=train_sizes, accuracies=np.array(accuracies))
    plt.plot(train_sizes, accuracies)
    plt.title('PARA')
    plt.xlabel('Number training sample')
    plt.ylabel('MSE')
    plt.show()