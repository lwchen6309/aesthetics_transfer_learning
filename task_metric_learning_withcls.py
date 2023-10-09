import logging

import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import umap
from cycler import cycler
from PIL import Image
from torchvision import datasets, transforms
from torchvision.models import resnet50
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm

import pytorch_metric_learning
import pytorch_metric_learning.utils.logging_presets as logging_presets
from pytorch_metric_learning import losses, miners, samplers, testers, trainers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning.distances import CosineSimilarity, SNRDistance
from pytorch_metric_learning.reducers import ThresholdReducer
from pytorch_metric_learning.regularizers import LpRegularizer

from PARA_histogram_dataloader import PARA_HistogramDataset, PARA_GIAA_HistogramDataset, PARA_PIAA_HistogramDataset
from PARA_PIAA_dataloader import PARA_PIAADataset, split_dataset_by_user, split_dataset_by_images


logging.getLogger().setLevel(logging.INFO)
logging.info("VERSION %s" % pytorch_metric_learning.__version__)


class MLP(nn.Module):
    # layer_sizes[0] is the dimension of the input
    # layer_sizes[-1] is the dimension of the output
    def __init__(self, layer_sizes, final_relu=False):
        super().__init__()
        layer_list = []
        layer_sizes = [int(x) for x in layer_sizes]
        num_layers = len(layer_sizes) - 1
        final_relu_layer = num_layers if final_relu else num_layers - 1
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            curr_size = layer_sizes[i + 1]
            if i < final_relu_layer:
                layer_list.append(nn.ReLU(inplace=False))
            layer_list.append(nn.Linear(input_size, curr_size))
        self.net = nn.Sequential(*layer_list)
        self.last_linear = self.net[-1]

    def forward(self, x):
        return self.net(x)


class TruncModel(nn.Module):
    def __init__(self, num_bins, num_pt, image_feature_dim=2048):
        super(TruncModel, self).__init__()
        # self.resnet = resnet50(pretrained=True)
        # self.feature_extractor = create_feature_extractor(self.resnet, return_nodes={'layer4': 'layer4', 'fc': 'fc'})
        self.num_bins = num_bins
        self.num_pt = num_pt
        self.embedding_layer = nn.Sequential(
            # nn.Linear(self.resnet.fc.in_features + self.num_bins, 512),  # Assuming the trait and target histograms are 512-dimensional each
            nn.Linear(image_feature_dim + self.num_bins + self.num_pt, 512),  # Assuming the trait and target histograms are 512-dimensional each
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )
    
    def forward(self, x):
        # image, traits_histogram, target_histogram = data
        # with torch.no_grad():
        # output_dict = self.feature_extractor(image)
        # resnet_feature = F.adaptive_avg_pool2d(output_dict['layer4'], (1,1))[:,:,0,0]
        # x = torch.cat((resnet_feature, traits_histogram, target_histogram), dim=1)
        return self.embedding_layer(x)


class TripletDataset(PARA_GIAA_HistogramDataset):
    def __init__(self, root_dir, transform=None, data=None, map_file=None, precompute_file=None, precomputed_data_path=None):
        super().__init__(root_dir, transform, data, map_file, precompute_file)
        self.num_tasks = 9
        self.targets = np.concatenate([np.arange(self.num_tasks) for i in range(super().__len__())], axis=0)
        self.resnet = resnet50(pretrained=True)
        self.feature_extractor = create_feature_extractor(self.resnet, return_nodes={'layer4': 'layer4', 'fc': 'fc'})
        self.precomputed_data_path = precomputed_data_path
        
        # Check if precomputed data exists
        if self.precomputed_data_path is not None and os.path.exists(self.precomputed_data_path):
            self._precomputed_data = self._load_precomputed_data()
        else:
            # Precompute the get_task_data for each image with a progress bar
            self._precomputed_data = [self.get_task_data(i) for i in tqdm(range(super().__len__()), desc="Precomputing data")]
            self._save_precomputed_data()

    def _save_precomputed_data(self):
        torch.save(self._precomputed_data, self.precomputed_data_path)
    
    def _load_precomputed_data(self):
        return torch.load(self.precomputed_data_path)

    def __len__(self):
        return super().__len__() * self.num_tasks

    def get_task_data(self, index):
        sample = super().__getitem__(index)
        
        images = sample['image']
        aesthetic_score_histogram = sample['aestheticScore']
        aesthetic_score_histogram = torch.cat([aesthetic_score_histogram[0].unsqueeze(0), 
                                              aesthetic_score_histogram[1:].reshape(-1,2).sum(dim=1)], dim=0)
        attributes_histogram = sample['attributes']
        total_task = torch.cat([aesthetic_score_histogram.unsqueeze(0), attributes_histogram.reshape(-1,5)])

        traits_histogram = sample['traits']
        onehot_traits_histogram = sample['onehot_traits']
        traits_histogram = torch.cat([traits_histogram, onehot_traits_histogram], dim=0)
        
        with torch.no_grad():
            output_dict = self.feature_extractor(images.unsqueeze(0))
            resnet_feature = F.adaptive_avg_pool2d(output_dict['layer4'], (1,1))[0,:,0,0]
        cat_data = torch.cat((resnet_feature, traits_histogram), dim=0).type(torch.float16)
        return cat_data, total_task
    
    def __getitem__(self, index):
        img_index, task_index = index//self.num_tasks, index%self.num_tasks
        # data, tasks = self.get_task_data(img_index)
        data, tasks = self._precomputed_data[img_index]
        data = torch.cat((data, tasks[task_index]), dim=0)
        return data, task_index


def load_triplet_data(root_dir):
    pkl_dir = './dataset_pkl'
    # Set the image transforms
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    # Create datasets with the appropriate transformations
    train_piaa_dataset = PARA_PIAADataset(root_dir, transform=train_transform)
    test_piaa_dataset = PARA_PIAADataset(root_dir, transform=train_transform)
    train_dataset, test_dataset = split_dataset_by_images(train_piaa_dataset, test_piaa_dataset, root_dir)

    # Create datasets with the appropriate transformations
    train_piaa_dataset = PARA_PIAADataset(root_dir, transform=train_transform)
    test_piaa_dataset = PARA_PIAADataset(root_dir, transform=train_transform)
    train_dataset, test_dataset = split_dataset_by_images(train_piaa_dataset, test_piaa_dataset, root_dir)

    # Turn the datasets into triplet datasets for contrastive learning
    
    train_dataset = TripletDataset(root_dir, transform=train_transform, data=train_dataset.data, 
            map_file=os.path.join(pkl_dir,'trainset_image_dct.pkl'), precompute_file=os.path.join(pkl_dir,'trainset_GIAA_dct.pkl'),
            precomputed_data_path=os.path.join(pkl_dir,'trainset_image_taskcls_dct.pkl'))
    test_dataset = TripletDataset(root_dir, transform=test_transform, data=test_dataset.data,
            map_file=os.path.join(pkl_dir,'testset_image_dct.pkl'), precompute_file=os.path.join(pkl_dir,'testset_GIAA_dct.pkl'),
            precomputed_data_path=os.path.join(pkl_dir,'testset_image_taskcls_dct.pkl'))
    # test_oneimg_dataset = TripletTestOneImageDataset(root_dir, transform=test_transform, data=test_dataset.data,
    #         map_file=os.path.join(pkl_dir,'testset_image_dct.pkl'), precompute_file=os.path.join(pkl_dir,'testset_GIAA_dct.pkl'))
    return train_dataset, test_dataset


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Set trunk model and replace the softmax layer with an identity function
    is_log = False
    random_seed = None
    num_bins = 5
    num_pt = 50 + 20
    trunk = TruncModel(num_bins=num_bins, num_pt=num_pt)
    trunk_output_size = 512
    trunk = torch.nn.DataParallel(trunk.to(device))
    embedding_dim = 512
    # Set other training parameters
    batch_size = 256
    num_epochs = 50
    nclasses = 9
    lr = 1e-3

    if is_log:
        wandb.init(project="resnet_PARA_PIAA_metric")
        wandb.config = {
            "learning_rate": lr,
            "batch_size": batch_size,
            "num_epochs": num_epochs
        }
        experiment_name = wandb.run.name
    else:
        experiment_name = 'local'
    
    train_dataset, test_dataset = load_triplet_data(root_dir = '/home/lwchen/datasets/PARA/')

    # Set embedder model. This takes in the output of the trunk and outputs 64 dimensional embeddings
    embedder = torch.nn.DataParallel(MLP([trunk_output_size, 512, embedding_dim]).to(device))

    # Set the classifier. The classifier will take the embeddings and output a 50 dimensional vector.
    # (Our training set will consist of the first 50 classes of the CIFAR100 dataset.)
    # We'll specify the classification loss further down in the code.
    classifier = torch.nn.DataParallel(MLP([embedding_dim, nclasses])).to(device)

    # Set optimizers
    trunk_optimizer = torch.optim.Adam(trunk.parameters(), lr=lr, weight_decay=0.0001)
    embedder_optimizer = torch.optim.Adam(
        embedder.parameters(), lr=lr, weight_decay=0.0001
    )
    classifier_optimizer = torch.optim.Adam(
        classifier.parameters(), lr=lr, weight_decay=0.0001
    )

    # Loss and optimizer
    margin = 0.2
    distance = CosineSimilarity()
    reducer = ThresholdReducer(low=0)
    loss = losses.TripletMarginLoss(margin=margin, distance=distance, reducer=reducer)
    miner = miners.TripletMarginMiner(
        margin=margin, distance=distance, type_of_triplets="semihard"
    )

    # Set the classification loss:
    classification_loss = torch.nn.CrossEntropyLoss()

    # Set the mining function
    # miner = miners.MultiSimilarityMiner(epsilon=0.1)

    # Set the dataloader sampler
    sampler = samplers.MPerClassSampler(
        train_dataset.targets, m=16, length_before_new_iter=len(train_dataset)
    )

    # Package the above stuff into dictionaries.
    models = {"trunk": trunk, "embedder": embedder, "classifier": classifier}
    optimizers = {
        "trunk_optimizer": trunk_optimizer,
        "embedder_optimizer": embedder_optimizer,
        "classifier_optimizer": classifier_optimizer,
    }
    loss_funcs = {"metric_loss": loss, "classifier_loss": classification_loss}
    mining_funcs = {"tuple_miner": miner}

    # We can specify loss weights if we want to. This is optional
    loss_weights = {"metric_loss": 1, "classifier_loss": 0.5}


    record_keeper, _, _ = logging_presets.get_record_keeper(
        "example_logs", "example_tensorboard"
    )
    hooks = logging_presets.get_hook_container(record_keeper)
    dataset_dict = {"val": test_dataset}
    model_folder = "example_saved_models"


    def visualizer_hook(umapper, umap_embeddings, labels, split_name, keyname, *args):
        return 
        logging.info(
            "UMAP plot for the {} split and label set {}".format(split_name, keyname)
        )
        label_set = np.unique(labels)
        num_classes = len(label_set)
        plt.figure(figsize=(20, 15))
        plt.gca().set_prop_cycle(
            cycler(
                "color", [plt.cm.nipy_spectral(i) for i in np.linspace(0, 0.9, num_classes)]
            )
        )
        for i in range(num_classes):
            idx = labels == label_set[i]
            plt.plot(umap_embeddings[idx, 0], umap_embeddings[idx, 1], ".", markersize=1)
        plt.show()


    # Create the tester
    tester = testers.GlobalEmbeddingSpaceTester(
        end_of_testing_hook=hooks.end_of_testing_hook,
        visualizer=umap.UMAP(),
        visualizer_hook=visualizer_hook,
        dataloader_num_workers=2,
        accuracy_calculator=AccuracyCalculator(k='max_bin_count'),
    )
    
    end_of_epoch_hook = hooks.end_of_epoch_hook(
        tester, dataset_dict, model_folder, 1, 1
    )

    trainer = trainers.TrainWithClassifier(
        models,
        optimizers,
        batch_size,
        loss_funcs,
        train_dataset,
        mining_funcs=mining_funcs,
        sampler=sampler,
        dataloader_num_workers=2,
        loss_weights=loss_weights,
        end_of_iteration_hook=hooks.end_of_iteration_hook,
        end_of_epoch_hook=end_of_epoch_hook,
    )

    trainer.train(num_epochs=num_epochs)
