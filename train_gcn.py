import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from torch_geometric.data import Data as Data_G, Batch as Batch_G
from gcn import graph_utils as GU
from gcn import gcn


def preprocess_features_to_graph_nodes(feat_cat_batch, device='cuda'):
    """
    Preprocess a batch of feature maps into batched graph nodes and positional data.

    Args:
        feat_cat_batch (list of torch.Tensor): List of concatenated feature maps for each image in the batch.
        coord_range (list of float): Range of coordinates for constructing the graph nodes.
        device (str): The device to which the graph data will be moved ('cuda' or 'cpu'). Defaults to 'cuda'.

    Returns:
        Batch_G: Batched PyTorch Geometric graph data object moved to the specified device.
    """
    # List to hold graph inputs for each sample in the batch
    graph_inputs = []

    # Iterate over each feature map in the batch
    for feat_cat in feat_cat_batch:
        # Construct the graph input using the concatenated feature for each image
        graph_input = GU.construct_graph_13(feat_cat[0])
        
        # Convert the graph input into a PyTorch Geometric `Data` object
        graph_data = Data_G(x=graph_input[0], pos=graph_input[1], y=None)
        
        # Append to the list
        graph_inputs.append(graph_data)
    
    # Create a batch of graphs using PyTorch Geometric's `Batch` class
    batch_graph_input = Batch_G.from_data_list(graph_inputs)

    # Move the graph data to the specified device (default is 'cuda')
    batch_graph_input.x = batch_graph_input.x.float().to(device)
    batch_graph_input.batch = batch_graph_input.batch.to(device)
    batch_graph_input.pos = batch_graph_input.pos.float().to(device)

    return batch_graph_input


# Efficient feature extractor from specific ResNet18 layers
class EfficientFeatureExtractor(nn.Module):
    def __init__(self):
        super(EfficientFeatureExtractor, self).__init__()
        self.resnet18 = resnet18(pretrained=True)
        
        # Break down ResNet18 into parts to capture features at specific layers
        self.layer_5 = nn.Sequential(*list(self.resnet18.children())[:5])  # until layer -5
        self.layer_4 = nn.Sequential(*list(self.resnet18.children())[5:6])  # layer -4
        self.layer_3 = nn.Sequential(*list(self.resnet18.children())[6:7])  # layer -3

    def forward(self, x):
        # Pass the input through the layers in a single forward pass
        feat_5 = self.layer_5(x)  # Smallest spatial dimensions
        feat_4 = self.layer_4(feat_5)  # Middle spatial dimensions
        feat_3 = self.layer_3(feat_4)  # Largest spatial dimensions

        # Resize feat_3 and feat_4 to the same size as feat_5
        feat_3_resized = F.interpolate(feat_3, size=feat_5.shape[-2:], mode='bilinear', align_corners=False)
        feat_4_resized = F.interpolate(feat_4, size=feat_5.shape[-2:], mode='bilinear', align_corners=False)

        # Concatenate the features along the channel dimension (dim=1)
        feat_cat = torch.cat([feat_5, feat_4_resized, feat_3_resized], dim=1)

        return feat_cat

# Initialize the GNN model
model_gnn = gcn.GAT_x3_GATP_MH(10, 448, 512, 10).cuda()

# Initialize the feature extractor
feature_extractor = EfficientFeatureExtractor().cuda()

with torch.no_grad():
    # Generate a random input image batch (batch size of 4 for example)
    x = torch.randn(50, 3, 224, 224).cuda()  # Example random input image batch
    
    # Efficiently extract features and concatenate them along the channel dimension for each sample
    feat_cat_batch = [feature_extractor(img.unsqueeze(0)) for img in x]  # Extract features for each image in the batch

    # Preprocess the concatenated features into batched graph nodes, specifying the device as 'cuda'
    batch_graph_input = preprocess_features_to_graph_nodes(feat_cat_batch, device='cuda')

    # Run the GNN model to get the final score predictions
    scores_dist = model_gnn(batch_graph_input, 'test')['A2']

