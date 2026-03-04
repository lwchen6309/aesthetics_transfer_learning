import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from timm import create_model


class NIMA_attr(nn.Module):
    def __init__(self, num_bins_aesthetic, num_attr, backbone="resnet50", pretrained=True):
        super().__init__()
        if backbone == "resnet50":
            self.backbone = resnet50(pretrained=pretrained)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == "vit_small_patch16_224":
            self.backbone = create_model("vit_small_patch16_224", pretrained=pretrained, num_classes=0)
            feature_dim = self.backbone.embed_dim
        elif backbone == "swin_tiny_patch4_window7_224":
            self.backbone = create_model("swin_tiny_patch4_window7_224", pretrained=pretrained, num_classes=0)
            feature_dim = self.backbone.num_features
        elif backbone == "swin_base_patch4_window7_224":
            self.backbone = create_model("swin_base_patch4_window7_224", pretrained=pretrained, num_classes=0)
            feature_dim = self.backbone.num_features
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.scale = torch.arange(1, 5.5, 0.5)
        self.fc_aesthetic = nn.Sequential(nn.Linear(feature_dim, num_bins_aesthetic))
        self.fc_attributes = nn.Sequential(nn.Linear(feature_dim, num_attr))

    def forward(self, images):
        x = self.backbone(images)
        aesthetic_logits = self.fc_aesthetic(x)
        attribute_logits = self.fc_attributes(x)
        return aesthetic_logits, attribute_logits


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class PIAA_MIR(nn.Module):
    def __init__(self, num_bins, num_attr, num_pt, hidden_size=1024, dropout=None, backbone="resnet50"):
        super().__init__()
        self.num_bins = num_bins
        self.num_attr = num_attr
        self.num_pt = num_pt
        self.scale = torch.arange(1, 5.5, 0.5)
        self.nima_attr = NIMA_attr(num_bins, num_attr, backbone)
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        self.mlp1 = MLP(num_attr * num_pt, hidden_size, 1)
        self.mlp2 = MLP(num_bins, hidden_size, 1)

    def forward(self, images, personal_traits):
        logit, attr_mean_pred = self.nima_attr(images)
        prob = F.softmax(logit, dim=1)
        A_ij = attr_mean_pred.unsqueeze(2) * personal_traits.unsqueeze(1)
        I_ij = A_ij.view(images.size(0), -1)
        if self.dropout and self.dropout > 0:
            I_ij = self.dropout_layer(I_ij)
        interaction_outputs = self.mlp1(I_ij)
        direct_outputs = self.mlp2(prob * self.scale.to(images.device))
        return interaction_outputs + direct_outputs


class InternalInteraction(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        self.input_dim = input_dim

    def forward(self, attribute_embeddings):
        num_attributes = attribute_embeddings.size(1)
        batch_size = attribute_embeddings.size(0)
        interaction_matrix = torch.zeros(batch_size, num_attributes, num_attributes, self.input_dim, device=attribute_embeddings.device)
        for i in range(num_attributes):
            for j in range(num_attributes):
                combined_features = attribute_embeddings[:, i] * attribute_embeddings[:, j]
                interaction_matrix[:, i, j] = self.mlp(combined_features)
        return torch.sum(interaction_matrix, dim=1)


class ExternalInteraction(nn.Module):
    def forward(self, user_attributes, image_attributes):
        interaction_results = user_attributes.unsqueeze(2) * image_attributes.unsqueeze(1)
        aggregated_interactions_user = torch.sum(interaction_results, dim=2)
        aggregated_interactions_img = torch.sum(interaction_results, dim=1)
        return aggregated_interactions_user, aggregated_interactions_img


class Interfusion_GRU(nn.Module):
    def __init__(self, input_dim=64):
        super().__init__()
        self.gru = nn.GRUCell(input_dim, input_dim)

    def forward(self, initial_node, internal_interaction, external_interaction):
        num_attr = initial_node.shape[1]
        results = []
        for i in range(num_attr):
            fused_node = self.gru(initial_node[:, i], None)
            fused_node = self.gru(internal_interaction[:, i], fused_node)
            fused_node = self.gru(external_interaction[:, i], fused_node)
            results.append(fused_node)
        return torch.stack(results, dim=1)


class PIAA_ICI(nn.Module):
    def __init__(self, num_bins, num_attr, num_pt, input_dim=64, hidden_size=256, dropout=None, backbone="resnet50"):
        super().__init__()
        self.num_bins = num_bins
        self.num_attr = num_attr
        self.num_pt = num_pt
        self.scale = torch.arange(1, 5.5, 0.5)
        self.nima_attr = NIMA_attr(num_bins, num_attr, backbone)
        self.input_dim = input_dim
        self.internal_interaction_img = InternalInteraction(input_dim=input_dim, hidden_dim=hidden_size)
        self.internal_interaction_user = InternalInteraction(input_dim=input_dim, hidden_dim=hidden_size)
        self.external_interaction = ExternalInteraction()
        self.interfusion_img = Interfusion_GRU(input_dim=input_dim)
        self.interfusion_user = Interfusion_GRU(input_dim=input_dim)
        self.node_attr_user = MLP(num_pt, hidden_size, num_attr * input_dim)
        self.node_attr_img = MLP(num_attr, hidden_size, num_attr * input_dim)
        self.mlp_dist = MLP(num_bins, hidden_size, 1)
        self.attr_corr = nn.Linear(input_dim, 1)
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(self.dropout) if self.dropout else None

    def forward(self, images, personal_traits):
        logit, attr_img = self.nima_attr(images)
        if self.dropout_layer is not None:
            personal_traits = self.dropout_layer(personal_traits)
        n_attr = attr_img.shape[1]
        attr_img = self.node_attr_img(attr_img).view(-1, n_attr, self.input_dim)
        attr_user = self.node_attr_user(personal_traits).view(-1, n_attr, self.input_dim)
        prob = F.softmax(logit, dim=1)
        internal_img = self.internal_interaction_img(attr_img)
        internal_user = self.internal_interaction_user(attr_user)
        agg_user, agg_img = self.external_interaction(attr_user, attr_img)
        fused_img = self.interfusion_img(attr_img, internal_img, agg_img)
        fused_user = self.interfusion_user(attr_user, internal_user, agg_user)
        interaction_outputs = torch.sum(fused_img, dim=1) + torch.sum(fused_user, dim=1)
        interaction_outputs = self.attr_corr(interaction_outputs)
        direct_outputs = self.mlp_dist(prob * self.scale.to(images.device))
        return interaction_outputs + direct_outputs
