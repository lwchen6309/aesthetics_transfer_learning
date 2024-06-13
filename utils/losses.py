import torch
import torch.nn as nn


class EarthMoverDistance(nn.Module):
    def __init__(self, dim=-1):
        super(EarthMoverDistance, self).__init__()
        self.dim = dim

    def forward(self, x, y):
        """
        Compute Earth Mover's Distance (EMD) between two 1D tensors x and y using 2-norm.
        """
        cdf_x = torch.cumsum(x, dim=self.dim)
        cdf_y = torch.cumsum(y, dim=self.dim)
        emd = torch.norm(cdf_x - cdf_y, p=2, dim=self.dim)
        return emd


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0, threshold=0.2):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.threshold = threshold

    def forward(self, pt_code, score_distributions):
        """
        Compute the contrastive loss based on score distributions in a vectorized manner.
        :param pt_code: The encoded point representations.
        :param score_distributions: The corresponding aesthetic score distributions.
        :return: The contrastive loss.
        """
        batch_size = pt_code.size(0)
        unsq_pt_code = pt_code.unsqueeze(0)
        unsq_cdf_score_distributions = torch.cumsum(score_distributions, dim=-1).unsqueeze(0)
        distance_matrix = torch.cdist(unsq_pt_code, unsq_pt_code, p=2)
        distribution_diff_matrix = torch.cdist(unsq_cdf_score_distributions, unsq_cdf_score_distributions, p=2)
        
        positive_pairs = distribution_diff_matrix < self.threshold
        negative_pairs = distribution_diff_matrix >= self.threshold
        
        positive_loss = (distance_matrix ** 2) * positive_pairs.float()
        negative_loss = (torch.clamp(self.margin - distance_matrix, min=0) ** 2) * negative_pairs.float()
        
        loss = positive_loss.sum() + negative_loss.sum()
        
        n_pospair = positive_pairs.sum()
        n_negpair = negative_pairs.sum()
        num_pairs = n_pospair + n_negpair
        
        loss /= num_pairs
        return loss
    