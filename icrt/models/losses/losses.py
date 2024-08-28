import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.loss import LabelSmoothingCrossEntropy

class GaussianLabelSmoothing(nn.Module):
    def __init__(self, sigma=1.0):
        super(GaussianLabelSmoothing, self).__init__()
        self.sigma = sigma

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Create Gaussian distribution around target
        num_classes = x.shape[-1]
        gauss_distrib = self.create_gaussian_distribution(target, num_classes)

        # Compute log probabilities
        logprobs = F.log_softmax(x, dim=-1)

        # Compute Gaussian Label Smoothing Loss
        loss = -torch.sum(gauss_distrib * logprobs, dim=-1)
        return loss.mean()

    def create_gaussian_distribution(self, target, num_classes):
        # Generate a sequence of class indices
        indices = torch.arange(num_classes).float().unsqueeze(0).to(target.device)

        # Expand target to match the shape of indices
        target_expanded = target.unsqueeze(1).expand(-1, num_classes).float()

        # Compute the Gaussian distribution in a vectorized way
        gauss_distrib = torch.exp(-0.5 * ((indices - target_expanded) / self.sigma) ** 2)

        # Normalize the distribution
        gauss_distrib = gauss_distrib / gauss_distrib.sum(dim=1, keepdim=True)

        return gauss_distrib

losses = {
    "l2" : nn.MSELoss,
    "l1" : nn.L1Loss,
    "smooth_l1" : nn.SmoothL1Loss,
    "bce_with_logits" : nn.BCEWithLogitsLoss,
    "cross_entropy" : nn.CrossEntropyLoss,
    "label_smoothing" : LabelSmoothingCrossEntropy,
    "gaussian_label_smoothing" : GaussianLabelSmoothing,
    "kl_div" : torch.distributions.kl.kl_divergence,
}
