"""Generalized Mean (GeM) pooling layer for Phase 1 of LC2 training.

Used as the pooling layer during contrastive pre-training (Phase 1) before
switching to NetVLAD for triplet fine-tuning (Phase 2).

GeM pooling generalizes average pooling (p=1) and max pooling (p→∞).
The learnable exponent p adapts during training.

Reference: Radenović et al., "Fine-tuning CNN image retrieval with no
human annotation", TPAMI 2018.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeM(nn.Module):
    """Generalized Mean pooling.

    Computes f_k = (1/|X| * Σ_i x_{k,i}^p)^{1/p} for each channel k.

    Aggregates ``(B, D, H, W)`` feature maps into ``(B, D)`` global descriptors.
    Output is L2-normalized to produce unit-length descriptors.

    Attributes:
        p: Learnable pooling exponent (initialized to 3.0).
        eps: Clamping epsilon to avoid numerical issues with pow.
    """

    def __init__(self, p: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute GeM descriptor from spatial feature map.

        Args:
            x: Feature map of shape ``(B, D, H, W)``.

        Returns:
            L2-normalized global descriptor of shape ``(B, D)``.
        """
        x_clamped = x.clamp(min=self.eps)
        gem = F.adaptive_avg_pool2d(x_clamped.pow(self.p), (1, 1)).pow(1.0 / self.p)
        gem = gem.squeeze(-1).squeeze(-1)  # (B, D)
        gem = F.normalize(gem, p=2, dim=1)
        return gem
