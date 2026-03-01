"""Loss functions for LC2 two-phase training.

Phase 1: Modified contrastive loss with degree of similarity ψ (Eq. 2).
Phase 2: Standard triplet margin loss (Eq. 3).

Reference: Lee et al., "(LC)²: LiDAR-Camera Loop Constraints for
Cross-Modal Place Recognition", RA-L 2023.
"""

import torch
import torch.nn as nn


class LC2ContrastiveLoss(nn.Module):
    r"""Modified contrastive loss weighted by degree of similarity ψ (Eq. 2).

    .. math::

        \mathcal{L}^M_{i,j} = \psi_{i,j} \cdot d(x_i, x_j)^2
            + (1 - \psi_{i,j}) \cdot \max(\tau - d(x_i, x_j),\, 0)^2

    where :math:`d(x_i, x_j) = \|\hat{f}(x_i) - \hat{f}(x_j)\|_2` is the
    L2 distance between pooled descriptors.

    When ψ ≈ 1 (high overlap), the loss pulls descriptors together.
    When ψ ≈ 0 (no overlap), the loss pushes descriptors apart beyond margin τ.

    Args:
        tau: Margin constant τ for dissimilar pairs.
    """

    def __init__(self, tau: float = 1.0) -> None:
        super().__init__()
        self.tau = tau

    def forward(
        self,
        desc_i: torch.Tensor,
        desc_j: torch.Tensor,
        psi: torch.Tensor,
    ) -> torch.Tensor:
        """Compute contrastive loss for a batch of pairs.

        Args:
            desc_i: L2-normalized descriptors for sample i, shape ``(B, D)``.
            desc_j: L2-normalized descriptors for sample j, shape ``(B, D)``.
            psi: Degree of similarity ψ ∈ [0, 1], shape ``(B,)``.

        Returns:
            Scalar loss averaged over the batch.
        """
        dist = torch.norm(desc_i - desc_j, p=2, dim=1)  # (B,)

        loss_attract = psi * dist.pow(2)
        loss_repel = (1.0 - psi) * torch.clamp(self.tau - dist, min=0.0).pow(2)

        return (loss_attract + loss_repel).mean()
