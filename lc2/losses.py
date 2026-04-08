"""Loss functions for LC2 two-phase training.

Phase 1 defaults to the paper's modified contrastive loss with degree of
similarity ψ (Eq. 2). We also expose a bidirectional InfoNCE loss for
diagnosing whether stronger in-batch ranking pressure improves cross-modal
memorization on VIVID.

Phase 2: Standard triplet margin loss (Eq. 3).

Reference: Lee et al., "(LC)²: LiDAR-Camera Loop Constraints for
Cross-Modal Place Recognition", RA-L 2023.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class BidirectionalInfoNCELoss(nn.Module):
    """Symmetric in-batch InfoNCE for paired cross-modal descriptors.

    Given paired descriptors ``(desc_i[k], desc_j[k])`` within a batch, treats
    the matching index as the positive and every other sample in the batch as a
    negative. The loss is the mean of i→j and j→i cross-entropy.
    """

    def __init__(self, temperature: float = 0.05) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, desc_i: torch.Tensor, desc_j: torch.Tensor) -> torch.Tensor:
        if desc_i.ndim != 2 or desc_j.ndim != 2:
            raise ValueError("InfoNCE expects 2D descriptor tensors")
        if desc_i.shape != desc_j.shape:
            raise ValueError("desc_i and desc_j must have the same shape")

        desc_i = F.normalize(desc_i, dim=1)
        desc_j = F.normalize(desc_j, dim=1)
        logits = desc_i @ desc_j.t()
        logits = logits / self.temperature
        targets = torch.arange(logits.size(0), device=logits.device)
        loss_i = F.cross_entropy(logits, targets)
        loss_j = F.cross_entropy(logits.t(), targets)
        return 0.5 * (loss_i + loss_j)


class MultiPositiveInfoNCELoss(nn.Module):
    """InfoNCE with multiple positives per query.

    For each range query, multiple depth entries can be positive (same location).
    Uses soft cross-entropy with a positive mask instead of single target.

    Args:
        temperature: Softmax temperature.
    """

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        desc_range: torch.Tensor,
        desc_depth: torch.Tensor,
        pos_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            desc_range: (N_r, D) range descriptors.
            desc_depth: (N_d, D) depth descriptors.
            pos_mask: (N_r, N_d) boolean mask. pos_mask[i,j]=True means
                depth j is a positive for range i.

        Returns:
            Scalar loss.
        """
        desc_range = F.normalize(desc_range, dim=1)
        desc_depth = F.normalize(desc_depth, dim=1)

        logits = desc_range @ desc_depth.t() / self.temperature  # (N_r, N_d)

        # Soft targets: uniform over positives per query
        pos_mask_f = pos_mask.float()
        n_pos = pos_mask_f.sum(dim=1, keepdim=True).clamp(min=1)
        soft_targets = pos_mask_f / n_pos  # (N_r, N_d)

        # Cross-entropy with soft targets
        log_probs = F.log_softmax(logits, dim=1)
        loss_r2d = -(soft_targets * log_probs).sum(dim=1).mean()

        # Reverse: depth → range (transpose)
        logits_t = logits.t()  # (N_d, N_r)
        pos_mask_t = pos_mask.t().float()
        n_pos_t = pos_mask_t.sum(dim=1, keepdim=True).clamp(min=1)
        soft_targets_t = pos_mask_t / n_pos_t
        log_probs_t = F.log_softmax(logits_t, dim=1)
        loss_d2r = -(soft_targets_t * log_probs_t).sum(dim=1).mean()

        return 0.5 * (loss_r2d + loss_d2r)
