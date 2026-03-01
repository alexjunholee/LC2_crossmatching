"""NetVLAD aggregation layer for visual place recognition.

Implements the NetVLAD pooling operation that converts a spatial feature map
into a fixed-size global descriptor by soft-assigning local features to learned
cluster centers and accumulating first-order residual statistics.

Reference: Arandjelović et al., "NetVLAD: CNN architecture for weakly supervised
place recognition", CVPR 2016.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

import faiss
from sklearn.neighbors import NearestNeighbors


class NetVLAD(nn.Module):
    """NetVLAD pooling layer.

    Aggregates ``(B, D, H, W)`` feature maps into ``(B, K*D)`` global descriptors
    where ``K`` is the number of visual words (clusters) and ``D`` is the feature
    dimension.

    The layer consists of:
        1. A 1x1 convolution for soft-assignment to ``K`` clusters.
        2. Learnable cluster centroids ``(K, D)``.
        3. VLAD residual accumulation with intra-normalization.
    """

    def __init__(
        self,
        num_clusters: int = 64,
        dim: int = 512,
        normalize_input: bool = True,
        vladv2: bool = False,
        use_faiss: bool = True,
    ) -> None:
        """
        Args:
            num_clusters: Number of visual words K.
            dim: Feature dimension D.
            normalize_input: If True, L2-normalize input descriptors.
            vladv2: If True, use VLADv2 formulation (learnable bias in
                soft-assignment conv). If False, use VLADv1 (no bias,
                alpha-scaled dot-product assignment).
            use_faiss: If True, use FAISS for nearest-neighbor computation
                during parameter initialization (faster than sklearn).
        """
        super().__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = 0.0
        self.vladv2 = vladv2
        self.normalize_input = normalize_input
        self.use_faiss = use_faiss

        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=vladv2)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))

    def init_params(self, clsts: np.ndarray, traindescs: np.ndarray) -> None:
        """Initialize NetVLAD parameters from K-means cluster centers.

        Computes the soft-assignment temperature ``alpha`` and initializes the
        1x1 conv weights and centroids from the provided cluster centers.

        Args:
            clsts: Cluster centers of shape ``(K, D)`` from K-means.
            traindescs: Training descriptors of shape ``(N, D)`` used to
                estimate the soft-assignment temperature.
        """
        if not self.vladv2:
            # VLADv1: alpha-scaled normalized dot-product assignment
            clsts_assign = clsts / np.linalg.norm(clsts, axis=1, keepdims=True)
            dots = np.dot(clsts_assign, traindescs.T)
            dots.sort(0)
            dots = dots[::-1, :]  # descending

            # alpha chosen so that P(correct assignment) ≈ 0.99
            self.alpha = (-np.log(0.01) / np.mean(dots[0, :] - dots[1, :])).item()

            self.centroids = nn.Parameter(torch.from_numpy(clsts))
            self.conv.weight = nn.Parameter(
                torch.from_numpy(self.alpha * clsts_assign)
                .unsqueeze(2).unsqueeze(3)
            )
            self.conv.bias = None
        else:
            # VLADv2: learnable bias, alpha from squared distances
            if not self.use_faiss:
                knn = NearestNeighbors(n_jobs=-1)
                knn.fit(traindescs)
                del traindescs
                ds_sq = np.square(knn.kneighbors(clsts, 2)[1])
                del knn
            else:
                index = faiss.IndexFlatL2(traindescs.shape[1])
                index.add(traindescs)
                del traindescs
                ds_sq = np.square(index.search(clsts, 2)[1])
                del index

            self.alpha = (-np.log(0.01) / np.mean(ds_sq[:, 1] - ds_sq[:, 0])).item()

            self.centroids = nn.Parameter(torch.from_numpy(clsts))
            del clsts, ds_sq

            self.conv.weight = nn.Parameter(
                (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
            )
            self.conv.bias = nn.Parameter(
                -self.alpha * self.centroids.norm(dim=1)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute VLAD descriptor from spatial feature map.

        Args:
            x: Feature map of shape ``(B, D, H, W)``.

        Returns:
            Global descriptor of shape ``(B, K*D)`` after intra-normalization
            and L2 normalization.
        """
        B, D = x.shape[:2]

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)

        # Soft assignment: (B, K, H*W)
        soft_assign = self.conv(x).view(B, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        # Flatten spatial dims: (B, D, H*W)
        x_flat = x.view(B, D, -1)

        # VLAD residual accumulation
        # Use loop to reduce peak memory (trading compute for memory)
        vlad = torch.zeros(
            [B, self.num_clusters, D],
            dtype=x.dtype,
            device=x.device,
        )
        for k in range(self.num_clusters):
            # Residual: (B, D, H*W) - (1, D, 1)
            residual = x_flat - self.centroids[k].unsqueeze(0).unsqueeze(2)
            # Weight by soft assignment: (B, 1, H*W)
            residual = residual * soft_assign[:, k : k + 1, :]
            # Sum over spatial locations: (B, D)
            vlad[:, k, :] = residual.sum(dim=-1)

        # Intra-normalization (per-cluster L2)
        vlad = F.normalize(vlad, p=2, dim=2)
        # Flatten to (B, K*D)
        vlad = vlad.view(B, -1)
        # Global L2 normalization
        vlad = F.normalize(vlad, p=2, dim=1)

        return vlad
