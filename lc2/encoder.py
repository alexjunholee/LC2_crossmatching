"""Dual VGG16 encoder for LC2 cross-modal place recognition.

Two separate VGG16 feature extractors: one for depth/disparity images (camera),
one for range images (LiDAR). Only conv5 layers are trainable; all earlier layers
are frozen at ImageNet-pretrained weights.

Reference: Lee et al., "(LC)²: LiDAR-Camera Loop Constraints for Cross-Modal
Place Recognition", RA-L 2023.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


class DualEncoder(nn.Module):
    """Dual-branch VGG16 encoder for cross-modal feature extraction.

    The depth branch (encoder_d) processes camera-derived depth/disparity images.
    The range branch (encoder_r) processes LiDAR-derived range images.
    Both branches share the same VGG16 architecture but have independent weights.

    During inference, each input sample is routed to exactly one branch based on
    the ``is_range`` mask. The outputs are combined via masked addition so that
    the batch can contain a mix of depth and range inputs.

    Attributes:
        enc_dim: Output feature dimension (512 for VGG16 conv5).
    """

    enc_dim: int = 512

    def __init__(self, pretrained_weights: Optional[str] = None) -> None:
        """
        Args:
            pretrained_weights: If ``"imagenet"`` or ``None``, initialize both
                branches from ImageNet-pretrained VGG16. Otherwise interpreted
                as a path or torchvision weight enum.
        """
        super().__init__()

        weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained_weights in (None, "imagenet") else pretrained_weights
        encoder_d = models.vgg16(weights=weights)
        encoder_r = models.vgg16(weights=weights)

        # VGG16 features has 31 children (conv, relu, pool layers).
        # features[:-2] removes the last MaxPool2d and preceding ReLU,
        # keeping through conv5_3 + ReLU (index 28).
        layers_d = list(encoder_d.features.children())[:-2]
        layers_r = list(encoder_r.features.children())[:-2]

        # Freeze all layers except the last 5 (conv5_1, relu, conv5_2, relu, conv5_3).
        # VGG16 features[:-2] has 29 layers; [-5:] = indices 24..28 = conv5 block.
        for layer in layers_d[:-5]:
            for p in layer.parameters():
                p.requires_grad = False
        for layer in layers_r[:-5]:
            for p in layer.parameters():
                p.requires_grad = False

        self.encoder_d = nn.Sequential(*layers_d)
        self.encoder_r = nn.Sequential(*layers_r)

    def forward(self, x: torch.Tensor, is_range: torch.Tensor) -> torch.Tensor:
        """Forward pass with modality-conditional routing.

        Args:
            x: Input images, shape ``(B, 3, H, W)``.
            is_range: Boolean tensor of shape ``(B,)``.
                ``True`` routes to range branch, ``False`` to depth branch.

        Returns:
            Feature maps of shape ``(B, 512, H/16, W/16)``.
        """
        # Cast to float for multiplication; shape (B, 1, 1, 1) for broadcasting
        idx_depth = (~is_range).float().unsqueeze(1).unsqueeze(2).unsqueeze(3)
        idx_range = is_range.float().unsqueeze(1).unsqueeze(2).unsqueeze(3)

        out_depth = self.encoder_d(x) * idx_depth
        out_range = self.encoder_r(x) * idx_range

        return out_depth + out_range

    def forward_single(self, x: torch.Tensor, is_range: bool) -> torch.Tensor:
        """Forward pass for a homogeneous batch (all same modality).

        More efficient than ``forward()`` when the entire batch shares a single
        modality, since only one branch is evaluated.

        Args:
            x: Input images, shape ``(B, 3, H, W)``.
            is_range: If ``True``, use range branch; otherwise depth branch.

        Returns:
            Feature maps of shape ``(B, 512, H/16, W/16)``.
        """
        if is_range:
            return self.encoder_r(x)
        return self.encoder_d(x)
