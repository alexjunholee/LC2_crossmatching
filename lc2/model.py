"""LC2 full model: DualEncoder + pooling assembled for cross-modal retrieval.

Provides the ``LC2Model`` class which combines the dual VGG16 encoder with
either GeM (Phase 1) or NetVLAD (Phase 2) pooling, plus a ``from_checkpoint``
factory for loading pretrained weights from the original LC2 release.

The model supports dynamic pooling layer swapping between training phases:
    - Phase 1: GeM pooling → 512-D descriptors (contrastive pre-training)
    - Phase 2: NetVLAD pooling → K*512-D descriptors (triplet fine-tuning)
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Union

from lc2.encoder import DualEncoder
from lc2.gem import GeM
from lc2.netvlad import NetVLAD
from lc2.utils.checkpoint import load_lc2_checkpoint


class LC2Model(nn.Module):
    """Complete LC2 cross-modal place recognition model.

    Combines :class:`DualEncoder` (two VGG16 branches) with a pooling layer
    (GeM or NetVLAD) to produce a fixed-size global descriptor from either
    depth/disparity or range images.

    The forward pass:
        1. Extracts spatial features via the modality-appropriate VGG16 branch.
        2. Aggregates into a global descriptor via the active pooling layer.

    Pooling modes:
        - ``"gem"``: GeM pooling → descriptor dim = encoder_dim (512).
        - ``"netvlad"``: NetVLAD pooling → descriptor dim = num_clusters * encoder_dim.
    """

    def __init__(
        self,
        num_clusters: int = 64,
        encoder_dim: int = 512,
        vladv2: bool = False,
        pooling: str = "netvlad",
    ) -> None:
        """
        Args:
            num_clusters: Number of NetVLAD cluster centers.
            encoder_dim: Feature dimension from the encoder (512 for VGG16).
            vladv2: Whether to use VLADv2 formulation for NetVLAD.
            pooling: Initial pooling mode, ``"gem"`` or ``"netvlad"``.
        """
        super().__init__()
        self.encoder = DualEncoder()
        self.num_clusters = num_clusters
        self.encoder_dim = encoder_dim
        self.vladv2 = vladv2

        # Create both pooling layers (only one is active at a time)
        self.gem = GeM(p=3.0)
        self.netvlad = NetVLAD(
            num_clusters=num_clusters,
            dim=encoder_dim,
            normalize_input=True,
            vladv2=vladv2,
        )

        # Active pooling layer (pointer)
        self._pooling_mode = pooling
        if pooling == "gem":
            self.pool = self.gem
            self.descriptor_dim = encoder_dim
        else:
            self.pool = self.netvlad
            self.descriptor_dim = num_clusters * encoder_dim

    def set_pooling(self, mode: str) -> None:
        """Switch the active pooling layer.

        Paper (Section III.B.2): "To achieve cross-modality learning, two
        training phases are proposed by changing the pooling layer."

        Phase 1 → ``set_pooling("gem")``
        Phase 2 → ``set_pooling("netvlad")``

        Args:
            mode: ``"gem"`` or ``"netvlad"``.
        """
        if mode == "gem":
            self.pool = self.gem
            self.descriptor_dim = self.encoder_dim
        elif mode == "netvlad":
            self.pool = self.netvlad
            self.descriptor_dim = self.num_clusters * self.encoder_dim
        else:
            raise ValueError(f"Unknown pooling mode: {mode}. Use 'gem' or 'netvlad'.")
        self._pooling_mode = mode

    @property
    def pooling_mode(self) -> str:
        return self._pooling_mode

    def forward(self, x: torch.Tensor, is_range: torch.Tensor) -> torch.Tensor:
        """Extract global descriptor from input images.

        Args:
            x: Batch of images, shape ``(B, 3, H, W)``.
            is_range: Boolean tensor ``(B,)`` indicating modality per sample.

        Returns:
            Global descriptors. Shape depends on pooling mode:
            - GeM: ``(B, D)`` where D=encoder_dim (512).
            - NetVLAD: ``(B, K*D)`` where K=num_clusters.
        """
        features = self.encoder(x, is_range)
        descriptors = self.pool(features)
        return descriptors

    def forward_single(self, x: torch.Tensor, is_range: bool) -> torch.Tensor:
        """Extract descriptors for a homogeneous batch (single modality).

        Args:
            x: Batch of images, shape ``(B, 3, H, W)``.
            is_range: If True, all inputs are range images.

        Returns:
            Global descriptors with shape depending on active pooling mode.
        """
        features = self.encoder.forward_single(x, is_range)
        descriptors = self.pool(features)
        return descriptors

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        device: Union[str, torch.device] = "cpu",
        pooling: str = "netvlad",
    ) -> "LC2Model":
        """Load a pretrained LC2 model from a checkpoint file.

        Handles the original LC2 checkpoint format which wraps the model in
        ``nn.DataParallel``, stores encoder and NetVLAD in a single state_dict,
        and may use different key naming conventions.

        Args:
            checkpoint_path: Path to ``dual_encoder.pth.tar`` or similar.
            device: Target device for the loaded model.
            pooling: Initial pooling mode.

        Returns:
            An ``LC2Model`` instance with loaded weights in eval mode.
        """
        checkpoint_path = Path(checkpoint_path)
        state_dict, meta = load_lc2_checkpoint(checkpoint_path, device=device)

        # Infer model configuration from checkpoint
        num_clusters = 64
        encoder_dim = 512
        vladv2 = False

        for key, tensor in state_dict.items():
            if "centroids" in key:
                num_clusters, encoder_dim = tensor.shape
                break

        for key in state_dict:
            if "pool.conv.bias" in key:
                vladv2 = True
                break

        model = cls(
            num_clusters=num_clusters,
            encoder_dim=encoder_dim,
            vladv2=vladv2,
            pooling=pooling,
        )

        # Map old checkpoint keys to new structure
        # Old: pool.xxx → netvlad.xxx (since pool is now a reference)
        remapped_state = {}
        for key, value in state_dict.items():
            new_key = key
            # Remap pool.* keys to netvlad.* for the NetVLAD weights
            if key.startswith("pool."):
                new_key = "netvlad." + key[5:]
            remapped_state[new_key] = value
            # Also keep the original key mapping
            if key not in remapped_state:
                remapped_state[key] = value

        model_state = model.state_dict()
        loaded_keys = set()
        mismatched = []

        for key, value in remapped_state.items():
            if key in model_state:
                if model_state[key].shape == value.shape:
                    model_state[key] = value
                    loaded_keys.add(key)
                else:
                    mismatched.append(
                        f"  {key}: ckpt {value.shape} vs model {model_state[key].shape}"
                    )

        missing = set(model_state.keys()) - loaded_keys
        if missing:
            # Only warn about non-trivial missing keys
            important_missing = [k for k in missing if not k.startswith("gem.")]
            if important_missing:
                print(f"[LC2Model] Warning: {len(important_missing)} keys not loaded:")
                for k in sorted(important_missing):
                    print(f"  missing: {k}")
        if mismatched:
            print(f"[LC2Model] Warning: {len(mismatched)} shape mismatches:")
            for m in mismatched:
                print(m)

        model.load_state_dict(model_state, strict=False)
        model = model.to(device)
        model.eval()

        return model
