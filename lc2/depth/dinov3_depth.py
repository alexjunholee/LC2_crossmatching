"""DINOv3 DPT depth estimator.

Uses the DPT (Dense Prediction Transformer) depth head shipped with
DINOv3 (or DINOv2 as fallback) via ``torch.hub``.  This gives a
self-supervised monocular depth backbone without requiring any
additional pip packages beyond PyTorch and torchvision.

References:
    - DINOv2: https://github.com/facebookresearch/dinov2
    - DPT: Ranftl et al., "Vision Transformers for Dense Prediction", ICCV 2021
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from lc2.depth.base import BaseDepthEstimator

# ---------------------------------------------------------------------------
# Variant registry
# ---------------------------------------------------------------------------
VARIANTS: Dict[str, str] = {
    "dinov3_vits14": "dinov2_vits14",   # torch.hub backbone name
    "dinov3_vitb14": "dinov2_vitb14",
    "dinov3_vitl14": "dinov2_vitl14",
    "dinov3_vitg14": "dinov2_vitg14",
}

# DPT head names corresponding to each backbone
_DPT_HEADS: Dict[str, str] = {
    "dinov2_vits14": "dinov2_vits14_ld",
    "dinov2_vitb14": "dinov2_vitb14_ld",
    "dinov2_vitl14": "dinov2_vitl14_ld",
    "dinov2_vitg14": "dinov2_vitg14_ld",
}

DEFAULT_VARIANT = "dinov3_vitl14"

# ViT patch size -- input spatial dims must be divisible by this
_PATCH_SIZE = 14

# ImageNet normalization constants
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _pad_to_multiple(h: int, w: int, m: int) -> Tuple[int, int, int, int]:
    """Compute (left, right, top, bottom) padding so h,w become multiples of *m*."""
    pad_h = (m - h % m) % m
    pad_w = (m - w % m) % m
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    return left, right, top, bottom


class DINOv3DepthEstimator(BaseDepthEstimator):
    """Monocular depth via DINOv3 / DINOv2 DPT head.

    The model is loaded through ``torch.hub``.  We first try the
    ``facebookresearch/dinov3`` hub repo; if that is not available
    (e.g. the repo has not been published yet) we transparently fall
    back to ``facebookresearch/dinov2``.

    Args:
        variant: One of the keys in :data:`VARIANTS`
            (``"dinov3_vits14"``, ``"dinov3_vitb14"``, ``"dinov3_vitl14"``,
            ``"dinov3_vitg14"``).
        device: CUDA device string.  Auto-detected if *None*.
    """

    def __init__(
        self,
        variant: str = DEFAULT_VARIANT,
        device: Optional[str] = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.variant = variant

        backbone_name = VARIANTS.get(variant)
        if backbone_name is None:
            raise ValueError(
                f"Unknown DINOv3 variant '{variant}'. "
                f"Choose from: {list(VARIANTS.keys())}"
            )

        dpt_name = _DPT_HEADS[backbone_name]

        # Try dinov3 hub first, fall back to dinov2
        model = None
        for repo in ("facebookresearch/dinov3", "facebookresearch/dinov2"):
            try:
                model = torch.hub.load(repo, dpt_name, pretrained=True)
                break
            except Exception:
                continue

        if model is None:
            raise RuntimeError(
                f"Could not load DPT head '{dpt_name}' from torch.hub.  "
                "Ensure you have internet access or a local hub cache for "
                "facebookresearch/dinov2 (or dinov3)."
            )

        self.model = model.to(self.device).eval()

        # Pre-build normalisation tensors (on device)
        self._mean = torch.tensor(_IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(_IMAGENET_STD, device=self.device).view(1, 3, 1, 1)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _preprocess(self, image: np.ndarray) -> Tuple[torch.Tensor, int, int]:
        """HWC uint8 -> NCHW float32, normalised, padded to patch-multiple.

        Returns:
            (tensor, orig_h, orig_w) so the caller can crop back.
        """
        h, w = image.shape[:2]
        tensor = (
            torch.from_numpy(image)
            .permute(2, 0, 1)        # HWC -> CHW
            .unsqueeze(0)             # -> NCHW
            .float()
            .div_(255.0)
            .to(self.device)
        )
        tensor = (tensor - self._mean) / self._std

        # Pad so spatial dims are multiples of _PATCH_SIZE
        left, right, top, bottom = _pad_to_multiple(h, w, _PATCH_SIZE)
        if left or right or top or bottom:
            tensor = F.pad(tensor, (left, right, top, bottom), mode="reflect")

        return tensor, h, w

    def _postprocess(
        self, depth_tensor: torch.Tensor, orig_h: int, orig_w: int,
    ) -> np.ndarray:
        """Resize back to original resolution and return float32 numpy."""
        # depth_tensor may be (1,1,H',W') or (1,H',W') or (H',W')
        if depth_tensor.dim() == 4:
            depth_tensor = depth_tensor.squeeze(0).squeeze(0)
        elif depth_tensor.dim() == 3:
            depth_tensor = depth_tensor.squeeze(0)

        # Bilinear resize to original (h, w)
        depth_tensor = F.interpolate(
            depth_tensor.unsqueeze(0).unsqueeze(0),
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze()

        return depth_tensor.cpu().numpy().astype(np.float32)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @torch.no_grad()
    def estimate(self, image_path: Union[str, Path]) -> np.ndarray:
        """Estimate depth for a single image.

        Args:
            image_path: Path to an RGB image file.

        Returns:
            Depth map ``(H, W)`` float32.
        """
        from PIL import Image as _PILImage

        img = np.asarray(_PILImage.open(str(image_path)).convert("RGB"))
        tensor, h, w = self._preprocess(img)
        depth = self.model(tensor)
        return self._postprocess(depth, h, w)

    @torch.no_grad()
    def estimate_batch(
        self,
        image_paths: List[Union[str, Path]],
        batch_size: int = 16,
    ) -> List[np.ndarray]:
        """Estimate depth for multiple images.

        .. note::
            Because input images may differ in resolution, images are
            processed individually (batch_size controls how many are
            kept in flight, but each forward pass uses N=1).
            Override with a custom collation strategy if all images
            share the same resolution.

        Args:
            image_paths: List of paths to RGB images.
            batch_size: Ignored for now (individual inference).

        Returns:
            List of ``(H, W)`` float32 depth maps.
        """
        return [self.estimate(p) for p in image_paths]
