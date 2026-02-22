"""Depth Anything 3 (DA3) depth estimator.

Wraps the DA3 monocular depth estimation model for batch inference on
camera images.  Supports multiple DA3 variants via HuggingFace Hub.

DA3 replaces the outdated ManyDepth used in the original LC2 pipeline.
Key advantages:
    - Single-image inference (no temporal context / pose network needed)
    - Superior cross-domain generalization
    - Much simpler integration (~80 lines)
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from lc2.depth.base import BaseDepthEstimator

# ---------------------------------------------------------------------------
# Variant registry: short name -> HuggingFace model identifier
# ---------------------------------------------------------------------------
VARIANTS: Dict[str, str] = {
    "da3-small": "depth-anything/DA3-SMALL",
    "da3": "depth-anything/DA3-LARGE",
    "da3-large": "depth-anything/DA3-LARGE",
    "da3-giant": "depth-anything/DA3-GIANT",
}

# Default variant when none is specified
DEFAULT_VARIANT = "da3-large"


class DA3DepthEstimator(BaseDepthEstimator):
    """Monocular depth estimation using Depth Anything 3.

    Produces per-pixel relative depth maps from single RGB images.

    Args:
        variant: Short name (``"da3-small"``, ``"da3"``, ``"da3-large"``,
            ``"da3-giant"``) or a full HuggingFace model identifier.
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

        # Resolve short name -> HF model id (pass through unknown names)
        self.model_name = VARIANTS.get(variant, variant)

        try:
            from depth_anything_3.api import DepthAnything3
        except ImportError:
            raise ImportError(
                "depth-anything-3 is not installed.  Install via:\n"
                "  pip install awesome-depth-anything-3\n"
                "or:\n"
                "  git clone https://github.com/DepthAnything/Depth-Anything-3\n"
                "  cd Depth-Anything-3 && pip install -e ."
            )

        self.model = DepthAnything3.from_pretrained(
            self.model_name, device=device,
        )
        self.model.eval()

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    @torch.no_grad()
    def estimate(self, image_path: Union[str, Path]) -> np.ndarray:
        """Estimate depth for a single image.

        Args:
            image_path: Path to an RGB image file (PNG, JPG, etc.).

        Returns:
            Depth map of shape ``(H, W)`` as float32 numpy array.
            Values represent relative depth (larger = farther).
        """
        image_path = str(image_path)
        prediction = self.model.inference([image_path])
        depth = prediction.depth[0]  # (H, W)

        if isinstance(depth, torch.Tensor):
            depth = depth.cpu().numpy()

        return depth.astype(np.float32)

    @torch.no_grad()
    def estimate_batch(
        self,
        image_paths: List[Union[str, Path]],
        batch_size: int = 16,
    ) -> List[np.ndarray]:
        """Estimate depth for multiple images.

        Processes images in batches for GPU efficiency.

        Args:
            image_paths: List of paths to RGB image files.
            batch_size: Number of images per inference batch.

        Returns:
            List of depth maps, each ``(H, W)`` float32 numpy arrays.
        """
        all_depths: List[np.ndarray] = []
        paths_str = [str(p) for p in image_paths]

        for i in range(0, len(paths_str), batch_size):
            batch_paths = paths_str[i : i + batch_size]
            prediction = self.model.inference(batch_paths)

            for j in range(len(batch_paths)):
                d = prediction.depth[j]
                if isinstance(d, torch.Tensor):
                    d = d.cpu().numpy()
                all_depths.append(d.astype(np.float32))

        return all_depths

    @torch.no_grad()
    def estimate_from_array(self, image: np.ndarray) -> np.ndarray:
        """Estimate depth from an in-memory numpy image array.

        Saves to a temporary file, runs inference, and cleans up.
        For bulk processing, prefer :meth:`estimate_batch` with file paths.

        Args:
            image: RGB image array of shape ``(H, W, 3)``, dtype uint8.

        Returns:
            Depth map of shape ``(H, W)`` as float32 numpy array.
        """
        import tempfile
        from PIL import Image

        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
            pil_img = Image.fromarray(image)
            pil_img.save(tmp.name)
            return self.estimate(tmp.name)
