"""Abstract base class for depth estimators.

All depth estimation backends must subclass :class:`BaseDepthEstimator`
and implement at minimum :meth:`estimate` and :meth:`estimate_batch`.
"""

import abc
import tempfile
from pathlib import Path
from typing import List, Union

import numpy as np


class BaseDepthEstimator(abc.ABC):
    """Abstract interface for monocular / stereo depth estimation.

    Subclasses wrap a specific depth model (DA3, DINOv3-DPT, DUSt3R, ...)
    and expose a uniform ``estimate`` / ``estimate_batch`` API that always
    returns ``(H, W)`` float32 numpy depth maps.
    """

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def estimate(self, image_path: Union[str, Path]) -> np.ndarray:
        """Estimate depth for a single image.

        Args:
            image_path: Path to an RGB image file (PNG, JPG, etc.).

        Returns:
            Depth map of shape ``(H, W)`` as float32 numpy array.
        """

    @abc.abstractmethod
    def estimate_batch(
        self,
        image_paths: List[Union[str, Path]],
        batch_size: int = 16,
    ) -> List[np.ndarray]:
        """Estimate depth for multiple images.

        Args:
            image_paths: List of paths to RGB image files.
            batch_size: Number of images per inference batch.

        Returns:
            List of ``(H, W)`` float32 depth maps, one per input path.
        """

    # ------------------------------------------------------------------
    # Default convenience method (can be overridden for efficiency)
    # ------------------------------------------------------------------

    def estimate_from_array(self, image: np.ndarray) -> np.ndarray:
        """Estimate depth from an in-memory numpy image array.

        Default implementation: save to a temporary file, call
        :meth:`estimate`, then clean up.  Subclasses that support direct
        tensor input should override this for better performance.

        Args:
            image: RGB image array of shape ``(H, W, 3)``, dtype uint8.

        Returns:
            Depth map of shape ``(H, W)`` as float32 numpy array.
        """
        from PIL import Image as _PILImage

        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
            pil_img = _PILImage.fromarray(image)
            pil_img.save(tmp.name)
            return self.estimate(tmp.name)
