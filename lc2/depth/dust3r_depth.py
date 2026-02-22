"""DUSt3R-based depth estimator.

DUSt3R (Dense and Unconstrained Stereo 3D Reconstruction) jointly
predicts depth and correspondences from image *pairs*.  This wrapper
exposes the standard :class:`BaseDepthEstimator` interface by:

    - ``estimate(path)``: self-pairing (same image as both views).
    - ``estimate_batch(paths)``: using consecutive frames as stereo pairs.
    - ``estimate_pair(path1, path2)``: explicit stereo pair depth.

Reference:
    Wang et al., "DUSt3R: Geometric 3D Vision Made Easy", CVPR 2024.
    https://github.com/naver/dust3r
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from lc2.depth.base import BaseDepthEstimator

DEFAULT_MODEL = "DUSt3R_ViTLarge_BaseDecoder_512_dpt"


class DUSt3RDepthEstimator(BaseDepthEstimator):
    """Depth estimation via DUSt3R stereo matching.

    Args:
        model_name: DUSt3R checkpoint identifier.  Passed to
            ``load_model``.  Defaults to the ViT-Large DPT variant.
        device: CUDA device string.  Auto-detected if *None*.
        image_size: Input resolution fed to DUSt3R (must match the
            checkpoint's expected size, typically 512).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        image_size: int = 512,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_name = model_name
        self.image_size = image_size

        # Lazy import -- DUSt3R is an optional heavy dependency
        try:
            from dust3r.model import AsymmetricCroCo3DStereo  # noqa: F401
            from dust3r.utils.device import to_numpy  # noqa: F401
        except ImportError:
            raise ImportError(
                "DUSt3R is not installed.  Install via:\n"
                "  git clone --recursive https://github.com/naver/dust3r\n"
                "  cd dust3r && pip install -e .\n"
                "See https://github.com/naver/dust3r for details."
            )

        from dust3r.model import AsymmetricCroCo3DStereo  # noqa: F811

        self.model = AsymmetricCroCo3DStereo.from_pretrained(
            model_name,
        ).to(self.device).eval()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_image_pair(
        self, path1: Union[str, Path], path2: Union[str, Path],
    ) -> list:
        """Load two images and prepare them as a DUSt3R image pair."""
        from dust3r.image_pairs import make_pairs
        from dust3r.utils.image import load_images

        images = load_images(
            [str(path1), str(path2)], size=self.image_size,
        )
        pairs = make_pairs(
            images, scene_graph="complete", prefilter=None, symmetrize=True,
        )
        return pairs

    def _run_pair(
        self, path1: Union[str, Path], path2: Union[str, Path],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run DUSt3R on a pair and return (depth1, depth2).

        Returns:
            Tuple of two ``(H, W)`` float32 depth maps, one per input view.
        """
        from dust3r.inference import inference
        from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

        pairs = self._load_image_pair(path1, path2)

        with torch.no_grad():
            output = inference(pairs, self.model, self.device, batch_size=1)

        scene = global_aligner(
            output,
            device=self.device,
            mode=GlobalAlignerMode.PairViewer,
        )

        depths = scene.get_depthmaps()
        results = []
        for d in depths:
            if isinstance(d, torch.Tensor):
                d = d.cpu().numpy()
            results.append(d.astype(np.float32))

        return results[0], results[1]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @torch.no_grad()
    def estimate(self, image_path: Union[str, Path]) -> np.ndarray:
        """Estimate depth for a single image via self-pairing.

        Uses the same image as both views in the stereo pair, which
        degenerates to monocular-like depth.  For best results, use
        :meth:`estimate_pair` with an actual stereo pair.

        Args:
            image_path: Path to an RGB image file.

        Returns:
            Depth map ``(H, W)`` float32.
        """
        depth, _ = self._run_pair(image_path, image_path)
        return depth

    @torch.no_grad()
    def estimate_pair(
        self,
        path1: Union[str, Path],
        path2: Union[str, Path],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate depth from an explicit stereo image pair.

        Args:
            path1: Path to the first RGB image.
            path2: Path to the second RGB image.

        Returns:
            Tuple ``(depth1, depth2)`` of ``(H, W)`` float32 depth maps.
        """
        return self._run_pair(path1, path2)

    @torch.no_grad()
    def estimate_batch(
        self,
        image_paths: List[Union[str, Path]],
        batch_size: int = 16,
    ) -> List[np.ndarray]:
        """Estimate depth for multiple images using consecutive pairing.

        Consecutive frames ``(i, i+1)`` are treated as stereo pairs.
        The depth for the *first* view of each pair is kept.  The last
        image uses a self-pair.

        Args:
            image_paths: Ordered list of image paths (e.g. a video sequence).
            batch_size: Ignored (pair-wise inference).

        Returns:
            List of ``(H, W)`` float32 depth maps, one per input path.
        """
        if len(image_paths) == 0:
            return []

        if len(image_paths) == 1:
            return [self.estimate(image_paths[0])]

        depths: List[np.ndarray] = []

        for i in range(len(image_paths) - 1):
            d1, d2 = self._run_pair(image_paths[i], image_paths[i + 1])
            depths.append(d1)

        # Last frame: use (last, last-1) pair and keep the *first* view
        d_last, _ = self._run_pair(
            image_paths[-1], image_paths[-2],
        )
        depths.append(d_last)

        return depths
