"""Augmentations for LiDAR range images.

Provides composable, randomised augmentation transforms that operate on
normalised ``(H, W)`` float32 range images (values in [0, 1], zeros denoting
unoccupied pixels).  Designed for training-time data augmentation of range
images produced by :func:`lc2.lidar.projection.lidar_to_range_image`.

Each augmentation callable follows the signature::

    (range_img, mask=None) -> (result, mask)

where *result* is the augmented range image (float32, clipped to [0, 1]) and
*mask* is a boolean array indicating occupied pixels.

Classes
-------
- :class:`RangeAugmentation` -- composable pipeline
- :class:`RandomBeamDrop` -- drop entire rows (simulate sparse sensor)
- :class:`RandomPointJitter` -- Gaussian noise on occupied pixels
- :class:`RandomRangeNoise` -- multiplicative range noise
- :class:`RandomOcclusion` -- random rectangular occlusion patches
- :class:`RandomFOVCrop` -- random azimuth sub-crop
- :class:`SensorTransfer` -- downsample to simulate a different sensor
- :class:`RandomIntensityShift` -- affine shift on range values
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Type

import numpy as np

from lc2.lidar.sensors import SensorConfig

__all__ = [
    "RangeAugmentation",
    "RandomBeamDrop",
    "RandomPointJitter",
    "RandomRangeNoise",
    "RandomOcclusion",
    "RandomFOVCrop",
    "SensorTransfer",
    "RandomIntensityShift",
]


# ──────────────────────────────────────────────────────────────
# Type alias
# ──────────────────────────────────────────────────────────────

AugResult = Tuple[np.ndarray, np.ndarray]
"""(range_img float32 [0,1], mask bool) tuple returned by each augmentation."""


# ──────────────────────────────────────────────────────────────
# Composable pipeline
# ──────────────────────────────────────────────────────────────


class RangeAugmentation:
    """Composable augmentation pipeline for range images.

    Sequentially applies a list of augmentation callables.  Each callable
    must accept ``(range_img, mask)`` and return ``(result, mask)``.

    Example::

        aug = RangeAugmentation([
            RandomBeamDrop(p=0.1),
            RandomPointJitter(sigma=0.02),
        ])
        result = aug(range_img)       # returns augmented (H, W) ndarray
        result, mask = aug(range_img, mask=valid)

    Args:
        augmentations: Ordered sequence of augmentation callables.
    """

    def __init__(self, augmentations: Sequence) -> None:
        self.augmentations = list(augmentations)

    def __call__(
        self,
        range_img: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Apply all augmentations in sequence.

        Args:
            range_img: ``(H, W)`` float32 range image in [0, 1].
            mask: Optional ``(H, W)`` boolean occupancy mask.  If *None*,
                the mask is inferred from ``range_img > 0``.

        Returns:
            Augmented range image ``(H, W)`` float32 in [0, 1].
        """
        img = range_img.copy().astype(np.float32)
        if mask is None:
            mask = img > 0
        else:
            mask = mask.copy()

        for aug in self.augmentations:
            img, mask = aug(img, mask)

        return img

    @classmethod
    def from_config(cls, cfg: dict) -> "RangeAugmentation":
        """Build a :class:`RangeAugmentation` pipeline from a config dict.

        Supported keys and their corresponding augmentation classes:

        ====================  ============================  ===========
        Key                   Class                         Value type
        ====================  ============================  ===========
        ``beam_drop``         :class:`RandomBeamDrop`       float (p)
        ``point_jitter``      :class:`RandomPointJitter`    float (sigma)
        ``range_noise``       :class:`RandomRangeNoise`     float (sigma)
        ``occlusion_patches`` :class:`RandomOcclusion`      int (n_patches)
        ``fov_crop_min``      :class:`RandomFOVCrop`        float (min_fov_frac)
        ``intensity_shift``   :class:`RandomIntensityShift` float (max_shift)
        ====================  ============================  ===========

        Args:
            cfg: Dictionary mapping config keys to parameter values.

        Returns:
            Configured :class:`RangeAugmentation` pipeline.
        """
        _KEY_MAP: Dict[str, Tuple[Type, str]] = {
            "beam_drop": (RandomBeamDrop, "p"),
            "point_jitter": (RandomPointJitter, "sigma"),
            "range_noise": (RandomRangeNoise, "sigma"),
            "occlusion_patches": (RandomOcclusion, "n_patches"),
            "fov_crop_min": (RandomFOVCrop, "min_fov_frac"),
            "intensity_shift": (RandomIntensityShift, "max_shift"),
        }

        augmentations: List = []
        for key, (aug_cls, param_name) in _KEY_MAP.items():
            if key in cfg:
                augmentations.append(aug_cls(**{param_name: cfg[key]}))

        return cls(augmentations)

    def __repr__(self) -> str:
        lines = [f"{self.__class__.__name__}(["]
        for aug in self.augmentations:
            lines.append(f"    {aug!r},")
        lines.append("])")
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# Individual augmentations
# ──────────────────────────────────────────────────────────────


class RandomBeamDrop:
    """Drop entire rows to simulate a sparser LiDAR sensor.

    Each row (beam) is independently dropped with probability *p*.  A minimum
    fraction of rows is always retained to avoid fully empty images.

    Args:
        p: Per-row drop probability.
        min_remaining: Minimum fraction of rows to keep (0, 1].
    """

    def __init__(self, p: float = 0.1, min_remaining: float = 0.5) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")
        if not 0.0 < min_remaining <= 1.0:
            raise ValueError(f"min_remaining must be in (0, 1], got {min_remaining}")
        self.p = p
        self.min_remaining = min_remaining

    def __call__(self, range_img: np.ndarray, mask: np.ndarray) -> AugResult:
        H, W = range_img.shape
        min_keep = max(1, int(np.ceil(H * self.min_remaining)))

        # Generate per-row drop decisions
        drop = np.random.random(H) < self.p

        # Ensure we keep at least min_keep rows
        keep_indices = np.where(~drop)[0]
        if len(keep_indices) < min_keep:
            # Randomly un-drop rows until we meet the minimum
            dropped_indices = np.where(drop)[0]
            n_restore = min_keep - len(keep_indices)
            restore = np.random.choice(dropped_indices, size=n_restore, replace=False)
            drop[restore] = False

        result = range_img.copy()
        result[drop] = 0.0
        new_mask = mask.copy()
        new_mask[drop] = False

        return result, new_mask

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p}, min_remaining={self.min_remaining})"


class RandomPointJitter:
    """Additive Gaussian noise on occupied pixels only.

    Applies i.i.d. Gaussian noise :math:`\\mathcal{N}(0, \\sigma^2)` to each
    occupied pixel value, then clips the result to [0, 1].

    Args:
        sigma: Standard deviation of the Gaussian noise.
    """

    def __init__(self, sigma: float = 0.02) -> None:
        if sigma < 0:
            raise ValueError(f"sigma must be >= 0, got {sigma}")
        self.sigma = sigma

    def __call__(self, range_img: np.ndarray, mask: np.ndarray) -> AugResult:
        result = range_img.copy()
        occupied = mask & (range_img > 0)
        n_occupied = int(occupied.sum())
        if n_occupied > 0 and self.sigma > 0:
            noise = np.random.normal(0.0, self.sigma, size=n_occupied).astype(np.float32)
            result[occupied] += noise
            result = np.clip(result, 0.0, 1.0)
        # Ensure unoccupied pixels remain zero
        result[~mask] = 0.0
        return result, mask

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(sigma={self.sigma})"


class RandomRangeNoise:
    """Multiplicative range noise: :math:`r \\leftarrow r \\cdot (1 + \\mathcal{N}(0, \\sigma))`.

    Simulates range measurement noise that scales with distance.

    Args:
        sigma: Standard deviation of the multiplicative noise factor.
    """

    def __init__(self, sigma: float = 0.01) -> None:
        if sigma < 0:
            raise ValueError(f"sigma must be >= 0, got {sigma}")
        self.sigma = sigma

    def __call__(self, range_img: np.ndarray, mask: np.ndarray) -> AugResult:
        result = range_img.copy()
        occupied = mask & (range_img > 0)
        n_occupied = int(occupied.sum())
        if n_occupied > 0 and self.sigma > 0:
            factors = (1.0 + np.random.normal(0.0, self.sigma, size=n_occupied)).astype(np.float32)
            result[occupied] *= factors
            result = np.clip(result, 0.0, 1.0)
        result[~mask] = 0.0
        return result, mask

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(sigma={self.sigma})"


class RandomOcclusion:
    """Random rectangular occlusion patches.

    Places *n_patches* axis-aligned rectangles of random size and position
    onto the range image, setting all covered pixels to zero.

    Args:
        n_patches: Number of occlusion rectangles per call.
        max_size: Maximum fraction of each image dimension that a single
            patch can span (independently for height and width).
    """

    def __init__(self, n_patches: int = 3, max_size: float = 0.1) -> None:
        if n_patches < 0:
            raise ValueError(f"n_patches must be >= 0, got {n_patches}")
        if not 0.0 < max_size <= 1.0:
            raise ValueError(f"max_size must be in (0, 1], got {max_size}")
        self.n_patches = n_patches
        self.max_size = max_size

    def __call__(self, range_img: np.ndarray, mask: np.ndarray) -> AugResult:
        result = range_img.copy()
        new_mask = mask.copy()
        H, W = result.shape

        for _ in range(self.n_patches):
            ph = max(1, int(np.random.uniform(1, H * self.max_size)))
            pw = max(1, int(np.random.uniform(1, W * self.max_size)))
            y0 = np.random.randint(0, max(1, H - ph + 1))
            x0 = np.random.randint(0, max(1, W - pw + 1))
            result[y0 : y0 + ph, x0 : x0 + pw] = 0.0
            new_mask[y0 : y0 + ph, x0 : x0 + pw] = False

        return result, new_mask

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(n_patches={self.n_patches}, "
            f"max_size={self.max_size})"
        )


class RandomFOVCrop:
    """Random azimuth (horizontal) field-of-view sub-crop.

    Selects a contiguous azimuth range spanning at least *min_fov_frac* of
    the full 360-degree view and zeros out all columns outside the crop.
    The crop wraps around the image boundary (azimuth is cyclic).

    Args:
        min_fov_frac: Minimum fraction of columns to keep, in (0, 1].
    """

    def __init__(self, min_fov_frac: float = 0.7) -> None:
        if not 0.0 < min_fov_frac <= 1.0:
            raise ValueError(f"min_fov_frac must be in (0, 1], got {min_fov_frac}")
        self.min_fov_frac = min_fov_frac

    def __call__(self, range_img: np.ndarray, mask: np.ndarray) -> AugResult:
        H, W = range_img.shape
        min_cols = max(1, int(np.ceil(W * self.min_fov_frac)))

        # Random crop width in [min_cols, W]
        crop_w = np.random.randint(min_cols, W + 1)
        # Random start column (wrapping handled via modular indexing)
        start = np.random.randint(0, W)

        result = range_img.copy()
        new_mask = mask.copy()

        if crop_w >= W:
            # Full FOV -- no cropping
            return result, new_mask

        # Build column mask for the kept region (handles wraparound)
        keep_cols = np.zeros(W, dtype=bool)
        end = start + crop_w
        if end <= W:
            keep_cols[start:end] = True
        else:
            keep_cols[start:] = True
            keep_cols[: end - W] = True

        drop_cols = ~keep_cols
        result[:, drop_cols] = 0.0
        new_mask[:, drop_cols] = False

        return result, new_mask

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(min_fov_frac={self.min_fov_frac})"


class SensorTransfer:
    """Downsample a range image to simulate a different LiDAR sensor.

    Resizes the range image from the *source* sensor resolution to the
    *target* sensor resolution using area-based downsampling (block-max for
    range images). Only downsampling is supported; if the target is larger
    in either dimension, that dimension is left unchanged.

    The occupancy mask is recomputed after resampling.

    Args:
        source: Source sensor configuration (defines input resolution).
        target: Target sensor configuration (defines output resolution).
    """

    def __init__(self, source: SensorConfig, target: SensorConfig) -> None:
        self.source = source
        self.target = target

    def __call__(self, range_img: np.ndarray, mask: np.ndarray) -> AugResult:
        H_src, W_src = range_img.shape
        H_tgt = min(H_src, self.target.H)
        W_tgt = min(W_src, self.target.W)

        if H_tgt == H_src and W_tgt == W_src:
            return range_img.copy(), mask.copy()

        # Block-based downsampling: partition into blocks and take the max
        # (closest range within block, preserving structure).
        result = _block_downsample(range_img, H_tgt, W_tgt)
        new_mask = result > 0

        return result, new_mask

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"source='{self.source.name}' ({self.source.H}x{self.source.W}), "
            f"target='{self.target.name}' ({self.target.H}x{self.target.W}))"
        )


def _block_downsample(img: np.ndarray, H_out: int, W_out: int) -> np.ndarray:
    """Downsample a 2D image by partitioning into blocks and taking the max.

    For range images where zero = unoccupied, the maximum within each block
    preserves the most prominent structure.

    Args:
        img: Input ``(H, W)`` float32 image.
        H_out: Target number of rows.
        W_out: Target number of columns.

    Returns:
        Downsampled ``(H_out, W_out)`` float32 image.
    """
    H_in, W_in = img.shape
    result = np.zeros((H_out, W_out), dtype=np.float32)

    # Row and column bin edges
    row_edges = np.linspace(0, H_in, H_out + 1).astype(int)
    col_edges = np.linspace(0, W_in, W_out + 1).astype(int)

    for i in range(H_out):
        for j in range(W_out):
            block = img[row_edges[i] : row_edges[i + 1], col_edges[j] : col_edges[j + 1]]
            if block.size > 0:
                # Among occupied pixels, take the maximum range value.
                # If the entire block is unoccupied (all zeros), result stays 0.
                result[i, j] = block.max()

    return result


class RandomIntensityShift:
    """Affine transform on range values: :math:`r \\leftarrow r + \\text{shift}`.

    Applies a random global additive shift drawn uniformly from
    ``[-max_shift, +max_shift]`` to all occupied pixel values.  The result
    is clipped to [0, 1].

    Args:
        max_shift: Maximum absolute shift magnitude.
    """

    def __init__(self, max_shift: float = 0.1) -> None:
        if max_shift < 0:
            raise ValueError(f"max_shift must be >= 0, got {max_shift}")
        self.max_shift = max_shift

    def __call__(self, range_img: np.ndarray, mask: np.ndarray) -> AugResult:
        result = range_img.copy()
        if self.max_shift > 0:
            shift = np.random.uniform(-self.max_shift, self.max_shift)
            occupied = mask & (range_img > 0)
            result[occupied] += shift
            result = np.clip(result, 0.0, 1.0)
        result[~mask] = 0.0
        return result, mask

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(max_shift={self.max_shift})"
