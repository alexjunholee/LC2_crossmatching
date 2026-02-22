"""Sparse-region infill strategies for LiDAR range images.

LiDAR range images are inherently sparse — many pixels have no return due to
beam spacing, occlusion, or absorptive surfaces.  This module provides several
infill (hole-filling) strategies that operate on ``(H, W)`` float32 range
images with an accompanying boolean occupancy mask.

Strategies
----------
====================  ==========================================================
Name                  Description
====================  ==========================================================
``none``              Identity (no infill).
``nearest_neighbor``  EDT-based nearest occupied pixel within ``max_dist``.
``bilateral``         Edge-preserving bilateral-like iterative filter.
``morphological``     Grey dilation followed by empty-pixel fill.
``adaptive``          Block-wise occupancy selects morphological vs. NN infill.
====================  ==========================================================

All public functions share the same signature::

    fn(range_img: ndarray[H, W], mask: ndarray[H, W], ...) -> ndarray[H, W]

where *range_img* is float32, *mask* is bool (True = occupied), and the return
is the filled float32 range image.
"""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np
from scipy.ndimage import (
    distance_transform_edt,
    gaussian_filter,
    grey_dilation,
    label,
    uniform_filter,
)

__all__ = [
    "get_infill_fn",
    "nearest_neighbor_infill",
    "bilateral_infill",
    "morphological_infill",
    "adaptive_infill",
]

# Type alias for infill callables
InfillFn = Callable[..., np.ndarray]


# ──────────────────────────────────────────────────────────────
# Registry / lookup
# ──────────────────────────────────────────────────────────────


def _no_infill(range_img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Identity infill — returns the input unchanged."""
    return range_img.copy()


_INFILL_REGISTRY: Dict[str, InfillFn] = {}


def _build_registry() -> None:
    _INFILL_REGISTRY["none"] = _no_infill
    _INFILL_REGISTRY["nearest_neighbor"] = nearest_neighbor_infill
    _INFILL_REGISTRY["bilateral"] = bilateral_infill
    _INFILL_REGISTRY["morphological"] = morphological_infill
    _INFILL_REGISTRY["adaptive"] = adaptive_infill


def get_infill_fn(name: str) -> InfillFn:
    """Look up an infill function by name.

    Args:
        name: One of ``"none"``, ``"nearest_neighbor"``, ``"bilateral"``,
            ``"morphological"``, ``"adaptive"``.

    Returns:
        The corresponding infill callable.

    Raises:
        KeyError: If *name* is not registered.
    """
    if not _INFILL_REGISTRY:
        _build_registry()
    if name not in _INFILL_REGISTRY:
        available = ", ".join(sorted(_INFILL_REGISTRY))
        raise KeyError(
            f"Unknown infill method '{name}'. Available: {available}"
        )
    return _INFILL_REGISTRY[name]


# ──────────────────────────────────────────────────────────────
# Nearest-neighbor infill (EDT-based)
# ──────────────────────────────────────────────────────────────


def nearest_neighbor_infill(
    range_img: np.ndarray,
    mask: np.ndarray,
    max_dist: int = 3,
) -> np.ndarray:
    """Fill empty pixels with the value of the nearest occupied pixel.

    Uses :func:`scipy.ndimage.distance_transform_edt` to compute, for every
    empty pixel, the indices of the closest occupied pixel in the Euclidean
    (pixel-grid) sense.  Only empty pixels within *max_dist* pixels of an
    occupied pixel are filled; the rest remain zero.

    Args:
        range_img: ``(H, W)`` float32 range image (0 = empty).
        mask: ``(H, W)`` boolean occupancy mask (True = occupied).
        max_dist: Maximum Euclidean pixel distance for infill.  Empty pixels
            farther than this from any occupied pixel are left at zero.

    Returns:
        ``(H, W)`` float32 filled range image.
    """
    out = range_img.copy()
    empty = ~mask

    if not empty.any():
        return out

    # EDT on the *empty* mask gives distance-to-nearest-occupied for each pixel.
    # `indices` gives the row, col of that nearest occupied pixel.
    dist, indices = distance_transform_edt(empty, return_distances=True, return_indices=True)

    # Only fill pixels within max_dist of an occupied pixel
    fill_mask = empty & (dist <= max_dist)
    out[fill_mask] = range_img[indices[0][fill_mask], indices[1][fill_mask]]

    return out


# ──────────────────────────────────────────────────────────────
# Bilateral-like infill
# ──────────────────────────────────────────────────────────────


def bilateral_infill(
    range_img: np.ndarray,
    mask: np.ndarray,
    sigma_s: float = 2.0,
    sigma_r: float = 0.1,
    iterations: int = 3,
) -> np.ndarray:
    """Edge-preserving bilateral-like iterative infill.

    At each iteration the algorithm:

    1. Computes a Gaussian-blurred version of the current image (spatial
       kernel with standard deviation *sigma_s*).
    2. Computes a Gaussian-blurred weight map from the current occupancy
       (same spatial kernel).
    3. Divides the blurred image by the weight map to get a normalised
       estimate for empty pixels.
    4. For empty pixels whose normalised estimate is within *sigma_r* of a
       neighbour (range consistency), fills the pixel and marks it as occupied.

    This produces a diffusion that respects depth edges — large range
    discontinuities block the spread.

    Args:
        range_img: ``(H, W)`` float32 range image (0 = empty).
        mask: ``(H, W)`` boolean occupancy mask (True = occupied).
        sigma_s: Spatial standard deviation (pixels) for Gaussian kernel.
        sigma_r: Range tolerance — maximum normalised range difference
            allowed between the interpolated value and at least one occupied
            neighbour for the fill to be accepted.
        iterations: Number of diffusion iterations.

    Returns:
        ``(H, W)`` float32 filled range image.
    """
    out = range_img.copy()
    occupied = mask.copy()

    for _ in range(iterations):
        weight = occupied.astype(np.float32)
        # Gaussian blur of the value field and the weight field
        blurred_val = gaussian_filter(out * weight, sigma=sigma_s)
        blurred_w = gaussian_filter(weight, sigma=sigma_s)

        # Avoid division by zero — where weight is negligible, keep zero
        safe_w = np.where(blurred_w > 1e-8, blurred_w, 1.0)
        estimate = blurred_val / safe_w

        # Determine which empty pixels to fill
        empty = ~occupied
        if not empty.any():
            break

        # Range consistency check: compare estimate against the blurred
        # occupied-only field (a proxy for local occupied-pixel mean).
        # Accept if the absolute difference is below sigma_r.
        local_mean = blurred_val / safe_w
        diff = np.abs(estimate - local_mean)
        accept = empty & (blurred_w > 1e-8) & (diff <= sigma_r)

        out[accept] = estimate[accept]
        occupied[accept] = True

    return out


# ──────────────────────────────────────────────────────────────
# Morphological infill
# ──────────────────────────────────────────────────────────────


def morphological_infill(
    range_img: np.ndarray,
    mask: np.ndarray,
    kernel_size: int = 3,
    iterations: int = 2,
) -> np.ndarray:
    """Grey dilation infill for small gaps.

    Applies :func:`scipy.ndimage.grey_dilation` with a square structuring
    element of side *kernel_size* for *iterations* rounds.  After each
    round only previously-empty pixels are updated (occupied pixels keep
    their original values).

    This is fast and effective for 1-2 pixel gaps between scan lines, but
    tends to over-fill large holes.

    Args:
        range_img: ``(H, W)`` float32 range image (0 = empty).
        mask: ``(H, W)`` boolean occupancy mask (True = occupied).
        kernel_size: Side length of the square structuring element.
        iterations: Number of dilation rounds.

    Returns:
        ``(H, W)`` float32 filled range image.
    """
    out = range_img.copy()
    occupied = mask.copy()

    for _ in range(iterations):
        dilated = grey_dilation(out, size=(kernel_size, kernel_size))
        empty = ~occupied
        out[empty] = dilated[empty]
        # Mark newly filled pixels (where dilation produced a nonzero value)
        newly_filled = empty & (dilated > 0)
        occupied[newly_filled] = True

    return out


# ──────────────────────────────────────────────────────────────
# Adaptive infill
# ──────────────────────────────────────────────────────────────


def adaptive_infill(
    range_img: np.ndarray,
    mask: np.ndarray,
    occupancy_threshold: float = 0.3,
    small_gap_kernel: int = 3,
    large_gap_max_dist: int = 5,
) -> np.ndarray:
    """Adaptive infill that switches strategy based on local occupancy.

    The image is partitioned into non-overlapping blocks.  For each block
    the local occupancy ratio is computed.

    - **Dense blocks** (occupancy >= *occupancy_threshold*): morphological
      closing (grey dilation) with *small_gap_kernel* fills the small gaps
      between adjacent scan lines.
    - **Sparse blocks** (occupancy < *occupancy_threshold*): nearest-neighbour
      infill up to *large_gap_max_dist* pixels handles the larger holes
      without introducing artefacts from aggressive dilation.

    The block size is chosen automatically as ``(H // 4, W // 16)`` (clamped
    to a minimum of 4 pixels per side) to roughly match the spatial scale at
    which occupancy varies in typical outdoor LiDAR range images.

    Args:
        range_img: ``(H, W)`` float32 range image (0 = empty).
        mask: ``(H, W)`` boolean occupancy mask (True = occupied).
        occupancy_threshold: Fraction of occupied pixels above which a block
            is considered *dense*.
        small_gap_kernel: Kernel size for morphological closing in dense
            blocks.
        large_gap_max_dist: Maximum distance for nearest-neighbour infill in
            sparse blocks.

    Returns:
        ``(H, W)`` float32 filled range image.
    """
    H, W = range_img.shape
    out = range_img.copy()

    # Block dimensions
    bh = max(H // 4, 4)
    bw = max(W // 16, 4)

    # Pre-compute full-image results for both strategies
    morph_filled = morphological_infill(range_img, mask, kernel_size=small_gap_kernel, iterations=2)
    nn_filled = nearest_neighbor_infill(range_img, mask, max_dist=large_gap_max_dist)

    # Compute local occupancy via uniform filter on the mask
    occ_float = mask.astype(np.float32)

    for r0 in range(0, H, bh):
        r1 = min(r0 + bh, H)
        for c0 in range(0, W, bw):
            c1 = min(c0 + bw, W)

            block_mask = mask[r0:r1, c0:c1]
            block_occ = block_mask.sum() / max(block_mask.size, 1)
            empty_block = ~mask[r0:r1, c0:c1]

            if block_occ >= occupancy_threshold:
                # Dense: use morphological
                out[r0:r1, c0:c1][empty_block] = morph_filled[r0:r1, c0:c1][empty_block]
            else:
                # Sparse: use nearest neighbour
                out[r0:r1, c0:c1][empty_block] = nn_filled[r0:r1, c0:c1][empty_block]

    return out
