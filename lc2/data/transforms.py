"""Image transforms for LC2 depth and range image preprocessing.

Matches the original LC2 preprocessing pipeline exactly:
    1. ``ToTensor()`` — (H, W) float32 → (1, H, W) tensor
    2. ``repeat(3, 1, 1)`` — single channel → 3 identical channels
    3. ``Normalize([0.5]*3, [0.5]*3)`` — centers around 0 with range [-1, 1]

The original LC2 does NOT use ImageNet normalization; it uses a uniform
mean=0.5, std=0.5 for all channels since the inputs are single-channel
depth/range maps replicated to 3 channels.
"""

import numpy as np
import torch
from scipy.ndimage import uniform_filter
from torchvision import transforms
from typing import Dict, List, Optional, Tuple


# LC2 normalization: (x - 0.5) / 0.5  →  maps [0, 1] to [-1, 1]
LC2_MEAN = [0.5, 0.5, 0.5]
LC2_STD = [0.5, 0.5, 0.5]


def _repeat_channels(x: torch.Tensor) -> torch.Tensor:
    """Replicate single-channel tensor to 3 channels."""
    if x.shape[0] == 1:
        return x.repeat(3, 1, 1)
    return x


def get_transform(
    input_size: Optional[Tuple[int, int]] = None,
) -> transforms.Compose:
    """LC2 input transform pipeline.

    Converts a single-channel numpy array (H, W) to a (3, H', W') tensor
    normalized with mean=0.5, std=0.5 per channel.

    Args:
        input_size: If provided, resize to (H, W). If None, keep original
            resolution (matching the original LC2 code which does NOT resize).

    Returns:
        Composed transform: ToTensor → Repeat3ch → [Resize] → Normalize.
    """
    t = [
        transforms.ToTensor(),
        transforms.Lambda(_repeat_channels),
    ]
    if input_size is not None:
        t.append(transforms.Resize(input_size, antialias=True))
    t.append(transforms.Normalize(mean=LC2_MEAN, std=LC2_STD))
    return transforms.Compose(t)


def squeeze_depth(depth: np.ndarray) -> np.ndarray:
    """Squeeze ManyDepth/DA3 output to 2D and optionally crop.

    ManyDepth saves as (1, 1, H, W). DA3 saves as (H, W).
    Both are reduced to (H, W) float32.

    Args:
        depth: Depth array, possibly with batch/channel dims.

    Returns:
        2D array of shape ``(H, W)``, float32.
    """
    depth = np.asarray(depth, dtype=np.float32).squeeze()
    if depth.ndim != 2:
        raise ValueError(f"Expected 2D depth after squeeze, got shape {depth.shape}")
    return depth


def normalize_depth(depth: np.ndarray) -> np.ndarray:
    """Per-frame min-max normalization of depth to [0, 1].

    DA3 outputs metric-scale depth values (e.g., [0.2, 7.08]) which are
    incompatible with the LC2 normalization pipeline that assumes inputs in
    [0, 1]. The original ManyDepth used a sigmoid output, so values were
    inherently in [0, 1]. This function re-creates that [0, 1] range for DA3.

    Args:
        depth: 2D depth array of shape ``(H, W)``.

    Returns:
        Normalized depth in [0, 1], float32.
    """
    d_min = depth.min()
    d_max = depth.max()
    if d_max - d_min < 1e-8:
        return np.zeros_like(depth, dtype=np.float32)
    return ((depth - d_min) / (d_max - d_min)).astype(np.float32)


def depth_to_disparity(depth: np.ndarray, min_depth: float = 0.1) -> np.ndarray:
    """Convert depth map to disparity (inverse depth).

    Paper: "These depth images are fed into the network in a disparity form.
    Each depth value follows the inverse depth parametrization to make closer
    objects numerically dominant."

    Disparity = 1 / depth. Closer objects have higher values, which makes
    them numerically dominant in the feature representation.

    Args:
        depth: 2D metric depth array of shape ``(H, W)``, values > 0.
        min_depth: Minimum depth clamp to avoid division by zero.

    Returns:
        Disparity array in [0, 1/min_depth], float32.
    """
    depth_clamped = np.maximum(depth, min_depth)
    disparity = 1.0 / depth_clamped
    return disparity.astype(np.float32)


def normalize_disparity(disparity: np.ndarray) -> np.ndarray:
    """Per-frame min-max normalization of disparity to [0, 1].

    Args:
        disparity: 2D disparity array of shape ``(H, W)``.

    Returns:
        Normalized disparity in [0, 1], float32.
    """
    d_min = disparity.min()
    d_max = disparity.max()
    if d_max - d_min < 1e-8:
        return np.zeros_like(disparity, dtype=np.float32)
    return ((disparity - d_min) / (d_max - d_min)).astype(np.float32)


def depth_to_normalized_disparity(depth: np.ndarray, min_depth: float = 0.1) -> np.ndarray:
    """Convert depth to disparity and normalize to [0, 1] in one step.

    Convenience function combining depth_to_disparity + normalize_disparity.

    Args:
        depth: Metric depth array ``(H, W)``.
        min_depth: Minimum depth clamp.

    Returns:
        Normalized disparity in [0, 1], float32.
    """
    disp = depth_to_disparity(depth, min_depth=min_depth)
    return normalize_disparity(disp)


def range_to_normalized_disparity(
    range_img: np.ndarray, min_range: float = 0.01
) -> np.ndarray:
    """Convert range image to disparity and normalize to [0, 1].

    Following the LC2 paper, range images are converted to disparity form
    (1/range) so that closer objects are numerically dominant, matching
    the depth disparity representation.

    Empty pixels (value 0 = no LiDAR return) remain 0.

    Args:
        range_img: Range image ``(H, W)`` with values in [0, max_range].
            0 means no return.
        min_range: Threshold below which pixels are treated as empty.

    Returns:
        Normalized disparity in [0, 1], float32. Empty pixels stay 0.
    """
    valid = range_img > min_range
    if not valid.any():
        return np.zeros_like(range_img, dtype=np.float32)

    out = np.zeros_like(range_img, dtype=np.float32)
    out[valid] = 1.0 / range_img[valid]

    r_min = out[valid].min()
    r_max = out[valid].max()
    if r_max - r_min < 1e-8:
        out[valid] = 0.5
    else:
        out[valid] = (out[valid] - r_min) / (r_max - r_min)

    return out


def scale_augment_disparity(
    disparity: np.ndarray, max_scale_pct: float = 20.0
) -> np.ndarray:
    """Random scale augmentation for disparity images (Phase 1).

    Paper (Section III.C.2): "We multiplied a random scaling constant r to
    the estimated depth values from the monocular disparity image, to
    randomly scale ±r% of the depth image."

    Multiplies disparity by (1 + r) where r ~ Uniform(-max_scale_pct/100,
    max_scale_pct/100).

    Args:
        disparity: Disparity array ``(H, W)``.
        max_scale_pct: Maximum scale percentage (default 20 → ±20%).

    Returns:
        Scaled disparity, float32.
    """
    r = np.random.uniform(-max_scale_pct / 100.0, max_scale_pct / 100.0)
    return (disparity * (1.0 + r)).astype(np.float32)


# ---------------------------------------------------------------------------
# Composable depth augmentation pipeline
# ---------------------------------------------------------------------------


class RandomDepthScale:
    """Random global scale augmentation wrapping :func:`scale_augment_disparity`.

    Multiplies the depth map by ``(1 + r)`` where
    ``r ~ Uniform(-max_pct/100, max_pct/100)``.

    Args:
        max_pct: Maximum scale percentage (default 20 -> +/-20%).
    """

    def __init__(self, max_pct: float = 20.0) -> None:
        self.max_pct = max_pct

    def __call__(self, depth: np.ndarray) -> np.ndarray:
        return scale_augment_disparity(depth, max_scale_pct=self.max_pct)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(max_pct={self.max_pct})"


class RandomDepthNoise:
    """Additive Gaussian noise on depth values.

    Adds i.i.d. ``N(0, sigma^2)`` noise to every pixel and clips the result
    to ``[0, 1]``.

    Args:
        sigma: Standard deviation of the Gaussian noise.
    """

    def __init__(self, sigma: float = 0.02) -> None:
        self.sigma = sigma

    def __call__(self, depth: np.ndarray) -> np.ndarray:
        noise = np.random.normal(0.0, self.sigma, size=depth.shape).astype(
            np.float32
        )
        return np.clip(depth + noise, 0.0, 1.0).astype(np.float32)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(sigma={self.sigma})"


class RandomDepthHoles:
    """Simulate depth estimation failures by zeroing random rectangular patches.

    The number of holes is proportional to the image area:
    ``n_holes = int(p * H * W / max_hole_size^2)``.

    Each hole is a rectangle of random size up to
    ``max_hole_size x max_hole_size`` placed at a uniformly random location.

    Args:
        p: Probability-like scaling factor controlling hole density.
        max_hole_size: Maximum side length of each rectangular hole in pixels.
    """

    def __init__(self, p: float = 0.05, max_hole_size: int = 16) -> None:
        self.p = p
        self.max_hole_size = max_hole_size

    def __call__(self, depth: np.ndarray) -> np.ndarray:
        H, W = depth.shape
        n_holes = int(self.p * H * W / (self.max_hole_size ** 2))
        if n_holes == 0:
            return depth
        out = depth.copy()
        for _ in range(n_holes):
            h = np.random.randint(1, self.max_hole_size + 1)
            w = np.random.randint(1, self.max_hole_size + 1)
            y = np.random.randint(0, H)
            x = np.random.randint(0, W)
            out[y : min(y + h, H), x : min(x + w, W)] = 0.0
        return out

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(p={self.p}, "
            f"max_hole_size={self.max_hole_size})"
        )


class RandomEdgeBlur:
    """Blur depth edges to simulate depth estimator artifacts.

    Detects edges via the gradient magnitude of the depth map and applies a
    ``scipy.ndimage.uniform_filter`` only at edge pixels. Applied
    stochastically with probability ``p``.

    Args:
        kernel: Kernel size for the uniform filter (must be odd).
        p: Probability of applying the augmentation per call.
    """

    def __init__(self, kernel: int = 5, p: float = 0.5) -> None:
        self.kernel = kernel
        self.p = p

    def __call__(self, depth: np.ndarray) -> np.ndarray:
        if np.random.rand() > self.p:
            return depth

        # Gradient magnitude to detect edges
        gy = np.gradient(depth, axis=0)
        gx = np.gradient(depth, axis=1)
        grad_mag = np.sqrt(gx ** 2 + gy ** 2)

        # Threshold: pixels with gradient > mean + 1 std are "edges"
        threshold = grad_mag.mean() + grad_mag.std()
        edge_mask = grad_mag > threshold

        if not edge_mask.any():
            return depth

        blurred = uniform_filter(depth, size=self.kernel).astype(np.float32)
        out = depth.copy()
        out[edge_mask] = blurred[edge_mask]
        return out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(kernel={self.kernel}, p={self.p})"


class DepthAugmentation:
    """Composable depth augmentation pipeline.

    Applies a sequence of augmentation transforms to a depth image.

    Args:
        augmentations: List of augmentation callables. Each must accept
            and return an ``(H, W)`` float32 numpy array in ``[0, 1]``.

    Example::

        aug = DepthAugmentation([
            RandomDepthScale(20.0),
            RandomDepthNoise(0.02),
            RandomDepthHoles(0.05),
            RandomEdgeBlur(5),
        ])
        out = aug(depth)  # (H, W) float32
    """

    def __init__(self, augmentations: List[object]) -> None:
        self.augmentations = augmentations

    def __call__(self, depth: np.ndarray) -> np.ndarray:
        for aug in self.augmentations:
            depth = aug(depth)
        return depth

    @classmethod
    def from_config(cls, cfg: Dict[str, float]) -> "DepthAugmentation":
        """Build a pipeline from a flat config dict.

        Supported keys and their defaults:

        - ``scale_pct`` (float): max scale % -> :class:`RandomDepthScale`
        - ``noise`` (float): noise sigma -> :class:`RandomDepthNoise`
        - ``holes`` (float): hole density p -> :class:`RandomDepthHoles`
        - ``edge_blur`` (int|float): kernel size -> :class:`RandomEdgeBlur`

        Args:
            cfg: Configuration dictionary.

        Returns:
            Configured :class:`DepthAugmentation` instance.
        """
        augs: List[object] = []
        if "scale_pct" in cfg:
            augs.append(RandomDepthScale(max_pct=float(cfg["scale_pct"])))
        if "noise" in cfg:
            augs.append(RandomDepthNoise(sigma=float(cfg["noise"])))
        if "holes" in cfg:
            augs.append(RandomDepthHoles(p=float(cfg["holes"])))
        if "edge_blur" in cfg:
            augs.append(RandomEdgeBlur(kernel=int(cfg["edge_blur"])))
        return cls(augs)

    def __repr__(self) -> str:
        items = ", ".join(repr(a) for a in self.augmentations)
        return f"{self.__class__.__name__}([{items}])"


def crop_range_panoramic(
    range_img: np.ndarray,
    crop_idx: int,
    n_crops: int = 8,
    crop_fov_deg: float = 90.0,
) -> np.ndarray:
    """Crop a panoramic range image into one of ``n_crops`` overlapping views.

    Paper (Section III.C.1): "we divide the panoramic range image into eight
    FoV-masked overlapping images."

    The panoramic range image covers 360°. Each crop covers
    ``crop_fov_deg`` degrees (default 90°), and crops are evenly spaced
    by ``360 / n_crops`` degrees.

    Args:
        range_img: Panoramic range image ``(H, W)`` where W covers 360°.
        crop_idx: Which crop to extract, 0 to ``n_crops - 1``.
        n_crops: Number of crops (default 8, step = 45°).
        crop_fov_deg: Angular width of each crop in degrees.

    Returns:
        Cropped range image ``(H, crop_width)``.
    """
    W = range_img.shape[-1]
    crop_w = int(W * crop_fov_deg / 360.0)
    step = W // n_crops
    start = (crop_idx * step) % W

    # Handle wraparound
    if start + crop_w <= W:
        if range_img.ndim == 2:
            return range_img[:, start : start + crop_w]
        return range_img[..., start : start + crop_w]
    else:
        overflow = crop_w - (W - start)
        if range_img.ndim == 2:
            part1 = range_img[:, start:]
            part2 = range_img[:, :overflow]
        else:
            part1 = range_img[..., start:]
            part2 = range_img[..., :overflow]
        return np.concatenate([part1, part2], axis=-1)


def crop_range_to_camera_fov(
    range_img: np.ndarray,
    camera_hfov_deg: float = 90.0,
    forward_col_frac: float = 0.5,
) -> np.ndarray:
    """Crop panoramic range image to camera FoV for Phase 2.

    Paper (Section III.B.4): "the range images from a LiDAR are cropped
    by the size of the camera FoV to ensure that the descriptors converge."

    Args:
        range_img: Panoramic range image ``(H, W)`` covering 360°.
        camera_hfov_deg: Camera horizontal FoV in degrees.
        forward_col_frac: Fraction of W corresponding to the forward
            direction (default 0.5, i.e., center of the image).

    Returns:
        Cropped range image aligned with camera FoV.
    """
    W = range_img.shape[-1]
    crop_w = int(W * camera_hfov_deg / 360.0)
    center = int(W * forward_col_frac)

    start = center - crop_w // 2
    end = start + crop_w

    # Handle wraparound
    if start < 0:
        if range_img.ndim == 2:
            part1 = range_img[:, start % W :]
            part2 = range_img[:, : end]
        else:
            part1 = range_img[..., start % W :]
            part2 = range_img[..., : end]
        return np.concatenate([part1, part2], axis=-1)
    elif end > W:
        if range_img.ndim == 2:
            part1 = range_img[:, start:]
            part2 = range_img[:, : end % W]
        else:
            part1 = range_img[..., start:]
            part2 = range_img[..., : end % W]
        return np.concatenate([part1, part2], axis=-1)
    else:
        if range_img.ndim == 2:
            return range_img[:, start:end]
        return range_img[..., start:end]
