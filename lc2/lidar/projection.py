"""Spherical projection of LiDAR point clouds to range images.

Converts 3D point clouds into 2D range images via azimuth-elevation
spherical projection. Each 3D point ``(x, y, z)`` is mapped to spherical
coordinates ``(r, phi, theta)`` and then quantised to a pixel ``(u, v)``
in the output grid. A z-buffer keeps only the closest point per pixel.
"""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np

from lc2.lidar.sensors import SensorConfig

__all__ = ["lidar_to_range_image"]


def lidar_to_range_image(
    points: np.ndarray,
    config: SensorConfig,
    return_mask: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Project a 3D point cloud onto a 2D range image via spherical projection.

    The projection maps each 3D point (x, y, z) to spherical coordinates
    (range r, azimuth phi, elevation theta) and then to pixel coordinates (u, v)
    in the range image grid.

    For overlapping projections (multiple points mapping to the same pixel),
    the closest point (smallest range) is retained (z-buffer).

    Output is a **single-channel** ``(H, W)`` range image with normalized
    distance values, matching the format used by the original LC2 training
    pipeline (single channel replicated to 3 via ``transforms``).

    Args:
        points: Point cloud of shape ``(N, C)`` where C >= 3.
            Columns: [x, y, z, (intensity, ...)].
            x: forward, y: left, z: up (right-hand, vehicle frame).
        config: Sensor configuration defining FOV, resolution, and range limits.
        return_mask: If True, also return the valid-pixel boolean mask.

    Returns:
        Range image of shape ``(H, W)`` float32, normalized range (r / max_range)
        in [0, 1], unoccupied pixels = 0.
        If return_mask is True, returns tuple (range_image, mask) where mask
        is ``(H, W)`` boolean array of occupied pixels.
    """
    assert points.ndim == 2 and points.shape[1] >= 3, \
        f"Expected (N, C>=3), got {points.shape}"

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Compute range
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    # Filter invalid points
    valid = (r > 1e-6) & (r < config.max_range)
    x, y, z, r = x[valid], y[valid], z[valid], r[valid]

    if len(r) == 0:
        img = np.zeros((config.H, config.W), dtype=np.float32)
        if return_mask:
            return img, np.zeros((config.H, config.W), dtype=bool)
        return img

    # Spherical coordinates
    # Elevation theta in [fov_down, fov_up] (degrees)
    theta = np.degrees(np.arcsin(np.clip(z / r, -1.0, 1.0)))
    # Azimuth phi in [-pi, pi]
    phi = np.arctan2(y, x)

    # FOV parameters in degrees
    fov_total = config.fov_up - config.fov_down

    # Map to pixel coordinates
    # u: azimuth -> column [0, W-1]
    u = 0.5 * (1.0 - phi / np.pi) * config.W
    u = np.clip(u, 0, config.W - 1).astype(np.int32)

    # v: elevation -> row [0, H-1]
    # theta=fov_up -> v=0 (top), theta=fov_down -> v=H-1 (bottom)
    v = (1.0 - (theta - config.fov_down) / fov_total) * config.H
    v = np.clip(v, 0, config.H - 1).astype(np.int32)

    # Z-buffer: keep closest point per pixel
    range_img = np.full((config.H, config.W), config.max_range, dtype=np.float32)

    # Sort by range descending so closer points overwrite farther ones
    order = np.argsort(-r)
    range_img[v[order], u[order]] = r[order]

    # Occupancy mask and normalization
    occupied = range_img < config.max_range
    range_norm = range_img / config.max_range
    range_norm[~occupied] = 0.0

    if return_mask:
        return range_norm.astype(np.float32), occupied
    return range_norm.astype(np.float32)
