"""LiDAR sensor configurations for spherical range-image projection.

Defines :class:`SensorConfig` — a frozen dataclass holding the vertical FOV,
resolution, and maximum-range parameters required by the projection function —
together with factory presets for commonly used sensors and a lightweight
registry that lets users add custom configurations at runtime.

Preset sensors
--------------
======================  ====  ======  ======================
Name                    H     W       Notes
======================  ====  ======  ======================
VELODYNE_HDL64          64    2048    KITTI-360 (HDL-64E)
VELODYNE_VLP32C         32    1024    HeLiPR
VELODYNE_VLP16          16    1024    Most common Velodyne
OUSTER_OS1_64           64    1024    VIVID
OUSTER_OS2_128          128   1024    HeLiPR
LIVOX_AVIA              6     4000    Non-repetitive, approx.
HESAI_PANDAR64          64    1800    PandarQT / AT128
ROBOSENSE_RS128         128   1800    RS-Ruby / RS-Helios
======================  ====  ======  ======================

Registry API
------------
- ``get_sensor_config(name)``  -- look up by canonical name
- ``register_sensor(name, config)`` -- add / override at runtime
- ``list_sensors()`` -- list all registered names
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

__all__ = [
    "SensorConfig",
    # presets
    "VELODYNE_HDL64",
    "VELODYNE_VLP32C",
    "VELODYNE_VLP16",
    "OUSTER_OS1_64",
    "OUSTER_OS2_128",
    "LIVOX_AVIA",
    "HESAI_PANDAR64",
    "ROBOSENSE_RS128",
    # registry
    "get_sensor_config",
    "register_sensor",
    "list_sensors",
]


# ──────────────────────────────────────────────────────────────
# Sensor configuration dataclass
# ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SensorConfig:
    """LiDAR sensor configuration for spherical projection.

    Attributes:
        name: Human-readable sensor identifier.
        fov_up: Upper vertical FOV limit in degrees (positive = above horizon).
        fov_down: Lower vertical FOV limit in degrees (negative = below horizon).
        H: Number of rows (elevation bins) in the range image.
        W: Number of columns (azimuth bins) in the range image.
        max_range: Maximum range in meters for normalization and filtering.
    """

    name: str
    fov_up: float
    fov_down: float
    H: int
    W: int
    max_range: float

    @property
    def fov_total(self) -> float:
        """Total vertical field of view in degrees (``fov_up - fov_down``)."""
        return self.fov_up - self.fov_down


# ──────────────────────────────────────────────────────────────
# Factory presets
# ──────────────────────────────────────────────────────────────

# Velodyne HDL-64E: KITTI-360 sensor
# 64 channels, ~26.8deg vertical FOV (+2deg to -24.8deg), 0.09deg azimuth res.
VELODYNE_HDL64 = SensorConfig(
    name="Velodyne HDL-64E",
    fov_up=2.0,
    fov_down=-24.8,
    H=64,
    W=2048,
    max_range=120.0,
)

# Velodyne VLP-32C: HeLiPR sensor
# 32 channels, 40deg vertical FOV (+15deg to -25deg)
VELODYNE_VLP32C = SensorConfig(
    name="Velodyne VLP-32C",
    fov_up=15.0,
    fov_down=-25.0,
    H=32,
    W=1024,
    max_range=200.0,
)

# Velodyne VLP-16 (Puck)
# 16 channels, 30deg vertical FOV (+15deg to -15deg)
VELODYNE_VLP16 = SensorConfig(
    name="Velodyne VLP-16",
    fov_up=15.0,
    fov_down=-15.0,
    H=16,
    W=1024,
    max_range=100.0,
)

# Ouster OS1-64: VIVID sensor
# 64 channels, 45deg vertical FOV (+22.5deg to -22.5deg)
OUSTER_OS1_64 = SensorConfig(
    name="Ouster OS1-64",
    fov_up=22.5,
    fov_down=-22.5,
    H=64,
    W=1024,
    max_range=120.0,
)

# Ouster OS2-128: HeLiPR sensor
# 128 channels, 45deg vertical FOV (+22.5deg to -22.5deg)
OUSTER_OS2_128 = SensorConfig(
    name="Ouster OS2-128",
    fov_up=22.5,
    fov_down=-22.5,
    H=128,
    W=1024,
    max_range=200.0,
)

# Livox Avia — non-repetitive scanning pattern
# Approximated as 6 effective rows (triple-return lines) x 4000 azimuth bins
# 70.4deg horizontal FOV, ~77.2deg vertical FOV (-0.3deg to +76.9deg is full)
# Typical mapping config uses smaller FOV; values below are for outdoor VPR use.
LIVOX_AVIA = SensorConfig(
    name="Livox Avia",
    fov_up=38.6,
    fov_down=-38.6,
    H=6,
    W=4000,
    max_range=190.0,
)

# Hesai Pandar64 (PandarQT / AT128 family)
# 64 channels, ~40deg vertical FOV (+15deg to -25deg)
HESAI_PANDAR64 = SensorConfig(
    name="Hesai Pandar64",
    fov_up=15.0,
    fov_down=-25.0,
    H=64,
    W=1800,
    max_range=200.0,
)

# RoboSense RS-128 (Ruby / Helios family)
# 128 channels, 40deg vertical FOV (+15deg to -25deg)
ROBOSENSE_RS128 = SensorConfig(
    name="RoboSense RS-128",
    fov_up=15.0,
    fov_down=-25.0,
    H=128,
    W=1800,
    max_range=200.0,
)


# ──────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────

_SENSOR_REGISTRY: Dict[str, SensorConfig] = {}


def _populate_registry() -> None:
    """Seed the registry from module-level presets."""
    for cfg in [
        VELODYNE_HDL64,
        VELODYNE_VLP32C,
        VELODYNE_VLP16,
        OUSTER_OS1_64,
        OUSTER_OS2_128,
        LIVOX_AVIA,
        HESAI_PANDAR64,
        ROBOSENSE_RS128,
    ]:
        _SENSOR_REGISTRY[cfg.name] = cfg


_populate_registry()


def get_sensor_config(name: str) -> SensorConfig:
    """Look up a sensor configuration by name.

    Args:
        name: Canonical sensor name (e.g. ``"Velodyne HDL-64E"``).

    Returns:
        Matching :class:`SensorConfig`.

    Raises:
        KeyError: If *name* is not in the registry.
    """
    if name not in _SENSOR_REGISTRY:
        available = ", ".join(sorted(_SENSOR_REGISTRY))
        raise KeyError(
            f"Unknown sensor '{name}'. Registered sensors: {available}"
        )
    return _SENSOR_REGISTRY[name]


def register_sensor(name: str, config: SensorConfig) -> None:
    """Register (or override) a sensor configuration.

    Args:
        name: Key used for subsequent :func:`get_sensor_config` look-ups.
        config: The sensor configuration to register.
    """
    if not isinstance(config, SensorConfig):
        raise TypeError(f"Expected SensorConfig, got {type(config).__name__}")
    _SENSOR_REGISTRY[name] = config


def list_sensors() -> List[str]:
    """Return a sorted list of all registered sensor names."""
    return sorted(_SENSOR_REGISTRY)
