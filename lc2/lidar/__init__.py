"""LiDAR sensor configuration, spherical projection, point cloud I/O, infill, and augmentation.

Submodules
----------
- :mod:`lc2.lidar.sensors`        -- sensor configs, presets, registry
- :mod:`lc2.lidar.projection`     -- spherical range-image projection
- :mod:`lc2.lidar.io`             -- point cloud file loaders
- :mod:`lc2.lidar.infill`         -- sparse-region infill for range images
- :mod:`lc2.lidar.augmentation`   -- range-image augmentation transforms
"""

from lc2.lidar.sensors import (
    SensorConfig,
    # presets
    VELODYNE_HDL64,
    VELODYNE_VLP32C,
    VELODYNE_VLP16,
    OUSTER_OS1_64,
    OUSTER_OS2_128,
    LIVOX_AVIA,
    HESAI_PANDAR64,
    ROBOSENSE_RS128,
    # registry
    get_sensor_config,
    register_sensor,
    list_sensors,
)
from lc2.lidar.projection import lidar_to_range_image
from lc2.lidar.io import load_point_cloud
from lc2.lidar.infill import get_infill_fn, nearest_neighbor_infill, adaptive_infill
from lc2.lidar.augmentation import RangeAugmentation

__all__ = [
    # sensors
    "SensorConfig",
    "VELODYNE_HDL64",
    "VELODYNE_VLP32C",
    "VELODYNE_VLP16",
    "OUSTER_OS1_64",
    "OUSTER_OS2_128",
    "LIVOX_AVIA",
    "HESAI_PANDAR64",
    "ROBOSENSE_RS128",
    "get_sensor_config",
    "register_sensor",
    "list_sensors",
    # projection
    "lidar_to_range_image",
    # io
    "load_point_cloud",
    # infill
    "get_infill_fn",
    "nearest_neighbor_infill",
    "adaptive_infill",
]
