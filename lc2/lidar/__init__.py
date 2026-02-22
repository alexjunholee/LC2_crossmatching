"""LiDAR sensor configuration, spherical projection, and point cloud I/O.

Submodules
----------
- :mod:`lc2.lidar.sensors`     -- sensor configs, presets, registry
- :mod:`lc2.lidar.projection`  -- spherical range-image projection
- :mod:`lc2.lidar.io`          -- point cloud file loaders
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
]
