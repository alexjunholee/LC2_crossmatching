"""Backward-compatible shim -- all symbols re-exported from :mod:`lc2.lidar`.

.. deprecated::
    Import directly from :mod:`lc2.lidar` (or its submodules) instead.
    This file is kept only so existing ``from lc2.range_projection import ...``
    statements continue to work without modification.
"""

# Sensor configs & presets
from lc2.lidar.sensors import (  # noqa: F401
    SensorConfig,
    VELODYNE_HDL64,
    VELODYNE_VLP32C,
    OUSTER_OS1_64,
    OUSTER_OS2_128,
)

# Projection
from lc2.lidar.projection import lidar_to_range_image  # noqa: F401

# Point cloud I/O
from lc2.lidar.io import (  # noqa: F401
    load_kitti360_bin,
    load_pcd as load_vivid_pcd,
    load_helipr_ouster_bin,
    load_helipr_velodyne_bin,
    DTYPE_HELIPR_OUSTER,
    DTYPE_HELIPR_VELODYNE,
)
