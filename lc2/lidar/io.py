"""Point cloud I/O utilities.

Loaders for common LiDAR point cloud file formats used in autonomous-driving
and robotics datasets. Each loader returns an ``(N, 4)`` float32 array with
columns ``[x, y, z, intensity]``.

Supported formats
-----------------
==========  ================  =====================================
Extension   Loader            Notes
==========  ================  =====================================
``.bin``    auto-detect       KITTI-360 (float32x4) or HeLiPR
``.pcd``    :func:`load_pcd`  Binary PCD v0.7 (VIVID, general)
``.ply``    :func:`load_ply`  Binary/ASCII PLY (Open3D, CloudCmp)
==========  ================  =====================================

The universal entry point :func:`load_point_cloud` dispatches on file
extension by default, or on an explicit ``format`` string.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Optional

import numpy as np

__all__ = [
    # dtypes
    "DTYPE_HELIPR_OUSTER",
    "DTYPE_HELIPR_VELODYNE",
    # loaders
    "load_kitti360_bin",
    "load_pcd",
    "load_helipr_ouster_bin",
    "load_helipr_velodyne_bin",
    "load_ply",
    "load_point_cloud",
    # backward-compat alias
    "load_vivid_pcd",
]

# ──────────────────────────────────────────────────────────────
# HeLiPR binary dtypes
# ──────────────────────────────────────────────────────────────

DTYPE_HELIPR_OUSTER = np.dtype([
    ("x", "f4"), ("y", "f4"), ("z", "f4"), ("intensity", "f4"),
    ("timestamp", "u4"),
    ("reflectivity", "u2"), ("ring", "u2"), ("ambient", "u2"),
])  # 26 bytes per point

DTYPE_HELIPR_VELODYNE = np.dtype([
    ("x", "f4"), ("y", "f4"), ("z", "f4"), ("intensity", "f4"),
    ("ring", "u2"),
    ("time", "f4"),
])  # 22 bytes per point


# ──────────────────────────────────────────────────────────────
# KITTI-360
# ──────────────────────────────────────────────────────────────


def load_kitti360_bin(path: str) -> np.ndarray:
    """Load a KITTI-360 Velodyne point cloud from a binary .bin file.

    The binary format stores N points as a flat array of float32 values:
    [x0, y0, z0, reflectance0, x1, y1, z1, reflectance1, ...].

    Args:
        path: Path to the ``.bin`` file.

    Returns:
        Point cloud of shape ``(N, 4)`` with columns [x, y, z, reflectance].
    """
    points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    return points


# ──────────────────────────────────────────────────────────────
# PCD (binary v0.7) — VIVID and general
# ──────────────────────────────────────────────────────────────


def load_pcd(path: str) -> np.ndarray:
    """Load a point cloud from a binary PCD v0.7 file.

    PCD files use a text header followed by packed binary point data.
    The header specifies fields, types, and counts. We extract x, y, z,
    and intensity (if available).

    Args:
        path: Path to the ``.pcd`` file.

    Returns:
        Point cloud of shape ``(N, 4)`` with columns [x, y, z, intensity].
        If intensity is not available, the 4th column is zeros.
    """
    path = Path(path)
    with open(path, "rb") as f:
        # Parse header
        fields = []
        sizes = []
        types = []
        counts = []
        num_points = 0
        data_format = "ascii"
        width = 0
        height = 0

        while True:
            line = f.readline().decode("ascii", errors="ignore").strip()
            if line.startswith("FIELDS"):
                fields = line.split()[1:]
            elif line.startswith("SIZE"):
                sizes = [int(s) for s in line.split()[1:]]
            elif line.startswith("TYPE"):
                types = line.split()[1:]
            elif line.startswith("COUNT"):
                counts = [int(c) for c in line.split()[1:]]
            elif line.startswith("WIDTH"):
                width = int(line.split()[1])
            elif line.startswith("HEIGHT"):
                height = int(line.split()[1])
            elif line.startswith("POINTS"):
                num_points = int(line.split()[1])
            elif line.startswith("DATA"):
                data_format = line.split()[1]
                break

        if num_points == 0:
            num_points = width * height

        if data_format != "binary":
            raise NotImplementedError(f"Only binary PCD supported, got: {data_format}")

        # Build dtype from header
        dtype_list = []
        for field, size, typ, count in zip(fields, sizes, types, counts):
            if typ == "F":
                np_type = np.float32 if size == 4 else np.float64
            elif typ == "U":
                np_type = np.uint8 if size == 1 else (np.uint16 if size == 2 else np.uint32)
            elif typ == "I":
                np_type = np.int8 if size == 1 else (np.int16 if size == 2 else np.int32)
            else:
                np_type = np.float32

            for i in range(count):
                name = field if count == 1 else f"{field}_{i}"
                # Handle duplicate field names (PCD may have unnamed padding fields)
                if name == "_":
                    name = f"_pad_{len(dtype_list)}"
                dtype_list.append((name, np_type))

        dtype = np.dtype(dtype_list)
        raw_data = np.frombuffer(f.read(num_points * dtype.itemsize), dtype=dtype, count=num_points)

    # Extract x, y, z
    x = raw_data["x"].astype(np.float32)
    y = raw_data["y"].astype(np.float32)
    z = raw_data["z"].astype(np.float32)

    # Extract intensity (field name may vary)
    intensity = np.zeros(num_points, dtype=np.float32)
    for iname in ("intensity", "i", "reflectance", "reflectivity"):
        if iname in raw_data.dtype.names:
            intensity = raw_data[iname].astype(np.float32)
            break

    # Filter NaN / Inf
    points = np.column_stack([x, y, z, intensity])
    valid = np.isfinite(points).all(axis=1)
    points = points[valid]

    return points


# Backward-compatible alias
load_vivid_pcd = load_pcd
"""Alias for :func:`load_pcd` (legacy name from VIVID dataset code)."""


# ──────────────────────────────────────────────────────────────
# HeLiPR
# ──────────────────────────────────────────────────────────────


def load_helipr_ouster_bin(path: str) -> np.ndarray:
    """Load a HeLiPR Ouster OS2-128 point cloud from binary .bin file.

    Args:
        path: Path to the ``.bin`` file (26 bytes per point, Ouster layout).

    Returns:
        Point cloud of shape ``(N, 4)`` with columns [x, y, z, intensity].
    """
    raw = np.fromfile(str(path), dtype=DTYPE_HELIPR_OUSTER)
    return np.column_stack([raw["x"], raw["y"], raw["z"], raw["intensity"]])


def load_helipr_velodyne_bin(path: str) -> np.ndarray:
    """Load a HeLiPR Velodyne VLP-32C point cloud from binary .bin file.

    Args:
        path: Path to the ``.bin`` file (22 bytes per point, Velodyne layout).

    Returns:
        Point cloud of shape ``(N, 4)`` with columns [x, y, z, intensity].
    """
    raw = np.fromfile(str(path), dtype=DTYPE_HELIPR_VELODYNE)
    return np.column_stack([raw["x"], raw["y"], raw["z"], raw["intensity"]])


# ──────────────────────────────────────────────────────────────
# PLY
# ──────────────────────────────────────────────────────────────

_PLY_TYPE_MAP = {
    "char": "i1", "uchar": "u1",
    "short": "i2", "ushort": "u2",
    "int": "i4", "uint": "u4",
    "int8": "i1", "uint8": "u1",
    "int16": "i2", "uint16": "u2",
    "int32": "i4", "uint32": "u4",
    "float": "f4", "float32": "f4",
    "double": "f8", "float64": "f8",
}


def load_ply(path: str) -> np.ndarray:
    """Load a point cloud from a PLY file (binary_little_endian or ASCII).

    Reads vertex properties ``x, y, z`` (required) and ``intensity``
    (optional; falls back to ``scalar_intensity``, ``reflectance``,
    ``scalar_Intensity``, or zeros).

    Args:
        path: Path to the ``.ply`` file.

    Returns:
        Point cloud of shape ``(N, 4)`` with columns [x, y, z, intensity].

    Raises:
        ValueError: If the PLY header is malformed or the file is
            ``binary_big_endian`` (not supported).
    """
    path = Path(path)
    with open(path, "rb") as f:
        # --- parse header ---
        header_lines: list[str] = []
        while True:
            line = f.readline().decode("ascii", errors="ignore").strip()
            header_lines.append(line)
            if line == "end_header":
                break

        data_format = "ascii"
        num_vertices = 0
        vertex_props: list[tuple[str, str]] = []  # (name, ply_type)
        in_vertex_element = False

        for line in header_lines:
            tokens = line.split()
            if not tokens:
                continue
            if tokens[0] == "format":
                data_format = tokens[1]
            elif tokens[0] == "element":
                in_vertex_element = tokens[1] == "vertex"
                if in_vertex_element:
                    num_vertices = int(tokens[2])
            elif tokens[0] == "property" and in_vertex_element:
                # "property <type> <name>"  — skip list properties
                if tokens[1] == "list":
                    continue
                vertex_props.append((tokens[2], tokens[1]))

        if data_format == "binary_big_endian":
            raise ValueError("binary_big_endian PLY is not supported")

        # Build numpy dtype for vertex
        dtype_list = []
        for prop_name, ply_type in vertex_props:
            np_type = _PLY_TYPE_MAP.get(ply_type)
            if np_type is None:
                raise ValueError(f"Unknown PLY type '{ply_type}' for property '{prop_name}'")
            dtype_list.append((prop_name, np_type))
        vertex_dtype = np.dtype(dtype_list)

        # --- read data ---
        if data_format == "binary_little_endian":
            raw = np.frombuffer(f.read(num_vertices * vertex_dtype.itemsize),
                                dtype=vertex_dtype, count=num_vertices)
        elif data_format == "ascii":
            # Read all remaining lines for vertices
            raw_lines = []
            for _ in range(num_vertices):
                raw_lines.append(f.readline().decode("ascii", errors="ignore").strip())
            # Parse to structured array via intermediate float array
            arr = np.loadtxt(raw_lines, dtype=np.float64)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            raw = np.empty(num_vertices, dtype=vertex_dtype)
            for idx, (prop_name, _) in enumerate(vertex_props):
                raw[prop_name] = arr[:, idx]
        else:
            raise ValueError(f"Unsupported PLY format: {data_format}")

    x = raw["x"].astype(np.float32)
    y = raw["y"].astype(np.float32)
    z = raw["z"].astype(np.float32)

    intensity = np.zeros(num_vertices, dtype=np.float32)
    for iname in ("intensity", "scalar_intensity", "scalar_Intensity",
                   "reflectance", "reflectivity"):
        if iname in raw.dtype.names:
            intensity = raw[iname].astype(np.float32)
            break

    points = np.column_stack([x, y, z, intensity])
    valid = np.isfinite(points).all(axis=1)
    return points[valid]


# ──────────────────────────────────────────────────────────────
# Universal dispatcher
# ──────────────────────────────────────────────────────────────

_FORMAT_LOADERS = {
    "kitti360": load_kitti360_bin,
    "pcd": load_pcd,
    "ply": load_ply,
    "helipr_ouster": load_helipr_ouster_bin,
    "helipr_velodyne": load_helipr_velodyne_bin,
}


def load_point_cloud(
    path: str,
    format: str = "auto",
) -> np.ndarray:
    """Load a point cloud file, auto-detecting the format from extension.

    Supported extensions: ``.bin`` (KITTI-360 float32x4), ``.pcd`` (binary PCD),
    ``.ply`` (binary/ASCII PLY). For ``.bin`` files the default assumption is
    KITTI-360 layout (float32 x 4 per point). Use ``format="helipr_ouster"``
    or ``format="helipr_velodyne"`` explicitly for HeLiPR binary files.

    Args:
        path: Path to the point cloud file.
        format: One of ``"auto"``, ``"kitti360"``, ``"pcd"``, ``"ply"``,
            ``"helipr_ouster"``, ``"helipr_velodyne"``.
            When ``"auto"``, the format is inferred from the file extension.

    Returns:
        Point cloud of shape ``(N, 4)`` float32 — columns [x, y, z, intensity].

    Raises:
        ValueError: If the format is unknown or cannot be inferred.
    """
    path_obj = Path(path)

    if format != "auto":
        loader = _FORMAT_LOADERS.get(format)
        if loader is None:
            available = ", ".join(sorted(_FORMAT_LOADERS))
            raise ValueError(
                f"Unknown format '{format}'. Available: {available}"
            )
        return loader(str(path))

    # Auto-detect from extension
    ext = path_obj.suffix.lower()
    if ext == ".bin":
        return load_kitti360_bin(str(path))
    elif ext == ".pcd":
        return load_pcd(str(path))
    elif ext == ".ply":
        return load_ply(str(path))
    else:
        raise ValueError(
            f"Cannot infer point cloud format from extension '{ext}'. "
            f"Supported: .bin (KITTI-360), .pcd, .ply. "
            f"For HeLiPR .bin files, pass format='helipr_ouster' or 'helipr_velodyne'."
        )
