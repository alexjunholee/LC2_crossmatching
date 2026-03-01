"""Batch range image generation for LC2.

Pre-computes range images from LiDAR point clouds via spherical projection.
Results are cached as .npy files for efficient reuse during evaluation.

Usage::

    python preprocess_range.py --dataset kitti360 --sequences 0000 0002 \\
        --output_dir cache/range/kitti360

    python preprocess_range.py --dataset helipr --sequences DCC04 \\
        --sensor "Ouster OS2-128" --output_dir cache/range/helipr/Ouster

    python preprocess_range.py --dataset helipr --sequences DCC04 \\
        --sensor "Velodyne VLP-32C" --output_dir cache/range/helipr/Velodyne

    # With infill:
    python preprocess_range.py --dataset kitti360 --sequences 0000 \\
        --infill nearest_neighbor --output_dir cache/range/kitti360_filled
"""

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from lc2.lidar import get_sensor_config, lidar_to_range_image, load_point_cloud, list_sensors
from lc2.lidar.infill import get_infill_fn, list_infill_methods
from lc2.lidar.io import (
    load_kitti360_bin,
    load_pcd as load_vivid_pcd,
    load_helipr_ouster_bin,
    load_helipr_velodyne_bin,
)


# ──────────────────────────────────────────────────────────────
# Dataset-specific scan discovery
# ──────────────────────────────────────────────────────────────

def get_kitti360_scans(root: str, sequence: str) -> List[Path]:
    """Glob LiDAR scans for a KITTI-360 sequence."""
    dirname = f"2013_05_28_drive_{sequence}_sync"
    scan_dir = Path(root) / "data_3d_raw" / dirname / "velodyne_points" / "data"
    if not scan_dir.exists():
        raise FileNotFoundError(f"Velodyne directory not found: {scan_dir}")
    return sorted(scan_dir.glob("*.bin"))


def get_vivid_scans(root: str, sequence: str) -> List[Path]:
    """Glob LiDAR scans for a VIVID sequence."""
    pcd_dir = Path(root) / sequence / "pcd"
    if not pcd_dir.exists():
        raise FileNotFoundError(f"PCD directory not found: {pcd_dir}")
    return sorted(pcd_dir.glob("*.pcd"))


def get_helipr_scans(root: str, sequence: str, sensor: str) -> List[Path]:
    """Glob LiDAR scans for a HeLiPR sequence and sensor."""
    scan_dir = Path(root) / sequence / "LiDAR" / sensor
    if not scan_dir.exists():
        raise FileNotFoundError(f"HeLiPR LiDAR directory not found: {scan_dir}")
    return sorted(scan_dir.glob("*.bin"))


# ──────────────────────────────────────────────────────────────
# Default dataset -> sensor name mapping
# ──────────────────────────────────────────────────────────────

_DATASET_SENSOR_DEFAULTS = {
    "kitti360": "Velodyne HDL-64E",
    "vivid": "Ouster OS1-64",
    # HeLiPR requires explicit --sensor (Ouster OS2-128 or Velodyne VLP-32C)
}

# Map sensor registry names to per-dataset load functions.
# When load_point_cloud cannot auto-detect (e.g. HeLiPR .bin files),
# we dispatch through this table.
_SENSOR_LOAD_FN = {
    "Velodyne HDL-64E": load_kitti360_bin,
    "Ouster OS1-64": load_vivid_pcd,
    "Ouster OS2-128": load_helipr_ouster_bin,
    "Velodyne VLP-32C": load_helipr_velodyne_bin,
}

# HeLiPR subdirectory names per sensor
_HELIPR_SENSOR_SUBDIR = {
    "Ouster OS2-128": "Ouster",
    "Velodyne VLP-32C": "Velodyne",
}


def main():
    available_sensors = list_sensors()
    available_infill = list_infill_methods()

    parser = argparse.ArgumentParser(
        description="Pre-compute range images from LiDAR point clouds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            f"Available sensors:\n  {', '.join(available_sensors)}\n\n"
            f"Available infill methods:\n  {', '.join(available_infill)}"
        ),
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        choices=["vivid", "kitti360", "helipr"],
    )
    parser.add_argument(
        "--sequences", nargs="+", required=True,
        help="Sequence names/IDs",
    )
    parser.add_argument(
        "--root", type=str, default=None,
        help="Dataset root (auto-detected if not set)",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for .npy files",
    )
    parser.add_argument(
        "--sensor", type=str, default=None,
        help=(
            "Sensor configuration name from the registry "
            "(e.g. 'Velodyne HDL-64E', 'Ouster OS2-128'). "
            "Auto-detected per dataset if not set. "
            "Required for helipr."
        ),
    )
    parser.add_argument(
        "--infill", type=str, default="none",
        choices=["none"] + [m for m in available_infill if m != "none"],
        help="Range image infill strategy (default: none)",
    )
    parser.add_argument(
        "--skip_existing", action="store_true",
        help="Skip already-computed frames",
    )
    args = parser.parse_args()

    # ── Auto-detect root ──────────────────────────────────────
    if args.root is None:
        if args.dataset == "vivid":
            args.root = "/media/jhlee/EVO4TB/vivid_projects/data"
        elif args.dataset == "kitti360":
            args.root = "/media/jhlee/QVO4TB/data/kitti360/KITTI-360"
        elif args.dataset == "helipr":
            args.root = "/media/jhlee/EVO4TB/HeLiPR"

    # ── Resolve sensor config ─────────────────────────────────
    if args.sensor is None:
        if args.dataset in _DATASET_SENSOR_DEFAULTS:
            args.sensor = _DATASET_SENSOR_DEFAULTS[args.dataset]
        else:
            parser.error(
                f"--sensor is required for dataset '{args.dataset}'. "
                f"Available: {', '.join(available_sensors)}"
            )

    # Validate sensor name against registry
    config = get_sensor_config(args.sensor)

    # ── Select loader and scan discovery ──────────────────────
    load_fn = _SENSOR_LOAD_FN.get(args.sensor)
    if load_fn is None:
        # Fallback: use load_point_cloud (auto-detect by extension)
        load_fn = load_point_cloud

    if args.dataset == "kitti360":
        get_scans_fn = lambda root, seq: get_kitti360_scans(root, seq)
    elif args.dataset == "vivid":
        get_scans_fn = lambda root, seq: get_vivid_scans(root, seq)
    elif args.dataset == "helipr":
        subdir = _HELIPR_SENSOR_SUBDIR.get(args.sensor)
        if subdir is None:
            parser.error(
                f"Sensor '{args.sensor}' has no HeLiPR subdirectory mapping. "
                f"Supported HeLiPR sensors: {', '.join(_HELIPR_SENSOR_SUBDIR)}"
            )
        get_scans_fn = lambda root, seq, _s=subdir: get_helipr_scans(root, seq, _s)

    # ── Resolve infill ────────────────────────────────────────
    infill_fn = None
    if args.infill != "none":
        infill_fn = get_infill_fn(args.infill)

    # ── Print configuration ───────────────────────────────────
    print(f"Sensor: {config.name}")
    print(f"Range image: {config.H} x {config.W}, FOV [{config.fov_down}, {config.fov_up}] deg")
    print(f"Infill: {args.infill}")

    # ── Process sequences ─────────────────────────────────────
    for seq in args.sequences:
        print(f"\n{'=' * 60}")
        print(f"Processing sequence: {seq}")

        # Get scan list
        scans = get_scans_fn(args.root, seq)
        print(f"  Found {len(scans)} scans")

        # Output directory
        out_dir = Path(args.output_dir) / seq
        out_dir.mkdir(parents=True, exist_ok=True)

        # Filter existing
        if args.skip_existing:
            pending = []
            for scan_path in scans:
                out_path = out_dir / f"{scan_path.stem}.npy"
                if not out_path.exists():
                    pending.append(scan_path)
            print(f"  Skipping {len(scans) - len(pending)} existing, {len(pending)} remaining")
            scans = pending

        if len(scans) == 0:
            print("  Nothing to process.")
            continue

        for scan_path in tqdm(scans, desc=f"  {seq}"):
            points = load_fn(str(scan_path))

            if infill_fn is not None:
                range_img, mask = lidar_to_range_image(points, config, return_mask=True)
                range_img = infill_fn(range_img, mask)
            else:
                range_img = lidar_to_range_image(points, config)

            out_path = out_dir / f"{scan_path.stem}.npy"
            np.save(str(out_path), range_img)

    print(f"\nDone. Range images saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
