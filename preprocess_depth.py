"""Batch depth estimation preprocessing for LC2.

Pre-computes depth maps for camera images using a pluggable depth
estimation backend (DA3, DINOv3, DUSt3R).  Results are cached as .npy
files for efficient reuse during evaluation.

Usage::

    # VIVID — default DA3
    python preprocess_depth.py --dataset vivid --sequences campus_day1 \\
        --output_dir cache/depth/vivid

    # KITTI-360 — DA3-Small variant
    python preprocess_depth.py --dataset kitti360 --sequences 0000 0002 \\
        --output_dir cache/depth/kitti360 --estimator da3 --variant da3-small

    # DINOv3 DPT depth
    python preprocess_depth.py --dataset vivid --sequences campus_day1 \\
        --output_dir cache/depth/vivid_dinov3 --estimator dinov3

    # DUSt3R stereo depth
    python preprocess_depth.py --dataset vivid --sequences campus_day1 \\
        --output_dir cache/depth/vivid_dust3r --estimator dust3r
"""

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))


def get_vivid_images(root: str, sequence: str) -> List[Path]:
    """Glob camera images for a VIVID sequence."""
    img_dir = Path(root) / sequence / "img"
    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    return sorted(img_dir.glob("*.png"))


def get_kitti360_images(root: str, sequence: str) -> List[Path]:
    """Glob camera images for a KITTI-360 sequence."""
    dirname = f"2013_05_28_drive_{sequence}_sync"
    img_dir = Path(root) / "data_2d_raw" / dirname / "image_00" / "data_rgb"
    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    return sorted(img_dir.glob("*.png"))


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute depth maps for LC2 using a pluggable depth backend",
    )
    parser.add_argument("--dataset", type=str, required=True, choices=["vivid", "kitti360"])
    parser.add_argument("--sequences", nargs="+", required=True, help="Sequence names/IDs")
    parser.add_argument("--root", type=str, default=None, help="Dataset root (auto-detected if not set)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for .npy files")
    parser.add_argument(
        "--estimator", type=str, default="da3",
        choices=["da3", "dinov3", "dust3r"],
        help="Depth estimation backend (default: da3)",
    )
    parser.add_argument(
        "--variant", type=str, default=None,
        help=(
            "Model variant within the chosen estimator. "
            "DA3: da3-small, da3-large (default), da3-giant. "
            "DINOv3: dinov3_vits14, dinov3_vitb14, dinov3_vitl14 (default), dinov3_vitg14. "
            "DUSt3R: ignored (single variant)."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Inference batch size")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip_existing", action="store_true", help="Skip already-computed frames")
    args = parser.parse_args()

    # Auto-detect root
    if args.root is None:
        if args.dataset == "vivid":
            args.root = "/media/jhlee/EVO4TB/vivid_projects/data"
        elif args.dataset == "kitti360":
            args.root = "/media/jhlee/QVO4TB/data/kitti360/KITTI-360"

    # Initialize depth estimator via factory
    from lc2.depth import get_depth_estimator

    factory_kwargs = {}
    if args.variant is not None:
        factory_kwargs["variant"] = args.variant

    print(f"Loading depth estimator: {args.estimator}"
          + (f" (variant={args.variant})" if args.variant else ""))
    estimator = get_depth_estimator(args.estimator, device=args.device, **factory_kwargs)
    print("Model loaded.")

    for seq in args.sequences:
        print(f"\n{'=' * 60}")
        print(f"Processing sequence: {seq}")

        # Get image list
        if args.dataset == "vivid":
            images = get_vivid_images(args.root, seq)
        else:
            images = get_kitti360_images(args.root, seq)

        print(f"  Found {len(images)} images")

        # Output directory
        out_dir = Path(args.output_dir) / seq
        out_dir.mkdir(parents=True, exist_ok=True)

        # Filter existing
        if args.skip_existing:
            pending = []
            for img_path in images:
                out_path = out_dir / f"{img_path.stem}.npy"
                if not out_path.exists():
                    pending.append(img_path)
            print(f"  Skipping {len(images) - len(pending)} existing, {len(pending)} remaining")
            images = pending

        if len(images) == 0:
            print("  Nothing to process.")
            continue

        # Batch inference
        for i in tqdm(range(0, len(images), args.batch_size), desc=f"  {seq}"):
            batch_paths = images[i : i + args.batch_size]
            depths = estimator.estimate_batch(batch_paths, batch_size=args.batch_size)

            for img_path, depth in zip(batch_paths, depths):
                out_path = out_dir / f"{img_path.stem}.npy"
                np.save(str(out_path), depth)

    print(f"\nDone. Depth maps saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
