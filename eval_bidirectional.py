#!/usr/bin/env python3
"""Bidirectional cross-modal retrieval evaluation for LC2.

Evaluates both directions:
  - Forward:  range(LiDAR) query → depth(camera) DB   (original LC2 protocol)
  - Reverse:  depth(camera) query → range(LiDAR) DB   (LC2++ direction)

Usage:
    python eval_bidirectional.py \
        --config configs/train_kitti360_multi.yaml \
        --checkpoint checkpoints/kitti360_multi_v3/best.pth.tar
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from lc2.model import LC2Model
from lc2.data.kitti360 import KITTI360LC2Dataset
from lc2.data.vivid import VIVIDLC2Dataset
from lc2.data.helipr import HeLiPRLC2Dataset
from lc2.utils.retrieval import evaluate_retrieval


def load_config(path: str) -> Dict:
    with open(path) as f:
        return yaml.safe_load(f)


@torch.no_grad()
def extract_descriptors(model, dataset, device, batch_size=64, is_range=True):
    """Extract global descriptors for all samples in dataset."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    descs = []
    for batch in loader:
        imgs = batch["image"].to(device)
        d = model.forward_single(imgs, is_range=is_range)
        descs.append(d.cpu().numpy())
    return np.concatenate(descs, axis=0)


def build_datasets(cfg, seq):
    """Build range and depth datasets for a given sequence."""
    dataset_cfg = cfg["dataset"]
    input_cfg = cfg.get("input", {})
    val_cfg = cfg.get("val", cfg.get("eval", {}))
    dataset_name = dataset_cfg.get("name", "vivid")

    resize_cfg = input_cfg.get("resize", None)
    input_size = tuple(resize_cfg) if resize_cfg else None
    camera_hfov_deg = input_cfg.get("camera_hfov_deg", None)

    if dataset_name == "kitti360":
        ds_range = KITTI360LC2Dataset(
            root=dataset_cfg["root"], sequences=[seq],
            modality="range",
            range_cache_dir=dataset_cfg.get("range_cache_dir"),
            subsample=10, input_size=input_size,
            camera_hfov_deg=camera_hfov_deg,
        )
        ds_depth = KITTI360LC2Dataset(
            root=dataset_cfg["root"], sequences=[seq],
            modality="depth",
            depth_cache_dir=dataset_cfg.get("depth_cache_dir"),
            subsample=10, input_size=input_size,
        )
    elif dataset_name == "helipr":
        ds_range = HeLiPRLC2Dataset(
            root=dataset_cfg["root"], sequences=[seq],
            modality="ouster",
            ouster_cache_dir=dataset_cfg.get("ouster_cache_dir"),
            subsample=10, input_size=input_size,
        )
        ds_depth = HeLiPRLC2Dataset(
            root=dataset_cfg["root"], sequences=[seq],
            modality="velodyne",
            velodyne_cache_dir=dataset_cfg.get("velodyne_cache_dir"),
            subsample=10, input_size=input_size,
        )
    else:
        forward_col_frac = input_cfg.get("forward_col_frac", 0.5)
        val_subsample = val_cfg.get("subsample", 10)
        ds_range = VIVIDLC2Dataset(
            root=dataset_cfg["root"], sequence=seq,
            modality="range", subsample=val_subsample,
            range_cache_dir=dataset_cfg.get("range_cache_dir"),
            input_size=input_size,
            camera_hfov_deg=camera_hfov_deg,
            forward_col_frac=forward_col_frac,
        )
        ds_depth = VIVIDLC2Dataset(
            root=dataset_cfg["root"], sequence=seq,
            modality="depth",
            depth_cache_dir=dataset_cfg.get("depth_cache_dir"),
            subsample=val_subsample, input_size=input_size,
        )

    return ds_range, ds_depth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sequences", nargs="*", default=None,
                        help="Override eval sequences (default: use config val sequences)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--pos_threshold", type=float, default=None)
    parser.add_argument("--gem", action="store_true", help="Use GeM pooling instead of NetVLAD")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_cfg = cfg["dataset"]
    dataset_name = dataset_cfg.get("name", "vivid")

    # Load model
    num_clusters = cfg.get("model", {}).get("num_clusters", 16)
    encoder_dim = cfg.get("model", {}).get("encoder_dim", 512)
    model = LC2Model(num_clusters=num_clusters, encoder_dim=encoder_dim, pooling="netvlad")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    # Strip DataParallel module. prefix if present
    state = {k.replace('.module.', '.'): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)

    # Detect pooling mode from checkpoint or args
    if args.gem:
        model.set_pooling("gem")
    elif isinstance(ckpt, dict) and ckpt.get("phase") in (1, "1", "1+vlad_init"):
        model.set_pooling("gem")
        print("  Auto-detected Phase 1 checkpoint → GeM pooling")
    elif "gem" in str(ckpt.get("config", {}).get("train", {})):
        model.set_pooling("gem")
    print(f"  Pooling mode: {model.pooling_mode}")

    model = model.to(device).eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Sequences to evaluate
    val_cfg = cfg.get("val", cfg.get("eval", {}))
    sequences = args.sequences or val_cfg.get("sequences", [])
    pos_threshold = args.pos_threshold or dataset_cfg.get("pos_threshold", 25.0)
    ks = [1, 5, 10, 20]

    # HeLiPR: db uses range encoder too
    db_is_range = (dataset_name == "helipr")

    for seq in sequences:
        print(f"\n{'='*65}")
        print(f"  Sequence: {seq}  (pos_thr={pos_threshold}m)")
        print(f"{'='*65}")

        try:
            ds_range, ds_depth = build_datasets(cfg, seq)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        range_pos = ds_range.get_positions()
        depth_pos = ds_depth.get_positions()
        print(f"  Range: {len(ds_range)} samples, Depth: {len(ds_depth)} samples")

        # Extract descriptors
        print("  Extracting range descriptors...")
        range_desc = extract_descriptors(model, ds_range, device, args.batch_size, is_range=True)
        print("  Extracting depth descriptors...")
        depth_desc = extract_descriptors(model, ds_depth, device, args.batch_size, is_range=db_is_range)

        # Forward: range(Q) → depth(DB)  [original LC2]
        print(f"\n  [Forward] range(Q) → depth(DB)")
        fwd = evaluate_retrieval(range_desc, depth_desc, range_pos, depth_pos,
                                 pos_threshold=pos_threshold, ks=ks, top_k=max(ks))
        for k in ks:
            print(f"    R@{k}@{pos_threshold:.0f}m = {fwd[f'recall@{k}']*100:.1f}%")

        # Reverse: depth(Q) → range(DB)  [LC2++ direction]
        print(f"\n  [Reverse] depth(Q) → range(DB)")
        rev = evaluate_retrieval(depth_desc, range_desc, depth_pos, range_pos,
                                 pos_threshold=pos_threshold, ks=ks, top_k=max(ks))
        for k in ks:
            print(f"    R@{k}@{pos_threshold:.0f}m = {rev[f'recall@{k}']*100:.1f}%")

        # Delta
        print(f"\n  Delta (Rev - Fwd):")
        for k in ks:
            delta = (rev[f"recall@{k}"] - fwd[f"recall@{k}"]) * 100
            print(f"    R@{k}: {delta:+.1f}%")


if __name__ == "__main__":
    main()
