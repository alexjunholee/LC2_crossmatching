"""KITTI-360 evaluation for trained LC2 models.

Within-sequence cross-modal: query=range, DB=depth from same sequence.
Supports both original LC2 checkpoints and our training checkpoints.

Usage::
    python eval_kitti360.py --checkpoint checkpoints/kitti360_v9/best.pth.tar
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from lc2.model import LC2Model
from lc2.data.kitti360 import KITTI360LC2Dataset
from lc2.utils.retrieval import (
    build_index, retrieve, compute_gt_positives, compute_recall,
)


def load_model(checkpoint_path, device, num_clusters=16, encoder_dim=512):
    """Load model from either training checkpoint or original LC2 format."""
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "phase" in ckpt:
        print(f"  Training checkpoint: phase={ckpt['phase']}, "
              f"epoch={ckpt['epoch']}, best={ckpt.get('best_score', 0):.4f}")
        model = LC2Model(
            num_clusters=num_clusters, encoder_dim=encoder_dim,
            vladv2=False, pooling="gem",
        )
        model.load_state_dict(ckpt["state_dict"], strict=False)
    else:
        model = LC2Model.from_checkpoint(checkpoint_path, device=device, pooling="gem")

    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def extract_descriptors(model, dataset, is_range, device, batch_size=64):
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )
    all_desc = []
    for batch in tqdm(loader, desc=f"{'Range' if is_range else 'Depth'}"):
        images = batch["image"].to(device)
        desc = model.forward_single(images, is_range=is_range)
        all_desc.append(desc.cpu().numpy())
    return np.concatenate(all_desc, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--root", type=str,
                        default="/media/jhlee/QVO4TB/data/kitti360/KITTI-360")
    parser.add_argument("--sequences", type=str, nargs="+", default=["0010"])
    parser.add_argument("--depth_cache_dir", type=str,
                        default="cache/depth/kitti360")
    parser.add_argument("--range_cache_dir", type=str,
                        default="cache/range/kitti360")
    parser.add_argument("--subsample", type=int, default=1)
    parser.add_argument("--resize", type=int, nargs=2, default=[480, 640])
    parser.add_argument("--camera_hfov_deg", type=float, default=90.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--exclude_self", type=float, default=0.0,
                        help="Exclude DB entries within this distance of query (m)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    input_size = tuple(args.resize)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Sequences: {args.sequences}")
    print(f"Subsample: {args.subsample}")
    print(f"Self-exclusion: {args.exclude_self}m")
    print()

    # Load model
    print("Loading model...")
    model = load_model(args.checkpoint, device)
    print()

    # Build datasets
    print("Building datasets...")
    ds_range = KITTI360LC2Dataset(
        root=args.root, sequences=args.sequences,
        modality="range",
        range_cache_dir=args.range_cache_dir,
        subsample=args.subsample,
        input_size=input_size,
        camera_hfov_deg=args.camera_hfov_deg,
    )
    ds_depth = KITTI360LC2Dataset(
        root=args.root, sequences=args.sequences,
        modality="depth",
        depth_cache_dir=args.depth_cache_dir,
        subsample=args.subsample,
        input_size=input_size,
    )
    print(f"  Range (query): {len(ds_range)} samples")
    print(f"  Depth (DB): {len(ds_depth)} samples")
    print()

    # Extract descriptors
    print("Extracting descriptors...")
    t0 = time.time()
    q_desc = extract_descriptors(model, ds_range, is_range=True, device=device)
    db_desc = extract_descriptors(model, ds_depth, is_range=False, device=device)
    dt = time.time() - t0
    print(f"  Query: {q_desc.shape}, DB: {db_desc.shape}")
    print(f"  Time: {dt:.1f}s")
    print()

    # Get positions
    q_pos = ds_range.get_positions()
    db_pos = ds_depth.get_positions()

    # Evaluate
    ks = [1, 5, 10, 15, 20]
    top_k = 25
    pos_thresholds = [5.0, 10.0, 25.0]

    index = build_index(db_desc)
    distances, predictions = retrieve(q_desc, index, top_k=top_k)

    # Optional: exclude self-matches (entries too close in trajectory)
    if args.exclude_self > 0:
        print(f"Excluding DB entries within {args.exclude_self}m of query...")
        for i in range(len(predictions)):
            valid_preds = []
            for j in range(predictions.shape[1]):
                pred_idx = predictions[i, j]
                dist = np.linalg.norm(q_pos[i] - db_pos[pred_idx])
                # Only exclude if it's likely the same frame (very close in index too)
                if dist > args.exclude_self or abs(i - pred_idx) > 5:
                    valid_preds.append(pred_idx)
            # Pad if needed
            while len(valid_preds) < top_k:
                valid_preds.append(valid_preds[-1] if valid_preds else 0)
            predictions[i] = valid_preds[:top_k]

    print("=" * 60)
    print(f"  KITTI-360 seq {args.sequences}")
    print(f"  Query (range): {len(q_pos)}, DB (depth): {len(db_pos)}")
    print("-" * 60)

    for thr in pos_thresholds:
        gt = compute_gt_positives(q_pos, db_pos, thr)
        recalls = compute_recall(predictions, gt, ks=ks)
        print(f"  pos_threshold = {thr}m:")
        for k in ks:
            print(f"    Recall@{k:>2d}: {recalls[k]*100:6.2f}%")
        print()

    print("=" * 60)

    # Top-1 retrieval distance stats
    top1_dists = []
    for i in range(len(predictions)):
        pred_idx = predictions[i, 0]
        dist = np.linalg.norm(q_pos[i] - db_pos[pred_idx])
        top1_dists.append(dist)
    top1_dists = np.array(top1_dists)
    print(f"\nTop-1 retrieval distance:")
    print(f"  median={np.median(top1_dists):.1f}m, "
          f"mean={np.mean(top1_dists):.1f}m")
    print(f"  <5m: {np.mean(top1_dists<5)*100:.1f}%, "
          f"<10m: {np.mean(top1_dists<10)*100:.1f}%, "
          f"<25m: {np.mean(top1_dists<25)*100:.1f}%")


if __name__ == "__main__":
    main()
