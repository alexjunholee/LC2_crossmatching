"""Cross-sequence evaluation for trained LC2 models.

Evaluates query=range from seq_q vs DB=depth from seq_db.
Supports both original LC2 checkpoints and our training checkpoints.

Usage::
    python eval_crossseq.py --checkpoint checkpoints/vivid_v20/best.pth.tar \
        --query_seq campus_day2 --db_seq campus_day1
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from lc2.model import LC2Model
from lc2.data.vivid import VIVIDLC2Dataset
from lc2.data.transforms import get_transform
from lc2.utils.retrieval import (
    build_index, retrieve, compute_gt_positives, compute_recall,
)


def load_model(checkpoint_path, device, num_clusters=16, encoder_dim=512):
    """Load model from either training checkpoint or original LC2 format."""
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "phase" in ckpt:
        # Our training checkpoint
        print(f"  Training checkpoint: phase={ckpt['phase']}, "
              f"epoch={ckpt['epoch']}, best={ckpt.get('best_score', 0):.4f}")
        model = LC2Model(
            num_clusters=num_clusters, encoder_dim=encoder_dim,
            vladv2=False, pooling="gem",
        )
        model.load_state_dict(ckpt["state_dict"], strict=False)
    else:
        # Original LC2 checkpoint
        model = LC2Model.from_checkpoint(checkpoint_path, device=device, pooling="gem")

    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def extract_descriptors(model, dataset, is_range, device, batch_size=64):
    """Extract descriptors from a dataset."""
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
    parser.add_argument("--query_seq", type=str, default="campus_day2")
    parser.add_argument("--db_seq", type=str, default="campus_day1")
    parser.add_argument("--root", type=str,
                        default="/media/jhlee/EVO4TB/vivid_projects/data")
    parser.add_argument("--depth_cache_dir", type=str,
                        default="cache/depth/vivid")
    parser.add_argument("--subsample", type=int, default=1,
                        help="Subsample factor (1=all, 5=every 5th, 10=every 10th)")
    parser.add_argument("--resize", type=int, nargs=2, default=[480, 640])
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    input_size = tuple(args.resize)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Query: {args.query_seq} (range)")
    print(f"DB: {args.db_seq} (depth)")
    print(f"Subsample: {args.subsample}")
    print()

    # Load model
    print("Loading model...")
    model = load_model(args.checkpoint, device)
    print()

    # Build datasets
    print("Building datasets...")
    ds_query = VIVIDLC2Dataset(
        root=args.root, sequence=args.query_seq,
        modality="range", subsample=args.subsample,
        input_size=input_size,
    )
    ds_db = VIVIDLC2Dataset(
        root=args.root, sequence=args.db_seq,
        modality="depth", depth_cache_dir=args.depth_cache_dir,
        subsample=args.subsample,
        input_size=input_size,
    )
    print(f"  Query (range): {len(ds_query)} samples")
    print(f"  DB (depth): {len(ds_db)} samples")
    print()

    # Extract descriptors
    print("Extracting descriptors...")
    t0 = time.time()
    q_desc = extract_descriptors(model, ds_query, is_range=True, device=device)
    db_desc = extract_descriptors(model, ds_db, is_range=False, device=device)
    dt = time.time() - t0
    print(f"  Query: {q_desc.shape}, DB: {db_desc.shape}")
    print(f"  Time: {dt:.1f}s")
    print()

    # Get positions
    q_pos = ds_query.get_positions()
    db_pos = ds_db.get_positions()

    # Evaluate
    ks = [1, 5, 10, 15, 20]
    top_k = 25
    pos_thresholds = [5.0, 10.0, 25.0]

    index = build_index(db_desc)
    distances, predictions = retrieve(q_desc, index, top_k=top_k)

    print("=" * 60)
    print(f"  Cross-sequence: {args.query_seq} (range) → {args.db_seq} (depth)")
    print(f"  Queries: {len(q_pos)}, Database: {len(db_pos)}")
    print("-" * 60)

    for thr in pos_thresholds:
        gt = compute_gt_positives(q_pos, db_pos, thr)
        recalls = compute_recall(predictions, gt, ks=ks)
        print(f"  pos_threshold = {thr}m:")
        for k in ks:
            print(f"    Recall@{k:>2d}: {recalls[k]*100:6.2f}%")
        print()

    # Also do reverse: DB=range from db_seq, Query=depth from query_seq
    print("-" * 60)
    print(f"  Reverse: {args.db_seq} (range) → {args.query_seq} (depth)")

    ds_query_rev = VIVIDLC2Dataset(
        root=args.root, sequence=args.db_seq,
        modality="range", subsample=args.subsample,
        input_size=input_size,
    )
    ds_db_rev = VIVIDLC2Dataset(
        root=args.root, sequence=args.query_seq,
        modality="depth", depth_cache_dir=args.depth_cache_dir,
        subsample=args.subsample,
        input_size=input_size,
    )
    print(f"  Query (range): {len(ds_query_rev)} samples")
    print(f"  DB (depth): {len(ds_db_rev)} samples")

    q_desc_rev = extract_descriptors(model, ds_query_rev, is_range=True, device=device)
    db_desc_rev = extract_descriptors(model, ds_db_rev, is_range=False, device=device)
    q_pos_rev = ds_query_rev.get_positions()
    db_pos_rev = ds_db_rev.get_positions()

    index_rev = build_index(db_desc_rev)
    distances_rev, predictions_rev = retrieve(q_desc_rev, index_rev, top_k=top_k)

    for thr in pos_thresholds:
        gt = compute_gt_positives(q_pos_rev, db_pos_rev, thr)
        recalls = compute_recall(predictions_rev, gt, ks=ks)
        print(f"  pos_threshold = {thr}m:")
        for k in ks:
            print(f"    Recall@{k:>2d}: {recalls[k]*100:6.2f}%")
        print()

    print("=" * 60)

    # Compute descriptor statistics
    print("\nDescriptor statistics:")
    q_norms = np.linalg.norm(q_desc, axis=1)
    db_norms = np.linalg.norm(db_desc, axis=1)
    print(f"  Query norm: mean={q_norms.mean():.4f}, std={q_norms.std():.6f}")
    print(f"  DB norm: mean={db_norms.mean():.4f}, std={db_norms.std():.6f}")

    # Median retrieval distance
    for thr in [25.0]:
        gt = compute_gt_positives(q_pos, db_pos, thr)
        top1_dists = []
        for i in range(len(predictions)):
            pred_idx = predictions[i, 0]
            dist = np.linalg.norm(q_pos[i] - db_pos[pred_idx])
            top1_dists.append(dist)
        top1_dists = np.array(top1_dists)
        print(f"\n  Top-1 retrieval distance: "
              f"median={np.median(top1_dists):.1f}m, "
              f"mean={np.mean(top1_dists):.1f}m, "
              f"<25m={np.mean(top1_dists<25)*100:.1f}%")


if __name__ == "__main__":
    main()
