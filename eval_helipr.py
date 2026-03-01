"""HeLiPR evaluation for trained LC2 models.

Cross-sensor: query=ouster, DB=velodyne from same sequence.

Usage::
    python eval_helipr.py --checkpoint checkpoints/helipr_v5/best.pth.tar
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
from lc2.data.helipr import HeLiPRLC2Dataset
from lc2.utils.retrieval import (
    build_index, retrieve, compute_gt_positives, compute_recall,
)


def load_model(checkpoint_path, device, num_clusters=16, encoder_dim=512):
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
    for batch in tqdm(loader, desc=f"{'Ouster' if is_range else 'Velodyne'}"):
        images = batch["image"].to(device)
        desc = model.forward_single(images, is_range=is_range)
        all_desc.append(desc.cpu().numpy())
    return np.concatenate(all_desc, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--root", type=str, default="/media/jhlee/EVO4TB/HeLiPR")
    parser.add_argument("--sequences", type=str, nargs="+", default=["DCC04"])
    parser.add_argument("--ouster_cache_dir", type=str,
                        default="cache/range/helipr/Ouster")
    parser.add_argument("--velodyne_cache_dir", type=str,
                        default="cache/range/helipr/Velodyne")
    parser.add_argument("--subsample", type=int, default=5)
    parser.add_argument("--resize", type=int, nargs=2, default=[128, 1024])
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    input_size = tuple(args.resize)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Sequences: {args.sequences}")
    print(f"Subsample: {args.subsample}")
    print()

    print("Loading model...")
    model = load_model(args.checkpoint, device)
    print()

    print("Building datasets...")
    ds_ouster = HeLiPRLC2Dataset(
        root=args.root, sequences=args.sequences,
        modality="ouster",
        ouster_cache_dir=args.ouster_cache_dir,
        subsample=args.subsample,
        input_size=input_size,
    )
    ds_velodyne = HeLiPRLC2Dataset(
        root=args.root, sequences=args.sequences,
        modality="velodyne",
        velodyne_cache_dir=args.velodyne_cache_dir,
        subsample=args.subsample,
        input_size=input_size,
    )
    print(f"  Ouster (query): {len(ds_ouster)} samples")
    print(f"  Velodyne (DB): {len(ds_velodyne)} samples")
    print()

    print("Extracting descriptors...")
    t0 = time.time()
    # Both sensors are range images → route all through encoder_r
    q_desc = extract_descriptors(model, ds_ouster, is_range=True, device=device)
    db_desc = extract_descriptors(model, ds_velodyne, is_range=True, device=device)
    dt = time.time() - t0
    print(f"  Query: {q_desc.shape}, DB: {db_desc.shape}")
    print(f"  Time: {dt:.1f}s")
    print()

    q_pos = ds_ouster.get_positions()
    db_pos = ds_velodyne.get_positions()

    ks = [1, 5, 10, 15, 20]
    top_k = 25
    pos_thresholds = [5.0, 10.0, 25.0]

    index = build_index(db_desc)
    distances, predictions = retrieve(q_desc, index, top_k=top_k)

    print("=" * 60)
    print(f"  HeLiPR {args.sequences}: Ouster → Velodyne")
    print(f"  Queries: {len(q_pos)}, Database: {len(db_pos)}")
    print("-" * 60)

    for thr in pos_thresholds:
        gt = compute_gt_positives(q_pos, db_pos, thr)
        recalls = compute_recall(predictions, gt, ks=ks)
        print(f"  pos_threshold = {thr}m:")
        for k in ks:
            print(f"    Recall@{k:>2d}: {recalls[k]*100:6.2f}%")
        print()

    print("=" * 60)

    top1_dists = []
    for i in range(len(predictions)):
        pred_idx = predictions[i, 0]
        dist = np.linalg.norm(q_pos[i] - db_pos[pred_idx])
        top1_dists.append(dist)
    top1_dists = np.array(top1_dists)
    print(f"\nTop-1 retrieval distance:")
    print(f"  median={np.median(top1_dists):.1f}m, mean={np.mean(top1_dists):.1f}m")
    print(f"  <5m: {np.mean(top1_dists<5)*100:.1f}%, "
          f"<10m: {np.mean(top1_dists<10)*100:.1f}%, "
          f"<25m: {np.mean(top1_dists<25)*100:.1f}%")


if __name__ == "__main__":
    main()
