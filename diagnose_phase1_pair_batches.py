#!/usr/bin/env python3
"""Diagnose Phase 1 pair construction and InfoNCE batch semantics.

Recreates the exact Phase 1 pool/miner/pair dataset from a training config,
prints pair modality/distance statistics, and inspects a few shuffled batches.

Optionally loads a checkpoint to report GeM.p and verify mixed-batch encoder
routing against per-sample branch-specific forwards.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from train import load_config
from lc2.model import LC2Model
from lc2.data.transforms import get_transform
from lc2.data.train_dataset import (
    ContrastivePairMiner,
    LC2ContrastivePairDataset,
    build_vivid_phase1_pool,
)


class IndexedPairDataset(torch.utils.data.Dataset):
    """Wraps LC2ContrastivePairDataset to expose sampled pair indices."""

    def __init__(self, base: LC2ContrastivePairDataset, pairs, positions: np.ndarray) -> None:
        self.base = base
        self.pairs = pairs
        self.positions = positions

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        item = self.base[idx]
        i, j, psi = self.pairs[idx]
        dist = float(np.linalg.norm(self.positions[i] - self.positions[j]))
        item["pair_index"] = idx
        item["pool_i"] = i
        item["pool_j"] = j
        item["pair_dist_m"] = dist
        item["pair_kind"] = pair_kind(self.base.pool.is_range[i], self.base.pool.is_range[j])
        return item


def modality_name(is_range: bool) -> str:
    return "range" if is_range else "depth"


def pair_kind(is_range_i: bool, is_range_j: bool) -> str:
    if is_range_i and is_range_j:
        return "rr"
    if (not is_range_i) and (not is_range_j):
        return "dd"
    if is_range_i and (not is_range_j):
        return "rd"
    return "dr"


def tensor_stats(x: torch.Tensor) -> Dict[str, float]:
    return {
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
    }


def count_modalities(pairs: Iterable[Tuple[int, int, float]], is_range: list[bool]) -> Dict[str, int]:
    counts = {"rr": 0, "dd": 0, "rd": 0, "dr": 0}
    for i, j, _ in pairs:
        counts[pair_kind(is_range[i], is_range[j])] += 1
    return counts


def load_model_from_checkpoint(cfg: Dict, checkpoint_path: Path, device: torch.device) -> LC2Model:
    model_cfg = cfg["model"]
    model = LC2Model(
        num_clusters=model_cfg.get("num_clusters", 16),
        encoder_dim=model_cfg.get("encoder_dim", 512),
        pooling="gem",
        freeze_until=model_cfg.get("freeze_until", 24),
    ).to(device)

    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num-batches", type=int, default=10)
    parser.add_argument("--num-pairs-print", type=int, default=12)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_cfg = cfg["train"]
    dataset_cfg = cfg["dataset"]
    input_cfg = cfg.get("input", {})
    model_cfg = cfg["model"]

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    resize_cfg = input_cfg.get("resize", None)
    input_size = tuple(resize_cfg) if resize_cfg else None
    transform = get_transform(input_size)

    print("=== CONFIG SUMMARY ===")
    print(f"config={args.config}")
    print(f"dataset.name={dataset_cfg.get('name')}")
    print(f"train.sequences={train_cfg.get('sequences', dataset_cfg.get('sequences'))}")
    print(f"train.batch_size={train_cfg.get('batch_size')}")
    print(f"train.phase1_loss={train_cfg.get('phase1_loss')}")
    print(f"train.phase1_positive_mode={train_cfg.get('phase1_positive_mode')}")
    print(f"train.spatial_radius={train_cfg.get('spatial_radius')}")
    print(f"train.n_crops={train_cfg.get('n_crops')}")
    print(f"train.scale_augment_pct={train_cfg.get('scale_augment_pct')}")
    print(f"model.freeze_until={model_cfg.get('freeze_until')}")
    print()

    print("=== BUILD PHASE1 POOL (exact train.py path) ===")
    pool = build_vivid_phase1_pool(
        root=dataset_cfg["root"],
        sequences=train_cfg.get("sequences", dataset_cfg.get("sequences")),
        depth_cache_dir=dataset_cfg.get("depth_cache_dir"),
        range_cache_dir=dataset_cfg.get("range_cache_dir"),
        range_subsample=train_cfg.get("range_subsample", 10),
        depth_subsample=train_cfg.get("depth_subsample", 10),
        n_crops=1 if input_cfg.get("range_is_camproj", False) else train_cfg.get("n_crops", 8),
        crop_fov_deg=train_cfg.get("crop_fov_deg", 90.0),
        camera_hfov_deg=input_cfg.get("camera_hfov_deg", 90.0),
        range_is_camproj=input_cfg.get("range_is_camproj", False),
        crop_px=input_cfg.get("crop_px", 0),
    )
    n_range = sum(1 for x in pool.is_range if x)
    n_depth = len(pool) - n_range
    print(f"pool entries={len(pool)} range={n_range} depth={n_depth}")
    print("NOTE: train.spatial_radius is NOT applied in build_vivid_phase1_pool(); only validation/phase2 crop uses it.")
    print()

    positions = pool.position_array()
    print("=== MINE PAIRS ===")
    miner = ContrastivePairMiner(
        pool=pool,
        max_range_m=50.0,
        camera_fov_deg=input_cfg.get("camera_hfov_deg", 90.0) or 90.0,
        positive_mode=train_cfg.get("phase1_positive_mode", "all"),
    )
    pos_pairs = [(i, j, psi) for (i, j, psi) in miner.pairs if psi > 0.0]
    neg_pairs = [(i, j, psi) for (i, j, psi) in miner.pairs if psi == 0.0]
    print(f"total_pairs={len(miner.pairs)} positives={len(pos_pairs)} negatives={len(neg_pairs)}")
    print(f"positive modality counts={count_modalities(pos_pairs, pool.is_range)}")
    print(f"negative modality counts={count_modalities(neg_pairs, pool.is_range)}")
    if neg_pairs:
        print("CRITICAL: InfoNCE ignores psi, so any psi=0 pairs in the dataset are still treated as diagonal positives.")
    print()

    print("=== SAMPLE POSITIVE PAIRS ===")
    for idx, (i, j, psi) in enumerate(pos_pairs[: args.num_pairs_print]):
        dist = float(np.linalg.norm(positions[i] - positions[j]))
        print(
            f"pos[{idx}] i={i}({modality_name(pool.is_range[i])}) "
            f"j={j}({modality_name(pool.is_range[j])}) "
            f"kind={pair_kind(pool.is_range[i], pool.is_range[j])} "
            f"dist={dist:.3f}m psi={psi:.4f}"
        )
    print()

    print("=== SAMPLE NEGATIVE PAIRS ===")
    for idx, (i, j, psi) in enumerate(neg_pairs[: args.num_pairs_print]):
        dist = float(np.linalg.norm(positions[i] - positions[j]))
        print(
            f"neg[{idx}] i={i}({modality_name(pool.is_range[i])}) "
            f"j={j}({modality_name(pool.is_range[j])}) "
            f"kind={pair_kind(pool.is_range[i], pool.is_range[j])} "
            f"dist={dist:.3f}m psi={psi:.1f}"
        )
    print()

    dataset = LC2ContrastivePairDataset(
        pairs=miner.pairs,
        pool=pool,
        transform=transform,
        scale_augment=True,
        max_scale_pct=train_cfg.get("scale_augment_pct", 20.0),
        n_crops=train_cfg.get("n_crops", 8),
        crop_fov_deg=train_cfg.get("crop_fov_deg", 90.0),
        range_augmentation=None,
        depth_augmentation=None,
    )
    indexed_dataset = IndexedPairDataset(dataset, miner.pairs, positions)
    loader = DataLoader(
        indexed_dataset,
        batch_size=train_cfg.get("batch_size", 4),
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    model = None
    if args.checkpoint:
        print("=== CHECKPOINT / GEM / ROUTING ===")
        ckpt_path = Path(args.checkpoint)
        model = load_model_from_checkpoint(cfg, ckpt_path, device)
        print(f"checkpoint={ckpt_path}")
        print(f"gem.p={float(model.gem.p.detach().cpu().item()):.6f}")
        print()

    print("=== FIRST BATCHES ===")
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= args.num_batches:
            break

        img_i = batch["image_i"]
        img_j = batch["image_j"]
        is_range_i = batch["is_range_i"]
        is_range_j = batch["is_range_j"]
        psi = batch["psi"]
        pair_index = batch["pair_index"]
        pool_i = batch["pool_i"]
        pool_j = batch["pool_j"]
        pair_dist = batch["pair_dist_m"]
        pair_kind_batch = batch["pair_kind"]

        # Recover distances from the sampled dataset indices by re-sampling current batch from the shuffled batch itself.
        # Since DataLoader shuffles indices internally, distance reporting below uses on-batch modality/psi plus image stats.
        stats_i = tensor_stats(img_i)
        stats_j = tensor_stats(img_j)

        rr = int(((is_range_i == 1) & (is_range_j == 1)).sum().item())
        dd = int(((is_range_i == 0) & (is_range_j == 0)).sum().item())
        rd = int(((is_range_i == 1) & (is_range_j == 0)).sum().item())
        dr = int(((is_range_i == 0) & (is_range_j == 1)).sum().item())

        print(f"batch[{batch_idx}]")
        print(f"  is_range_i={is_range_i.tolist()}")
        print(f"  is_range_j={is_range_j.tolist()}")
        print(f"  psi={psi.tolist()}")
        print(f"  pair_index={pair_index.tolist()}")
        print(f"  pool_i={pool_i.tolist()}")
        print(f"  pool_j={pool_j.tolist()}")
        print(f"  pair_kind_str={pair_kind_batch}")
        print(f"  pair_dist_m={[round(float(x), 4) for x in pair_dist.tolist()]}")
        print(f"  pair_kinds: rr={rr} dd={dd} rd={rd} dr={dr}")
        print(f"  image_i_stats={stats_i}")
        print(f"  image_j_stats={stats_j}")

        if model is not None:
            with torch.no_grad():
                img_i_dev = img_i.to(device)
                img_j_dev = img_j.to(device)
                iri = is_range_i.to(device)
                irj = is_range_j.to(device)
                desc_i = model(img_i_dev, iri)
                desc_j = model(img_j_dev, irj)
                diag_cos = torch.sum(desc_i * desc_j, dim=1)
                print(
                    "  desc_diag_cos="
                    f"{[round(float(x), 6) for x in diag_cos.detach().cpu().tolist()]}"
                )

                # Mixed-routing check: batched forward should match per-sample branch-specific forward.
                routed = model.encoder(img_i_dev, iri)
                singles = []
                for k in range(img_i_dev.size(0)):
                    singles.append(
                        model.encoder.forward_single(
                            img_i_dev[k:k + 1], bool(iri[k].item())
                        )
                    )
                singles = torch.cat(singles, dim=0)
                max_abs_diff = (routed - singles).abs().max().item()
                print(f"  routing_max_abs_diff={max_abs_diff:.8f}")
        print()


if __name__ == "__main__":
    main()
