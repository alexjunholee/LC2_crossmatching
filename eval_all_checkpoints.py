#!/usr/bin/env python3
"""Batch-evaluate all checkpoints for a dataset, producing a summary YAML."""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from lc2.model import LC2Model
from evaluate import build_dataset, extract_descriptors, get_all_positions
from lc2.utils.retrieval import build_index, retrieve, compute_gt_positives, compute_recall


def clean_name(p: Path) -> str:
    """Strip .pth.tar → p2a_epoch_0."""
    name = p.name
    for suffix in (".pth.tar", ".pth", ".tar"):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break
    return name


def natural_sort_key(p: Path):
    """Sort p2a_epoch_0 < p2a_epoch_1 < ... < p2b_epoch_0 < ..."""
    name = clean_name(p)
    if name == "best":
        return (2, 999)  # sort last
    parts = name.split("_")
    phase = 0 if "p2a" in name else (1 if "p2b" in name else 2)
    try:
        epoch = int(parts[-1])
    except ValueError:
        epoch = 999
    return (phase, epoch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--output", default=None, help="Output YAML path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip_best", action="store_true",
                        help="Skip best.pth.tar (it duplicates one of the epoch files)")
    parser.add_argument("--stride", type=int, default=1,
                        help="Evaluate every N-th checkpoint (1=all)")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(args.checkpoint_dir)

    # Collect checkpoints
    ckpts = sorted(ckpt_dir.glob("p2*_epoch_*.pth.tar"), key=natural_sort_key)
    if not args.skip_best and (ckpt_dir / "best.pth.tar").exists():
        ckpts.append(ckpt_dir / "best.pth.tar")

    # Apply stride (keep first P2a, last P2a, then every N-th P2b)
    if args.stride > 1:
        p2a = [c for c in ckpts if "p2a" in c.stem]
        p2b = [c for c in ckpts if "p2b" in c.stem]
        best = [c for c in ckpts if c.stem == "best"]
        # Keep first and last P2a
        p2a_selected = []
        if p2a:
            p2a_selected = [p2a[0]]
            if len(p2a) > 1:
                p2a_selected.append(p2a[-1])
        # Every N-th P2b
        p2b_selected = p2b[::args.stride]
        if p2b and p2b[-1] not in p2b_selected:
            p2b_selected.append(p2b[-1])
        ckpts = p2a_selected + p2b_selected + best

    print(f"Dataset: {cfg['dataset']['name']}")
    print(f"Checkpoints: {len(ckpts)} files from {ckpt_dir}")
    print(f"Evaluating: {[clean_name(c) for c in ckpts]}")
    print()

    # Build datasets once
    query_modality = cfg["dataset"].get("query_modality", "range")
    db_modality = cfg["dataset"].get("db_modality", "depth")

    print(f"Building datasets (query={query_modality}, db={db_modality})...")
    query_dataset = build_dataset(cfg, query_modality)
    db_dataset = build_dataset(cfg, db_modality)
    print(f"  Query: {len(query_dataset)}, DB: {len(db_dataset)}")

    query_positions = get_all_positions(query_dataset)
    db_positions = get_all_positions(db_dataset)

    batch_size = cfg["eval"].get("batch_size", 64)
    num_workers = cfg["eval"].get("num_workers", 4)

    query_loader = torch.utils.data.DataLoader(
        query_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    db_loader = torch.utils.data.DataLoader(
        db_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    pos_thresholds = cfg["eval"].get("pos_thresholds", [5.0, 10.0, 25.0])
    ks = cfg["eval"].get("recall_ks", [1, 5, 10])
    top_k = cfg["eval"].get("top_k", 25)

    # Pre-compute GT
    gt_by_thr = {}
    for thr in pos_thresholds:
        gt_by_thr[thr] = compute_gt_positives(query_positions, db_positions, thr)

    results = {}

    for ci, ckpt_path in enumerate(ckpts):
        name = clean_name(ckpt_path)
        print(f"\n[{ci+1}/{len(ckpts)}] {name}")
        t0 = time.time()

        # Load model
        model = LC2Model.from_checkpoint(ckpt_path, device=device)

        # Extract descriptors
        q_desc = extract_descriptors(model, query_loader,
                                     is_range=(query_modality == "range"),
                                     device=device)
        d_desc = extract_descriptors(model, db_loader,
                                     is_range=(db_modality == "range"),
                                     device=device)

        # Retrieve
        index = build_index(d_desc)
        distances, predictions = retrieve(q_desc, index, top_k=top_k)

        # Compute recalls at each threshold
        ckpt_results = {}
        for thr in pos_thresholds:
            recalls = compute_recall(predictions, gt_by_thr[thr], ks=ks)
            for k in ks:
                ckpt_results[f"R@{k}@{int(thr)}m"] = float(recalls[k])

        results[name] = ckpt_results
        dt = time.time() - t0

        # Print summary
        for thr in pos_thresholds:
            r1 = ckpt_results[f"R@1@{int(thr)}m"]
            r5 = ckpt_results[f"R@5@{int(thr)}m"]
            r10 = ckpt_results[f"R@10@{int(thr)}m"]
            print(f"  @{int(thr)}m: R@1={r1*100:.1f}%, R@5={r5*100:.1f}%, R@10={r10*100:.1f}%")
        print(f"  time: {dt:.1f}s")

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

    # Save
    output_path = args.output or str(
        Path("results") / f"{cfg['dataset']['name']}_all_checkpoints.yaml"
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(results, f, default_flow_style=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
