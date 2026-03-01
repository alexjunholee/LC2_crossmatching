"""LC2 cross-modal place recognition evaluation.

Main entry point for evaluating the LC2 model on VIVID or KITTI-360 datasets.
Loads a pretrained dual encoder + NetVLAD checkpoint, extracts descriptors
for query (range) and database (depth) modalities, and computes Recall@K.

Usage::

    python evaluate.py --config configs/vivid.yaml
    python evaluate.py --config configs/kitti360.yaml
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from lc2.model import LC2Model
from lc2.data.vivid import VIVIDLC2Dataset
from lc2.data.kitti360 import KITTI360LC2Dataset
from lc2.data.helipr import HeLiPRLC2Dataset
from lc2.utils.retrieval import evaluate_retrieval


def load_config(config_path: str) -> Dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_dataset(cfg: Dict, modality: str):
    """Build dataset from config for the specified modality."""
    dataset_name = cfg["dataset"]["name"]
    resize_cfg = cfg.get("input", {}).get("resize", None)
    input_size = tuple(resize_cfg) if resize_cfg else None

    subsample = cfg.get("eval", {}).get("subsample", 1)

    if dataset_name == "vivid":
        camera_hfov_deg = cfg.get("input", {}).get("camera_hfov_deg", None)
        datasets = []
        for seq in cfg["dataset"]["sequences"]:
            ds = VIVIDLC2Dataset(
                root=cfg["dataset"]["root"],
                sequence=seq,
                modality=modality,
                depth_cache_dir=cfg["dataset"].get("depth_cache_dir"),
                range_cache_dir=cfg["dataset"].get("range_cache_dir"),
                input_size=input_size,
                subsample=subsample,
                camera_hfov_deg=camera_hfov_deg if modality == "range" else None,
            )
            datasets.append(ds)
        if len(datasets) == 1:
            return datasets[0]
        return torch.utils.data.ConcatDataset(datasets)

    elif dataset_name == "kitti360":
        camera_hfov_deg = cfg.get("input", {}).get("camera_hfov_deg", None)
        return KITTI360LC2Dataset(
            root=cfg["dataset"]["root"],
            sequences=cfg["dataset"]["sequences"],
            modality=modality,
            depth_cache_dir=cfg["dataset"].get("depth_cache_dir"),
            range_cache_dir=cfg["dataset"].get("range_cache_dir"),
            input_size=input_size,
            camera_hfov_deg=camera_hfov_deg,
        )

    elif dataset_name == "helipr":
        # For HeLiPR: query_modality=ouster, db_modality=velodyne
        return HeLiPRLC2Dataset(
            root=cfg["dataset"]["root"],
            sequences=cfg["dataset"]["sequences"],
            modality=modality,
            ouster_cache_dir=cfg["dataset"].get("ouster_cache_dir"),
            velodyne_cache_dir=cfg["dataset"].get("velodyne_cache_dir"),
            input_size=input_size,
        )

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


@torch.no_grad()
def extract_descriptors(
    model: LC2Model,
    dataloader: DataLoader,
    is_range: bool,
    device: torch.device,
    use_fp16: bool = False,
) -> np.ndarray:
    """Extract global descriptors for an entire dataset.

    Args:
        model: LC2 model in eval mode.
        dataloader: DataLoader yielding batches of samples.
        is_range: Whether inputs are range images (True) or depth (False).
        device: Computation device.
        use_fp16: If True, use automatic mixed precision.

    Returns:
        Descriptor matrix of shape ``(N, D)`` where D = num_clusters * encoder_dim.
    """
    all_descriptors = []

    for batch in tqdm(dataloader, desc=f"{'Range' if is_range else 'Depth'} descriptors"):
        images = batch["image"].to(device)

        if use_fp16:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                desc = model.forward_single(images, is_range=is_range)
        else:
            desc = model.forward_single(images, is_range=is_range)

        all_descriptors.append(desc.cpu().numpy())

    return np.concatenate(all_descriptors, axis=0)


def get_all_positions(dataset) -> np.ndarray:
    """Extract positions from a dataset (handles ConcatDataset)."""
    if hasattr(dataset, "get_positions"):
        return dataset.get_positions()

    # ConcatDataset
    if hasattr(dataset, "datasets"):
        all_pos = []
        for ds in dataset.datasets:
            all_pos.append(ds.get_positions())
        return np.concatenate(all_pos, axis=0)

    raise ValueError("Dataset does not provide positions")


def main():
    parser = argparse.ArgumentParser(description="LC2 Cross-Modal Place Recognition Evaluation")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Override checkpoint path")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Resolve checkpoint path
    checkpoint_path = args.checkpoint or cfg["model"]["checkpoint"]
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path(__file__).parent / checkpoint_path

    print(f"Config: {args.config}")
    print(f"Dataset: {cfg['dataset']['name']}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print()

    # Load model
    print("Loading LC2 model...")
    model = LC2Model.from_checkpoint(checkpoint_path, device=device)
    print(f"  Descriptor dimension: {model.descriptor_dim}")
    print(f"  Clusters: {model.num_clusters}, Encoder dim: {model.encoder_dim}")
    print()

    # Build datasets
    query_modality = cfg["dataset"].get("query_modality", "range")
    db_modality = cfg["dataset"].get("db_modality", "depth")

    print(f"Building query dataset (modality={query_modality})...")
    query_dataset = build_dataset(cfg, query_modality)
    print(f"  Query samples: {len(query_dataset)}")

    print(f"Building database dataset (modality={db_modality})...")
    db_dataset = build_dataset(cfg, db_modality)
    print(f"  Database samples: {len(db_dataset)}")
    print()

    # DataLoaders
    batch_size = cfg["eval"].get("batch_size", 100)
    num_workers = cfg["eval"].get("num_workers", 4)

    query_loader = DataLoader(
        query_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    db_loader = DataLoader(
        db_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Extract descriptors
    print("Extracting descriptors...")
    t0 = time.time()

    query_desc = extract_descriptors(
        model, query_loader,
        is_range=(query_modality == "range"),
        device=device,
        use_fp16=args.fp16,
    )
    db_desc = extract_descriptors(
        model, db_loader,
        is_range=(db_modality == "range"),
        device=device,
        use_fp16=args.fp16,
    )

    t1 = time.time()
    print(f"  Query descriptors: {query_desc.shape}")
    print(f"  Database descriptors: {db_desc.shape}")
    print(f"  Extraction time: {t1 - t0:.1f}s")
    print()

    # Get positions
    query_positions = get_all_positions(query_dataset)
    db_positions = get_all_positions(db_dataset)

    # Evaluate at multiple thresholds
    pos_threshold = cfg["dataset"].get("pos_threshold", 25.0)
    pos_thresholds = cfg["eval"].get("pos_thresholds", [5.0, 10.0, 25.0])
    ks = cfg["eval"].get("recall_ks", [1, 5, 10, 15, 20])
    top_k = cfg["eval"].get("top_k", 25)

    # Build index and retrieve once
    from lc2.utils.retrieval import build_index, retrieve, compute_gt_positives, compute_recall
    index = build_index(db_desc)
    distances, predictions = retrieve(query_desc, index, top_k=top_k)

    print()
    print("=" * 60)
    print(f"  Dataset: {cfg['dataset']['name']}")
    print(f"  Sequences: {cfg['dataset']['sequences']}")
    print(f"  Queries: {len(query_positions)}, Database: {len(db_positions)}")
    print("-" * 60)

    all_recalls = {}
    for thr in pos_thresholds:
        gt_positives = compute_gt_positives(query_positions, db_positions, thr)
        recalls = compute_recall(predictions, gt_positives, ks=ks)
        print(f"  pos_threshold = {thr}m:")
        for k in ks:
            val = recalls[k]
            print(f"    Recall@{k:>2d}: {val * 100:6.2f}%")
            all_recalls[f"R@{k}@{int(thr)}m"] = val
    print("=" * 60)

    # Also compute with primary threshold for backward compatibility
    gt_positives = compute_gt_positives(query_positions, db_positions, pos_threshold)
    results = {
        "predictions": predictions,
        "distances": distances,
        "gt_positives": gt_positives,
        "num_queries": len(query_positions),
        "num_db": len(db_positions),
    }
    top1_correct = np.array([
        predictions[i, 0] in gt_positives[i]
        for i in range(len(gt_positives))
    ], dtype=bool)
    results["top1_correct"] = top1_correct
    primary_recalls = compute_recall(predictions, gt_positives, ks=ks)
    for k, v in primary_recalls.items():
        results[f"recall@{k}"] = v

    # Save results
    output_dir = Path(args.output_dir) / f"{cfg['dataset']['name']}_{'_'.join(cfg['dataset']['sequences'])}"
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "query_descriptors.npy", query_desc)
    np.save(output_dir / "db_descriptors.npy", db_desc)
    np.save(output_dir / "predictions.npy", results["predictions"])
    np.save(output_dir / "distances.npy", results["distances"])
    np.save(output_dir / "query_positions.npy", query_positions)
    np.save(output_dir / "db_positions.npy", db_positions)
    np.save(output_dir / "top1_correct.npy", results["top1_correct"])

    # Save recall summary (all thresholds)
    recall_summary = {}
    recall_summary.update(all_recalls)
    recall_summary["pos_threshold"] = pos_threshold
    recall_summary["num_queries"] = results["num_queries"]
    recall_summary["num_db"] = results["num_db"]

    with open(output_dir / "results.yaml", "w") as f:
        yaml.dump(recall_summary, f, default_flow_style=False)

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
