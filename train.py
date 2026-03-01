"""LC2 paper-faithful two-phase training pipeline.

Phase 1 — Contrastive pre-training (Section III.B.3):
    - Pooling: GeM (512-D descriptors)
    - Loss: Modified contrastive loss (Eq. 2) weighted by degree of similarity psi
    - Range: 8 FoV-masked overlapping crops (Section III.C.1)
    - Depth: converted to disparity + scale augmentation r=20% (Section III.C.2)
    - Encoder conv5 trainable

Phase 2 — Triplet fine-tuning (Section III.B.4):
    - Pooling: NetVLAD (K*512-D descriptors)
    - Loss: Triplet margin loss (Eq. 3), m=0.1
    - Range: cropped to camera FoV
    - Depth: converted to disparity
    - Hard negative mining, conv5 trainable

Usage::

    # Full paper pipeline (Phase 1 + Phase 2):
    python train.py --config configs/train_vivid.yaml --resume pretrained/dual_encoder.pth.tar

    # Skip Phase 1 (use pretrained encoder, Phase 2 only):
    python train.py --config configs/train_vivid.yaml --resume pretrained/dual_encoder.pth.tar --skip_phase1

Reference: Lee et al., "(LC)^2: LiDAR-Camera Loop Constraints for
Cross-Modal Place Recognition", RA-L 2023.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from lc2.model import LC2Model
from lc2.losses import LC2ContrastiveLoss
from lc2.data.transforms import (
    get_transform, depth_to_normalized_disparity,
    range_to_normalized_disparity, normalize_disparity, squeeze_depth, crop_range_to_camera_fov,
    DepthAugmentation,
)
from lc2.lidar.augmentation import RangeAugmentation
from lc2.data.train_dataset import (
    ImagePool,
    Phase1Pool,
    TripletMiner,
    ContrastivePairMiner,
    LC2TripletDataset,
    LC2ContrastivePairDataset,
    build_vivid_pool,
    build_vivid_phase1_pool,
    build_kitti360_pool,
    build_kitti360_phase1_pool,
    build_helipr_pool,
)
from lc2.data.vivid import VIVIDLC2Dataset
from lc2.data.kitti360 import KITTI360LC2Dataset
from lc2.data.helipr import HeLiPRLC2Dataset
from lc2.utils.retrieval import evaluate_retrieval


def load_config(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def init_netvlad_from_data(
    model: LC2Model,
    pool: ImagePool,
    transform,
    device: torch.device,
    num_clusters: int = 16,
    batch_size: int = 64,
    camera_hfov_deg: float = 90.0,
) -> None:
    """Initialize NetVLAD centroids via K-means on encoder features."""
    from sklearn.cluster import MiniBatchKMeans

    print(f"Initializing NetVLAD clusters (K={num_clusters})...")
    model.eval()

    n_samples = min(len(pool), 500)
    indices = np.random.choice(len(pool), size=n_samples, replace=False)
    bs = min(batch_size, 16)

    all_feats = []
    with torch.no_grad():
        for start in tqdm(range(0, len(indices), bs), desc="Collecting features"):
            batch_idx = indices[start : start + bs]
            images = []
            is_range_flags = []
            for idx in batch_idx:
                data = np.load(str(pool.paths[idx]))
                if data.ndim > 2:
                    data = squeeze_depth(data)

                if pool.is_range[idx]:
                    if camera_hfov_deg < 360.0:
                        data = crop_range_to_camera_fov(data, camera_hfov_deg=camera_hfov_deg)
                    data = normalize_disparity(data)
                else:
                    data = depth_to_normalized_disparity(data)

                img = transform(data)
                images.append(img)
                is_range_flags.append(pool.is_range[idx])

            images_t = torch.stack(images).to(device)
            is_range_t = torch.tensor(is_range_flags, device=device)
            features = model.encoder(images_t, is_range_t)
            B, D, H, W = features.shape
            feats_flat = features.permute(0, 2, 3, 1).reshape(-1, D)
            all_feats.append(feats_flat.cpu().numpy())

    all_feats = np.concatenate(all_feats, axis=0)
    print(f"  Feature matrix: {all_feats.shape}")

    if all_feats.shape[0] > 50000:
        idx = np.random.choice(all_feats.shape[0], 50000, replace=False)
        all_feats = all_feats[idx]

    print(f"  Running K-means (K={num_clusters})...")
    kmeans = MiniBatchKMeans(
        n_clusters=num_clusters, batch_size=4096,
        max_iter=100, random_state=42,
    )
    kmeans.fit(all_feats)

    model.netvlad.init_params(kmeans.cluster_centers_, all_feats)
    print(f"  NetVLAD initialized. alpha = {model.netvlad.alpha:.4f}")


@torch.no_grad()
def validate(
    model: LC2Model,
    cfg: Dict,
    device: torch.device,
    transform,
) -> Dict:
    """Validate with cross-modal Recall@K.

    Uses the active pooling mode (GeM in Phase 1, NetVLAD in Phase 2).
    Applies paper-faithful preprocessing (disparity, camera FoV crop).
    """
    model.eval()

    val_cfg = cfg.get("val", cfg.get("eval", {}))
    dataset_cfg = cfg["dataset"]
    input_cfg = cfg.get("input", {})
    train_cfg = cfg.get("train", {})

    val_seqs = val_cfg.get("sequences", dataset_cfg.get("val_sequences", []))
    if not val_seqs:
        return {}

    batch_size = val_cfg.get("batch_size", 64)
    pos_threshold = dataset_cfg.get("pos_threshold", 25.0)
    ks = val_cfg.get("recall_ks", [1, 5, 10])
    top_k = val_cfg.get("top_k", max(ks))

    camera_hfov_deg = input_cfg.get("camera_hfov_deg", None)

    resize_cfg = input_cfg.get("resize", None)
    input_size = tuple(resize_cfg) if resize_cfg else None

    query_desc_list, db_desc_list = [], []
    query_pos_list, db_pos_list = [], []

    dataset_name = dataset_cfg.get("name", "vivid")

    for seq in val_seqs:
        try:
            if dataset_name == "kitti360":
                ds_q = KITTI360LC2Dataset(
                    root=dataset_cfg["root"], sequences=[seq],
                    modality="range",
                    range_cache_dir=dataset_cfg.get("range_cache_dir"),
                    subsample=10, input_size=input_size,
                    camera_hfov_deg=camera_hfov_deg,
                )
                ds_d = KITTI360LC2Dataset(
                    root=dataset_cfg["root"], sequences=[seq],
                    modality="depth",
                    depth_cache_dir=dataset_cfg.get("depth_cache_dir"),
                    subsample=10, input_size=input_size,
                )
            elif dataset_name == "helipr":
                ds_q = HeLiPRLC2Dataset(
                    root=dataset_cfg["root"], sequences=[seq],
                    modality="ouster",
                    ouster_cache_dir=dataset_cfg.get("ouster_cache_dir"),
                    subsample=10, input_size=input_size,
                )
                ds_d = HeLiPRLC2Dataset(
                    root=dataset_cfg["root"], sequences=[seq],
                    modality="velodyne",
                    velodyne_cache_dir=dataset_cfg.get("velodyne_cache_dir"),
                    subsample=10, input_size=input_size,
                )
            else:
                ds_q = VIVIDLC2Dataset(
                    root=dataset_cfg["root"], sequence=seq,
                    modality="range", subsample=10,
                    range_cache_dir=dataset_cfg.get("range_cache_dir"),
                    input_size=input_size,
                    camera_hfov_deg=camera_hfov_deg,
                )
                ds_d = VIVIDLC2Dataset(
                    root=dataset_cfg["root"], sequence=seq,
                    modality="depth",
                    depth_cache_dir=dataset_cfg.get("depth_cache_dir"),
                    subsample=10, input_size=input_size,
                )
        except FileNotFoundError:
            continue

        # Range queries
        range_descs = []
        loader_q = DataLoader(ds_q, batch_size=batch_size, shuffle=False, num_workers=2)
        for batch in loader_q:
            imgs = batch["image"].to(device)
            d = model.forward_single(imgs, is_range=True)
            range_descs.append(d.cpu().numpy())
        query_desc_list.append(np.concatenate(range_descs, axis=0))
        query_pos_list.append(ds_q.get_positions())

        # Depth/secondary database
        # HeLiPR: both sensors are range images → route both through encoder_r
        db_is_range = (dataset_name == "helipr")
        depth_descs = []
        loader = DataLoader(ds_d, batch_size=batch_size, shuffle=False, num_workers=2)
        for batch in loader:
            imgs = batch["image"].to(device)
            d = model.forward_single(imgs, is_range=db_is_range)
            depth_descs.append(d.cpu().numpy())
        db_desc_list.append(np.concatenate(depth_descs, axis=0))
        db_pos_list.append(ds_d.get_positions())

    if not query_desc_list:
        return {}

    query_desc = np.concatenate(query_desc_list, axis=0)
    db_desc = np.concatenate(db_desc_list, axis=0)
    query_pos = np.concatenate(query_pos_list, axis=0)
    db_pos = np.concatenate(db_pos_list, axis=0)

    return evaluate_retrieval(
        query_desc, db_desc, query_pos, db_pos,
        pos_threshold=pos_threshold, ks=ks, top_k=top_k,
    )


# -----------------------------------------------------------------
# Phase 1: Contrastive training epoch (GeM + psi-weighted loss)
# -----------------------------------------------------------------

def train_phase1_epoch(
    model: LC2Model,
    pair_dataset: LC2ContrastivePairDataset,
    criterion: LC2ContrastiveLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_size: int = 4,
    epoch: int = 0,
) -> float:
    """Train one epoch with contrastive loss (Phase 1)."""
    loader = DataLoader(
        pair_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )

    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in tqdm(loader, desc=f"P1 Epoch {epoch}"):
        img_i = batch["image_i"].to(device)
        img_j = batch["image_j"].to(device)
        is_range_i = batch["is_range_i"].to(device)
        is_range_j = batch["is_range_j"].to(device)
        psi = batch["psi"].float().to(device)

        desc_i = model(img_i, is_range_i)
        desc_j = model(img_j, is_range_j)

        loss = criterion(desc_i, desc_j, psi)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


# -----------------------------------------------------------------
# Phase 2: Triplet training epoch (NetVLAD + hard negative mining)
# -----------------------------------------------------------------

def train_phase2_epoch(
    model: LC2Model,
    miner: TripletMiner,
    transform,
    criterion: nn.TripletMarginLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: Dict,
    epoch: int,
    range_augmentation=None,
    depth_augmentation=None,
) -> float:
    """Train one epoch with triplet loss and hard negative mining (Phase 2)."""
    train_cfg = cfg["train"]
    input_cfg = cfg.get("input", {})
    batch_size = train_cfg.get("batch_size", 4)
    n_neg = train_cfg.get("n_neg", 5)
    margin = train_cfg.get("margin", 0.1)
    camera_hfov_deg = input_cfg.get("camera_hfov_deg", None)
    effective_fov = camera_hfov_deg if camera_hfov_deg is not None else 360.0

    # Mine hard triplets from current descriptors
    print(f"[Epoch {epoch}] Mining hard triplets...")
    t0 = time.time()
    descriptors = miner.compute_descriptors(
        model, transform, device,
        batch_size=train_cfg.get("cache_batch_size", 64),
        camera_hfov_deg=effective_fov,
    )
    triplets = miner.mine(descriptors, margin=margin)
    t1 = time.time()
    print(f"  Mined {len(triplets)} triplets ({t1 - t0:.1f}s)")

    if len(triplets) == 0:
        print("  No triplets found. Skipping epoch.")
        return 0.0

    triplet_dataset = LC2TripletDataset(
        triplets, miner.pool, transform,
        camera_hfov_deg=effective_fov,
        range_augmentation=range_augmentation,
        depth_augmentation=depth_augmentation,
    )
    triplet_loader = DataLoader(
        triplet_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )

    model.train()
    total_loss = 0.0
    n_batches = 0

    for images, is_range in tqdm(triplet_loader, desc=f"Epoch {epoch}"):
        B, T, C, H, W = images.shape
        images_flat = images.view(B * T, C, H, W).to(device)
        is_range_flat = is_range.view(B * T).to(device)

        desc = model(images_flat, is_range_flat)
        desc = desc.view(B, T, -1)

        q_desc = desc[:, 0]
        p_desc = desc[:, 1]
        n_desc = desc[:, 2:]

        loss = torch.tensor(0.0, device=device)
        for n in range(n_neg):
            loss = loss + criterion(q_desc, p_desc, n_desc[:, n])
        loss = loss / n_neg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


# -----------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------

def freeze_encoder(model: LC2Model) -> None:
    """Freeze all encoder parameters."""
    for p in model.encoder.parameters():
        p.requires_grad = False


def unfreeze_encoder_conv5(model: LC2Model) -> None:
    """Unfreeze conv5 layers in both encoder branches."""
    for branch in [model.encoder.encoder_d, model.encoder.encoder_r]:
        layers = list(branch.children())
        for layer in layers[-5:]:
            for p in layer.parameters():
                p.requires_grad = True


def unfreeze_encoder_all(model: LC2Model) -> None:
    """Unfreeze all encoder parameters."""
    for p in model.encoder.parameters():
        p.requires_grad = True


def make_optimizer(model, optim_name, lr, momentum, weight_decay):
    """Create optimizer with only trainable parameters."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    if optim_name == "sgd":
        return torch.optim.SGD(
            trainable, lr=lr, momentum=momentum, weight_decay=weight_decay,
        )
    elif optim_name == "adam":
        return torch.optim.Adam(trainable, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {optim_name}")


def count_trainable(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# -----------------------------------------------------------------
# Main
# -----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LC2 Paper-Faithful Training")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume encoder weights from checkpoint")
    parser.add_argument("--skip_phase1", action="store_true",
                        help="Skip Phase 1 contrastive pre-training")
    parser.add_argument("--gem_phase2", action="store_true",
                        help="Use GeM pooling for Phase 2 (skip NetVLAD)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_cfg = cfg["train"]
    model_cfg = cfg["model"]
    dataset_cfg = cfg["dataset"]
    input_cfg = cfg.get("input", {})

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_clusters = model_cfg.get("num_clusters", 16)
    encoder_dim = model_cfg.get("encoder_dim", 512)

    resize_cfg = input_cfg.get("resize", None)
    input_size = tuple(resize_cfg) if resize_cfg else None
    transform = get_transform(input_size)

    # Config
    camera_hfov_deg = input_cfg.get("camera_hfov_deg", None)  # None = no FoV crop

    # Build augmentation pipelines from config
    lidar_cfg = cfg.get("lidar", {})
    range_aug = None
    aug_cfg = lidar_cfg.get("augmentation", {})
    if aug_cfg:
        range_aug = RangeAugmentation.from_config(aug_cfg)

    depth_aug_cfg = input_cfg.get("depth_augmentation", {})
    depth_aug = None
    if depth_aug_cfg:
        depth_aug = DepthAugmentation.from_config(depth_aug_cfg)

    optim_name = train_cfg.get("optimizer", "sgd").lower()
    weight_decay = train_cfg.get("weight_decay", 1e-3)
    momentum = train_cfg.get("momentum", 0.9)
    lr_step = train_cfg.get("lr_step", 5)
    lr_gamma = train_cfg.get("lr_gamma", 0.5)
    patience = train_cfg.get("patience", 10)
    margin = train_cfg.get("margin", 0.1)

    # Phase 1 config
    phase1_epochs = train_cfg.get("phase1_epochs", 20)
    phase1_lr = train_cfg.get("phase1_lr", 1e-3)
    contrastive_tau = train_cfg.get("contrastive_tau", 1.0)
    n_crops = train_cfg.get("n_crops", 8)
    crop_fov_deg = train_cfg.get("crop_fov_deg", 90.0)
    scale_augment_pct = train_cfg.get("scale_augment_pct", 20.0)

    # Phase 2 config
    phase2_frozen_epochs = train_cfg.get("phase2_frozen_epochs", 5)
    phase2_frozen_lr = train_cfg.get("phase2_frozen_lr", 1e-3)
    phase2_epochs = train_cfg.get("phase2_epochs", 25)
    phase2_lr = train_cfg.get("phase2_lr", 1e-4)

    # ─── Model ───────────────────────────────────────────────────
    if args.resume:
        resume_path = Path(args.resume)
        print(f"Loading encoder weights from: {resume_path}")

        # Detect checkpoint format: our training format has 'phase' key
        ckpt_probe = torch.load(str(resume_path), map_location="cpu", weights_only=False)
        if isinstance(ckpt_probe, dict) and "phase" in ckpt_probe:
            # Our own training checkpoint — load state_dict directly
            print(f"  Detected training checkpoint (phase={ckpt_probe['phase']}, "
                  f"epoch={ckpt_probe['epoch']}, best={ckpt_probe.get('best_score', 0):.4f})")
            model = LC2Model(
                num_clusters=num_clusters, encoder_dim=encoder_dim,
                vladv2=False, pooling="gem",
            )
            model.load_state_dict(ckpt_probe["state_dict"], strict=False)
            model = model.to(device)
        else:
            # Original LC2 pretrained checkpoint
            model = LC2Model.from_checkpoint(args.resume, device=device, pooling="gem")
    else:
        model = LC2Model(
            num_clusters=num_clusters, encoder_dim=encoder_dim,
            vladv2=False, pooling="gem",
        )
        model = model.to(device)

    print(f"Model: {num_clusters} clusters, {encoder_dim}-dim")
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Preprocessing: always disparity, camera_hfov_deg={camera_hfov_deg}")

    best_recall = 0.0
    total_epoch = 0

    # ╔══════════════════════════════════════════════════════════════╗
    # ║  PHASE 1: GeM + Contrastive Loss with ψ (Section III.B.3)   ║
    # ╚══════════════════════════════════════════════════════════════╝

    if not args.skip_phase1 and phase1_epochs > 0:
        model.set_pooling("gem")
        print(f"\n{'='*60}")
        print(f"  PHASE 1: GeM + Contrastive (ψ-weighted)")
        print(f"  Epochs: {phase1_epochs}")
        print(f"  Loss: LC2ContrastiveLoss, tau={contrastive_tau}")
        print(f"  Range: {n_crops} FoV crops @ {crop_fov_deg}°")
        print(f"  Depth: disparity + scale augment ±{scale_augment_pct}%")
        print(f"  LR: {phase1_lr}")
        print(f"{'='*60}\n")

        # Build Phase 1 pool with FoV crops
        print("Building Phase 1 pool (range crops + depth)...")
        dataset_name = dataset_cfg.get("name", "vivid")
        if dataset_name == "kitti360":
            phase1_pool = build_kitti360_phase1_pool(
                root=dataset_cfg["root"],
                sequences=train_cfg.get("sequences", dataset_cfg["sequences"]),
                depth_cache_dir=dataset_cfg.get("depth_cache_dir"),
                range_cache_dir=dataset_cfg.get("range_cache_dir"),
                range_subsample=train_cfg.get("range_subsample", 10),
                depth_subsample=train_cfg.get("depth_subsample", 10),
                n_crops=n_crops,
                crop_fov_deg=crop_fov_deg,
                camera_hfov_deg=camera_hfov_deg if camera_hfov_deg else 90.0,
            )
        else:
            phase1_pool = build_vivid_phase1_pool(
                root=dataset_cfg["root"],
                sequences=train_cfg.get("sequences", dataset_cfg["sequences"]),
                depth_cache_dir=dataset_cfg.get("depth_cache_dir"),
                range_cache_dir=dataset_cfg.get("range_cache_dir"),
                range_subsample=train_cfg.get("range_subsample", 10),
                depth_subsample=train_cfg.get("depth_subsample", 10),
                n_crops=n_crops,
                crop_fov_deg=crop_fov_deg,
                camera_hfov_deg=camera_hfov_deg if camera_hfov_deg else 90.0,
            )
        n_range_p1 = sum(1 for r in phase1_pool.is_range if r)
        n_depth_p1 = len(phase1_pool) - n_range_p1
        print(f"  Phase 1 pool: {len(phase1_pool)} entries "
              f"({n_range_p1} range crops, {n_depth_p1} depth)")

        # Mine contrastive pairs with ψ
        # Default to 90° for ψ computation if camera_hfov_deg not set
        # (e.g. when range images are already in camera view)
        pair_miner = ContrastivePairMiner(
            pool=phase1_pool,
            max_range_m=50.0,
            camera_fov_deg=camera_hfov_deg if camera_hfov_deg else 90.0,
        )

        # Phase 1 dataset with scale augmentation
        phase1_dataset = LC2ContrastivePairDataset(
            pairs=pair_miner.pairs,
            pool=phase1_pool,
            transform=transform,
            scale_augment=True,
            max_scale_pct=scale_augment_pct,
            n_crops=n_crops,
            crop_fov_deg=crop_fov_deg,
            range_augmentation=range_aug,
            depth_augmentation=depth_aug,
        )

        # Phase 1: unfreeze conv5 for training
        unfreeze_encoder_conv5(model)
        # Also make GeM learnable
        for p in model.gem.parameters():
            p.requires_grad = True
        print(f"  Trainable params (conv5 + GeM): {count_trainable(model):,}")

        phase1_criterion = LC2ContrastiveLoss(tau=contrastive_tau)
        optimizer = make_optimizer(model, optim_name, phase1_lr, momentum, weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=lr_step, gamma=lr_gamma,
        )

        for epoch in range(phase1_epochs):
            t_epoch = time.time()

            avg_loss = train_phase1_epoch(
                model, phase1_dataset, phase1_criterion,
                optimizer, device,
                batch_size=train_cfg.get("batch_size", 4),
                epoch=total_epoch,
            )
            scheduler.step()

            # Validate (GeM descriptors)
            val_results = validate(model, cfg, device, transform)
            r1 = val_results.get("recall@1", 0.0)
            dt = time.time() - t_epoch
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"[P1 Epoch {epoch}] loss={avg_loss:.4f}, R@1={r1*100:.2f}%, "
                  f"lr={lr_now:.6f}, time={dt:.0f}s")

            is_best = r1 > best_recall
            if is_best:
                best_recall = r1

            checkpoint = {
                "epoch": total_epoch, "phase": 1,
                "state_dict": model.state_dict(),
                "best_score": best_recall,
                "optimizer": optimizer.state_dict(),
                "config": cfg,
            }
            torch.save(checkpoint, output_dir / f"p1_epoch_{epoch}.pth.tar")
            if is_best:
                torch.save(checkpoint, output_dir / "best.pth.tar")
                print(f"  ** New best: R@1={best_recall*100:.2f}%")

            total_epoch += 1

        print(f"\nPhase 1 complete. Best R@1: {best_recall*100:.2f}%")

    elif args.skip_phase1:
        print("\nSkipping Phase 1 (--skip_phase1 flag).")

    # ╔══════════════════════════════════════════════════════════════╗
    # ║  PHASE 2: Triplet Loss Fine-tuning (Section III.B.4)        ║
    # ╚══════════════════════════════════════════════════════════════╝

    use_gem_phase2 = args.gem_phase2

    if not use_gem_phase2:
        # Switch to NetVLAD (original paper pipeline)
        model.set_pooling("netvlad")
    else:
        # Keep GeM pooling (avoids NetVLAD mode collapse on range/depth data)
        model.set_pooling("gem")

    pooling_name = "GeM" if use_gem_phase2 else "NetVLAD"
    print(f"\n{'='*60}")
    print(f"  PHASE 2: {pooling_name} + Triplet Loss")
    print(f"  Frozen epochs: {phase2_frozen_epochs}, Frozen LR: {phase2_frozen_lr}")
    print(f"  Fine-tune epochs: {phase2_epochs}, Fine-tune LR: {phase2_lr}")
    print(f"  Loss: TripletMarginLoss, margin={margin}")
    fov_str = f"camera FoV crop ({camera_hfov_deg}°)" if camera_hfov_deg else "no FoV crop"
    print(f"  Range: {fov_str}")
    print(f"  Depth: disparity (always)")
    print(f"{'='*60}\n")

    # Build Phase 2 pool (no FoV crops — cropping done at load time)
    print("Building Phase 2 pool...")
    dataset_name = dataset_cfg.get("name", "vivid")
    if dataset_name == "kitti360":
        pool = build_kitti360_pool(
            root=dataset_cfg["root"],
            sequences=train_cfg.get("sequences", dataset_cfg["sequences"]),
            depth_cache_dir=dataset_cfg.get("depth_cache_dir"),
            range_cache_dir=dataset_cfg.get("range_cache_dir"),
            range_subsample=train_cfg.get("range_subsample", 10),
            depth_subsample=train_cfg.get("depth_subsample", 10),
        )
    elif dataset_name == "helipr":
        pool = build_helipr_pool(
            root=dataset_cfg["root"],
            sequences=train_cfg.get("sequences", dataset_cfg["sequences"]),
            ouster_cache_dir=dataset_cfg.get("ouster_cache_dir"),
            velodyne_cache_dir=dataset_cfg.get("velodyne_cache_dir"),
            ouster_subsample=train_cfg.get("ouster_subsample", 10),
            velodyne_subsample=train_cfg.get("velodyne_subsample", 10),
        )
    else:
        pool = build_vivid_pool(
            root=dataset_cfg["root"],
            sequences=train_cfg.get("sequences", dataset_cfg["sequences"]),
            depth_cache_dir=dataset_cfg.get("depth_cache_dir"),
            range_cache_dir=dataset_cfg.get("range_cache_dir"),
            range_subsample=train_cfg.get("range_subsample", 10),
            depth_subsample=train_cfg.get("depth_subsample", 10),
        )
    n_range = sum(1 for r in pool.is_range if r)
    n_depth = len(pool) - n_range
    print(f"  Pool: {len(pool)} entries ({n_range} range, {n_depth} depth)")

    if not use_gem_phase2:
        # Initialize NetVLAD centroids via K-means on current data.
        # Always re-init even when resuming — pretrained centroids from a different
        # dataset/resolution won't match the current feature distribution.
        freeze_encoder(model)
        init_netvlad_from_data(
            model, pool, transform, device,
            num_clusters=num_clusters,
            batch_size=train_cfg.get("cache_batch_size", 64),
            camera_hfov_deg=camera_hfov_deg if camera_hfov_deg else 360.0,
        )
        model = model.to(device)

    # Triplet miner
    miner = TripletMiner(
        pool=pool,
        pos_dist_thr=train_cfg.get("pos_dist_thr", 10.0),
        neg_dist_thr=train_cfg.get("neg_dist_thr", 25.0),
        n_neg=train_cfg.get("n_neg", 5),
    )
    print(f"  Miner: pos<{miner.pos_dist_thr}m, neg>={miner.neg_dist_thr}m")

    # Triplet loss
    triplet_criterion = nn.TripletMarginLoss(
        margin=margin, p=2, reduction="sum",
    )

    epochs_no_improve = 0

    # ─── Phase 2a: Frozen encoder, train NetVLAD only ────────────
    # (Skip Phase 2a when using GeM — no new pooling layer to warm up)
    if phase2_frozen_epochs > 0 and not use_gem_phase2:
        print(f"\n  Phase 2a: Frozen encoder, NetVLAD-only ({phase2_frozen_epochs} epochs)")
        freeze_encoder(model)
        print(f"  Trainable params (encoder frozen): {count_trainable(model):,}")

        optimizer = make_optimizer(model, optim_name, phase2_frozen_lr, momentum, weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=lr_step, gamma=lr_gamma,
        )

        for epoch in range(phase2_frozen_epochs):
            t_epoch = time.time()

            avg_loss = train_phase2_epoch(
                model, miner, transform, triplet_criterion,
                optimizer, device, cfg, total_epoch,
                range_augmentation=range_aug,
                depth_augmentation=depth_aug,
            )
            scheduler.step()

            val_results = validate(model, cfg, device, transform)
            r1 = val_results.get("recall@1", 0.0)
            dt = time.time() - t_epoch
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"[P2a Epoch {epoch}] loss={avg_loss:.4f}, R@1={r1*100:.2f}%, "
                  f"lr={lr_now:.6f}, time={dt:.0f}s")

            is_best = r1 > best_recall
            if is_best:
                best_recall = r1

            checkpoint = {
                "epoch": total_epoch, "phase": "2a",
                "state_dict": model.state_dict(),
                "best_score": best_recall,
                "optimizer": optimizer.state_dict(),
                "config": cfg,
            }
            torch.save(checkpoint, output_dir / f"p2a_epoch_{epoch}.pth.tar")
            if is_best:
                torch.save(checkpoint, output_dir / "best.pth.tar")
                print(f"  ** New best: R@1={best_recall*100:.2f}%")

            total_epoch += 1

    # ─── Phase 2b: Unfreeze conv5, fine-tune all ─────────────────
    print(f"\n  Phase 2b: Unfreeze conv5, fine-tune all ({phase2_epochs} epochs)")

    unfreeze_encoder_conv5(model)
    print(f"  Trainable params (conv5 unfrozen): {count_trainable(model):,}")

    optimizer = make_optimizer(model, optim_name, phase2_lr, momentum, weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=lr_step, gamma=lr_gamma,
    )

    for epoch in range(phase2_epochs):
        t_epoch = time.time()

        avg_loss = train_phase2_epoch(
            model, miner, transform, triplet_criterion,
            optimizer, device, cfg, total_epoch,
            range_augmentation=range_aug,
            depth_augmentation=depth_aug,
        )
        scheduler.step()

        val_results = validate(model, cfg, device, transform)
        r1 = val_results.get("recall@1", 0.0)
        dt = time.time() - t_epoch
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"[P2b Epoch {epoch}] loss={avg_loss:.4f}, R@1={r1*100:.2f}%, "
              f"lr={lr_now:.6f}, time={dt:.0f}s")

        is_best = r1 > best_recall
        if is_best:
            best_recall = r1
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        checkpoint = {
            "epoch": total_epoch, "phase": "2b",
            "state_dict": model.state_dict(),
            "best_score": best_recall,
            "optimizer": optimizer.state_dict(),
            "config": cfg,
        }
        torch.save(checkpoint, output_dir / f"p2b_epoch_{epoch}.pth.tar")
        if is_best:
            torch.save(checkpoint, output_dir / "best.pth.tar")
            print(f"  ** New best: R@1={best_recall*100:.2f}%")

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {total_epoch} "
                  f"(no improvement for {patience} epochs)")
            break

        total_epoch += 1

    print(f"\nTraining complete. Best R@1: {best_recall*100:.2f}%")
    print(f"Checkpoints saved to: {output_dir}")


if __name__ == "__main__":
    main()
