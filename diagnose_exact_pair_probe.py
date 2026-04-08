#!/usr/bin/env python3
"""
Exact-pair probe: compare encoder representations across checkpoints.

For each checkpoint, loads VIVID campus_day1 data and measures:
1. Encoder feature diversity (intra-modal cosine sim)
2. Cross-modal alignment (range-depth cosine sim for same location)
3. Retrieval: R@1 on exact pairs (same location, cross-modal)
4. Layer-wise feature statistics

Usage:
    python diagnose_exact_pair_probe.py
"""

import sys, torch, numpy as np
from pathlib import Path
from scipy.spatial.distance import pdist, squareform

sys.path.insert(0, str(Path(__file__).parent))

from lc2.model import LC2Model
from lc2.data.train_dataset import build_vivid_pool
from lc2.data.transforms import (
    get_transform, range_to_normalized_disparity,
    depth_to_normalized_disparity, crop_range_to_camera_fov, squeeze_depth,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_DIR = Path("/home/jhlee/ws_xloc/lc2pp/checkpoints/probe_compare")
CHECKPOINTS = {
    "pretrained":    CKPT_DIR / "pretrained.pth.tar",
    "old_best_948":  CKPT_DIR / "old_best_948.pth.tar",
    "new_best_841":  CKPT_DIR / "new_best_841.pth.tar",
}

CAMERA_HFOV = 85.9
FORWARD_COL_FRAC = 0.996
N_SAMPLES = 20  # per modality


def load_model(ckpt_path):
    """Load LC2 model from checkpoint."""
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    model = LC2Model(num_clusters=16, encoder_dim=512, vladv2=False, pooling="gem")

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
        # Strip DataParallel 'module.' prefix if present
        sd = {k.replace('.module.', '.'): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f"  Warning: {len(missing)} missing keys (e.g. {missing[:2]})")
        info = f"phase={ckpt.get('phase','?')}, epoch={ckpt.get('epoch','?')}, best={ckpt.get('best_score',0):.4f}"
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        info = "pretrained (original LC2)"
    else:
        # Try direct load (original LC2 format)
        try:
            model.load_state_dict(ckpt, strict=False)
            info = "direct state_dict"
        except Exception:
            # Original LC2 checkpoint format
            model = LC2Model.from_checkpoint(str(ckpt_path), device="cpu", pooling="gem")
            info = "LC2 original format"

    model = model.to(DEVICE).eval()
    return model, info


def prepare_sample(pool, idx, transform):
    """Load and preprocess a single sample."""
    data = np.load(str(pool.paths[idx]))
    if data.ndim > 2:
        data = squeeze_depth(data)

    if pool.is_range[idx]:
        data = crop_range_to_camera_fov(
            data, camera_hfov_deg=CAMERA_HFOV,
            forward_col_frac=FORWARD_COL_FRAC)
        data = range_to_normalized_disparity(data)
    else:
        data = depth_to_normalized_disparity(data)

    img = transform(data).unsqueeze(0).to(DEVICE)
    is_range = torch.tensor([pool.is_range[idx]], device=DEVICE)
    return img, is_range


def extract_features(model, pool, indices, transform):
    """Extract encoder features and GeM descriptors for given indices."""
    enc_feats = []  # (N, 512, H, W) -> GAP to (N, 512)
    gem_descs = []  # (N, 512)

    with torch.no_grad():
        for idx in indices:
            img, is_range = prepare_sample(pool, idx, transform)
            feat = model.encoder(img, is_range)  # (1, 512, H, W)
            desc = model(img, is_range)  # (1, 512) via GeM

            # GAP pooled encoder feature
            gap = feat.mean(dim=(2, 3)).squeeze()  # (512,)
            enc_feats.append(gap.cpu())
            gem_descs.append(desc.squeeze().cpu())

    return torch.stack(enc_feats), torch.stack(gem_descs)


def cosine_sim_matrix(feats):
    """Compute pairwise cosine similarity."""
    feats_n = feats / feats.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return (feats_n @ feats_n.T).numpy()


def find_exact_pairs(pool, n_pairs=20):
    """Find range-depth pairs at the same location (<5m)."""
    positions = np.array(pool.positions)
    is_range = np.array(pool.is_range)

    range_idx = np.where(is_range)[0]
    depth_idx = np.where(~is_range)[0]

    pairs = []
    for ri in range_idx:
        dists = np.linalg.norm(positions[depth_idx] - positions[ri], axis=1)
        best_di = depth_idx[np.argmin(dists)]
        if dists.min() < 5.0:
            pairs.append((ri, best_di, dists.min()))
        if len(pairs) >= n_pairs:
            break

    return pairs


def main():
    print("=" * 70)
    print("  Exact-Pair Probe: Comparing 3 Checkpoints")
    print("=" * 70)

    # Build pool
    pool = build_vivid_pool(
        root="/media/jhlee/EVO4TB/vivid_projects/data",
        sequences=["campus_day1"],
        depth_cache_dir="cache/depth/vivid",
        range_cache_dir="cache/range/vivid",
        range_subsample=10, depth_subsample=10,
    )
    print(f"Pool: {len(pool)} entries")

    transform = get_transform((480, 640))

    # Find exact pairs
    pairs = find_exact_pairs(pool, n_pairs=N_SAMPLES)
    print(f"Exact pairs found: {len(pairs)} (range-depth within 5m)")
    if not pairs:
        print("ERROR: No exact pairs found!")
        return

    range_indices = [p[0] for p in pairs]
    depth_indices = [p[1] for p in pairs]
    pair_dists = [p[2] for p in pairs]
    print(f"  Mean pair distance: {np.mean(pair_dists):.2f}m")

    # Also get some random samples for intra-modal diversity
    all_range = [i for i in range(len(pool)) if pool.is_range[i]][:N_SAMPLES]
    all_depth = [i for i in range(len(pool)) if not pool.is_range[i]][:N_SAMPLES]

    for name, ckpt_path in CHECKPOINTS.items():
        print(f"\n{'─' * 70}")
        print(f"  Checkpoint: {name}")
        print(f"{'─' * 70}")

        model, info = load_model(ckpt_path)
        print(f"  Info: {info}")

        # Extract features for exact pairs
        range_enc, range_gem = extract_features(model, pool, range_indices, transform)
        depth_enc, depth_gem = extract_features(model, pool, depth_indices, transform)

        # 1. Intra-modal diversity (encoder)
        range_sim = cosine_sim_matrix(range_enc)
        depth_sim = cosine_sim_matrix(depth_enc)
        # Off-diagonal mean
        n = len(range_sim)
        mask = ~np.eye(n, dtype=bool)
        range_intra = range_sim[mask].mean()
        depth_intra = depth_sim[mask].mean()
        print(f"\n  [Encoder] Intra-modal cosine sim (off-diag mean):")
        print(f"    Range: {range_intra:.4f}  (1.0 = collapsed)")
        print(f"    Depth: {depth_intra:.4f}")

        # 2. Cross-modal alignment for exact pairs (encoder)
        cross_sims_enc = []
        for i in range(len(pairs)):
            r = range_enc[i] / range_enc[i].norm().clamp(min=1e-8)
            d = depth_enc[i] / depth_enc[i].norm().clamp(min=1e-8)
            cross_sims_enc.append((r @ d).item())
        print(f"\n  [Encoder] Cross-modal cosine sim (exact pairs):")
        print(f"    Mean: {np.mean(cross_sims_enc):.4f}, Std: {np.std(cross_sims_enc):.4f}")

        # 3. Same for GeM descriptors
        range_sim_g = cosine_sim_matrix(range_gem)
        depth_sim_g = cosine_sim_matrix(depth_gem)
        range_intra_g = range_sim_g[mask].mean()
        depth_intra_g = depth_sim_g[mask].mean()
        print(f"\n  [GeM] Intra-modal cosine sim:")
        print(f"    Range: {range_intra_g:.4f}")
        print(f"    Depth: {depth_intra_g:.4f}")

        cross_sims_gem = []
        for i in range(len(pairs)):
            r = range_gem[i] / range_gem[i].norm().clamp(min=1e-8)
            d = depth_gem[i] / depth_gem[i].norm().clamp(min=1e-8)
            cross_sims_gem.append((r @ d).item())
        print(f"\n  [GeM] Cross-modal cosine sim (exact pairs):")
        print(f"    Mean: {np.mean(cross_sims_gem):.4f}, Std: {np.std(cross_sims_gem):.4f}")

        # 4. Retrieval: R@1 on exact pairs
        # range query → depth DB
        all_range_gem = range_gem / range_gem.norm(dim=1, keepdim=True).clamp(min=1e-8)
        all_depth_gem = depth_gem / depth_gem.norm(dim=1, keepdim=True).clamp(min=1e-8)
        sim_r2d = (all_range_gem @ all_depth_gem.T).numpy()
        r1_r2d = sum(1 for i in range(n) if np.argmax(sim_r2d[i]) == i) / n * 100
        # depth query → range DB
        sim_d2r = sim_r2d.T
        r1_d2r = sum(1 for i in range(n) if np.argmax(sim_d2r[i]) == i) / n * 100
        print(f"\n  [GeM] Exact-pair retrieval R@1:")
        print(f"    Range→Depth: {r1_r2d:.1f}%")
        print(f"    Depth→Range: {r1_d2r:.1f}%")

        # 5. Feature statistics
        print(f"\n  [Encoder] Feature stats:")
        print(f"    Range: mean={range_enc.mean():.6f}, std={range_enc.std():.6f}, norm={range_enc.norm(dim=1).mean():.4f}")
        print(f"    Depth: mean={depth_enc.mean():.6f}, std={depth_enc.std():.6f}, norm={depth_enc.norm(dim=1).mean():.4f}")

        del model
        torch.cuda.empty_cache()

    print(f"\n{'=' * 70}")
    print("  Probe complete.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
