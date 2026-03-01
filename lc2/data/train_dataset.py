"""Cross-modal training datasets for LC2 two-phase training.

Phase 1 — Contrastive pre-training:
    Manages pairs of depth/range images with degree of similarity ψ.
    Range images are divided into 8 FoV-masked overlapping crops.
    Depth images receive random scale augmentation.

Phase 2 — Triplet fine-tuning:
    Hard-negative-mined triplets (query, positive, negatives) with
    cross-modal positive matching. Range images are cropped to camera FoV.
"""

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from lc2.data.transforms import (
    get_transform,
    squeeze_depth,
    depth_to_normalized_disparity,
    range_to_normalized_disparity,
    scale_augment_disparity,
    depth_to_disparity,
    normalize_disparity,
    crop_range_panoramic,
    crop_range_to_camera_fov,
)
from lc2.fov_overlap import compute_headings, build_psi_pairs


# ─────────────────────────────────────────────────────────────────
# Phase 1: Contrastive pair dataset
# ─────────────────────────────────────────────────────────────────

class Phase1Pool:
    """Pool of all images for Phase 1 with FoV metadata.

    Each entry has:
        - ``path``: Path to ``.npy`` file.
        - ``position``: (2,) UTM coordinates.
        - ``heading``: heading angle (radians).
        - ``is_range``: bool.
        - ``fov_deg``: horizontal FoV in degrees.
        - ``crop_idx``: -1 for depth, 0-7 for range crops.
        - ``parent_idx``: index into the original (uncropped) range file list.
    """

    def __init__(self) -> None:
        self.paths: List[Path] = []
        self.positions: List[np.ndarray] = []
        self.headings: List[float] = []
        self.is_range: List[bool] = []
        self.fov_deg: List[float] = []
        self.crop_idx: List[int] = []

    def add(
        self,
        path: Path,
        position: np.ndarray,
        heading: float,
        is_range: bool,
        fov_deg: float,
        crop_idx: int = -1,
    ) -> None:
        self.paths.append(path)
        self.positions.append(position[:2].astype(np.float64))
        self.headings.append(heading)
        self.is_range.append(is_range)
        self.fov_deg.append(fov_deg)
        self.crop_idx.append(crop_idx)

    def __len__(self) -> int:
        return len(self.paths)

    def position_array(self) -> np.ndarray:
        return np.array(self.positions, dtype=np.float64)


class ContrastivePairMiner:
    """Mines pairs with degree of similarity ψ for Phase 1 contrastive training.

    For each pair (i, j):
        - ψ > 0: overlapping views → attract
        - ψ ≈ 0: non-overlapping → repel

    Uses sparse ψ computation — no dense N×N matrix is built.
    """

    def __init__(
        self,
        pool: Phase1Pool,
        max_range_m: float = 50.0,
        camera_fov_deg: float = 90.0,
        lidar_range_m: float = 50.0,
        camera_range_m: float = 50.0,
        psi_grid_res: int = 64,
    ) -> None:
        self.pool = pool
        print("Computing pairwise ψ pairs (sparse)...")

        positions = pool.position_array()
        headings = np.array(pool.headings)
        fovs = np.array(pool.fov_deg)
        max_ranges = np.array(
            [lidar_range_m if r else camera_range_m for r in pool.is_range]
        )

        # Get sparse (i, j, ψ) tuples directly — no dense matrix
        positive_pairs = build_psi_pairs(
            positions=positions,
            headings=headings,
            fovs_deg=fovs,
            max_ranges=max_ranges,
            max_dist=max_range_m,
        )

        # Build set of positive pair indices for fast negative sampling
        N = len(pool)
        pos_set = set()
        for i, j, _ in positive_pairs:
            pos_set.add((min(i, j), max(i, j)))

        self.pairs: List[Tuple[int, int, float]] = list(positive_pairs)

        # Sample negative pairs (ψ = 0) for the repulsion term
        n_neg_pairs = min(len(positive_pairs), N * 5)
        neg_count = 0
        max_attempts = n_neg_pairs * 3
        for _ in range(max_attempts):
            if neg_count >= n_neg_pairs:
                break
            i = random.randint(0, N - 1)
            j = random.randint(0, N - 1)
            if i != j and (min(i, j), max(i, j)) not in pos_set:
                self.pairs.append((i, j, 0.0))
                neg_count += 1

        n_pos = len(positive_pairs)
        n_neg = len(self.pairs) - n_pos
        print(f"  Pairs: {len(self.pairs)} (positive: {n_pos}, negative: {n_neg})")


class LC2ContrastivePairDataset(Dataset):
    """Dataset that yields pairs with ψ for Phase 1 contrastive training.

    Each ``__getitem__`` returns:
        - image_i, image_j: transformed tensors (3, H, W)
        - is_range_i, is_range_j: booleans
        - psi: degree of similarity ∈ [0, 1]
    """

    def __init__(
        self,
        pairs: List[Tuple[int, int, float]],
        pool: Phase1Pool,
        transform,
        scale_augment: bool = True,
        max_scale_pct: float = 20.0,
        n_crops: int = 8,
        crop_fov_deg: float = 90.0,
        range_augmentation=None,
        depth_augmentation=None,
    ) -> None:
        self.pairs = pairs
        self.pool = pool
        self.transform = transform
        self.scale_augment = scale_augment
        self.max_scale_pct = max_scale_pct
        self.n_crops = n_crops
        self.crop_fov_deg = crop_fov_deg
        self.range_augmentation = range_augmentation
        self.depth_augmentation = depth_augmentation

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_image(self, idx: int) -> Tuple[torch.Tensor, bool]:
        """Load and preprocess a single image from the pool."""
        data = np.load(str(self.pool.paths[idx]))
        if data.ndim > 2:
            data = squeeze_depth(data)

        is_range = self.pool.is_range[idx]
        crop_idx = self.pool.crop_idx[idx]

        if is_range:
            # Range image: apply crop (Phase 1 augmentation)
            # Skip crop for n_crops=1 (camera-view range, not panoramic)
            if crop_idx >= 0 and self.n_crops > 1:
                data = crop_range_panoramic(
                    data, crop_idx,
                    n_crops=self.n_crops,
                    crop_fov_deg=self.crop_fov_deg,
                )
            # Apply range augmentation (after crop, before transform)
            if self.range_augmentation is not None:
                data = self.range_augmentation(data)
            data = normalize_disparity(data)
        else:
            # Depth: convert to disparity (paper Section III.A.2)
            # + optional scale augmentation (paper Section III.C.2)
            disp = depth_to_disparity(data)
            if self.scale_augment:
                disp = scale_augment_disparity(disp, self.max_scale_pct)
            data = normalize_disparity(disp)
            # Apply depth augmentation (after disparity normalization, before transform)
            if self.depth_augmentation is not None:
                data = self.depth_augmentation(data)

        img = self.transform(data)
        return img, is_range

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        i, j, psi = self.pairs[idx]

        img_i, is_range_i = self._load_image(i)
        img_j, is_range_j = self._load_image(j)

        return {
            "image_i": img_i,
            "image_j": img_j,
            "is_range_i": is_range_i,
            "is_range_j": is_range_j,
            "psi": psi,
        }


# ─────────────────────────────────────────────────────────────────
# Phase 2: Triplet dataset (kept from original implementation)
# ─────────────────────────────────────────────────────────────────

class ImagePool:
    """Flat pool of all images (both modalities) for Phase 2.

    Each entry has:
        - ``path``: Path to ``.npy`` file
        - ``position``: (2,) UTM coordinates
        - ``is_range``: bool indicating modality
    """

    def __init__(self, all_range_preprocess: bool = False,
                 forward_all_as_range: bool = False) -> None:
        self.paths: List[Path] = []
        self.positions: List[np.ndarray] = []
        self.is_range: List[bool] = []
        self.all_range_preprocess = all_range_preprocess
        self.forward_all_as_range = forward_all_as_range

    def add(self, path: Path, position: np.ndarray, is_range: bool) -> None:
        self.paths.append(path)
        self.positions.append(position[:2].astype(np.float64))
        self.is_range.append(is_range)

    def __len__(self) -> int:
        return len(self.paths)

    def position_array(self) -> np.ndarray:
        return np.array(self.positions, dtype=np.float64)


class TripletMiner:
    """Mines hard cross-modal triplets for Phase 2 training.

    Cross-modal only: positive must be different modality from query.
    """

    def __init__(
        self,
        pool: ImagePool,
        pos_dist_thr: float = 10.0,
        neg_dist_thr: float = 25.0,
        n_neg: int = 5,
    ) -> None:
        self.pool = pool
        self.pos_dist_thr = pos_dist_thr
        self.neg_dist_thr = neg_dist_thr
        self.n_neg = n_neg

        pos = pool.position_array()
        diff = pos[:, None, :] - pos[None, :, :]
        self.dist_matrix = np.sqrt((diff ** 2).sum(axis=2))

        # Cross-modal positive and negative pools
        self.positive_pool: List[List[int]] = []
        self.negative_pool: List[List[int]] = []
        is_range_arr = np.array(pool.is_range)
        N = len(pool)
        for i in range(N):
            pos_mask = self.dist_matrix[i] < pos_dist_thr
            neg_mask = self.dist_matrix[i] >= neg_dist_thr
            pos_mask[i] = False
            cross_modal_mask = is_range_arr != is_range_arr[i]
            pos_mask = pos_mask & cross_modal_mask
            self.positive_pool.append(np.where(pos_mask)[0].tolist())
            self.negative_pool.append(np.where(neg_mask)[0].tolist())

    @torch.no_grad()
    def compute_descriptors(
        self,
        model: torch.nn.Module,
        transform,
        device: torch.device,
        batch_size: int = 64,
        camera_hfov_deg: float = 90.0,
    ) -> np.ndarray:
        """Extract descriptors for all images in the pool.

        Phase 2: range images are cropped to camera FoV before encoding.
        Both modalities are always converted to disparity.
        """
        model.eval()
        all_desc = []

        indices = list(range(len(self.pool)))
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            images = []
            is_range_flags = []
            for idx in batch_idx:
                data = np.load(str(self.pool.paths[idx]))
                if data.ndim > 2:
                    data = squeeze_depth(data)

                if self.pool.is_range[idx]:
                    if camera_hfov_deg < 360.0:
                        data = crop_range_to_camera_fov(data, camera_hfov_deg=camera_hfov_deg)
                    data = normalize_disparity(data)
                elif self.pool.all_range_preprocess:
                    # HeLiPR: both modalities are range images
                    data = normalize_disparity(data)
                else:
                    data = depth_to_normalized_disparity(data)

                img = transform(data)
                images.append(img)
                is_range_flags.append(self.pool.is_range[idx])

            images_t = torch.stack(images).to(device)
            # HeLiPR: both sensors are range → route all through encoder_r
            if self.pool.forward_all_as_range:
                is_range_t = torch.ones(len(is_range_flags), dtype=torch.bool, device=device)
            else:
                is_range_t = torch.tensor(is_range_flags, device=device)
            desc = model(images_t, is_range_t)
            all_desc.append(desc.cpu().numpy())

        return np.concatenate(all_desc, axis=0)

    def mine(
        self,
        descriptors: np.ndarray,
        margin: float = 0.3162,
    ) -> List[Tuple[int, int, List[int]]]:
        """Mine hard triplets from precomputed descriptors."""
        N = len(self.pool)
        sim_matrix = descriptors @ descriptors.T
        desc_dist = 2.0 - 2.0 * sim_matrix

        triplets: List[Tuple[int, int, List[int]]] = []

        for q in range(N):
            pos_candidates = self.positive_pool[q]
            neg_candidates = self.negative_pool[q]

            if len(pos_candidates) == 0 or len(neg_candidates) < self.n_neg:
                continue

            pos_dists = desc_dist[q, pos_candidates]
            p_idx = pos_candidates[np.argmin(pos_dists)]
            d_pos = desc_dist[q, p_idx]

            neg_dists = desc_dist[q, neg_candidates]
            sorted_neg = np.argsort(neg_dists)

            hard_negs = []
            for ni in sorted_neg:
                if len(hard_negs) >= self.n_neg:
                    break
                if neg_dists[ni] < d_pos + margin:
                    hard_negs.append(neg_candidates[ni])

            if len(hard_negs) < self.n_neg:
                for ni in sorted_neg:
                    if len(hard_negs) >= self.n_neg:
                        break
                    nidx = neg_candidates[ni]
                    if nidx not in hard_negs:
                        hard_negs.append(nidx)

            if len(hard_negs) >= self.n_neg:
                triplets.append((q, p_idx, hard_negs[: self.n_neg]))

        return triplets


class LC2TripletDataset(Dataset):
    """Dataset that yields pre-mined triplets for Phase 2.

    Each ``__getitem__`` returns stacked tensor of
    [query, positive, neg_1, ..., neg_K] and their modality flags.
    """

    def __init__(
        self,
        triplets: List[Tuple[int, int, List[int]]],
        pool: ImagePool,
        transform,
        camera_hfov_deg: float = 90.0,
        range_augmentation=None,
        depth_augmentation=None,
    ) -> None:
        self.triplets = triplets
        self.pool = pool
        self.transform = transform
        self.camera_hfov_deg = camera_hfov_deg
        self.range_augmentation = range_augmentation
        self.depth_augmentation = depth_augmentation

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        q_idx, p_idx, n_idxs = self.triplets[idx]
        all_idx = [q_idx, p_idx] + n_idxs

        images = []
        is_range = []
        for i in all_idx:
            data = np.load(str(self.pool.paths[i]))
            if data.ndim > 2:
                data = squeeze_depth(data)

            if self.pool.is_range[i]:
                # Range: crop to camera FoV (paper Section III.B.4)
                if self.camera_hfov_deg < 360.0:
                    data = crop_range_to_camera_fov(data, camera_hfov_deg=self.camera_hfov_deg)
                if self.range_augmentation is not None:
                    data = self.range_augmentation(data)
                data = normalize_disparity(data)
            elif self.pool.all_range_preprocess:
                # HeLiPR: both modalities are range images
                data = normalize_disparity(data)
            else:
                data = depth_to_normalized_disparity(data)
                if self.depth_augmentation is not None:
                    data = self.depth_augmentation(data)

            img = self.transform(data)
            images.append(img)
            is_range.append(self.pool.is_range[i])

        # HeLiPR: both sensors are range → route all through encoder_r
        if self.pool.forward_all_as_range:
            is_range_t = torch.ones(len(is_range), dtype=torch.bool)
        else:
            is_range_t = torch.tensor(is_range)
        return torch.stack(images), is_range_t


# ─────────────────────────────────────────────────────────────────
# Pool builders
# ─────────────────────────────────────────────────────────────────

def build_kitti360_phase1_pool(
    root: str,
    sequences: List[str],
    depth_cache_dir: Optional[str] = None,
    range_cache_dir: Optional[str] = None,
    range_subsample: int = 10,
    depth_subsample: int = 10,
    n_crops: int = 8,
    crop_fov_deg: float = 90.0,
    camera_hfov_deg: float = 90.0,
) -> Phase1Pool:
    """Build a Phase1Pool from KITTI-360 sequences."""
    from lc2.data.kitti360 import KITTI360LC2Dataset
    from lc2.fov_overlap import compute_headings

    pool = Phase1Pool()
    crop_step_deg = 360.0 / n_crops

    # Range images → 8 crops each
    try:
        ds_r = KITTI360LC2Dataset(
            root=root, sequences=sequences, modality="range",
            range_cache_dir=range_cache_dir, subsample=range_subsample,
        )
        positions = ds_r.get_positions()
        headings = compute_headings(positions)

        for i in range(len(ds_r)):
            for crop_k in range(n_crops):
                crop_heading = headings[i] + np.radians(crop_k * crop_step_deg)
                pool.add(
                    path=ds_r.samples[i][0],
                    position=positions[i],
                    heading=crop_heading,
                    is_range=True,
                    fov_deg=crop_fov_deg,
                    crop_idx=crop_k,
                )
    except FileNotFoundError as e:
        print(f"  Warning: range data not found: {e}")

    # Depth images
    if depth_cache_dir:
        try:
            ds_d = KITTI360LC2Dataset(
                root=root, sequences=sequences, modality="depth",
                depth_cache_dir=depth_cache_dir, subsample=depth_subsample,
            )
            positions_d = ds_d.get_positions()
            headings_d = compute_headings(positions_d)

            for i in range(len(ds_d)):
                pool.add(
                    path=ds_d.samples[i][0],
                    position=positions_d[i],
                    heading=headings_d[i],
                    is_range=False,
                    fov_deg=camera_hfov_deg,
                    crop_idx=-1,
                )
        except FileNotFoundError as e:
            print(f"  Warning: depth data not found: {e}")

    return pool


def build_kitti360_pool(
    root: str,
    sequences: List[str],
    depth_cache_dir: Optional[str] = None,
    range_cache_dir: Optional[str] = None,
    range_subsample: int = 10,
    depth_subsample: int = 10,
) -> ImagePool:
    """Build an ImagePool from KITTI-360 sequences for Phase 2."""
    from lc2.data.kitti360 import KITTI360LC2Dataset

    pool = ImagePool()

    try:
        ds_r = KITTI360LC2Dataset(
            root=root, sequences=sequences, modality="range",
            range_cache_dir=range_cache_dir, subsample=range_subsample,
        )
        for i in range(len(ds_r)):
            pool.add(ds_r.samples[i][0], ds_r.positions[i], is_range=True)
    except FileNotFoundError as e:
        print(f"  Warning: range data not found: {e}")

    if depth_cache_dir:
        try:
            ds_d = KITTI360LC2Dataset(
                root=root, sequences=sequences, modality="depth",
                depth_cache_dir=depth_cache_dir, subsample=depth_subsample,
            )
            for i in range(len(ds_d)):
                pool.add(ds_d.samples[i][0], ds_d.positions[i], is_range=False)
        except FileNotFoundError as e:
            print(f"  Warning: depth data not found: {e}")

    return pool


def build_vivid_phase1_pool(
    root: str,
    sequences: List[str],
    depth_cache_dir: Optional[str] = None,
    range_cache_dir: Optional[str] = None,
    range_subsample: int = 10,
    depth_subsample: int = 10,
    n_crops: int = 8,
    crop_fov_deg: float = 90.0,
    camera_hfov_deg: float = 90.0,
) -> Phase1Pool:
    """Build a Phase1Pool from VIVID sequences with range crops and FoV metadata.

    Range images are expanded into ``n_crops`` overlapping FoV crops.
    Depth images keep their native camera FoV.

    Args:
        root: VIVID data root.
        sequences: List of sequence names.
        depth_cache_dir: Directory with pre-computed depth .npy files.
        range_subsample: Use every N-th range frame.
        depth_subsample: Use every N-th depth frame.
        n_crops: Number of FoV crops per range image (default 8).
        crop_fov_deg: Angular width of each crop (default 90°).
        camera_hfov_deg: Camera horizontal FoV (default 90°).
    """
    from lc2.data.vivid import VIVIDLC2Dataset

    pool = Phase1Pool()
    crop_step_deg = 360.0 / n_crops  # 45° for 8 crops

    for seq in sequences:
        # Range images → 8 crops each
        try:
            ds_r = VIVIDLC2Dataset(
                root=root, sequence=seq, modality="range",
                range_cache_dir=range_cache_dir,
                subsample=range_subsample,
            )
            positions = ds_r.get_positions()
            headings = compute_headings(positions)

            for i in range(len(ds_r)):
                for crop_k in range(n_crops):
                    # Each crop's heading = vehicle heading + crop offset
                    crop_heading = headings[i] + np.radians(crop_k * crop_step_deg)
                    pool.add(
                        path=ds_r.samples[i],
                        position=positions[i],
                        heading=crop_heading,
                        is_range=True,
                        fov_deg=crop_fov_deg,
                        crop_idx=crop_k,
                    )
        except FileNotFoundError:
            pass

        # Depth images (single, camera FoV)
        if depth_cache_dir:
            try:
                ds_d = VIVIDLC2Dataset(
                    root=root, sequence=seq, modality="depth",
                    depth_cache_dir=depth_cache_dir,
                    subsample=depth_subsample,
                )
                positions_d = ds_d.get_positions()
                headings_d = compute_headings(positions_d)

                for i in range(len(ds_d)):
                    pool.add(
                        path=ds_d.samples[i],
                        position=positions_d[i],
                        heading=headings_d[i],
                        is_range=False,
                        fov_deg=camera_hfov_deg,
                        crop_idx=-1,
                    )
            except FileNotFoundError:
                pass

    return pool


def build_helipr_pool(
    root: str,
    sequences: List[str],
    ouster_cache_dir: Optional[str] = None,
    velodyne_cache_dir: Optional[str] = None,
    ouster_subsample: int = 10,
    velodyne_subsample: int = 10,
) -> ImagePool:
    """Build an ImagePool from HeLiPR sequences for Phase 2.

    Ouster is treated as "range" (is_range=True) and Velodyne as "depth"
    (is_range=False) for cross-modal triplet mining.
    Both are range images so use normalize_disparity for both.
    """
    from lc2.data.helipr import HeLiPRLC2Dataset

    pool = ImagePool(all_range_preprocess=True, forward_all_as_range=True)

    if ouster_cache_dir:
        try:
            ds_o = HeLiPRLC2Dataset(
                root=root, sequences=sequences, modality="ouster",
                ouster_cache_dir=ouster_cache_dir,
                subsample=ouster_subsample,
            )
            for i in range(len(ds_o)):
                pool.add(ds_o.samples[i][0], ds_o.positions[i], is_range=True)
        except FileNotFoundError as e:
            print(f"  Warning: Ouster data not found: {e}")

    if velodyne_cache_dir:
        try:
            ds_v = HeLiPRLC2Dataset(
                root=root, sequences=sequences, modality="velodyne",
                velodyne_cache_dir=velodyne_cache_dir,
                subsample=velodyne_subsample,
            )
            for i in range(len(ds_v)):
                pool.add(ds_v.samples[i][0], ds_v.positions[i], is_range=False)
        except FileNotFoundError as e:
            print(f"  Warning: Velodyne data not found: {e}")

    return pool


def build_vivid_pool(
    root: str,
    sequences: List[str],
    depth_cache_dir: Optional[str] = None,
    range_cache_dir: Optional[str] = None,
    range_subsample: int = 10,
    depth_subsample: int = 10,
) -> ImagePool:
    """Build an ImagePool from VIVID sequences for Phase 2.

    Loads both range and depth images from each sequence.
    Range images are NOT cropped here — cropping happens in the dataset/miner.
    """
    from lc2.data.vivid import VIVIDLC2Dataset

    pool = ImagePool()

    for seq in sequences:
        try:
            ds_r = VIVIDLC2Dataset(
                root=root, sequence=seq, modality="range",
                range_cache_dir=range_cache_dir,
                subsample=range_subsample,
            )
            for i in range(len(ds_r)):
                pool.add(ds_r.samples[i], ds_r.positions[i], is_range=True)
        except FileNotFoundError:
            pass

        if depth_cache_dir:
            try:
                ds_d = VIVIDLC2Dataset(
                    root=root, sequence=seq, modality="depth",
                    depth_cache_dir=depth_cache_dir,
                    subsample=depth_subsample,
                )
                for i in range(len(ds_d)):
                    pool.add(ds_d.samples[i], ds_d.positions[i], is_range=False)
            except FileNotFoundError:
                pass

    return pool
