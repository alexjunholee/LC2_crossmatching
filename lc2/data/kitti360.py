"""KITTI-360 dataset loader for LC2 cross-modal place recognition.

Provides synchronized access to pre-computed depth maps (from camera via DA3)
and range images (from LiDAR via spherical projection) for LC2 evaluation
on the KITTI-360 autonomous driving dataset.

All ``.npy`` files are single-channel ``(H, W)`` float32, loaded as-is and
passed through the LC2 transform (single-channel -> 3ch repeat -> Normalize).

Poses from ``data_poses/{sequence}/poses.txt`` provide SE(3) ground truth.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from lc2.data.transforms import (
    get_transform, squeeze_depth,
    depth_to_normalized_disparity, range_to_normalized_disparity, normalize_disparity,
    crop_range_to_camera_fov,
)


# KITTI-360 sequence IDs (drive numbers without the 2013_05_28_drive_ prefix)
KITTI360_SEQUENCES = ["0000", "0002", "0003", "0004", "0005", "0006", "0007", "0009", "0010"]


def _seq_to_dirname(seq_id: str) -> str:
    """Convert short sequence ID to full KITTI-360 directory name."""
    return f"2013_05_28_drive_{seq_id}_sync"


class KITTI360LC2Dataset(Dataset):
    """KITTI-360 dataset for LC2 evaluation.

    Loads pre-computed range or depth images from cache directories,
    along with poses for ground truth matching.

    Cache directories contain per-frame ``.npy`` files named by the
    zero-padded 10-digit frame index (e.g., ``0000000042.npy``).
    """

    def __init__(
        self,
        root: str,
        sequences: List[str],
        modality: str,
        depth_cache_dir: Optional[str] = None,
        range_cache_dir: Optional[str] = None,
        input_size: Optional[Tuple[int, int]] = None,
        subsample: int = 1,
        camera_hfov_deg: Optional[float] = None,
    ) -> None:
        """
        Args:
            root: KITTI-360 root directory.
            sequences: List of short sequence IDs (e.g., ``["0000", "0002"]``).
            modality: Either ``"depth"`` or ``"range"``.
            depth_cache_dir: Directory with pre-computed depth ``.npy`` files.
            range_cache_dir: Directory with pre-computed range ``.npy`` files.
            input_size: If provided, resize to (H, W). If None, keep native resolution.
            subsample: Load every N-th frame.
            camera_hfov_deg: If set, crop panoramic range images to this FoV.
        """
        super().__init__()
        self.root = Path(root)
        self.modality = modality
        self.camera_hfov_deg = camera_hfov_deg
        self.transform = get_transform(input_size)

        if modality == "depth" and depth_cache_dir is None:
            raise ValueError("depth_cache_dir required for 'depth' modality")
        if modality == "range" and range_cache_dir is None:
            raise ValueError("range_cache_dir required for 'range' modality")

        cache_dir = Path(depth_cache_dir) if modality == "depth" else Path(range_cache_dir)

        # Build sample list across all sequences
        self.samples: List[Tuple[Path, str, int]] = []  # (npy_path, seq_id, frame_id)
        self.positions: List[np.ndarray] = []  # (x, y, z) from pose translation

        for seq_id in sequences:
            dirname = _seq_to_dirname(seq_id)

            # Load poses
            pose_file = self.root / "data_poses" / dirname / "poses.txt"
            if not pose_file.exists():
                continue
            poses = self._load_poses(pose_file)

            # Find available cache files for this sequence
            seq_cache = cache_dir / seq_id
            if not seq_cache.exists():
                seq_cache = cache_dir / dirname
            if not seq_cache.exists():
                continue

            npy_files = {int(p.stem): p for p in seq_cache.glob("*.npy")}

            # Intersect with available poses
            common_frames = sorted(set(poses.keys()) & set(npy_files.keys()))

            if subsample > 1:
                common_frames = common_frames[::subsample]

            for frame_id in common_frames:
                self.samples.append((npy_files[frame_id], seq_id, frame_id))
                T = poses[frame_id]
                self.positions.append(T[:3, 3])

        self._positions_array = np.array(self.positions, dtype=np.float64) if self.positions else np.zeros((0, 3))

    @staticmethod
    def _load_poses(pose_file: Path) -> Dict[int, np.ndarray]:
        """Load poses from KITTI-360 poses.txt.

        Format: ``frame_id T11 T12 T13 T14 T21 T22 T23 T24 T31 T32 T33 T34``
        """
        poses: Dict[int, np.ndarray] = {}
        with open(pose_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 13:
                    continue
                try:
                    frame_id = int(parts[0])
                    values = [float(v) for v in parts[1:13]]
                except ValueError:
                    continue

                T = np.eye(4, dtype=np.float64)
                T[:3, :] = np.array(values).reshape(3, 4)
                poses[frame_id] = T

        return poses

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load a single sample.

        The numpy array is loaded as-is (single-channel float32) and passed
        through the LC2 transform: ToTensor -> Repeat3ch -> Normalize([0.5]*3).
        """
        npy_path, seq_id, frame_id = self.samples[idx]
        data = np.load(str(npy_path))

        # Ensure 2D single-channel
        if data.ndim > 2:
            data = squeeze_depth(data)

        if self.modality == "range":
            if self.camera_hfov_deg is not None:
                data = crop_range_to_camera_fov(data, camera_hfov_deg=self.camera_hfov_deg)
            data = normalize_disparity(data)
        elif self.modality == "depth":
            data = depth_to_normalized_disparity(data)

        image = self.transform(data)
        position = torch.from_numpy(self._positions_array[idx].copy()).float()

        return {
            "image": image,
            "is_range": self.modality == "range",
            "position": position,
            "seq": seq_id,
            "frame_id": frame_id,
            "index": idx,
        }

    def get_positions(self) -> np.ndarray:
        """Return all positions as numpy array of shape ``(N, 3)``."""
        return self._positions_array.copy()

    def get_paths(self) -> List[Path]:
        """Return all sample file paths."""
        return [s[0] for s in self.samples]
