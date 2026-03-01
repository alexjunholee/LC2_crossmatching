"""HeLiPR dataset loader for LC2 cross-sensor place recognition.

Loads pre-computed range images from Ouster OS2-128 and Velodyne VLP-32C
sensors. Ouster is treated as "range" (encoder_r) and Velodyne as "depth"
(encoder_d) for cross-modal training, even though both are LiDAR range images.

Poses from ``LiDAR_GT/global_{sensor}_gt.txt`` provide UTM ground truth.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from lc2.data.transforms import (
    get_transform, range_to_normalized_disparity, normalize_disparity,
)


class HeLiPRLC2Dataset(Dataset):
    """HeLiPR dataset for LC2 training and evaluation.

    Loads pre-computed range images from cache directories,
    along with GT poses for ground truth matching.

    Cache directories contain per-scan ``.npy`` files named by timestamp.
    """

    def __init__(
        self,
        root: str,
        sequences: List[str],
        modality: str,
        ouster_cache_dir: Optional[str] = None,
        velodyne_cache_dir: Optional[str] = None,
        input_size: Optional[Tuple[int, int]] = None,
        subsample: int = 1,
    ) -> None:
        """
        Args:
            root: HeLiPR root directory.
            sequences: List of sequence names (e.g., ``["DCC04"]``).
            modality: ``"ouster"`` (treated as range/query) or
                ``"velodyne"`` (treated as depth/database).
            ouster_cache_dir: Directory with pre-computed Ouster range .npy files.
            velodyne_cache_dir: Directory with pre-computed Velodyne range .npy files.
            input_size: If provided, resize to (H, W).
            subsample: Load every N-th frame.
        """
        super().__init__()
        self.root = Path(root)
        self.modality = modality
        self.transform = get_transform(input_size)

        if modality == "ouster":
            sensor_name = "Ouster"
            if ouster_cache_dir is None:
                raise ValueError("ouster_cache_dir required for 'ouster' modality")
            cache_dir = Path(ouster_cache_dir)
        elif modality == "velodyne":
            sensor_name = "Velodyne"
            if velodyne_cache_dir is None:
                raise ValueError("velodyne_cache_dir required for 'velodyne' modality")
            cache_dir = Path(velodyne_cache_dir)
        else:
            raise ValueError(f"Unknown modality: {modality}. Use 'ouster' or 'velodyne'.")

        self.samples: List[Tuple[Path, str, int]] = []  # (npy_path, seq, timestamp)
        self.positions: List[np.ndarray] = []

        for seq in sequences:
            # Load GT poses
            gt_file = self.root / seq / "LiDAR_GT" / f"global_{sensor_name}_gt.txt"
            if not gt_file.exists():
                print(f"  Warning: GT file not found: {gt_file}")
                continue
            poses = self._load_gt_poses(gt_file)

            # Find available cache files
            seq_cache = cache_dir / seq
            if not seq_cache.exists():
                print(f"  Warning: cache dir not found: {seq_cache}")
                continue

            npy_files = {int(p.stem): p for p in seq_cache.glob("*.npy")}

            # Intersect with available poses
            common_ts = sorted(set(poses.keys()) & set(npy_files.keys()))

            if subsample > 1:
                common_ts = common_ts[::subsample]

            for ts in common_ts:
                self.samples.append((npy_files[ts], seq, ts))
                self.positions.append(poses[ts])

        self._positions_array = (
            np.array(self.positions, dtype=np.float64)
            if self.positions else np.zeros((0, 3))
        )

    @staticmethod
    def _load_gt_poses(gt_file: Path) -> Dict[int, np.ndarray]:
        """Load GT poses from HeLiPR global GT file.

        Format: ``timestamp utm_x utm_y utm_z qx qy qz qw``
        """
        poses: Dict[int, np.ndarray] = {}
        with open(gt_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                try:
                    ts = int(parts[0])
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                except (ValueError, IndexError):
                    continue
                poses[ts] = np.array([x, y, z], dtype=np.float64)
        return poses

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        npy_path, seq, ts = self.samples[idx]
        data = np.load(str(npy_path))

        if data.ndim > 2:
            data = data.squeeze()

        # Both ouster and velodyne are range images — use same preprocessing
        data = normalize_disparity(data)

        image = self.transform(data)
        position = torch.from_numpy(self._positions_array[idx].copy()).float()

        return {
            "image": image,
            "is_range": True,  # Both sensors are range → route all through encoder_r
            "position": position,
            "seq": seq,
            "frame_id": ts,
            "index": idx,
        }

    def get_positions(self) -> np.ndarray:
        return self._positions_array.copy()

    def get_paths(self) -> List[Path]:
        return [s[0] for s in self.samples]
