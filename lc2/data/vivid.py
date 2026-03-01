"""VIVID dataset loader for LC2 cross-modal place recognition.

The VIVID dataset contains synchronized camera images and LiDAR range data.
For LC2 evaluation and training:
    - Query: LiDAR range images (``range_npy/*.npy``, single-channel (H, W) float32)
    - Database: Camera depth maps (pre-computed via ManyDepth/DA3, single-channel)

Positions from ``range_timelists.txt`` (LiDAR) and ``img_timelists.txt`` (camera)
provide UTM coordinates for ground truth matching.

Timelist format (3 columns, space-separated)::

    utm_x utm_y timestamp
    318.673129514 306.487684032 1586.935135885
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


def _parse_timelists(path: Path) -> List[Tuple[str, float, float]]:
    """Parse a VIVID timelists file.

    Format: ``utm_x utm_y timestamp`` per line.

    Returns:
        List of (timestamp_str, utm_x, utm_y) tuples.
    """
    entries: List[Tuple[str, float, float]] = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                utm_x = float(parts[0])
                utm_y = float(parts[1])
                timestamp = parts[2]
                entries.append((timestamp, utm_x, utm_y))
            except (ValueError, IndexError):
                continue
    return entries


class VIVIDLC2Dataset(Dataset):
    """VIVID dataset for LC2 evaluation and training.

    Loads pre-processed range images or depth maps, applies LC2 transforms
    (single-channel → 3ch repeat → Normalize([0.5]*3)), and provides
    UTM positions for retrieval evaluation.

    Directory structure expected::

        {root}/{sequence}/
        ├── img/                  # Camera images (*.png)
        ├── range_npy/            # Pre-processed range images (*.npy, single-channel)
        ├── range_timelists.txt   # "utm_x utm_y timestamp" per LiDAR frame
        ├── img_timelists.txt     # "utm_x utm_y timestamp" per camera frame
        └── ...

    The ``.npy`` filenames match the timestamp field from timelists.
    """

    def __init__(
        self,
        root: str,
        sequence: str,
        modality: str,
        depth_cache_dir: Optional[str] = None,
        range_cache_dir: Optional[str] = None,
        input_size: Optional[Tuple[int, int]] = None,
        subsample: int = 1,
        camera_hfov_deg: Optional[float] = None,
    ) -> None:
        """
        Args:
            root: Root directory containing VIVID sequences.
            sequence: Sequence name (e.g., ``"campus_day1"``).
            modality: Either ``"range"`` (LiDAR) or ``"depth"`` (camera).
            depth_cache_dir: Directory containing pre-computed depth .npy files.
                Required when modality is ``"depth"``.
            range_cache_dir: Directory containing proper panoramic range .npy files.
                If provided, used instead of ``{root}/{seq}/range_npy/``.
            input_size: If provided, resize to (H, W). If None, keep native
                resolution (original LC2 does not resize).
            subsample: Load every N-th frame (for faster evaluation).
            camera_hfov_deg: If set, crop range images to this FoV (paper III.B.4).
        """
        super().__init__()
        self.root = Path(root)
        self.sequence = sequence
        self.modality = modality
        self.depth_cache_dir = Path(depth_cache_dir) if depth_cache_dir else None
        self.range_cache_dir = Path(range_cache_dir) if range_cache_dir else None
        self.transform = get_transform(input_size)
        self.camera_hfov_deg = camera_hfov_deg

        seq_dir = self.root / sequence

        if modality == "range":
            self.samples, self.positions = self._load_range_data(seq_dir, subsample)
        elif modality == "depth":
            self.samples, self.positions = self._load_depth_data(seq_dir, subsample)
        else:
            raise ValueError(f"Unknown modality: {modality}. Use 'range' or 'depth'.")

    def _load_range_data(
        self, seq_dir: Path, subsample: int
    ) -> Tuple[List[Path], np.ndarray]:
        """Load range image paths and UTM positions from range_timelists.txt."""
        # Prefer range_cache_dir (proper panoramic) over legacy range_npy/
        if self.range_cache_dir is not None:
            range_npy_dir = self.range_cache_dir / self.sequence
            if not range_npy_dir.exists():
                range_npy_dir = self.range_cache_dir
        else:
            range_npy_dir = seq_dir / "range_npy"
        timelists_file = seq_dir / "range_timelists.txt"

        if not range_npy_dir.exists():
            raise FileNotFoundError(f"Range NPY directory not found: {range_npy_dir}")
        if not timelists_file.exists():
            raise FileNotFoundError(f"Range timelists not found: {timelists_file}")

        entries = _parse_timelists(timelists_file)
        npy_by_stem = {p.stem: p for p in range_npy_dir.glob("*.npy")}

        samples: List[Path] = []
        positions: List[Tuple[float, float]] = []

        for timestamp, utm_x, utm_y in entries:
            if timestamp in npy_by_stem:
                samples.append(npy_by_stem[timestamp])
                positions.append((utm_x, utm_y))

        # Fallback: if timestamp matching fails, align by sorted order
        if len(samples) == 0 and len(npy_by_stem) > 0 and len(entries) > 0:
            npy_sorted = sorted(npy_by_stem.values())
            n = min(len(npy_sorted), len(entries))
            for i in range(n):
                samples.append(npy_sorted[i])
                positions.append((entries[i][1], entries[i][2]))

        if subsample > 1:
            samples = samples[::subsample]
            positions = positions[::subsample]

        pos_array = np.array(positions, dtype=np.float64) if positions else np.zeros((0, 2))
        return samples, pos_array

    def _load_depth_data(
        self, seq_dir: Path, subsample: int
    ) -> Tuple[List[Path], np.ndarray]:
        """Load depth cache paths and UTM positions from img_timelists.txt."""
        if self.depth_cache_dir is None:
            raise ValueError("depth_cache_dir is required for 'depth' modality")

        timelists_file = seq_dir / "img_timelists.txt"
        if not timelists_file.exists():
            timelists_file = seq_dir / "imglist.txt"
        if not timelists_file.exists():
            raise FileNotFoundError(f"Image timelists not found in {seq_dir}")

        entries = _parse_timelists(timelists_file)

        # Locate depth cache directory
        depth_dir = Path(self.depth_cache_dir) / self.sequence
        if not depth_dir.exists():
            depth_dir = Path(self.depth_cache_dir)

        depth_files = {p.stem: p for p in depth_dir.glob("*.npy")}

        samples: List[Path] = []
        positions: List[Tuple[float, float]] = []

        for timestamp, utm_x, utm_y in entries:
            if timestamp in depth_files:
                samples.append(depth_files[timestamp])
                positions.append((utm_x, utm_y))

        if subsample > 1:
            samples = samples[::subsample]
            positions = positions[::subsample]

        pos_array = np.array(positions, dtype=np.float64) if positions else np.zeros((0, 2))
        return samples, pos_array

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load a single sample.

        The numpy array is loaded as-is (single-channel float32) and passed
        through the LC2 transform: ToTensor → Repeat3ch → Normalize([0.5]*3).

        Returns:
            Dict with keys:
            - ``"image"``: Transformed tensor ``(3, H, W)``.
            - ``"is_range"``: Boolean, True if range modality.
            - ``"position"``: UTM position tensor ``(2,)``.
            - ``"index"``: Dataset index.
        """
        npy_path = self.samples[idx]
        data = np.load(str(npy_path))

        # Ensure 2D single-channel
        if data.ndim > 2:
            data = squeeze_depth(data)

        if self.modality == "depth":
            data = depth_to_normalized_disparity(data)
        elif self.modality == "range":
            if self.camera_hfov_deg is not None:
                data = crop_range_to_camera_fov(data, camera_hfov_deg=self.camera_hfov_deg)
            data = normalize_disparity(data)

        image = self.transform(data)
        position = torch.from_numpy(self.positions[idx].copy()).float()

        return {
            "image": image,
            "is_range": self.modality == "range",
            "position": position,
            "index": idx,
        }

    def get_positions(self) -> np.ndarray:
        """Return all positions as numpy array of shape ``(N, 2)``."""
        return self.positions.copy()

    def get_paths(self) -> List[Path]:
        """Return all sample file paths."""
        return list(self.samples)
