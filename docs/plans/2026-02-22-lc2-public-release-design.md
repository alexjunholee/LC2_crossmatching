# LC2 Public Release — Clean Re-architecture Design

**Date:** 2026-02-22
**Repo:** `alexjunholee/LC2_crossmatching` (existing, update in-place)
**Goal:** Production-quality public release with multi-depth-estimator support, comprehensive LiDAR augmentation, and artifact handling.

---

## 1. Repository Structure

```
LC2_crossmatching/
├── README.md                    # Full usage guide (install, train, eval, preprocess)
├── pyproject.toml               # Package metadata + dependencies
├── requirements.txt             # Pinned dependencies
├── LICENSE                      # MIT or Apache 2.0
│
├── lc2/
│   ├── __init__.py              # Version, convenience imports
│   ├── model.py                 # LC2Model (DualEncoder + pooling)
│   ├── encoder.py               # DualEncoder (VGG16 depth/range branches)
│   ├── netvlad.py               # NetVLAD pooling
│   ├── gem.py                   # GeM pooling
│   ├── losses.py                # ContrastiveLoss + TripletLoss
│   │
│   ├── depth/                   # Pluggable depth estimation
│   │   ├── __init__.py          # get_depth_estimator(name) factory
│   │   ├── base.py              # BaseDepthEstimator ABC
│   │   ├── depth_anything.py    # Depth Anything 3 (DA3)
│   │   ├── dinov3_depth.py      # DINOv3 monocular depth head
│   │   └── dust3r_depth.py      # DUSt3R stereo/multi-view depth
│   │
│   ├── lidar/                   # LiDAR processing + augmentation
│   │   ├── __init__.py          # get_sensor_config(name) registry
│   │   ├── sensors.py           # SensorConfig dataclass + presets
│   │   ├── projection.py        # lidar_to_range_image()
│   │   ├── io.py                # Point cloud loaders (KITTI, PCD, HeLiPR, etc.)
│   │   ├── infill.py            # Sparse region interpolation
│   │   └── augmentation.py      # Beam drop, jitter, occlusion, sensor transfer
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── transforms.py        # Image transforms + augmentation pipeline
│   │   ├── train_dataset.py     # Phase1/Phase2 datasets + miners
│   │   ├── vivid.py             # VIVID dataset
│   │   ├── kitti360.py          # KITTI-360 dataset
│   │   └── helipr.py            # HeLiPR dataset
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── retrieval.py         # FAISS index, evaluate_retrieval
│   │   └── checkpoint.py        # Checkpoint loading/saving
│   │
│   └── fov_overlap.py           # FoV overlap computation
│
├── configs/
│   ├── train_vivid.yaml
│   ├── train_kitti360.yaml
│   ├── train_helipr.yaml
│   ├── eval_vivid.yaml
│   ├── eval_kitti360.yaml
│   └── eval_helipr.yaml
│
├── train.py                     # Training entry point
├── evaluate.py                  # Evaluation entry point
├── preprocess_depth.py          # Depth map preprocessing (all estimators)
├── preprocess_range.py          # Range image preprocessing
│
└── scripts/
    └── download_weights.sh      # Download pretrained weights from GitHub Releases
```

## 2. Depth Estimator Plugin Architecture

### 2.1 Base Interface

```python
# lc2/depth/base.py
class BaseDepthEstimator(ABC):
    """Abstract base for monocular/stereo depth estimation."""

    @abstractmethod
    def estimate(self, image_path: str) -> np.ndarray:
        """Single image → (H, W) float32 depth map."""

    @abstractmethod
    def estimate_batch(self, paths: List[str], batch_size: int = 16) -> List[np.ndarray]:
        """Batch inference for efficiency."""

    def estimate_from_array(self, image: np.ndarray) -> np.ndarray:
        """In-memory image → depth map (default: save temp + estimate)."""
```

### 2.2 Factory

```python
# lc2/depth/__init__.py
DEPTH_ESTIMATORS = {
    "da3": "lc2.depth.depth_anything.DA3DepthEstimator",
    "da3-small": "lc2.depth.depth_anything.DA3DepthEstimator",
    "da3-large": "lc2.depth.depth_anything.DA3DepthEstimator",
    "da3-giant": "lc2.depth.depth_anything.DA3DepthEstimator",
    "dinov3": "lc2.depth.dinov3_depth.DINOv3DepthEstimator",
    "dust3r": "lc2.depth.dust3r_depth.DUSt3RDepthEstimator",
}

def get_depth_estimator(name: str, **kwargs) -> BaseDepthEstimator:
    """Factory: name → estimator instance."""
```

### 2.3 Implementations

**DA3** (existing, refactored):
- Wraps Depth Anything 3 from HuggingFace
- Variants: SMALL (fast), LARGE (balanced), GIANT (best)
- Single-image relative depth

**DINOv3 Depth**:
- DINOv3 ViT backbone + DPT depth head
- Artifact characteristics similar to LiDAR range images (soft edges, consistent structure)
- Better cross-modal compatibility than DA3 for LC2 matching

**DUSt3R Depth**:
- Stereo/multi-view metric depth from DUSt3R
- Scale-accurate (unlike monocular methods)
- Requires 2+ views — use sequential frames from driving data
- Fallback to monocular if single image provided

## 3. LiDAR Processing Module

### 3.1 Sensor Registry

```python
# lc2/lidar/sensors.py
SENSOR_REGISTRY = {
    "velodyne_hdl64":   SensorConfig("Velodyne HDL-64E",   fov_up=2.0,  fov_down=-24.8, H=64,  W=2048, max_range=120.0),
    "ouster_os1_64":    SensorConfig("Ouster OS1-64",      fov_up=22.5, fov_down=-22.5, H=64,  W=1024, max_range=120.0),
    "ouster_os2_128":   SensorConfig("Ouster OS2-128",     fov_up=22.5, fov_down=-22.5, H=128, W=1024, max_range=200.0),
    "velodyne_vlp32c":  SensorConfig("Velodyne VLP-32C",   fov_up=15.0, fov_down=-25.0, H=32,  W=1024, max_range=200.0),
    "velodyne_vlp16":   SensorConfig("Velodyne VLP-16",    fov_up=15.0, fov_down=-15.0, H=16,  W=1824, max_range=100.0),
    "livox_avia":       SensorConfig("Livox Avia",         fov_up=38.4, fov_down=-4.0,  H=6,   W=1000, max_range=450.0),  # Non-repetitive
    "hesai_pandar64":   SensorConfig("Hesai Pandar64",     fov_up=15.0, fov_down=-25.0, H=64,  W=1800, max_range=200.0),
    "robosense_rs128":  SensorConfig("RoboSense RS-128",   fov_up=15.0, fov_down=-25.0, H=128, W=1800, max_range=200.0),
}

def get_sensor_config(name: str) -> SensorConfig: ...
def register_sensor(name: str, config: SensorConfig) -> None: ...
```

### 3.2 Point Cloud I/O

```python
# lc2/lidar/io.py
def load_point_cloud(path: str, format: str = "auto") -> np.ndarray:
    """Universal loader: auto-detect format from extension/header.

    Supported formats:
    - .bin (KITTI, HeLiPR Ouster/Velodyne)
    - .pcd (VIVID, general PCD v0.7 binary)
    - .ply (general PLY)
    - .npy (numpy arrays)
    - .las/.laz (LAS/LAZ point clouds)
    """
```

### 3.3 Sparse Infill

```python
# lc2/lidar/infill.py
def nearest_neighbor_infill(range_img: np.ndarray, mask: np.ndarray, max_dist: int = 3) -> np.ndarray:
    """Fill empty pixels from nearest occupied neighbors."""

def bilateral_infill(range_img: np.ndarray, mask: np.ndarray, sigma_s: float = 2.0, sigma_r: float = 0.1) -> np.ndarray:
    """Edge-preserving bilateral filter infill."""

def morphological_infill(range_img: np.ndarray, mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Morphological closing for small gaps."""

def adaptive_infill(range_img: np.ndarray, mask: np.ndarray, occupancy_threshold: float = 0.3) -> np.ndarray:
    """Adaptive infill based on local occupancy rate."""
```

### 3.4 Range Image Augmentation

```python
# lc2/lidar/augmentation.py
class RangeAugmentation:
    """Composable augmentation pipeline for range images.

    Usage:
        aug = RangeAugmentation([
            RandomBeamDrop(p=0.1),
            RandomPointJitter(sigma=0.02),
            RandomRangeNoise(sigma=0.01),
            RandomOcclusion(n_patches=3, max_size=0.1),
            RandomFOVCrop(min_fov_frac=0.7),
        ])
        augmented = aug(range_img, mask)
    """

class RandomBeamDrop:
    """Drop entire LiDAR beams (rows) to simulate sparse sensors."""
    def __init__(self, p: float = 0.1, min_remaining: float = 0.5): ...

class RandomPointJitter:
    """Add Gaussian noise to occupied range values."""
    def __init__(self, sigma: float = 0.02): ...

class RandomRangeNoise:
    """Additive range noise simulating sensor measurement error."""
    def __init__(self, sigma: float = 0.01): ...

class RandomOcclusion:
    """Random rectangular occlusion patches (simulates dynamic objects)."""
    def __init__(self, n_patches: int = 3, max_size: float = 0.1): ...

class RandomFOVCrop:
    """Random FOV sub-crop for viewpoint robustness."""
    def __init__(self, min_fov_frac: float = 0.7): ...

class SensorTransfer:
    """Simulate different sensor by downsampling channels/resolution."""
    def __init__(self, source_config: SensorConfig, target_config: SensorConfig): ...

class RandomIntensityShift:
    """Shift intensity/reflectance values (if using intensity channel)."""
    def __init__(self, max_shift: float = 0.1): ...
```

## 4. Depth Augmentation (Enhanced)

Existing scale augmentation (±20%) extended with:

```python
# In lc2/data/transforms.py (additions)
class DepthAugmentation:
    """Composable depth image augmentation."""

    - RandomScaleAug(max_pct=20)         # Existing ±20% scale
    - RandomDepthNoise(sigma=0.02)       # Gaussian noise on depth values
    - RandomDepthHoles(p=0.05)           # Simulate depth estimation failures
    - RandomEdgeBlur(kernel=5)           # Blur depth edges (artifact matching)
    - RandomArtifactPatches(n=3)         # Simulate depth estimator artifacts
```

## 5. Weight Distribution

- **Single pretrained weight**: `lc2_pretrained.pth.tar`
  - Well-trained base encoder (Phase 1 contrastive + Phase 2 fine-tune)
  - Augmentation-robust: trained with full augmentation pipeline
  - Uploaded to GitHub Releases
- Users fine-tune on their target dataset using provided training code
- `scripts/download_weights.sh`: auto-download from GitHub Releases via `gh release download` or `curl`

## 6. Config Structure (Updated)

```yaml
# configs/train_vivid.yaml (example)
dataset:
  name: vivid
  root: /path/to/vivid
  train_seq: campus_day1
  eval_seq: campus_day2
  subsample: 1

model:
  num_clusters: 16
  encoder_dim: 512

depth:
  estimator: dinov3           # da3 | dinov3 | dust3r
  model_name: dinov3_vitb16   # estimator-specific variant
  cache_dir: cache/depth/

lidar:
  sensor: ouster_os1_64
  infill: nearest_neighbor    # none | nearest_neighbor | bilateral | adaptive
  augmentation:
    beam_drop: 0.1
    point_jitter: 0.02
    range_noise: 0.01
    occlusion_patches: 3
    fov_crop_min: 0.7

input:
  resize: [64, 512]
  camera_hfov_deg: 90.0
  use_disparity: true
  depth_augmentation:
    scale_pct: 20
    noise: 0.02
    holes: 0.05

training:
  phase1:
    epochs: 5
    lr: 1.0e-4
    batch_size: 32
    loss_margin: 1.0
  phase2:
    epochs: 15
    lr: 5.0e-5
    batch_size: 16
    triplet_margin: 0.1
    hard_negatives: 10

eval:
  top_k: 25
  recall_ks: [1, 5, 10]
  pos_thresholds: [5.0, 10.0, 25.0]
```

## 7. Implementation Phases

### Phase 1: Core Restructure (Code cleanup)
1. Create `lc2/depth/` plugin architecture
2. Create `lc2/lidar/` module (sensors, projection, io from existing code)
3. Refactor `train.py` to use new module structure
4. Refactor `evaluate.py` similarly
5. Update configs to new format

### Phase 2: New Features
6. Implement `lc2/depth/dinov3_depth.py`
7. Implement `lc2/depth/dust3r_depth.py`
8. Implement `lc2/lidar/infill.py`
9. Implement `lc2/lidar/augmentation.py`
10. Implement depth augmentation extensions

### Phase 3: Integration & Polish
11. Update `preprocess_depth.py` for all estimators
12. Update `preprocess_range.py` with infill + augmentation
13. Write comprehensive README.md
14. Add pyproject.toml
15. Test full pipeline (preprocess → train → eval)

### Phase 4: Release
16. Train with full augmentation pipeline → produce best weight
17. Upload weight to GitHub Releases
18. Final code review + cleanup
19. Tag release v2.0

## 8. Dependencies

```
torch >= 2.0
torchvision >= 0.15
numpy >= 1.24
faiss-cpu >= 1.7 (or faiss-gpu)
scikit-learn >= 1.3
pyyaml >= 6.0
Pillow >= 10.0
tqdm >= 4.65
scipy >= 1.11  # for infill operations

# Optional (depth estimators):
# depth-anything-3  (for DA3)
# dinov3             (for DINOv3 depth)
# dust3r             (for DUSt3R depth)
```

## 9. README Structure

1. Overview + method diagram
2. Installation (pip install -e .)
3. Quick Start (eval with pretrained weight)
4. Dataset Preparation (KITTI-360, VIVID, HeLiPR)
5. Preprocessing (depth estimation + range projection)
6. Training (Phase 1 + Phase 2)
7. Evaluation
8. Custom Dataset Guide
9. Citation
