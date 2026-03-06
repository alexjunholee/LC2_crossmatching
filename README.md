# LC2: LiDAR-Camera Cross-Modal Place Recognition

## Overview

LC2 is a cross-modal place recognition system that matches LiDAR range images against camera-derived depth images for loop closure detection. The method converts both modalities into a shared representation space using a dual VGG16 encoder with modality-specific branches, trained through a two-phase pipeline: contrastive pre-training with GeM pooling followed by triplet fine-tuning with NetVLAD aggregation.

**Key features:**

- Cross-modal retrieval between LiDAR range images and monocular depth estimates
- Dual VGG16 encoder with independent branches for each modality
- Two-phase training: contrastive loss with FoV-overlap weighting (Phase 1) + triplet loss with hard negative mining (Phase 2)
- Pluggable monocular depth estimation backends (Depth Anything 3, DINOv3 DPT, DUSt3R)
- 8 built-in LiDAR sensor presets with runtime-extensible registry
- Comprehensive range image augmentation and sparse-region infill
- Support for KITTI-360, VIVID, and HeLiPR datasets

**Paper:** Lee et al., "(LC)^2: LiDAR-Camera Loop Constraints for Cross-Modal Place Recognition", IEEE Robotics and Automation Letters (RA-L), 2023.

## Installation

### From source (recommended)

```bash
git clone https://github.com/alexjunholee/LC2_crossmatching.git
cd LC2_crossmatching
pip install -e .
```

### With optional depth estimation backends

```bash
# Depth Anything 3
pip install -e ".[da3]"

# DUSt3R stereo depth
pip install -e ".[dust3r]"

# All optional dependencies (DA3, DUSt3R, GTSAM)
pip install -e ".[all]"
```

### Core dependencies

- Python >= 3.10
- PyTorch >= 2.0
- torchvision >= 0.15
- faiss-cpu >= 1.7
- scikit-learn >= 1.3
- scipy >= 1.11
- NumPy, PyYAML, Pillow, tqdm

## Quick Start

### Download pretrained weights

```bash
bash scripts/download_weights.sh
```

This downloads `lc2_pretrained.pth.tar` into the `pretrained/` directory. The script supports `gh`, `curl`, and `wget` backends.

### Evaluate on VIVID

```bash
python evaluate.py --config configs/eval_vivid.yaml \
    --checkpoint pretrained/lc2_pretrained.pth.tar
```

### Evaluate on KITTI-360

```bash
python evaluate.py --config configs/eval_kitti360.yaml \
    --checkpoint pretrained/lc2_pretrained.pth.tar
```

### Evaluate on HeLiPR

```bash
python evaluate.py --config configs/eval_helipr.yaml \
    --checkpoint pretrained/lc2_pretrained.pth.tar
```

The evaluation script extracts descriptors for query (range) and database (depth) modalities, builds a FAISS index, retrieves top-K candidates, and reports Recall@K at multiple distance thresholds. Results are saved to the `results/` directory.

## Dataset Preparation

### KITTI-360

```
KITTI-360/
  data_2d_raw/
    2013_05_28_drive_0000_sync/
      image_00/
        data_rgb/
          0000000000.png
          ...
  data_3d_raw/
    2013_05_28_drive_0000_sync/
      velodyne_points/
        data/
          0000000000.bin
          ...
```

Download from [KITTI-360 website](http://www.cvlibs.net/datasets/kitti-360/).

### VIVID

```
vivid/
  campus_day1/
    img/
      000000.png
      ...
    pcd/
      000000.pcd
      ...
  campus_day2/
    img/
      ...
    pcd/
      ...
```

### HeLiPR

```
HeLiPR/
  DCC04/
    LiDAR/
      Ouster/
        *.bin
      Velodyne/
        *.bin
    img/
      *.png
  DCC05/
    ...
```

## Preprocessing

Before training or evaluation, pre-compute depth maps from camera images and range images from LiDAR point clouds.

### Depth Estimation

```bash
# VIVID with Depth Anything 3 (default)
python preprocess_depth.py --dataset vivid --sequences campus_day1 campus_day2 \
    --output_dir cache/depth/vivid

# KITTI-360 with DA3-Small variant
python preprocess_depth.py --dataset kitti360 --sequences 0000 0002 \
    --output_dir cache/depth/kitti360 --estimator da3 --variant da3-small

# Using DINOv3 DPT depth
python preprocess_depth.py --dataset vivid --sequences campus_day1 \
    --output_dir cache/depth/vivid_dinov3 --estimator dinov3

# Using DUSt3R stereo depth
python preprocess_depth.py --dataset vivid --sequences campus_day1 \
    --output_dir cache/depth/vivid_dust3r --estimator dust3r

# Resume interrupted preprocessing
python preprocess_depth.py --dataset vivid --sequences campus_day1 \
    --output_dir cache/depth/vivid --skip_existing
```

Available depth estimators:

| Name | Variants | Description |
|------|----------|-------------|
| `da3` | `da3-small`, `da3-large` (default), `da3-giant` | Depth Anything 3 monocular depth |
| `dinov3` | `dinov3_vits14`, `dinov3_vitb14`, `dinov3_vitl14` (default), `dinov3_vitg14` | DINOv3 DPT depth head |
| `dust3r` | single variant | DUSt3R stereo depth |

### Range Image Generation

```bash
# KITTI-360 (Velodyne HDL-64E auto-detected)
python preprocess_range.py --dataset kitti360 --sequences 0000 0002 \
    --output_dir cache/range/kitti360

# VIVID (Ouster OS1-64 auto-detected)
python preprocess_range.py --dataset vivid --sequences campus_day1 campus_day2 \
    --output_dir cache/range/vivid

# HeLiPR — requires explicit sensor selection
python preprocess_range.py --dataset helipr --sequences DCC04 \
    --sensor "Ouster OS2-128" --output_dir cache/range/helipr/Ouster

python preprocess_range.py --dataset helipr --sequences DCC04 \
    --sensor "Velodyne VLP-32C" --output_dir cache/range/helipr/Velodyne

# With sparse-region infill
python preprocess_range.py --dataset kitti360 --sequences 0000 \
    --infill nearest_neighbor --output_dir cache/range/kitti360_filled

python preprocess_range.py --dataset vivid --sequences campus_day1 \
    --infill adaptive --output_dir cache/range/vivid_adaptive
```

Available infill methods: `none` (default), `nearest_neighbor`, `bilateral`, `morphological`, `adaptive`.

## Training

The training pipeline follows the paper with two phases:

- **Phase 1** -- Contrastive pre-training with GeM pooling. Uses a modified contrastive loss (Eq. 2 in the paper) weighted by the degree of FoV overlap (psi). Range images are augmented with 8 overlapping FoV-masked crops (90 degrees each). Depth images are converted to disparity with scale augmentation (+/-20%).

- **Phase 2** -- Triplet fine-tuning with NetVLAD pooling. Hard negative mining with triplet margin loss (m=0.1). Sub-phase 2a freezes the encoder and trains only NetVLAD cluster centroids. Sub-phase 2b unfreezes conv5 layers for joint fine-tuning with early stopping.

```bash
# Full pipeline (Phase 1 + Phase 2) on VIVID
python train.py --config configs/train_vivid.yaml \
    --resume pretrained/lc2_pretrained.pth.tar

# Phase 2 only (skip contrastive pre-training)
python train.py --config configs/train_vivid.yaml \
    --resume pretrained/lc2_pretrained.pth.tar --skip_phase1

# KITTI-360 training
python train.py --config configs/train_kitti360.yaml \
    --resume pretrained/lc2_pretrained.pth.tar

# HeLiPR training
python train.py --config configs/train_helipr.yaml \
    --resume pretrained/lc2_pretrained.pth.tar --skip_phase1
```

Checkpoints are saved to `checkpoints/` by default. The best model (by validation Recall@1) is saved as `checkpoints/best.pth.tar`.

## Evaluation

```bash
# Standard evaluation
python evaluate.py --config configs/eval_vivid.yaml \
    --checkpoint checkpoints/best.pth.tar

# With mixed precision for faster inference
python evaluate.py --config configs/eval_kitti360.yaml \
    --checkpoint checkpoints/best.pth.tar --fp16

# Custom output directory
python evaluate.py --config configs/eval_helipr.yaml \
    --checkpoint checkpoints/best.pth.tar --output_dir results/helipr_exp1
```

The evaluator reports Recall@K at multiple distance thresholds and saves:
- `query_descriptors.npy`, `db_descriptors.npy` -- extracted global descriptors
- `predictions.npy`, `distances.npy` -- retrieval results
- `results.yaml` -- recall summary

## Supported Sensors

The LiDAR sensor registry ships with 8 presets. Custom sensors can be registered at runtime.

| Sensor | Channels | Azimuth Bins | Vertical FOV | Max Range | Used By |
|--------|----------|-------------|--------------|-----------|---------|
| Velodyne HDL-64E | 64 | 2048 | -24.8 to +2.0 deg | 120 m | KITTI-360 |
| Velodyne VLP-32C | 32 | 1024 | -25.0 to +15.0 deg | 200 m | HeLiPR |
| Velodyne VLP-16 | 16 | 1024 | -15.0 to +15.0 deg | 100 m | General |
| Ouster OS1-64 | 64 | 1024 | -22.5 to +22.5 deg | 120 m | VIVID |
| Ouster OS2-128 | 128 | 1024 | -22.5 to +22.5 deg | 200 m | HeLiPR |
| Livox Avia | 6 | 4000 | -38.6 to +38.6 deg | 190 m | General |
| Hesai Pandar64 | 64 | 1800 | -25.0 to +15.0 deg | 200 m | General |
| RoboSense RS-128 | 128 | 1800 | -25.0 to +15.0 deg | 200 m | General |

### Registering a custom sensor

```python
from lc2.lidar import SensorConfig, register_sensor, get_sensor_config

my_sensor = SensorConfig(
    name="My Custom LiDAR",
    fov_up=15.0,
    fov_down=-25.0,
    H=64,
    W=2048,
    max_range=150.0,
)
register_sensor("My Custom LiDAR", my_sensor)

# Use it
config = get_sensor_config("My Custom LiDAR")
```

## Custom Dataset Guide

To add a new dataset:

1. **Register your LiDAR sensor** (if not already in the registry):

```python
from lc2.lidar import SensorConfig, register_sensor

register_sensor("My Sensor", SensorConfig(
    name="My Sensor", fov_up=15.0, fov_down=-25.0,
    H=64, W=1024, max_range=100.0,
))
```

2. **Implement a dataset class** following the interface used by `VIVIDLC2Dataset`:

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, root, modality, input_size=None, ...):
        # Load file paths and positions
        ...

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Return {"image": tensor, "position": ndarray, "index": int}
        ...

    def get_positions(self):
        # Return (N, 2) or (N, 3) array of UTM/metric positions
        ...
```

3. **Create a YAML config file**:

```yaml
dataset:
  name: mydataset
  root: /path/to/data
  sequences: [seq01, seq02]
  pos_threshold: 25.0

model:
  num_clusters: 16
  encoder_dim: 512

depth:
  estimator: da3
  variant: da3-large

lidar:
  sensor: my_sensor
  infill: none
  augmentation:
    beam_drop: 0.1
    point_jitter: 0.02
    range_noise: 0.01
    occlusion_patches: 3

input:
  resize: [480, 640]
  use_disparity: true
  depth_augmentation:
    scale_pct: 20
    noise: 0.02
    holes: 0.05

train:
  sequences: [seq01]
  batch_size: 4
  phase1_epochs: 20
  phase2_frozen_epochs: 5
  phase2_epochs: 25
  pos_dist_thr: 10.0
  neg_dist_thr: 25.0
  margin: 0.1
  optimizer: sgd
  momentum: 0.9
  weight_decay: 0.001
  lr_step: 5
  lr_gamma: 0.5
  patience: 10

val:
  sequences: [seq02]
  batch_size: 64
  recall_ks: [1, 5, 10]

eval:
  batch_size: 64
  top_k: 25
  recall_ks: [1, 5, 10, 15, 20]
  pos_thresholds: [5.0, 10.0, 25.0]
```

## Configuration

YAML config files control all aspects of the pipeline. The main sections are:

| Section | Purpose |
|---------|---------|
| `dataset` | Dataset name, root path, sequences, positive distance threshold |
| `model` | Number of NetVLAD clusters (K), encoder feature dimension (512) |
| `depth` | Depth estimator backend (`da3`, `dinov3`, `dust3r`) and variant |
| `lidar` | Sensor preset name, infill method, augmentation parameters |
| `input` | Input resolution, disparity conversion flag, camera FoV crop, depth augmentation |
| `train` | Phase 1/2 epochs, learning rates, triplet mining thresholds, optimizer settings |
| `val` | Validation sequences, batch size, Recall@K values to track |
| `eval` | Evaluation batch size, top-K retrieval depth, distance thresholds |

See `configs/train_vivid.yaml` for a fully annotated example.

## Project Structure

```
LC2_crossmatching/
  lc2/
    __init__.py              # Package root (exports LC2Model, DualEncoder, NetVLAD)
    model.py                 # LC2Model: dual encoder + GeM/NetVLAD pooling
    encoder.py               # DualEncoder: two VGG16 branches (depth + range)
    netvlad.py               # NetVLAD aggregation layer
    gem.py                   # Generalized Mean (GeM) pooling
    losses.py                # LC2 contrastive loss with psi weighting
    depth/
      __init__.py            # Lazy-loading depth estimator factory
      base.py                # BaseDepthEstimator abstract class
      depth_anything.py      # Depth Anything 3 backend
      dinov3_depth.py        # DINOv3 DPT depth backend
      dust3r_depth.py        # DUSt3R stereo depth backend
    lidar/
      __init__.py            # Sensor registry, projection, IO, infill, augmentation
      sensors.py             # SensorConfig dataclass + 8 presets + registry API
      projection.py          # Spherical range-image projection
      io.py                  # Point cloud loaders (.bin, .pcd, .ply, .npy)
      infill.py              # Sparse-region infill (NN, bilateral, morphological, adaptive)
      augmentation.py        # Range image augmentations (beam drop, jitter, occlusion, ...)
    data/
      transforms.py          # Depth/range preprocessing transforms and augmentations
      train_dataset.py       # Training dataset classes, pool builders, triplet/pair miners
      vivid.py               # VIVID dataset loader
      kitti360.py            # KITTI-360 dataset loader
      helipr.py              # HeLiPR dataset loader
    utils/
      checkpoint.py          # Checkpoint loading with key remapping
      retrieval.py           # FAISS index building, retrieval, Recall@K computation
  configs/                   # YAML configuration files
  scripts/
    download_weights.sh      # Pretrained weight downloader
  train.py                   # Two-phase training pipeline
  evaluate.py                # Evaluation with Recall@K
  preprocess_depth.py        # Batch depth estimation preprocessing
  preprocess_range.py        # Batch range image generation
  pretrained/                # Pretrained model weights
  checkpoints/               # Training checkpoints
  results/                   # Evaluation outputs
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{lee20232,
  title={${(LC)}^{2}$: LiDAR-camera loop constraints for cross-modal place recognition},
  author={Lee, Alex Junho and Song, Seungwon and Lim, Hyungtae and Lee, Woojoo and Myung, Hyun},
  journal={IEEE Robotics and Automation Letters},
  volume={8},
  number={6},
  pages={3589--3596},
  year={2023},
  publisher={IEEE}
}
```

## License

This project is released under the [MIT License](LICENSE).
