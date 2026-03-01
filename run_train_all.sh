#!/bin/bash
set -e
cd /home/jhlee/ws_xloc/external/LC2_crossmatching

PRETRAINED=pretrained/dual_encoder.pth.tar

echo "=== VIVID Training Start (v6: resize 192x512) ==="
python train.py --config configs/train_vivid.yaml --output_dir checkpoints/vivid_v6 --resume $PRETRAINED --device cuda 2>&1 | tee logs/train_vivid_v6.log
echo "=== VIVID Training Done ==="

echo "=== KITTI-360 Training Start (v5: resize 192x640) ==="
python train.py --config configs/train_kitti360.yaml --output_dir checkpoints/kitti360_v5 --resume $PRETRAINED --device cuda 2>&1 | tee logs/train_kitti360_v5.log
echo "=== KITTI-360 Training Done ==="

echo "=== HeLiPR Training Start (v4: resize 128x1024) ==="
python train.py --config configs/train_helipr.yaml --output_dir checkpoints/helipr_v4 --resume $PRETRAINED --device cuda 2>&1 | tee logs/train_helipr_v4.log
echo "=== HeLiPR Training Done ==="
