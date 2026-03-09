#!/bin/bash
# Train LC2 on multiple KITTI-360 sequences
# Train: 0002, 0004, 0005, 0006
# Test:  0000, 0009, 0010 (following LIP-Loc protocol: seq0000 as primary test)
set -e

cd /home/jhlee/ws_xloc/external/LC2_crossmatching

OUTPUT_DIR=checkpoints/kitti360_multi_v3
mkdir -p $OUTPUT_DIR

echo "=== LC2 Multi-Sequence Training ==="
echo "Train: 0002, 0004, 0005, 0006 (subsample=3)"
echo "Val: 0000"
echo "Test: 0000, 0009, 0010"
echo "Started at $(date)"

# Verify caches exist
for seq in 0002 0004 0005 0006 0000 0009 0010; do
    dc=$(ls cache/depth/kitti360/$seq/*.npy 2>/dev/null | wc -l)
    rc=$(ls cache/range/kitti360/$seq/*.npy 2>/dev/null | wc -l)
    echo "  seq$seq: depth=$dc range=$rc"
done

# Phase 1 skip (pretrained encoder) + Phase 2 GeM+Triplet
PYTHONUNBUFFERED=1 python train.py \
    --config configs/train_kitti360_multi.yaml \
    --resume pretrained/dual_encoder.pth.tar \
    --skip_phase1 \
    --gem_phase2 \
    --output_dir $OUTPUT_DIR \
    --device cuda \
    2>&1 | tee $OUTPUT_DIR/train.log

echo "=== Done at $(date) ==="
