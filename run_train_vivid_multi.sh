#!/bin/bash
# Train LC2 on multiple VIVID sequences (KITTI v3 recipe)
# Train: campus_day1, campus_day2, city_day1, city_day2 (subsample=3, ~20K pool)
# Val:   campus_val1, campus_val2, city_val1, city_val2
set -e

cd /home/jhlee/ws_xloc/external/LC2_crossmatching

OUTPUT_DIR=checkpoints/vivid_multi_v1
mkdir -p $OUTPUT_DIR

echo "=== LC2 VIVID Multi-Sequence Training ==="
echo "Train: campus_day1, campus_day2, city_day1, city_day2 (subsample=3)"
echo "Val:   campus_val1, campus_val2, city_val1, city_val2"
echo "Started at $(date)"

# ── Step 1: Generate caches if missing ──

NEED_CACHE=false
for seq in campus_day1 campus_day2 city_day1 city_day2 campus_val1 campus_val2 city_val1 city_val2; do
    rc=$(ls cache/range/vivid/$seq/*.npy 2>/dev/null | wc -l)
    dc=$(ls cache/depth/vivid/$seq/*.npy 2>/dev/null | wc -l)
    if [ "$rc" -eq 0 ] || [ "$dc" -eq 0 ]; then
        NEED_CACHE=true
        echo "  $seq: range=$rc depth=$dc (MISSING)"
    else
        echo "  $seq: range=$rc depth=$dc (OK)"
    fi
done

if [ "$NEED_CACHE" = true ]; then
    echo ""
    echo "=== Generating missing caches ==="

    # Range caches
    echo "--- Range image cache ---"
    python preprocess_range.py \
        --dataset vivid \
        --sequences campus_day1 campus_day2 city_day1 city_day2 \
            campus_val1 campus_val2 city_val1 city_val2 \
        --output_dir cache/range/vivid \
        --skip_existing

    # Depth caches (GPU required)
    echo "--- Depth map cache ---"
    python preprocess_depth.py \
        --dataset vivid \
        --sequences campus_day1 campus_day2 city_day1 city_day2 \
            campus_val1 campus_val2 city_val1 city_val2 \
        --output_dir cache/depth/vivid \
        --skip_existing \
        --device cuda

    echo "=== Cache generation done at $(date) ==="
    echo ""
fi

# ── Step 2: Verify caches ──

echo "=== Verifying caches ==="
for seq in campus_day1 campus_day2 city_day1 city_day2 campus_val1 campus_val2 city_val1 city_val2; do
    rc=$(ls cache/range/vivid/$seq/*.npy 2>/dev/null | wc -l)
    dc=$(ls cache/depth/vivid/$seq/*.npy 2>/dev/null | wc -l)
    echo "  $seq: range=$rc depth=$dc"
done

# ── Step 3: Train ──

echo ""
echo "=== Starting training ==="
PYTHONUNBUFFERED=1 python train.py \
    --config configs/train_vivid_multi.yaml \
    --resume pretrained/dual_encoder.pth.tar \
    --skip_phase1 \
    --gem_phase2 \
    --output_dir $OUTPUT_DIR \
    --device cuda \
    2>&1 | tee $OUTPUT_DIR/train.log

echo "=== Done at $(date) ==="
