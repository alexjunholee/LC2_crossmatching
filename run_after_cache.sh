#!/bin/bash
# Wait for cache generation processes to finish, then start training
set -e

echo "Waiting for cache generation to finish..."

# Wait for range cache (preprocess_range.py)
RANGE_PID=$(pgrep -f "preprocess_range.py" || true)
if [ -n "$RANGE_PID" ]; then
    echo "  Range cache PID: $RANGE_PID"
    while kill -0 $RANGE_PID 2>/dev/null; do sleep 30; done
    echo "  Range cache done at $(date)"
fi

# Wait for depth cache (preprocess_depth.py)
DEPTH_PID=$(pgrep -f "preprocess_depth.py" || true)
if [ -n "$DEPTH_PID" ]; then
    echo "  Depth cache PID: $DEPTH_PID"
    while kill -0 $DEPTH_PID 2>/dev/null; do sleep 30; done
    echo "  Depth cache done at $(date)"
fi

echo "All caches ready at $(date)"

# Verify
cd /home/jhlee/ws_xloc/external/LC2_crossmatching
for seq in 0000 0002 0004 0005 0006 0009 0010; do
    dc=$(ls cache/depth/kitti360/$seq/*.npy 2>/dev/null | wc -l)
    rc=$(ls cache/range/kitti360/$seq/*.npy 2>/dev/null | wc -l)
    echo "  seq$seq: depth=$dc range=$rc"
done

# Start training
echo ""
echo "Starting LC2 multi-sequence training..."
bash run_train_multi.sh
