#!/bin/bash
# Wait for DA3 extraction to finish, then start training
echo "Waiting for DA3 extraction (PID 395083) to finish..."
while kill -0 395083 2>/dev/null; do
    sleep 60
done
echo "DA3 extraction done. Starting VIVID training with DA3 depth..."
sleep 5

cd /home/jhlee/ws_xloc/external/LC2_crossmatching
python train.py \
    --config configs/train_vivid.yaml \
    --output_dir checkpoints/vivid_da3_v1 \
    --device cuda \
    2>&1 | tee logs/vivid_da3_v1_train.log
