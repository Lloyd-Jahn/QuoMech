#!/usr/bin/env bash
set -euo pipefail
export PATH="/etc/anaconda3/bin:${PATH}"
export OMP_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=4,5,6,7
cd /home/data/jiangyi/QuoMech
mkdir -p results/phase2/logs results/phase2/models

echo "[$(date)] === Stage 2a Slot 2 (GPU 4-7, NUMA 1): Flat NPE p2_256 Seed 42 (bs/gpu=512, eff_bs=2048, 60k) ==="
PYTHONPATH=. torchrun --nproc_per_node=4 --master_port=29602 \
    scripts/p2_stage2_train.py \
    --head flat \
    --arch p2_256 \
    --data-dir data/pool_multicore_527 \
    --out-dir results/phase2/models/flat_s42 \
    --steps 60000 \
    --bs 512 \
    --grad-accum 1 \
    --lr 8e-4 \
    --warmup 2000 \
    --eval-every 2500 \
    --seed 42 \
    --resume auto \
    2>&1 | tee results/phase2/logs/stage2a_flat_s42.log

echo "[$(date)] === Stage 2a Slot 2 (Flat NPE s42) COMPLETE! ==="
