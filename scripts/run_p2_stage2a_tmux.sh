#!/usr/bin/env bash
set -euo pipefail
export PATH="/etc/anaconda3/bin:${PATH}"
export OMP_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=0,1,2,3
cd /home/data/jiangyi/QuoMech
mkdir -p results/phase2/logs results/phase2/models

echo "[$(date)] === Stage 2a Slot 1 (GPU 0-3, NUMA 0): QuoMech p2_256 Seed 42 (bs/gpu=512, eff_bs=2048, 60k) ==="
PYTHONPATH=. torchrun --nproc_per_node=4 --master_port=29601 \
    scripts/p2_stage2_train.py \
    --head factorized \
    --arch p2_256 \
    --data-dir data/pool_multicore_527 \
    --out-dir results/phase2/models/quomech_s42 \
    --steps 60000 \
    --bs 512 \
    --grad-accum 1 \
    --lr 8e-4 \
    --warmup 2000 \
    --eval-every 2500 \
    --seed 42 \
    --resume auto \
    2>&1 | tee results/phase2/logs/stage2a_quomech_s42.log

echo "[$(date)] === Stage 2a Smoke Test: E4 Real Literature (11 published datasets) ==="
EQUIV_ARG=""
if [ -f results/phase2/equiv/equiv_Xi_basic.json ]; then
    EQUIV_ARG="--equiv-json results/phase2/equiv/equiv_Xi_basic.json"
fi
PYTHONPATH=. python scripts/p2_e4_real_eval.py \
    --ckpt results/phase2/models/quomech_s42/best.pt \
    --arch p2_256 \
    --head factorized \
    $EQUIV_ARG \
    --data-dir data/real_literature \
    --out-json results/phase2/e4_smoke_s42.json \
    --gpu 0 \
    2>&1 | tee results/phase2/logs/stage2a_e4_smoke_s42.log

echo "[$(date)] === Stage 2a Slot 1 (QuoMech s42 + E4 Smoke) COMPLETE! ==="
