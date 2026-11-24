#!/usr/bin/env bash
set -euo pipefail
export PATH="/etc/anaconda3/bin:${PATH}"
cd /home/data/jiangyi/QuoMech
mkdir -p results/phase2/logs

echo "[$(date)] === Stage 1b-Part 1: Filtered Pool (data/pool_multicore_527, c3_branch_min=0.10) ==="
PYTHONPATH=. python scripts/p2_stage1b_gen_pools.py \
    --c3-branch-min 0.10 \
    --n-train 6000 \
    --n-val 100 \
    --n-test 100 \
    --n-threads 64 \
    --skip-unfiltered \
    2>&1 | tee results/phase2/logs/stage1b_filtered.log

touch data/pool_multicore_527/.READY
echo "[$(date)] === Filtered Pool READY! Starting Stage 1b-Part 2: Unfiltered Pool ==="

PYTHONPATH=. python scripts/p2_stage1b_gen_pools.py \
    --c3-branch-min 0.10 \
    --n-train 6000 \
    --n-val 100 \
    --n-test 100 \
    --n-threads 64 \
    --skip-filtered \
    2>&1 | tee results/phase2/logs/stage1b_unfiltered.log

touch data/pool_multicore_527_unfiltered/.READY
echo "[$(date)] === Both Pools READY! Starting Stage 1c (E0): 6-Set Equivalence Classes ==="

PYTHONPATH=. python scripts/p2_stage1c_equiv.py \
    --c3-branch-min 0.10 \
    --n-threads 64 \
    --n-lib 64 \
    --k1 8 \
    --k2 32 \
    --seed 2026 \
    2>&1 | tee results/phase2/logs/stage1c_equiv.log

echo "[$(date)] === CPU Pipeline (Stage 1b + Stage 1c) COMPLETE! ==="
