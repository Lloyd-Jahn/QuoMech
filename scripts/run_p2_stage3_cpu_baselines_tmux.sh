#!/usr/bin/env bash
set -euo pipefail
export PATH="/etc/anaconda3/bin:${PATH}"
export OMP_NUM_THREADS=4
cd /home/data/jiangyi/QuoMech
mkdir -p results/phase2/logs

echo "[$(date)] === Phase 2 Stage 3 CPU Classical Baselines (64-core CPU, zero GPU contention) ==="
PYTHONPATH=. python3 scripts/p2_stage3_cpu_baselines.py \
    --n-threads 64 \
    --n-bic-cases 200 \
    --n-smc-cases 50 \
    --n-smc-particles 300 \
    --n-sindy-cases 100 \
    --c3-branch-min 0.10 \
    --nn-device cpu \
    2>&1 | tee results/phase2/logs/stage3_cpu_baselines.log
echo "[$(date)] === Stage 3 CPU Baselines ALL COMPLETE! ==="
