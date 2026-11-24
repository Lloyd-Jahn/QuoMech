#!/usr/bin/env bash
set -euo pipefail
export PATH="/etc/anaconda3/bin:${PATH}"
export OMP_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=0,1,2,3
cd /home/data/jiangyi/QuoMech
mkdir -p results/phase2/logs results/phase2/models

STEPS="${P2_STEPS:-60000}"

run_quomech() {
    local name="$1"; shift
    local out_dir="results/phase2/models/${name}"
    if [ -f "${out_dir}/train_summary.json" ]; then
        echo "[$(date)] [Slot1] Skipping completed ${name}"
        return 0
    fi
    echo "[$(date)] [Slot1] Starting ${name} (steps=${STEPS})..."
    PYTHONPATH=. torchrun --nproc_per_node=4 --master_port=29611 \
        scripts/p2_stage2_train.py \
        --arch p2_256 --steps "${STEPS}" --bs 512 --grad-accum 1 --lr 8e-4 --warmup 2000 --eval-every 2500 --resume auto \
        --out-dir "${out_dir}" "$@" \
        2>&1 | tee "results/phase2/logs/stage2b_${name}.log"
    echo "[$(date)] [Slot1] Finished ${name}"
}

run_bl() {
    local name="$1"; shift
    local out_dir="results/phase2/models/${name}"
    if [ -f "${out_dir}/train_summary.json" ]; then
        echo "[$(date)] [Slot1] Skipping completed ${name}"
        return 0
    fi
    echo "[$(date)] [Slot1] Starting ${name}..."
    PYTHONPATH=. torchrun --nproc_per_node=4 --master_port=29611 \
        scripts/p2_train_bl_classifier.py \
        --bs 512 --lr 1e-3 --warmup 1500 --eval-every 2500 --resume auto \
        --out-dir "${out_dir}" "$@" \
        2>&1 | tee "results/phase2/logs/stage2b_${name}.log"
    echo "[$(date)] [Slot1] Finished ${name}"
}

# === arXiv v1 Fast-Track (seed=42 complete coverage, ~10.5h total) ===
# 1. Bures & Larrosa closed-set 527-class classifier (seed 42, ~25 min)
run_bl "bl527_s42" --protocol full527 --steps 40000 --seed 42

# 2. E2 OOD 34 held-out 3-module ablation: Factorized head (seed 42, ~5.1h)
run_quomech "e2_ood_factorized_s42" --head factorized --data-dir data/pool_multicore_527 --ood-holdout --seed 42

# 3. E2 Discrete fixed-window (Xi_full, t_end=1.0) ablation (seed 42, ~5.1h)
run_quomech "e2_disc_fixed_s42" --head factorized --data-dir data/pool_multicore_527 --discrete-fixed-window --seed 42

echo "[$(date)] === arXiv v1 Fast-Track Slot 1 (GPU 0-3, seed=42) ALL COMPLETE! ==="
