#!/usr/bin/env bash
set -euo pipefail
export PATH="/etc/anaconda3/bin:${PATH}"
export OMP_NUM_THREADS=4
export MPLCONFIGDIR="/tmp/mpl_quomech"
cd /home/data/jiangyi/QuoMech
mkdir -p results/phase2/logs results/phase2/figures

SEEDS="${P2_EVAL_SEEDS:-42}"
GPU="${P2_EVAL_GPU:-0}"

echo "[$(date)] === Phase 2 Stage 3-5 Full Evaluation Pipeline (seeds=${SEEDS}, gpu=${GPU}) ==="

# 1. Stage 3 E1 + BL-20 + E2 Ablations (5,270 test cases)
echo "[$(date)] [1/4] Running E1 + BL-20 + E2 Ablations..."
PYTHONPATH=. python3 scripts/p2_stage3_e1_e2.py \
    --seeds "${SEEDS}" --gpu "${GPU}" --n-threads 64 \
    2>&1 | tee results/phase2/logs/stage3_e1_e2.log

# 2. Stage 3 E3 Quotient BOED + Continuous LHS-128 + Cost/Upgrade + C_macro (527 cases)
echo "[$(date)] [2/4] Running E3 Quotient BOED + C_macro..."
PYTHONPATH=. python3 scripts/p2_stage3_e3_boed.py \
    --seeds "${SEEDS}" --gpu "${GPU}" --n-threads 64 \
    2>&1 | tee results/phase2/logs/stage3_e3_boed.log

# 3. Stage 4 E4 Real Literature (11 systems) + 3x100 Synthetic Misspecification Stress Tests
echo "[$(date)] [3/4] Running E4 Real Literature & Misspecification Stress Tests..."
PYTHONPATH=. python3 scripts/p2_stage4_e4_full.py \
    --ckpt results/phase2/models/quomech_s42/best.pt \
    --equiv-basic-json results/phase2/equiv/equiv_Xi_basic.json \
    --gpu "${GPU}" --n-threads 64 \
    2>&1 | tee results/phase2/logs/stage4_e4_full.log

# 4. Stage 5 Publication Figures (5 PDF+PNG) + Full Markdown Report
echo "[$(date)] [4/4] Generating 5 Publication Figures & Full Report..."
PYTHONPATH=. python3 scripts/p2_make_figures_and_report.py \
    --res-dir results/phase2 \
    2>&1 | tee results/phase2/logs/stage5_figures_report.log

echo "[$(date)] === ALL STAGE 3-5 EVALUATIONS, FIGURES & REPORT COMPLETE! ==="
