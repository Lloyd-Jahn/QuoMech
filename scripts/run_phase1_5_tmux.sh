#!/usr/bin/env bash
set -euo pipefail

export PATH="/etc/anaconda3/bin:${PATH}"
cd /home/data/jiangyi/QuoMech
mkdir -p results/phase1_5/eval_out

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Step 1: 8-GPU 60,000-step QuoMech Training (Pre-LN KV-Norm + QK-Norm, lr=8e-4) ===" | tee -a results/phase1_5/train_60k.log
/etc/anaconda3/bin/torchrun \
  --nproc_per_node=8 \
  --master_port=29541 \
  /home/data/jiangyi/QuoMech/scripts/step4b_train_models.py \
  --head factorized \
  --steps 60000 \
  --batch-size 256 \
  --grad-accum 1 \
  --lr 8e-4 \
  --eval-every 5000 \
  --workers 0 \
  --seed 42 \
  --resume auto \
  --out-dir /home/data/jiangyi/QuoMech/results/phase1_5/model_factorized_60k_s42 \
  2>&1 | tee -a results/phase1_5/train_60k.log

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Step 2: Running Full Phase 1.5 Tasks 1-4 Evaluation ===" | tee -a results/phase1_5/eval_all.log
/etc/anaconda3/bin/python3 -u \
  /home/data/jiangyi/QuoMech/scripts/phase1_5_eval.py \
  --gpu 0 \
  --model-dir /home/data/jiangyi/QuoMech/results/phase1_5/model_factorized_60k_s42 \
  --out-dir /home/data/jiangyi/QuoMech/results/phase1_5/eval_out \
  --n-test-e1 1510 \
  --n-per-mech-t2 2 \
  --n-mc-t2 512 \
  --n-cases-t3 151 \
  --n-lhs-t3 128 \
  --n-mc-t3 256 \
  --n-mc-t4 512 \
  2>&1 | tee -a results/phase1_5/eval_all.log

date '+%Y-%m-%d %H:%M:%S' > /home/data/jiangyi/QuoMech/results/phase1_5/PHASE1_5_DONE
echo "[$(date '+%Y-%m-%d %H:%M:%S')] === ALL PHASE 1.5 TRAINING & EVALUATIONS COMPLETE ===" | tee -a results/phase1_5/eval_all.log
