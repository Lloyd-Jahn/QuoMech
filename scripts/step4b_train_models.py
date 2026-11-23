#!/usr/bin/env python3
"""Step 4b: Train 1-seed QuoMech (factorized) and 1-seed Flat Softmax NPE models on RTX 3090."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quomech.grammar import build_default_grammar
from quomech.train import train_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", type=str, choices=["factorized", "flat"], required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--steps", type=int, default=35000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--eval-every", type=int, default=2500)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", type=str, default=None, help="'auto' to resume from out_dir/latest_ckpt.pt, or explicit path")
    ap.add_argument("--out-dir", type=str, default=None, help="Custom output directory for checkpoints")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data" / "pool_v1"
    out_dir = Path(args.out_dir) if args.out_dir else (root / "results" / "gate3b" / f"model_{args.head}_s{args.seed}")

    grammar = build_default_grammar()
    summary = train_model(
        grammar=grammar,
        data_dir=data_dir,
        out_dir=out_dir,
        head_type=args.head,
        gpu_id=args.gpu,
        total_steps=args.steps,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        eval_every=args.eval_every,
        num_workers=args.workers,
        seed=args.seed,
        resume=args.resume,
    )
    if summary:
        print(f"\n[Train:{args.head}] COMPLETE! Best val loss={summary['best_val_loss']:.4f}, "
              f"elapsed={summary['elapsed_s']:.1f}s")


if __name__ == "__main__":
    main()
