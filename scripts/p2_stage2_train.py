#!/usr/bin/env python3
"""Phase 2 Stage 2a/2b unified DDP launcher for QuoMech, Flat NPE, and E2 ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from quomech.grammar import build_multicore_grammar
from quomech.train import train_model


def get_ood_heldout_3mod_ids(grammar, n_holdout: int = 34, seed: int = 2026) -> np.ndarray:
    """Deterministically select 34 3-module combinations stratified across all 4 cores (~10% of 334)."""
    rng = np.random.default_rng(seed)
    k_active = np.sum(grammar.mech_options > 0, axis=1)
    # Stratified quotas proportional to 3-mod counts (98, 98, 98, 40 -> 10, 10, 10, 4 = 34!)
    quotas = {0: 10, 1: 10, 2: 10, 3: 4}
    held = []
    for c_id, q in quotas.items():
        cands = np.where((grammar.mech_core_id == c_id) & (k_active == 3))[0]
        chosen = rng.choice(cands, size=q, replace=False)
        held.extend(chosen.tolist())
    held_arr = np.sort(np.array(held, dtype=np.int64))
    assert len(held_arr) == n_holdout
    return held_arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", type=str, default="factorized", choices=["factorized", "flat"])
    ap.add_argument("--arch", type=str, default="p2_256", choices=["p1_192", "p2_256"])
    ap.add_argument("--data-dir", type=str, default="data/pool_multicore_527")
    ap.add_argument("--val-data-dir", type=str, default=None)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--eval-every", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", type=str, default="auto")
    ap.add_argument("--discrete-fixed-window", action="store_true")
    ap.add_argument("--ood-holdout", action="store_true", help="Hold out 34 stratified 3-module mechs for E2 OOD ablation")
    args = ap.parse_args()

    g = build_multicore_grammar()
    allowed_mech_ids = None
    if args.ood_holdout:
        held = get_ood_heldout_3mod_ids(g, n_holdout=34, seed=2026)
        allowed_mech_ids = np.array([m for m in range(g.n_mechs) if m not in set(held.tolist())], dtype=np.int64)
        out_p = Path(args.out_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        with open(out_p / "ood_split.json", "w") as f:
            json.dump(
                {
                    "n_total": g.n_mechs,
                    "n_train_mechs": len(allowed_mech_ids),
                    "n_ood_mechs": len(held),
                    "ood_mech_ids": held.tolist(),
                    "ood_mech_names": [g.mech_names[m] for m in held.tolist()],
                },
                f,
                indent=2,
            )

    train_model(
        grammar=g,
        data_dir=Path(args.data_dir),
        val_data_dir=Path(args.val_data_dir) if args.val_data_dir else None,
        out_dir=Path(args.out_dir),
        head_type=args.head,
        arch=args.arch,
        total_steps=args.steps,
        batch_size=args.bs,
        grad_accum=args.grad_accum,
        lr=args.lr,
        warmup_steps=args.warmup,
        eval_every=args.eval_every,
        seed=args.seed,
        resume=args.resume,
        discrete_fixed_window=args.discrete_fixed_window,
        allowed_mech_ids=allowed_mech_ids,
    )


if __name__ == "__main__":
    main()
