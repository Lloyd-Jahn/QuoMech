#!/usr/bin/env python3
"""Step 4a: Generate 64-core CPU offline trajectory pool (train/val/test) per Section 7.1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quomech.data_gen import generate_split
from quomech.grammar import build_default_grammar
from quomech.simulator import ODESimulator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train-per-mech", type=int, default=6000)
    ap.add_argument("--n-val-per-mech", type=int, default=100)
    ap.add_argument("--n-test-per-mech", type=int, default=100)
    ap.add_argument("--n-threads", type=int, default=64)
    args = ap.parse_args()

    out_dir = Path(__file__).resolve().parents[1] / "data" / "pool_v1"
    grammar = build_default_grammar()
    sim = ODESimulator(grammar, rtol=1e-5, atol=1e-9, n_threads=args.n_threads)

    metas = {}
    metas["train"] = generate_split(
        grammar, sim, out_dir, "train", n_per_mech=args.n_train_per_mech, n_rand_designs=4, seed=101
    )
    metas["val"] = generate_split(
        grammar, sim, out_dir, "val", n_per_mech=args.n_val_per_mech, n_rand_designs=4, seed=202
    )
    metas["test"] = generate_split(
        grammar, sim, out_dir, "test", n_per_mech=args.n_test_per_mech, n_rand_designs=4, seed=303
    )

    with open(out_dir / "dataset_summary.json", "w") as f:
        json.dump(metas, f, indent=2)
    print("\n=== DATASET GENERATION COMPLETE ===")
    print(json.dumps(metas, indent=2))


if __name__ == "__main__":
    main()
