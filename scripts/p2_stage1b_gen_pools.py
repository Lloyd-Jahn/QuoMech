#!/usr/bin/env python3
"""Phase 2 Stage 1b: Generate both 527-mechanism offline trajectory pools
  1) data/pool_multicore_527/            (filtered: train 6000/mech, val 100/mech, test 100/mech)
  2) data/pool_multicore_527_unfiltered/ (unfiltered: train 6000/mech, D0 conv 25-98% + solver ok)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import numpy as np
from scipy.stats import chi2_contingency

from quomech.data_gen import generate_split
from quomech.grammar import build_multicore_grammar
from quomech.simulator import ODESimulator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c3-branch-min", type=float, default=0.15)
    ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--n-val", type=int, default=100)
    ap.add_argument("--n-test", type=int, default=100)
    ap.add_argument("--n-threads", type=int, default=64)
    ap.add_argument("--skip-filtered", action="store_true")
    ap.add_argument("--skip-unfiltered", action="store_true")
    args = ap.parse_args()

    g = build_multicore_grammar()
    sim = ODESimulator(g, n_threads=args.n_threads)
    t_all = time.perf_counter()
    summary = {
        "c3_branch_min": args.c3_branch_min,
        "n_mechs": g.n_mechs,
        "n_train_per_mech": args.n_train,
        "n_val_per_mech": args.n_val,
        "n_test_per_mech": args.n_test,
        "pools": {},
    }

    if not args.skip_filtered:
        out_filt = Path("data/pool_multicore_527")
        print(f"=== Generating Filtered Pool: {out_filt} (c3_branch_min={args.c3_branch_min}) ===", flush=True)
        m_val = generate_split(
            g, sim, out_filt, "val", n_per_mech=args.n_val, seed=2001,
            unfiltered=False, c3_branch_min=args.c3_branch_min,
        )
        m_test = generate_split(
            g, sim, out_filt, "test", n_per_mech=args.n_test, seed=3001,
            unfiltered=False, c3_branch_min=args.c3_branch_min,
        )
        m_train = generate_split(
            g, sim, out_filt, "train", n_per_mech=args.n_train, seed=1001,
            unfiltered=False, c3_branch_min=args.c3_branch_min,
        )
        summary["pools"]["filtered"] = {"train": m_train, "val": m_val, "test": m_test}

    if not args.skip_unfiltered:
        out_unfilt = Path("data/pool_multicore_527_unfiltered")
        print(f"\n=== Generating Unfiltered Pool: {out_unfilt} ===", flush=True)
        m_train_u = generate_split(
            g, sim, out_unfilt, "train", n_per_mech=args.n_train, seed=4001,
            unfiltered=True, c3_branch_min=args.c3_branch_min,
        )
        summary["pools"]["unfiltered"] = {"train": m_train_u}

    out_meta = Path("results/phase2/stage1b_pools_summary.json")
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    if out_meta.exists():
        prev = json.loads(out_meta.read_text())
        merged_pools = prev.get("pools", {})
        merged_pools.update(summary["pools"])
        summary["pools"] = merged_pools
    summary["elapsed_total_s"] = time.perf_counter() - t_all
    with open(out_meta, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[Stage 1b DONE] Total elapsed: {summary['elapsed_total_s']/60:.1f} min -> {out_meta}", flush=True)


if __name__ == "__main__":
    main()
