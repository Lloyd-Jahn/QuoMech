#!/usr/bin/env python3
"""Step 2b: Supplemental fixed-loading control set {D0, D5} for Gate 2 equivalence comparison.

Explicitly compares {D0}, {D0, D5}, Ξ_no-P, Ξ_basic, Ξ_full on:
  - 1st-order / complex deactivation (M2:free1, M2:complex) vs Product inhibition (M4)
  - 2nd-order deactivation (M2:free2) vs Product inhibition (M4)
  - Product inhibition (M4) vs Reversible product release (M7)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from quomech.equivalence import run_two_stage_equivalence
from quomech.grammar import build_default_grammar
from quomech.simulator import ODESimulator
from scripts.pre_phase2_scripts.step2_gate2_equiv import inspect_chem_pairs


def main():
    out_dir = Path(__file__).resolve().parents[1] / "results" / "gate2"
    pool_path = out_dir / "equiv_pools.npz"
    loaded = np.load(pool_path)
    pools = {k: loaded[k] for k in ("lib", "s1", "s2")}

    grammar = build_default_grammar()
    sim = ODESimulator(grammar, rtol=1e-5, atol=1e-9, n_threads=64)

    res = run_two_stage_equivalence(
        grammar=grammar,
        sim=sim,
        design_set_name="D0_D5",
        design_indices=[0, 5],  # D0 + D5 (same excess, fixed 1x catalyst loading)
        pools=pools,
        sigma_ref_base=0.01,
        n_t=20,
        nn_prescreen_thresh=3.0,
        stage1_rms_cut=2.0,
        stage1_min_pass=0.50,
        next_tier_keep_rms=None,
        allowed_pairs=None,
        seed=1002,
    )
    res.pop("next_tier_pairs", None)
    res["chem_inspection"] = inspect_chem_pairs(grammar, res)
    with open(out_dir / "equiv_D0_D5.json", "w") as f:
        json.dump(res, f, indent=2)

    # Print detailed comparison across all 5 design sets
    print("\n=== CROSS-TIER EQUIVALENCE REFINEMENT SUMMARY ===")
    for name in ["D0", "D0_D5", "no_P", "basic", "full"]:
        p = out_dir / f"equiv_{name}.json"
        if not p.exists():
            continue
        with open(p) as f:
            d = json.load(f)
        pr = d["primary"]
        s05 = d["sensitivity"]["sig_0.5pct_pr_85pct"]["n_classes"]
        s20 = d["sensitivity"]["sig_2.0pct_pr_85pct"]["n_classes"]
        print(
            f"  {name:6s} ({len(d['design_names']):2d} des): "
            f"1%={pr['n_classes']:3d} cls ({pr['n_nontrivial_classes']:2d} non-triv, {pr['n_mechs_in_nontrivial']:3d} mechs, max={pr['max_class_size']:2d}, "
            f"non-trans={pr['n_non_transitive_components']}/{pr['n_components_gt2']}) | 0.5%={s05:3d}, 2%={s20:3d}"
        )

    print("\n=== SPOTLIGHT CHEMICAL PAIRS ({D0, D5} vs Ξ_no-P vs Ξ_basic vs Ξ_full) ===")
    key_pairs = [
        "M2:free1 vs M4",
        "M2:complex vs M4",
        "M2:free2 vs M4",
        "M4 vs M7",
        "M2:free1+M4 vs M2:free1+M7",
        "M1+M4 vs M1+M7",
        "M3+M4 vs M3+M7",
        "M2:free1 vs M2:free2",
        "M2:free1 vs M2:complex",
        "M2:free2 vs M5",
        "M5 vs M8",
    ]
    tiers_loaded = {}
    for name in ["D0", "D0_D5", "no_P", "basic", "full"]:
        p = out_dir / f"equiv_{name}.json"
        if p.exists():
            with open(p) as f:
                tiers_loaded[name] = json.load(f)

    header = f"  {'Pair':28s} | " + " | ".join(f"{t:8s}" for t in tiers_loaded)
    print(header)
    print("  " + "-" * len(header))
    for pair in key_pairs:
        cols = []
        for t, d in tiers_loaded.items():
            info = d["chem_inspection"]["spotlight_pairs"].get(pair, {})
            if info.get("same_class"):
                tag = "MERGED"
            elif info.get("direct_edge"):
                tag = "EDGE*"
            else:
                tag = "split"
            cols.append(f"{tag:8s}")
        print(f"  {pair:28s} | " + " | ".join(cols))

    # Also count how many multi-module pairs of type (M2:free1/complex vs M4), (M2:free2 vs M4), (M4 vs M7) merge!
    print("\n=== FAMILY-WIDE MERGE COUNTS (same_class across all 151 mechs, 1% >=85%) ===")
    mopts = grammar.mech_options
    for t, d in tiers_loaded.items():
        m2c = d["primary"]["mech_to_class"]
        c_m2f1_m4 = 0
        c_m2c_m4 = 0
        c_m2f2_m4 = 0
        c_m4_m7 = 0
        for i in range(grammar.n_mechs):
            for j in range(i + 1, grammar.n_mechs):
                if m2c[i] != m2c[j]:
                    continue
                oi, oj = mopts[i], mopts[j]
                # Check if they differ ONLY in M2 vs M4 (i has M2, j has M4 or vice versa)
                other_same_24 = all(oi[k] == oj[k] for k in range(grammar.n_modules) if k not in (1, 3))
                if other_same_24:
                    if (oi[1] == 1 and oi[3] == 0 and oj[1] == 0 and oj[3] == 1) or (oj[1] == 1 and oj[3] == 0 and oi[1] == 0 and oi[3] == 1):
                        c_m2f1_m4 += 1
                    if (oi[1] == 3 and oi[3] == 0 and oj[1] == 0 and oj[3] == 1) or (oj[1] == 3 and oj[3] == 0 and oi[1] == 0 and oi[3] == 1):
                        c_m2c_m4 += 1
                    if (oi[1] == 2 and oi[3] == 0 and oj[1] == 0 and oj[3] == 1) or (oj[1] == 2 and oj[3] == 0 and oi[1] == 0 and oi[3] == 1):
                        c_m2f2_m4 += 1
                other_same_47 = all(oi[k] == oj[k] for k in range(grammar.n_modules) if k not in (3, 6))
                if other_same_47:
                    if (oi[3] == 1 and oi[6] == 0 and oj[3] == 0 and oj[6] == 1) or (oj[3] == 1 and oj[6] == 0 and oi[3] == 0 and oi[6] == 1):
                        c_m4_m7 += 1
        print(f"  {t:6s}: (M2:free1 vs M4)={c_m2f1_m4:2d}, (M2:complex vs M4)={c_m2c_m4:2d}, "
              f"(M2:free2 vs M4)={c_m2f2_m4:2d}, (M4 vs M7)={c_m4_m7:2d}")


if __name__ == "__main__":
    main()
