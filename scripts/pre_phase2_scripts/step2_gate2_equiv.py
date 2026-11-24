#!/usr/bin/env python3
"""Step 2 (Gate 2): Network-independent two-stage ε-equivalence class construction.

Evaluates {D0}, Ξ_no-P, Ξ_basic, Ξ_full at primary σ_ref=1% (with 0.5% and 2% sensitivity)
plus analytic verification of Proposition 1 (catalyst-order non-additivity).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from quomech.equivalence import run_two_stage_equivalence, sample_mechanism_pools
from quomech.grammar import build_default_grammar
from quomech.simulator import ODESimulator


def verify_analytic_non_additivity(out_dir: Path) -> dict:
    """Analytic calibration check of Proposition 1: rate = k * [cat]^n * [A], n in {1, 2}.
    Under a single catalyst loading c1=0.05, n=1 and n=2 are EXACTLY equivalent (RMS = 0).
    Under two loadings {c1=0.05, c2=0.10}, ratio of decay constants is 2^n -> strictly separated.
    """
    rng = np.random.default_rng(7)
    t = np.linspace(0.0, 1.0, 20)
    sigma_ref = 0.01
    n_test = 32

    # n=1: A(t) = exp(-k1 * c * t), k1 ~ 10^N(1.3, 0.3^2) => a1 = k1*c ~ 1.0
    # n=2: A(t) = exp(-k2 * c^2 * t), k2 ~ 10^N(2.6, 0.3^2) => a2 = k2*c^2 ~ 1.0
    k1_samples = 10.0 ** rng.normal(1.3, 0.3, size=n_test)
    k2_samples = 10.0 ** rng.normal(2.6, 0.3, size=n_test)

    c1, c2 = 0.05, 0.10
    # Single loading c1: exact fit exists -> k2_fit = k1 / c1, k1_fit = k2 * c1
    rms_single_1to2 = []
    rms_single_2to1 = []
    for k1, k2 in zip(k1_samples, k2_samples):
        y1 = np.exp(-k1 * c1 * t)
        y1_fit = np.exp(-(k1 / c1) * (c1 ** 2) * t)
        rms_single_1to2.append(float(np.sqrt(np.mean(((y1_fit - y1) / sigma_ref) ** 2))))
        y2 = np.exp(-k2 * (c1 ** 2) * t)
        y2_fit = np.exp(-(k2 * c1) * c1 * t)
        rms_single_2to1.append(float(np.sqrt(np.mean(((y2_fit - y2) / sigma_ref) ** 2))))

    # Two loadings {c1, c2}: minimize ||exp(-k2*c^2*t) - exp(-k1*c*t)||^2 over k2 > 0
    from scipy.optimize import minimize_scalar
    rms_pair_1to2 = []
    rms_pair_2to1 = []
    for k1, k2 in zip(k1_samples, k2_samples):
        y1 = np.concatenate([np.exp(-k1 * c1 * t), np.exp(-k1 * c2 * t)])
        res = minimize_scalar(
            lambda lk2: np.mean(((np.concatenate([np.exp(-(10**lk2)*c1**2*t), np.exp(-(10**lk2)*c2**2*t)]) - y1) / sigma_ref) ** 2),
            bounds=(0.0, 5.0), method="bounded"
        )
        rms_pair_1to2.append(float(np.sqrt(res.fun)))

        y2 = np.concatenate([np.exp(-k2 * c1**2 * t), np.exp(-k2 * c2**2 * t)])
        res2 = minimize_scalar(
            lambda lk1: np.mean(((np.concatenate([np.exp(-(10**lk1)*c1*t), np.exp(-(10**lk1)*c2*t)]) - y2) / sigma_ref) ** 2),
            bounds=(-1.0, 4.0), method="bounded"
        )
        rms_pair_2to1.append(float(np.sqrt(res2.fun)))

    return {
        "single_loading_c0.05": {
            "pass_1to2": float(np.mean(np.array(rms_single_1to2) <= 1.0)),
            "pass_2to1": float(np.mean(np.array(rms_single_2to1) <= 1.0)),
            "max_rms": max(max(rms_single_1to2), max(rms_single_2to1)),
        },
        "two_loadings_0.05_0.10": {
            "pass_1to2": float(np.mean(np.array(rms_pair_1to2) <= 1.0)),
            "pass_2to1": float(np.mean(np.array(rms_pair_2to1) <= 1.0)),
            "median_rms_1to2": float(np.median(rms_pair_1to2)),
            "median_rms_2to1": float(np.median(rms_pair_2to1)),
        },
    }


def inspect_chem_pairs(grammar, result: dict) -> dict:
    """Inspect specific chemically important mechanism pairs and largest merged classes."""
    p = result["primary"]
    m2c = p["mech_to_class"]
    edge_map = {}
    for e in p["passing_edges"]:
        edge_map[(e["i"], e["j"])] = e
        edge_map[(e["j"], e["i"])] = e

    spotlight_names = [
        ("M2:free1", "M4"),
        ("M2:free2", "M4"),
        ("M2:complex", "M4"),
        ("M4", "M7"),
        ("M2:free1", "M2:free2"),
        ("M2:free1", "M2:complex"),
        ("M2:free2", "M2:complex"),
        ("M2:free2", "M5"),
        ("M5", "M8"),
        ("Core", "M4"),
        ("Core", "M7"),
        ("Core", "M2:free1"),
        ("M2:free1+M4", "M2:free1+M7"),
        ("M1+M4", "M1+M7"),
        ("M3+M4", "M3+M7"),
        ("M2:free1+M5", "M2:free2+M5"),
    ]
    spotlight = {}
    for n1, n2 in spotlight_names:
        i = grammar.get_mech_id_by_name(n1)
        j = grammar.get_mech_id_by_name(n2)
        same_cls = (m2c[i] == m2c[j])
        e = edge_map.get((i, j))
        spotlight[f"{n1} vs {n2}"] = {
            "same_class": same_cls,
            "direct_edge": e is not None,
            "pass_rates": (e["pass_i2j"], e["pass_j2i"]) if e else None,
            "mean_rms_1pct": e["mean_rms_1pct"] if e else None,
        }

    top_classes = [
        {
            "class_id": cid,
            "size": len(members),
            "members": [grammar.mech_names[m] for m in members],
        }
        for cid, members in enumerate(p["classes"])
        if len(members) >= 2
    ][:25]

    return {"spotlight_pairs": spotlight, "top_nontrivial_classes": top_classes}


def main():
    out_dir = Path(__file__).resolve().parents[1] / "results" / "gate2"
    out_dir.mkdir(parents=True, exist_ok=True)

    grammar = build_default_grammar()
    sim = ODESimulator(grammar, rtol=1e-5, atol=1e-9, n_threads=64)

    print("[Gate 2] Analytic check of Proposition 1 (catalyst order non-additivity)...")
    analytic_check = verify_analytic_non_additivity(out_dir)
    print(json.dumps(analytic_check, indent=2))

    pool_path = out_dir / "equiv_pools.npz"
    if pool_path.exists():
        print(f"\n[Gate 2] Loading cached accepted parameter pools from {pool_path}...")
        loaded = np.load(pool_path)
        pools = {k: loaded[k] for k in ("lib", "s1", "s2")}
    else:
        print("\n[Gate 2] Sampling independent accepted parameter pools (151 mechs x (128 lib + 8 S1 + 32 S2))...")
        t0 = time.perf_counter()
        pools = sample_mechanism_pools(grammar, sim, n_lib=128, k1=8, k2=32, seed=2026)
        print(f"[Gate 2] Pools sampled in {time.perf_counter() - t0:.1f}s.")
        np.savez_compressed(pool_path, **pools)

    # Run two-stage ε-equivalence for the 3 nested laboratory catalogs first: no_P ⊂ basic ⊂ full,
    # using the exact sum-of-squares lower bound RMS(D_sup) >= sqrt(n_ch_sub / n_ch_sup) * RMS(D_sub)
    # with 1.25x safety margin (plus union with target-tier NN_med <= 3.0) -> zero false negatives.
    # Then run {D0} independently.
    catalog_chain = ["no_P", "basic", "full"]
    tier_order = ["D0", "no_P", "basic", "full"]
    tier_n_ch = {
        t: sum(len(grammar.catalog[i].obs_species) for i in grammar.tiers[t])
        for t in tier_order
    }
    all_results = {"analytic_check": analytic_check, "tiers": {}}
    carried_pairs = None

    for t_idx, tier in enumerate(catalog_chain):
        design_indices = grammar.tiers[tier]
        if t_idx + 1 < len(catalog_chain):
            next_tier = catalog_chain[t_idx + 1]
            ratio = float(np.sqrt(tier_n_ch[next_tier] / tier_n_ch[tier]))
            next_keep = 2.0 * ratio * 1.25
        else:
            next_keep = None

        res = run_two_stage_equivalence(
            grammar=grammar,
            sim=sim,
            design_set_name=tier,
            design_indices=design_indices,
            pools=pools,
            sigma_ref_base=0.01,
            n_t=20,
            nn_prescreen_thresh=3.0,
            stage1_rms_cut=2.0,
            stage1_min_pass=0.50,
            next_tier_keep_rms=next_keep,
            allowed_pairs=carried_pairs,
            seed=1000 + len(design_indices),
        )
        carried_pairs = res.pop("next_tier_pairs")
        res["chem_inspection"] = inspect_chem_pairs(grammar, res)
        all_results["tiers"][tier] = res
        with open(out_dir / f"equiv_{tier}.json", "w") as f:
            json.dump(res, f, indent=2)

    # Finally run {D0} (single-experiment baseline): confirm all pairs passing <=1.0 in Stage 2
    # (and also <=2.0 if <=2500 pairs; otherwise compute 2.0% sensitivity from Stage 1 + Stage 2).
    res_d0 = run_two_stage_equivalence(
        grammar=grammar,
        sim=sim,
        design_set_name="D0",
        design_indices=grammar.tiers["D0"],
        pools=pools,
        sigma_ref_base=0.01,
        n_t=20,
        nn_prescreen_thresh=3.0,
        stage1_rms_cut=1.2,   # covers 0.5% and 1.0% with margin without exploding into 9000 trivial 2% D0 pairs
        stage1_min_pass=0.50,
        next_tier_keep_rms=None,
        allowed_pairs=None,
        seed=1001,
    )
    res_d0.pop("next_tier_pairs", None)
    res_d0["chem_inspection"] = inspect_chem_pairs(grammar, res_d0)
    all_results["tiers"]["D0"] = res_d0
    with open(out_dir / "equiv_D0.json", "w") as f:
        json.dump(res_d0, f, indent=2)

    # Save compact summary across all 4 tiers
    summary = {
        "analytic_check": analytic_check,
        "tier_summary": {},
    }
    for tier in tier_order:
        r = all_results["tiers"][tier]
        p = r["primary"]
        summary["tier_summary"][tier] = {
            "design_names": r["design_names"],
            "n_classes": p["n_classes"],
            "n_singleton": p["n_singleton"],
            "n_nontrivial_classes": p["n_nontrivial_classes"],
            "n_mechs_in_nontrivial": p["n_mechs_in_nontrivial"],
            "max_class_size": p["max_class_size"],
            "size_histogram": p["size_histogram"],
            "n_passing_edges": p["n_passing_edges"],
            "n_components_gt2": p["n_components_gt2"],
            "n_non_transitive_components": p["n_non_transitive_components"],
            "non_transitive_fraction_of_gt2": p["non_transitive_fraction_of_gt2"],
            "sensitivity": r["sensitivity"],
            "elapsed_total_s": r["elapsed_total_s"],
            "spotlight_pairs": r["chem_inspection"]["spotlight_pairs"],
            "top_nontrivial_classes": r["chem_inspection"]["top_nontrivial_classes"][:15],
        }

    with open(out_dir / "gate2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== GATE 2 SUMMARY ===")
    for tier in tier_order:
        ts = summary["tier_summary"][tier]
        print(
            f"  {tier:6s} ({len(ts['design_names']):2d} designs): "
            f"{ts['n_classes']:3d} classes | {ts['n_nontrivial_classes']:2d} non-trivial "
            f"({ts['n_mechs_in_nontrivial']:3d} mechs, max={ts['max_class_size']:2d}) | "
            f"non-trans={ts['n_non_transitive_components']}/{ts['n_components_gt2']} | "
            f"0.5%={ts['sensitivity']['sig_0.5pct_pr_85pct']['n_classes']} cls, "
            f"2.0%={ts['sensitivity']['sig_2.0pct_pr_85pct']['n_classes']} cls | "
            f"time={ts['elapsed_total_s']:.1f}s"
        )


if __name__ == "__main__":
    main()
