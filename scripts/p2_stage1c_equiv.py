#!/usr/bin/env python3
"""Phase 2 Stage 1c (E0): Build exact two-stage ε-equivalence classes for all 527 mechanisms
across 6 design sets on 64-core CPU (runs in parallel with 8x RTX 3090 Stage 2a):
  Reverse-nested discrete chain (with 0-iter forward inheritance):
    1) Xi_full  (D0-D9)
    2) Xi_basic (D0-D6)
    3) Xi_noP   (D0-D5)
    4) D0_D5    ({D0, D5})
    5) D0       ({D0})
  Independent continuous reference set:
    6) X_basic_ref (9 extreme-coverage continuous designs with 0.25x/4x cat & t_end=2.5)

Also writes results/phase2/stage1_multicore_summary.md.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import numpy as np

from quomech.equivalence import (
    build_x_basic_ref_designs,
    run_two_stage_equivalence_multicore,
    sample_mechanism_pools,
)
from quomech.grammar import build_multicore_grammar
from quomech.simulator import ODESimulator


def write_stage1_summary(out_dir: Path, results: dict, c3_branch_min: float):
    lines = [
        "# QuoMech Phase 2 Stage 1 Summary: 4-Core 527-Mechanism Library & Quotient Hierarchy (E0)",
        "",
        f"- **Library**: 4 catalytic cores (`C1` Ordered A-first: 151, `C2` Ordered B-first: 151, `C3` Random-order: 151, `C4` Eley–Rideal: 74) × 8 optional modules (`k ≤ 3`) = **527 mechanisms** (13 species, 24 elementary reactions, 19 parameter slots).",
        f"- **C3 dual-branch threshold**: `{c3_branch_min:.2f}`.",
        "",
        "## 1. Stage 1a Probe & Stage 1b Pools",
        "",
    ]
    probe_path = out_dir / "stage1a_probe.json"
    if probe_path.exists():
        pr = json.loads(probe_path.read_text())
        rc = pr["radau_check"]
        f_key = "filtered_c3_10" if (abs(c3_branch_min - 0.10) < 1e-6 and pr.get("filtered_c3_10")) else "filtered_c3_15"
        fp = pr[f_key]
        lines.extend([
            f"- **ROS2 vs SciPy Radau (`1e-10`)** on {rc['n_samples']} multicore samples: `max_err = {rc['max_abs_err']:.2e}`, `p99_err = {rc['p99_abs_err']:.2e}`, `max_atom_err = {rc['max_atom_err']:.2e}`, `ros2_fails = {rc['ros2_fails']}`.",
            f"- **Filtered acceptance** (all 527 mechs): median `{fp['overall_median']*100:.2f}%`, min `{fp['overall_min']*100:.2f}%`, `<3%` count = `{fp['n_below_3pct']}`, solver fails = `{fp['total_solver_fails']}`.",
            f"- **Unfiltered acceptance** (D0 conv 25–98%): median `{pr['unfiltered']['overall_median']*100:.2f}%`, min `{pr['unfiltered']['overall_min']*100:.2f}%`.",
            "",
        ])

    pools_path = out_dir / "stage1b_pools_summary.json"
    if pools_path.exists():
        ps = json.loads(pools_path.read_text())
        lines.append("### Offline Trajectory Pools (`t_max=4.0`, 80-pt early-log-dense, 14 designs/sample, fp16)")
        lines.append("")
        lines.append("| Pool | Split | Samples | Size (GB) | Sim Speed (traj/s) | Rej Fails | Sim Fails |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for pname, splits in ps.get("pools", {}).items():
            for sname, sm in splits.items():
                lines.append(
                    f"| `{pname}` | `{sname}` | {sm['n_samples']:,} | {sm['size_gb']:.2f} | "
                    f"{sm['sim_traj_per_sec']:,.0f} | {sm['total_rej_solver_fails']} | {sm['total_sim_solver_fails']} |"
                )
        lines.append("")

    lines.extend([
        "## 2. E0 Quotient Hierarchy Across 6 Design Sets (`σ_ref = 1%`, pass rate `≥ 85%`)",
        "",
        "| Design Set | #Designs | Classes | Singletons | Non-Trivial Classes (Mechs) | Max Size | Within / Cross-Core Edges | Cross-Core Classes | 0-Iter Inherited | Elapsed (s) |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|",
    ])
    order = ["D0", "D0_D5", "Xi_noP", "Xi_basic", "X_basic_ref", "Xi_full"]
    for name in order:
        if name not in results:
            continue
        r = results[name]
        p = r["primary"]
        cm = r["core_merge_stats"]
        lines.append(
            f"| `{name}` | {r['n_designs']} | **{p['n_classes']}** | {p['n_singleton']} | "
            f"{p['n_nontrivial_classes']} ({p['n_mechs_in_nontrivial']}) | {p['max_class_size']} | "
            f"{cm['n_within_core_edges']} / {cm['n_cross_core_edges']} | {cm['n_cross_core_classes']} | "
            f"{r['n_inherited_exact_all']}/{r['n_inherited_input']} | {r['elapsed_total_s']:.1f} |"
        )
    lines.append("")

    lines.extend([
        "## 3. Sensitivity Analysis (`σ_ref ∈ {0.5%, 1%, 2%}` × pass rate `∈ {80%, 85%, 90%}`)",
        "",
        "| Design Set | 0.5% (85%) | 1.0% (80% / **85%** / 90%) | 2.0% (85%) | Non-Trans (1%, 85%) |",
        "|---|---:|---|---:|---:|",
    ])
    for name in order:
        if name not in results:
            continue
        s = results[name]["sensitivity"]
        p = results[name]["primary"]
        lines.append(
            f"| `{name}` | {s['sig_0.5pct_pr_85pct']['n_classes']} | "
            f"{s['sig_1.0pct_pr_80pct']['n_classes']} / **{s['sig_1.0pct_pr_85pct']['n_classes']}** / "
            f"{s['sig_1.0pct_pr_90pct']['n_classes']} | "
            f"{s['sig_2.0pct_pr_85pct']['n_classes']} | "
            f"{p['n_non_transitive_components']}/{p['n_components_gt2']} |"
        )
    lines.append("")

    lines.append("## 4. Core-Pair Equivalence Edge Matrix (`σ_ref = 1%`, `≥ 85%`)")
    lines.append("")
    for name in ["D0", "Xi_basic", "X_basic_ref", "Xi_full"]:
        if name not in results:
            continue
        cm = results[name]["core_merge_stats"]
        codes = cm["core_codes"]
        mat = cm["edge_matrix"]
        lines.append(f"### `{name}`")
        lines.append("| | " + " | ".join(codes) + " |")
        lines.append("|---|" + "---:|" * len(codes))
        for i, c in enumerate(codes):
            lines.append(f"| **{c}** | " + " | ".join(str(mat[i][j]) for j in range(len(codes))) + " |")
        lines.append("")

    md_path = out_dir / "stage1_multicore_summary.md"
    md_path.write_text("\n".join(lines))
    print(f"\nWrote Stage 1 summary report to {md_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c3-branch-min", type=float, default=0.15)
    ap.add_argument("--n-threads", type=int, default=64)
    ap.add_argument("--n-lib", type=int, default=64)
    ap.add_argument("--k1", type=int, default=8)
    ap.add_argument("--k2", type=int, default=32)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out-dir", type=str, default="results/phase2")
    ap.add_argument("--no-resume", action="store_true", help="Recompute even if equiv_<tier>.json exists")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    equiv_dir = out_dir / "equiv"
    equiv_dir.mkdir(parents=True, exist_ok=True)

    g = build_multicore_grammar()
    sim = ODESimulator(g, n_threads=args.n_threads)

    t0 = time.perf_counter()
    print(
        f"=== Stage 1c: Sampling accepted pools (527 mechs x (lib={args.n_lib}+k1={args.k1}+k2={args.k2}), "
        f"c3_branch_min={args.c3_branch_min}) ===",
        flush=True,
    )
    pools = sample_mechanism_pools(
        g, sim, n_lib=args.n_lib, k1=args.k1, k2=args.k2, seed=args.seed, c3_branch_min=args.c3_branch_min
    )
    print(f"Sampled pools in {time.perf_counter() - t0:.1f}s", flush=True)

    t_eval_1d = np.linspace(0.0, 1.0, 20)
    all_results = {}

    # Independent continuous reference set X_basic^ref FIRST (fast, unblocks E3 BOED!)
    x_json = equiv_dir / "equiv_X_basic_ref.json"
    if (not args.no_resume) and x_json.exists():
        print(f"\n=== Resuming Independent Continuous Set: X_basic_ref (loaded {x_json}) ===", flush=True)
        all_results["X_basic_ref"] = json.loads(x_json.read_text())
    else:
        print("\n=== Running Independent Continuous Set: X_basic_ref ===", flush=True)
        x_designs, x_t_eval_2d = build_x_basic_ref_designs(g, n_t=20)
        res_x, _ = run_two_stage_equivalence_multicore(
            g, sim, "X_basic_ref", x_designs, x_t_eval_2d, pools,
            inherited_fits=None, seed=args.seed + 999,
        )
        all_results["X_basic_ref"] = res_x
        with open(x_json, "w") as f:
            json.dump(res_x, f, indent=2)
        print(f"  Saved {x_json}", flush=True)

    # Reverse-nested discrete chain: Xi_full -> Xi_basic -> Xi_noP -> D0_D5 -> D0
    nested_chain = [
        ("Xi_full",  g.tiers["full"]),
        ("Xi_basic", g.tiers["basic"]),
        ("Xi_noP",   g.tiers["no_P"]),
        ("D0_D5",    [0, 5]),
        ("D0",       [0]),
    ]

    inherited_fits = None

    for idx_tier, (tier_name, design_indices) in enumerate(nested_chain):
        out_json = equiv_dir / f"equiv_{tier_name}.json"
        if (not args.no_resume) and out_json.exists():
            print(f"\n=== Resuming Nested Tier [{idx_tier + 1}/5]: {tier_name} (loaded {out_json}) ===", flush=True)
            all_results[tier_name] = json.loads(out_json.read_text())
            continue
        designs = [g.catalog[i] for i in design_indices]
        print(f"\n=== Running Nested Tier [{idx_tier + 1}/5]: {tier_name} ({[d.name for d in designs]}) ===", flush=True)
        res, inherited_fits = run_two_stage_equivalence_multicore(
            g, sim, tier_name, designs, t_eval_1d, pools,
            inherited_fits=inherited_fits, seed=args.seed + 100 * (idx_tier + 1),
        )
        all_results[tier_name] = res
        with open(out_json, "w") as f:
            json.dump(res, f, indent=2)
        print(f"  Saved {out_json}", flush=True)

    # Save compact index + write markdown summary
    compact = {
        k: {
            "n_designs": v["n_designs"],
            "design_names": v["design_names"],
            "n_classes": v["primary"]["n_classes"],
            "n_singleton": v["primary"]["n_singleton"],
            "n_nontrivial_classes": v["primary"]["n_nontrivial_classes"],
            "n_mechs_in_nontrivial": v["primary"]["n_mechs_in_nontrivial"],
            "max_class_size": v["primary"]["max_class_size"],
            "core_merge_stats": v["core_merge_stats"],
            "sensitivity": v["sensitivity"],
            "elapsed_total_s": v["elapsed_total_s"],
        }
        for k, v in all_results.items()
    }
    with open(equiv_dir / "equiv_summary_all6.json", "w") as f:
        json.dump(compact, f, indent=2)

    write_stage1_summary(out_dir, all_results, args.c3_branch_min)
    print(f"\n[Stage 1c DONE] Total elapsed: {(time.perf_counter() - t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
