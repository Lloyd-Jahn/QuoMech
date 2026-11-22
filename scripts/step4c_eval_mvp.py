#!/usr/bin/env python3
"""Step 4c: Run full E1 (inference + calibration + 50-case Enum+BIC) and E3 (BOED + RPKA/VTNA) MVP eval."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from quomech.boed import run_boed_benchmark
from quomech.evaluate import evaluate_e1_suite, evaluate_enum_bic_comparison
from quomech.grammar import build_default_grammar
from quomech.simulator import ODESimulator
from quomech.train import build_model


def load_trained(grammar, ckpt_path: Path, head_type: str, device: torch.device):
    model = build_model(grammar, head_type=head_type).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt.get("step", -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-test-e1", type=int, default=1510)
    ap.add_argument("--n-bic", type=int, default=50)
    ap.add_argument("--n-boed", type=int, default=151)
    ap.add_argument("--n-mc", type=int, default=256)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data" / "pool_v1"
    g2_dir = root / "results" / "gate2"
    g3b_dir = root / "results" / "gate3b"
    g3b_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    grammar = build_default_grammar()
    sim = ODESimulator(grammar, rtol=1e-5, atol=1e-9, n_threads=64)

    equiv_tiers = {}
    for tier in ["D0", "no_P", "basic", "full"]:
        with open(g2_dir / f"equiv_{tier}.json") as f:
            equiv_tiers[tier] = json.load(f)

    ckpt_q = g3b_dir / f"model_factorized_s{args.seed}" / "best.pt"
    ckpt_f = g3b_dir / f"model_flat_s{args.seed}" / "best.pt"
    model_q, step_q = load_trained(grammar, ckpt_q, "factorized", device)
    model_f, step_f = load_trained(grammar, ckpt_f, "flat", device)
    print(f"[Eval] Loaded QuoMech (step={step_q}) and Flat NPE (step={step_f}) on {device}")

    print("\n=== E1: Amortized Inference Suite (QuoMech) ===")
    e1_q = evaluate_e1_suite(model_q, grammar, sim, data_dir / "test", equiv_tiers, device, n_test=args.n_test_e1)
    with open(g3b_dir / "e1_quomech.json", "w") as f:
        json.dump(e1_q, f, indent=2)
    for rname, r in e1_q["regimes"].items():
        sbc = f"SBC-KS(med)={r['sbc_ks_median_param']:.4f} PPE={r['heldout_D7_ppe_rmse_median']:.4f}" if "sbc_ks_median_param" in r else ""
        print(f"  {rname:20s} | clsTop1={r['class_top1']:.3f} clsTop3={r['class_top3']:.3f} clsECE={r['class_ece']:.4f} | "
              f"mTop1={r['mech_top1']:.3f} mECE={r['mech_ece']:.4f} | {r['infer_ms_per_case']:.2f}ms {sbc}")

    print("\n=== E1: Amortized Inference Suite (Flat NPE) ===")
    e1_f = evaluate_e1_suite(model_f, grammar, sim, data_dir / "test", equiv_tiers, device, n_test=args.n_test_e1)
    with open(g3b_dir / "e1_flat.json", "w") as f:
        json.dump(e1_f, f, indent=2)
    for rname, r in e1_f["regimes"].items():
        sbc = f"SBC-KS(med)={r['sbc_ks_median_param']:.4f} PPE={r['heldout_D7_ppe_rmse_median']:.4f}" if "sbc_ks_median_param" in r else ""
        print(f"  {rname:20s} | clsTop1={r['class_top1']:.3f} clsTop3={r['class_top3']:.3f} clsECE={r['class_ece']:.4f} | "
              f"mTop1={r['mech_top1']:.3f} mECE={r['mech_ece']:.4f} | {r['infer_ms_per_case']:.2f}ms {sbc}")

    print(f"\n=== E1: {args.n_bic}-case Head-to-Head vs 64-core CPU Enum + LM + BIC ===")
    bic_cmp = evaluate_enum_bic_comparison(
        model_q, model_f, grammar, sim, data_dir / "test", equiv_tiers, device,
        n_cases=args.n_bic, design_indices=(0, 2, 5, 6), tier_key="basic",
    )
    with open(g3b_dir / "e1_enum_bic_cmp.json", "w") as f:
        json.dump(bic_cmp, f, indent=2)
    print(json.dumps(bic_cmp, indent=2))

    print(f"\n=== E3: Closed-Loop BOED Benchmark ({args.n_boed} cases, N_MC={args.n_mc}) ===")
    boed_res = run_boed_benchmark(
        model_q, grammar, sim, data_dir / "test", equiv_tiers, device,
        n_cases=args.n_boed, budget=4, n_mc=args.n_mc, sigma_rel=0.01,
    )
    with open(g3b_dir / "e3_boed_results.json", "w") as f:
        json.dump(boed_res, f, indent=2)

    print("\n=== Spontaneous RPKA / VTNA Diagnostic Logic Emergence ===")
    print(json.dumps(boed_res["rpka_vtna_emergence"], indent=2))


if __name__ == "__main__":
    main()
