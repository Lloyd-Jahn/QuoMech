#!/usr/bin/env python3
"""Evaluate strictly global reaction-type prior mean adjustments (no per-mechanism tuning)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from quomech.grammar import build_default_grammar, ReactionSpec
from quomech.simulator import ODESimulator
from scripts.step1_gate1_probe import run_prior_probe


def build_grammar_with_catcat(prior_means: dict, use_catcat_type: bool = False):
    g = build_default_grammar(prior_means_override={
        k: v for k, v in prior_means.items() if k in ("bimol", "unimol", "act", "deact1", "deact2", "bg")
    })
    if not use_catcat_type:
        return g
    # Assign reaction type "bimol_cat" to all catalyst-catalyst bimolecular reactions (data-driven!)
    new_rxns = []
    for r in g.reactions:
        if r.rxn_type == "bimol" and len(r.reactants) == 2:
            s0, s1 = g.species[r.reactants[0]], g.species[r.reactants[1]]
            if s0.is_catalyst and s1.is_catalyst:
                new_rxns.append(ReactionSpec(
                    r.name, r.reactants, r.stoich, r.param_slot,
                    r.module_id, r.module_option, "bimol_cat"
                ))
                continue
        new_rxns.append(r)
    from quomech.grammar import ReactionGrammar
    return ReactionGrammar(
        species=g.species,
        reactions=new_rxns,
        modules=g.modules,
        param_names=g.param_names,
        prior_means_by_type=dict(prior_means),
        prior_std=g.prior_std,
        c0_base=g.c0_base,
        max_active_modules=g.max_active_modules,
        catalog=g.catalog,
    )


def main():
    configs = [
        (
            "A1_6type_modest",
            False,
            {"bimol": 2.2, "unimol": 1.6, "act": 0.6, "deact1": 0.4, "deact2": 1.8, "bg": -1.2},
        ),
        (
            "A2_6type_shift",
            False,
            {"bimol": 2.4, "unimol": 1.4, "act": 0.6, "deact1": 0.4, "deact2": 1.8, "bg": -1.1},
        ),
        (
            "B1_dim_catcat_plus1.0",
            True,
            {"bimol": 2.2, "bimol_cat": 3.2, "unimol": 1.6, "act": 0.6, "deact1": 0.4, "deact2": 1.8, "bg": -1.2},
        ),
        (
            "B2_dim_catcat_plus1.3",
            True,
            {"bimol": 2.2, "bimol_cat": 3.5, "unimol": 1.6, "act": 0.6, "deact1": 0.4, "deact2": 1.8, "bg": -1.2},
        ),
        (
            "B3_dim_catcat_plus1.3_c0exact",
            True,
            {"bimol": 2.1, "bimol_cat": 3.4, "unimol": 1.6, "act": 0.6, "deact1": 0.5, "deact2": 1.8, "bg": -1.1},
        ),
    ]

    results = {}
    for name, use_cc, means in configs:
        g = build_grammar_with_catcat(means, use_catcat_type=use_cc)
        sim = ODESimulator(g, rtol=1e-5, atol=1e-9, n_threads=64)
        summary, _ = run_prior_probe(g, sim, n_draws_per_mech=1000, seed=77)
        oa = summary["overall_acceptance"]
        mp = {k: round(v["mean_pass"], 3) for k, v in summary["by_module_individual_pass"].items()}
        print(
            f"\n[{name}] min={oa['min']:.3f}, p05={oa['p05']:.3f}, med={oa['median']:.3f}, "
            f"mean={oa['mean']:.3f}, <3%={oa['n_below_3pct']}, fail={summary['solver_fail_overall']['max']:.4f}"
        )
        print(f"  module means: {mp}")
        print(f"  worst 3: {[(w['name'], round(w['acc_rate'], 4)) for w in summary['worst_10'][:3]]}")
        results[name] = summary

    out = Path(__file__).resolve().parents[1] / "results" / "gate1" / "prior_tuning_ablation.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
