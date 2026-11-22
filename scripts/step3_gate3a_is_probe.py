#!/usr/bin/env python3
"""Step 3 (Gate 3a): Pure Bayesian Importance-Sampling Probe for Proposition 3.

Compares Quotient EIG I(C_Xi; Y_xi | D) vs Fine-grained EIG I(M; Y_xi | D) on 30 test cases
drawn from non-trivial equivalence classes (|C_Xi| >= 2) under Ξ_no-P and Ξ_basic, using exact
IS posteriors (zero neural networks).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from quomech.bayes_probe import (
    build_is_particle_bank,
    compute_exact_eig_candidates,
    compute_exact_is_posterior,
)
from quomech.equivalence import build_design_targets
from quomech.grammar import build_default_grammar
from quomech.simulator import ODESimulator, sample_accepted_thetas


def run_tier_probe(
    grammar,
    sim,
    bank: dict,
    tier_name: str,
    equiv_data: dict,
    n_cases: int = 30,
    max_steps: int = 4,
    n_mc_outer: int = 512,
    seed: int = 202603,
) -> dict:
    rng = np.random.default_rng(seed)
    primary = equiv_data["primary"]
    classes = primary["classes"]
    mech_to_class = np.array(primary["mech_to_class"], dtype=np.int32)
    n_classes = len(classes)
    tier_designs = grammar.tiers[tier_name]
    t_eval = bank["t_eval"]
    sigma_ref = bank["sigma_ref"]
    curves_scaled = bank["curves_scaled"]

    # Gather all mechanisms in non-trivial equivalence classes (|C_Xi| >= 2)
    nontrivial_mechs = [m for cls in classes if len(cls) >= 2 for m in cls]
    if not nontrivial_mechs:
        return {"tier": tier_name, "error": "No non-trivial equivalence classes found!"}

    # Sample n_cases test mechanisms balanced across non-trivial classes
    nontrivial_classes = [cls for cls in classes if len(cls) >= 2]
    chosen_mechs = []
    for k in range(n_cases):
        cls = nontrivial_classes[k % len(nontrivial_classes)]
        chosen_mechs.append(int(rng.choice(cls)))

    # Draw independent ground-truth accepted theta for each case
    case_thetas = np.zeros((n_cases, grammar.n_params), dtype=np.float64)
    for idx, m in enumerate(chosen_mechs):
        th, _, _ = sample_accepted_thetas(grammar, sim, m, 1, rng, batch_size=512)
        case_thetas[idx] = th[0]

    # Simulate true noiseless trajectories on all 10 designs, then add fixed 1% Gaussian noise
    true_obs_scaled = []
    mids_arr = np.array(chosen_mechs, dtype=np.int32)
    for d_idx, des in enumerate(grammar.catalog):
        tgt, inv_sig, ok = build_design_targets(
            grammar, sim, mids_arr, case_thetas, [des], t_eval, sigma_ref=sigma_ref
        )
        assert np.all(ok)
        clean_sc = (tgt * inv_sig[None, :]).astype(np.float32)
        noisy_sc = clean_sc + rng.normal(0.0, 1.0, size=clean_sc.shape).astype(np.float32)
        true_obs_scaled.append(noisy_sc)

    def run_policy_episode(case_i: int, policy: str):
        """Run closed-loop BOED for one test case under 'quotient' or 'fine' policy."""
        true_m = chosen_mechs[case_i]
        true_c = int(mech_to_class[true_m])
        obs_d = [0]  # Start from D0
        y_list = [true_obs_scaled[0][case_i]]
        M, S, _ = curves_scaled[0].shape

        # Maintain exact cumulative log-likelihood across (M, S)
        diff0 = curves_scaled[0] - y_list[0][None, None, :]
        ll_curr = -0.5 * np.einsum("msd,msd->ms", diff0, diff0, optimize=True)

        def get_post(ll):
            w = np.exp(ll - np.max(ll))
            w /= np.sum(w)
            pm = np.sum(w, axis=1)
            pc = np.bincount(mech_to_class, weights=pm, minlength=n_classes)
            ess = float(1.0 / np.sum(w * w))
            return pm, pc, w, ess

        pm, pc, w, ess = get_post(ll_curr)
        pc_traj = [float(pc[true_c])]
        pm_traj = [float(pm[true_m])]
        max_pc_traj = [float(np.max(pc))]
        max_pm_traj = [float(np.max(pm))]
        step_logs = []

        for step in range(max_steps):
            cands = [d for d in tier_designs if d not in obs_d]
            if not cands:
                break
            eigs = compute_exact_eig_candidates(
                curves_scaled, ll_curr, w, mech_to_class, n_classes,
                cands, n_mc_outer=n_mc_outer, n_active_particles=4096,
                noise_mult=1.0, rng=rng,
            )
            best_q_idx = int(np.argmax(eigs["eig_quot"]))
            best_f_idx = int(np.argmax(eigs["eig_fine"]))
            d_q = cands[best_q_idx]
            d_f = cands[best_f_idx]
            chosen_d = d_q if policy == "quotient" else d_f

            step_logs.append({
                "step": step + 1,
                "chosen_design": grammar.catalog[chosen_d].name,
                "quot_best": grammar.catalog[d_q].name,
                "fine_best": grammar.catalog[d_f].name,
                "disagree": bool(d_q != d_f),
                "eig_quot_at_q": float(eigs["eig_quot"][best_q_idx]),
                "eig_quot_at_f": float(eigs["eig_quot"][best_f_idx]),
                "eig_fine_at_f": float(eigs["eig_fine"][best_f_idx]),
                "eig_intra_at_f": float(eigs["eig_intra"][best_f_idx]),
                "intra_frac_at_f": float(eigs["eig_intra"][best_f_idx] / max(eigs["eig_fine"][best_f_idx], 1e-9)),
                "ess_before": ess,
                "max_pc_before": float(np.max(pc)),
                "max_pm_before": float(np.max(pm)),
            })

            # Execute chosen design and update exact log-likelihood
            obs_d.append(chosen_d)
            y_new = true_obs_scaled[chosen_d][case_i]
            diff = curves_scaled[chosen_d] - y_new[None, None, :]
            ll_curr = ll_curr - 0.5 * np.einsum("msd,msd->ms", diff, diff, optimize=True)
            pm, pc, w, ess = get_post(ll_curr)
            pc_traj.append(float(pc[true_c]))
            pm_traj.append(float(pm[true_m]))
            max_pc_traj.append(float(np.max(pc)))
            max_pm_traj.append(float(np.max(pm)))

        # Compute stopping statistics
        # 1) Unified class stopping: first t in 0..max_steps where max_C p(C|D_t) >= 0.9
        t_stop_class = max_steps
        for t_i, mp in enumerate(max_pc_traj):
            if mp >= 0.90:
                t_stop_class = t_i
                break
        # 2) Fine-grained stopping: first t where max_m p(m|D_t) >= 0.9
        t_stop_fine = max_steps
        for t_i, mp in enumerate(max_pm_traj):
            if mp >= 0.90:
                t_stop_fine = t_i
                break
        # 3) Experiments to reach TRUE class >= 0.9
        t_true_90 = max_steps + 1  # > max_steps means not reached within budget
        for t_i, pt in enumerate(pc_traj):
            if pt >= 0.90:
                t_true_90 = t_i
                break

        return {
            "true_mech": grammar.mech_names[true_m],
            "class_size": len(classes[true_c]),
            "class_members": [grammar.mech_names[m] for m in classes[true_c]],
            "pc_traj": pc_traj,
            "pm_traj": pm_traj,
            "max_pc_traj": max_pc_traj,
            "max_pm_traj": max_pm_traj,
            "auc_pc": float(np.mean(pc_traj)),
            "t_stop_class": t_stop_class,
            "t_stop_fine": t_stop_fine,
            "t_true_90": t_true_90,
            "wasted_after_class_90": max(0, t_stop_fine - t_stop_class),
            "steps": step_logs,
        }

    q_episodes = []
    f_episodes = []
    for i in range(n_cases):
        q_episodes.append(run_policy_episode(i, "quotient"))
        f_episodes.append(run_policy_episode(i, "fine"))

    # Step-1 direct decision comparison (identical D={D0} state!)
    step1_disagrees = [ep["steps"][0]["disagree"] for ep in q_episodes]
    step1_delta_eig_q = [ep["steps"][0]["eig_quot_at_q"] - ep["steps"][0]["eig_quot_at_f"] for ep in q_episodes]
    step1_intra_frac = [ep["steps"][0]["intra_frac_at_f"] for ep in q_episodes]

    # All-step decision comparison on quotient trajectory
    all_disagrees = [s["disagree"] for ep in q_episodes for s in ep["steps"]]
    all_disagree_when_unsolved = [
        s["disagree"] for ep in q_episodes for s in ep["steps"] if s["max_pc_before"] < 0.90
    ]

    # Closed-loop metrics
    q_pc_mat = np.array([ep["pc_traj"] for ep in q_episodes])  # (n_cases, T+1)
    f_pc_mat = np.array([ep["pc_traj"] for ep in f_episodes])

    # Conditional on cases where step 1 actually disagreed
    dis_idx = [i for i, d in enumerate(step1_disagrees) if d]
    dis_summary = {}
    if dis_idx:
        dis_summary = {
            "n_disagree_step1": len(dis_idx),
            "quot_mean_pc_step1": float(np.mean(q_pc_mat[dis_idx, 1])),
            "fine_mean_pc_step1": float(np.mean(f_pc_mat[dis_idx, 1])),
            "quot_mean_auc": float(np.mean([q_episodes[i]["auc_pc"] for i in dis_idx])),
            "fine_mean_auc": float(np.mean([f_episodes[i]["auc_pc"] for i in dis_idx])),
            "mean_eig_quot_loss_on_disagree": float(np.mean([step1_delta_eig_q[i] for i in dis_idx])),
            "mean_intra_frac_on_disagree": float(np.mean([step1_intra_frac[i] for i in dis_idx])),
        }

    summary = {
        "tier": tier_name,
        "n_cases": n_cases,
        "n_nontrivial_classes": len(nontrivial_classes),
        "n_nontrivial_mechs": len(nontrivial_mechs),
        "step1_disagree_rate": float(np.mean(step1_disagrees)),
        "all_steps_disagree_rate": float(np.mean(all_disagrees)),
        "unsolved_steps_disagree_rate": float(np.mean(all_disagree_when_unsolved)) if all_disagree_when_unsolved else 0.0,
        "step1_mean_intra_frac_at_fine_choice": float(np.mean(step1_intra_frac)),
        "step1_mean_eig_quot_loss": float(np.mean(step1_delta_eig_q)),
        "quotient_mean_pc_by_step": [float(x) for x in np.mean(q_pc_mat, axis=0)],
        "fine_mean_pc_by_step": [float(x) for x in np.mean(f_pc_mat, axis=0)],
        "quotient_auc_pc": float(np.mean([ep["auc_pc"] for ep in q_episodes])),
        "fine_auc_pc": float(np.mean([ep["auc_pc"] for ep in f_episodes])),
        "quotient_mean_exp_unified_stop": float(np.mean([ep["t_stop_class"] for ep in q_episodes])),
        "fine_mean_exp_unified_stop": float(np.mean([ep["t_stop_class"] for ep in f_episodes])),
        "fine_mean_exp_fine_stop": float(np.mean([ep["t_stop_fine"] for ep in f_episodes])),
        "fine_mean_wasted_exps_after_class_90": float(np.mean([ep["wasted_after_class_90"] for ep in f_episodes])),
        "quotient_reach_true90_rate": float(np.mean([ep["t_true_90"] <= max_steps for ep in q_episodes])),
        "fine_reach_true90_rate": float(np.mean([ep["t_true_90"] <= max_steps for ep in f_episodes])),
        "quotient_mean_t_true90_capped": float(np.mean([min(ep["t_true_90"], max_steps) for ep in q_episodes])),
        "fine_mean_t_true90_capped": float(np.mean([min(ep["t_true_90"], max_steps) for ep in f_episodes])),
        "disagree_step1_breakdown": dis_summary,
        "example_disagreements": [
            {
                "case": i,
                "true_mech": q_episodes[i]["true_mech"],
                "class_members": q_episodes[i]["class_members"],
                "step1": q_episodes[i]["steps"][0],
                "q_pc_traj": [round(x, 3) for x in q_episodes[i]["pc_traj"]],
                "f_pc_traj": [round(x, 3) for x in f_episodes[i]["pc_traj"]],
            }
            for i in dis_idx[:8]
        ],
    }
    return summary


def main():
    root = Path(__file__).resolve().parents[1]
    g2_dir = root / "results" / "gate2"
    out_dir = root / "results" / "gate3a"
    out_dir.mkdir(parents=True, exist_ok=True)

    grammar = build_default_grammar()
    sim = ODESimulator(grammar, rtol=1e-5, atol=1e-9, n_threads=64)

    print("[Gate 3a] Building exact IS particle bank (151 mechs x 2048 particles x 10 designs)...")
    t0 = time.perf_counter()
    bank = build_is_particle_bank(grammar, sim, n_per_mech=2048, n_t=20, sigma_ref=0.01, seed=31415)
    print(f"[Gate 3a] IS bank built in {time.perf_counter() - t0:.1f}s.")

    all_out = {}
    for tier in ["no_P", "basic", "full"]:
        with open(g2_dir / f"equiv_{tier}.json") as f:
            eq_data = json.load(f)
        print(f"\n[Gate 3a] Running Proposition 3 IS probe on tier={tier} (30 non-trivial cases)...")
        t1 = time.perf_counter()
        res = run_tier_probe(
            grammar, sim, bank, tier_name=tier, equiv_data=eq_data,
            n_cases=30, max_steps=4, n_mc_outer=512, seed=202603 + len(tier),
        )
        res["elapsed_s"] = time.perf_counter() - t1
        all_out[tier] = res
        print(json.dumps({k: v for k, v in res.items() if k != "example_disagreements"}, indent=2))

    with open(out_dir / "gate3a_summary.json", "w") as f:
        json.dump(all_out, f, indent=2)


if __name__ == "__main__":
    main()
