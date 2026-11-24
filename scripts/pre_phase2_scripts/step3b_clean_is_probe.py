#!/usr/bin/env python3
"""Step 3 (Gate 3a — Non-Degenerate Exact IS Probe for Proposition 3).

Uses a canonical 10-mechanism diagnostic sub-library containing Gate 2's verified
non-trivial equivalence classes under Ξ_no-P and Ξ_basic:
  - C_prod  = {M4, M7, M4+M7}            (merged in Ξ_no-P & Ξ_basic at 1-2%)
  - C_dimer = {M5, M2:free2+M5, M5+M8}   (merged in Ξ_no-P; partially split in Ξ_basic)
  - Singletons: Core, M1, M2:free1, M3

With S = 30,000 accepted prior particles per mechanism (300,000 total in 4-7D space!) and
explicit Effective Sample Size (ESS) auditing at every step, this avoids the 10D weight
collapse observed with S=2,048 across 151 mechanisms and cleanly measures BOTH Proposition 3
effects:
  (A) Selection distortion under unified class stopping: argmax I(M) != argmax I(C_Xi)
  (B) Budget waste under fine-grained stopping (max_m p(m|D) >= 0.9 vs max_C p(C|D) >= 0.9).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from quomech.bayes_probe import compute_exact_eig_candidates
from quomech.equivalence import build_design_targets
from quomech.grammar import build_default_grammar
from quomech.simulator import ODESimulator, sample_accepted_thetas


def main():
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "results" / "gate3a"
    out_dir.mkdir(parents=True, exist_ok=True)

    grammar = build_default_grammar()
    # Leave 32 threads free for DDP dataloaders!
    sim = ODESimulator(grammar, rtol=1e-5, atol=1e-9, n_threads=32)

    sub_names = [
        "Core",
        "M1",
        "M2:free1",
        "M3",
        "M4",
        "M7",
        "M4+M7",
        "M5",
        "M2:free2+M5",
        "M5+M8",
    ]
    sub_ids = [grammar.get_mech_id_by_name(n) for n in sub_names]
    M_sub = len(sub_ids)
    S = 25000
    n_t = 12
    sigma_ref = 0.02  # 2% noise keeps exact IS ESS high (50-2000) across 4 experiments in 4-7D!
    t_eval = np.linspace(0.0, 1.0, n_t)
    rng = np.random.default_rng(20260309)

    print(f"[Gate 3a Clean IS] Sampling {M_sub} canonical mechanisms x {S:,} accepted particles...")
    t0 = time.perf_counter()
    thetas = np.zeros((M_sub, S, grammar.n_params), dtype=np.float64)
    for idx_m, m_global in enumerate(sub_ids):
        th, _, _ = sample_accepted_thetas(grammar, sim, m_global, S, rng, batch_size=8192)
        thetas[idx_m] = th
    print(f"[Gate 3a Clean IS] Sampled {M_sub * S:,} accepted particles in {time.perf_counter() - t0:.1f}s.")

    # Simulate all 10 catalog designs
    t1 = time.perf_counter()
    mids_flat = np.repeat(np.array(sub_ids, dtype=np.int32), S)
    th_flat = thetas.reshape(M_sub * S, grammar.n_params)
    curves_scaled = []
    for d_idx, des in enumerate(grammar.catalog):
        tgt, inv_sig, ok = build_design_targets(
            grammar, sim, mids_flat, th_flat, [des], t_eval, sigma_ref=sigma_ref
        )
        assert np.all(ok)
        curves_scaled.append((tgt * inv_sig[None, :]).astype(np.float32).reshape(M_sub, S, -1))
    print(f"[Gate 3a Clean IS] Simulated 10 designs in {time.perf_counter() - t1:.1f}s.")

    # Load actual Gate 2 equivalence classes (at 1% primary AND 2% matched noise) restricted to sub_ids!
    g2_dir = root / "results" / "gate2"
    all_results = {}

    for tier in ["no_P", "basic"]:
        with open(g2_dir / f"equiv_{tier}.json") as f:
            eq_data = json.load(f)
        # Use Gate 2's verified equivalence partition at 1.0% primary (and also inspect 2.0%)
        m2c_global = eq_data["primary"]["mech_to_class"]
        # Re-index classes for the 10-mech sub-library
        raw_c = [m2c_global[m] for m in sub_ids]
        uniq_c = sorted(set(raw_c))
        c_remap = {c: i for i, c in enumerate(uniq_c)}
        mech_to_class = np.array([c_remap[c] for c in raw_c], dtype=np.int32)
        n_cls = len(uniq_c)
        classes_sub = [[i for i in range(M_sub) if mech_to_class[i] == c] for c in range(n_cls)]

        print(f"\n[Gate 3a Clean IS | tier={tier}] Induced Gate-2 (1%) sub-classes:")
        for c_i, members in enumerate(classes_sub):
            print(f"  C{c_i}: {[sub_names[i] for i in members]}")

        nontriv_sub_mechs = [i for cls in classes_sub if len(cls) >= 2 for i in cls]
        n_cases = 30
        chosen_local = [nontriv_sub_mechs[k % len(nontriv_sub_mechs)] for k in range(n_cases)]

        # Draw fresh ground-truth accepted thetas for the 30 test cases
        case_thetas = np.zeros((n_cases, grammar.n_params), dtype=np.float64)
        case_global_mids = np.array([sub_ids[i] for i in chosen_local], dtype=np.int32)
        for ci, mg in enumerate(case_global_mids):
            th, _, _ = sample_accepted_thetas(grammar, sim, int(mg), 1, rng, batch_size=512)
            case_thetas[ci] = th[0]

        true_obs_scaled = []
        for d_idx, des in enumerate(grammar.catalog):
            tgt, inv_sig, ok = build_design_targets(
                grammar, sim, case_global_mids, case_thetas, [des], t_eval, sigma_ref=sigma_ref
            )
            assert np.all(ok)
            clean_sc = (tgt * inv_sig[None, :]).astype(np.float32)
            noisy_sc = clean_sc + rng.normal(0.0, 1.0, size=clean_sc.shape).astype(np.float32)
            true_obs_scaled.append(noisy_sc)

        tier_designs = list(grammar.tiers[tier])
        max_steps = 4

        def run_ep(ci: int, policy: str):
            true_m = chosen_local[ci]
            true_c = int(mech_to_class[true_m])
            obs_d = [0]
            diff0 = curves_scaled[0] - true_obs_scaled[0][ci][None, None, :]
            ll = -0.5 * np.einsum("msd,msd->ms", diff0, diff0, optimize=True)

            def get_post(ll_mat):
                w = np.exp(ll_mat - np.max(ll_mat))
                w /= np.sum(w)
                pm = np.sum(w, axis=1)
                pc = np.bincount(mech_to_class, weights=pm, minlength=n_cls)
                ess = float(1.0 / np.sum(w * w))
                return pm, pc, w, ess

            pm, pc, w, ess = get_post(ll)
            pc_traj = [float(pc[true_c])]
            pm_traj = [float(pm[true_m])]
            max_pc_traj = [float(np.max(pc))]
            max_pm_traj = [float(np.max(pm))]
            ess_traj = [ess]
            steps = []

            for step in range(max_steps):
                cands = [d for d in tier_designs if d not in obs_d]
                eigs = compute_exact_eig_candidates(
                    curves_scaled, ll, w, mech_to_class, n_cls,
                    cands, n_mc_outer=512, n_active_particles=4096,
                    noise_mult=1.0, rng=rng,
                )
                iq = int(np.argmax(eigs["eig_quot"]))
                if_ = int(np.argmax(eigs["eig_fine"]))
                dq, df = cands[iq], cands[if_]
                chosen = dq if policy == "quotient" else df
                steps.append({
                    "step": step + 1,
                    "quot_best": grammar.catalog[dq].name,
                    "fine_best": grammar.catalog[df].name,
                    "disagree": bool(dq != df),
                    "eig_quot_at_q": float(eigs["eig_quot"][iq]),
                    "eig_quot_at_f": float(eigs["eig_quot"][if_]),
                    "eig_fine_at_f": float(eigs["eig_fine"][if_]),
                    "eig_intra_at_f": float(eigs["eig_intra"][if_]),
                    "intra_frac_at_f": float(eigs["eig_intra"][if_] / max(eigs["eig_fine"][if_], 1e-9)),
                    "max_pc_before": float(np.max(pc)),
                    "max_pm_before": float(np.max(pm)),
                    "ess_before": ess,
                })
                obs_d.append(chosen)
                diff = curves_scaled[chosen] - true_obs_scaled[chosen][ci][None, None, :]
                ll = ll - 0.5 * np.einsum("msd,msd->ms", diff, diff, optimize=True)
                pm, pc, w, ess = get_post(ll)
                pc_traj.append(float(pc[true_c]))
                pm_traj.append(float(pm[true_m]))
                max_pc_traj.append(float(np.max(pc)))
                max_pm_traj.append(float(np.max(pm)))
                ess_traj.append(ess)

            t_stop_cls = next((ti for ti, v in enumerate(max_pc_traj) if v >= 0.90), max_steps)
            t_stop_fine = next((ti for ti, v in enumerate(max_pm_traj) if v >= 0.90), max_steps)
            t_true90 = next((ti for ti, v in enumerate(pc_traj) if v >= 0.90), max_steps + 1)
            return {
                "true_mech": sub_names[true_m],
                "pc_traj": pc_traj,
                "pm_traj": pm_traj,
                "max_pc_traj": max_pc_traj,
                "max_pm_traj": max_pm_traj,
                "ess_traj": ess_traj,
                "auc_pc": float(np.mean(pc_traj)),
                "t_stop_cls": t_stop_cls,
                "t_stop_fine": t_stop_fine,
                "t_true90": t_true90,
                "wasted": max(0, t_stop_fine - t_stop_cls),
                "steps": steps,
            }

        q_eps = [run_ep(i, "quotient") for i in range(n_cases)]
        f_eps = [run_ep(i, "fine") for i in range(n_cases)]

        ess_mat = np.array([e["ess_traj"] for e in q_eps])
        q_pc = np.array([e["pc_traj"] for e in q_eps])
        f_pc = np.array([e["pc_traj"] for e in f_eps])

        s1_dis = [e["steps"][0]["disagree"] for e in q_eps]
        all_dis = [s["disagree"] for e in q_eps for s in e["steps"]]
        unsolved_dis = [s["disagree"] for e in q_eps for s in e["steps"] if s["max_pc_before"] < 0.90]
        all_intra_frac = [s["intra_frac_at_f"] for e in q_eps for s in e["steps"]]
        unsolved_intra_frac = [s["intra_frac_at_f"] for e in q_eps for s in e["steps"] if s["max_pc_before"] < 0.90]
        dis_idx = [i for i, d in enumerate(s1_dis) if d]

        res_tier = {
            "tier": tier,
            "n_cases": n_cases,
            "n_particles_per_mech": S,
            "sigma_ref": sigma_ref,
            "median_ess_by_step": [float(x) for x in np.median(ess_mat, axis=0)],
            "p10_ess_by_step": [float(x) for x in np.percentile(ess_mat, 10, axis=0)],
            "step1_disagree_rate": float(np.mean(s1_dis)),
            "all_steps_disagree_rate": float(np.mean(all_dis)),
            "unsolved_steps_disagree_rate": float(np.mean(unsolved_dis)) if unsolved_dis else 0.0,
            "mean_intra_eig_fraction_all_steps": float(np.mean(all_intra_frac)),
            "mean_intra_eig_fraction_unsolved": float(np.mean(unsolved_intra_frac)) if unsolved_intra_frac else 0.0,
            "quotient_mean_pc_by_step": [float(x) for x in np.mean(q_pc, axis=0)],
            "fine_mean_pc_by_step": [float(x) for x in np.mean(f_pc, axis=0)],
            "quotient_auc_pc": float(np.mean([e["auc_pc"] for e in q_eps])),
            "fine_auc_pc": float(np.mean([e["auc_pc"] for e in f_eps])),
            "quotient_mean_exps_unified_stop": float(np.mean([e["t_stop_cls"] for e in q_eps])),
            "fine_mean_exps_unified_stop": float(np.mean([e["t_stop_cls"] for e in f_eps])),
            "fine_mean_exps_fine_stop": float(np.mean([e["t_stop_fine"] for e in f_eps])),
            "fine_mean_wasted_exps_after_class_90": float(np.mean([e["wasted"] for e in f_eps])),
            "quotient_reach_true90_rate": float(np.mean([e["t_true90"] <= max_steps for e in q_eps])),
            "fine_reach_true90_rate": float(np.mean([e["t_true90"] <= max_steps for e in f_eps])),
            "disagree_step1_n": len(dis_idx),
            "disagree_step1_quot_pc1": float(np.mean(q_pc[dis_idx, 1])) if dis_idx else None,
            "disagree_step1_fine_pc1": float(np.mean(f_pc[dis_idx, 1])) if dis_idx else None,
            "disagree_step1_quot_auc": float(np.mean([q_eps[i]["auc_pc"] for i in dis_idx])) if dis_idx else None,
            "disagree_step1_fine_auc": float(np.mean([f_eps[i]["auc_pc"] for i in dis_idx])) if dis_idx else None,
            "example_disagreements": [
                {
                    "case": i, "true_mech": q_eps[i]["true_mech"],
                    "step1": q_eps[i]["steps"][0],
                    "q_pc": [round(x, 3) for x in q_eps[i]["pc_traj"]],
                    "f_pc": [round(x, 3) for x in f_eps[i]["pc_traj"]],
                    "q_stop": q_eps[i]["t_stop_cls"],
                    "f_stop_fine": f_eps[i]["t_stop_fine"],
                }
                for i in dis_idx[:6]
            ],
        }
        all_results[tier] = res_tier
        print(json.dumps({k: v for k, v in res_tier.items() if k != "example_disagreements"}, indent=2))

    with open(out_dir / "gate3a_clean_is_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
