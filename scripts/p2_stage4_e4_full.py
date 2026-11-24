#!/usr/bin/env python3
"""Phase 2 Stage 4 (E4): Full Real Published Catalysis Evaluation (11 Systems, 48 Runs)
+ Next-Experiment BOED Comparison + 3x100 Synthetic Misspecification Stress Tests.

Part 1 — All 11 Published Homogeneous Catalysis Datasets (data/real_literature/manifest.json):
  - Zero retraining, zero case-specific hacks, zero cherry-picking:
    * Physical single-substrate core support (C1/C4 when reaction_type == "pseudo_unimolecular")
    * Reports BOTH uniform & hierarchical Occam complexity-prior posteriors, plus quotient-class
      posterior under Xi_basic and coarse C_macro!
    * Fast batched (~100-200ms) Top-5 local posterior predictive overlap clique (ONE C++ batch per run)
    * Amortized warm-start (3 NSF draws + 1 prior mean) + 20-iter bounded C++ LM refinement on
      Top-5 U {physical clean cores} -> BIC*, clean-core fit, and per-channel signal-normalized
      residual z_sig (sigma_{d,c} = 0.02 * max(span_{d,c}, 0.02)) + conversion regime guard!
    * Leave-One-Out Posterior Predictive Error (LOO-PPE) across all 48 verbatim published runs
    * Next-Experiment BOED Recommendation starting ONLY from each system's reference_run_id vs
      the published paper's actual follow-up experiments!
    * Saves verbatim & posterior predictive curve coordinates for Fig. 5 publication plots.

Part 2 — 3x100 Synthetic Model-Misspecification Stress Tests (+ 100 Matched In-Distribution Controls):
  1) Temperature drift: k_r(t) = k_{r,0} * exp(alpha * t), |alpha| in [0.45, 0.75] (SciPy Radau)
  2) Sensor calibration bias: systematic +/-8% multiplicative response-factor error on A / P
  3) Unmodeled out-of-library side reaction: competing 2nd-order substrate degradation 2A -> X
     consuming 12-28% of A into unmodeled byproduct X (SciPy Radau)
  Reports >3sigma True Positive Rate (TPR) vs In-Distribution FPR & AUROC for BOTH pure amortized
  Top-5 posterior predictive residual AND amortized warm-start local LM refinement!
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.integrate import solve_ivp
from sklearn.metrics import roc_auc_score
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quomech.boed import _score_eig_designs_batched
from quomech.data_gen import get_observable_channel_indices
from quomech.evaluate import aggregate_class_probs, pack_fixed_test_batch
from quomech.grammar import DesignSpec, ReactionGrammar, build_multicore_grammar
from quomech.simulator import ODESimulator
from quomech.train import build_model
from scripts.p2_e4_real_eval import load_real_case, runs_to_tensors
from scripts.p2_stage3_e3_boed import build_c_macro_partition


# =============================================================================
# Part 1: All 11 Published Homogeneous Catalysis Datasets
# =============================================================================
def _build_per_channel_sigmas(runs: List[Dict], sigma_rel: float = 0.02) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """Compute per-run, per-channel signal-normalized sigma_c = sigma_rel * max(span_c, 0.02).

    Properly scales high-concentration co-substrate channels (e.g. wei_2020 B0=4.64) AND
    low-conversion OOD regimes (e.g. ser_2025 3.4% conversion) without ad-hoc caps.
    """
    obs_chunks = []
    sig_chunks = []
    masks_per_run = []
    for r in runs:
        y_obs = r["y_raw_dim"]  # (n_raw, n_ch)
        n_raw, n_ch = y_obs.shape
        mask_2d = (r["ch_mask"][None, :] > 0.5) & (~np.isnan(y_obs))
        masks_per_run.append(mask_2d)
        sig_2d = np.zeros((n_raw, n_ch), dtype=np.float64)
        for c_i in range(n_ch):
            if r["ch_mask"][c_i] > 0.5:
                vals = y_obs[mask_2d[:, c_i], c_i]
                span = float(np.max(vals) - np.min(vals)) if len(vals) > 1 else 0.0
                sig_c = sigma_rel * max(span, 0.02)
                sig_2d[:, c_i] = sig_c
        obs_chunks.append(y_obs[mask_2d])
        sig_chunks.append(sig_2d[mask_2d])
    return np.concatenate(obs_chunks), np.concatenate(sig_chunks), masks_per_run


@torch.no_grad()
def evaluate_real_case_full(
    grammar: ReactionGrammar,
    sim: ODESimulator,
    model,
    case: Dict,
    meta: Dict,
    device: torch.device,
    equiv_basic_m2c: Optional[np.ndarray],
    equiv_basic_classes: Optional[List[List[int]]],
    c_macro: Dict,
    n_post_samples: int = 16,
    sigma_rel: float = 0.02,
    seed: int = 42,
) -> Tuple[Dict, Dict]:
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    runs = case["runs"]
    obs_ch = case["obs_ch"]
    ch_pos = {grammar.species[sp].name: i for i, sp in enumerate(obs_ch)}
    is_uni = (case["reaction_type"] == "pseudo_unimolecular")

    k_arr = np.sum(grammar.mech_options > 0, axis=1)
    nk = {k: int(np.sum(k_arr == k)) for k in range(4)}
    pi_hier = np.array([1.0 / (4.0 * nk[int(k)]) for k in k_arr], dtype=np.float64)
    phys_mask = np.isin(grammar.mech_core_id, [0, 3]).astype(np.float64) if is_uni else np.ones(grammar.n_mechs, dtype=np.float64)

    # 1. Full-dataset neural posterior (uniform & hierarchical Occam, under physical core support)
    tensors = runs_to_tensors(grammar, runs, device)
    probs_t, f_t, G_t = model.predict_posterior(*tensors)
    p_raw = probs_t[0].cpu().numpy().astype(np.float64)
    p_unif = (p_raw * phys_mask)
    p_unif /= np.maximum(p_unif.sum(), 1e-12)
    p_hier = (p_raw * pi_hier * phys_mask)
    p_hier /= np.maximum(p_hier.sum(), 1e-12)

    def summarize_post(p: np.ndarray) -> Dict:
        top5 = np.argsort(p)[::-1][:5].tolist()
        out = {
            "core_probs": {c.code: float(np.sum(p[grammar.mech_core_id == c.id])) for c in grammar.cores},
            "module_probs": {mod.code: float(np.sum(p[grammar.mech_options[:, mod.id] > 0])) for mod in grammar.modules},
            "expected_module_probs": {
                m: float(np.sum(p[grammar.mech_options[:, next(mod.id for mod in grammar.modules if mod.code == m)] > 0]))
                for m in case["expected_modules"]
            },
            "clean_core_prob": float(np.sum(p[k_arr == 0])),
            "top5": [{"mech_id": int(m), "name": grammar.mech_names[m], "prob": float(p[m])} for m in top5],
        }
        if equiv_basic_m2c is not None and equiv_basic_classes is not None:
            cp = aggregate_class_probs(p[None, :], equiv_basic_m2c, len(equiv_basic_classes))[0]
            ctop = np.argsort(cp)[::-1][:3]
            out["quotient_basic_top3"] = [
                {
                    "class_id": int(cid), "prob": float(cp[cid]), "size": len(equiv_basic_classes[cid]),
                    "rep_names": [grammar.mech_names[m] for m in equiv_basic_classes[cid][:4]],
                }
                for cid in ctop
            ]
        cm_p = aggregate_class_probs(p[None, :], c_macro["mech_to_class"], c_macro["n_classes"])[0]
        cm_top = np.argsort(cm_p)[::-1][:3]
        out["c_macro_top3"] = [
            {"class_id": int(cid), "label": c_macro["class_labels"][cid], "prob": float(cm_p[cid])}
            for cid in cm_top
        ]
        return out

    post_unif_sum = summarize_post(p_unif)
    post_hier_sum = summarize_post(p_hier)

    # 2. Fast Batched Top-5 Local Posterior Predictive Overlap (~100ms: 1 C++ call per run for all 5*S draws!)
    t_loc0 = time.perf_counter()
    top5_idx = [x["mech_id"] for x in post_hier_sum["top5"]]
    obs_flat_all, sig_flat_all, masks_per_run = _build_per_channel_sigmas(runs, sigma_rel=sigma_rel)

    f_rep5 = f_t.expand(len(top5_idx), -1)
    m_t5 = torch.tensor(top5_idx, dtype=torch.long, device=device)
    th_s5 = model.sample_theta(f_rep5, G_t, m_t5, num_samples=n_post_samples).cpu().numpy().astype(np.float64)
    # (S, 5, P) -> (5 * S, P)
    th_flat5 = th_s5.transpose(1, 0, 2).reshape(len(top5_idx) * n_post_samples, grammar.n_params)
    m_flat5 = np.repeat(np.array(top5_idx, dtype=np.int32), n_post_samples)
    lb = grammar.mech_prior_mean - 4.5 * grammar.mech_prior_std
    ub = grammar.mech_prior_mean + 4.5 * grammar.mech_prior_std
    th_flat5 = np.clip(th_flat5, lb[m_flat5], ub[m_flat5])

    pred_chunks_5 = []
    valid_5 = np.ones(len(m_flat5), dtype=bool)
    map_run_curves = {}
    for r_i, r in enumerate(runs):
        y0s = np.stack([grammar.initial_state(int(m), r["design"]) for m in m_flat5], axis=0)
        y_full, st, _, _ = sim.simulate_batch(
            m_flat5, th_flat5, y0s, r["t_raw_dim"], compute_flux=False, rtol=5e-5, atol=5e-9
        )
        valid_5 &= (st == 0)
        y_ch = np.take(y_full, obs_ch, axis=-1)  # (5*S, n_raw, n_ch)
        pred_chunks_5.append(y_ch[:, masks_per_run[r_i]])
        # Save MAP posterior predictive bands (first n_post_samples rows) for plotting
        ok0 = (st[:n_post_samples] == 0)
        if np.any(ok0):
            y0_ok = y_ch[:n_post_samples][ok0]
            map_run_curves[r["run_id"]] = {
                "t": r["t_raw_dim"].tolist(),
                "y_obs": np.where(np.isnan(r["y_raw_dim"]), None, r["y_raw_dim"]).tolist(),
                "y_pred_med": np.median(y0_ok, axis=0).tolist(),
                "y_pred_p05": np.percentile(y0_ok, 5, axis=0).tolist(),
                "y_pred_p95": np.percentile(y0_ok, 95, axis=0).tolist(),
            }
    preds_all_5 = np.concatenate(pred_chunks_5, axis=1).reshape(len(top5_idx), n_post_samples, -1)
    valid_2d = valid_5.reshape(len(top5_idx), n_post_samples)

    top5_meds, top5_z_med, top5_z_best = [], [], []
    for k_i in range(len(top5_idx)):
        ok_k = valid_2d[k_i]
        if np.any(ok_k):
            pk = preds_all_5[k_i, ok_k]
            errs = np.sqrt(np.mean(((pk - obs_flat_all[None, :]) / sig_flat_all[None, :]) ** 2, axis=1))
            med_c = np.median(pk, axis=0)
            top5_meds.append(med_c)
            top5_z_med.append(float(np.sqrt(np.mean(((med_c - obs_flat_all) / sig_flat_all) ** 2))))
            top5_z_best.append(float(np.min(errs)))
        else:
            top5_meds.append(np.zeros_like(obs_flat_all))
            top5_z_med.append(999.0)
            top5_z_best.append(999.0)

    overlap_clique = [top5_idx[0]]
    for k_i in range(1, len(top5_idx)):
        z_diff = float(np.sqrt(np.mean(((top5_meds[k_i] - top5_meds[0]) / sig_flat_all) ** 2)))
        if z_diff <= 1.0:
            overlap_clique.append(top5_idx[k_i])
    dt_local_ms = (time.perf_counter() - t_loc0) * 1e3

    # 3. Amortized Warm-Start + 20-iter Local LM Refinement on Top-5 U {physical clean cores}
    core_ids = [grammar.get_mech_id_by_name(c) for c in (["C1", "C4"] if is_uni else ["C1", "C2", "C3", "C4"])]
    lm_cands = list(dict.fromkeys(top5_idx + core_ids))
    n_t_fit = min(15, min(len(r["t_raw_dim"]) for r in runs))
    designs_lm = [r["design"] for r in runs]
    t_eval_2d = np.zeros((len(runs), n_t_fit), dtype=np.float64)
    tgt_chunks, inv_sig_chunks = [], []
    for d_i, r in enumerate(runs):
        n_raw = len(r["t_raw_dim"])
        idx = np.round(np.linspace(0, n_raw - 1, n_t_fit)).astype(int)
        t_eval_2d[d_i] = r["t_raw_dim"][idx]
        for sp in r["design"].obs_species:
            c_i = ch_pos[grammar.species[sp].name]
            yv_full = r["y_raw_dim"][:, c_i]
            valid_full = yv_full[~np.isnan(yv_full)]
            span_c = float(np.max(valid_full) - np.min(valid_full)) if len(valid_full) > 1 else 0.0
            sig_c = sigma_rel * max(span_c, 0.02)
            yv = r["y_raw_dim"][idx, c_i]
            # Handle any isolated NaN cleanly by zeroing inverse sigma at missing points
            is_ok_pt = ~np.isnan(yv)
            tgt_chunks.append(np.where(is_ok_pt, yv, 0.0))
            inv_sig_chunks.append(np.where(is_ok_pt, 1.0 / sig_c, 0.0))
    tgt_1d = np.concatenate(tgt_chunks)
    inv_sig_1d = np.concatenate(inv_sig_chunks)
    n_obs_lm = int(np.sum(inv_sig_1d > 0))

    n_c = len(lm_cands)
    inits = np.zeros((n_c, 4, grammar.n_params), dtype=np.float64)
    f_repc = f_t.expand(n_c, -1)
    m_tc = torch.tensor(lm_cands, dtype=torch.long, device=device)
    th_sc = model.sample_theta(f_repc, G_t, m_tc, num_samples=3).cpu().numpy().astype(np.float64)  # (3, n_c, P)
    for ci, m_id in enumerate(lm_cands):
        inits[ci, :3] = th_sc[:, ci, :]
        inits[ci, 3] = grammar.mech_prior_mean[m_id]

    t_lm0 = time.perf_counter()
    rms_fit, theta_fit = sim.fit_lm_batch(
        np.array(lm_cands, dtype=np.int32), designs_lm, t_eval_2d,
        np.broadcast_to(tgt_1d[None, :], (n_c, len(tgt_1d))).copy(),
        np.broadcast_to(inv_sig_1d[None, :], (n_c, len(inv_sig_1d))).copy(),
        inits, lb_all=grammar.mech_prior_mean - 4.0 * grammar.mech_prior_std,
        ub_all=grammar.mech_prior_mean + 4.0 * grammar.mech_prior_std,
        rtol=1e-4, atol=1e-8, max_ode_steps=8000, max_lm_iters=20, early_stop_rms=0.5,
    )
    dt_lm_ms = (time.perf_counter() - t_lm0) * 1e3

    # Re-evaluate best-fit theta on ALL verbatim raw points for exact RMSE & plotting!
    bics = []
    for ci, m_id in enumerate(lm_cands):
        k_p = int(grammar.mech_param_mask[m_id].sum())
        rss_z = (rms_fit[ci] ** 2) * len(tgt_1d)
        bics.append(float(rss_z + k_p * np.log(max(n_obs_lm, 2))))
    best_ci = int(np.argmin(rms_fit))
    best_bic_ci = int(np.argmin(bics))
    core_cis = [ci for ci, m_id in enumerate(lm_cands) if k_arr[m_id] == 0]
    best_core_ci = min(core_cis, key=lambda ci: rms_fit[ci])

    # Simulate BIC* fitted curve on verbatim raw timestamps for plotting & exact raw RMSE
    bic_mid = lm_cands[best_bic_ci]
    bic_th = theta_fit[best_bic_ci]
    bic_raw_errs = []
    for r_i, r in enumerate(runs):
        y0 = grammar.initial_state(bic_mid, r["design"])[None, :]
        y_s, st, _, _ = sim.simulate_batch(
            np.array([bic_mid], dtype=np.int32), bic_th[None, :], y0, r["t_raw_dim"], compute_flux=False, n_threads=1
        )
        y_ch = np.take(y_s[0], obs_ch, axis=-1)
        if r["run_id"] in map_run_curves:
            map_run_curves[r["run_id"]]["y_lm_bic_fit"] = y_ch.tolist()
        if st[0] == 0:
            m2d = masks_per_run[r_i]
            bic_raw_errs.append(y_ch[m2d] - r["y_raw_dim"][m2d])
    bic_raw_rmse = float(np.sqrt(np.mean(np.concatenate(bic_raw_errs) ** 2))) if bic_raw_errs else float("nan")

    ref_meta = next(r for r in meta["runs"] if r["run_id"] == meta["reference_run_id"])
    ref_conv = float(ref_meta["final_conversion_pct"])
    regime_ood = bool((ref_conv < 20.0) or (ref_conv > 99.0))
    resid_ood_lm = bool(float(rms_fit[best_ci]) > 3.0)
    resid_ood_amort = bool(min(top5_z_best) > 3.0)

    # 4. Leave-One-Out Posterior Predictive Error (LOO-PPE) across all runs
    loo_rmses = {}
    for hold_i, hold_run in enumerate(runs):
        train_runs = [r for j, r in enumerate(runs) if j != hold_i]
        p_lt, f_lt, G_lt = model.predict_posterior(*runs_to_tensors(grammar, train_runs, device))
        p_l = p_lt[0].cpu().numpy().astype(np.float64) * pi_hier * phys_mask
        p_l /= np.maximum(p_l.sum(), 1e-12)
        samp_m = rng.choice(grammar.n_mechs, size=n_post_samples, p=p_l).astype(np.int32)
        f_rep = f_lt.expand(n_post_samples, -1)
        th_l = model.sample_theta(f_rep, G_lt, torch.from_numpy(samp_m).long().to(device), num_samples=1).cpu().numpy().astype(np.float64)
        th_l = np.clip(th_l, lb[samp_m], ub[samp_m])
        y0s = np.stack([grammar.initial_state(int(m), hold_run["design"]) for m in samp_m], axis=0)
        y_full, st, _, _ = sim.simulate_batch(samp_m, th_l, y0s, hold_run["t_raw_dim"], compute_flux=False)
        ok = (st == 0)
        y_obs = hold_run["y_raw_dim"]
        m2d = masks_per_run[hold_i]
        if np.any(ok):
            y_med = np.median(np.take(y_full[ok], obs_ch, axis=-1), axis=0)
            loo_rmses[hold_run["run_id"]] = float(np.sqrt(np.mean((y_med[m2d] - y_obs[m2d]) ** 2)))
        else:
            loo_rmses[hold_run["run_id"]] = float("nan")

    # 5. Next-Experiment BOED Recommendation starting ONLY from reference_run_id
    ref_run = next(r for r in runs if r["run_id"] == case["reference_run_id"])
    ref_tensors = runs_to_tensors(grammar, [ref_run], device)
    p_ref_t, f_ref_t, G_ref_t = model.predict_posterior(*ref_tensors)
    # Obtain single-experiment embedding h_ref
    df_r, tm_r, ov_r, om_r, tp_r, _ = ref_tensors
    h_ref = model.exp_encoder(df_r[0], tm_r[0], ov_r[0], om_r[0], tp_r[0]).float().unsqueeze(0)  # (1, 1, d_exp)
    p_ref = p_ref_t[0].cpu().numpy().astype(np.float64) * phys_mask
    p_ref /= np.maximum(p_ref.sum(), 1e-12)

    # Candidate follow-up designs matching the system's observed species and physical reaction type!
    # Exclude D3 (high B) for pseudo-unimolecular systems where B does not physically exist.
    obs_sp_t = ref_run["design"].obs_species
    cand_defs = []
    for d_idx in [1, 2, 3, 4, 5, 6, 8]:
        d_cat = grammar.catalog[d_idx]
        if is_uni and d_idx == 3:
            continue
        b0_use = 10.0 if is_uni else d_cat.B0
        t_use = float(np.clip(1.0 / d_cat.cat_mult, 0.5, 2.5))
        cand_defs.append((
            DesignSpec(d_cat.name, d_cat.A0, b0_use, d_cat.P0, d_cat.cat_mult, d_cat.a_pre, obs_sp_t, 1.0, ()),
            t_use,
        ))
    m2c_boed = equiv_basic_m2c if equiv_basic_m2c is not None else c_macro["mech_to_class"]
    n_cls_boed = int(np.max(m2c_boed)) + 1
    pc_ref = aggregate_class_probs(p_ref[None, :], m2c_boed, n_cls_boed)
    eq_boed, _ = _score_eig_designs_batched(
        model, grammar, sim, h_ref, f_ref_t.float(), G_ref_t.float(),
        p_ref[None, :], pc_ref, cand_defs, [list(range(len(cand_defs)))],
        m2c_boed, n_cls_boed, n_t=20, sigma_rel=0.02, n_mc=128, device=device,
        rng=np.random.default_rng(seed + 500), crn_seed=seed + 500, case_ids=[0],
    )
    scores_boed = eq_boed[0]
    order_boed = np.argsort(scores_boed)[::-1]
    boed_ranked = [
        {"design": cand_defs[k][0].name, "eig_quotient": float(scores_boed[k])}
        for k in order_boed[:3]
    ]
    actual_followups = [
        {"run_id": r["run_id"], "role": r["role"], "A0": r["design"].A0, "B0": r["design"].B0, "P0": r["design"].P0, "cat_mult": r["design"].cat_mult}
        for r in runs if r["run_id"] != case["reference_run_id"]
    ]

    summary = {
        "case_id": case["case_id"],
        "in_library": case["in_library"],
        "reaction_type": case["reaction_type"],
        "n_runs": len(runs),
        "observed_species": case["observed_species"],
        "expected_modules": case["expected_modules"],
        "ref_conversion_pct": ref_conv,
        "posterior_uniform": post_unif_sum,
        "posterior_hierarchical": post_hier_sum,
        "local_overlap": {
            "clique_names": [grammar.mech_names[m] for m in overlap_clique],
            "clique_size": len(overlap_clique),
            "elapsed_ms": dt_local_ms,
            "amortized_min_z_best": float(min(top5_z_best)),
            "amortized_min_z_med": float(min(top5_z_med)),
            "amortized_alert_3sigma": resid_ood_amort,
        },
        "warmstart_local_lm": {
            "elapsed_ms": dt_lm_ms,
            "n_candidates": n_c,
            "best_fit_mech": grammar.mech_names[lm_cands[best_ci]],
            "best_fit_z_sig": float(rms_fit[best_ci]),
            "best_core_mech": grammar.mech_names[lm_cands[best_core_ci]],
            "best_core_z_sig": float(rms_fit[best_core_ci]),
            "bic_winner_mech": grammar.mech_names[bic_mid],
            "bic_winner_z_sig": float(rms_fit[best_bic_ci]),
            "bic_winner_raw_rmse": bic_raw_rmse,
            "resid_alert_3sigma": resid_ood_lm,
            "regime_ood_alert": regime_ood,
            "combined_ood_alert": bool(resid_ood_lm or regime_ood),
        },
        "loo_ppe": {
            "by_run": loo_rmses,
            "mean": float(np.nanmean(list(loo_rmses.values()))),
        },
        "boed_next_exp_from_ref": {
            "reference_run_id": case["reference_run_id"],
            "top3_recommended": boed_ranked,
            "actual_paper_followups": actual_followups,
        },
        "elapsed_s": time.perf_counter() - t0,
    }
    plot_bundle = {
        "case_id": case["case_id"],
        "in_library": case["in_library"],
        "observed_species": case["observed_species"],
        "obs_ch_names": [grammar.species[sp].name for sp in obs_ch],
        "map_mech": post_hier_sum["top5"][0]["name"],
        "bic_mech": grammar.mech_names[bic_mid],
        "runs": map_run_curves,
    }
    return summary, plot_bundle


# =============================================================================
# Part 2: 3x100 Synthetic Misspecification Stress Tests (Parallel SciPy Radau)
# =============================================================================
_G_WORKER: Optional[ReactionGrammar] = None


def _init_radau_worker():
    global _G_WORKER
    _G_WORKER = build_multicore_grammar()


def _sim_misspec_case_worker(task: Tuple) -> Tuple[int, np.ndarray, bool]:
    """Simulate 4 designs (D0, D2, D5, D6) under a specified physical misspecification via SciPy Radau."""
    ci, mode, mech_id, theta, param_val, design_indices = task
    g = _G_WORKER
    obs_ch = get_observable_channel_indices(g)
    t_eval = np.linspace(0.0, 1.0, 20)
    rmask = g.mech_rxn_mask[mech_id]
    k_base = np.where(rmask == 1, 10.0 ** theta[g.rxn_param_slot], 0.0)
    r1, r2, S = g.reactant1, g.reactant2, g.stoich_matrix
    act_rxns = np.where(rmask == 1)[0]
    sA = g.species_idx["A"]

    out_y = np.zeros((len(design_indices), 20, len(obs_ch)), dtype=np.float64)
    for d_i, cat_idx in enumerate(design_indices):
        des = g.catalog[cat_idx]
        y0 = g.initial_state(int(mech_id), des)
        if mode == "temp_drift":
            alpha = float(param_val)
            def rhs(t, y):
                scale_t = np.exp(alpha * t)
                rates = np.zeros(g.n_reactions, dtype=np.float64)
                for r in act_rxns:
                    a, b = r1[r], r2[r]
                    rates[r] = scale_t * (k_base[r] * y[a] if b < 0 else k_base[r] * y[a] * y[b])
                return S @ rates
        elif mode == "unmodeled_side_rxn":
            # Out-of-library competing 2nd-order substrate degradation 2A -> X (unmodeled byproduct!)
            k_leak = float(param_val)
            def rhs(_t, y):
                rates = np.zeros(g.n_reactions, dtype=np.float64)
                for r in act_rxns:
                    a, b = r1[r], r2[r]
                    rates[r] = k_base[r] * y[a] if b < 0 else k_base[r] * y[a] * y[b]
                dy = S @ rates
                a_pos = max(y[sA], 0.0)
                dy[sA] -= 2.0 * k_leak * a_pos * a_pos
                return dy
        else:
            raise ValueError(mode)

        sol = solve_ivp(rhs, (0.0, 1.0), y0, method="Radau", t_eval=t_eval, rtol=1e-6, atol=1e-10)
        if not sol.success or sol.y.shape[1] != 20 or not np.all(np.isfinite(sol.y)):
            return ci, out_y, False
        out_y[d_i] = sol.y.T[:, obs_ch]
    return ci, out_y, True


@torch.no_grad()
def evaluate_misspec_batch(
    grammar: ReactionGrammar,
    sim: ODESimulator,
    model,
    raw_noisy: np.ndarray,      # (N, 4, 20, n_ch)
    df: np.ndarray,
    tm: np.ndarray,
    ov: np.ndarray,
    om: np.ndarray,
    tp: np.ndarray,
    ep: np.ndarray,
    designs: List[DesignSpec],
    device: torch.device,
    sigma_rel: float = 0.01,
    n_ppc: int = 16,
) -> Dict:
    """Compute both pure amortized Top-5 min z_sigma AND amortized warm-start 20-iter LM min z_sigma."""
    N = len(raw_noisy)
    obs_ch = get_observable_channel_indices(grammar)
    t_eval = np.linspace(0.0, 1.0, 20)
    lb = grammar.mech_prior_mean - 4.0 * grammar.mech_prior_std
    ub = grammar.mech_prior_mean + 4.0 * grammar.mech_prior_std

    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    with torch.cuda.amp.autocast(enabled=use_bf16, dtype=torch.bfloat16):
        pr, f_all, G_mat = model.predict_posterior(
            torch.from_numpy(df).to(device), torch.from_numpy(tm).to(device),
            torch.from_numpy(ov).to(device), torch.from_numpy(om).to(device),
            torch.from_numpy(tp).to(device), torch.from_numpy(ep).to(device),
        )
    probs_m = pr.cpu().numpy()
    f_all, G_mat = f_all.float(), G_mat.float()

    inv_sig_chunks = []
    for des in designs:
        cat0 = des.cat_mult * grammar.c0_base
        for sp in des.obs_species:
            scale = cat0 if grammar.species[sp].is_catalyst else des.A0
            inv_sig_chunks.append(np.full(20, 1.0 / (sigma_rel * scale), dtype=np.float64))
    inv_sig_1d = np.concatenate(inv_sig_chunks)
    n_res = len(inv_sig_1d)

    targets = np.zeros((N, n_res), dtype=np.float64)
    for ci in range(N):
        chunks = []
        for e, des in enumerate(designs):
            for sp in des.obs_species:
                chunks.append(raw_noisy[ci, e, :, obs_ch.index(sp)])
        targets[ci] = np.concatenate(chunks)

    # Top-5 mechanisms per case -> batched sampling & simulation across ALL N cases!
    top_k = 5
    top5_mat = np.argsort(probs_m, axis=1)[:, -top_k:][:, ::-1].copy()  # (N, 5)
    f_rep = f_all.unsqueeze(1).expand(N, top_k, -1).reshape(N * top_k, -1)
    m_flat_k = top5_mat.ravel()
    th_s = model.sample_theta(f_rep, G_mat, torch.from_numpy(m_flat_k).long().to(device), num_samples=n_ppc).cpu().numpy().astype(np.float64)
    # (n_ppc, N*5, P) -> (N*5*n_ppc, P)
    th_ppc = th_s.transpose(1, 0, 2).reshape(N * top_k * n_ppc, grammar.n_params)
    m_ppc = np.repeat(m_flat_k.astype(np.int32), n_ppc)
    th_ppc = np.clip(th_ppc, lb[m_ppc], ub[m_ppc])

    y_sim, st_sim, _, _ = sim.simulate_designs(
        m_ppc, th_ppc, designs, t_eval, compute_flux=False, rtol=5e-5, atol=5e-9
    )
    ok_ppc = np.all(st_sim == 0, axis=1).reshape(N, top_k, n_ppc)
    pred_chunks = []
    for d_i, des in enumerate(designs):
        for sp in des.obs_species:
            pred_chunks.append(y_sim[:, d_i, :, sp])
    pred_all = np.concatenate(pred_chunks, axis=1).reshape(N, top_k, n_ppc, n_res)

    z_amort_med = np.zeros(N, dtype=np.float64)
    z_amort_best = np.zeros(N, dtype=np.float64)
    for ci in range(N):
        best_med, best_dr = 999.0, 999.0
        for k_i in range(top_k):
            ok_k = ok_ppc[ci, k_i]
            if np.any(ok_k):
                pk = pred_all[ci, k_i, ok_k]
                errs = np.sqrt(np.mean(((pk - targets[ci][None, :]) * inv_sig_1d[None, :]) ** 2, axis=1))
                med_c = np.median(pk, axis=0)
                zm = float(np.sqrt(np.mean(((med_c - targets[ci]) * inv_sig_1d) ** 2)))
                best_med = min(best_med, zm)
                best_dr = min(best_dr, float(np.min(errs)))
        z_amort_med[ci] = best_med
        z_amort_best[ci] = best_dr

    # Batched Amortized Warm-Start + 20-iter C++ LM Refinement (3 NSF draws + 1 prior mean per Top-5 mech)
    n_tasks = N * top_k
    inits = np.zeros((n_tasks, 4, grammar.n_params), dtype=np.float64)
    th_s_trans = th_s.transpose(1, 0, 2)  # (N*5, n_ppc, P)
    inits[:, :3, :] = th_s_trans[:, :3, :]
    inits[:, 3, :] = grammar.mech_prior_mean[m_flat_k]

    rms_lm, _ = sim.fit_lm_batch(
        m_flat_k.astype(np.int32), designs, t_eval,
        np.repeat(targets, top_k, axis=0),
        np.broadcast_to(inv_sig_1d[None, :], (n_tasks, n_res)).copy(),
        inits, lb_all=lb, ub_all=ub,
        rtol=5e-5, atol=5e-9, max_ode_steps=8000, max_lm_iters=20, early_stop_rms=0.8,
    )
    z_lm = np.min(rms_lm.reshape(N, top_k), axis=1)

    return {
        "n": N,
        "z_amort_best": z_amort_best,
        "z_amort_med": z_amort_med,
        "z_lm": z_lm,
    }


def run_synthetic_misspec_suite(
    grammar: ReactionGrammar,
    sim: ODESimulator,
    model,
    test_dir: Path,
    device: torch.device,
    n_cases: int = 100,
    seed: int = 4444,
) -> Dict:
    print(f"\n=== [E4 Part 2] 3x{n_cases} Synthetic Misspecification Stress Tests (+ {n_cases} In-Dist Controls) ===", flush=True)
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    t_grid = np.load(test_dir / "t_grid.npy")
    all_mids = np.load(test_dir / "mech_ids.npy")
    all_thetas = np.load(test_dir / "thetas.npy")
    all_rand = np.load(test_dir / "rand_des.npy")
    all_trajs = np.load(test_dir / "trajs.npy", mmap_mode="r")

    sel_mechs = np.linspace(0, grammar.n_mechs - 1, n_cases, dtype=int)
    sel_idx = np.array([np.where(all_mids == m)[0][7] for m in sel_mechs], dtype=np.int64)
    mids = all_mids[sel_idx].astype(np.int64)
    thetas = all_thetas[sel_idx].astype(np.float64)
    trajs = np.array(all_trajs[sel_idx])
    rand_des = all_rand[sel_idx]

    design_indices = [0, 2, 5, 6]
    designs = [grammar.catalog[i] for i in design_indices]
    slots_list = [design_indices for _ in range(n_cases)]
    sigma_rel = 0.01

    # 0. In-Distribution Control
    df_id, tm_id, ov_id, om_id, tp_id, ep_id, raw_id, _ = pack_fixed_test_batch(
        grammar, t_grid, trajs, rand_des, slots_list, sigma_rel=sigma_rel, n_t_obs=20, seed=seed,
    )
    res_id = evaluate_misspec_batch(grammar, sim, model, raw_id, df_id, tm_id, ov_id, om_id, tp_id, ep_id, designs, device, sigma_rel=sigma_rel)

    # Helper to repack custom 4-design clean trajectories (N, 4, 20, n_ch) with 1% Gaussian noise into model tensors
    obs_ch = get_observable_channel_indices(grammar)
    ch_is_cat = np.array([grammar.species[s].is_catalyst for s in obs_ch], dtype=bool)
    ch_A = obs_ch.index(grammar.species_idx["A"])
    ch_P = obs_ch.index(grammar.species_idx["P"])

    def pack_custom_4(clean_4: np.ndarray, noise_seed: int, mult_bias: Optional[np.ndarray] = None):
        rng_n = np.random.default_rng(noise_seed)
        raw_n = np.zeros_like(clean_4)
        ov_c = np.zeros_like(ov_id)
        for e_i, des in enumerate(designs):
            cat0 = des.cat_mult * grammar.c0_base
            n_sc = np.where(ch_is_cat, cat0, des.A0)
            norm_sc = np.where(ch_is_cat, cat0, 1.0)
            y_c = clean_4[:, e_i].copy()
            if mult_bias is not None:
                y_c = y_c * mult_bias[:, None, :]
            y_n = y_c + rng_n.normal(0.0, 1.0, size=y_c.shape) * (sigma_rel * n_sc[None, None, :])
            raw_n[:, e_i] = y_n
            ov_c[:, e_i] = (y_n / norm_sc[None, None, :]).astype(np.float32) * om_id[:, e_i]
        return raw_n, ov_c

    # 1. Sensor Response-Factor Calibration Bias (+/-8% systematic multiplicative offset on A and P)
    # Build exact clean 4-design curves from test pool via interpolation
    t_eval = np.linspace(0.0, 1.0, 20)
    hi_i = np.clip(np.searchsorted(t_grid, t_eval, side="right"), 1, len(t_grid) - 1)
    lo_i = hi_i - 1
    wh = (t_eval - t_grid[lo_i]) / np.maximum(t_grid[hi_i] - t_grid[lo_i], 1e-12)
    clean_id = np.zeros((n_cases, 4, 20, len(obs_ch)), dtype=np.float64)
    for e_i, d_slot in enumerate(design_indices):
        raw_d = trajs[:, d_slot].astype(np.float64)
        clean_id[:, e_i] = raw_d[:, lo_i, :] * (1.0 - wh[None, :, None]) + raw_d[:, hi_i, :] * wh[None, :, None]

    mult_bias = np.ones((n_cases, len(obs_ch)), dtype=np.float64)
    signs_a = rng.choice([-1.0, 1.0], size=n_cases)
    signs_p = rng.choice([-1.0, 1.0], size=n_cases)
    mult_bias[:, ch_A] = 1.0 + 0.08 * signs_a
    mult_bias[:, ch_P] = 1.0 + 0.08 * signs_p
    raw_sb, ov_sb = pack_custom_4(clean_id, seed + 101, mult_bias=mult_bias)
    res_sb = evaluate_misspec_batch(grammar, sim, model, raw_sb, df_id, tm_id, ov_sb, om_id, tp_id, ep_id, designs, device, sigma_rel=sigma_rel)

    # 2. Temperature Drift (alpha in +/- [0.45, 0.75]) & 3. Unmodeled Side Reaction (2A -> X, k_leak in [0.18, 0.40])
    alphas = rng.choice([-1.0, 1.0], size=n_cases) * rng.uniform(0.45, 0.75, size=n_cases)
    kleaks = rng.uniform(0.18, 0.40, size=n_cases)

    tasks_td = [(ci, "temp_drift", int(mids[ci]), thetas[ci], float(alphas[ci]), design_indices) for ci in range(n_cases)]
    tasks_sr = [(ci, "unmodeled_side_rxn", int(mids[ci]), thetas[ci], float(kleaks[ci]), design_indices) for ci in range(n_cases)]

    clean_td = np.zeros_like(clean_id)
    ok_td = np.zeros(n_cases, dtype=bool)
    clean_sr = np.zeros_like(clean_id)
    ok_sr = np.zeros(n_cases, dtype=bool)

    with ProcessPoolExecutor(max_workers=32, initializer=_init_radau_worker) as ex:
        for ci, y_out, ok in ex.map(_sim_misspec_case_worker, tasks_td):
            clean_td[ci] = y_out
            ok_td[ci] = ok
        for ci, y_out, ok in ex.map(_sim_misspec_case_worker, tasks_sr):
            clean_sr[ci] = y_out
            ok_sr[ci] = ok

    raw_td, ov_td = pack_custom_4(clean_td, seed + 202)
    idx_td = np.where(ok_td)[0]
    res_td = evaluate_misspec_batch(
        grammar, sim, model, raw_td[idx_td], df_id[idx_td], tm_id[idx_td], ov_td[idx_td],
        om_id[idx_td], tp_id[idx_td], ep_id[idx_td], designs, device, sigma_rel=sigma_rel,
    )

    raw_sr, ov_sr = pack_custom_4(clean_sr, seed + 303)
    idx_sr = np.where(ok_sr)[0]
    res_sr = evaluate_misspec_batch(
        grammar, sim, model, raw_sr[idx_sr], df_id[idx_sr], tm_id[idx_sr], ov_sr[idx_sr],
        om_id[idx_sr], tp_id[idx_sr], ep_id[idx_sr], designs, device, sigma_rel=sigma_rel,
    )

    def summarize_regime(name: str, r: Dict, r_ctrl: Optional[Dict] = None) -> Dict:
        out = {
            "n": r["n"],
            "amort_best_rate_gt3sigma": float(np.mean(r["z_amort_best"] > 3.0)),
            "amort_best_median_z": float(np.median(r["z_amort_best"])),
            "amort_med_rate_gt3sigma": float(np.mean(r["z_amort_med"] > 3.0)),
            "amort_med_median_z": float(np.median(r["z_amort_med"])),
            "warmstart_lm_rate_gt3sigma": float(np.mean(r["z_lm"] > 3.0)),
            "warmstart_lm_median_z": float(np.median(r["z_lm"])),
            "z_amort_best_values": [round(float(x), 3) for x in r["z_amort_best"]],
            "z_lm_values": [round(float(x), 3) for x in r["z_lm"]],
        }
        if r_ctrl is not None:
            y_true = np.concatenate([np.zeros(r_ctrl["n"]), np.ones(r["n"])])
            out["auroc_amort_best"] = float(roc_auc_score(y_true, np.concatenate([r_ctrl["z_amort_best"], r["z_amort_best"]])))
            out["auroc_warmstart_lm"] = float(roc_auc_score(y_true, np.concatenate([r_ctrl["z_lm"], r["z_lm"]])))
        print(
            f"  [Misspec:{name:20s}] n={out['n']:3d} | Amort >3s={out['amort_best_rate_gt3sigma']*100:5.1f}% (med={out['amort_best_median_z']:.2f}s) | "
            f"LM >3s={out['warmstart_lm_rate_gt3sigma']*100:5.1f}% (med={out['warmstart_lm_median_z']:.2f}s)"
            + (f" | AUROC(A/LM)={out['auroc_amort_best']:.3f}/{out['auroc_warmstart_lm']:.3f}" if r_ctrl else ""),
            flush=True,
        )
        return out

    return {
        "designs": [d.name for d in designs],
        "sigma_rel": sigma_rel,
        "elapsed_s": time.perf_counter() - t0,
        "in_dist_control": summarize_regime("in_dist_control", res_id),
        "temp_drift": summarize_regime("temp_drift", res_td, res_id),
        "sensor_bias_8pct": summarize_regime("sensor_bias_8pct", res_sb, res_id),
        "unmodeled_side_rxn": summarize_regime("unmodeled_side_rxn", res_sr, res_id),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="results/phase2/models/quomech_s42/best.pt")
    ap.add_argument("--equiv-basic-json", type=str, default="results/phase2/equiv/equiv_Xi_basic.json")
    ap.add_argument("--lit-dir", type=str, default="data/real_literature")
    ap.add_argument("--test-dir", type=str, default="data/pool_multicore_527/test")
    ap.add_argument("--out-dir", type=str, default="results/phase2")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--n-threads", type=int, default=32)
    ap.add_argument("--n-misspec", type=int, default=100)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    g = build_multicore_grammar()
    sim = ODESimulator(g, n_threads=args.n_threads)
    c_macro = build_c_macro_partition(g)

    ckpt = torch.load(args.ckpt, map_location=device)
    model = build_model(g, head_type=ckpt.get("head_type", "factorized"), arch=ckpt.get("arch", "p2_256")).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    eq_m2c, eq_classes = None, None
    if Path(args.equiv_basic_json).exists():
        eq = json.loads(Path(args.equiv_basic_json).read_text())["primary"]
        eq_m2c = np.array(eq["mech_to_class"], dtype=np.int64)
        eq_classes = eq["classes"]

    lit_dir = Path(args.lit_dir)
    manifest = json.loads((lit_dir / "manifest.json").read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== [E4 Part 1] Real Literature Evaluation ({len(manifest['cases'])} systems, ckpt={args.ckpt}) ===", flush=True)
    real_cases, plot_curves = [], {}
    for idx, entry in enumerate(manifest["cases"]):
        cdir = lit_dir / entry["case_id"]
        meta = json.loads((cdir / "meta.json").read_text())
        case = load_real_case(g, cdir, meta)
        res, pbundle = evaluate_real_case_full(
            g, sim, model, case, meta, device, eq_m2c, eq_classes, c_macro, seed=42 + idx
        )
        real_cases.append(res)
        plot_curves[entry["case_id"]] = pbundle
        tag = "IN " if res["in_library"] else "OOD"
        lm = res["warmstart_local_lm"]
        loc = res["local_overlap"]
        bo = res["boed_next_exp_from_ref"]["top3_recommended"][0]
        print(
            f"  [{tag}] {res['case_id']:42s} | BIC*={lm['bic_winner_mech']:20s} "
            f"z_lm={lm['best_fit_z_sig']:4.2f}s (core={lm['best_core_z_sig']:4.2f}s) "
            f"alert={lm['combined_ood_alert']} | LOO={res['loo_ppe']['mean']:.4f} | "
            f"loc={loc['elapsed_ms']:5.0f}ms | next={bo['design']}",
            flush=True,
        )

    misspec_res = run_synthetic_misspec_suite(g, sim, model, Path(args.test_dir), device, n_cases=args.n_misspec)

    full_out = {
        "ckpt": args.ckpt,
        "real_literature": {
            "n_cases": len(real_cases),
            "n_in_library": sum(1 for c in real_cases if c["in_library"]),
            "n_ood": sum(1 for c in real_cases if not c["in_library"]),
            "median_local_overlap_ms": float(np.median([c["local_overlap"]["elapsed_ms"] for c in real_cases])),
            "median_warmstart_lm_ms": float(np.median([c["warmstart_local_lm"]["elapsed_ms"] for c in real_cases])),
            "mean_in_lib_loo_ppe": float(np.mean([c["loo_ppe"]["mean"] for c in real_cases if c["in_library"]])),
            "cases": real_cases,
        },
        "synthetic_misspecification": misspec_res,
    }
    (out_dir / "stage4_e4_full.json").write_text(json.dumps(full_out, indent=2))
    (out_dir / "stage4_e4_plot_curves.json").write_text(json.dumps(plot_curves, indent=2))
    print(f"\n[Stage 4 E4 DONE] Saved to {out_dir / 'stage4_e4_full.json'}", flush=True)


if __name__ == "__main__":
    main()
