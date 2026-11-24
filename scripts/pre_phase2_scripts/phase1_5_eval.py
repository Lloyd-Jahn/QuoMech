#!/usr/bin/env python3
"""Phase 1.5 Comprehensive Validation Suite (Tasks 1, 2, 3, 4).

Executes:
  Task 1: 60k-step convergence & overfitting analysis (16k / 30k / 45k / 60k train vs val loss + E1 suite).
  Task 2: Un-truncated discrete 3-tier BOED (302 cases = 2/mech, T_eff = min(6, |Xi|-1), N_MC=512,
          strict CRN, paired Bootstrap 95% CI, 0.85 & 0.90 stopping thresholds, same-state counterfactual
          disagreement test on xi_Q* != xi_F*).
  Task 3: Continuous design space X_basic (LHS-128 candidate pool, t_end = clip(1/cat_mult, 0.5, 2.5),
          151 cases, T=6, N_MC=256, strict CRN) head-to-head: Continuous Random vs Continuous Fine EIG
          vs Continuous Quotient EIG + Step-1 VTNA & Step-2 RPKA spatial clustering.
  Task 4: (a) Cost-normalized BOED under Xi_full (total cost <= 6, cost(D9)=3): Quotient EIG vs
              Quotient EIG/cost vs Fine EIG/cost (AUC, final Top-1, D9 selection rate).
          (b) Instrument capability upgrade recommendation (Xi_basic -> Xi_full \ Xi_basic = {D7,D8,D9})
              on converged non-trivial Xi_basic cases (and M4/M7 family): D9 Top-1 hit rate & posterior gain.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
from scipy.stats import qmc
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quomech.boed import _build_single_exp_tensors, _score_eig_designs_batched
from quomech.data_gen import get_observable_channel_indices
from quomech.evaluate import aggregate_class_probs, evaluate_e1_suite, interp_traj_to_times
from quomech.grammar import DesignSpec, ReactionGrammar, build_default_grammar
from quomech.models import QuoMechNet
from quomech.simulator import ODESimulator
from quomech.train import build_model


def paired_bootstrap_ci(
    diff_arr: np.ndarray, n_boot: int = 10000, seed: int = 42
) -> Dict[str, float]:
    """Compute mean, std, and 95% paired bootstrap confidence interval for a 1D difference array."""
    x = np.asarray(diff_arr, dtype=np.float64)
    n = len(x)
    if n == 0:
        return {"n": 0, "mean": 0.0, "ci95_lo": 0.0, "ci95_hi": 0.0, "win_rate": 0.0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = np.mean(x[idx], axis=1)
    return {
        "n": int(n),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "ci95_lo": float(np.percentile(boot_means, 2.5)),
        "ci95_hi": float(np.percentile(boot_means, 97.5)),
        "win_rate": float(np.mean(x > 0.0)),
    }


def load_ckpt_model(grammar: ReactionGrammar, ckpt_path: Path, device: torch.device) -> Tuple[QuoMechNet, dict]:
    model = build_model(grammar, head_type="factorized").to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def eval_task1_convergence(
    grammar: ReactionGrammar,
    sim: ODESimulator,
    test_dir: Path,
    equiv_tiers: Dict[str, Dict],
    model_60k_dir: Path,
    p1_e1_path: Path,
    device: torch.device,
    n_test: int = 1510,
) -> Dict:
    """Task 1: Compare 16k / 30k / 45k / 60k milestones and full E1 suite vs Phase 1 16k model."""
    summary_path = model_60k_dir / "train_summary.json"
    train_summary = json.load(open(summary_path)) if summary_path.exists() else {}
    history = train_summary.get("history", [])

    milestone_table = {}
    for h in history:
        st = h["step"]
        if st in (16000, 30000, 45000, 60000) or st == train_summary.get("total_steps"):
            milestone_table[str(st)] = h

    # Evaluate final 60k best checkpoint on E1 suite
    best_path = model_60k_dir / "best.pt"
    model_60k, ckpt_best = load_ckpt_model(grammar, best_path, device)
    e1_60k = evaluate_e1_suite(
        model_60k, grammar, sim, test_dir, equiv_tiers, device,
        n_test=n_test, run_local_overlap=True, seed=2026,
    )

    best_step = int(ckpt_best.get("step", 60000))
    e1_60k_final = None
    final_cp = model_60k_dir / "ckpt_60000.pt"
    if best_step != 60000 and final_cp.exists():
        m_fin, _ = load_ckpt_model(grammar, final_cp, device)
        e1_60k_final = evaluate_e1_suite(
            m_fin, grammar, sim, test_dir, equiv_tiers, device,
            n_test=n_test, run_local_overlap=False, seed=2026,
        )

    # Also evaluate intermediate 16k/30k/45k/60k checkpoints on key regimes if saved
    milestone_e1 = {}
    for st in [16000, 30000, 45000, 60000]:
        cp = model_60k_dir / f"ckpt_{st}.pt"
        if cp.exists():
            m_st, _ = load_ckpt_model(grammar, cp, device)
            e1_st = evaluate_e1_suite(
                m_st, grammar, sim, test_dir, equiv_tiers, device,
                n_test=n_test, run_local_overlap=False, seed=2026,
            )
            milestone_e1[str(st)] = {
                rn: {
                    "class_top1": e1_st["regimes"][rn]["class_top1"],
                    "class_top3": e1_st["regimes"][rn]["class_top3"],
                    "class_ece": e1_st["regimes"][rn]["class_ece"],
                    "mech_top1": e1_st["regimes"][rn]["mech_top1"],
                    "sbc_ks_median": e1_st["regimes"][rn].get("sbc_ks_median_param"),
                    "ppe_median": e1_st["regimes"][rn].get("heldout_D7_ppe_rmse_median"),
                }
                for rn in [
                    "cat_4_D0_D2_D5_D6", "cat_7_basic_all", "cat_basic_var_window",
                    "cat_10_full_all", "cont_4_var_window", "P_only_basic",
                ]
            }

    p1_e1 = json.load(open(p1_e1_path)) if p1_e1_path.exists() else {}
    return {
        "best_step": best_step,
        "train_milestones": milestone_table,
        "full_history": history,
        "milestone_e1_key_regimes": milestone_e1,
        "e1_60k_best": e1_60k,
        "e1_60k_final": e1_60k_final,
        "e1_phase1_16k_ref": p1_e1.get("regimes", {}),
    }


def _summarize_boed_trajectories(
    pc_traj: np.ndarray,       # (N, T_max + 1)
    pm_traj: np.ndarray,       # (N, T_max + 1)
    max_pc_traj: np.ndarray,   # (N, T_max + 1)
    max_pm_traj: np.ndarray,   # (N, T_max + 1)
    top1_traj: np.ndarray,     # (N, T_max + 1)
    horizons: Sequence[int],   # e.g. [4, T_eff]
    thresholds: Sequence[float] = (0.85, 0.90),
) -> Tuple[Dict, Dict[str, np.ndarray]]:
    """Compute step-by-step and horizon-specific metrics + per-case arrays for paired bootstrap."""
    N, T_plus_1 = pc_traj.shape
    T_max = T_plus_1 - 1
    out = {
        "n": int(N),
        "mean_pc_by_step": [float(x) for x in np.mean(pc_traj, axis=0)],
        "top1_by_step": [float(x) for x in np.mean(top1_traj, axis=0)],
        "horizons": {},
    }
    raw_arrays: Dict[str, np.ndarray] = {}

    for H in horizons:
        if H > T_max:
            continue
        sub_pc = pc_traj[:, : H + 1]
        sub_max_pc = max_pc_traj[:, : H + 1]
        sub_max_pm = max_pm_traj[:, : H + 1]
        sub_top1 = top1_traj[:, : H + 1]

        auc_arr = np.mean(sub_pc, axis=1)
        top1_H_arr = sub_top1[:, H]
        pc_H_arr = sub_pc[:, H]
        raw_arrays[f"auc@{H}"] = auc_arr
        raw_arrays[f"top1@{H}"] = top1_H_arr
        raw_arrays[f"pc@{H}"] = pc_H_arr

        h_dict = {
            "auc_true_class_prob": float(np.mean(auc_arr)),
            "top1_at_H": float(np.mean(top1_H_arr)),
            "true_class_prob_at_H": float(np.mean(pc_H_arr)),
            "stopping": {},
        }

        for tau in thresholds:
            t_key = f"tau_{int(round(tau * 100))}"
            t_stop_cls = np.full(N, H, dtype=np.int64)
            t_stop_fine = np.full(N, H, dtype=np.int64)
            reached_cls = np.zeros(N, dtype=bool)
            reached_fine = np.zeros(N, dtype=bool)
            t_true_tau = np.full(N, H + 1, dtype=np.int64)
            acc_stop_cls = np.zeros(N, dtype=np.float64)
            acc_stop_fine = np.zeros(N, dtype=np.float64)

            for i in range(N):
                for ti in range(H + 1):
                    if sub_max_pc[i, ti] >= tau:
                        t_stop_cls[i] = ti
                        reached_cls[i] = True
                        break
                for ti in range(H + 1):
                    if sub_max_pm[i, ti] >= tau:
                        t_stop_fine[i] = ti
                        reached_fine[i] = True
                        break
                for ti in range(H + 1):
                    if sub_pc[i, ti] >= tau:
                        t_true_tau[i] = ti
                        break
                acc_stop_cls[i] = sub_top1[i, t_stop_cls[i]]
                acc_stop_fine[i] = sub_top1[i, t_stop_fine[i]]

            wasted = np.maximum(0, t_stop_fine - t_stop_cls).astype(np.float64)
            early_cls_mask = reached_cls & (t_stop_cls < H)
            wasted_on_early = float(np.mean(wasted[early_cls_mask])) if np.any(early_cls_mask) else 0.0

            raw_arrays[f"cls_stop@{H}_{t_key}"] = t_stop_cls.astype(np.float64)
            raw_arrays[f"fine_stop@{H}_{t_key}"] = t_stop_fine.astype(np.float64)
            raw_arrays[f"wasted@{H}_{t_key}"] = wasted

            h_dict["stopping"][t_key] = {
                "reach_true_tau_rate": float(np.mean(t_true_tau <= H)),
                "mean_exps_to_true_tau_capped": float(np.mean(np.minimum(t_true_tau, H))),
                "cls_stop_reach_rate": float(np.mean(reached_cls)),
                "cls_early_stop_rate": float(np.mean(early_cls_mask)),
                "fine_stop_reach_rate": float(np.mean(reached_fine)),
                "unified_class_stop_mean_exps": float(np.mean(t_stop_cls)),
                "unified_class_stop_top1": float(np.mean(acc_stop_cls)),
                "fine_stop_mean_exps": float(np.mean(t_stop_fine)),
                "fine_stop_top1": float(np.mean(acc_stop_fine)),
                "wasted_exps_all": float(np.mean(wasted)),
                "wasted_exps_given_cls_early_stopped": wasted_on_early,
            }

        out["horizons"][f"T={H}"] = h_dict

    return out, raw_arrays


@torch.no_grad()
def eval_task2_discrete_boed(
    model: QuoMechNet,
    grammar: ReactionGrammar,
    sim: ODESimulator,
    test_dir: Path,
    equiv_tiers: Dict[str, Dict],
    device: torch.device,
    n_per_mech: int = 2,
    n_mc: int = 512,
    sigma_rel: float = 0.01,
    seed: int = 202615,
) -> Dict:
    """Task 2: 302-case (2/mech) discrete 3-tier BOED with T_eff = min(6, |Xi|-1), N_MC=512, strict CRN."""
    model.eval()
    rng_obs = np.random.default_rng(seed)
    obs_ch = get_observable_channel_indices(grammar)
    ch_is_cat = np.array([grammar.species[s].is_catalyst for s in obs_ch], dtype=bool)
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    t_grid = np.load(test_dir / "t_grid.npy")
    all_mids = np.load(test_dir / "mech_ids.npy")
    all_trajs = np.load(test_dir / "trajs.npy")

    # Select n_per_mech=2 cases per mechanism (indices 5 and 6) -> 302 cases
    sel_list = []
    for m in range(grammar.n_mechs):
        idxs = np.where(all_mids == m)[0]
        sel_list.extend(idxs[5 : 5 + n_per_mech].tolist())
    sel_idx = np.array(sel_list, dtype=np.int64)
    N = len(sel_idx)
    true_mids = all_mids[sel_idx].astype(np.int64)
    trajs_sel = all_trajs[sel_idx]

    n_t = 20
    t_eval = np.linspace(0.0, 1.0, n_t)
    catalog_pool = [(d, 1.0) for d in grammar.catalog]

    # Strict CRN: pre-generate fixed noisy observations y*(i, d) for all N x 10 catalog designs
    exp_embs_all = torch.empty((N, 10, model.exp_encoder.d_model), device=device, dtype=torch.float32)
    for d_idx in range(10):
        des = grammar.catalog[d_idx]
        clean = np.stack(
            [interp_traj_to_times(t_grid, trajs_sel[i, d_idx].astype(np.float64), t_eval) for i in range(N)],
            axis=0,
        )
        cat0 = des.cat_mult * grammar.c0_base
        noise_scale = np.where(ch_is_cat, cat0, des.A0)
        y_n = clean + rng_obs.normal(0.0, 1.0, size=clean.shape) * (sigma_rel * noise_scale[None, None, :])
        df_np, tm_np, ov_np, om_np, tp_np = _build_single_exp_tensors(grammar, des, 1.0, y_n, t_eval)
        with torch.cuda.amp.autocast(enabled=use_bf16, dtype=torch.bfloat16):
            emb = model.exp_encoder(
                torch.from_numpy(df_np).to(device), torch.from_numpy(tm_np).to(device),
                torch.from_numpy(ov_np).to(device), torch.from_numpy(om_np).to(device),
                torch.from_numpy(tp_np).to(device),
            )
        exp_embs_all[:, d_idx] = emb.float()

    _, G_mat = model.mech_head(torch.zeros((1, 256), device=device))
    G_mat = G_mat.float()

    def eval_state(chosen_hist: List[List[int]]) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        E = len(chosen_hist[0])
        idx_t = torch.tensor(chosen_hist, dtype=torch.long, device=device)
        ar = torch.arange(N, device=device).unsqueeze(1).expand(N, E)
        h_seq = exp_embs_all[ar, idx_t]
        ep = torch.zeros((N, E), dtype=torch.bool, device=device)
        with torch.cuda.amp.autocast(enabled=use_bf16, dtype=torch.bfloat16):
            f = model.set_encoder(h_seq, exp_pad_mask=ep)
            logits, _ = model.mech_head(f)
        probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        return probs, f.float(), h_seq

    probs_m_d0, f_d0, h_d0 = eval_state([[0] for _ in range(N)])

    out_tiers = {}
    for tier in ["no_P", "basic", "full"]:
        t0_tier = time.perf_counter()
        tier_designs = list(grammar.tiers[tier])
        T_eff = min(6, len(tier_designs) - 1)
        horizons = sorted(list({4, T_eff}))

        eq = equiv_tiers[tier]["primary"]
        m2c = np.array(eq["mech_to_class"], dtype=np.int64)
        classes = eq["classes"]
        n_cls = len(classes)
        true_c = m2c[true_mids]
        nontriv_mask = np.array([len(classes[c]) >= 2 for c in true_c], dtype=bool)
        probs_c_d0 = aggregate_class_probs(probs_m_d0, m2c, n_cls)

        # Step 1 (from D0) shared strict-CRN EIG evaluation for both Quotient and Fine-grained
        cand_d0 = [[d for d in tier_designs if d != 0] for _ in range(N)]
        rng_s0 = np.random.default_rng(seed + 100)
        eig_q_d0, eig_f_d0 = _score_eig_designs_batched(
            model, grammar, sim, h_d0, f_d0, G_mat, probs_m_d0, probs_c_d0,
            catalog_pool, cand_d0, m2c, n_cls, n_t, sigma_rel, n_mc=n_mc, device=device, rng=rng_s0,
            crn_seed=seed + 100,
        )

        # Track same-state counterfactual disagreements along Quotient trajectory
        cf_dq_diff_all, cf_dlogq_diff_all = [], []
        cf_dq_q_all, cf_dq_f_all = [], []
        cf_dlogq_q_all, cf_dlogq_f_all = [], []
        cf_dq_diff_nontriv, cf_dlogq_diff_nontriv = [], []
        cf_dq_q_nontriv, cf_dq_f_nontriv = [], []
        cf_dq_diff_unsolved90, cf_dlogq_diff_unsolved90 = [], []
        cf_dq_q_unsolved90, cf_dq_f_unsolved90 = [], []
        cf_dq_diff_unsolved85, cf_dlogq_diff_unsolved85 = [], []
        disagree_by_step = []
        disagree_count_by_step = []
        disagree_nontriv_by_step = []
        disagree_unsolved90_by_step = []
        disagree_unsolved85_by_step = []
        intra_frac_by_step = []
        intra_frac_nontriv_by_step = []

        def run_policy(policy_name: str, record_cf: bool = False):
            hist = [[0] for _ in range(N)]
            pc_traj = np.zeros((N, T_eff + 1), dtype=np.float64)
            pm_traj = np.zeros((N, T_eff + 1), dtype=np.float64)
            max_pc_traj = np.zeros((N, T_eff + 1), dtype=np.float64)
            max_pm_traj = np.zeros((N, T_eff + 1), dtype=np.float64)
            top1_traj = np.zeros((N, T_eff + 1), dtype=np.float64)

            pm_curr = probs_m_d0.copy()
            pc_curr = probs_c_d0.copy()
            f_curr = f_d0
            h_curr = h_d0

            pc_traj[:, 0] = pc_curr[np.arange(N), true_c]
            pm_traj[:, 0] = pm_curr[np.arange(N), true_mids]
            max_pc_traj[:, 0] = np.max(pc_curr, axis=1)
            max_pm_traj[:, 0] = np.max(pm_curr, axis=1)
            top1_traj[:, 0] = (np.argmax(pc_curr, axis=1) == true_c).astype(np.float64)
            rng_rand = np.random.default_rng(seed + 7777)

            for step in range(T_eff):
                cands_all = [[d for d in tier_designs if d not in hist[i]] for i in range(N)]
                if policy_name == "Random":
                    chosen = [int(rng_rand.choice(cands_all[i])) for i in range(N)]
                else:
                    if step == 0:
                        eq_list, ef_list = eig_q_d0, eig_f_d0
                    else:
                        step_seed = seed + 100 + step * 1000
                        rng_step = np.random.default_rng(step_seed)
                        eq_list, ef_list = _score_eig_designs_batched(
                            model, grammar, sim, h_curr, f_curr, G_mat, pm_curr, pc_curr,
                            catalog_pool, cands_all, m2c, n_cls, n_t, sigma_rel, n_mc=n_mc, device=device,
                            rng=rng_step, crn_seed=step_seed,
                        )

                    q_picks = [cands_all[i][int(np.argmax(eq_list[i]))] for i in range(N)]
                    f_picks = [cands_all[i][int(np.argmax(ef_list[i]))] for i in range(N)]

                    if record_cf:
                        dis_mask = [q_picks[i] != f_picks[i] for i in range(N)]
                        unsolved90 = [max_pc_traj[i, step] < 0.90 for i in range(N)]
                        unsolved85 = [max_pc_traj[i, step] < 0.85 for i in range(N)]
                        disagree_by_step.append(float(np.mean(dis_mask)))
                        disagree_count_by_step.append(int(sum(dis_mask)))
                        nt_dis = [dis_mask[i] for i in range(N) if nontriv_mask[i]]
                        disagree_nontriv_by_step.append(float(np.mean(nt_dis)) if nt_dis else 0.0)
                        u90_dis = [dis_mask[i] for i in range(N) if unsolved90[i]]
                        u85_dis = [dis_mask[i] for i in range(N) if unsolved85[i]]
                        disagree_unsolved90_by_step.append(float(np.mean(u90_dis)) if u90_dis else 0.0)
                        disagree_unsolved85_by_step.append(float(np.mean(u85_dis)) if u85_dis else 0.0)

                        # Intra-class EIG artifact fraction at fine-grained choice: (EIG_M - EIG_C) / EIG_M
                        ifracs, ifracs_nt = [], []
                        for i in range(N):
                            idx_f = int(np.argmax(ef_list[i]))
                            em_val = float(ef_list[i][idx_f])
                            ec_val = float(eq_list[i][idx_f])
                            if em_val > 1e-6:
                                frac = max(0.0, (em_val - ec_val) / em_val)
                                ifracs.append(frac)
                                if nontriv_mask[i]:
                                    ifracs_nt.append(frac)
                        intra_frac_by_step.append(float(np.mean(ifracs)) if ifracs else 0.0)
                        intra_frac_nontriv_by_step.append(float(np.mean(ifracs_nt)) if ifracs_nt else 0.0)

                        dis_indices = [i for i in range(N) if dis_mask[i]]
                        if dis_indices:
                            hist_q = [hist[i] + [q_picks[i]] for i in range(N)]
                            hist_f = [hist[i] + [f_picks[i]] for i in range(N)]
                            pm_q_cf, _, _ = eval_state(hist_q)
                            pm_f_cf, _, _ = eval_state(hist_f)
                            pc_q_cf = aggregate_class_probs(pm_q_cf, m2c, n_cls)
                            pc_f_cf = aggregate_class_probs(pm_f_cf, m2c, n_cls)
                            for i in dis_indices:
                                p0 = float(np.clip(pc_traj[i, step], 1e-12, 1.0))
                                pq = float(np.clip(pc_q_cf[i, true_c[i]], 1e-12, 1.0))
                                pf = float(np.clip(pc_f_cf[i, true_c[i]], 1e-12, 1.0))
                                dq_q, dq_f = pq - p0, pf - p0
                                dlogq_q, dlogq_f = float(np.log(pq) - np.log(p0)), float(np.log(pf) - np.log(p0))
                                dq = dq_q - dq_f
                                dlogq = dlogq_q - dlogq_f
                                cf_dq_q_all.append(dq_q)
                                cf_dq_f_all.append(dq_f)
                                cf_dlogq_q_all.append(dlogq_q)
                                cf_dlogq_f_all.append(dlogq_f)
                                cf_dq_diff_all.append(dq)
                                cf_dlogq_diff_all.append(dlogq)
                                if nontriv_mask[i]:
                                    cf_dq_q_nontriv.append(dq_q)
                                    cf_dq_f_nontriv.append(dq_f)
                                    cf_dq_diff_nontriv.append(dq)
                                    cf_dlogq_diff_nontriv.append(dlogq)
                                if unsolved90[i]:
                                    cf_dq_q_unsolved90.append(dq_q)
                                    cf_dq_f_unsolved90.append(dq_f)
                                    cf_dq_diff_unsolved90.append(dq)
                                    cf_dlogq_diff_unsolved90.append(dlogq)
                                if unsolved85[i]:
                                    cf_dq_diff_unsolved85.append(dq)
                                    cf_dlogq_diff_unsolved85.append(dlogq)

                    chosen = q_picks if policy_name == "Quotient_EIG" else f_picks

                for i in range(N):
                    hist[i].append(chosen[i])

                pm_curr, f_curr, h_curr = eval_state(hist)
                pc_curr = aggregate_class_probs(pm_curr, m2c, n_cls)
                pc_traj[:, step + 1] = pc_curr[np.arange(N), true_c]
                pm_traj[:, step + 1] = pm_curr[np.arange(N), true_mids]
                max_pc_traj[:, step + 1] = np.max(pc_curr, axis=1)
                max_pm_traj[:, step + 1] = np.max(pm_curr, axis=1)
                top1_traj[:, step + 1] = (np.argmax(pc_curr, axis=1) == true_c).astype(np.float64)

            all_sum, all_raw = _summarize_boed_trajectories(
                pc_traj, pm_traj, max_pc_traj, max_pm_traj, top1_traj, horizons=horizons
            )
            nt_sum, nt_raw = _summarize_boed_trajectories(
                pc_traj[nontriv_mask], pm_traj[nontriv_mask],
                max_pc_traj[nontriv_mask], max_pm_traj[nontriv_mask],
                top1_traj[nontriv_mask], horizons=horizons
            )
            return {"all_cases": all_sum, "nontrivial_cases": nt_sum}, {"all": all_raw, "nontriv": nt_raw}

        q_res, q_raw = run_policy("Quotient_EIG", record_cf=True)
        f_res, f_raw = run_policy("Fine_EIG", record_cf=False)
        r_res, r_raw = run_policy("Random", record_cf=False)

        # Compute paired bootstrap 95% CIs (Quotient - Fine, and Fine_stop - Quotient_cls_stop waste)
        paired_ci = {"all_cases": {}, "nontrivial_cases": {}}
        for sub_k, out_k in [("all", "all_cases"), ("nontriv", "nontrivial_cases")]:
            qr, fr, rr = q_raw[sub_k], f_raw[sub_k], r_raw[sub_k]
            for H in horizons:
                paired_ci[out_k][f"delta_auc@{H}"] = paired_bootstrap_ci(qr[f"auc@{H}"] - fr[f"auc@{H}"])
                paired_ci[out_k][f"delta_top1@{H}"] = paired_bootstrap_ci(qr[f"top1@{H}"] - fr[f"top1@{H}"])
                paired_ci[out_k][f"delta_pc@{H}"] = paired_bootstrap_ci(qr[f"pc@{H}"] - fr[f"pc@{H}"])
                paired_ci[out_k][f"Q_minus_Rand_auc@{H}"] = paired_bootstrap_ci(qr[f"auc@{H}"] - rr[f"auc@{H}"])
                paired_ci[out_k][f"Q_minus_Rand_top1@{H}"] = paired_bootstrap_ci(qr[f"top1@{H}"] - rr[f"top1@{H}"])
                for t_key in ["tau_85", "tau_90"]:
                    paired_ci[out_k][f"delta_cls_stop@{H}_{t_key}_F_minus_Q"] = paired_bootstrap_ci(
                        fr[f"cls_stop@{H}_{t_key}"] - qr[f"cls_stop@{H}_{t_key}"]
                    )
                    paired_ci[out_k][f"fine_wasted_exps@{H}_{t_key}"] = paired_bootstrap_ci(
                        fr[f"wasted@{H}_{t_key}"]
                    )
                    paired_ci[out_k][f"end2end_budget_saving_Ffine_minus_Qcls@{H}_{t_key}"] = paired_bootstrap_ci(
                        fr[f"fine_stop@{H}_{t_key}"] - qr[f"cls_stop@{H}_{t_key}"]
                    )

        dt_tier = time.perf_counter() - t0_tier
        out_tiers[tier] = {
            "n_cases": N,
            "n_nontrivial_cases": int(np.sum(nontriv_mask)),
            "T_eff": T_eff,
            "n_mc": n_mc,
            "elapsed_s": dt_tier,
            "Quotient_EIG": q_res,
            "Fine_EIG": f_res,
            "Random": r_res,
            "paired_bootstrap_ci": paired_ci,
            "same_state_counterfactual": {
                "disagree_rate_by_step": disagree_by_step,
                "disagree_count_by_step": disagree_count_by_step,
                "disagree_rate_mean_all_steps": float(np.mean(disagree_by_step)),
                "disagree_nontriv_by_step": disagree_nontriv_by_step,
                "disagree_nontriv_mean": float(np.mean(disagree_nontriv_by_step)),
                "disagree_rate_unsolved90_by_step": disagree_unsolved90_by_step,
                "disagree_rate_mean_unsolved90": float(np.mean(disagree_unsolved90_by_step)),
                "disagree_rate_unsolved85_by_step": disagree_unsolved85_by_step,
                "disagree_rate_mean_unsolved85": float(np.mean(disagree_unsolved85_by_step)),
                "intra_class_eig_frac_by_step": intra_frac_by_step,
                "intra_class_eig_frac_mean": float(np.mean(intra_frac_by_step)),
                "intra_class_eig_frac_nontriv_by_step": intra_frac_nontriv_by_step,
                "intra_class_eig_frac_nontriv_mean": float(np.mean(intra_frac_nontriv_by_step)),
                "mean_step_gain_q_on_disagree": float(np.mean(cf_dq_q_all)) if cf_dq_q_all else 0.0,
                "mean_step_gain_f_on_disagree": float(np.mean(cf_dq_f_all)) if cf_dq_f_all else 0.0,
                "mean_step_gain_q_nontriv": float(np.mean(cf_dq_q_nontriv)) if cf_dq_q_nontriv else 0.0,
                "mean_step_gain_f_nontriv": float(np.mean(cf_dq_f_nontriv)) if cf_dq_f_nontriv else 0.0,
                "mean_step_gain_q_unsolved90": float(np.mean(cf_dq_q_unsolved90)) if cf_dq_q_unsolved90 else 0.0,
                "mean_step_gain_f_unsolved90": float(np.mean(cf_dq_f_unsolved90)) if cf_dq_f_unsolved90 else 0.0,
                "mean_step_log_gain_q_on_disagree": float(np.mean(cf_dlogq_q_all)) if cf_dlogq_q_all else 0.0,
                "mean_step_log_gain_f_on_disagree": float(np.mean(cf_dlogq_f_all)) if cf_dlogq_f_all else 0.0,
                "all_disagreements_delta_q": paired_bootstrap_ci(np.array(cf_dq_diff_all)),
                "all_disagreements_delta_logq": paired_bootstrap_ci(np.array(cf_dlogq_diff_all)),
                "nontriv_disagreements_delta_q": paired_bootstrap_ci(np.array(cf_dq_diff_nontriv)),
                "nontriv_disagreements_delta_logq": paired_bootstrap_ci(np.array(cf_dlogq_diff_nontriv)),
                "unsolved90_disagreements_delta_q": paired_bootstrap_ci(np.array(cf_dq_diff_unsolved90)),
                "unsolved90_disagreements_delta_logq": paired_bootstrap_ci(np.array(cf_dlogq_diff_unsolved90)),
                "unsolved85_disagreements_delta_q": paired_bootstrap_ci(np.array(cf_dq_diff_unsolved85)),
                "unsolved85_disagreements_delta_logq": paired_bootstrap_ci(np.array(cf_dlogq_diff_unsolved85)),
            },
        }
        print(
            f"[Task2:{tier}] Done in {dt_tier:.1f}s | "
            f"Q AUC@{T_eff}={q_res['all_cases']['horizons'][f'T={T_eff}']['auc_true_class_prob']:.3f} "
            f"top1@{T_eff}={q_res['all_cases']['horizons'][f'T={T_eff}']['top1_at_H']:.3f} vs "
            f"F AUC@{T_eff}={f_res['all_cases']['horizons'][f'T={T_eff}']['auc_true_class_prob']:.3f} "
            f"top1@{T_eff}={f_res['all_cases']['horizons'][f'T={T_eff}']['top1_at_H']:.3f} | "
            f"CF disagree={np.mean(disagree_by_step):.3f} (n={sum(disagree_count_by_step)}), "
            f"CF delta_q={np.mean(cf_dq_diff_all) if cf_dq_diff_all else 0:.4f}",
            flush=True,
        )

    return out_tiers


@torch.no_grad()
def eval_task3_continuous_lhs128(
    model: QuoMechNet,
    grammar: ReactionGrammar,
    sim: ODESimulator,
    test_dir: Path,
    equiv_basic: Dict,
    device: torch.device,
    n_cases: int = 151,
    n_lhs: int = 128,
    budget: int = 6,
    n_mc: int = 256,
    sigma_rel: float = 0.01,
    seed: int = 202633,
) -> Dict:
    """Task 3: Continuous design space X_basic (LHS-128, t_end=clip(1/cat_mult, 0.5, 2.5)) head-to-head."""
    t0 = time.perf_counter()
    model.eval()
    rng_obs = np.random.default_rng(seed)
    obs_ch = get_observable_channel_indices(grammar)
    ch_is_cat = np.array([grammar.species[s].is_catalyst for s in obs_ch], dtype=bool)
    sA, sP = grammar.species_idx["A"], grammar.species_idx["P"]
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    # Build LHS-128 candidate pool in X_basic: a_pre=0, obs={A,P}, t_end = clip(1.0 / cat_mult, 0.5, 2.5)
    sampler = qmc.LatinHypercube(d=4, seed=seed)
    u = sampler.random(n=n_lhs)
    A0_arr = 0.25 + u[:, 0] * (2.0 - 0.25)
    B0_arr = 0.25 + u[:, 1] * (3.0 - 0.25)
    P0_arr = np.where(u[:, 2] < 0.5, 0.0, 0.1 + ((u[:, 2] - 0.5) / 0.5) * 0.9)
    cat_arr = 10.0 ** (np.log10(0.25) + u[:, 3] * (np.log10(4.0) - np.log10(0.25)))
    tend_arr = np.clip(1.0 / cat_arr, 0.5, 2.5)

    lhs_pool: List[Tuple[DesignSpec, float]] = [
        (
            DesignSpec(
                f"LHS_{k:03d}", float(A0_arr[k]), float(B0_arr[k]), float(P0_arr[k]),
                float(cat_arr[k]), 0, (sA, sP), 1.0, (),
            ),
            float(tend_arr[k]),
        )
        for k in range(n_lhs)
    ]

    t_grid = np.load(test_dir / "t_grid.npy")
    all_mids = np.load(test_dir / "mech_ids.npy")
    all_thetas = np.load(test_dir / "thetas.npy")
    all_trajs = np.load(test_dir / "trajs.npy")
    if n_cases == 2 * grammar.n_mechs:
        sel_list = []
        for m in range(grammar.n_mechs):
            idxs = np.where(all_mids == m)[0]
            sel_list.extend(idxs[5:7].tolist())
        sel_idx = np.array(sel_list, dtype=np.int64)
    else:
        sel_mechs = np.linspace(0, grammar.n_mechs - 1, n_cases, dtype=int)
        # Align with Task 2's first case per mechanism (index 5)
        sel_idx = np.array([np.where(all_mids == m)[0][5] for m in sel_mechs], dtype=np.int64)
    N = len(sel_idx)
    true_mids = all_mids[sel_idx].astype(np.int64)
    true_thetas = all_thetas[sel_idx].astype(np.float64)

    eq = equiv_basic["primary"]
    m2c = np.array(eq["mech_to_class"], dtype=np.int64)
    classes = eq["classes"]
    n_cls = len(classes)
    true_c = m2c[true_mids]
    nontriv_mask = np.array([len(classes[c]) >= 2 for c in true_c], dtype=bool)
    n_t = 20

    # 1. Pre-simulate & pre-encode fixed CRN ground-truth observations y*(i, k) for all N x 128 LHS candidates!
    exp_embs_lhs = torch.empty((N, n_lhs, model.exp_encoder.d_model), device=device, dtype=torch.float32)
    for k, (des, t_end) in enumerate(lhs_pool):
        t_q = np.linspace(0.0, t_end, n_t)
        y0s = np.array([grammar.initial_state(int(m), des) for m in true_mids], dtype=np.float64)
        y_sim, _, _, _ = sim.simulate_batch(true_mids.astype(np.int32), true_thetas, y0s, t_q, compute_flux=False)
        y_obs = y_sim[:, :, obs_ch]
        n_sc = np.where(ch_is_cat, des.cat_mult * grammar.c0_base, des.A0)
        y_n = y_obs + rng_obs.normal(0.0, 1.0, size=y_obs.shape) * (sigma_rel * n_sc[None, None, :])
        df_k, tm_k, ov_k, om_k, tp_k = _build_single_exp_tensors(grammar, des, t_end, y_n, t_q)
        with torch.cuda.amp.autocast(enabled=use_bf16, dtype=torch.bfloat16):
            exp_embs_lhs[:, k] = model.exp_encoder(
                torch.from_numpy(df_k).to(device), torch.from_numpy(tm_k).to(device),
                torch.from_numpy(ov_k).to(device), torch.from_numpy(om_k).to(device),
                torch.from_numpy(tp_k).to(device),
            ).float()

    # Initial D0 observation embedding
    t_eval_1 = np.linspace(0.0, 1.0, n_t)
    d0 = grammar.catalog[0]
    clean_d0 = np.stack(
        [interp_traj_to_times(t_grid, all_trajs[sel_idx[i], 0].astype(np.float64), t_eval_1) for i in range(N)],
        axis=0,
    )
    noise_sc_d0 = np.where(ch_is_cat, d0.cat_mult * grammar.c0_base, d0.A0)
    y_d0 = clean_d0 + rng_obs.normal(0.0, 1.0, size=clean_d0.shape) * (sigma_rel * noise_sc_d0[None, None, :])
    df0, tm0, ov0, om0, tp0 = _build_single_exp_tensors(grammar, d0, 1.0, y_d0, t_eval_1)
    with torch.cuda.amp.autocast(enabled=use_bf16, dtype=torch.bfloat16):
        h_d0 = model.exp_encoder(
            torch.from_numpy(df0).to(device), torch.from_numpy(tm0).to(device),
            torch.from_numpy(ov0).to(device), torch.from_numpy(om0).to(device),
            torch.from_numpy(tp0).to(device),
        ).float().unsqueeze(1)  # (N, 1, d_exp)
        f_d0 = model.set_encoder(h_d0, exp_pad_mask=torch.zeros((N, 1), dtype=torch.bool, device=device))
        logits_d0, G_mat = model.mech_head(f_d0)
    probs_m_d0 = torch.softmax(logits_d0.float(), dim=-1).cpu().numpy()
    probs_c_d0 = aggregate_class_probs(probs_m_d0, m2c, n_cls)
    f_d0 = f_d0.float()
    G_mat = G_mat.float()

    def eval_lhs_state(h_prev: torch.Tensor, new_lhs_indices: Sequence[int]) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        idx_t = torch.tensor(new_lhs_indices, dtype=torch.long, device=device)
        new_emb = exp_embs_lhs[torch.arange(N, device=device), idx_t].unsqueeze(1)
        h_cat = torch.cat([h_prev, new_emb], dim=1)
        ep = torch.zeros((N, h_cat.shape[1]), dtype=torch.bool, device=device)
        with torch.cuda.amp.autocast(enabled=use_bf16, dtype=torch.bfloat16):
            f = model.set_encoder(h_cat, exp_pad_mask=ep)
            logits, _ = model.mech_head(f)
        probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        return probs, f.float(), h_cat

    # Step 1 shared strict-CRN EIG across all 128 LHS candidates
    all_cands_0 = [list(range(n_lhs)) for _ in range(N)]
    rng_s0 = np.random.default_rng(seed + 500)
    eig_q_s0, eig_f_s0 = _score_eig_designs_batched(
        model, grammar, sim, h_d0, f_d0, G_mat, probs_m_d0, probs_c_d0,
        lhs_pool, all_cands_0, m2c, n_cls, n_t, sigma_rel, n_mc=n_mc, device=device, rng=rng_s0,
        crn_seed=seed + 500,
    )

    cf_dq_diff_all, cf_dlogq_diff_all = [], []
    cf_dq_q_all, cf_dq_f_all = [], []
    cf_dlogq_q_all, cf_dlogq_f_all = [], []
    cf_dq_diff_nontriv, cf_dlogq_diff_nontriv = [], []
    cf_dq_q_nontriv, cf_dq_f_nontriv = [], []
    cf_dq_diff_unsolved90, cf_dlogq_diff_unsolved90 = [], []
    cf_dq_q_unsolved90, cf_dq_f_unsolved90 = [], []
    cf_dq_diff_unsolved85, cf_dlogq_diff_unsolved85 = [], []
    disagree_by_step, disagree_count_by_step, disagree_nontriv_by_step = [], [], []
    disagree_unsolved90_by_step, disagree_unsolved85_by_step = [], []
    intra_frac_by_step, intra_frac_nontriv_by_step = [], []
    step1_q_picks_saved, step2_q_picks_saved = [], []
    step1_f_picks_saved, step2_f_picks_saved = [], []
    pm_step1_q_saved, pc_step1_q_saved = None, None
    pm_step1_f_saved, pc_step1_f_saved = None, None

    horizons = [4, budget]

    def run_cont_policy(policy: str, record_cf: bool = False):
        nonlocal pm_step1_q_saved, pc_step1_q_saved, pm_step1_f_saved, pc_step1_f_saved
        h_curr = h_d0.clone()
        f_curr = f_d0.clone()
        pm_curr = probs_m_d0.copy()
        pc_curr = probs_c_d0.copy()
        used = [set() for _ in range(N)]

        pc_traj = np.zeros((N, budget + 1), dtype=np.float64)
        pm_traj = np.zeros((N, budget + 1), dtype=np.float64)
        max_pc_traj = np.zeros((N, budget + 1), dtype=np.float64)
        max_pm_traj = np.zeros((N, budget + 1), dtype=np.float64)
        top1_traj = np.zeros((N, budget + 1), dtype=np.float64)

        pc_traj[:, 0] = pc_curr[np.arange(N), true_c]
        pm_traj[:, 0] = pm_curr[np.arange(N), true_mids]
        max_pc_traj[:, 0] = np.max(pc_curr, axis=1)
        max_pm_traj[:, 0] = np.max(pm_curr, axis=1)
        top1_traj[:, 0] = (np.argmax(pc_curr, axis=1) == true_c).astype(np.float64)

        rng_rand = np.random.default_rng(seed + 777)

        for step in range(budget):
            cands = [[k for k in range(n_lhs) if k not in used[i]] for i in range(N)]
            if policy == "Continuous_Random":
                picks = [int(rng_rand.choice(cands[i])) for i in range(N)]
            else:
                if step == 0:
                    eq_l, ef_l = eig_q_s0, eig_f_s0
                else:
                    step_seed = seed + 500 + step * 1000
                    rng_step = np.random.default_rng(step_seed)
                    eq_l, ef_l = _score_eig_designs_batched(
                        model, grammar, sim, h_curr, f_curr, G_mat, pm_curr, pc_curr,
                        lhs_pool, cands, m2c, n_cls, n_t, sigma_rel, n_mc=n_mc, device=device,
                        rng=rng_step, crn_seed=step_seed,
                    )
                q_picks = [cands[i][int(np.argmax(eq_l[i]))] for i in range(N)]
                f_picks = [cands[i][int(np.argmax(ef_l[i]))] for i in range(N)]

                if step == 0 and policy == "Continuous_Quotient_EIG":
                    step1_q_picks_saved.extend(q_picks)
                    step1_f_picks_saved.extend(f_picks)
                elif step == 1 and policy == "Continuous_Quotient_EIG":
                    step2_q_picks_saved.extend(q_picks)
                elif step == 1 and policy == "Continuous_Fine_EIG":
                    step2_f_picks_saved.extend(f_picks)

                if record_cf:
                    dis_mask = [q_picks[i] != f_picks[i] for i in range(N)]
                    unsolved90 = [max_pc_traj[i, step] < 0.90 for i in range(N)]
                    unsolved85 = [max_pc_traj[i, step] < 0.85 for i in range(N)]
                    disagree_by_step.append(float(np.mean(dis_mask)))
                    disagree_count_by_step.append(int(sum(dis_mask)))
                    nt_dis = [dis_mask[i] for i in range(N) if nontriv_mask[i]]
                    disagree_nontriv_by_step.append(float(np.mean(nt_dis)) if nt_dis else 0.0)
                    u90_dis = [dis_mask[i] for i in range(N) if unsolved90[i]]
                    u85_dis = [dis_mask[i] for i in range(N) if unsolved85[i]]
                    disagree_unsolved90_by_step.append(float(np.mean(u90_dis)) if u90_dis else 0.0)
                    disagree_unsolved85_by_step.append(float(np.mean(u85_dis)) if u85_dis else 0.0)

                    ifracs, ifracs_nt = [], []
                    for i in range(N):
                        idx_f = int(np.argmax(ef_l[i]))
                        em_val = float(ef_l[i][idx_f])
                        ec_val = float(eq_l[i][idx_f])
                        if em_val > 1e-6:
                            frac = max(0.0, (em_val - ec_val) / em_val)
                            ifracs.append(frac)
                            if nontriv_mask[i]:
                                ifracs_nt.append(frac)
                    intra_frac_by_step.append(float(np.mean(ifracs)) if ifracs else 0.0)
                    intra_frac_nontriv_by_step.append(float(np.mean(ifracs_nt)) if ifracs_nt else 0.0)

                    dis_indices = [i for i in range(N) if dis_mask[i]]
                    if dis_indices:
                        pm_q_cf, _, _ = eval_lhs_state(h_curr, q_picks)
                        pm_f_cf, _, _ = eval_lhs_state(h_curr, f_picks)
                        pc_q_cf = aggregate_class_probs(pm_q_cf, m2c, n_cls)
                        pc_f_cf = aggregate_class_probs(pm_f_cf, m2c, n_cls)
                        for i in dis_indices:
                            p0 = float(np.clip(pc_traj[i, step], 1e-12, 1.0))
                            pq = float(np.clip(pc_q_cf[i, true_c[i]], 1e-12, 1.0))
                            pf = float(np.clip(pc_f_cf[i, true_c[i]], 1e-12, 1.0))
                            dq_q, dq_f = pq - p0, pf - p0
                            dlogq_q, dlogq_f = float(np.log(pq) - np.log(p0)), float(np.log(pf) - np.log(p0))
                            dq = dq_q - dq_f
                            dlogq = dlogq_q - dlogq_f
                            cf_dq_q_all.append(dq_q)
                            cf_dq_f_all.append(dq_f)
                            cf_dlogq_q_all.append(dlogq_q)
                            cf_dlogq_f_all.append(dlogq_f)
                            cf_dq_diff_all.append(dq)
                            cf_dlogq_diff_all.append(dlogq)
                            if nontriv_mask[i]:
                                cf_dq_q_nontriv.append(dq_q)
                                cf_dq_f_nontriv.append(dq_f)
                                cf_dq_diff_nontriv.append(dq)
                                cf_dlogq_diff_nontriv.append(dlogq)
                            if unsolved90[i]:
                                cf_dq_q_unsolved90.append(dq_q)
                                cf_dq_f_unsolved90.append(dq_f)
                                cf_dq_diff_unsolved90.append(dq)
                                cf_dlogq_diff_unsolved90.append(dlogq)
                            if unsolved85[i]:
                                cf_dq_diff_unsolved85.append(dq)
                                cf_dlogq_diff_unsolved85.append(dlogq)

                picks = q_picks if policy == "Continuous_Quotient_EIG" else f_picks

            for i, k in enumerate(picks):
                used[i].add(k)
            pm_curr, f_curr, h_curr = eval_lhs_state(h_curr, picks)
            pc_curr = aggregate_class_probs(pm_curr, m2c, n_cls)
            if step == 0 and policy == "Continuous_Quotient_EIG":
                pm_step1_q_saved = pm_curr.copy()
                pc_step1_q_saved = pc_curr.copy()
            elif step == 0 and policy == "Continuous_Fine_EIG":
                pm_step1_f_saved = pm_curr.copy()
                pc_step1_f_saved = pc_curr.copy()

            pc_traj[:, step + 1] = pc_curr[np.arange(N), true_c]
            pm_traj[:, step + 1] = pm_curr[np.arange(N), true_mids]
            max_pc_traj[:, step + 1] = np.max(pc_curr, axis=1)
            max_pm_traj[:, step + 1] = np.max(pm_curr, axis=1)
            top1_traj[:, step + 1] = (np.argmax(pc_curr, axis=1) == true_c).astype(np.float64)

        all_sum, all_raw = _summarize_boed_trajectories(
            pc_traj, pm_traj, max_pc_traj, max_pm_traj, top1_traj, horizons=horizons
        )
        nt_sum, nt_raw = _summarize_boed_trajectories(
            pc_traj[nontriv_mask], pm_traj[nontriv_mask],
            max_pc_traj[nontriv_mask], max_pm_traj[nontriv_mask],
            top1_traj[nontriv_mask], horizons=horizons
        )
        return {"all_cases": all_sum, "nontrivial_cases": nt_sum}, {"all": all_raw, "nontriv": nt_raw}

    q_res, q_raw = run_cont_policy("Continuous_Quotient_EIG", record_cf=True)
    f_res, f_raw = run_cont_policy("Continuous_Fine_EIG", record_cf=False)
    r_res, r_raw = run_cont_policy("Continuous_Random", record_cf=False)

    # Paired bootstrap CIs
    paired_ci = {"all_cases": {}, "nontrivial_cases": {}}
    for sub_k, out_k in [("all", "all_cases"), ("nontriv", "nontrivial_cases")]:
        qr, fr, rr = q_raw[sub_k], f_raw[sub_k], r_raw[sub_k]
        for H in horizons:
            paired_ci[out_k][f"Q_minus_F_auc@{H}"] = paired_bootstrap_ci(qr[f"auc@{H}"] - fr[f"auc@{H}"])
            paired_ci[out_k][f"Q_minus_F_top1@{H}"] = paired_bootstrap_ci(qr[f"top1@{H}"] - fr[f"top1@{H}"])
            paired_ci[out_k][f"Q_minus_F_pc@{H}"] = paired_bootstrap_ci(qr[f"pc@{H}"] - fr[f"pc@{H}"])
            paired_ci[out_k][f"Q_minus_Rand_auc@{H}"] = paired_bootstrap_ci(qr[f"auc@{H}"] - rr[f"auc@{H}"])
            paired_ci[out_k][f"Q_minus_Rand_top1@{H}"] = paired_bootstrap_ci(qr[f"top1@{H}"] - rr[f"top1@{H}"])
            paired_ci[out_k][f"F_minus_Rand_auc@{H}"] = paired_bootstrap_ci(fr[f"auc@{H}"] - rr[f"auc@{H}"])
            paired_ci[out_k][f"F_minus_Rand_top1@{H}"] = paired_bootstrap_ci(fr[f"top1@{H}"] - rr[f"top1@{H}"])
            for t_key in ["tau_85", "tau_90"]:
                paired_ci[out_k][f"delta_cls_stop@{H}_{t_key}_F_minus_Q"] = paired_bootstrap_ci(
                    fr[f"cls_stop@{H}_{t_key}"] - qr[f"cls_stop@{H}_{t_key}"]
                )
                paired_ci[out_k][f"fine_wasted_exps@{H}_{t_key}"] = paired_bootstrap_ci(fr[f"wasted@{H}_{t_key}"])
                paired_ci[out_k][f"end2end_budget_saving_Ffine_minus_Qcls@{H}_{t_key}"] = paired_bootstrap_ci(
                    fr[f"fine_stop@{H}_{t_key}"] - qr[f"cls_stop@{H}_{t_key}"]
                )

    # Step 1 (VTNA) and Step 2 (RPKA) spatial clustering analysis
    def analyze_spatial(picks: List[int], pm_state: np.ndarray, pc_state: np.ndarray) -> Dict:
        vtna_zone = [(cat_arr[k] <= 0.55 or cat_arr[k] >= 1.8) for k in picks]
        preP_zone = [P0_arr[k] >= 0.20 for k in picks]
        redA_zone = [A0_arr[k] <= 0.80 for k in picks]
        rpka_zone = [(P0_arr[k] >= 0.20 or A0_arr[k] <= 0.80) for k in picks]
        highA_zone = [A0_arr[k] >= 1.40 for k in picks]

        vtna_amb, rpka_amb, suba_amb = [], [], []
        for i in range(N):
            if pc_state[i].max() >= 0.90:
                continue
            t2 = np.argsort(pc_state[i])[-2:][::-1]
            ma = max(classes[t2[0]], key=lambda m: pm_state[i, m])
            mb = max(classes[t2[1]], key=lambda m: pm_state[i, m])
            oa, ob = grammar.mech_options[ma], grammar.mech_options[mb]
            k = picks[i]
            if (oa[4] > 0, oa[7] > 0, oa[1] == 2, oa[5] > 0) != (ob[4] > 0, ob[7] > 0, ob[1] == 2, ob[5] > 0):
                vtna_amb.append(k)
            if ((oa[1] > 0) != (ob[1] > 0)) or (((oa[3] > 0) or (oa[6] > 0)) != ((ob[3] > 0) or (ob[6] > 0))):
                rpka_amb.append(k)
            if oa[2] != ob[2]:
                suba_amb.append(k)

        return {
            "all_cases_vtna_extreme_cat_frac": float(np.mean(vtna_zone)),
            "all_cases_preadded_P_frac": float(np.mean(preP_zone)),
            "all_cases_reduced_A0_frac": float(np.mean(redA_zone)),
            "all_cases_rpka_zone_frac": float(np.mean(rpka_zone)),
            "all_cases_high_A0_frac": float(np.mean(highA_zone)),
            "VTNA_ambig_subset": {
                "n": len(vtna_amb),
                "vtna_extreme_cat_frac": float(np.mean([(cat_arr[k] <= 0.55 or cat_arr[k] >= 1.8) for k in vtna_amb])) if vtna_amb else 0.0,
                "mean_abs_log2_cat": float(np.mean([abs(np.log2(cat_arr[k])) for k in vtna_amb])) if vtna_amb else 0.0,
            },
            "RPKA_M2_M4M7_ambig_subset": {
                "n": len(rpka_amb),
                "preadded_P_frac": float(np.mean([P0_arr[k] >= 0.20 for k in rpka_amb])) if rpka_amb else 0.0,
                "reduced_A0_frac": float(np.mean([A0_arr[k] <= 0.80 for k in rpka_amb])) if rpka_amb else 0.0,
                "rpka_zone_frac": float(np.mean([(P0_arr[k] >= 0.20 or A0_arr[k] <= 0.80) for k in rpka_amb])) if rpka_amb else 0.0,
                "mean_P0": float(np.mean([P0_arr[k] for k in rpka_amb])) if rpka_amb else 0.0,
            },
            "SubA_M3_ambig_subset": {
                "n": len(suba_amb),
                "high_A0_frac": float(np.mean([A0_arr[k] >= 1.40 for k in suba_amb])) if suba_amb else 0.0,
                "mean_A0": float(np.mean([A0_arr[k] for k in suba_amb])) if suba_amb else 0.0,
            },
        }

    spatial_clustering = {
        "pool_size": n_lhs,
        "uniform_pool_baseline": {
            "vtna_extreme_cat_frac": float(np.mean((cat_arr <= 0.55) | (cat_arr >= 1.8))),
            "preadded_P_frac": float(np.mean(P0_arr >= 0.20)),
            "reduced_A0_frac": float(np.mean(A0_arr <= 0.80)),
            "rpka_zone_frac": float(np.mean((P0_arr >= 0.20) | (A0_arr <= 0.80))),
            "high_A0_frac": float(np.mean(A0_arr >= 1.40)),
        },
        "Step1_Quotient": analyze_spatial(step1_q_picks_saved, probs_m_d0, probs_c_d0),
        "Step1_Fine": analyze_spatial(step1_f_picks_saved, probs_m_d0, probs_c_d0),
        "Step2_Quotient": analyze_spatial(step2_q_picks_saved, pm_step1_q_saved, pc_step1_q_saved),
        "Step2_Fine": analyze_spatial(step2_f_picks_saved, pm_step1_f_saved, pc_step1_f_saved),
    }

    dt = time.perf_counter() - t0
    print(
        f"[Task3:LHS-128] Done in {dt:.1f}s | "
        f"Q AUC@6={q_res['all_cases']['horizons']['T=6']['auc_true_class_prob']:.3f} "
        f"top1@6={q_res['all_cases']['horizons']['T=6']['top1_at_H']:.3f} | "
        f"F AUC@6={f_res['all_cases']['horizons']['T=6']['auc_true_class_prob']:.3f} "
        f"top1@6={f_res['all_cases']['horizons']['T=6']['top1_at_H']:.3f} | "
        f"Rand AUC@6={r_res['all_cases']['horizons']['T=6']['auc_true_class_prob']:.3f} "
        f"top1@6={r_res['all_cases']['horizons']['T=6']['top1_at_H']:.3f}",
        flush=True,
    )
    return {
        "n_cases": N,
        "n_nontrivial_cases": int(np.sum(nontriv_mask)),
        "n_lhs": n_lhs,
        "budget": budget,
        "n_mc": n_mc,
        "elapsed_s": dt,
        "Continuous_Quotient_EIG": q_res,
        "Continuous_Fine_EIG": f_res,
        "Continuous_Random": r_res,
        "paired_bootstrap_ci": paired_ci,
        "same_state_counterfactual": {
            "disagree_rate_by_step": disagree_by_step,
            "disagree_count_by_step": disagree_count_by_step,
            "disagree_rate_mean_all_steps": float(np.mean(disagree_by_step)),
            "disagree_nontriv_by_step": disagree_nontriv_by_step,
            "disagree_nontriv_mean": float(np.mean(disagree_nontriv_by_step)),
            "disagree_rate_unsolved90_by_step": disagree_unsolved90_by_step,
            "disagree_rate_mean_unsolved90": float(np.mean(disagree_unsolved90_by_step)),
            "disagree_rate_unsolved85_by_step": disagree_unsolved85_by_step,
            "disagree_rate_mean_unsolved85": float(np.mean(disagree_unsolved85_by_step)),
            "intra_class_eig_frac_by_step": intra_frac_by_step,
            "intra_class_eig_frac_mean": float(np.mean(intra_frac_by_step)),
            "intra_class_eig_frac_nontriv_by_step": intra_frac_nontriv_by_step,
            "intra_class_eig_frac_nontriv_mean": float(np.mean(intra_frac_nontriv_by_step)),
            "mean_step_gain_q_on_disagree": float(np.mean(cf_dq_q_all)) if cf_dq_q_all else 0.0,
            "mean_step_gain_f_on_disagree": float(np.mean(cf_dq_f_all)) if cf_dq_f_all else 0.0,
            "mean_step_gain_q_nontriv": float(np.mean(cf_dq_q_nontriv)) if cf_dq_q_nontriv else 0.0,
            "mean_step_gain_f_nontriv": float(np.mean(cf_dq_f_nontriv)) if cf_dq_f_nontriv else 0.0,
            "mean_step_gain_q_unsolved90": float(np.mean(cf_dq_q_unsolved90)) if cf_dq_q_unsolved90 else 0.0,
            "mean_step_gain_f_unsolved90": float(np.mean(cf_dq_f_unsolved90)) if cf_dq_f_unsolved90 else 0.0,
            "mean_step_log_gain_q_on_disagree": float(np.mean(cf_dlogq_q_all)) if cf_dlogq_q_all else 0.0,
            "mean_step_log_gain_f_on_disagree": float(np.mean(cf_dlogq_f_all)) if cf_dlogq_f_all else 0.0,
            "all_disagreements_delta_q": paired_bootstrap_ci(np.array(cf_dq_diff_all)),
            "all_disagreements_delta_logq": paired_bootstrap_ci(np.array(cf_dlogq_diff_all)),
            "nontriv_disagreements_delta_q": paired_bootstrap_ci(np.array(cf_dq_diff_nontriv)),
            "nontriv_disagreements_delta_logq": paired_bootstrap_ci(np.array(cf_dlogq_diff_nontriv)),
            "unsolved90_disagreements_delta_q": paired_bootstrap_ci(np.array(cf_dq_diff_unsolved90)),
            "unsolved90_disagreements_delta_logq": paired_bootstrap_ci(np.array(cf_dlogq_diff_unsolved90)),
            "unsolved85_disagreements_delta_q": paired_bootstrap_ci(np.array(cf_dq_diff_unsolved85)),
            "unsolved85_disagreements_delta_logq": paired_bootstrap_ci(np.array(cf_dlogq_diff_unsolved85)),
        },
        "spatial_clustering": spatial_clustering,
    }


@torch.no_grad()
def eval_task4_cost_and_upgrade(
    model: QuoMechNet,
    grammar: ReactionGrammar,
    sim: ODESimulator,
    test_dir: Path,
    equiv_tiers: Dict[str, Dict],
    device: torch.device,
    n_per_mech: int = 2,
    n_mc: int = 256,
    max_cost: int = 6,
    sigma_rel: float = 0.01,
    seed: int = 202644,
) -> Dict:
    """Task 4: (a) Cost-normalized Xi_full BOED (cost<=6, D9 cost=3); (b) Xi_basic -> Xi_full upgrade recommendation."""
    t0 = time.perf_counter()
    model.eval()
    rng_obs = np.random.default_rng(seed)
    obs_ch = get_observable_channel_indices(grammar)
    ch_is_cat = np.array([grammar.species[s].is_catalyst for s in obs_ch], dtype=bool)
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    t_grid = np.load(test_dir / "t_grid.npy")
    all_mids = np.load(test_dir / "mech_ids.npy")
    all_trajs = np.load(test_dir / "trajs.npy")

    sel_list = []
    for m in range(grammar.n_mechs):
        idxs = np.where(all_mids == m)[0]
        sel_list.extend(idxs[5 : 5 + n_per_mech].tolist())
    sel_idx = np.array(sel_list, dtype=np.int64)
    N = len(sel_idx)
    true_mids = all_mids[sel_idx].astype(np.int64)
    trajs_sel = all_trajs[sel_idx]

    n_t = 20
    t_eval = np.linspace(0.0, 1.0, n_t)
    catalog_pool = [(d, 1.0) for d in grammar.catalog]
    costs = np.array([int(round(d.cost)) for d in grammar.catalog], dtype=np.int64)  # D9=3, others=1

    exp_embs_all = torch.empty((N, 10, model.exp_encoder.d_model), device=device, dtype=torch.float32)
    for d_idx in range(10):
        des = grammar.catalog[d_idx]
        clean = np.stack(
            [interp_traj_to_times(t_grid, trajs_sel[i, d_idx].astype(np.float64), t_eval) for i in range(N)],
            axis=0,
        )
        cat0 = des.cat_mult * grammar.c0_base
        noise_scale = np.where(ch_is_cat, cat0, des.A0)
        y_n = clean + rng_obs.normal(0.0, 1.0, size=clean.shape) * (sigma_rel * noise_scale[None, None, :])
        df_np, tm_np, ov_np, om_np, tp_np = _build_single_exp_tensors(grammar, des, 1.0, y_n, t_eval)
        with torch.cuda.amp.autocast(enabled=use_bf16, dtype=torch.bfloat16):
            exp_embs_all[:, d_idx] = model.exp_encoder(
                torch.from_numpy(df_np).to(device), torch.from_numpy(tm_np).to(device),
                torch.from_numpy(ov_np).to(device), torch.from_numpy(om_np).to(device),
                torch.from_numpy(tp_np).to(device),
            ).float()

    _, G_mat = model.mech_head(torch.zeros((1, 256), device=device))
    G_mat = G_mat.float()

    def eval_var_hist(hist_list: List[List[int]]) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        E_max = max(len(h) for h in hist_list)
        d_exp = model.exp_encoder.d_model
        h_seq = torch.zeros((N, E_max, d_exp), device=device, dtype=torch.float32)
        ep = torch.ones((N, E_max), device=device, dtype=torch.bool)
        for i, h in enumerate(hist_list):
            for e_i, d_i in enumerate(h):
                h_seq[i, e_i] = exp_embs_all[i, d_i]
                ep[i, e_i] = False
        with torch.cuda.amp.autocast(enabled=use_bf16, dtype=torch.bfloat16):
            f = model.set_encoder(h_seq, exp_pad_mask=ep)
            logits, _ = model.mech_head(f)
        probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        return probs, f.float(), h_seq

    # ------------------------------------------------------------------
    # Task 4a: Cost-normalized design under Xi_full (total cost budget <= 6)
    # ------------------------------------------------------------------
    eq_full = equiv_tiers["full"]["primary"]
    m2c_full = np.array(eq_full["mech_to_class"], dtype=np.int64)
    classes_full = eq_full["classes"]
    n_cls_full = eq_full["n_classes"]
    true_c_full = m2c_full[true_mids]
    full_designs = list(grammar.tiers["full"])
    is_nontriv_full = np.array([len(classes_full[c]) >= 2 for c in true_c_full], dtype=bool)

    eq_basic = equiv_tiers["basic"]["primary"]
    m2c_basic = np.array(eq_basic["mech_to_class"], dtype=np.int64)
    classes_basic = eq_basic["classes"]
    n_cls_basic = len(classes_basic)
    true_c_basic = m2c_basic[true_mids]

    # Exact identification of M4/M7 splittable classes in Xi_basic -> Xi_full:
    # members in the same Xi_basic class must actually DIFFER in (M4, M7) AND split in Xi_full!
    mopts = grammar.mech_options
    m4m7_split_class_ids = set()
    for cid, members in enumerate(classes_basic):
        if len(members) >= 2:
            full_cids = {m2c_full[m] for m in members}
            m47_variants = {(int(mopts[m, 3]), int(mopts[m, 6])) for m in members}
            if len(full_cids) > 1 and len(m47_variants) > 1:
                m4m7_split_class_ids.add(cid)
    is_m4m7_nontriv_basic = np.array([c in m4m7_split_class_ids for c in true_c_basic], dtype=bool)

    # Pre-compute shared Step-0 strict-CRN EIG across all 3 cost policies
    pm_d0_4a, f_d0_4a, h_d0_4a = eval_var_hist([[0] for _ in range(N)])
    pc_d0_4a = aggregate_class_probs(pm_d0_4a, m2c_full, n_cls_full)
    cands_0_4a = [[d for d in full_designs if d != 0 and costs[d] <= max_cost] for _ in range(N)]
    rng_4a_0 = np.random.default_rng(seed + 900)
    eq_0_4a, ef_0_4a = _score_eig_designs_batched(
        model, grammar, sim, h_d0_4a, f_d0_4a, G_mat, pm_d0_4a, pc_d0_4a,
        catalog_pool, cands_0_4a, m2c_full, n_cls_full, n_t, sigma_rel, n_mc=n_mc, device=device,
        rng=rng_4a_0, crn_seed=seed + 900, case_ids=list(range(N)),
    )

    def run_cost_policy(mode: str) -> Tuple[Dict, Dict[str, np.ndarray]]:
        hist = [[0] for _ in range(N)]
        spent = np.zeros(N, dtype=np.int64)
        pc_by_cost = np.zeros((N, max_cost + 1), dtype=np.float64)
        top1_by_cost = np.zeros((N, max_cost + 1), dtype=np.float64)
        d9_picked_ever = np.zeros(N, dtype=bool)
        d9_picked_step1 = np.zeros(N, dtype=bool)
        n_exps_run = np.zeros(N, dtype=np.int64)

        pm_curr, f_curr, h_curr = pm_d0_4a.copy(), f_d0_4a.clone(), h_d0_4a.clone()
        pc_curr = pc_d0_4a.copy()
        pc_by_cost[:, 0] = pc_curr[np.arange(N), true_c_full]
        top1_by_cost[:, 0] = (np.argmax(pc_curr, axis=1) == true_c_full).astype(np.float64)

        for step in range(max_cost):
            active = [i for i in range(N) if spent[i] < max_cost]
            if not active:
                break
            cands_act = [
                [d for d in full_designs if (d not in hist[i]) and (costs[d] <= max_cost - spent[i])]
                for i in active
            ]
            valid_pairs = [(i, c) for i, c in zip(active, cands_act) if len(c) > 0]
            if not valid_pairs:
                break
            act_idx = [p[0] for p in valid_pairs]
            cands_use = [p[1] for p in valid_pairs]

            if step == 0:
                eq_l, ef_l = eq_0_4a, ef_0_4a
            else:
                h_act = torch.stack([exp_embs_all[i, torch.tensor(hist[i], device=device)] for i in act_idx], dim=0)
                f_act = f_curr[act_idx]
                pm_act = pm_curr[act_idx]
                pc_act = pc_curr[act_idx]
                step_seed = seed + 900 + step * 1000
                rng_step = np.random.default_rng(step_seed)
                eq_l, ef_l = _score_eig_designs_batched(
                    model, grammar, sim, h_act, f_act, G_mat, pm_act, pc_act,
                    catalog_pool, cands_use, m2c_full, n_cls_full, n_t, sigma_rel, n_mc=n_mc, device=device,
                    rng=rng_step, crn_seed=step_seed, case_ids=act_idx,
                )

            for loc_i, i in enumerate(act_idx):
                c_list = cands_use[loc_i]
                c_costs = np.array([float(costs[d]) for d in c_list])
                if mode == "Quotient_EIG":
                    scores = eq_l[loc_i]
                elif mode == "Quotient_EIG_per_cost":
                    scores = np.maximum(eq_l[loc_i], 0.0) / c_costs
                elif mode == "Fine_EIG_per_cost":
                    scores = np.maximum(ef_l[loc_i], 0.0) / c_costs
                else:
                    raise ValueError(mode)
                pick = c_list[int(np.argmax(scores))]
                old_c = int(spent[i])
                new_c = old_c + int(costs[pick])
                for cc in range(old_c + 1, new_c):
                    pc_by_cost[i, cc] = pc_by_cost[i, old_c]
                    top1_by_cost[i, cc] = top1_by_cost[i, old_c]
                hist[i].append(pick)
                spent[i] = new_c
                n_exps_run[i] += 1
                if pick == 9:
                    d9_picked_ever[i] = True
                    if step == 0:
                        d9_picked_step1[i] = True

            pm_curr, f_curr, h_curr = eval_var_hist(hist)
            pc_curr = aggregate_class_probs(pm_curr, m2c_full, n_cls_full)
            for loc_i, i in enumerate(act_idx):
                cc = int(spent[i])
                pc_by_cost[i, cc] = pc_curr[i, true_c_full[i]]
                top1_by_cost[i, cc] = float(np.argmax(pc_curr[i]) == true_c_full[i])
                for rem_c in range(cc + 1, max_cost + 1):
                    pc_by_cost[i, rem_c] = pc_by_cost[i, cc]
                    top1_by_cost[i, rem_c] = top1_by_cost[i, cc]

        cost_auc = np.mean(pc_by_cost, axis=1)
        return {
            "cost_auc_mean": float(np.mean(cost_auc)),
            "cost_auc_nontriv_full": float(np.mean(cost_auc[is_nontriv_full])),
            "pc_by_cost": [float(x) for x in np.mean(pc_by_cost, axis=0)],
            "top1_by_cost": [float(x) for x in np.mean(top1_by_cost, axis=0)],
            "final_pc_at_cost6": float(np.mean(pc_by_cost[:, max_cost])),
            "final_top1_at_cost6": float(np.mean(top1_by_cost[:, max_cost])),
            "final_top1_nontriv_full": float(np.mean(top1_by_cost[is_nontriv_full, max_cost])),
            "mean_exps_executed": float(np.mean(n_exps_run)),
            "d9_ever_rate_all": float(np.mean(d9_picked_ever)),
            "d9_step1_rate_all": float(np.mean(d9_picked_step1)),
            "d9_ever_rate_m4m7_splittable": float(np.mean(d9_picked_ever[is_m4m7_nontriv_basic])),
            "d9_ever_rate_non_m4m7_splittable": float(np.mean(d9_picked_ever[~is_m4m7_nontriv_basic])),
            "d9_ever_rate_xi_full_nontriv": float(np.mean(d9_picked_ever[is_nontriv_full])),
        }, {
            "cost_auc": cost_auc,
            "final_top1": top1_by_cost[:, max_cost],
            "final_pc": pc_by_cost[:, max_cost],
            "d9_ever": d9_picked_ever.astype(np.float64),
        }

    c_q_unw, raw_q_unw = run_cost_policy("Quotient_EIG")
    c_q_norm, raw_q_norm = run_cost_policy("Quotient_EIG_per_cost")
    c_f_norm, raw_f_norm = run_cost_policy("Fine_EIG_per_cost")

    cost_ci = {
        "Qnorm_minus_Qunw_cost_auc": paired_bootstrap_ci(raw_q_norm["cost_auc"] - raw_q_unw["cost_auc"]),
        "Qnorm_minus_Qunw_top1@cost6": paired_bootstrap_ci(raw_q_norm["final_top1"] - raw_q_unw["final_top1"]),
        "Qnorm_minus_Fnorm_cost_auc": paired_bootstrap_ci(raw_q_norm["cost_auc"] - raw_f_norm["cost_auc"]),
        "Qnorm_minus_Fnorm_top1@cost6": paired_bootstrap_ci(raw_q_norm["final_top1"] - raw_f_norm["final_top1"]),
        "Qnorm_minus_Fnorm_cost_auc_nontriv_full": paired_bootstrap_ci(
            raw_q_norm["cost_auc"][is_nontriv_full] - raw_f_norm["cost_auc"][is_nontriv_full]
        ),
        "Fnorm_minus_Qnorm_d9_rate_all": paired_bootstrap_ci(raw_f_norm["d9_ever"] - raw_q_norm["d9_ever"]),
        "Fnorm_minus_Qnorm_d9_rate_non_m4m7": paired_bootstrap_ci(
            raw_f_norm["d9_ever"][~is_m4m7_nontriv_basic] - raw_q_norm["d9_ever"][~is_m4m7_nontriv_basic]
        ),
        "Fnorm_minus_Qnorm_d9_rate_xi_full_nontriv": paired_bootstrap_ci(
            raw_f_norm["d9_ever"][is_nontriv_full] - raw_q_norm["d9_ever"][is_nontriv_full]
        ),
    }

    # ------------------------------------------------------------------
    # Task 4b: Instrument capability upgrade recommendation (Xi_basic -> {D7, D8, D9})
    # ------------------------------------------------------------------
    basic_designs = list(grammar.tiers["basic"])
    upgrade_cands = [d for d in full_designs if d not in basic_designs]  # [7, 8, 9] = D7, D8, D9

    # Evaluate posterior after observing all of Xi_basic (D0..D6)
    hist_basic = [basic_designs for _ in range(N)]
    pm_basic, f_basic, h_basic = eval_var_hist(hist_basic)
    pc_basic = aggregate_class_probs(pm_basic, m2c_basic, n_cls_basic)
    pc_full_before = aggregate_class_probs(pm_basic, m2c_full, n_cls_full)

    is_nontriv_basic = np.array([len(classes_basic[c]) >= 2 for c in true_c_basic], dtype=bool)
    is_converged_basic90 = (np.max(pc_basic, axis=1) >= 0.90) & is_nontriv_basic
    is_converged_basic80 = (np.max(pc_basic, axis=1) >= 0.80) & is_nontriv_basic

    # Compute out-of-catalog Quotient EIG targeting C_{Xi_full} across {D7, D8, D9}
    upg_cand_lists = [upgrade_cands for _ in range(N)]
    rng_upg = np.random.default_rng(seed + 1234)
    eig_upg_q, _ = _score_eig_designs_batched(
        model, grammar, sim, h_basic, f_basic, G_mat, pm_basic, pc_full_before,
        catalog_pool, upg_cand_lists, m2c_full, n_cls_full, n_t, sigma_rel, n_mc=n_mc, device=device,
        rng=rng_upg, crn_seed=seed + 1234, case_ids=list(range(N)),
    )
    upg_picks = [upgrade_cands[int(np.argmax(eig_upg_q[i]))] for i in range(N)]

    # Execute recommended upgrade experiment and measure posterior jump in Xi_full
    hist_after_upg = [basic_designs + [upg_picks[i]] for i in range(N)]
    pm_after_upg, _, _ = eval_var_hist(hist_after_upg)
    pc_full_after = aggregate_class_probs(pm_after_upg, m2c_full, n_cls_full)

    # Also evaluate executing ALL 3 upgrade experiments (D7+D8+D9 = Xi_full)
    hist_all_full = [full_designs for _ in range(N)]
    pm_all_full, _, _ = eval_var_hist(hist_all_full)
    pc_full_all10 = aggregate_class_probs(pm_all_full, m2c_full, n_cls_full)

    def summarize_upgrade_subset(mask: np.ndarray) -> Dict:
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            return {"n": 0}
        picks_sub = [upg_picks[i] for i in idxs]
        p_before = pc_full_before[idxs, true_c_full[idxs]]
        p_after1 = pc_full_after[idxs, true_c_full[idxs]]
        p_after3 = pc_full_all10[idxs, true_c_full[idxs]]
        top1_before = (np.argmax(pc_full_before[idxs], axis=1) == true_c_full[idxs]).astype(np.float64)
        top1_after1 = (np.argmax(pc_full_after[idxs], axis=1) == true_c_full[idxs]).astype(np.float64)
        top1_after3 = (np.argmax(pc_full_all10[idxs], axis=1) == true_c_full[idxs]).astype(np.float64)
        return {
            "n": int(len(idxs)),
            "rec_dist": {grammar.catalog[d].name: int(picks_sub.count(d)) for d in upgrade_cands},
            "D9_top1_rec_rate": float(picks_sub.count(9) / len(idxs)),
            "D7_or_D9_rec_rate": float((picks_sub.count(7) + picks_sub.count(9)) / len(idxs)),
            "xi_full_true_class_prob_before": float(np.mean(p_before)),
            "xi_full_true_class_prob_after_1_rec_exp": float(np.mean(p_after1)),
            "xi_full_true_class_prob_after_all_3_exps": float(np.mean(p_after3)),
            "mean_posterior_gain_1_exp": float(np.mean(p_after1 - p_before)),
            "xi_full_top1_before": float(np.mean(top1_before)),
            "xi_full_top1_after_1_rec_exp": float(np.mean(top1_after1)),
            "xi_full_top1_after_all_3_exps": float(np.mean(top1_after3)),
            "top1_gain_1_exp": float(np.mean(top1_after1 - top1_before)),
        }

    upgrade_res = {
        "converged_90_nontriv_basic": summarize_upgrade_subset(is_converged_basic90),
        "converged_80_nontriv_basic": summarize_upgrade_subset(is_converged_basic80),
        "all_nontriv_basic": summarize_upgrade_subset(is_nontriv_basic),
        "m4m7_family_converged_90": summarize_upgrade_subset(is_converged_basic90 & is_m4m7_nontriv_basic),
        "m4m7_family_converged_80": summarize_upgrade_subset(is_converged_basic80 & is_m4m7_nontriv_basic),
        "m4m7_family_all_nontriv": summarize_upgrade_subset(is_m4m7_nontriv_basic),
    }

    print(
        f"[Task4] Done in {time.perf_counter()-t0:.1f}s | "
        f"Cost6 Top1: Q_norm={c_q_norm['final_top1_at_cost6']:.3f} (D9={c_q_norm['d9_ever_rate_all']:.2f}) vs "
        f"Q_unw={c_q_unw['final_top1_at_cost6']:.3f} (D9={c_q_unw['d9_ever_rate_all']:.2f}) vs "
        f"F_norm={c_f_norm['final_top1_at_cost6']:.3f} (D9={c_f_norm['d9_ever_rate_all']:.2f}) | "
        f"Upgrade M4/M7 D9 rec={upgrade_res['m4m7_family_all_nontriv'].get('D9_top1_rec_rate', 0):.3f}",
        flush=True,
    )

    return {
        "task4a_cost_normalized": {
            "max_cost": max_cost,
            "n_cases": N,
            "Quotient_EIG_unweighted": c_q_unw,
            "Quotient_EIG_per_cost": c_q_norm,
            "Fine_EIG_per_cost": c_f_norm,
            "paired_bootstrap_ci": cost_ci,
        },
        "task4b_instrument_upgrade": upgrade_res,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--model-dir", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--n-test-e1", type=int, default=1510)
    ap.add_argument("--n-per-mech-t2", type=int, default=2)
    ap.add_argument("--n-mc-t2", type=int, default=512)
    ap.add_argument("--n-cases-t3", type=int, default=151)
    ap.add_argument("--n-lhs-t3", type=int, default=128)
    ap.add_argument("--n-mc-t3", type=int, default=256)
    ap.add_argument("--n-mc-t4", type=int, default=512)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data" / "pool_v1"
    g2_dir = root / "results" / "gate2"
    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    grammar = build_default_grammar()
    sim = ODESimulator(grammar, rtol=1e-5, atol=1e-9, n_threads=64)

    equiv_tiers = {t: json.load(open(g2_dir / f"equiv_{t}.json")) for t in ["D0", "no_P", "basic", "full"]}

    print("=== Phase 1.5 Task 1: 60k Convergence & Overfitting Analysis ===", flush=True)
    t1_res = eval_task1_convergence(
        grammar, sim, data_dir / "test", equiv_tiers, model_dir,
        root / "results" / "gate3b" / "e1_quomech.json", device, n_test=args.n_test_e1,
    )
    with open(out_dir / "task1_convergence.json", "w") as f:
        json.dump(t1_res, f, indent=2)

    model_best, _ = load_ckpt_model(grammar, model_dir / "best.pt", device)

    print("\n=== Phase 1.5 Task 2: Un-truncated Discrete 3-Tier BOED (302 cases, N_MC=512, Strict CRN) ===", flush=True)
    t2_res = eval_task2_discrete_boed(
        model_best, grammar, sim, data_dir / "test", equiv_tiers, device,
        n_per_mech=args.n_per_mech_t2, n_mc=args.n_mc_t2,
    )
    with open(out_dir / "task2_discrete_boed.json", "w") as f:
        json.dump(t2_res, f, indent=2)

    print("\n=== Phase 1.5 Task 3: Continuous X_basic LHS-128 Head-to-Head (T=6, N_MC=256, Strict CRN) ===", flush=True)
    t3_res = eval_task3_continuous_lhs128(
        model_best, grammar, sim, data_dir / "test", equiv_tiers["basic"], device,
        n_cases=args.n_cases_t3, n_lhs=args.n_lhs_t3, budget=6, n_mc=args.n_mc_t3,
    )
    with open(out_dir / "task3_continuous_lhs128.json", "w") as f:
        json.dump(t3_res, f, indent=2)

    print("\n=== Phase 1.5 Task 4: Cost-Normalized Design & Instrument Upgrade Recommendation ===", flush=True)
    t4_res = eval_task4_cost_and_upgrade(
        model_best, grammar, sim, data_dir / "test", equiv_tiers, device,
        n_per_mech=args.n_per_mech_t2, n_mc=args.n_mc_t4, max_cost=6,
    )
    with open(out_dir / "task4_cost_and_upgrade.json", "w") as f:
        json.dump(t4_res, f, indent=2)

    print("\n=== ALL PHASE 1.5 EVALUATIONS COMPLETE! ===", flush=True)


if __name__ == "__main__":
    main()
