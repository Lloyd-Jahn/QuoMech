#!/usr/bin/env python3
"""Phase 2 Stage 3 (E3): Complete 527-Mechanism Quotient BOED + Continuous LHS-128 + Cost-Normalized
+ 1-Shot Instrument Upgrade + Zero-Retrain Coarse Task Quotient C_macro Benchmark.

Supports both single-seed arXiv v1 fast-track (--seeds 42) and full 3-seed backfill (--seeds 42,43,44).
All comparisons enforce bit-exact Strict Common Random Numbers (CRN, verified programmatically)
and compute 10,000-resample paired Bootstrap 95% confidence intervals.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
from scipy.stats import qmc
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quomech.boed import _build_single_exp_tensors, _score_eig_designs_batched
from quomech.data_gen import get_observable_channel_indices
from quomech.evaluate import aggregate_class_probs, interp_traj_to_times
from quomech.grammar import DesignSpec, ReactionGrammar, build_multicore_grammar
from quomech.models import QuoMechNet
from quomech.simulator import ODESimulator
from quomech.train import build_model


def paired_bootstrap_ci(diff_arr: np.ndarray, n_boot: int = 10000, seed: int = 42) -> Dict[str, float]:
    x = np.asarray(diff_arr, dtype=np.float64)
    n = len(x)
    if n == 0:
        return {"n": 0, "mean": 0.0, "median": 0.0, "ci95_lo": 0.0, "ci95_hi": 0.0, "win_rate": 0.0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    bm = np.mean(x[idx], axis=1)
    return {
        "n": int(n),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "ci95_lo": float(np.percentile(bm, 2.5)),
        "ci95_hi": float(np.percentile(bm, 97.5)),
        "win_rate": float(np.mean(x > 0.0)),
    }


def load_all_equiv(equiv_dir: Path) -> Dict[str, Dict]:
    mapping = {
        "D0": "equiv_D0.json",
        "D0_D5": "equiv_D0_D5.json",
        "no_P": "equiv_Xi_noP.json",
        "basic": "equiv_Xi_basic.json",
        "full": "equiv_Xi_full.json",
        "X_basic_ref": "equiv_X_basic_ref.json",
    }
    return {k: json.loads((equiv_dir / fn).read_text()) for k, fn in mapping.items() if (equiv_dir / fn).exists()}


def build_c_macro_partition(grammar: ReactionGrammar) -> Dict:
    """Coarse chemical process-diagnostic task quotient C_macro (zero retraining):
    6-bit presence/absence of macro side-reaction phenomena:
      (M1>0, M2>0, M3>0, (M4>0 or M7>0), M5>0, (M6>0 or M8>0))
    Merges C1-C4 core binding order, M2 subtypes (free1/free2/complex), M4/M7, and M6/M8.
    """
    mopts = grammar.mech_options
    sig_to_cid: Dict[Tuple[int, ...], int] = {}
    classes: List[List[int]] = []
    m2c = np.zeros(grammar.n_mechs, dtype=np.int64)
    labels: List[str] = []
    bit_names = ["M1(act)", "M2(deact)", "M3(subInh)", "M4/7(prodEff)", "M5(dimer)", "M6/8(bg/bimet)"]
    for m in range(grammar.n_mechs):
        o = mopts[m]
        sig = (
            int(o[0] > 0),
            int(o[1] > 0),
            int(o[2] > 0),
            int((o[3] > 0) or (o[6] > 0)),
            int(o[4] > 0),
            int((o[5] > 0) or (o[7] > 0)),
        )
        if sig not in sig_to_cid:
            sig_to_cid[sig] = len(classes)
            classes.append([])
            active_lbl = "+".join(b for b, v in zip(bit_names, sig) if v) or "CleanCore"
            labels.append(active_lbl)
        cid = sig_to_cid[sig]
        m2c[m] = cid
        classes[cid].append(m)
    return {
        "n_classes": len(classes),
        "mech_to_class": m2c,
        "classes": classes,
        "class_labels": labels,
        "mean_class_size": float(grammar.n_mechs / len(classes)),
        "max_class_size": int(max(len(c) for c in classes)),
    }


def _summarize_trajectories(
    pc_traj: np.ndarray,
    pm_traj: np.ndarray,
    max_pc_traj: np.ndarray,
    max_pm_traj: np.ndarray,
    top1_traj: np.ndarray,
    horizons: Sequence[int],
    thresholds: Sequence[float] = (0.85, 0.90),
) -> Tuple[Dict, Dict[str, np.ndarray]]:
    N, T_plus_1 = pc_traj.shape
    T_max = T_plus_1 - 1
    out = {
        "n": int(N),
        "mean_pc_by_step": [float(x) for x in np.mean(pc_traj, axis=0)],
        "top1_by_step": [float(x) for x in np.mean(top1_traj, axis=0)],
        "horizons": {},
    }
    raw: Dict[str, np.ndarray] = {}
    for H in horizons:
        if H > T_max:
            continue
        sub_pc = pc_traj[:, : H + 1]
        sub_max_pc = max_pc_traj[:, : H + 1]
        sub_max_pm = max_pm_traj[:, : H + 1]
        sub_top1 = top1_traj[:, : H + 1]

        auc_arr = np.mean(sub_pc, axis=1)
        top1_H = sub_top1[:, H]
        pc_H = sub_pc[:, H]
        raw[f"auc@{H}"] = auc_arr
        raw[f"top1@{H}"] = top1_H
        raw[f"pc@{H}"] = pc_H

        h_dict = {
            "auc_true_class_prob": float(np.mean(auc_arr)),
            "top1_at_H": float(np.mean(top1_H)),
            "true_class_prob_at_H": float(np.mean(pc_H)),
            "stopping": {},
        }
        for tau in thresholds:
            t_key = f"tau_{int(round(tau * 100))}"
            t_stop_cls = np.full(N, H, dtype=np.int64)
            t_stop_fine = np.full(N, H, dtype=np.int64)
            reached_cls = np.zeros(N, dtype=bool)
            reached_fine = np.zeros(N, dtype=bool)
            t_true_tau = np.full(N, H + 1, dtype=np.int64)
            acc_cls = np.zeros(N, dtype=np.float64)
            acc_fine = np.zeros(N, dtype=np.float64)
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
                acc_cls[i] = sub_top1[i, t_stop_cls[i]]
                acc_fine[i] = sub_top1[i, t_stop_fine[i]]

            wasted = np.maximum(0, t_stop_fine - t_stop_cls).astype(np.float64)
            early_mask = reached_cls & (t_stop_cls < H)
            raw[f"cls_stop@{H}_{t_key}"] = t_stop_cls.astype(np.float64)
            raw[f"fine_stop@{H}_{t_key}"] = t_stop_fine.astype(np.float64)
            raw[f"wasted@{H}_{t_key}"] = wasted
            h_dict["stopping"][t_key] = {
                "reach_true_tau_rate": float(np.mean(t_true_tau <= H)),
                "mean_exps_to_true_tau_capped": float(np.mean(np.minimum(t_true_tau, H))),
                "cls_stop_reach_rate": float(np.mean(reached_cls)),
                "cls_early_stop_rate": float(np.mean(early_mask)),
                "fine_stop_reach_rate": float(np.mean(reached_fine)),
                "unified_class_stop_mean_exps": float(np.mean(t_stop_cls)),
                "unified_class_stop_top1": float(np.mean(acc_cls)),
                "fine_stop_mean_exps": float(np.mean(t_stop_fine)),
                "fine_stop_top1": float(np.mean(acc_fine)),
                "wasted_exps_all": float(np.mean(wasted)),
                "wasted_exps_given_cls_early_stopped": float(np.mean(wasted[early_mask])) if np.any(early_mask) else 0.0,
            }
        out["horizons"][f"T={H}"] = h_dict
    return out, raw


def _package_cf_stats(
    disagree_by_step, disagree_count_by_step, disagree_nontriv_by_step,
    disagree_unsolved90_by_step, intra_frac_by_step, intra_frac_nontriv_by_step,
    cf_dq_q_all, cf_dq_f_all, cf_dlogq_q_all, cf_dlogq_f_all,
    cf_dq_diff_all, cf_dlogq_diff_all, cf_dq_diff_nontriv, cf_dlogq_diff_nontriv,
    cf_dq_diff_unsolved90, cf_dlogq_diff_unsolved90,
) -> Dict:
    return {
        "disagree_rate_by_step": disagree_by_step,
        "disagree_count_by_step": disagree_count_by_step,
        "disagree_rate_mean_all_steps": float(np.mean(disagree_by_step)) if disagree_by_step else 0.0,
        "disagree_nontriv_by_step": disagree_nontriv_by_step,
        "disagree_nontriv_mean": float(np.mean(disagree_nontriv_by_step)) if disagree_nontriv_by_step else 0.0,
        "disagree_rate_unsolved90_by_step": disagree_unsolved90_by_step,
        "disagree_rate_mean_unsolved90": float(np.mean(disagree_unsolved90_by_step)) if disagree_unsolved90_by_step else 0.0,
        "intra_class_eig_frac_by_step": intra_frac_by_step,
        "intra_class_eig_frac_mean": float(np.mean(intra_frac_by_step)) if intra_frac_by_step else 0.0,
        "intra_class_eig_frac_nontriv_by_step": intra_frac_nontriv_by_step,
        "intra_class_eig_frac_nontriv_mean": float(np.mean(intra_frac_nontriv_by_step)) if intra_frac_nontriv_by_step else 0.0,
        "mean_step_gain_q_on_disagree": float(np.mean(cf_dq_q_all)) if cf_dq_q_all else 0.0,
        "mean_step_gain_f_on_disagree": float(np.mean(cf_dq_f_all)) if cf_dq_f_all else 0.0,
        "mean_step_log_gain_q_on_disagree": float(np.mean(cf_dlogq_q_all)) if cf_dlogq_q_all else 0.0,
        "mean_step_log_gain_f_on_disagree": float(np.mean(cf_dlogq_f_all)) if cf_dlogq_f_all else 0.0,
        "all_disagreements_delta_q": paired_bootstrap_ci(np.array(cf_dq_diff_all)),
        "all_disagreements_delta_logq": paired_bootstrap_ci(np.array(cf_dlogq_diff_all)),
        "nontriv_disagreements_delta_q": paired_bootstrap_ci(np.array(cf_dq_diff_nontriv)),
        "nontriv_disagreements_delta_logq": paired_bootstrap_ci(np.array(cf_dlogq_diff_nontriv)),
        "unsolved90_disagreements_delta_q": paired_bootstrap_ci(np.array(cf_dq_diff_unsolved90)),
        "unsolved90_disagreements_delta_logq": paired_bootstrap_ci(np.array(cf_dlogq_diff_unsolved90)),
    }


@torch.no_grad()
def precompute_catalog_embs(
    model: QuoMechNet,
    grammar: ReactionGrammar,
    t_grid: np.ndarray,
    trajs_sel: np.ndarray,
    device: torch.device,
    sigma_rel: float,
    rng_obs: np.random.Generator,
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, torch.Tensor, torch.Tensor]:
    """Pre-encode fixed CRN observations y*(i, d) for all N x 10 catalog designs + initial D0 state."""
    N = len(trajs_sel)
    obs_ch = get_observable_channel_indices(grammar)
    ch_is_cat = np.array([grammar.species[s].is_catalyst for s in obs_ch], dtype=bool)
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    n_t = 20
    t_eval = np.linspace(0.0, 1.0, n_t)

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

    h_d0 = exp_embs_all[:, 0:1, :]
    ep0 = torch.zeros((N, 1), dtype=torch.bool, device=device)
    with torch.cuda.amp.autocast(enabled=use_bf16, dtype=torch.bfloat16):
        f_d0 = model.set_encoder(h_d0, exp_pad_mask=ep0)
        logits_d0, G_mat = model.mech_head(f_d0)
    probs_m_d0 = torch.softmax(logits_d0.float(), dim=-1).cpu().numpy()
    return exp_embs_all, G_mat.float(), probs_m_d0, f_d0.float(), h_d0


@torch.no_grad()
def run_discrete_partition_boed(
    model: QuoMechNet,
    grammar: ReactionGrammar,
    sim: ODESimulator,
    exp_embs_all: torch.Tensor,
    G_mat: torch.Tensor,
    probs_m_d0: np.ndarray,
    f_d0: torch.Tensor,
    h_d0: torch.Tensor,
    true_mids: np.ndarray,
    tier_designs: List[int],
    m2c: np.ndarray,
    classes: List[List[int]],
    policies: Sequence[str],
    device: torch.device,
    tier_name: str,
    n_mc: int = 512,
    sigma_rel: float = 0.01,
    seed: int = 202615,
    horizons_override: Optional[List[int]] = None,
    verify_crn: bool = False,
) -> Dict:
    """Generic strict-CRN discrete BOED runner for any design tier and any quotient partition (E0 or C_macro)."""
    t0 = time.perf_counter()
    N = len(true_mids)
    n_t = 20
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    catalog_pool = [(d, 1.0) for d in grammar.catalog]
    T_eff = min(6, len(tier_designs) - 1)
    horizons = horizons_override if horizons_override is not None else sorted(list({4, T_eff}))
    n_cls = len(classes)
    true_c = m2c[true_mids]
    nontriv_mask = np.array([len(classes[c]) >= 2 for c in true_c], dtype=bool)
    probs_c_d0 = aggregate_class_probs(probs_m_d0, m2c, n_cls)

    def eval_state(chosen_hist: List[List[int]]) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        E = len(chosen_hist[0])
        idx_t = torch.tensor(chosen_hist, dtype=torch.long, device=device)
        ar = torch.arange(N, device=device).unsqueeze(1).expand(N, E)
        h_seq = exp_embs_all[ar, idx_t]
        ep = torch.zeros((N, E), dtype=torch.bool, device=device)
        with torch.cuda.amp.autocast(enabled=use_bf16, dtype=torch.bfloat16):
            f = model.set_encoder(h_seq, exp_pad_mask=ep)
            logits, _ = model.mech_head(f)
        return torch.softmax(logits.float(), dim=-1).cpu().numpy(), f.float(), h_seq

    cand_d0 = [[d for d in tier_designs if d != 0] for _ in range(N)]
    rng_s0 = np.random.default_rng(seed + 100)
    eig_q_d0, eig_f_d0 = _score_eig_designs_batched(
        model, grammar, sim, h_d0, f_d0, G_mat, probs_m_d0, probs_c_d0,
        catalog_pool, cand_d0, m2c, n_cls, n_t, sigma_rel, n_mc=n_mc, device=device,
        rng=rng_s0, crn_seed=seed + 100, case_ids=list(range(N)),
    )

    crn_max_diff = None
    if verify_crn:
        # Bit-exact Strict CRN audit on 16 cases
        eq_chk, ef_chk = _score_eig_designs_batched(
            model, grammar, sim, h_d0[:16], f_d0[:16], G_mat, probs_m_d0[:16], probs_c_d0[:16],
            catalog_pool, cand_d0[:16], m2c, n_cls, n_t, sigma_rel, n_mc=n_mc, device=device,
            rng=np.random.default_rng(seed + 100), crn_seed=seed + 100, case_ids=list(range(16)),
        )
        crn_max_diff = float(max(
            max(float(np.max(np.abs(eq_chk[i] - eig_q_d0[i]))) for i in range(16)),
            max(float(np.max(np.abs(ef_chk[i] - eig_f_d0[i]))) for i in range(16)),
        ))

    eig_q_d0_mat = np.stack(eig_q_d0, axis=0)
    static_seq = [cand_d0[0][i] for i in np.argsort(np.mean(eig_q_d0_mat, axis=0))[::-1][:T_eff]]
    if tier_name == "no_P":
        textbook_seq = [2, 5, 3, 4, 1][:T_eff]
    elif tier_name == "basic":
        textbook_seq = [2, 5, 6, 3, 4, 1][:T_eff]
    else:
        textbook_seq = [2, 5, 6, 3, 4, 8][:T_eff]

    # Step-1 emergent RPKA/VTNA/Cross-Core diagnostics
    step1_q = [cand_d0[i][int(np.argmax(eig_q_d0[i]))] for i in range(N)]
    rpka_h, rpka_t, vtna_h, vtna_t, sub_h, sub_t, act_h, act_t, xcore_h, xcore_t = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    for i in range(N):
        if probs_c_d0[i].max() >= 0.90:
            continue
        t2 = np.argsort(probs_c_d0[i])[-2:][::-1]
        ma = max(classes[t2[0]], key=lambda m: probs_m_d0[i, m])
        mb = max(classes[t2[1]], key=lambda m: probs_m_d0[i, m])
        oa, ob = grammar.mech_options[ma], grammar.mech_options[mb]
        ca, cb = int(grammar.mech_core_id[ma]), int(grammar.mech_core_id[mb])
        ch = step1_q[i]
        if ((oa[1] > 0) != (ob[1] > 0)) and (((oa[3] > 0) or (oa[6] > 0)) != ((ob[3] > 0) or (ob[6] > 0))):
            rpka_t += 1
            if ch in (5, 6, 7):
                rpka_h += 1
        if (oa[4] > 0, oa[7] > 0, oa[1] == 2, oa[5] > 0) != (ob[4] > 0, ob[7] > 0, ob[1] == 2, ob[5] > 0):
            vtna_t += 1
            if ch in (1, 2):
                vtna_h += 1
        if oa[2] != ob[2]:
            sub_t += 1
            if ch in (3, 4):
                sub_h += 1
        if oa[0] != ob[0] and 8 in tier_designs:
            act_t += 1
            if ch == 8:
                act_h += 1
        if ca != cb:
            xcore_t += 1
            if ch in (3, 4, 9):
                xcore_h += 1

    emergence = {
        "step1_choice_dist": {grammar.catalog[d].name: int(step1_q.count(d)) for d in tier_designs if d != 0},
        "RPKA_M2_vs_M4M7_D5D6D7": {"hits": rpka_h, "total": rpka_t, "rate": rpka_h / max(rpka_t, 1)},
        "VTNA_cat_order_D1D2": {"hits": vtna_h, "total": vtna_t, "rate": vtna_h / max(vtna_t, 1)},
        "Sub_M3_D3D4": {"hits": sub_h, "total": sub_t, "rate": sub_h / max(sub_t, 1)},
        "Activation_M1_D8": {"hits": act_h, "total": act_t, "rate": act_h / max(act_t, 1)},
        "CrossCore_D3D4D9": {"hits": xcore_h, "total": xcore_t, "rate": xcore_h / max(xcore_t, 1)},
    }

    cf_dq_q_all, cf_dq_f_all, cf_dlogq_q_all, cf_dlogq_f_all = [], [], [], []
    cf_dq_diff_all, cf_dlogq_diff_all, cf_dq_diff_nt, cf_dlogq_diff_nt = [], [], [], []
    cf_dq_diff_u90, cf_dlogq_diff_u90 = [], []
    dis_step, dis_cnt_step, dis_nt_step, dis_u90_step = [], [], [], []
    ifrac_step, ifrac_nt_step = [], []

    def run_policy(pname: str, record_cf: bool = False):
        hist = [[0] for _ in range(N)]
        pc_traj = np.zeros((N, T_eff + 1), dtype=np.float64)
        pm_traj = np.zeros((N, T_eff + 1), dtype=np.float64)
        max_pc = np.zeros((N, T_eff + 1), dtype=np.float64)
        max_pm = np.zeros((N, T_eff + 1), dtype=np.float64)
        top1_traj = np.zeros((N, T_eff + 1), dtype=np.float64)

        pm_curr, pc_curr, f_curr, h_curr = probs_m_d0.copy(), probs_c_d0.copy(), f_d0, h_d0
        pc_traj[:, 0] = pc_curr[np.arange(N), true_c]
        pm_traj[:, 0] = pm_curr[np.arange(N), true_mids]
        max_pc[:, 0] = np.max(pc_curr, axis=1)
        max_pm[:, 0] = np.max(pm_curr, axis=1)
        top1_traj[:, 0] = (np.argmax(pc_curr, axis=1) == true_c).astype(np.float64)
        rng_r = np.random.default_rng(seed + 7777)

        for step in range(T_eff):
            cands_all = [[d for d in tier_designs if d not in hist[i]] for i in range(N)]
            if pname == "Random":
                chosen = [int(rng_r.choice(cands_all[i])) for i in range(N)]
            elif pname == "Textbook_RPKA_VTNA":
                chosen = [textbook_seq[step] for _ in range(N)]
            elif pname == "Static_EIG":
                chosen = [static_seq[step] for _ in range(N)]
            else:
                if step == 0:
                    eq_l, ef_l = eig_q_d0, eig_f_d0
                else:
                    step_seed = seed + 100 + step * 1000
                    eq_l, ef_l = _score_eig_designs_batched(
                        model, grammar, sim, h_curr, f_curr, G_mat, pm_curr, pc_curr,
                        catalog_pool, cands_all, m2c, n_cls, n_t, sigma_rel, n_mc=n_mc, device=device,
                        rng=np.random.default_rng(step_seed), crn_seed=step_seed, case_ids=list(range(N)),
                    )
                q_p = [cands_all[i][int(np.argmax(eq_l[i]))] for i in range(N)]
                f_p = [cands_all[i][int(np.argmax(ef_l[i]))] for i in range(N)]

                if record_cf:
                    dmask = [q_p[i] != f_p[i] for i in range(N)]
                    u90 = [max_pc[i, step] < 0.90 for i in range(N)]
                    dis_step.append(float(np.mean(dmask)))
                    dis_cnt_step.append(int(sum(dmask)))
                    dis_nt_step.append(float(np.mean([dmask[i] for i in range(N) if nontriv_mask[i]])) if np.any(nontriv_mask) else 0.0)
                    dis_u90_step.append(float(np.mean([dmask[i] for i in range(N) if u90[i]])) if any(u90) else 0.0)

                    ifs, ifs_nt = [], []
                    for i in range(N):
                        idx_f = int(np.argmax(ef_l[i]))
                        em_v, ec_v = float(ef_l[i][idx_f]), float(eq_l[i][idx_f])
                        if em_v > 1e-6:
                            fr = max(0.0, (em_v - ec_v) / em_v)
                            ifs.append(fr)
                            if nontriv_mask[i]:
                                ifs_nt.append(fr)
                    ifrac_step.append(float(np.mean(ifs)) if ifs else 0.0)
                    ifrac_nt_step.append(float(np.mean(ifs_nt)) if ifs_nt else 0.0)

                    dis_idx = [i for i in range(N) if dmask[i]]
                    if dis_idx:
                        pm_q_cf, _, _ = eval_state([hist[i] + [q_p[i]] for i in range(N)])
                        pm_f_cf, _, _ = eval_state([hist[i] + [f_p[i]] for i in range(N)])
                        pc_q_cf = aggregate_class_probs(pm_q_cf, m2c, n_cls)
                        pc_f_cf = aggregate_class_probs(pm_f_cf, m2c, n_cls)
                        for i in dis_idx:
                            p0 = float(np.clip(pc_traj[i, step], 1e-12, 1.0))
                            pq = float(np.clip(pc_q_cf[i, true_c[i]], 1e-12, 1.0))
                            pf = float(np.clip(pc_f_cf[i, true_c[i]], 1e-12, 1.0))
                            dq_q, dq_f = pq - p0, pf - p0
                            dl_q, dl_f = float(np.log(pq) - np.log(p0)), float(np.log(pf) - np.log(p0))
                            cf_dq_q_all.append(dq_q); cf_dq_f_all.append(dq_f)
                            cf_dlogq_q_all.append(dl_q); cf_dlogq_f_all.append(dl_f)
                            cf_dq_diff_all.append(dq_q - dq_f); cf_dlogq_diff_all.append(dl_q - dl_f)
                            if nontriv_mask[i]:
                                cf_dq_diff_nt.append(dq_q - dq_f); cf_dlogq_diff_nt.append(dl_q - dl_f)
                            if u90[i]:
                                cf_dq_diff_u90.append(dq_q - dq_f); cf_dlogq_diff_u90.append(dl_q - dl_f)
                chosen = q_p if pname == "Quotient_EIG" else f_p

            for i in range(N):
                hist[i].append(chosen[i])
            pm_curr, f_curr, h_curr = eval_state(hist)
            pc_curr = aggregate_class_probs(pm_curr, m2c, n_cls)
            pc_traj[:, step + 1] = pc_curr[np.arange(N), true_c]
            pm_traj[:, step + 1] = pm_curr[np.arange(N), true_mids]
            max_pc[:, step + 1] = np.max(pc_curr, axis=1)
            max_pm[:, step + 1] = np.max(pm_curr, axis=1)
            top1_traj[:, step + 1] = (np.argmax(pc_curr, axis=1) == true_c).astype(np.float64)

        all_s, all_r = _summarize_trajectories(pc_traj, pm_traj, max_pc, max_pm, top1_traj, horizons=horizons)
        nt_s, nt_r = (
            _summarize_trajectories(
                pc_traj[nontriv_mask], pm_traj[nontriv_mask],
                max_pc[nontriv_mask], max_pm[nontriv_mask], top1_traj[nontriv_mask], horizons=horizons,
            )
            if np.any(nontriv_mask) else ({}, {})
        )
        return {"all_cases": all_s, "nontrivial_cases": nt_s}, {"all": all_r, "nontriv": nt_r}

    pol_res, pol_raw = {}, {}
    for pname in policies:
        pol_res[pname], pol_raw[pname] = run_policy(pname, record_cf=(pname == "Quotient_EIG"))

    # Paired Bootstrap 95% CIs
    paired_ci = {"all_cases": {}, "nontrivial_cases": {}}
    if "Quotient_EIG" in pol_raw and "Fine_EIG" in pol_raw:
        for sub_k, out_k in [("all", "all_cases"), ("nontriv", "nontrivial_cases")]:
            if not pol_raw["Quotient_EIG"][sub_k]:
                continue
            qr, fr = pol_raw["Quotient_EIG"][sub_k], pol_raw["Fine_EIG"][sub_k]
            for H in horizons:
                paired_ci[out_k][f"Q_minus_F_auc@{H}"] = paired_bootstrap_ci(qr[f"auc@{H}"] - fr[f"auc@{H}"])
                paired_ci[out_k][f"Q_minus_F_top1@{H}"] = paired_bootstrap_ci(qr[f"top1@{H}"] - fr[f"top1@{H}"])
                paired_ci[out_k][f"Q_minus_F_pc@{H}"] = paired_bootstrap_ci(qr[f"pc@{H}"] - fr[f"pc@{H}"])
                if "Random" in pol_raw:
                    rr = pol_raw["Random"][sub_k]
                    paired_ci[out_k][f"Q_minus_Rand_auc@{H}"] = paired_bootstrap_ci(qr[f"auc@{H}"] - rr[f"auc@{H}"])
                    paired_ci[out_k][f"Q_minus_Rand_top1@{H}"] = paired_bootstrap_ci(qr[f"top1@{H}"] - rr[f"top1@{H}"])
                if "Textbook_RPKA_VTNA" in pol_raw:
                    tr = pol_raw["Textbook_RPKA_VTNA"][sub_k]
                    paired_ci[out_k][f"Q_minus_Textbook_auc@{H}"] = paired_bootstrap_ci(qr[f"auc@{H}"] - tr[f"auc@{H}"])
                    paired_ci[out_k][f"Q_minus_Textbook_top1@{H}"] = paired_bootstrap_ci(qr[f"top1@{H}"] - tr[f"top1@{H}"])
                for t_key in ["tau_85", "tau_90"]:
                    paired_ci[out_k][f"delta_cls_stop@{H}_{t_key}_F_minus_Q"] = paired_bootstrap_ci(
                        fr[f"cls_stop@{H}_{t_key}"] - qr[f"cls_stop@{H}_{t_key}"]
                    )
                    paired_ci[out_k][f"fine_wasted_exps@{H}_{t_key}"] = paired_bootstrap_ci(fr[f"wasted@{H}_{t_key}"])
                    paired_ci[out_k][f"end2end_budget_saving_Ffine_minus_Qcls@{H}_{t_key}"] = paired_bootstrap_ci(
                        fr[f"fine_stop@{H}_{t_key}"] - qr[f"cls_stop@{H}_{t_key}"]
                    )

    dt = time.perf_counter() - t0
    out = {
        "n_cases": N,
        "n_classes": n_cls,
        "n_nontrivial_cases": int(np.sum(nontriv_mask)),
        "T_eff": T_eff,
        "n_mc": n_mc,
        "crn_bit_exact_max_diff": crn_max_diff,
        "static_seq": [grammar.catalog[d].name for d in static_seq],
        "textbook_seq": [grammar.catalog[d].name for d in textbook_seq],
        "elapsed_s": dt,
        "strategies": pol_res,
        "paired_bootstrap_ci": paired_ci,
        "same_state_counterfactual": _package_cf_stats(
            dis_step, dis_cnt_step, dis_nt_step, dis_u90_step, ifrac_step, ifrac_nt_step,
            cf_dq_q_all, cf_dq_f_all, cf_dlogq_q_all, cf_dlogq_f_all,
            cf_dq_diff_all, cf_dlogq_diff_all, cf_dq_diff_nt, cf_dlogq_diff_nt,
            cf_dq_diff_u90, cf_dlogq_diff_u90,
        ),
        "emergence": emergence,
    }
    qH = pol_res["Quotient_EIG"]["all_cases"]["horizons"][f"T={T_eff}"]
    fH = pol_res["Fine_EIG"]["all_cases"]["horizons"][f"T={T_eff}"]
    print(
        f"  [Discrete:{tier_name}] {dt:.1f}s | Q AUC@{T_eff}={qH['auc_true_class_prob']:.3f} top1={qH['top1_at_H']:.3f} "
        f"vs F AUC@{T_eff}={fH['auc_true_class_prob']:.3f} top1={fH['top1_at_H']:.3f} | "
        f"CF dis={np.mean(dis_step):.3f} d_q={np.mean(cf_dq_diff_all) if cf_dq_diff_all else 0:+.4f}",
        flush=True,
    )
    return out


@torch.no_grad()
def run_continuous_lhs128_suite(
    model: QuoMechNet,
    grammar: ReactionGrammar,
    sim: ODESimulator,
    t_grid: np.ndarray,
    trajs_sel: np.ndarray,
    true_mids: np.ndarray,
    true_thetas: np.ndarray,
    equiv_xref: Dict,
    equiv_basic: Dict,
    c_macro_part: Dict,
    device: torch.device,
    n_lhs: int = 128,
    budget: int = 6,
    n_mc: int = 256,
    sigma_rel: float = 0.01,
    seed: int = 202633,
) -> Tuple[Dict, Dict]:
    """Continuous X_basic LHS-128 BOED evaluated on:
      (1) Standard quotient C_{X_basic^ref} (also cross-scored on C_{Xi_basic}!), AND
      (2) Zero-retrain coarse task quotient C_macro (reusing the exact same pre-encoded LHS-128 pool!).
    """
    t0_all = time.perf_counter()
    N = len(true_mids)
    obs_ch = get_observable_channel_indices(grammar)
    ch_is_cat = np.array([grammar.species[s].is_catalyst for s in obs_ch], dtype=bool)
    sA, sP = grammar.species_idx["A"], grammar.species_idx["P"]
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    rng_obs = np.random.default_rng(seed)

    # LHS-128 over X_basic: A0 in [0.25,2.0], B0 in [0.25,3.0], P0 in {0} U [0.1,1.0], cat in [0.25,4.0]
    sampler = qmc.LatinHypercube(d=4, seed=seed)
    u = sampler.random(n=n_lhs)
    A0_arr = 0.25 + u[:, 0] * 1.75
    B0_arr = 0.25 + u[:, 1] * 2.75
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

    # Pre-simulate & pre-encode CRN ground-truth observations for all N x 128 LHS candidates ONCE!
    n_t = 20
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

    # Initial D0 embedding
    t_eval_1 = np.linspace(0.0, 1.0, n_t)
    d0 = grammar.catalog[0]
    clean_d0 = np.stack([interp_traj_to_times(t_grid, trajs_sel[i, 0].astype(np.float64), t_eval_1) for i in range(N)], axis=0)
    n_sc0 = np.where(ch_is_cat, d0.cat_mult * grammar.c0_base, d0.A0)
    y_d0 = clean_d0 + rng_obs.normal(0.0, 1.0, size=clean_d0.shape) * (sigma_rel * n_sc0[None, None, :])
    df0, tm0, ov0, om0, tp0 = _build_single_exp_tensors(grammar, d0, 1.0, y_d0, t_eval_1)
    with torch.cuda.amp.autocast(enabled=use_bf16, dtype=torch.bfloat16):
        h_d0 = model.exp_encoder(
            torch.from_numpy(df0).to(device), torch.from_numpy(tm0).to(device),
            torch.from_numpy(ov0).to(device), torch.from_numpy(om0).to(device),
            torch.from_numpy(tp0).to(device),
        ).float().unsqueeze(1)
        f_d0 = model.set_encoder(h_d0, exp_pad_mask=torch.zeros((N, 1), dtype=torch.bool, device=device))
        logits_d0, G_mat = model.mech_head(f_d0)
    probs_m_d0 = torch.softmax(logits_d0.float(), dim=-1).cpu().numpy()
    f_d0, G_mat = f_d0.float(), G_mat.float()

    def eval_lhs_state(h_prev: torch.Tensor, new_idx: Sequence[int]) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        idx_t = torch.tensor(new_idx, dtype=torch.long, device=device)
        new_emb = exp_embs_lhs[torch.arange(N, device=device), idx_t].unsqueeze(1)
        h_cat = torch.cat([h_prev, new_emb], dim=1)
        ep = torch.zeros((N, h_cat.shape[1]), dtype=torch.bool, device=device)
        with torch.cuda.amp.autocast(enabled=use_bf16, dtype=torch.bfloat16):
            f = model.set_encoder(h_cat, exp_pad_mask=ep)
            logits, _ = model.mech_head(f)
        return torch.softmax(logits.float(), dim=-1).cpu().numpy(), f.float(), h_cat

    def _run_lhs_for_partition(
        tag: str,
        m2c: np.ndarray,
        classes: List[List[int]],
        horizons: List[int],
        secondary_m2c: Optional[np.ndarray] = None,
        secondary_classes: Optional[List[List[int]]] = None,
        save_spatial: bool = False,
        part_seed: int = 500,
    ) -> Dict:
        t0_p = time.perf_counter()
        n_cls = len(classes)
        true_c = m2c[true_mids]
        nontriv_mask = np.array([len(classes[c]) >= 2 for c in true_c], dtype=bool)
        probs_c_d0 = aggregate_class_probs(probs_m_d0, m2c, n_cls)

        if secondary_m2c is not None:
            n_cls2 = len(secondary_classes)
            true_c2 = secondary_m2c[true_mids]
            nontriv2 = np.array([len(secondary_classes[c]) >= 2 for c in true_c2], dtype=bool)
            probs_c2_d0 = aggregate_class_probs(probs_m_d0, secondary_m2c, n_cls2)

        all_cands_0 = [list(range(n_lhs)) for _ in range(N)]
        eig_q_s0, eig_f_s0 = _score_eig_designs_batched(
            model, grammar, sim, h_d0, f_d0, G_mat, probs_m_d0, probs_c_d0,
            lhs_pool, all_cands_0, m2c, n_cls, n_t, sigma_rel, n_mc=n_mc, device=device,
            rng=np.random.default_rng(seed + part_seed), crn_seed=seed + part_seed, case_ids=list(range(N)),
        )

        cf_dq_q_all, cf_dq_f_all, cf_dlogq_q_all, cf_dlogq_f_all = [], [], [], []
        cf_dq_diff_all, cf_dlogq_diff_all, cf_dq_diff_nt, cf_dlogq_diff_nt = [], [], [], []
        cf_dq_diff_u90, cf_dlogq_diff_u90 = [], []
        dis_step, dis_cnt_step, dis_nt_step, dis_u90_step = [], [], [], []
        ifrac_step, ifrac_nt_step = [], []
        step1_q_p, step2_q_p, step1_f_p, step2_f_p = [], [], [], []
        pm_s1_q = pc_s1_q = pm_s1_f = pc_s1_f = None

        def run_p(policy: str, record_cf: bool = False):
            nonlocal pm_s1_q, pc_s1_q, pm_s1_f, pc_s1_f
            h_curr, f_curr = h_d0.clone(), f_d0.clone()
            pm_curr, pc_curr = probs_m_d0.copy(), probs_c_d0.copy()
            used = [set() for _ in range(N)]
            pc_traj = np.zeros((N, budget + 1), dtype=np.float64)
            pm_traj = np.zeros((N, budget + 1), dtype=np.float64)
            max_pc = np.zeros((N, budget + 1), dtype=np.float64)
            max_pm = np.zeros((N, budget + 1), dtype=np.float64)
            top1_traj = np.zeros((N, budget + 1), dtype=np.float64)
            pc_traj[:, 0] = pc_curr[np.arange(N), true_c]
            pm_traj[:, 0] = pm_curr[np.arange(N), true_mids]
            max_pc[:, 0] = np.max(pc_curr, axis=1)
            max_pm[:, 0] = np.max(pm_curr, axis=1)
            top1_traj[:, 0] = (np.argmax(pc_curr, axis=1) == true_c).astype(np.float64)

            if secondary_m2c is not None:
                pc2_traj = np.zeros((N, budget + 1), dtype=np.float64)
                max_pc2 = np.zeros((N, budget + 1), dtype=np.float64)
                top1_2_traj = np.zeros((N, budget + 1), dtype=np.float64)
                pc2_traj[:, 0] = probs_c2_d0[np.arange(N), true_c2]
                max_pc2[:, 0] = np.max(probs_c2_d0, axis=1)
                top1_2_traj[:, 0] = (np.argmax(probs_c2_d0, axis=1) == true_c2).astype(np.float64)

            rng_r = np.random.default_rng(seed + part_seed + 777)
            for step in range(budget):
                cands = [[k for k in range(n_lhs) if k not in used[i]] for i in range(N)]
                if policy == "Continuous_Random":
                    picks = [int(rng_r.choice(cands[i])) for i in range(N)]
                else:
                    if step == 0:
                        eq_l, ef_l = eig_q_s0, eig_f_s0
                    else:
                        s_seed = seed + part_seed + step * 1000
                        eq_l, ef_l = _score_eig_designs_batched(
                            model, grammar, sim, h_curr, f_curr, G_mat, pm_curr, pc_curr,
                            lhs_pool, cands, m2c, n_cls, n_t, sigma_rel, n_mc=n_mc, device=device,
                            rng=np.random.default_rng(s_seed), crn_seed=s_seed, case_ids=list(range(N)),
                        )
                    qp = [cands[i][int(np.argmax(eq_l[i]))] for i in range(N)]
                    fp = [cands[i][int(np.argmax(ef_l[i]))] for i in range(N)]
                    if save_spatial:
                        if step == 0 and policy == "Continuous_Quotient_EIG":
                            step1_q_p.extend(qp); step1_f_p.extend(fp)
                        elif step == 1 and policy == "Continuous_Quotient_EIG":
                            step2_q_p.extend(qp)
                        elif step == 1 and policy == "Continuous_Fine_EIG":
                            step2_f_p.extend(fp)
                    if record_cf:
                        dmask = [qp[i] != fp[i] for i in range(N)]
                        u90 = [max_pc[i, step] < 0.90 for i in range(N)]
                        dis_step.append(float(np.mean(dmask)))
                        dis_cnt_step.append(int(sum(dmask)))
                        dis_nt_step.append(float(np.mean([dmask[i] for i in range(N) if nontriv_mask[i]])) if np.any(nontriv_mask) else 0.0)
                        dis_u90_step.append(float(np.mean([dmask[i] for i in range(N) if u90[i]])) if any(u90) else 0.0)
                        ifs, ifs_nt = [], []
                        for i in range(N):
                            idx_f = int(np.argmax(ef_l[i]))
                            em_v, ec_v = float(ef_l[i][idx_f]), float(eq_l[i][idx_f])
                            if em_v > 1e-6:
                                fr = max(0.0, (em_v - ec_v) / em_v)
                                ifs.append(fr)
                                if nontriv_mask[i]:
                                    ifs_nt.append(fr)
                        ifrac_step.append(float(np.mean(ifs)) if ifs else 0.0)
                        ifrac_nt_step.append(float(np.mean(ifs_nt)) if ifs_nt else 0.0)

                        dis_idx = [i for i in range(N) if dmask[i]]
                        if dis_idx:
                            pm_q_cf, _, _ = eval_lhs_state(h_curr, qp)
                            pm_f_cf, _, _ = eval_lhs_state(h_curr, fp)
                            pc_q_cf = aggregate_class_probs(pm_q_cf, m2c, n_cls)
                            pc_f_cf = aggregate_class_probs(pm_f_cf, m2c, n_cls)
                            for i in dis_idx:
                                p0 = float(np.clip(pc_traj[i, step], 1e-12, 1.0))
                                pq = float(np.clip(pc_q_cf[i, true_c[i]], 1e-12, 1.0))
                                pf = float(np.clip(pc_f_cf[i, true_c[i]], 1e-12, 1.0))
                                dq_q, dq_f = pq - p0, pf - p0
                                dl_q, dl_f = float(np.log(pq) - np.log(p0)), float(np.log(pf) - np.log(p0))
                                cf_dq_q_all.append(dq_q); cf_dq_f_all.append(dq_f)
                                cf_dlogq_q_all.append(dl_q); cf_dlogq_f_all.append(dl_f)
                                cf_dq_diff_all.append(dq_q - dq_f); cf_dlogq_diff_all.append(dl_q - dl_f)
                                if nontriv_mask[i]:
                                    cf_dq_diff_nt.append(dq_q - dq_f); cf_dlogq_diff_nt.append(dl_q - dl_f)
                                if u90[i]:
                                    cf_dq_diff_u90.append(dq_q - dq_f); cf_dlogq_diff_u90.append(dl_q - dl_f)
                    picks = qp if policy == "Continuous_Quotient_EIG" else fp

                for i, k in enumerate(picks):
                    used[i].add(k)
                pm_curr, f_curr, h_curr = eval_lhs_state(h_curr, picks)
                pc_curr = aggregate_class_probs(pm_curr, m2c, n_cls)
                if save_spatial and step == 0:
                    if policy == "Continuous_Quotient_EIG":
                        pm_s1_q, pc_s1_q = pm_curr.copy(), pc_curr.copy()
                    elif policy == "Continuous_Fine_EIG":
                        pm_s1_f, pc_s1_f = pm_curr.copy(), pc_curr.copy()

                pc_traj[:, step + 1] = pc_curr[np.arange(N), true_c]
                pm_traj[:, step + 1] = pm_curr[np.arange(N), true_mids]
                max_pc[:, step + 1] = np.max(pc_curr, axis=1)
                max_pm[:, step + 1] = np.max(pm_curr, axis=1)
                top1_traj[:, step + 1] = (np.argmax(pc_curr, axis=1) == true_c).astype(np.float64)
                if secondary_m2c is not None:
                    pc2 = aggregate_class_probs(pm_curr, secondary_m2c, n_cls2)
                    pc2_traj[:, step + 1] = pc2[np.arange(N), true_c2]
                    max_pc2[:, step + 1] = np.max(pc2, axis=1)
                    top1_2_traj[:, step + 1] = (np.argmax(pc2, axis=1) == true_c2).astype(np.float64)

            all_s, all_r = _summarize_trajectories(pc_traj, pm_traj, max_pc, max_pm, top1_traj, horizons=horizons)
            nt_s, nt_r = (
                _summarize_trajectories(
                    pc_traj[nontriv_mask], pm_traj[nontriv_mask],
                    max_pc[nontriv_mask], max_pm[nontriv_mask], top1_traj[nontriv_mask], horizons=horizons,
                )
                if np.any(nontriv_mask) else ({}, {})
            )
            res_obj = {"all_cases": all_s, "nontrivial_cases": nt_s}
            if secondary_m2c is not None:
                s2_all, _ = _summarize_trajectories(pc2_traj, pm_traj, max_pc2, max_pm, top1_2_traj, horizons=horizons)
                s2_nt, _ = _summarize_trajectories(
                    pc2_traj[nontriv2], pm_traj[nontriv2], max_pc2[nontriv2], max_pm[nontriv2], top1_2_traj[nontriv2], horizons=horizons
                )
                res_obj["eval_under_Xi_basic"] = {"all_cases": s2_all, "nontrivial_cases": s2_nt}
            return res_obj, {"all": all_r, "nontriv": nt_r}

        q_res, q_raw = run_p("Continuous_Quotient_EIG", record_cf=True)
        f_res, f_raw = run_p("Continuous_Fine_EIG", record_cf=False)
        r_res, r_raw = run_p("Continuous_Random", record_cf=False)

        paired_ci = {"all_cases": {}, "nontrivial_cases": {}}
        for sub_k, out_k in [("all", "all_cases"), ("nontriv", "nontrivial_cases")]:
            if not q_raw[sub_k]:
                continue
            qr, fr, rr = q_raw[sub_k], f_raw[sub_k], r_raw[sub_k]
            for H in horizons:
                paired_ci[out_k][f"Q_minus_F_auc@{H}"] = paired_bootstrap_ci(qr[f"auc@{H}"] - fr[f"auc@{H}"])
                paired_ci[out_k][f"Q_minus_F_top1@{H}"] = paired_bootstrap_ci(qr[f"top1@{H}"] - fr[f"top1@{H}"])
                paired_ci[out_k][f"Q_minus_Rand_auc@{H}"] = paired_bootstrap_ci(qr[f"auc@{H}"] - rr[f"auc@{H}"])
                paired_ci[out_k][f"Q_minus_Rand_top1@{H}"] = paired_bootstrap_ci(qr[f"top1@{H}"] - rr[f"top1@{H}"])
                for t_key in ["tau_85", "tau_90"]:
                    paired_ci[out_k][f"delta_cls_stop@{H}_{t_key}_F_minus_Q"] = paired_bootstrap_ci(
                        fr[f"cls_stop@{H}_{t_key}"] - qr[f"cls_stop@{H}_{t_key}"]
                    )
                    paired_ci[out_k][f"fine_wasted_exps@{H}_{t_key}"] = paired_bootstrap_ci(fr[f"wasted@{H}_{t_key}"])
                    paired_ci[out_k][f"end2end_budget_saving_Ffine_minus_Qcls@{H}_{t_key}"] = paired_bootstrap_ci(
                        fr[f"fine_stop@{H}_{t_key}"] - qr[f"cls_stop@{H}_{t_key}"]
                    )

        out = {
            "n_cases": N, "n_classes": n_cls, "n_nontrivial_cases": int(np.sum(nontriv_mask)),
            "n_lhs": n_lhs, "budget": budget, "n_mc": n_mc, "elapsed_s": time.perf_counter() - t0_p,
            "Continuous_Quotient_EIG": q_res, "Continuous_Fine_EIG": f_res, "Continuous_Random": r_res,
            "paired_bootstrap_ci": paired_ci,
            "same_state_counterfactual": _package_cf_stats(
                dis_step, dis_cnt_step, dis_nt_step, dis_u90_step, ifrac_step, ifrac_nt_step,
                cf_dq_q_all, cf_dq_f_all, cf_dlogq_q_all, cf_dlogq_f_all,
                cf_dq_diff_all, cf_dlogq_diff_all, cf_dq_diff_nt, cf_dlogq_diff_nt,
                cf_dq_diff_u90, cf_dlogq_diff_u90,
            ),
        }

        if save_spatial:
            def analyze_sp(picks: List[int], pm_s: np.ndarray, pc_s: np.ndarray) -> Dict:
                vtna_a, rpka_a, sub_a = [], [], []
                for i in range(N):
                    if pc_s[i].max() >= 0.90:
                        continue
                    t2 = np.argsort(pc_s[i])[-2:][::-1]
                    ma = max(classes[t2[0]], key=lambda m: pm_s[i, m])
                    mb = max(classes[t2[1]], key=lambda m: pm_s[i, m])
                    oa, ob = grammar.mech_options[ma], grammar.mech_options[mb]
                    k = picks[i]
                    if (oa[4] > 0, oa[7] > 0, oa[1] == 2, oa[5] > 0) != (ob[4] > 0, ob[7] > 0, ob[1] == 2, ob[5] > 0):
                        vtna_a.append(k)
                    if ((oa[1] > 0) != (ob[1] > 0)) or (((oa[3] > 0) or (oa[6] > 0)) != ((ob[3] > 0) or (ob[6] > 0))):
                        rpka_a.append(k)
                    if oa[2] != ob[2]:
                        sub_a.append(k)
                return {
                    "all_vtna_extreme_cat_frac": float(np.mean([(cat_arr[k] <= 0.55 or cat_arr[k] >= 1.8) for k in picks])),
                    "all_preadded_P_frac": float(np.mean([P0_arr[k] >= 0.20 for k in picks])),
                    "all_rpka_zone_frac": float(np.mean([(P0_arr[k] >= 0.20 or A0_arr[k] <= 0.80) for k in picks])),
                    "all_high_A0_or_B0_frac": float(np.mean([(A0_arr[k] >= 1.40 or B0_arr[k] >= 2.20) for k in picks])),
                    "VTNA_ambig_subset": {
                        "n": len(vtna_a),
                        "extreme_cat_frac": float(np.mean([(cat_arr[k] <= 0.55 or cat_arr[k] >= 1.8) for k in vtna_a])) if vtna_a else 0.0,
                        "mean_abs_log2_cat": float(np.mean([abs(np.log2(cat_arr[k])) for k in vtna_a])) if vtna_a else 0.0,
                    },
                    "RPKA_ambig_subset": {
                        "n": len(rpka_a),
                        "preadded_P_frac": float(np.mean([P0_arr[k] >= 0.20 for k in rpka_a])) if rpka_a else 0.0,
                        "rpka_zone_frac": float(np.mean([(P0_arr[k] >= 0.20 or A0_arr[k] <= 0.80) for k in rpka_a])) if rpka_a else 0.0,
                    },
                    "Sub_M3_ambig_subset": {
                        "n": len(sub_a),
                        "high_A0_or_B0_frac": float(np.mean([(A0_arr[k] >= 1.40 or B0_arr[k] >= 2.20) for k in sub_a])) if sub_a else 0.0,
                    },
                }

            out["spatial_clustering"] = {
                "uniform_pool_baseline": {
                    "vtna_extreme_cat_frac": float(np.mean((cat_arr <= 0.55) | (cat_arr >= 1.8))),
                    "preadded_P_frac": float(np.mean(P0_arr >= 0.20)),
                    "rpka_zone_frac": float(np.mean((P0_arr >= 0.20) | (A0_arr <= 0.80))),
                    "high_A0_or_B0_frac": float(np.mean((A0_arr >= 1.40) | (B0_arr >= 2.20))),
                },
                "Step1_Quotient": analyze_sp(step1_q_p, probs_m_d0, probs_c_d0),
                "Step1_Fine": analyze_sp(step1_f_p, probs_m_d0, probs_c_d0),
                "Step2_Quotient": analyze_sp(step2_q_p, pm_s1_q, pc_s1_q),
                "Step2_Fine": analyze_sp(step2_f_p, pm_s1_f, pc_s1_f),
                # Compact scatter arrays for fig4
                "plot_lhs_coords": {
                    "A0": A0_arr.tolist(), "B0": B0_arr.tolist(),
                    "P0": P0_arr.tolist(), "cat_mult": cat_arr.tolist(),
                    "step1_q_picks": step1_q_p, "step2_q_picks": step2_q_p,
                },
            }

        qH = q_res["all_cases"]["horizons"][f"T={budget}"]
        fH = f_res["all_cases"]["horizons"][f"T={budget}"]
        rH = r_res["all_cases"]["horizons"][f"T={budget}"]
        print(
            f"  [LHS-128:{tag}] {out['elapsed_s']:.1f}s | Q AUC@6={qH['auc_true_class_prob']:.3f} top1={qH['top1_at_H']:.3f} "
            f"| F AUC@6={fH['auc_true_class_prob']:.3f} top1={fH['top1_at_H']:.3f} "
            f"| Rand AUC@6={rH['auc_true_class_prob']:.3f} top1={rH['top1_at_H']:.3f}",
            flush=True,
        )
        return out

    # 1. Standard continuous quotient C_{X_basic^ref} (also cross-evaluated under C_{Xi_basic})
    eq_x = equiv_xref["primary"]
    eq_b = equiv_basic["primary"]
    res_xbasic = _run_lhs_for_partition(
        "X_basic_ref",
        np.array(eq_x["mech_to_class"], dtype=np.int64), eq_x["classes"],
        horizons=[4, budget],
        secondary_m2c=np.array(eq_b["mech_to_class"], dtype=np.int64),
        secondary_classes=eq_b["classes"],
        save_spatial=True, part_seed=500,
    )

    # 2. Zero-retrain coarse task quotient C_macro on LHS-128!
    res_cmacro_lhs = _run_lhs_for_partition(
        "C_macro",
        c_macro_part["mech_to_class"], c_macro_part["classes"],
        horizons=[2, 3, 4, budget],
        save_spatial=False, part_seed=5500,
    )
    print(f"  [LHS-128 Suite Total] {(time.perf_counter()-t0_all):.1f}s", flush=True)
    return res_xbasic, res_cmacro_lhs


@torch.no_grad()
def run_cost_and_upgrade_suite(
    model: QuoMechNet,
    grammar: ReactionGrammar,
    sim: ODESimulator,
    exp_embs_all: torch.Tensor,
    G_mat: torch.Tensor,
    true_mids: np.ndarray,
    equiv_tiers: Dict[str, Dict],
    device: torch.device,
    n_mc: int = 512,
    max_cost: int = 6,
    sigma_rel: float = 0.01,
    seed: int = 202644,
) -> Dict:
    """Cost-normalized Xi_full BOED (cost<=6, D9=3) + 1-Shot Instrument Upgrade (Xi_basic -> {D7,D8,D9}),
    explicitly tracking BOTH M4/M7-splittable AND Cross-Core Eley-Rideal (C4) vs C1 subsets!
    """
    t0 = time.perf_counter()
    N = len(true_mids)
    n_t = 20
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    catalog_pool = [(d, 1.0) for d in grammar.catalog]
    costs = np.array([int(round(d.cost)) for d in grammar.catalog], dtype=np.int64)

    def eval_var_hist(hist_list: List[List[int]]) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        E_max = max(len(h) for h in hist_list)
        h_seq = torch.zeros((N, E_max, model.exp_encoder.d_model), device=device, dtype=torch.float32)
        ep = torch.ones((N, E_max), device=device, dtype=torch.bool)
        for i, h in enumerate(hist_list):
            for e_i, d_i in enumerate(h):
                h_seq[i, e_i] = exp_embs_all[i, d_i]
                ep[i, e_i] = False
        with torch.cuda.amp.autocast(enabled=use_bf16, dtype=torch.bfloat16):
            f = model.set_encoder(h_seq, exp_pad_mask=ep)
            logits, _ = model.mech_head(f)
        return torch.softmax(logits.float(), dim=-1).cpu().numpy(), f.float(), h_seq

    eq_full = equiv_tiers["full"]["primary"]
    m2c_full = np.array(eq_full["mech_to_class"], dtype=np.int64)
    classes_full = eq_full["classes"]
    n_cls_full = len(classes_full)
    true_c_full = m2c_full[true_mids]
    full_designs = list(grammar.tiers["full"])
    is_nontriv_full = np.array([len(classes_full[c]) >= 2 for c in true_c_full], dtype=bool)

    eq_basic = equiv_tiers["basic"]["primary"]
    m2c_basic = np.array(eq_basic["mech_to_class"], dtype=np.int64)
    classes_basic = eq_basic["classes"]
    n_cls_basic = len(classes_basic)
    true_c_basic = m2c_basic[true_mids]

    mopts = grammar.mech_options
    mcids = grammar.mech_core_id
    m4m7_split_cids, c4c1_split_cids, any_xcore_split_cids = set(), set(), set()
    for cid, members in enumerate(classes_basic):
        if len(members) < 2:
            continue
        if len({m2c_full[m] for m in members}) <= 1:
            continue
        if len({(int(mopts[m, 3]), int(mopts[m, 6])) for m in members}) > 1:
            m4m7_split_cids.add(cid)
        cores_in = {int(mcids[m]) for m in members}
        if len(cores_in) > 1:
            any_xcore_split_cids.add(cid)
            if (3 in cores_in) and (0 in cores_in):
                c4c1_split_cids.add(cid)

    is_m4m7_split = np.array([c in m4m7_split_cids for c in true_c_basic], dtype=bool)
    is_c4c1_split = np.array([c in c4c1_split_cids for c in true_c_basic], dtype=bool)
    is_xcore_split = np.array([c in any_xcore_split_cids for c in true_c_basic], dtype=bool)

    # --- Part A: Cost-Normalized Xi_full ---
    pm_d0, f_d0, h_d0 = eval_var_hist([[0] for _ in range(N)])
    pc_d0 = aggregate_class_probs(pm_d0, m2c_full, n_cls_full)
    cands_0 = [[d for d in full_designs if d != 0 and costs[d] <= max_cost] for _ in range(N)]
    eq_0, ef_0 = _score_eig_designs_batched(
        model, grammar, sim, h_d0, f_d0, G_mat, pm_d0, pc_d0,
        catalog_pool, cands_0, m2c_full, n_cls_full, n_t, sigma_rel, n_mc=n_mc, device=device,
        rng=np.random.default_rng(seed + 900), crn_seed=seed + 900, case_ids=list(range(N)),
    )

    def run_cost_pol(mode: str):
        hist = [[0] for _ in range(N)]
        spent = np.zeros(N, dtype=np.int64)
        pc_by_c = np.zeros((N, max_cost + 1), dtype=np.float64)
        top1_by_c = np.zeros((N, max_cost + 1), dtype=np.float64)
        d9_ever = np.zeros(N, dtype=bool)
        d9_step1 = np.zeros(N, dtype=bool)
        n_run = np.zeros(N, dtype=np.int64)
        pm_curr, f_curr, h_curr, pc_curr = pm_d0.copy(), f_d0.clone(), h_d0.clone(), pc_d0.copy()
        pc_by_c[:, 0] = pc_curr[np.arange(N), true_c_full]
        top1_by_c[:, 0] = (np.argmax(pc_curr, axis=1) == true_c_full).astype(np.float64)

        for step in range(max_cost):
            active = [i for i in range(N) if spent[i] < max_cost]
            valid_pairs = [
                (i, [d for d in full_designs if (d not in hist[i]) and (costs[d] <= max_cost - spent[i])])
                for i in active
            ]
            valid_pairs = [(i, c) for i, c in valid_pairs if len(c) > 0]
            if not valid_pairs:
                break
            act_idx = [p[0] for p in valid_pairs]
            cands_use = [p[1] for p in valid_pairs]
            if step == 0:
                eq_l, ef_l = eq_0, ef_0
            else:
                # Pad variable-length histories cleanly for active cases if needed
                max_h = max(len(hist[i]) for i in act_idx)
                # Group by history length (only 1 or 2 lengths possible because D9=3, others=1!)
                eq_l, ef_l = [None] * len(act_idx), [None] * len(act_idx)
                by_len: Dict[int, List[int]] = {}
                for loc_i, i in enumerate(act_idx):
                    by_len.setdefault(len(hist[i]), []).append(loc_i)
                s_seed = seed + 900 + step * 1000
                for hlen, loc_list in by_len.items():
                    sub_i = [act_idx[l] for l in loc_list]
                    h_act = torch.stack([exp_embs_all[i, torch.tensor(hist[i], device=device)] for i in sub_i], dim=0)
                    eq_s, ef_s = _score_eig_designs_batched(
                        model, grammar, sim, h_act, f_curr[sub_i], G_mat, pm_curr[sub_i], pc_curr[sub_i],
                        catalog_pool, [cands_use[l] for l in loc_list], m2c_full, n_cls_full, n_t, sigma_rel,
                        n_mc=n_mc, device=device, rng=np.random.default_rng(s_seed), crn_seed=s_seed, case_ids=sub_i,
                    )
                    for k_l, l_idx in enumerate(loc_list):
                        eq_l[l_idx], ef_l[l_idx] = eq_s[k_l], ef_s[k_l]

            for loc_i, i in enumerate(act_idx):
                cl = cands_use[loc_i]
                cc = np.array([float(costs[d]) for d in cl])
                if mode == "Quotient_EIG":
                    sc = eq_l[loc_i]
                elif mode == "Quotient_EIG_per_cost":
                    sc = np.maximum(eq_l[loc_i], 0.0) / cc
                else:
                    sc = np.maximum(ef_l[loc_i], 0.0) / cc
                pick = cl[int(np.argmax(sc))]
                old_c, new_c = int(spent[i]), int(spent[i]) + int(costs[pick])
                for c_fill in range(old_c + 1, new_c):
                    pc_by_c[i, c_fill] = pc_by_c[i, old_c]
                    top1_by_c[i, c_fill] = top1_by_c[i, old_c]
                hist[i].append(pick)
                spent[i] = new_c
                n_run[i] += 1
                if pick == 9:
                    d9_ever[i] = True
                    if step == 0:
                        d9_step1[i] = True

            pm_curr, f_curr, h_curr = eval_var_hist(hist)
            pc_curr = aggregate_class_probs(pm_curr, m2c_full, n_cls_full)
            for i in act_idx:
                cc = int(spent[i])
                pc_by_c[i, cc] = pc_curr[i, true_c_full[i]]
                top1_by_c[i, cc] = float(np.argmax(pc_curr[i]) == true_c_full[i])
                for rc in range(cc + 1, max_cost + 1):
                    pc_by_c[i, rc] = pc_by_c[i, cc]
                    top1_by_c[i, rc] = top1_by_c[i, cc]

        c_auc = np.mean(pc_by_c, axis=1)
        return {
            "cost_auc_mean": float(np.mean(c_auc)),
            "cost_auc_nontriv_full": float(np.mean(c_auc[is_nontriv_full])),
            "pc_by_cost": [float(x) for x in np.mean(pc_by_c, axis=0)],
            "top1_by_cost": [float(x) for x in np.mean(top1_by_c, axis=0)],
            "final_pc_at_cost6": float(np.mean(pc_by_c[:, max_cost])),
            "final_top1_at_cost6": float(np.mean(top1_by_c[:, max_cost])),
            "final_top1_nontriv_full": float(np.mean(top1_by_c[is_nontriv_full, max_cost])),
            "mean_exps_executed": float(np.mean(n_run)),
            "d9_ever_rate_all": float(np.mean(d9_ever)),
            "d9_step1_rate_all": float(np.mean(d9_step1)),
            "d9_ever_rate_m4m7_splittable": float(np.mean(d9_ever[is_m4m7_split])) if np.any(is_m4m7_split) else 0.0,
            "d9_ever_rate_c4c1_splittable": float(np.mean(d9_ever[is_c4c1_split])) if np.any(is_c4c1_split) else 0.0,
            "d9_ever_rate_xi_full_nontriv": float(np.mean(d9_ever[is_nontriv_full])),
        }, {"cost_auc": c_auc, "final_top1": top1_by_c[:, max_cost], "d9_ever": d9_ever.astype(np.float64)}

    c_qu, raw_qu = run_cost_pol("Quotient_EIG")
    c_qn, raw_qn = run_cost_pol("Quotient_EIG_per_cost")
    c_fn, raw_fn = run_cost_pol("Fine_EIG_per_cost")
    cost_ci = {
        "Qnorm_minus_Qunw_cost_auc": paired_bootstrap_ci(raw_qn["cost_auc"] - raw_qu["cost_auc"]),
        "Qnorm_minus_Qunw_top1@cost6": paired_bootstrap_ci(raw_qn["final_top1"] - raw_qu["final_top1"]),
        "Qnorm_minus_Fnorm_cost_auc": paired_bootstrap_ci(raw_qn["cost_auc"] - raw_fn["cost_auc"]),
        "Qnorm_minus_Fnorm_top1@cost6": paired_bootstrap_ci(raw_qn["final_top1"] - raw_fn["final_top1"]),
        "Fnorm_minus_Qnorm_d9_rate_all": paired_bootstrap_ci(raw_fn["d9_ever"] - raw_qn["d9_ever"]),
        "Fnorm_minus_Qnorm_d9_rate_nontriv": paired_bootstrap_ci(raw_fn["d9_ever"][is_nontriv_full] - raw_qn["d9_ever"][is_nontriv_full]),
    }

    # --- Part B: 1-Shot Instrument Upgrade (Xi_basic -> {D7, D8, D9}) ---
    basic_designs = list(grammar.tiers["basic"])
    upg_cands = [d for d in full_designs if d not in basic_designs]  # [7, 8, 9]
    pm_b, f_b, h_b = eval_var_hist([basic_designs for _ in range(N)])
    pc_b = aggregate_class_probs(pm_b, m2c_basic, n_cls_basic)
    pc_f_before = aggregate_class_probs(pm_b, m2c_full, n_cls_full)

    is_nt_b = np.array([len(classes_basic[c]) >= 2 for c in true_c_basic], dtype=bool)
    is_conv90 = (np.max(pc_b, axis=1) >= 0.90) & is_nt_b
    is_conv80 = (np.max(pc_b, axis=1) >= 0.80) & is_nt_b

    eig_upg_q, _ = _score_eig_designs_batched(
        model, grammar, sim, h_b, f_b, G_mat, pm_b, pc_f_before,
        catalog_pool, [upg_cands for _ in range(N)], m2c_full, n_cls_full, n_t, sigma_rel, n_mc=n_mc, device=device,
        rng=np.random.default_rng(seed + 1234), crn_seed=seed + 1234, case_ids=list(range(N)),
    )
    upg_picks = [upg_cands[int(np.argmax(eig_upg_q[i]))] for i in range(N)]

    pm_after1, _, _ = eval_var_hist([basic_designs + [upg_picks[i]] for i in range(N)])
    pc_f_after1 = aggregate_class_probs(pm_after1, m2c_full, n_cls_full)
    pm_all10, _, _ = eval_var_hist([full_designs for _ in range(N)])
    pc_f_all10 = aggregate_class_probs(pm_all10, m2c_full, n_cls_full)

    def sum_upg(mask: np.ndarray) -> Dict:
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            return {"n": 0}
        pk = [upg_picks[i] for i in idxs]
        pb = pc_f_before[idxs, true_c_full[idxs]]
        p1 = pc_f_after1[idxs, true_c_full[idxs]]
        p3 = pc_f_all10[idxs, true_c_full[idxs]]
        t_b = (np.argmax(pc_f_before[idxs], axis=1) == true_c_full[idxs]).astype(np.float64)
        t_1 = (np.argmax(pc_f_after1[idxs], axis=1) == true_c_full[idxs]).astype(np.float64)
        t_3 = (np.argmax(pc_f_all10[idxs], axis=1) == true_c_full[idxs]).astype(np.float64)
        return {
            "n": int(len(idxs)),
            "rec_dist": {grammar.catalog[d].name: int(pk.count(d)) for d in upg_cands},
            "D9_top1_rec_rate": float(pk.count(9) / len(idxs)),
            "xi_full_true_class_prob_before": float(np.mean(pb)),
            "xi_full_true_class_prob_after_1_rec_exp": float(np.mean(p1)),
            "xi_full_true_class_prob_after_all_3_exps": float(np.mean(p3)),
            "mean_posterior_gain_1_exp": float(np.mean(p1 - pb)),
            "xi_full_top1_before": float(np.mean(t_b)),
            "xi_full_top1_after_1_rec_exp": float(np.mean(t_1)),
            "xi_full_top1_after_all_3_exps": float(np.mean(t_3)),
            "top1_gain_1_exp": float(np.mean(t_1 - t_b)),
        }

    upg_res = {
        "all_nontriv_basic": sum_upg(is_nt_b),
        "converged_80_nontriv_basic": sum_upg(is_conv80),
        "converged_90_nontriv_basic": sum_upg(is_conv90),
        "m4m7_family_all_nontriv": sum_upg(is_m4m7_split),
        "m4m7_family_converged_80": sum_upg(is_conv80 & is_m4m7_split),
        "m4m7_family_converged_90": sum_upg(is_conv90 & is_m4m7_split),
        "cross_core_C4_vs_C1_all": sum_upg(is_c4c1_split),
        "cross_core_C4_vs_C1_converged_80": sum_upg(is_conv80 & is_c4c1_split),
        "cross_core_C4_vs_C1_converged_90": sum_upg(is_conv90 & is_c4c1_split),
        "any_cross_core_splittable_all": sum_upg(is_xcore_split),
        "any_cross_core_splittable_converged_80": sum_upg(is_conv80 & is_xcore_split),
    }
    print(
        f"  [Cost&Upgrade] {time.perf_counter()-t0:.1f}s | Cost6 top1: Qn={c_qn['final_top1_at_cost6']:.3f} "
        f"vs Qu={c_qu['final_top1_at_cost6']:.3f} vs Fn={c_fn['final_top1_at_cost6']:.3f} | "
        f"Upg D9 hit: M4/M7={upg_res['m4m7_family_all_nontriv'].get('D9_top1_rec_rate',0):.3f}, "
        f"C4vC1={upg_res['cross_core_C4_vs_C1_all'].get('D9_top1_rec_rate',0):.3f}",
        flush=True,
    )
    return {
        "cost_normalized_Xi_full": {
            "max_cost": max_cost, "n_cases": N,
            "Quotient_EIG_unweighted": c_qu, "Quotient_EIG_per_cost": c_qn, "Fine_EIG_per_cost": c_fn,
            "paired_bootstrap_ci": cost_ci,
        },
        "instrument_upgrade_Xi_basic_to_full": upg_res,
    }


def run_seed_boed(
    grammar: ReactionGrammar,
    sim: ODESimulator,
    test_dir: Path,
    equiv_tiers: Dict[str, Dict],
    models_dir: Path,
    device: torch.device,
    seed: int,
    n_mc_disc: int = 512,
    n_mc_lhs: int = 256,
) -> Dict:
    print(f"\n================ E3 Quotient BOED + C_macro (Seed {seed}, N={grammar.n_mechs}) ================", flush=True)
    ckpt_p = models_dir / f"quomech_s{seed}" / "best.pt"
    ckpt = torch.load(ckpt_p, map_location=device)
    model = build_model(grammar, head_type="factorized", arch=ckpt.get("arch", "p2_256")).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    t_grid = np.load(test_dir / "t_grid.npy")
    all_mids = np.load(test_dir / "mech_ids.npy")
    all_thetas = np.load(test_dir / "thetas.npy")
    all_trajs = np.load(test_dir / "trajs.npy", mmap_mode="r")

    # 1 test case per mechanism (index 5 of each mechanism's 100 test samples -> 527 cases spanning all 4 cores)
    sel_idx = np.array([np.where(all_mids == m)[0][5] for m in range(grammar.n_mechs)], dtype=np.int64)
    true_mids = all_mids[sel_idx].astype(np.int64)
    true_thetas = all_thetas[sel_idx].astype(np.float64)
    trajs_sel = np.array(all_trajs[sel_idx])

    rng_obs = np.random.default_rng(202615 + seed)
    exp_embs_all, G_mat, probs_m_d0, f_d0, h_d0 = precompute_catalog_embs(
        model, grammar, t_grid, trajs_sel, device, sigma_rel=0.01, rng_obs=rng_obs,
    )

    # 1. 3-tier discrete catalog BOED
    disc_res = {}
    for t_i, tier in enumerate(["no_P", "basic", "full"]):
        eq = equiv_tiers[tier]["primary"]
        disc_res[tier] = run_discrete_partition_boed(
            model, grammar, sim, exp_embs_all, G_mat, probs_m_d0, f_d0, h_d0, true_mids,
            tier_designs=list(grammar.tiers[tier]),
            m2c=np.array(eq["mech_to_class"], dtype=np.int64),
            classes=eq["classes"],
            policies=["Random", "Textbook_RPKA_VTNA", "Static_EIG", "Fine_EIG", "Quotient_EIG"],
            device=device, tier_name=tier, n_mc=n_mc_disc,
            seed=202615 + seed * 100 + t_i * 10, verify_crn=(t_i == 0),
        )

    # 2. Zero-retrain Coarse Task Quotient C_macro on Xi_full
    c_macro = build_c_macro_partition(grammar)
    print(f"\n  [C_macro] Partition: {c_macro['n_classes']} macro classes (mean size={c_macro['mean_class_size']:.1f}, max={c_macro['max_class_size']})", flush=True)
    cmacro_xifull = run_discrete_partition_boed(
        model, grammar, sim, exp_embs_all, G_mat, probs_m_d0, f_d0, h_d0, true_mids,
        tier_designs=list(grammar.tiers["full"]),
        m2c=c_macro["mech_to_class"], classes=c_macro["classes"],
        policies=["Random", "Fine_EIG", "Quotient_EIG"],
        device=device, tier_name="C_macro_Xi_full", n_mc=n_mc_disc,
        seed=202688 + seed * 100, horizons_override=[2, 3, 4, 6],
    )

    # 3. Continuous LHS-128 BOED (X_basic_ref + C_macro reusing pre-encoded LHS-128 pool)
    lhs_xbasic, lhs_cmacro = run_continuous_lhs128_suite(
        model, grammar, sim, t_grid, trajs_sel, true_mids, true_thetas,
        equiv_xref=equiv_tiers["X_basic_ref"], equiv_basic=equiv_tiers["basic"],
        c_macro_part=c_macro, device=device, n_lhs=128, budget=6, n_mc=n_mc_lhs,
        seed=202633 + seed * 100,
    )

    # 4. Cost-normalized Xi_full & 1-Shot Instrument Upgrade
    cost_upg = run_cost_and_upgrade_suite(
        model, grammar, sim, exp_embs_all, G_mat, true_mids, equiv_tiers, device,
        n_mc=n_mc_disc, max_cost=6, seed=202644 + seed * 100,
    )

    return {
        "seed": seed,
        "n_cases": grammar.n_mechs,
        "discrete_3tier": disc_res,
        "continuous_lhs128": lhs_xbasic,
        "cost_and_upgrade": cost_upg,
        "c_macro_task_quotient": {
            "n_classes": c_macro["n_classes"],
            "mean_class_size": c_macro["mean_class_size"],
            "max_class_size": c_macro["max_class_size"],
            "Xi_full": cmacro_xifull,
            "LHS_128": lhs_cmacro,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="42")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--n-threads", type=int, default=64)
    ap.add_argument("--n-mc-disc", type=int, default=512)
    ap.add_argument("--n-mc-lhs", type=int, default=256)
    ap.add_argument("--data-dir", type=str, default="data/pool_multicore_527")
    ap.add_argument("--equiv-dir", type=str, default="results/phase2/equiv")
    ap.add_argument("--models-dir", type=str, default="results/phase2/models")
    ap.add_argument("--out-dir", type=str, default="results/phase2")
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    g = build_multicore_grammar()
    sim = ODESimulator(g, rtol=1e-5, atol=1e-9, n_threads=args.n_threads)
    equiv_tiers = load_all_equiv(Path(args.equiv_dir))
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "stage3_e3_boed.json"
    per_seed = json.loads(out_json.read_text()).get("per_seed", {}) if out_json.exists() else {}

    for s in seeds:
        per_seed[str(s)] = run_seed_boed(
            g, sim, Path(args.data_dir) / "test", equiv_tiers, Path(args.models_dir),
            device, seed=s, n_mc_disc=args.n_mc_disc, n_mc_lhs=args.n_mc_lhs,
        )
        out_json.write_text(json.dumps({"seeds_completed": sorted([int(k) for k in per_seed]), "per_seed": per_seed}, indent=2))

    print(f"\n[Stage 3 E3 BOED + C_macro DONE] Saved to {out_json}", flush=True)


if __name__ == "__main__":
    main()
