"""E3 Closed-Loop Bayesian Experimental Design (Section 7.4 & 8.2):
  1. Discrete 3-tier catalog BOED (Ξ_no-P, Ξ_basic, Ξ_full) comparing:
     - Random, Textbook fixed RPKA/VTNA, Static EIG,
     - Fine-grained EIG (reporting BOTH unified class stopping AND fine-grained stopping),
     - Quotient EIG (QuoMech),
     plus spontaneous RPKA/VTNA expert diagnostic mapping statistics.
  2. Continuous design space X BOED (96-point Latin Hypercube candidate pool with load-adapted
     variable windows t_end(xi) in [0.4, 4.0]), verifying spontaneous spatial clustering of
     optimal continuous xi* into VTNA (variable cat loading) and RPKA (pre-added P / same-excess) zones.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
from scipy.stats import qmc
import torch

from quomech.data_gen import build_design_feature_vec, get_observable_channel_indices
from quomech.evaluate import aggregate_class_probs, interp_traj_to_times
from quomech.grammar import DesignSpec, ReactionGrammar
from quomech.models import QuoMechNet
from quomech.simulator import ODESimulator


def _build_single_exp_tensors(
    grammar: ReactionGrammar,
    des: DesignSpec,
    t_end: float,
    y_noisy: np.ndarray,  # (B, n_t, n_ch) raw noisy concentrations
    t_grid: np.ndarray,   # (n_t,)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pack B observations of ONE design (discrete or continuous) into ExperimentEncoder tensors."""
    B, n_t, n_ch = y_noisy.shape
    obs_ch = get_observable_channel_indices(grammar)
    ch_is_cat = np.array([grammar.species[s].is_catalyst for s in obs_ch], dtype=bool)

    ch_mask = np.zeros(n_ch, dtype=np.float32)
    for c_i, sp in enumerate(obs_ch):
        if sp in des.obs_species:
            ch_mask[c_i] = 1.0

    df1 = build_design_feature_vec(
        grammar, des.A0, des.B0, des.P0, des.cat_mult, float(des.a_pre), t_end, obs_ch, ch_mask
    )
    df = np.broadcast_to(df1[None, :], (B, len(df1))).copy()
    tm = np.broadcast_to(t_grid.astype(np.float32)[None, :], (B, n_t)).copy()

    cat0 = des.cat_mult * grammar.c0_base
    norm_scale = np.where(ch_is_cat, cat0, 1.0).astype(np.float32)
    ov = (y_noisy.astype(np.float32) / norm_scale[None, None, :]) * ch_mask[None, None, :]
    om = np.broadcast_to(ch_mask[None, None, :], (B, n_t, n_ch)).copy()
    tp = np.zeros((B, n_t), dtype=bool)
    return df, tm, ov, om, tp


@torch.no_grad()
def _score_eig_designs_batched(
    model: QuoMechNet,
    grammar: ReactionGrammar,
    sim: ODESimulator,
    hist_embs: torch.Tensor,      # (B_act, E_hist, d_exp) on GPU
    f_curr: torch.Tensor,         # (B_act, d_set) on GPU
    G_mat: torch.Tensor,          # (M, d_g) on GPU
    probs_m_curr: np.ndarray,     # (B_act, M)
    probs_c_curr: np.ndarray,     # (B_act, n_classes)
    design_pool: Sequence[Tuple[DesignSpec, float]],  # pool of (DesignSpec, t_end)
    cand_lists: List[List[int]],  # len B_act: indices into design_pool
    mech_to_class: np.ndarray,
    n_classes: int,
    n_t: int,
    sigma_rel: float,
    n_mc: int,
    device: torch.device,
    rng: np.random.Generator,
    gpu_batch: int = 4096,
    crn_seed: Optional[int] = None,
    case_ids: Optional[Sequence[int]] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Compute Barber-Agakov Quotient EIG and Fine-grained EIG simultaneously for B_act cases.

    When `crn_seed` is provided, enforces strict Common Random Numbers across policies for:
      1. Mechanism draws m ~ q(m | D) (seeded per case_id),
      2. NSF parameter draws theta ~ q(theta | m, D) (seeded via forked torch RNG),
      3. Candidate observation noise eps ~ N(0, 1) (seeded deterministically per design & case_id).
    """
    B_act = len(cand_lists)
    M = grammar.n_mechs
    obs_ch = get_observable_channel_indices(grammar)
    n_ch = len(obs_ch)
    ch_is_cat = np.array([grammar.species[s].is_catalyst for s in obs_ch], dtype=bool)
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    cids = list(range(B_act)) if case_ids is None else list(case_ids)
    max_cid = max(cids) if cids else 0

    sampled_m = np.empty((B_act, n_mc), dtype=np.int32)
    for b in range(B_act):
        p = probs_m_curr[b]
        p = p / np.sum(p)
        r_b = np.random.default_rng(crn_seed + cids[b] * 1009) if crn_seed is not None else rng
        sampled_m[b] = r_b.choice(M, size=n_mc, replace=True, p=p)

    f_rep = f_curr.unsqueeze(1).expand(B_act, n_mc, -1).reshape(B_act * n_mc, -1)
    m_flat_t = torch.from_numpy(sampled_m.ravel()).long().to(device)
    th_chunks = []
    fork_ctx = torch.random.fork_rng(devices=[device] if device.type == "cuda" else [])
    with fork_ctx:
        if crn_seed is not None:
            torch.manual_seed(crn_seed + 99991)
            if device.type == "cuda":
                torch.cuda.manual_seed(crn_seed + 99991)
        for st in range(0, B_act * n_mc, 4096):
            ed = min(B_act * n_mc, st + 4096)
            th_chunks.append(model.sample_theta(f_rep[st:ed], G_mat, m_flat_t[st:ed], num_samples=1).cpu().numpy())
    sampled_th = np.concatenate(th_chunks, axis=0).astype(np.float64).reshape(B_act, n_mc, grammar.n_params)
    lb = grammar.mech_prior_mean - 4.5 * grammar.mech_prior_std
    ub = grammar.mech_prior_mean + 4.5 * grammar.mech_prior_std
    m_flat = sampled_m.ravel()
    th_flat = np.clip(sampled_th.reshape(B_act * n_mc, grammar.n_params), lb[m_flat], ub[m_flat])
    sampled_th = th_flat.reshape(B_act, n_mc, grammar.n_params)

    cases_by_design: Dict[int, List[int]] = {}
    for b, cands in enumerate(cand_lists):
        for d in cands:
            cases_by_design.setdefault(d, []).append(b)

    eig_q_map: Dict[Tuple[int, int], float] = {}
    eig_f_map: Dict[Tuple[int, int], float] = {}
    E_hist = hist_embs.shape[1]
    m2c_t = torch.from_numpy(mech_to_class).long().to(device)

    for d_idx in sorted(cases_by_design.keys()):
        b_list = cases_by_design[d_idx]
        des, t_end = design_pool[d_idx]
        t_eval = np.linspace(0.0, t_end, n_t)
        n_b = len(b_list)
        b_arr = np.array(b_list, dtype=np.int64)
        m_sub = sampled_m[b_arr].ravel()
        th_sub = sampled_th[b_arr].reshape(n_b * n_mc, -1)

        y0_table = np.array([grammar.initial_state(m, des) for m in range(M)], dtype=np.float64)
        y0_sub = y0_table[m_sub]

        y_sim, st_sim, _, _ = sim.simulate_batch(
            m_sub, th_sub, y0_sub, t_eval, compute_flux=False, rtol=3e-5, atol=3e-9, max_steps=10000
        )
        y_obs = y_sim[:, :, obs_ch]
        cat0 = des.cat_mult * grammar.c0_base
        noise_scale = np.where(ch_is_cat, cat0, des.A0)
        if crn_seed is not None:
            rng_d = np.random.default_rng(crn_seed + d_idx * 100003 + 7)
            eps_all = rng_d.standard_normal((max_cid + 1, n_mc, n_t, n_ch), dtype=np.float32)
            cid_arr = np.array([cids[b] for b in b_list], dtype=np.int64)
            eps = eps_all[cid_arr].reshape(n_b * n_mc, n_t, n_ch)
            y_noisy = y_obs + eps * (sigma_rel * noise_scale[None, None, :])
        else:
            y_noisy = y_obs + rng.normal(0.0, 1.0, size=y_obs.shape) * (sigma_rel * noise_scale[None, None, :])

        df_np, tm_np, ov_np, om_np, tp_np = _build_single_exp_tensors(grammar, des, t_end, y_noisy, t_eval)

        tot_n = n_b * n_mc
        pm_prior = probs_m_curr[np.repeat(b_arr, n_mc), m_sub]
        c_sub = mech_to_class[m_sub]
        pc_prior = probs_c_curr[np.repeat(b_arr, n_mc), c_sub]
        log_pm_prior = np.log(np.maximum(pm_prior, 1e-12))
        log_pc_prior = np.log(np.maximum(pc_prior, 1e-12))

        log_pm_post = np.empty(tot_n, dtype=np.float64)
        log_pc_post = np.empty(tot_n, dtype=np.float64)
        b_idx_rep_t = torch.from_numpy(np.repeat(b_arr, n_mc)).long().to(device)
        m_sub_t = torch.from_numpy(m_sub).long().to(device)
        c_sub_t = torch.from_numpy(c_sub).long().to(device)

        for st in range(0, tot_n, gpu_batch):
            ed = min(tot_n, st + gpu_batch)
            bs = ed - st
            with torch.cuda.amp.autocast(enabled=use_bf16, dtype=torch.bfloat16):
                new_emb = model.exp_encoder(
                    torch.from_numpy(df_np[st:ed]).to(device),
                    torch.from_numpy(tm_np[st:ed]).to(device),
                    torch.from_numpy(ov_np[st:ed]).to(device),
                    torch.from_numpy(om_np[st:ed]).to(device),
                    torch.from_numpy(tp_np[st:ed]).to(device),
                )
                h_prev = hist_embs[b_idx_rep_t[st:ed]]
                h_cat = torch.cat([h_prev, new_emb.unsqueeze(1).to(h_prev.dtype)], dim=1)
                ep_zero = torch.zeros((bs, E_hist + 1), dtype=torch.bool, device=device)
                probs_new = model.probs_from_exp_embs(h_cat, ep_zero)

            ar = torch.arange(bs, device=device)
            pm_new = probs_new[ar, m_sub_t[st:ed]].clamp(min=1e-12)
            pc_mat = torch.zeros((bs, n_classes), device=device, dtype=torch.float32)
            pc_mat.scatter_add_(1, m2c_t.unsqueeze(0).expand(bs, -1), probs_new)
            pc_new = pc_mat[ar, c_sub_t[st:ed]].clamp(min=1e-12)

            log_pm_post[st:ed] = torch.log(pm_new).cpu().numpy()
            log_pc_post[st:ed] = torch.log(pc_new).cpu().numpy()

        ok_2d = (st_sim == 0).reshape(n_b, n_mc)
        mi_f_2d = (log_pm_post - log_pm_prior).reshape(n_b, n_mc)
        mi_q_2d = (log_pc_post - log_pc_prior).reshape(n_b, n_mc)

        for local_i, b in enumerate(b_list):
            ok_m = ok_2d[local_i]
            eig_q_map[(b, d_idx)] = float(np.mean(mi_q_2d[local_i, ok_m])) if np.any(ok_m) else -1e9
            eig_f_map[(b, d_idx)] = float(np.mean(mi_f_2d[local_i, ok_m])) if np.any(ok_m) else -1e9

    eig_q_out = [np.array([eig_q_map[(b, d)] for d in cands]) for b, cands in enumerate(cand_lists)]
    eig_f_out = [np.array([eig_f_map[(b, d)] for d in cands]) for b, cands in enumerate(cand_lists)]
    return eig_q_out, eig_f_out


@torch.no_grad()
def run_boed_benchmark(
    model: QuoMechNet,
    grammar: ReactionGrammar,
    sim: ODESimulator,
    test_dir: Path,
    equiv_tiers: Dict[str, Dict],
    device: torch.device,
    n_cases: int = 151,
    budget: int = 4,
    n_mc: int = 256,
    sigma_rel: float = 0.01,
    seed: int = 202604,
) -> Dict:
    """Full E3 discrete 3-tier catalog BOED benchmark across Ξ_no-P, Ξ_basic, Ξ_full."""
    model.eval()
    rng = np.random.default_rng(seed)
    obs_ch = get_observable_channel_indices(grammar)
    ch_is_cat = np.array([grammar.species[s].is_catalyst for s in obs_ch], dtype=bool)
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    t_grid = np.load(test_dir / "t_grid.npy")
    all_mids = np.load(test_dir / "mech_ids.npy")
    all_trajs = np.load(test_dir / "trajs.npy")
    sel_idx = np.array([np.where(all_mids == m)[0][5] for m in range(grammar.n_mechs)], dtype=np.int64)
    if n_cases < len(sel_idx):
        sel_idx = sel_idx[:n_cases]
    N = len(sel_idx)
    true_mids = all_mids[sel_idx].astype(np.int64)
    trajs_sel = all_trajs[sel_idx]

    n_t = 20
    t_eval = np.linspace(0.0, 1.0, n_t)
    catalog_pool = [(d, 1.0) for d in grammar.catalog]

    exp_embs_all = torch.empty((N, 10, model.exp_encoder.d_model), device=device, dtype=torch.float32)
    for d_idx in range(10):
        des = grammar.catalog[d_idx]
        clean = np.stack(
            [interp_traj_to_times(t_grid, trajs_sel[i, d_idx].astype(np.float64), t_eval) for i in range(N)],
            axis=0,
        )
        cat0 = des.cat_mult * grammar.c0_base
        noise_scale = np.where(ch_is_cat, cat0, des.A0)
        y_n = clean + rng.normal(0.0, 1.0, size=clean.shape) * (sigma_rel * noise_scale[None, None, :])

        df_np, tm_np, ov_np, om_np, tp_np = _build_single_exp_tensors(grammar, des, 1.0, y_n, t_eval)
        with torch.cuda.amp.autocast(enabled=use_bf16, dtype=torch.bfloat16):
            emb = model.exp_encoder(
                torch.from_numpy(df_np).to(device),
                torch.from_numpy(tm_np).to(device),
                torch.from_numpy(ov_np).to(device),
                torch.from_numpy(om_np).to(device),
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

    init_hist = [[0] for _ in range(N)]
    probs_m_d0, f_d0, h_d0 = eval_state(init_hist)

    all_tier_results = {}
    rpka_vtna_stats = {}

    for tier in ["no_P", "basic", "full"]:
        t_tier = time.perf_counter()
        tier_designs = list(grammar.tiers[tier])
        eq = equiv_tiers[tier]["primary"]
        m2c = np.array(eq["mech_to_class"], dtype=np.int64)
        classes = eq["classes"]
        n_cls = len(classes)
        true_c = m2c[true_mids]
        nontriv_mask = np.array([len(classes[c]) >= 2 for c in true_c], dtype=bool)
        probs_c_d0 = aggregate_class_probs(probs_m_d0, m2c, n_cls)

        cand_d0 = [[d for d in tier_designs if d != 0] for _ in range(N)]
        eig_q_d0, eig_f_d0 = _score_eig_designs_batched(
            model, grammar, sim, h_d0, f_d0, G_mat, probs_m_d0, probs_c_d0,
            catalog_pool, cand_d0, m2c, n_cls, n_t, sigma_rel, n_mc=n_mc, device=device, rng=rng,
        )
        eig_q_d0_mat = np.stack(eig_q_d0, axis=0)
        eig_f_d0_mat = np.stack(eig_f_d0, axis=0)
        static_order_idx = np.argsort(np.mean(eig_q_d0_mat, axis=0))[::-1][:budget]
        static_seq = [cand_d0[0][i] for i in static_order_idx]
        textbook_seq = [2, 5, 3, 4] if tier == "no_P" else [2, 5, 6, 3]

        step1_q_choices = [cand_d0[i][int(np.argmax(eig_q_d0_mat[i]))] for i in range(N)]
        step1_f_choices = [cand_d0[i][int(np.argmax(eig_f_d0_mat[i]))] for i in range(N)]
        step1_disagree = float(np.mean([a != b for a, b in zip(step1_q_choices, step1_f_choices)]))
        step1_disagree_nontriv = float(
            np.mean([step1_q_choices[i] != step1_f_choices[i] for i in range(N) if nontriv_mask[i]])
        ) if np.any(nontriv_mask) else 0.0

        rpka_hits, rpka_tot = 0, 0
        vtna_hits, vtna_tot = 0, 0
        subA_hits, subA_tot = 0, 0
        act_hits, act_tot = 0, 0
        for i in range(N):
            if probs_c_d0[i].max() >= 0.90:
                continue
            top2_c = np.argsort(probs_c_d0[i])[-2:][::-1]
            m_a = max(classes[top2_c[0]], key=lambda m: probs_m_d0[i, m])
            m_b = max(classes[top2_c[1]], key=lambda m: probs_m_d0[i, m])
            opt_a, opt_b = grammar.mech_options[m_a], grammar.mech_options[m_b]
            ch = step1_q_choices[i]
            m2_diff = (opt_a[1] > 0) != (opt_b[1] > 0)
            p_diff = ((opt_a[3] > 0) or (opt_a[6] > 0)) != ((opt_b[3] > 0) or (opt_b[6] > 0))
            if m2_diff and p_diff:
                rpka_tot += 1
                if ch in (5, 6, 7):
                    rpka_hits += 1
            cat_ord_a = (opt_a[4] > 0, opt_a[7] > 0, opt_a[1] == 2, opt_a[5] > 0)
            cat_ord_b = (opt_b[4] > 0, opt_b[7] > 0, opt_b[1] == 2, opt_b[5] > 0)
            if cat_ord_a != cat_ord_b:
                vtna_tot += 1
                if ch in (1, 2):
                    vtna_hits += 1
            if opt_a[2] != opt_b[2]:
                subA_tot += 1
                if ch in (3, 4):
                    subA_hits += 1
            if opt_a[0] != opt_b[0] and tier == "full":
                act_tot += 1
                if ch == 8:
                    act_hits += 1

        rpka_vtna_stats[tier] = {
            "step1_choice_dist": {
                grammar.catalog[d].name: int(sum(1 for x in step1_q_choices if x == d))
                for d in tier_designs if d != 0
            },
            "step1_disagree_all": step1_disagree,
            "step1_disagree_nontriv": step1_disagree_nontriv,
            "RPKA_M2_vs_M4M7_D5D6D7": {"hits": rpka_hits, "total": rpka_tot, "rate": rpka_hits / max(rpka_tot, 1)},
            "VTNA_cat_order_D1D2": {"hits": vtna_hits, "total": vtna_tot, "rate": vtna_hits / max(vtna_tot, 1)},
            "SubA_M3_D3D4": {"hits": subA_hits, "total": subA_tot, "rate": subA_hits / max(subA_tot, 1)},
            "Activation_M1_D8": {"hits": act_hits, "total": act_tot, "rate": act_hits / max(act_tot, 1)},
        }

        def run_strategy(strat_name: str) -> Dict:
            hist = [[0] for _ in range(N)]
            pc_traj = np.zeros((N, budget + 1), dtype=np.float64)
            pm_traj = np.zeros((N, budget + 1), dtype=np.float64)
            max_pc_traj = np.zeros((N, budget + 1), dtype=np.float64)
            max_pm_traj = np.zeros((N, budget + 1), dtype=np.float64)
            top1_traj = np.zeros((N, budget + 1), dtype=np.float64)

            pm_curr = probs_m_d0.copy()
            pc_curr = probs_c_d0.copy()
            f_curr = f_d0
            h_curr = h_d0

            pc_traj[:, 0] = pc_curr[np.arange(N), true_c]
            pm_traj[:, 0] = pm_curr[np.arange(N), true_mids]
            max_pc_traj[:, 0] = np.max(pc_curr, axis=1)
            max_pm_traj[:, 0] = np.max(pm_curr, axis=1)
            top1_traj[:, 0] = (np.argmax(pc_curr, axis=1) == true_c).astype(np.float64)

            for step in range(budget):
                if strat_name == "Random":
                    for i in range(N):
                        cands = [d for d in tier_designs if d not in hist[i]]
                        hist[i].append(int(rng.choice(cands)))
                elif strat_name == "Textbook_RPKA_VTNA":
                    for i in range(N):
                        hist[i].append(textbook_seq[step])
                elif strat_name == "Static_EIG":
                    for i in range(N):
                        hist[i].append(static_seq[step])
                elif strat_name in ("Quotient_EIG", "Fine_EIG"):
                    if step == 0:
                        choices = step1_q_choices if strat_name == "Quotient_EIG" else step1_f_choices
                        for i in range(N):
                            hist[i].append(choices[i])
                    else:
                        cands_all = [[d for d in tier_designs if d not in hist[i]] for i in range(N)]
                        eq_list, ef_list = _score_eig_designs_batched(
                            model, grammar, sim, h_curr, f_curr, G_mat, pm_curr, pc_curr,
                            catalog_pool, cands_all, m2c, n_cls, n_t, sigma_rel, n_mc=n_mc, device=device, rng=rng,
                        )
                        scores_use = eq_list if strat_name == "Quotient_EIG" else ef_list
                        for i in range(N):
                            hist[i].append(cands_all[i][int(np.argmax(scores_use[i]))])

                pm_curr, f_curr, h_curr = eval_state(hist)
                pc_curr = aggregate_class_probs(pm_curr, m2c, n_cls)
                pc_traj[:, step + 1] = pc_curr[np.arange(N), true_c]
                pm_traj[:, step + 1] = pm_curr[np.arange(N), true_mids]
                max_pc_traj[:, step + 1] = np.max(pc_curr, axis=1)
                max_pm_traj[:, step + 1] = np.max(pm_curr, axis=1)
                top1_traj[:, step + 1] = (np.argmax(pc_curr, axis=1) == true_c).astype(np.float64)

            def summarize_subset(mask: np.ndarray) -> Dict:
                sub_pc = pc_traj[mask]
                sub_max_pc = max_pc_traj[mask]
                sub_max_pm = max_pm_traj[mask]
                sub_top1 = top1_traj[mask]
                n_s = len(sub_pc)
                if n_s == 0:
                    return {}
                t_stop_cls = np.full(n_s, budget, dtype=np.int64)
                t_stop_fine = np.full(n_s, budget, dtype=np.int64)
                t_true90 = np.full(n_s, budget + 1, dtype=np.int64)
                acc_at_stop_cls = np.zeros(n_s, dtype=np.float64)
                acc_at_stop_fine = np.zeros(n_s, dtype=np.float64)
                for i in range(n_s):
                    for t_i in range(budget + 1):
                        if sub_max_pc[i, t_i] >= 0.90:
                            t_stop_cls[i] = t_i
                            break
                    for t_i in range(budget + 1):
                        if sub_max_pm[i, t_i] >= 0.90:
                            t_stop_fine[i] = t_i
                            break
                    for t_i in range(budget + 1):
                        if sub_pc[i, t_i] >= 0.90:
                            t_true90[i] = t_i
                            break
                    acc_at_stop_cls[i] = sub_top1[i, t_stop_cls[i]]
                    acc_at_stop_fine[i] = sub_top1[i, t_stop_fine[i]]
                return {
                    "n": int(n_s),
                    "auc_true_class_prob": float(np.mean(sub_pc)),
                    "mean_pc_by_step": [float(x) for x in np.mean(sub_pc, axis=0)],
                    "top1_by_step": [float(x) for x in np.mean(sub_top1, axis=0)],
                    "reach_true90_rate": float(np.mean(t_true90 <= budget)),
                    "mean_exps_to_true90_capped": float(np.mean(np.minimum(t_true90, budget))),
                    "unified_class_stop_mean_exps": float(np.mean(t_stop_cls)),
                    "unified_class_stop_top1": float(np.mean(acc_at_stop_cls)),
                    "fine_stop_mean_exps": float(np.mean(t_stop_fine)),
                    "fine_stop_top1": float(np.mean(acc_at_stop_fine)),
                    "wasted_exps_fine_minus_class_stop": float(np.mean(np.maximum(0, t_stop_fine - t_stop_cls))),
                }

            return {
                "all_cases": summarize_subset(np.ones(N, dtype=bool)),
                "nontrivial_class_cases": summarize_subset(nontriv_mask),
                "singleton_class_cases": summarize_subset(~nontriv_mask),
            }

        tier_res = {
            "n_classes": n_cls,
            "n_nontrivial_cases": int(np.sum(nontriv_mask)),
            "static_seq": [grammar.catalog[d].name for d in static_seq],
            "textbook_seq": [grammar.catalog[d].name for d in textbook_seq],
            "strategies": {},
        }
        for sname in ["Random", "Textbook_RPKA_VTNA", "Static_EIG", "Fine_EIG", "Quotient_EIG"]:
            tier_res["strategies"][sname] = run_strategy(sname)
        tier_res["elapsed_s"] = time.perf_counter() - t_tier
        all_tier_results[tier] = tier_res

        print(f"\n[BOED:{tier}] Done in {tier_res['elapsed_s']:.1f}s (n_nontriv={tier_res['n_nontrivial_cases']}/{N}):")
        for sname, sres in tier_res["strategies"].items():
            a = sres["all_cases"]
            nt = sres["nontrivial_class_cases"]
            nt_str = (
                f"nontriv AUC={nt['auc_true_class_prob']:.3f}, clsStop={nt['unified_class_stop_mean_exps']:.2f}, "
                f"fineStop={nt['fine_stop_mean_exps']:.2f} (waste={nt['wasted_exps_fine_minus_class_stop']:.2f})"
                if nt else "n/a"
            )
            print(
                f"  {sname:20s} | ALL AUC={a['auc_true_class_prob']:.3f} top1@4={a['top1_by_step'][-1]:.3f} "
                f"true90={a['reach_true90_rate']:.3f} clsStop={a['unified_class_stop_mean_exps']:.2f} | {nt_str}"
            )

    return {"tiers": all_tier_results, "rpka_vtna_emergence": rpka_vtna_stats}


@torch.no_grad()
def run_continuous_lhs_boed(
    model: QuoMechNet,
    grammar: ReactionGrammar,
    sim: ODESimulator,
    test_dir: Path,
    equiv_basic: Dict,
    device: torch.device,
    n_cases: int = 80,
    n_lhs: int = 64,
    budget: int = 4,
    n_mc: int = 128,
    sigma_rel: float = 0.01,
    seed: int = 202605,
) -> Dict:
    """Section 8.2 continuous design space X BOED on a 64-point Latin Hypercube candidate pool.

    Tests:
      1. Closed-loop Quotient EIG vs Random Continuous on 80 test cases (from D0, budget=4).
      2. Spontaneous spatial clustering of optimal continuous xi* at step 1 (from D0):
         - Catalyst-order ambiguity -> clusters into VTNA extreme catalyst loading (<=0.55x or >=1.8x)
         - M2 vs M4/M7 ambiguity -> clusters into RPKA pre-added P (P0>=0.2) and/or reduced A0<=0.8
         - M3 ambiguity -> clusters into high [A]_0 >= 1.4
    """
    t0 = time.perf_counter()
    model.eval()
    rng = np.random.default_rng(seed)
    obs_ch = get_observable_channel_indices(grammar)
    ch_is_cat = np.array([grammar.species[s].is_catalyst for s in obs_ch], dtype=bool)
    sA, sP = grammar.species_idx["A"], grammar.species_idx["P"]
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    # Build 64-point Latin Hypercube over continuous X: (A0, B0, P0, log10_cat_mult) + load-adapted t_end
    sampler = qmc.LatinHypercube(d=4, seed=seed)
    u = sampler.random(n=n_lhs)
    A0_arr = 0.25 + u[:, 0] * (2.0 - 0.25)
    B0_arr = 0.25 + u[:, 1] * (3.0 - 0.25)
    # Half at P0=0, half spanning [0.1, 1.0]
    P0_arr = np.where(u[:, 2] < 0.5, 0.0, 0.1 + ((u[:, 2] - 0.5) / 0.5) * 0.9)
    cat_arr = 10.0 ** (np.log10(0.25) + u[:, 3] * (np.log10(4.0) - np.log10(0.25)))
    tend_arr = np.clip(1.0 / cat_arr, 0.4, 4.0)

    lhs_pool: List[Tuple[DesignSpec, float]] = [
        (
            DesignSpec(
                f"LHS_{k:02d}", float(A0_arr[k]), float(B0_arr[k]), float(P0_arr[k]),
                float(cat_arr[k]), 0, (sA, sP), 1.0, (),
            ),
            float(tend_arr[k]),
        )
        for k in range(n_lhs)
    ]

    # Select n_cases diverse test mechanisms (from test split index 8)
    t_grid = np.load(test_dir / "t_grid.npy")
    all_mids = np.load(test_dir / "mech_ids.npy")
    all_thetas = np.load(test_dir / "thetas.npy")
    all_trajs = np.load(test_dir / "trajs.npy")
    sel_mechs = np.linspace(0, grammar.n_mechs - 1, n_cases, dtype=int)
    sel_idx = np.array([np.where(all_mids == m)[0][8] for m in sel_mechs], dtype=np.int64)
    true_mids = all_mids[sel_idx].astype(np.int64)
    true_thetas = all_thetas[sel_idx].astype(np.float64)

    eq = equiv_basic["primary"]
    m2c = np.array(eq["mech_to_class"], dtype=np.int64)
    classes = eq["classes"]
    n_cls = len(classes)
    true_c = m2c[true_mids]
    n_t = 20

    # Initial D0 observation (t_end = 1.0)
    t_eval_1 = np.linspace(0.0, 1.0, n_t)
    d0 = grammar.catalog[0]
    clean_d0 = np.stack(
        [interp_traj_to_times(t_grid, all_trajs[sel_idx[i], 0].astype(np.float64), t_eval_1) for i in range(n_cases)],
        axis=0,
    )
    noise_sc_d0 = np.where(ch_is_cat, d0.cat_mult * grammar.c0_base, d0.A0)
    y_d0 = clean_d0 + rng.normal(0.0, 1.0, size=clean_d0.shape) * (sigma_rel * noise_sc_d0[None, None, :])
    df0, tm0, ov0, om0, tp0 = _build_single_exp_tensors(grammar, d0, 1.0, y_d0, t_eval_1)

    with torch.cuda.amp.autocast(enabled=use_bf16, dtype=torch.bfloat16):
        h_d0 = model.exp_encoder(
            torch.from_numpy(df0).to(device), torch.from_numpy(tm0).to(device),
            torch.from_numpy(ov0).to(device), torch.from_numpy(om0).to(device),
            torch.from_numpy(tp0).to(device),
        ).float().unsqueeze(1)  # (N, 1, d_exp)
        f_d0 = model.set_encoder(h_d0, exp_pad_mask=torch.zeros((n_cases, 1), dtype=torch.bool, device=device))
        logits_d0, G_mat = model.mech_head(f_d0)
    probs_m_d0 = torch.softmax(logits_d0.float(), dim=-1).cpu().numpy()
    probs_c_d0 = aggregate_class_probs(probs_m_d0, m2c, n_cls)
    f_d0 = f_d0.float()
    G_mat = G_mat.float()

    # Step 1 continuous Quotient EIG across all 64 LHS points
    all_cands = [list(range(n_lhs)) for _ in range(n_cases)]
    eig_q_step1, _ = _score_eig_designs_batched(
        model, grammar, sim, h_d0, f_d0, G_mat, probs_m_d0, probs_c_d0,
        lhs_pool, all_cands, m2c, n_cls, n_t, sigma_rel, n_mc=n_mc, device=device, rng=rng,
    )
    step1_best = [int(np.argmax(eig_q_step1[i])) for i in range(n_cases)]

    # Spatial clustering diagnostics on step-1 optimal continuous xi*
    base_vtna_frac = float(np.mean((cat_arr <= 0.55) | (cat_arr >= 1.8)))
    base_prod_frac = float(np.mean(P0_arr >= 0.2))
    base_rpka_frac = float(np.mean((P0_arr >= 0.2) | (A0_arr <= 0.8)))
    base_highA_frac = float(np.mean(A0_arr >= 1.4))

    vtna_sel, rpka_sel, highA_sel = [], [], []
    for i in range(n_cases):
        if probs_c_d0[i].max() >= 0.90:
            continue
        top2_c = np.argsort(probs_c_d0[i])[-2:][::-1]
        m_a = max(classes[top2_c[0]], key=lambda m: probs_m_d0[i, m])
        m_b = max(classes[top2_c[1]], key=lambda m: probs_m_d0[i, m])
        oa, ob = grammar.mech_options[m_a], grammar.mech_options[m_b]
        k = step1_best[i]

        cat_ord_a = (oa[4] > 0, oa[7] > 0, oa[1] == 2, oa[5] > 0)
        cat_ord_b = (ob[4] > 0, ob[7] > 0, ob[1] == 2, ob[5] > 0)
        if cat_ord_a != cat_ord_b:
            vtna_sel.append(k)
        m2_diff = (oa[1] > 0) != (ob[1] > 0)
        p_diff = ((oa[3] > 0) or (oa[6] > 0)) != ((ob[3] > 0) or (ob[6] > 0))
        if m2_diff or p_diff:
            rpka_sel.append(k)
        if oa[2] != ob[2]:
            highA_sel.append(k)

    spatial_stats = {
        "pool_size": n_lhs,
        "overall_step1_mean_cat_extremity_frac": float(np.mean([(cat_arr[k] <= 0.55 or cat_arr[k] >= 1.8) for k in step1_best])),
        "overall_step1_preadded_P_frac": float(np.mean([P0_arr[k] >= 0.2 for k in step1_best])),
        "baseline_uniform_vtna_frac": base_vtna_frac,
        "baseline_uniform_preadded_P_frac": base_prod_frac,
        "baseline_uniform_rpka_frac": base_rpka_frac,
        "baseline_uniform_highA_frac": base_highA_frac,
        "VTNA_ambig_cases": {
            "n": len(vtna_sel),
            "extreme_cat_loading_frac": float(np.mean([(cat_arr[k] <= 0.55 or cat_arr[k] >= 1.8) for k in vtna_sel])) if vtna_sel else 0.0,
            "mean_abs_log2_cat": float(np.mean([abs(np.log2(cat_arr[k])) for k in vtna_sel])) if vtna_sel else 0.0,
        },
        "RPKA_M2_M4M7_ambig_cases": {
            "n": len(rpka_sel),
            "preadded_P_frac": float(np.mean([P0_arr[k] >= 0.2 for k in rpka_sel])) if rpka_sel else 0.0,
            "rpka_zone_frac": float(np.mean([(P0_arr[k] >= 0.2 or A0_arr[k] <= 0.8) for k in rpka_sel])) if rpka_sel else 0.0,
            "mean_P0": float(np.mean([P0_arr[k] for k in rpka_sel])) if rpka_sel else 0.0,
        },
        "SubA_M3_ambig_cases": {
            "n": len(highA_sel),
            "high_A0_frac": float(np.mean([A0_arr[k] >= 1.4 for k in highA_sel])) if highA_sel else 0.0,
            "mean_A0": float(np.mean([A0_arr[k] for k in highA_sel])) if highA_sel else 0.0,
        },
    }

    # Closed-loop 4-step rollout comparing Continuous Quotient EIG vs Continuous Random
    def rollout_cont(policy: str) -> Dict:
        h_curr = h_d0.clone()
        f_curr = f_d0.clone()
        pm_curr = probs_m_d0.copy()
        pc_curr = probs_c_d0.copy()
        used = [set() for _ in range(n_cases)]

        pc_traj = np.zeros((n_cases, budget + 1), dtype=np.float64)
        top1_traj = np.zeros((n_cases, budget + 1), dtype=np.float64)
        max_pc_traj = np.zeros((n_cases, budget + 1), dtype=np.float64)
        pc_traj[:, 0] = pc_curr[np.arange(n_cases), true_c]
        top1_traj[:, 0] = (np.argmax(pc_curr, axis=1) == true_c).astype(np.float64)
        max_pc_traj[:, 0] = np.max(pc_curr, axis=1)

        for step in range(budget):
            if policy == "Continuous_Random":
                picks = [int(rng.choice([k for k in range(n_lhs) if k not in used[i]])) for i in range(n_cases)]
            elif step == 0:
                picks = list(step1_best)
            else:
                cands = [[k for k in range(n_lhs) if k not in used[i]] for i in range(n_cases)]
                eq_l, _ = _score_eig_designs_batched(
                    model, grammar, sim, h_curr, f_curr, G_mat, pm_curr, pc_curr,
                    lhs_pool, cands, m2c, n_cls, n_t, sigma_rel, n_mc=n_mc, device=device, rng=rng,
                )
                picks = [cands[i][int(np.argmax(eq_l[i]))] for i in range(n_cases)]

            # Simulate real noisy observation for each case's chosen continuous design on 64-core CPU
            new_embs = torch.empty((n_cases, model.exp_encoder.d_model), device=device, dtype=torch.float32)
            for i, k in enumerate(picks):
                used[i].add(k)
                des, t_end = lhs_pool[k]
                t_q = np.linspace(0.0, t_end, n_t)
                y0 = grammar.initial_state(int(true_mids[i]), des)[None, :]
                y_sim, _, _, _ = sim.simulate_batch(
                    np.array([true_mids[i]], dtype=np.int32), true_thetas[i : i + 1], y0, t_q, n_threads=1
                )
                y_obs = y_sim[:, :, obs_ch]
                n_sc = np.where(ch_is_cat, des.cat_mult * grammar.c0_base, des.A0)
                y_n = y_obs + rng.normal(0.0, 1.0, size=y_obs.shape) * (sigma_rel * n_sc[None, None, :])
                df_i, tm_i, ov_i, om_i, tp_i = _build_single_exp_tensors(grammar, des, t_end, y_n, t_q)
                with torch.cuda.amp.autocast(enabled=use_bf16, dtype=torch.bfloat16):
                    new_embs[i] = model.exp_encoder(
                        torch.from_numpy(df_i).to(device), torch.from_numpy(tm_i).to(device),
                        torch.from_numpy(ov_i).to(device), torch.from_numpy(om_i).to(device),
                        torch.from_numpy(tp_i).to(device),
                    )[0].float()

            h_curr = torch.cat([h_curr, new_embs.unsqueeze(1)], dim=1)
            ep = torch.zeros((n_cases, h_curr.shape[1]), dtype=torch.bool, device=device)
            with torch.cuda.amp.autocast(enabled=use_bf16, dtype=torch.bfloat16):
                f_curr = model.set_encoder(h_curr, exp_pad_mask=ep)
                logits, _ = model.mech_head(f_curr)
            f_curr = f_curr.float()
            pm_curr = torch.softmax(logits.float(), dim=-1).cpu().numpy()
            pc_curr = aggregate_class_probs(pm_curr, m2c, n_cls)
            pc_traj[:, step + 1] = pc_curr[np.arange(n_cases), true_c]
            top1_traj[:, step + 1] = (np.argmax(pc_curr, axis=1) == true_c).astype(np.float64)
            max_pc_traj[:, step + 1] = np.max(pc_curr, axis=1)

        t_stop = np.full(n_cases, budget, dtype=np.int64)
        for i in range(n_cases):
            for ti in range(budget + 1):
                if max_pc_traj[i, ti] >= 0.90:
                    t_stop[i] = ti
                    break
        return {
            "n_cases": n_cases,
            "auc_true_class_prob": float(np.mean(pc_traj)),
            "mean_pc_by_step": [float(x) for x in np.mean(pc_traj, axis=0)],
            "top1_by_step": [float(x) for x in np.mean(top1_traj, axis=0)],
            "unified_class_stop_mean_exps": float(np.mean(t_stop)),
        }

    res_q = rollout_cont("Continuous_Quotient_EIG")
    res_r = rollout_cont("Continuous_Random")
    elapsed = time.perf_counter() - t0
    print(
        f"\n[BOED:Continuous LHS-{n_lhs}] Done in {elapsed:.1f}s | "
        f"Cont_Quot AUC={res_q['auc_true_class_prob']:.3f} top1@4={res_q['top1_by_step'][-1]:.3f} vs "
        f"Cont_Rand AUC={res_r['auc_true_class_prob']:.3f} top1@4={res_r['top1_by_step'][-1]:.3f}"
    )
    return {
        "n_cases": n_cases,
        "n_lhs": n_lhs,
        "elapsed_s": elapsed,
        "Continuous_Quotient_EIG": res_q,
        "Continuous_Random": res_r,
        "spatial_clustering": spatial_stats,
    }
