"""E3 Closed-Loop Bayesian Experimental Design (Section 7.4 & Step 4.2).

Compares 5 strategies across Ξ_no-P, Ξ_basic, Ξ_full on 150 test cases (from D0, budget T=4):
  1. Random
  2. Textbook fixed RPKA/VTNA (D2 -> D5 -> D6 -> D3; D4 fallback when D6 unavailable)
  3. Static EIG (pre-selected prior-optimal sequence)
  4. Fine-grained EIG (Barber-Agakov I(M; Y_xi | D)), reporting BOTH:
       - unified class-level stopping (max_C q(C_Xi | D) >= 0.90)
       - fine-grained stopping (max_m q(m | D) >= 0.90)
  5. Quotient EIG (Barber-Agakov I(C_Xi; Y_xi | D))

Also quantifies spontaneous emergence of RPKA & VTNA diagnostic logic (Section 8.2).
Uses fast cached experiment embeddings + batched 64-core CPU ROS2 simulation + GPU BF16 scoring.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import numpy as np
import torch

from quomech.data_gen import get_observable_channel_indices
from quomech.evaluate import aggregate_class_probs
from quomech.grammar import ReactionGrammar
from quomech.models import QuoMechNet
from quomech.simulator import ODESimulator


def _build_single_exp_tensors(
    grammar: ReactionGrammar,
    design_idx: int,
    y_noisy: np.ndarray,  # (B, n_t, n_ch) raw noisy concentrations
    t_grid: np.ndarray,   # (n_t,)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pack a batch of B observations of ONE catalog design into ExperimentEncoder tensors."""
    B, n_t, n_ch = y_noisy.shape
    obs_ch = get_observable_channel_indices(grammar)
    ch_is_cat = np.array([grammar.species[s].is_catalyst for s in obs_ch], dtype=bool)
    des = grammar.catalog[design_idx]

    ch_mask = np.zeros(n_ch, dtype=np.float32)
    obs_full = np.zeros(grammar.n_species, dtype=np.float32)
    for c_i, sp in enumerate(obs_ch):
        if sp in des.obs_species:
            ch_mask[c_i] = 1.0
            obs_full[sp] = 1.0

    cat0 = des.cat_mult * grammar.c0_base
    df1 = np.concatenate([
        np.array([des.A0, des.B0, des.P0, np.log10(max(cat0, 1e-6)), float(des.a_pre)], dtype=np.float32),
        obs_full,
    ])
    df = np.broadcast_to(df1[None, :], (B, len(df1))).copy()
    tm = np.broadcast_to(t_grid[None, :], (B, n_t)).copy()

    norm_scale = np.where(ch_is_cat, cat0, 1.0).astype(np.float32)
    ov = (y_noisy.astype(np.float32) / norm_scale[None, None, :]) * ch_mask[None, None, :]
    om = np.broadcast_to(ch_mask[None, None, :], (B, n_t, n_ch)).copy()
    tp = np.zeros((B, n_t), dtype=bool)
    return df, tm, ov, om, tp


@torch.no_grad()
def _score_eig_batched(
    model: QuoMechNet,
    grammar: ReactionGrammar,
    sim: ODESimulator,
    hist_embs: torch.Tensor,      # (B_act, E_hist, d_exp) on GPU
    f_curr: torch.Tensor,         # (B_act, d_set) on GPU
    G_mat: torch.Tensor,          # (M, d_g) on GPU
    probs_m_curr: np.ndarray,     # (B_act, M)
    probs_c_curr: np.ndarray,     # (B_act, n_classes)
    cand_lists: List[List[int]],  # len B_act: available candidate design indices for each active case
    mech_to_class: np.ndarray,    # (M,)
    n_classes: int,
    t_eval: np.ndarray,
    sigma_rel: float,
    n_mc: int,
    device: torch.device,
    rng: np.random.Generator,
    gpu_batch: int = 2048,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Compute Barber-Agakov Quotient EIG and Fine-grained EIG simultaneously for B_act active cases."""
    B_act = len(cand_lists)
    M = grammar.n_mechs
    obs_ch = get_observable_channel_indices(grammar)
    ch_is_cat = np.array([grammar.species[s].is_catalyst for s in obs_ch], dtype=bool)
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    # 1. Sample (m_i, theta_i) ~ q(. | D) for each active case: shape (B_act, n_mc)
    sampled_m = np.empty((B_act, n_mc), dtype=np.int32)
    for b in range(B_act):
        p = probs_m_curr[b]
        p = p / np.sum(p)
        sampled_m[b] = rng.choice(M, size=n_mc, replace=True, p=p)

    # Batched flow sampling on GPU
    f_rep = f_curr.unsqueeze(1).expand(B_act, n_mc, -1).reshape(B_act * n_mc, -1)
    m_flat_t = torch.from_numpy(sampled_m.ravel()).long().to(device)
    th_chunks = []
    for st in range(0, B_act * n_mc, 4096):
        ed = min(B_act * n_mc, st + 4096)
        th_chunks.append(model.sample_theta(f_rep[st:ed], G_mat, m_flat_t[st:ed], num_samples=1).cpu().numpy())
    sampled_th = np.concatenate(th_chunks, axis=0).astype(np.float64).reshape(B_act, n_mc, grammar.n_params)
    # Clamp extreme sampled log10_k safely to +/- 5 std around prior mean so ODE stays bounded
    lb = grammar.mech_prior_mean - 4.5 * grammar.mech_prior_std
    ub = grammar.mech_prior_mean + 4.5 * grammar.mech_prior_std
    m_flat = sampled_m.ravel()
    th_flat = sampled_th.reshape(B_act * n_mc, grammar.n_params)
    th_flat = np.clip(th_flat, lb[m_flat], ub[m_flat])
    sampled_th = th_flat.reshape(B_act, n_mc, grammar.n_params)

    # Group candidate evaluations by design index so each catalog design is simulated & encoded in bulk
    # Map design_idx -> list of active case indices b that include design_idx in cand_lists[b]
    cases_by_design: Dict[int, List[int]] = {}
    for b, cands in enumerate(cand_lists):
        for d in cands:
            cases_by_design.setdefault(d, []).append(b)

    eig_q_map: Dict[Tuple[int, int], float] = {}
    eig_f_map: Dict[Tuple[int, int], float] = {}

    E_hist = hist_embs.shape[1]
    t_eval_f32 = t_eval.astype(np.float32)
    m2c_t = torch.from_numpy(mech_to_class).long().to(device)

    for d_idx, b_list in cases_by_design.items():
        des = grammar.catalog[d_idx]
        n_b = len(b_list)
        b_arr = np.array(b_list, dtype=np.int64)
        m_sub = sampled_m[b_arr].ravel()                       # (n_b * n_mc,)
        th_sub = sampled_th[b_arr].reshape(n_b * n_mc, -1)     # (n_b * n_mc, P)

        # Build y0 for design d_idx
        y0_table = np.array([grammar.initial_state(m, des) for m in range(M)], dtype=np.float64)
        y0_sub = y0_table[m_sub]

        # 64-core CPU ROS2 simulation
        y_sim, st_sim, _, _ = sim.simulate_batch(
            m_sub, th_sub, y0_sub, t_eval, compute_flux=False, rtol=2e-5, atol=2e-9, max_steps=10000
        )
        y_obs = y_sim[:, :, obs_ch]  # (n_b * n_mc, n_t, n_ch)
        cat0 = des.cat_mult * grammar.c0_base
        noise_scale = np.where(ch_is_cat, cat0, des.A0)
        y_noisy = y_obs + rng.normal(0.0, 1.0, size=y_obs.shape) * (sigma_rel * noise_scale[None, None, :])

        # Pack into ExperimentEncoder input tensors
        df_np, tm_np, ov_np, om_np, tp_np = _build_single_exp_tensors(grammar, d_idx, y_noisy, t_eval_f32)

        # GPU batched scoring: encode new exp -> concat with cached hist_embs -> SetEncoder + MechHead
        tot_n = n_b * n_mc
        # Pre-gather prior log-probs for each (b, i)
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
                )  # (bs, d_exp)
                h_prev = hist_embs[b_idx_rep_t[st:ed]]  # (bs, E_hist, d_exp)
                h_cat = torch.cat([h_prev, new_emb.unsqueeze(1).to(h_prev.dtype)], dim=1)
                ep_zero = torch.zeros((bs, E_hist + 1), dtype=torch.bool, device=device)
                probs_new = model.probs_from_exp_embs(h_cat, ep_zero)  # (bs, M) float32

            ar = torch.arange(bs, device=device)
            pm_new = probs_new[ar, m_sub_t[st:ed]].clamp(min=1e-12)
            # Sum class probability cleanly on GPU via scatter_add
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
    """Run full E3 closed-loop BOED benchmark across Ξ_no-P, Ξ_basic, Ξ_full."""
    model.eval()
    rng = np.random.default_rng(seed)
    obs_ch = get_observable_channel_indices(grammar)
    ch_is_cat = np.array([grammar.species[s].is_catalyst for s in obs_ch], dtype=bool)
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    all_mids = np.load(test_dir / "mech_ids.npy")
    all_trajs = np.load(test_dir / "trajs.npy")
    # Pick 1 fresh test sample per mechanism (151 cases, completely covering all 151 mechanisms!)
    # Use index 5 per mech so distinct from Enum+BIC index 0
    sel_idx = np.array([np.where(all_mids == m)[0][5] for m in range(grammar.n_mechs)], dtype=np.int64)
    if n_cases < len(sel_idx):
        sel_idx = sel_idx[:n_cases]
    N = len(sel_idx)
    true_mids = all_mids[sel_idx].astype(np.int64)
    trajs_sel = all_trajs[sel_idx]  # (N, 14, 64, C)

    n_t = 20
    t_idx = np.round(np.linspace(0, trajs_sel.shape[2] - 1, n_t)).astype(np.int64)
    t_eval = np.linspace(0.0, 1.0, n_t)
    t_eval_f32 = t_eval.astype(np.float32)

    # Pre-generate FIXED noisy observations for all N cases x 10 catalog designs (paired CRN comparison!)
    noisy_obs = np.empty((N, 10, n_t, len(obs_ch)), dtype=np.float64)
    exp_embs_all = torch.empty((N, 10, model.exp_encoder.d_model), device=device, dtype=torch.float32)

    for d_idx in range(10):
        des = grammar.catalog[d_idx]
        clean = trajs_sel[:, d_idx, :, :][:, t_idx, :].astype(np.float64)
        cat0 = des.cat_mult * grammar.c0_base
        noise_scale = np.where(ch_is_cat, cat0, des.A0)
        y_n = clean + rng.normal(0.0, 1.0, size=clean.shape) * (sigma_rel * noise_scale[None, None, :])
        noisy_obs[:, d_idx] = y_n

        df_np, tm_np, ov_np, om_np, tp_np = _build_single_exp_tensors(grammar, d_idx, y_n, t_eval_f32)
        with torch.cuda.amp.autocast(enabled=use_bf16, dtype=torch.bfloat16):
            emb = model.exp_encoder(
                torch.from_numpy(df_np).to(device),
                torch.from_numpy(tm_np).to(device),
                torch.from_numpy(ov_np).to(device),
                torch.from_numpy(om_np).to(device),
                torch.from_numpy(tp_np).to(device),
            )
        exp_embs_all[:, d_idx] = emb.float()

    # Precompute mechanism embeddings G once
    _, G_mat = model.mech_head(torch.zeros((1, 256), device=device))
    G_mat = G_mat.float()

    def eval_state(chosen_hist: List[List[int]]) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        """Given list of observed design index lists (all same length E), return (probs_m, f, hist_embs)."""
        E = len(chosen_hist[0])
        idx_t = torch.tensor(chosen_hist, dtype=torch.long, device=device)  # (N, E)
        ar = torch.arange(N, device=device).unsqueeze(1).expand(N, E)
        h_seq = exp_embs_all[ar, idx_t]  # (N, E, d_exp)
        ep = torch.zeros((N, E), dtype=torch.bool, device=device)
        with torch.cuda.amp.autocast(enabled=use_bf16, dtype=torch.bfloat16):
            f = model.set_encoder(h_seq, exp_pad_mask=ep)
            logits, _ = model.mech_head(f)
        probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        return probs, f.float(), h_seq

    # Compute initial D0 posterior once
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

        # Compute Static EIG sequence (greedy prior-average Quotient EIG across the N cases' D0 state -> fixed 4 designs!)
        # Actually per Section 8.3: "静态 EIG: 仅按先验一次性选定 4 组实验" — select top-4 designs by average step-1 Quotient EIG!
        cand_d0 = [[d for d in tier_designs if d != 0] for _ in range(N)]
        eig_q_d0, eig_f_d0 = _score_eig_batched(
            model, grammar, sim, h_d0, f_d0, G_mat, probs_m_d0, probs_c_d0,
            cand_d0, m2c, n_cls, t_eval, sigma_rel, n_mc=n_mc, device=device, rng=rng,
        )
        eig_q_d0_mat = np.stack(eig_q_d0, axis=0)  # (N, n_cands)
        eig_f_d0_mat = np.stack(eig_f_d0, axis=0)
        mean_q_d0 = np.mean(eig_q_d0_mat, axis=0)
        static_order_idx = np.argsort(mean_q_d0)[::-1][:budget]
        static_seq = [cand_d0[0][i] for i in static_order_idx]

        # Textbook fixed RPKA/VTNA sequence (D2 -> D5 -> D6 -> D3; substitute D4 in no_P where D6 unavailable)
        if tier == "no_P":
            textbook_seq = [2, 5, 3, 4]  # D2, D5, D3, D4
        else:
            textbook_seq = [2, 5, 6, 3]  # D2, D5, D6, D3

        # Spontaneous RPKA / VTNA emergence statistics from step-1 Quotient EIG choices on D0!
        step1_q_choices = [cand_d0[i][int(np.argmax(eig_q_d0_mat[i]))] for i in range(N)]
        step1_f_choices = [cand_d0[i][int(np.argmax(eig_f_d0_mat[i]))] for i in range(N)]
        step1_disagree = float(np.mean([a != b for a, b in zip(step1_q_choices, step1_f_choices)]))
        step1_disagree_nontriv = float(
            np.mean([step1_q_choices[i] != step1_f_choices[i] for i in range(N) if nontriv_mask[i]])
        ) if np.any(nontriv_mask) else 0.0

        # Analyze spontaneous RPKA/VTNA diagnostic logic on cases where top-2 classes after D0 differ by key chemical motif
        rpka_hits, rpka_tot = 0, 0
        vtna_hits, vtna_tot = 0, 0
        subA_hits, subA_tot = 0, 0
        act_hits, act_tot = 0, 0
        for i in range(N):
            if probs_c_d0[i].max() >= 0.90:
                continue  # already confident from D0
            top2_c = np.argsort(probs_c_d0[i])[-2:][::-1]
            # Representative top mechanism in each of top-2 classes
            m_a = max(classes[top2_c[0]], key=lambda m: probs_m_d0[i, m])
            m_b = max(classes[top2_c[1]], key=lambda m: probs_m_d0[i, m])
            opt_a, opt_b = grammar.mech_options[m_a], grammar.mech_options[m_b]
            ch = step1_q_choices[i]
            # 1) M2 vs M4/M7 (deactivation vs product inhibition/rev release)
            m2_diff = (opt_a[1] > 0) != (opt_b[1] > 0)
            p_diff = ((opt_a[3] > 0) or (opt_a[6] > 0)) != ((opt_b[3] > 0) or (opt_b[6] > 0))
            if m2_diff and p_diff:
                rpka_tot += 1
                if ch in (5, 6, 7):  # D5/D6/D7
                    rpka_hits += 1
            # 2) Catalyst order ambiguity (M5, M8, M2:free2, M6)
            cat_ord_a = (opt_a[4] > 0, opt_a[7] > 0, opt_a[1] == 2, opt_a[5] > 0)
            cat_ord_b = (opt_b[4] > 0, opt_b[7] > 0, opt_b[1] == 2, opt_b[5] > 0)
            if cat_ord_a != cat_ord_b:
                vtna_tot += 1
                if ch in (1, 2):  # D1/D2
                    vtna_hits += 1
            # 3) M3 substrate A inhibition
            if opt_a[2] != opt_b[2]:
                subA_tot += 1
                if ch in (3, 4):  # D3/D4
                    subA_hits += 1
            # 4) M1 activation
            if opt_a[0] != opt_b[0] and tier == "full":
                act_tot += 1
                if ch == 8:  # D8
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
                    d_pick = textbook_seq[step]
                    for i in range(N):
                        hist[i].append(d_pick)
                elif strat_name == "Static_EIG":
                    d_pick = static_seq[step]
                    for i in range(N):
                        hist[i].append(d_pick)
                elif strat_name in ("Quotient_EIG", "Fine_EIG"):
                    if step == 0:
                        choices = step1_q_choices if strat_name == "Quotient_EIG" else step1_f_choices
                        for i in range(N):
                            hist[i].append(choices[i])
                    else:
                        cands_all = [[d for d in tier_designs if d not in hist[i]] for i in range(N)]
                        eq_list, ef_list = _score_eig_batched(
                            model, grammar, sim, h_curr, f_curr, G_mat, pm_curr, pc_curr,
                            cands_all, m2c, n_cls, t_eval, sigma_rel, n_mc=n_mc, device=device, rng=rng,
                        )
                        scores_use = eq_list if strat_name == "Quotient_EIG" else ef_list
                        for i in range(N):
                            best_d = cands_all[i][int(np.argmax(scores_use[i]))]
                            hist[i].append(best_d)

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

                # Stopping times (0..budget additional experiments)
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
