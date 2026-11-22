"""E1 Evaluation Suite: Equivalence-class Top-1/3, ECE, SBC-KS, Held-out PPE, and 50-case Enum+BIC."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import numpy as np
from scipy.stats import kstest
import torch

from quomech.data_gen import get_observable_channel_indices
from quomech.equivalence import build_design_targets
from quomech.grammar import DesignSpec, ReactionGrammar
from quomech.models import QuoMechNet
from quomech.simulator import ODESimulator
from quomech.train import compute_ece


def aggregate_class_probs(probs_m: np.ndarray, mech_to_class: np.ndarray, n_classes: int) -> np.ndarray:
    """Exact quotient posterior q(C | D) = sum_{m in C} q(m | D) for batch (N, M)."""
    N = probs_m.shape[0]
    probs_c = np.zeros((N, n_classes), dtype=np.float64)
    for m, c in enumerate(mech_to_class):
        probs_c[:, c] += probs_m[:, m]
    return probs_c


def pack_fixed_test_batch(
    grammar: ReactionGrammar,
    trajs: np.ndarray,       # (N, 14, 64, C)
    rand_des: np.ndarray,    # (N, 4, 5)
    design_slots_list: List[List[int]],
    sigma_rel: float = 0.01,
    n_t_obs: int = 20,
    p_only: bool = False,
    seed: int = 777,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create standardized noisy observation tensors for N test cases.

    Also returns raw noisy trajectories `raw_noisy` for Enum+BIC comparison when design sets match.
    """
    rng = np.random.default_rng(seed)
    N = len(design_slots_list)
    obs_ch = get_observable_channel_indices(grammar)
    n_ch = len(obs_ch)
    n_cat = len(grammar.catalog)
    ch_is_cat = np.array([grammar.species[s].is_catalyst for s in obs_ch], dtype=bool)
    ch_pos_A = obs_ch.index(grammar.species_idx["A"])
    ch_pos_P = obs_ch.index(grammar.species_idx["P"])

    cat_params = np.array(
        [[d.A0, d.B0, d.P0, d.cat_mult, float(d.a_pre)] for d in grammar.catalog],
        dtype=np.float32,
    )
    cat_obs_mask = np.zeros((n_cat, n_ch), dtype=np.float32)
    for d_i, des in enumerate(grammar.catalog):
        for c_i, sp in enumerate(obs_ch):
            if sp in des.obs_species:
                cat_obs_mask[d_i, c_i] = 1.0

    E_max = max(len(ds) for ds in design_slots_list)
    des_dim = 5 + grammar.n_species
    t_idx = np.round(np.linspace(0, trajs.shape[2] - 1, n_t_obs)).astype(np.int64)
    t_grid = np.linspace(0.0, 1.0, trajs.shape[2], dtype=np.float32)[t_idx]

    df = np.zeros((N, E_max, des_dim), dtype=np.float32)
    tm = np.zeros((N, E_max, n_t_obs), dtype=np.float32)
    ov = np.zeros((N, E_max, n_t_obs, n_ch), dtype=np.float32)
    om = np.zeros((N, E_max, n_t_obs, n_ch), dtype=np.float32)
    tp = np.ones((N, E_max, n_t_obs), dtype=bool)
    ep = np.ones((N, E_max), dtype=bool)
    raw_noisy = np.zeros((N, E_max, n_t_obs, n_ch), dtype=np.float64)

    for i, slots in enumerate(design_slots_list):
        for e, d_slot in enumerate(slots):
            ep[i, e] = False
            if d_slot < n_cat:
                p5 = cat_params[d_slot]
                ch_mask = cat_obs_mask[d_slot].copy()
            else:
                p5 = rand_des[i, d_slot - n_cat]
                ch_mask = np.zeros(n_ch, dtype=np.float32)
                ch_mask[ch_pos_A] = 1.0
                ch_mask[ch_pos_P] = 1.0
            if p_only and d_slot != 9:
                ch_mask[:] = 0.0
                ch_mask[ch_pos_P] = 1.0

            obs_full = np.zeros(grammar.n_species, dtype=np.float32)
            for c_i, sp in enumerate(obs_ch):
                if ch_mask[c_i] > 0.5:
                    obs_full[sp] = 1.0
            cat0 = float(p5[3]) * grammar.c0_base
            df[i, e] = np.concatenate([
                np.array([p5[0], p5[1], p5[2], np.log10(max(cat0, 1e-6)), p5[4]], dtype=np.float32),
                obs_full,
            ])

            y_clean = trajs[i, d_slot, t_idx, :].astype(np.float64)
            A0 = float(p5[0])
            noise_scale = np.where(ch_is_cat, cat0, A0)
            norm_scale = np.where(ch_is_cat, cat0, 1.0)
            y_noisy = y_clean + rng.normal(0.0, 1.0, size=y_clean.shape) * (sigma_rel * noise_scale[None, :])
            raw_noisy[i, e] = y_noisy

            tm[i, e] = t_grid
            ov[i, e] = (y_noisy / norm_scale[None, :]).astype(np.float32) * ch_mask[None, :]
            om[i, e] = np.broadcast_to(ch_mask[None, :], (n_t_obs, n_ch))
            tp[i, e] = False

    return df, tm, ov, om, tp, ep, raw_noisy


@torch.no_grad()
def evaluate_e1_suite(
    model: QuoMechNet,
    grammar: ReactionGrammar,
    sim: ODESimulator,
    test_dir: Path,
    equiv_tiers: Dict[str, Dict],
    device: torch.device,
    n_test: int = 1510,
    batch_size: int = 256,
    seed: int = 2026,
) -> Dict:
    """Comprehensive E1 evaluation on held-out test split."""
    model.eval()
    rng = np.random.default_rng(seed)

    all_mids = np.load(test_dir / "mech_ids.npy")
    all_thetas = np.load(test_dir / "thetas.npy")
    all_rand = np.load(test_dir / "rand_des.npy")
    all_trajs = np.load(test_dir / "trajs.npy")

    # Stratified subsample across all 151 mechanisms
    per_m = max(1, n_test // grammar.n_mechs)
    chosen_idx = []
    for m in range(grammar.n_mechs):
        m_idxs = np.where(all_mids == m)[0]
        chosen_idx.extend(m_idxs[:per_m].tolist())
    chosen_idx = np.array(chosen_idx, dtype=np.int64)
    N = len(chosen_idx)

    mids = all_mids[chosen_idx].astype(np.int64)
    thetas = all_thetas[chosen_idx].astype(np.float64)
    rand_des = all_rand[chosen_idx]
    trajs = all_trajs[chosen_idx]

    # Define evaluation regimes
    regimes = {
        "cat_1_D0": ([[0] for _ in range(N)], False, "D0"),
        "cat_3_D0_D2_D5": ([[0, 2, 5] for _ in range(N)], False, "no_P"),
        "cat_4_D0_D2_D5_D6": ([[0, 2, 5, 6] for _ in range(N)], False, "basic"),
        "cat_6_noP_all": ([list(grammar.tiers["no_P"]) for _ in range(N)], False, "no_P"),
        "cat_7_basic_all": ([list(grammar.tiers["basic"]) for _ in range(N)], False, "basic"),
        "cat_10_full_all": ([list(grammar.tiers["full"]) for _ in range(N)], False, "full"),
        "cont_4_random": ([[10, 11, 12, 13] for _ in range(N)], False, "full"),
        "P_only_basic": ([list(grammar.tiers["basic"]) for _ in range(N)], True, "basic"),
    }

    results = {"n_test": N, "regimes": {}}
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    for rname, (slots_list, p_only, tier_key) in regimes.items():
        df, tm, ov, om, tp, ep, _ = pack_fixed_test_batch(
            grammar, trajs, rand_des, slots_list, sigma_rel=0.01, n_t_obs=20, p_only=p_only, seed=seed + len(rname)
        )
        t0 = time.perf_counter()
        probs_list = []
        f_list = []
        G_last = None
        for st in range(0, N, batch_size):
            ed = min(N, st + batch_size)
            with torch.cuda.amp.autocast(enabled=use_bf16, dtype=torch.bfloat16):
                pr, f_b, G_b = model.predict_posterior(
                    torch.from_numpy(df[st:ed]).to(device),
                    torch.from_numpy(tm[st:ed]).to(device),
                    torch.from_numpy(ov[st:ed]).to(device),
                    torch.from_numpy(om[st:ed]).to(device),
                    torch.from_numpy(tp[st:ed]).to(device),
                    torch.from_numpy(ep[st:ed]).to(device),
                )
            probs_list.append(pr.cpu().numpy())
            f_list.append(f_b.cpu())
            G_last = G_b
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        probs_m = np.concatenate(probs_list, axis=0)
        f_all = torch.cat(f_list, dim=0)

        # Fine-grained metrics
        m_top1 = float(np.mean(np.argmax(probs_m, axis=1) == mids))
        m_top3 = float(np.mean(np.any(np.argpartition(probs_m, -3, axis=1)[:, -3:] == mids[:, None], axis=1)))
        m_ece = compute_ece(probs_m, mids)
        m_true_p = float(np.mean(probs_m[np.arange(N), mids]))

        # Quotient-class metrics under target tier
        eq = equiv_tiers[tier_key]["primary"]
        m2c = np.array(eq["mech_to_class"], dtype=np.int64)
        n_cls = eq["n_classes"]
        true_c = m2c[mids]
        probs_c = aggregate_class_probs(probs_m, m2c, n_cls)
        c_top1 = float(np.mean(np.argmax(probs_c, axis=1) == true_c))
        k3 = min(3, n_cls)
        c_top3 = float(np.mean(np.any(np.argpartition(probs_c, -k3, axis=1)[:, -k3:] == true_c[:, None], axis=1)))
        c_ece = compute_ece(probs_c, true_c)
        c_true_p = float(np.mean(probs_c[np.arange(N), true_c]))

        entry = {
            "tier": tier_key,
            "n_classes": n_cls,
            "mech_top1": m_top1,
            "mech_top3": m_top3,
            "mech_ece": m_ece,
            "mech_mean_true_prob": m_true_p,
            "class_top1": c_top1,
            "class_top3": c_top3,
            "class_ece": c_ece,
            "class_mean_true_prob": c_true_p,
            "infer_ms_per_case": 1000.0 * dt / N,
        }

        # Compute SBC-KS and Held-out PPE on key regimes (cat_4_D0_D2_D5_D6 and cat_7_basic_all)
        if rname in ("cat_4_D0_D2_D5_D6", "cat_7_basic_all"):
            # 1) Exact SBC-KS (L=99 posterior draws per test case conditioned on true m)
            L_sbc = 99
            ranks_by_param: List[List[float]] = [[] for _ in range(grammar.n_params)]
            for st in range(0, N, batch_size):
                ed = min(N, st + batch_size)
                f_b = f_all[st:ed].to(device)
                m_b = torch.from_numpy(mids[st:ed]).to(device)
                th_samp = model.sample_theta(f_b, G_last, m_b, num_samples=L_sbc).cpu().numpy()  # (L, B, P)
                th_true = thetas[st:ed]  # (B, P)
                for bi in range(ed - st):
                    pmask = grammar.mech_param_mask[mids[st + bi]]
                    for p_i in range(grammar.n_params):
                        if pmask[p_i]:
                            cnt = int(np.sum(th_samp[:, bi, p_i] < th_true[bi, p_i]))
                            u = (cnt + rng.random()) / (L_sbc + 1.0)
                            ranks_by_param[p_i].append(u)

            ks_per_param = {}
            ks_vals = []
            for p_i, u_list in enumerate(ranks_by_param):
                if len(u_list) >= 30:
                    stat = float(kstest(u_list, "uniform").statistic)
                    ks_per_param[grammar.param_names[p_i]] = stat
                    ks_vals.append(stat)
            all_u = [u for ul in ranks_by_param for u in ul]
            entry["sbc_ks_overall"] = float(kstest(all_u, "uniform").statistic)
            entry["sbc_ks_median_param"] = float(np.median(ks_vals))
            entry["sbc_ks_mean_param"] = float(np.mean(ks_vals))
            entry["sbc_ks_max_param"] = float(np.max(ks_vals))
            entry["sbc_ks_by_param"] = ks_per_param

            # 2) Held-out Posterior Predictive Error (PPE) on unseen design D7 (or D3 if basic_all)
            heldout_d_idx = 7  # D7 (Standard + pre-added P) is unseen in BOTH D0+D2+D5+D6 and basic_all!
            heldout_des = grammar.catalog[heldout_d_idx]
            n_ppe_cases = min(300, N)
            n_ppc_draws = 16
            t_eval_ppe = np.linspace(0.0, 1.0, 20)
            t_idx_ppe = np.round(np.linspace(0, trajs.shape[2] - 1, 20)).astype(np.int64)
            obs_ch = get_observable_channel_indices(grammar)
            ch_AP = [obs_ch.index(grammar.species_idx["A"]), obs_ch.index(grammar.species_idx["P"])]

            # Sample (m_i, theta_i) ~ q(m, theta | D) for n_ppe_cases
            draw_mids = []
            draw_f_idx = []
            for ci in range(n_ppe_cases):
                ms = rng.choice(grammar.n_mechs, size=n_ppc_draws, p=probs_m[ci])
                draw_mids.append(ms)
                draw_f_idx.append(np.full(n_ppc_draws, ci, dtype=np.int64))
            draw_mids_flat = np.concatenate(draw_mids).astype(np.int32)
            draw_f_idx_flat = np.concatenate(draw_f_idx)

            f_draw = f_all[draw_f_idx_flat].to(device)
            m_draw_t = torch.from_numpy(draw_mids_flat).long().to(device)
            th_draw = model.sample_theta(f_draw, G_last, m_draw_t, num_samples=1).cpu().numpy().astype(np.float64)

            y_pred, st_pred, _, _ = sim.simulate_designs(
                draw_mids_flat, th_draw, [heldout_des], t_eval_ppe, compute_flux=False
            )
            y_pred_AP = y_pred[:, 0, :, [grammar.species_idx["A"], grammar.species_idx["P"]]]
            y_pred_AP = y_pred_AP.reshape(n_ppe_cases, n_ppc_draws, 20, 2)
            st_ok = (st_pred[:, 0] == 0).reshape(n_ppe_cases, n_ppc_draws)

            y_true_AP = trajs[:n_ppe_cases, heldout_d_idx][:, t_idx_ppe][:, :, ch_AP].astype(np.float64)
            ppe_rmses = []
            for ci in range(n_ppe_cases):
                ok_c = st_ok[ci]
                if np.any(ok_c):
                    mean_pred = np.mean(y_pred_AP[ci, ok_c], axis=0)
                    ppe_rmses.append(float(np.sqrt(np.mean((mean_pred - y_true_AP[ci]) ** 2))))
            entry["heldout_D7_ppe_rmse_median"] = float(np.median(ppe_rmses))
            entry["heldout_D7_ppe_rmse_mean"] = float(np.mean(ppe_rmses))

        results["regimes"][rname] = entry

    return results


def evaluate_enum_bic_comparison(
    model_quomech: QuoMechNet,
    model_flat: QuoMechNet,
    grammar: ReactionGrammar,
    sim: ODESimulator,
    test_dir: Path,
    equiv_tiers: Dict[str, Dict],
    device: torch.device,
    n_cases: int = 50,
    design_indices: Sequence[int] = (0, 2, 5, 6),  # D0, D2, D5, D6 (core basic subset)
    tier_key: str = "basic",
    sigma_rel: float = 0.01,
    seed: int = 4242,
) -> Dict:
    """50-case head-to-head comparison of QuoMech vs Flat NPE vs 64-core CPU Enum + LM + BIC."""
    rng = np.random.default_rng(seed)
    all_mids = np.load(test_dir / "mech_ids.npy")
    all_thetas = np.load(test_dir / "thetas.npy")
    all_rand = np.load(test_dir / "rand_des.npy")
    all_trajs = np.load(test_dir / "trajs.npy")

    # Select 50 diverse test cases spread across mechanisms
    sel_mechs = np.linspace(0, grammar.n_mechs - 1, n_cases, dtype=int)
    sel_idx = np.array([np.where(all_mids == m)[0][0] for m in sel_mechs], dtype=np.int64)

    mids = all_mids[sel_idx].astype(np.int64)
    trajs = all_trajs[sel_idx]
    rand_des = all_rand[sel_idx]
    slots_list = [list(design_indices) for _ in range(n_cases)]

    df, tm, ov, om, tp, ep, raw_noisy = pack_fixed_test_batch(
        grammar, trajs, rand_des, slots_list, sigma_rel=sigma_rel, n_t_obs=20, p_only=False, seed=seed
    )

    # 1. Score QuoMech and Flat NPE on GPU
    obs_ch = get_observable_channel_indices(grammar)
    eq = equiv_tiers[tier_key]["primary"]
    m2c = np.array(eq["mech_to_class"], dtype=np.int64)
    n_cls = eq["n_classes"]
    true_c = m2c[mids]

    nn_res = {}
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    for name, net in [("QuoMech", model_quomech), ("Flat_NPE", model_flat)]:
        net.eval()
        # Warmup
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_bf16, dtype=torch.bfloat16):
            _ = net.predict_posterior(
                torch.from_numpy(df[:4]).to(device),
                torch.from_numpy(tm[:4]).to(device),
                torch.from_numpy(ov[:4]).to(device),
                torch.from_numpy(om[:4]).to(device),
                torch.from_numpy(tp[:4]).to(device),
                torch.from_numpy(ep[:4]).to(device),
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_bf16, dtype=torch.bfloat16):
            pr, _, _ = net.predict_posterior(
                torch.from_numpy(df).to(device),
                torch.from_numpy(tm).to(device),
                torch.from_numpy(ov).to(device),
                torch.from_numpy(om).to(device),
                torch.from_numpy(tp).to(device),
                torch.from_numpy(ep).to(device),
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        probs_m = pr.cpu().numpy()
        probs_c = aggregate_class_probs(probs_m, m2c, n_cls)
        nn_res[name] = {
            "mech_top1": float(np.mean(np.argmax(probs_m, axis=1) == mids)),
            "mech_top3": float(np.mean(np.any(np.argpartition(probs_m, -3, axis=1)[:, -3:] == mids[:, None], axis=1))),
            "class_top1": float(np.mean(np.argmax(probs_c, axis=1) == true_c)),
            "class_top3": float(np.mean(np.any(np.argpartition(probs_c, -3, axis=1)[:, -3:] == true_c[:, None], axis=1))),
            "class_ece": compute_ece(probs_c, true_c),
            "total_time_s": dt,
            "ms_per_case": 1000.0 * dt / n_cases,
        }

    # 2. 64-core CPU Enum + LM + BIC (fit all 151 mechanisms x 4 starts per case)
    designs = [grammar.catalog[i] for i in design_indices]
    t_eval = np.linspace(0.0, 1.0, 20)
    M = grammar.n_mechs
    P = grammar.n_params

    # Flatten raw_noisy targets and inv_sigma for each case in the exact order expected by fit_lm_batch
    targets_cases = []
    inv_sig_chunks = []
    for e, des in enumerate(designs):
        cat0 = des.cat_mult * grammar.c0_base
        for sp in des.obs_species:
            scale = cat0 if grammar.species[sp].is_catalyst else des.A0
            inv_sig_chunks.append(np.full(20, 1.0 / (sigma_rel * scale), dtype=np.float64))
    inv_sig_1d = np.concatenate(inv_sig_chunks)
    n_res = len(inv_sig_1d)

    for ci in range(n_cases):
        chunks = []
        for e, des in enumerate(designs):
            for sp in des.obs_species:
                ch_i = obs_ch.index(sp)
                chunks.append(raw_noisy[ci, e, :, ch_i])
        targets_cases.append(np.concatenate(chunks))
    targets_cases = np.stack(targets_cases, axis=0)  # (n_cases, n_res)

    # Build a small curve library (64 per mech from test split excluded indices or prior) for 2 NN starts + 2 random starts
    # Use test indices NOT in sel_idx (or fresh prior draws -> zero test leakage!)
    lib_th = np.zeros((M, 64, P), dtype=np.float64)
    for m in range(M):
        m_idxs = np.where(all_mids == m)[0]
        # Take indices 1..65 (sel_idx used index 0!)
        lib_th[m] = all_thetas[m_idxs[1:65]].astype(np.float64)
    mids_lib = np.repeat(np.arange(M, dtype=np.int32), 64)
    lib_flat, _, _ = build_design_targets(
        grammar, sim, mids_lib, lib_th.reshape(M * 64, P), designs, t_eval, sigma_rel
    )
    lib_scaled = (lib_flat * inv_sig_1d[None, :]).reshape(M, 64, n_res)
    lib_sq = np.sum(lib_scaled * lib_scaled, axis=2)

    lb_all = grammar.mech_prior_mean - 3.5 * grammar.mech_prior_std
    ub_all = grammar.mech_prior_mean + 3.5 * grammar.mech_prior_std
    k_params_by_mech = np.sum(grammar.mech_param_mask, axis=1).astype(np.float64)

    # Build all n_cases * 151 tasks
    n_tasks = n_cases * M
    task_mids = np.tile(np.arange(M, dtype=np.int32), n_cases)
    task_tgts = np.repeat(targets_cases, M, axis=0)
    task_invs = np.broadcast_to(inv_sig_1d[None, :], (n_tasks, n_res)).copy()
    task_inits = np.empty((n_tasks, 4, P), dtype=np.float64)

    t_bic_start = time.perf_counter()
    tgts_scaled = targets_cases * inv_sig_1d[None, :]
    for ci in range(n_cases):
        q = tgts_scaled[ci]
        q_sq = float(np.dot(q, q))
        for m in range(M):
            d2 = q_sq + lib_sq[m] - 2.0 * (lib_scaled[m] @ q)
            nn2 = np.argpartition(d2, 2)[:2]
            r2 = rng.integers(0, 64, size=2)
            t_i = ci * M + m
            task_inits[t_i, 0:2] = lib_th[m, nn2]
            task_inits[t_i, 2:4] = lib_th[m, r2]

    rms_all, _ = sim.fit_lm_batch(
        task_mids, designs, t_eval, task_tgts, task_invs, task_inits,
        lb_all=lb_all, ub_all=ub_all,
        rtol=1e-5, atol=1e-9, max_ode_steps=10000, max_lm_iters=35,
        early_stop_rms=-1.0,  # full minimization for exact BIC!
    )
    dt_bic = time.perf_counter() - t_bic_start

    rms_mat = rms_all.reshape(n_cases, M)
    chi2_mat = (rms_mat ** 2) * n_res
    bic_mat = chi2_mat + k_params_by_mech[None, :] * np.log(n_res)
    # Convert BIC to posterior weights w_m proportional to exp(-0.5 * (BIC - min_BIC))
    bic_shift = bic_mat - np.min(bic_mat, axis=1, keepdims=True)
    w_bic = np.exp(-0.5 * np.clip(bic_shift, 0.0, 500.0))
    w_bic /= np.sum(w_bic, axis=1, keepdims=True)
    w_bic_c = aggregate_class_probs(w_bic, m2c, n_cls)

    bic_res = {
        "mech_top1": float(np.mean(np.argmax(w_bic, axis=1) == mids)),
        "mech_top3": float(np.mean(np.any(np.argpartition(w_bic, -3, axis=1)[:, -3:] == mids[:, None], axis=1))),
        "class_top1": float(np.mean(np.argmax(w_bic_c, axis=1) == true_c)),
        "class_top3": float(np.mean(np.any(np.argpartition(w_bic_c, -3, axis=1)[:, -3:] == true_c[:, None], axis=1))),
        "class_ece": compute_ece(w_bic_c, true_c),
        "total_time_s": dt_bic,
        "ms_per_case": 1000.0 * dt_bic / n_cases,
        "speedup_quomech_vs_bic": dt_bic / max(nn_res["QuoMech"]["total_time_s"], 1e-9),
    }

    return {
        "n_cases": n_cases,
        "designs": [grammar.catalog[i].name for i in design_indices],
        "tier": tier_key,
        "QuoMech": nn_res["QuoMech"],
        "Flat_NPE": nn_res["Flat_NPE"],
        "Enum_BIC": bic_res,
    }
