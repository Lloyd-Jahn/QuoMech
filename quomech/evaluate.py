"""E1 Evaluation Suite: Equivalence-class Top-1/3, ECE, SBC-KS, Held-out PPE,
50-case Enum+BIC comparison, and millisecond Top-5 Local Posterior Predictive Overlap test (Section 7.4).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
from scipy.stats import kstest
import torch

from quomech.data_gen import build_design_feature_vec, get_observable_channel_indices
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


def interp_traj_to_times(t_grid: np.ndarray, traj_1d_or_2d: np.ndarray, t_query: np.ndarray) -> np.ndarray:
    """Linear interpolation of (n_t_full, C) trajectory onto arbitrary t_query in [0, t_grid[-1]]."""
    hi = np.clip(np.searchsorted(t_grid, t_query, side="right"), 1, len(t_grid) - 1)
    lo = hi - 1
    w_hi = (t_query - t_grid[lo]) / np.maximum(t_grid[hi] - t_grid[lo], 1e-12)
    return (1.0 - w_hi[:, None]) * traj_1d_or_2d[lo] + w_hi[:, None] * traj_1d_or_2d[hi]


def pack_fixed_test_batch(
    grammar: ReactionGrammar,
    t_grid: np.ndarray,
    trajs: np.ndarray,       # (N, 14, 80, C)
    rand_des: np.ndarray,    # (N, 4, 5)
    design_slots_list: List[List[int]],
    sigma_rel: float = 0.01,
    n_t_obs: int = 20,
    p_only: bool = False,
    variable_window: bool = False,
    seed: int = 777,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create standardized noisy observation tensors (with design_dim = 6 + n_species) for N test cases.

    If `variable_window=True`, assigns load-adapted observation windows t_end(xi) = clip(1/cat_mult, 0.4, 4.0)
    matching real multi-window VTNA/RPKA protocols.
    Returns (df, tm, ov, om, tp, ep, raw_noisy, t_ends).
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
    des_dim = 6 + grammar.n_species

    df = np.zeros((N, E_max, des_dim), dtype=np.float32)
    tm = np.zeros((N, E_max, n_t_obs), dtype=np.float32)
    ov = np.zeros((N, E_max, n_t_obs, n_ch), dtype=np.float32)
    om = np.zeros((N, E_max, n_t_obs, n_ch), dtype=np.float32)
    tp = np.ones((N, E_max, n_t_obs), dtype=bool)
    ep = np.ones((N, E_max), dtype=bool)
    raw_noisy = np.zeros((N, E_max, n_t_obs, n_ch), dtype=np.float64)
    t_ends = np.ones((N, E_max), dtype=np.float64)

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

            cat_mult = float(p5[3])
            t_end = float(np.clip(1.0 / max(cat_mult, 0.25), 0.4, 4.0)) if variable_window else 1.0
            t_ends[i, e] = t_end
            t_q = np.linspace(0.0, t_end, n_t_obs)

            df[i, e] = build_design_feature_vec(
                grammar, float(p5[0]), float(p5[1]), float(p5[2]),
                cat_mult, float(p5[4]), t_end, obs_ch, ch_mask,
            )

            y_clean = interp_traj_to_times(t_grid, trajs[i, d_slot].astype(np.float64), t_q)
            A0 = float(p5[0])
            cat0 = cat_mult * grammar.c0_base
            noise_scale = np.where(ch_is_cat, cat0, A0)
            norm_scale = np.where(ch_is_cat, cat0, 1.0)
            y_noisy = y_clean + rng.normal(0.0, 1.0, size=y_clean.shape) * (sigma_rel * noise_scale[None, :])
            raw_noisy[i, e] = y_noisy

            tm[i, e] = t_q.astype(np.float32)
            ov[i, e] = (y_noisy / norm_scale[None, :]).astype(np.float32) * ch_mask[None, :]
            om[i, e] = np.broadcast_to(ch_mask[None, :], (n_t_obs, n_ch))
            tp[i, e] = False

    return df, tm, ov, om, tp, ep, raw_noisy, t_ends


@torch.no_grad()
def evaluate_local_top5_overlap(
    model: QuoMechNet,
    grammar: ReactionGrammar,
    sim: ODESimulator,
    f_all: torch.Tensor,          # (N_cases, d_set) on CPU
    G_mat: torch.Tensor,          # (M, d_g) on GPU
    probs_m: np.ndarray,          # (N_cases, M)
    true_mids: np.ndarray,        # (N_cases,)
    designs_per_case: List[List[Tuple[DesignSpec, float]]],  # (DesignSpec, t_end) per exp
    raw_noisy: np.ndarray,        # (N_cases, E, n_t, n_ch)
    device: torch.device,
    sigma_rel: float = 0.01,
    n_ppc: int = 16,
    top_k: int = 5,
    n_cases: int = 100,
) -> Dict:
    """Section 7.4 real-time millisecond local posterior predictive overlap test on Top-5 mechanisms.

    For each case:
      1. Selects Top-5 mechanisms under q(m | D).
      2. Draws n_ppc=16 parameter samples theta ~ q(theta | m, D) per mechanism (80 total).
      3. Simulates them on D via 64-core CPU ROS2 and computes each mechanism's posterior
         predictive mean curve y_bar_m(D).
      4. Forms the local indistinguishable cluster of mechanisms whose y_bar_m overlaps the MAP
         mechanism within 1-sigma normalized RMS (<= 1.0), and checks misspecification (> 3-sigma
         residual from observed data across all Top-5).
    """
    obs_ch = get_observable_channel_indices(grammar)
    ch_is_cat = np.array([grammar.species[s].is_catalyst for s in obs_ch], dtype=bool)
    n_cases = min(n_cases, len(probs_m))
    n_t = raw_noisy.shape[2]
    lb = grammar.mech_prior_mean - 4.5 * grammar.mech_prior_std
    ub = grammar.mech_prior_mean + 4.5 * grammar.mech_prior_std

    t_per_case_ms = []
    cluster_sizes = []
    cluster_contains_true = []
    cluster_mass = []
    misspec_flags = []
    min_obs_rms_list = []

    for ci in range(n_cases):
        t0 = time.perf_counter()
        top_m = np.argsort(probs_m[ci])[-top_k:][::-1]  # (5,)
        f_rep = f_all[ci : ci + 1].to(device).expand(top_k, -1)
        m_t = torch.from_numpy(top_m.copy()).long().to(device)
        th_samp = model.sample_theta(f_rep, G_mat, m_t, num_samples=n_ppc).cpu().numpy().astype(np.float64)
        # (n_ppc, top_k, P) -> (top_k * n_ppc, P)
        th_flat = th_samp.transpose(1, 0, 2).reshape(top_k * n_ppc, grammar.n_params)
        m_flat = np.repeat(top_m.astype(np.int32), n_ppc)
        th_flat = np.clip(th_flat, lb[m_flat], ub[m_flat])

        # Simulate all experiments in D for this case
        des_list = designs_per_case[ci]
        pred_chunks = []
        obs_chunks = []
        inv_sig_chunks = []

        for e_i, (des, t_end) in enumerate(des_list):
            t_q = np.linspace(0.0, t_end, n_t)
            y0s = np.array([grammar.initial_state(int(m), des) for m in m_flat], dtype=np.float64)
            y_sim, st, _, _ = sim.simulate_batch(
                m_flat, th_flat, y0s, t_q, compute_flux=False, rtol=5e-5, atol=5e-9
            )
            cat0 = des.cat_mult * grammar.c0_base
            for sp in des.obs_species:
                ch_i = obs_ch.index(sp)
                scale = cat0 if grammar.species[sp].is_catalyst else des.A0
                inv_s = 1.0 / (sigma_rel * scale)
                # Posterior predictive mean across valid draws for each of Top-5 mechanisms
                y_sp = y_sim[:, :, sp].reshape(top_k, n_ppc, n_t)
                st_2d = (st == 0).reshape(top_k, n_ppc)
                y_mean = np.zeros((top_k, n_t), dtype=np.float64)
                for k_i in range(top_k):
                    ok_k = st_2d[k_i]
                    y_mean[k_i] = np.mean(y_sp[k_i, ok_k], axis=0) if np.any(ok_k) else 0.0
                pred_chunks.append(y_mean * inv_s)
                obs_chunks.append(raw_noisy[ci, e_i, :, ch_i] * inv_s)

        pred_mat = np.concatenate(pred_chunks, axis=1)  # (top_k, n_res)
        obs_vec = np.concatenate(obs_chunks, axis=0)    # (n_res,)

        # Pairwise RMS between MAP (index 0) and Top-5 mechanisms
        diff_map = pred_mat - pred_mat[0:1]
        rms_to_map = np.sqrt(np.mean(diff_map * diff_map, axis=1))
        local_cluster = [int(top_m[k_i]) for k_i in range(top_k) if rms_to_map[k_i] <= 1.0]

        # Residual to actual noisy observation (misspecification check > 3 sigma)
        obs_rms = np.sqrt(np.mean((pred_mat - obs_vec[None, :]) ** 2, axis=1))
        min_obs_rms = float(np.min(obs_rms))

        t_per_case_ms.append(1000.0 * (time.perf_counter() - t0))
        cluster_sizes.append(len(local_cluster))
        cluster_contains_true.append(int(true_mids[ci]) in local_cluster)
        cluster_mass.append(float(np.sum(probs_m[ci, local_cluster])))
        misspec_flags.append(min_obs_rms > 3.0)
        min_obs_rms_list.append(min_obs_rms)

    return {
        "n_cases": n_cases,
        "median_ms_per_case": float(np.median(t_per_case_ms)),
        "mean_ms_per_case": float(np.mean(t_per_case_ms)),
        "mean_local_cluster_size": float(np.mean(cluster_sizes)),
        "frac_nontrivial_local_cluster": float(np.mean(np.array(cluster_sizes) >= 2)),
        "local_cluster_true_mech_coverage": float(np.mean(cluster_contains_true)),
        "mean_local_cluster_prob_mass": float(np.mean(cluster_mass)),
        "in_dist_false_misspec_alarm_rate": float(np.mean(misspec_flags)),
        "median_min_obs_rms_sigma": float(np.median(min_obs_rms_list)),
    }


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
    run_local_overlap: bool = True,
    seed: int = 2026,
) -> Dict:
    """Comprehensive E1 evaluation on held-out test split (catalog, continuous, variable-window, P-only)."""
    model.eval()
    rng = np.random.default_rng(seed)

    t_grid = np.load(test_dir / "t_grid.npy")
    all_mids = np.load(test_dir / "mech_ids.npy")
    all_thetas = np.load(test_dir / "thetas.npy")
    all_rand = np.load(test_dir / "rand_des.npy")
    all_trajs = np.load(test_dir / "trajs.npy")

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

    # (slots_list, p_only, variable_window, tier_key)
    regimes = {
        "cat_1_D0": ([[0] for _ in range(N)], False, False, "D0"),
        "cat_3_D0_D2_D5": ([[0, 2, 5] for _ in range(N)], False, False, "no_P"),
        "cat_4_D0_D2_D5_D6": ([[0, 2, 5, 6] for _ in range(N)], False, False, "basic"),
        "cat_6_noP_all": ([list(grammar.tiers["no_P"]) for _ in range(N)], False, False, "no_P"),
        "cat_7_basic_all": ([list(grammar.tiers["basic"]) for _ in range(N)], False, False, "basic"),
        "cat_10_full_all": ([list(grammar.tiers["full"]) for _ in range(N)], False, False, "full"),
        "cont_4_var_window": ([[10, 11, 12, 13] for _ in range(N)], False, True, "full"),
        "cat_basic_var_window": ([list(grammar.tiers["basic"]) for _ in range(N)], False, True, "basic"),
        "P_only_basic": ([list(grammar.tiers["basic"]) for _ in range(N)], True, False, "basic"),
    }

    results = {"n_test": N, "regimes": {}}
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    sA, sP = grammar.species_idx["A"], grammar.species_idx["P"]

    for rname, (slots_list, p_only, var_win, tier_key) in regimes.items():
        df, tm, ov, om, tp, ep, raw_noisy, t_ends = pack_fixed_test_batch(
            grammar, t_grid, trajs, rand_des, slots_list,
            sigma_rel=0.01, n_t_obs=20, p_only=p_only, variable_window=var_win, seed=seed + len(rname)
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

        m_top1 = float(np.mean(np.argmax(probs_m, axis=1) == mids))
        m_top3 = float(np.mean(np.any(np.argpartition(probs_m, -3, axis=1)[:, -3:] == mids[:, None], axis=1)))
        m_ece = compute_ece(probs_m, mids)
        m_true_p = float(np.mean(probs_m[np.arange(N), mids]))

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

        if rname in ("cat_4_D0_D2_D5_D6", "cat_7_basic_all", "cont_4_var_window"):
            # 1) Exact SBC-KS (L=99 posterior draws conditioned on true m)
            L_sbc = 99
            ranks_by_param: List[List[float]] = [[] for _ in range(grammar.n_params)]
            for st in range(0, N, batch_size):
                ed = min(N, st + batch_size)
                f_b = f_all[st:ed].to(device)
                m_b = torch.from_numpy(mids[st:ed]).to(device)
                th_samp = model.sample_theta(f_b, G_last, m_b, num_samples=L_sbc).cpu().numpy()
                th_true = thetas[st:ed]
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

            # 2) Held-out Posterior Predictive Error (PPE) on unseen design D7 (t in [0, 1.0])
            heldout_d_idx = 7
            heldout_des = grammar.catalog[heldout_d_idx]
            n_ppe_cases = min(300, N)
            n_ppc_draws = 16
            t_eval_ppe = np.linspace(0.0, 1.0, 20)
            obs_ch = get_observable_channel_indices(grammar)
            ch_AP = [obs_ch.index(sA), obs_ch.index(sP)]

            draw_mids = [rng.choice(grammar.n_mechs, size=n_ppc_draws, p=probs_m[ci]) for ci in range(n_ppe_cases)]
            draw_mids_flat = np.concatenate(draw_mids).astype(np.int32)
            draw_f_idx_flat = np.repeat(np.arange(n_ppe_cases, dtype=np.int64), n_ppc_draws)

            f_draw = f_all[draw_f_idx_flat].to(device)
            m_draw_t = torch.from_numpy(draw_mids_flat).long().to(device)
            th_draw = model.sample_theta(f_draw, G_last, m_draw_t, num_samples=1).cpu().numpy().astype(np.float64)
            lb = grammar.mech_prior_mean - 4.5 * grammar.mech_prior_std
            ub = grammar.mech_prior_mean + 4.5 * grammar.mech_prior_std
            th_draw = np.clip(th_draw, lb[draw_mids_flat], ub[draw_mids_flat])

            y_pred, st_pred, _, _ = sim.simulate_designs(
                draw_mids_flat, th_draw, [heldout_des], t_eval_ppe, compute_flux=False
            )
            # Note: use np.take to keep axis order (B, n_t, 2) instead of NumPy fancy-index axis transpose!
            y_pred_AP = np.take(y_pred[:, 0], [sA, sP], axis=-1).reshape(n_ppe_cases, n_ppc_draws, 20, 2)
            st_ok = (st_pred[:, 0] == 0).reshape(n_ppe_cases, n_ppc_draws)

            ppe_rmses = []
            for ci in range(n_ppe_cases):
                ok_c = st_ok[ci]
                if np.any(ok_c):
                    y_true_AP = interp_traj_to_times(t_grid, trajs[ci, heldout_d_idx].astype(np.float64), t_eval_ppe)[:, ch_AP]
                    mean_pred = np.mean(y_pred_AP[ci, ok_c], axis=0)
                    ppe_rmses.append(float(np.sqrt(np.mean((mean_pred - y_true_AP) ** 2))))
            entry["heldout_D7_ppe_rmse_median"] = float(np.median(ppe_rmses))
            entry["heldout_D7_ppe_rmse_mean"] = float(np.mean(ppe_rmses))

        # 3) Local Top-5 posterior predictive overlap test on continuous variable-window regime (Section 7.4)
        if run_local_overlap and rname == "cont_4_var_window":
            des_per_case = []
            for ci in range(min(100, N)):
                c_list = []
                for e_i in range(4):
                    p5 = rand_des[ci, e_i]
                    d_spec = DesignSpec(
                        "rand", float(p5[0]), float(p5[1]), float(p5[2]),
                        float(p5[3]), int(p5[4] > 0.5), (sA, sP), 1.0, ()
                    )
                    c_list.append((d_spec, float(t_ends[ci, e_i])))
                des_per_case.append(c_list)
            entry["local_top5_overlap"] = evaluate_local_top5_overlap(
                model, grammar, sim, f_all, G_last, probs_m, mids, des_per_case, raw_noisy, device,
                sigma_rel=0.01, n_ppc=16, top_k=5, n_cases=100,
            )

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
    design_indices: Sequence[int] = (0, 2, 5, 6),
    tier_key: str = "basic",
    sigma_rel: float = 0.01,
    seed: int = 4242,
) -> Dict:
    """50-case head-to-head comparison of QuoMech vs Flat NPE vs 64-core CPU Enum + LM + BIC."""
    rng = np.random.default_rng(seed)
    t_grid = np.load(test_dir / "t_grid.npy")
    all_mids = np.load(test_dir / "mech_ids.npy")
    all_thetas = np.load(test_dir / "thetas.npy")
    all_rand = np.load(test_dir / "rand_des.npy")
    all_trajs = np.load(test_dir / "trajs.npy")

    sel_mechs = np.linspace(0, grammar.n_mechs - 1, n_cases, dtype=int)
    sel_idx = np.array([np.where(all_mids == m)[0][0] for m in sel_mechs], dtype=np.int64)

    mids = all_mids[sel_idx].astype(np.int64)
    trajs = all_trajs[sel_idx]
    rand_des = all_rand[sel_idx]
    slots_list = [list(design_indices) for _ in range(n_cases)]

    df, tm, ov, om, tp, ep, raw_noisy, _ = pack_fixed_test_batch(
        grammar, t_grid, trajs, rand_des, slots_list,
        sigma_rel=sigma_rel, n_t_obs=20, p_only=False, variable_window=False, seed=seed
    )

    obs_ch = get_observable_channel_indices(grammar)
    eq = equiv_tiers[tier_key]["primary"]
    m2c = np.array(eq["mech_to_class"], dtype=np.int64)
    n_cls = eq["n_classes"]
    true_c = m2c[mids]

    nn_res = {}
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    for name, net in [("QuoMech", model_quomech), ("Flat_NPE", model_flat)]:
        net.eval()
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

    # 64-core CPU Enum + LM + BIC (all 151 mechanisms x 4 starts per case)
    designs = [grammar.catalog[i] for i in design_indices]
    t_eval = np.linspace(0.0, 1.0, 20)
    M = grammar.n_mechs
    P = grammar.n_params

    inv_sig_chunks = []
    for des in designs:
        cat0 = des.cat_mult * grammar.c0_base
        for sp in des.obs_species:
            scale = cat0 if grammar.species[sp].is_catalyst else des.A0
            inv_sig_chunks.append(np.full(20, 1.0 / (sigma_rel * scale), dtype=np.float64))
    inv_sig_1d = np.concatenate(inv_sig_chunks)
    n_res = len(inv_sig_1d)

    targets_cases = []
    for ci in range(n_cases):
        chunks = []
        for e, des in enumerate(designs):
            for sp in des.obs_species:
                ch_i = obs_ch.index(sp)
                chunks.append(raw_noisy[ci, e, :, ch_i])
        targets_cases.append(np.concatenate(chunks))
    targets_cases = np.stack(targets_cases, axis=0)

    lib_th = np.zeros((M, 64, P), dtype=np.float64)
    for m in range(M):
        m_idxs = np.where(all_mids == m)[0]
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
        rtol=5e-5, atol=5e-9, max_ode_steps=10000, max_lm_iters=25,
        early_stop_rms=-1.0,
    )
    dt_bic = time.perf_counter() - t_bic_start

    rms_mat = rms_all.reshape(n_cases, M)
    chi2_mat = (rms_mat ** 2) * n_res
    bic_mat = chi2_mat + k_params_by_mech[None, :] * np.log(n_res)
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
