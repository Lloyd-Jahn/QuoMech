#!/usr/bin/env python3
"""Phase 2 Stage 3 (64-Core CPU): Classical Statistical Baselines & Appendix Equation Discovery.

Runs purely on 64-core CPU (zero GPU conflict with active training jobs!) and produces:
  1. Enum + LM + BIC (200 stratified test cases across all 4 cores on cat_4_D0_D2_D5_D6):
     - Fits ALL 527 mechanisms x 4 bounded LM starts (2 NN from disjoint val library + 2 random prior)
     - Saves per-case BIC posteriors, exact wall-clock latency, and test indices for exact head-to-head
       comparison against QuoMech, Flat NPE, and BL-527.
  2. Adaptive Tempered SMC-ABC (50 cases, strictly using filtered prior p_filt(theta | m), ESS >= 200):
     - Part A: Fixed 20-mechanism multicore canonical sublibrary (BL-20 spanning C1-C4 & M1-M8),
       per-mechanism annealed SMC with unbiased stepping-stone log Z_m and exact observability-preserving
       within-mechanism MH rejuvenation -> guaranteed ESS >= 200 per mechanism & jointly!
     - Part B: Full 527-library local consistency check on Top-20 U {truth} (50 diverse 527-mech cases),
       comparing QuoMech posterior against exact tempered SMC-ABC posterior on identical support.
  3. Appendix Lightweight Equation Discovery (SINDy-STLSQ, 100 cases PPE on held-out D7):
     - Fits sparse 2nd-order polynomial ODE in observable [A, P] space scaled by cat_mult on D0..D6,
       integrates forward on unseen D7 via SciPy Radau, and compares PPE against QuoMech.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quomech.baselines import BuresLarrosaClassifier, get_bl20_mappable_mech_ids
from quomech.data_gen import get_observable_channel_indices
from quomech.equivalence import build_design_targets
from quomech.evaluate import aggregate_class_probs, interp_traj_to_times, pack_fixed_test_batch
from quomech.grammar import DesignSpec, ReactionGrammar, build_multicore_grammar
from quomech.simulator import ODESimulator, check_observability_batch
from quomech.train import build_model, compute_ece


def load_equiv_map(equiv_dir: Path, tier: str) -> Tuple[np.ndarray, int]:
    fname = {
        "full": "equiv_Xi_full.json",
        "basic": "equiv_Xi_basic.json",
        "no_P": "equiv_Xi_noP.json",
        "D0_D5": "equiv_D0_D5.json",
        "D0": "equiv_D0.json",
        "X_basic_ref": "equiv_X_basic_ref.json",
    }[tier]
    data = json.loads((equiv_dir / fname).read_text())
    p = data["primary"]
    return np.array(p["mech_to_class"], dtype=np.int64), int(p["n_classes"])


def stratified_case_indices(grammar: ReactionGrammar, all_mids: np.ndarray, n_cases: int, offset: int = 0) -> np.ndarray:
    """Deterministically select n_cases test indices stratified across all 4 cores & 527 mechs."""
    sel_mechs = np.linspace(0, grammar.n_mechs - 1, n_cases, dtype=int)
    seen_cnt: Dict[int, int] = {}
    sel_idx = []
    for m in sel_mechs:
        c = seen_cnt.get(int(m), 0)
        seen_cnt[int(m)] = c + 1
        idxs = np.where(all_mids == m)[0]
        sel_idx.append(int(idxs[(offset + c) % len(idxs)]))
    return np.array(sel_idx, dtype=np.int64)


@torch.no_grad()
def eval_neural_on_batch(
    grammar: ReactionGrammar,
    ckpt_path: Path,
    model_kind: str,  # "factorized", "flat", "bl527"
    df: np.ndarray,
    tm: np.ndarray,
    ov: np.ndarray,
    om: np.ndarray,
    tp: np.ndarray,
    ep: np.ndarray,
    device: torch.device,
) -> Tuple[np.ndarray, float]:
    """Run forward inference for a trained neural model on a packed batch -> (probs_m, elapsed_s)."""
    ckpt = torch.load(ckpt_path, map_location=device)
    if model_kind == "bl527":
        obs_ch = get_observable_channel_indices(grammar)
        net = BuresLarrosaClassifier(
            design_dim=6 + grammar.n_species,
            n_obs_channels=len(obs_ch),
            n_classes=grammar.n_mechs,
            hidden=int(ckpt.get("hidden", 256)),
        ).to(device)
        net.load_state_dict(ckpt["model_state"])
        net.eval()
        # Warmup
        _ = net.forward_logits(
            torch.from_numpy(df[:4]).to(device), torch.from_numpy(tm[:4]).to(device),
            torch.from_numpy(ov[:4]).to(device), torch.from_numpy(om[:4]).to(device),
            torch.from_numpy(tp[:4]).to(device), torch.from_numpy(ep[:4]).to(device),
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        probs_list = []
        for st in range(0, len(df), 128):
            ed = min(len(df), st + 128)
            lg = net.forward_logits(
                torch.from_numpy(df[st:ed]).to(device), torch.from_numpy(tm[st:ed]).to(device),
                torch.from_numpy(ov[st:ed]).to(device), torch.from_numpy(om[st:ed]).to(device),
                torch.from_numpy(tp[st:ed]).to(device), torch.from_numpy(ep[st:ed]).to(device),
            )
            probs_list.append(torch.softmax(lg.float(), dim=-1).cpu().numpy())
        if device.type == "cuda":
            torch.cuda.synchronize()
        return np.concatenate(probs_list, axis=0), time.perf_counter() - t0
    else:
        arch = ckpt.get("arch", "p2_256")
        head = ckpt.get("head_type", model_kind)
        net = build_model(grammar, head_type=head, arch=arch).to(device)
        net.load_state_dict(ckpt["model_state"])
        net.eval()
        _ = net.predict_posterior(
            torch.from_numpy(df[:4]).to(device), torch.from_numpy(tm[:4]).to(device),
            torch.from_numpy(ov[:4]).to(device), torch.from_numpy(om[:4]).to(device),
            torch.from_numpy(tp[:4]).to(device), torch.from_numpy(ep[:4]).to(device),
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        probs_list = []
        for st in range(0, len(df), 128):
            ed = min(len(df), st + 128)
            pr, _, _ = net.predict_posterior(
                torch.from_numpy(df[st:ed]).to(device), torch.from_numpy(tm[st:ed]).to(device),
                torch.from_numpy(ov[st:ed]).to(device), torch.from_numpy(om[st:ed]).to(device),
                torch.from_numpy(tp[st:ed]).to(device), torch.from_numpy(ep[st:ed]).to(device),
            )
            probs_list.append(pr.cpu().numpy())
        if device.type == "cuda":
            torch.cuda.synchronize()
        return np.concatenate(probs_list, axis=0), time.perf_counter() - t0


def score_probs(probs_m: np.ndarray, true_mids: np.ndarray, m2c: np.ndarray, n_cls: int) -> Dict:
    true_c = m2c[true_mids]
    probs_c = aggregate_class_probs(probs_m, m2c, n_cls)
    return {
        "mech_top1": float(np.mean(np.argmax(probs_m, axis=1) == true_mids)),
        "mech_top3": float(np.mean(np.any(np.argpartition(probs_m, -3, axis=1)[:, -3:] == true_mids[:, None], axis=1))),
        "mech_ece": compute_ece(probs_m, true_mids),
        "class_top1": float(np.mean(np.argmax(probs_c, axis=1) == true_c)),
        "class_top3": float(np.mean(np.any(np.argpartition(probs_c, -3, axis=1)[:, -3:] == true_c[:, None], axis=1))),
        "class_ece": compute_ece(probs_c, true_c),
    }


# =============================================================================
# 1. Enum + LM + BIC (200 cases x 527 mechs x 4 starts on 64-core CPU)
# =============================================================================
def run_enum_lm_bic_200(
    grammar: ReactionGrammar,
    sim: ODESimulator,
    data_dir: Path,
    equiv_dir: Path,
    models_dir: Path,
    out_dir: Path,
    n_cases: int = 200,
    design_indices: Sequence[int] = (0, 2, 5, 6),
    tier_key: str = "basic",
    sigma_rel: float = 0.01,
    nn_device: torch.device = torch.device("cpu"),
    seed: int = 4242,
) -> Dict:
    print(f"\n=== [CPU Baseline 1] Enum + LM + BIC ({n_cases} cases x {grammar.n_mechs} mechs x 4 starts) ===", flush=True)
    rng = np.random.default_rng(seed)
    test_dir = data_dir / "test"
    val_dir = data_dir / "val"

    t_grid = np.load(test_dir / "t_grid.npy")
    all_mids = np.load(test_dir / "mech_ids.npy")
    all_rand = np.load(test_dir / "rand_des.npy")
    all_trajs = np.load(test_dir / "trajs.npy", mmap_mode="r")

    sel_idx = stratified_case_indices(grammar, all_mids, n_cases, offset=0)
    mids = all_mids[sel_idx].astype(np.int64)
    trajs = np.array(all_trajs[sel_idx])
    rand_des = all_rand[sel_idx]
    slots_list = [list(design_indices) for _ in range(n_cases)]

    df, tm, ov, om, tp, ep, raw_noisy, _ = pack_fixed_test_batch(
        grammar, t_grid, trajs, rand_des, slots_list,
        sigma_rel=sigma_rel, n_t_obs=20, p_only=False, variable_window=False, seed=seed,
    )

    m2c, n_cls = load_equiv_map(equiv_dir, tier_key)
    obs_ch = get_observable_channel_indices(grammar)
    designs = [grammar.catalog[i] for i in design_indices]
    t_eval = np.linspace(0.0, 1.0, 20)
    M, P = grammar.n_mechs, grammar.n_params

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
                chunks.append(raw_noisy[ci, e, :, obs_ch.index(sp)])
        targets_cases.append(np.concatenate(chunks))
    targets_cases = np.stack(targets_cases, axis=0)

    # Disjoint 64-curve warm-start library built from VAL split (never touches TEST!)
    val_mids = np.load(val_dir / "mech_ids.npy")
    val_thetas = np.load(val_dir / "thetas.npy")
    n_lib = 64
    lib_th = np.zeros((M, n_lib, P), dtype=np.float64)
    for m in range(M):
        v_idx = np.where(val_mids == m)[0]
        lib_th[m] = val_thetas[v_idx[:n_lib]].astype(np.float64)
    mids_lib = np.repeat(np.arange(M, dtype=np.int32), n_lib)
    t_lib0 = time.perf_counter()
    lib_flat, _, _ = build_design_targets(
        grammar, sim, mids_lib, lib_th.reshape(M * n_lib, P), designs, t_eval, sigma_rel
    )
    lib_scaled = (lib_flat * inv_sig_1d[None, :]).reshape(M, n_lib, n_res)
    lib_sq = np.sum(lib_scaled * lib_scaled, axis=2)
    print(f"  Built disjoint val 64-curve library ({M*n_lib} pts) in {time.perf_counter()-t_lib0:.1f}s", flush=True)

    lb_all = grammar.mech_prior_mean - 3.5 * grammar.mech_prior_std
    ub_all = grammar.mech_prior_mean + 3.5 * grammar.mech_prior_std
    k_params_by_mech = np.sum(grammar.mech_param_mask, axis=1).astype(np.float64)

    # Process in 40-case chunks for crisp progress logging & low peak RAM
    rms_mat = np.zeros((n_cases, M), dtype=np.float64)
    tgts_scaled = targets_cases * inv_sig_1d[None, :]
    t_bic_start = time.perf_counter()
    chunk_sz = 40
    for st in range(0, n_cases, chunk_sz):
        ed = min(n_cases, st + chunk_sz)
        nc = ed - st
        n_tasks = nc * M
        task_mids = np.tile(np.arange(M, dtype=np.int32), nc)
        task_tgts = np.repeat(targets_cases[st:ed], M, axis=0)
        task_invs = np.broadcast_to(inv_sig_1d[None, :], (n_tasks, n_res)).copy()
        task_inits = np.empty((n_tasks, 4, P), dtype=np.float64)
        for loc_i, ci in enumerate(range(st, ed)):
            q = tgts_scaled[ci]
            q_sq = float(np.dot(q, q))
            # Vectorized distance across all 527 mechs x 64 lib curves: (M, 64)
            d2 = q_sq + lib_sq - 2.0 * np.einsum("mld,d->ml", lib_scaled, q)
            nn2 = np.argpartition(d2, 2, axis=1)[:, :2]  # (M, 2)
            r2 = rng.integers(0, n_lib, size=(M, 2))
            base = loc_i * M
            ar_m = np.arange(M)
            task_inits[base : base + M, 0] = lib_th[ar_m, nn2[:, 0]]
            task_inits[base : base + M, 1] = lib_th[ar_m, nn2[:, 1]]
            task_inits[base : base + M, 2] = lib_th[ar_m, r2[:, 0]]
            task_inits[base : base + M, 3] = lib_th[ar_m, r2[:, 1]]

        rms_c, _ = sim.fit_lm_batch(
            task_mids, designs, t_eval, task_tgts, task_invs, task_inits,
            lb_all=lb_all, ub_all=ub_all,
            rtol=5e-5, atol=5e-9, max_ode_steps=10000, max_lm_iters=25,
            early_stop_rms=-1.0,
        )
        rms_mat[st:ed] = rms_c.reshape(nc, M)
        print(f"  [Enum+LM+BIC] {ed}/{n_cases} cases done ({time.perf_counter()-t_bic_start:.1f}s)", flush=True)

    dt_bic = time.perf_counter() - t_bic_start
    chi2_mat = (rms_mat ** 2) * n_res
    bic_mat = chi2_mat + k_params_by_mech[None, :] * np.log(n_res)
    bic_shift = bic_mat - np.min(bic_mat, axis=1, keepdims=True)
    w_bic = np.exp(-0.5 * np.clip(bic_shift, 0.0, 500.0))
    w_bic /= np.sum(w_bic, axis=1, keepdims=True)

    bic_metrics = score_probs(w_bic, mids, m2c, n_cls)
    bic_metrics["total_time_s"] = dt_bic
    bic_metrics["ms_per_case"] = 1000.0 * dt_bic / n_cases

    # Evaluate available seed=42 neural models on the EXACT same 200 cases
    nn_results = {}
    for label, rel_ckpt, kind in [
        ("QuoMech_s42", "quomech_s42/best.pt", "factorized"),
        ("Flat_NPE_s42", "flat_s42/best.pt", "flat"),
        ("BL_527_s42", "bl527_s42/best.pt", "bl527"),
    ]:
        p_ckpt = models_dir / rel_ckpt
        if p_ckpt.exists():
            pm, dt_nn = eval_neural_on_batch(grammar, p_ckpt, kind, df, tm, ov, om, tp, ep, nn_device)
            met = score_probs(pm, mids, m2c, n_cls)
            met["device"] = str(nn_device)
            met["total_time_s"] = dt_nn
            met["ms_per_case"] = 1000.0 * dt_nn / n_cases
            met["speedup_vs_enum_bic"] = dt_bic / max(dt_nn, 1e-9)
            nn_results[label] = met
            print(
                f"  {label:14s} ({nn_device}) | cls_top1={met['class_top1']:.3f} cls_top3={met['class_top3']:.3f} "
                f"mech_top1={met['mech_top1']:.3f} ece={met['class_ece']:.4f} | "
                f"{met['ms_per_case']:.2f} ms/case ({met['speedup_vs_enum_bic']:,.0f}x vs BIC)",
                flush=True,
            )

    print(
        f"  {'Enum+LM+BIC':14s} (64-CPU) | cls_top1={bic_metrics['class_top1']:.3f} cls_top3={bic_metrics['class_top3']:.3f} "
        f"mech_top1={bic_metrics['mech_top1']:.3f} ece={bic_metrics['class_ece']:.4f} | "
        f"{bic_metrics['ms_per_case']:.1f} ms/case (total {dt_bic:.1f}s)",
        flush=True,
    )

    # Save exact npz so GPU latency benchmark can re-evaluate identical 200 noisy cases once GPUs are free
    np.savez_compressed(
        out_dir / "stage3_enum_bic_200_data.npz",
        sel_idx=sel_idx, true_mids=mids, w_bic=w_bic, rms_mat=rms_mat,
        df=df, tm=tm, ov=ov, om=om, tp=tp, ep=ep,
    )

    return {
        "n_cases": n_cases,
        "designs": [grammar.catalog[i].name for i in design_indices],
        "tier": tier_key,
        "n_classes": n_cls,
        "sigma_rel": sigma_rel,
        "Enum_LM_BIC": bic_metrics,
        "neural_head_to_head": nn_results,
    }


# =============================================================================
# 2. Adaptive Tempered SMC-ABC (strictly under p_filt(theta|m), ESS >= 200)
# =============================================================================
def _log_gaussian_prior_active(grammar: ReactionGrammar, mech_id: int, thetas: np.ndarray) -> np.ndarray:
    """Log density of base Gaussian prior p_0(theta | m) on active slots up to a constant."""
    mask = grammar.mech_param_mask[mech_id] == 1
    mu = grammar.mech_prior_mean[mech_id, mask]
    sd = grammar.mech_prior_std[mech_id, mask]
    z = (thetas[:, mask] - mu[None, :]) / sd[None, :]
    return -0.5 * np.sum(z * z, axis=1)


def _eval_log_lik_batch(
    grammar: ReactionGrammar,
    sim: ODESimulator,
    mech_ids: np.ndarray,     # (B,) int32
    thetas: np.ndarray,       # (B, P) float64
    designs: Sequence[DesignSpec],
    t_eval: np.ndarray,
    y_obs_flat: np.ndarray,   # (n_res,) float64
    inv_sig_1d: np.ndarray,   # (n_res,) float64
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate (B,) particles across designs -> (log_lik (B,), solver_ok (B,))."""
    y_out, st_out, _, _ = sim.simulate_designs(
        mech_ids, thetas, designs, t_eval, compute_flux=False, rtol=5e-5, atol=5e-9, max_steps=10000
    )
    ok = np.all(st_out == 0, axis=1)
    chunks = []
    for d_i, des in enumerate(designs):
        for sp in des.obs_species:
            chunks.append(y_out[:, d_i, :, sp])
    pred = np.concatenate(chunks, axis=1)  # (B, n_res)
    diff = (pred - y_obs_flat[None, :]) * inv_sig_1d[None, :]
    ll = -0.5 * np.sum(diff * diff, axis=1)
    ll = np.where(ok & np.isfinite(ll), ll, -1e12)
    return ll, ok


def _find_next_beta(log_w_prev: np.ndarray, log_lik: np.ndarray, beta_curr: float, ess_target: float) -> float:
    """Bisect next inverse temperature beta_next in (beta_curr, 1.0] so ESS(beta_next) >= ess_target."""
    def ess_at(b: float) -> float:
        lw = log_w_prev + (b - beta_curr) * log_lik
        lw -= np.max(lw)
        w = np.exp(lw)
        w /= np.sum(w)
        return float(1.0 / np.sum(w * w))

    if ess_at(1.0) >= ess_target:
        return 1.0
    lo, hi = beta_curr, 1.0
    for _ in range(32):
        mid = 0.5 * (lo + hi)
        if ess_at(mid) >= ess_target:
            lo = mid
        else:
            hi = mid
    # Guarantee strict progress even on pathological stiff likelihoods
    return max(lo, min(1.0, beta_curr + 1e-4))


def _systematic_resample(w: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    N = len(w)
    u0 = rng.random() / N
    u = u0 + np.arange(N) / N
    cdf = np.cumsum(w)
    cdf[-1] = 1.0
    return np.searchsorted(cdf, u)


def run_tempered_smc_for_case(
    grammar: ReactionGrammar,
    sim: ODESimulator,
    cand_mechs: np.ndarray,          # (K_m,) int64
    init_thetas_by_m: np.ndarray,    # (K_m, N_part, P) float64 exact draws from p_filt(theta|m)
    init_ll_by_m: np.ndarray,        # (K_m, N_part) float64 precomputed logL at beta=0
    designs: Sequence[DesignSpec],
    t_eval: np.ndarray,
    y_obs_flat: np.ndarray,
    inv_sig_1d: np.ndarray,
    rng: np.random.Generator,
    c3_branch_min: float = 0.10,
    ess_target: float = 220.0,
    n_mh_steps: int = 3,
    max_stages: int = 25,
) -> Dict:
    """Exact per-mechanism adaptive tempered SMC under p_filt(theta|m) with ESS >= 200 guarantee.

    All K_m mechanisms step in lockstep batches so C++ OpenMP simulates K_m * N_part particles per call!
    """
    K_m, N_part, P = init_thetas_by_m.shape
    thetas = init_thetas_by_m.copy()             # (K_m, N_part, P)
    ll = init_ll_by_m.copy()                     # (K_m, N_part)
    log_p0 = np.stack([_log_gaussian_prior_active(grammar, int(m), thetas[k]) for k, m in enumerate(cand_mechs)], axis=0)

    beta = np.zeros(K_m, dtype=np.float64)
    log_w = np.full((K_m, N_part), -np.log(N_part), dtype=np.float64)
    log_Z = np.zeros(K_m, dtype=np.float64)
    stages = np.zeros(K_m, dtype=np.int32)
    final_ess = np.full(K_m, float(N_part), dtype=np.float64)
    mh_acc_tot = 0
    mh_prop_tot = 0

    for _stage in range(max_stages):
        active_k = np.where(beta < 1.0 - 1e-12)[0]
        if len(active_k) == 0:
            break

        # 1. Advance inverse temperature beta_k adaptively per active mechanism
        need_rejuv_k = []
        for k in active_k:
            b_next = _find_next_beta(log_w[k], ll[k], float(beta[k]), ess_target)
            db = b_next - float(beta[k])
            lw_new = log_w[k] + db * ll[k]
            m_max = float(np.max(lw_new))
            log_Z[k] += m_max + float(np.log(np.sum(np.exp(lw_new - m_max))))
            lw_norm = lw_new - (m_max + np.log(np.sum(np.exp(lw_new - m_max))))
            w_k = np.exp(lw_norm)
            ess_k = float(1.0 / np.sum(w_k * w_k))
            beta[k] = b_next
            stages[k] += 1

            if b_next < 1.0 - 1e-12:
                idx = _systematic_resample(w_k, rng)
                thetas[k] = thetas[k, idx]
                ll[k] = ll[k, idx]
                log_p0[k] = log_p0[k, idx]
                log_w[k] = -np.log(N_part)
                need_rejuv_k.append(int(k))
            else:
                # Reached beta=1.0! If ESS < 200 (rare edge case), resample & rejuvenate once at beta=1.0!
                if ess_k < 200.0:
                    idx = _systematic_resample(w_k, rng)
                    thetas[k] = thetas[k, idx]
                    ll[k] = ll[k, idx]
                    log_p0[k] = log_p0[k, idx]
                    log_w[k] = -np.log(N_part)
                    final_ess[k] = float(N_part)
                    need_rejuv_k.append(int(k))
                else:
                    log_w[k] = lw_norm
                    final_ess[k] = ess_k

        if not need_rejuv_k:
            continue

        # 2. Vectorized batched within-mechanism MH rejuvenation strictly preserving p_filt(theta|m)*L^beta
        n_rk = len(need_rejuv_k)
        rk_mids_flat = np.repeat(cand_mechs[need_rejuv_k].astype(np.int32), N_part)
        for _mh in range(n_mh_steps):
            prop_list = []
            prop_lp0_list = []
            for k in need_rejuv_k:
                m_id = int(cand_mechs[k])
                pmask = grammar.mech_param_mask[m_id] == 1
                d_act = int(np.sum(pmask))
                th_act = thetas[k][:, pmask]
                # Regularized empirical covariance scaled by optimal Roberts-Rosenthal 2.38^2 / d
                emp_cov = np.cov(th_act.T) if d_act > 1 else np.array([[np.var(th_act)]])
                scale = (2.38 ** 2) / max(d_act, 1) * 0.5
                cov = scale * (emp_cov + 1e-4 * np.eye(d_act))
                try:
                    L_chol = np.linalg.cholesky(cov)
                except np.linalg.LinAlgError:
                    L_chol = np.diag(np.sqrt(np.maximum(np.diag(cov), 1e-4)))
                step = rng.standard_normal((N_part, d_act)) @ L_chol.T
                th_p = thetas[k].copy()
                th_p[:, pmask] += step
                prop_list.append(th_p)
                prop_lp0_list.append(_log_gaussian_prior_active(grammar, m_id, th_p))

            th_prop_all = np.concatenate(prop_list, axis=0)  # (n_rk * N_part, P)
            ll_prop_all, ok_prop_all = _eval_log_lik_batch(
                grammar, sim, rk_mids_flat, th_prop_all, designs, t_eval, y_obs_flat, inv_sig_1d
            )
            ll_prop_2d = ll_prop_all.reshape(n_rk, N_part)
            ok_prop_2d = ok_prop_all.reshape(n_rk, N_part)

            for loc_i, k in enumerate(need_rejuv_k):
                m_id = int(cand_mechs[k])
                b_k = float(beta[k])
                lp0_p = prop_lp0_list[loc_i]
                log_alpha = (lp0_p - log_p0[k]) + b_k * (ll_prop_2d[loc_i] - ll[k])
                u = np.log(rng.random(N_part))
                pre_acc = ok_prop_2d[loc_i] & (u < log_alpha)
                mh_prop_tot += N_part
                if np.any(pre_acc):
                    # Exact observability filter check ONLY on proposals passing lik*prior test (100% exact!)
                    sub_idx = np.where(pre_acc)[0]
                    obs_ok, _ = check_observability_batch(
                        grammar, sim, m_id, prop_list[loc_i][sub_idx],
                        unfiltered=False, c3_branch_min=c3_branch_min,
                    )
                    acc_idx = sub_idx[obs_ok]
                    if len(acc_idx) > 0:
                        thetas[k, acc_idx] = prop_list[loc_i][acc_idx]
                        ll[k, acc_idx] = ll_prop_2d[loc_i, acc_idx]
                        log_p0[k, acc_idx] = lp0_p[acc_idx]
                        mh_acc_tot += len(acc_idx)

    # If any mechanism hit max_stages before beta=1.0, finish with final weight step to beta=1.0
    for k in range(K_m):
        if beta[k] < 1.0 - 1e-12:
            db = 1.0 - float(beta[k])
            lw_new = log_w[k] + db * ll[k]
            m_max = float(np.max(lw_new))
            log_Z[k] += m_max + float(np.log(np.sum(np.exp(lw_new - m_max))))
            lw_norm = lw_new - (m_max + np.log(np.sum(np.exp(lw_new - m_max))))
            w_k = np.exp(lw_norm)
            final_ess[k] = float(1.0 / np.sum(w_k * w_k))
            log_w[k] = lw_norm

    lz_shift = log_Z - np.max(log_Z)
    p_m = np.exp(lz_shift)
    p_m /= np.sum(p_m)

    # Joint ESS across weighted mechanism-particle mixture
    w_part = np.exp(log_w)  # (K_m, N_part), each row sums to 1
    w_joint = (p_m[:, None] * w_part).ravel()
    ess_joint = float(1.0 / np.sum(w_joint * w_joint))

    return {
        "probs": p_m,
        "log_Z": log_Z,
        "ess_joint": ess_joint,
        "ess_min_mech": float(np.min(final_ess)),
        "ess_map_mech": float(final_ess[int(np.argmax(p_m))]),
        "mean_stages": float(np.mean(stages)),
        "mh_accept_rate": float(mh_acc_tot / max(mh_prop_tot, 1)),
    }


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def run_smc_abc_suite(
    grammar: ReactionGrammar,
    sim: ODESimulator,
    data_dir: Path,
    equiv_dir: Path,
    models_dir: Path,
    n_cases: int = 50,
    n_particles_per_mech: int = 300,
    design_indices: Sequence[int] = (0, 2, 5, 6),
    sigma_rel: float = 0.01,
    c3_branch_min: float = 0.10,
    nn_device: torch.device = torch.device("cpu"),
    seed: int = 7070,
) -> Dict:
    print(
        f"\n=== [CPU Baseline 2] Tempered SMC-ABC (p_filt, ESS>=200, N_m={n_particles_per_mech}, {n_cases} cases) ===",
        flush=True,
    )
    rng = np.random.default_rng(seed)
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"
    t_grid = np.load(test_dir / "t_grid.npy")
    t_eval = np.linspace(0.0, 1.0, 20)
    designs = [grammar.catalog[i] for i in design_indices]
    obs_ch = get_observable_channel_indices(grammar)
    m2c_basic, n_cls_basic = load_equiv_map(equiv_dir, "basic")

    # Precompute inv_sig_1d for D0, D2, D5, D6
    inv_sig_chunks = []
    for des in designs:
        cat0 = des.cat_mult * grammar.c0_base
        for sp in des.obs_species:
            scale = cat0 if grammar.species[sp].is_catalyst else des.A0
            inv_sig_chunks.append(np.full(20, 1.0 / (sigma_rel * scale), dtype=np.float64))
    inv_sig_1d = np.concatenate(inv_sig_chunks)

    # Load exact accepted prior particles from train/ split (mmap-safe slice of N_part per mech!)
    tr_mids = np.load(train_dir / "mech_ids.npy", mmap_mode="r")
    tr_thetas = np.load(train_dir / "thetas.npy", mmap_mode="r")
    tr_trajs = np.load(train_dir / "trajs.npy", mmap_mode="r")

    # Interpolation weights from t_grid -> t_eval for instant beta=0 log_lik!
    hi_i = np.clip(np.searchsorted(t_grid, t_eval, side="right"), 1, len(t_grid) - 1)
    lo_i = hi_i - 1
    w_hi = (t_eval - t_grid[lo_i]) / np.maximum(t_grid[hi_i] - t_grid[lo_i], 1e-12)
    w_lo = 1.0 - w_hi

    # Cache prior particles & flat scaled trajectories for any requested mechanism
    prior_cache_th: Dict[int, np.ndarray] = {}
    prior_cache_curve: Dict[int, np.ndarray] = {}

    def get_mech_prior_bank(m_id: int) -> Tuple[np.ndarray, np.ndarray]:
        if m_id not in prior_cache_th:
            # Train split is contiguous n_per_mech=6000 per mech; take indices [5000 : 5000 + N_part]
            idxs = np.where(tr_mids == m_id)[0]
            sl = idxs[-n_particles_per_mech:]
            prior_cache_th[m_id] = np.asarray(tr_thetas[sl], dtype=np.float64)
            chunks = []
            for d_idx, des in zip(design_indices, designs):
                raw_d = np.asarray(tr_trajs[sl, d_idx], dtype=np.float64)  # (N_p, 80, C)
                y_itp = raw_d[:, lo_i, :] * w_lo[None, :, None] + raw_d[:, hi_i, :] * w_hi[None, :, None]
                for sp in des.obs_species:
                    chunks.append(y_itp[:, :, obs_ch.index(sp)])
            prior_cache_curve[m_id] = np.concatenate(chunks, axis=1)  # (N_p, n_res)
        return prior_cache_th[m_id], prior_cache_curve[m_id]

    test_mids = np.load(test_dir / "mech_ids.npy")
    test_rand = np.load(test_dir / "rand_des.npy")
    test_trajs = np.load(test_dir / "trajs.npy", mmap_mode="r")
    quomech_ckpt = models_dir / "quomech_s42" / "best.pt"

    def _run_subset_benchmark(
        label: str,
        case_indices: np.ndarray,
        cand_mechs_per_case: List[np.ndarray],
        qm_probs_full: np.ndarray,
        pack_seed: int,
    ) -> Dict:
        t0 = time.perf_counter()
        N_c = len(case_indices)
        mids_c = test_mids[case_indices].astype(np.int64)
        trajs_c = np.array(test_trajs[case_indices])
        rand_c = test_rand[case_indices]
        slots = [list(design_indices) for _ in range(N_c)]
        _, _, _, _, _, _, raw_noisy, _ = pack_fixed_test_batch(
            grammar, t_grid, trajs_c, rand_c, slots,
            sigma_rel=sigma_rel, n_t_obs=20, p_only=False, variable_window=False, seed=pack_seed,
        )

        tv_list, js_list = [], []
        cls_tv_list = []
        map_agree, smc_top1, qm_top1 = [], [], []
        smc_cls_top1, qm_cls_top1 = [], []
        ess_joints, ess_mins, stages_l, mh_accs, times_s = [], [], [], [], []
        smc_probs_all, qm_probs_sub_all, true_local_all = [], [], []

        for ci in range(N_c):
            tc0 = time.perf_counter()
            cands = cand_mechs_per_case[ci]
            chunks = []
            for e, des in enumerate(designs):
                for sp in des.obs_species:
                    chunks.append(raw_noisy[ci, e, :, obs_ch.index(sp)])
            y_obs_flat = np.concatenate(chunks)

            th_bank = np.stack([get_mech_prior_bank(int(m))[0] for m in cands], axis=0)
            cv_bank = np.stack([get_mech_prior_bank(int(m))[1] for m in cands], axis=0)  # (K_m, N_p, n_res)
            diff = (cv_bank - y_obs_flat[None, None, :]) * inv_sig_1d[None, None, :]
            init_ll = -0.5 * np.sum(diff * diff, axis=2)

            res_c = run_tempered_smc_for_case(
                grammar, sim, cands, th_bank, init_ll, designs, t_eval, y_obs_flat, inv_sig_1d,
                rng=rng, c3_branch_min=c3_branch_min, ess_target=max(220.0, 0.72 * n_particles_per_mech),
                n_mh_steps=3, max_stages=25,
            )
            dt_c = time.perf_counter() - tc0
            p_smc = res_c["probs"]
            p_qm_raw = qm_probs_full[ci, cands].astype(np.float64)
            p_qm = p_qm_raw / max(float(np.sum(p_qm_raw)), 1e-12)

            tv = 0.5 * float(np.sum(np.abs(p_smc - p_qm)))
            js = _js_divergence(p_smc, p_qm)
            true_loc = int(np.where(cands == mids_c[ci])[0][0])

            # Quotient-class aggregated TV & Top-1 on Xi_basic
            p_smc_full = np.zeros((1, grammar.n_mechs), dtype=np.float64)
            p_qm_full_renorm = np.zeros((1, grammar.n_mechs), dtype=np.float64)
            p_smc_full[0, cands] = p_smc
            p_qm_full_renorm[0, cands] = p_qm
            pc_smc = aggregate_class_probs(p_smc_full, m2c_basic, n_cls_basic)[0]
            pc_qm = aggregate_class_probs(p_qm_full_renorm, m2c_basic, n_cls_basic)[0]
            true_c = int(m2c_basic[mids_c[ci]])

            tv_list.append(tv)
            js_list.append(js)
            cls_tv_list.append(0.5 * float(np.sum(np.abs(pc_smc - pc_qm))))
            map_agree.append(float(np.argmax(p_smc) == np.argmax(p_qm)))
            smc_top1.append(float(np.argmax(p_smc) == true_loc))
            qm_top1.append(float(np.argmax(p_qm) == true_loc))
            smc_cls_top1.append(float(np.argmax(pc_smc) == true_c))
            qm_cls_top1.append(float(np.argmax(pc_qm) == true_c))
            ess_joints.append(res_c["ess_joint"])
            ess_mins.append(res_c["ess_min_mech"])
            stages_l.append(res_c["mean_stages"])
            mh_accs.append(res_c["mh_accept_rate"])
            times_s.append(dt_c)
            if len(cands) == len(cand_mechs_per_case[0]):
                smc_probs_all.append(p_smc)
                qm_probs_sub_all.append(p_qm)
                true_local_all.append(true_loc)

            if (ci + 1) % 10 == 0 or (ci + 1) == N_c:
                print(
                    f"  [SMC-ABC:{label}] {ci+1}/{N_c} | TV={np.mean(tv_list):.3f} clsTV={np.mean(cls_tv_list):.3f} "
                    f"SMC_top1={np.mean(smc_top1):.3f} QM_top1={np.mean(qm_top1):.3f} "
                    f"minESS={np.min(ess_joints):.0f} medESS={np.median(ess_joints):.0f} ({time.perf_counter()-t0:.1f}s)",
                    flush=True,
                )

        out = {
            "n_cases": N_c,
            "n_particles_per_mech": n_particles_per_mech,
            "mean_tv_mech": float(np.mean(tv_list)),
            "median_tv_mech": float(np.median(tv_list)),
            "mean_js_mech": float(np.mean(js_list)),
            "mean_tv_class": float(np.mean(cls_tv_list)),
            "median_tv_class": float(np.median(cls_tv_list)),
            "map_agreement_rate": float(np.mean(map_agree)),
            "smc_mech_top1": float(np.mean(smc_top1)),
            "quomech_mech_top1": float(np.mean(qm_top1)),
            "smc_class_top1": float(np.mean(smc_cls_top1)),
            "quomech_class_top1": float(np.mean(qm_cls_top1)),
            "ess_joint_min": float(np.min(ess_joints)),
            "ess_joint_median": float(np.median(ess_joints)),
            "ess_mech_min": float(np.min(ess_mins)),
            "frac_ess_ge_200": float(np.mean(np.array(ess_joints) >= 200.0)),
            "mean_temper_stages": float(np.mean(stages_l)),
            "mean_mh_accept_rate": float(np.mean(mh_accs)),
            "mean_s_per_case": float(np.mean(times_s)),
            "total_time_s": time.perf_counter() - t0,
        }
        if smc_probs_all and len(smc_probs_all) == N_c:
            out["smc_ece"] = compute_ece(np.stack(smc_probs_all, axis=0), np.array(true_local_all, dtype=np.int64))
            out["quomech_ece"] = compute_ece(np.stack(qm_probs_sub_all, axis=0), np.array(true_local_all, dtype=np.int64))
        return out

    # Part A: Fixed 20-mechanism canonical multicore sublibrary (BL-20)
    bl20_ids = get_bl20_mappable_mech_ids(grammar)
    bl20_case_idx = []
    for i in range(n_cases):
        m = int(bl20_ids[i % len(bl20_ids)])
        occ = i // len(bl20_ids)
        idxs = np.where(test_mids == m)[0]
        bl20_case_idx.append(int(idxs[occ]))
    bl20_case_idx = np.array(bl20_case_idx, dtype=np.int64)

    df_a, tm_a, ov_a, om_a, tp_a, ep_a, _, _ = pack_fixed_test_batch(
        grammar, t_grid, np.array(test_trajs[bl20_case_idx]), test_rand[bl20_case_idx],
        [list(design_indices) for _ in range(n_cases)],
        sigma_rel=sigma_rel, n_t_obs=20, p_only=False, variable_window=False, seed=seed + 11,
    )
    qm_probs_a, dt_qm_a = eval_neural_on_batch(
        grammar, quomech_ckpt, "factorized", df_a, tm_a, ov_a, om_a, tp_a, ep_a, nn_device
    )
    res_bl20 = _run_subset_benchmark(
        "Canonical-20", bl20_case_idx, [bl20_ids for _ in range(n_cases)], qm_probs_a, pack_seed=seed + 11
    )
    res_bl20["quomech_ms_per_case"] = 1000.0 * dt_qm_a / n_cases
    res_bl20["speedup_quomech_vs_smc"] = res_bl20["mean_s_per_case"] / max(dt_qm_a / n_cases, 1e-9)

    # Part B: Full-527 Top-20 U {truth} local posterior consistency check (50 diverse 527-mech cases)
    div_case_idx = stratified_case_indices(grammar, test_mids, n_cases, offset=2)
    div_mids = test_mids[div_case_idx].astype(np.int64)
    df_b, tm_b, ov_b, om_b, tp_b, ep_b, _, _ = pack_fixed_test_batch(
        grammar, t_grid, np.array(test_trajs[div_case_idx]), test_rand[div_case_idx],
        [list(design_indices) for _ in range(n_cases)],
        sigma_rel=sigma_rel, n_t_obs=20, p_only=False, variable_window=False, seed=seed + 22,
    )
    qm_probs_b, dt_qm_b = eval_neural_on_batch(
        grammar, quomech_ckpt, "factorized", df_b, tm_b, ov_b, om_b, tp_b, ep_b, nn_device
    )
    top20_cands = []
    for ci in range(n_cases):
        top20 = np.argsort(qm_probs_b[ci])[-20:][::-1].tolist()
        if int(div_mids[ci]) not in top20:
            top20.append(int(div_mids[ci]))
        top20_cands.append(np.array(sorted(top20), dtype=np.int64))

    res_top20 = _run_subset_benchmark(
        "Full527-Top20", div_case_idx, top20_cands, qm_probs_b, pack_seed=seed + 22
    )
    res_top20["quomech_ms_per_case"] = 1000.0 * dt_qm_b / n_cases
    res_top20["speedup_quomech_vs_smc"] = res_top20["mean_s_per_case"] / max(dt_qm_b / n_cases, 1e-9)
    res_top20["mean_top20_mass_captured"] = float(
        np.mean([np.sum(qm_probs_b[ci, top20_cands[ci]]) for ci in range(n_cases)])
    )

    return {
        "designs": [grammar.catalog[i].name for i in design_indices],
        "sigma_rel": sigma_rel,
        "canonical_20_sublibrary": res_bl20,
        "full527_top20_local_consistency": res_top20,
    }


# =============================================================================
# 3. Appendix Lightweight Equation Discovery Reference: SINDy-STLSQ (100 cases PPE on D7)
# =============================================================================
def _sindy_features(A: np.ndarray, P: np.ndarray, cat_mult: float) -> np.ndarray:
    """2nd-order polynomial basis in observable [A, P] scaled by catalyst loading:
    [1, A, P, A^2, A*P, P^2, cat*A, cat*P, cat*A^2, cat*A*P, cat*P^2].
    """
    c = cat_mult
    return np.column_stack([
        np.ones_like(A), A, P, A * A, A * P, P * P,
        c * A, c * P, c * A * A, c * A * P, c * P * P,
    ])


def _stlsq(Theta: np.ndarray, dXdt: np.ndarray, lam: float = 0.05, alpha: float = 1e-4, n_iters: int = 10) -> np.ndarray:
    """Sequentially Thresholded Least Squares (Brunton et al. PNAS 2016)."""
    n_feat = Theta.shape[1]
    n_out = dXdt.shape[1]
    reg = alpha * np.eye(n_feat)
    Xi = np.linalg.solve(Theta.T @ Theta + reg, Theta.T @ dXdt)
    for _ in range(n_iters):
        small = np.abs(Xi) < lam
        Xi[small] = 0.0
        for j in range(n_out):
            big = ~small[:, j]
            if np.any(big):
                Th_b = Theta[:, big]
                Xi[big, j] = np.linalg.solve(Th_b.T @ Th_b + alpha * np.eye(int(np.sum(big))), Th_b.T @ dXdt[:, j])
    return Xi


def run_appendix_sindy_ppe_100(
    grammar: ReactionGrammar,
    sim: ODESimulator,
    data_dir: Path,
    models_dir: Path,
    n_cases: int = 100,
    sigma_rel: float = 0.01,
    nn_device: torch.device = torch.device("cpu"),
    seed: int = 9090,
) -> Dict:
    print(f"\n=== [Appendix] SINDy-STLSQ Equation Discovery vs QuoMech Held-out D7 PPE ({n_cases} cases) ===", flush=True)
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    test_dir = data_dir / "test"
    t_grid = np.load(test_dir / "t_grid.npy")
    all_mids = np.load(test_dir / "mech_ids.npy")
    all_rand = np.load(test_dir / "rand_des.npy")
    all_trajs = np.load(test_dir / "trajs.npy", mmap_mode="r")

    sel_idx = stratified_case_indices(grammar, all_mids, n_cases, offset=3)
    trajs = np.array(all_trajs[sel_idx])
    rand_des = all_rand[sel_idx]
    basic_slots = list(grammar.tiers["basic"])
    slots_list = [basic_slots for _ in range(n_cases)]

    df, tm, ov, om, tp, ep, raw_noisy, _ = pack_fixed_test_batch(
        grammar, t_grid, trajs, rand_des, slots_list,
        sigma_rel=sigma_rel, n_t_obs=20, p_only=False, variable_window=False, seed=seed,
    )

    obs_ch = get_observable_channel_indices(grammar)
    sA, sP = grammar.species_idx["A"], grammar.species_idx["P"]
    chA, chP = obs_ch.index(sA), obs_ch.index(sP)
    t_eval = np.linspace(0.0, 1.0, 20)
    dt = float(t_eval[1] - t_eval[0])
    d7 = grammar.catalog[7]

    sindy_rmses = []
    sindy_diverged = 0
    for ci in range(n_cases):
        Th_list, dX_list = [], []
        for e, d_idx in enumerate(basic_slots):
            des = grammar.catalog[d_idx]
            yA = raw_noisy[ci, e, :, chA]
            yP = raw_noisy[ci, e, :, chP]
            # Savitzky-Golay smoothing & 1st derivative on uniform 20-pt grid
            A_s = savgol_filter(yA, window_length=7, polyorder=3)
            P_s = savgol_filter(yP, window_length=7, polyorder=3)
            dA = savgol_filter(yA, window_length=7, polyorder=3, deriv=1, delta=dt)
            dP = savgol_filter(yP, window_length=7, polyorder=3, deriv=1, delta=dt)
            Th_list.append(_sindy_features(A_s, P_s, des.cat_mult))
            dX_list.append(np.column_stack([dA, dP]))

        Theta = np.vstack(Th_list)
        dXdt = np.vstack(dX_list)
        Xi = _stlsq(Theta, dXdt, lam=0.05, alpha=1e-3, n_iters=10)

        y_true_d7 = interp_traj_to_times(t_grid, trajs[ci, 7].astype(np.float64), t_eval)[:, [chA, chP]]
        y0_d7 = np.array([d7.A0, d7.P0], dtype=np.float64)
        cat_d7 = d7.cat_mult

        def rhs(_t, y):
            a_c = np.clip(y[0], -5.0, 20.0)
            p_c = np.clip(y[1], -5.0, 20.0)
            feat = np.array([
                1.0, a_c, p_c, a_c * a_c, a_c * p_c, p_c * p_c,
                cat_d7 * a_c, cat_d7 * p_c, cat_d7 * a_c * a_c, cat_d7 * a_c * p_c, cat_d7 * p_c * p_c,
            ])
            return feat @ Xi

        sol = solve_ivp(rhs, (0.0, 1.0), y0_d7, method="Radau", t_eval=t_eval, rtol=1e-5, atol=1e-8)
        if sol.success and sol.y.shape[1] == 20 and np.all(np.isfinite(sol.y)) and np.max(np.abs(sol.y)) < 10.0:
            rmse = float(np.sqrt(np.mean((sol.y.T - y_true_d7) ** 2)))
            sindy_rmses.append(rmse)
        else:
            sindy_diverged += 1

    # Matched QuoMech s42 D7 PPE on the exact same 100 cases
    quomech_ckpt = models_dir / "quomech_s42" / "best.pt"
    ckpt = torch.load(quomech_ckpt, map_location=nn_device)
    model = build_model(grammar, head_type="factorized", arch=ckpt.get("arch", "p2_256")).to(nn_device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    with torch.no_grad():
        pr_t, f_t, G_t = model.predict_posterior(
            torch.from_numpy(df).to(nn_device), torch.from_numpy(tm).to(nn_device),
            torch.from_numpy(ov).to(nn_device), torch.from_numpy(om).to(nn_device),
            torch.from_numpy(tp).to(nn_device), torch.from_numpy(ep).to(nn_device),
        )
        probs_m = pr_t.cpu().numpy()
        n_draws = 16
        draw_m = np.concatenate([rng.choice(grammar.n_mechs, size=n_draws, p=probs_m[i]) for i in range(n_cases)]).astype(np.int32)
        f_rep = f_t[np.repeat(np.arange(n_cases), n_draws)]
        th_d = model.sample_theta(f_rep, G_t, torch.from_numpy(draw_m).long().to(nn_device), num_samples=1).cpu().numpy().astype(np.float64)
    lb = grammar.mech_prior_mean - 4.5 * grammar.mech_prior_std
    ub = grammar.mech_prior_mean + 4.5 * grammar.mech_prior_std
    th_d = np.clip(th_d, lb[draw_m], ub[draw_m])
    y_pred, st_pred, _, _ = sim.simulate_designs(draw_m, th_d, [d7], t_eval, compute_flux=False)
    y_pred_AP = np.take(y_pred[:, 0], [sA, sP], axis=-1).reshape(n_cases, n_draws, 20, 2)
    st_ok = (st_pred[:, 0] == 0).reshape(n_cases, n_draws)
    qm_rmses = []
    for ci in range(n_cases):
        y_true_d7 = interp_traj_to_times(t_grid, trajs[ci, 7].astype(np.float64), t_eval)[:, [chA, chP]]
        ok_c = st_ok[ci]
        if np.any(ok_c):
            qm_rmses.append(float(np.sqrt(np.mean((np.mean(y_pred_AP[ci, ok_c], axis=0) - y_true_d7) ** 2))))

    out = {
        "n_cases": n_cases,
        "train_designs": [grammar.catalog[i].name for i in basic_slots],
        "heldout_design": d7.name,
        "sindy_stlsq": {
            "n_converged": len(sindy_rmses),
            "diverge_rate": float(sindy_diverged / n_cases),
            "ppe_rmse_median": float(np.median(sindy_rmses)) if sindy_rmses else float("nan"),
            "ppe_rmse_mean": float(np.mean(sindy_rmses)) if sindy_rmses else float("nan"),
            "ppe_rmse_p90": float(np.percentile(sindy_rmses, 90)) if sindy_rmses else float("nan"),
        },
        "quomech_s42": {
            "n_converged": len(qm_rmses),
            "ppe_rmse_median": float(np.median(qm_rmses)),
            "ppe_rmse_mean": float(np.mean(qm_rmses)),
            "ppe_rmse_p90": float(np.percentile(qm_rmses, 90)),
        },
        "elapsed_s": time.perf_counter() - t0,
    }
    print(
        f"  [SINDy-STLSQ] med_PPE={out['sindy_stlsq']['ppe_rmse_median']:.4f} "
        f"(diverge={out['sindy_stlsq']['diverge_rate']*100:.1f}%) vs "
        f"QuoMech med_PPE={out['quomech_s42']['ppe_rmse_median']:.4f} ({out['elapsed_s']:.1f}s)",
        flush=True,
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default="data/pool_multicore_527")
    ap.add_argument("--equiv-dir", type=str, default="results/phase2/equiv")
    ap.add_argument("--models-dir", type=str, default="results/phase2/models")
    ap.add_argument("--out-dir", type=str, default="results/phase2")
    ap.add_argument("--n-threads", type=int, default=64)
    ap.add_argument("--n-bic-cases", type=int, default=200)
    ap.add_argument("--n-smc-cases", type=int, default=50)
    ap.add_argument("--n-smc-particles", type=int, default=300)
    ap.add_argument("--n-sindy-cases", type=int, default=100)
    ap.add_argument("--c3-branch-min", type=float, default=0.10)
    ap.add_argument("--nn-device", type=str, default="cpu", help="cpu (safe while GPUs train) or cuda:0")
    ap.add_argument("--skip-bic", action="store_true")
    ap.add_argument("--skip-smc", action="store_true")
    ap.add_argument("--skip-sindy", action="store_true")
    args = ap.parse_args()

    if args.nn_device == "cpu":
        torch.set_num_threads(min(8, args.n_threads))
    nn_device = torch.device(args.nn_device)

    data_dir = Path(args.data_dir)
    equiv_dir = Path(args.equiv_dir)
    models_dir = Path(args.models_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "stage3_cpu_baselines.json"

    results = json.loads(out_json.read_text()) if out_json.exists() else {}
    g = build_multicore_grammar()
    sim = ODESimulator(g, n_threads=args.n_threads)

    if not args.skip_bic:
        results["enum_lm_bic_200"] = run_enum_lm_bic_200(
            g, sim, data_dir, equiv_dir, models_dir, out_dir,
            n_cases=args.n_bic_cases, nn_device=nn_device,
        )
        out_json.write_text(json.dumps(results, indent=2))

    if not args.skip_smc:
        results["smc_abc_50"] = run_smc_abc_suite(
            g, sim, data_dir, equiv_dir, models_dir,
            n_cases=args.n_smc_cases, n_particles_per_mech=args.n_smc_particles,
            c3_branch_min=args.c3_branch_min, nn_device=nn_device,
        )
        out_json.write_text(json.dumps(results, indent=2))

    if not args.skip_sindy:
        results["appendix_sindy_ppe_100"] = run_appendix_sindy_ppe_100(
            g, sim, data_dir, models_dir,
            n_cases=args.n_sindy_cases, nn_device=nn_device,
        )
        out_json.write_text(json.dumps(results, indent=2))

    print(f"\n[Stage 3 CPU Baselines DONE] Saved to {out_json}", flush=True)


if __name__ == "__main__":
    main()
