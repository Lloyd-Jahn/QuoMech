#!/usr/bin/env python3
"""Phase 2 Stage 3: Full E1 Amortized Inference Main Table (5,270 cases) + BL-20 Protocol + E2 Ablations.

Supports both single-seed arXiv v1 fast-track (--seeds 42) and full 3-seed backfill (--seeds 42,43,44),
computing per-seed results AND multi-seed mean +/- std + 10,000-resample paired Bootstrap 95% CIs.

Evaluates:
  1. Table 1 (E1 Main Suite, 5,270 held-out test cases = 10/mech):
     - QuoMech vs Flat NPE vs Burés & Larrosa (BL-527) across 11 regimes:
       cat_1_D0, cat_3_D0_D2_D5, cat_4_D0_D2_D5_D6, cat_6_noP_all, cat_7_basic_all,
       cat_basic_var_window, cat_10_full_all, cont_4_var_window, P_only_basic,
       plus high-noise robustness (sigma = 2%, 3%).
     - 19D NSF SBC-KS, Held-out D7 PPE (527 cases = 1/mech), and 100ms batched Top-5 Local Overlap (200 cases).
     - Also times GPU inference on the exact 200 Enum+LM+BIC cases (if stage3_enum_bic_200_data.npz exists)
       for apples-to-apples GPU vs 64-core CPU speedup!
  2. BL-20 External Protocol (200 test cases = 10/mech across canonical 20 mechs):
     - BL-20 closed-set CNN vs QuoMech (restricted to BL-20 & open-527) vs Flat NPE (restricted).
  3. Table 2 (E2 Controlled Ablations, 5,270 test cases):
     - (a) OOD 34 held-out 3-module combinations: e2_ood_factorized vs e2_ood_flat
           on both 340 OOD test cases & 4,930 seen test cases (mech Top-1/3, class Top-1, core Top-1, mod F1).
     - (b) Discrete fixed-window ablation: e2_disc_fixed vs QuoMech on catalog vs continuous/variable-window.
     - (c) Unfiltered-prior ablation: e2_unfiltered vs QuoMech on filtered test split (class Top-1, ECE, SBC-KS, D7 PPE).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
from scipy.stats import kstest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quomech.baselines import BuresLarrosaClassifier, get_bl20_mappable_mech_ids
from quomech.data_gen import get_observable_channel_indices
from quomech.evaluate import (
    aggregate_class_probs,
    evaluate_local_top5_overlap,
    interp_traj_to_times,
    pack_fixed_test_batch,
)
from quomech.grammar import DesignSpec, ReactionGrammar, build_multicore_grammar
from quomech.simulator import ODESimulator
from quomech.train import build_model, compute_ece
from scripts.p2_stage2_train import get_ood_heldout_3mod_ids


def paired_bootstrap_ci(diff_arr: np.ndarray, n_boot: int = 10000, seed: int = 42) -> Dict[str, float]:
    x = np.asarray(diff_arr, dtype=np.float64)
    n = len(x)
    if n == 0:
        return {"n": 0, "mean": 0.0, "ci95_lo": 0.0, "ci95_hi": 0.0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    bm = np.mean(x[idx], axis=1)
    return {
        "n": int(n),
        "mean": float(np.mean(x)),
        "ci95_lo": float(np.percentile(bm, 2.5)),
        "ci95_hi": float(np.percentile(bm, 97.5)),
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


def load_model(grammar: ReactionGrammar, ckpt_path: Path, kind: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if kind in ("bl527", "bl20"):
        obs_ch = get_observable_channel_indices(grammar)
        net = BuresLarrosaClassifier(
            design_dim=6 + grammar.n_species,
            n_obs_channels=len(obs_ch),
            n_classes=int(ckpt["n_classes"]),
            hidden=int(ckpt.get("hidden", 256)),
        ).to(device)
        net.load_state_dict(ckpt["model_state"])
        net.eval()
        return net, ckpt, False
    arch = ckpt.get("arch", "p2_256")
    head = ckpt.get("head_type", kind)
    net = build_model(grammar, head_type=head, arch=arch).to(device)
    net.load_state_dict(ckpt["model_state"])
    net.eval()
    return net, ckpt, True


@torch.no_grad()
def predict_batch_probs(
    net,
    has_flow: bool,
    df: np.ndarray,
    tm: np.ndarray,
    ov: np.ndarray,
    om: np.ndarray,
    tp: np.ndarray,
    ep: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> Tuple[np.ndarray, Optional[torch.Tensor], Optional[torch.Tensor], float]:
    """Run batched GPU inference -> (probs (N, C), f_all or None, G_mat or None, elapsed_s)."""
    N = len(df)
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    # Small warmup
    with torch.cuda.amp.autocast(enabled=use_bf16, dtype=torch.bfloat16):
        if has_flow:
            _ = net.predict_posterior(
                torch.from_numpy(df[:4]).to(device), torch.from_numpy(tm[:4]).to(device),
                torch.from_numpy(ov[:4]).to(device), torch.from_numpy(om[:4]).to(device),
                torch.from_numpy(tp[:4]).to(device), torch.from_numpy(ep[:4]).to(device),
            )
        else:
            _ = net.forward_logits(
                torch.from_numpy(df[:4]).to(device), torch.from_numpy(tm[:4]).to(device),
                torch.from_numpy(ov[:4]).to(device), torch.from_numpy(om[:4]).to(device),
                torch.from_numpy(tp[:4]).to(device), torch.from_numpy(ep[:4]).to(device),
            )
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    probs_list, f_list = [], []
    G_last = None
    for st in range(0, N, batch_size):
        ed = min(N, st + batch_size)
        with torch.cuda.amp.autocast(enabled=use_bf16, dtype=torch.bfloat16):
            if has_flow:
                pr, f_b, G_b = net.predict_posterior(
                    torch.from_numpy(df[st:ed]).to(device), torch.from_numpy(tm[st:ed]).to(device),
                    torch.from_numpy(ov[st:ed]).to(device), torch.from_numpy(om[st:ed]).to(device),
                    torch.from_numpy(tp[st:ed]).to(device), torch.from_numpy(ep[st:ed]).to(device),
                )
                probs_list.append(pr.cpu().numpy())
                f_list.append(f_b.cpu())
                G_last = G_b
            else:
                lg = net.forward_logits(
                    torch.from_numpy(df[st:ed]).to(device), torch.from_numpy(tm[st:ed]).to(device),
                    torch.from_numpy(ov[st:ed]).to(device), torch.from_numpy(om[st:ed]).to(device),
                    torch.from_numpy(tp[st:ed]).to(device), torch.from_numpy(ep[st:ed]).to(device),
                )
                probs_list.append(torch.softmax(lg.float(), dim=-1).cpu().numpy())
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    probs = np.concatenate(probs_list, axis=0)
    f_all = torch.cat(f_list, dim=0) if f_list else None
    return probs, f_all, G_last, dt


@torch.no_grad()
def compute_sbc_and_ppe(
    model,
    grammar: ReactionGrammar,
    sim: ODESimulator,
    f_all: torch.Tensor,
    G_mat: torch.Tensor,
    probs_m: np.ndarray,
    mids: np.ndarray,
    thetas: np.ndarray,
    trajs: np.ndarray,
    t_grid: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
    n_ppe_cases: int = 527,
    seed: int = 2026,
) -> Dict:
    """Compute exact 19D NSF SBC-KS (L=99) on all N cases + Held-out D7 PPE on n_ppe_cases (1/mech)."""
    rng = np.random.default_rng(seed)
    N = len(mids)
    L_sbc = 99
    ranks_by_param: List[List[float]] = [[] for _ in range(grammar.n_params)]
    for st in range(0, N, batch_size):
        ed = min(N, st + batch_size)
        f_b = f_all[st:ed].to(device)
        m_b = torch.from_numpy(mids[st:ed]).to(device)
        th_samp = model.sample_theta(f_b, G_mat, m_b, num_samples=L_sbc).cpu().numpy()
        th_true = thetas[st:ed]
        for bi in range(ed - st):
            pmask = grammar.mech_param_mask[mids[st + bi]]
            for p_i in range(grammar.n_params):
                if pmask[p_i]:
                    cnt = int(np.sum(th_samp[:, bi, p_i] < th_true[bi, p_i]))
                    ranks_by_param[p_i].append((cnt + rng.random()) / (L_sbc + 1.0))

    ks_per_param = {}
    ks_vals = []
    for p_i, u_list in enumerate(ranks_by_param):
        if len(u_list) >= 30:
            stat = float(kstest(u_list, "uniform").statistic)
            ks_per_param[grammar.param_names[p_i]] = stat
            ks_vals.append(stat)
    all_u = [u for ul in ranks_by_param for u in ul]

    # Held-out D7 PPE (stratified 1 case per mech across all 527 mechs!)
    ppe_idx = np.array([np.where(mids == m)[0][0] for m in range(grammar.n_mechs)], dtype=np.int64)[:n_ppe_cases]
    n_ppe = len(ppe_idx)
    heldout_des = grammar.catalog[7]
    n_ppc_draws = 16
    t_eval_ppe = np.linspace(0.0, 1.0, 20)
    obs_ch = get_observable_channel_indices(grammar)
    sA, sP = grammar.species_idx["A"], grammar.species_idx["P"]
    ch_AP = [obs_ch.index(sA), obs_ch.index(sP)]

    draw_mids_flat = np.concatenate(
        [rng.choice(grammar.n_mechs, size=n_ppc_draws, p=probs_m[ci] / np.sum(probs_m[ci])) for ci in ppe_idx]
    ).astype(np.int32)
    draw_f_idx = np.repeat(ppe_idx, n_ppc_draws)

    th_chunks = []
    for st in range(0, len(draw_mids_flat), 2048):
        ed = min(len(draw_mids_flat), st + 2048)
        th_chunks.append(
            model.sample_theta(
                f_all[draw_f_idx[st:ed]].to(device), G_mat,
                torch.from_numpy(draw_mids_flat[st:ed]).long().to(device), num_samples=1,
            ).cpu().numpy()
        )
    th_draw = np.concatenate(th_chunks, axis=0).astype(np.float64)
    lb = grammar.mech_prior_mean - 4.5 * grammar.mech_prior_std
    ub = grammar.mech_prior_mean + 4.5 * grammar.mech_prior_std
    th_draw = np.clip(th_draw, lb[draw_mids_flat], ub[draw_mids_flat])

    y_pred, st_pred, _, _ = sim.simulate_designs(draw_mids_flat, th_draw, [heldout_des], t_eval_ppe, compute_flux=False)
    y_pred_AP = np.take(y_pred[:, 0], [sA, sP], axis=-1).reshape(n_ppe, n_ppc_draws, 20, 2)
    st_ok = (st_pred[:, 0] == 0).reshape(n_ppe, n_ppc_draws)

    ppe_rmses = []
    for loc_i, ci in enumerate(ppe_idx):
        ok_c = st_ok[loc_i]
        if np.any(ok_c):
            y_true_AP = interp_traj_to_times(t_grid, trajs[ci, 7].astype(np.float64), t_eval_ppe)[:, ch_AP]
            ppe_rmses.append(float(np.sqrt(np.mean((np.mean(y_pred_AP[loc_i, ok_c], axis=0) - y_true_AP) ** 2))))

    return {
        "sbc_ks_overall": float(kstest(all_u, "uniform").statistic),
        "sbc_ks_median_param": float(np.median(ks_vals)),
        "sbc_ks_mean_param": float(np.mean(ks_vals)),
        "sbc_ks_max_param": float(np.max(ks_vals)),
        "sbc_ks_by_param": ks_per_param,
        "heldout_D7_ppe_n": len(ppe_rmses),
        "heldout_D7_ppe_rmse_median": float(np.median(ppe_rmses)),
        "heldout_D7_ppe_rmse_mean": float(np.mean(ppe_rmses)),
        "heldout_D7_ppe_rmse_p90": float(np.percentile(ppe_rmses, 90)),
    }


def score_predictions(probs_m: np.ndarray, mids: np.ndarray, equiv_tier: Dict) -> Tuple[Dict, np.ndarray, np.ndarray]:
    N = len(mids)
    m_hit = (np.argmax(probs_m, axis=1) == mids).astype(np.float64)
    m_top3 = float(np.mean(np.any(np.argpartition(probs_m, -3, axis=1)[:, -3:] == mids[:, None], axis=1)))
    m_ece = compute_ece(probs_m, mids)
    m_tp = float(np.mean(probs_m[np.arange(N), mids]))

    eq = equiv_tier["primary"]
    m2c = np.array(eq["mech_to_class"], dtype=np.int64)
    n_cls = int(eq["n_classes"])
    true_c = m2c[mids]
    probs_c = aggregate_class_probs(probs_m, m2c, n_cls)
    c_hit = (np.argmax(probs_c, axis=1) == true_c).astype(np.float64)
    k3 = min(3, n_cls)
    c_top3 = float(np.mean(np.any(np.argpartition(probs_c, -k3, axis=1)[:, -k3:] == true_c[:, None], axis=1)))
    c_ece = compute_ece(probs_c, true_c)
    c_tp = float(np.mean(probs_c[np.arange(N), true_c]))

    return {
        "n_classes": n_cls,
        "mech_top1": float(np.mean(m_hit)),
        "mech_top3": m_top3,
        "mech_ece": m_ece,
        "mech_mean_true_prob": m_tp,
        "class_top1": float(np.mean(c_hit)),
        "class_top3": c_top3,
        "class_ece": c_ece,
        "class_mean_true_prob": c_tp,
    }, m_hit, c_hit


def compute_compositional_metrics(grammar: ReactionGrammar, probs_m: np.ndarray, mids: np.ndarray) -> Dict:
    """Core Top-1 + 8-module presence Macro-F1 / mean AUROC-free F1 for OOD compositional analysis."""
    N = len(mids)
    core_p = np.zeros((N, grammar.n_cores), dtype=np.float64)
    for c_i in range(grammar.n_cores):
        core_p[:, c_i] = np.sum(probs_m[:, grammar.mech_core_id == c_i], axis=1)
    core_top1 = float(np.mean(np.argmax(core_p, axis=1) == grammar.mech_core_id[mids]))

    f1s = []
    exact_mod_match = np.ones(N, dtype=bool)
    for mod in grammar.modules:
        true_pres = (grammar.mech_options[mids, mod.id] > 0)
        prob_pres = np.sum(probs_m[:, grammar.mech_options[:, mod.id] > 0], axis=1)
        pred_pres = (prob_pres >= 0.5)
        exact_mod_match &= (pred_pres == true_pres)
        tp = float(np.sum(pred_pres & true_pres))
        fp = float(np.sum(pred_pres & ~true_pres))
        fn = float(np.sum(~pred_pres & true_pres))
        f1 = (2.0 * tp) / max(2.0 * tp + fp + fn, 1.0)
        f1s.append(f1)

    return {
        "core_top1": core_top1,
        "module_macro_f1": float(np.mean(f1s)),
        "exact_8mod_presence_match": float(np.mean(exact_mod_match)),
        "module_f1_by_mod": {mod.code: float(f1s[mod.id]) for mod in grammar.modules},
    }


def run_seed_evaluation(
    grammar: ReactionGrammar,
    sim: ODESimulator,
    test_dir: Path,
    equiv_tiers: Dict[str, Dict],
    models_dir: Path,
    out_dir: Path,
    device: torch.device,
    seed: int,
    n_per_mech: int = 10,
) -> Dict:
    print(f"\n================ Evaluating Seed {seed} (n_test={n_per_mech * grammar.n_mechs}) ================", flush=True)
    t_grid = np.load(test_dir / "t_grid.npy")
    all_mids = np.load(test_dir / "mech_ids.npy")
    all_thetas = np.load(test_dir / "thetas.npy")
    all_rand = np.load(test_dir / "rand_des.npy")
    all_trajs = np.load(test_dir / "trajs.npy", mmap_mode="r")

    chosen_idx = []
    for m in range(grammar.n_mechs):
        idxs = np.where(all_mids == m)[0]
        chosen_idx.extend(idxs[:n_per_mech].tolist())
    chosen_idx = np.array(chosen_idx, dtype=np.int64)
    N = len(chosen_idx)
    mids = all_mids[chosen_idx].astype(np.int64)
    thetas = all_thetas[chosen_idx].astype(np.float64)
    rand_des = all_rand[chosen_idx]
    trajs = np.array(all_trajs[chosen_idx])

    # Pack all 11 regimes ONCE with fixed noise seed so every model sees bit-identical noisy test data!
    regime_specs = {
        "cat_1_D0":             ([[0] for _ in range(N)], False, False, 0.01, "D0"),
        "cat_3_D0_D2_D5":       ([[0, 2, 5] for _ in range(N)], False, False, 0.01, "no_P"),
        "cat_4_D0_D2_D5_D6":    ([[0, 2, 5, 6] for _ in range(N)], False, False, 0.01, "basic"),
        "cat_6_noP_all":        ([list(grammar.tiers["no_P"]) for _ in range(N)], False, False, 0.01, "no_P"),
        "cat_7_basic_all":      ([list(grammar.tiers["basic"]) for _ in range(N)], False, False, 0.01, "basic"),
        "cat_basic_var_window": ([list(grammar.tiers["basic"]) for _ in range(N)], False, True,  0.01, "basic"),
        "cat_10_full_all":      ([list(grammar.tiers["full"]) for _ in range(N)], False, False, 0.01, "full"),
        "cont_4_var_window":    ([[10, 11, 12, 13] for _ in range(N)], False, True,  0.01, "full"),
        "P_only_basic":         ([list(grammar.tiers["basic"]) for _ in range(N)], True,  False, 0.01, "basic"),
        "cat_7_basic_noise2pct":([list(grammar.tiers["basic"]) for _ in range(N)], False, False, 0.02, "basic"),
        "cat_7_basic_noise3pct":([list(grammar.tiers["basic"]) for _ in range(N)], False, False, 0.03, "basic"),
    }

    packed_cache: Dict[str, Tuple] = {}
    for r_i, (rname, (slots, p_only, var_w, sig, _)) in enumerate(regime_specs.items()):
        packed_cache[rname] = pack_fixed_test_batch(
            grammar, t_grid, trajs, rand_des, slots,
            sigma_rel=sig, n_t_obs=20, p_only=p_only, variable_window=var_w, seed=202600 + r_i * 37,
        )

    seed_out: Dict = {"seed": seed, "n_test": N, "e1_main": {}, "e1_paired_ci": {}, "bl20_protocol": {}, "e2_ablations": {}}

    # -------------------------------------------------------------------------
    # Part 1: E1 Main Table (QuoMech vs Flat NPE vs BL-527)
    # -------------------------------------------------------------------------
    sA, sP = grammar.species_idx["A"], grammar.species_idx["P"]
    e1_hits_cache: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}

    for m_label, subdir, kind in [
        ("QuoMech", f"quomech_s{seed}", "factorized"),
        ("Flat_NPE", f"flat_s{seed}", "flat"),
        ("BL_527", f"bl527_s{seed}", "bl527"),
    ]:
        ckpt_p = models_dir / subdir / "best.pt"
        if not ckpt_p.exists():
            print(f"  [SKIP] {m_label}: missing {ckpt_p}", flush=True)
            continue
        net, _, has_flow = load_model(grammar, ckpt_p, kind, device)
        m_res = {}
        e1_hits_cache[m_label] = {}
        for rname, (_, _, _, sig, tier_k) in regime_specs.items():
            df, tm, ov, om, tp, ep, raw_noisy, t_ends = packed_cache[rname]
            probs_m, f_all, G_mat, dt = predict_batch_probs(net, has_flow, df, tm, ov, om, tp, ep, device)
            entry, m_hit, c_hit = score_predictions(probs_m, mids, equiv_tiers[tier_k])
            entry["tier"] = tier_k
            entry["sigma_rel"] = sig
            entry["infer_ms_per_case"] = 1000.0 * dt / N
            e1_hits_cache[m_label][rname] = (m_hit, c_hit)

            if has_flow and rname in ("cat_4_D0_D2_D5_D6", "cat_7_basic_all", "cont_4_var_window"):
                entry.update(
                    compute_sbc_and_ppe(
                        net, grammar, sim, f_all, G_mat, probs_m, mids, thetas, trajs, t_grid,
                        device=device, n_ppe_cases=grammar.n_mechs, seed=2026 + seed,
                    )
                )

            if has_flow and m_label == "QuoMech" and rname == "cont_4_var_window":
                n_loc = min(200, N)
                des_per_case = []
                for ci in range(n_loc):
                    c_list = []
                    for e_i in range(4):
                        p5 = rand_des[ci, e_i]
                        d_spec = DesignSpec(
                            "rand", float(p5[0]), float(p5[1]), float(p5[2]),
                            float(p5[3]), int(p5[4] > 0.5), (sA, sP), 1.0, (),
                        )
                        c_list.append((d_spec, float(t_ends[ci, e_i])))
                    des_per_case.append(c_list)
                entry["local_top5_overlap"] = evaluate_local_top5_overlap(
                    net, grammar, sim, f_all, G_mat, probs_m, mids, des_per_case, raw_noisy, device,
                    sigma_rel=0.01, n_ppc=16, top_k=5, n_cases=n_loc,
                )

            m_res[rname] = entry
        seed_out["e1_main"][m_label] = m_res
        print(
            f"  [E1:{m_label:8s}] cat7_cls={m_res['cat_7_basic_all']['class_top1']:.3f} "
            f"cat10_cls={m_res['cat_10_full_all']['class_top1']:.3f} "
            f"cont4_cls={m_res['cont_4_var_window']['class_top1']:.3f} "
            f"ece7={m_res['cat_7_basic_all']['class_ece']:.4f} "
            f"ms={m_res['cat_7_basic_all']['infer_ms_per_case']:.3f}",
            flush=True,
        )

        # GPU latency benchmark on exact 200 Enum+LM+BIC cases if present!
        bic_npz = out_dir / "stage3_enum_bic_200_data.npz"
        if bic_npz.exists():
            bz = np.load(bic_npz)
            pm_b, _, _, dt_b = predict_batch_probs(
                net, has_flow, bz["df"], bz["tm"], bz["ov"], bz["om"], bz["tp"], bz["ep"], device
            )
            e_b, _, _ = score_predictions(pm_b, bz["true_mids"].astype(np.int64), equiv_tiers["basic"])
            e_b["total_time_s"] = dt_b
            e_b["ms_per_case"] = 1000.0 * dt_b / len(bz["true_mids"])
            e_b["device"] = str(device)
            seed_out.setdefault("enum_bic_200_gpu_matched", {})[m_label] = e_b

        del net
        torch.cuda.empty_cache()

    # Paired Bootstrap 95% CIs for E1 (QuoMech vs Flat NPE & QuoMech vs BL-527)
    if "QuoMech" in e1_hits_cache:
        for rival in ["Flat_NPE", "BL_527"]:
            if rival not in e1_hits_cache:
                continue
            ci_map = {}
            for rname in regime_specs:
                qm_m, qm_c = e1_hits_cache["QuoMech"][rname]
                rv_m, rv_c = e1_hits_cache[rival][rname]
                ci_map[rname] = {
                    "delta_class_top1": paired_bootstrap_ci(qm_c - rv_c, seed=seed + 1),
                    "delta_mech_top1": paired_bootstrap_ci(qm_m - rv_m, seed=seed + 2),
                }
            seed_out["e1_paired_ci"][f"QuoMech_minus_{rival}"] = ci_map

    # -------------------------------------------------------------------------
    # Part 2: BL-20 External Protocol (200 test cases = 10/mech across BL-20)
    # -------------------------------------------------------------------------
    bl20_ids = get_bl20_mappable_mech_ids(grammar)
    bl20_mask = np.isin(mids, bl20_ids)
    bl20_mids_true = mids[bl20_mask]
    remap = np.full(grammar.n_mechs, -1, dtype=np.int64)
    for ci, mid in enumerate(bl20_ids):
        remap[mid] = ci
    bl20_targets = remap[bl20_mids_true]
    bl20_regimes = ["cat_4_D0_D2_D5_D6", "cat_7_basic_all", "cont_4_var_window", "P_only_basic"]

    for m_label, subdir, kind in [
        ("BL_20", f"bl20_s{seed}", "bl20"),
        ("QuoMech", f"quomech_s{seed}", "factorized"),
        ("Flat_NPE", f"flat_s{seed}", "flat"),
    ]:
        ckpt_p = models_dir / subdir / "best.pt"
        if not ckpt_p.exists():
            continue
        net, _, has_flow = load_model(grammar, ckpt_p, kind, device)
        b_res = {}
        for rname in bl20_regimes:
            df, tm, ov, om, tp, ep, _, _ = packed_cache[rname]
            probs, _, _, dt = predict_batch_probs(
                net, has_flow, df[bl20_mask], tm[bl20_mask], ov[bl20_mask], om[bl20_mask],
                tp[bl20_mask], ep[bl20_mask], device,
            )
            if kind == "bl20":
                p20 = probs
                b_res[rname] = {
                    "bl20_top1": float(np.mean(np.argmax(p20, axis=1) == bl20_targets)),
                    "bl20_top3": float(np.mean(np.any(np.argpartition(p20, -3, axis=1)[:, -3:] == bl20_targets[:, None], axis=1))),
                    "bl20_ece": compute_ece(p20, bl20_targets),
                    "infer_ms_per_case": 1000.0 * dt / len(bl20_targets),
                }
            else:
                p20 = probs[:, bl20_ids].astype(np.float64)
                p20 /= np.maximum(np.sum(p20, axis=1, keepdims=True), 1e-12)
                tier_k = regime_specs[rname][4]
                open_sc, _, _ = score_predictions(probs, bl20_mids_true, equiv_tiers[tier_k])
                b_res[rname] = {
                    "restricted_bl20_top1": float(np.mean(np.argmax(p20, axis=1) == bl20_targets)),
                    "restricted_bl20_top3": float(np.mean(np.any(np.argpartition(p20, -3, axis=1)[:, -3:] == bl20_targets[:, None], axis=1))),
                    "restricted_bl20_ece": compute_ece(p20, bl20_targets),
                    "open527_mech_top1": open_sc["mech_top1"],
                    "open527_class_top1": open_sc["class_top1"],
                    "open527_class_top3": open_sc["class_top3"],
                    "open527_class_ece": open_sc["class_ece"],
                    "infer_ms_per_case": 1000.0 * dt / len(bl20_targets),
                }
        seed_out["bl20_protocol"][m_label] = b_res
        del net
        torch.cuda.empty_cache()

    if "BL_20" in seed_out["bl20_protocol"] and "QuoMech" in seed_out["bl20_protocol"]:
        print(
            f"  [BL-20] cat7: BL20={seed_out['bl20_protocol']['BL_20']['cat_7_basic_all']['bl20_top1']:.3f} vs "
            f"QuoMech(restr)={seed_out['bl20_protocol']['QuoMech']['cat_7_basic_all']['restricted_bl20_top1']:.3f} "
            f"(open527_cls={seed_out['bl20_protocol']['QuoMech']['cat_7_basic_all']['open527_class_top1']:.3f})",
            flush=True,
        )

    # -------------------------------------------------------------------------
    # Part 3: E2 Controlled Ablations
    # -------------------------------------------------------------------------
    # 3a. OOD 34 held-out 3-module combinations
    ood_ids = get_ood_heldout_3mod_ids(grammar, n_holdout=34, seed=2026)
    ood_mask = np.isin(mids, ood_ids)
    seen_mask = ~ood_mask
    ood_regimes = ["cat_4_D0_D2_D5_D6", "cat_7_basic_all", "cat_10_full_all"]
    ood_out = {"n_ood_mechs": len(ood_ids), "n_ood_cases": int(np.sum(ood_mask)), "n_seen_cases": int(np.sum(seen_mask))}

    for m_label, subdir, kind in [
        ("OOD_Factorized", f"e2_ood_factorized_s{seed}", "factorized"),
        ("OOD_Flat", f"e2_ood_flat_s{seed}", "flat"),
    ]:
        ckpt_p = models_dir / subdir / "best.pt"
        if not ckpt_p.exists():
            print(f"  [SKIP] {m_label}: missing {ckpt_p}", flush=True)
            continue
        net, _, has_flow = load_model(grammar, ckpt_p, kind, device)
        o_res = {}
        for rname in ood_regimes:
            df, tm, ov, om, tp, ep, _, _ = packed_cache[rname]
            probs_m, _, _, _ = predict_batch_probs(net, has_flow, df, tm, ov, om, tp, ep, device)
            tier_k = regime_specs[rname][4]
            sc_ood, _, _ = score_predictions(probs_m[ood_mask], mids[ood_mask], equiv_tiers[tier_k])
            sc_ood.update(compute_compositional_metrics(grammar, probs_m[ood_mask], mids[ood_mask]))
            sc_seen, _, _ = score_predictions(probs_m[seen_mask], mids[seen_mask], equiv_tiers[tier_k])
            o_res[rname] = {"unseen_34_ood": sc_ood, "seen_493": sc_seen}
        ood_out[m_label] = o_res
        print(
            f"  [E2:{m_label:14s}] OOD34 cat10: mech_top1={o_res['cat_10_full_all']['unseen_34_ood']['mech_top1']:.3f} "
            f"top3={o_res['cat_10_full_all']['unseen_34_ood']['mech_top3']:.3f} "
            f"cls_top1={o_res['cat_10_full_all']['unseen_34_ood']['class_top1']:.3f} "
            f"modF1={o_res['cat_10_full_all']['unseen_34_ood']['module_macro_f1']:.3f} | "
            f"Seen493 cls_top1={o_res['cat_10_full_all']['seen_493']['class_top1']:.3f}",
            flush=True,
        )
        del net
        torch.cuda.empty_cache()
    seed_out["e2_ablations"]["ood_34_3mod"] = ood_out

    # 3b. Discrete fixed-window ablation
    disc_ckpt = models_dir / f"e2_disc_fixed_s{seed}" / "best.pt"
    if disc_ckpt.exists():
        net, _, has_flow = load_model(grammar, disc_ckpt, "factorized", device)
        d_regimes = ["cat_4_D0_D2_D5_D6", "cat_7_basic_all", "cat_10_full_all", "cat_basic_var_window", "cont_4_var_window", "P_only_basic"]
        d_res = {}
        for rname in d_regimes:
            df, tm, ov, om, tp, ep, _, _ = packed_cache[rname]
            probs_m, f_all, G_mat, dt = predict_batch_probs(net, has_flow, df, tm, ov, om, tp, ep, device)
            tier_k = regime_specs[rname][4]
            entry, _, _ = score_predictions(probs_m, mids, equiv_tiers[tier_k])
            if rname == "cont_4_var_window":
                entry.update(
                    compute_sbc_and_ppe(
                        net, grammar, sim, f_all, G_mat, probs_m, mids, thetas, trajs, t_grid,
                        device=device, n_ppe_cases=grammar.n_mechs, seed=2026 + seed,
                    )
                )
            d_res[rname] = entry
        seed_out["e2_ablations"]["discrete_fixed_window"] = d_res
        print(
            f"  [E2:DiscFixed] cat10_cls={d_res['cat_10_full_all']['class_top1']:.3f} | "
            f"varwin_cls={d_res['cat_basic_var_window']['class_top1']:.3f} | "
            f"cont4_cls={d_res['cont_4_var_window']['class_top1']:.3f}",
            flush=True,
        )
        del net
        torch.cuda.empty_cache()

    # 3c. Unfiltered-prior ablation
    unf_ckpt = models_dir / f"e2_unfiltered_s{seed}" / "best.pt"
    if unf_ckpt.exists():
        net, _, has_flow = load_model(grammar, unf_ckpt, "factorized", device)
        u_regimes = ["cat_4_D0_D2_D5_D6", "cat_7_basic_all", "cat_10_full_all", "cont_4_var_window"]
        u_res = {}
        for rname in u_regimes:
            df, tm, ov, om, tp, ep, _, _ = packed_cache[rname]
            probs_m, f_all, G_mat, dt = predict_batch_probs(net, has_flow, df, tm, ov, om, tp, ep, device)
            tier_k = regime_specs[rname][4]
            entry, _, _ = score_predictions(probs_m, mids, equiv_tiers[tier_k])
            if rname in ("cat_7_basic_all", "cont_4_var_window"):
                entry.update(
                    compute_sbc_and_ppe(
                        net, grammar, sim, f_all, G_mat, probs_m, mids, thetas, trajs, t_grid,
                        device=device, n_ppe_cases=grammar.n_mechs, seed=2026 + seed,
                    )
                )
            u_res[rname] = entry
        seed_out["e2_ablations"]["unfiltered_prior"] = u_res
        print(
            f"  [E2:Unfiltered] cat7_cls={u_res['cat_7_basic_all']['class_top1']:.3f} "
            f"(ece={u_res['cat_7_basic_all']['class_ece']:.4f}, "
            f"sbc={u_res['cat_7_basic_all']['sbc_ks_median_param']:.4f}, "
            f"ppe={u_res['cat_7_basic_all']['heldout_D7_ppe_rmse_median']:.4f})",
            flush=True,
        )
        del net
        torch.cuda.empty_cache()

    return seed_out


def aggregate_across_seeds(per_seed: Dict[str, Dict]) -> Dict:
    """Recursively aggregate numeric leaves across seeds into {mean, std, values}."""
    seeds = sorted(per_seed.keys(), key=int)
    if len(seeds) == 1:
        return per_seed[seeds[0]]

    def _agg(objs: List):
        first = objs[0]
        if isinstance(first, dict):
            keys = set.intersection(*(set(o.keys()) for o in objs if isinstance(o, dict)))
            return {k: _agg([o[k] for o in objs]) for k in sorted(keys)}
        if isinstance(first, (int, float)) and not isinstance(first, bool):
            vals = [float(o) for o in objs]
            return {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)), "values": vals}
        return first

    return _agg([per_seed[s] for s in seeds])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="42", help="Comma-separated seeds, e.g. 42 or 42,43,44")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--n-threads", type=int, default=32)
    ap.add_argument("--n-per-mech", type=int, default=10)
    ap.add_argument("--data-dir", type=str, default="data/pool_multicore_527")
    ap.add_argument("--equiv-dir", type=str, default="results/phase2/equiv")
    ap.add_argument("--models-dir", type=str, default="results/phase2/models")
    ap.add_argument("--out-dir", type=str, default="results/phase2")
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    g = build_multicore_grammar()
    sim = ODESimulator(g, n_threads=args.n_threads)
    equiv_tiers = load_all_equiv(Path(args.equiv_dir))
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "stage3_e1_e2.json"

    per_seed = {}
    if out_json.exists():
        prev = json.loads(out_json.read_text())
        per_seed = prev.get("per_seed", {})

    for s in seeds:
        per_seed[str(s)] = run_seed_evaluation(
            g, sim, Path(args.data_dir) / "test", equiv_tiers,
            Path(args.models_dir), out_dir, device, seed=s, n_per_mech=args.n_per_mech,
        )
        payload = {
            "seeds_completed": sorted([int(k) for k in per_seed.keys()]),
            "per_seed": per_seed,
            "summary": aggregate_across_seeds(per_seed),
        }
        out_json.write_text(json.dumps(payload, indent=2))

    print(f"\n[Stage 3 E1+E2 DONE] Saved to {out_json}", flush=True)


if __name__ == "__main__":
    main()
