#!/usr/bin/env python3
"""Phase 2 E4: Real Published Catalysis Literature Evaluation & Stage 2a Smoke Test.

Reads all 11 curated published datasets from data/real_literature/manifest.json
(9 in-library + 2 OOD, 48 total runs, verbatim published points) and computes:
  - Fine-grained posterior q(m | D), core marginals (C1-C4), module presence marginals (M1-M8),
    and quotient-class posterior q(C | D) (if equiv JSON is provided)
  - Posterior predictive residual (RMSE & sigma multiples) + >3sigma misspecification alert
  - Top-5 local posterior predictive overlap clique (Section 7.4)
  - Leave-One-Out Posterior Predictive Error (LOO-PPE) across all runs
  - Optional next-experiment BOED recommendation from reference run vs paper's actual follow-ups
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch

from quomech.data_gen import build_design_feature_vec, get_observable_channel_indices
from quomech.grammar import DesignSpec, ReactionGrammar, build_multicore_grammar
from quomech.simulator import ODESimulator
from quomech.train import build_model


def load_real_case(
    grammar: ReactionGrammar,
    case_dir: Path,
    meta: Dict,
    max_len: int = 30,
) -> Dict:
    """Load and nondimensionalize a real literature case strictly per manifest.json convention."""
    obs_ch = get_observable_channel_indices(grammar)
    n_ch = len(obs_ch)
    si = grammar.species_idx
    ch_pos = {grammar.species[sp].name: i for i, sp in enumerate(obs_ch)}

    # Load curves grouped by run_id
    curves_by_run: Dict[str, List[Dict[str, float]]] = {}
    with open(case_dir / "curves.csv", "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = row["run_id"]
            curves_by_run.setdefault(rid, []).append(row)

    runs_meta = meta["runs"]
    ref_id = meta["reference_run_id"]
    ref_run = next(r for r in runs_meta if r["run_id"] == ref_id)
    A_ref = float(ref_run["A0"])
    t_ref = float(ref_run["t_end"])
    cat_ref = float(ref_run["cat0"])
    is_pseudo_uni = (meta["reaction_type"] == "pseudo_unimolecular")
    obs_names = list(meta["observed_species"])

    ch_mask = np.zeros(n_ch, dtype=np.float32)
    obs_sp_indices = []
    for sp_name in obs_names:
        if sp_name in ch_pos:
            ch_mask[ch_pos[sp_name]] = 1.0
            obs_sp_indices.append(si[sp_name])
    obs_sp_tuple = tuple(sorted(obs_sp_indices))

    parsed_runs = []
    for r in runs_meta:
        rid = r["run_id"]
        rows = curves_by_run[rid]
        n_raw = len(rows)

        t_raw = np.array([float(x["t"]) for x in rows], dtype=np.float64)
        t_dim = t_raw / t_ref

        # Raw observed curves (normalized by A_ref)
        y_raw_ch = np.zeros((n_raw, n_ch), dtype=np.float64)
        for sp_name in obs_names:
            vals = np.array(
                [float(x[sp_name]) if (sp_name in x and x[sp_name] not in ("", "None", "nan")) else np.nan for x in rows],
                dtype=np.float64,
            )
            y_raw_ch[:, ch_pos[sp_name]] = vals / A_ref

        A0_dim = float(r["A0"]) / A_ref
        B0_dim = 10.0 if (is_pseudo_uni or r["B0"] is None) else (float(r["B0"]) / A_ref)
        P0_dim = float(r["P0"]) / A_ref
        cat_mult = float(r["cat0"]) / cat_ref
        a_pre = 1 if r.get("preactivated", False) else 0
        t_end_dim = float(t_dim[-1])

        # Deterministic striding without interpolation if n_raw > max_len (keeps t=0 and t_end)
        if n_raw > max_len:
            idx_sub = np.unique(np.round(np.linspace(0, n_raw - 1, max_len)).astype(int))
        else:
            idx_sub = np.arange(n_raw, dtype=int)

        des_spec = DesignSpec(
            name=rid,
            A0=A0_dim,
            B0=B0_dim,
            P0=P0_dim,
            cat_mult=cat_mult,
            a_pre=a_pre,
            obs_species=obs_sp_tuple,
            cost=1.0,
            tiers=(),
        )
        des_feat = build_design_feature_vec(
            grammar, A0_dim, B0_dim, P0_dim, cat_mult, float(a_pre), t_end_dim, obs_ch, ch_mask
        )

        parsed_runs.append({
            "run_id": rid,
            "role": r.get("role", ""),
            "design": des_spec,
            "design_feat": des_feat,
            "ch_mask": ch_mask,
            "t_raw_dim": t_dim,
            "y_raw_dim": y_raw_ch,
            "idx_sub": idx_sub,
            "t_sub": t_dim[idx_sub].astype(np.float32),
            "y_sub": y_raw_ch[idx_sub].astype(np.float32),
        })

    return {
        "case_id": meta["case_id"],
        "in_library": bool(meta["in_library"]),
        "reaction_type": meta["reaction_type"],
        "observed_species": obs_names,
        "expected_cores": meta.get("expected_cores", []),
        "expected_modules": meta.get("expected_modules", []),
        "reference_run_id": ref_id,
        "A_ref": A_ref,
        "t_ref": t_ref,
        "cat_ref": cat_ref,
        "obs_ch": obs_ch,
        "runs": parsed_runs,
    }


def runs_to_tensors(
    grammar: ReactionGrammar,
    runs: List[Dict],
    device: torch.device,
    max_len: int = 30,
) -> Tuple[torch.Tensor, ...]:
    """Pack a list of parsed runs into a single-example batch (1, E, ...) for QuoMechNet."""
    E = len(runs)
    n_ch = len(runs[0]["ch_mask"])
    des_dim = len(runs[0]["design_feat"])

    df = np.zeros((1, E, des_dim), dtype=np.float32)
    tm = np.zeros((1, E, max_len), dtype=np.float32)
    ov = np.zeros((1, E, max_len, n_ch), dtype=np.float32)
    om = np.zeros((1, E, max_len, n_ch), dtype=np.float32)
    tp = np.ones((1, E, max_len), dtype=bool)
    ep = np.zeros((1, E), dtype=bool)

    for e_i, r in enumerate(runs):
        L = len(r["t_sub"])
        df[0, e_i] = r["design_feat"]
        tm[0, e_i, :L] = r["t_sub"]
        y_s = r["y_sub"]
        valid_val = ~np.isnan(y_s)
        m_mat = np.broadcast_to(r["ch_mask"][None, :], (L, n_ch)) * valid_val.astype(np.float32)
        y_clean = np.where(valid_val, y_s, 0.0)
        ov[0, e_i, :L] = y_clean * m_mat
        om[0, e_i, :L] = m_mat
        tp[0, e_i, :L] = False

    return (
        torch.from_numpy(df).to(device),
        torch.from_numpy(tm).to(device),
        torch.from_numpy(ov).to(device),
        torch.from_numpy(om).to(device),
        torch.from_numpy(tp).to(device),
        torch.from_numpy(ep).to(device),
    )


def simulate_run_predictions(
    grammar: ReactionGrammar,
    sim: ODESimulator,
    mech_ids: np.ndarray,   # (S,)
    thetas: np.ndarray,     # (S, P)
    run: Dict,
    obs_ch: Tuple[int, ...],
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate (S,) trajectories on exact raw timestamps of `run` -> (S, n_raw, n_ch), ok_mask (S,)."""
    S = len(mech_ids)
    t_eval = run["t_raw_dim"]
    des = run["design"]
    y0s = np.stack([grammar.initial_state(int(m), des) for m in mech_ids], axis=0)
    y_full, st, _, _ = sim.simulate_batch(
        mech_ids.astype(np.int32), thetas.astype(np.float64), y0s, t_eval, compute_flux=False
    )
    ok = (st == 0)
    y_ch = np.take(y_full, obs_ch, axis=-1)  # (S, n_raw, n_ch)
    return y_ch, ok


def evaluate_real_case(
    grammar: ReactionGrammar,
    sim: ODESimulator,
    model,
    case: Dict,
    device: torch.device,
    n_post_samples: int = 32,
    sigma_alert_scale: float = 0.02,
    equiv_mech_to_class: Optional[np.ndarray] = None,
    seed: int = 42,
) -> Dict:
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    runs = case["runs"]
    obs_ch = case["obs_ch"]

    # 1. Full-dataset posterior
    tensors = runs_to_tensors(grammar, runs, device)
    with torch.no_grad():
        probs_t, f_t, G_t = model.predict_posterior(*tensors)
    probs = probs_t[0].cpu().numpy()  # (M,)

    # Core & module marginals
    core_probs = {
        c.code: float(np.sum(probs[grammar.mech_core_id == c.id]))
        for c in grammar.cores
    }
    mod_probs = {
        mod.code: float(np.sum(probs[grammar.mech_options[:, mod.id] > 0]))
        for mod in grammar.modules
    }
    top5_idx = np.argsort(probs)[::-1][:5].tolist()
    top5_info = [
        {"mech_id": int(m), "name": grammar.mech_names[m], "prob": float(probs[m])}
        for m in top5_idx
    ]

    quotient_top3 = None
    if equiv_mech_to_class is not None:
        n_cls = int(np.max(equiv_mech_to_class)) + 1
        c_probs = np.bincount(equiv_mech_to_class, weights=probs, minlength=n_cls)
        c_top = np.argsort(c_probs)[::-1][:3]
        quotient_top3 = []
        for cid in c_top:
            members = np.where(equiv_mech_to_class == cid)[0].tolist()
            quotient_top3.append({
                "class_id": int(cid),
                "prob": float(c_probs[cid]),
                "size": len(members),
                "rep_names": [grammar.mech_names[m] for m in members[:4]],
            })

    # 2. Top-5 Local Posterior Predictive Overlap & >3sigma Misspecification Check
    t_loc0 = time.perf_counter()
    top5_pred_means = []   # list of flat observed vectors for each of Top-5 mechs
    top5_res_rmse = []
    top5_res_sigma = []
    obs_flat_ref = []
    scale_flat_ref = []
    for r in runs:
        y_obs = r["y_raw_dim"]  # (n_raw, n_ch)
        mask_2d = (r["ch_mask"][None, :] > 0.5) & (~np.isnan(y_obs))
        obs_flat_ref.append(y_obs[mask_2d])
        scale_flat_ref.append(np.full(int(np.sum(mask_2d)), sigma_alert_scale * r["design"].A0, dtype=np.float64))
    obs_flat_all = np.concatenate(obs_flat_ref)
    sig_flat_all = np.concatenate(scale_flat_ref)

    for m_id in top5_idx:
        with torch.no_grad():
            m_t = torch.tensor([m_id], dtype=torch.long, device=device)
            th_s = model.sample_theta(f_t, G_t, m_t, num_samples=n_post_samples)[:, 0, :].cpu().numpy()  # (S, P)
        mids_s = np.full(n_post_samples, m_id, dtype=np.int32)

        pred_chunks = []
        valid_s = np.ones(n_post_samples, dtype=bool)
        for r in runs:
            y_ch, ok = simulate_run_predictions(grammar, sim, mids_s, th_s, r, obs_ch)
            valid_s &= ok
            mask_2d = (r["ch_mask"][None, :] > 0.5) & (~np.isnan(r["y_raw_dim"]))
            pred_chunks.append(y_ch[:, mask_2d])  # (S, n_obs_r)
        preds_all = np.concatenate(pred_chunks, axis=1)  # (S, N_obs)
        if np.any(valid_s):
            preds_ok = preds_all[valid_s]
            # Pick best-fitting posterior draw AND posterior median curve for robustness
            err_per_s = np.sqrt(np.mean(((preds_ok - obs_flat_all[None, :]) / sig_flat_all[None, :]) ** 2, axis=1))
            best_s = int(np.argmin(err_per_s))
            pred_med = np.median(preds_ok, axis=0)
            rmse_med = float(np.sqrt(np.mean((pred_med - obs_flat_all) ** 2)))
            z_med = float(np.sqrt(np.mean(((pred_med - obs_flat_all) / sig_flat_all) ** 2)))
            z_best = float(err_per_s[best_s])
            top5_pred_means.append(pred_med)
            top5_res_rmse.append(rmse_med)
            top5_res_sigma.append({"z_median_curve": z_med, "z_best_post_draw": z_best})
        else:
            top5_pred_means.append(np.zeros_like(obs_flat_all))
            top5_res_rmse.append(999.0)
            top5_res_sigma.append({"z_median_curve": 999.0, "z_best_post_draw": 999.0})

    # Local overlap clique relative to MAP (top5_idx[0]) within 1.0 * sigma_alert_scale
    overlap_with_map = [top5_idx[0]]
    p0 = top5_pred_means[0]
    for k in range(1, len(top5_idx)):
        z_diff = float(np.sqrt(np.mean(((top5_pred_means[k] - p0) / sig_flat_all) ** 2)))
        if z_diff <= 1.0:
            overlap_with_map.append(top5_idx[k])

    min_z_med = float(min(x["z_median_curve"] for x in top5_res_sigma))
    min_z_best = float(min(x["z_best_post_draw"] for x in top5_res_sigma))
    misspec_alert_3sigma = bool(min_z_best > 3.0)
    dt_local_ms = (time.perf_counter() - t_loc0) * 1e3

    # 3. Leave-One-Out Posterior Predictive Error (LOO-PPE) across all runs
    loo_rmses = {}
    for hold_i, hold_run in enumerate(runs):
        train_runs = [r for j, r in enumerate(runs) if j != hold_i]
        t_loo = runs_to_tensors(grammar, train_runs, device)
        with torch.no_grad():
            p_loo_t, f_loo_t, G_loo_t = model.predict_posterior(*t_loo)
            p_loo = p_loo_t[0].cpu().numpy()
            sampled_m = rng.choice(grammar.n_mechs, size=n_post_samples, p=p_loo / np.sum(p_loo))
            m_t = torch.from_numpy(sampled_m).long().to(device)
            f_rep = f_loo_t.expand(n_post_samples, -1)
            th_loo = model.sample_theta(f_rep, G_loo_t, m_t, num_samples=1).cpu().numpy()
        y_pred, ok = simulate_run_predictions(grammar, sim, sampled_m, th_loo, hold_run, obs_ch)
        y_obs = hold_run["y_raw_dim"]
        mask_2d = (hold_run["ch_mask"][None, :] > 0.5) & (~np.isnan(y_obs))
        if np.any(ok):
            y_med = np.median(y_pred[ok], axis=0)
            rmse_loo = float(np.sqrt(np.mean((y_med[mask_2d] - y_obs[mask_2d]) ** 2)))
        else:
            rmse_loo = float("nan")
        loo_rmses[hold_run["run_id"]] = rmse_loo

    mean_loo_ppe = float(np.nanmean(list(loo_rmses.values())))

    # Check expected module hit
    exp_mods = case["expected_modules"]
    exp_mod_probs = {m: mod_probs[m] for m in exp_mods}

    return {
        "case_id": case["case_id"],
        "in_library": case["in_library"],
        "n_runs": len(runs),
        "observed_species": case["observed_species"],
        "expected_cores": case["expected_cores"],
        "expected_modules": exp_mods,
        "expected_module_probs": exp_mod_probs,
        "core_probs": core_probs,
        "module_probs": mod_probs,
        "top5": top5_info,
        "quotient_top3": quotient_top3,
        "top5_res_rmse": top5_res_rmse,
        "top5_res_sigma": top5_res_sigma,
        "min_z_median": min_z_med,
        "min_z_best_draw": min_z_best,
        "misspec_alert_3sigma": misspec_alert_3sigma,
        "local_overlap_clique": [grammar.mech_names[m] for m in overlap_with_map],
        "local_overlap_ms": dt_local_ms,
        "loo_ppe_by_run": loo_rmses,
        "mean_loo_ppe": mean_loo_ppe,
        "elapsed_s": time.perf_counter() - t0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--arch", type=str, default="p2_256")
    ap.add_argument("--head", type=str, default="factorized")
    ap.add_argument("--equiv-json", type=str, default=None)
    ap.add_argument("--data-dir", type=str, default="data/real_literature")
    ap.add_argument("--out-json", type=str, required=True)
    ap.add_argument("--n-post-samples", type=int, default=32)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    g = build_multicore_grammar()
    sim = ODESimulator(g, n_threads=32)

    ckpt = torch.load(args.ckpt, map_location=device)
    arch = ckpt.get("arch", args.arch)
    head = ckpt.get("head_type", args.head)
    model = build_model(g, head_type=head, arch=arch).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    equiv_m2c = None
    if args.equiv_json and Path(args.equiv_json).exists():
        eq = json.loads(Path(args.equiv_json).read_text())
        equiv_m2c = np.array(eq["primary"]["mech_to_class"], dtype=np.int32)

    lit_dir = Path(args.data_dir)
    manifest = json.loads((lit_dir / "manifest.json").read_text())

    results = []
    print(f"=== E4 Real Literature Evaluation ({len(manifest['cases'])} cases, ckpt={args.ckpt}) ===", flush=True)
    for idx, c_entry in enumerate(manifest["cases"]):
        case_dir = lit_dir / c_entry["case_id"]
        meta = json.loads((case_dir / "meta.json").read_text())
        case = load_real_case(g, case_dir, meta)
        res = evaluate_real_case(
            g, sim, model, case, device,
            n_post_samples=args.n_post_samples,
            equiv_mech_to_class=equiv_m2c,
            seed=42 + idx,
        )
        results.append(res)
        tag = "IN-LIB" if res["in_library"] else "OOD   "
        top1 = res["top5"][0]
        print(
            f"  [{tag}] {res['case_id']:44s} | MAP={top1['name']:24s} ({top1['prob']:.2f}) | "
            f"LOO-PPE={res['mean_loo_ppe']:.4f} | z_best={res['min_z_best_draw']:.2f}s "
            f"alert={res['misspec_alert_3sigma']} | exp_mods={res['expected_module_probs']}",
            flush=True,
        )

    out_p = Path(args.out_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        json.dump({"ckpt": args.ckpt, "n_cases": len(results), "cases": results}, f, indent=2)
    print(f"\nSaved E4 real literature results to {out_p}", flush=True)


if __name__ == "__main__":
    main()
