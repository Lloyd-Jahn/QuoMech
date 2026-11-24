#!/usr/bin/env python3
"""Fast diagnostic on real published datasets (data/real_literature/) using quomech_s42/best.pt:
  1. Inspect empirical catalyst-order & curves on stroek_2021, albright_2019, elsby_2021, herbort_2021
  2. Test B0_dim sensitivity (2.5, 5.0, 10.0) + physical C1/C4 single-substrate mask on pseudo-unimolecular cases
  3. Test Amortized Warm-Start + 20-iter Local LM refinement (Top-5 + clean cores C1/C2/C4) + signal-normalized z-score
  4. Test 1-step BOED next-experiment recommendation from reference run vs paper's actual follow-up runs
"""

from __future__ import annotations

import json
import time
from pathlib import Path
import numpy as np
import torch

from quomech.data_gen import build_design_feature_vec, get_observable_channel_indices
from quomech.grammar import DesignSpec, build_multicore_grammar
from quomech.simulator import ODESimulator
from quomech.train import build_model
from scripts.p2_e4_real_eval import load_real_case, runs_to_tensors


def reload_with_b0(g, cdir, meta, b0_uni: float):
    case = load_real_case(g, cdir, meta)
    if meta["reaction_type"] != "pseudo_unimolecular":
        return case
    obs_ch = case["obs_ch"]
    for r in case["runs"]:
        d = r["design"]
        r["design"] = DesignSpec(d.name, d.A0, b0_uni, d.P0, d.cat_mult, d.a_pre, d.obs_species, d.cost, d.tiers)
        r["design_feat"] = build_design_feature_vec(
            g, d.A0, b0_uni, d.P0, d.cat_mult, float(d.a_pre), float(r["t_raw_dim"][-1]), obs_ch, r["ch_mask"]
        )
    return case


def check_empirical_cat_order(lit_dir: Path):
    print("=== 1. Empirical Half-Life / Rate Scaling in Raw Published Curves ===", flush=True)
    import csv
    for cid in ["stroek_2021_fe_ch_amination", "albright_2019_fe_carbonyl_olefin_metathesis", "elsby_2021_mn_hydroboration", "herbort_2021_co_hydroalkenylation"]:
        meta = json.loads((lit_dir / cid / "meta.json").read_text())
        rows_by_run = {}
        with open(lit_dir / cid / "curves.csv") as f:
            for row in csv.DictReader(f):
                rows_by_run.setdefault(row["run_id"], []).append(row)
        print(f"  {cid} (expected={meta['expected_modules']}):")
        for r in meta["runs"]:
            rid = r["run_id"]
            rows = rows_by_run[rid]
            t = np.array([float(x["t"]) for x in rows])
            sp = "A" if "A" in meta["observed_species"] else "P"
            y = np.array([float(x[sp]) for x in rows])
            if sp == "A":
                conv = 1.0 - y / float(r["A0"])
            else:
                conv = y / float(r["A0"])
            # time to reach 30% and 50% conversion
            t30 = float(np.interp(0.30, conv, t)) if np.max(conv) >= 0.30 else float("nan")
            t50 = float(np.interp(0.50, conv, t)) if np.max(conv) >= 0.50 else float("nan")
            print(f"    {rid:22s} A0={r['A0']:.4g} B0={str(r['B0']):>6s} cat0={r['cat0']:.5g} t_end={r['t_end']:.1f} conv={conv[-1]*100:5.1f}% t30={t30:7.1f} t50={t50:7.1f}")


def main():
    torch.set_num_threads(4)
    lit_dir = Path("data/real_literature")
    manifest = json.loads((lit_dir / "manifest.json").read_text())
    check_empirical_cat_order(lit_dir)

    g = build_multicore_grammar()
    sim = ODESimulator(g, n_threads=8)
    dev = torch.device("cpu")
    ckpt = torch.load("results/phase2/models/quomech_s42/best.pt", map_location=dev)
    model = build_model(g, "factorized", "p2_256").to(dev)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    k_arr = np.sum(g.mech_options > 0, axis=1)
    nk = {k: int(np.sum(k_arr == k)) for k in range(4)}
    pi_hier = np.array([1.0 / (4.0 * nk[int(k)]) for k in k_arr], dtype=np.float64)
    uni_mask = np.isin(g.mech_core_id, [0, 3]).astype(np.float64)

    print("\n=== 2. B0_dim Sensitivity on Pseudo-Unimolecular Systems (with C1/C4 mask + Hier prior) ===", flush=True)
    for entry in manifest["cases"]:
        if entry["reaction_type"] != "pseudo_unimolecular":
            continue
        cid = entry["case_id"]
        cdir = lit_dir / cid
        meta = json.loads((cdir / "meta.json").read_text())
        print(f"  [{ 'IN ' if meta['in_library'] else 'OOD' }] {cid} (exp={meta['expected_modules']})")
        for b0_val in [2.5, 5.0, 10.0]:
            case = reload_with_b0(g, cdir, meta, b0_val)
            with torch.no_grad():
                p_t, _, _ = model.predict_posterior(*runs_to_tensors(g, case["runs"], dev))
            p_u = p_t[0].numpy().astype(np.float64)
            p_h = p_u * pi_hier * uni_mask
            p_h /= p_h.sum()
            top = int(np.argmax(p_h))
            mp = {m.code: round(float(p_h[g.mech_options[:, m.id] > 0].sum()), 2) for m in g.modules}
            exp = {m: mp[m] for m in meta["expected_modules"]}
            top_mods = sorted(mp.items(), key=lambda kv: -kv[1])[:3]
            print(f"    B0={b0_val:4.1f} -> MAP={g.mech_names[top]:20s}({p_h[top]:.2f}) k0={p_h[k_arr==0].sum():.2f} exp={exp} top3_mods={top_mods}")

    print("\n=== 3. Amortized Warm-Start + 20-iter Local LM Refinement & Signal-Normalized Misspec Alert ===", flush=True)
    # Build uniform-n_t strided verbatim point targets per case so C++ fit_lm_batch runs in ONE call!
    core_ids = [g.get_mech_id_by_name(c) for c in ["C1", "C2", "C3", "C4"]]
    obs_ch = get_observable_channel_indices(g)
    ch_pos = {g.species[sp].name: i for i, sp in enumerate(obs_ch)}
    lb_all = g.mech_prior_mean - 4.0 * g.mech_prior_std
    ub_all = g.mech_prior_mean + 4.0 * g.mech_prior_std

    for entry in manifest["cases"]:
        cid = entry["case_id"]
        cdir = lit_dir / cid
        meta = json.loads((cdir / "meta.json").read_text())
        case = load_real_case(g, cdir, meta)
        runs = case["runs"]
        is_uni = (meta["reaction_type"] == "pseudo_unimolecular")

        t0 = time.perf_counter()
        with torch.no_grad():
            p_t, f_t, G_t = model.predict_posterior(*runs_to_tensors(g, runs, dev))
        p_u = p_t[0].numpy().astype(np.float64)
        mask = uni_mask if is_uni else np.ones_like(p_u)
        p_h = p_u * pi_hier * mask
        p_h /= p_h.sum()

        # Candidate set: Top-5 posterior (Hier) + clean cores (C1,C4 for uni; C1-C4 for bimol)
        top5 = np.argsort(p_h)[::-1][:5].tolist()
        cands = list(dict.fromkeys(top5 + ([core_ids[0], core_ids[3]] if is_uni else core_ids)))

        # Build verbatim strided points with uniform n_t_fit = min(15, min_run_pts) per run
        n_t_fit = min(15, min(len(r["t_raw_dim"]) for r in runs))
        designs = [r["design"] for r in runs]
        t_eval_2d = np.zeros((len(runs), n_t_fit), dtype=np.float64)
        tgt_chunks = []
        sig_chunks = []
        obs_vals_all = []
        for d_i, r in enumerate(runs):
            n_raw = len(r["t_raw_dim"])
            idx = np.round(np.linspace(0, n_raw - 1, n_t_fit)).astype(int)
            t_eval_2d[d_i] = r["t_raw_dim"][idx]
            for sp in r["design"].obs_species:
                sp_name = g.species[sp].name
                yv = r["y_raw_dim"][idx, ch_pos[sp_name]]
                tgt_chunks.append(yv)
                obs_vals_all.append(yv)
                sig_chunks.append(np.full(n_t_fit, 0.02 * r["design"].A0, dtype=np.float64))

        tgt_1d = np.concatenate(tgt_chunks)
        obs_all_vec = np.concatenate(obs_vals_all)
        sig_span = max(float(np.nanmax(obs_all_vec) - np.nanmin(obs_all_vec)), 0.01)
        # Signal-amplitude-scaled sigma (2% of dynamic signal range, capped at 0.02)
        sig_rel = min(0.02, 0.02 * sig_span)
        inv_sig_1d = np.full_like(tgt_1d, 1.0 / sig_rel)

        # Sample 3 amortized warm-start thetas + 1 prior mean per candidate mechanism
        n_c = len(cands)
        inits = np.zeros((n_c, 4, g.n_params), dtype=np.float64)
        with torch.no_grad():
            for ci, m_id in enumerate(cands):
                m_t = torch.tensor([m_id], dtype=torch.long, device=dev)
                th_s = model.sample_theta(f_t, G_t, m_t, num_samples=3)[:, 0, :].numpy()
                inits[ci, :3] = th_s
                inits[ci, 3] = g.mech_prior_mean[m_id]

        t_lm0 = time.perf_counter()
        rms_fit, _ = sim.fit_lm_batch(
            np.array(cands, dtype=np.int32),
            designs,
            t_eval_2d,
            np.broadcast_to(tgt_1d[None, :], (n_c, len(tgt_1d))).copy(),
            np.broadcast_to(inv_sig_1d[None, :], (n_c, len(tgt_1d))).copy(),
            inits,
            lb_all=lb_all, ub_all=ub_all,
            rtol=1e-4, atol=1e-8, max_ode_steps=8000, max_lm_iters=20,
            early_stop_rms=0.5,
        )
        dt_lm_ms = (time.perf_counter() - t_lm0) * 1e3

        # Compute BIC for each candidate using signal-scaled variance (or 2% abs)
        n_obs = len(tgt_1d)
        bics = []
        for ci, m_id in enumerate(cands):
            k_p = int(g.mech_param_mask[m_id].sum())
            rss_z = (rms_fit[ci] ** 2) * n_obs
            bics.append(rss_z + k_p * np.log(n_obs))
        best_ci = int(np.argmin(rms_fit))
        best_bic_ci = int(np.argmin(bics))

        # Clean core best RMS
        core_idxs = [ci for ci, m_id in enumerate(cands) if k_arr[m_id] == 0]
        best_core_ci = min(core_idxs, key=lambda ci: rms_fit[ci])
        best_core_z = float(rms_fit[best_core_ci])
        best_core_rmse = best_core_z * sig_rel
        best_any_z = float(rms_fit[best_ci])
        best_any_rmse = best_any_z * sig_rel

        ref_run = next(r for r in meta["runs"] if r["run_id"] == meta["reference_run_id"])
        ref_conv = float(ref_run["final_conversion_pct"])
        regime_ood = (ref_conv < 20.0) or (ref_conv > 99.0)
        resid_ood = (best_any_z > 3.0)

        tag = "IN " if meta["in_library"] else "OOD"
        print(
            f"  [{tag}] {cid:42s} | {dt_lm_ms:5.0f}ms | span={sig_span:.2f} ref_conv={ref_conv:5.1f}% | "
            f"best={g.mech_names[cands[best_ci]]:20s} rmse={best_any_rmse:.4f} (z_sig={best_any_z:.2f}) | "
            f"core={g.mech_names[cands[best_core_ci]]:4s} rmse={best_core_rmse:.4f} (z={best_core_z:.2f}) | "
            f"BIC*={g.mech_names[cands[best_bic_ci]]:20s} | alert(res/reg)={resid_ood}/{regime_ood}",
            flush=True,
        )


if __name__ == "__main__":
    main()
