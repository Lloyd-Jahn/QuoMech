#!/usr/bin/env python3
"""Phase 2 Stage 1a: 4-core 527-mechanism verification, SciPy Radau accuracy check,
C++17 engine extended API tests, and full 527-mechanism acceptance probe.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
import time
from pathlib import Path
import numpy as np

from quomech.grammar import build_default_grammar, build_multicore_grammar
from quomech.simulator import ODESimulator, check_observability_batch


_G_WORKER = None
_SIM_WORKER = None


def _init_radau_worker():
    global _G_WORKER, _SIM_WORKER
    _G_WORKER = build_multicore_grammar()
    _SIM_WORKER = ODESimulator(_G_WORKER, n_threads=1)


def _radau_one(args):
    idx, mid, theta, y0, t_eval = args
    y_ref, ok = _SIM_WORKER.simulate_scipy_radau(int(mid), theta, y0, t_eval, rtol=1e-10, atol=1e-14)
    return idx, y_ref, ok


def check_grammar_invariants():
    print("=== 1. Grammar Invariant Checks ===", flush=True)
    g151 = build_default_grammar()
    assert g151.n_mechs == 151 and g151.n_species == 11 and g151.n_reactions == 17 and g151.n_params == 15
    print(f"  Legacy single-core grammar intact: {g151.n_mechs} mechs, {g151.n_species} sp, {g151.n_reactions} rxns, {g151.n_params} params")

    g = build_multicore_grammar()
    assert g.n_cores == 4
    assert g.n_species == 13
    assert g.n_reactions == 24
    assert g.n_params == 19
    assert g.n_mechs == 527
    core_counts = {c.code: int(np.sum(g.mech_core_id == c.id)) for c in g.cores}
    assert core_counts == {"C1": 151, "C2": 151, "C3": 151, "C4": 74}, core_counts
    max_active_params = int(np.max(np.sum(g.mech_param_mask, axis=1)))
    assert max_active_params == 13, max_active_params
    atom_err = float(np.max(np.abs(g.atom_matrix @ g.stoich_matrix)))
    assert atom_err < 1e-12, atom_err
    n_3mod = int(np.sum(np.sum(g.mech_options > 0, axis=1) == 3))
    assert n_3mod == 334, n_3mod

    # Check every reaction is used by at least one mechanism (zero dead reactions)
    rxn_use = np.sum(g.mech_rxn_mask, axis=0)
    assert np.all(rxn_use > 0), f"Unused reactions: {np.where(rxn_use == 0)[0]}"

    # Verify C2 M2:complex and M3 bind catB / catB2, while C1/C3/C4 bind catA / catA2
    m_c2_m2c_m3 = g.get_mech_id_by_name("C2+M2:complex+M3")
    active_c2 = [g.reactions[r].name for r in np.where(g.mech_rxn_mask[m_c2_m2c_m3] == 1)[0]]
    assert "R12_deactCB" in active_c2 and "R11_deactCA" not in active_c2
    assert "R15_bindB2" in active_c2 and "R16_dissB2" in active_c2 and "R13_bindA2" not in active_c2

    # Verify C4 has no M7/M8 and no catAB reactions
    c4_ids = np.where(g.mech_core_id == 3)[0]
    assert np.all(g.mech_options[c4_ids, 6] == 0) and np.all(g.mech_options[c4_ids, 7] == 0)
    r_relP = g.rxn_idx_by_role["relP"]
    assert np.all(g.mech_rxn_mask[c4_ids, r_relP] == 0)

    print(
        f"  Multicore grammar OK: {g.n_mechs} mechs {core_counts}, "
        f"{g.n_species} sp, {g.n_reactions} rxns, {g.n_params} params (max active={max_active_params}), "
        f"3-mod={n_3mod}, max_atom_err={atom_err:.2e}",
        flush=True,
    )
    return g


def check_radau_accuracy(g, sim: ODESimulator, n_samples: int = 1000, seed: int = 2026):
    print(f"\n=== 2. ROS2 vs SciPy Radau (rtol=1e-10, atol=1e-14) on {n_samples} Multicore Samples ===", flush=True)
    rng = np.random.default_rng(seed)
    t_eval = np.linspace(0.0, 4.0, 80)

    # Sample balanced across all 4 cores & mix of catalog + continuous designs
    mids = np.empty(n_samples, dtype=np.int32)
    thetas = np.empty((n_samples, g.n_params), dtype=np.float64)
    y0s = np.empty((n_samples, g.n_species), dtype=np.float64)
    for i in range(n_samples):
        core_id = i % 4
        core_mechs = np.where(g.mech_core_id == core_id)[0]
        m = int(rng.choice(core_mechs))
        mids[i] = m
        thetas[i] = g.sample_prior_theta(m, 1, rng)[0]
        des = g.catalog[int(rng.integers(0, len(g.catalog)))] if (i % 2 == 0) else g.sample_continuous_design(rng)
        y0s[i] = g.initial_state(m, des)

    t0 = time.perf_counter()
    y_ros2, status, _, _ = sim.simulate_batch(mids, thetas, y0s, t_eval, compute_flux=False)
    dt_ros2 = time.perf_counter() - t0
    n_ros2_fails = int(np.sum(status != 0))

    max_abs_errs = []
    max_atom_errs = []
    n_radau_fails = 0
    t1 = time.perf_counter()
    tasks = [(i, int(mids[i]), thetas[i], y0s[i], t_eval) for i in range(n_samples)]
    with ProcessPoolExecutor(max_workers=32, initializer=_init_radau_worker) as ex:
        for i, y_ref, ok in ex.map(_radau_one, tasks, chunksize=8):
            if not ok:
                n_radau_fails += 1
                continue
            err = float(np.max(np.abs(y_ros2[i] - y_ref)))
            max_abs_errs.append(err)
            atoms0 = g.atom_matrix @ y0s[i]
            atoms_t = y_ros2[i] @ g.atom_matrix.T
            max_atom_errs.append(float(np.max(np.abs(atoms_t - atoms0[None, :]))))
    dt_radau = time.perf_counter() - t1

    max_err = float(np.max(max_abs_errs))
    p99_err = float(np.percentile(max_abs_errs, 99))
    med_err = float(np.median(max_abs_errs))
    max_atom_err = float(np.max(max_atom_errs))
    print(
        f"  ROS2 ({dt_ros2*1e3:.1f} ms) vs Radau ({dt_radau:.1f} s): "
        f"ros2_fails={n_ros2_fails}, radau_fails={n_radau_fails}, "
        f"med_err={med_err:.2e}, p99_err={p99_err:.2e}, max_err={max_err:.2e}, max_atom_err={max_atom_err:.2e}",
        flush=True,
    )
    assert n_ros2_fails == 0, f"ROS2 failures: {n_ros2_fails}"
    assert max_err < 1e-5, f"ROS2 max error {max_err:.2e} >= 1e-5"
    assert max_atom_err < 1e-10, f"Atom conservation error {max_atom_err:.2e} >= 1e-10"
    return {
        "n_samples": n_samples,
        "ros2_ms": dt_ros2 * 1e3,
        "radau_s": dt_radau,
        "ros2_fails": n_ros2_fails,
        "radau_fails": n_radau_fails,
        "median_abs_err": med_err,
        "p99_abs_err": p99_err,
        "max_abs_err": max_err,
        "max_atom_err": max_atom_err,
    }


def check_extended_cpp_apis(g, sim: ODESimulator):
    print("\n=== 3. Extended C++ Engine API Checks (var_t, return_theta, 0-iter forward) ===", flush=True)
    rng = np.random.default_rng(77)
    designs = [g.catalog[0], g.catalog[1], g.catalog[2]]  # D0, D1 (0.5x), D2 (2.0x)
    t_eval_2d = np.stack([
        np.linspace(0.0, 1.0, 20),
        np.linspace(0.0, 2.5, 20),
        np.linspace(0.0, 0.5, 20),
    ], axis=0)  # (3, 20)

    # Test simulate_designs with 2D per-design t_eval vs per-design 1D calls
    mids = np.array([0, 151, 302, 453], dtype=np.int32)
    thetas = np.stack([g.sample_prior_theta(int(m), 1, rng)[0] for m in mids], axis=0)
    y_var, st_var, _, _ = sim.simulate_designs(mids, thetas, designs, t_eval_2d)
    assert np.all(st_var == 0)
    for d_i, des in enumerate(designs):
        y_ref, st_ref, _, _ = sim.simulate_designs(mids, thetas, [des], t_eval_2d[d_i])
        assert np.all(st_ref == 0)
        diff = float(np.max(np.abs(y_var[:, d_i] - y_ref[:, 0])))
        assert diff < 1e-14, f"var_t mismatch on design {d_i}: {diff}"

    # Test fit_lm_pairs_dir(return_theta=True) + eval_pairs_forward exact RMS match
    # Fit C1 (mech 0) onto C3 (mech 302) curves across variable t_eval
    n_pairs, K = 2, 4
    src_mids = np.array([0, 453], dtype=np.int32)
    tgt_mids = np.array([302, 0], dtype=np.int32)
    n_res = sum(len(d.obs_species) for d in designs) * t_eval_2d.shape[1]
    targets = np.zeros((n_pairs, K, n_res), dtype=np.float64)
    inv_sig = np.ones(n_res, dtype=np.float64) / 0.01
    inits = np.zeros((n_pairs, K, 2, g.n_params), dtype=np.float64)
    for p in range(n_pairs):
        th_src = g.sample_prior_theta(int(src_mids[p]), K, rng)
        y_src, _, _, _ = sim.simulate_designs(np.full(K, src_mids[p], dtype=np.int32), th_src, designs, t_eval_2d)
        chunks = []
        for d_i, des in enumerate(designs):
            for sp in des.obs_species:
                chunks.append(y_src[:, d_i, :, sp])
        targets[p] = np.concatenate(chunks, axis=1)
        for s in range(2):
            inits[p, :, s] = g.sample_prior_theta(int(tgt_mids[p]), K, rng)

    rms_fit, theta_fit = sim.fit_lm_pairs_dir(
        tgt_mids, designs, t_eval_2d, targets, inv_sig, inits,
        max_lm_iters=10, early_stop_rms=0.49, fail_rms_thresh=1.0, max_fails=-1,
        return_theta=True,
    )
    rms_fwd = sim.eval_pairs_forward(tgt_mids, designs, t_eval_2d, targets, inv_sig, theta_fit)
    diff_fwd = float(np.max(np.abs(rms_fit - rms_fwd)))
    assert diff_fwd < 1e-10, f"Forward RMS mismatch vs fit RMS: {diff_fwd}"
    print(f"  Extended C++ APIs OK: var_t diff=0.0, fit-vs-forward 0-iter diff={diff_fwd:.2e}", flush=True)


def probe_acceptance(g, sim: ODESimulator, draws_per_mech: int = 2000, c3_branch_min: float = 0.15, unfiltered: bool = False, seed: int = 1234):
    rng = np.random.default_rng(seed)
    M = g.n_mechs
    t0 = time.perf_counter()
    acc_rates = np.zeros(M, dtype=np.float64)
    solver_fails = np.zeros(M, dtype=np.int64)
    for m in range(M):
        cand = g.sample_prior_theta(m, draws_per_mech, rng)
        acc, diag = check_observability_batch(
            g, sim, m, cand, unfiltered=unfiltered, c3_branch_min=c3_branch_min
        )
        acc_rates[m] = float(np.mean(acc))
        solver_fails[m] = int(np.sum(~diag["solver_ok"]))
    dt = time.perf_counter() - t0

    core_stats = {}
    for c in g.cores:
        mask = (g.mech_core_id == c.id)
        rates_c = acc_rates[mask]
        core_stats[c.code] = {
            "n_mechs": int(np.sum(mask)),
            "median": float(np.median(rates_c)),
            "mean": float(np.mean(rates_c)),
            "min": float(np.min(rates_c)),
            "p05": float(np.percentile(rates_c, 5)),
            "n_below_3pct": int(np.sum(rates_c < 0.03)),
            "n_below_5pct": int(np.sum(rates_c < 0.05)),
            "solver_fails": int(np.sum(solver_fails[mask])),
        }

    k_active = np.sum(g.mech_options > 0, axis=1)
    mod_count_stats = {}
    for k in range(4):
        rates_k = acc_rates[k_active == k]
        mod_count_stats[f"k={k}"] = {
            "n_mechs": int(len(rates_k)),
            "median": float(np.median(rates_k)),
            "min": float(np.min(rates_k)),
        }

    order = np.argsort(acc_rates)
    bottom15 = [
        {"mech_id": int(m), "name": g.mech_names[m], "core": g.cores[int(g.mech_core_id[m])].code, "acc": float(acc_rates[m])}
        for m in order[:15]
    ]

    label = "UNFILTERED" if unfiltered else f"FILTERED (c3_branch_min={c3_branch_min:.2f})"
    print(
        f"\n=== 4. Acceptance Probe [{label}] ({draws_per_mech} draws/mech, {dt:.1f}s) ===",
        flush=True,
    )
    print(
        f"  Overall (527 mechs): median={np.median(acc_rates)*100:.2f}%, mean={np.mean(acc_rates)*100:.2f}%, "
        f"min={np.min(acc_rates)*100:.2f}%, <3%={int(np.sum(acc_rates < 0.03))}, solver_fails={int(np.sum(solver_fails))}",
        flush=True,
    )
    for code, st in core_stats.items():
        print(
            f"    {code} ({st['n_mechs']}): median={st['median']*100:.2f}%, min={st['min']*100:.2f}%, "
            f"p05={st['p05']*100:.2f}%, <3%={st['n_below_3pct']}, fails={st['solver_fails']}",
            flush=True,
        )
    print("  Bottom 10 mechanisms:", flush=True)
    for b in bottom15[:10]:
        print(f"    [{b['mech_id']:3d}] {b['name']:32s} acc={b['acc']*100:.2f}%", flush=True)

    return {
        "label": label,
        "unfiltered": unfiltered,
        "c3_branch_min": c3_branch_min,
        "draws_per_mech": draws_per_mech,
        "elapsed_s": dt,
        "overall_median": float(np.median(acc_rates)),
        "overall_mean": float(np.mean(acc_rates)),
        "overall_min": float(np.min(acc_rates)),
        "n_below_3pct": int(np.sum(acc_rates < 0.03)),
        "total_solver_fails": int(np.sum(solver_fails)),
        "core_stats": core_stats,
        "mod_count_stats": mod_count_stats,
        "bottom15": bottom15,
        "all_rates": acc_rates.tolist(),
    }


def main():
    out_dir = Path("results/phase2")
    out_dir.mkdir(parents=True, exist_ok=True)

    g = check_grammar_invariants()
    sim = ODESimulator(g, n_threads=64)

    radau_res = check_radau_accuracy(g, sim, n_samples=1000, seed=2026)
    check_extended_cpp_apis(g, sim)

    probe_15 = probe_acceptance(g, sim, draws_per_mech=2000, c3_branch_min=0.15, unfiltered=False, seed=1234)
    probe_10 = None
    recommended_c3_branch_min = 0.15
    if probe_15["core_stats"]["C3"]["min"] < 0.03:
        print("\n  -> Detected C3 mechanism(s) < 3% at c3_branch_min=0.15; probing uniform c3_branch_min=0.10...", flush=True)
        probe_10 = probe_acceptance(g, sim, draws_per_mech=2000, c3_branch_min=0.10, unfiltered=False, seed=1234)
        recommended_c3_branch_min = 0.10

    probe_unfilt = probe_acceptance(g, sim, draws_per_mech=1000, c3_branch_min=0.15, unfiltered=True, seed=5678)

    report = {
        "radau_check": radau_res,
        "recommended_c3_branch_min": recommended_c3_branch_min,
        "filtered_c3_15": probe_15,
        "filtered_c3_10": probe_10,
        "unfiltered": probe_unfilt,
    }
    out_path = out_dir / "stage1a_probe.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved Stage 1a probe report to {out_path}", flush=True)


if __name__ == "__main__":
    main()
