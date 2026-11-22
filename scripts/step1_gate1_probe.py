#!/usr/bin/env python3
"""Step 1 (Gate 1): ROS2 ODE Solver Accuracy, Conservation, Throughput & 151-Mechanism Prior Probe."""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from quomech.grammar import build_default_grammar
from quomech.simulator import ODESimulator, check_observability_batch


def _radau_worker(args):
    mech_id, log10_k, y0, t_eval = args
    g = build_default_grammar()
    sim = ODESimulator(g, n_threads=1)
    y_ref, ok = sim.simulate_scipy_radau(mech_id, log10_k, y0, t_eval, rtol=1e-10, atol=1e-14)
    return y_ref, ok


def run_accuracy_and_throughput(grammar, sim, out_dir: Path, seed: int = 42):
    rng = np.random.default_rng(seed)
    t_eval = np.linspace(0.0, 1.0, 64)

    # 1. Accuracy & conservation on 1,000 random (mechanism, prior theta, design) triples
    n_acc = 1000
    mech_ids = rng.integers(0, grammar.n_mechs, size=n_acc, dtype=np.int32)
    thetas = np.zeros((n_acc, grammar.n_params), dtype=np.float64)
    y0s = np.zeros((n_acc, grammar.n_species), dtype=np.float64)
    for i in range(n_acc):
        m = int(mech_ids[i])
        thetas[i] = grammar.sample_prior_theta(m, 1, rng)[0]
        if i % 2 == 0:
            des = grammar.catalog[int(rng.integers(0, len(grammar.catalog)))]
        else:
            des = grammar.sample_continuous_design(rng)
        y0s[i] = grammar.initial_state(m, des)

    t0 = time.perf_counter()
    y_ros2, st_ros2, _, _ = sim.simulate_batch(mech_ids, thetas, y0s, t_eval)
    dt_ros2_1k = time.perf_counter() - t0

    # Mass conservation error across all 3 invariants (A, B, cat)
    ok_ros2 = st_ros2 == 0
    atom_mat = grammar.atom_matrix  # (3, n_species)
    atoms_init = np.einsum("as,bs->ba", atom_mat, y0s[ok_ros2])  # (B_ok, 3)
    atoms_traj = np.einsum("as,bts->bta", atom_mat, y_ros2[ok_ros2])  # (B_ok, T, 3)
    cons_err = np.max(np.abs(atoms_traj - atoms_init[:, None, :]), axis=(1, 2))

    # Parallel SciPy Radau reference on 32 workers
    print(f"[Gate 1] Running SciPy Radau (rtol=1e-10, atol=1e-14) on {n_acc} cases...")
    t0 = time.perf_counter()
    tasks = [(int(mech_ids[i]), thetas[i], y0s[i], t_eval) for i in range(n_acc)]
    with ProcessPoolExecutor(max_workers=32) as ex:
        radau_res = list(ex.map(_radau_worker, tasks, chunksize=8))
    dt_radau = time.perf_counter() - t0

    y_ref = np.array([r[0] for r in radau_res])
    ok_ref = np.array([r[1] for r in radau_res], dtype=bool)
    both_ok = ok_ros2 & ok_ref

    abs_err = np.max(np.abs(y_ros2[both_ok] - y_ref[both_ok]), axis=(1, 2))
    obs_idx = [grammar.species_idx["A"], grammar.species_idx["B"], grammar.species_idx["P"], grammar.species_idx["catAB"]]
    obs_err = np.max(np.abs(y_ros2[both_ok][:, :, obs_idx] - y_ref[both_ok][:, :, obs_idx]), axis=(1, 2))

    # 2. 64-core Throughput benchmark (100,000 trajectories)
    n_bench = 100000
    b_mids = rng.integers(0, grammar.n_mechs, size=n_bench, dtype=np.int32)
    b_thetas = np.zeros((n_bench, grammar.n_params), dtype=np.float64)
    b_y0s = np.zeros((n_bench, grammar.n_species), dtype=np.float64)
    d0 = grammar.catalog[0]
    for m in range(grammar.n_mechs):
        idx = np.where(b_mids == m)[0]
        if len(idx) > 0:
            b_thetas[idx] = grammar.sample_prior_theta(m, len(idx), rng)
            b_y0s[idx] = grammar.initial_state(m, d0)

    # Warmup
    sim.simulate_batch(b_mids[:1000], b_thetas[:1000], b_y0s[:1000], t_eval)

    # Timed runs at both rtol=1e-6 (high acc) and rtol=1e-5 (fast)
    bench_results = {}
    for label, rtol, atol in [("rtol1e-6", 1e-6, 1e-10), ("rtol1e-5", 1e-5, 1e-9)]:
        t0 = time.perf_counter()
        _, st_b, _, _ = sim.simulate_batch(b_mids, b_thetas, b_y0s, t_eval, rtol=rtol, atol=atol)
        dt = time.perf_counter() - t0
        bench_results[label] = {
            "n_traj": n_bench,
            "elapsed_s": dt,
            "traj_per_sec": n_bench / dt,
            "fail_count": int(np.sum(st_b != 0)),
            "fail_rate": float(np.mean(st_b != 0)),
        }

    # Also test rtol=1e-5 accuracy on the 1,000 cases
    y_ros2_e5, st_e5, _, _ = sim.simulate_batch(mech_ids, thetas, y0s, t_eval, rtol=1e-5, atol=1e-9)
    both_ok_e5 = (st_e5 == 0) & ok_ref
    abs_err_e5 = np.max(np.abs(y_ros2_e5[both_ok_e5] - y_ref[both_ok_e5]), axis=(1, 2))

    summary = {
        "n_acc_cases": n_acc,
        "ros2_rtol1e-6": {
            "ok_count": int(np.sum(ok_ros2)),
            "radau_ok_count": int(np.sum(ok_ref)),
            "both_ok_count": int(np.sum(both_ok)),
            "max_abs_err_all_species": float(np.max(abs_err)),
            "p99_abs_err_all_species": float(np.percentile(abs_err, 99)),
            "median_abs_err_all_species": float(np.median(abs_err)),
            "max_abs_err_obs_species": float(np.max(obs_err)),
            "max_conservation_err": float(np.max(cons_err)),
            "median_conservation_err": float(np.median(cons_err)),
            "elapsed_1k_s": dt_ros2_1k,
            "radau_elapsed_1k_s": dt_radau,
        },
        "ros2_rtol1e-5": {
            "both_ok_count": int(np.sum(both_ok_e5)),
            "max_abs_err_all_species": float(np.max(abs_err_e5)),
            "p99_abs_err_all_species": float(np.percentile(abs_err_e5, 99)),
            "median_abs_err_all_species": float(np.median(abs_err_e5)),
        },
        "throughput": bench_results,
    }
    print(json.dumps(summary, indent=2))
    return summary


def run_prior_probe(grammar, sim, n_draws_per_mech: int = 2000, seed: int = 123):
    rng = np.random.default_rng(seed)
    t_eval = np.linspace(0.0, 1.0, 64)
    t0 = time.perf_counter()

    per_mech = []
    acc_rates = np.zeros(grammar.n_mechs)
    fail_rates = np.zeros(grammar.n_mechs)
    module_pass_accum = {m.code: [] for m in grammar.modules}
    conv_low_rates = []
    conv_high_rates = []

    for m in range(grammar.n_mechs):
        thetas = grammar.sample_prior_theta(m, n_draws_per_mech, rng)
        acc, diag = check_observability_batch(grammar, sim, m, thetas, t_eval=t_eval)
        acc_rate = float(np.mean(acc))
        fail_rate = float(np.mean(~diag["solver_ok"]))
        acc_rates[m] = acc_rate
        fail_rates[m] = fail_rate
        conv_low_rates.append(float(np.mean(diag["conv_low"])))
        conv_high_rates.append(float(np.mean(diag["conv_high"])))

        diag_summary = {k: float(np.mean(v)) for k, v in diag.items()}
        for mod in grammar.modules:
            if mod.code in diag:
                module_pass_accum[mod.code].append(diag_summary[mod.code])

        per_mech.append({
            "mech_id": m,
            "name": grammar.mech_names[m],
            "k_active": int(np.sum(grammar.mech_options[m] > 0)),
            "acc_rate": acc_rate,
            "solver_fail_rate": fail_rate,
            "diag": diag_summary,
        })

    dt = time.perf_counter() - t0
    order = np.argsort(acc_rates)
    worst_10 = [per_mech[i] for i in order[:10]]
    best_5 = [per_mech[i] for i in order[-5:][::-1]]

    by_k = {}
    for k in range(grammar.max_active_modules + 1):
        ks = [p["acc_rate"] for p in per_mech if p["k_active"] == k]
        by_k[f"k={k}"] = {
            "count": len(ks),
            "min": float(np.min(ks)),
            "median": float(np.median(ks)),
            "mean": float(np.mean(ks)),
            "max": float(np.max(ks)),
        }

    mod_pass_summary = {
        code: {
            "mean_pass": float(np.mean(vals)),
            "min_pass": float(np.min(vals)),
            "median_pass": float(np.median(vals)),
        }
        for code, vals in module_pass_accum.items()
    }

    probe_summary = {
        "prior_means": grammar.prior_means_by_type,
        "n_draws_per_mech": n_draws_per_mech,
        "elapsed_s": dt,
        "overall_acceptance": {
            "min": float(np.min(acc_rates)),
            "p05": float(np.percentile(acc_rates, 5)),
            "p10": float(np.percentile(acc_rates, 10)),
            "median": float(np.median(acc_rates)),
            "mean": float(np.mean(acc_rates)),
            "max": float(np.max(acc_rates)),
            "n_below_3pct": int(np.sum(acc_rates < 0.03)),
            "n_below_5pct": int(np.sum(acc_rates < 0.05)),
            "n_below_10pct": int(np.sum(acc_rates < 0.10)),
        },
        "solver_fail_overall": {
            "mean": float(np.mean(fail_rates)),
            "max": float(np.max(fail_rates)),
        },
        "conv_pass_mean": {
            "ge_25pct": float(np.mean(conv_low_rates)),
            "le_98pct": float(np.mean(conv_high_rates)),
        },
        "by_k_active": by_k,
        "by_module_individual_pass": mod_pass_summary,
        "worst_10": [
            {"mech_id": p["mech_id"], "name": p["name"], "acc_rate": p["acc_rate"], "diag": p["diag"]}
            for p in worst_10
        ],
        "best_5": [
            {"mech_id": p["mech_id"], "name": p["name"], "acc_rate": p["acc_rate"]}
            for p in best_5
        ],
    }
    return probe_summary, per_mech


def main():
    out_dir = Path(__file__).resolve().parents[1] / "results" / "gate1"
    out_dir.mkdir(parents=True, exist_ok=True)

    grammar = build_default_grammar()
    print(f"Loaded grammar: {grammar.n_species} species, {grammar.n_reactions} reactions, "
          f"{grammar.n_params} params, {grammar.n_mechs} mechanisms.")
    sim = ODESimulator(grammar, rtol=1e-6, atol=1e-10, n_threads=64)

    acc_summary = run_accuracy_and_throughput(grammar, sim, out_dir)
    print("\n[Gate 1] Running 151-mechanism prior acceptance probe (2,000 draws/mech)...")
    probe_summary, per_mech = run_prior_probe(grammar, sim, n_draws_per_mech=2000)

    print("\n=== Prior Acceptance Summary (Default Proposal Means) ===")
    print(json.dumps(probe_summary, indent=2))

    with open(out_dir / "gate1_summary.json", "w") as f:
        json.dump({"solver": acc_summary, "prior_probe": probe_summary}, f, indent=2)
    with open(out_dir / "gate1_per_mech.json", "w") as f:
        json.dump(per_mech, f, indent=2)


if __name__ == "__main__":
    main()
