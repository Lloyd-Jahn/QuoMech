"""Network-independent two-stage ε-equivalence class builder (Section 7.3).

Strictly zero neural network involvement:
  Stage 1: All module-Hamming <= 2 pairs + curve-library NN pre-screen (RMS_NN <= 3.0)
           -> coarse bidirectional LM fit (K1=8 samples x 2 starts).
  Stage 2: High-precision bidirectional LM confirmation (K2=32 independent samples x 4 starts).
  Clustering: Connected components + diameter check + deterministic max-clique split
              ordered by mean edge residual for any non-transitive component.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
from scipy.stats import beta as beta_dist

from quomech.grammar import DesignSpec, ReactionGrammar
from quomech.simulator import ODESimulator, sample_accepted_thetas


def clopper_pearson_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Exact 95% Clopper-Pearson binomial confidence interval."""
    lo = 0.0 if k == 0 else float(beta_dist.ppf(alpha / 2.0, k, n - k + 1))
    hi = 1.0 if k == n else float(beta_dist.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return lo, hi


def build_design_targets(
    grammar: ReactionGrammar,
    sim: ODESimulator,
    mech_ids: np.ndarray,
    thetas: np.ndarray,
    designs: Sequence[DesignSpec],
    t_eval: np.ndarray,
    sigma_ref: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate (mech_ids, thetas) on designs and flatten observed channels + 1/sigma weights.

    Noise scale convention (data-driven via SpeciesSpec.is_catalyst):
      - Macroscopic species (A, B, P): sigma = sigma_ref * design.A0
      - Catalyst species (e.g. catAB in D9): sigma = sigma_ref * (design.cat_mult * c0_base)

    Returns:
        targets_flat: (B, n_res) float64
        inv_sigma_1d: (n_res,) float64 (same for all samples sharing `designs`)
        ok_mask: (B,) bool
    """
    y_all, status, _, _ = sim.simulate_designs(mech_ids, thetas, designs, t_eval, compute_flux=False)
    ok_mask = np.all(status == 0, axis=1)
    B = len(mech_ids)
    n_t = len(t_eval)

    chunks = []
    inv_sig_chunks = []
    for d_idx, des in enumerate(designs):
        cat0 = des.cat_mult * grammar.c0_base
        for sp in des.obs_species:
            scale = cat0 if grammar.species[sp].is_catalyst else des.A0
            sig = sigma_ref * scale
            chunks.append(y_all[:, d_idx, :, sp])  # (B, n_t)
            inv_sig_chunks.append(np.full(n_t, 1.0 / sig, dtype=np.float64))

    targets_flat = np.concatenate(chunks, axis=1)  # (B, n_res)
    inv_sigma_1d = np.concatenate(inv_sig_chunks, axis=0)  # (n_res,)
    return targets_flat, inv_sigma_1d, ok_mask


def sample_mechanism_pools(
    grammar: ReactionGrammar,
    sim: ODESimulator,
    n_lib: int = 128,
    k1: int = 8,
    k2: int = 32,
    seed: int = 2026,
) -> Dict[str, np.ndarray]:
    """Draw independent accepted parameter pools for library, Stage 1, and Stage 2."""
    rng = np.random.default_rng(seed)
    n_tot = n_lib + k1 + k2
    M = grammar.n_mechs
    P = grammar.n_params

    lib_thetas = np.zeros((M, n_lib, P), dtype=np.float64)
    s1_thetas = np.zeros((M, k1, P), dtype=np.float64)
    s2_thetas = np.zeros((M, k2, P), dtype=np.float64)

    for m in range(M):
        th, _, _ = sample_accepted_thetas(grammar, sim, m, n_tot, rng, batch_size=2048)
        if len(th) < n_tot:
            raise RuntimeError(f"Mech {m} ({grammar.mech_names[m]}) only got {len(th)}/{n_tot} accepted draws")
        lib_thetas[m] = th[:n_lib]
        s1_thetas[m] = th[n_lib : n_lib + k1]
        s2_thetas[m] = th[n_lib + k1 : n_tot]

    return {"lib": lib_thetas, "s1": s1_thetas, "s2": s2_thetas}


def _bron_kerbosch_all_maximal_cliques(adj: Dict[int, set]) -> List[List[int]]:
    """Deterministic Bron-Kerbosch with pivoting to enumerate all maximal cliques."""
    cliques: List[List[int]] = []

    def recurse(R: set, P: set, X: set):
        if not P and not X:
            cliques.append(sorted(R))
            return
        # Pivot from P U X with max neighbors in P
        union_px = P | X
        pivot = max(union_px, key=lambda u: len(adj[u] & P))
        for v in sorted(P - adj[pivot]):
            recurse(R | {v}, P & adj[v], X & adj[v])
            P = P - {v}
            X = X | {v}

    recurse(set(), set(adj.keys()), set())
    return cliques


def partition_equivalence_graph(
    n_mechs: int,
    pair_rms_i2j: Dict[Tuple[int, int], np.ndarray],
    pair_rms_j2i: Dict[Tuple[int, int], np.ndarray],
    rms_thresh: float = 1.0,
    pass_rate_thresh: float = 0.85,
) -> Dict:
    """Construct equivalence classes from Stage 2 bidirectional RMS arrays.

    Args:
        pair_rms_i2j: dict[(i, j)] -> (K2,) normalized RMS at sigma_ref=1%
        pair_rms_j2i: dict[(i, j)] -> (K2,) normalized RMS at sigma_ref=1%
        rms_thresh: 1.0 for sigma_ref=1%, 0.5 for sigma_ref=0.5%, 2.0 for sigma_ref=2%
        pass_rate_thresh: e.g. 0.85 (>= 28/32)
    """
    adj: Dict[int, set] = {m: set() for m in range(n_mechs)}
    edge_weight: Dict[Tuple[int, int], float] = {}
    passing_edges = []

    for (i, j), rms_ij in pair_rms_i2j.items():
        rms_ji = pair_rms_j2i[(i, j)]
        k_ij = int(np.sum(rms_ij <= rms_thresh))
        k_ji = int(np.sum(rms_ji <= rms_thresh))
        n_k = len(rms_ij)
        pr_ij = k_ij / n_k
        pr_ji = k_ji / n_k
        if pr_ij >= pass_rate_thresh and pr_ji >= pass_rate_thresh:
            adj[i].add(j)
            adj[j].add(i)
            w = 0.5 * (float(np.mean(rms_ij)) + float(np.mean(rms_ji)))
            edge_weight[(i, j)] = w
            edge_weight[(j, i)] = w
            ci_ij = clopper_pearson_ci(k_ij, n_k)
            ci_ji = clopper_pearson_ci(k_ji, n_k)
            passing_edges.append({
                "i": i, "j": j,
                "pass_i2j": pr_ij, "pass_j2i": pr_ji,
                "k_i2j": k_ij, "k_j2i": k_ji, "n": n_k,
                "ci_i2j": ci_ij, "ci_j2i": ci_ji,
                "mean_rms_1pct": w,
            })

    # Find connected components
    visited = set()
    components: List[List[int]] = []
    for m in range(n_mechs):
        if m not in visited:
            comp = []
            stack = [m]
            visited.add(m)
            while stack:
                u = stack.pop()
                comp.append(u)
                for v in sorted(adj[u]):
                    if v not in visited:
                        visited.add(v)
                        stack.append(v)
            components.append(sorted(comp))

    classes: List[List[int]] = []
    n_gt2 = 0
    n_non_trans = 0
    non_trans_details = []

    for comp in components:
        if len(comp) <= 2:
            classes.append(comp)
            continue
        n_gt2 += 1
        # Diameter check: is every pair inside comp connected?
        is_clique = True
        for a_idx in range(len(comp)):
            for b_idx in range(a_idx + 1, len(comp)):
                if comp[b_idx] not in adj[comp[a_idx]]:
                    is_clique = False
                    break
            if not is_clique:
                break
        if is_clique:
            classes.append(comp)
        else:
            n_non_trans += 1
            # Deterministic max-clique partition ordered by largest size, then lowest mean edge residual
            rem = set(comp)
            sub_cliques = []
            while rem:
                sub_adj = {u: (adj[u] & rem) for u in rem}
                max_cliques = _bron_kerbosch_all_maximal_cliques(sub_adj)

                def clique_key(clq: List[int]):
                    if len(clq) <= 1:
                        avg_w = 0.0
                    else:
                        ws = [
                            edge_weight[(clq[a], clq[b])]
                            for a in range(len(clq))
                            for b in range(a + 1, len(clq))
                        ]
                        avg_w = float(np.mean(ws))
                    return (-len(clq), avg_w, clq)

                best = min(max_cliques, key=clique_key)
                sub_cliques.append(best)
                rem -= set(best)
            classes.extend(sub_cliques)
            non_trans_details.append({"original_comp": comp, "split_into": sub_cliques})

    # Sort classes canonically by (-size, min_member)
    classes.sort(key=lambda c: (-len(c), c[0]))
    mech_to_class = np.zeros(n_mechs, dtype=np.int32)
    for c_id, members in enumerate(classes):
        for m in members:
            mech_to_class[m] = c_id

    sizes = [len(c) for c in classes]
    size_hist: Dict[int, int] = {}
    for s in sizes:
        size_hist[s] = size_hist.get(s, 0) + 1

    return {
        "rms_thresh_1pct": rms_thresh,
        "pass_rate_thresh": pass_rate_thresh,
        "n_classes": len(classes),
        "n_singleton": sum(1 for s in sizes if s == 1),
        "n_nontrivial_classes": sum(1 for s in sizes if s >= 2),
        "n_mechs_in_nontrivial": sum(s for s in sizes if s >= 2),
        "max_class_size": max(sizes),
        "size_histogram": {str(k): v for k, v in sorted(size_hist.items())},
        "n_passing_edges": len(passing_edges),
        "n_components_gt2": n_gt2,
        "n_non_transitive_components": n_non_trans,
        "non_transitive_fraction_of_gt2": (n_non_trans / n_gt2) if n_gt2 > 0 else 0.0,
        "non_transitive_details": non_trans_details,
        "classes": classes,
        "mech_to_class": mech_to_class.tolist(),
        "passing_edges": passing_edges,
    }


def run_two_stage_equivalence(
    grammar: ReactionGrammar,
    sim: ODESimulator,
    design_set_name: str,
    design_indices: Sequence[int],
    pools: Dict[str, np.ndarray],
    sigma_ref_base: float = 0.01,
    n_t: int = 20,
    nn_prescreen_thresh: float = 3.0,
    stage1_rms_cut: float = 2.0,
    stage1_min_pass: float = 0.50,
    next_tier_keep_rms: Optional[float] = None,
    allowed_pairs: Optional[Sequence[Tuple[int, int]]] = None,
    seed: int = 999,
) -> Dict:
    """Execute full network-independent two-stage ε-equivalence analysis for one design set.

    If `allowed_pairs` is provided (from a provable subset-design lower bound on a strict
    subset tier D_sub ⊂ D), restricts Stage 1 to `allowed_pairs` union any pair passing
    the fast curve-library NN pre-screen on the current design set.
    """
    t_start = time.perf_counter()
    rng = np.random.default_rng(seed)
    M = grammar.n_mechs
    P = grammar.n_params
    designs = [grammar.catalog[i] for i in design_indices]
    t_eval = np.linspace(0.0, 1.0, n_t)

    lib_th = pools["lib"]  # (M, N_lib, P)
    s1_th = pools["s1"]    # (M, K1, P)
    s2_th = pools["s2"]    # (M, K2, P)
    N_lib, K1, K2 = lib_th.shape[1], s1_th.shape[1], s2_th.shape[1]

    # Simulate library, S1, and S2 curves for all 151 mechanisms on this design set
    mids_lib = np.repeat(np.arange(M, dtype=np.int32), N_lib)
    lib_flat, inv_sig_1d, _ = build_design_targets(
        grammar, sim, mids_lib, lib_th.reshape(M * N_lib, P), designs, t_eval, sigma_ref_base
    )
    n_res = lib_flat.shape[1]
    lib_curves = lib_flat.reshape(M, N_lib, n_res)
    lib_scaled = lib_curves * inv_sig_1d[None, None, :]  # scaled by 1/sigma_1pct

    mids_s1 = np.repeat(np.arange(M, dtype=np.int32), K1)
    s1_flat, _, _ = build_design_targets(
        grammar, sim, mids_s1, s1_th.reshape(M * K1, P), designs, t_eval, sigma_ref_base
    )
    s1_curves = s1_flat.reshape(M, K1, n_res)
    s1_scaled = s1_curves * inv_sig_1d[None, None, :]

    mids_s2 = np.repeat(np.arange(M, dtype=np.int32), K2)
    s2_flat, _, _ = build_design_targets(
        grammar, sim, mids_s2, s2_th.reshape(M * K2, P), designs, t_eval, sigma_ref_base
    )
    s2_curves = s2_flat.reshape(M, K2, n_res)
    s2_scaled = s2_curves * inv_sig_1d[None, None, :]

    # Bounds: shared prior mean +/- 3.5 * std
    lb_all = grammar.mech_prior_mean - 3.5 * grammar.mech_prior_std
    ub_all = grammar.mech_prior_mean + 3.5 * grammar.mech_prior_std

    # ------------------------------------------------------------------
    # Candidate pair selection (100% deterministic, zero NN)
    # ------------------------------------------------------------------
    hamming = grammar.module_hamming_matrix()
    X = lib_scaled.reshape(M * N_lib, n_res)
    X_sq = np.sum(X * X, axis=1)
    nn_med = np.zeros((M, M), dtype=np.float64)
    for i in range(M):
        Xi = lib_scaled[i]
        Xi_sq = np.sum(Xi * Xi, axis=1)[:, None]
        d2 = np.maximum(Xi_sq + X_sq[None, :] - 2.0 * (Xi @ X.T), 0.0)
        d2 = d2.reshape(N_lib, M, N_lib)
        min_rms = np.sqrt(np.min(d2, axis=2) / n_res)
        # Median NN RMS (Section 7.3)
        nn_med[i] = np.median(min_rms, axis=0)

    allowed_set = set(allowed_pairs) if allowed_pairs is not None else None
    cand_pairs: List[Tuple[int, int]] = []
    n_hamming_pairs = 0
    n_nn_extra_pairs = 0
    for i in range(M):
        for j in range(i + 1, M):
            in_nn = min(nn_med[i, j], nn_med[j, i]) <= nn_prescreen_thresh
            if allowed_set is not None:
                if (i, j) in allowed_set or in_nn:
                    cand_pairs.append((i, j))
                    if hamming[i, j] <= 2:
                        n_hamming_pairs += 1
                    else:
                        n_nn_extra_pairs += 1
            else:
                if hamming[i, j] <= 2:
                    cand_pairs.append((i, j))
                    n_hamming_pairs += 1
                elif in_nn:
                    cand_pairs.append((i, j))
                    n_nn_extra_pairs += 1

    t_prescreen = time.perf_counter() - t_start
    print(
        f"[Equiv:{design_set_name}] Pre-screen ({t_prescreen:.1f}s): "
        f"{n_hamming_pairs} Hamming<=2 + {n_nn_extra_pairs} NN-extra = {len(cand_pairs)} pairs"
    )

    lib_sq_per_m = np.sum(lib_scaled * lib_scaled, axis=2)  # (M, N_lib)

    def get_nn_thetas(query_scaled: np.ndarray, target_m: int, topk: int) -> np.ndarray:
        q_sq = np.sum(query_scaled * query_scaled, axis=1)[:, None]
        d2 = q_sq + lib_sq_per_m[target_m][None, :] - 2.0 * (query_scaled @ lib_scaled[target_m].T)
        idx = np.argpartition(d2, topk, axis=1)[:, :topk]
        rows = np.arange(len(query_scaled))[:, None]
        ord_sub = np.argsort(d2[rows, idx], axis=1)
        idx_sorted = idx[rows, ord_sub]
        return lib_th[target_m][idx_sorted]

    # ------------------------------------------------------------------
    # Stage 1: Coarse bidirectional LM (K1=8 x 2 starts) with threshold-matched early stop
    # ------------------------------------------------------------------
    t_s1_start = time.perf_counter()
    n_cand = len(cand_pairs)
    keep_cut = max(stage1_rms_cut, next_tier_keep_rms) if next_tier_keep_rms else stage1_rms_cut
    max_fails_s1 = K1 - int(np.ceil(stage1_min_pass * K1))  # 8 - 4 = 4
    # Stop Stage 1 LM immediately once rms <= 0.95 (already passes 1.0% AND 2.0% AND next_tier!)
    s1_early_stop = 0.95

    # Pass 1a: direction i -> j
    s1a_mids = np.empty(n_cand, dtype=np.int32)
    s1a_tgt = np.empty((n_cand, K1, n_res), dtype=np.float64)
    s1a_inits = np.empty((n_cand, K1, 2, P), dtype=np.float64)
    for idx_p, (i, j) in enumerate(cand_pairs):
        s1a_mids[idx_p] = j
        s1a_tgt[idx_p] = s1_curves[i]
        s1a_inits[idx_p, :, 0] = get_nn_thetas(s1_scaled[i], j, topk=1)[:, 0, :]
        s1a_inits[idx_p, :, 1] = lib_th[j][rng.integers(0, N_lib, size=K1)]

    s1a_rms = sim.fit_lm_pairs_dir(
        s1a_mids, designs, t_eval, s1a_tgt, inv_sig_1d, s1a_inits,
        lb_all=lb_all, ub_all=ub_all,
        rtol=1e-4, atol=1e-8, max_ode_steps=8000, max_lm_iters=14,
        early_stop_rms=s1_early_stop, fail_rms_thresh=keep_cut, max_fails=max_fails_s1,
    )

    pass_1a_indices = [
        idx_p for idx_p in range(n_cand)
        if float(np.mean(s1a_rms[idx_p] <= keep_cut)) >= stage1_min_pass
    ]

    # Pass 1b: direction j -> i for Pass 1a survivors
    n_1b = len(pass_1a_indices)
    stage2_1pct_pairs: List[Tuple[int, int]] = []
    stage2_2pct_extra_pairs: List[Tuple[int, int]] = []
    next_tier_pairs: List[Tuple[int, int]] = []

    if n_1b > 0:
        s1b_mids = np.empty(n_1b, dtype=np.int32)
        s1b_tgt = np.empty((n_1b, K1, n_res), dtype=np.float64)
        s1b_inits = np.empty((n_1b, K1, 2, P), dtype=np.float64)
        for k_b, idx_p in enumerate(pass_1a_indices):
            i, j = cand_pairs[idx_p]
            s1b_mids[k_b] = i
            s1b_tgt[k_b] = s1_curves[j]
            s1b_inits[k_b, :, 0] = get_nn_thetas(s1_scaled[j], i, topk=1)[:, 0, :]
            s1b_inits[k_b, :, 1] = lib_th[i][rng.integers(0, N_lib, size=K1)]

        s1b_rms = sim.fit_lm_pairs_dir(
            s1b_mids, designs, t_eval, s1b_tgt, inv_sig_1d, s1b_inits,
            lb_all=lb_all, ub_all=ub_all,
            rtol=1e-4, atol=1e-8, max_ode_steps=8000, max_lm_iters=14,
            early_stop_rms=s1_early_stop, fail_rms_thresh=keep_cut, max_fails=max_fails_s1,
        )

        for k_b, idx_p in enumerate(pass_1a_indices):
            pair = cand_pairs[idx_p]
            r_ij = s1a_rms[idx_p]
            r_ji = s1b_rms[k_b]
            if float(np.mean(r_ij <= keep_cut)) >= stage1_min_pass and float(np.mean(r_ji <= keep_cut)) >= stage1_min_pass:
                next_tier_pairs.append(pair)
            pass_1 = (float(np.mean(r_ij <= 1.0)) >= stage1_min_pass and float(np.mean(r_ji <= 1.0)) >= stage1_min_pass)
            if pass_1:
                stage2_1pct_pairs.append(pair)
            elif stage1_rms_cut >= 2.0 and float(np.mean(r_ij <= 2.0)) >= stage1_min_pass and float(np.mean(r_ji <= 2.0)) >= stage1_min_pass:
                stage2_2pct_extra_pairs.append(pair)

    dt_s1 = time.perf_counter() - t_s1_start
    print(
        f"[Equiv:{design_set_name}] Stage 1 done ({dt_s1:.1f}s): "
        f"1a->{n_1b}, S2(1%)={len(stage2_1pct_pairs)}, S2(2% extra)={len(stage2_2pct_extra_pairs)}, next_tier={len(next_tier_pairs)}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Stage 2: High-precision bidirectional LM (K2=32 x 4 starts)
    # Threshold-matched exact confirmation (10x faster, ZERO approximation):
    #   (A) Primary sigma_ref=1.0%: test stage2_1pct_pairs with early_stop=0.98, fail_thresh=1.0
    #   (B) Sensitivity 0.5%: re-test ONLY 1.0%-passers with early_stop=0.49, fail_thresh=0.5
    #   (C) Sensitivity 2.0%: 1.0%-passers auto-pass; test (1.0%-failers U 2%-extra) with early_stop=1.96, fail_thresh=2.0
    # ------------------------------------------------------------------
    t_s2_start = time.perf_counter()
    max_fails_s2 = K2 - int(np.ceil(0.80 * K2))  # 32 - 26 = 6 (covers 80%, 85%, 90% exactly!)

    def run_s2_level(
        pair_list: List[Tuple[int, int]], rms_thresh: float, early_stop: float, label: str
    ) -> Tuple[Dict[Tuple[int, int], np.ndarray], Dict[Tuple[int, int], np.ndarray]]:
        out_ij: Dict[Tuple[int, int], np.ndarray] = {}
        out_ji: Dict[Tuple[int, int], np.ndarray] = {}
        n_p = len(pair_list)
        if n_p == 0:
            return out_ij, out_ji
        t0_l = time.perf_counter()
        # Pass a: i -> j
        mids_a = np.empty(n_p, dtype=np.int32)
        tgt_a = np.empty((n_p, K2, n_res), dtype=np.float64)
        init_a = np.empty((n_p, K2, 4, P), dtype=np.float64)
        for idx_p, (i, j) in enumerate(pair_list):
            mids_a[idx_p] = j
            tgt_a[idx_p] = s2_curves[i]
            init_a[idx_p, :, 0:2] = get_nn_thetas(s2_scaled[i], j, topk=2)
            init_a[idx_p, :, 2:4] = lib_th[j][rng.integers(0, N_lib, size=(K2, 2))]
        rms_a = sim.fit_lm_pairs_dir(
            mids_a, designs, t_eval, tgt_a, inv_sig_1d, init_a,
            lb_all=lb_all, ub_all=ub_all,
            rtol=1e-4, atol=1e-8, max_ode_steps=10000, max_lm_iters=20,
            early_stop_rms=early_stop, fail_rms_thresh=rms_thresh, max_fails=max_fails_s2,
        )
        surv_a = [
            idx_p for idx_p in range(n_p)
            if int(np.sum(rms_a[idx_p] <= rms_thresh)) >= (K2 - max_fails_s2)
        ]
        # Pass b: j -> i only for Pass a survivors
        n_b = len(surv_a)
        if n_b > 0:
            mids_b = np.empty(n_b, dtype=np.int32)
            tgt_b = np.empty((n_b, K2, n_res), dtype=np.float64)
            init_b = np.empty((n_b, K2, 4, P), dtype=np.float64)
            for kb, idx_p in enumerate(surv_a):
                i, j = pair_list[idx_p]
                mids_b[kb] = i
                tgt_b[kb] = s2_curves[j]
                init_b[kb, :, 0:2] = get_nn_thetas(s2_scaled[j], i, topk=2)
                init_b[kb, :, 2:4] = lib_th[i][rng.integers(0, N_lib, size=(K2, 2))]
            rms_b = sim.fit_lm_pairs_dir(
                mids_b, designs, t_eval, tgt_b, inv_sig_1d, init_b,
                lb_all=lb_all, ub_all=ub_all,
                rtol=1e-4, atol=1e-8, max_ode_steps=10000, max_lm_iters=20,
                early_stop_rms=early_stop, fail_rms_thresh=rms_thresh, max_fails=max_fails_s2,
            )
            for kb, idx_p in enumerate(surv_a):
                pair = pair_list[idx_p]
                out_ij[pair] = rms_a[idx_p]
                out_ji[pair] = rms_b[kb]
        print(
            f"  [Equiv:{design_set_name} S2 {label}] {n_p} cand -> {n_b} passed 2a ({time.perf_counter()-t0_l:.1f}s)",
            flush=True,
        )
        return out_ij, out_ji

    # (A) Primary 1.0%
    ij_1pct, ji_1pct = run_s2_level(stage2_1pct_pairs, rms_thresh=1.0, early_stop=0.98, label="1.0%")
    pass_1pct_pairs = [
        p for p in ij_1pct
        if int(np.sum(ij_1pct[p] <= 1.0)) >= (K2 - max_fails_s2)
        and int(np.sum(ji_1pct[p] <= 1.0)) >= (K2 - max_fails_s2)
    ]

    # (B) Sensitivity 0.5% (strict subset of 1.0%-passers!)
    ij_05pct, ji_05pct = run_s2_level(pass_1pct_pairs, rms_thresh=0.5, early_stop=0.49, label="0.5%")

    # (C) Sensitivity 2.0% (1.0%-passers automatically pass <= 2.0; check remaining candidates)
    pass_1_set = set(pass_1pct_pairs)
    cand_2pct_rem = [p for p in stage2_1pct_pairs if p not in pass_1_set] + stage2_2pct_extra_pairs
    ij_2pct_rem, ji_2pct_rem = run_s2_level(cand_2pct_rem, rms_thresh=2.0, early_stop=1.96, label="2.0%")
    ij_2pct = dict(ij_1pct)
    ji_2pct = dict(ji_1pct)
    ij_2pct.update(ij_2pct_rem)
    ji_2pct.update(ji_2pct_rem)

    dt_s2 = time.perf_counter() - t_s2_start
    n_s2 = len(stage2_1pct_pairs) + len(stage2_2pct_extra_pairs)
    print(f"[Equiv:{design_set_name}] Stage 2 all levels done ({dt_s2:.1f}s)", flush=True)

    rms_maps = {
        "0.5pct": (0.5, ij_05pct, ji_05pct),
        "1.0pct": (1.0, ij_1pct, ji_1pct),
        "2.0pct": (2.0, ij_2pct, ji_2pct),
    }
    partitions = {}
    for sig_label, (rms_cut, map_ij, map_ji) in rms_maps.items():
        for pr_label, pr_cut in [("80pct", 0.80), ("85pct", 0.85), ("90pct", 0.90)]:
            key = f"sig_{sig_label}_pr_{pr_label}"
            partitions[key] = partition_equivalence_graph(
                M, map_ij, map_ji, rms_thresh=rms_cut, pass_rate_thresh=pr_cut
            )

    primary = partitions["sig_1.0pct_pr_85pct"]
    print(
        f"[Equiv:{design_set_name}] Primary (1%, >=85%): {primary['n_classes']} classes "
        f"({primary['n_nontrivial_classes']} non-trivial covering {primary['n_mechs_in_nontrivial']} mechs, "
        f"max size {primary['max_class_size']}, non-trans={primary['n_non_transitive_components']}/{primary['n_components_gt2']})"
    )

    return {
        "design_set": design_set_name,
        "design_names": [grammar.catalog[i].name for i in design_indices],
        "n_designs": len(design_indices),
        "n_hamming_pairs": n_hamming_pairs,
        "n_nn_extra_pairs": n_nn_extra_pairs,
        "n_stage1_pairs": n_cand,
        "n_stage2_pairs": n_s2,
        "next_tier_pairs": next_tier_pairs,
        "elapsed_prescreen_s": t_prescreen,
        "elapsed_stage1_s": dt_s1,
        "elapsed_stage2_s": dt_s2,
        "elapsed_total_s": time.perf_counter() - t_start,
        "primary": primary,
        "sensitivity": {
            k: {
                "n_classes": v["n_classes"],
                "n_nontrivial_classes": v["n_nontrivial_classes"],
                "n_mechs_in_nontrivial": v["n_mechs_in_nontrivial"],
                "max_class_size": v["max_class_size"],
                "n_passing_edges": v["n_passing_edges"],
                "n_non_transitive": v["n_non_transitive_components"],
            }
            for k, v in partitions.items()
        },
        "partitions_full": {
            k: {
                "classes": partitions[k]["classes"],
                "mech_to_class": partitions[k]["mech_to_class"],
                "size_histogram": partitions[k]["size_histogram"],
            }
            for k in ["sig_0.5pct_pr_85pct", "sig_1.0pct_pr_85pct", "sig_2.0pct_pr_85pct"]
        },
    }
