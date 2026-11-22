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
    # Stage 1: Coarse bidirectional LM (K1=8 x 2 starts) with exact binomial early exit
    # ------------------------------------------------------------------
    t_s1_start = time.perf_counter()
    n_cand = len(cand_pairs)
    keep_cut = max(stage1_rms_cut, next_tier_keep_rms) if next_tier_keep_rms else stage1_rms_cut
    max_fails_s1 = K1 - int(np.ceil(stage1_min_pass * K1))  # e.g. 8 - 4 = 4

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
        rtol=1e-4, atol=1e-8, max_ode_steps=8000, max_lm_iters=16,
        early_stop_rms=0.45, fail_rms_thresh=keep_cut, max_fails=max_fails_s1,
    )

    pass_1a_indices = [
        idx_p for idx_p in range(n_cand)
        if float(np.mean(s1a_rms[idx_p] <= keep_cut)) >= stage1_min_pass
    ]

    # Pass 1b: direction j -> i for Pass 1a survivors
    n_1b = len(pass_1a_indices)
    stage2_pairs: List[Tuple[int, int]] = []
    next_tier_pairs: List[Tuple[int, int]] = []
    n_pass_1pct_s1 = 0

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
            rtol=1e-4, atol=1e-8, max_ode_steps=8000, max_lm_iters=16,
            early_stop_rms=0.45, fail_rms_thresh=keep_cut, max_fails=max_fails_s1,
        )

        for k_b, idx_p in enumerate(pass_1a_indices):
            pair = cand_pairs[idx_p]
            r_ij = s1a_rms[idx_p]
            r_ji = s1b_rms[k_b]
            if float(np.mean(r_ij <= keep_cut)) >= stage1_min_pass and float(np.mean(r_ji <= keep_cut)) >= stage1_min_pass:
                next_tier_pairs.append(pair)
            if float(np.mean(r_ij <= stage1_rms_cut)) >= stage1_min_pass and float(np.mean(r_ji <= stage1_rms_cut)) >= stage1_min_pass:
                stage2_pairs.append(pair)
            if float(np.mean(r_ij <= 1.0)) >= stage1_min_pass and float(np.mean(r_ji <= 1.0)) >= stage1_min_pass:
                n_pass_1pct_s1 += 1

    dt_s1 = time.perf_counter() - t_s1_start
    print(
        f"[Equiv:{design_set_name}] Stage 1 done ({dt_s1:.1f}s): "
        f"1a->{n_1b}, Stage2={len(stage2_pairs)} (<={stage1_rms_cut}; {n_pass_1pct_s1} <=1.0), next_tier={len(next_tier_pairs)}"
    )

    # ------------------------------------------------------------------
    # Stage 2: High-precision bidirectional LM (K2=32 x 4 starts)
    # Two-pass (i->j first, then j->i only for i->j passers) + exact binomial early exit
    # at max sensitivity cut (fail_rms_thresh = stage1_rms_cut, max_fails = 6 for 80% min thresh).
    # Zero approximation across all 9 (sigma_ref in {0.5,1,2}%, pr in {80,85,90}%) settings!
    # ------------------------------------------------------------------
    t_s2_start = time.perf_counter()
    n_s2 = len(stage2_pairs)
    pair_rms_i2j: Dict[Tuple[int, int], np.ndarray] = {}
    pair_rms_j2i: Dict[Tuple[int, int], np.ndarray] = {}
    max_fails_s2 = K2 - int(np.ceil(0.80 * K2))  # 32 - 26 = 6 (covers 80%, 85%, 90% exactly!)

    if n_s2 > 0:
        # Pass 2a: direction i -> j
        s2a_mids = np.empty(n_s2, dtype=np.int32)
        s2a_tgt = np.empty((n_s2, K2, n_res), dtype=np.float64)
        s2a_inits = np.empty((n_s2, K2, 4, P), dtype=np.float64)
        for idx_p, (i, j) in enumerate(stage2_pairs):
            s2a_mids[idx_p] = j
            s2a_tgt[idx_p] = s2_curves[i]
            s2a_inits[idx_p, :, 0:2] = get_nn_thetas(s2_scaled[i], j, topk=2)
            s2a_inits[idx_p, :, 2:4] = lib_th[j][rng.integers(0, N_lib, size=(K2, 2))]

        s2a_rms = sim.fit_lm_pairs_dir(
            s2a_mids, designs, t_eval, s2a_tgt, inv_sig_1d, s2a_inits,
            lb_all=lb_all, ub_all=ub_all,
            rtol=5e-5, atol=5e-9, max_ode_steps=10000, max_lm_iters=22,
            early_stop_rms=0.45, fail_rms_thresh=stage1_rms_cut, max_fails=max_fails_s2,
        )

        pass_2a_idx = [
            idx_p for idx_p in range(n_s2)
            if int(np.sum(s2a_rms[idx_p] <= stage1_rms_cut)) >= (K2 - max_fails_s2)
        ]

        # Pass 2b: direction j -> i ONLY for Pass 2a survivors
        n_2b = len(pass_2a_idx)
        if n_2b > 0:
            s2b_mids = np.empty(n_2b, dtype=np.int32)
            s2b_tgt = np.empty((n_2b, K2, n_res), dtype=np.float64)
            s2b_inits = np.empty((n_2b, K2, 4, P), dtype=np.float64)
            for kb, idx_p in enumerate(pass_2a_idx):
                i, j = stage2_pairs[idx_p]
                s2b_mids[kb] = i
                s2b_tgt[kb] = s2_curves[j]
                s2b_inits[kb, :, 0:2] = get_nn_thetas(s2_scaled[j], i, topk=2)
                s2b_inits[kb, :, 2:4] = lib_th[i][rng.integers(0, N_lib, size=(K2, 2))]

            s2b_rms = sim.fit_lm_pairs_dir(
                s2b_mids, designs, t_eval, s2b_tgt, inv_sig_1d, s2b_inits,
                lb_all=lb_all, ub_all=ub_all,
                rtol=5e-5, atol=5e-9, max_ode_steps=10000, max_lm_iters=22,
                early_stop_rms=0.45, fail_rms_thresh=stage1_rms_cut, max_fails=max_fails_s2,
            )

            for kb, idx_p in enumerate(pass_2a_idx):
                pair = stage2_pairs[idx_p]
                pair_rms_i2j[pair] = s2a_rms[idx_p]
                pair_rms_j2i[pair] = s2b_rms[kb]

    dt_s2 = time.perf_counter() - t_s2_start
    print(f"[Equiv:{design_set_name}] Stage 2 done ({dt_s2:.1f}s): 2a->{len(pair_rms_i2j)}/{n_s2} pairs checked bidirectionally")

    partitions = {}
    for sig_label, rms_cut in [("0.5pct", 0.5), ("1.0pct", 1.0), ("2.0pct", 2.0)]:
        for pr_label, pr_cut in [("80pct", 0.80), ("85pct", 0.85), ("90pct", 0.90)]:
            key = f"sig_{sig_label}_pr_{pr_label}"
            partitions[key] = partition_equivalence_graph(
                M, pair_rms_i2j, pair_rms_j2i, rms_thresh=rms_cut, pass_rate_thresh=pr_cut
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
