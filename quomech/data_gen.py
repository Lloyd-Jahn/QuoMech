"""Offline trajectory dataset generation (Section 7.1) and PyTorch online observation augmentation.

Supports cross-experiment variable observation windows t_end(xi) in [0.4, 4.0]:
  - All offline trajectories are integrated to t_max = 4.0 on an early-log-dense 80-point grid
    with exact knots at t = 0.0, 0.4, 1.0, 4.0 (50 points in [0, 1.0], 25 points in [0, 0.4]).
  - During training, each experiment dynamically draws t_end(xi) in [0.4, 4.0] (low catalyst loading
    biased toward longer windows, high loading biased toward shorter windows; plus exact t_end=1.0
    coverage for standard catalog protocols).
  - Design feature vector (6 + n_species): [A0, B0, P0, log10(cat0), a_pre, log10(t_end), obs_mask...].
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

from quomech.grammar import DesignSpec, ReactionGrammar
from quomech.simulator import ODESimulator, sample_accepted_thetas


def build_dense_time_grid(t_max: float = 4.0, n_t: int = 80) -> np.ndarray:
    """Early-log-dense 80-point grid on [0, 4.0] with exact knots at 0.0, 0.4, 1.0, 4.0."""
    assert n_t == 80 and abs(t_max - 4.0) < 1e-9
    t0 = np.array([0.0], dtype=np.float64)
    t1 = np.geomspace(0.005, 0.4, 24)        # 24 pts in [0.005, 0.4] (25 pts <= 0.4)
    t2 = np.linspace(0.4, 1.0, 26)[1:]       # 25 pts in (0.4, 1.0]  (50 pts <= 1.0)
    t3 = np.linspace(1.0, 4.0, 31)[1:]       # 30 pts in (1.0, 4.0]  (80 pts total)
    grid = np.concatenate([t0, t1, t2, t3])
    assert len(grid) == 80
    return grid


def get_observable_channel_indices(grammar: ReactionGrammar) -> Tuple[int, ...]:
    """Data-driven sorted union of all species indices observed in any catalog design."""
    obs = set()
    for d in grammar.catalog:
        obs.update(d.obs_species)
    return tuple(sorted(obs))


def build_design_feature_vec(
    grammar: ReactionGrammar,
    A0: float,
    B0: float,
    P0: float,
    cat_mult: float,
    a_pre: float,
    t_end: float,
    obs_ch: Tuple[int, ...],
    ch_mask: np.ndarray,
) -> np.ndarray:
    """Continuous design feature vector (6 + n_species):
    [A0, B0, P0, log10(cat0), a_pre, log10(t_end), obs_mask(n_species)].
    """
    obs_full = np.zeros(grammar.n_species, dtype=np.float32)
    for c_i, sp in enumerate(obs_ch):
        if ch_mask[c_i] > 0.5:
            obs_full[sp] = 1.0
    cat0 = cat_mult * grammar.c0_base
    head = np.array(
        [A0, B0, P0, np.log10(max(cat0, 1e-6)), float(a_pre), np.log10(max(t_end, 1e-4))],
        dtype=np.float32,
    )
    return np.concatenate([head, obs_full])


def generate_split(
    grammar: ReactionGrammar,
    sim: ODESimulator,
    out_dir: Path,
    split_name: str,
    n_per_mech: int,
    n_rand_designs: int = 4,
    seed: int = 42,
    unfiltered: bool = False,
    c3_branch_min: float = 0.15,
) -> Dict:
    """Generate accepted (m, theta) samples simulated to t_max=4.0 on 80-pt early-log-dense grid.

    Stores compact uncompressed .npy files (fast mmap / RAM load):
      - t_grid.npy:   (80,) float64
      - mech_ids.npy: (N,) uint16
      - thetas.npy:   (N, n_params) float32
      - rand_des.npy: (N, n_rand_designs, 5) float32  [A0, B0, P0, cat_mult, a_pre]
      - trajs.npy:    (N, n_cat + n_rand_designs, 80, n_obs_ch) float16
    """
    t0 = time.perf_counter()
    split_dir = out_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    M = grammar.n_mechs
    P = grammar.n_params
    n_cat = len(grammar.catalog)
    n_des_tot = n_cat + n_rand_designs
    obs_ch = get_observable_channel_indices(grammar)
    n_ch = len(obs_ch)
    N = M * n_per_mech
    t_grid = build_dense_time_grid(t_max=4.0, n_t=80)
    n_t = len(t_grid)

    mech_ids = np.repeat(np.arange(M, dtype=np.uint16), n_per_mech)
    thetas = np.empty((N, P), dtype=np.float32)
    rand_des = np.empty((N, n_rand_designs, 5), dtype=np.float32)
    trajs = np.empty((N, n_des_tot, n_t, n_ch), dtype=np.float16)

    total_draws_per_mech = np.zeros(M, dtype=np.int64)
    rej_solver_fails_per_mech = np.zeros(M, dtype=np.int64)
    sim_solver_fails_per_mech = np.zeros(M, dtype=np.int64)

    # Step 1: Rejection sample accepted thetas per mechanism (observability checked on standard [0,1] window)
    t_rej = time.perf_counter()
    for m in range(M):
        th, tot_d, sfail = sample_accepted_thetas(
            grammar,
            sim,
            m,
            n_per_mech,
            rng,
            batch_size=max(4096, min(16384, n_per_mech * 4)),
            max_draws=max(500000, n_per_mech * 150),
            unfiltered=unfiltered,
            c3_branch_min=c3_branch_min,
        )
        if len(th) < n_per_mech:
            raise RuntimeError(
                f"[{split_name}] Mech {m} ({grammar.mech_names[m]}) got {len(th)}/{n_per_mech} "
                f"(draws={tot_d}, acc={len(th)/max(tot_d,1):.4f})"
            )
        thetas[m * n_per_mech : (m + 1) * n_per_mech] = th.astype(np.float32)
        total_draws_per_mech[m] = tot_d
        rej_solver_fails_per_mech[m] = sfail
        if (m + 1) % 100 == 0 or (m + 1) == M:
            acc_so_far = ((m + 1) * n_per_mech) / max(1, int(np.sum(total_draws_per_mech[: m + 1])))
            print(
                f"  [DataGen:{split_name}] Rejection {m + 1}/{M} mechs ({time.perf_counter() - t_rej:.1f}s, "
                f"cum_acc={acc_so_far:.3f}, rej_fails={int(np.sum(rej_solver_fails_per_mech[: m + 1]))})",
                flush=True,
            )
    dt_rej = time.perf_counter() - t_rej
    acc_rates = n_per_mech / np.maximum(total_draws_per_mech, 1)
    print(
        f"[DataGen:{split_name}] Rejection sampled {N:,} (m, theta) in {dt_rej:.1f}s "
        f"(median_acc={float(np.median(acc_rates)):.3f}, min_acc={float(np.min(acc_rates)):.4f}, "
        f"mean_acc={N / np.sum(total_draws_per_mech):.3f}, rej_fails={int(np.sum(rej_solver_fails_per_mech))})",
        flush=True,
    )

    # Step 2: Sample 4 continuous random designs per sample from X
    A0_r = rng.uniform(0.25, 2.0, size=(N, n_rand_designs))
    large_B = rng.random(size=(N, n_rand_designs)) < 0.10
    B0_r = np.where(
        large_B,
        rng.uniform(5.0, 10.0, size=(N, n_rand_designs)),
        rng.uniform(0.25, 3.0, size=(N, n_rand_designs)),
    )
    zero_P = rng.random(size=(N, n_rand_designs)) < 0.50
    P0_r = np.where(zero_P, 0.0, rng.uniform(0.1, 1.0, size=(N, n_rand_designs)))
    cat_r = 10.0 ** rng.uniform(np.log10(0.25), np.log10(4.0), size=(N, n_rand_designs))
    apre_r = (rng.random(size=(N, n_rand_designs)) < 0.15).astype(np.float64)
    rand_des[:, :, 0] = A0_r
    rand_des[:, :, 1] = B0_r
    rand_des[:, :, 2] = P0_r
    rand_des[:, :, 3] = cat_r
    rand_des[:, :, 4] = apre_r

    # Step 3: Simulate all 14 designs to t_max = 4.0 across 64 CPU threads
    t_sim = time.perf_counter()
    sA, sB, sP = grammar.species_idx["A"], grammar.species_idx["B"], grammar.species_idx["P"]
    sCat, sPrecat = grammar.species_idx["cat"], grammar.species_idx["precat"]
    m1_active_by_mech = (grammar.mech_options[:, 0] > 0)

    cat_y0 = np.zeros((M, n_cat, grammar.n_species), dtype=np.float64)
    for m in range(M):
        for d_i, des in enumerate(grammar.catalog):
            cat_y0[m, d_i] = grammar.initial_state(m, des)

    chunk = 16384
    for start in range(0, N, chunk):
        end = min(N, start + chunk)
        B = end - start
        mids_c = mech_ids[start:end].astype(np.int32)
        th_c = thetas[start:end].astype(np.float64)

        y0_c = np.zeros((B, n_des_tot, grammar.n_species), dtype=np.float64)
        y0_c[:, :n_cat, :] = cat_y0[mids_c]

        rd = rand_des[start:end].astype(np.float64)
        y0_c[:, n_cat:, sA] = rd[:, :, 0]
        y0_c[:, n_cat:, sB] = rd[:, :, 1]
        y0_c[:, n_cat:, sP] = rd[:, :, 2]
        cat0 = rd[:, :, 3] * grammar.c0_base
        use_precat = m1_active_by_mech[mids_c][:, None] & (rd[:, :, 4] < 0.5)
        y0_c[:, n_cat:, sPrecat] = np.where(use_precat, cat0, 0.0)
        y0_c[:, n_cat:, sCat] = np.where(use_precat, 0.0, cat0)

        mids_rep = np.repeat(mids_c, n_des_tot)
        th_rep = np.repeat(th_c, n_des_tot, axis=0)
        y0_rep = y0_c.reshape(B * n_des_tot, grammar.n_species)

        y_out, st_out, _, _ = sim.simulate_batch(mids_rep, th_rep, y0_rep, t_grid, compute_flux=False)
        st_2d = st_out.reshape(B, n_des_tot)
        if np.any(st_2d != 0):
            fail_mids = mids_c[np.any(st_2d != 0, axis=1)]
            for fm in fail_mids:
                sim_solver_fails_per_mech[fm] += 1
            y_out = y_out.reshape(B, n_des_tot, n_t, grammar.n_species)
            y_out[st_2d != 0] = 0.0
        else:
            y_out = y_out.reshape(B, n_des_tot, n_t, grammar.n_species)

        trajs[start:end] = y_out[:, :, :, obs_ch].astype(np.float16)

    dt_sim = time.perf_counter() - t_sim
    n_tot_traj = N * n_des_tot
    print(
        f"[DataGen:{split_name}] Simulated {n_tot_traj:,} trajs (t_max=4.0, 80-pt) in {dt_sim:.1f}s "
        f"({n_tot_traj / dt_sim:,.0f} traj/s, sim_fails={int(np.sum(sim_solver_fails_per_mech))})",
        flush=True,
    )

    np.save(split_dir / "t_grid.npy", t_grid)
    np.save(split_dir / "mech_ids.npy", mech_ids)
    np.save(split_dir / "thetas.npy", thetas)
    np.save(split_dir / "rand_des.npy", rand_des)
    np.save(split_dir / "trajs.npy", trajs)

    meta = {
        "split": split_name,
        "unfiltered": unfiltered,
        "c3_branch_min": c3_branch_min,
        "n_mechs": M,
        "n_per_mech": n_per_mech,
        "n_samples": N,
        "n_cat": n_cat,
        "n_rand_designs": n_rand_designs,
        "t_max": 4.0,
        "n_t": n_t,
        "obs_channels": list(obs_ch),
        "obs_channel_names": [grammar.species[i].name for i in obs_ch],
        "elapsed_rej_s": dt_rej,
        "elapsed_sim_s": dt_sim,
        "elapsed_total_s": time.perf_counter() - t0,
        "sim_traj_per_sec": n_tot_traj / dt_sim,
        "median_acc_rate": float(np.median(acc_rates)),
        "min_acc_rate": float(np.min(acc_rates)),
        "mean_acc_rate": float(N / np.sum(total_draws_per_mech)),
        "total_rej_solver_fails": int(np.sum(rej_solver_fails_per_mech)),
        "total_sim_solver_fails": int(np.sum(sim_solver_fails_per_mech)),
        "rej_fails_by_core": {
            c.code: int(np.sum(rej_solver_fails_per_mech[grammar.mech_core_id == c.id]))
            for c in grammar.cores
        },
        "sim_fails_by_core": {
            c.code: int(np.sum(sim_solver_fails_per_mech[grammar.mech_core_id == c.id]))
            for c in grammar.cores
        },
        "size_gb": float(trajs.nbytes / (1024 ** 3)),
    }
    with open(split_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return meta


class OnlineKineticsDataset(Dataset):
    """Online observation augmentation with dynamic cross-experiment variable time windows t_end(xi) in [0.4, 4.0]."""

    def __init__(
        self,
        grammar: ReactionGrammar,
        split_dir: Path,
        max_exps: int = 8,
        max_len: int = 30,
        in_memory: bool = True,
    ) -> None:
        self.grammar = grammar
        self.max_exps = max_exps
        self.max_len = max_len
        self.obs_ch = get_observable_channel_indices(grammar)
        self.n_ch = len(self.obs_ch)
        self.ch_is_cat = np.array([grammar.species[s].is_catalyst for s in self.obs_ch], dtype=bool)
        self.ch_pos_P = self.obs_ch.index(grammar.species_idx["P"])
        self.ch_pos_A = self.obs_ch.index(grammar.species_idx["A"])

        mmap = None if in_memory else "r"
        self.t_grid = np.load(split_dir / "t_grid.npy").astype(np.float32)
        self.mech_ids = np.load(split_dir / "mech_ids.npy", mmap_mode=mmap)
        self.thetas = np.load(split_dir / "thetas.npy", mmap_mode=mmap)
        self.rand_des = np.load(split_dir / "rand_des.npy", mmap_mode=mmap)
        self.trajs = np.load(split_dir / "trajs.npy", mmap_mode=mmap)
        self.n_samples = len(self.mech_ids)
        self.n_cat = len(grammar.catalog)
        self.n_t = len(self.t_grid)

        self.cat_params = np.array(
            [[d.A0, d.B0, d.P0, d.cat_mult, float(d.a_pre)] for d in grammar.catalog],
            dtype=np.float32,
        )
        self.cat_obs_mask = np.zeros((self.n_cat, self.n_ch), dtype=np.float32)
        for d_i, des in enumerate(grammar.catalog):
            for c_i, sp in enumerate(self.obs_ch):
                if sp in des.obs_species:
                    self.cat_obs_mask[d_i, c_i] = 1.0

    def __len__(self) -> int:
        return self.n_samples

    def _sample_t_end(self, cat_mult: float, is_standard_catalog_mode: bool, rng) -> float:
        """Sample variable observation duration t_end(xi) in [0.4, 4.0].
        In standard catalog mode, 50% use exact t_end=1.0; otherwise load-adapted inverse scaling
        (low catalyst -> longer window, high catalyst -> shorter window) + 15% broad LogU.
        """
        if is_standard_catalog_mode and rng.random() < 0.50:
            return 1.0
        if rng.random() < 0.15:
            return float(10.0 ** rng.uniform(np.log10(0.4), np.log10(4.0)))
        log_t = -np.log10(max(cat_mult, 0.25)) + rng.normal(0.0, 0.25)
        return float(np.clip(10.0 ** log_t, 0.4, 4.0))

    def __getitem__(self, idx: int):
        rng = np.random
        E = int(rng.randint(1, self.max_exps + 1))

        # 50% catalog-only containing D0; 50% mixed with continuous random designs
        is_cat_only = (rng.random() < 0.50)
        if is_cat_only:
            if E == 1:
                chosen = np.array([0], dtype=np.int64)
            else:
                others = rng.choice(np.arange(1, self.n_cat), size=E - 1, replace=False)
                chosen = np.concatenate([[0], others])
                rng.shuffle(chosen)
        else:
            chosen = rng.choice(self.n_cat + self.rand_des.shape[1], size=E, replace=False)

        sigma_a_rel = float(10.0 ** rng.uniform(np.log10(0.003), np.log10(0.03)))
        sigma_m_rel = float(rng.uniform(0.0, 0.03))
        p_only_dataset = (rng.random() < 0.15)

        des_dim = 6 + self.grammar.n_species
        design_feats = np.zeros((self.max_exps, des_dim), dtype=np.float32)
        times = np.zeros((self.max_exps, self.max_len), dtype=np.float32)
        obs_vals = np.zeros((self.max_exps, self.max_len, self.n_ch), dtype=np.float32)
        obs_masks = np.zeros((self.max_exps, self.max_len, self.n_ch), dtype=np.float32)
        time_pad = np.ones((self.max_exps, self.max_len), dtype=bool)
        exp_pad = np.ones((self.max_exps,), dtype=bool)

        for e_i, d_slot in enumerate(chosen):
            exp_pad[e_i] = False
            if d_slot < self.n_cat:
                p5 = self.cat_params[d_slot]
                ch_mask = self.cat_obs_mask[d_slot].copy()
            else:
                p5 = self.rand_des[idx, d_slot - self.n_cat]
                ch_mask = np.zeros(self.n_ch, dtype=np.float32)
                if rng.random() < 0.15:
                    ch_mask[:] = 1.0
                else:
                    ch_mask[self.ch_pos_A] = 1.0
                    ch_mask[self.ch_pos_P] = 1.0

            if p_only_dataset and d_slot != 9:
                ch_mask[:] = 0.0
                ch_mask[self.ch_pos_P] = 1.0

            cat_mult = float(p5[3])
            t_end_target = self._sample_t_end(cat_mult, is_cat_only and (d_slot < self.n_cat), rng)
            # Find eligible grid indices in [0, t_end_target]
            max_idx = int(np.searchsorted(self.t_grid, t_end_target, side="right")) - 1
            max_idx = max(12, min(max_idx, self.n_t - 1))
            t_end_actual = float(self.t_grid[max_idx])

            design_feats[e_i] = build_design_feature_vec(
                self.grammar, float(p5[0]), float(p5[1]), float(p5[2]),
                cat_mult, float(p5[4]), t_end_actual, self.obs_ch, ch_mask,
            )

            L = int(rng.randint(8, min(self.max_len, max_idx + 1) + 1))
            mode = rng.random()
            if mode < 0.45:
                # Approximately uniform in physical time on [0, t_end_actual]
                t_targets = np.linspace(0.0, t_end_actual, L)
                t_idx = np.unique(np.clip(np.searchsorted(self.t_grid[: max_idx + 1], t_targets), 0, max_idx))
            elif mode < 0.75:
                # Log-spaced in physical time (denser near t=0)
                t_targets = np.concatenate([[0.0], np.geomspace(max(0.005, 0.01 * t_end_actual), t_end_actual, L - 1)])
                t_idx = np.unique(np.clip(np.searchsorted(self.t_grid[: max_idx + 1], t_targets), 0, max_idx))
            else:
                t_idx = np.sort(rng.choice(max_idx + 1, size=L, replace=False))
            L_eff = len(t_idx)

            y_clean = self.trajs[idx, d_slot, t_idx, :].astype(np.float32)
            A0 = float(p5[0])
            cat0 = cat_mult * self.grammar.c0_base
            noise_scale = np.where(self.ch_is_cat, cat0, A0).astype(np.float32)
            norm_scale = np.where(self.ch_is_cat, cat0, 1.0).astype(np.float32)

            std_tot = np.sqrt((sigma_a_rel * noise_scale[None, :]) ** 2 + (sigma_m_rel * np.abs(y_clean)) ** 2)
            y_noisy = y_clean + rng.normal(0.0, 1.0, size=y_clean.shape).astype(np.float32) * std_tot
            y_norm = y_noisy / norm_scale[None, :]

            p_miss = float(rng.uniform(0.0, 0.20))
            m_mat = np.broadcast_to(ch_mask[None, :], (L_eff, self.n_ch)).copy()
            if p_miss > 0.0:
                drop = rng.random(size=m_mat.shape) < p_miss
                m_mat[drop] = 0.0
                if m_mat.sum() < 2.0:
                    m_mat = np.broadcast_to(ch_mask[None, :], (L_eff, self.n_ch)).copy()

            times[e_i, :L_eff] = self.t_grid[t_idx]
            obs_vals[e_i, :L_eff] = y_norm * m_mat
            obs_masks[e_i, :L_eff] = m_mat
            time_pad[e_i, :L_eff] = False

        return (
            torch.from_numpy(design_feats),
            torch.from_numpy(times),
            torch.from_numpy(obs_vals),
            torch.from_numpy(obs_masks),
            torch.from_numpy(time_pad),
            torch.from_numpy(exp_pad),
            torch.tensor(int(self.mech_ids[idx]), dtype=torch.long),
            torch.from_numpy(self.thetas[idx].copy()),
        )
