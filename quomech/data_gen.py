"""Offline trajectory dataset generation (Section 7.1) and PyTorch online observation augmentation."""

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


def get_observable_channel_indices(grammar: ReactionGrammar) -> Tuple[int, ...]:
    """Data-driven sorted union of all species indices observed in any catalog design."""
    obs = set()
    for d in grammar.catalog:
        obs.update(d.obs_species)
    return tuple(sorted(obs))


def generate_split(
    grammar: ReactionGrammar,
    sim: ODESimulator,
    out_dir: Path,
    split_name: str,
    n_per_mech: int,
    n_rand_designs: int = 4,
    n_t: int = 64,
    seed: int = 42,
) -> Dict:
    """Generate accepted (m, theta) samples simulated across all catalog + random continuous designs.

    Stores compact uncompressed .npy files (fast mmap / RAM load):
      - mech_ids.npy: (N,) uint16
      - thetas.npy:   (N, n_params) float32
      - rand_des.npy: (N, n_rand_designs, 5) float32  [A0, B0, P0, cat_mult, a_pre]
      - trajs.npy:    (N, n_cat + n_rand_designs, n_t, n_obs_ch) float16
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
    t_eval = np.linspace(0.0, 1.0, n_t)

    mech_ids = np.repeat(np.arange(M, dtype=np.uint16), n_per_mech)
    thetas = np.empty((N, P), dtype=np.float32)
    rand_des = np.empty((N, n_rand_designs, 5), dtype=np.float32)
    trajs = np.empty((N, n_des_tot, n_t, n_ch), dtype=np.float16)

    # Audit tracking
    total_draws_per_mech = np.zeros(M, dtype=np.int64)
    rej_solver_fails_per_mech = np.zeros(M, dtype=np.int64)
    sim_solver_fails_per_mech = np.zeros(M, dtype=np.int64)

    # Step 1: Rejection sample accepted thetas per mechanism
    t_rej = time.perf_counter()
    for m in range(M):
        th, tot_d, sfail = sample_accepted_thetas(
            grammar, sim, m, n_per_mech, rng, batch_size=max(2048, min(8192, n_per_mech * 4))
        )
        if len(th) < n_per_mech:
            raise RuntimeError(f"[{split_name}] Mech {m} got {len(th)}/{n_per_mech}")
        thetas[m * n_per_mech : (m + 1) * n_per_mech] = th.astype(np.float32)
        total_draws_per_mech[m] = tot_d
        rej_solver_fails_per_mech[m] = sfail
    dt_rej = time.perf_counter() - t_rej
    print(f"[DataGen:{split_name}] Rejection sampled {N:,} (m, theta) in {dt_rej:.1f}s "
          f"(mean acc={N / np.sum(total_draws_per_mech):.3f}, rej_fails={int(np.sum(rej_solver_fails_per_mech))})")

    # Step 2: Sample 4 continuous random designs per sample
    A0_r = rng.uniform(0.25, 2.0, size=(N, n_rand_designs))
    large_B = rng.random(size=(N, n_rand_designs)) < 0.10
    B0_r = np.where(large_B, rng.uniform(5.0, 10.0, size=(N, n_rand_designs)), rng.uniform(0.25, 3.0, size=(N, n_rand_designs)))
    zero_P = rng.random(size=(N, n_rand_designs)) < 0.50
    P0_r = np.where(zero_P, 0.0, rng.uniform(0.1, 1.0, size=(N, n_rand_designs)))
    cat_r = 10.0 ** rng.uniform(np.log10(0.25), np.log10(4.0), size=(N, n_rand_designs))
    apre_r = (rng.random(size=(N, n_rand_designs)) < 0.15).astype(np.float64)
    rand_des[:, :, 0] = A0_r
    rand_des[:, :, 1] = B0_r
    rand_des[:, :, 2] = P0_r
    rand_des[:, :, 3] = cat_r
    rand_des[:, :, 4] = apre_r

    # Step 3: Simulate all 14 designs in memory-friendly chunks across 64 CPU threads
    t_sim = time.perf_counter()
    sA, sB, sP = grammar.species_idx["A"], grammar.species_idx["B"], grammar.species_idx["P"]
    sCat, sPrecat = grammar.species_idx["cat"], grammar.species_idx["precat"]
    m1_active_by_mech = (grammar.mech_options[:, 0] > 0)

    # Precompute catalog y0 table: (M, n_cat, ns)
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

        # Build y0 for all 14 designs of this chunk: (B, 14, ns)
        y0_c = np.zeros((B, n_des_tot, grammar.n_species), dtype=np.float64)
        y0_c[:, :n_cat, :] = cat_y0[mids_c]

        # Fill random designs y0 vectorized
        rd = rand_des[start:end].astype(np.float64)  # (B, n_rand, 5)
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

        y_out, st_out, _, _ = sim.simulate_batch(mids_rep, th_rep, y0_rep, t_eval, compute_flux=False)
        st_2d = st_out.reshape(B, n_des_tot)
        if np.any(st_2d != 0):
            fail_mids = mids_c[np.any(st_2d != 0, axis=1)]
            for fm in fail_mids:
                sim_solver_fails_per_mech[fm] += 1
            # Zero out any failed trajectory safely (rare/zero expected)
            y_out = y_out.reshape(B, n_des_tot, n_t, grammar.n_species)
            y_out[st_2d != 0] = 0.0
        else:
            y_out = y_out.reshape(B, n_des_tot, n_t, grammar.n_species)

        trajs[start:end] = y_out[:, :, :, obs_ch].astype(np.float16)

    dt_sim = time.perf_counter() - t_sim
    n_tot_traj = N * n_des_tot
    print(f"[DataGen:{split_name}] Simulated {n_tot_traj:,} trajectories in {dt_sim:.1f}s "
          f"({n_tot_traj / dt_sim:,.0f} traj/s, sim_fails={int(np.sum(sim_solver_fails_per_mech))})")

    # Save arrays
    np.save(split_dir / "mech_ids.npy", mech_ids)
    np.save(split_dir / "thetas.npy", thetas)
    np.save(split_dir / "rand_des.npy", rand_des)
    np.save(split_dir / "trajs.npy", trajs)

    meta = {
        "split": split_name,
        "n_mechs": M,
        "n_per_mech": n_per_mech,
        "n_samples": N,
        "n_cat": n_cat,
        "n_rand_designs": n_rand_designs,
        "n_t": n_t,
        "obs_channels": list(obs_ch),
        "obs_channel_names": [grammar.species[i].name for i in obs_ch],
        "elapsed_rej_s": dt_rej,
        "elapsed_sim_s": dt_sim,
        "elapsed_total_s": time.perf_counter() - t0,
        "sim_traj_per_sec": n_tot_traj / dt_sim,
        "total_rej_solver_fails": int(np.sum(rej_solver_fails_per_mech)),
        "total_sim_solver_fails": int(np.sum(sim_solver_fails_per_mech)),
        "size_gb": float(trajs.nbytes / (1024 ** 3)),
    }
    with open(split_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return meta


class OnlineKineticsDataset(Dataset):
    """In-memory / mmap PyTorch dataset with Section 6.4 & 7.1 online observation augmentation.

    For each item:
      - Draws subset size E in 1..8 (50% catalog-only containing D0; 50% mixed with continuous designs)
      - Adds additive noise sigma_a ~ LogU[0.3%, 3%] * scale_s and multiplicative noise sigma_m ~ U[0, 3%]
      - Subsamples L ~ U[8, 30] irregular timestamps (mixture of uniform and log-spaced indices)
      - Applies per-point missingness p ~ U[0, 0.2] and 15% P-only channel masking
      - Normalizes each observed channel by species scale (A0 for A/B/P? Actually 1.0 for macro,
        cat0 for catalyst species so A0 amplitude variation remains directly visible AND catAB is O(1)!)
    """

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
        self.mech_ids = np.load(split_dir / "mech_ids.npy", mmap_mode=mmap)
        self.thetas = np.load(split_dir / "thetas.npy", mmap_mode=mmap)
        self.rand_des = np.load(split_dir / "rand_des.npy", mmap_mode=mmap)
        self.trajs = np.load(split_dir / "trajs.npy", mmap_mode=mmap)
        self.n_samples = len(self.mech_ids)
        self.n_cat = len(grammar.catalog)
        self.n_t = self.trajs.shape[2]
        self.t_grid = np.linspace(0.0, 1.0, self.n_t, dtype=np.float32)

        # Precompute catalog static metadata
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

    def _build_design_feat(self, params5: np.ndarray, ch_mask: np.ndarray) -> np.ndarray:
        """Build (5 + n_species) design feature vector matching DesignSpec.to_feature_vec."""
        obs_full = np.zeros(self.grammar.n_species, dtype=np.float32)
        for c_i, sp in enumerate(self.obs_ch):
            if ch_mask[c_i] > 0.5:
                obs_full[sp] = 1.0
        cat0 = params5[3] * self.grammar.c0_base
        head = np.array(
            [params5[0], params5[1], params5[2], np.log10(max(cat0, 1e-6)), params5[4]],
            dtype=np.float32,
        )
        return np.concatenate([head, obs_full])

    def __getitem__(self, idx: int):
        rng = np.random  # fast worker-seeded numpy RNG (PyTorch seeds numpy per worker!)
        E = int(rng.randint(1, self.max_exps + 1))

        # 50% catalog-only containing D0; 50% mixed with continuous random designs
        if rng.random() < 0.50:
            if E == 1:
                chosen = np.array([0], dtype=np.int64)
            else:
                others = rng.choice(np.arange(1, self.n_cat), size=E - 1, replace=False)
                chosen = np.concatenate([[0], others])
                rng.shuffle(chosen)
        else:
            chosen = rng.choice(self.n_cat + self.rand_des.shape[1], size=E, replace=False)

        # Shared noise level across experiments in this dataset (realistic lab setting)
        sigma_a_rel = float(10.0 ** rng.uniform(np.log10(0.003), np.log10(0.03)))
        sigma_m_rel = float(rng.uniform(0.0, 0.03))
        p_only_dataset = (rng.random() < 0.15)

        des_dim = 5 + self.grammar.n_species
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
                    ch_mask[:] = 1.0  # full in-situ obs
                else:
                    ch_mask[self.ch_pos_A] = 1.0
                    ch_mask[self.ch_pos_P] = 1.0

            if p_only_dataset and d_slot != 9:  # D9 is full spectroscopy; otherwise P-only
                ch_mask[:] = 0.0
                ch_mask[self.ch_pos_P] = 1.0

            design_feats[e_i] = self._build_design_feat(p5, ch_mask)

            # Sample L in [8, 30] irregular timestamps (mix uniform + log-spaced)
            L = int(rng.randint(8, self.max_len + 1))
            mode = rng.random()
            if mode < 0.4:
                t_idx = np.unique(np.round(np.linspace(0, self.n_t - 1, L)).astype(np.int64))
            elif mode < 0.7:
                # Log-spaced (denser near t=0 to resolve activation/initial rate)
                raw = np.geomspace(1.0, float(self.n_t), L) - 1.0
                t_idx = np.unique(np.clip(np.round(raw).astype(np.int64), 0, self.n_t - 1))
            else:
                t_idx = np.sort(rng.choice(self.n_t, size=L, replace=False))
            L_eff = len(t_idx)

            y_clean = self.trajs[idx, d_slot, t_idx, :].astype(np.float32)  # (L_eff, C)
            A0 = float(p5[0])
            cat0 = float(p5[3]) * self.grammar.c0_base
            # Noise scales: sigma_a * A0 for macro, sigma_a * cat0 for catalyst
            noise_scale = np.where(self.ch_is_cat, cat0, A0).astype(np.float32)
            # Encoder normalization scale: 1.0 for macro (preserves A0 scale), cat0 for catalyst (O(1))
            norm_scale = np.where(self.ch_is_cat, cat0, 1.0).astype(np.float32)

            std_tot = np.sqrt((sigma_a_rel * noise_scale[None, :]) ** 2 + (sigma_m_rel * np.abs(y_clean)) ** 2)
            y_noisy = y_clean + rng.normal(0.0, 1.0, size=y_clean.shape).astype(np.float32) * std_tot
            y_norm = y_noisy / norm_scale[None, :]

            # Per-point random missingness p ~ U[0, 0.2]
            p_miss = float(rng.uniform(0.0, 0.20))
            m_mat = np.broadcast_to(ch_mask[None, :], (L_eff, self.n_ch)).copy()
            if p_miss > 0.0:
                drop = rng.random(size=m_mat.shape) < p_miss
                # Ensure at least 2 observations remain per active channel
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


def encode_observations_for_model(
    grammar: ReactionGrammar,
    designs: Sequence[DesignSpec],
    times_list: Sequence[np.ndarray],
    obs_y_list: Sequence[np.ndarray],  # each (L_e, n_obs_ch) raw concentrations
    obs_mask_list: Sequence[np.ndarray],  # each (L_e, n_obs_ch) binary mask
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack a single variable-length dataset D into model input tensors (batch=1)."""
    obs_ch = get_observable_channel_indices(grammar)
    n_ch = len(obs_ch)
    ch_is_cat = np.array([grammar.species[s].is_catalyst for s in obs_ch], dtype=bool)
    E = len(designs)
    L_max = max(len(t) for t in times_list)
    des_dim = 5 + grammar.n_species

    df = np.zeros((1, E, des_dim), dtype=np.float32)
    tm = np.zeros((1, E, L_max), dtype=np.float32)
    ov = np.zeros((1, E, L_max, n_ch), dtype=np.float32)
    om = np.zeros((1, E, L_max, n_ch), dtype=np.float32)
    tp = np.ones((1, E, L_max), dtype=bool)
    ep = np.zeros((1, E), dtype=bool)

    for e, (des, t_arr, y_arr, m_arr) in enumerate(zip(designs, times_list, obs_y_list, obs_mask_list)):
        L = len(t_arr)
        ch_active = (m_arr.sum(axis=0) > 0).astype(np.float32)
        obs_full = np.zeros(grammar.n_species, dtype=np.float32)
        for c_i, sp in enumerate(obs_ch):
            if ch_active[c_i] > 0.5:
                obs_full[sp] = 1.0
        cat0 = des.cat_mult * grammar.c0_base
        df[0, e] = np.concatenate([
            np.array([des.A0, des.B0, des.P0, np.log10(max(cat0, 1e-6)), float(des.a_pre)], dtype=np.float32),
            obs_full,
        ])
        norm_scale = np.where(ch_is_cat, cat0, 1.0).astype(np.float32)
        tm[0, e, :L] = t_arr.astype(np.float32)
        ov[0, e, :L] = (y_arr.astype(np.float32) / norm_scale[None, :]) * m_arr.astype(np.float32)
        om[0, e, :L] = m_arr.astype(np.float32)
        tp[0, e, :L] = False

    return (
        torch.from_numpy(df).to(device),
        torch.from_numpy(tm).to(device),
        torch.from_numpy(ov).to(device),
        torch.from_numpy(om).to(device),
        torch.from_numpy(tp).to(device),
        torch.from_numpy(ep).to(device),
    )
