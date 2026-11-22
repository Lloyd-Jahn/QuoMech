"""Python ctypes wrapper for the C++17/OpenMP float64 ROS2 ODE solver and LM fitter,
plus SciPy Radau reference integrator and observability-filtered rejection sampler.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
from scipy.integrate import solve_ivp

from quomech.grammar import DesignSpec, ReactionGrammar

_CSRC_DIR = Path(__file__).resolve().parent / "csrc"
_LIB_PATH = _CSRC_DIR / "libquomech_ode.so"
_LIB: Optional[ctypes.CDLL] = None


def _load_lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB
    if not _LIB_PATH.exists():
        subprocess.run(["bash", str(_CSRC_DIR / "build.sh")], check=True)
    lib = ctypes.CDLL(str(_LIB_PATH))

    c_int_p = ctypes.POINTER(ctypes.c_int)
    c_dbl_p = ctypes.POINTER(ctypes.c_double)

    lib.simulate_batch_ros2.argtypes = [
        ctypes.c_int,   # batch
        ctypes.c_int,   # n_species
        ctypes.c_int,   # n_reactions
        ctypes.c_int,   # n_params
        c_int_p,        # reactant1
        c_int_p,        # reactant2
        c_int_p,        # rxn_param_slot
        c_dbl_p,        # stoich_matrix
        c_int_p,        # rxn_masks
        c_dbl_p,        # log10_ks
        c_dbl_p,        # y0s
        c_dbl_p,        # t_eval
        ctypes.c_int,   # n_t
        ctypes.c_double,# rtol
        ctypes.c_double,# atol
        ctypes.c_int,   # max_steps
        ctypes.c_int,   # n_threads
        c_dbl_p,        # y_out
        c_dbl_p,        # flux1_out
        c_dbl_p,        # flux2_out
        c_int_p,        # status_out
    ]
    lib.simulate_batch_ros2.restype = None

    lib.fit_lm_batch.argtypes = [
        ctypes.c_int,   # n_tasks
        ctypes.c_int,   # n_starts
        ctypes.c_int,   # n_species
        ctypes.c_int,   # n_reactions
        ctypes.c_int,   # n_params
        c_int_p,        # reactant1
        c_int_p,        # reactant2
        c_int_p,        # rxn_param_slot
        c_dbl_p,        # stoich_matrix
        c_int_p,        # mech_rxn_masks
        c_int_p,        # mech_param_masks
        c_int_p,        # task_mech_ids
        ctypes.c_int,   # n_designs
        c_dbl_p,        # mech_y0_all
        c_int_p,        # obs_flat
        c_int_p,        # obs_offsets
        c_dbl_p,        # t_eval
        ctypes.c_int,   # n_t
        ctypes.c_int,   # n_res
        c_dbl_p,        # targets_flat
        c_dbl_p,        # inv_sigmas_flat
        c_dbl_p,        # init_thetas
        c_dbl_p,        # lb_all
        c_dbl_p,        # ub_all
        ctypes.c_double,# rtol
        ctypes.c_double,# atol
        ctypes.c_int,   # max_ode_steps
        ctypes.c_int,   # max_lm_iters
        ctypes.c_double,# early_stop_rms
        ctypes.c_int,   # n_threads
        c_dbl_p,        # best_rms_out
        c_dbl_p,        # best_theta_out
    ]
    lib.fit_lm_batch.restype = None

    lib.fit_lm_pairs_dir.argtypes = [
        ctypes.c_int,   # n_pairs
        ctypes.c_int,   # K
        ctypes.c_int,   # n_starts
        ctypes.c_int,   # n_species
        ctypes.c_int,   # n_reactions
        ctypes.c_int,   # n_params
        c_int_p,        # reactant1
        c_int_p,        # reactant2
        c_int_p,        # rxn_param_slot
        c_dbl_p,        # stoich_matrix
        c_int_p,        # mech_rxn_masks
        c_int_p,        # mech_param_masks
        c_int_p,        # pair_target_mech_ids
        ctypes.c_int,   # n_designs
        c_dbl_p,        # mech_y0_all
        c_int_p,        # obs_flat
        c_int_p,        # obs_offsets
        c_dbl_p,        # t_eval
        ctypes.c_int,   # n_t
        ctypes.c_int,   # n_res
        c_dbl_p,        # targets_flat
        c_dbl_p,        # inv_sigma_1d
        c_dbl_p,        # init_thetas
        c_dbl_p,        # lb_all
        c_dbl_p,        # ub_all
        ctypes.c_double,# rtol
        ctypes.c_double,# atol
        ctypes.c_int,   # max_ode_steps
        ctypes.c_int,   # max_lm_iters
        ctypes.c_double,# early_stop_rms
        ctypes.c_double,# fail_rms_thresh
        ctypes.c_int,   # max_fails
        ctypes.c_int,   # n_threads
        c_dbl_p,        # best_rms_out
    ]
    lib.fit_lm_pairs_dir.restype = None

    _LIB = lib
    return lib


def _ptr_i32(arr: np.ndarray):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int))


def _ptr_f64(arr: np.ndarray):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


class ODESimulator:
    """High-throughput CPU OpenMP float64 ROS2 simulator & LM fitter for a ReactionGrammar."""

    def __init__(
        self,
        grammar: ReactionGrammar,
        rtol: float = 1e-6,
        atol: float = 1e-10,
        max_steps: int = 20000,
        n_threads: int = 64,
    ) -> None:
        self.grammar = grammar
        self.rtol = rtol
        self.atol = atol
        self.max_steps = max_steps
        self.n_threads = n_threads
        self.lib = _load_lib()

        # Contiguous grammar arrays cached for ctypes
        self._r1 = np.ascontiguousarray(grammar.reactant1, dtype=np.int32)
        self._r2 = np.ascontiguousarray(grammar.reactant2, dtype=np.int32)
        self._rps = np.ascontiguousarray(grammar.rxn_param_slot, dtype=np.int32)
        self._S = np.ascontiguousarray(grammar.stoich_matrix, dtype=np.float64)
        self._mech_rmask = np.ascontiguousarray(grammar.mech_rxn_mask, dtype=np.int32)
        self._mech_pmask = np.ascontiguousarray(grammar.mech_param_mask, dtype=np.int32)

    def simulate_batch(
        self,
        mech_ids: np.ndarray,
        log10_ks: np.ndarray,
        y0s: np.ndarray,
        t_eval: np.ndarray,
        compute_flux: bool = False,
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
        max_steps: Optional[int] = None,
        n_threads: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Simulate a batch of trajectories.

        Args:
            mech_ids: (B,) int32 mechanism indices
            log10_ks: (B, n_params) float64 log10 rate constants
            y0s: (B, n_species) float64 initial states
            t_eval: (n_t,) float64 sorted evaluation times in [0, T]
            compute_flux: if True, also return integrated fluxes over [0, T/2] and [T/2, T]

        Returns:
            y_out: (B, n_t, n_species) float64
            status: (B,) int32 (0 = success)
            flux1: (B, n_reactions) float64 or None
            flux2: (B, n_reactions) float64 or None
        """
        g = self.grammar
        B = len(mech_ids)
        n_t = len(t_eval)
        rmasks = np.ascontiguousarray(self._mech_rmask[mech_ids], dtype=np.int32)
        lks = np.ascontiguousarray(log10_ks, dtype=np.float64)
        y0_c = np.ascontiguousarray(y0s, dtype=np.float64)
        te_c = np.ascontiguousarray(t_eval, dtype=np.float64)

        y_out = np.empty((B, n_t, g.n_species), dtype=np.float64)
        status = np.empty(B, dtype=np.int32)
        if compute_flux:
            flux1 = np.empty((B, g.n_reactions), dtype=np.float64)
            flux2 = np.empty((B, g.n_reactions), dtype=np.float64)
            f1_ptr = _ptr_f64(flux1)
            f2_ptr = _ptr_f64(flux2)
        else:
            flux1 = None
            flux2 = None
            f1_ptr = ctypes.POINTER(ctypes.c_double)()
            f2_ptr = ctypes.POINTER(ctypes.c_double)()

        self.lib.simulate_batch_ros2(
            B, g.n_species, g.n_reactions, g.n_params,
            _ptr_i32(self._r1), _ptr_i32(self._r2), _ptr_i32(self._rps), _ptr_f64(self._S),
            _ptr_i32(rmasks), _ptr_f64(lks), _ptr_f64(y0_c), _ptr_f64(te_c), n_t,
            self.rtol if rtol is None else rtol,
            self.atol if atol is None else atol,
            self.max_steps if max_steps is None else max_steps,
            self.n_threads if n_threads is None else n_threads,
            _ptr_f64(y_out), f1_ptr, f2_ptr, _ptr_i32(status),
        )
        return y_out, status, flux1, flux2

    def simulate_designs(
        self,
        mech_ids: np.ndarray,
        log10_ks: np.ndarray,
        designs: Sequence[DesignSpec],
        t_eval: np.ndarray,
        compute_flux: bool = False,
        n_threads: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Simulate B (mech, theta) samples across D designs -> (B, D, n_t, n_species)."""
        g = self.grammar
        B = len(mech_ids)
        D = len(designs)
        n_t = len(t_eval)

        # Precompute y0 table per (mech, design)
        mids_rep = np.repeat(mech_ids, D)
        lks_rep = np.repeat(log10_ks, D, axis=0)
        y0_table = np.zeros((g.n_mechs, D, g.n_species), dtype=np.float64)
        for m in range(g.n_mechs):
            for d_idx, des in enumerate(designs):
                y0_table[m, d_idx] = g.initial_state(m, des)
        y0s_rep = y0_table[mech_ids].reshape(B * D, g.n_species)

        y_out, status, f1, f2 = self.simulate_batch(
            mids_rep, lks_rep, y0s_rep, t_eval, compute_flux=compute_flux, n_threads=n_threads
        )
        y_out = y_out.reshape(B, D, n_t, g.n_species)
        status = status.reshape(B, D)
        if compute_flux:
            f1 = f1.reshape(B, D, g.n_reactions)
            f2 = f2.reshape(B, D, g.n_reactions)
        return y_out, status, f1, f2

    def fit_lm_batch(
        self,
        task_mech_ids: np.ndarray,
        designs: Sequence[DesignSpec],
        t_eval: np.ndarray,
        targets_flat: np.ndarray,
        inv_sigmas_flat: np.ndarray,
        init_thetas: np.ndarray,
        lb_all: Optional[np.ndarray] = None,
        ub_all: Optional[np.ndarray] = None,
        rtol: float = 1e-5,
        atol: float = 1e-9,
        max_ode_steps: int = 10000,
        max_lm_iters: int = 40,
        early_stop_rms: float = -1.0,
        n_threads: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Batch multi-start bounded LM fitting for a fixed design list."""
        g = self.grammar
        n_tasks = len(task_mech_ids)
        n_starts = init_thetas.shape[1]
        n_designs = len(designs)
        n_t = len(t_eval)
        n_res = targets_flat.shape[1]

        mech_y0 = np.zeros((g.n_mechs, n_designs, g.n_species), dtype=np.float64)
        for m in range(g.n_mechs):
            for d_idx, des in enumerate(designs):
                mech_y0[m, d_idx] = g.initial_state(m, des)

        obs_list = []
        obs_offsets = [0]
        for des in designs:
            obs_list.extend(des.obs_species)
            obs_offsets.append(len(obs_list))
        obs_flat = np.ascontiguousarray(obs_list, dtype=np.int32)
        obs_off = np.ascontiguousarray(obs_offsets, dtype=np.int32)
        assert obs_offsets[-1] * n_t == n_res

        if lb_all is None:
            lb_all = g.mech_prior_mean - 4.0 * g.mech_prior_std
        if ub_all is None:
            ub_all = g.mech_prior_mean + 4.0 * g.mech_prior_std

        t_mids = np.ascontiguousarray(task_mech_ids, dtype=np.int32)
        te_c = np.ascontiguousarray(t_eval, dtype=np.float64)
        tgt_c = np.ascontiguousarray(targets_flat, dtype=np.float64)
        invs_c = np.ascontiguousarray(inv_sigmas_flat, dtype=np.float64)
        init_c = np.ascontiguousarray(init_thetas, dtype=np.float64)
        lb_c = np.ascontiguousarray(lb_all, dtype=np.float64)
        ub_c = np.ascontiguousarray(ub_all, dtype=np.float64)

        best_rms = np.empty(n_tasks, dtype=np.float64)
        best_theta = np.empty((n_tasks, g.n_params), dtype=np.float64)

        self.lib.fit_lm_batch(
            n_tasks, n_starts, g.n_species, g.n_reactions, g.n_params,
            _ptr_i32(self._r1), _ptr_i32(self._r2), _ptr_i32(self._rps), _ptr_f64(self._S),
            _ptr_i32(self._mech_rmask), _ptr_i32(self._mech_pmask),
            _ptr_i32(t_mids), n_designs, _ptr_f64(mech_y0),
            _ptr_i32(obs_flat), _ptr_i32(obs_off), _ptr_f64(te_c), n_t, n_res,
            _ptr_f64(tgt_c), _ptr_f64(invs_c), _ptr_f64(init_c),
            _ptr_f64(lb_c), _ptr_f64(ub_c),
            rtol, atol, max_ode_steps, max_lm_iters, early_stop_rms,
            self.n_threads if n_threads is None else n_threads,
            _ptr_f64(best_rms), _ptr_f64(best_theta),
        )
        return best_rms, best_theta

    def fit_lm_pairs_dir(
        self,
        pair_target_mech_ids: np.ndarray,  # (n_pairs,)
        designs: Sequence[DesignSpec],
        t_eval: np.ndarray,
        targets_flat: np.ndarray,          # (n_pairs, K, n_res)
        inv_sigma_1d: np.ndarray,          # (n_res,)
        init_thetas: np.ndarray,           # (n_pairs, K, n_starts, n_params)
        lb_all: Optional[np.ndarray] = None,
        ub_all: Optional[np.ndarray] = None,
        rtol: float = 5e-5,
        atol: float = 5e-9,
        max_ode_steps: int = 8000,
        max_lm_iters: int = 22,
        early_stop_rms: float = 0.40,
        fail_rms_thresh: float = 2.0,
        max_fails: int = -1,
        n_threads: Optional[int] = None,
    ) -> np.ndarray:
        """Pair-grouped directional LM fitting with exact binomial early exit -> (n_pairs, K)."""
        g = self.grammar
        n_pairs, K, n_res = targets_flat.shape
        n_starts = init_thetas.shape[2]
        n_designs = len(designs)
        n_t = len(t_eval)

        mech_y0 = np.zeros((g.n_mechs, n_designs, g.n_species), dtype=np.float64)
        for m in range(g.n_mechs):
            for d_idx, des in enumerate(designs):
                mech_y0[m, d_idx] = g.initial_state(m, des)

        obs_list = []
        obs_offsets = [0]
        for des in designs:
            obs_list.extend(des.obs_species)
            obs_offsets.append(len(obs_list))
        obs_flat = np.ascontiguousarray(obs_list, dtype=np.int32)
        obs_off = np.ascontiguousarray(obs_offsets, dtype=np.int32)

        if lb_all is None:
            lb_all = g.mech_prior_mean - 3.5 * g.mech_prior_std
        if ub_all is None:
            ub_all = g.mech_prior_mean + 3.5 * g.mech_prior_std

        pmids_c = np.ascontiguousarray(pair_target_mech_ids, dtype=np.int32)
        te_c = np.ascontiguousarray(t_eval, dtype=np.float64)
        tgt_c = np.ascontiguousarray(targets_flat, dtype=np.float64)
        invs_c = np.ascontiguousarray(inv_sigma_1d, dtype=np.float64)
        init_c = np.ascontiguousarray(init_thetas, dtype=np.float64)
        lb_c = np.ascontiguousarray(lb_all, dtype=np.float64)
        ub_c = np.ascontiguousarray(ub_all, dtype=np.float64)
        best_rms = np.empty((n_pairs, K), dtype=np.float64)

        self.lib.fit_lm_pairs_dir(
            n_pairs, K, n_starts, g.n_species, g.n_reactions, g.n_params,
            _ptr_i32(self._r1), _ptr_i32(self._r2), _ptr_i32(self._rps), _ptr_f64(self._S),
            _ptr_i32(self._mech_rmask), _ptr_i32(self._mech_pmask),
            _ptr_i32(pmids_c), n_designs, _ptr_f64(mech_y0),
            _ptr_i32(obs_flat), _ptr_i32(obs_off), _ptr_f64(te_c), n_t, n_res,
            _ptr_f64(tgt_c), _ptr_f64(invs_c), _ptr_f64(init_c),
            _ptr_f64(lb_c), _ptr_f64(ub_c),
            rtol, atol, max_ode_steps, max_lm_iters, early_stop_rms,
            fail_rms_thresh, max_fails,
            self.n_threads if n_threads is None else n_threads,
            _ptr_f64(best_rms),
        )
        return best_rms

    def simulate_scipy_radau(
        self,
        mech_id: int,
        log10_k: np.ndarray,
        y0: np.ndarray,
        t_eval: np.ndarray,
        rtol: float = 1e-10,
        atol: float = 1e-14,
    ) -> Tuple[np.ndarray, bool]:
        """High-precision SciPy Radau reference solution with analytic Jacobian."""
        g = self.grammar
        rmask = g.mech_rxn_mask[mech_id]
        k_rxn = np.where(rmask == 1, 10.0 ** log10_k[g.rxn_param_slot], 0.0)
        r1, r2, S = g.reactant1, g.reactant2, g.stoich_matrix
        active_rxns = np.where(rmask == 1)[0]

        def rhs(_t, y):
            rates = np.zeros(g.n_reactions, dtype=np.float64)
            for r in active_rxns:
                a, b = r1[r], r2[r]
                rates[r] = k_rxn[r] * y[a] if b < 0 else k_rxn[r] * y[a] * y[b]
            return S @ rates

        def jac(_t, y):
            dv = np.zeros((g.n_reactions, g.n_species), dtype=np.float64)
            for r in active_rxns:
                a, b = r1[r], r2[r]
                if b < 0:
                    dv[r, a] = k_rxn[r]
                elif a == b:
                    dv[r, a] = 2.0 * k_rxn[r] * y[a]
                else:
                    dv[r, a] = k_rxn[r] * y[b]
                    dv[r, b] = k_rxn[r] * y[a]
            return S @ dv

        sol = solve_ivp(
            rhs, (0.0, float(t_eval[-1])), y0,
            method="Radau", t_eval=t_eval, jac=jac, rtol=rtol, atol=atol
        )
        if not sol.success or sol.y.shape[1] != len(t_eval):
            return np.zeros((len(t_eval), g.n_species)), False
        return sol.y.T, True


def check_observability_batch(
    grammar: ReactionGrammar,
    sim: ODESimulator,
    mech_id: int,
    log10_ks: np.ndarray,
    t_eval: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Evaluate Section 6.2 & Appendix A lightweight physical/observability filters.

    Returns:
        accepted: (B,) bool array
        diagnostics: dict of boolean arrays per criterion (True = passed)
    """
    B = len(log10_ks)
    if t_eval is None:
        t_eval = np.linspace(0.0, 1.0, 64)
    opts = grammar.mech_options[mech_id]
    si = grammar.species_idx
    c0 = grammar.c0_base

    # Determine which probe designs are needed for this mechanism
    # D0 always needed; D2 needed if M8; D4 needed if M3; D7 needed if M4 or M7
    needed = ["D0"]
    if opts[7] > 0:  # M8
        needed.append("D2")
    if opts[2] > 0:  # M3
        needed.append("D4")
    if opts[3] > 0 or opts[6] > 0:  # M4 or M7
        needed.append("D7")

    designs = [grammar.catalog[grammar.catalog_by_name[name]] for name in needed]
    d_map = {name: idx for idx, name in enumerate(needed)}
    need_flux = bool(opts[5] > 0 or opts[6] > 0 or opts[7] > 0)

    mids = np.full(B, mech_id, dtype=np.int32)
    y_all, st_all, f1_all, f2_all = sim.simulate_designs(
        mids, log10_ks, designs, t_eval, compute_flux=need_flux
    )

    diag: Dict[str, np.ndarray] = {}
    ok_solver = np.all(st_all == 0, axis=1)
    diag["solver_ok"] = ok_solver

    # D0 conversion in [0.25, 0.98]
    y_d0 = y_all[:, d_map["D0"]]  # (B, n_t, ns)
    conv = 1.0 - y_d0[:, -1, si["A"]] / grammar.catalog[grammar.catalog_by_name["D0"]].A0
    diag["conv_low"] = conv >= 0.25
    diag["conv_high"] = conv <= 0.98
    accepted = ok_solver & diag["conv_low"] & diag["conv_high"]

    # M1: activation (precat >= 20% in first 10% window, <= 50% at end in D0)
    if opts[0] > 0:
        idx_10 = np.searchsorted(t_eval, 0.10)
        precat_10 = y_d0[:, idx_10, si["precat"]] / c0
        precat_end = y_d0[:, -1, si["precat"]] / c0
        m1_ok = (precat_10 >= 0.20) & (precat_end <= 0.50)
        diag["M1"] = m1_ok
        accepted &= m1_ok

    # M2: deactivation (catI at end >= 10% of c0 in D0)
    if opts[1] > 0:
        catI_end = y_d0[:, -1, si["catI"]] / c0
        m2_ok = catI_end >= 0.10
        diag["M2"] = m2_ok
        accepted &= m2_ok

    # M3: substrate A inhibition (peak catA2 >= 10% of c0 in first half in D4)
    if opts[2] > 0:
        y_d4 = y_all[:, d_map["D4"]]
        half_mask = t_eval <= 0.50 + 1e-12
        peak_catA2 = np.max(y_d4[:, half_mask, si["catA2"]], axis=1) / c0
        m3_ok = peak_catA2 >= 0.10
        diag["M3"] = m3_ok
        accepted &= m3_ok

    # M4: product inhibition (end catP >= 10% of c0 in D0 or D7)
    if opts[3] > 0:
        y_d7 = y_all[:, d_map["D7"]]
        catP_d0 = y_d0[:, -1, si["catP"]] / c0
        catP_d7 = y_d7[:, -1, si["catP"]] / c0
        m4_ok = (catP_d0 >= 0.10) | (catP_d7 >= 0.10)
        diag["M4"] = m4_ok
        accepted &= m4_ok

    # M5: dimeric resting state (peak 2*cat2 >= 15% of c0 in early window [0, 0.5] in D0)
    if opts[4] > 0:
        half_mask = t_eval <= 0.50 + 1e-12
        peak_dimer = np.max(2.0 * y_d0[:, half_mask, si["cat2"]], axis=1) / c0
        m5_ok = peak_dimer >= 0.15
        diag["M5"] = m5_ok
        accepted &= m5_ok

    # M6: background reaction (flux_bg / total_P in [0.05, 0.70] in D0)
    if opts[5] > 0:
        r_bg = next(i for i, r in enumerate(grammar.reactions) if r.name == "R14_bg")
        d0_i = d_map["D0"]
        flux_bg = f1_all[:, d0_i, r_bg] + f2_all[:, d0_i, r_bg]
        P_end = np.maximum(y_d0[:, -1, si["P"]], 1e-12)
        bg_frac = flux_bg / P_end
        m6_ok = (bg_frac >= 0.05) & (bg_frac <= 0.70)
        diag["M6"] = m6_ok
        accepted &= m6_ok

    # M7: reversible product release (second-half rev flux / fwd flux >= 10% in D7)
    if opts[6] > 0:
        r_fwd = next(i for i, r in enumerate(grammar.reactions) if r.name == "R3_relP")
        r_rev = next(i for i, r in enumerate(grammar.reactions) if r.name == "R15_revP")
        d7_i = d_map["D7"]
        fwd2 = np.maximum(f2_all[:, d7_i, r_fwd], 1e-12)
        rev2 = f2_all[:, d7_i, r_rev]
        m7_ok = (rev2 / fwd2) >= 0.10
        diag["M7"] = m7_ok
        accepted &= m7_ok

    # M8: bimetallic product release (bimet flux / total catalytic P flux >= 10% in D2)
    if opts[7] > 0:
        r_fwd = next(i for i, r in enumerate(grammar.reactions) if r.name == "R3_relP")
        r_bi = next(i for i, r in enumerate(grammar.reactions) if r.name == "R16_bimet")
        d2_i = d_map["D2"]
        f_uni = f1_all[:, d2_i, r_fwd] + f2_all[:, d2_i, r_fwd]
        f_bi = f1_all[:, d2_i, r_bi] + f2_all[:, d2_i, r_bi]
        m8_ok = (f_bi / np.maximum(f_uni + f_bi, 1e-12)) >= 0.10
        diag["M8"] = m8_ok
        accepted &= m8_ok

    return accepted, diag


def sample_accepted_thetas(
    grammar: ReactionGrammar,
    sim: ODESimulator,
    mech_id: int,
    n_target: int,
    rng: np.random.Generator,
    batch_size: int = 2048,
    max_draws: int = 500000,
    t_eval: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, int, int]:
    """Rejection-sample n_target accepted log10_k vectors for mech_id.

    Returns:
        thetas: (n_collected, n_params) float64
        total_draws: int
        solver_fails: int
    """
    collected: List[np.ndarray] = []
    n_got = 0
    total_draws = 0
    solver_fails = 0
    while n_got < n_target and total_draws < max_draws:
        draw_n = min(batch_size, max_draws - total_draws)
        cand = grammar.sample_prior_theta(mech_id, draw_n, rng)
        acc, diag = check_observability_batch(grammar, sim, mech_id, cand, t_eval=t_eval)
        total_draws += draw_n
        solver_fails += int(np.sum(~diag["solver_ok"]))
        if np.any(acc):
            collected.append(cand[acc])
            n_got += int(np.sum(acc))
    if not collected:
        return np.zeros((0, grammar.n_params), dtype=np.float64), total_draws, solver_fails
    out = np.concatenate(collected, axis=0)[:n_target]
    return out, total_draws, solver_fails
