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

    lib.simulate_batch_ros2_var_t.argtypes = [
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
        c_dbl_p,        # t_eval_sets
        c_int_p,        # t_set_ids
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
    lib.simulate_batch_ros2_var_t.restype = None

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

    lib.fit_lm_batch_ex.argtypes = [
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
        ctypes.c_int,   # per_design_t_eval
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
    lib.fit_lm_batch_ex.restype = None

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

    lib.fit_lm_pairs_dir_ex.argtypes = [
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
        ctypes.c_int,   # per_design_t_eval
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
        c_dbl_p,        # best_theta_out (nullable)
    ]
    lib.fit_lm_pairs_dir_ex.restype = None

    lib.eval_pairs_forward_ex.argtypes = [
        ctypes.c_int,   # n_pairs
        ctypes.c_int,   # K
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
        ctypes.c_int,   # per_design_t_eval
        ctypes.c_int,   # n_res
        c_dbl_p,        # targets_flat
        c_dbl_p,        # inv_sigma_1d
        c_dbl_p,        # fitted_thetas
        ctypes.c_double,# rtol
        ctypes.c_double,# atol
        ctypes.c_int,   # max_ode_steps
        ctypes.c_int,   # n_threads
        c_dbl_p,        # rms_out
    ]
    lib.eval_pairs_forward_ex.restype = None

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
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
        max_steps: Optional[int] = None,
        n_threads: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Simulate B (mech, theta) samples across D designs -> (B, D, n_t, n_species).

        Supports both 1D t_eval (n_t,) shared across designs and 2D t_eval (D, n_t) per design.
        """
        g = self.grammar
        B = len(mech_ids)
        D = len(designs)
        te_arr = np.asarray(t_eval, dtype=np.float64)
        per_design_t = (te_arr.ndim == 2)
        if per_design_t:
            assert te_arr.shape[0] == D
            n_t = te_arr.shape[1]
        else:
            n_t = te_arr.shape[0]

        # Precompute y0 table only for active mechanism IDs in this batch
        mids_rep = np.repeat(mech_ids, D)
        lks_rep = np.repeat(log10_ks, D, axis=0)
        uniq_m = np.unique(mech_ids)
        if len(uniq_m) == 1:
            m0 = int(uniq_m[0])
            y0_single = np.stack([g.initial_state(m0, des) for des in designs], axis=0)  # (D, ns)
            y0s_rep = np.tile(y0_single, (B, 1))
        else:
            y0_table = np.zeros((g.n_mechs, D, g.n_species), dtype=np.float64)
            for m in uniq_m:
                mi = int(m)
                for d_idx, des in enumerate(designs):
                    y0_table[mi, d_idx] = g.initial_state(mi, des)
            y0s_rep = y0_table[mech_ids].reshape(B * D, g.n_species)

        if not per_design_t:
            y_out, status, f1, f2 = self.simulate_batch(
                mids_rep, lks_rep, y0s_rep, te_arr,
                compute_flux=compute_flux, rtol=rtol, atol=atol, max_steps=max_steps, n_threads=n_threads,
            )
        else:
            rmasks = np.ascontiguousarray(self._mech_rmask[mids_rep], dtype=np.int32)
            lks_c = np.ascontiguousarray(lks_rep, dtype=np.float64)
            y0_c = np.ascontiguousarray(y0s_rep, dtype=np.float64)
            te_c = np.ascontiguousarray(te_arr, dtype=np.float64)
            t_set_ids = np.ascontiguousarray(np.tile(np.arange(D, dtype=np.int32), B), dtype=np.int32)
            y_out = np.empty((B * D, n_t, g.n_species), dtype=np.float64)
            status = np.empty(B * D, dtype=np.int32)
            if compute_flux:
                f1 = np.empty((B * D, g.n_reactions), dtype=np.float64)
                f2 = np.empty((B * D, g.n_reactions), dtype=np.float64)
                f1_ptr, f2_ptr = _ptr_f64(f1), _ptr_f64(f2)
            else:
                f1 = f2 = None
                f1_ptr = ctypes.POINTER(ctypes.c_double)()
                f2_ptr = ctypes.POINTER(ctypes.c_double)()

            self.lib.simulate_batch_ros2_var_t(
                B * D, g.n_species, g.n_reactions, g.n_params,
                _ptr_i32(self._r1), _ptr_i32(self._r2), _ptr_i32(self._rps), _ptr_f64(self._S),
                _ptr_i32(rmasks), _ptr_f64(lks_c), _ptr_f64(y0_c),
                _ptr_f64(te_c), _ptr_i32(t_set_ids), n_t,
                self.rtol if rtol is None else rtol,
                self.atol if atol is None else atol,
                self.max_steps if max_steps is None else max_steps,
                self.n_threads if n_threads is None else n_threads,
                _ptr_f64(y_out), f1_ptr, f2_ptr, _ptr_i32(status),
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
        """Batch multi-start bounded LM fitting for a fixed design list (1D or 2D per-design t_eval)."""
        g = self.grammar
        n_tasks = len(task_mech_ids)
        n_starts = init_thetas.shape[1]
        n_designs = len(designs)
        te_arr = np.asarray(t_eval, dtype=np.float64)
        per_design_t = 1 if te_arr.ndim == 2 else 0
        n_t = te_arr.shape[1] if per_design_t else te_arr.shape[0]
        if per_design_t:
            assert te_arr.shape[0] == n_designs
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
        te_c = np.ascontiguousarray(te_arr, dtype=np.float64)
        tgt_c = np.ascontiguousarray(targets_flat, dtype=np.float64)
        invs_c = np.ascontiguousarray(inv_sigmas_flat, dtype=np.float64)
        init_c = np.ascontiguousarray(init_thetas, dtype=np.float64)
        lb_c = np.ascontiguousarray(lb_all, dtype=np.float64)
        ub_c = np.ascontiguousarray(ub_all, dtype=np.float64)

        best_rms = np.empty(n_tasks, dtype=np.float64)
        best_theta = np.empty((n_tasks, g.n_params), dtype=np.float64)

        self.lib.fit_lm_batch_ex(
            n_tasks, n_starts, g.n_species, g.n_reactions, g.n_params,
            _ptr_i32(self._r1), _ptr_i32(self._r2), _ptr_i32(self._rps), _ptr_f64(self._S),
            _ptr_i32(self._mech_rmask), _ptr_i32(self._mech_pmask),
            _ptr_i32(t_mids), n_designs, _ptr_f64(mech_y0),
            _ptr_i32(obs_flat), _ptr_i32(obs_off), _ptr_f64(te_c), n_t, per_design_t, n_res,
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
        return_theta: bool = False,
    ):
        """Pair-grouped directional LM fitting with exact binomial early exit.

        Returns best_rms (n_pairs, K), or (best_rms, best_theta) if return_theta=True.
        Supports both 1D t_eval (n_t,) and 2D per-design t_eval (n_designs, n_t).
        """
        g = self.grammar
        n_pairs, K, n_res = targets_flat.shape
        n_starts = init_thetas.shape[2]
        n_designs = len(designs)
        te_arr = np.asarray(t_eval, dtype=np.float64)
        per_design_t = 1 if te_arr.ndim == 2 else 0
        n_t = te_arr.shape[1] if per_design_t else te_arr.shape[0]
        if per_design_t:
            assert te_arr.shape[0] == n_designs

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
        te_c = np.ascontiguousarray(te_arr, dtype=np.float64)
        tgt_c = np.ascontiguousarray(targets_flat, dtype=np.float64)
        invs_c = np.ascontiguousarray(inv_sigma_1d, dtype=np.float64)
        init_c = np.ascontiguousarray(init_thetas, dtype=np.float64)
        lb_c = np.ascontiguousarray(lb_all, dtype=np.float64)
        ub_c = np.ascontiguousarray(ub_all, dtype=np.float64)
        best_rms = np.empty((n_pairs, K), dtype=np.float64)

        if return_theta:
            best_theta = np.empty((n_pairs, K, g.n_params), dtype=np.float64)
            th_ptr = _ptr_f64(best_theta)
        else:
            best_theta = None
            th_ptr = ctypes.POINTER(ctypes.c_double)()

        self.lib.fit_lm_pairs_dir_ex(
            n_pairs, K, n_starts, g.n_species, g.n_reactions, g.n_params,
            _ptr_i32(self._r1), _ptr_i32(self._r2), _ptr_i32(self._rps), _ptr_f64(self._S),
            _ptr_i32(self._mech_rmask), _ptr_i32(self._mech_pmask),
            _ptr_i32(pmids_c), n_designs, _ptr_f64(mech_y0),
            _ptr_i32(obs_flat), _ptr_i32(obs_off), _ptr_f64(te_c), n_t, per_design_t, n_res,
            _ptr_f64(tgt_c), _ptr_f64(invs_c), _ptr_f64(init_c),
            _ptr_f64(lb_c), _ptr_f64(ub_c),
            rtol, atol, max_ode_steps, max_lm_iters, early_stop_rms,
            fail_rms_thresh, max_fails,
            self.n_threads if n_threads is None else n_threads,
            _ptr_f64(best_rms), th_ptr,
        )
        if return_theta:
            return best_rms, best_theta
        return best_rms

    def eval_pairs_forward(
        self,
        pair_target_mech_ids: np.ndarray,  # (n_pairs,)
        designs: Sequence[DesignSpec],
        t_eval: np.ndarray,
        targets_flat: np.ndarray,          # (n_pairs, K, n_res)
        inv_sigma_1d: np.ndarray,          # (n_res,)
        fitted_thetas: np.ndarray,         # (n_pairs, K, n_params)
        rtol: float = 5e-5,
        atol: float = 5e-9,
        max_ode_steps: int = 8000,
        n_threads: Optional[int] = None,
    ) -> np.ndarray:
        """0-iteration forward RMS verification of inherited fitted parameters -> (n_pairs, K)."""
        g = self.grammar
        n_pairs, K, n_res = targets_flat.shape
        n_designs = len(designs)
        te_arr = np.asarray(t_eval, dtype=np.float64)
        per_design_t = 1 if te_arr.ndim == 2 else 0
        n_t = te_arr.shape[1] if per_design_t else te_arr.shape[0]
        if per_design_t:
            assert te_arr.shape[0] == n_designs

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

        pmids_c = np.ascontiguousarray(pair_target_mech_ids, dtype=np.int32)
        te_c = np.ascontiguousarray(te_arr, dtype=np.float64)
        tgt_c = np.ascontiguousarray(targets_flat, dtype=np.float64)
        invs_c = np.ascontiguousarray(inv_sigma_1d, dtype=np.float64)
        th_c = np.ascontiguousarray(fitted_thetas, dtype=np.float64)
        rms_out = np.empty((n_pairs, K), dtype=np.float64)

        self.lib.eval_pairs_forward_ex(
            n_pairs, K, g.n_species, g.n_reactions, g.n_params,
            _ptr_i32(self._r1), _ptr_i32(self._r2), _ptr_i32(self._rps), _ptr_f64(self._S),
            _ptr_i32(self._mech_rmask), _ptr_i32(self._mech_pmask),
            _ptr_i32(pmids_c), n_designs, _ptr_f64(mech_y0),
            _ptr_i32(obs_flat), _ptr_i32(obs_off), _ptr_f64(te_c), n_t, per_design_t, n_res,
            _ptr_f64(tgt_c), _ptr_f64(invs_c), _ptr_f64(th_c),
            rtol, atol, max_ode_steps,
            self.n_threads if n_threads is None else n_threads,
            _ptr_f64(rms_out),
        )
        return rms_out

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
    unfiltered: bool = False,
    c3_branch_min: float = 0.15,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Evaluate Section 6.2 & Appendix A lightweight physical/observability filters.

    Supports both single-core (151-mech) and 4-core (527-mech) grammars:
      - unfiltered=True keeps only solver validity + D0 conversion in [0.25, 0.98].
      - For C3 (Random-order, core_id == 2), requires both binding branches
        (catA+B->catAB and catB+A->catAB) >= c3_branch_min of total catalytic turnover in D0.
      - For C2 (Ordered B-first, core_id == 1), M3 probes peak catB2 in D3 instead of catA2 in D4.

    Returns:
        accepted: (B,) bool array
        diagnostics: dict of boolean arrays per criterion (True = passed)
    """
    B = len(log10_ks)
    if t_eval is None:
        t_eval = np.linspace(0.0, 1.0, 64)
    opts = grammar.mech_options[mech_id]
    core_id = int(grammar.mech_core_id[mech_id])
    is_multicore = (grammar.n_cores > 1)
    is_c2 = is_multicore and (core_id == 1)
    is_c3 = is_multicore and (core_id == 2)
    si = grammar.species_idx
    c0 = grammar.c0_base

    # Stage A: Always simulate D0 first (rejects ~65-75% before extra probe designs!)
    d0_design = grammar.catalog[grammar.catalog_by_name["D0"]]
    need_d0_flux = (not unfiltered) and bool(is_c3 or opts[5] > 0)
    mids_all = np.full(B, mech_id, dtype=np.int32)
    y_d0_all, st_d0, f1_d0, f2_d0 = sim.simulate_designs(
        mids_all, log10_ks, [d0_design], t_eval, compute_flux=need_d0_flux
    )

    diag: Dict[str, np.ndarray] = {}
    ok_solver = (st_d0[:, 0] == 0)
    diag["solver_ok"] = ok_solver.copy()

    # D0 conversion in [0.25, 0.98]
    y_d0 = y_d0_all[:, 0]  # (B, n_t, ns)
    conv = 1.0 - y_d0[:, -1, si["A"]] / d0_design.A0
    diag["conv_low"] = conv >= 0.25
    diag["conv_high"] = conv <= 0.98
    accepted = ok_solver & diag["conv_low"] & diag["conv_high"]

    if unfiltered:
        return accepted, diag

    # C3 Random-order dual-branch turnover check in D0
    if is_c3:
        r_ba = grammar.rxn_idx_by_role["bindB_fromA"]
        r_ab = grammar.rxn_idx_by_role["bindA_fromB"]
        f_a = f1_d0[:, 0, r_ba] + f2_d0[:, 0, r_ba]
        f_b = f1_d0[:, 0, r_ab] + f2_d0[:, 0, r_ab]
        tot_cat = np.maximum(f_a + f_b, 1e-12)
        c3_ok = ((f_a / tot_cat) >= c3_branch_min) & ((f_b / tot_cat) >= c3_branch_min)
        diag["C3_branches"] = c3_ok
        accepted &= c3_ok

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

    # M5: dimeric resting state (peak 2*cat2 >= 15% of c0 in early window [0, 0.5] in D0)
    if opts[4] > 0:
        half_mask = t_eval <= 0.50 + 1e-12
        peak_dimer = np.max(2.0 * y_d0[:, half_mask, si["cat2"]], axis=1) / c0
        m5_ok = peak_dimer >= 0.15
        diag["M5"] = m5_ok
        accepted &= m5_ok

    # M6: background reaction (flux_bg / total_P in [0.05, 0.70] in D0)
    if opts[5] > 0:
        r_bg = grammar.rxn_idx_by_role["bg"]
        flux_bg = f1_d0[:, 0, r_bg] + f2_d0[:, 0, r_bg]
        P_end = np.maximum(y_d0[:, -1, si["P"]], 1e-12)
        bg_frac = flux_bg / P_end
        m6_ok = (bg_frac >= 0.05) & (bg_frac <= 0.70)
        diag["M6"] = m6_ok
        accepted &= m6_ok

    # Stage B: Simulate extra probe designs (D2/D3/D4/D7) ONLY for D0 survivors (100% lossless!)
    extra_needed = []
    if opts[7] > 0:  # M8
        extra_needed.append("D2")
    if opts[2] > 0:  # M3: D3 for C2 (catB2), D4 for C1/C3/C4 (catA2)
        extra_needed.append("D3" if is_c2 else "D4")
    if opts[3] > 0 or opts[6] > 0:  # M4 or M7
        extra_needed.append("D7")

    if extra_needed and np.any(accepted):
        surv_idx = np.where(accepted)[0]
        extra_designs = [grammar.catalog[grammar.catalog_by_name[name]] for name in extra_needed]
        e_map = {name: idx for idx, name in enumerate(extra_needed)}
        need_extra_flux = bool(opts[6] > 0 or opts[7] > 0)

        y_ex, st_ex, f1_ex, f2_ex = sim.simulate_designs(
            np.full(len(surv_idx), mech_id, dtype=np.int32),
            log10_ks[surv_idx],
            extra_designs,
            t_eval,
            compute_flux=need_extra_flux,
        )
        ok_ex = np.all(st_ex == 0, axis=1)
        diag["solver_ok"][surv_idx] &= ok_ex
        surv_ok = ok_ex.copy()

        # M3: excess substrate inhibition
        if opts[2] > 0:
            half_mask = t_eval <= 0.50 + 1e-12
            if is_c2:
                y_sub = y_ex[:, e_map["D3"]]
                peak_sub = np.max(y_sub[:, half_mask, si["catB2"]], axis=1) / c0
            else:
                y_sub = y_ex[:, e_map["D4"]]
                peak_sub = np.max(y_sub[:, half_mask, si["catA2"]], axis=1) / c0
            m3_sub = peak_sub >= 0.10
            m3_full = np.zeros(B, dtype=bool)
            m3_full[surv_idx] = m3_sub
            diag["M3"] = m3_full
            surv_ok &= m3_sub

        # M4: product inhibition (end catP >= 10% of c0 in D0 or D7)
        if opts[3] > 0:
            y_d7 = y_ex[:, e_map["D7"]]
            catP_d0 = y_d0[surv_idx, -1, si["catP"]] / c0
            catP_d7 = y_d7[:, -1, si["catP"]] / c0
            m4_sub = (catP_d0 >= 0.10) | (catP_d7 >= 0.10)
            m4_full = np.zeros(B, dtype=bool)
            m4_full[surv_idx] = m4_sub
            diag["M4"] = m4_full
            surv_ok &= m4_sub

        # M7: reversible product release (second-half rev flux / fwd flux >= 10% in D7)
        if opts[6] > 0:
            r_fwd = grammar.rxn_idx_by_role["relP"]
            r_rev = grammar.rxn_idx_by_role["revP"]
            d7_i = e_map["D7"]
            fwd2 = np.maximum(f2_ex[:, d7_i, r_fwd], 1e-12)
            rev2 = f2_ex[:, d7_i, r_rev]
            m7_sub = (rev2 / fwd2) >= 0.10
            m7_full = np.zeros(B, dtype=bool)
            m7_full[surv_idx] = m7_sub
            diag["M7"] = m7_full
            surv_ok &= m7_sub

        # M8: bimetallic product release (bimet flux / total catalytic P flux >= 10% in D2)
        if opts[7] > 0:
            r_fwd = grammar.rxn_idx_by_role["relP"]
            r_bi = grammar.rxn_idx_by_role["bimet"]
            d2_i = e_map["D2"]
            f_uni = f1_ex[:, d2_i, r_fwd] + f2_ex[:, d2_i, r_fwd]
            f_bi = f1_ex[:, d2_i, r_bi] + f2_ex[:, d2_i, r_bi]
            m8_sub = (f_bi / np.maximum(f_uni + f_bi, 1e-12)) >= 0.10
            m8_full = np.zeros(B, dtype=bool)
            m8_full[surv_idx] = m8_sub
            diag["M8"] = m8_full
            surv_ok &= m8_sub

        accepted[surv_idx] &= surv_ok

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
    unfiltered: bool = False,
    c3_branch_min: float = 0.15,
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
        acc, diag = check_observability_batch(
            grammar, sim, mech_id, cand,
            t_eval=t_eval, unfiltered=unfiltered, c3_branch_min=c3_branch_min,
        )
        total_draws += draw_n
        solver_fails += int(np.sum(~diag["solver_ok"]))
        if np.any(acc):
            collected.append(cand[acc])
            n_got += int(np.sum(acc))
    if not collected:
        return np.zeros((0, grammar.n_params), dtype=np.float64), total_draws, solver_fails
    out = np.concatenate(collected, axis=0)[:n_target]
    return out, total_draws, solver_fails
