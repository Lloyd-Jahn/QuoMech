"""Data-driven chemical reaction grammar, mechanism library, priors, and experiment catalogs.

All chemical species, elementary reactions, stoichiometric matrices, sparse mechanism masks,
shared reaction-type priors, observability filters, and experimental design catalogs are
defined as structured data tables so the ODE solver, LM fitter, neural network, and BOED
loop remain 100% network-agnostic and seamlessly extensible to larger mechanism libraries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations, product
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np


@dataclass(frozen=True)
class SpeciesSpec:
    name: str
    atoms: Tuple[int, int, int]  # (A, B, cat)
    is_catalyst: bool = False
    cat_stoich: int = 0          # catalyst equivalents per molecule (1 for mono, 2 for cat2)


@dataclass(frozen=True)
class ReactionSpec:
    name: str
    reactants: Tuple[int, ...]         # 1 or 2 species indices
    stoich: Tuple[Tuple[int, int], ...]  # (species_idx, net_delta)
    param_slot: int                    # index into shared parameter vector
    module_id: int                     # -1 for core cycle, 0..n_modules-1 for optional
    module_option: int                 # 0 for core; >=1 for active module option
    rxn_type: str                      # reaction-type key for shared Gaussian prior


@dataclass(frozen=True)
class ModuleSpec:
    id: int
    code: str
    name: str
    options: Tuple[str, ...]           # option 0 is always "off"


@dataclass(frozen=True)
class DesignSpec:
    name: str
    A0: float
    B0: float
    P0: float
    cat_mult: float                    # multiplier of c0 (=0.05)
    a_pre: int                         # 0 = precat if M1 active; 1 = pre-activated (all in cat)
    obs_species: Tuple[int, ...]       # indices of observed species
    cost: float = 1.0
    tiers: Tuple[str, ...] = ("no_P", "basic", "full")

    def to_feature_vec(self, n_species: int, c0_base: float = 0.05) -> np.ndarray:
        """Continuous design feature vector: [A0, B0, P0, log10(cat0), a_pre, obs_mask...]."""
        obs_mask = np.zeros(n_species, dtype=np.float32)
        for s in self.obs_species:
            obs_mask[s] = 1.0
        cat0 = self.cat_mult * c0_base
        head = np.array(
            [self.A0, self.B0, self.P0, np.log10(cat0), float(self.a_pre)],
            dtype=np.float32,
        )
        return np.concatenate([head, obs_mask])


@dataclass
class ReactionGrammar:
    """Complete data-driven specification of a catalytic reaction mechanism space."""
    species: List[SpeciesSpec]
    reactions: List[ReactionSpec]
    modules: List[ModuleSpec]
    param_names: List[str]
    prior_means_by_type: Dict[str, float]
    prior_std: float = 0.7
    c0_base: float = 0.05
    max_active_modules: int = 3
    catalog: List[DesignSpec] = field(default_factory=list)
    # Populated in __post_init__
    species_idx: Dict[str, int] = field(init=False)
    n_species: int = field(init=False)
    n_reactions: int = field(init=False)
    n_params: int = field(init=False)
    n_modules: int = field(init=False)
    stoich_matrix: np.ndarray = field(init=False)       # (n_species, n_reactions) float64
    atom_matrix: np.ndarray = field(init=False)         # (3, n_species) float64
    reactant1: np.ndarray = field(init=False)           # (n_reactions,) int32
    reactant2: np.ndarray = field(init=False)           # (n_reactions,) int32 (-1 if unimol)
    rxn_param_slot: np.ndarray = field(init=False)      # (n_reactions,) int32
    # Sparse mechanism library
    mech_options: np.ndarray = field(init=False)        # (n_mechs, n_modules) int32
    mech_rxn_mask: np.ndarray = field(init=False)       # (n_mechs, n_reactions) int32
    mech_param_mask: np.ndarray = field(init=False)     # (n_mechs, n_params) int32
    mech_prior_mean: np.ndarray = field(init=False)     # (n_mechs, n_params) float64
    mech_prior_std: np.ndarray = field(init=False)      # (n_mechs, n_params) float64
    mech_names: List[str] = field(init=False)
    n_mechs: int = field(init=False)
    catalog_by_name: Dict[str, int] = field(init=False)
    tiers: Dict[str, List[int]] = field(init=False)

    def __post_init__(self) -> None:
        self.species_idx = {s.name: i for i, s in enumerate(self.species)}
        self.n_species = len(self.species)
        self.n_reactions = len(self.reactions)
        self.n_params = len(self.param_names)
        self.n_modules = len(self.modules)

        # Build stoichiometry and reactant arrays
        self.stoich_matrix = np.zeros((self.n_species, self.n_reactions), dtype=np.float64)
        self.reactant1 = np.full(self.n_reactions, -1, dtype=np.int32)
        self.reactant2 = np.full(self.n_reactions, -1, dtype=np.int32)
        self.rxn_param_slot = np.zeros(self.n_reactions, dtype=np.int32)

        for r_idx, rxn in enumerate(self.reactions):
            assert 1 <= len(rxn.reactants) <= 2, f"Only 1st/2nd order supported: {rxn.name}"
            self.reactant1[r_idx] = rxn.reactants[0]
            if len(rxn.reactants) == 2:
                self.reactant2[r_idx] = rxn.reactants[1]
            for s_idx, delta in rxn.stoich:
                self.stoich_matrix[s_idx, r_idx] += delta
            self.rxn_param_slot[r_idx] = rxn.param_slot

        # Verify strict atomic conservation for every reaction
        self.atom_matrix = np.array([s.atoms for s in self.species], dtype=np.float64).T
        atom_err = np.abs(self.atom_matrix @ self.stoich_matrix)
        if np.max(atom_err) > 1e-12:
            bad = np.where(atom_err.max(axis=0) > 1e-12)[0]
            raise ValueError(f"Atom conservation violated in reactions: {[self.reactions[i].name for i in bad]}")

        # Enumerate sparse mechanisms with <= max_active_modules active modules
        mech_opt_list: List[Tuple[int, ...]] = []
        for k in range(self.max_active_modules + 1):
            for mod_subset in combinations(range(self.n_modules), k):
                opt_ranges = [range(1, len(self.modules[m].options)) for m in mod_subset]
                for chosen_opts in product(*opt_ranges):
                    vec = [0] * self.n_modules
                    for m, opt in zip(mod_subset, chosen_opts):
                        vec[m] = opt
                    mech_opt_list.append(tuple(vec))

        self.n_mechs = len(mech_opt_list)
        self.mech_options = np.array(mech_opt_list, dtype=np.int32)
        self.mech_rxn_mask = np.zeros((self.n_mechs, self.n_reactions), dtype=np.int32)
        self.mech_param_mask = np.zeros((self.n_mechs, self.n_params), dtype=np.int32)
        self.mech_prior_mean = np.zeros((self.n_mechs, self.n_params), dtype=np.float64)
        self.mech_prior_std = np.full((self.n_mechs, self.n_params), self.prior_std, dtype=np.float64)
        self.mech_names = []

        for m_idx, opt_vec in enumerate(mech_opt_list):
            parts = []
            for mod_id, opt in enumerate(opt_vec):
                if opt > 0:
                    mod = self.modules[mod_id]
                    parts.append(f"{mod.code}:{mod.options[opt]}" if len(mod.options) > 2 else mod.code)
            self.mech_names.append("Core" if not parts else "+".join(parts))

            for r_idx, rxn in enumerate(self.reactions):
                active = (rxn.module_id < 0) or (opt_vec[rxn.module_id] == rxn.module_option)
                if active:
                    self.mech_rxn_mask[m_idx, r_idx] = 1
                    p_slot = rxn.param_slot
                    self.mech_param_mask[m_idx, p_slot] = 1
                    self.mech_prior_mean[m_idx, p_slot] = self.prior_means_by_type[rxn.rxn_type]

        # Catalog indices and tiers
        self.catalog_by_name = {d.name: i for i, d in enumerate(self.catalog)}
        self.tiers = {
            "D0": [0],
            "no_P": [i for i, d in enumerate(self.catalog) if "no_P" in d.tiers],
            "basic": [i for i, d in enumerate(self.catalog) if "basic" in d.tiers],
            "full": [i for i, d in enumerate(self.catalog) if "full" in d.tiers],
        }

    def initial_state(self, mech_id: int, design: DesignSpec) -> np.ndarray:
        """Build initial concentration vector y0 for (mech_id, design)."""
        y0 = np.zeros(self.n_species, dtype=np.float64)
        y0[self.species_idx["A"]] = design.A0
        y0[self.species_idx["B"]] = design.B0
        y0[self.species_idx["P"]] = design.P0
        cat0 = design.cat_mult * self.c0_base
        # M1 (module 0) active and not pre-activated -> catalyst starts as precat
        m1_active = (self.mech_options[mech_id, 0] > 0)
        if m1_active and design.a_pre == 0:
            y0[self.species_idx["precat"]] = cat0
        else:
            y0[self.species_idx["cat"]] = cat0
        return y0

    def sample_prior_theta(
        self, mech_id: int, n_samples: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Sample log10 rate constants (n_samples, n_params). Inactive slots are N(0, 1)."""
        z = rng.standard_normal((n_samples, self.n_params))
        means = self.mech_prior_mean[mech_id]
        stds = self.mech_prior_std[mech_id]
        mask = self.mech_param_mask[mech_id].astype(np.float64)
        return mask * (means + stds * z) + (1.0 - mask) * z

    def sample_continuous_design(
        self, rng: np.random.Generator, obs_mode: str = "AP"
    ) -> DesignSpec:
        """Sample a continuous experimental design from X (Section 6.3)."""
        A0 = float(rng.uniform(0.25, 2.0))
        if rng.random() < 0.10:
            B0 = float(rng.uniform(5.0, 10.0))  # quasi-single-substrate large excess
        else:
            B0 = float(rng.uniform(0.25, 3.0))
        cat_mult = float(10.0 ** rng.uniform(np.log10(0.25), np.log10(4.0)))
        if rng.random() < 0.50:
            P0 = 0.0
        else:
            P0 = float(rng.uniform(0.1, 1.0))
        a_pre = int(rng.random() < 0.15)
        sA, sB, sP, sCatAB = (
            self.species_idx["A"],
            self.species_idx["B"],
            self.species_idx["P"],
            self.species_idx["catAB"],
        )
        if obs_mode == "full":
            obs = (sA, sB, sP, sCatAB)
        elif obs_mode == "P":
            obs = (sP,)
        else:
            obs = (sA, sP)
        return DesignSpec(
            name="rand",
            A0=A0,
            B0=B0,
            P0=P0,
            cat_mult=cat_mult,
            a_pre=a_pre,
            obs_species=obs,
            cost=1.0,
            tiers=(),
        )

    def module_hamming_matrix(self) -> np.ndarray:
        """Pairwise module Hamming distance matrix (n_mechs, n_mechs).
        Two mechanisms differ at module j iff their option values at module j differ.
        """
        opts = self.mech_options  # (M, n_mod)
        return np.sum(opts[:, None, :] != opts[None, :, :], axis=-1)

    def active_param_indices(self, mech_id: int) -> np.ndarray:
        return np.where(self.mech_param_mask[mech_id] == 1)[0]

    def get_mech_id_by_name(self, name: str) -> int:
        return self.mech_names.index(name)


def build_default_grammar(
    prior_means_override: Optional[Dict[str, float]] = None,
    prior_std: float = 0.7,
) -> ReactionGrammar:
    """Build the 11-species, 17-reaction, 8-module, 151-mechanism catalytic grammar (Section 6)."""
    species = [
        SpeciesSpec("A", (1, 0, 0)),
        SpeciesSpec("B", (0, 1, 0)),
        SpeciesSpec("P", (1, 1, 0)),
        SpeciesSpec("cat", (0, 0, 1), is_catalyst=True, cat_stoich=1),
        SpeciesSpec("catA", (1, 0, 1), is_catalyst=True, cat_stoich=1),
        SpeciesSpec("catAB", (1, 1, 1), is_catalyst=True, cat_stoich=1),
        SpeciesSpec("precat", (0, 0, 1), is_catalyst=True, cat_stoich=1),
        SpeciesSpec("catI", (0, 0, 1), is_catalyst=True, cat_stoich=1),
        SpeciesSpec("catA2", (2, 0, 1), is_catalyst=True, cat_stoich=1),
        SpeciesSpec("catP", (1, 1, 1), is_catalyst=True, cat_stoich=1),
        SpeciesSpec("cat2", (0, 0, 2), is_catalyst=True, cat_stoich=2),
    ]
    si = {s.name: i for i, s in enumerate(species)}

    modules = [
        ModuleSpec(0, "M1", "Activation", ("off", "on")),
        ModuleSpec(1, "M2", "Deactivation", ("off", "free1", "free2", "complex")),
        ModuleSpec(2, "M3", "SubA_Inhib", ("off", "on")),
        ModuleSpec(3, "M4", "Prod_Inhib", ("off", "on")),
        ModuleSpec(4, "M5", "Dimer_Rest", ("off", "on")),
        ModuleSpec(5, "M6", "Background", ("off", "on")),
        ModuleSpec(6, "M7", "Rev_Release", ("off", "on")),
        ModuleSpec(7, "M8", "Bimetallic", ("off", "on")),
    ]

    param_names = [
        "k_1", "k_-1", "k_2", "k_3",       # 0..3: core
        "k_a",                              # 4: M1
        "k_d",                              # 5: M2 (shared slot across 3 mutually exclusive opts)
        "k_3a", "k_-3a",                    # 6..7: M3
        "k_4", "k_-4",                      # 8..9: M4
        "k_5", "k_-5",                      # 10..11: M5
        "k_bg",                             # 12: M6
        "k_-3",                             # 13: M7
        "k_bi",                             # 14: M8
    ]

    reactions = [
        # Core cycle (4 reactions)
        ReactionSpec("R0_bindA", (si["cat"], si["A"]), ((si["cat"], -1), (si["A"], -1), (si["catA"], 1)), 0, -1, 0, "bimol"),
        ReactionSpec("R1_dissA", (si["catA"],), ((si["catA"], -1), (si["cat"], 1), (si["A"], 1)), 1, -1, 0, "unimol"),
        ReactionSpec("R2_bindB", (si["catA"], si["B"]), ((si["catA"], -1), (si["B"], -1), (si["catAB"], 1)), 2, -1, 0, "bimol"),
        ReactionSpec("R3_relP",  (si["catAB"],), ((si["catAB"], -1), (si["cat"], 1), (si["P"], 1)), 3, -1, 0, "unimol"),
        # M1: Activation
        ReactionSpec("R4_act",   (si["precat"],), ((si["precat"], -1), (si["cat"], 1)), 4, 0, 1, "act"),
        # M2: Deactivation (3 mutually exclusive options sharing param slot 5)
        ReactionSpec("R5_deact1", (si["cat"],), ((si["cat"], -1), (si["catI"], 1)), 5, 1, 1, "deact1"),
        ReactionSpec("R6_deact2", (si["cat"], si["cat"]), ((si["cat"], -2), (si["catI"], 2)), 5, 1, 2, "deact2"),
        ReactionSpec("R7_deactC", (si["catA"],), ((si["catA"], -1), (si["catI"], 1), (si["A"], 1)), 5, 1, 3, "deact1"),
        # M3: Substrate A inhibition
        ReactionSpec("R8_bindA2", (si["catA"], si["A"]), ((si["catA"], -1), (si["A"], -1), (si["catA2"], 1)), 6, 2, 1, "bimol"),
        ReactionSpec("R9_dissA2", (si["catA2"],), ((si["catA2"], -1), (si["catA"], 1), (si["A"], 1)), 7, 2, 1, "unimol"),
        # M4: Product inhibition
        ReactionSpec("R10_bindP", (si["cat"], si["P"]), ((si["cat"], -1), (si["P"], -1), (si["catP"], 1)), 8, 3, 1, "bimol"),
        ReactionSpec("R11_dissP", (si["catP"],), ((si["catP"], -1), (si["cat"], 1), (si["P"], 1)), 9, 3, 1, "unimol"),
        # M5: Dimeric resting state (cat + cat -> bimol_cat, scaled by -log10(c0) = +1.3)
        ReactionSpec("R12_dimer", (si["cat"], si["cat"]), ((si["cat"], -2), (si["cat2"], 1)), 10, 4, 1, "bimol_cat"),
        ReactionSpec("R13_monom", (si["cat2"],), ((si["cat2"], -1), (si["cat"], 2)), 11, 4, 1, "unimol"),
        # M6: Background non-catalytic
        ReactionSpec("R14_bg",    (si["A"], si["B"]), ((si["A"], -1), (si["B"], -1), (si["P"], 1)), 12, 5, 1, "bg"),
        # M7: Reversible product release
        ReactionSpec("R15_revP",  (si["cat"], si["P"]), ((si["cat"], -1), (si["P"], -1), (si["catAB"], 1)), 13, 6, 1, "bimol"),
        # M8: Bimetallic product release (catAB + cat -> bimol_cat, scaled by -log10(c0) = +1.3)
        ReactionSpec("R16_bimet", (si["catAB"], si["cat"]), ((si["catAB"], -1), (si["cat"], 1), (si["P"], 1)), 14, 7, 1, "bimol_cat"),
    ]

    # Global shared reaction-type prior means (strictly no per-mechanism tuning).
    # Catalyst-catalyst bimolecular steps (bimol_cat, deact2) share the exact dimensional
    # scaling -log10(c0) = +1.3 relative to cat-substrate / 1st-order counterparts (Section 6.2).
    prior_means = {
        "bimol": 2.2,
        "bimol_cat": 3.5,   # bimol + 1.3 (-log10(c0))
        "unimol": 1.6,
        "act": 0.6,
        "deact1": 0.4,
        "deact2": 1.7,      # deact1 + 1.3 (-log10(c0))
        "bg": -1.2,
    }
    if prior_means_override:
        prior_means.update(prior_means_override)

    obs_AP = (si["A"], si["P"])
    obs_full = (si["A"], si["B"], si["P"], si["catAB"])

    catalog = [
        DesignSpec("D0", 1.0, 1.5, 0.0, 1.0, 0, obs_AP,   1.0, ("no_P", "basic", "full")),
        DesignSpec("D1", 1.0, 1.5, 0.0, 0.5, 0, obs_AP,   1.0, ("no_P", "basic", "full")),
        DesignSpec("D2", 1.0, 1.5, 0.0, 2.0, 0, obs_AP,   1.0, ("no_P", "basic", "full")),
        DesignSpec("D3", 1.0, 2.5, 0.0, 1.0, 0, obs_AP,   1.0, ("no_P", "basic", "full")),
        DesignSpec("D4", 2.0, 2.5, 0.0, 1.0, 0, obs_AP,   1.0, ("no_P", "basic", "full")),
        DesignSpec("D5", 0.5, 1.0, 0.0, 1.0, 0, obs_AP,   1.0, ("no_P", "basic", "full")),
        DesignSpec("D6", 0.5, 1.0, 0.5, 1.0, 0, obs_AP,   1.0, ("basic", "full")),
        DesignSpec("D7", 1.0, 1.5, 0.5, 1.0, 0, obs_AP,   1.0, ("full",)),
        DesignSpec("D8", 1.0, 1.5, 0.0, 1.0, 1, obs_AP,   1.0, ("full",)),
        DesignSpec("D9", 1.0, 1.5, 0.0, 1.0, 0, obs_full, 3.0, ("full",)),
    ]

    return ReactionGrammar(
        species=species,
        reactions=reactions,
        modules=modules,
        param_names=param_names,
        prior_means_by_type=prior_means,
        prior_std=prior_std,
        c0_base=0.05,
        max_active_modules=3,
        catalog=catalog,
    )
