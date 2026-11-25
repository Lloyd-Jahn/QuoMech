# QuoMech Phase 2 Comprehensive Experimental Report (4-Core 527-Mechanism Library)

- **Reporting Mode**: arXiv v1 Fast-Track (`seed = 42`, 10,000-resample paired Bootstrap 95% CIs)
- **Mechanism Library**: 4 catalytic cores (`C1` Ordered A-first: 151, `C2` Ordered B-first: 151, `C3` Random-order: 151, `C4` Eley–Rideal: 74) × 8 optional modules (`k ≤ 3`) = **527 mechanisms** (`13` species, `24` elementary reactions, `19` parameter slots).
- **Architecture**: `QuoMechNet (p2_256)` (`15.36M` params, `Pre-LN KV-LayerNorm + QK-Norm` + Factorized Core/Module Head + 19D 10L RQS-NSF, `60,000` steps, `eff_bs=2,048`, `lr=8e-4`).

## 1. E0: 527-Mechanism Quotient Hierarchy Across 6 Design Sets

| Design Set | #Designs | Classes (`1%, 85%`) | Singletons | Non-Trivial Classes (Mechs) | Max Size | Within / Cross-Core Edges | `0.5%` / `2.0%` Sensitivity |
|---|---:|---:|---:|---|---:|---:|---|
| `D0` | 1 | **26** | 0 | 26 (527) | 95 | 12534 / 29235 | 63 / 13 |
| `D0_D5` | 2 | **99** | 23 | 76 (504) | 30 | 5480 / 6689 | 210 / 52 |
| `Xi_noP` | 6 | **290** | 147 | 143 (380) | 7 | 548 / 286 | 403 / 172 |
| `Xi_basic` | 7 | **306** | 160 | 146 (367) | 6 | 417 / 229 | 428 / 195 |
| `X_basic_ref` | 9 | **417** | 323 | 94 (204) | 4 | 148 / 8 | 484 / 298 |
| `Xi_full` | 10 | **392** | 289 | 103 (238) | 5 | 162 / 116 | 506 / 214 |

## 2. E1: Amortized Inference Main Table (`N = 5,270` Held-Out Test Cases)

| Regime | Model | Class Top-1 | Class Top-3 | Class ECE | Mech Top-1 | 19D SBC-KS (med) | D7 PPE RMSE | ms/case |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `cat_1_D0` | QuoMech | 45.46% | 79.66% | 0.0073 | 4.99% | — | — | 0.044 |
| `cat_1_D0` | Flat_NPE | 45.39% | 80.13% | 0.0130 | 4.63% | — | — | 0.027 |
| `cat_1_D0` | BL_527 | 43.66% | 78.61% | 0.0280 | 4.10% | — | — | 0.020 |
| `cat_4_D0_D2_D5_D6` | QuoMech | 45.39% | 76.76% | 0.0131 | 35.16% | 0.0336 | 0.0024 | 0.061 |
| `cat_4_D0_D2_D5_D6` | Flat_NPE | 44.04% | 74.59% | 0.0172 | 33.55% | 0.0300 | 0.0025 | 0.053 |
| `cat_4_D0_D2_D5_D6` | BL_527 | 33.06% | 61.86% | 0.0483 | 23.47% | — | — | 0.038 |
| `cat_7_basic_all` | QuoMech | 68.48% | 92.58% | 0.0184 | 56.81% | 0.0337 | 0.0017 | 0.089 |
| `cat_7_basic_all` | Flat_NPE | 66.47% | 91.50% | 0.0413 | 54.57% | 0.0379 | 0.0019 | 0.082 |
| `cat_7_basic_all` | BL_527 | 50.06% | 79.11% | 0.0360 | 37.46% | — | — | 0.061 |
| `cat_basic_var_window` | QuoMech | 70.11% | 93.74% | 0.0209 | 58.52% | — | — | 0.094 |
| `cat_basic_var_window` | Flat_NPE | 67.74% | 92.35% | 0.0442 | 56.17% | — | — | 0.081 |
| `cat_basic_var_window` | BL_527 | 49.15% | 77.23% | 0.0500 | 37.89% | — | — | 0.057 |
| `cat_10_full_all` | QuoMech | 81.63% | 97.55% | 0.0144 | 77.55% | — | — | 0.119 |
| `cat_10_full_all` | Flat_NPE | 80.00% | 96.96% | 0.0446 | 75.81% | — | — | 0.111 |
| `cat_10_full_all` | BL_527 | 65.50% | 89.60% | 0.0239 | 60.06% | — | — | 0.081 |
| `cont_4_var_window` | QuoMech | 37.89% | 67.27% | 0.0206 | 34.16% | 0.0251 | 0.0066 | 0.061 |
| `cont_4_var_window` | Flat_NPE | 36.93% | 65.35% | 0.0147 | 32.81% | 0.0271 | 0.0067 | 0.054 |
| `cont_4_var_window` | BL_527 | 22.35% | 46.28% | 0.0626 | 18.50% | — | — | 0.035 |
| `P_only_basic` | QuoMech | 53.59% | 82.69% | 0.0146 | 42.75% | — | — | 0.095 |
| `P_only_basic` | Flat_NPE | 51.86% | 81.50% | 0.0176 | 40.55% | — | — | 0.083 |
| `P_only_basic` | BL_527 | 36.98% | 65.43% | 0.0368 | 26.62% | — | — | 0.056 |
| `cat_7_basic_noise2pct` | QuoMech | 55.33% | 83.53% | 0.0104 | 43.95% | — | — | 0.092 |
| `cat_7_basic_noise2pct` | Flat_NPE | 53.70% | 82.39% | 0.0219 | 42.03% | — | — | 0.081 |
| `cat_7_basic_noise2pct` | BL_527 | 42.54% | 71.25% | 0.0409 | 31.02% | — | — | 0.056 |
| `cat_7_basic_noise3pct` | QuoMech | 44.35% | 73.81% | 0.0186 | 33.47% | — | — | 0.090 |
| `cat_7_basic_noise3pct` | Flat_NPE | 43.49% | 72.79% | 0.0139 | 32.64% | — | — | 0.081 |
| `cat_7_basic_noise3pct` | BL_527 | 35.18% | 62.52% | 0.0526 | 25.24% | — | — | 0.056 |

- **Real-Time Local Top-5 Posterior Predictive Overlap (`cont_4_var_window`, `N=200`)**: median `102.6 ms/case`, mean clique size `4.45`, true-mech coverage `77.0%`, in-dist false `>3σ` alarm rate `0.0%`.

### 2.1 Nature 2023 BL-20 External Protocol (`200` Test Cases)

| Regime | BL-20 Closed-Set CNN | QuoMech (Restricted BL-20) | QuoMech (Open-527 Class Top-1) | Flat NPE (Restricted BL-20) |
|---|---:|---:|---:|---:|
| `cat_4_D0_D2_D5_D6` | 74.00% | **77.00%** | 53.50% | 78.00% |
| `cat_7_basic_all` | 81.50% | **89.50%** | 78.00% | 86.50% |
| `cont_4_var_window` | 65.50% | **78.00%** | 55.00% | 76.00% |
| `P_only_basic` | 77.00% | **78.00%** | 60.00% | 82.00% |

### 2.2 E2 Controlled Ablations (`5,270` Test Cases)

#### (a) OOD 34 Held-Out 3-Module Combinations (`340` Unseen OOD / `4,930` Seen Cases)

| Regime | Model | Unseen-34 Mech Top-1 | Unseen-34 Mech Top-3 | Unseen-34 Class Top-1 | Unseen-34 Core Top-1 | Unseen-34 Mod Macro-F1 | Seen-493 Class Top-1 |
|---|---|---:|---:|---:|---:|---:|---:|
| `cat_4_D0_D2_D5_D6` | OOD_Factorized | 24.41% | 57.06% | 34.71% | 57.94% | 82.77% | 44.79% |
| `cat_4_D0_D2_D5_D6` | OOD_Flat | 0.00% | 0.00% | 13.53% | 35.00% | 78.31% | 45.40% |
| `cat_7_basic_all` | OOD_Factorized | 47.65% | 80.88% | 60.88% | 79.12% | 89.71% | 67.10% |
| `cat_7_basic_all` | OOD_Flat | 0.00% | 0.00% | 24.12% | 51.18% | 81.78% | 67.00% |
| `cat_10_full_all` | OOD_Factorized | 73.82% | 94.41% | 79.71% | 89.12% | 96.61% | 80.32% |
| `cat_10_full_all` | OOD_Flat | 0.00% | 0.00% | 18.82% | 41.76% | 90.54% | 79.78% |

#### (b) Discrete Fixed-Window (`Ξ_full, t_end=1.0`) vs Full Continuous Variable-Window `QuoMech`

| Regime | QuoMech (Full) Class Top-1 | Discrete Fixed-Window Class Top-1 | Delta |
|---|---:|---:|---:|
| `cat_7_basic_all` | **68.48%** | 68.35% | +0.13 pp |
| `cat_10_full_all` | **81.63%** | 81.59% | +0.04 pp |
| `cat_basic_var_window` | **70.11%** | 19.05% | +51.06 pp |
| `cont_4_var_window` | **37.89%** | 0.57% | +37.32 pp |
| `P_only_basic` | **53.59%** | 53.13% | +0.46 pp |

#### (c) Unfiltered-Prior vs Filtered-Prior `QuoMech` (Evaluated on Filtered Test Split)

| Regime | Model | Class Top-1 | Class ECE | 19D SBC-KS (med) | D7 PPE RMSE |
|---|---|---:|---:|---:|---:|
| `cat_4_D0_D2_D5_D6` | QuoMech (Filtered) | 45.39% | 0.0131 | 0.0336 | 0.0024 |
| `cat_4_D0_D2_D5_D6` | Unfiltered Prior | 40.57% | 0.0639 | — | — |
| `cat_7_basic_all` | QuoMech (Filtered) | 68.48% | 0.0184 | 0.0337 | 0.0017 |
| `cat_7_basic_all` | Unfiltered Prior | 63.00% | 0.1104 | 0.0487 | 0.0019 |
| `cat_10_full_all` | QuoMech (Filtered) | 81.63% | 0.0144 | — | — |
| `cat_10_full_all` | Unfiltered Prior | 75.10% | 0.1102 | — | — |
| `cont_4_var_window` | QuoMech (Filtered) | 37.89% | 0.0206 | 0.0251 | 0.0066 |
| `cont_4_var_window` | Unfiltered Prior | 33.89% | 0.0565 | 0.0445 | 0.0069 |

### 2.3 Classical Statistical Baselines (64-Core CPU) & Appendix Equation Discovery

- **`Enum + LM + BIC` (`N=200`, 527 mechs × 4 starts on `cat_4_D0_D2_D5_D6`)**: Class Top-1 = `32.00%`, Mech Top-1 = `23.50%`, Class ECE = `0.2413`, Latency = `14736.2 ms/case` (`2947.2s` total).
  - Matched GPU `QuoMech s42`: Class Top-1 = **`44.00%`**, Mech Top-1 = **`35.00%`**, Class ECE = **`0.0986`**, Latency = **`0.064 ms/case`** (**`231,173×`** faster than 64-core CPU `Enum+LM+BIC`!).
- **Adaptive Tempered `SMC-ABC` (`p_filt`, `N=50`, guaranteed `ESS ≥ 200`)**:
  - Canonical 20-Mech Sublibrary: `min ESS = 221` (`100% ≥ 200`: `100%`), Mean TV (`QuoMech` vs `SMC-ABC`) = `0.141` (Class TV = `0.119`), MAP agreement = `82.0%`, Top-1: SMC `72.0%` vs QuoMech `78.0%`, Latency = `162.99 s/case`.
  - Full-527 `Top-20 ∪ {truth}` Local Consistency: `min ESS = 260`, Mean Mech TV = `0.479` (Class TV = `0.392`), MAP agreement = `40.0%`, Class Top-1: SMC `36.0%` vs QuoMech `48.0%`, Latency = `239.55 s/case`.
- **Appendix Equation Discovery (`SINDy-STLSQ` vs `QuoMech`, `N=100` Held-out `D7` PPE)**: `SINDy-STLSQ` median PPE = `0.0587` (diverge rate `1.0%`) vs `QuoMech` median PPE = **`0.0016`** (`0%` diverge).

## 3. E3: Quotient Closed-Loop Experimental Design (`N = 527`, Bit-Exact Strict CRN + 10k Paired Bootstrap 95% CI)

### 3.1 Three-Tier Discrete Catalog BOED (`Ξ_no-P`, `Ξ_basic`, `Ξ_full`)

| Tier | Policy | AUC@T | Top-1@T | Cls-Stop (`τ=0.90`) | Fine-Stop (`τ=0.90`) | NonTriv Cls-Stop | NonTriv Fine-Stop |
|---|---|---:|---:|---:|---:|---:|---:|
| `no_P` (`T=5`) | Quotient_EIG | 0.3283 | 60.72% | 4.63 | 4.85 | 4.61 | 4.91 |
| `no_P` (`T=5`) | Fine_EIG | 0.3283 | 60.72% | 4.63 | 4.85 | 4.61 | 4.91 |
| `no_P` (`T=5`) | Static_EIG | 0.2853 | 60.72% | 4.82 | 4.92 | 4.81 | 4.95 |
| `no_P` (`T=5`) | Textbook_RPKA_VTNA | 0.3152 | 60.72% | 4.69 | 4.87 | 4.66 | 4.91 |
| `no_P` (`T=5`) | Random | 0.2841 | 60.72% | 4.80 | 4.91 | 4.79 | 4.95 |
| `basic` (`T=6`) | Quotient_EIG | 0.3704 | 66.79% | 5.37 | 5.68 | 5.35 | 5.79 |
| `basic` (`T=6`) | Fine_EIG | 0.3703 | 66.79% | 5.37 | 5.68 | 5.35 | 5.78 |
| `basic` (`T=6`) | Static_EIG | 0.3135 | 66.79% | 5.69 | 5.87 | 5.66 | 5.92 |
| `basic` (`T=6`) | Textbook_RPKA_VTNA | 0.3405 | 66.79% | 5.55 | 5.80 | 5.51 | 5.86 |
| `basic` (`T=6`) | Random | 0.3183 | 66.79% | 5.61 | 5.80 | 5.60 | 5.87 |
| `full` (`T=6`) | Quotient_EIG | 0.4267 | 79.32% | 5.06 | 5.28 | 5.04 | 5.53 |
| `full` (`T=6`) | Fine_EIG | 0.4259 | 79.13% | 5.06 | 5.25 | 5.06 | 5.47 |
| `full` (`T=6`) | Static_EIG | 0.3548 | 77.42% | 5.51 | 5.69 | 5.46 | 5.87 |
| `full` (`T=6`) | Textbook_RPKA_VTNA | 0.2905 | 58.25% | 5.69 | 5.80 | 5.61 | 5.84 |
| `full` (`T=6`) | Random | 0.2789 | 62.81% | 5.61 | 5.74 | 5.58 | 5.86 |

- **`no_P` (`CRN diff=3.026798367500305e-09`)**: End-to-end budget saving (`Fine-stop − Quot-cls-stop`, `τ=0.90`): All = **`+0.21`** exps `[+0.16, +0.27]`, Non-trivial = **`+0.29`** exps `[+0.22, +0.37]`; Same-state counterfactual disagreement `6.0%`, `Δq = +0.0010` `[-0.0256, +0.0266]` (win rate `52.9%`).
- **`basic` (`CRN diff=None`)**: End-to-end budget saving (`Fine-stop − Quot-cls-stop`, `τ=0.90`): All = **`+0.31`** exps `[+0.24, +0.38]`, Non-trivial = **`+0.43`** exps `[+0.33, +0.54]`; Same-state counterfactual disagreement `8.0%`, `Δq = +0.0083` `[-0.0109, +0.0275]` (win rate `57.5%`).
- **`full` (`CRN diff=None`)**: End-to-end budget saving (`Fine-stop − Quot-cls-stop`, `τ=0.90`): All = **`+0.20`** exps `[+0.14, +0.25]`, Non-trivial = **`+0.43`** exps `[+0.32, +0.55]`; Same-state counterfactual disagreement `10.2%`, `Δq = +0.0097` `[-0.0081, +0.0285]` (win rate `55.7%`).

### 3.2 Continuous Design Space `𝒳_basic` (LHS-128, `T=6`, `N=527`)

| Policy | `C_{X_basic^ref}` AUC@6 | `C_{X_basic^ref}` Top-1@6 | `C_{Xi_basic}` AUC@6 | `C_{Xi_basic}` Top-1@6 | Cls-Stop (`τ=0.90`) | Fine-Stop (`τ=0.90`) |
|---|---:|---:|---:|---:|---:|---:|
| Continuous_Quotient_EIG | 0.4144 | 78.56% | 0.4434 | 79.51% | 5.23 | 5.49 |
| Continuous_Fine_EIG | 0.4163 | 78.18% | 0.4447 | 77.99% | 5.22 | 5.47 |
| Continuous_Random | 0.2564 | 59.96% | 0.2889 | 63.95% | 5.80 | 5.87 |

- **Spontaneous Spatial Clustering (Uniform baseline vs Step-1 / Step-2 Quotient EIG)**: VTNA extreme cat loading `57.0%` → Step-1 **`98.7%`** (`n=298`); RPKA zone `59.4%` → Step-2 **`66.3%`** (`n=92`).

### 3.3 Cost-Normalized `Ξ_full` (`cost ≤ 6`, `D9 = 3`) & 1-Shot Instrument Upgrade (`Ξ_basic → {D7, D8, D9}`)

| Cost Policy | Cost-AUC | Top-1@cost6 | Mean Exps | `D9` Ever (All) | `D9` Ever (M4/M7) | `D9` Ever (C4 vs C1) |
|---|---:|---:|---:|---:|---:|---:|
| Quotient_EIG_per_cost | 0.3354 | 72.11% | 4.90 | 55.0% | 83.2% | 66.7% |
| Fine_EIG_per_cost | 0.3337 | 70.40% | 4.97 | 51.6% | 78.5% | 66.7% |
| Quotient_EIG_unweighted | 0.2757 | 73.62% | 4.07 | 96.6% | 100.0% | 100.0% |

| Upgrade Subset (`Ξ_basic → Ξ_full`) | `n` | `D9` Top-1 Hit Rate | `Ξ_full` Prob Before → After 1 Rec (All 3) | `Ξ_full` Top-1 Before → After 1 Rec (All 3) |
|---|---:|---:|---|---|
| `all_nontriv_basic` | 367 | **91.3%** | 47.2% → **69.9%** (71.7%) | 55.9% → **77.7%** (78.2%) |
| `converged_80_nontriv_basic` | 142 | **89.4%** | 68.9% → **89.5%** (90.6%) | 78.2% → **94.4%** (94.4%) |
| `m4m7_family_all_nontriv` | 149 | **100.0%** | 36.6% → **71.0%** (72.3%) | 44.3% → **79.2%** (79.2%) |
| `m4m7_family_converged_80` | 63 | **100.0%** | 51.0% → **88.8%** (90.1%) | 61.9% → **95.2%** (95.2%) |
| `cross_core_C4_vs_C1_all` | 6 | **83.3%** | 42.6% → **96.1%** (95.6%) | 33.3% → **100.0%** (100.0%) |
| `any_cross_core_splittable_all` | 78 | **82.1%** | 49.7% → **70.6%** (72.5%) | 57.7% → **80.8%** (82.1%) |

### 3.4 Zero-Retrain Coarse Task Quotient `C_macro` (42 Classes, Mean Size `12.5`)

| Action Space | Policy | AUC@3 | AUC@6 | Top-1@2 | Top-1@3 | Top-1@6 | Cls-Stop (`τ=0.90`) | `Q − F` AUC@3 95% CI |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `Ξ_full` | Quotient_EIG | 0.4523 | 0.6003 | 69.26% | 78.37% | 85.77% | 3.99 | **`+0.0194`** `[+0.0105, +0.0281]` |
| `Ξ_full` | Fine_EIG | 0.4329 | 0.5864 | 65.84% | 77.23% | 85.01% | 4.06 | — |
| `Ξ_full` | Random | 0.3332 | 0.4832 | 48.96% | 63.38% | 80.46% | 4.85 | — |
| `LHS-128` | Quotient_EIG | 0.4813 | 0.6286 | 73.62% | 82.54% | 89.75% | 3.81 | **`+0.0225`** `[+0.0122, +0.0333]` |
| `LHS-128` | Fine_EIG | 0.4588 | 0.6085 | 69.07% | 79.51% | 88.61% | 4.03 | — |
| `LHS-128` | Random | 0.3253 | 0.4565 | 50.28% | 62.24% | 77.61% | 5.10 | — |

## 4. E4: Real Published Catalysis Benchmarks (`11` Systems: `9` In-Library + `2` OOD) & Misspecification Stress Tests

- **Runtime**: Median batched Top-5 local overlap = **`120 ms`**, warm-start 20-iter LM = **`1952 ms`**, mean in-library LOO-PPE = **`0.0680`**.

| Case | Type | Exp. Mods | Hier MAP (`p`) | Warm-Start `BIC*` | `z_lm` (`core`) | Alert | LOO-PPE | BOED Next Rec |
|---|---|---|---|---|---:|:---:|---:|---|
| `elsby_2021_mn_hydroboration` | IN | `Core` | `C4+M1+M2:complex+M6` (0.43) | `C4+M1+M2:complex+M6` | 1.66σ (2.37σ) | No | 0.1143 | `D8` |
| `herbort_2021_co_hydroalkenylation` | IN | `Core` | `C2+M2:complex+M4+M8` (0.24) | `C2+M2:complex+M4+M8` | 1.03σ (1.17σ) | YES | 0.0391 | `D4` |
| `wei_2020_rh_cyclopropanation_dcm` | IN | `M3,M2` | `C2+M2:complex+M3+M8` (0.62) | `C3` | 3.12σ (3.12σ) | YES | 0.0968 | `D1` |
| `antico_2023_mn_silane_oxidation` | IN | `M1,M3` | `C2+M1+M2:complex+M3` (1.00) | `C3+M1+M2:free1+M8` | 1.78σ (3.21σ) | No | 0.0518 | `D2` |
| `aikonen_2018_au_ynone_transposition` | IN | `M4,M2` | `C4+M2:free2+M5` (0.19) | `C4+M2:free2+M5` | 2.35σ (4.18σ) | No | 0.0871 | `D2` |
| `thiel_2013_ru_rcm` | IN | `M2,M4` | `C1+M2:free1` (0.14) | `C4+M1+M2:free2` | 1.14σ (4.08σ) | No | 0.0303 | `D2` |
| `joannou_2020_fe_cycloaddition` | IN | `M2` | `C1+M8` (0.32) | `C1+M8` | 2.63σ (2.85σ) | No | 0.0629 | `D1` |
| `stroek_2021_fe_ch_amination` | IN | `M1,M5` | `C1+M1+M2:free2+M8` (0.51) | `C1+M1+M2:free2+M8` | 3.60σ (7.09σ) | YES | 0.0840 | `D8` |
| `albright_2019_fe_carbonyl_olefin_metathesis` | IN | `M8` | `C4+M2:free1+M3` (0.10) | `C4+M2:free1+M3` | 0.88σ (6.46σ) | No | 0.0455 | `D4` |
| `pfeffer_2021_pd_alkyne_dimerization_ood` | **OOD** | `Core` | `C4+M2:free1` (0.15) | `C4+M2:free1+M5` | 1.11σ (1.66σ) | No | 0.0206 | `D1` |
| `ser_2025_pd_protodeboronation_ood` | **OOD** | `Core` | `C1+M2:free2+M8` (0.08) | `C1+M1+M2:free2+M8` | 8.35σ (8.79σ) | YES | 0.1226 | `D5` |

### 4.1 3×100 Synthetic Model-Misspecification Stress Tests (`>3σ` Residual Alert)

| Stress Regime | `n` | Pure Amortized `>3σ` Rate (med `z`) | Warm-Start LM `>3σ` Rate (med `z`) | AUROC (Amort / LM) |
|---|---:|---:|---:|---:|
| In-Distribution Control (FPR) | 100 | 0.0% (0.99σ) | 0.0% (0.97σ) | — |
| Temperature Drift (TPR) | 100 | 0.0% (1.05σ) | 0.0% (1.02σ) | 0.758 / 0.715 |
| Sensor Bias ±8% (TPR) | 100 | 79.0% (3.58σ) | 65.0% (3.30σ) | 1.000 / 1.000 |
| Unmodeled Side Rxn `2A→X` (TPR) | 100 | 67.0% (3.64σ) | 54.0% (3.14σ) | 1.000 / 1.000 |
