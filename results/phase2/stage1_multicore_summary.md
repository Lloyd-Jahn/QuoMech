# QuoMech Phase 2 Stage 1 Summary: 4-Core 527-Mechanism Library & Quotient Hierarchy (E0)

- **Library**: 4 catalytic cores (`C1` Ordered A-first: 151, `C2` Ordered B-first: 151, `C3` Random-order: 151, `C4` Eley–Rideal: 74) × 8 optional modules (`k ≤ 3`) = **527 mechanisms** (13 species, 24 elementary reactions, 19 parameter slots).
- **C3 dual-branch threshold**: `0.10`.

## 1. Stage 1a Probe & Stage 1b Pools

- **ROS2 vs SciPy Radau (`1e-10`)** on 1000 multicore samples: `max_err = 5.97e-07`, `p99_err = 3.34e-07`, `max_atom_err = 1.28e-13`, `ros2_fails = 0`.
- **Filtered acceptance** (all 527 mechs): median `17.30%`, min `1.80%`, `<3%` count = `4`, solver fails = `0`.
- **Unfiltered acceptance** (D0 conv 25–98%): median `58.40%`, min `31.30%`.

### Offline Trajectory Pools (`t_max=4.0`, 80-pt early-log-dense, 14 designs/sample, fp16)

| Pool | Split | Samples | Size (GB) | Sim Speed (traj/s) | Rej Fails | Sim Fails |
|---|---|---:|---:|---:|---:|---:|
| `filtered` | `train` | 3,162,000 | 26.39 | 11,435 | 0 | 0 |
| `filtered` | `val` | 52,700 | 0.44 | 11,400 | 0 | 0 |
| `filtered` | `test` | 52,700 | 0.44 | 11,414 | 0 | 0 |
| `unfiltered` | `train` | 3,162,000 | 26.39 | 9,677 | 0 | 0 |

## 2. E0 Quotient Hierarchy Across 6 Design Sets (`σ_ref = 1%`, pass rate `≥ 85%`)

| Design Set | #Designs | Classes | Singletons | Non-Trivial Classes (Mechs) | Max Size | Within / Cross-Core Edges | Cross-Core Classes | 0-Iter Inherited | Elapsed (s) |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| `D0` | 1 | **26** | 0 | 26 (527) | 95 | 12534 / 29235 | 24 | 502/18502 | 9945.2 |
| `D0_D5` | 2 | **99** | 23 | 76 (504) | 30 | 5480 / 6689 | 56 | 0/0 | 12606.9 |
| `Xi_noP` | 6 | **290** | 147 | 143 (380) | 7 | 548 / 286 | 54 | 94/1014 | 6682.1 |
| `Xi_basic` | 7 | **306** | 160 | 146 (367) | 6 | 417 / 229 | 50 | 1/522 | 5950.3 |
| `X_basic_ref` | 9 | **417** | 323 | 94 (204) | 4 | 148 / 8 | 4 | 0/0 | 4327.3 |
| `Xi_full` | 10 | **392** | 289 | 103 (238) | 5 | 162 / 116 | 51 | 0/0 | 6835.3 |

## 3. Sensitivity Analysis (`σ_ref ∈ {0.5%, 1%, 2%}` × pass rate `∈ {80%, 85%, 90%}`)

| Design Set | 0.5% (85%) | 1.0% (80% / **85%** / 90%) | 2.0% (85%) | Non-Trans (1%, 85%) |
|---|---:|---|---:|---:|
| `D0` | 63 | 23 / **26** / 38 | 13 | 1/1 |
| `D0_D5` | 210 | 79 / **99** / 124 | 52 | 2/2 |
| `Xi_noP` | 403 | 247 / **290** / 318 | 172 | 21/23 |
| `Xi_basic` | 428 | 261 / **306** / 335 | 195 | 27/29 |
| `X_basic_ref` | 484 | 378 / **417** / 435 | 298 | 18/27 |
| `Xi_full` | 506 | 334 / **392** / 426 | 214 | 33/38 |

## 4. Core-Pair Equivalence Edge Matrix (`σ_ref = 1%`, `≥ 85%`)

### `D0`
| | C1 | C2 | C3 | C4 |
|---|---:|---:|---:|---:|
| **C1** | 3491 | 7062 | 7331 | 2404 |
| **C2** | 7062 | 4625 | 7807 | 2272 |
| **C3** | 7331 | 7807 | 3884 | 2359 |
| **C4** | 2404 | 2272 | 2359 | 534 |

### `Xi_basic`
| | C1 | C2 | C3 | C4 |
|---|---:|---:|---:|---:|
| **C1** | 109 | 15 | 69 | 11 |
| **C2** | 15 | 113 | 134 | 0 |
| **C3** | 69 | 134 | 183 | 0 |
| **C4** | 11 | 0 | 0 | 12 |

### `X_basic_ref`
| | C1 | C2 | C3 | C4 |
|---|---:|---:|---:|---:|
| **C1** | 44 | 1 | 2 | 0 |
| **C2** | 1 | 51 | 5 | 0 |
| **C3** | 2 | 5 | 49 | 0 |
| **C4** | 0 | 0 | 0 | 4 |

### `Xi_full`
| | C1 | C2 | C3 | C4 |
|---|---:|---:|---:|---:|
| **C1** | 27 | 5 | 40 | 0 |
| **C2** | 5 | 47 | 71 | 0 |
| **C3** | 40 | 71 | 77 | 0 |
| **C4** | 0 | 0 | 0 | 11 |
