# Phase 2 Two-Milestone Strategy: arXiv v1 (`seed=42`) Fast-Track -> Camera-Ready (`seeds=[42,43,44]`)

## Milestone 1: Complete arXiv v1 Preprint (Priority — Target ~16–18h from 2026-11-24 07:55)
Complete **every single experiment, baseline, ablation, real-literature case, misspecification stress test, and publication figure** using **`seed = 42`** (with 10,000-resample paired Bootstrap 95% CIs on BOED & E1):
- **E0 (64-core CPU)**: Full 527-mech 6-set quotient hierarchy (`Ξ_full, Ξ_basic, Ξ_noP, {D0,D5}, {D0}, 𝒳_basic^ref`) — currently running in `quomech_p2_cpu`.
- **Neural Models (`seed=42`, 8× RTX 3090 dual NUMA 4-GPU DDP)**:
  1. `QuoMech s42` (60k) — DONE (`val_loss=15.489`)
  2. `Flat NPE s42` (60k) — DONE (`val_loss=15.703`)
  3. `BL-527 s42` (40k) — Slot 1
  4. `BL-20 s42` (30k) — Slot 2
  5. `E2 OOD Factorized s42` (60k, 34 held-out 3-mod) — Slot 1
  6. `E2 OOD Flat s42` (60k, 34 held-out 3-mod) — Slot 2
  7. `E2 Discrete Fixed-Window s42` (60k, `Ξ_full, t_end=1.0`) — Slot 1
  8. `E2 Unfiltered Prior s42` (60k, `pool_multicore_527_unfiltered`) — Slot 2
- **Classical Baselines (64-core CPU, parallel with GPUs once Stage 1c finishes)**:
  - `Enum + LM + BIC` (200 cases, 527 mechs × 4 starts)
  - `SMC-ABC` (50 cases, `p_filt`, `ESS >= 200`, 20-mech canonical sublibrary + `Top-20 ∪ {truth}`)
- **Stage 3 (`seed=42` full E1 + E3 + `C_macro`) & Stage 4 (E4 11 real literature + 300 synthetic misspecification)** -> `results/phase2/phase2_arxiv_v1_report.md` + 5 publication figures (`fig1`–`fig5`, PDF+PNG).

## Milestone 2: Post-arXiv Camera-Ready Backfill (`seeds = [43, 44]`)
- Resume `quomech_s43` cleanly from its saved step-12,500 checkpoint (`results/phase2/models/quomech_s43/latest_ckpt.pt`), plus fresh `flat_s43` (note: earlier `flat_s43` hit an early flat-head symmetry saddle and was discarded) and remaining `s43/s44` models -> upgrade all tables to `3-seed mean ± std` for arXiv v2 / conference submission.
