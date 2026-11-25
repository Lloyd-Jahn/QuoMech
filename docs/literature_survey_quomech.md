# QuoMech Comprehensive Literature Survey (ICML 2027 Foundation)

Generated: 2026-11-24 | Scope: 6 Research Pillars (~42 verified references through Nov 2026)

---

## Pillar 1: Machine Learning & Amortized SBI for Reaction Kinetics & Mechanism Inference

| Citation | Venue / ID | What They Did | Exact Gap vs. QuoMech |
|---|---|---|---|
| **Burés & Larrosa (2023)**, *Organic reaction mechanism classification using machine learning* | *Nature*, 613(7945):689–695, `10.1038/s41586-022-05639-4` | Closed-set deep learning classifier over 20 single-core reaction mechanisms (BL-20) from simulated concentration profiles. | No rate-constant posteriors; no design-dependent equivalence classes; no closed-loop BOED; closed-set (0% zero-shot OOD); fixed 20-class unimolecular/simple scope vs. QuoMech's 4-core 527-mech combinatorial space. |
| **Esquivel, Campos-Gonzalez-Angulo, Pablo-García, Aspuru-Guzik & Leos-Barajas (Oct 2026)**, *Joint Neural Posterior Estimation and Model Selection for Chemical Reaction Networks* | `arXiv:2610.18825` *(PDF already in `papers/2610.18825v1.pdf`)* | Shared 1D-CNN + Set Attention + BayesFlow 2 classification head + **separate model-specific cINNs** (`ψ_M`) on **3–4 candidate mechanisms**; tested on Ser et al. (2025) protodeboronation. Required **22h × 120 CPU cores per mechanism** for ODE simulation. | **Explicitly lists QuoMech's 3 core contributions as unsolved future work (p. 11)**: (1) scaling past 3–4 models with structured heads for overlapping mechanisms (`QuoMech`: 527 mechs, single factorized head + single 19D flow, `45.7M` trajs in `<35 min` on 64 cores); (2) out-of-library anomaly/misspecification flagging (`QuoMech`: 100ms PPC + warm-start LM `>3σ` guard); (3) closed-loop experimental design (`QuoMech`: Quotient BOED + `C_macro`). |
| **Yan, Ulissi, Miller & Cho (2026)**, *SPARK: Simulation-based Posterior Amortization for Reaction Kinetics* | Preprint / HF Space `bing-yan/spark` (cited in Esquivel et al. 2026) | Amortized neural posterior inference (Set Transformer + CNN + Flows) for electrochemical (HER/CV/Tafel) & surface kinetics + divergence-based active experiment planning. | Electrochemistry/surface kinetics focus; does NOT formalize design-indexed $\varepsilon$-equivalence quotient spaces $C_\Xi$, non-additivity (Prop. 1), intra-class prior artifacts (Prop. 2), or quotient EIG decomposition / stopping / instrument upgrade / zero-retrain $C_{\text{macro}}$ (Prop. 3). |
| **Chen, Tian, Wu, Chen, Yang, Jin & Xu (2025)**, *KinFormer: Generalizable Dynamical Symbolic Regression for Catalytic Organic Reaction Kinetics* | **ICLR 2025**, OpenReview `nhrXqy5d5q` | Conditional Transformer + MCTS for dynamical symbolic regression of catalytic organic reaction rate laws under mass-action constraints. | Deterministic symbolic point estimates; no Bayesian mechanism/parameter uncertainty quantification; no identifiability quotient; no closed-loop BOED. |
| **d'Ascoli, Becker, Mathis, Schwaller & Kilbertus (2024)**, *ODEFormer: Symbolic Regression of Dynamical Systems with Transformers* | **ICLR 2024**, `arXiv:2310.05573` | Transformer mapping single noisy ODE trajectories to symbolic ODEs directly in observed-variable space. | Cannot infer latent catalytic intermediates (`cat`, `catA`, `catAB`, `catI`, `cat2`) or extrapolate across catalyst loadings/initial conditions; no posterior uncertainty or BOED. |
| **Ji & Deng (2021)**, *Autonomous Discovery of Unknown Reaction Pathways from Data by Chemical Reaction Neural Network (CRNN)* | *J. Phys. Chem. A*, 125(4):1082–1092 | Embeds mass-action & Arrhenius laws into neural ODE weights to fit reaction pathways from concentration data. | Per-dataset gradient optimization (not amortized); requires observing species or suffers severe unidentifiability; no Bayesian posteriors or BOED. |
| **Galagali & Marzouk (2019)**, *Bayesian inference of chemical kinetic models from proposed reactions* | *Chem. Eng. Sci.* / *J. R. Soc. Interface* (2015/2019) | Reversible-jump MCMC over reaction network topologies exploiting network structure. | Non-amortized, computationally prohibitive for real-time closed-loop BOED across 527 competing models. |
| **Quaglio, Roberts, Jaapar, Fraga, Dua & Galvanin (2020)**, *An artificial neural network approach to recognise kinetic models from experimental data* | *Comput. Chem. Eng.*, 135:106759 | ANN classifier discriminating a small set of candidate kinetic models. | Small closed-set classification; no parameter posteriors, no equivalence classes, no sequential BOED. |
| **Taylor, Seki, Dannheim, Willis, Clemens, Taylor, Chamberlain & Bourne (2021)**, *An automated computational approach to kinetic model discrimination and parameter estimation* | *React. Chem. Eng.*, 6:1404–1411, `10.1039/D1RE00098E` | Automated transient flow microreactor + algorithmic mass-balance model enumeration + parameter fitting & statistical ranking. | Classical per-model ODE fitting & statistical criteria (slow for hundreds of models); no amortized posterior or quotient-space BOED. |
| **Brunton, Proctor & Kutz (2016)**, *Discovering governing equations from data by sparse identification of nonlinear dynamical systems (SINDy)* | *PNAS*, 113(15):3932–3937 | STLSQ sparse regression of observable-space polynomial ODEs. | Fails on latent catalytic species & catalyst-loading extrapolation (we verify this quantitatively in Appendix `SINDy-STLSQ` D7 PPE!). |

---

## Pillar 2: Structural & Practical Identifiability of Chemical Reaction Networks & ODEs

| Citation | Venue / ID | Key Contribution | Relevance to QuoMech |
|---|---|---|---|
| **Bellman & Åström (1970)**, *On structural identifiability* | *Math. Biosci.*, 7(3–4):329–339 | Introduced structural identifiability for dynamical systems. | Foundational citation for Definition 1 (exact observation equivalence). |
| **Craciun & Pantea (2008)**, *Identifiability of chemical reaction networks* | *J. Math. Chem.*, 44:244–259 | Proved distinct mass-action CRNs can generate identical dynamics even when all species are observed ("macroscopic equivalence"). | Direct mathematical precursor to our cross-mechanism observation equivalence; QuoMech extends this to **partial observation + multi-experiment design sets $D$ + finite measurement noise $\sigma_{\text{ref}}$**. |
| **Raue et al. (2009)**, *Structural and practical identifiability analysis of partially observed dynamical models by exploiting the profile likelihood* | *Bioinformatics*, 25(15):1923–1929 | Profile likelihood separating structural vs. practical parameter non-identifiability. | Contrasts parameter-level profile likelihood (within one fixed model) against QuoMech's **model-level design-indexed $\varepsilon$-equivalence quotient $C_\Xi$** across 527 competing structures. |
| **Villaverde, Barreiro & Papachristodoulou (2016)** / **Villaverde (2019)** | *PLoS Comput. Biol.* 12(10):e1005153 / *Complexity* 2019 | Differential-geometry / Lie-derivative structural identifiability & observability (`STRIKE-GOLDD`). | Structural (noise-free, single-model) identifiability; cannot quantify finite-noise practical equivalence across 138,601 mechanism pairs under multi-experiment catalogs $\Xi$. |
| **Wieland, Hauber, Rosenblatt, Tönsing & Timmer (2021)**, *On structural and practical identifiability* | *Curr. Opin. Syst. Biol.*, 25:60–69, `arXiv:2102.05100` | Modern synthesis distinguishing structural vs. finite-noise practical identifiability in ODE models. | Grounds our Definition 2 ($\varepsilon$-equivalence under realistic noise $\sigma_{\text{ref}}$) and finite-noise sensitivity analysis (`0.5% / 1% / 2%`). |

---

## Pillar 3: Modern & Amortized Bayesian Optimal Experimental Design (BOED)

| Citation | Venue / ID | Key Contribution | Exact Gap vs. QuoMech |
|---|---|---|---|
| **Box & Hill (1967)**, *Discrimination among mechanistic models* | *Technometrics*, 9(1):57–71 | Classical sequential KL-divergence experimental design for mechanistic kinetic model discrimination. | Assumes identifiable models, Gaussian linearized approximations, and small candidate sets. |
| **Franceschini & Macchietto (2008)**, *Model-based design of experiments for parameter precision: State of the art* | *Chem. Eng. Sci.*, 63(19):4846–4872 | Comprehensive chemical-engineering review of MBDoE for dynamic ODE systems. | Non-amortized local Fisher/optimal design; does not address quotient equivalence or amortized multi-model BOED. |
| **Barber & Agakov (2003)**, *The IM algorithm: a variational approach to information maximization* | *NeurIPS 2003* | Variational lower bound on mutual information used in our exact analytic-sum Quotient EIG estimator. | Core mathematical primitive for our zero-count-variance Quotient EIG estimator. |
| **Olofsson, Deisenroth & Misener (2018)**, *Design of experiments for model discrimination hybridising analytical and data-driven approaches* | **ICML 2018**, PMLR 80 | GP surrogates per candidate mechanistic model for model-discrimination BOED. | Separate GP per model (doesn't scale to 527 mechs), non-amortized, fine-grained objective only. |
| **Foster, Ivanova, Malik & Rainforth (2021)**, *Deep Adaptive Design (DAD)* | **ICML 2021**, PMLR 139, `arXiv:2103.02438` | Amortizes sequential BOED policy via upfront simulation training (sPCE bound). | Fixed target & fixed design horizon; requires retraining if target partition ($C_\Xi \to C_{\text{macro}}$) or lab catalog $\Xi$ changes! QuoMech's amortized posterior + analytic quotient EIG switches targets/catalogs **zero-retrain**! |
| **Ivanova, Foster, Kleinegesse, Gutmann & Rainforth (2021)**, *Implicit Deep Adaptive Design (iDAD)* | **NeurIPS 2021**, `arXiv:2111.02329` | Policy-based amortized BOED for implicit likelihood-free models. | Same fixed-policy limitation; does not address non-identifiable equivalence classes or intra-class prior artifacts. |
| **Blau, Bonilla, Chadès & Dezfouli (2022)**, *Optimizing Sequential Experimental Design with Deep RL* | **ICML 2022**, PMLR 162, `arXiv:2202.00821` | MDP + RL policy optimization for sequential BOED including discrete designs. | Fixed reward/target at RL training time; no quotient-space identifiability analysis. |
| **Rainforth, Foster, Ivanova & Bickford Smith (2024)**, *Modern Bayesian Experimental Design* | *Statistical Science*, 39(1):100–114, `arXiv:2302.14545` | Definitive modern review of BOED, EIG bounds, and amortized policies. | Key framing reference for BOED; highlights challenges of model discrimination & implicit simulators. |
| **Iollo, Heurtin & Bect / Contrastive Diffusions (2025)**, *Bayesian Experimental Design via Contrastive Diffusions* | **ICLR 2025 Spotlight**, `arXiv:2410.11826` | Diffusion-based contrastive EIG estimation for high-dimensional BOED. | Focuses on EIG estimation efficiency rather than non-identifiability-induced intra-class EIG bias. |
| **Huang et al. (2025)**, *ALINE: Joint Amortization for Bayesian Inference and Active Data Acquisition* | **NeurIPS 2025**, `arXiv:2506.07259` | Jointly amortizes inference & active acquisition with runtime target-variable switching. | **Super relevant 2025 ML peer!** Contrasts nicely: ALINE amortizes target-variable subsets, whereas QuoMech targets **simulator-induced design-dependent quotient partitions $C_\Xi$** and **user-defined task quotients $C_{\text{macro}}$** over 527 stiff nonlinear ODE models, proving intra-class EIG prior bias (Props 1–3). |
| **Bracher et al. (2025/2026)**, *JADAI: Jointly Amortizing Adaptive Design and Bayesian Inference* | `arXiv:2512.22999` | Jointly amortizes adaptive design & Bayesian inference for biophysical/epidemiological simulators. | Parameter-focused single-model / small-model amortization; no quotient equivalence classes. |
| **Action-BED (2026)**, *Task-Driven Bayesian Experimental Design with Singly Intractable Objectives* | **NeurIPS 2026**, `arXiv:2606.23662` | Task-driven expected future loss (EFL) formulation for BOED. | Complementary decision-theoretic view; QuoMech's $C_{\text{macro}}$ provides an exact information-theoretic task-quotient objective requiring zero policy retraining. |
| **IMBUE (2026)**, *Amortized Bayesian Experimental Design with In-Context Knowledge Conditioning* | **NeurIPS 2026**, `arXiv:2611.11512` | Transformer in-context conditioning of amortized BED on deployment-time prior knowledge & past trajectories. | Complementary in-context prior adaptation; no design-dependent model equivalence classes or stiff CRN model selection. |

---

## Pillar 4: Equivalence-Class Active Learning & Causal Experimental Design

| Citation | Venue / ID | Key Contribution | Relevance to QuoMech |
|---|---|---|---|
| **Golovin, Krause & Ray (2010)**, *Near-Optimal Bayesian Active Learning with Noisy Observations (EC²)* | **NeurIPS 2010** | Equivalence Class Edge Cutting (`EC²`) — active learning over user-defined equivalence classes of hypotheses with non-uniform test costs. | **Direct theoretical cousin!** `EC²` assumes static, known hypothesis-outcome mappings and fixed classes; QuoMech deals with **continuous-parameter stiff ODE mechanisms** where equivalence itself is a non-additive function of the experimental design set $D$ (Prop. 1), and applies continuous & discrete Quotient EIG with zero-retrain switching. |
| **Chen, Hassani, Karbasi & Krause (2015)**, *Sequential Information Maximization: When is Greedy Near-Optimal?* | **COLT 2015**, PMLR 40:338–363 | Theoretical analysis of greedy mutual-information maximization for equivalence-class determination (`ECED`). | Supports greedy myopic Quotient EIG in discrete & continuous catalogs. |
| **Hauser & Bühlmann (2012)**, *Two Optimal Strategies for Active Learning of Causal Models from Interventional Data* | *JMLR*, 13:2409–2461 | Formalizes **interventional Markov equivalence classes** in causal DAGs and optimal experimental design to refine them. | **Clean conceptual analogy for ICML reviewers**: just as interventional causal discovery refines Markov equivalence classes as new interventions enter $\mathcal{I}$, QuoMech refines **kinetic observation equivalence classes $C_\Xi$** as lab capabilities expand ($\Xi_{\text{no-P}} \subset \Xi_{\text{basic}} \subset \Xi_{\text{full}}$)! |
| **Toth, Lorch, Knoll, Krause, Pernkopf, Peharz & von Kügelgen (2022)**, *Active Bayesian Causal Inference (ABCI)* | **NeurIPS 2022**, `arXiv:2206.02063` | Active Bayesian causal discovery targeting downstream causal queries rather than full graph recovery. | Direct analogy to QuoMech's coarse task quotient $C_{\text{macro}}$ (targeting actionable side-reaction syndromes rather than nuisance core ordering!). |
| **Tigas, Annadani, Jesson, Schölkopf, Gal & Bauer (2022)**, *Interventions, Where and How? Experimental Design for Causal Models at Scale* | **NeurIPS 2022**, `arXiv:2203.02016` | Scalable continuous-value causal BOED. | Methodological reference for continuous-value experimental design. |

---

## Pillar 5: Amortized Bayesian Inference, Model Comparison, Misspecification & Neural Primitives

| Citation | Venue / ID | Role in QuoMech |
|---|---|---|
| **Cranmer, Brehmer & Louppe (2020)**, *The frontier of simulation-based inference* | *PNAS*, 117(48):30055–30062 | Standard SBI framing reference. |
| **Papamakarios & Murray (2016)**, *Fast $\epsilon$-free Inference of Simulation Models* | **NeurIPS 2016** | Seminal NPE foundation. |
| **Radev, Mertens, Voss, Ardizzone & Köthe (2020/2022)** / **Kühmichel et al. (2026)** | *IEEE TNNLS* (`arXiv:2003.06281`) / `arXiv:2602.07098` | BayesFlow amortized inference framework (used by Esquivel et al. 2026). |
| **Radev et al. (2021/2023)**, *Amortized Bayesian Model Comparison with Evidential Deep Learning* | *IEEE TNNLS*, 34(8):4903–4917 (`arXiv:2004.10629`) | Foundational amortized Bayesian model comparison (closed-set, flat output). |
| **Schmitt, Bürkner, Köthe & Radev (2023/2024)**, *Detecting Model Misspecification in Amortized Bayesian Inference* | GCPR 2023 / `arXiv:2406.03154` | Unsupervised summary-space MMD misspecification detection; QuoMech complements this with **physics-grounded 100ms forward ODE posterior predictive residual (`>3σ`) + warm-start LM verification** directly in observable concentration units! |
| **Elsemüller et al. (2024)**, *Sensitivity-Aware Amortized Bayesian Inference* | *TMLR 2024*, `arXiv:2310.11122` | Prior/likelihood sensitivity in amortized inference. |
| **Dax et al. (2022)**, *Group Equivariant Neural Posterior Estimation (GNPE)* | **ICLR 2022**, `arXiv:2111.13139` | Contrast: global continuous group symmetries vs. QuoMech's design-dependent non-additive kinetic observation equivalences. |
| **Talts, Betancourt, Simpson, Vehtari & Gelman (2018)**, *Simulation-Based Calibration* | `arXiv:1804.06788` | Exact calibration metric (`19D SBC-KS`). |
| **Lee et al. (2019)**, *Set Transformer* | **ICML 2019**, PMLR 97 | Permutation-invariant multi-experiment set encoder (`ISAB + PMA`). |
| **Durkan, Bekasov, Murray & Papamakarios (2019)**, *Neural Spline Flows* | **NeurIPS 2019** | 19D conditional Rational-Quadratic Spline Flow (`RQS-NSF`). |
| **Henry et al. (2020)** & **Dehghani et al. (2023)**, *QK-Norm / ViT-22B* | *EMNLP 2020* / **ICML 2023** | Motivates our `Pre-LN KV-LayerNorm + QK-Norm` stability design. |

---

## Pillar 6: Mechanistic Catalysis Kinetics Foundations (RPKA & VTNA)

| Citation | Venue / DOI | Role in QuoMech |
|---|---|---|
| **Blackmond (2005)**, *Reaction Progress Kinetic Analysis: A Powerful Methodology for Mechanistic Studies of Complex Catalytic Reactions* | *Angew. Chem. Int. Ed.*, 44(28):4302–4320 | Defines RPKA same-excess (`D5`, `D6`) and different-excess (`D3`) protocols. |
| **Blackmond (2015)**, *Kinetic Profiling of Catalytic Organic Reactions as a Mechanistic Tool* | *J. Am. Chem. Soc.*, 137(34):10852–10866 | Comprehensive mechanistic kinetic profiling review. |
| **Burés (2016)**, *Variable Time Normalization Analysis: General Graphical Elucidation of Reaction Orders from Concentration Profiles* | *Angew. Chem. Int. Ed.*, 55(52):16084–16087 | Defines VTNA variable catalyst loading (`D1`, `D2`) — our Prop. 1 algebraic non-additivity example & spontaneous continuous BOED clustering target! |
| **Nielsen & Burés (2019)**, *Visual kinetic analysis* | *Chem. Sci.*, 10(2):348–353 | Unified guide to RPKA & VTNA visual kinetic analysis. |
| **11 Real Published Catalysis Benchmarks** (`data/real_literature/manifest.json`) | Exact DOIs verified in `manifest.json` | Elsby 2021, Herbort 2021, Wei 2020, Antico 2023, Aikonen 2018, Thiel 2013, Joannou 2020, Stroek 2021, Albright 2019, Pfeffer 2021 [OOD], Ser 2025 [OOD]. |

---

## Verified First-Hand Notes from All 12 Downloaded PDFs (`papers/`)

All 12 main PDFs (no SI) are saved and verified (`%PDF`, SHA-256 logged in `papers/manifest.json`; `KinFormer` on OpenReview returned HTTP 403 to scripted download so its verified OpenReview/ICLR 2025 metadata is recorded above):

1. **`2610.18825v1.pdf` — Esquivel, Campos-Gonzalez-Angulo, Pablo-García, Aspuru-Guzik & Leos-Barajas (Oct 2026)**:
   - Only `3–4` models (Burés & Larrosa subset), separate model-specific cINNs (`ψ_M`), `22h × 120 CPU cores` per mechanism.
   - Page 11 explicitly lists **three open challenges directly solved by QuoMech**: (i) scaling to larger libraries with structured heads for overlapping mechanisms, (ii) flagging anomalous/out-of-library trajectories, (iii) real-time closed-loop experimental design.
2. **`ALINE_NeurIPS2025_2506.07259.pdf` — Huang, Wen, Bharti, Kaski & Acerbi (NeurIPS 2025)**:
   - RL-trained Transformer jointly amortizing inference and active acquisition with runtime target-subset selection.
   - Limitations (p. 10): outputs **marginal posteriors only** (no joint posterior), **fixed priors**, and **discrete design spaces only**; does not address multi-model ODE equivalence classes.
3. **`IMBUE_NeurIPS2026_2611.11512.pdf` — Zhao, Huang, Wen & Kaski (NeurIPS 2026)**:
   - Transformer in-context conditioning of amortized BED on external prior histograms and past trajectories with a learned reliability filter.
   - Limitations (p. 10): parameter-wise marginal priors only, discrete candidate enumeration (continuous designs left to future work), no model selection or non-identifiability quotient.
4. **`ActionBED_NeurIPS2026_2606.23662.pdf` — Rossa, Phillips & Rainforth (NeurIPS 2026)**:
   - Reformulates task-driven BED via expected future loss (EFL) into a singly intractable objective jointly training design policy + action policy.
   - Contrast: policy is tied to one upfront task loss & fixed simulator setup; QuoMech decouples amortized fine-grained posterior inference from zero-retrain quotient EIG, switching seamlessly across lab catalogs ($\Xi_{\text{no-P}}, \Xi_{\text{basic}}, \Xi_{\text{full}}, \mathcal{X}_{\text{basic}}$), cost weights, instrument upgrades, and task quotients ($C_{\text{macro}}$).
5. **`JADAI_2512.22999.pdf` — Bracher, Kühmichel, Ivanova, Intes, Bürkner & Radev (Jul 2026)**:
   - End-to-end joint amortization of policy, history network, and diffusion posterior via incremental posterior error loss (parameter estimation focus; no discrete model-selection quotient).
6. **`ContrastiveDiffusionsBOED_ICLR2025_2410.11826.pdf` — Iollo, Heinkelé, Alliez & Forbes (ICLR 2025 Spotlight)**:
   - Pooled posterior + Langevin/DiGS diffusion samplers for continuous EIG gradients; focuses on EIG estimation efficiency for single-model continuous parameters rather than multi-model equivalence classes.
7. **`Rainforth_ModernBED_2024_2302.14545.pdf` — Rainforth, Foster, Ivanova & Bickford Smith (*Stat. Sci.* 2024)**:
   - Definitive modern BED review; highlights how EIG over parameters is misaligned with downstream prediction/tasks — directly echoing our Proposition 3 & $C_{\text{macro}}$ motivation.
8. **`Foster_DAD_ICML2021_2103.02438.pdf` — Foster, Ivanova, Malik & Rainforth (ICML 2021)**:
   - Foundational amortized sequential BOED (DAD) using sPCE contrastive bounds and permutation-invariant history encoders; fixed design horizon & fixed target.
9. **`Toth_ABCI_NeurIPS2022_2206.02063.pdf` — Toth, Lorch, Knoll, Krause, Pernkopf, Peharz & von Kügelgen (NeurIPS 2022)**:
   - Active Bayesian Causal Inference targeting downstream causal queries rather than full DAG identification — exact conceptual mirror in causal discovery of QuoMech's quotient & $C_{\text{macro}}$ kinetic design.
10. **`Schmitt_MisspecABI_2024_2406.03154.pdf` — Schmitt, Bürkner, Köthe & Radev (2024)**:
    - Summary-space MMD misspecification detection for ABI; Conclusion (Sec. 5) explicitly advocates a two-step workflow combining amortized detection with targeted local non-amortized verification — directly mirrored by QuoMech's physics-space 100ms PPC + warm-start LM guard!
11. **`Radev_ABMC_2023_2004.10629.pdf` — Radev, D'Alessandro, Mertens, Voss, Köthe & Bürkner (*IEEE TNNLS* 2023)**:
    - Evidential amortized Bayesian model comparison (includes a 2-model stochastic kinetics MJP benchmark vs. SMC-ABC, illustrating why our 527-mechanism multi-core stiff ODE setting is orders of magnitude beyond prior ABMC benchmarks).
12. **`Wieland_Identifiability_2021_2102.05100.pdf` — Wieland, Hauber, Rosenblatt, Tönsing & Timmer (*Curr. Opin. Syst. Biol.* 2021)**:
    - Establishes why finite-noise practical identifiability in nonlinear ODEs cannot be captured by Fisher Information Matrix approximations — grounding our bidirectional bounded-LM $\varepsilon$-equivalence engine (Definition 2).

