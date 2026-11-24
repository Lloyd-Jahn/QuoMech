Phase 1.5 专项验证报告与图表已审核通过，现在正式启动 **Phase 2（多核心 527 类扩库 + 3 Seeds 全量正式实验）**。请先同步阅读更新后的 `docs/proposal_QuoMech.md`。

### 一、Phase 2 总体范围与工程标准
- **随机种子与统计报告**：全部正式神经模型与消融统一运行 **3 个随机种子（`seeds = [42, 43, 44]`）**，所有主表与消融表均汇报 `mean ± std` 及配对 Bootstrap 95% CI。
- **沿用 Phase 1.5 验证通过的工程配置**：
  - 全部 `float64` 刚性 ODE 求解、两阶段 LM 拟合、`Enum + BIC`、`SMC-ABC` 均在 **64 核 CPU 的 C++17+OpenMP 引擎**运行；
  - 全部神经网络采用 **`Pre-LN KV-LayerNorm + QK-Norm` + 19D 无分支 RQS-NSF**，并在 **8× RTX 3090（BF16/FP32 DDP）**上训练；
  - 全部 BOED 对比采用**位级 CRN** 与 **10,000 次配对 Bootstrap 95% CI**；所有结果落盘结构化 JSON，完整记录指标与异常边界。
- **目录隔离**：保留 Phase 1 / Phase 1.5 结果，Phase 2 数据池、模型权重与报告统一存至独立目录（`data/pool_multicore_527/`、`data/pool_multicore_527_unfiltered/`、`results/phase2/`）。

---

### 二、分阶段流水线执行清单（CPU 与 GPU 并行）

#### Stage 1a–1b：527 类准入校验与双数据池生成（64 核 CPU，预计 ~50 分钟）
1. **扩展 `grammar.py` 与 C++17 ROS2 引擎至 4 核心 × 13 物种 × 24 反应超集 × 19 参数槽位 = 527 类机理**（按 `docs/proposal_QuoMech.md` 6.1–6.2 节与附录 A）：
   - C1 `Ordered A-first`（151）、C2 `Ordered B-first`（151，`M2:complex` 绑定 `catB → catI + B`，`M3` 绑定 `catB + B ⇌ catB2`）、C3 `Random-order`（151，D0 中两条支路各占总催化周转 `≥ 15%`）、C4 `Eley–Rideal`（74，禁用 M7/M8）；
   - 先验采用**全局 7 类型量纲一致高斯先验**（`bimol=2.2, bimol_cat=3.5, unimol=1.6, act=0.6, deact1=0.4, deact2=1.7, bg=-1.2, σ=0.7`）。
2. **Stage 1a 快速准入检验**：
   - 在 1,000 组多核心随机样本上对照 `SciPy Radau (1e-10)` 校验 ROS2 精度（最大误差 `< 1e-5`、原子守恒 `< 1e-10`、零失败）；
   - 核验全部 527 类接受率（目标中位 `≥ 15%`、最低 `≥ 3%`；若 C3 双支路 `15%` 约束导致极少数 3 模块 C3 机理低于 `3%`，可将 C3 双支路下限统一调整为 `10%` 并记录，保持全局统一规则）。
3. **同步生成两个离线数据池（均为 `6,000`/机理 × 527 = 3,162,000 训练样本 × 14 设计，`t_max=4.0` 早期对数加密 80 点网格，fp16 分片存储）**：
   - **主数据池 `data/pool_multicore_527/`**：经过附录 A 效应量过滤（另生成独立 `val` `52,700` 与 `test` `52,700`）；
   - **无过滤消融池 `data/pool_multicore_527_unfiltered/`**：关闭附录 A 模块效应量过滤（保留基础数值有效性与 D0 转化率 `25%–98%` 过滤，供 Stage 2 无过滤先验消融使用，预计 ~15 分钟）。

---

#### Stage 1c ‖ Stage 2a（CPU 与 GPU 并行执行）
细粒度神经网络训练不依赖 `C_Ξ`，两个数据池生成完毕后，同时并行启动 CPU 商空间构建与 GPU 首个 Seed 训练：

- **【64 核 CPU 后台并行】Stage 1c：两阶段 `ε`-等价类构建（E0 完整产出）**
  - 按 7.3 节覆盖 6 组集合，启用两项无损加速策略：
    1. **无损早停**：Stage 2（`K2=32 × 4` 多起点 `float64` LM）中，单样本 4 起点首次命中归一化 `RMS ≤ 0.5` 即早退（同时满足 `0.5% / 1% / 2%` 三档敏感性）；任一方向累计满 **7 次失败**（上限 `25/32 < 80%`）即早停该对（保留 `{80%, 85%, 90%}` 阈值敏感性）；
    2. **嵌套倒序继承 + 0 迭代验算**：离散集合按 **`Ξ_full → Ξ_basic → Ξ_no-P → {D0, D5} → {D0}`** 倒序运行——在大集合 $D'$ 已确认为等价的对，将其保存的最优参数 $\hat{\theta}$ 在子集 $D \subset D'$ 上做 **0 次迭代前向验算**，通过则直接继承免重拟合；
    3. **`𝒳_basic^ref` 独立计算**（含 `0.25×/4×` 负载与 `t_end=2.5`，不属于 `Ξ_full` 子集，独立于离散嵌套流程计算）。
  - 完成后落盘 `results/phase2/stage1_multicore_summary.md`（汇总 527 类接受率、双数据池与 6 组商空间细化曲线及同核/跨核合并热力图数据）。

- **【8× RTX 3090 前台同步】Stage 2a：训练 `QuoMech Seed 42` 与 E4 冒烟测试**
  - 架构配置：**`d_model = 256`（约 15–18M 参数**，4 层实验编码器 + 2 层 Set Transformer 含 `Pre-LN KV-LayerNorm + QK-Norm` + 因子化核心/模块头 + 19D 无分支 RQS-NSF 8–10 层 coupling，`eff_bs = 2,048`, `lr = 8e-4`, `warmup = 2,000`, `grad_clip = 1.0`, `60,000` 步完整 Cosine LR——若 60k 时验证集 `loss_m` 或 `loss_th` 仍在明显下降，可将正式步数调至 `80,000` 步）；
  - **训练完 `QuoMech Seed 42`（约 3.5 小时）后，执行一次 E4 冒烟测试**：直接读取工作目录下 `data/real_literature/` 中的真实文献数据（若目录暂空则以合成连续可变窗样本替代），跑一遍 LOO-PPE、Top-5 局部重叠与 `>3σ` 错配残差检查，确认无量纲接口与推理管线正常后继续 Stage 2b。

---

#### Stage 2b：补齐全部 3 Seeds 正式模型与 E2 消融（8× RTX 3090 DDP）
沿用 Stage 2a 确认的同一架构与步数，补全 **3 个随机种子（`42, 43, 44`）**：
1. **`QuoMech` 主模型**：补齐 `seed 43, 44`（共 3 seeds）；
2. **`Flat NPE` 对照模型（3 seeds）**：同编码器 + 扁平 527 类 Softmax 头 + 19D 参数流；
3. **`Burés & Larrosa` 闭集分类器（3 seeds）**：分别在 527 类与 **Nature 2023 BL-20 原版可映射子集协议**上训练对照；
4. **E2 消融（各 3 seeds）**：
   - **OOD 组合泛化**：跨 4 核心留出 **34 种三模块组合（~10%）**，训练因子化头与扁平 Softmax 对照，评估未见组合 Top-1/Top-3 与校准 ECE；
   - **离散固定窗消融**：仅在离散目录 `Ξ_full` 且固定 `t_end=1.0` 上训练；
   - **无过滤先验消融**：使用 Stage 1b 生成的 `data/pool_multicore_527_unfiltered/` 训练，对比过滤先验与无过滤先验对等价类准确率与校准的影响。

---

#### Stage 3：E1 / E3 全量主表评测、粗粒度任务商空间 `C_macro` 与经典统计基线
1. **E1 摊销推断主表（Table 1，3 seeds `mean ± std`）**：
   - 在 `5,270` 留出测试例（10 例/机理）上评测 `cat_1_D0`、`cat_4`、`cat_6_noP`、`cat_7_basic`、`cat_basic_var_window`、`cat_10_full`、`cont_4_var_window`、`P_only_basic` 与高噪声稳健性（类别 Top-1/Top-3、ECE、19D SBC-KS、留出 PPE、百毫秒级 Top-5 局部重叠覆盖率与单例耗时）；
   - **BL-20 外部协议对比**：在 Nature 2023 BL-20 可映射类别上对比 `QuoMech`（开放 527 类 & 限制到 BL-20）与闭集分类器；
   - **经典统计基线（64 核 CPU）**：
     - **`Enum + LM + BIC`（200 例）**：对全部 527 类各跑 4 起点有界 LM，对比类别/机理 Top-1 与加速比；
     - **`SMC-ABC`（50 例）**：使用过滤后先验 `p_filt(θ|m)` 并要求 **`ESS ≥ 200`**，在**固定的 20 机理多核心规范子库**（覆盖 C1–C4 与 M1–M8）上执行序列回火 `SMC-ABC` 评测精确校准，并辅以全 527 类 `Top-20 ∪ {truth}` 局部后验一致性核验；
   - **附录轻量方程发现参考（100 例 PPE）**：用 `PySINDy`（如环境已装 `odeformer` 亦测 `ODEFormer`）评估留出实验曲线预测误差（PPE）。
2. **E3 商空间闭环设计主表（Table 3 & Table 4，3 seeds，位级 CRN + 配对 Bootstrap 95% CI）**：
   - 在 **527 例测试集（1 例/机理）**上评测：
     - **三级离散目录**（`Ξ_no-P` `T_eff=5`、`Ξ_basic` `T_eff=6`、`Ξ_full` `T_eff=6`，`N_MC=512`）：对比 `Random`、`Textbook RPKA/VTNA`、`Static EIG`、`Fine-grained EIG`（统一类别停止 & 细粒度停止）与 `Quotient EIG`；
     - **连续设计空间 `𝒳_basic` LHS-128 候选池**（`T=6`, `N_MC=256`，同步核对 `C_{𝒳_basic^ref}` 与 `C_{Ξ_basic}`）：对比 `Cont. Random`、`Cont. Fine-grained EIG` 与 `Cont. Quotient EIG`，并统计 Step 1–2 自发多因子 VTNA/RPKA/底物抑制聚集比例；
     - **成本归一化设计**（`Ξ_full`，总成本 `≤ 6`，`D9=3`）：对比 `Quotient EIG`、`Fine EIG / cost` 与 `Quotient EIG / cost`；
     - **单步仪器能力升级建议**（`Ξ_basic → Ξ_full \ Ξ_basic`）：汇报 `M4/M7`、跨核心 `Eley–Rideal vs C1` 与全非平凡收敛子集上的 `D9` Top-1 命中率及仅补做 1 组实验后的 `Ξ_full` 后验/Top-1 跃升；
     - **粗粒度任务商空间 `C_macro` 选点对照（复用已训 527 类权重，约 15 分钟）**：定义化学工艺诊断任务商空间 $C_{\text{macro}}$（仅判别 6 类宏副反应现象：活化 M1 / 失活 M2 / 底物抑制 M3 / 产物抑制与可逆释放 M4+M7 / 二聚 M5 / 背景与双核 M6+M8 是否存在，合并 C1–C4 核心顺序、失活亚型与 M4/M7 差异），在 `Ξ_full` 与 `LHS-128` 行动集上评测统一停止条件下的 **`Quotient EIG (C_macro)` vs `Fine-grained EIG`** 早期纯选点 AUC、Top-1@k 与步级反事实分歧增益。

---

#### Stage 4：E4 真实催化文献迁移与模型错配压测
1. **真实文献验证**：
   - 直接读取 `data/real_literature/` 中整理好的 `manifest.json` 与各体系 `curves.csv` / `meta.json`（全部使用原始发表观测数据点），对每套真实体系汇报：商空间类别后验、留一实验预测误差（LOO-PPE）、Top-5 局部后验预测重叠团、观测残差（`σ` 倍数）与 `>3σ` 错配告警状态，以及 QuoMech 推荐的下一组实验与原文实际补充实验对照；
2. **合成模型错配（Misspecification）压测**：
   - 构造 3 类真实世界失配测试集（每类 100 例）：① 温度微小漂移导致速率随时间缓慢变化；② 传感器响应因子系统标定偏移（±8%）；③ 库外未建模微量副反应（如可逆失活或三阶竞争步）；
   - 定量汇报 `>3σ` 后验预测残差告警的真阳性检出率 vs 分布内误报率（FPR）。

---

### 三、最终交付物
完成后输出：
1. 汇总全部 3-seed `mean ± std` 主表、消融表、配对 Bootstrap 95% CI 与真实文献案例的 **`results/phase2/phase2_full_report.md`**；
2. 面向 8 页顶会论文的高分辨率出版级图表（PDF + PNG）：
   - `fig1_nonadditivity_and_concept`（不可加性与商空间概念验证）
   - `fig2_quotient_hierarchy`（527 类五级商空间细化曲线 + 同核/跨核合并热力图）
   - `fig3_e1_calibration_and_speed`（E1 精度/校准 SBC-KS/PPE 与 vs BIC 加速比）
   - `fig4_e3_quotient_boed`（连续 vs 离散轨迹、停止预算节省、成本归一化曲线、单步仪器升级跃升与 `C_macro` 纯选点对比）
   - `fig5_e4_real_kinetics_and_misspec`（真实文献案例后验预测拟合与错配压测）

现在请先同步阅读更新后的 `docs/proposal_QuoMech.md`，确认 Stage 1a–1b 的快速检查与 CPU/GPU 并行流水线排期后，启动 Stage 1a。
