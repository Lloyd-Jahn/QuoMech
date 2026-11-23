已审核 `go_nogo_report.md`，Phase 1 全部 4 个 Gate 表现非常扎实！在正式启动 Phase 2（多核心催化拓扑扩库 + 3 seeds 全量正式实验）之前，我们先在**现有 151 类数据池 `pool_v1` 上做一轮 2–3 小时的快速专项验证（Phase 1.5）**，重点彻底验证命题 3 在“充分收敛 + 解除 `T=4` 截断 + 连续候选池”下的真实优势。

**不需要重造数据池，也不需要跑多 Seed**，请只执行以下 4 项任务，完成后输出 `phase1_5_validation_report.md` 等待我审核后立刻启动 Phase 2：

1. **充分收敛单 Seed `QuoMech`（约 1–1.5 小时）**：
   - 基于现有 `pool_v1` **从头（from scratch）按 `total_steps = 60,000` 重新配置完整 Cosine LR 调度**训练单 Seed `QuoMech` 主模型（避免沿用 16k 已衰减完的 LR schedule）；
   - 在 `16k / 30k / 45k / 60k` 同步记录 **train loss 与 val loss**（诊断 `pool_v1` 是否开始过拟合），并汇报最终验证集 loss、Top-1/Top-3、ECE、15D NSF 中位 SBC-KS 及 `cat_7_basic` / `cat_10_full` / `cat_basic_var_window` 精度，对比 16k 步的收敛增益。
2. **解除 `T=4` 截断的离散三目录 E3 对比（`N_MC = 512` + 严格 CRN 降方差）**：
   - 在 **302 例（2 例/机理）**测试集上，将闭环预算扩展至 **`T_eff = min(6, |Ξ| - 1)`**（即 `Ξ_no-P` 最多追加 5 步，`Ξ_basic` 与 `Ξ_full` 最多追加 6 步，避免耗尽候选集越界；同时记录每步 `k = 1..T_eff` 指标，重点报告 `T=4` 与 `T_eff` 两档，以及 `0.85` 与 `0.90` 两档停止阈值）；
   - **严格 CRN 规范**：对每个测试案例预生成一份固定的真实含噪观测表 $y^\star(\xi)$，且每步共享相同 MC 采样种子；
   - 在 `Ξ_no-P`、`Ξ_basic`、`Ξ_full` 上对比 `Quotient EIG` 与 `Fine-grained EIG`，同步报告全 302 例与**非平凡等价类子集（`|C_Ξ| ≥ 2`）**的 AUC、Top-1@k、步级分歧率、统一类别停止实验数、细粒度停止浪费实验数及**配对 Bootstrap 95% 置信区间**；
   - **同状态反事实分歧检验**：在 `Quotient` 轨迹的每一步状态 $D_t$ 上，复用同一批 MC 样本同步求取 $\xi_F^\star = \arg\max \mathrm{EIG}_M$；**仅针对 $\xi_Q^\star \neq \xi_F^\star$ 的分歧步**，直接对比执行 $\xi_Q^\star$ vs $\xi_F^\star$ 在真实数据上的单步类后验增益 $\Delta q(C_\Xi^{\text{true}})$ 与 $\Delta \log q(C_\Xi^{\text{true}})$。
3. **补齐连续设计空间 `𝒳` 的头对头核心对照（最关键！）**：
   - 在与 `Ξ_basic` 对应的离线连续空间 `𝒳_basic`（`a_pre=0`, 观测 `{A, P}`，连续候选观测时长统一按负载自适应 `t_end = clip(1.0 / cat_mult, 0.5, 2.5)`）的 **LHS-128 连续候选池**上，用相同的 151 例（1 例/机理，或若耗时允许用 302 例）测试集、`T=6`、`N_MC=256`（启用上述严格 CRN 与同状态反事实分歧检验）头对头对比：
     - `Continuous Random`
     - **`Continuous Fine-grained EIG`**（目标 `M`）
     - **`Continuous Quotient EIG`**（目标 `C_{Ξ_basic}`）
   - 统计三者的 AUC、Top-1@4/6、配对 Bootstrap 95% CI、步级选点分歧率、分歧步单步真实类后验增益、类内伪影 EIG 占比、统一类别停止 vs 细粒度停止实验数，以及 **Step 1（VTNA 负载区）和 Step 2（RPKA 预加产物/同过量区）**的连续最优设计 `ξ*` 自发聚集比例。
4. **顺手跑 2 个 5 分钟高价值附加检验**：
   - **成本归一化设计（`Ξ_full`，总成本预算 `≤ 6`，`D9` 成本=3）**：在相同总成本预算下同时对比 **`Quotient EIG`**、**`Quotient EIG / cost`** 与 **`Fine-grained EIG / cost`** 的类后验 AUC、最终 Top-1 与 `D9` 选用率（检验细粒度目标是否因类内伪影更频繁误选高价 `D9`）；
   - **仪器能力升级建议验证**：筛选在 `Ξ_basic` 下已收敛（`q(C_{Ξ_basic}|D) ≥ 0.90`）且属于非平凡类（`|C_{Ξ_basic}| ≥ 2`，单独统计 `M4/M7` 家族）的案例，计算目录外高阶实验 `Ξ_full \ Ξ_basic`（D7/D8/D9）的 EIG，量化汇报针对 `M4/M7` 家族推荐 **`D9`（原位 `catAB`）的 Top-1 命中率**，以及实际执行所推荐升级实验后的 `Ξ_full` 真类后验提升幅度。
