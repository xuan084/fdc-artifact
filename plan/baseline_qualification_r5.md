# 第 5 轮严格资格表（`r5_baseline_qualification`，2026-10-03）

> 范围：methodology §4.2–4.3 列出的每一个方法。中文为控制面说明，方法名、定理号、引文保持英文原文。
> 证据：`exp/results/pilots/r5_baseline_qualification/{summary.json, toy_fwer.csv, timing.csv}`；代码 `exp/code/dsswm/baselines/{b4_bal,hait_sw,molitor_wor}.py`、`exp/code/dsswm/streams/r5_registry.py`；单测 `exp/code/dsswm/tests/test_r5_rivals.py`（20/20 通过）。r4 冻结模块（6297e7f2）一行未改。

## 0. 资格判据（methodology §4.1，原样）

(i) 已发表定理或 r4/r5 已签字证明；(ii) 假设覆盖本设定：**WoR** 有限池回放、**adaptive counts**（数据依赖计数）、**checkpoint stopping**（K = 20 固定检查点）、**one δ over 15 problems**；(iii) 移植变更逐条列明；(iv) toy FWER 0/200（单侧 CP 上界 0.0149）；(v) 至少与 FDC 同等的调参机会。

**所有矩形（radius-sum）证书共用的有效性论证**（qfc_lemma §6 "Valid route (R1, rectangle)"）：每个单元池按自己的预置换顺序被读取，"the j-th draw from cell c is y_{c,Π_c(j)} no matter when it is drawn"；WSR20（Waudby-Smith & Ramdas, NeurIPS 2020, Thm 4）的 WoR 经验 Bernstein CS 在 n ∈ {1..N_c} 上时间一致，所以在任意数据依赖计数、任意检查点停止下都成立。对 S·A = 18 个单元各取 δ/(S·A) 双侧并集，得到一个总概率 ≥ 1 − δ 的单元事件；在这个事件上，15 题的所有证书、所有检查点**同时**成立，因此 15 题天然共用一个 δ，无需按题再拆。下文记作 **[RECT]**。

## 1. 严格对手集 𝓡（锁前冻结，C1 的 IUT 对象）

| ID | 论文 / 定理 | 适用假设 | WoR | 自适应计数 | 检查点停止 | 15 题单 δ | 移植变更 | 有效性类 | 调参网格（开发 900–949） | 计入 𝓡 |
|---|---|---|---|---|---|---|---|---|---|---|
| **B1** CLUCB-joint | Chen, Lin, King, Lyu & Chen, NeurIPS 2014（CLUCB）；证书 [RECT] | 原文为 i.i.d. 有放回 + 自身置信半径；证书改为 WSR20 后假设只剩"单元按预置换顺序读取" | ✔（WSR20） | ✔（时间一致 CS） | ✔ | ✔（单元事件） | 15 题联合候选集；段为外生到达，段内选 CS 最宽的候选臂；耗尽时按 CS 宽度降序重选（r4 `clucb_joint.py`） | rigorous | 无自由度 | **是** |
| **B4** uniform（池比例） | [RECT] | 同上 | ✔ | ✔（池比例本身非自适应） | ✔ | ✔ | 日志随机顺序读取（r4 `uniform_rs.py`） | rigorous | 无 | **是**（工业基线） |
| **B4-bal** | [RECT]，冻结设计 | 同上 | ✔ | ✔（冻结份额，计数是 𝒜 的函数） | ✔ | ✔ | 新增 `b4_bal.py`：对照份额 `share`，其余臂均分；`make_schedule` 预生成；耗尽臂用预抽均匀数重选（`StreamEngine._resel`）。share = 0.5 时设计与 FDC **完全相同**，FDC / B4-bal 隔离"证书形状"（C4） | rigorous | share ∈ {0.40, 0.45, 0.50} | **是** |
| **B2-rect** | Jourdan, Mutný, Kirschner & Krause, ALT 2021（CombGame 采样）+ [RECT] | 采样规则不承担有效性 | ✔ | ✔ | ✔ | ✔ | 15 题联合 best-response；段内 AdaHedge + C-tracking；强制探索 √n_s（r4 `combgame_joint.py`） | rigorous | 强制探索常数 ∈ {√n_s, 0.5√n_s}（由 `r5_rival_tuning` 以子类实现；r4 模块冻结，见 §4 待办） | **是** |
| **B3-rect** | Fiez, Jain, Jamieson & Ratliff, NeurIPS 2019（RAGE）+ [RECT] | 分阶段采样不承担有效性 | ✔ | ✔ | ✔ | ✔ | 帧前沿 XY 设计、阶段内 C-tracking、15 题并集（r4 `rage_frontier.py`） | rigorous | r4 冻结配置（原文常数，不调） | **是** |
| **Peace-rect** | Katz-Samuels, Jain, Karnin & Jamieson, NeurIPS 2020（arXiv 2006.11685, Alg. 1/2）+ [RECT] | 同上 | ✔ | ✔ | ✔ | ✔ | Alg. 2 推荐常数 α = 4 的 -fav 调度作为采样；证书换成 [RECT]（r4 `peace_frontier.py`） | rigorous | r4 冻结配置（不调） | **是** |
| **Hait-SW** | Hait, arXiv 2609.37873：§4.2 式 (4.4)、Prop. 4.2 式 (4.8)/(4.11)、Lemma 6.1、**Thm 6.2**、§7 Cor. 7.1 | 原文局部 CS 为 PrPl-EB（有放回鞅）；Thm 6.2 要求每个局部 CS 时间一致错误 ≤ α/J，允许 "any predictable adaptive allocation rule" | ✔（局部 CS 换成 WSR20，其 WoR 版本） | ✔（Lemma 6.1 的 WoR 类比：单元局部时间 + 预置换） | ✔ | ✔（单元级 δ/(S·A)，比原文 α/(HJ) 更紧） | 见 §3.1 | rigorous | p_min ∈ {0.02, 0.05, 0.10} × γ ∈ {2/3, 1}（**扩展**，见 §3.1(1)） | **是** |
| **Molitor-WoR** | Molitor, arXiv 2606.17515：**Prop. 1**（引 Waudby-Smith et al. 2024）、**Thm 1** | 原文为无限 i.i.d. 上下文 bandit 流，逐策略 CS 取 α/m 并集 | ✔（逐策略 CS = Σ_s w_s · 单元 WSR20 CS） | ✔（池比例；即使自适应也成立） | ✔ | ✔ | 见 §3.2 | rigorous | 无自由度 | **是**（按构造被 B4 支配，如实标注） |
| **QFC-pool** | r4 qfc_lemma **Thm 1**（已签字） | 非自适应调度（计数为 𝒜 的函数） | ✔（Lemma L1：Hoeffding 凸序） | 不适用（池比例非自适应，Thm 1 覆盖） | ✔（K 检查点并集） | ✔（Q\*-并集 L1 = 15.161，r4 账本 0.04/0.01/48） | 无（r4 原样） | rigorous | 无（r4 账本） | **是**（我们自己的前作） |

**主方法 FDC**：定理 FDC-1（`plan/theory/fdc_theorem.md`，`r5_fdc_spec_theory` 签字中），不调参，账本由 ctx 自动计算；不在 𝓡 中（它是被比较的一方）。在本任务时 `dsswm/baselines/fdc.py` 尚未存在（`r5_fdc_impl`），注册表对 "FDC" 做惰性解析。

## 2. 排除或只作非严格对照

| 方法 | 归类 | 依据 / 理由 |
|---|---|---|
| **QFC-half**（50/50 + r4 账本） | **ablation（FDC 的消融）** | 设计、证书、代码路径与 FDC 相同，只差账本常数（δ_main, δ_var, C_var, 并集对象）。给它同等合法的会计它就是 FDC，比值恒为 1。进入 C4 2×2×2 析因，不计入 C1。注册表 role = `ablation`。 |
| **B2-nominal / B2-CG**（σ² = 1/4；CombGame 理论阈值 d0 = 2S） | **nominal** | `combgame_joint.py:23` 标注 "nominal -- valid with replacement, unproven under WoR + adaptive counts"。qfc_lemma §6："Kaufmann–Koolen mixture thresholds assume i.i.d. (with-replacement) draws per arm. Under WoR, the conditional mean of the next draw is the mean of the remaining pool, not μ_c"；固定计数 + 检查点并集的论证在自适应计数下被击穿（"crossing rate is 0.0845, against the claimed bound e^{−3} = 0.0498 (ratio 1.70)"）。经验 FWER 再低也不晋升。 |
| **B3-nominal / Peace-nominal** | nominal | 同上（"Phase-wise arguments (RAGE / Peace) break for the same reason"）。 |
| **Hait-pooled**（目标级 PrPl-EB，Hait §3 式 (3.3)） | **排除** | 原文的 pooled 序列要求有放回抽样：§12.1 "It treats the released PromptEval correctness matrix as a finite empirical population and samples prompt-item cells **with replacement**"；式 (3.2) 的鞅差性质 E(Z_t − Δ_Q) = 0 依赖每次抽样的条件均值等于总体均值，WoR 下条件均值是剩余池均值，不成立。对重要性加权混合流的 WoR 目标级 CS 没有现成证明；改为有放回又改变估计对象（不再是日志有限总体上的回放）。另外它只针对**一个**预设目标，而本设定是 15 题 × |Π_{B_q}| 个方向，需要的正是 Hait 所说的 "stratum-resolved" 架构（§4.2：当需要 "a family of target distributions will be queried from the same local confidence event"）。 |
| **Shekhar & Howard 2605.21736** | related work | 原文（已用 arXiv MCP 核对）：Thm 4.5 的有限目录下界淘汰用 "a Bonferroni-adjusted normal interval over the finite catalog, LCB_π = Δ̂_π − z_{1−α/(2|𝒫|)} se_π"，即**固定样本、渐近正态**，不是任意时刻有效。移植到检查点停止只能对 K 再取并集且仍是渐近的；有限样本化后与 Molitor-WoR / B4 同构且被它们支配。 |
| **B2-fav、Peace-fav（发表配置）、B3-fav** | none（plug-in） | 无有限样本保证，作 E-cost 描述性对照，如实报告速度与错误流。 |
| **B5 maq** | asymptotic | Qini 曲线 bootstrap，仅渐近。描述性。 |
| **H8-\***（JPC1、Tboot、MisLid） | model | 只在参数模型类内有效。描述性。 |
| **B2-fav-tight、FIX-bal-fav、Peace-fav-bal** | none（tuned plug-in） | 由 `r5_setup_plugin_variants` 实现并通过 `r5_registry.register` 注册；用于估计"保证的代价"，不计入 𝓡。 |

## 3. 新对手的移植细节与原文核对（arXiv MCP，2026-10-03）

### 3.1 Hait-SW（`baselines/hait_sw.py`）

原文段落（逐字摘录）：
- §4.2 式 (4.4)："A conservative target-level confidence sequence is obtained by weighting the local intervals, with half-width W(n) = Σ_{j=1}^{J} q_j b_j(n_j). (4.4)"
- Prop. 4.2 / 式 (4.11)："For a variance-adaptive local sequence with leading constant proportional to σ_j, the analogous first-order rule is approximately n_j^{id} ∝ (q_j σ_j)^{2/3}. (4.11) … **Neither rule is the Neyman allocation q_j σ_j.** This difference is a property of the stratum-resolved additive-width construction"
- Lemma 6.1："Suppose π_t is H_{t−1}-measurable, A_t is drawn before observing D_t … Then the time-changed process M̃_{j,t}(δ_j) = M_{j,N_j(t)}(δ_j) is a nonnegative supermartingale with respect to the global filtration."
- **Thm 6.2**（Anytime-valid target interval）："if the J local confidence sequences each have time-uniform error probability at most α/J, then P{Δ_Q ∈ [L_{Q,t}, U_{Q,t}] for every t} ≥ 1 − α. This statement remains valid under any predictable adaptive allocation rule."
- §7 Cor. 7.1："Construct local pair-by-stratum confidence sequences at error level α_0 = α/(HJ) … P(∃t, ∃w: w is identified as uniquely best at t and min_{k≠w} Δ_{wk,Q} ≤ 0) ≤ α."
- §12.1："samples prompt-item cells with replacement"（原文回放是有放回的，故局部 CS 必须换成 WoR 版本）。

移植变更：
1. **分配律（与 methodology §4.2 的文字不一致，需写入决策日志）**：methodology 写"段内按 σ 做 plug-in 分配，即 Neyman"。但 Hait 的宽度最优律在本架构（加性汇总的矩形宽度）下是 (q σ)^{2/3}，并且原文明说"Neither rule is the Neyman allocation"。同一段的两臂在每个挑战者的矩形宽度中权重相同（都是 w_s），所以段内版本为 p_{s,a} ∝ σ̂_{s,a}^{2/3}。实现：默认 γ = 2/3（忠实于 Hait），γ = 1（Neyman，即 methodology 文字）作为调参选项。调参网格因此从 3 点扩到 p_min × γ = 6 点。这只增加对手的调参机会，属于保守方向。
2. σ̂：Laplace 平滑 plug-in m = (sum+1)/(n+2)，σ̂ = R√(m(1−m))，用**上一个重规划批次之前**的数据（滞后一批，忠实于 methodology 措辞；分配因此可预测）。有效性不依赖这一点（[RECT] 对任意采样规则成立）。
3. 探索下限：Hait Thm 6.1 用递减下限 ε_t = min{1, 2t^{−0.6}}；这里用常数下限 p = p_min + (1 − A·p_min)·p_raw，p_min 在网格上调。
4. 份额通过批边界的 C-tracking 实现（段到达是外生的，方法只能选段内臂）：目标臂 = 最大缺口 T_{s,a} + p_{s,a}·E[批内到达] − n_{s,a}；耗尽时按缺口降序重选。单测验证实现份额跟随目标（σ 小的臂少采）。
5. 局部 CS：PrPl-EB → WSR20 WoR EB CS（Thm 4，同一可预测插入下注构造的不放回版本），每单元 δ/(S·A) 双侧。
6. 决策：§7 式 (7.1) 的成对下界规则加 ε 松弛，对每题全部可行挑战者同时取最大，即 `rect_certificate`。所有成对比较共享同一个单元事件，所以不需要 H 因子。

### 3.2 Molitor-WoR（`baselines/molitor_wor.py`）

原文段落（逐字摘录）：
- **Prop. 1**（waudby-smith_anytime-valid_2024）："Let (X_t, A_t, R_t)_{t=1}^{∞} be an infinite sequence of contextual bandit data with rewards in [0,1], generated by the logging policies (h_t) … Then L_t(π; α) := (1+k)(…) ∨ 0 forms a lower (1−α)-CS for ν(π)"
- **Thm 1**（Anytime-valid optimal policy set）："For each t ≥ 1, define the optimal policy candidate set S_t := {π ∈ Π : U_t(π; α/m) ≥ max_{π′∈Π} L_t(π′; α/m)}. Then P(Π* ⊆ S_t ∀t ≥ 1) ≥ 1 − α."
- §3.1："by modifying the elimination criterion, the analyst can target … ε-optimal policy sets … We do not develop these extensions further here"。

移植变更：
1. 数据 = RCT 日志按池比例随机顺序读取（日志策略），与 B4 相同的调度。
2. 逐策略 CS = Σ_s w_s·[lo_{s,π(s)}, hi_{s,π(s)}]，由单元 WSR20 WoR CS 合成。单元事件上所有策略 CS 同时覆盖，取代 Thm 1 的 α/m 并集（这里 18 个单元 vs |Π| = 512 个策略，更紧）。Prop. 1 的 i.i.d. CS 换成 WSR20，因为回放是不放回的。
3. ε 松弛的 Thm 1 规则：领先者 π̂_q = argmax_{Π_{B_q}} LCB（即 S_t 定义中 max L_t 的那个策略，并列取最小序号），U_q = max_{π′∈Π_{B_q}, π′≠π̂_q} UCB(π′) − LCB(π̂_q)，U_q ≤ ε 即认证。**与 methodology §4.2 的公式差异**：methodology 写的是对 Π_{B_q} 全体取最大（包括 π′ = π̂）。包括 π̂ 只会额外加上领先者自身的 CS 宽度，对有效性没有任何必要，所以采用忠实于 Thm 1 的 π′ ≠ π̂ 形式。这对对手更有利。只有一个可行策略时 U_q = 0。
4. 有效性：单元事件上 J(π*) ≤ UCB(π*)，J(π̂) ≥ LCB(π̂)，所以 π* ≠ π̂ 时 J(π*) − J(π̂) ≤ U_q ≤ ε。
5. 支配关系：同一领先者下，Molitor 界保留了 π′ 与 π̂ 一致的段上的 CS 宽度，所以 ≥ 矩形界（单测 `test_molitor_matches_bruteforce_and_dominates_rectangle` 在随机状态上验证）。按构造被 B4 支配。之所以仍列入 𝓡，是为了回应"只和矩形证书比"的稻草人质疑：它是文献中唯一直接针对"从日志中任意时刻有效地识别最优策略"的已发表方法。

### 3.3 B4-bal（`baselines/b4_bal.py`）
与 FDC 同一冻结设计族；share = 0.5 时与 FDC 设计逐位相同（同一 `make_schedule` 路径、同一重选均匀数）。证书为 [RECT]。share = 0 或 1 会饿死单元（r5_t0_alloc_audit 已演示 U = +∞ 直到兄弟池耗尽），构造函数拒绝。

## 4. 验证结果（pilot，`exp/results/pilots/r5_baseline_qualification/`）

**toy FWER**（r4 近并列合成池 `_toy_population`，开发种子 900–919 × 置换 0–9 = 200 流，ε = 0.02，5 题，停在 4/5；含自适应池耗尽）：

| 配置 | 错误流 / 200 | CP 上界 | 到达停止 | 含耗尽的流 | 计费不一致 |
|---|---|---|---|---|---|
| B4-bal share 0.40 / 0.45 / 0.50 | 0 / 0 / 0 | 0.0149 | 200 | 100 / 93 / 78 | 0 |
| Hait-SW γ = 2/3，p_min 0.02 / 0.05 / 0.10 | 0 / 0 / 0 | 0.0149 | 200 | 76 / 75 / 74 | 0 |
| Hait-SW γ = 1，p_min 0.02 / 0.05 / 0.10 | 0 / 0 / 0 | 0.0149 | 200 | 78 / 78 / 76 | 0 |
| Molitor-WoR | 0 | 0.0149 | 200（中位数停在 τ_R） | 0 | 0 |
| B4（r4 参照） | 0 | 0.0149 | 200 | 0 | 0 |
| NAIVE-control（plug-in GLR，β = ln(1/δ)，功效对照） | **4**（9 个错误证书） | 0.0452 | 200 | 0 | 0 |

全部 10 个新严格配置都是 0/200，满足门（iv）；NAIVE-control 出现 4 条错误流，说明这个 toy 能检出坏证书，并非空测。0/200 只是实证冒烟，不能证明错误率为零；有效性依据仍是上面的 [RECT] 论证。

**计费一致性**：所有 toy 流和计时流的 billing_ok 均为真。
**CR9 开发计时冒烟**（种子 900–903，ε* = 0.001，并发运行，只记录秒数）：B4-bal 平均 3.0 s/流，Hait-SW 1.3 s/流，Molitor-WoR 1.8 s/流（不含 CR9 环境构建时间）。

## 5. 待办与需写入决策日志 / lock 的事项
1. Hait-SW 分配律：methodology "Neyman" → Hait 原文 2/3 律为默认，Neyman 保留为调参点；网格扩到 6 点（`r5_decision_log` 记录，`r5_prereg_lock` 冻结网格）。
2. Molitor-WoR 采用 π′ ≠ π̂ 和 argmax-LCB 领先者（对对手有利，记录）。
3. B2-rect 的强制探索常数网格需要在 `r5_rival_tuning` 中以子类实现（r4 `combgame_joint.py` 冻结，常数硬编码为 √n_s）。
4. 计时冒烟的开发种子 900–903 与调参种子 900–949 重叠，这是计划所定；本任务没有读取或汇总任何 N80 比较结果。
