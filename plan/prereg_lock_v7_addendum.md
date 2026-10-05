# Lock v7 附录：FDC-BF 的确认性检验（新 eval 种子）

状态：DRAFT（2026-10-04）。机器可读版本：`plan/prereg_lock_v7_addendum_DRAFT.json`（锁定后为 `plan/prereg_lock_v7_addendum.json`，由 `dsswm.stats.prereg_v7.finalize_addendum` 生成，哈希由 `exp/code/build_v7_addendum_draft.py` 计算，从不手填）。
代码：`exp/code/run_r5s_v7.py`、`dsswm/stats/{prereg_v7,v7_analysis,v7_replica,v7_seal}.py`、`dsswm/envs/lenta_v7.py`、`dsswm/baselines/{fdc_bet,fdc_mr}.py`；测试 `dsswm/tests/test_v7_addendum.py`、`test_fdc_bet.py`。v5 锁、v6 附录及其所有文件都不改动；v7 gate 每次运行都重新校验 v5 与 v6。

## 0. 披露（偏离声明）

- **FDC-BF 是事后方法。** 它在 2026-10-04 v6 eval 结果公开之后设计（v6 块 A：FDC/RECT-ck-HG = 0.851 [UB 0.871]，FDC/HC-WoR = 0.869，判 downgrade）。它取代 FDC 成为主方法，这是对 v5/v6 主方法的事后偏离，论文必须如实写明日期、理由和选择过程，不得写成事前注册的方法。
- **选择只用 dev 数据。** CR9 dev 种子 900–949（FDC-MR 权重调优）、950–999（约 22 个证书变体比较，变体清单见 json `primary_method.selection_disclosure`），Lenta LR9 dev 半边种子 950–999，以及 toy FWER 检查。没有任何 v7 eval 种子或 LR9 eval 结局被使用。
- **选择规则不是事先声明的。** 主会话依据 dev 结果判断：FDC-BF 跨数据集最稳、结构最简单；MR[front3] 在 CR9 dev 略好、在 Lenta dev 差约 0.1。dev 950–999 上的数字因此**不是证据**。
- **只有块 A 是确认性的**，而且只针对新的重放随机化（排列种子 33000–33199），条件于已被 v5/v6 用过的 Criteo CR9 eval 表；块 B 是描述性的。方法的**设计**参考了已公开的 v6 eval 结果（哪种对手构造紧），只有候选变体之间的比较用的是 dev 数据。
- **阈值放宽。** 成功阈值 0.80 比 v6 块 A 的 keep 阈值 0.70 宽松；这里事前固定，不推翻 v6 的 downgrade 判定。
- **v7 runner 检查。** 用 v7 runner 在 dev 950–999 上重跑（v7a_pilot / v7b_pilot），确认它复现了选择时的数字；该检查同样不是证据。

## 1. 主方法 FDC-BF（冻结）

`make_variant('FDC-BF')` = `FDCBet(kind='bennett', var_box='HG', fpc=True, rect=False, split=(0.045, 0.005))`：
- 设计：冻结 50/50，日程摘要与 FDC 逐位相同；K = 20 个预设检查点，粘性答案，12/15 停止；
- 单元界：Bennett + 精确有限总体方差，Ψ(u; m) = n m(1−m)(N−n)/(N−1) · φ(|u|/n)；
- 方差盒：精确超几何双侧检验反演，α_side = δ_var/(2SAK)，检查点间运行交集；
- 并集：FDC 的 (q, π, k) 并集，β = ln(Σ_q |Π_{B_q}| K / δ_main)，δ_main = 0.045，δ_var = 0.005；
- 宽度：λ 取 λ₀·G 网格上 (β + Σ_c sup_box Ψ_c(λ ã_c))/λ 的最小值，G = exp(linspace(ln 1/200, ln 10, 161))，λ₀ = √(2β/V̄_pair)；
- 保证：定理 FDC-bet-1（`plan/fdc_bet_exploration.md` §2.6），在 K 个预设检查点上 FWER ≤ 0.05。保证强度与 FDC、RECT-ck-HG(-live) 相同，弱于时间一致的 HC-WoR。
- 无任何数据拟合的参数。代码哈希与来源 commit（`ec56177f`）写入 json。

次要描述性方法：FDC-MR[front3]（两块都只作描述）。

## 2. 块 A（确认性；本附录唯一的判定）

- **数据与设定**：CR9 eval 半边，ε = 0.001（与 v6 块 A 相同），K = 20，δ = 0.05，Q = 15。
- **种子**：33000–33199，200 条**新**流。2026-10-04 在整个工作区的结果 json/jsonl 中检索 seed / perm_seed 33000–33199、34000–34199：0 次命中；代码中只有 v7 runner/分析引用这些范围。
- **运行**：每个 (方法, 流) 跑一次，stop_k = 15；N80_pen、x12 与 N100_pen 读自同一轨迹。12/15 之前的轨迹与停在 12 的运行逐位相同（构造保证；v6 R3 已对 v5 对手验证）。
- **方法（16 个）**：FDC-BF、FDC-MR[front3]、FDC（v5）、B4-bal、RECT-ck-HG、RECT-ck-HG-live、RECT-ck-Bern、HC-WoR，以及 v5 有保证对手 B1、B4、B2-rect、B3-rect、Peace-rect、Hait-SW、Molitor-WoR、QFC-pool。配置：v5 冻结配置 + v6 调好的 HC-WoR；不重新调参。
- **主终点**：N80_pen（v5 定义）。配对几何均值比 FDC-BF/r，配对百分位 bootstrap B = 10⁴、seed 42（所有对手和终点共用同一重抽样矩阵），单侧 95% UB，双侧 95% CI。
- **主家族**：F = {RECT-ck-HG, RECT-ck-HG-live, HC-WoR}（同设计、同或更强保证的矩形）。
- **判定规则（IUT）**：
  - `positive_result_achieved` 当且仅当：对 F 中**每个** r，UB95(FDC-BF/r) < 0.80；**且** FDC-BF 在 200 条流上 0 条错误流（直到其运行结束，即 15/15 或 τ_R，任何错误证书都算；报告 Clopper–Pearson 95% UB）；**且** replica 通过。
  - 否则 `positive_result_not_achieved`，列出失败分量：faster_below_1（0.80 ≤ UB* < 1）、not_faster（UB* ≥ 1）、validity_failure、replica_fail。所有估计照实报告。
  - 阈值 0.80 由主会话在任何 v7 运行前给出（dev UB 为 0.677 / 0.695）。
- **次要（描述性，无判定）**：FDC-BF 对 RECT-ck-Bern、FDC、B4-bal 和九个 v5 对手的 N80_pen 比；所有对手的 x12（第 12 个粘性穿越的插值到达数，不惩罚）与 N100_pen 比；FDC-MR[front3] 的同样比值；每个方法的几何均值 N/τ_R、τ_R 处占比、错误流。

## 3. 块 B（描述性，Lenta LR9 eval 半边，结局已暴露）

- 与 v6 块 B 相同的数据、分层、拆分种子 6006；eval 结局只对通过 `prereg_v7` gate 的 `v7b_*` 任务开放（`LentaLayerEnvV7`）。
- 种子 34000–34199；ε ∈ {0.002, 0.003, 0.004} 全部报告；12/15 停止。
- 方法：FDC-BF、FDC-MR[front3]、FDC、B4-bal（share 0.4）、QFC-pool、B2-rect、Peace-rect、Hait-SW、RECT-ck-HG、RECT-ck-HG-live、RECT-ck-Bern、HC-WoR；配置为 v6 LR9 冻结配置。
- 这不是未触碰的留出集：r4 计算过全表 LR8 均值，v6 块 B 用过同一 eval 半边（种子不同）。只报告，不判定，不与 Criteo 合并。

## 4. 复核、封存与威胁模型（同 v6）

- **R1**：独立进程重跑每个 eval 任务的前 10 个种子，规范行（除计时字段）必须完全相同。
- **R2**：在 50/50 组内，每个 (seed, ε) 的 schedule_digest 必须相同（完整矩阵）。组 A = {FDC-BF, FDC-MR[front3], FDC, B4-bal, RECT-ck-HG, RECT-ck-HG-live, RECT-ck-Bern, HC-WoR}；组 B 同上去掉 B4-bal（share 0.4）。
- 分析程序从实际行重算 R1/R2，存储的报告必须完全对上；eval 行必须等于 git 提交的封存（`exp/results/full/v7_seals`）。
- 块 A replica 失败 → 判定分量 replica_fail；块 B 失败 → 只放附录并标注。
- 威胁模型：能检测意外漂移（代码、输入、原始数据、配置）、部分/续跑以及对结果、报告、封存的事后修改；故意改写 git 历史不在范围内。

## 5. 任务

| 任务 | 块 | 种子 | ε | 方法 |
|---|---|---|---|---|
| v7a_full_a | A | 33000–33199 | 0.001 | Peace-rect |
| v7a_full_b | A | 33000–33199 | 0.001 | 其余 15 个 |
| v7b_full_a | B | 34000–34199 | 0.002/0.003/0.004 | Peace-rect |
| v7b_full_b | B | 34000–34199 | 0.002/0.003/0.004 | 其余 11 个 |

命令：`cd exp/code && python run_r5s_v7.py --task <t> --workers 4`，然后 `--task <t> --replica`，然后 `--analyse A|B`。最多 3 个进程并行，每个 ≤ 4 个 worker。
