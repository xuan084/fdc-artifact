# Lock v8 附录：未触碰的新表 X5 RetailHero 上的确认性检验 + 相位自适应联合对手 PJC-BF

状态：由主会话于 2026-10-04 批准设计（`prereg_lock_v8_addendum_DRAFT.md`）。机器可读版本：`plan/prereg_lock_v8_addendum.json`（由 `dsswm.stats.prereg_v8.finalize_addendum` 从 `prereg_lock_v8_addendum_DRAFT.json` 生成；所有哈希由 `exp/code/build_v8_addendum_draft.py` 计算，从不手填）。
代码：`exp/code/run_r5s_v8.py`、`run_v8_select.py`、`dsswm/stats/{prereg_v8,v8_analysis,v8_replica,v8_seal}.py`、`dsswm/envs/{x5_v8,data_v8}.py`、`dsswm/baselines/pjc_bf.py`；测试 `dsswm/tests/test_v8_addendum.py`、`test_pjc_bf.py`。v5 锁、v6/v7 附录及其文件一律不改；v8 gate 每次运行都经 v7 gate 级联校验 v7、v6、v5。

## 0. 披露（偏离声明）

- **FDC-BF 仍是事后方法。** 它在 v6 eval 之后设计，v7 确认；v8 不改它的任何参数，也不在 X5 上调参。
- **PJC 族是事后加入的对手。** 它在看过 v7 eval 结果之后、为回应两份 v7 审稿（P0-3：最强的合法有保证对手）而设计。调参只用 dev 数据（CR9 dev 900–949、X5 dev 900–949）。
- **X5 RetailHero 是在 v7 之后选定的。**
  - Criteo 没有未触碰的行：本地 `criteo_uplift` 是合成替身；真实全表（13,979,592 行）已全部分进 dev/eval 两半，eval 半边已被 v5–v7 使用；
  - MegaFon 被 scikit-uplift 文档标为生成的合成数据，不满足"真实数据"要求，弃用。
- **阈值与 ε 依据 dev 选定。** 0.60（F）、1.05（G）和判定 ε = 0.02 都依据 X5 dev 数字（种子 900–949）确定。之后在 X5 dev 950–999 上做的 runner 检查不改变它们。
- **一个 v7 绑定输入在 v7 锁之后被改动。** `exp/results/full/v6_summary.md` 在 commit e740f421（论文 v7-r2 修订）中追加了一行带日期的勘误（v6 块 A 中 FDC 对每个矩形各有 2/200 条流更慢；原句保留）。没有任何注册结果的数字被改动。v8 gate 只接受这一处、且哈希与记录一致的漂移（json 字段 `v7_known_input_drift`），任何其他漂移都拒绝。
- **X5 盲化靠流程保证，不是技术上不可能读取。**
  - 原始 csv（含全部结局）作为来源证明保留在 `x5_retailhero/raw/`；
  - `ingest_v8_fresh.py` 解析了整列结局，并把 eval 结局不经任何汇总写入 `eval_outcome.npy`；完整性校验（data_sha256、PROVENANCE）会读取该文件的字节来算哈希；
  - 没有计算、查看或使用任何 eval 结局的统计量，设计、调参和阈值都没有用到；`X5LayerEnv('eval')` 受 v8 gate 保护，并按 PROVENANCE.json 校验文件哈希；
  - 这一点在代码与 git 历史之外无法事后核实。
- **继承的校验范围。** v6/v7/v8 gate 链校验 v5 锁文件本身、v5 冻结代码和 v6/v7/v8 的全部绑定，但不重新校验 v5 锁自己的输入清单（例如 `plan/task_plan.json` 已与其 v5 哈希不同）。v8 的任何计算都不依赖这些文件。
- **external reviewer 第 1 轮审查在锁前改动了设计**（`reviews/v8_review.md`）：
  - PJC 局部族加入"无重置"成员（b = ()），它在 X5 与 CR9 dev 上都胜出，成为冻结的 PJC-local；
  - R2b 改为对运行实际使用的日程求哈希，跨任务 R2b 计入 replica 状态；
  - PROVENANCE.json 与本 md 被绑定；
  - PJC 的相位延迟生效按实现注册（见 §1）；
  - 盲化措辞改为"流程保证"。
  - X5 dev runner 检查（950–999）用最终代码重跑。

## 1. 主方法与对手

- **主方法**：FDC-BF（`make_variant('FDC-BF')`，冻结 50/50 设计，δ = 0.05，与 v7 绑定的 `fdc_bet.py` 哈希相同）。
- **F（矩形，优效检验）**，均在 X5 dev 900–949、ε = 0.02 上按 v5 规则调参（argmin 几何均值 N80_pen，约束 0 条错误流；平局看 x12，再看网格顺序）：
  - RECT-ck-HG：计划网格 {0.3, 0.4, 0.5, 0.6, 0.7, ney, ney23}，选中 **0.5**；
  - RECT-ck-HG-live：无参数（按批准设计不调）；
  - HC-WoR：9 个日程（v6 网格加 x5tune 的目标比例）× 7 个计划，共 63 个配置，选中 **nstar, c = 0.75, target_frac = 0.6, Neyman 计划**。
- **G（相位自适应联合证书，非劣检验）**，同样在 X5 dev 900–949 上调参：
  - PJC-local（17 个配置：16 个带相位重置的配置，加 external reviewer 第 1 轮后补入的无重置成员）选中 **`PJC-BF[local,b=,half]`**。它只有一个相位、按计数确定性跟踪 50/50、用联合 Bennett-FPC 宽度，与 FDC-BF 的区别只在分配规则（确定性跟踪 vs iid 50/50 抽取）。因此 **G-local 不是自适应设计**，G-menu 才是；
  - PJC-menu（10 个配置）选中 `PJC-BF-M[b=2,menu=0.4/0.5,proj]`；
  - 两者的保证与 FDC-BF 同级：在 K 个预设检查点上 FWER ≤ 0.05；
  - **相位延迟生效（按实现注册）**：新相位的计划在边界检查点由截至该点的数据算出，但要到 runner 的下一个重规划批次才生效（最多 replan_interval 个到达之后；X5 eval 上为检查点 3020、重规划 3200）。中间的到达沿用上一相位只依赖计数的偏好，相位内计数只由相位起点之前的历史和与结局无关的后续到达序列决定，保证不受影响；dev 调参用的是同一行为。
- **Neyman / sd^{2/3} 计划矩阵** 由 X5 dev 半边的单元方差冻结，相对 eval 半边不含结局信息。
- 全部网格成员与选择表见 `exp/results/v8_gates/x9_configs.json`。

## 2. 块 A（确认性，唯一判定）

- **数据**：X5 RetailHero eval 半边（100,393 行），X9 分段（性别 × 年龄段，只用特征），问题族 `fr.cr_problems`（15 个预算，512 个策略）。K = 20，n_min = 2000，δ = 0.05。
- **种子**：35000–35199（200 条新流）。ε ∈ {0.015, 0.02, 0.03}，判定在 **ε = 0.02**；stop_k = 15，N80 取自同一轨迹。
- **任务**：
  - `v8a_full_a`：FDC-BF、FDC-MR[front3]、FDC、F、G、RECT-ck-BF、RECT-ck-BF+box，三档 ε；
  - `v8a_full_b`：plug-in（B2-fav、B3-fav、Peace-fav、B5、B2-fav-tight、FIX-bal-fav、Peace-fav-bal）与 v5 文献对手（沿用 v5 CR9 配置，不在 X5 上重调），只跑 ε = 0.02，仅作描述。
- **统计**：主终点 N80_pen，配对几何均值比 FDC-BF/r；配对百分位 bootstrap，B = 10⁴，seed 42，所有对手共用同一重抽样矩阵；报告单侧 UB95。
- **判定（IUT，全部满足才是 `positive_result_achieved`）**：
  1. 对 F 中每个 r，UB95(FDC-BF/r) < **0.60**；
  2. 对 G 中每个 r，UB95(FDC-BF/r) < **1.05**（非劣，界值 5%）；
  3. FDC-BF 在 ε = 0.02 的 200 条流的整个运行内 0 条错误流（同时报告 CP 95% UB）；
  4. `v8a_full_a` 与 `v8a_full_b` 的 replica 都通过。
- **失败分量**：`rectangle_superiority_failed`（子标签 faster_below_1 / not_faster）、`joint_adaptive_rival_not_inferior_failed`、`validity_failure`、`replica_fail`。
- **措辞约束**：
  - 通过只支持"比矩形快"和"不比 PJC 慢 5% 以上"，**不支持**"比 PJC 快"（G 的优效 UB < 1 只作描述）；
  - 若只有分量 2 失败，头条须改为"联合聚合本身（含其相位自适应形式）是收益来源"。

## 3. 块 B（描述）：CR12 规模

- CR12 eval 半边已被 r5 暴露（FDC 种子 30000–30049），只作描述。
- 种子 35200–35399，ε = 0.001，12/15 停止。
- 方法：FDC-BF、FDC-MR[front3]、FDC、RECT-ck-HG、RECT-ck-HG-live、HC-WoR（CR9 配置）、RECT-ck-BF+box、PJC-local/menu（CR9 dev 选出）。
- 另报告每个检查点的证书耗时（4,096 个策略）。

## 4. 块 C（描述，事后）：v7 块 A 的 CR9 流

- 种子 33000–33199，ε = 0.001，stop_k = 15。
- 新跑：PJC-local/menu（CR9 配置）、RECT-ck-BF(+box)、RECT-ck-HG[ney]、HC-WoR[ney]（CR9 dev 冻结的 Neyman 计划）、七个 plug-in，以及 FDC-BF 重跑。
- 与 git 封存的 v7 块 A 行配对（读取前按 v7 seal 校验）。FDC-BF 重跑与 v7 封存行逐字段比对，作为复现检查。
- PJC-local 在 CR9 dev 上同样选中无重置成员。CR9 dev 上矩形与 HC 的自选计划都选中 50/50，与 v7 行相同；Neyman 变体只作描述。
- 明确标注"事后、条件于已用三次的表"，不进任何判定。

## 5. replica 与完整性

- **R1**：独立进程重跑每个 eval 任务的前 10 个种子，规范行（除计时字段）逐位相同。
- **R2**：每个已注册设计组（冻结 50/50 组；自适应 PJC 组）在每个 (seed, ε) 上 schedule_digest 一致。
- **R2b（v8 新增）**：任务内所有方法的 arrival_digest（到达段序列 + 各池记录顺序）在每个 (seed, ε) 上一致；分析时还跨同一块的任务检查。PJC 与 FDC-BF 的配对正是靠这一点。
- 封存、锁的单次提交锚定与 v7 相同（`v8_seal`、`prereg_v8.lock_history_check`）。
- 代码、输入和数据哈希：数据绑定 Criteo、Lenta 与 X5 的三个文件；X5 文件另按 PROVENANCE.json 校验。

## 6. dev runner 检查（不是证据；X5 dev 950–999，最终代码与冻结配置）

- ε = 0.02 时：
  - FDC-BF/RECT-ck-HG = 0.336（UB 0.347），FDC-BF/HC-WoR* = 0.329（UB 0.342）；
  - FDC-BF/PJC-local*（无重置）= 0.988（UB 1.004），FDC-BF/PJC-menu* = 0.929（UB 0.952）；
  - FDC-BF/FDC-MR[front3] = 0.828；
  - 0/50 条错误流。若在这些 dev 种子上套用判定函数，结果为 positive。
- 其他 ε（只作描述，判定 ε 不变）：对 PJC-local 的 UB 在 ε = 0.015 为 1.042、ε = 0.03 为 1.025。非劣余量在 dev 上只有约 0.05，分量 2 是本判定中最可能失败的部分。
- plug-in B5（渐近方法）在 dev 上比 FDC-BF 快约 3.8 倍；dev 上 0 条错误流。论文须写明"plug-in 更快但无保证"。
- R1 / R2 / R2b（含跨任务 R2b）在四个 dev 任务上全部通过。
