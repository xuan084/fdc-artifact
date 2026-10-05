# Hillstrom 暴露审计（v9 块 C 准备，2026-10-04）

**结论：Hillstrom 只能作描述性证据，不具备确认性资格。** 在项目自己的标准下（见 §3），表中没有任何一行、任何一个二元结局、任何一个半边可以算作"未触碰"。块 C 应当注册为**"预注册的描述性复用"**：设计、调参和分析规则在 dev 半边上冻结，种子全新，复用已暴露日志的事实写明，**不下任何判定**。这与 v6 块 B（Lenta）的处理完全相同，暴露机制也相同。

数据：`$DATA_DIR/hillstrom/tidy.pkl`（64,000 行，sha256 `3dab9ed7…`）。列包括 `binary`（visit）、`conversion`、`outcome`（spend），以及协变量 recency、history、history_segment、mens、womens、newbie、channel、zip。

## 1. 已计算、打印或保存的结局统计量

以下逐项来自代码和结果文件，不是凭记忆。

| # | 来源（代码 → 产出） | 行范围 | 分段 | 结局 | 保存的统计量 |
|---|---|---|---|---|---|
| E1 | `exp/code/run_r4_setup_pool_replay.py` → `exp/results/pilots/r4_setup_pool_replay/cell_tables.csv` | **全表**（`half="full"`，64,000 行）与 r4 dev 半边（split 种子 4242，按 S16 × 臂分层） | HR6 = hs3 × newbie；HR16 = hs3 × newbie × 购买类别（mens-only / womens-only / both） | visit、conversion、spend 三个**全部** | 每个（段, 臂）单元的均值和样本量：HR6 18 个单元，HR16 48 个单元，dev 与 full 各一份 |
| E2 | 同一脚本 → `trivial_policy_regret.csv`、`summary.json` | 全表与 r4 dev 半边 | HR6、HR16 | visit、conversion | 15 个问题（成本档 C1/C2/C3 × 预算 0.2…1.0）的 J*、π*、ε 答案集大小、平凡策略 regret、各 ε 下的非平凡问题数 |
| E3 | 同一脚本 `reviewer_proxy_widths()` → `summary.json:reviewer_repro` | 全表 | HR6、HR16 | visit | 用全表单元均值算出的 oracle 宽度代理 |
| E4 | `exp/code/run_r4_setup_criteo_replay.py` → `pilots/r4_setup_criteo_replay/{truth_tables,trivial_guard}.csv` | r4 dev 半边与 r4 **eval** 半边 | HR8 = hs3 × 类别 | visit、conversion | 真值表（J*、π*、ε 集合）与平凡策略守卫 |
| E5 | `exp/code/run_r4_g_eps_select.py` → `pilots/r4_g_eps_select/guard_tables.json` | r4 dev 与 r4 eval 半边 | HR8 | visit | 守卫表（ε 网格 0.01–0.0175 下的非平凡数） |
| E6 | `run_r4_g_eps_select.py`、`run_r4_g1_shape_a.py`、`run_r4_setup_baselines_a.py` → 各自的 `results.jsonl` | r4 dev 半边（种子 880–883、900–999） | HR8 | visit | QFC、B1、B4、B2、A-Ney、A-XY 的认证轨迹；A-Ney / A-XY 的分配由 dev 单元方差冻结 |
| E7 | `plan/feasibility_r4/{feas_diag,coarse_diag}.py` → `hr_coarse_visit.jsonl`、`feas_HR6_visit_{dev,full}.json`、`hr16_*.log` | r4 dev 半边与互补半边 | **cat、cat × newbie（= v9 的 CN6）**、hs3 × cat、hs3 × newbie | visit（另有 HR16 conversion dev 的日志） | 非平凡数、最大平凡 regret、oracle 宽度下 12/15 的到达点 |
| E8 | `exp/code/dsswm/theory_checks/mc_l1.py::certificate_problem_hillstrom` → `pilots/r4_theory_qfc/*` | **全表** | HR6 | visit | 每个单元中 1 的个数（等价于全表单元均值），用于 MC 覆盖检验 |
| E9 | `idea/r4_synth_shape_check.py` | **全表** | S6、S16 | visit、conversion | 分组均值与宽度代理（打印） |
| E10 | `idea/history/round4_r3_debate/perspectives/innovator_sims/sim.py`、`sim2.py` | **全表** | hs3 × newbie × cat | visit | 单元均值，用于模拟 |
| E11 | 同目录 `sim3.py` | 随机一半（`rng(0)`）拟合，另一半计算分箱均值 | **所有协变量**（recency、history、mens、womens、zip、newbie、channel）上的 T-learner 打分，分 K 箱 | visit，仅 Mens 臂 vs 对照 | 在 eval 那一半上打印 uplift 分箱均值 τ_k 和单元样本量 |
| E12 | `dev_scripts/q.py`、`q4.py`、`q5.py`、`r4_contrarian_devhalf_sim.py` | 随机 32,000 行（两种不同的种子） | hs3 × newbie × cat 及其粗化 | visit、conversion | 单元均值，用于模拟 |

v9 之前没有执行过任何 Hillstrom eval 流。r4 锁里写有 `r4_hr8_secondary`（HR8 eval 种子 32000–32099），但 `exp/results/full/` 下没有它的产出；r4 门决定后未运行（`r4_gate_decision.json`："HR8 网格上 QFC 未删失比例全部为 0，无 ε*"）。

## 2. 有没有哪一部分能算作未触碰

逐个检查可能的论据，结论都是否定的。

- **哈希切分出来的某一半？不能。** E1、E3、E8、E9、E10 用的是**全表**，任何切分（包括 v9 的哈希切分）下的 eval 行都在其中。而且 r4 的 eval 半边已经被 E4、E5 读过真值。
- **另一个结局？不能。** E1 保存了 visit、conversion、spend 三个结局在 HR6 和 HR16 上的全表单元均值。数据集里没有第四个结局。
- **另一种分段？不能。** 购买类别 cat3（mens-only / womens-only / both）的全表 cat3 × 臂均值，可以由 E1 的 HR16 全表单元按权重相加精确得到，cat × newbie（CN6）在 E7 里被直接算过。v9 的两个候选层，设计 A 的 CR6（cat3 × recency ≤5 / ≥6）和设计 B 的 CZ6（cat3 × urban / 非 urban），作为单元表都从未被计算过，但有三点使它们也站不住：
  - 它对 recency 求和后的边缘（cat3 × 臂）已经暴露；
  - recency 和 zip 作为协变量都进过 E11 的 T-learner：该模型拟合 visit，并在另一半上打印了按 uplift 分箱的均值；
  - 臂的总体均值（全表）也已暴露。

  channel 的情况相同：它没有被做成单元表，但进过 E11。
- **只用协变量、不用结局？** 只有切分和分段本身满足这一点，这对确认性资格不够。确认性要求的是：在冻结设计、阈值和对手配置之前，评价总体中任何结局都未被看过。

## 3. 判定口径

本项目已有的判例：v6 附录块 B（Lenta，`plan/prereg_lock_v6_addendum.json`）写明："r4 计算过包含 LR9 eval 半边每一行的全表单元均值；新的分段和切分种子不能恢复一个未触碰的留出集"，因此块 B 被定为描述性复用。Hillstrom 的暴露机制与之相同（同一个脚本 E1、同一个"全表"半边），而且更重：三个结局全部暴露，并且有 E11 那种全协变量的模型探索。v6 当时排除 Hillstrom 的理由也写在附录里："A = 3（FDC 只定义了 A = 2），且其 eval 半边汇总被 r4 的 ε 守卫读过"。

**判定：只能作描述（descriptive-only）。** 下面给出可以使用的口径。

- **可以写的（预注册描述性复用）：**
  - 层的设计（分段、成本族、ε、问题族）只用 v9 哈希 dev 半边定下，选择规则与修订记录见 `exp/code/run_hillstrom_v9_dev.py` 的文档串和 `exp/results/pilots/hillstrom_v9/design*.json`；
  - 对手只在 dev 种子 900–949 上调参，dev 报告种子为 950–999；
  - eval 半边（31,881 行）在 v9 锁之前不经任何评价接口读取：`HillstromV9Env('eval')` 需要 `dsswm.stats.prereg_v9.addendum_gate`，该模块在锁前不存在，单测验证会抛 PermissionError；
  - eval 种子 37400–37599 全新；
  - 分析函数与"描述性阈值"在锁前写死，只用于措辞，不构成检验。
- **不能写的：** 不能称"未触碰的留出集""确认性""第三个独立验证"，不能把它计入任何 IUT 或 Holm 族，也不能与 Criteo / X5 合并。
- **建议的论文措辞：** "Hillstrom (A = 3) is an outcome-exposed public log (its full-table cell means were computed during round-4 setup); we report it as a pre-registered descriptive replication with a dev-frozen design, fresh streams and no verdict."

## 4. 若要恢复确认性资格

在这张表上做不到，这是表本身的属性。要拿到第三个确认性真实日志，只能换一张**从未读过结局**的公开 RCT 表，并沿用 X5 的盲化入库流程（`ingest_v8_fresh.py`：只解析标签，eval 结局不经任何汇总直接落盘）。本地 `shared/datasets/` 下的 criteo_uplift、hillstrom、lenta、aact_clinical_subset 全部已经暴露或不合适：本地 criteo_uplift 是合成替身，aact 只有 1.2k 行。
