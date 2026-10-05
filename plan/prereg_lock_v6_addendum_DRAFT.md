# Lock v6 附录（DRAFT rev. 5，未锁定）：第 5 轮补充块 A / B / C

状态：**draft**。rev. 2 于 2026-10-04 完成，已落实 external reviewer 附录审查（`reviews/v6_addendum_review.md`）提出的全部 7 条 REQUIRED，以及可选项 RECT-ck-HG-live。机器可读版本是 `plan/prereg_lock_v6_addendum_DRAFT.json`，由 `exp/code/build_v6_addendum_draft.py` 从磁盘文件生成，所有哈希都是计算得到的，没有手填。

锁定方法：主 session 调用 `dsswm.stats.prereg_v6.finalize_addendum(git_commit)`，生成 `plan/prereg_lock_v6_addendum.json`。这个函数有三条保护：

- 拒绝覆盖已存在的锁；
- 校验该 commit 确实包含所绑定的代码，并且代码和输入的哈希都没有漂移；
- 锁后如需修复，必须另立新版本，不能重锁。

## 0. 与 v5 的关系（不可协商）

- 本附录只追加内容，不改 `plan/prereg_lock.json`（sha 64196b…948），也不改 v5 的 R、Holm 家族、终点和判定。
- **A、B、C 的任何结果都不能改变 v5 判定 `positive_result_achieved`。** 它们只能改变三处措辞：证书归因、终点范围、外部效度。
- **结论范围只限于已注册的对手及其冻结配置。** 具体有三点限制：
  - "keep" 只说明所检验的证据门槛已经达到，不能自动推出"同保证下约 2×"；
  - RECT-ck-HG 是精确的固定 n 检验反演，不是所有合法构造中最紧的；
  - HG、HG-live、HC、Bern 之间同时改变了多个构造因素，因此不能据此做"主要来自保证强度"这类定量归因。

## 1. 块 A：同强度矩形对手

四个新对手都使用冻结的 50/50 设计，与 FDC 的日程摘要逐位相同（dev 上的 R2 检查通过）。

| 名称 | 单元区间 | δ 分配 | 保证 | 调参 |
|---|---|---|---|---|
| RECT-ck-HG | 每个（单元，检查点）做精确超几何反演 | α_side = 0.05/720 | K 个检查点 | 无 |
| RECT-ck-HG-live（新增） | 同上 | 每个检查点 δ/K，只在 0 < n < N 的活单元间平分；总误覆盖 ≤ δ（某个 L_k = 0 时严格小于） | K 个检查点；弱支配 HG | 无 |
| RECT-ck-Bern | L1 Bernstein + L2 方差 UCB | 0.045 / 0.005，与 FDC 相同 | K 个检查点 | 无 |
| HC-WoR | 对冲资本 WoR 下注 CS；**rev. 2 修复了反演**：两个单调单侧区域在完整逻辑区间上分别反演，空交集时退回逻辑区间 | 每单元 δ/18 | 任意时刻 | 6 点网格。用修复后的代码重调，仍选中 `nstar, c = 0.75, t* = 0.2` |

**HC 修复。** 旧代码假设样本均值处总被接受，这个假设不成立。external reviewer 给出的反例下，旧代码给出下界 0.5，正确的根是 0.552007，现在已作为单测。这个错误会让对手变松（对 FDC 有利），不会造成欠覆盖。修复后 CR9 的调参表和 dev 比值与旧代码在 3 位小数上完全一致，可见这个错误在真实流上几乎没有被触发。

**资格检查。** 10 个严格配置（HG、HG-live、Bern、HC 的 6 个网格点、B4-bal 参照）在 toy 上都是 0/200 错误流，CP UB 0.0149。NAIVE-control 出现 4/200 错误流。这只说明 harness 能检出坏证书；它的 CP UB 是 0.045，不能说明 FWER 超过了 0.05。

**设置。** CR9 eval 半，ε = 0.001，K = 20，12/15 停止，种子用新的 31000–31199（200 条流）。

**判定**（`v6_analysis.analyse_block('A')`）：UB* = max_{HG, HG-live, Bern, HC} UB95(FDC/r)。

| UB* | 判定 | 含义 |
|---|---|---|
| < 0.70 | keep | 在已注册对手上达到预设的证据门槛 |
| 0.70 ≤ UB* < 1.0 | downgrade | FDC 比每个已注册对手都快，但未达预设门槛；每个点估计和 CI 照实报告 |
| ≥ 1.0 | withdraw | 不再主张对已注册对手的证书优效 |

replica 失败时**强制判 withdraw**。判定函数会拒绝以下输入：缺失、多余或重复的行，NaN 或非正值，对手集不完整，以及 replica 未完成。

**dev 结果（rev. 2 代码，seeds 950–999，探索性）：**

| 比较 | 比值 | 95% CI | UB |
|---|---|---|---|
| FDC/RECT-ck-HG | 0.804 | [0.763, 0.847] | 0.838 |
| FDC/RECT-ck-HG-live | 0.804 | 同上 | 0.838 |
| FDC/HC-WoR | 0.821 | [0.779, 0.864] | **0.860** |
| FDC/RECT-ck-Bern | 0.403 | — | 0.424 |
| FDC/B4-bal | 0.396 | — | 0.420 |

dev 上的 UB* 是 0.860，由 HC-WoR 决定，判定为 **downgrade**。HG-live 在 CR9 上与 HG 逐流相同，原因是 FDC 和 HG 停止之前没有任何单元耗尽或为空，活单元数一直是 18。

## 2. 块 B：Lenta LR9，描述性分析，属于**已有结局暴露的数据复用**

- **结局暴露。** r4 在**全表**上计算过 LR8 的单元均值，这些均值包含了 LR9 eval 半的每一行。新分段和新拆分种子都不能恢复一个未经接触的 holdout，所以 B 定性为"已暴露日志的描述性复用"，不作确认性检验。
- **访问控制。** eval 结局只有在 `eval_task_id` 通过 `addendum_gate` 后才会暴露。代码中已删除 `allow_eval` 旁路，并有测试保证。
- **加载方式的准确说法。** pandas 会把整个 pickle 读入内存，所以结局列在加载时短暂存在于内存中，但不会被缓存，eval 行的结局也不会被物化。准确的措辞是"锁定前不通过任何评估接口暴露"，而不是"从未加载"。
- **数据校验。** 数据集的 sha256 默认会被校验。
- **ε 与种子。** ε 取 {0.002, 0.003, 0.004}，三档全部报告，不做选择。种子用新的 32000–32199。
- **对手配置。** 在 LR9 dev 上用修复后的代码重调，结果是 B4-bal share 0.4、Hait 0.02 / 2/3、B2 1.0、HC `nstar t* = 0.5`。
  - B4-bal、Hait、B2 的网格点全部打平在 τ_R，选择由网格顺序决定，这一点已披露。
  - 因为 B4-bal 的 share 是 0.4，它**不在** B 的 50/50 比较组里。
- **HC 网格边界敏感性（只报告，不改网格）。** t* = 0.5 / 0.65 / 0.8 / 1.0 时，N80/τ 分别为 0.790 / 0.786 / 0.786 / 0.786，相差不到 0.5%。
- **可行性。** 结论与 rev. 1 相同（rev. 2 的 dev 数据：ε = 0.002 时 FDC/HG = 1.14、FDC/HG-live = 1.22、FDC/HC = 1.04，即同强度矩形比 FDC 更快；ε = 0.003 时为 0.91 / 0.94 / 0.90；ε = 0.004 时为 0.84 / 0.86 / 0.84）。ε = 0.002 时，FDC 和 R 中的所有对手都要到 τ_R 才能完成；ε ≥ 0.003 时 15 个问题全部是平凡的，比较结果由终点堆积决定。
- **replica 失败时**：B 的表格只放附录，并标记 `replica_fail`。
- Hillstrom 排除；不与 Criteo 合并。

## 3. 块 C：N100 补充延续

- **性质。** 这是在部分终点已经公开之后、用事先固定的规则做的 supplementary continuation：200 条流上所有方法的 N80，以及 `r5_cr_fwer_audit` 中 FDC 的 N100_pen 都已经公开。它不是独立的盲态确认，只用来决定终点范围的措辞；如需新的确认证据，必须使用未用过的流。
- **方法集。** 显式绑定 FDC 加 v5 的 9 个对手：B1、B4、B4-bal、B2-rect、B3-rect、Peace-rect、Hait-SW、Molitor-WoR、QFC-pool。runner 会检查这个集合是否完全一致。
- **判定。** 只有当 9 个对手的 UB 全部小于 1、且 replica 通过时，判 `n100_pass`；否则判 `narrow_to_N80`，replica 失败时强制判 `narrow_to_N80`。

## 4. replica 规则与报告完整性（rev. 3）

实现位置：`dsswm/stats/v6_replica.py`，以及 `run_r5s_v6.py --task T --replica` 和 `--analyse`。

**三条规则**

- **R1**：由独立进程重跑每个任务的前 10 个种子（A 为 31000–31009，B 为 32000–32009 且三档 ε 全跑，C 为 30000–30009），覆盖任务中的全部方法。除计时字段 `sec, rs_sec_plan, rs_sec_cert, rs_sec_total` 外，所有行必须完全一致。
- **R2**：每个比较组内，schedule_digest 必须一致；并且每个**预定的（种子，ε）流必须齐全**。各块的比较组：
  - A：FDC、B4-bal、HG、HG-live、Bern、HC；
  - B：FDC、HG、HG-live、Bern、HC；
  - C：FDC、B4-bal。
- **R3**（只适用于 C）：先确定 v5 的停止点 k_stop。r5_cr_main_a 中它在顶层字段 `rs_k_stop`，r5_cr_main_b 中它在嵌套字段 `run_stream.k_stop`，两种 schema 都支持。截断点 k_cut 取 k_stop；若 v5 从未达到 12/15，则取曲线的最后一个索引。字段对应如下：

| v6 字段 | 对应的 v5 字段 / 条件 |
|---|---|
| N80_pen | v5 N80_pen |
| k80 | v5 k_stop |
| n_cert_curve[:k_cut+1] | v5 n_cert_curve[:k_cut+1]（v5 曲线在停止点之后用补齐值填充，比较时忽略这段尾巴） |
| cert_k | v5 已认证的题必须一致；v5 未认证的题在 v6 中为 -1 或 > k_cut |
| FDC 的 N100_pen | r5_cr_fwer_audit 中的 N100_pen |

R3 有基于真实 v5 记录的回归测试，覆盖两种 schema，包括 Peace-rect 的种子 30037、30112、30137，以及 B4-bal 的种子 30150。

**报告完整性**

- **生成前**：先验证当前代码、附录和数据下，任务的**完整预定矩阵**（方法 × ε × 种子，每组恰好一条）是否齐全；不齐全就拒绝生成报告。
- **报告绑定的内容**：task_id、is_eval、addendum_sha256（eval 时不能为空）、逐文件的代码 sha、数据 sha、方法 / ε / 种子 / R1 种子、所需规则集（R1、R2，C 的 eval 任务另加 R3），以及结果行内容的 sha256。
- **失效规则**：只要向任务写入任何新行（追加、替换或续跑），就删除旧报告。
- **分析时的校验（rev. 4）**：分析程序用实际的 replica 行、当前主结果和冻结的 v5 参照（两个矩阵都必须齐全）**重新计算** R1–R3（`compute_checks`），判定只使用重算得到的状态。存档报告只是一份缓存，它的 checks、status、绑定字段（包括 replica 行的内容哈希）必须与重算结果完全一致，否则拒绝分析。因此，只改报告里的 pass/status 字段无法改变任何判定。external reviewer 给出的反例（同时改了 status 和 check、`bad` 仍不为空）已加为回归测试。
- **块状态**：块内每个任务的报告都有效且为 pass 时，块才算 pass；任何一份报告为 fail，块即为 fail。fail 的后果按块不同：A 强制判 withdraw，C 强制判 narrow_to_N80，B 只放附录并加标记。

dev 上的演示：v6a、v6c、v6b 三个 pilot 的 replica 都是 pass，A 和 C 的 dev 分析都通过了报告校验。

## 4b. eval 封存与威胁模型（rev. 5）

**封存流程**

1. 每个 eval 任务跑完时（续跑补齐后也一样），runner 写出 `exp/results/full/v6_seals/<task>.seal.json`。内容包括：全部主结果行的内容哈希（不含计时字段）、行数、方法 / ε / 种子的覆盖情况，以及代码、数据和附录的哈希。
2. 写完**立即单独 git commit** 这个文件。
3. 一个文件无法包含添加它的那次 commit 的哈希，所以 commit sha 另外记在 `<task>.seal_ref.json` 里（随后同样单独提交），同时写进任务 summary。

**校验（`v6_seal.verify_seal`）**

replica 步骤和分析都会调用这个校验，任何一条不满足都拒绝：

- 封存 commit 存在，并且在 HEAD 的历史中；
- 封存文件的当前内容等于该 commit 中的内容；
- 封存中的绑定字段一致；
- **当前主结果的内容哈希等于封存的哈希**。

此外，每个 eval 任务的 `validate_report` 都必须拿到这个封存哈希。R2 在完整主矩阵上运行（全部种子 × ε），R1 在前 10 个种子上运行。external reviewer 指出的反例是：改动 R1 种子之外的一条主结果行，同时重写报告。这种情况已加入回归测试，会被拒绝。

**威胁模型**

完整性检查能发现以下几类问题，并且都可以对照 git 历史检出：

- 意外漂移：代码、输入、原始数据、配置；
- 部分运行或续跑；
- 事后改动结果、replica 报告或封存。

所有 eval 产物在完成后立即提交：封存在任务结束时提交，结果、replica 报告和分析在各步骤之后提交。蓄意改写 git 历史不在保护范围内，这与标准预注册实践一致。

## 5. 绑定与溯源

- **原始数据（rev. 3）**：在 v6 层冻结以下文件的 sha256，具体在 `dsswm/envs/data_v6.py`，v5 的 `pool_replay` 未改动：
  - Criteo `criteo_uplift_real/tidy.pkl`，即回放实际读取的文件；
  - Criteo `criteo-research-uplift-v2.1.csv.gz`；
  - Lenta `tidy.pkl`。
  
  这些哈希写在锁的 `data_sha256` 中，在 eval 启动、每次续跑和分析时都会经 `check_addendum` 校验；runner 在构建任何 env 之前也会校验一次。每层的组合数据哈希写入每条结果行、每份 summary 和每份 replica 报告；续跑时数据哈希不同的行会被丢弃。

- 每一步都会重新检查锁：eval 启动、断点续跑和分析三个环节，都会校验状态、sha256、与 v5 的链接、代码哈希和**输入哈希**（冻结配置、pilot 汇总、v5 参照结果）。
- 配置从锁中的 `frozen_configs` 读取，并与磁盘上的 gate 文件比对；没有默认值回退。
- 每条结果行和每份 summary 都写入 `addendum_sha256`。
- schema 是必填的，`code_sha256` 或 `input_sha256` 为空时拒绝。

## 6. 成本（4 个 worker）

| 任务 | CPU·h | 墙钟 |
|---|---|---|
| v6a_full | 0.9 | ≈ 15 min |
| v6b_full_a（Peace） | 3.3 | ≈ 50 min |
| v6b_full_b | 1.3 | ≈ 20 min |
| v6c_full_a | 2.3 | ≈ 36 min |
| v6c_full_b | 1.3 | ≈ 20 min |
| 各块 replica | 约 10% | — |

## 7. 主 session 审阅要点

1. 块 A 在 dev 上的结果是 downgrade（UB* 0.86），需确认接受；阈值在锁定后不得修改。
2. B 为描述性分析，并承认是已暴露数据的复用。
3. C 是补充延续，不是独立确认。
