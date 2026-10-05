# 第 5 轮决策日志与 deviation 披露（r5_decision_log，交付物 6）

> 机器可读版：`plan/decision_log_r5.json`（含每个来源文件的 sha256 与全部核对记录），供 lock v5 引用。核对脚本：`exp/code/run_r5_decision_log.py`，核对明细：`exp/results/pilots/r5_decision_log/checks.jsonl`。本文件只读来源文件写成，未运行任何实验。所有时间为 +10:00 本地时间。

本日志要证明一件事：第 5 轮从 r4 失败走到 FDC 规格的每一个选择，哪些是事前的、哪些是看过开发数据之后作出的，并且每一条都能回到具体文件和字段。结论先写在这里：**DL-01 至 DL-08 全部是事后决策或探索，没有一条进入确认性证据；确认性证据只能来自 lock v5 之后的评价种子 30000+。**

## 时间线（可核实锚点）

| 时间（10-03） | 事件 | 来源 |
|---|---|---|
| 02:12:14 | r4 门判定写出：G1-shape 未过，真实层判死 | `exp/results/r4_gate_decision.json` 字段 written_at |
| 02:25:18 | r4 lock v4 与冻结代码提交（6297e7f2） | git log |
| 02:30:19 | idea_debate 阶段开始 | `logs/events.jsonl` |
| 02:33:51 | pragmatist 写出 QFC-half / QFC-ney 离线脚本 | `idea/r5_offline/pragmatist/offline_qfc_alloc.py` mtime |
| 02:35:45 | contrarian run2（种子 900–939）写出 | `idea/r5_offline/contrarian/run2.json` mtime |
| 02:36:16 | offline_ctr.py 最后修改（含 δ 拆分 0.045/0.005） | `idea/r5_offline/contrarian/offline_ctr.py:29` |
| 02:37:51 / 02:38:33 | contrarian run3（种子 900–999）与 ci_out | `idea/r5_offline/contrarian/ci_out.txt` |
| 03:28:13 | novelty v8 写出，仍按旧 H1（对 B2-fav / Peace-fav）评估 | `idea/novelty_report.md:56` |
| 03:28–12:56 之间 | **用户裁决：主张收窄**（精确时刻未记录） | `.pipeline/project/MEMORY.md:43` |
| 12:56–12:57 | 收敛编辑前的提案快照（含旧 H1 与第 13 节上报） | `idea/history/convergence/proposal_20261003-1256_pre.md:117` |
| 13:09:26 | idea_debate 结束；裁决首次进入 git（47a13eef） | git log |
| 13:16:23 | methodology 第 5 轮版写出 | `plan/methodology.md` |

这条时间线说明，用户裁决同时晚于 r4 开发块结果和 r5 全部探索性开发比较，所以下面每一条都要按事后决策披露。

## DL-01（a）用户裁决：头条收窄为"列明严格对手集中最快"

- **内容**：头条从"对发表配置 plug-in（B2-fav / Peace-fav）优效"改为"在列明并核验过的严格有效对手集 𝓡 中最快"；B2-fav、Peace-fav、B3-fav、B5 maq、H8 降为描述性对照，必须如实报告"plug-in 更快但无保证"。来源：`.pipeline/project/MEMORY.md:43`、`.pipeline/project/prompt_overlays/planner.md:3`、`plan/methodology.md:3`。
- **时间**：2026-10-03，下界 03:28:13（novelty v8 仍在评估旧 H1），上界 12:56:34（MEMORY.md mtime）。git 证据：该节在 ee02cd9f（02:30:19）中不存在，在 47a13eef（13:09:26）中存在。精确分钟没有被任何工作区文件记录，lock 中只能写这个区间。
- **理由**：综合者在提案第 13 节上报（`idea/history/convergence/proposal_20261003-1256_pre.md:117`）：如果"公平基线"包括我们替对手移植平衡读表与紧并集后的 plug-in 变体，则没有候选能达到显著改进。用户裁决头条只针对有保证的方法，plug-in 的速度如实报告。
- **裁决晚于以下开发证据**（必须在 lock 与论文中写明）：
  - r4 开发块 QFC 对各对手的配对几何均值 N80 比：B2-fav 1.669、Peace-fav 0.987、B2-rect 0.769、Peace-rect 0.753、B3-rect 0.553、B1 0.582、B4 0.387、B2-nominal 0.499（`exp/results/r4_gate_decision.json` 字段 variant_ratios_QFC_over_X）；
  - r5 探索性 CI，包括 FIX0.40-rig-b14.1 对 B2-fav 的 0.856 [0.827, 0.885]（`idea/r5_offline/contrarian/ci_out.txt`）。
- **分类**：事后 deviation，不进入确认性证据。

这一条是所有其他 deviation 的源头，所以论文的 deviation 段落必须从它写起。

## DL-02（b）份额、β、K 与变体的全部探索

下表列出 r5 期间在开发半、开发种子上跑过的全部离线变体。它们**全部是探索性的，不进入确认性证据**。几何均值 N80 已从原始流记录重算核对。

| 文件 | 种子 | 变体 | 几何均值 N80 | 错误流 |
|---|---|---|---|---|
| `idea/r5_offline/contrarian/run2.json` | 900–939（40 流） | B2-fav | 1,132,450 | 0 |
| 同上 | 同上 | FIX0.30-rig / 0.40-rig / 0.50-rig / 0.60-rig（默认严格 β） | 1,117,823 / 950,162 / 956,358 / 987,952 | 0 |
| 同上 | 同上 | FIX0.40-rig-b10 / FIX0.40-rig-b8 | 682,062 / 529,329 | 0 |
| 同上 | 同上 | FIX0.40-fav-b10 / FIX0.50-fav-tight | 572,272 / 856,307 | 0 |
| `idea/r5_offline/contrarian/run3.json` | 900–999（100 流） | B2-fav | 1,120,733 | 0 |
| 同上 | 同上 | B2fav-b14.1 / B2fav-b10 | 864,135 / 592,716 | 各 1（种子 973 / 934） |
| 同上 | 同上 | FIX0.40-rig-b11 / b12 / b14.1 | 747,043 / 831,081 / 958,848 | 0 |
| 同上 | 同上 | FIX0.45-rig-b12 | 809,751 | 0 |

对 B2-fav 的配对比（`idea/r5_offline/contrarian/ci_out.txt`，已用 ci.py 的同一 bootstrap 从 run3 原始记录逐位复现）：

| 变体 | 比值 | 95% CI | 更快流占比 |
|---|---|---|---|
| B2fav-b14.1 | 0.771 | [0.730, 0.821] | 0.80 |
| B2fav-b10 | 0.529 | [0.489, 0.575] | 0.97 |
| FIX0.40-rig-b12 | 0.742 | [0.711, 0.771] | 0.80 |
| FIX0.40-rig-b11 | 0.667 | [0.638, 0.697] | 0.92 |
| FIX0.40-rig-b14.1 | 0.856 | [0.827, 0.885] | 0.51 |
| FIX0.45-rig-b12 | 0.723 | [0.695, 0.749] | 0.89 |

pragmatist 的探索：

- K = 20、种子 900–939（40 流）：QFC-half 对 B2-fav 0.8896，对 B2-rect 0.4185，对 r4 QFC 0.5323，对 Peace-fav 0.5393（`idea/r5_offline/pragmatist/offline_qfc_alloc.json`）。oracle Neyman 分配（QFC-ney，**用了真值**）中位 N80 = 6,989,793 = τ_R，对 B2-fav 6.172，这就是 T0 要排查的异常。
- K = 60、种子 900–929（30 流）：QFC-half 对 B2-fav 0.8869 [0.8505, 0.9222]，对 B2-rect 0.4451，对 Peace-fav 0.5706（`idea/r5_offline/pragmatist/offline_k60.json`）。

**没有原始记录的数字**：contrarian 观点表中的"POOL + plug-in GLR，β = 18.5 → 1.74"与"冻结 50/50 + plug-in GLR，β = 18.5 → 0.97"只出现在文本里（`idea/history/round5_debate/perspectives/contrarian.md:8`、`idea/history/round5_debate/perspectives/contrarian.md:9`），`idea/r5_offline/contrarian/` 下没有对应的流级文件。T1 对账必须重新生成它，不能引用文本数字。另外，methodology T1 引用的"0.887"对应 K = 60、30 流的版本（`plan/methodology.md:134`），K = 20 版本是 0.8896，两者不可混引。

探索覆盖了份额 {0.30, 0.40, 0.45, 0.50, 0.60}、β ∈ {8, 10, 11, 12, 14.1, 默认}、K ∈ {20, 60} 和 rig / fav / tight 三种口径，比任务描述列出的范围更宽，所以 forking path 的披露范围以本表为准。

## DL-03（c）优效对象与效应量门槛的变更

- **优效对象**：旧 H1 = FDC 对 B2-fav 与 Peace-fav（发表配置）（`idea/history/convergence/proposal_20261003-1256_pre.md:46`）→ 现 C1 = 对严格集 𝓡 中每一个对手（`plan/methodology.md:13`）。𝓡 = {B1, B4, B4-bal, B2-rect, B3-rect, Peace-rect, Hait-SW, Molitor-WoR, QFC-pool}，其中 B4-bal、Hait-SW、Molitor-WoR 为 r5 新增。
- **效应量门槛**：r4 G1-shape 与 r5 innovator 提案用比值 ≤ 0.67（`plan/history/r4_final/pilot_plan.json:15`、`idea/history/round5_debate/perspectives/innovator.md:18`）→ 裁决前提案用"上界 < 1.0 且点估计 ≤ 0.90" → 现规则：每个对手单侧 95% 上界 < 1.0（IUT），点估计 ≤ 0.90 只决定能否写"faster by ≥10%"（`plan/methodology.md:118`）。
- **理由**：0.67 在严格口径下已被 r5 理论分析判为不可达，而 0.90 作为硬门在提案中悬而未决。两次变更都发生在看过 DL-01 所列开发数据之后。

门槛从 0.67 放宽到 1.0 是本轮最需要审稿人看到的一处放宽，所以论文必须同时给出 0.90 措辞门槛的结果。

## DL-04（d）C3 由门降为估计量（E-cost）

- **之前**：裁决前 H2 是共同主终点，非劣上界 ≤ 1.15（`idea/history/convergence/proposal_20261003-1256_pre.md:47`）；提案 C3 与 T2 仍写 ≤ 1.15（`idea/proposal.md:17`、`idea/proposal.md:86`）。
- **之后**：只报告配对比与双侧 95% CI，不设通过线（`plan/methodology.md:16`）。
- **理由**：裁决把 plug-in 定为描述性对照，若保留通过线，就等于把 plug-in 比较重新写成主张。开发上 FDC 代理（FIX0.40-rig-b14.1）对 B2-tight（B2fav-b14.1）的配对几何均值为 1.110（run3 重算，无 CI），1.15 线本来就很勉强，这一点同样披露。
- **文字更正**：`plan/methodology.md:9` 写"原因见 §6.3"，但 §6.3 是 C4 析因；真正的原因是用户裁决（同文件第 18 行）。

这一降级不改变任何数据，但它让论文不再承诺一个开发上已接近失败的门槛，所以必须写成 deviation 而不是事前设计。

## DL-05（e）r4_g_eps_select 对评价半的接触

r4 在 2026-10-02 23:22:39 选 ε 时接触过评价半。按三类分别披露：

| 接触类别 | 是否接触 | 具体内容 | 来源 |
|---|---|---|---|
| 结果记录（逐行 outcome） | **是** | CR9 与 HR8 的评价半被整体载入（`PoolReplayEnv(layer, "eval")`），用来计算每格真均值；没有回放任何评价半的流 | `exp/code/run_r4_g_eps_select.py:285`、`exp/code/run_r4_g_eps_select.py:298` |
| 汇总真值 | **是** | `exp/results/pilots/r4_g_eps_select/guard_tables.json` 的 CR9 评价半 15 行，含 J_star、second_gap、eps_set_size、min_trivial_regret；非平凡题计数 n_nontrivial_eval 进入资格规则 guard_ok。CR9 在 ε ∈ {0.001, 0.0015, 0.002, 0.003} 上都是 15/15；HR8 在 {0.01, 0.0125, 0.015, 0.0175} 上为 10 / 5 / 5 / 5 | `exp/results/r4_gates/eps.json:17`、`exp/results/pilots/r4_g_eps_select/summary.json` 字段 guard_counts、`exp/code/run_r4_g_eps_select.py:229` |
| 随机种子（置换） | **否** | 只用 900–999；integrity.eval_seeds_touched = false | `exp/results/r4_gates/eps.json` |

在 CR9 上这个守卫没有起约束作用（四个 ε 都是 15/15），所以 ε* = 0.001 在去掉守卫后不变；但"评价数据从未参与设计"这句话不能写。论文只能写：评价种子与评价半的流级结果从未用于任何方法或门的比较，评价半的汇总真值曾用于 ε 守卫（与 `plan/methodology.md:26` 一致）。

## DL-06（f）C_var 48 → 18，δ 拆分 0.04/0.01 → 0.045/0.005

- **r4 账本**：δ_main = 0.04，δ_var = 0.01，C_var = 48，L1 = 15.1610，x_v = 12.1653（`exp/code/dsswm/baselines/frontier_common.py:204`；r4 summary 字段 layer_info.CR9.params）。C_var = 48 是为 S = 16 预留的保守值（`plan/theory/qfc_lemma.md:68`）。
- **FDC 账本**：Σ_q|Π_{B_q}| = 3386（从 CR9 开发 ctx 只读重算核对），β = ln(3386·20/0.045) = 14.2242；C_var = S·A = 18，x_v = ln(720/0.005) = 11.8776。并集对象从 Q·|Π| = 7680 收紧为 3386。硬编码 β = 14.1 只能给出 3386·20·e^{−14.1} + 0.005 = 0.05595 > δ。
- **δ 拆分首次出现的证据与限制**（需要更正 methodology 的措辞）：
  - 全工作区最早出现 0.045/0.005 账本的位置是 `idea/r5_offline/contrarian/offline_ctr.py:29`（δ_main）与第 30 行（δ_var）。external reviewer 引用的 offline_ctr.py 第 26 行实际是 setup() 函数头。更早文件里的"0.045"都是 CP 上界或 σ²，与账本无关。
  - 该文件 mtime 为 02:36:16，**比 run2.json 晚 31 秒**，比 run3.json 早；它第一次进入 git 是 47a13eef（13:09:26）。所以时间戳和提交都无法证明拆分早于 run2。
  - 旁证：run2 中用默认 β 的 FIX0.40-rig 在 40/40 条流上 N80 ≥ run3 中 β = 14.1 的同名变体（1 条严格更大），与运行时默认 β > 14.1 一致，但这不是证明。没有任何一次运行改变过 δ 拆分，所以不存在对拆分的搜索。
  - **更正**：`plan/methodology.md:44` 的"在开发结果出来之前就已写进提案和离线代码的默认值"只能按以下口径成立：晚于 r4 开发结果（02:12:14），与 r5 首批探索运行处于同一时段，早于 r5 的全部门（T0–T3）。lock v5 应按这个口径写。

C_var 改为 18 是精确值替换保守值，属于合法收紧；δ 拆分没有被搜索过，但它出现的时间不能被说成早于全部开发数据。

## DL-07（g）QFC-half 的 50/50 是看过 r4 之后选定的

- `idea/r5_offline/pragmatist/offline_qfc_alloc.py:1` 标注"EXPLORATORY (r5 pragmatist)"，写于 02:33:51，比 r4 门判定晚 21 分钟；它直接读取 r4_g1_shape 开发流结果作参照（同文件第 36 行）。
- 50/50 的动机来自 r4 失败机制："慢在池比例读法饿死对照池"（`idea/proposal.md:47`）。
- 开发上 0.40 略快于 0.50（run2 几何均值 950,162 对 956,358），FDC 仍锁 0.50，不调份额。

选择 50/50 是看过 r4 后的研究者自由度；它的代价是如实承受 0.50 可能慢于 0.40，而不是在评价前再挑份额。

## DL-08（h）两阶段读表与 CS 紧会计模块删除

- **两阶段读表**（前 5% 池比例读、估方差后冻结）原为预注册备选（`idea/history/convergence/proposal_20261003-1256_pre.md:54`、`idea/proposal.md:63`），在 methodology 中删除（`plan/methodology.md:32`）。这去掉了一条 forking path。
- **A = 2 的 Cauchy–Schwarz 9 维 ℓ2 紧会计**原有"10-06 前证不出即砍"规则（`idea/proposal.md:66`），在 10-03 planning 时删除（`plan/methodology.md:61`）。
- **文字更正**：methodology 写"10-06 前没有证明，按提案规则删除"，但删除发生在 10-03，早于截止日，也没有尝试证明。lock 应写成"planning 阶段主动删除"。

两项删除都只减少了可选路径，没有增加任何依赖开发数据的选择，所以它们降低而不是增加 forking 风险。

## DL-09（补充）β = 14.1 结果全部降级

0.856 等结果用的是 contrarian 的逐检查点经验 Bernstein 方差公式和硬编码 β = 14.1，两者都不是 FDC 的证书（`plan/methodology.md:33`、`reviews/idea_debate_review.md:37`）。FDC 用 Lemma L2 反演 UCB 与自动计算的 β；任何参数被覆盖的实例改名并标记 validity = "none"。所以 0.856 不能作为 FDC 的开发证据引用，T2 才是 FDC 对严格对手的第一份直接测量。

## DL-10 门判定 T0–T3（r5_gate_decision，2026-10-03T16:12:42）

这一条要证明：进入 lock v5 前的五道门都按事前冻结的判据过了，评价流数也按事前规则定下。结论是 **verdict = GO**，c4_mode = **predictive**，cr_n_streams = **200**（评价种子 30000-30199，扩展块 30200–30399 不启用）。

| 门 | 结果 | 来源字段 |
|---|---|---|
| T0 | PASS：根因是 oracle Neyman 份额对零方差单元为 0（输入问题，非代码 bug），17 条回归单测通过，不影响 r4 数字 | `exp/results/pilots/r5_t0_alloc_audit/summary.json`: verdict, root_cause, tests_passed |
| T-spec | PASS：自检签字；60 个合成配置 × 500 流 × 3 个 ε 的证书 MC 0 违例（1,350,000 次认证），无效对照能检出违例；§10 第 15–16 项由控制面在 MC 结束后补签，数字未改 | `exp/results/pilots/r5_fdc_spec_theory/summary.json`: gate.*, mc.fdc_false_streams_all_windows_eps, control_plane_note |
| T1 | 交付物完成：0.856 / 0.887 / 0.97 复现为 0.8556 / 0.8869 / 0.9794（均在 ±2% 内）；C4 冻结模型 M1 在 950–999 上 8 格最大 |log 误差| = 0.038 ≤ 0.095 ⇒ c4_mode = predictive。其中 6 格的 95% CI 端点越过 ±0.095，判据按点估计执行，论文须同时报告 CI | `exp/results/pilots/r5_t1_reconcile/summary.json`: reproduction.*, c4_model_validation.max_abs_log_error_primary |
| T2 | PASS：𝓡 的 9 个对手单侧 95% 上界全部 < 1.0，最难的是 QFC-pool（0.5344） | `exp/results/pilots/r5_t2_gate_strict/summary.json`: gate_T2.per_rival.*.ub95_one_sided |
| T3 | PASS：FDC 错误流停 12/15 时 0/100，全视界 0/100（CP 上界 0.0295），已审计 | `exp/results/pilots/r5_t2_gate_plugin_t3/summary.json`: gate_T3.* |

功效按 §5.3 的事前规则计算：对最难对手 QFC-pool 用开发点估计 θ̂ = -0.668、σ̂ = 0.257（log 比），10³ 复制 × 2000 次 bootstrap。N = 200 时 IUT 功效 = 1.000，N = 400 时 = 1.000，所以评价用 200 流。对全部 9 个对手做联合经验重抽样，以及把 θ 放到开发上界、σ 放大 1.5 倍的压力情景，功效也都是 1.000。C2 的 CP 临界值：200 流至多 4 条错误流（CP(4,200) = 0.0452），400 流至多 12 条。

需要在 lock v5 中写明的事项：Hait-SW（p_min = 0.02，γ = 2/3）与 B4-bal（share = 0.5）在 100/100 条开发流上 N80 逐流相同，𝓡 列出 9 个方法但只有 8 个不同的 N80 轮廓；Molitor-WoR 100/100 删失、B3-rect 33/100 删失，它们的比值来自对手删失。功效接近 1 说明 C1 的主要风险是开发到评价的分布漂移，而不是抽样噪声。以上全部是开发种子结果，确认性证据仍只来自 lock v5 之后的评价种子 30000+。详表：`exp/results/pilots/r5_gate_decision/gate_table.md`。

## 贡献说明

### 中文

r4 的 QFC 已经使用 Q\*-并集（L1 = S ln A + ln(QK/δ_main) = 15.161，`exp/code/dsswm/certify/quadknap.py:21`；external reviewer P0-4，`reviews/idea_debate_review.md:102`）。所以 FDC 在会计上相对 r4 新增的只有两处：并集对象从 Q·|Π| = 7680 收紧为 Σ_q|Π_{B_q}| = 3386，以及 δ 拆分与 C_var 的调整。"对固定最优策略的备选取并集"是方法描述，先例为 Garivier & Kaufmann (2016)、Peace（Katz-Samuels et al. 2020）与 de Heide 2608.19903；"设计优先"是 Hait 2609.37873 框架在组合预算前沿、不放回回放上的实例化；冻结平衡读表的先例是 Neyman 分配、Kato 2405.19317 与 Dang 2608.06512（TWNA）。本工作唯一主张的可测能力是：**冻结、与结果无关的读表使方向级 Bernstein 证书在不放回 + 检查点停止下合法**，这一能力由 C4 中 FDC 对 B4-bal（设计相同、矩形证书）的配对比，以及 CR12（|Π| = 4096）规模层上差距的变化来度量。

### English (for the paper's deviation paragraph)

The headline claim of this paper was narrowed after we had seen development data. On 2026-10-03, after the round-4 development block (in which our prior certificate QFC was 1.67× slower than the plug-in CombGame rule B2-fav but 0.39–0.77× the stopping time of every rigorous rival) and after exploratory development runs of FDC-like variants, the claim "faster than published plug-in configurations" was replaced by "fastest among a listed set of rigorous competitors", and plug-in rules were reclassified as descriptive comparators whose speed and error streams we report. The effect-size gate moved from a ratio of 0.67 to a one-sided 95% upper bound below 1.0 per rival, with a point estimate of at most 0.90 required for the wording "faster by at least 10%", and the non-inferiority bound of 1.15 against tuned plug-in variants became an estimate without a pass line. The 50/50 read schedule, the per-cell variance budget (18 cells instead of a conservative 48), and the split δ_main = 0.045 / δ_var = 0.005 were chosen after round 4; the split first appears in exploratory code written during those runs, and no run varied it. All share (0.30–0.60), threshold (β ∈ {8, …, 14.1}) and checkpoint-grid (K ∈ {20, 60}) explorations used development seeds only and are excluded from confirmatory evidence. Round 4's ε-selection read aggregate ground truth on the evaluation half (counts of non-trivial problems) but no evaluation seeds or evaluation streams. Relative to QFC, which already used a union over alternatives to the per-question optimum, FDC's accounting changes are limited to tightening that union from Q·|Π| to Σ_q|Π_{B_q}| and re-splitting δ; the alternative-union itself follows Garivier & Kaufmann (2016), Peace and de Heide et al. (2608.19903), and the design-first view instantiates Hait (2609.37873) for combinatorial budget frontiers under without-replacement replay, with balanced and Neyman-type frozen designs following Kato et al. (2405.19317) and Dang et al. (2608.06512). The capability we claim, and measure via FDC versus a rectangular certificate under the same design (B4-bal) and on the larger CR12 class, is that a frozen, outcome-independent schedule makes a direction-level Bernstein certificate valid under without-replacement sampling with checkpoint stopping.

这段英文按"每段第一句说要证明什么"的收敛规则写成，可直接放进论文的 deviation 段落；数字均与上文和 JSON 中的核对记录一致。
