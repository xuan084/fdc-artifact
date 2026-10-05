# Lock v8 附录（草案，未锁定）：未触碰新表上的确认性检验 + 相位自适应联合对手 PJC-BF

状态：**DRAFT，不是锁**（2026-10-04，methodologist）。本文件只规划；没有生成 json 锁，没有触碰任何 eval 种子、eval 半边或 X5 eval 结局。v5/v6/v7 的锁和文件一律不改。
依据：`exp/results/pilots/pjc_v8/`（dev 结果，见 §1）、`exp/code/dsswm/baselines/pjc_bf.py`、`dsswm/tests/test_pjc_bf.py`、`ingest_v8_fresh.py`、`dsswm/envs/x5_v8.py`。

## 0. 为什么要 v8（对两份 v7 审稿的回应）

| 审稿要求 | v8 的处理 |
|---|---|
| P0-3 最强的合法有保证对手：相位自适应联合证书 | 新增 PJC-BF（RAGE 式逐相位冻结，剩余池 Bennett-FPC，联合宽度）和 PJC-BF-M（菜单路径并集，使用全部数据）；作为主家族成员之一，按非劣/优效判定（§3） |
| P1-2 未触碰的新表 | X5 RetailHero（真实 SMS 营销 RCT，200,039 行），按哈希行号盲拆分；eval 结局从未被读取（§2） |
| P1-1 机制消融（同单元不等式） | RECT-ck-BF：与 FDC-BF 同一 Bennett-FPC 单元界、同一 HG 方差盒、同一 δ 拆分、同一设计，只换聚合方式 |
| P1-5 矩形用自己的读取计划 | 矩形和 HC-WoR 在 dev 上调 control share {0.3,…,0.7} 与 Neyman / sd^{2/3} 逐单元计划 |
| P1-3 规模 | CR12（4,096 个策略）描述性块 |
| P1-9 plug-in 配对 | 在块 C 的同一流上配对运行 plug-in（仅描述） |

## 1. dev 证据（不是证据，只用于设计；全部 dev 种子 900–999）

### 1.1 数据可得性

- **Criteo 没有未触碰的部分。** 本地 `criteo_uplift`（100k）是合成替身，不是真实表的子样本；真实全表（13,979,592 行）就是 `criteo_uplift_real`，sha256 与 HF 镜像的 LFS oid 一致（`2716e1bf…`）；原 URL 已 404。CR9/CR12 的 split 把每一行都分进 dev 或 eval，eval 的 visit 已被 v5/v6/v7 和 r5 CR12 使用。r4 setup 还写出了 CR9/CR12 eval 半边的 conversion 单元均值。conversion 在 ε ≥ 0.001 下也没有非平凡问题。审计记录在 `shared/datasets/criteo_full/PROVENANCE.json`。
- **X5 RetailHero（选为确认性新表）。** 真实数据，来自 HF `pytorch-lifestream/retailhero-uplift`，`uplift_train` 与 `clients` 合并。
  - 盲拆分：键为 sha256("dsswm-v8-2026-10-04|x5_retailhero|client_id")，dev 99,646 行，eval 100,393 行；
  - eval 结局只在 `eval_outcome.npy` 里（只读，sha256 已记录），没有计算任何统计量；
  - dev 的处理组比例为 0.500，两组结局率为 0.604 / 0.637。
- **MegaFon。** 600k 行，已同样盲拆分，但 scikit-uplift 文档说明它是生成的合成数据，**不满足"真实数据"要求**，最多作为压力测试。
- **其他可得数据。**
  - Open Bandit Dataset（ZOZOTOWN，均匀随机 logging，413 MB，可下载）：80 个臂，需要另做 A = 2 的问题构造，留作将来；
  - Upworthy Research Archive（OSF 可访问；自带 exploratory / confirmatory 划分的标题 A/B 测试）：数据是聚合计数，需要另建环境；
  - Starbucks（Udacity）：无可靠公开镜像；
  - X5 S3 镜像已 403，用 HF 镜像。

### 1.2 PJC-BF 与其他对手（FDC-BF/r 的 N80_pen 配对几何均值比，括号为 UB95）

PJC 的两个调好配置在 CR9 tune 种子 900–949 上按 v5 规则选出：局部族选 `PJC-BF[local,b=0,half]`，菜单族选 `PJC-BF-M[b=2,menu=0.4/0.5,proj]`。全部 26 个网格成员都报告在 `analysis_dev.json`。

| 块（dev） | vs PJC-local* | vs PJC-menu* | vs RECT-ck-HG | vs HC-WoR | vs RECT-ck-BF(+box) | vs FDC-MR[front3] |
|---|---|---|---|---|---|---|
| CR9 950–999，ε = 0.001 | 0.964 (0.990) | 0.974 (0.990) | 0.649 (0.677) | 0.663 (0.695) | 0.525 / 0.539 | 1.042（MR 更快） |
| CR12 950–969，ε = 0.001 | 0.974 (1.000) | 0.949 (0.987) | 0.483 (0.509) | 0.496 (0.522) | 0.441 | 0.867 |
| LR9 dev 950–999，ε = 0.002/0.003/0.004 | 0.996 / 0.982 / 0.987 (≤ 1.009) | 0.991 / 0.961 / 0.969 (≤ 1.000) | 0.772 / 0.679 / 0.670 | 0.703 / 0.670 / 0.667 | 0.682 / 0.664 / 0.571 | 0.86–0.91 |
| X5 dev 930–949，ε = 0.02 | 0.940 (0.970) | 0.950 (0.980) | 0.333 (0.354) | 0.329 (0.350)† | 0.303 | 0.814 |
| X5 dev 930–949，ε = 0.015 / 0.03 | 0.960 / 0.893 | 0.970 / 0.950 | 0.458 / 0.239 | 0.453 / 0.244† | 0.453 / 0.205 | 0.85 / 0.80 |

† HC-WoR 在 X5 dev 900–929 上重新调参（nstar，c = 0.75，target_frac 0.6）。

- **结论 1：FDC-BF 不输给相位自适应联合对手，但只快 0–6%。** 在所有 dev 块上，FDC-BF/PJC* 的点估计都在 0.89–1.00。
  - 最差的 UB 是 X5 ε = 0.01（20 个种子）的 1.042；
  - 原因：外生到达下唯一可调的是段内拆分，而方差最优的 Neyman 拆分本来就接近 50/50（CR9 dev 上 control share 为 0.41–0.50）；
  - 相位局部估计丢弃早期样本，菜单并集要付 β 代价（3 个菜单项时约 +1.0）；
  - 因此 v8 对 PJC 只能注册**非劣**，不能注册优效。
- **结论 2：联合聚合本身（同单元不等式）约省 1.9 倍读取。** CR9 上 FDC-BF/RECT-ck-BF = 0.525（UB 0.556）。它与 FDC-BF/RECT-ck-HG = 0.649 一起，把"聚合"与"更紧的单元界"分开了。
- **结论 3：矩形自选计划无收益。** CR9 tune 上 RECT-ck-HG 和 HC-WoR 的最优计划仍是 50/50；Neyman 与 sd^{2/3} 计划在 dev 上都更慢（0.1770–0.1807 vs 0.1724）。
- **结论 4：优势随规模和段数增大。** CR12 上为 0.48–0.50，X5（9 个规模相近的段，结局率约 0.6）上为 0.24–0.46。
- **自家兄弟方法。** FDC-MR[front3] 在 CR9 上比 FDC-BF 快 4%，在 CR12、LR9、X5 上慢 9–20%。论文须如实写出。
- **有效性。**
  - 所有方法、所有 dev 块 0 条错误流；
  - toy FWER（200 条流，ε = 0.02）上 PJC 全部 5 个变体与 RECT-ck-BF(+box) 都是 0 条，NAIVE-joint 为 5 条（检测有效）；
  - 单测 9 个通过：相位局部尾概率 MC（相位 2 的计数由相位 1 数据对抗性决定）、账本恰好用完 δ、plan 只读计数。

## 2. 块 A（确认性，唯一判定）：X5 RetailHero X9，eval 半边（未触碰）

- **数据**：`x5_retailhero/eval_labels.pkl` + `eval_outcome.npy`（100,393 行）。
  - 只能经 v8 gate 读取：`X5LayerEnv("eval")` 在锁 json 不存在时抛 PermissionError；
  - eval loader 与锁一起写，并校验 PROVENANCE 中的 sha256。
- **分段与问题**：X9 = 性别 {U, F, M} × 年龄段 {<38, [38,53) 或无效, ≥53}，2026-10-04 只按特征固定；问题族沿用 `fr.cr_problems`（15 个预算）。
- **设定**：K = 20，n_min = 2000，δ = 0.05，12/15 停止。主 ε = 0.02（dev 上所有主对手都在 τ_R 之前完成，删失少），次要 ε 为 0.015 与 0.03。ε 只依据 dev 选定。
- **种子**：35000–35199（200 条新流；2026-10-04 在工作区 json/jsonl 与代码中检索 35xxx，0 次命中）。
- **方法**：
  - 主方法：FDC-BF（v7 冻结，不改任何参数）；
  - 主家族 F：RECT-ck-HG（计划在 X5 dev 900–949 上从 {0.3, 0.4, 0.5, 0.6, 0.7, ney, ney23} 调出）、RECT-ck-HG-live、HC-WoR（在 X5 dev 900–949 上按 v6 网格重调）；
  - 联合自适应家族 G：PJC-BF-local* 与 PJC-BF-M*，两个网格（§1.2，共 26 个配置）都在 **X5 dev 900–949** 上按 v5 规则重调；
  - 次要（描述）：RECT-ck-BF、RECT-ck-BF+box、FDC-MR[front3]、FDC，以及 plug-in（FIX-bal-fav、B2-fav-tight、B5），同流配对，只描述。
- **终点与统计**：主终点 N80_pen，配对几何均值比 FDC-BF/r。配对百分位 bootstrap，B = 10⁴，seed 42，所有对手共用同一重抽样矩阵；报告单侧 UB95。
- **判定（IUT，全部满足才算 `positive_result_achieved`）**：
  1. 对 F 中每个 r，UB95(FDC-BF/r) < **0.60**；
  2. 对 G 中每个 r，UB95(FDC-BF/r) < **1.05**（非劣，界值 5%）；
  3. FDC-BF 在 200 条流的整个运行内 0 条错误流（同时报告 CP 95% UB）；
  4. replica（R1/R2）通过。
- **阈值依据（如实）**：
  - 0.60：dev（X5 900–949 与 930–949，ε = 0.02）上对 F 的 UB 为 0.350–0.360；留出约 0.25 的余量，是因为 eval 半边是另一组 10 万行，段方差可能不同。v7 的 0.80 不再适用，因为在这张表上会过松。
  - 1.05：dev 上对 G 的最差 UB 为 1.042（X5 ε = 0.01，20 个种子），在 ε = 0.02 时为 0.97–0.98。这是非劣界值，**不支持**"比 PJC 更快"的措辞。G 的优效（UB < 1）只作次要描述。
  - 若分量 2 失败而分量 1 成立：判为 `positive_result_not_achieved`（失败分量为 joint_adaptive_rival_not_inferior_failed），论文头条须改为"联合聚合本身（含相位自适应形式）是收益来源"。

## 3. 块 B（描述）：CR12 规模块

- CR12 eval 半边：已被 r5 的 FDC 暴露（种子 30000–30049），因此只作描述。种子 35200–35399，ε = 0.001。
- 方法：FDC-BF、FDC-MR[front3]、FDC、RECT-ck-HG、RECT-ck-HG-live、HC-WoR（CR9 配置）、RECT-ck-BF+box、PJC*（沿用 CR9 dev 调出的配置）。
- dev 预期：对 HG 约 0.48，对 PJC 约 0.95–0.97。
- 另报告每个检查点的证书耗时（4,096 个策略），回应成本问题。

## 4. 块 C（描述，事后）：CR9 v7 块 A 的流（种子 33000–33199）

- 在已封存的 v7 流上，**事后**加跑 PJC*（CR9 调好的配置）、RECT-ck-BF(+box)、矩形与 HC 的自选计划（CR9 调好的配置）、plug-in 配对。
- 明确标注"事后、条件于已用三次的表"，不进任何判定。这一块回应审稿人"在同一流上配对"的要求。

## 5. 其余规则（沿用 v7）

- R1：独立进程重跑每个任务的前 10 个种子，规范行逐位相同。R2：所有冻结 50/50 方法的 schedule_digest 一致。
- PJC 是 adaptive 方法，日程摘要不同；它与 FDC-BF 的配对靠相同的到达序列与池内记录顺序（make_schedule 中 r_arr / r_pool 与分配类型无关），R2 对它改为检查到达序列摘要。
- 代码哈希、输入哈希（X5 PROVENANCE 中三个文件的 sha256）、seal 与 v7 相同机制。gate 级联校验 v5/v6/v7。
- 偏离披露：
  - FDC-BF 仍是 v7 的事后方法；
  - PJC 族是在看过 v7 eval 结果之后、按审稿要求设计的；
  - 主对手的调参全部在 X5 dev 上进行，FDC-BF 不调参；
  - ε、X9 分段、阈值都只依据 dev 选定（本文件 §1 的数字）。

## 6. 时间线（WWW'27：摘要 10-18，全文 10-25 AoE）

| 日期 | 工作 |
|---|---|
| 10-05 | X5 dev 900–949 全网格调参（PJC 两族、矩形计划、HC 网格），约 1 小时 CPU；写 v8 gate、X5 eval loader、seal、replica、分析代码与单测 |
| 10-06 | external reviewer 审查锁；锁 v8 并提交（锁文件只提交一次） |
| 10-06 至 10-07 | 跑块 A：约 12 个方法 × 200 条流 × 3 个 ε，每行约 2 秒，3 进程 × 4 worker 下约 1 小时，加 replica。块 B：约 1–2 小时。块 C：约 1 小时 |
| 10-08 | 分析、summary、交给写作；10-08 至 10-17 改论文（新表头条、PJC 与机制 2×2、规模）；10-18 提交摘要；10-25 提交全文 |

余量：整个计算不到一天，失败重跑也来得及。主要风险是锁代码与 external reviewer 审查的轮次。
