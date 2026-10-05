# external reviewer independent review: v6 prereg addendum DRAFT. Model: gpt-6-astra (config.yaml `reviewer_model`, no model param passed; approval-policy never, sandbox read-only)
# Timestamp: 2026-10-03T23:42:36+10:00 | external reviewer threadId 01a101fa-784c-77c2-be71-7a313772e9ff

**目前不宜锁定。** 三个 rival 的概率论构造基本成立，但存在 HC 实现被额外放宽、判定函数接受不完整输入、replica 条款冲突，以及锁后配置和揭盲保护不完整的问题。以下修改只涉及 v6；v5 的 `positive_result_achieved` 保持不变。

有效性核查结论：

- `RECT-ck-HG`：给定独立于 outcomes 的完整 schedule，checkpoint 的 cell count 固定，成功数确实服从 hypergeometric。`2×9×2×20=720`，每侧 `0.05/720` 的分配正确；无需再为 15 个问题分配 δ。同一同时覆盖事件允许跨 checkpoint 取交集，不增加错误概率。
- `RECT-ck-Bern`：先建立真方差的 Bernstein 事件，再与 variance UCB 事件相交，论证成立。`0.045+0.005` 合法，也适合作为匹配 FDC 的对照；但这不是最强 rectangular interval 的必要分配。
- `HC-WoR`：predictable betting、WoR conditional mean 和非负资本过程支持其保证。`max(θK⁺,(1−θ)K⁻)` 本身不必是鞅；其越界蕴含非负鞅 `θK⁺+(1−θ)K⁻` 越界，因此 Ville 论证成立。全序列预计算 λ 的实现仅使用各位置之前的 prefix，没有发现未来 outcomes 影响早期 λ。

**REQUIRED：锁定前必须修改**

1. **修复 HC 反演中错误的 accepted-center 假设。**  
   [wor_betting_v6.py:119](exp/code/dsswm/baselines/wor_betting_v6.py:119) 把搜索分割在 `mhat`，并声称它总被接受，这不成立。例如 `N=1000`、观测序列为 100 个 1 后接 100 个 0、`nstar=200`、`c=.75`、`alpha=.05/18`：按实现公式，`log K⁺(.5)=20.986`，超过阈值 `log(720)=6.579`；真正下侧根约为 `.552007`，当前代码却返回 `.5`。这是**保守地放宽 rival**，不是已发现的欠覆盖，但会损害比较公平性。应在完整逻辑区间上分别反演两个单调函数，明确空交集处理，并补充这类反例测试。

2. **让 A/C 判定必须通过完整性和 replica 检查。**  
   [v6_analysis.py:56](exp/code/dsswm/stats/v6_analysis.py:56) 的实际行为包括：`decide_block_c({}) → n100_pass`；C 仅提供 `B1` 也可通过；A 仅提供一个 rival 可返回 `keep`；某些含 `NaN` 的输入仍可返回 `keep`。  
   必须校验固定 rival 集合、每方法完整的 200 个指定 seeds、唯一配对、正且有限的 endpoints，以及完整 replica 状态。缺失、重复、非有限值或 replica 未完成都不得通过。把这些检查与最终聚合程序一并锁定。

3. **修正并落实 replica 规则。**  
   [runner:376](exp/code/run_r5s_v6.py:376) 比较所有 `alloc_kind=="fixed"` 方法，而 B 的 `B4-bal` 已固定为 `share=.4`。因此 `v6b_pilot` 的对应字段已经是 `false`，与草案要求冲突。  
   应逐块列明实际同为 50/50 的比较组；B 保留已选 `.4` 配置，并从该同设计组排除。冻结 R1 的 seeds、B 的全部 ε、计时字段排除清单，以及 R3 对 v5 字段的准确映射。当前没有完整的 v6 R1–R3 执行和汇总实现。`replica_fail` 后 A 应确定进入 `withdraw`，不能留下由 authors 酌情保留的空间；B 的失败标记也需规定。

4. **补齐配置、数据和锁文件的不可漂移绑定。**  
   [prereg_v6.py:40](exp/code/dsswm/stats/prereg_v6.py:40) 在 eval 时不检查 `input_sha256`；runner 却重新从文件读取 HC/LR9 配置，文件缺失还会回退默认配置。`LentaLayerEnv(check_sha=False)` 也是默认值。  
   必须在 eval、resume 和分析时核验实际输入，缺失配置直接拒绝，并将 locked-addendum hash 写入结果 provenance。增加必填 schema，拒绝空 `code_sha256`；当前 gate 会接受这种不完整锁。`finalize_addendum` 还应拒绝覆盖已有锁，验证 commit 确实包含所绑定代码，而不只是检查 40 位十六进制格式。锁后的修复应另立版本，不能重置 `eval_touched=False` 后覆盖重锁。

5. **落实 Lenta 揭盲保护，并准确限定“盲态”。**  
   [lenta_v6.py:88](exp/code/dsswm/envs/lenta_v6.py:88) 允许直接使用 `allow_eval=True`，没有执行 docstring 声称的 `addendum_gate`；现有测试也没有覆盖这个入口。此外，`load_table` 会读取并缓存含 outcomes 的整个 pickle，所以“不加载任何结局”并不准确。应在环境入口验证 locked task，补充拒绝测试，并区分“未向评估接口暴露”与“未加载”。  
   r4 的 LR8 full-table means 已包含后来 LR9 eval 的记录；更换 segmentation 和 split **不能恢复独立、未接触的 holdout**。这不妨碍 B 作描述性分析，但必须明确写成“既有 outcome exposure 的数据复用”。三档 ε 全报、不做 eval 后选择的规定应保留。

6. **保留阈值，修正阈值所支持的措辞。**  
   数值规则应统一写成 `UB*<.70`、`.70≤UB*<1`、`UB*≥1`；边界本身清楚，不需要调整。但 [草案:65](plan/prereg_lock_v6_addendum_DRAFT.json:65) 的推论不正确：`UB*≥.70` **不能推出点估计 speedup<1.43×**。例如点比值 `.60`、UB `.75` 应当 downgrade，但点 speedup 是 `1.67×`。应表述为“未达到预设优势证据门槛”，照实报告每个点估计和 CI。  
   同样，`keep` 门槛只能支持所检验的优势幅度，不能自动保留“同保证下约 2×”。HG 不是所有合法区间中的“最紧者”；HG、HC、Bern 比较同时改变界的构造等因素，不能据此定量归因“主要来自保证强度”。结论须限定为**已注册的具体 rivals 和配置**。

7. **完整披露 C 的已有 endpoint exposure。**  
   [草案 C](plan/prereg_lock_v6_addendum_DRAFT.json:131) 主要强调 N80 已揭盲，但 `r5_cr_fwer_audit/results.jsonl` 已包含同 200 个 seeds 的 **FDC `N100_pen`**。所以这不只是尚未观察部分的延续。应改称“在部分 endpoints 已公开后固定规则的 supplementary continuation”，不能称为独立盲态确认。可以保留 `n100_pass` 的措辞门，但如需新的 confirmatory evidence，应另用未使用 streams。C 的集合应显式绑定 v5 的全部 9 个 rivals。

**OPTIONAL：建议改进**

- **增加更强的合法 rival，但不必无限扩充。** 可增加 `RECT-ck-HG-live`：给定 outcome-free schedule，只向 `0<n_c(k)<N_c` 的随机区间分配 δ；空样本和耗尽区间无误覆盖成本。这保持相同 checkpoint 保证，并弱支配当前统一分配。更进一步可用预先固定的 checkpoint spending 或 exact finite-horizon boundary。
- **HC 调参可做锁前敏感性检查。** CR9 选中的 `.2` 位于网格内部，未见刻意选择差配置的证据；LR9 的 `.5` 位于边界，可检查更大 target。也可考虑固定权重 capital mixture。任何扩展均须只用 dev、披露搜索并重新冻结；不能把多个各自用满 δ 的 CS 直接取交集。
- **限定 qualification 和推断的含义。** `NAIVE-control` 的 `4/200` 只能说明检测到了错误证书，不能说明检出了 FWER 超过 `.05`——其 CP upper bound 反而约 `.0452`。paired percentile bootstrap 是近似推断；A/C 的 conjunctive IUT 无需 rival 内 multiplicity correction，但不能据此宣称全部独立措辞共同享有 95% 保证。

核查记录：当前 v5 canonical hash、v5 冻结模块、v6 草案代码及列出的输入 hash 全部匹配；结果目录内指定 A/B seeds 的 `seed`/`perm_seed` 检索为零。这个证据支持“未见已保存运行”，不能证明历史上绝无未保存访问。未修改任何文件，也未揭盲 LR9 outcomes；当前环境缺少指定虚拟环境和 `pytest`，未能独立重跑记录中的 23 项测试。

LOCK_READY: NO
