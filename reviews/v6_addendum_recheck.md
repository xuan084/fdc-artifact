<!-- model: gpt-6-astra (config.yaml reviewer_model; MCP call without model param), thread 01a10226-8391-7921-b56e-9818e9285f2c -->
<!-- timestamp: 2026-10-04T00:30:07+10:00 -->
**仍不宜锁定。7 项中，4 项 FIXED、3 项 PARTIAL。** `b618cdb8` 修复了主要统计构造和措辞问题，但 replica 验证与数据绑定尚未闭合。

1. **HC 反演：FIXED。**  
   [wor_betting_v6.py:127](exp/code/dsswm/baselines/wor_betting_v6.py:127) 现在分别在完整逻辑区间内反演两个单调函数，不再假设样本均值被接受；整段拒绝或空交集时返回逻辑区间并标记 `empty`。原反例的下界 `.552007` 和空交集处理均有回归测试：[test_v6_addendum.py:360](exp/code/dsswm/tests/test_v6_addendum.py:360)。

2. **A/C 完整性与 replica 门控：PARTIAL。**  
   固定方法集、200 个指定 seeds、唯一配对、正且有限 endpoints 已由 [v6_analysis.py:83](exp/code/dsswm/stats/v6_analysis.py:83) 和正式分析入口落实；空集合、部分 rivals、`NaN` 的旧问题已修复。  
   **但 replica 仍只检查一个字符串。** [run_r5s_v6.py:435](exp/code/run_r5s_v6.py:435) 读取报告后，只聚合 `status`，不验证 `task_id`、`addendum_sha256`、检查项、覆盖范围或对应结果。只读反例确认：携带错误任务、`addendum_sha256=None`、`checks=[]` 的 `status="pass"` 仍可支持 `keep`。旧报告或不完整报告可以充当最终判定的通行证。

3. **replica 规则与执行：PARTIAL。**  
   B 的 `B4-bal share=.4` 已正确排除；R1 seeds、三档 ε、计时排除字段已冻结：[v6_replica.py:17](exp/code/dsswm/stats/v6_replica.py:17)。A/C 的失败判定和 B 的失败标记也已明确。  
   **R3 对真实 v5 文件仍有两个确定错误：**[v6_replica.py:85](exp/code/dsswm/stats/v6_replica.py:85) 直接读取顶层 `rs_k_stop`，而 `r5_cr_main_b` 将其保存为 `run_stream.k_stop`，见 [run_r5_cr_main_b.py:167](exp/code/run_r5_cr_main_b.py:167)；实际记录触发 `KeyError`。此外，v5 的 `n_cert_curve` 被补齐至 20 个 checkpoints，见 [run_r5_cr_main.py:106](exp/code/run_r5_cr_main.py:106)。按整个数组长度比较，会把合法的停止后继续认证误判为失败。用 `Peace-rect, seed=30037` 的真实 v5 记录构造合法延续，已复现该误报。  
   R2 也不检查预定 streams 是否齐全：[v6_replica.py:60](exp/code/dsswm/stats/v6_replica.py:60)。只包含 A 前 10/200 个 seeds 的主结果，足以生成 R1、R2 均通过的报告；后续补齐结果不会自动使旧报告失效。现有 C pilot 的 R3 明确为“不适用”，不能验证上述映射。

4. **配置、数据与锁绑定：PARTIAL。**  
   配置缺失拒绝、配置比对、输入 hash 检查、结果中的 `addendum_sha256`、拒绝空 hash 表、拒绝覆盖锁和 commit 内容检查均已实现：[run_r5s_v6.py:126](exp/code/run_r5s_v6.py:126)、[prereg_v6.py:141](exp/code/dsswm/stats/prereg_v6.py:141)。  
   **实际数据尚未完整绑定。** [build_v6_addendum_draft.py:29](exp/code/build_v6_addendum_draft.py:29) 的输入列表没有原始 Criteo/Lenta 数据；CR9 入口直接构造 `PoolReplayEnv`，其 [pool_replay.py:77](exp/code/dsswm/envs/pool_replay.py:77) 直接读取 pickle，没有核验数据 hash。因此 Criteo 数据变化不会被当前 v6 gate 拒绝，resume 还可能混用不同数据产生的结果。Lenta 的构造器默认校验有所改善，但分析入口没有统一的数据绑定检查。

5. **Lenta 揭盲保护与暴露披露：FIXED。**  
   [lenta_v6.py:103](exp/code/dsswm/envs/lenta_v6.py:103) 已删除 `allow_eval`，要求合法 block-B task 通过 addendum gate；拒绝路径有测试：[test_v6_addendum.py:403](exp/code/dsswm/tests/test_v6_addendum.py:403)。草案明确披露完整 pickle 的短暂加载，以及 r4 full-table means 导致的既有 outcome exposure，并将 B 定为描述性复用：[DRAFT.json:122](plan/prereg_lock_v6_addendum_DRAFT.json:122)。三档 ε 全报规则保留。

6. **阈值及其支持的措辞：FIXED。**  
   [v6_analysis.py:126](exp/code/dsswm/stats/v6_analysis.py:126) 保持 `<.70`、`.70≤UB*<1`、`≥1`；只读边界检查结果正确。[DRAFT.json:14](plan/prereg_lock_v6_addendum_DRAFT.json:14) 已删除错误的点估计推论，限制结论于 registered rivals/configs，并明确 `keep` 不自动支持“同保证下 2×”或保证强度的定量归因。

7. **C 的 endpoint exposure 与 rival 集合：FIXED。**  
   [DRAFT.json:187](plan/prereg_lock_v6_addendum_DRAFT.json:187) 明确披露同 200 条流上已知的所有方法 N80 和 FDC `N100_pen`，定性为部分 endpoints 公开后的 supplementary continuation。全部九个 v5 rivals 已列出，runner 也检查与 v5 集合一致：[run_r5s_v6.py:339](exp/code/run_r5s_v6.py:339)。

**RECT-ck-HG-live 的概率构造有效。**  
[rect_v6.py:226](exp/code/dsswm/baselines/rect_v6.py:226) 的实现与论证一致：给定 outcome-free schedule，`L_k` 固定；每个 live cell 每侧分配 `δ/(2KL_k)`，每个 checkpoint 的总误覆盖预算至多 `δ/K`。空样本和耗尽区间没有误覆盖成本，跨 checkpoints 取交集仍在同一同时覆盖事件内。`L_k≤SA` 使单次区间不宽于 HG；无需额外为 15 个问题分配 δ。`L_k=0` 时总预算为零，因此文中的总和严格应写 **≤δ**。已有 qualification 记录为 `0/200` 错误流、CP UB `.014867`：[summary.json:18](exp/results/pilots/v6_baseline_qualification/summary.json:18)。

**不能确认“eval 后没有判定漂移空间”。** 数值阈值、bootstrap 参数和正式 rival 集合已固定；但未绑定当前结果的 replica 报告，以及未校验的 Criteo 数据，仍能在不改变这些阈值的情况下改变判定依据。R3 的确定性错误也必须在 eval 前修复。

剩余 **REQUIRED**：

1. **闭合 replica 完整性与溯源。** 报告生成前验证完整任务矩阵；绑定 task、lock、code 和实际结果内容 hash。分析时核验所需 R1–R3、seeds/ε/methods 与当前结果一致；结果追加、替换或 resume 后，旧报告必须失效。
2. **修正 R3 的真实字段映射。** 同时支持两种 v5 schema，只比较到原始 `k_stop` 的真实前缀，排除补齐尾部；用现有真实 v5 records 加回归测试，并同步 JSON、Markdown 和 builder。
3. **补齐原始数据绑定。** 在 v6 层冻结 Criteo 与 Lenta 数据 hash，在 eval、resume、分析中验证，并绑定结果 provenance；无需修改 v5 冻结模块。

核查范围：11 个 v6 code hashes、16 个 input hashes 均匹配；`b618cdb8` 包含全部绑定代码，v5 冻结模块无漂移。结果 JSONL 中指定 A/B eval seeds 检索为零。未修改文件或运行新 eval。记录声称 `45 passed`；当前环境缺少 `.venv`、NumPy/SciPy/pytest，未能独立重跑完整测试，以上反例使用标准库和现有结果完成。

LOCK_READY: NO
