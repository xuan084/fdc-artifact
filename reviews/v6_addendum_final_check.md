Model: gpt-6-astra (config.yaml reviewer_model)
Timestamp: 2026-10-04T00:59:29+10:00

**仍不宜锁定：3 个 REQUIRED 项中，2 项 FIXED、1 项 NOT FIXED；措辞 FIXED。**

已确认 HEAD 为 `c7469c67`，draft canonical SHA256 为 `9b448004bd503a4fa3bd678ed3dd69f3a7b94b85e99bea620e8c8425c9d065b4`。12 个 code hashes、16 个 input hashes、3 个 raw-data hashes 全部匹配；commit 包含全部绑定代码。

1. **replica integrity/provenance：NOT FIXED。**  
   完整矩阵、task、lock、code、data 和主结果 content hash 的绑定已实现：[v6_replica.py:165](exp/code/dsswm/stats/v6_replica.py:165)。但 validator 仍信任报告内部的 `checks[].pass`：仅核验规则名称、主结果 hash 和布尔值，没有重算 R1–R3，也没有绑定实际 replica rows：[v6_replica.py:205](exp/code/dsswm/stats/v6_replica.py:205)。分析直接使用该返回状态：[run_r5s_v6.py:465](exp/code/run_r5s_v6.py:465)。

2. **R3 双 v5 schema 与 padded tail：FIXED。**  
   `v5_k_stop` 支持顶层 `rs_k_stop` 和嵌套 `run_stream.k_stop`：[v6_replica.py:91](exp/code/dsswm/stats/v6_replica.py:91)。比较截断到 `k_cut+1`，保留原有认证时间，并允许停止后的合法延续：[v6_replica.py:114](exp/code/dsswm/stats/v6_replica.py:114)。真实记录回归覆盖两种 schema：[test_v6_addendum.py:473](exp/code/dsswm/tests/test_v6_addendum.py:473)。我独立执行了四条真实记录的合法延续及前缀破坏检查，均符合预期。

3. **v6 raw-data hash binding：FIXED。**  
   三份指定数据均明确列入绑定：[data_v6.py:22](exp/code/dsswm/envs/data_v6.py:22)。schema 强制齐全，gate 校验实际文件：[prereg_v6.py:73](exp/code/dsswm/stats/prereg_v6.py:73)、[prereg_v6.py:108](exp/code/dsswm/stats/prereg_v6.py:108)。结果记录 `data_sha256`，resume 检查一致性：[run_r5s_v6.py:223](exp/code/run_r5s_v6.py:223)、[run_r5s_v6.py:241](exp/code/run_r5s_v6.py:241)。正式分析重新加载并验证锁：[run_r5s_v6.py:454](exp/code/run_r5s_v6.py:454)。

4. **RECT-ck-HG-live 的 `≤δ`：FIXED。**  
   实现说明、builder、JSON 和 Markdown 已同步，并说明存在 `L_k=0` 时预算严格小于 δ：[rect_v6.py:232](exp/code/dsswm/baselines/rect_v6.py:232)、[builder:104](exp/code/build_v6_addendum_draft.py:104)、[JSON:53](plan/prereg_lock_v6_addendum_DRAFT.json:53)、[Markdown:27](plan/prereg_lock_v6_addendum_DRAFT.md:27)。

**仍存在 eval 后改动报告、无检测地改变 verdict 的路径。** 只读内存反例使用完整 A 矩阵（6 methods × 200 seeds），人为制造一条 schedule mismatch；真实 `check_r2` 返回失败，原报告通过 validator 并返回 `fail`。仅将报告的 `status` 和 R2 的 `pass` 改为通过，**甚至保留非空 `bad` 列表**，validator 仍返回 `pass`。在全部 UB 为 `.5` 的决策测试中，verdict 随之从 `withdraw` 变成 `keep`；结果、阈值和所有 provenance hashes 均未改变。

锁定前应让分析从实际 replica rows、当前主结果及冻结 v5 references 重算 R1–R3，并核对报告。现有测试只覆盖 `status=pass` 而检查仍为失败的情况，未覆盖两者一起修改：[test_v6_addendum.py:549](exp/code/dsswm/tests/test_v6_addendum.py:549)。

本次未修改文件、未运行 eval；环境缺少 NumPy/SciPy/pytest，未重跑完整测试套件。上述 hash 校验和反例均已用标准库独立执行。

LOCK_READY: NO
