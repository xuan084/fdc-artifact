<!-- model: gpt-6-astra (config.yaml reviewer_model; MCP call without model param) -->
<!-- timestamp: 2026-10-03T15:26:10Z | reviewer threadId 01a1025f-8031-7c40-ae00-ebeb0b07671a -->
**REQUIRED(1)：NOT FIXED。原先仅改 `checks[].pass/status` 的漏洞已 FIXED，但仍存在联合修改主结果与报告、无检测地翻转 verdict 的路径。**

已确认 HEAD 为 `5571d5ed`，draft SHA256 为 `0f3607c9…`，12 个 code hashes、16 个 input hashes 均匹配。

原反例现在被拒绝：`validate_report` 重算 checks/status 并核对所有 bindings，不一致即抛出 `ValueError`：[v6_replica.py:226](exp/code/dsswm/stats/v6_replica.py:226)。正式分析使用其返回状态：[run_r5s_v6.py:472](exp/code/run_r5s_v6.py:472)。标准库实测也确认 honest pass 通过，篡改 replica rows、v5 references 或缺失矩阵行均被拒绝。

**剩余反例已用标准库复现：**

1. 完整 A 矩阵为 6 methods × 200 seeds；令 `FDC / seed 31010` 的 `schedule_digest` 与其他方法不同。R2 失败，所有 UB 为 `.5` 时 verdict 为 `withdraw`。
2. 只修饰该主结果行的 `schedule_digest`，再更新报告的 `checks`、`status`、`results_content_sha256`。
3. 验证返回 `pass`，verdict 变成 `keep`。replica rows、全部 endpoints、lock/code/data bindings 均未改变。

原因是 R1 只复核前 10 seeds，未覆盖 `31010`：[v6_replica.py:180](exp/code/dsswm/stats/v6_replica.py:180)。主结果 hash 从**当前可修改的 rows** 计算，并仅与可同时修改的报告比较：[v6_replica.py:193](exp/code/dsswm/stats/v6_replica.py:193)。因此它证明当前文件之间一致，不能识别这次 eval 后的联合修改。

若要求涵盖上述 provenance，仍需独立保存、冻结并验证 eval 完成时的结果摘要。此次仅运行内存中的标准库验证，未运行 eval 或完整 pytest。

LOCK_READY: NO
