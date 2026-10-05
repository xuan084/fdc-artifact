# external reviewer review of the v8 addendum (lock + PJC validity + X5 blinding)

Reviewer: external reviewer (MCP `reviewer`, config default model, approval-policy never, read-only sandbox). Reviewer did not read
`eval_outcome.npy` or the raw outcome csv.

## Round 1 (2026-10-04) — VERDICT: fix before lock

Reviewer summary: the core certificate argument (PJC local and menu) is valid; ordinary eval entry points reject drift in
bound code/configs; the decision rule matches the registration; all 30 code and 23 input bindings verified; the v7 drift
record matches the one-line erratum of commit e740f421; the X5 eval loader gates before loading outcomes, dev/eval client
ids are disjoint and match the hash split, eval labels contain no target column, and both frozen plan matrices reproduce
from dev outcomes alone. 17 eligible tests passed.

| # | Sev | Finding | Resolution |
|---|---|---|---|
| 1 | P0 | Cross-task R2b failures were written to the analysis but did not enter the replica status, so they could not block a positive verdict. | Fixed: `run_r5s_v8.analyse` combines task replica statuses AND every cross-task R2b check into `replica_status` (field `replica_status_combined`). |
| 2 | P1 | R2b hashed a separately regenerated schedule, not the one consumed by the run. | Fixed: `_CaptureSchedule` wraps `frontier_runner.make_schedule` for the duration of each run and hashes the schedule the engine actually used (exactly one per run, asserted). Test `test_arrival_digest_comes_from_the_consumed_schedule` (incl. a different-seed fault check and wrapper removal). No v5-v7 file modified. |
| 3 | P1 | The PJC grid excluded the valid no-reset member (b = ()), which is materially competitive (exploratory dev 900-909: 0.940 vs the selected local rival). | Fixed: `PJC-BF[local,b=,half]` added to the local grid and run on X5 dev 900-949 and CR9 dev 900-949. It wins the local family on both (X5 N80/tau 0.2541 vs 0.2681; CR9 0.1103 vs 0.1155) and is the frozen PJC-local. Disclosed: G-local is then a non-adaptive joint certificate (differs from FDC-BF only by deterministic count tracking); G-menu remains adaptive. |
| 4 | P1 | Phase plans take effect at the next replan batch, not exactly at the boundary checkpoint (X5 eval: checkpoint 3020, replan 3200). | Registered as implemented (lock `rivals.G.phase_activation`): validity unaffected (counts in between follow a count-only preference fixed before the boundary); the dev tuning used the same behaviour. Not changed, so tuning and eval are consistent. |
| 5 | P1 | Final lock markdown and PROVENANCE.json were not enforced. | Fixed: `plan/prereg_lock_v8_addendum.md` is in `input_sha256`; `x5_provenance` (PROVENANCE.json) is in `data_v8.DATA_FILES_V8` / `LAYER_DATA_V8['X9']`, hence in `data_sha256`, checked at finalisation, at every gate check and in every X9 row's data hash. |
| 6 | P1 | "No code path reads eval outcomes before the lock" overstated: ingest parsed/copied them and hashing reads bytes. | Reworded in the lock (md s0, json `threat_model`): procedural blinding; ingest parsed the column and wrote it unreduced; hashing reads bytes; no eval-outcome statistic computed, inspected or used; limits of retrospective verification stated. |
| 7 | P2 | The gate chain does not re-verify the v5 lock's own input manifest (plan/task_plan.json differs from its v5 hash). | Disclosed as inherited scope in the lock (`threat_model`); no v8 computation depends on task_plan.json. |

## Round 2 (2026-10-04) — VERDICT: OK to lock

Reviewer verified all seven resolutions: cross-task R2b now blocks a positive verdict (fault injection turned an
otherwise-positive result into `positive_result_not_achieved` / `replica_fail`); `_CaptureSchedule` captures the exact
engine schedule for FDC-BF, HC (composes with the HC `StreamEngine` patch) and PJC under fork workers, and is restored
after an exception (not valid for threaded concurrency inside one worker, which the runner does not use); the no-reset
pick was recomputed independently on X5 and CR9 and its disclosure is accurate; 30 code and 24 input bindings match;
binding the md with dev numbers creates no circularity (post-lock corrections must go to a new addendum). 18 tests
passed. No new P0/P1.

| # | Sev | Finding | Resolution |
|---|---|---|---|
| 8 | P2 | Statement still called both G rivals "phased adaptive". | Reworded: "two PJC-BF joint-certificate rivals, one outcome-adaptive". |
| 9 | P2 | "F_{T_j}-measurable" counts: they also depend on future arrivals. | Reworded (json and md): counts are determined by the phase-start history and the outcome-free future arrival sequence. |
