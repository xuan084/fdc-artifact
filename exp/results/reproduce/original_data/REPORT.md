# Reproduction runner report

> REPRODUCTION RUN OUTSIDE THE LOCK GATES. The lock gates are replaced by a stand-in that only looks the task up in the shipped lock; this run does not recreate, and is no evidence of, the historical authorisation or chronology of the sealed eval runs.

Run 2026-10-06T04:08:10 to 2026-10-06T04:21:12; Python 3.12.14; numpy 2.5.3, pandas 3.0.6, scipy 1.18.1, scikit-learn 1.9.1.

Each regenerated row is compared field by field (exact, after a JSON round trip) with the sealed row of the same (method, eps, seed). Fields excluded: `code_sha256`, `rs_sec_cert`, `rs_sec_plan`, `rs_sec_total`, `sec`. `Sealed file = seal` is the content hash of the shipped sealed rows against the seal's `results_content_sha256`.

| Task | Lock | Seeds | Methods | Rows compared | Exact match | Mismatch | Fields compared | Sealed file = seal | Run (s) |
|---|---|---|---|---|---|---|---|---|---|
| v9a_full_d | v9 | 10 (37000..37199) | 8 | 80 | 80 | 0 | 40 | True | 138.1 |
| v9a_full_k | v9 | 10 (37000..37199) | 7 | 70 | 70 | 0 | 38 | True | 58.0 |
| v9a_full_x | v9 | 10 (37000..37199) | 2 | 20 | 20 | 0 | 38 | True | 55.2 |
| v9b_full_d | v9 | 10 (37200..37399) | 8 | 80 | 80 | 0 | 40 | True | 98.5 |
| v9b_full_k | v9 | 10 (37200..37399) | 7 | 70 | 70 | 0 | 38 | True | 31.0 |
| v9b_full_x | v9 | 10 (37200..37399) | 2 | 20 | 20 | 0 | 38 | True | 68.6 |
| v9c_full_b | v9 | 10 (37400..37599) | 9 | 270 | 270 | 0 | 37 | True | 91.7 |
| v9c_full_a | v9 | 10 (37600..37799) | 7 | 210 | 210 | 0 | 37 | True | 59.0 |
| v10a_full_s16 | v10 | 10 (38000..38199) | 4 | 40 | 40 | 0 | 39 | True | 7.4 |
| v10a_full_s32 | v10 | 10 (38000..38199) | 4 | 40 | 40 | 0 | 39 | True | 12.1 |
| v10b_full_s16 | v10 | 10 (38200..38399) | 4 | 40 | 40 | 0 | 39 | True | 15.4 |
| v10b_full_s32 | v10 | 10 (38200..38399) | 4 | 40 | 40 | 0 | 39 | True | 20.4 |
| v10b_full_s64 | v10 | 10 (38200..38399) | 4 | 40 | 40 | 0 | 39 | True | 34.7 |
| v10d_full_x5s64 | v10 | 10 (38000..38199) | 4 | 40 | 40 | 0 | 39 | True | 18.5 |
| v11_obd_full | v11 | 10 (39000..39199) | 4 | 40 | 40 | 0 | 39 | True | 20.7 |
| v12_women_full | v12 | 10 (39200..39399) | 4 | 40 | 40 | 0 | 41 | True | 17.6 |
| v12_men_full | v12 | 10 (39400..39599) | 4 | 40 | 40 | 0 | 41 | True | 13.6 |

Total: 1180 of 1180 regenerated rows equal their sealed rows in every compared field; files changed under exp/results during the run (outside the runner's output tree): 0. Overall: **all match**.

Lock files used by the gate stand-in (the stand-in only looks the task up in `eval_tasks`; the readers' own layer, lock-hash and data-hash checks still run):

- v9: `plan/prereg_lock_v9_addendum.json` (file sha256 `e2d7e8f114eb4b25...`, sha256 field `9540de0c7266e931...`)
- v10: `plan/prereg_lock_v10_addendum.json` (file sha256 `b7b9f16c0f001530...`, sha256 field `ecaf78e1c5424d99...`)
- v11: `plan/prereg_lock_v11_addendum.json` (file sha256 `1902174c8fb2b737...`, sha256 field `4ee733915bf2ac77...`)
- v12: `plan/prereg_lock_v12_addendum.json` (file sha256 `aba6ef975dedcefa...`, sha256 field `9a99ef28b3c22138...`)

Post-run note: All files that changed under exp/results during this run were outputs of a second reproduction run written concurrently to exp/results/reproduce/reingested_data/ (listed in exp_results_changed_files_raw). The runner then excluded only its own output directory from the snapshot; it now excludes exp/results/reproduce/. No file outside exp/results/reproduce/ changed; all_match recomputed accordingly on 2026-10-06.
