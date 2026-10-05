# Reproduction runner report

> REPRODUCTION RUN OUTSIDE THE LOCK GATES. The lock gates are replaced by a stand-in that only looks the task up in the shipped lock; this run does not recreate, and is no evidence of, the historical authorisation or chronology of the sealed eval runs.

Run 2026-10-06T04:14:55 to 2026-10-06T04:19:06; Python 3.12.14; numpy 2.5.3, pandas 3.0.6, scipy 1.18.1, scikit-learn 1.9.1.

Each regenerated row is compared field by field (exact, after a JSON round trip) with the sealed row of the same (method, eps, seed). Fields excluded: `code_sha256`, `rs_sec_cert`, `rs_sec_plan`, `rs_sec_total`, `sec`. `Sealed file = seal` is the content hash of the shipped sealed rows against the seal's `results_content_sha256`.

| Task | Lock | Seeds | Methods | Rows compared | Exact match | Mismatch | Fields compared | Sealed file = seal | Run (s) |
|---|---|---|---|---|---|---|---|---|---|
| v9a_full_d | v9 | 3 (37000..37199) | 8 | 24 | 24 | 0 | 40 | True | 39.2 |
| v9a_full_k | v9 | 3 (37000..37199) | 7 | 21 | 21 | 0 | 38 | True | 18.3 |
| v9a_full_x | v9 | 3 (37000..37199) | 2 | 6 | 6 | 0 | 38 | True | 15.9 |
| v9b_full_d | v9 | 3 (37200..37399) | 8 | 24 | 24 | 0 | 40 | True | 25.6 |
| v9b_full_k | v9 | 3 (37200..37399) | 7 | 21 | 21 | 0 | 38 | True | 8.4 |
| v9b_full_x | v9 | 3 (37200..37399) | 2 | 6 | 6 | 0 | 38 | True | 18.2 |
| v9c_full_b | v9 | 3 (37400..37599) | 9 | 81 | 81 | 0 | 37 | True | 26.0 |
| v9c_full_a | v9 | 3 (37600..37799) | 7 | 63 | 63 | 0 | 37 | True | 19.5 |
| v10a_full_s16 | v10 | 3 (38000..38199) | 4 | 12 | 12 | 0 | 39 | True | 2.3 |
| v10a_full_s32 | v10 | 3 (38000..38199) | 4 | 12 | 12 | 0 | 39 | True | 3.8 |
| v10b_full_s16 | v10 | 3 (38200..38399) | 4 | 12 | 12 | 0 | 39 | True | 5.0 |
| v10b_full_s32 | v10 | 3 (38200..38399) | 4 | 12 | 12 | 0 | 39 | True | 7.3 |
| v10b_full_s64 | v10 | 3 (38200..38399) | 4 | 12 | 12 | 0 | 39 | True | 13.5 |
| v10d_full_x5s64 | v10 | 3 (38000..38199) | 4 | 12 | 12 | 0 | 39 | True | 7.1 |
| v11_obd_full | v11 | 3 (39000..39199) | 4 | 12 | 12 | 0 | 39 | True | 7.7 |
| v12_women_full | v12 | 3 (39200..39399) | 4 | 12 | 12 | 0 | 41 | True | 6.3 |
| v12_men_full | v12 | 3 (39400..39599) | 4 | 12 | 12 | 0 | 41 | True | 4.5 |

Total: 354 of 354 regenerated rows equal their sealed rows in every compared field; files changed under exp/results during the run (outside the runner's output tree): 0. Overall: **all match**.

Option --provenance-by-files was set; PROVENANCE.json records accepted by their file hashes in place of the bound bytes: dd/open_bandit/PROVENANCE.json, open_bandit/women/PROVENANCE.json, open_bandit/men/PROVENANCE.json.

Lock files used by the gate stand-in (the stand-in only looks the task up in `eval_tasks`; the readers' own layer, lock-hash and data-hash checks still run):

- v9: `plan/prereg_lock_v9_addendum.json` (file sha256 `e2d7e8f114eb4b25...`, sha256 field `9540de0c7266e931...`)
- v10: `plan/prereg_lock_v10_addendum.json` (file sha256 `b7b9f16c0f001530...`, sha256 field `ecaf78e1c5424d99...`)
- v11: `plan/prereg_lock_v11_addendum.json` (file sha256 `1902174c8fb2b737...`, sha256 field `4ee733915bf2ac77...`)
- v12: `plan/prereg_lock_v12_addendum.json` (file sha256 `aba6ef975dedcefa...`, sha256 field `9a99ef28b3c22138...`)
