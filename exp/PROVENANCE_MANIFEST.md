# Provenance manifest (anonymous copy)

This file says, for each pre-registration lock and its eval blocks, which claims a reader of this anonymous copy can
check from the files alone, and which claims rest on material that the copy cannot contain. The machine-readable
version is `exp/PROVENANCE_MANIFEST.json`, written by `exp/code/reproduce/build_provenance_manifest.py` from the shipped
files (rerun it to recompute every entry below; with `$DATA_DIR` set it also hashes the data files). The historical
locks, seals and analyses are unchanged; nothing here re-authorises or re-dates an eval run.

## 1. What this copy lets a reader verify

| Check | How | Result in this copy |
|---|---|---|
| Sealed rows equal their seals | `cd exp/code && python check_sealed_rows.py` recomputes `results_content_sha256` (timing fields excluded, rows sorted) for every shipped row file and checks it against the seal, the planned method x eps x seed matrix and the lock hash in every row | 21 of 21 shipped row files pass: all 8 v9, 6 v10, 1 v11, 2 v12 tasks, and v7a_full_b, v8a_full_a, v8a_full_b, v8c_full. Rows of 10 v6-v8 tasks are not shipped (seals and analyses are) |
| Analyses follow from the sealed rows | `recompute_v10_from_sealed_rows.py` (v10 A, B, D); `verify_r8d_numbers.py` chain (v6-v12 printed numbers, from analyses and sealed rows) | 0 mismatches (README, "Reproducing the paper tables") |
| The shipped code, frozen gate files and data regenerate the sealed rows | `exp/code/reproduce/reproduce_frozen.py` re-executes the frozen eval configuration of each headline task for registered eval seeds, outside the lock gates, and compares every regenerated row field by field (all fields except timing and `code_sha256`) | see section 3: every compared row equal, on our data files and on data rebuilt from the public raw files |
| Data files are the bound ones | `ingest_tidy.py`, `ingest_v8_fresh.py x5`, `split_obd_all.py`, `split_obd_v12.py` rebuild every derived data file from the public raw files; the locks bind their SHA-256 | all 15 derived data files (3 tidy tables, 3 X5 split files, 3 + 6 Open Bandit split files) reproduced byte for byte on 2026-10-06 with the pinned versions; content hashes for version-independent checks in `check_tidy_content.py` |
| Frozen gate files are the bound ones | file SHA-256 against each lock's `frozen_gates` / `input_sha256` entries (manifest builder) | v11: 3 of 3 and v12: 8 of 8 frozen gate files byte-identical; v10 score models and segmentation json byte-identical (their hashes are checked at load) |
| Replica agreement | each task's `replica_report.json` (an independent re-run of the first seeds at eval time, compared with the sealed rows) | status `pass` in all 31 shipped eval-task replica reports (v6-v12) |
| v5 base lock | the canonical hash recomputes from the shipped bytes | yes (the file was not edited) |

## 2. What this copy cannot establish, and why

| Claim | Why it cannot be checked here | What is shipped instead |
|---|---|---|
| Each lock was written and committed before its block's eval half was read | The gates anchor every lock to a single commit of the original git history (`lock_history_check`), and every seal to its own commit (`*.seal_ref.json`). The history identifies the authors and is not part of this copy. | Lock times and code-freeze commits in the README lock index; the seals' `sealed_at`; the access logs of lock v11, lock v12 and post-hoc block G. Checkable at de-anonymisation. |
| No eval outcome was read outside the gated, logged readers | A negative cannot be shown from files. The access logs list recorded reads only, and the eval readers of locks v6-v10 kept no access log (their gates refused unauthorised reads but did not log authorised ones). | The gated reader code (`dsswm/envs/*_eval.py`, `x5_v8.py`, `x5_v9.py`, `hillstrom_v9.py`, `lenta_v6.py`), the logs, the disclosures in supplement S2. |
| The scrubbed lock files are the locked bytes | Scrubbing edited string leaves (paths, author and tool labels), so the lock bodies of v6-v12 no longer hash to their `sha256` fields (the canonical hash every row, seal and analysis binds; the field itself is unchanged). | `exp/lock_scrub_diff.json`: per lock, the edited leaf paths. It reports 0 edited non-string leaves and 0 edited SHA-256 values in every lock (numbers, seeds, thresholds, hashes and decision rules untouched). This is the authors' statement, computed from the originals, and checkable at de-anonymisation. |
| The scrubbed code and input files are the bound bytes | Files that contained paths or tool names were edited (manifest JSON: per lock, the list of differing files; e.g. lock v10: 16 of 39 bound code files byte-identical). | Behaviour equality: the reproduction runner regenerates the sealed rows with the scrubbed code (section 3). |
| The Open Bandit provenance records are the bound bytes | They are hash-bound by locks v11 / v12 but not shipped (their free text names the authors' tooling or machine paths). | Their file-hash records are checked indirectly: the runner's `--provenance-by-files` accepts a re-made record only if it lists, and the disk holds, exactly the bound data-file hashes. |
| The development work used only development halves | Development rows, logs and selection files are partly shipped (`exp/results/pilots/`); unlogged exploration cannot be excluded. | Development analyses and rule files bound by the locks; supplement S2 disclosures. |

## 3. Reproduction-runner agreement

The reports are `exp/results/reproduce/original_data/REPORT.md` (our data files) and
`exp/results/reproduce/reingested_data/REPORT.md` (every derived data file rebuilt from the public raw files with
the shipped scripts; Open Bandit provenance records accepted by file hashes). Both ran from this scrubbed copy.
Summary: original data, 10 registered eval seeds per task (spread over each task's 200), all 17 eval tasks of locks v9-v12 and every registered method: 1180 of 1180 regenerated rows equal their sealed rows in every compared field (about 13 min wall clock with 4 workers); re-ingested data, 3 seeds per task: 354 of 354 rows equal (about 4 min). Mismatches: 0. Files changed under `exp/results` outside the runner's output tree: 0 (see the post-run note in original_data/REPORT.md about the concurrent second run). The shipped sealed rows equal their seals for all 17 tasks.

The runner replaces each lock gate by a stand-in that only looks the task up in the shipped lock's `eval_tasks`; the
readers' own checks (layer, campaign, lock sha256, data and frozen-file hashes, frozen configurations against the
lock) still run. It used only registered eval seeds of already-sealed tasks and wrote nothing under
`exp/results/full`. An exact match shows that the shipped code, gate files and data determine the sealed rows; it is
not evidence about when those rows were first computed.

## 4. Per-lock summary

| Lock | Lock file = canonical hash | Bound code byte-identical | Bound inputs byte-identical (absent) | Frozen gate files | Data files (our copies) | Eval tasks: rows shipped / = seal / reproduced |
|---|---|---|---|---|---|---|
| v5 | yes | 148/161 | 13/18 (0) | - | none bound | no eval task |
| v6 | no (17 string leaves scrubbed) | 6/13 | 13/16 (3) | - | 3/3 | 0 / 0 / 0 of 5 |
| v7 | no (8 string leaves scrubbed) | 11/20 | 13/19 (3) | - | 3/3 | 1 / 1 / 0 of 4 |
| v8 | no (12 string leaves scrubbed) | 15/30 | 13/24 (5) | - | 7/7 | 3 / 3 / 0 of 5 |
| v9 | no (16 string leaves scrubbed) | 21/43 | 16/29 (0) | - | 8/8 | 8 / 8 / 8 of 8 |
| v10 | no (17 string leaves scrubbed) | 16/39 | 11/21 (0) | - | 5/5 | 6 / 6 / 6 of 6 |
| v11 | no (9 string leaves scrubbed) | 21/45 | 17/27 (0) | 3/3 | 4/4 | 1 / 1 / 1 of 1 |
| v12 | no (11 string leaves scrubbed) | 25/50 | 28/41 (0) | 8/8 | 8/8 | 2 / 2 / 2 of 2 |

Inputs absent from this copy are development row files bound by locks v6-v8 (`exp/results/full/r5_*`, `pilots/fdc_bet/*`, `pilots/pjc_v8/*`; their summaries are shipped). Data files were hashed under our `$DATA_DIR`; with data rebuilt by the shipped scripts the same files are byte-identical except the three Open Bandit provenance records (section 2). In the authors' unscrubbed tree the same builder reports every lock's canonical hash recomputing and every bound code file byte-identical; two inputs differ there from their lock-time hashes: `exp/results/full/v6_summary.md` (bound by v7; a dated erratum line was appended later, recorded with both hashes as known input drift in lock v8) and `plan/task_plan.json` (bound by the v5 base lock; lock v8 states that the v6-v12 gate chain does not re-verify v5 inputs).
