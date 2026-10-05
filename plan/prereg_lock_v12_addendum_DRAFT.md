# Pre-registration lock v12 addendum (DRAFT): second OBD replication, random/women (A) and random/men (B)

Status: **DRAFT** (`plan/prereg_lock_v12_addendum_DRAFT.json`, built by `exp/code/build_v12_addendum_draft.py`; the
draft never authorises an eval read). Append-only to lock v5 and the v6–v11 addenda. Plan: `plan/v12_obd2_plan.md`
(317619b4 / 5b3eb048, before any split or dev read). Dev report: `plan/v12_dev_report.md`.

## 0. Outcome of the pre-stated block-status gate (read this first)

**Both blocks are DESCRIPTIVE.** The gate (plan §6) was evaluated on each campaign's dev half at the rule-selected ε,
from the dev truth. The gate files were written before the registered, saved dev cells, which hold the first saved
FDC-DP dev rows, in both dev runs. The exceptions are an unsaved pre-rule smoke test and the test suite's one-stream
check; both are disclosed in §4 and neither fed any rule.

| block | campaign | ε (rule) | (i) ε selected | (ii) all-control not ε-optimal somewhere | (iii) mean ε-optimal share ≤ 0.5 | status |
|---|---|---|---|---|---|---|
| A (primary) | random/women | 7.5e-4 | pass | pass (3/15 problems all-control ε-optimal) | **fail (0.661)** | descriptive |
| B | random/men | 1e-3 | pass | pass (5/15) | **fail (0.858)** | descriptive |

- **What still happens.** Both blocks are locked, run once on their eval halves (fresh seeds), sealed, replicated and
  analysed exactly as a confirmatory block would be.
- **No verdict.** The analysis record carries `verdict = "descriptive"` and no positive or negative wording. THRESH is
  computed by the pre-stated rule (women 0.83, men 0.90) and recorded with `applicable: false`.
  - `v12_analysis.decide_block` calls the unchanged v11 `decide` only to validate its inputs. Its internal THRESH
    comparison and verdict are discarded and have no reporting or decision effect.
  - A replica failure is reported explicitly as `replica_status = 'fail'`, and the verdict stays `descriptive`.
- **What the descriptive report gives, per block** (pre-stated):
  - the paired geometric-mean ratio TU-FDC-DP(b) / RECT-BF-DP-TU with one-sided UB95 and two-sided CI;
  - per-method false streams over the whole run, with Clopper–Pearson UB;
  - the eval ε-optimal share and all-control count;
  - exhaustion at N80;
  - the full-frontier ratio (rows to 15/15; horizon-sensitive);
  - the uncensored-subset ratio (streams with no exhausted pool for either method at N80);
  - descriptive ratios against HC-WoR-DP[tuned] and RECT-HG-DP (CP);
  - the replica status.
- **Dated observation (not a design change).** The identical protocol gives near-trivial frontiers on the smaller
  campaigns, because the rival-success rule pushes ε to 7.5e-4 and 1e-3 (15–20 % of base CTR, against 8.6 % in v11).
  Lock v11 remains the only confirmatory OBD block.

## 1. Blocks

| | block A: random/women | block B: random/men |
|---|---|---|
| eval task / layer | `v12_women_full` / `OBD-WOMEN-UF0X3-A2` | `v12_men_full` / `OBD-MEN-UF0X3-A2` |
| eval rows (τ) | 432,576 | 226,130 |
| eval seeds | 39200–39399 | 39400–39599 |
| S, items (arms) | 8, 46 (23 / 23) | 10, 34 (17 / 17) |
| κ[s,1] | 18 17 14 5 4 3 2 1 | 23 20 14 7 4 3 3 2 2 1 |
| ε, HC frozen | 7.5e-4, (0.5, 0.6) | 1e-3, (0.5, 0.6) |
| dev ratio [CI], UB95 | 0.7194 [0.706, 0.733], 0.7294 | 0.7923 [0.782, 0.803], 0.8017 |
| THRESH (computed, n/a) | 0.83 | 0.90 |
| status | descriptive | descriptive |

- **No pooling.** Each block is reported separately, and neither can rescue the other. A confirmatory block gets a
  verdict; a descriptive block gets a descriptive record with `verdict = 'descriptive'`.

## 2. Protocol (identical to lock v11; re-derived per campaign on its dev half)

- **Design.**
  - Segments: uf0 × uf3, with cells < 2 % merged into their uf0 "other" cell and residual cells < 1 % merged into the
    largest cell. Frozen as explicit maps, with fallback routing counted.
  - Arms: the top half of the campaign's items by dev CTR vs the rest. Frozen lists; an unknown eval item stops the
    run.
  - Costs: traffic-share knapsack κ[s,1] = max(1, round(8 S w_s^dev)), with 15 budgets 0.10–0.80.
  - K = 40, δ = 0.05, a 50/50 frozen design, and every stream runs to 15/15 or τ.
- **Methods.**
  - TU-FDC-DP(b) (primary).
  - RECT-BF-DP-TU (governing rival).
  - HC-WoR-DP[tuned] (descriptive).
  - RECT-HG-DP (descriptive, checkpoint-strength CP; its rows are labelled `checkpoint_strength_CP`).
- **ε and HC.** The v11 rival-success rule on {3e-4, 5e-4, 7.5e-4, 1e-3, 1.5e-3} and the block-G3 HC tuning, both on
  50 dev streams (seeds 950–999). The rule never reads FDC-DP.
- **Endpoint and statistic.** N80_pen at 12/15, with false certificates penalised to τ. The statistic is the paired
  geometric-mean ratio with B = 10^4 bootstrap resamples, seed 42, and a one-sided UB95.
- **Replica.** R1 (first 10 eval seeds re-run independently), R2 (schedule digests) and R2b (arrival digests), with the
  `v11_replica` rules imported unchanged.

## 3. Gating, sealing and binding

- **Reader order.** `obd_v12_eval.read_eval_rows(campaign, task, layer, lock_sha256)` proceeds in this order:
  1. caller arguments;
  2. the `prereg_v12` gate (locked addendum, canonical hash, inherited v11 → v5 chain, code / input / non-eval data
     drift, task registered, single-commit history), with no eval byte touched;
  3. the task registered for exactly this campaign and layer;
  4. the caller's sha equal to the lock's;
  5. an fsync'ed access-log line;
  6. `_verify_eval_bytes`, which is private and not exported (v11 r2 P2-1);
  7. deserialisation.

  The module docstring states this order (v11 r2 P2-2).
- **Data binding.** The lock binds 8 data files: dev.pkl, eval_labels.pkl, eval_outcome.npy and PROVENANCE.json for
  each campaign. The post-split draft and binding checks take the four eval hashes from the hash-bound campaign
  PROVENANCE files without opening any eval file. Split preparation did write and hash those files (§4, item 9).
- **Code binding.** The new files are `run_v12.py`, `obd_v12_eval.py`, `prereg_v12.py`, `v12_analysis.py` and
  `v12_seal.py`. The v11 modules are imported unchanged and hash-bound. No locked file was modified.
- **Seals.** `v12_seal`: each eval task is sealed alone at completion and never re-run.

## 4. Declarations

1. FDC-DP's certificate is FDC-BF's. The contribution is computational, and the claims are limited to the named
   implementations and to methods with proven guarantees.
2. The protocol was chosen on random/all dev for v11 and is applied mechanically here. The arm grouping uses dev
   outcomes, so each eval gap shrinks by selection.
3. The block-status gate is new in v12 and was pre-stated before any split. It can only demote a block.
4. **Row-level split.**
   - Dev rows sharing a timestamp with the campaign's own eval half: women 3.27 %, men 3.04 %.
   - Cross-campaign timestamp sharing with the random/all eval half: 35 rows (women) and 29 rows (men). Women and men
     share 27 rows.
   - User ids are absent.
5. This is the first use of both eval halves. Inference is conditional on each finite table.
6. HC is tuned on the rule's dev streams (selection-optimistic). RECT-HG-DP is CP only.
7. **Dev block regenerated once.**
   - Run 1 (d23b7aaa) preceded 39dace28, which only relabels descriptive outputs (coordinator instruction).
   - All 4,980 dev rows reproduce run 1 exactly. The ε/HC, block-status and THRESH gates and MANIFEST were rewritten
     once, for this reason only (frozen designs unchanged; THRESH source and MANIFEST hashes changed as expected).
8. **Smoke and test streams.**
   - A pre-rule smoke test ran one dev stream per method per campaign, TU-FDC-DP(b) included (seed 950, ε 5e-4). It
     was not saved and cannot be compared independently.
   - The test suite runs one women dev stream per method at 1.5e-3.
   - Neither fed any rule. Together they are a disclosed exception to plan §6's "before any FDC-DP dev row" wording.
9. **Split preparation.** The splitter loaded the source CSV (clicks included), wrote the eval click and label files,
    and opened and hashed them for PROVENANCE. It computed no eval-outcome statistic. After the split, no v12 code path
    opened an eval file.
10. **Plan clarification.** `plan/v12_obd2_plan.md` carries a dated appended note (append-only) on the descriptive-block
    output change (39dace28) and the smoke-test exception.
11. The top-level OBD PROVENANCE.json is not amended (v11 binds its hash). The pointer to the v12 splits is
   `open_bandit/V12_SPLITS.json`. bts/* stays unread.

## 5. Lock procedure (not executed in this step)

1. external reviewer lock review (at most two rounds).
2. Rebuild the draft.
3. Code-freeze commit.
4. `prereg_v12.finalize_addendum(<freeze commit>)`.
5. Commit the lock file alone.
6. Per block: `run_v12.py --task v12_<C>_full`, then `--replica`, then `--campaign <C> --analyse`.
