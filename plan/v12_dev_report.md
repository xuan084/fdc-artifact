# v12 dev report: Open Bandit random/women (block A) and random/men (block B), dev halves only; eval untouched

Written 2026-10-05 after the v12 dev blocks. Plan: `plan/v12_obd2_plan.md` (317619b4, renamed and extended in
5b3eb048; both commits precede the split and every dev read). Split: `exp/code/split_obd_v12.py` (run once, after the
plan). Code: `exp/code/run_v12.py`, `dsswm/envs/obd_v12_eval.py`, `dsswm/stats/{prereg_v12,v12_analysis,v12_seal}.py`
(first version f281ab27; descriptive relabel 39dace28). Rows are bound to the code hash.

## Bottom line

**Both blocks are descriptive.** The pre-stated block-status gate (plan §6) fails criterion (iii) on both campaigns:
at the rule-selected ε the mean share of feasible policies that are ε-optimal is **0.661** (women, ε = 7.5e-4) and
**0.858** (men, ε = 1e-3), above the 0.5 limit. Criteria (i) (the rule selects an ε) and (ii) (all-control is not
ε-optimal in every problem) pass for both. As pre-stated, both blocks will still be locked, run once on their eval
halves, sealed, replicated and analysed, and reported **descriptively, with no verdict**. THRESH is computed by the
rule and recorded but marked not applicable.

| campaign (block) | dev rows | S | ε (rule) | ε / dev base CTR | ε-optimal share | all-control ε-opt. | status | dev ratio FDC/RECT | UB95 | THRESH (n/a) |
|---|---|---|---|---|---|---|---|---|---|---|
| women (A, primary) | 432,009 | 8 | 7.5e-4 | 15.5 % (base 0.484 %) | 0.661 | 3 / 15 | descriptive | 0.7194 | 0.7294 | 0.83 |
| men (B) | 226,819 | 10 | 1e-3 | 19.9 % (base 0.501 %) | 0.858 | 5 / 15 | descriptive | 0.7923 | 0.8017 | 0.90 |

## 1. Eval-file access and data

- **Split.** Salted 50/50 row split per campaign, written once by `split_obd_v12.py`. The split rule was committed
  in the plan before any outcome was inspected. The splitter itself loaded the full source CSV (clicks included),
  partitioned it, serialised the eval clicks to `eval_outcome.npy`, wrote `eval_labels.pkl`, and opened and hashed
  both eval files to record their sha256. It computed no eval-outcome statistic. PROVENANCE's
  `eval_outcomes_touched: false` means no eval-outcome statistic was computed or inspected; it does not deny this
  outcome-independent split processing (external reviewer v12 lock r1 P2-1).
  - Women: dev 432,009 / eval 432,576. Men: dev 226,819 / eval 226,130.
  - All three files of each campaign are `chmod 444`. Sha256s and dev-only statistics are in
    `open_bandit/{women,men}/PROVENANCE.json`, and `split_created_utc` was written once.
  - The only eval quantities recorded are the row count and the non-outcome timestamp-overlap counts.
- **Overlap (timestamps only).** All three campaigns cover 2019-11-24 to 2019-11-30.
  - Exact-timestamp sharing with the already-used random/all eval half: women 35 rows (0.004 %), men 29 rows
    (0.006 %). Women and men share 27 rows.
  - Within each campaign, dev rows that share a timestamp with their own eval half: women 3.27 %, men 3.04 %
    (multi-position impressions).
  - User ids are absent, so user overlap cannot be measured.
- **random/all untouched.** The random/all eval membership was recomputed from `all.csv`'s index and timestamp columns
  with the v11 salt rule (688,159 rows, as in PROVENANCE). No random/all eval file was opened, and the top-level
  PROVENANCE.json was not changed: lock v11 binds its hash, so the pointer is `open_bandit/V12_SPLITS.json`.
- **No eval file access after the split.** After the split, no v12 dev, builder, gate or test code path opened,
  hashed or deserialised any eval file. Eval hashes come from the campaign PROVENANCE files.

## 2. Frozen designs (`exp/results/v12_gates/obd_v12_{women,men}_frozen.json`)

- **Women.**
  - S = 8 uf0 × uf3 segments with dev shares 0.280, 0.269, 0.226, 0.077, 0.069, 0.042, 0.026, 0.011.
  - κ[s,1] = 18, 17, 14, 5, 4, 3, 2, 1.
  - Budgets: 6, 9, 12, 16, 19, 22, 25, 28, 32, 35, 38, 41, 44, 48, 51.
  - Arms: 23 / 23 items, dev CTR 0.623 % (treatment) vs 0.346 % (control).
  - Dev segment uplifts are +0.0009 to +0.0038, with one negative segment (−0.0014, w = 0.026). The smallest
    segment × arm cell has 9 dev clicks.
- **Men.**
  - S = 10 segments with shares 0.293, 0.248, 0.179, 0.086, 0.056, 0.034, 0.033, 0.027, 0.025, 0.018.
  - κ[s,1] = 23, 20, 14, 7, 4, 3, 3, 2, 2, 1.
  - Budgets: 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63.
  - Arms: 17 / 17 items, dev CTR 0.610 % vs 0.394 %.
  - Every dev segment uplift is positive (+0.0005 to +0.0051). The smallest cell has 2 dev clicks.
- **Rule checks.** Both maps reproduce `obd_v11.segmentation(dev, 'uf0x3')` row for row, and the item lists reproduce
  `obd_v11.item_groups(dev, 2)` (unit tests). There are no fallback rows on dev.

## 3. ε rule and HC tuning (`v12_{campaign}_eps_hc.json`)

The rule used 50 dev streams per campaign (seeds 950–999) and the RECT-BF-DP-TU and 8 HC rows only. `select_eps`
refuses FDC-DP rows.

| campaign | ε | RECT-BF-DP-TU successes | HC tuned (config) successes | dev-best | qualifies |
|---|---|---|---|---|---|
| women | 3e-4 | 0 / 50 | 0 / 50 (0.5, 0.2) | RECT | no |
| women | 5e-4 | 0 / 50 | 0 / 50 (0.5, 0.2) | RECT | no |
| women | **7.5e-4** | 50 / 50 | 50 / 50 (0.5, 0.6) | HC (geo N80 384,455 vs 385,335) | **yes → selected** |
| men | 3e-4 | 0 / 50 | 0 / 50 (0.5, 0.2) | RECT | no |
| men | 5e-4 | 0 / 50 | 0 / 50 (0.5, 0.2) | RECT | no |
| men | 7.5e-4 | 0 / 50 | 1 / 50 (0.5, 0.6) | HC | no |
| men | **1e-3** | 50 / 50 | 50 / 50 (0.5, 0.6) | RECT (tie → RECT) | **yes → selected** |

- **HC frozen for the block.** (c 0.5, tf 0.6) for both campaigns.
- **No false streams.** None of the 4,500 rule streams had a false certificate.

## 4. Block-status gate (`v12_{campaign}_block_status.json`)

The gate files were written by `--select` at 18:13:16 (women) and 18:13:17 (men), before the dev cells (the first
FDC-DP dev rows) started at 18:13:20. Run 1 had the same order.

- **Women at 7.5e-4.**
  - The rule selects an ε, so (i) passes.
  - All-control is ε-optimal in 3 of 15 problems (the three smallest budgets), so (ii) passes.
  - The mean ε-optimal share is 0.661 over 256 policies, so (iii) fails.
  - Per-problem shares: 1.0, 1.0, 1.0, 0.886, 0.681, 0.632, 0.702, 0.722, 0.669, 0.532, 0.488, 0.388, 0.392, 0.423,
    0.401.
- **Men at 1e-3.**
  - (i) passes.
  - All-control is ε-optimal in 5 of 15 problems, so (ii) passes.
  - The mean share is 0.858 over 1,024 policies, so (iii) fails.
  - Per-problem shares run from 1.0 down to 0.533.
- **Status.** Both blocks are **descriptive**.

**Dated observation (2026-10-05; not a design change).**
- The identical v11 protocol gives near-trivial frontiers on the two smaller campaigns.
- On these halves the rectangles cannot reach 12/15 before τ at ε ≤ 5e-4 (women) or ≤ 7.5e-4 (men). The
  rival-success rule therefore pushes ε to 7.5e-4 and 1e-3, which is 15–20 % of the base CTR, against 8.6 % in v11.
- At those tolerances most feasible policies are ε-optimal, because almost every segment has a positive uplift of the
  same order as ε.
- The gate exists to catch this and did so. The gate, the grid and the rule were not altered (coordinator instruction,
  2026-10-05).

## 5. Dev cells (all four methods at the block ε; `v12_{campaign}_analysis_dev.json`)

These numbers are descriptive. No verdict is issued for a descriptive block.

| comparison | women ratio [CI] | women UB95 | men ratio [CI] | men UB95 |
|---|---|---|---|---|
| TU-FDC-DP(b) / RECT-BF-DP-TU | 0.7194 [0.706, 0.733] | 0.7294 | 0.7923 [0.782, 0.803] | 0.8017 |
| TU-FDC-DP(b) / HC-WoR-DP[tuned] | 0.7211 | 0.7310 | 0.7923 | 0.8017 |
| TU-FDC-DP(b) / RECT-HG-DP (CP) | 0.7211 | 0.7310 | 0.7923 | 0.8017 |
| full frontier (rows to 15/15) | 0.7048 | 0.7113 | 0.7785 | 0.7892 |
| uncensored subset (n = 50 both) | 0.7194 | 0.7294 | 0.7923 | 0.8017 |

- **Speed.** TU-FDC-DP(b) is faster on 50/50 streams in both campaigns.
  - Women: FDC-DP reaches 12/15 at checkpoint index 34–36 (5 / 34 / 11 streams). RECT-BF-DP-TU reaches it at index
    38 on all 50 streams; HC-WoR-DP[tuned] and RECT-HG-DP reach it one step earlier on 1 stream.
  - Men: FDC-DP at index 35–36 (19 / 31 streams); all three rectangles at index 38 on all 50 streams.
  - The ratios are again a gain of 2–4 grid steps.
- **Validity and exhaustion.** No method has a false stream. The node limit was never hit, and no pool was exhausted
  at N80.
- **Horizon.** The rectangle reaches 15/15 only at τ on 48 (women) and 40 (men) streams. The full-frontier ratio is
  therefore horizon-sensitive, as pre-stated.
- **RECT-HG-DP label.** RECT-HG-DP rows carry `validity = checkpoint_strength_CP`.
- **Replica.** The dev runner check of R1 (seeds 950–959 in an independent process), R2 and R2b passes for both
  campaigns.

## 6. THRESH (`v12_{campaign}_thresh.json`): computed, not applicable

THRESH = min(0.90, UB95_dev + 0.10). Women: min(0.90, 0.8294) → **0.83**. Men: min(0.90, 0.9017) → **0.90**. Each file
records `applicable: false`. For a descriptive block, THRESH has no reporting or decision effect: the validation call
(the unchanged v11 `decide`) computes a THRESH comparison and a verdict and then discards both. The gate files are hashed in
`exp/results/v12_gates/MANIFEST.json`.

## 7. Regeneration and checks

- **Run 1 (d23b7aaa).** Run 1 was committed as-is. Commit 39dace28 then changed the analysis and THRESH files only, to
  relabel descriptive outputs (no verdict, THRESH not applicable), on coordinator instruction after the gate result.
- **Run 2.** Rule tasks, selection and status, dev cells, dev replica, dev analysis and THRESH were regenerated once
  on that code.
  - All **4,980** dev rows reproduce run 1 exactly (N80_pen, cert_k, decided policies, schedule and arrival digests,
    false flags, N_stop_pen, exhaustion, validity label).
  - The gate files differ only in code hash, timestamps and the new applicability fields, plus the expected changes
    to the THRESH `source_sha256` (the regenerated dev analysis) and the corresponding MANIFEST hashes. The frozen
    design files were not rewritten. "Rewritten once" refers to the ε/HC, block-status and THRESH gates and to
    MANIFEST.
- **Smoke test.** Before the ε rule, one dev stream per method, TU-FDC-DP(b) included (seed 950, ε 5e-4), was run on
  each campaign's frozen dev env to check the code path. Its rows were not saved, fed no rule and cannot be compared
  independently. This is a disclosed exception to the plan's wording that the status file is written "before any
  FDC-DP dev row exists" (§6). The status files do precede the registered, saved dev cells in both runs.
- **Test suite.** The suite runs one women dev stream per method at 1.5e-3, seed 950.

## 8. What the eval runs can and cannot say

- Each eval block will report the same quantities as above on 200 fresh streams: women 39200–39399, men 39400–39599.
  The report is descriptive only.
- At these tolerances a speed-up is a statement about how fast a near-trivial frontier is certified. The report must
  give the eval ε-optimal share next to every ratio.
- v11 (random/all) remains the only confirmatory OBD block.
