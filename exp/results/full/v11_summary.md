# Lock v11: results summary (Open Bandit Dataset, a never-replayed randomized log)

- **Lock.** `plan/prereg_lock_v11_addendum.json`, sha256 4ee73391…629963. Code freeze 1741ebcf, lock commit 91606679 (2026-10-05).
  - The lock was reviewed externally in 2 rounds; r2 verdict "lock" (lock reviews under `reviews/`).
  - Plan `plan/v11_obd_plan.md` was committed alone (40ee83d8) before the dev block. Dev report: `plan/v11_dev_report.md`.
- **Data.** ZOZOTOWN Open Bandit Dataset, `random/all`: uniform-random logging over 80 items, binary click outcome.
  - The salted 50/50 row split was written before any click was read (`shared/datasets/open_bandit/PROVENANCE.json`).
  - **The evaluation half was used exactly once, here.** It is the first table in this project whose evaluation outcomes informed nothing in the design.
- **Design (frozen from dev).**
  - Nine user-feature segments (uf0 × uf3; cells under 2% merged).
  - Two item-group arms: the top-40 items by dev CTR vs the other 40.
  - Traffic-share knapsack costs (κ = 2–26) and 15 budgets from 0.10 to 0.80.
  - K = 40 checkpoints and a 50/50 frozen plan for every method.
- **ε = 3e-4** absolute CTR, about 8.6–8.7% of the base rate.
  - It was set by the pre-stated rival-success rule on dev (better of RECT-BF-DP-TU and the tuned HC-WoR-DP), never reading FDC-DP.
  - On the evaluation half, 15.1% of the ~~512 policies~~ feasible policies [r7b] are ε-optimal on average, and no problem has all-control ε-optimal.
- **Eval run.** Seeds 39000–39199 (200 fresh streams), 800 rows, 2026-10-05 15:12–15:19.
  - ~~Sealed alone (6972e18f) before the replica check and the analysis.~~ Sealed alone (81ba8691; reference 36c82389; rows and run record 6972e18f) before the replica check and the analysis [r7b]. Replica: pass.

## Verdict: **positive_result_achieved**

Rule: UB95 of TU-FDC-DP(b) / RECT-BF-DP-TU (paired geometric-mean N80_pen) < THRESH = 0.90, AND 0 TU-FDC-DP false streams over the whole run, AND replica pass. THRESH was fixed on dev as min(0.90, dev UB95 + 0.10).

| Comparison | Ratio [95% CI] | UB95 | Faster / tied / slower | Role |
|---|---|---|---|---|
| TU-FDC-DP(b) / RECT-BF-DP-TU | **0.817 [0.808, 0.826]** | **0.824** | 99.5 / 0.5 / 0 % | primary (TU vs TU) |
| TU-FDC-DP(b) / HC-WoR-DP (dev-tuned) | 0.822 | 0.829 | 100 / 0 / 0 % | descriptive (TU) |
| TU-FDC-DP(b) / RECT-HG-DP | 0.826 | 0.833 | 98.5 % faster | descriptive (CP strength) |

| Method | False streams | N80 / τ (geomean) | Share N80 < τ | Mean exhausted fraction at N80 |
|---|---|---|---|---|
| TU-FDC-DP(b) | 0 | 0.726 | 1.00 | 0 |
| RECT-BF-DP-TU | 0 | 0.888 | 0.94 | 0.06 |
| HC-WoR-DP (tuned) | 0 | 0.882 | 0.97 | 0.03 |
| RECT-HG-DP | 0 | 0.879 | 1.00 | 0 |

**How to read this.**
- ~~On a randomized log whose evaluation outcomes never informed the method, the joint certificate certified the whole 15-budget frontier after about 73% of the table, against about 88–89% for every rectangle.~~ [r7b: 73% is the 12-of-15 endpoint, not the whole frontier; see Corrections.]
- That is 18% fewer outcome reads than the matched time-uniform rectangle, with no pool exhausted for FDC-DP (the rectangle reached the horizon, every pool exhausted, on 12 of 200 streams) [r7b].
- ~~**A pre-exhaustion gain.** The comparison meets the lock-v10 condition for a pre-exhaustion width claim: FDC-DP's exhausted fraction is 0, and the rival's is 0.06.~~ [r7b: withdrawn; the lock-v10 condition needs both fractions zero and is not met at block level; see Corrections.]
- **Effect size.** The dev ratio was ~~0.826~~ 0.8255, printed 0.825 [r7b] (UB 0.840). Eval reproduces it: 0.817 (UB 0.824).
- **Strongest tested rival.** All three rival constructions are within 1.1% of each other, so the gap does not depend on which rectangle is named.

## Disclosures (from the lock)

- **Design choices that used dev outcomes.** The arm grouping (top-40 items by dev CTR) and the choice of segmentation among the feasibility candidates both used dev outcomes. The evaluation half was used once, after the lock.
- **Split and dependence.** The split is row-level. 3.3% of dev rows share an exact timestamp with an evaluation row, and user ids are absent, so repeat users can fall in both halves. The halves are therefore ~~weakly dependent~~ not independent, and the log gives no quantitative bound on their dependence [r7b].
- **Changes from the v10 template.** K = 40 (not 20) is a pre-stated change for ratio resolution. Costs are traffic shares, not monetary prices; the log documents no price field.
- **Rival tuning.** HC-WoR-DP was tuned on dev over the block-G3 eight-configuration grid. RECT-HG-DP has checkpoint (CP) strength only.
- **Low base rate.** The base CTR is 0.35%. Each segment × arm cell has roughly 10–700 clicks per half.
- **Scope.** Claims cover the named rectangle constructions on this log.

## Corrections (r7 reviews, added 2026-10-05 in paper revision r7b)

This file is a results note, not a locked file. The lines above that were wrong are struck through and corrected in place; nothing else above was changed. Sources: the external r7 review, `writing/review_r7_critic.md`. Every number below is recomputed from `v11_obd/v11_obd_full/results.jsonl` (paired by seed, bootstrap B = 10^4, numpy `default_rng(42)`) and checked by `writing/scripts/verify_r7b_numbers.py`. The verdict and every registered number are unchanged.

- **Pre-exhaustion.** The lock-v10 condition for a pre-exhaustion width claim (`plan/prereg_lock_v10_addendum.md`, item 6) requires *both* methods' exhausted fractions to be 0. TU-FDC-DP(b) had no exhausted pool at N80 on all 200 streams; both methods had none on 188 streams; RECT-BF-DP-TU reached τ_R = 688,159, with every pool exhausted, on the other 12. The condition is therefore **not met** at block level. The v11 decision rule does not use it.
- **Where FDC-DP stops.** N80 = 415,272 / 471,164 / 534,578 / 606,527 rows (60.3 / 68.5 / 77.7 / 88.1% of the table, checkpoints 35–38) on 11 / 87 / 101 / 1 streams. RECT-BF-DP-TU stops at 606,527 (88.1%, checkpoint 38) on 188 streams and at τ_R on 12. The rival always certifies in the last eighth of the table; FDC-DP does so on one stream. One checkpoint step is a factor of about 1.13.
- **Uncensored subset (descriptive, post hoc).** On the 188 streams where neither method had an exhausted pool, TU-FDC-DP(b) / RECT-BF-DP-TU = 0.821 (UB95 0.829).
- **Full frontier (secondary, descriptive, not the registered endpoint).** Time to 15 of 15: geometric means 572,301 vs 670,998 rows, ratio 0.853 (UB95 0.859). The rectangle reaches 15 of 15 only at τ_R on 160 of 200 streams, so this ratio is horizon-sensitive.
- **Development ratio.** 0.8255 (UB95 0.8402), printed 0.825 in the paper; "0.826" above was a rounding slip.
- **Seal commits.** `81ba8691` committed the seal file alone, `36c82389` its reference, and `6972e18f` the 800 result rows and the run record; `089db6f3` then added the replica and the analysis. The access log has three records: the main run (15:12:18, HEAD 91606679), then the replica (15:19:33) and the analysis (15:20:00), both at HEAD 6972e18f. "Used exactly once" above means this one registered block.
- **Provenance timestamp.** `PROVENANCE.json`'s `split_created_utc` (2026-10-05T03:26Z = 14:26 local) records when the provenance record was last rewritten; the split files (`dev.pkl`, `eval_labels.pkl`, `eval_outcome.npy`) are dated 14:01, about 25 minutes earlier, and predate the feasibility pilot.
- **Public helper.** The frozen `dsswm/envs/obd_v11_eval.py` still exports `verify_eval_bytes` in `__all__`; it hashes the evaluation bytes without its own authorisation check, and the module overview describes the pre-r1 ordering. The registered reader `read_eval_rows` authorises and logs before calling it. The file is lock-bound, so this is documented, not repaired.
- **Validity label.** `v11_obd/v11_analysis.json` labels every method `validity: "rigorous"`, including RECT-HG-DP, whose guarantee is checkpoint strength (CP) only, as the paper and S21 state.
