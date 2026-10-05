# Lock v10: results summary (FDC-DP over exponentially large segment-policy classes)

- **Lock.** `plan/prereg_lock_v10_addendum.{json,md}`, sha256 ecaf78e1…69d82d8. Code freeze 2984121d, lock commit afd64bb9 (2026-10-05). external reviewer reviewed the method (`reviews/fdc_dp_review_r1.md`, B1–B5 fixed) and the lock, in 2 rounds (`reviews/v10_lock_review.md`; the r2 P1 was fixed as specified).
- **Runs.** Eval tasks ran 2026-10-05 06:36–07:20. All 6 tasks were sealed in a commit of their own and passed their replica checks. Analyses: `v10_analysis_{A,B,D}.json`.
- **Primary.** TU-FDC-DP(b): the exact branch-and-bound decision, read under the time-uniform guarantee and monitored at the K = 20 block grid.
- **Governing rival.** RECT-BF-DP-TU: the matched, parameter-free Bennett-FPC rectangle with an exact worst-case gap DP, time-uniform via Lemma TU.
- **Statistic.** N80_pen, paired geometric-mean ratio, 200 fresh streams per cell, paired percentile bootstrap (B = 10⁴, seed 42), one-sided UB95.
- **Decision rule.** Per block, an IUT over cells: UB95 < 0.80 in every cell, AND 0/200 false streams over the whole run to 15/15, AND replica pass.

## Verdicts

| Block | Verdict | Cells (ε) | TU-FDC-DP / RECT-BF-DP-TU (UB95) | TU-FDC-DP / HC-WoR-DP (UB95, descriptive) |
|---|---|---|---|---|
| **A: X5 eval half** (seeds 38000–38199; 3rd use of the table) | **positive_result_achieved** | S = 16 (0.03) | 0.191 (0.193) | 0.216 (0.219) |
| | | S = 32 (0.04) | 0.169 (0.171) | 0.169 (0.171) |
| **B: Lenta eval half** (38200–38399; 3rd fresh-stream use, 1st confirmatory) | **positive_result_achieved** | S = 16 (0.004) | 0.632 (0.636) | 0.508 (0.511) |
| | | S = 32 (0.006) | 0.516 (0.518) | 0.413 (0.415) |
| | | S = 64 (0.008) | 0.513 (0.513) | 0.410 (0.410) |
| D: X5, S = 64 (descriptive; rectangles mostly censored on dev) | descriptive | S = 64 (0.05) | 0.182 (0.184) | 0.173 (0.175) |

- **False streams.** Zero for every method in every cell (whole run to 15/15).
- **Scheme (b) vs scheme (a).** Scheme (b) equals scheme (a) on N80 in 199–200 of 200 streams per cell.
- **Replicas.** Replica status is pass for all six tasks.

## Exhaustion at N80 (mean fraction of pools exhausted; pre-registered qualification)

| Cell | TU-FDC-DP: all / control | RECT-BF-DP-TU: all / control | N80 / τ (FDC-DP vs RECT) |
|---|---|---|---|
| X5 S = 16 | 0 / 0 | 0 / 0 | 0.155 vs 0.814 |
| X5 S = 32 | 0 / 0 | 0.016 / 0.031 | 0.137 vs 0.814 |
| Lenta S = 16 | 0.449 / **0.898** | 0.508 / 1.000 | 0.508 vs 0.803 |
| Lenta S = 32 | 0.011 / 0.021 | 0.500 / 1.000 | 0.413 vs 0.800 |
| Lenta S = 64 | 0 / 0 | 0.500 / 1.000 | 0.410 vs 0.800 |
| X5 S = 64 (D) | 0 / 0 | 0.097 / 0.104 | 0.151 vs 0.829 |

**How to read the exhaustion results.**
- On X5, FDC-DP certifies after reading about 14–16% of the table with no pool exhausted. The rectangles need about 81–83% of the table.
- On Lenta, the rectangle certifies only after **every control pool is exhausted** (Lenta is about 25% control and the design is 50/50).
- At S = 32 and 64, FDC-DP certifies before exhaustion (≤ 2% of control pools). At S = 16 its certification also depends on control exhaustion (90%).
- So the Lenta S = 16 gain is exhaustion-assisted. The S = 32/64 gains are a genuine pre-exhaustion width advantage.

**Lenta S = 64 has zero spread across streams (sd of the log ratio = 0).** Every stream certifies at the same checkpoints:
- for the rectangle, at control-pool exhaustion;
- for FDC-DP, at the same block point on every stream.

The bootstrap UB therefore equals the point ratio. This is checkpoint quantisation, not a computation error.

## Disclosures carried from the lock

**Scope of the claims**
- FDC-DP's statistical certificate is FDC-BF's (Theorem DP-1). The contribution is computational: exact certification over A^S implicit policies without enumeration.
- Claims are restricted to the named rectangle implementations.
- Inference is conditional on the reused finite tables:
  - The X5 eval half is in its 3rd use (v8, v9, v10).
  - The Lenta eval half is in its 3rd fresh-stream use (v6 B, v7 B descriptive) and is outcome-exposed through r4's full-table cell means.
  - The segmentations, scorers and cut points are new and were frozen from the dev halves.

**Choices made on dev**
- Each ε was chosen on dev by the pre-stated rival-success rule, which does not look at FDC-DP's speed.
- HC-WoR-DP is an untuned transfer of the v8 S = 9 configuration (descriptive). On Lenta it never reaches 12/15 before τ.

**Dev deviations and fixes before the lock**
- v1 dev streams stopped at 12/15; v2 dev and v10 eval run to 15/15.
- The weighted-centre bug (B1) was found by external reviewer and fixed before the lock; v1 dev numbers are superseded.
- The integrity scope is that no eval outcome was *used* before the lock. Literal reads are listed in the lock §5.

## Relation to locks v7–v9

Locks v7–v9 confirm FDC-BF on enumerable classes: 9 segments × 2 arms, ≤ 4,096 policies.
- At those sizes the joint certificate needs 0.53–0.71 of the rows of the time-uniform rivals (v9).
- v10 extends the result to 16–64 segments (up to 2^64 policies) via the knapsack DP.
- At these sizes the matched rectangle needs 1.6–5.9× as many rows as FDC-DP, against 1.9–3.3× at 9 segments (v9). *(Corrected 2026-10-05: the first version said 2.0–5.9×; Lenta S = 16 gives 1/0.632 ≈ 1.58.)*

---
**Note added 2026-10-05 (paper revision r6c; append-only, nothing above is edited).** Line 40 above ("The S = 32/64 gains are a genuine pre-exhaustion width advantage") is withdrawn. The rival RECT-BF-DP-TU has a mean exhausted-control fraction of 1 in every Lenta cell, so no Lenta comparison meets the lock's condition for a pre-exhaustion width claim (both fractions zero); the Lenta ratios are exhaustion- and checkpoint-sensitive stopping comparisons, quantised to the K = 20 checkpoint grid. Only X5 S = 16 meets the condition. The corrected interpretation is in the paper's supplement S17 ("How to read the exhaustion columns") and is indexed in the errata (S14).
