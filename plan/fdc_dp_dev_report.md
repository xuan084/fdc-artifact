# FDC-DP dev report (2026-10-05, authors)

> **Status: the v1 numbers below are SUPERSEDED by "v2 (post-external reviewer fixes)" at the end of this file.** v1 used the
> unweighted empirical centre (external reviewer FDC-DP r1 B1), stopped streams at 12/15 although the theory note prespecified
> stop_k = 15 (B2), counted "uncensored" with a mislabelled field and claimed identical (a)/(b) results (B2). The v1
> raw results stay unchanged in `exp/results/pilots/fdc_dp/` (code 63b17172).

## v1 (superseded)

**Setup.** Dev seeds 950–999 (50 streams), dev halves only. Code 63b17172 (commit 115d7bb7). Analysis in `exp/results/pilots/fdc_dp/analysis.json`. Theory and go criteria are in `plan/fdc_dp_theory.md`; the ε-selection rule was pre-stated in `plan/v10_plan.md` before the ε ≥ 0.03 runs.

## ε chosen by the rule
The rule picks the smallest ε on the grid at which the best valid rectangle has N80 < τ on at least 80% of dev streams.

| Table, S | Rule ε | Rectangle uncensored (RECT-BF-DP / HC-WoR-DP) | FDC-DP(b) / best rectangle, N80_pen ratio (UB95) |
|---|---|---|---|
| X5, 16 | 0.03 | 50/50 / 50/50 | 0.249 (0.255) |
| X5, 32 | 0.04 | 50/50 / 50/50 | 0.221 (0.224) |
| X5, 64 | none on the grid (0.05: 6/50 / 0/50) | — | descriptive: rectangles censored; FDC-DP/τ ≈ 0.20 at ε 0.05 |
| Lenta, 16 | 0.004 | 50/50 / 0/50 | 0.665 (0.665) |
| Lenta, 32 | 0.006 | 50/50 / 0/50 | 0.601 (0.616) |
| Lenta, 64 | 0.008 | 50/50 / 0/50 | 0.596 (0.611) |

## Other results
- **False certificates:** 0 for every method, cell and stream.
- **Scheme (b) vs (a):** scheme (b), the exact branch-and-bound, gave the same result as scheme (a), single λ per check, in every cell (ratio 0.993–1.000).
- **Runtime at S = 64:** about 0.4 s per checkpoint, against a target of under 10 s.

## Go criteria (plan/fdc_dp_theory.md)
1. Validity proof: done (Theorem DP-1).
2. Exactness against enumeration at S ≤ 9: passed, to 1e-9.
3. Runtime: passed.
4. Dev UB95 < 0.80 against the best rectangle at S ≥ 32 on at least one table: passed on both tables.

**Verdict: GO.**

## Caveats to carry into lock v10
1. **HC-WoR-DP is untuned at large S.** Its configuration is the frozen v8 S = 9 setting, transferred untuned. It never certifies on Lenta at the rule ε, so RECT-BF-DP (parameter-free, matched) is the governing rival.
2. **Lenta exhausts its control pools.** Lenta is about 25% control, so the 50/50 design exhausts control pools and certification is exhaustion-driven. Quantised ratios (UB = ratio at Lenta, S = 16) reflect checkpoint quantisation.
3. **Table reuse.** The X5 eval half has been used twice (v8, v9) and the Lenta eval half once (v7 B, descriptive). The new segmentations are new problems on reused rows.
4. **X5 at S = 64 is descriptive only.** Rectangles cannot certify 12/15 problems before exhaustion at any ε on the grid.


## v2 (post-external reviewer fixes), 2026-10-05

**What changed (external reviewer `reviews/fdc_dp_review_r1.md`, verdict "fix first").**
- **B1:** FDC-DP and RECT-BF-DP now use the weighted empirical centre `dp_argmax(w[:, None] * mu_hat)`; regressions
  (external reviewer counterexample, end-to-end weighted-centre enumeration) added to `dsswm/tests/test_fdc_dp.py`.
- **B2:** streams now run to stop_k = 15 (or τ); N80 is recorded at 12/15 from the same trajectory and a false stream
  is any false certificate over the whole run. **Dev deviation, disclosed:** v1 stopped at 12/15. The ε-selection rule
  is implemented in code (`run_fdc_dp_dev.select_eps`): full declared grid, exactly 50 unique seeds per rectangle and ε,
  strict N80_pen < τ, dev-best of the two named rectangles (tie → RECT-BF-DP), exported selection / rival / successes /
  fallback. The success count is "N80_pen < τ" (strict); the v1 column labelled "uncensored" meant exactly this.
  Runtime and node-cap statistics are part of the automated gate.
- **B3:** scheme (b) is an exact decision when the search completes and abstains (not certified) at the node cap; the
  reported U is the scheme-(a) upper bound U_a, and `last_detail` separates U_a, U_lower (violator leaf), U_exact
  (diagnostic mode) and the decision source; forced-cap and branch-only regressions added.
- **B4:** fresh-width FDC-DP / RECT-BF-DP calls off the predeclared checkpoint grid raise; dense monitoring only via the
  TU wrappers (v10 does not use dense monitoring).
- Code `e6061c609ef49ca5` (runner hash over the four runner files; commit 45ea10c4 + the runner change), dev seeds
  950–999 (50 streams), 4 methods × 4 ε × 6 tasks = 4,800 rows, 0 errors; `exp/results/pilots/fdc_dp_v2/analysis.json`.

**Result: every v1 number is reproduced.** On these segmentations the segment sizes are equal to ±1 row, so the weighted
and unweighted centres coincide on every stream, and no false certificate occurred between 12/15 and 15/15.

### ε chosen by the rule (applied mechanically)
| Table, S | RECT-BF-DP successes (N80_pen < τ, of 50) on the grid | Selected ε | FDC-DP(b) / RECT-BF-DP, N80_pen ratio (UB95) |
|---|---|---|---|
| X5, 16 | 0.02: 0; **0.03: 50** | 0.03 | 0.249 (0.255) |
| X5, 32 | 0.02: 0; 0.03: 0; **0.04: 50** | 0.04 | 0.221 (0.224) |
| X5, 64 | 0.02: 0; 0.03: 0; 0.04: 0; 0.05: 6 | none: descriptive | (0.201 (0.206) at 0.05; rivals censored) |
| Lenta, 16 | 0.003: 28; **0.004: 50** | 0.004 | 0.665 (0.665) |
| Lenta, 32 | 0.003: 0; 0.004: 2; **0.006: 50** | 0.006 | 0.601 (0.616) |
| Lenta, 64 | 0.003: 0; 0.004: 0; 0.006: 1; **0.008: 50** | 0.008 | 0.596 (0.611) |

The dev-best rectangle is RECT-BF-DP at every grid ε of the selection trace (HC-WoR-DP never had more successes).
HC-WoR-DP is an untuned transfer of the v8 S = 9 configuration; claims are restricted to the two named rectangle
implementations, not "the best valid rectangle".

### Gate (automated)
- **Validity:** 0 false streams for every method, cell, ε and stream, over the whole run to 15/15; every stream of
  FDC-DP(b) reached 15/15.
- **Criterion 3 (runtime, S = 64, both tables, every ε):** FDC-DP(b) median 0.29–0.33 s and maximum ≤ 0.46 s per
  checkpoint (all 15 problems), FDC-DP(a) median 0.18–0.22 s, node-limit hits 0 of 7,289–13,432 B&B calls per cell:
  **pass**.
- **Criterion 4:** at the v1 primary ε (X5 0.02, Lenta 0.003; S ≥ 32): X5 S=32 UB 0.463, X5 S=64 UB 0.571 → pass; at
  the rule-selected ε: X5 S=32 UB 0.224, Lenta S=32 / 64 UB 0.616 / 0.611 → pass.
- **Scheme (b) vs (a):** stopping performance nearly identical (ratio 0.993–1.000); scheme (b) produced 25
  branch-only certificates across the 24 (task, ε) cells (external reviewer counted 27 in the v1 logs). The v1 claim "same
  result in every cell" was too strong.

### Lenta exhaustion qualification (mean fraction of exhausted cells at the N80 checkpoint)
| Cell | FDC-DP(b): all / control / treatment | RECT-BF-DP: all / control / treatment | N80/τ FDC-DP(b), RECT-BF-DP |
|---|---|---|---|
| Lenta 16 @ 0.004 | 0.50 / 1.00 / 0.00 | 0.50 / 1.00 / 0.00 | 0.543, 0.816 |
| Lenta 32 @ 0.006 | 0.25 / 0.50 / 0.00 | 0.50 / 1.00 / 0.00 | 0.490, 0.816 |
| Lenta 64 @ 0.008 | 0.23 / 0.46 / 0.00 | 0.50 / 1.00 / 0.00 | 0.486, 0.816 |
| X5 16 @ 0.03 | 0 / 0 / 0 | 0.02 / 0.05 / 0.00 | 0.206, 0.829 |
| X5 32 @ 0.04 | 0 / 0 / 0 | 0.02 / 0.03 / 0.00 | 0.183, 0.829 |

On Lenta the rectangle certifies only after every control pool is exhausted (0.816 τ on every stream), and at S = 16
FDC-DP(b) also needs every control pool exhausted; at S = 32 / 64 about half of the control pools are exhausted when
FDC-DP(b) reaches 12/15. The Lenta comparisons are therefore exhaustion- and checkpoint-sensitive finite-table
comparisons, not evidence of a general pre-exhaustion width advantage; X5 certifications are pre-exhaustion for
FDC-DP(b). Because ε changes with S, the selected-ε ratios are not a controlled scaling experiment.

### Verdict v2
GO is unchanged: criteria 1–4 pass with the fixed code. Lock v10 registers the rule-selected cells (X5 S=16 @ 0.03,
S=32 @ 0.04; Lenta S=16 @ 0.004, S=32 @ 0.006, S=64 @ 0.008) with X5 S=64 @ 0.05 descriptive.
