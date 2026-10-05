# external reviewer review: FDC-HG theorem + implementation (2026-10-04)

Reviewer: external reviewer MCP (model from ~/.reviews/config.toml, no model parameter, approval-policy never, read-only sandbox).
Thread 01a10532-217c-7fc3-9486-821e368a3b5c. Two rounds (task cap). Objects reviewed: `plan/fdc_hg_theorem.md`,
`exp/code/dsswm/baselines/fdc_hg.py`, `exp/code/dsswm/tests/test_fdc_hg.py`, parent `fdc_bet.py`.

## Round 1 — verdict INVALID (as a rigorous numerical certificate; mathematics sound)

Mathematics confirmed: conditional product MGF, Chernoff with strict-tail limit, union ledger, Lemma T tail directions,
Lemma R identity + MLR monotonicity + [M1, M2−1] ranges + negative-tilt sign reversal + two-line max, Lemma C chords
(incl. origin chord), exhausted / empty cells, pointwise dominance for matching non-NaN inputs.

| # | Severity | Finding | Resolution (v2) |
|---|---|---|---|
| 1 | blocking | Fixed relative margin did not cover scipy `hypergeom.logpmf` error at N = 10⁶ (Λ underestimated by 2.7e-10 on a two-point support); h bounds could cross the truth | pmf rebuilt from the exact neighbour-ratio recurrence with explicit per-term error radii, normalised on the same window; directed h bounds; mpmath oracle tests. (scipy logpmf itself is biased by ~2e-9 at N = 2.66e6; new code brackets mpmath there.) |
| 2 | major | Bennett operand `expm1(x) − x` cancels at tiny x (returns 0 for a positive truth) | `bennett_sup_safe` with a series upper bound for x < 1e-3 and outward factors |
| 3 | major | Float re-integerisation `ceil(N lo − 1e-6)` can exclude the true M at N = 1e11 | FDCHG tracks the HG box in integer counts and passes integers |
| 4 | major | NaN at extreme tilts (`log1p(−1)`, `expm1` overflow) in Lemma R | `log((1−p)+pe^t)` via logaddexp; non-finite → +inf |
| 5 | minor | Proof (iv): inequality for arbitrary λ̂ vs grid minimum | Rewritten: bound holds at every grid λ, then take the min |
| 6 | minor | 1e-3 tightness only if B&B converged | Stated as conditional; converged flag exposed |

## Round 2 — verdict INVALID (residual floating-point enclosure gaps; mathematics and `_box` equivalence confirmed)

Confirmed: revised Chernoff step; normalised Λ upper bound needs only the untilted core lower bound; tail / moment
directions; integer `_box` bitwise-equal to FDC-BF's float box (external reviewer checked 11 checkpoints × 4 cells incl. empty,
exhausted, pure pools).

| # | Severity | Finding | Resolution (v3, final) |
|---|---|---|---|
| 1 | blocking | Bennett variance from float m: `1 − m` underestimates when m ≈ 1 (N = 10⁷, M = N − 1) | `var_sup_int`: exact integer M(N−M)(N−n)n / (N²(N−1)) via Fraction, outward 4ε |
| 2 | major | `eq` radius did not cover global-prefix-sum-then-subtract (6.3e-12 actual vs 2.1e-13 claimed) | log-weights accumulated left / right **from the anchor**, matching the radius derivation |
| 3 | major | Underflow: h_hi = 0 at t = −750; φ(1e-162) = 0 | absolute floors (h_hi + 1e-299, φ + 1e-300, Λ / interval / chord + 1e-300) |
| 4 | minor | Optional float-recovery path still excludes at N = 1e11 | Fraction floor/ceil outward recovery, N < 2⁵⁰ enforced; production path passes integers |
| 5 | minor | `converged` ≠ tolerance met | converged := max(up − best_lo) ≤ tol |
| 6 | minor | Dominance vs frozen FDC-BF not universal at tiny u (FDC-BF's own cancellation) | Theorem doc: constructive dominance only vs a BF with the same validated operand; vs frozen FDC-BF it is an empirical check (unit test on regular ranges, per-stream N80 check in dev) |

Every external reviewer counterexample from both rounds is a regression test in `test_fdc_hg.py` (21 tests, all pass). The dev
run was restarted after each round's fixes; all reported FDC-HG rows come from the v3 code (earlier rows kept only in
`exp/results/pilots/fdc_hg/cr9/results_*discard*.jsonl` / `results_v0_prefix.jsonl`, not analysed).

## Assessment after round 2

No issue found by external reviewer in either round touched the probabilistic argument; all were floating-point enclosure gaps of
size ≤ 1e-9 in Λ (orders of magnitude below any effect on certification). The cap of two external reviewer rounds was reached, so
the v3 fixes were not re-reviewed by external reviewer; they were verified by the mpmath / long-double regression tests. The
residual caveat, stated in the theorem doc, is that the floating-point analysis is a forward-error argument with
conservative constants (assumes ≤ 1-ulp `log`), not interval arithmetic.
