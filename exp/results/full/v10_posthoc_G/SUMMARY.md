# Block v10-posthoc-G: results summary (post hoc, descriptive, outside any lock)

**Purpose.** Answers the critic's r5 points R5-P0-2/3:
- Does the FDC-DP advantage depend on the ε-selection rule that pins the rival at a checkpoint?
- Does branch-and-bound matter, and does FDC-DP beat stronger rivals?

The rules, the eval access path, RECT-HG-DP and the runner were committed before any G stream (7d8efa47). The HC-WoR-DP tuning was selected on dev before any G3 eval row (db792bf5).

**Code.**
- Rows are keyed by code_sha 02f90692448956a9.
- G1 dev reproduces the lock-v10 pilot rows exactly (150/150 identical per cell; `summary.json` → `checks`).

**Integrity deviation.**
- The first eval launch (09:03–09:13, 2026-10-05) wrote gated access records for 6 cells, then crashed while writing cell metadata, before computing any eval row. A missing `describe()` on the post-hoc env caused the crash.
- Fixed in runner metadata only, outside the row hash (e0ea168c). All 12 eval cells were relaunched at 11:04.
- `eval_access_log.jsonl` therefore shows two access records for those 6 cells.

**False streams.** Zero for every method in every G cell, dev and eval:
- 8,400 eval streams each for TU-FDC-DP(b) and RECT-BF-DP-TU;
- 7,200 eval streams for HC-WoR-DP;
- 1,200 eval streams each for RECT-HG-DP and the tuned HC-WoR-DP.

## G1: reads against ε (eval seeds 38400–38599, 200 streams per (cell, ε))

**Advantage against ε.** The ratio TU-FDC-DP(b) / RECT-BF-DP-TU falls as ε grows (monotone except Lenta S16, where 0.621 → 0.623 between ε 0.003 and 0.004), in every (table, S):

| Table | ε range | Ratio |
|---|---|---|
| X5 | 0.02 → 0.08 | 0.29–0.50 → 0.08–0.09 |
| Lenta | 0.003 → 0.012 | 0.62–0.80 → 0.29–0.37 |

**On the tested grid the lock ε gives the largest ratio among the ε at which the rival passes the 80% pre-horizon criterion.** The lock rule picked the smallest ε at which the rival reaches 12 of 15 strictly before τ on at least 80% of development streams. Larger ε, where the rival is no longer stuck at τ or pinned to a single checkpoint, gives FDC-DP a larger lead. On this finite grid, then, the ε rule did not inflate the confirmatory ratios (X5 S64's lock ε was descriptive, since no development ε qualified); this says nothing outside the grid.

**Unpinned cells.** "Pinned" means ≥ 95% of RECT streams have the same k80. In the cells where RECT is not pinned, the ratio vs RECT is as below. Some of these are censored (RECT reaches N80 before τ on < 80% of streams: X5 S16 ε 0.02, X5 S32 ε 0.03, Lenta S32 ε 0.004). Passing the 80% pre-horizon criterion and unpinned: X5 0.093–0.183, Lenta 0.336–0.446.

| Cell | Ratio vs RECT |
|---|---|
| X5 S16, ε 0.02 | 0.294 |
| X5 S16, ε 0.05 / 0.06 | 0.119 / 0.093 |
| X5 S32, ε 0.03 | 0.224 |
| X5 S64, ε 0.05 | 0.183 |
| Lenta S16, ε 0.006 / 0.01 | 0.446 / 0.336 |
| Lenta S32, ε 0.004 / 0.008 | 0.557 / 0.420 |
| Lenta S64, ε 0.012 | 0.373 |

Full table, with 95% CIs: `table.tex`. Figure: `curves.{pdf,png}`.

## G2: heterogeneous knapsack costs (development halves only; dev seeds 950–999, 50 streams per (cell, ε); descriptive)

**Costs.**
- X5: seeded lognormal costs.
- Lenta: segment mean discount depth (a covariate-only promotion-cost proxy).
- Both use integer weights κ ∈ [1, 24] (X5 S64 has κ = 24 at segment index 20; corrected in r6c, see the errata note below).

**Exact branch-and-bound (b) vs min–max (a).**
- Scheme (b) gives the same N80 as scheme (a) in all 36 (cell, ε) pairs (ratio 1.000).
- Scheme (b) certifies a problem that (a) cannot in 0–3 of 750 problem instances (50 streams × 15 problems) per pair; summed over the six ε, 0–8 branch-only certification events among 4,500 problem instances per row.
- The node limit was never hit.
- Maximum time is 0.17 / 0.45 / 1.8 s per checkpoint at S = 16 / 32 / 64 (maxima across the two logs: 0.1737, 0.4512, 1.7777 s).

**Conclusion.** Under these knapsack constraints the conservative single-λ scheme is practically exact. Branch-and-bound is an exact-decision refinement (scheme (a) is already statistically sound), not a speed source, so the paper must not claim speed from it.

**FDC-DP against the rectangles under knapsack costs.**
- FDC-DP(b) / RECT-BF-DP: 0.09–0.82. Values near 0.8 occur only where the rectangle is censored at τ.
- FDC-DP(b) / HC-WoR-DP: 0.10–0.82.
- So the advantage holds beyond cardinality budgets.

## G3: stronger rivals at the lock-v10 cells (eval seeds 38600–38799)

Ratio TU-FDC-DP(b) / rival (one-sided UB95):

| Table | S | ε | vs RECT-BF-DP-TU | vs RECT-HG-DP | vs HC-WoR-DP (tuned on dev) |
|---|---|---|---|---|---|
| X5 | 16 | 0.03 | 0.188 (0.191) | 0.223 (0.227) | 0.200 (0.204) |
| X5 | 32 | 0.04 | 0.168 (0.170) | 0.168 (0.170) | 0.168 (0.170) |
| X5 | 64 | 0.05 | 0.182 (0.185) | 0.187 (0.188) | 0.152 (0.153) |
| Lenta | 16 | 0.004 | 0.629 (0.634) | 0.633 (0.636) | 0.632 (0.636) |
| Lenta | 32 | 0.006 | 0.518 (0.521) | 0.519 (0.523) | 0.518 (0.521) |
| Lenta | 64 | 0.008 | 0.513 (0.513) | 0.513 (0.513) | 0.513 (0.513) |

**Fresh seeds.** On fresh eval seeds 38600–38799, the RECT-BF-DP-TU ratios replicate lock v10 to within ±0.003.

**Rival ranking.** Tuning HC and the exact-hypergeometric rectangle shift the ratios by about 0.035 at most (0.0353, RECT-HG-DP, X5 S16). Neither closes the gap. Tuned HC is not uniformly stronger than the matched rectangle: at X5 S64 its ratio is 0.152 against 0.182, because it reaches 12 of 15 before τ on no stream. The Lenta cells stay pinned at rival control-pool exhaustion, as disclosed in v10.

## G4: union cost against S

| S | β_J | β_C | β_J/β_C |
|---|---|---|---|
| 9 | 14.21 | 9.68 | 1.47 |
| 16 | 19.08 | 10.26 | 1.86 |
| 32 | 30.16 | 10.95 | 2.75 |
| 64 | 52.34 | 11.64 | 4.50 |

The joint ledger's price grows linearly in S, whereas the per-cell price grows only logarithmically. The observed advantage does not shrink as S grows from 16 to 64, so the joint certificate's geometry pays for a ledger that is 4.5× larger at S = 64.

## Errata (paper revision r6c, 2026-10-05)

Corrected after the r6 reviews; no row, ratio or table value changed.
- G2 cost range: the generated costs span 1–24, not 1–22 (`summary.json` → `G2` → `G2-dev-x5-64` → `costs`; `rows/G2-dev-x5-64.meta.json`). The runner applies the stated rule `max(1, rint(8 c_s / mean c))` without an upper cap, so only the prose was wrong.
- Branch-only denominator: the 4,500 (750 per ε) counts final problem certifications, 50 streams × 15 problems (× 6 ε), i.e. problem instances; it is not the number of problem-checkpoint tests (`a_certs + b_only_certs = 750` per ε).
- S = 16 runtime: the cross-log maximum is Lenta's 0.1737 s, not X5's 0.11 s.
- Wording: scheme (b) is an exact-decision refinement rather than a "correctness device"; G1's lock-ε observation holds on the tested grid, at points that pass the 80% pre-horizon criterion, and X5 S64's lock ε was descriptive (no qualifying development ε); hollow (censored) G1 points are horizon-sensitive endpoint ratios, with no asserted bias direction; RECT-HG-DP is checkpoint-valid only (CP), HC-WoR-DP is time-uniform (TU). The paper's supplement S19 carries the corrected text.
- r6d (2026-10-05): the "least favourable uncensored point" heading and the "uncensored and unpinned" shorthand above were reworded in place to the 80% pre-horizon criterion (external reviewer r6c review, section 7); no number changed.

