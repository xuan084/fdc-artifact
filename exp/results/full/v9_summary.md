# Lock v9: results summary

Lock: `plan/prereg_lock_v9_addendum.{json,md}`, sha256 9540de0c…41fda79, code freeze 29be77b7, lock commit 91977fed (2026-10-05). external reviewer reviewed the addendum in 2 rounds (`reviews/v9_lock_review.md`; r2 verdict "lock"). Eval tasks ran 2026-10-05 02:30–03:53. All 8 eval tasks were sealed in a commit of their own and passed R1 / R2 / R2b / R2c. Analyses: `v9_analysis_{A,B,C}.json`.

Primary statistic: N80_pen, paired geometric-mean ratio TU-FDC@D / rival (200 fresh streams per block), paired percentile bootstrap B = 10⁴ seed 42, one-sided UB95. Every method in the "@D" rows is monitored at the same 77 dense times.

## Verdicts

| Block | Verdict | Governing comparison | UB95 | Threshold |
|---|---|---|---|---|
| **A: Criteo CR9** (ε 0.001, seeds 37000–37199, 4th fresh-stream use of the table) | **positive_result_achieved** | TU-FDC / HC-WoR* | 0.723 | < 0.85 |
| **B: X5 RetailHero** (ε 0.02, seeds 37200–37399, 2nd use of the table) | **positive_result_achieved** | TU-FDC / HC-WoR*; NI vs TU-PJC | 0.379; NI 1.009 | < 0.50; NI < 1.05 |
| C: Hillstrom 3-arm (descriptive) | wording "faster (descriptive)" for design B (main) and design A (supplement) | FDC-BF / tuned rectangles | 0.814 / 0.799 (B) | < 0.95 (wording only) |

TU-FDC had 0/200 false streams in A and in B. Every rigorous method in A and B also had 0 false streams in total, and every Hillstrom method had 0. Replica status is pass for all tasks, including the cross-task R2b / R2 and the cross-grid R2c checks.

## Block A — Criteo CR9 (TU-FDC@D / r; ratio, UB95)

| Rival | Role | Ratio | UB95 |
|---|---|---|---|
| HC-WoR*@D | primary rival (time-uniform, any predictable sampling) | 0.708 | 0.723 |
| RECT-ck-BF-TU@D | primary rival (time-uniform matched Bennett rectangle) | 0.530 | 0.542 |
| RECT-ck-HG*@D | descriptive (K = 77 ledger) | 0.623 | 0.636 |
| PJC-local*@D | descriptive in A | 0.997 | 1.009 |
| TU-PJC@D | descriptive in A | 1.000 | 1.006 |
| FDC-BF[K77]@D | descriptive (fresh widths, K = 77 ledger) | 0.997 | 1.008 |
| TU-LOC@D | descriptive | 1.063 | 1.076 |
| FDC-BF@K (same procedure on the K = 20 grid) | descriptive | 0.959 | 0.968 |

On the K = 20 grid (FDC-BF@K / r):

| Rival | Ratio | UB95 |
|---|---|---|
| HC-WoR*@K | 0.658 | 0.673 |
| RECT-ck-HG*@K | 0.645 | 0.661 |
| RECT-ck-BF+box@K | 0.533 | 0.546 |
| PJC-local*@K | 1.000 | 1.008 |
| FDC-MR[front3] | 1.033 | — |
| FDC-HG | 1.021 | — |
| FDC-LOC | 1.060 | — |

FDC-HG and FDC-LOC were NO-GO candidates and appear only as descriptive rows. On CR9 eval they read about 2% (FDC-HG) and 6% (FDC-LOC) fewer rows than FDC-BF (ratios FDC-BF/r = 1.021 and 1.060), and FDC-MR[front3] about 3% fewer. These gains are small next to the 0.53–0.66 ratios against the rectangles, so the joint geometry, not the cell inequality or the union localisation, carries the speed-up. *(Corrected 2026-10-05: the first version of this paragraph said these variants were "equal to or slightly worse" than FDC-BF, misreading the direction of the ratio; the numbers in the tables were always correct.)*

## Block B — X5 RetailHero (TU-FDC@D / r; ratio, UB95)

| Rival | Role | Ratio | UB95 |
|---|---|---|---|
| HC-WoR*@D | primary rival | 0.374 | 0.379 |
| RECT-ck-BF-TU@D | primary rival | 0.304 | 0.308 |
| PJC-local*@D | non-inferiority (margin 1.05) | 0.985 | 0.992 |
| TU-PJC@D | non-inferiority (margin 1.05) | 1.004 | 1.009 |
| RECT-ck-HG*@D | descriptive | 0.353 | 0.357 |
| FDC-BF[K77]@D | descriptive | 0.982 | 0.988 |
| TU-LOC@D | descriptive | 1.000 | 1.000 |
| FDC-BF@K | descriptive | 0.965 | 0.971 |

On the K = 20 grid (FDC-BF@K / r):

| Rival | Ratio | UB95 |
|---|---|---|
| HC-WoR*@K | 0.364 | 0.369 |
| RECT-ck-HG*@K | 0.365 | 0.370 |
| RECT-ck-BF+box@K | 0.309 | 0.313 |
| PJC-local*@K | 1.002 | 1.011 |
| FDC-MR[front3] | 0.856 | — |
| FDC-HG | 1.025 | — |
| FDC-LOC | 1.000 | — |

Wording permitted by the lock: TU-FDC is faster than the two registered time-uniform rectangles, and not slower by more than 5% than the frozen PJC-local configuration (K = 77 ledger) or its time-uniform form TU-PJC. The lock forbids "faster than PJC"; the TU-PJC point estimate is 1.004.

## Block C — Hillstrom 3-arm (descriptive only; outcome-exposed log)

Design B (main: CZ6 = cat3 × urban, visit, womens e-mail at an **unsourced** half price; ε 0.0125; seeds 37400–37599), FDC-BF / r at N80_pen:

| Rival | Ratio | UB95 |
|---|---|---|
| RECT-ck-HG*[0.2] | 0.807 | 0.814 |
| HC-WoR*[0.2] | 0.791 | 0.799 |
| RECT-ck-BF | 0.698 | 0.700 |
| RECT-ck-HG[bal] | 0.776 | 0.784 |
| PJC-local*[b=4, neyman] | 0.993 | 0.997 |
| FDC-BF[0.5] | 0.850 | — |
| FDC-BF[ney] | 1.004 | — |

Design A (supplement: CR6 / F3 / visit; ε 0.0075; seeds 37600–37799): FDC-BF / every rectangle 0.857 (UB 0.864), / PJC-local* 1.001. The identical ratios across rectangles arise because the rivals pile up at τ_R on this design (the ratios are quantised). This was already disclosed.

## Disclosures carried from the lock

- The time-uniform reading of FDC-BF is a post-hoc theoretical observation applied to an unchanged procedure. The 77-time dense monitoring design was promoted from a secondary experiment to the primary design.
- HC-WoR's guarantee also holds under adaptive (predictable) sampling, whereas TU-FDC and RECT-ck-BF-TU require the frozen, outcome-free schedule.
- Confirmation is conditional on two reused finite tables (fresh streams, not new populations). It covers only the two registered rectangle constructions.
- Thresholds were sized from dev. On dev, the B non-inferiority margin left only about 0.034 of headroom against TU-PJC; on eval the UB was 1.009.
- The toy dense invalid control had 0/200 false streams, so it is uninformative about ledger validity; validity rests on Lemma TU / Theorem TU-1.
- Hillstrom is outcome-exposed; its cost family is unsourced and its design rule was revised twice.
- external reviewer r2 residual P2 (non-blocking): the `word_block_c` helper accepts an arbitrary rectangle set when called without `rectangles`; the registered analysis path supplies the registered set.

## Comparison with earlier locks (same tables, different guarantee reading)

| Lock | Table | Comparison | Ratio | UB95 |
|---|---|---|---|---|
| v7 A (K = 20, checkpoint guarantee) | CR9 | FDC-BF / HC-WoR* | 0.655 | 0.670 |
| v9 A (dense, time-uniform) | CR9 | TU-FDC / HC-WoR* | 0.708 | 0.723 |
| v8 A | X5 | FDC-BF / HC-WoR* | 0.362 | 0.367 |
| v9 B | X5 | TU-FDC / HC-WoR* | 0.374 | 0.379 |

Under dense monitoring HC-WoR gains more than TU-FDC, because its fresh time-uniform widths can stop between block points. Matching the guarantee strength therefore costs about 5 points of the ratio on CR9 and about 1 point on X5.
