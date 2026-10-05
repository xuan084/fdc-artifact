# v11 workstream (main session, 2026-10-05): can the union cost be localised? (critic r6 P0-1)

## Starting point

- **r6 reviews.** external reviewer scored 7/10 (`reviews/paper_r6_review.md`): accept on scientific merit, bounded edits only. The critic scored 6/10 (`writing/review_r6_critic.md`).
- **Critic's route to 7 (P0-1).** A class-level threshold computed by the same DP that "tames the union cost growing with S" (β_J/β_C = 4.50 at S = 64).
- **external reviewer's alternative route to 8.** "A rigorously smaller complexity-dependent ledger with an efficiently evaluable certificate."
- **Bounded fixes.** All bounded r6 fixes went to a separate agent (revision r6c).
- **Status of the constructions below.** All are development-only. No evaluation row was read and no lock was involved.

## Constructions tried (all valid; tests in `exp/code/dsswm/tests/test_fdc_{pw,ls}.py`, all passing)

| # | Construction | Code | Dev result (seeds 950–999; ratio vs TU-FDC-DP(b), N80 geomean) |
|---|---|---|---|
| 1 | **Prior-weighted ledger (FDC-PW).** Frozen reference policy from the v10 score model (covariates only); Hamming-shell weights around it; candidate centres {argmax, reference}. | `dsswm/baselines/fdc_pw.py`, `dsswm/envs/pw_reference.py` | X5 S16: 1.04–1.09; X5 S32: 1.01–1.05; Lenta S16: 1.01–1.07 (all 6 ε complete); X5 S64: 1.00 (ε 0.02, 0.03 complete; 0.04 stopped at 33 streams); Lenta S32 and S64: 1.00 (ε 0.003 complete; 0.004 stopped at 14 and 46 streams). Rows: `exp/results/pilots/fdc_pw/rows_v1_twocentre/`. **No gain.** |
| 1b | Same ledger, candidate centres = argmax restricted to Hamming balls around the reference (one knapsack DP with a flip state). | `fdc_pw.py` (`centres="ball"`) | Not run on dev: the diagnostic below showed it cannot help. |
| 2 | **Shell-localised ledger around the unknown optimum (FDC-LS).** Weights ½·uniform + ½·ρ_d/(Q·C(S,d)) with d = Hamming(π, π*_q). The challenger-dependent β_d is evaluated exactly by a knapsack DP with a flip-count state. Worst case costs ln 2 over FDC-DP. β_1 = 15 against β_J = 52 at S = 64. | `dsswm/baselines/fdc_ls.py` | X5 S16: 1.02–1.05; Lenta S16: 1.00–1.03; X5 S32: 1.01–1.03; Lenta S32: 1.00–1.02 (S16 and S32 complete: 50 seeds × 6 ε each). S64 stopped after 3 streams at the smallest ε (slow, flat trend). Rows: `exp/results/pilots/fdc_pw/rows/`. **No gain.** |
| 3 | **Thresholded-sum event** (one Chernoff event for Σ_s(\|ξ_s\| − c_s)_+, giving an additive column). | Gaussian calculation only | Even with the exact Gaussian MGF, the width at S = 64 is 0.98–1.25× the union's for \|D\| ≥ 16 (reads 0.96–1.56×). The only gain is for \|D\| = 8 (reads 0.79×), which is not the binding regime. **Not built.** |

## Why none of them helps (dev diagnostics)

**The score-model reference is far from the optimum.** The Hamming distance from the frozen-score reference policy to the true dev optimum π*_q is about S/2 at S = 64 (X5: 10–34; Lenta: 6–32). Its value gap is 0.004–0.014 on X5 and 0.001–0.005 on Lenta. So the prior is ε-good in value but far in policy space. The certificate width grows with the number of segments in which the challenger and the centre differ, so a far reference never certifies.

**The binding challengers differ from the centre in many segments.** On these near-tie tables most (problem, policy) pairs are ε-optimal (v9 census: 3,076 of 3,347 on X5 dev). The challengers that maximise Δ̂ + W therefore differ from the centre in many near-tie segments. Shell localisation lowers β only for small Hamming distance, so it never touches the binding challengers. That is why FDC-LS ties FDC-DP.

**Data-dependent centres cost about a factor of two in any union-free bound.** Because the centre is chosen from the data, a union-free bound must control Σ_{s∈D}|ξ_s|, not Σ_s (ξ_s)_+. The mean of the former is twice the latter (about 0.8σ against 0.4σ per segment). The Chernoff union over (q, π, k) is then within about 1× of the best thresholded-sum bound at S ≤ 64.

## Conclusion

On these tables, the union price that FDC-DP pays is not an artefact that a localised or class-level ledger removes within the certificate family. Its source is the near-tie geometry together with data-dependent centres. This is a negative development result. It belongs in the supplement next to the converse remarks (S18) as descriptive context. It is not a new contribution, and it is not a reason to change the confirmatory claims.

## Consequence for the 10-15 decision

- The critic's P0-1 (new class-level threshold) has no viable construction that could be built and locked before 10-15.
- After r6c the expected scores are external reviewer 7 and critic 6 ("clean, strong 6").
- Under the user's rule (both ≥ 7), WWW'27 would then not qualify, and the project retargets to the next same-tier venue.
- The r6c critic re-review decides this. If it stays at 6, plan the retarget around external reviewer's route to 8: one prospectively specified scaling study on a never-replayed randomized log with documented heterogeneous costs. *(Corrected 2026-10-05, flagged by the r6c critic: the full Criteo release is NOT a candidate, because all 13,979,592 rows were already split into dev/eval and used (see `shared/datasets/criteo_full/PROVENANCE.json`); MegaFon is synthetic.)* The only never-replayed real randomized log found so far is the Open Bandit Dataset (ZOZOTOWN, uniform-random logging, CC BY 4.0). It was downloaded 2026-10-05 and has not been read (`shared/datasets/open_bandit/PROVENANCE.json`). Its fit (80 item arms, very low click rates, no documented costs) still has to be assessed.

## Note added 2026-10-05 (paper revision r6d)

The coverage figures in the table above were corrected from the rows (the r6c critic noted that the FDC-LS S = 32 cells are complete). The paper reports this workstream in supplement S20 at its tested scope: the ratios are development-only descriptions of three constructions on these tables, not a lower bound. The two general statements in "Why none of them helps" and "Conclusion" (that no localised or class-level ledger can remove the price within the certificate family, and that data-dependent centres cost a factor of two in any union-free bound) are conjectures suggested by the diagnostics; neither is proved here, and the paper does not make them.
