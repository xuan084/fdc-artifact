# v9 candidates: localised union (a) and time-uniform FDC (b): dev report and go / no-go

Date 2026-10-04.

| What | Where |
|---|---|
| Theory (written and committed before any run, 61ce3930; v2 adds external reviewer fixes) | `plan/v9_candidates_theory.md` |
| Code | `exp/code/dsswm/baselines/fdc_loc.py` (FDCLoc, FDCTimeUniform, dense_grid) |
| Tests | `exp/code/dsswm/tests/test_v9_candidates.py` (8 tests, all pass) |
| Runner | `exp/code/run_v9_candidates_dev.py` |
| Analysis | `exp/code/analyze_v9_candidates_dev.py`, output `exp/results/pilots/v9_candidates/analysis_dev.json` |
| external reviewer review (2 rounds) | `reviews/v9_candidates_review.md` |

**Discipline.**
- Data: dev halves only. Report seeds 950–999 (asserted). Toy population seeds 900–919.
- Nothing was tuned: neither candidate adds a statistical hyperparameter (FDC-LOC inherits FDC-BF's split, λ grid and boxes), so no tuning seeds were used.
- Comparator rows (FDC-BF, RECT-ck-HG*, HC-WoR*, RECT-ck-BF, PJC-local*) on the K = 20 grid are reused from the FDC-HG dev run. That run used the same frozen code, the frozen v8 dev configs and the same seeds.
- No locked file, eval half or `writing/` file was touched.

## 0. Verdicts

| Candidate | Pre-stated go criterion | Result | Verdict |
|---|---|---|---|
| (a) FDC-LOC, gap-localised union | FDC-LOC/FDC-BF UB95 < 0.90 on one of {CR9, X5 ε = 0.02} and ≤ 1.00 on the other; 0 false streams | CR9 0.925 (UB 0.949), X5 1.000 (UB 1.000); 0 false streams | **NO-GO** |
| (b) TU-FDC, time-uniform via reverse martingale | TU-FDC/HC-WoR* UB95 < 0.80 on both CR9 and X5 ε = 0.02 (K = 20 grid); 0 false streams (dev + dense-monitoring toy) | CR9 0.663 (UB 0.695), X5 0.329 (UB 0.342); 0 false streams | **GO** |

**Recommendation for the v9 lock primary.**
- **Primary method:** FDC-BF, unchanged.
- **Guarantee, restated:** time-uniform over all t ≥ t₁ under the frozen schedule (Theorem TU-1). Evaluated on the K = 20 grid, the procedure is the same algorithm, so no number changes and the guarantee strengthens at zero cost.
- **FDC-LOC:** report it as a valid, dominating, descriptive ablation (CR9 −7.5%), together with the oracle ceiling, which is a diagnostic of the near-tie structure.

## 1. Theory in brief (full statements and proofs in `plan/v9_candidates_theory.md`)

### (a) FDC-LOC, Theorem LOC-1 (external reviewer: valid; bookkeeping fixes applied)

**Construction.** At each checkpoint, β_J = ln(MK/δ_main) is replaced by a data-dependent β̂_k, the smallest β with Ĝ_k(β) ≤ δ_main/K, where

  Ĝ_k(β) = Σ_{q,π} exp(−β(1 + (Δ^lo − ε)₊ / W̄(β)))

- Δ^lo is a pairwise-rectangle gap lower bound.
- W̄ is a Bernstein upper bound on the width, maximised over the survivors.
- Policies whose gap upper bound is ≤ ε are dropped.
- Both bounds are computed on the exact-HG variance boxes FDC-BF already pays for, so the screen spends no δ.

**Why it is valid.**
- A deterministic comparator G_k(β), built from the true gaps and the true ideal widths t*, satisfies Ĝ_k ≥ G_k on E_var. Hence β̂_k ≥ β_k^det, and the data-dependent β̂ never enters a probability statement.
- A false certification needs a deviation that exceeds the width by the excess gap. By concavity of t*(β) with t*(0) = 0, that event costs exp(−β(1 + x/t*)).
- **Dominance:** β̂_k ≤ β_J always.

### (b) TU-FDC: Lemma TU and Theorem TU-1 (external reviewer: lemma valid; theorem valid after notation fixes)

**Lemma TU.** Fix the schedule. Each WoR cell mean is a reverse martingale in its count (Serfling 1974; Bardenet & Maillard 2015). Independent cells with deterministic nondecreasing counts therefore make Y_t = Σ_c a_c(μ_c − μ̂_c(t)) a reverse martingale for t ≥ t_k. Doob then gives

  P(sup_{t≥t_k} Y_t ≥ y) ≤ inf_λ e^{−λy} E e^{λY_{t_k}}.

**Consequences.**
- FDC-BF's per-(q, π, k) Chernoff event extends to all t ≥ t_k with the same ledger. At any time t ≥ t₁, the certificate uses the width frozen at the last block point.
- On the block grid this is numerically identical to FDC-BF (unit test `test_tu_on_block_grid_equals_fdc_bf`).
- The lemma also upgrades RECT-ck-BF.
- It does **not** upgrade the exact-HG quantile rectangle RECT-ck-HG. the external reviewer's counterexample: N = 4, M = 2, n₀ = 2 gives P(Y₂ > 0) = 1/6 but P(max_n Y_n > 0) = 1/2.
- **Remaining asymmetry, to disclose:** HC-WoR is uniform under any predictable sampling rule. TU-FDC is uniform under the frozen outcome-free schedule, which is the one both methods use here.

## 2. Validity checks

| Check | Result |
|---|---|
| Toy, K = 20 grid, 200 streams, ε = 0.02 | FDC-LOC 0/200 false streams, FDC-BF 0/200, invalid NAIVE-joint **5/200** (12 false certifications) |
| Toy, 4×-dense monitoring (77 evaluation times) | TU-FDC[K20] 0/200, TU-LOC[K20] 0/200, FDC-BF[K77] 0/200, invalid NAIVE-joint **6/200** |
| Lemma TU, single-cell Monte Carlo (N = 300, M = 90, n₀ = 30, 6000 permutations, target level a = 0.10) | Crossing frequency over all n ≥ n₀ of the frozen-n₀ Chernoff threshold: **0.036 ≤ 0.10** |
| Invalid control for the same Monte Carlo: exact fixed-n quantile refreshed at every n, no maximal inequality | **0.553 ≫ 0.10** |
| Dev, all blocks and methods (K20 and dense) | 0 false streams |
| Dominance | FDC-LOC ≤ FDC-BF on every stream (CR9 and X5) |
| external reviewer r1 bug: β̂ = 0 gave infinite widths when every policy is surely ε-good | Fixed (β̂ floor 10⁻³, admissible); regression test added. No dev row was affected (no β < 0.01 in any trace); both toys were rerun on the fixed code with identical numbers |

## 3. Dev results: N80_pen paired geometric-mean ratios (UB95 one-sided)

### 3.1 Candidate (a) on the frozen K = 20 grid

| Comparison | CR9, ε = 0.001 | X5, ε = 0.015 | X5, ε = 0.02 | X5, ε = 0.03 |
|---|---|---|---|---|
| **FDC-LOC/FDC-BF** | **0.925 (0.949)**; faster on 30%, tied on 70% | 1.000 (1.000) | **1.000 (1.000)** | 1.000 (1.000) |
| FDC-LOC/RECT-ck-HG* | 0.601 (0.630) | 0.463 (0.477) | 0.336 (0.347) | 0.249 (0.256) |
| FDC-LOC/HC-WoR* | 0.613 (0.646) | 0.460 (0.473) | 0.329 (0.342) | 0.253 (0.260) |
| FDC-LOC/PJC-local* | 0.925 (0.949) | 1.021 (1.042) | 0.988 (1.004) | 1.004 (1.025) |
| β̂/β_J at stop | 0.944 | 1.000 | 1.000 | 1.000 |
| Effective union e^{β̂}·Ĝ (M ≈ 3386 / 3347) | 1527 | 3344 | 3344 | ≈3344 |
| *ORACLE-LOC/FDC-BF (invalid, uses true gaps; ceiling only)* | *0.812 (0.847)* | *0.824 (0.852)* | *0.807 (0.834)* | *0.143 (0.146)* |
| *ORACLE β/β_J at stop; effective union* | *0.771; 127* | *0.804; 214* | *0.764; 120* | — |

The X5 ε = 0.03 oracle value is degenerate: almost every policy is ε-optimal there, so the oracle's union collapses.

**Why (a) fails: the near-tie barrier.**
- **True-gap census.**
  - CR9: of the 3386 (q, π) pairs, 437 are ε-optimal and 1531 lie within 3ε.
  - X9: 3076 of the 3347 pairs are ε-optimal, and every pair lies within 2ε.
- **What localisation must prove.** A large union reduction requires showing that hundreds of (q, π) pairs are either ε-good (Δ^up ≤ ε) or separated by more than about the certificate width.
- **Why the data cannot prove it in time.**
  - The free screen (δ_var = 0.005 boxes, an ℓ₁ rectangle) proves almost nothing at the time FDC-BF certifies. On X5 it proves essentially nothing.
  - A stronger joint pairwise screen on a δ split (δ_main/2, about 2.6·10⁵ ordered pairs, Bernstein) was prototyped on one stream per log. It is *worse*: β rises to 14.9 at the early checkpoints, because the δ split costs more than the screen recovers. The prototype was not run on 50 streams.
  - Gaps become provable only at roughly the precision that certification itself needs.
- **What is left.** The oracle shows the headroom is real (about 19% fewer rows), but no data-driven screen we built reaches it before certification.
- **Reading for the paper.** The union cost β_J/β_C ≈ 1.47 measures the size of the ε-near-optimal frontier set, which is intrinsic to these logs. It is not slack in the ledger, and it can be written as a supporting remark to the width proposition.

### 3.2 Candidate (b)

**Primary (frozen K = 20 grid, matched with every earlier block).** TU-FDC ≡ FDC-BF here (identity verified by unit test).

| Comparison | CR9, ε = 0.001 | X5, ε = 0.015 | **X5, ε = 0.02** | X5, ε = 0.03 |
|---|---|---|---|---|
| **TU-FDC/HC-WoR*** | **0.663 (0.695)** | 0.460 (0.473) | **0.329 (0.342)** | 0.253 (0.260) |
| TU-FDC/RECT-ck-HG* (stronger guarantee than the rival) | 0.649 (0.677) | 0.463 (0.477) | 0.336 (0.347) | 0.249 (0.256) |
| TU-FDC/PJC-local* | 1.000 (1.010) | 1.021 (1.042) | 0.988 (1.004) | 1.004 (1.025) |
| **Cost of the time-uniform guarantee vs checkpoint FDC-BF** | **1.000 (identical)** | 1.000 | 1.000 | 1.000 |

**Secondary (4×-dense evaluation grid, K_eval = 77, all methods monitored at the same 77 times).**

| Comparison | CR9, ε = 0.001 | X5, ε = 0.02 |
|---|---|---|
| TU-FDC[K20, stale widths]/HC-WoR* (dense) | **0.708 (0.737)** | **0.356 (0.365)** |
| TU-LOC[K20]/HC-WoR* (dense) | 0.661 (0.689) | 0.356 (0.365) |
| FDC-BF[K77, fresh widths, β + ln(77/20)]/HC-WoR* (dense) | 0.709 (0.734) | 0.366 (0.375) |
| TU-FDC[K20]/RECT-ck-HG (dense, K = 77 ledger) | 0.624 (0.653) | 0.341 (0.351) |
| TU-FDC[K20]/FDC-BF[K77] | 0.999 (1.016) | 0.972 (0.985) |
| TU-LOC[K20]/TU-FDC[K20] | 0.935 (0.954) | 1.000 |
| Gain from dense monitoring: TU-FDC dense / FDC-BF at K20 | 0.954 | 0.979 |
| Gain from dense monitoring: HC-WoR dense / HC-WoR at K20 | 0.894 | 0.906 |

**Reading.**
- At matched guarantee strength and a matched evaluation grid, the joint advantage survives intact: 0.66 on CR9 and 0.33 on X5 on the frozen grid; 0.71 and 0.36 on the dense grid.
- Dense monitoring helps HC-WoR somewhat more (−10%) than TU-FDC (−2% to −5%). The reason is that TU-FDC's widths are frozen between block points, so only Δ̂ moves. Even so, every dense ratio has UB95 < 0.75.
- Refreshing widths on a finer block grid (K' = 77) buys nothing: the ln(77/20) = 1.35 increase in β cancels the fresher widths.

## 4. Implications for the paper and for v9

1. **Objection (ii), guarantee asymmetry, is answered by a theorem and costs nothing.**
   - FDC-BF's certification is valid at every time t ≥ t₁ under the frozen schedule (Theorem TU-1), the same time-uniformity class HC-WoR has at a fixed design.
   - The existing confirmatory numbers (v7 / v8 blocks) stay as they are, because the evaluated procedure is unchanged. Only the stated guarantee strengthens, and this should be disclosed as a post-hoc theoretical observation, not a design change.
   - The one honest residual to state: HC-WoR's guarantee also covers adaptive sampling, and TU-FDC's does not. The paper's design is frozen by construction, which is why this does not affect the comparison.
2. **Objection (i), novelty.**
   - (b) adds a clean ingredient: the reverse-martingale maximal inequality applied to a direction-level multi-cell WoR certificate. That combination is not stated in the literature we have checked. The lemma itself is classical (Serfling), and the paper should say so.
   - (a) adds the near-tie barrier, an explanatory result: the union cost reflects the size of the ε-near-optimal frontier set, data-driven localisation recovers at most 7.5% (CR9) or 0% (X5), and an oracle gains about 19%.
   - Neither result changes the speed headline.
3. **Lock v9 (recommendation).**
   - Primary: FDC-BF with the guarantee stated as Theorem TU-1, versus RECT-ck-HG*, HC-WoR* and PJC-local* (non-inferiority) as in v8.
   - Descriptive rows:
     - FDC-LOC (dominates FDC-BF; CR9 0.925);
     - FDC-HG;
     - the dense-monitoring block (TU-FDC[K20] vs HC-WoR* on a 77-point grid);
     - ORACLE-LOC, clearly labelled invalid and used only as a ceiling.
   - Neither FDC-LOC nor FDC-HG enters a confirmatory comparison (both are NO-GO).
4. **Compute.** Per stream, FDC-LOC takes about 15 s (CR9) and 17 s (X5), against about 4 s for FDC-BF; the cost is the 512 × 512 pair tables at each checkpoint. TU-FDC costs the same as FDC-BF.
