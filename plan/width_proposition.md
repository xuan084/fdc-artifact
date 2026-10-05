# Width proposition: joint direction-level width vs matched per-cell rectangle

Status: v2 (2026-10-04), theorist. Revised after external reviewer review round 1 (`reviews/width_prop_review.md`).
Script `exp/code/predict_width_ratio.py`, output `exp/results/pilots/width_prop/prediction.json`. LaTeX:
`writing/snippets/width_prop.tex` (main text, ≈0.36 page) and `writing/snippets/width_prop_appendix.tex` (appendix).
Data discipline: DEV halves only (`PoolReplayEnv("CR9"|"CR12", "dev")`, `X5LayerEnv("dev")`). No evaluation outcome is
loaded. Observed ratios are copied from the sealed analysis JSONs and used only for a descriptive comparison.

## 0. One-sentence result

For the ideal widths, on each policy difference `a`, FDC-BF's width is `ρ(a) = √(β_J/β_C)·‖g‖₂/‖g‖₁` times the matched
rectangle's width, up to direction-dependent remainders that are a few percent for the directions examined in the fluid
calculation, where `g_c = |a_c|·sd_c` is the per-cell error scale. The matched rectangle uses the same
cell bound, boxes, plan and δ split. In the scaling regime the rows ratio is r/(1 − x(1 − r)). Here r is a ratio of order statistics that is sandwiched by
binding-direction values of `(β_J/β_C)/m_eff`, where `m_eff = ‖g‖₁²/‖g‖₂²` is the effective number of cells, and x is
the rectangle's stopping point divided by the horizon T, so exhaustion compresses the ratio toward 1. The union cost `β_J/β_C` is 1.47 to 1.64 on all three logs. The log-dependent spread comes from the
frontier-level `m_eff* = (β_J/β_C)/r`: 3.3 on CR9, 7.3 on CR12 and 11.7 to 13.4 on X5. This frontier summary need not
equal any single direction's m_eff.

## 1. Setting and the two widths

Fix a plan 𝒜 and a checkpoint k. Every quantity below is a function of 𝒜, the fixed table and the variance boxes, which
both methods share. For each cell c:

- `n_c` is the number of rows read from cell c and `N_c` is its pool size. The cell is live when `0 < n_c < N_c`.
- `σ̄_c² = sup_{m∈B_c(k)} m(1−m)`, where `B_c(k)` is the exact HG inversion at `α_var = δ_var/(2SAK)`, intersected over
  checkpoints.
- `v_c = σ̄_c² (N_c−n_c)/((N_c−1)n_c)` is the variance proxy of `μ̂_c`, and `b_c = 1/n_c`.

The Bennett-FPC bound (main.tex eq. (1)) is increasing in `m(1−m)`, so its box supremum is

  `Ψ̄_c(u) = (v_c/b_c²)·φ(b_c|u|)`, with `φ(x) = eˣ−1−x`.

This is exactly `fdc_bet.psi_bar(kind="bennett")`.

A policy difference `(π', π̂)` has coefficients `a_c = +w_s` on `(s, π̂(s))` and `a_c = −w_s` on `(s, π'(s))`, for
`s ∈ D`. Write `C(a) = {c : 0<n_c<N_c, a_c≠0, v_c>0}` and `m = |C(a)| ≤ 2|D|`. **Assume no touched cell is empty,
m ≥ 1, and M ≥ 2SA (true on all instances here).** Let `g_c = |a_c|√v_c`, with norms taken over `C(a)`.

**Ledgers.** Both methods use the same split `δ_main + δ_var = 0.045 + 0.005`.

- *Joint (FDC-BF).* One one-sided event per (problem q, feasible π, checkpoint k). The sign of each event is fixed by
  the comparison with `π*_q`. This gives `β_J = ln(MK/δ_main)`, with `M = Σ_q|Π_{B_q}|`.
- *Matched rectangle* (`pjc_bf.RectCkBF`). The centre is data-dependent, so a cell can enter as a challenger cell, which
  needs an upper bound, or as a centre cell, which needs a lower bound. Every cell therefore needs both tails. This
  gives 2SAK one-sided events and `β_C = ln(2SAK/δ_main)`.
- *Values.*
  - CR9: `M = 3386`, `β_J = 14.22`, `β_C = 9.68`, `β_J/β_C = 1.469`.
  - X9: dev `M = 3347`, `β_J/β_C = 1.468` (eval M = 3386 gives 1.469).
  - CR12: `M = 26940`, `β_J = 16.30`, `β_C = 9.97`, `β_J/β_C = 1.635`.
- The joint-only bound on η in Proposition 1(v) uses `β_J ≥ β_C`, which is equivalent to `M ≥ 2SA`. This holds on every
  instance here.

**Ideal widths.**

  `W_J(a) = inf_{λ>0} [β_J + Σ_{c∈C(a)} Ψ̄_c(λa_c)]/λ`

  `r_c = inf_{λ>0} [β_C + Ψ̄_c(λ)]/λ`,  `W_R(a) = Σ_{c∈C(a)} |a_c| r_c`

For the current checkpoint's unprocessed intervals `[μ̂_c − r_c, μ̂_c + r_c]`, the rectangle statistic is
`Σ_{s∈D} w_s(hi_{s,π'(s)} − lo_{s,π̂(s)}) = Δ̂ + W_R`. FDC-BF's statistic is `Δ̂ + w_k`.

**How the implementation departs from the ideal widths (no fixed sign).**

1. Both implementations minimise over finite λ grids of 161 points with ratio `q = 2000^{1/160} = 1.0487`. FDC-BF's grid
   is centred at `√(2β_J/V)` and the rectangle's at `√(2β_C/v_c)`, both spanning `[1/200, 10]` times the centre. Each
   grid enlarges its own width relative to the infimum. For a purely quadratic objective whose optimum the grid brackets,
   the enlargement factor is at most `(√q + 1/√q)/2 = 1.00028`. This is not a uniform bound for the Bennett objective,
   and because both the numerator and the denominator of the ratio are enlarged, the net effect on the ratio has no
   fixed sign.
2. The rectangle is shrunk by three operations: clipping to the pool's logical bounds `[s/N, (s+N−n)/N]`, the optional
   intersection with `B_c` ("+box"), and the running intersection of the final intervals across checkpoints
   (`_CkRect.certify`). These can bind away from exhaustion, for example at small n or for means near 0 or 1.
3. For a touched empty cell, the joint width and the unprocessed radius are +∞, whereas the implemented rectangle uses
   the logical interval [0, 1]. Proposition 1 excludes such policy differences.

Proposition 1 is therefore a statement about the two ideal constructions. It makes no uniform claim about how close the
implemented ratio is.

## 2. Proposition 1 (width ratio, non-asymptotic)

**Proposition 1.** Assume no touched cell is empty and `m(a) ≥ 1`. Then:

(i) `√(2β_J)‖g‖₂ ≤ W_J(a) ≤ √(2β_J)‖g‖₂ + β_J b*/3`, where `b* = max_{c∈C(a)}|a_c|b_c`.

(ii) `√(2β_C)‖g‖₁ ≤ W_R(a) ≤ √(2β_C)‖g‖₁ + β_C Σ_c|a_c|b_c/3`.

(iii) `ρ(a)/(1+η_R) ≤ W_J/W_R ≤ ρ(a)(1+η_J)`, where
- `ρ(a) = √(β_J/β_C)·κ(a)` and `κ = ‖g‖₂/‖g‖₁`,
- `η_J = √β_J·b*/(3√2‖g‖₂)` and `η_R = √β_C·Σ|a_c|b_c/(3√2‖g‖₁)`.

(iv) `m^{−1/2} ≤ κ ≤ 1`, equivalently `ρ² = (β_J/β_C)/m_eff` with `m_eff = ‖g‖₁²/‖g‖₂² ∈ [1, m]`. Equality holds at
`κ = 1` if and only if m = 1, and at `κ = m^{−1/2}` if and only if all g_c are equal.

(v) `η_J ≤ √(β_J/(18γ))` and `η_R ≤ √(β_C/(18γ))`, where `γ = min_{c∈C(a)} n_c σ̄_c² (N_c−n_c)/(N_c−1)`.

**Proof.**

*Step 1 (one cell).* For V, b, β > 0, let `h = inf_λ [β + (V/b²)φ(bλ)]/λ`.
- Since `φ(x) ≥ x²/2` for x ≥ 0, `h ≥ inf_λ (β + λ²V/2)/λ = √(2βV)`.
- For 0 ≤ x < 3, `j! ≥ 2·3^{j−2}` gives `φ(x) ≤ x²/(2(1−x/3))`.
- Take `s = √(2β/V)` and `λ = s/(1+bs/3)`. Then `bλ < 3` and `1−bλ/3 = 1/(1+bs/3)`, so
  `(V/b²)φ(bλ) ≤ β/(1+bs/3)`.
- The objective at this λ is therefore at most `β(2+bs/3)/s = √(2βV) + bβ/3`, and so is h.

*Step 2 (joint width).*
- Lower bound: each term satisfies `Ψ̄_c(λa_c) ≥ λ²a_c²v_c/2`, and Step 1 applies.
- Upper bound: for `λb* < 3`, `Ψ̄_c(λa_c) ≤ λ²a_c²v_c/(2(1−λb*/3))`. The objective is therefore at most
  `[β_J + λ²‖g‖₂²/(2(1−λb*/3))]/λ`, and Step 1's choice of λ applies with `V = ‖g‖₂²` and `b = b*`.

*Step 3 (rectangle).* Apply Step 1 to each cell, multiply by `|a_c|` and sum.

*Step 4 (ratio).* Divide the bounds in (i) by those in (ii).

*Step 5 (range of κ).* Cauchy–Schwarz gives `‖g‖₂ ≤ ‖g‖₁ ≤ √m‖g‖₂`.

*Step 6 (bounds on η).*
- For η_J: `b*/‖g‖₂ ≤ b_{c*}/√v_{c*}`, where c* is the cell attaining b*.
- For η_R: the mediant inequality gives `Σ|a|b/Σ|a|√v ≤ max_c b_c/√v_c`.
- In both cases `b_c/√v_c = (n_cσ̄_c²(N_c−n_c)/(N_c−1))^{−1/2} ≤ γ^{−1/2}`. ∎

On our logs η is small. At the fluid FDC-BF stopping time, for the joint-binding policy differences, η_J has median 0.010
(CR9 and CR12) and 0.016 to 0.021 (X9). η_R has median 0.019 to 0.059. The largest value of either is 0.075.

The two forces are separated. The union cost enters only through `√(β_J/β_C) > 1`. The aggregation gain enters only
through κ. To leading order, the joint width is narrower on a policy difference if and only if `m_eff(a) > β_J/β_C`.

## 3. Corollary 2 (rows ratio in the scaling model)

**Assumptions.** ε > 0. Every problem has a challenger with C_a > 0, and zero-width challengers are omitted from the
minimum. In addition, (S1)–(S5) hold on the range of t considered:

- (S1) `n_c = π_c t`.
- (S2) One common horizon: `N_c = π_c T` for all relevant cells, with T = ∞ allowed.
- (S3) Time-invariant variance proxies `σ̄_c²`.
- (S4) The widths are replaced by their Gaussian leading terms. The finite-sample remainders are positive, so this is an
  approximation, not "η = 0".
- (S5) Fluid statistics: the centre is `π*_q` and `Δ̂ = Δ`. A challenger π' is therefore ruled out once its width is at
  most `θ(π') = ε + J(π*_q) − J(π') ≥ ε`.

Under these assumptions:

- `v_c(t) = ṽ_c·u(t)`, with `u(t) = 1/t − 1/T` and `ṽ_c = σ̄_c²N_c/((N_c−1)π_c)`.
- The squared widths are `C^J_a·u(t)` and `C^R_a·u(t)`, with `C^J_a = ρ(a)²C^R_a`.
- Problem q is certified once `u(t) ≤ u_q = min_{π'} θ²/C`.

**Corollary 2.**

(a) For each problem q, `ρ(a_R*(q))² ≤ u^R_q/u^J_q ≤ ρ(a_J*(q))²`, where `a_J*(q)` and `a_R*(q)` are the minimisers for
the joint certificate and the rectangle.

(b) Let `u_[12]` denote the 12th largest threshold, so that the 12th certification time is `N80 = 1/(u_[12] + 1/T)`.
Then `r := u^R_[12]/u^J_[12]` satisfies

  `r ∈ [min_q ρ(a_R*(q))², max_q ρ(a_J*(q))²] ⊂ [(β_J/β_C)/(2S), β_J/β_C]`.

If `T = ∞`, then `N80^J/N80^R = r`.

(c) If `T < ∞` and `x = N80^R/T`, then `N80^J/N80^R = r/(1 − x(1−r))`.

For fixed r > 0, this value increases in x when r < 1, decreases in x when r > 1, and equals 1 when r = 1. It always lies
between r and 1, and it tends to 1 as x ↑ 1, because the derivative in x is `r(1−r)/(1−x(1−r))²`. Since it is also
increasing in r, the interval in (b) maps to `[g(r_min, x), g(r_max, x)]` for the observed x.

**Proof.**

*(a)*
- `u^R_q ≤ θ²_{a_J*}/C^R_{a_J*} = ρ(a_J*)²·u^J_q`.
- `u^J_q ≤ θ²_{a_R*}/C^J_{a_R*} = u^R_q/ρ(a_R*)²`.

*(b)* If `x_q ≤ c·y_q` holds for all q, then the same inequality holds between corresponding order statistics; the same
is true for ≥. The outer interval follows from Proposition 1(iv) with m ≤ 2S.

*(c)* Write `1/N80 = u_[12] + 1/T` for each method, and `u^R_[12] = (1−x)/(xT)`. Then
`N80^J/N80^R = (u^R_[12] + 1/T)/(u^J_[12] + 1/T) = r/(1 − x + xr)`. ∎

**Exact numerical check of the corollary.** `predict_width_ratio.py` computes the corollary analytically, with no time
grid, under (S1)–(S5), using dev means and uncapped counts `π_c = w_s/2`.
- In the T = ∞ model, every problem's `u^R_q/u^J_q` lies inside its sandwich (a) to a relative tolerance of 1e−9, on
  all five logs and tolerances.
- The common-horizon model (T = τ_R) reproduces formula (c) exactly.

**Which real logs satisfy (S2).**
- X5's cell horizons `N_c/(w_s/2)` lie in [0.9838, 1.0162]·τ_R, so (S2) holds approximately. In the realistic G-FPC
  computation (actual pools, capped counts, cell-specific FPC, log grid), all 15 problems at each ε pass the sandwich
  test at an absolute tolerance of 1e−3. Exact inclusion holds for 8, 10 and 11 of the 15 problems at ε = 0.015, 0.02
  and 0.03.
- On Criteo the log is 85/15 and the plan is 50/50, so the cell horizons range from 0.26 to 1.74 τ_R and (S2) fails.
  Only 5 of 15 problems pass the sandwich test there, so for Criteo we rely on the G-FPC and code fluid computations.

## 4. Converse remarks

**(a) Single-cell policy differences.**
- If m = 1, for example |D| = 1 with the other arm's pool exhausted, then `W_J/W_R ≥ √(β_J/β_C)/(1+η_R)`. This exceeds 1
  once `β_J > β_C(1+η_R)²`, so no joint width built on the same cell bound and union beats the rectangle on that policy
  difference.
- For two live cells, `β_J < 2β_C` is necessary for a leading-order advantage. The actual condition is
  `m_eff > β_J/β_C`. The CR9 problems whose joint-binding policy difference has |D| = 1 have `ρ² = 0.735 < 1`.
- A comparison on single policy differences does not by itself order the certification times of whole problems. Under
  (S1)–(S5) and T = ∞, Corollary 2(a) gives the problem-level converse: `m_eff(a_R*(q)) ≤ β_J/β_C` implies
  `t^J_q ≥ t^R_q`, where equality need not mean a strict loss.
- Since `M ≤ QA^S`, we have `β_J ≤ β_C + S ln A + ln(Q/(2SA))`. The union penalty grows at most linearly in S, and so can
  `m_eff ≤ 2S`. Whether aggregation pays is decided by the variance profile of the binding policy differences.

**(b) The ℓ2-versus-ℓ1 gap is intrinsic.**
- Conditional on 𝒜, take independent Gaussian cell errors `N(0, v_c)` with known variances, and widths that are deterministic given 𝒜. Then a
  one-sided bound on one fixed policy difference at level `α_J` has width at least `z_{α_J}‖g‖₂`. A sum of per-cell
  one-sided radii at level `α_C` has width at least `z_{α_C}‖g‖₁`.
- At the fixed Bonferroni levels `α_J = δ_main/(MK)` and `α_C = δ_main/(2SAK)`, the ratio of these two separately minimal
  widths is `(z_{α_J}/z_{α_C})κ`.
- Since `z_α² = 2ln(1/α) − ln ln(1/α) − ln 4π + o(1)`, we get `(z_J/z_C)² = (β_J/β_C)(1 + o(1))` as
  `min(β_J, β_C) → ∞`. A Gaussian approximation at fixed ledgers does not by itself imply this equivalence.
- On our ledgers, `(z_J/z_C)²` is 1.589 on CR9 (against `β_J/β_C = 1.469`) and 1.794 on CR12 (against 1.635). Exact
  Gaussian quantiles are slightly less favourable to the joint width.

**(c) Exact-HG rectangles.**
- RECT-ck-HG spends all of δ on exact inversion at `δ/(2SAK)`. With known variances, negligible FPC, proportional counts
  and leading terms, its radius is `z√v_c` with z = 3.810, against the Chernoff radius `√(2β_C v_c)` = `4.40√v_c` on
  CR9.
- The predicted HG/BF rows ratio is therefore `z²/(2β_C) = 0.750`. On CR9 the observed RECT-ck-HG/RECT-ck-BF rows ratio
  is 0.498/0.643 ≈ 0.775. Finite-population effects, box uncertainty and interval intersections all alter this
  translation.
- This is why FDC-BF's ratios against RECT-ck-HG (0.643, 0.364, 0.546) are larger than its matched ratios (0.498, 0.299,
  0.499 for +box).

## 5. Prediction vs observed (descriptive, post hoc)

The numerical CH model approximates N_c/(N_c−1) by 1. The identities and the sandwich are exact for this surrogate.
All predictions use dev pool sizes and dev means, the frozen 50/50 plan's expected counts, the code's ledgers and ε, and
no sampling noise. The observed values are paired geometric-mean ratios of FDC-BF to the method on N80_pen:
- CR9: v7 block A for RECT-ck-HG, v8 block C for the matched rectangle.
- X9: v8 block A, the untouched evaluation half.
- CR12: v8 block B, which has only the matched "+box" rectangle.

Columns:
- `m_eff*`: `(β_J/β_C)/r`, the frontier-effective number of cells, from the exact scaling model.
- `r (T=∞)`: Corollary 2(b), exact model.
- `CH`: Corollary 2(c) with common horizon T = τ_R, exact model, together with its sandwich interval
  `[g(r_min, x), g(r_max, x)]`.
- `G-FPC`: leading terms with the actual pools.
- `code fluid`: the width routines run at fluid counts, without running intersections.

| log | β_J/β_C | m_eff* | r (T=∞) | x | CH ratio [interval] | G-FPC | code fluid (BF) | observed FDC-BF/RECT-ck-BF [95% CI] | code fluid (HG) | observed FDC-BF/RECT-ck-HG |
|---|---|---|---|---|---|---|---|---|---|---|
| CR9, 0.001 | 1.469 | 3.3 | 0.441 | 0.21 | (S2 fails) | 0.555 | 0.558 | 0.498 [0.484, 0.512] | 0.710 | 0.643 |
| CR12, 0.001 | 1.635 | 7.3 | 0.224 | 0.36 | (S2 fails) | 0.396 | 0.402 (+box 0.427) | +box 0.499 [0.487, 0.512] | 0.478 | 0.546 |
| X9, 0.015 | 1.468 | 11.7 | 0.125 | 0.81 | 0.423 [0.355, 0.642] | 0.424 | 0.432 | 0.426 [0.420, 0.431] | 0.470 | 0.428 |
| X9, 0.02 | 1.468 | 12.6 | 0.117 | 0.73 | 0.329 [0.258, 0.564] | 0.329 | 0.338 | 0.299 [0.294, 0.304] | 0.379 | 0.364 |
| X9, 0.03 | 1.468 | 13.4 | 0.109 | 0.59 | 0.232 [0.187, 0.461] | 0.232 | 0.239 | 0.202 [0.199, 0.206] | 0.282 | 0.251 |

**Reading the table.**

1. **The union cost is about the same on every log.** `β_J/β_C` is 1.47 to 1.64.
2. **The variance profile sets the pre-FPC ratio.**
   - On Criteo the error variance sits in three high-rate segments (s = 2, 6, 8, with μ ≈ 0.12 to 0.30), while every
     other segment has μ ≤ 0.054. The result is `m_eff* ≈ 3.3` on CR9 and `r ≈ 0.44`. CR12 splits these segments
     further: `m_eff* ≈ 7.3`, `r ≈ 0.22`.
   - On X5 every segment has μ ≈ 0.6 and the binding policy differences flip about six segments, so `m_eff* ≈ 12` and
     `r ≈ 0.11` to 0.125.
3. **Exhaustion compresses the ratio toward 1, and this explains X5's dependence on ε.**
   - In the X5 rectangle runs, x is 0.59 to 0.81, so Corollary 2(c) lifts r from ≈0.12 to 0.23, 0.33 and 0.42 at
     ε = 0.03, 0.02 and 0.015.
   - The observed values are 0.20, 0.30 and 0.43, and each lies numerically inside the corresponding interval. The
     corollary does not guarantee this for noisy runs.
   - A pure scaling argument without FPC would predict a nearly constant ratio and could not produce this trend.
4. **Magnitudes.**
   - The code fluid predictions differ from the observed matched ratios by −14.5% to +18.3%, and from the observed HG
     ratios by −12.4% to +12.4%.
   - Both are too high on CR9 and too low on CR12. The observed CR9 and CR12 matched ratios are equal (0.498 and 0.499),
     whereas the fluid model ranks CR12 lower.
   - The model ignores sampling noise (which can advance or delay either certificate), the running intersections, and
     the difference between dev and eval means.
   - **The proposition explains the mechanism and the ε-trend. It does not forecast the second decimal.**

## 6. Limitations

- The predictions are descriptive and post hoc. They were computed after the observed ratios were known, and no lock
  threshold or claim depends on them.
- Proposition 1 bounds the ideal widths. The implemented widths differ through the λ grids, clipping and running
  intersections, and the net effect on the ratio has no fixed sign (§1).
- Corollary 2 is exact only in the idealised scaling model. (S2) is approximate on X5 and fails on Criteo, and (S5)
  ignores noise in which challenger binds.
- The converse in §4(b) holds only for known-variance Gaussian errors with deterministic widths and Bonferroni levels. It
  is not a minimax lower bound over all certificates; for example, a smaller union such as FDC-MR's changes β_J.
- Certificates are issued only at log-spaced checkpoints (ratio 1.297 on Criteo, 1.229 on X5), so each stream's ratio is
  a ratio of grid points. The continuous-time predictions approximate the geometric mean over streams.
