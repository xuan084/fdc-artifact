# external reviewer review: width proposition (Proposition 1 / Corollary 2)

Reviewer: external reviewer MCP (model from ~/.reviews/config.toml, approval-policy never, read-only sandbox). Thread 01a10489-df04-7cc0-b997-88dac5652423.
Files reviewed: plan/width_proposition.md, writing/snippets/width_prop{,_appendix}.tex, exp/code/predict_width_ratio.py,
exp/results/pilots/width_prop/prediction.json, with fdc_bet.py / pjc_bf.py / main.tex as context.

## Round 1 verdict: FLAWED as written (ideal-width proof sound; implementation/rows/prediction claims overstated)

external reviewer confirmed: phi bounds, lambda choice, joint and rectangle sandwiches, ratio bounds, kappa equality cases, box
supremum and both ledgers match the code; Corollary 2(a) and the order-statistic argument are correct.

Issues raised and how v2 resolved them:

| # | Sev. | Issue | Resolution in v2 |
|---|---|---|---|
| 1 | major | Claimed "implemented ratio only moves up": rectangle also uses a lambda grid; `_CkRect.certify` intersects final intervals across checkpoints; empty cells use [0,1] in the rectangle | Rewritten: grid / clipping / running intersection effects have no fixed sign; proposition is about ideal widths; empty-cell exclusion stated |
| 2 | minor | Grid ratio is 2000^(1/160)=1.0487 (not 1.069); quadratic factor 1.00028; Bennett optimum not guaranteed in grid; clipping can bind away from exhaustion | Constants fixed; caveats added |
| 3 | major | Main-text rows statement omitted (S3),(S5); r bound must be over all problems | Main proposition now lists all assumptions; r sandwich stated over all problems; "Gaussian terms" instead of "eta=0" |
| 4 | major | Monotonicity in x false when r>1 | Derivative r(1-r)/(1-x(1-r))^2 stated; "lies between r and 1" |
| 5 | major | "Closed form" substituted one binding rho^2 for the order-statistic ratio r | Replaced by exact analytic r12 = u^R_[12]/u^J_[12] and the theorem-implied interval [g(r_min,x), g(r_max,x)]; old substitution kept only as a labelled heuristic in JSON |
| 6 | major | "S2 holds"/"sandwich exact" contradicted (tolerance 1e-3 hid violations) | X5 S2 called approximate; exact check now done in the exact analytic model (all inside at rel. tol 1e-9); realistic G-FPC counts reported both ways (tol 1e-3: 15/15; exact: 8, 10, 11 of 15) |
| 7 | major | Predictor "G" kept capped counts/exhaustion, so not the T=inf model; gauss_limit rescalings unjustified | "G" removed; Corollary computed analytically with uncapped counts (T=inf and common T); gauss_limit reduced to z values and labelled heuristic |
| 8 | major | Converse promoted a direction-level necessary condition to problem level | Reworded; problem-level converse derived from Cor. 2(a): m_eff(a_R*(q)) <= beta_J/beta_C implies t^J_q >= t^R_q |
| 9 | minor | Gaussian asymptotic needs levels -> 0 | "as min(beta_J,beta_C) -> inf"; "separately minimal widths" |
| 10 | minor | HG rows translation needs negligible FPC | Assumptions stated |
| 11 | minor | m>=1 / no empty cell; joint gamma bound needs M>=2SA; "equals" -> "at most"; 0<=x<3 in sketch | All added |
| 12 | minor | m_eff statistic ambiguity; "code-faithful" overstated; noise can advance; error ranges -14.5%..+18.3% and -12.4%..+12.4%; 1.468 vs 1.469; 1.794; CI half-widths from 0.0036 | m_eff* := (beta_J/beta_C)/r12 used consistently; "code fluid (no running intersections)"; numbers fixed |

## Round 2 verdict: SOUND WITH MINOR FIXES

external reviewer re-checked Corollary 2(b)/(c) (u_[12] as 12th largest threshold, monotonicity of g(r,x) in r, sandwich mapping to
[g(r_min,x), g(r_max,x)]), the problem-level converse, and all numbers against prediction.json; no new errors in the
proofs. Remaining minor items, all applied:

1. Main proposition: state no empty touched cell, C(a) with v_c>0, m>=1, M>=2SA; corollary: eps>0, a positive-variance
   challenger per problem, zero-width challengers omitted. -> added to main snippet, appendix, plan.
2. Analytic CH computation omits N_c/(N_c-1); say identities/sandwich are exact for that surrogate. -> added.
3. Do not conflate directional m_eff with frontier m_eff* = (beta_J/beta_C)/r; CR12 is 7.3, not ~3. -> fixed.
4. Scope: remainders small only for examined directions; no uniform closeness claim for implemented ratios; "differ only
   in two respects" only at leading order; inclusion of observed ratios in CH intervals is numerical, not guaranteed. -> fixed.
5. Gaussian converse needs independence. -> "conditional on A, independent N(0, v_c)".
6. Error range wording: relative errors -14.5% to +18.3% (CR12 vs +box). -> fixed.

No round 3 was run (task limit of 2 rounds); all round-2 items were minor and applied verbatim or nearly so.
