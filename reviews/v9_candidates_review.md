# external reviewer review: v9 candidate validity arguments (FDC-LOC, TU-FDC)

Date 2026-10-04. Reviewer: external reviewer MCP (model from ~/.reviews/config.toml, no model parameter; approval-policy never;
read-only sandbox). Thread 01a10591-0bae-7ad1-9913-5c91a5ec0309. 2 rounds (the cap).
Objects reviewed: `plan/v9_candidates_theory.md` (v1 at commit 61ce3930, then v2), `exp/code/dsswm/baselines/fdc_loc.py`,
`exp/code/dsswm/tests/test_v9_candidates.py`.

## Round 1: verdicts and findings

**(a) Theorem LOC-1: VALID WITH FIXES.** The FWER argument is sound in real arithmetic. The deterministic comparator
G_k / β_det is legitimate, and no random threshold is substituted into a probability bound. Required fixes:
1. Gap lower bound: "π* is one candidate" is not enough justification. For EVERY feasible π′,
   L(π′, π) ≤ J(π′) − J(π) ≤ Δ_π. Taking the maximum gives the bound and puts π*_q in S_q.
2. Survivor-maximised W̄ ≥ t* is correct, since the pair (π*_q, π) is among the maximised pairs.
3. The surely-good refinement (drop policies with Δ^up ≤ ε) is valid and should be stated in the theorem.
4. Strictness and limit argument: correct once both cases are written out (β″ ≥ β_det uses concavity; β″ < β_det uses
   monotonicity).
5. Endpoint conventions: exclude Y ≡ 0 directions from G, define G(0) by its right limit, and treat touched-empty-cell
   directions separately.
6. Union accounting over (q, π, k) is correct.
- **Code bug (reproduced by external reviewer):** one segment, N = 1000, n = 900, s = 450 per arm, ε = 0.1. All policies are surely
  good, so `loc_beta` returned β = 0, and `direction_widths` then gave width +∞ against FDC-BF's 0.0219. This is
  conservative for FWER, but it breaks dominance and certification.

**(b) Lemma TU / Theorem TU-1: VALID WITH FIXES.** The filtration argument is correct:
- G^c_{n+1} ⊆ G^c_n.
- Exchangeability gives E[μ̂_c(m) | G^c_n] = μ̂_c(n) for m ≤ n.
- Conditional independence extends this to the joined filtration. Unchanged counts give identity steps, and exhausted
  cells contribute 0.
- Doob's inequality applies to the reversed-time submartingale, and the λ-infimum and limit step is valid.
- E_var is needed only at block points, and the snapshot logic faithfully implements the stated construction.

Required fix: a positive starting count for every touched cell. With an empty cell, the placeholder μ̂ = 1/2 breaks the
lemma (pool {0,1}: Y₀ = 0, but Y₁ = 0.5 w.p. 1/2). The infinite frozen widths already exclude such directions in the code.

**Exact-HG exclusion: VALID.** Counterexample: N = 4, M = 2, n₀ = 2 gives P(Y₂ > 0) = 1/6, but
P(max_{2≤n≤4} Y_n > 0) = 1/2.

## Fixes applied after round 1
- Theory v2: every fix above, marked [r1].
- Code: if Ĝ(0) ≤ target, β̂ = BETA_FLOOR = 10⁻³ instead of 0.
- Regression test `test_loc_beta_zero_branch_keeps_dominance`, using the external reviewer's example.
- No dev row was affected: no FDC-LOC beta trace on CR9 has β < 0.01.

## Round 2: verdicts
- **LOC-1: VALID WITH FIXES (bookkeeping only).**
  - List empty-cell directions in G's displayed index set.
  - State the empty-index-set case explicitly. Otherwise G(0) ≥ 1 > δ_main/K, so β_det > 0 and nothing divides by t*(0).
  - Write "Y ≡ 0, including when all touched cells are exhausted".
- **β-floor implementation: VALID.**
  - W̄(β)/β = max_j{√(2V_j/β) + b_j/3} is nonincreasing, so Ĝ is nonincreasing.
  - The code's conservative Ĝ(0) count C bounds every positive-β value. Because the target is below 1, C ≤ target means
    no retained terms, so `min(BETA_FLOOR, β_J)` is admissible and preserves dominance.
  - The comment should say "a positive admissible value".
- **Lemma TU: VALID.** TU-1: VALID WITH FIXES (notation only). Restrict E_main^TU to directions with non-empty touched
  cells, and define H_t over the touched cells.
- **Exact-HG limitation: VALID.**
- All 8 tests passed.

## Fixes applied after round 2
Theory note: the notational items, marked [r2]. Code: comment wording. No change to any construction or criterion.
