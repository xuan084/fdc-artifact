# FDC-DP: joint certificates over exponentially large segment-policy classes

Status: v1, 2026-10-05. This was written before any dev run of `exp/code/run_fdc_dp_dev.py`. The only runs before it
were unit tests and a 2-seed smoke check (§10). Setting, notation, cell inequality, variance box and δ split are FDC-BF's
(`plan/fdc_hg_theorem.md` §1–2, `plan/v9_candidates_theory.md` preamble). The code is in
`exp/code/dsswm/baselines/fdc_dp.py`, `rect_dp.py` and `envs/seg_v10.py`, with tests in `dsswm/tests/test_fdc_dp.py`.

## 1. Problem

There are S segments, A arms and cells c = (s, a) with pool sizes N_c. The cell means μ_c come from a fixed finite table.
The design is FDC-BF's frozen outcome-free schedule 𝒜 (50/50 plan, labels-only re-selection). Given 𝒜, every cell is
an independent uniform without-replacement (WoR) sample with deterministic counts n_c(t).

Costs are integers κ[s,a] ≥ 0. Problem q has an integer budget B_q, and its policy class is

  Π_q = {π ∈ [A]^S : Σ_s κ[s, π(s)] ≤ B_q}.

That class has up to A^S members (≈ 1.2·10²⁰ summed over the 15 budgets at S = 64) and is **never enumerated**.

J(π) = Σ_s w_s μ_{s,π(s)}, and π*_q is the optimal policy in Π_q (lowest index on ties; any deterministic rule works).
The centre π̂_q = argmax_{Π_q} Ĵ is computed by an exact multiple-choice knapsack DP with deterministic tie-breaking.

For the dev tables (`envs/seg_v10.py`, family SR), arm 0 is control with cost 0. Arm 1 costs κ[s,1] = round(8·S·w_s),
which is 8 for every segment because the segments are equal-size score quantiles. The budgets are
B_q = ⌊b_q·Σ_s κ[s,1]⌋ for b_q = 0.10, …, 0.80. So Π_q is "treat at most ⌊b_q S⌋ segments", a cardinality class. The
code handles general integer knapsacks; the tests use heterogeneous costs.

## 2. Ledger by counting, not enumeration

The main union is FDC-BF's: one one-sided event per (q, π ∈ Π_q, k), and **its cardinality is all that the union bound
uses**. With the counting DP N_0(0) = 1 and N_s(c) = Σ_a N_{s−1}(c − κ[s,a]), we get |Π_q| = Σ_{c ≤ B_q} N_S(c). This is
exact in Python integers, after dividing costs by their gcd g and budgets by ⌊·/g⌋ (lossless).

  M = Σ_q |Π_q|,  β_J = ln(M K / δ_main) ≤ S ln A + ln(Q K / δ_main),

so β_J is linear in S. The SR family with K = 20 and δ_main = 0.045 gives the values below. β_C = ln(2SAK/δ_main) is the
matched rectangle's per-cell exponent.

| S | M | β_J | β_C | β_J/β_C |
|---|---|---|---|---|
| 9 | 3.3·10³ | 14.21 | 9.68 | 1.47 |
| 16 | 4.3·10⁵ | 19.08 | 10.26 | 1.86 |
| 32 | 2.8·10¹⁰ | 30.16 | 10.95 | 2.75 |
| 64 | 1.2·10²⁰ | 52.34 | 11.64 | 4.50 |

The test `test_count_policies_*` checks the counts against enumeration and against the closed form Σ_{j ≤ m} C(S, j).

## 3. Certification condition and its column form

For challenger π ≠ π̂ with difference set D, FDC-BF's direction width is

  W(π) = min_{λ∈Λ} [β + Σ_{s∈D} (Ψ̄_{s,π(s)}(λ w_s) + Ψ̄_{s,π̂(s)}(λ w_s))]/λ.

Ψ̄_c(u) is the box supremum of the Bennett-FPC bound. It is symmetric in u, so challenger and centre cells use the same
argument. The certificate is U_q = max_{π∈Π_q} [Δ̂(π, π̂) + W(π)] ≤ ε, which is a **∀π ∃λ** condition. Write it in
*column* form. For every j ∈ 𝒥 = Λ ∪ {∞}, let C_j(π) = const_j + Σ_s T_s(j, π(s)), with T_s(j, π̂(s)) = 0, and:

- λ-column: const = β/λ, T_s(λ, a) = w_s(μ̂_{s,a} − μ̂_{s,π̂(s)}) + [Ψ̄_{s,a}(λw_s) + Ψ̄_{s,π̂(s)}(λw_s)]/λ;
- exact column (λ = ∞): const = 0. T_s(∞, a) = w_s(μ̂_{s,a} − μ̂_{s,π̂(s)}) if both cells are deterministic on E_var
  (exhausted, or zero box variance), and +∞ otherwise. This is FDC-BF's "width 0 when V_pair = 0" rule.
- cells with n_c = 0 give +∞ in every column (FDC-BF's zero_touch rule).

Then Δ̂ + W(π) = min_j C_j(π), and **certified ⇔ max_{π∈Π_q∖{π̂}} min_{j∈𝒥} C_j(π) ≤ ε**. The centre π̂ has U = 0 ≤ ε
and can be dropped.

**Lemma 0 (any λ set is valid).** On E_var, every λ > 0 gives
[β + F̄_π(λ)]/λ ≥ [β + F_π(λ)]/λ ≥ inf_{λ'} [β + F_π(λ')]/λ' = t*_π(β), where F is the true conditional log-MGF. So a
minimum over any set Λ, data-dependent or not, is ≥ t*. The probability statement concerns only the deterministic
t*. The grid here is global per checkpoint: geometric at FDC-BF's density 2000^{1/160}, running from
√(2β/V_max)/200 to max(10·√(2β/V_min), β/ε). V_max is the largest pair-variance proxy summed over segments and V_min
is the smallest single-cell one. It therefore covers every per-direction range FDC-BF uses, [λ_0/200, 10λ_0] with
λ_0 = √(2β/V_pair), and it also contains β/ε, which §4.3 needs.

## 4. Computable sufficient conditions

### 4.1 Scheme (a): one λ per check (conservative)
Because max min ≤ min max,

  U_q ≤ U_a := min_{j∈𝒥} [const_j + max_{π∈Π_q∖{π̂}} Σ_s T_s(j, π(s))].

For fixed j, the inner maximum is a multiple-choice knapsack over segments, with A options per segment, integer costs
and a two-state "differs from π̂" flag. A forward DP over capacity solves it exactly in O(S·A·|𝒥|·B) for all budgets ≤ B
at once. **Proposition A:** certifying when U_a ≤ ε is valid, because U_a ≥ U_q.

Scheme (a) is conservative only through the quantifier swap. Every λ-column is still minimised per check, so a single λ
serves the whole class, chosen after seeing the data.

### 4.2 Scheme (b): exact, by branch-and-bound over segments
Fix a segment order σ. A node fixes π on σ(1..i): it holds the used cost u, the flag h and the partial column vector
p_j = Σ_{l≤i} T_{σ(l)}(j, ·). Suffix DP tables give, for every j, every remaining capacity r and both flag states:

  F_i(j, r) = max over assignments of σ(i+1..S) with cost ≤ r of Σ T_s(j, ·)  (with "≥ 1 difference" when h = 0).

The node bound is UB = min_j [const_j + p_j + F_i(j, B_q − u)]. This is scheme (a) applied to the subtree, so UB ≥ the
maximum over the subtree of min_j C_j. At a leaf, UB equals min_j C_j(π) exactly. Depth-first search expands the
higher-bound child first and prunes nodes with UB ≤ ε (decision mode) or UB ≤ incumbent (max mode).

**Proposition B (exactness).** The tree is finite, every pruned subtree holds only policies with value ≤ the pruning
threshold, and every unpruned leaf is evaluated exactly. Decision mode therefore returns "certified" iff
max_{π≠π̂} min_j C_j(π) ≤ ε, and max mode returns that maximum. In both cases the result is exactly FDC-BF's certificate
with λ set 𝒥 (FDC-BF[𝒥]). A node cap (2·10⁵) returns "not certified" when reached, which is sound, and the hits are
counted and reported. Scheme (b) is applied only when scheme (a) fails.

**Exactness contract (external reviewer FDC-DP r1 B3).** The implementation gives the exact decision [U_q ≤ ε] *when the search
completes*; when the node cap is reached it abstains ("not certified"). The capped algorithm is therefore not an
unconditional iff implementation of FDC-BF[𝒥]; cap hits are logged (0 in every v2 dev cell). The U reported with an
answer is the scheme-(a) bound U_a ≥ U_q, never an exact certificate value; a violator found by scheme (b) gives a lower
bound U_lower ≤ U_q, and U_q itself is computed only in the diagnostic max mode. The centre π̂_q is the weighted
empirical argmax of Σ_s w_s μ̂(s, π(s)) (external reviewer r1 B1; the v1 code used unweighted means, see the dev report).

### 4.3 Dominance (decision mode only, lossless)
Drop an option a ≠ π̂(s) when T_s(j, a) ≤ 0 for every j and κ[s,a] ≥ κ[s,π̂(s)], provided min_j const_j ≤ ε. The grid
contains λ ≥ β/ε, so this proviso holds.

*Proof.* Take a violator π that uses dropped options. Replace them all by the centre arm. Feasibility is kept and every
column value weakly increases. If the result is π̂, then every difference of π was a dropped option and
min_j C_j(π) ≤ min_j const_j ≤ ε, so π was not a violator after all. Otherwise the result is a violator that uses no
dropped option. ∎

### 4.4 Exact enumeration (ground truth, S ≤ ~20)
`enum_U` evaluates the same columns on every π ∈ [A]^S. The tests also include an independent per-policy implementation
of FDC-BF[𝒥] built directly from `psi_bar`, with no column algebra. At S ∈ {3, 5, 7, 9} with random cells, boxes,
heterogeneous costs and budgets, scheme (b) equals it to 1e-9 relative, with every decision identical. Scheme (a) is
never below it.

Native FDC-BF (`direction_widths`, per-direction λ grid) differs from FDC-BF[𝒥] only by grid discretisation. On the test
instances the maximum difference is 0.2 % of max(|U|, 0.01), and the median is 0.03 %. Both variants are valid.

**A rejected alternative: partitioning challengers by their λ-optimal cell.** The set
{π : column j attains min_j C_j(π)} is defined by |𝒥| − 1 inequalities between sums. It is not a knapsack-separable set,
so it cannot be optimised by one DP per cell without the branching that scheme (b) already does.

## 5. Data-dependent centre and the union never being enumerated

**Theorem DP-1.** Under FDC-DP(a) or FDC-DP(b), P(some certification is ε-wrong) ≤ δ_main + δ_var = 0.05 at the K
checkpoints, for every problem and both schemes.

*Proof.* The index set I = {(q, π, k) : π ∈ Π_q, 1 ≤ k ≤ K} is deterministic: it depends on costs and budgets only,
not on data. For each index, the event E_{q,π,k} = {Y_{t_k}(π) ≤ t*_{k,q,π}(β_J)} concerns the fixed coefficient vector
of the direction (π*_q, π), where Y = J(π*_q) − J(π) − Δ̂(π*_q, π). By the Chernoff bound,
P(E^c | 𝒜) ≤ e^{−β_J}. The union bound is a sum over I, so P(∪ E^c) ≤ |I|·e^{−β_J} = M K e^{−β_J} = δ_main. Neither the
bound nor the method needs to list I; |I| = MK comes from §2. P(E_var^c) ≤ δ_var (§6).

On E_main ∩ E_var, suppose q is certified at k with centre π̂, which is data-dependent but lies in Π_q, so (q, π̂, k) ∈ I.
If π̂ = π*_q, the answer is exact. Otherwise π*_q is one challenger in Π_q∖{π̂}, and

  J(π*_q) − J(π̂) = Δ̂(π*_q, π̂) + Y(π̂) ≤ Δ̂ + t* ≤ Δ̂ + W(π*_q) = min_j C_j(π*_q) ≤ U_q ≤ U_a.

The second inequality is Lemma 0. Scheme (b) certifies only if U_q ≤ ε, and scheme (a) only if U_a ≤ ε, so the answer
is ε-good. Sticky answers, the stop rule and the K checkpoints are as in FDC-BF. ∎

The data enter only through the centre's index in I, which is covered for every possible value, and through computing
U_q. The argument is FDC-BF's proof verbatim. The new ingredient is that both |I| and max_π are obtained without listing
Π_q.

## 6. Variance boxes
There are S·A cells. The exact hypergeometric test inversion is run per cell and checkpoint at per-tail level
α = δ_var/(2SAK), with a running intersection over checkpoints. This gives P(E_var^c) ≤ 2SAK·α = δ_var. At S = 64,
α = 9.8·10⁻⁷. The box enters only through Ψ̄ and the "deterministic cell" flags, exactly as in FDC-BF.

## 7. Time-uniform extension
Lemma TU (`v9_candidates_theory.md` (b)) is a statement about one fixed coefficient vector. The event family here is
the same, indexed by (q, π, k), with the same vectors. Theorem TU-1 therefore applies unchanged.

At an evaluation time t, take the latest block point k(t). Freeze the counts, box, λ set, Ψ̄ tables and β there, and
recompute only the Δ̂ part of the T terms from current data. The certificate condition keeps the same column form, so
schemes (a) and (b) apply verbatim. This is `FDCDPTimeUniform`, and `test_tu_wrapper_identical_on_block_grid` checks it
is identical to FDC-DP on the block grid. Lemma TU's empty-cell restriction is inherited: such directions have
infinite frozen width.

## 8. Rectangle rival at scale: RECT-BF-DP (and HC-WoR-DP)

RECT-BF-DP is the matched Bennett rectangle `pjc_bf.RectCkBF(box=True)`. It shares FDC-DP's cell inequality, HG box,
(0.045, 0.005) split and design. Each cell gets the radius r_c = min_λ (β_C + Ψ̄_c(λ))/λ, with β_C = ln(2SAK/δ_main),
and the interval is clipped to the logical pool bounds and the box, with a running intersection. The certificate is

  U_q = max(0, max_{π∈Π_q∖{π̂}} Σ_{s∈D} w_s (hi[s, π(s)] − lo[s, π̂(s)])).

This is one additive column, so the DP is exact. Its union is over the 2SAK cell events and does not depend on |Π_q|.

HC-WoR-DP uses the same DP over the hedged-capital WoR CS, at δ/(SA) per cell and time-uniform. Its schedule
parameters are the frozen v8 ones, at the 50/50 plan: Lenta takes the CR config and X5 takes the X9 config without its
S = 9 Neyman matrix. These configs were tuned on S = 9, so for S ≥ 16 they are transferred, not tuned (disclosed).

The *best valid rectangle* for a (table, S, ε) cell is the one of {RECT-BF-DP, HC-WoR-DP} with the lower dev geomean
N80_pen, chosen per cell. That choice is made after seeing the data and favours the rival.

## 9. Width proposition at scale (Proposition 1 of `width_proposition.md`)

To leading order the joint width is narrower than the matched rectangle on a difference a iff
ρ(a)² = (β_J/β_C)/m_eff(a) < 1, where m_eff = ‖g‖₁²/‖g‖₂² ≤ 2|D|. Two effects grow with S:

1. **The union cost grows.** β_J/β_C rises from 1.47 at S = 9 to 4.50 at S = 64 (§2), because β_J ≈ S·H(b) while
   β_C ≈ ln S. On a local swap (|D| = 2, m_eff ≤ 4) the joint width is therefore *wider* at S = 64. The joint
   certificate does not win on every direction.
2. **The rectangle's binding directions get longer.** The rectangle has no ℓ2 aggregation, so its width grows like
   Σ_{s∈D} w_s·(r_{s,π(s)} + r_{s,π̂(s)}), linearly in |D|. Its worst challenger in Π_q is a large flip whose |D| scales
   with the budget, so it is O(b·S). Proposition 1's frontier sandwich puts the rows ratio between the ρ² of the
   rectangle's binding differences (small, since m_eff ∝ S) and the ρ² of the joint certificate's binding differences.
   The joint certificate's large flips are paid for by a very negative Δ̂.

Combining the two, ρ² on the rectangle's binding direction is ≈ (S·H(b)/β_C)/(c·b·S) = O(1/ln S). The ratio should
**improve with S, but only logarithmically, and only through the rectangle's long binding directions**. The joint
certificate's own binding directions can have m_eff below β_J/β_C. This is a mechanism statement, not a guarantee, and
the dev runs decide it.

The 2-seed smoke check fits the mechanism, but it is not evidence. On X5 at S = 64 and t ≈ 0.57N, the rectangle's
largest U over problems was 0.13, against 0.018 for FDC-DP. The rectangle's U grows with the budget, while FDC-DP's is
flat.

**Lenta caveat (pre-stated).** Lenta is 25 % control, so the 50/50 frozen design exhausts the control pools at about
t ≈ 0.5–0.67·N. After that the control cells are exact. On Lenta every method's certification is dominated by
exhaustion: in the smoke check certification jumps at k = 17, t ≈ 0.67N. X5 is the table where the width mechanism is
visible. Lenta is reported, but is expected to compress all ratios toward the exhaustion point.

## 10. Pre-stated go criteria
These are copied from `plan/v10_plan.md` workstream 1, with numeric targets added.

1. **Validity proof.** This document, Theorem DP-1 with Propositions A and B and Lemma 0, followed by a external reviewer review.
2. **Exact agreement on small S.** `dsswm/tests/test_fdc_dp.py` passes. At S ∈ {3, 5, 7, 9}, scheme (b) equals the
   enumeration FDC-BF[𝒥] (U to 1e-9 relative, 100 % of decisions). Scheme (a) never certifies where enumeration does
   not. The DP max equals the brute-force max, and the counts equal enumeration. In the toy Monte Carlo (12 populations ×
   8 permutations, ε = 0.02) every rigorous method has 0 false streams, and the invalid control (β = 0.5) has ≥ 1.
3. **Runtime.** For FDC-DP(b) at S = 64 with all 15 problems, the median is < 10 s per checkpoint and the maximum
   < 30 s. Node-limit hits must stay ≤ 1 % of B&B calls, on both tables. For FDC-DP(a), the median is < 2 s.
4. **Dev effect.** The one-sided UB95 of the paired geometric-mean ratio N80_pen(FDC-DP(b))/N80_pen(best valid
   rectangle) is < 0.80 at S ≥ 32 on at least one table, at the primary ε, with 0 false streams for every rigorous
   method.
   - Primary ε: X5 0.02, the decision ε of v8/v9. Lenta 0.003, the middle of LR9's grid.
   - Secondary ε, descriptive: X5 0.015, Lenta 0.004.
   - Seeds 950–999, stop_k = 15, v6 bootstrap (B = 10⁴, seed 42).

If all four pass, the next steps are a external reviewer review of this note, a v10 lock with fresh seeds, and a confirmatory block.
FDC-DP(b)+R, the hybrid min(joint, rectangle) with δ_main split equally, is descriptive only and is not part of the go
decision.

## 11. Data discipline and reuse
- `seg_v10` is dev-only: any `half != 'dev'` raises PermissionError.
- Lenta uses the LR9 dev half (split seed 6006). Pandas loads the whole pickle, and the eval rows' outcomes are discarded
  immediately, as in `lenta_v6`.
- X5 uses `dev.pkl`.
- Inside each dev half, a stratified 30 % model split (seed 10010) fits a T-learner of two HistGradientBoosting
  classifiers. Score = p̂₁ − p̂₀. The other 70 % is the replay population, whose outcomes the score model never sees.
  Replay N is 240,461 for Lenta and 69,752 for X5.
- Segments are the equal-size quantiles of the score on the replay pool. This uses covariates only, with ties broken by
  a seeded permutation.
- **Reuse disclosure.** These dev rows were used in earlier dev work: the Lenta dev half in v6/v7 and the X5 dev half in
  v8/v9. The new segmentation is a new problem on old rows. The eval halves are not touched in v10 dev.

## 12. Scope: what is new and what is not

**New:**
- certification of the FDC-BF joint certificate over *implicit* policy classes of size up to 10²⁰, by
  - exact union counting by DP,
  - a multiple-choice-knapsack form of the ∀π∃λ certificate,
  - an exact branch-and-bound for the quantifier order, with lossless dominance pruning;
- the resulting scaling regime, S = 16–64 segments, where per-cell rectangles pay ℓ1 over long challengers.

**Not new:**
- the cell inequality (Bennett with FPC);
- the union and Chernoff structure, and Theorem DP-1, which is FDC-BF's theorem;
- Lemma TU;
- the rectangle certificates;
- knapsack DP and branch-and-bound as tools.

The statistical guarantee is unchanged. The contribution is computational, and the empirical scaling claim depends on
§10.4.
