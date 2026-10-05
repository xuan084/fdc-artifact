# v9 candidate upgrades: validity arguments and pre-stated go criteria

Status: v2, 2026-10-04. v1 (commit 61ce3930) was written BEFORE any code or dev run; v2 adds the external reviewer r1 fixes
(marked [r1]; notational r2 fixes marked [r2]) without changing the constructions' validity logic or the go criteria. Setting, notation and ledgers as in
`plan/fdc_hg_theorem.md` §1-2 and `plan/width_proposition.md` §1: frozen outcome-free schedule 𝒜 (50/50, labels-only
re-selection), cells independent uniform WoR samples given 𝒜 with deterministic counts n_c(t), exact-HG variance box
B_c(k) at α_var = δ_var/(2SAK) with running intersection (event E_var, P(E_var^c) ≤ δ_var), (δ_main, δ_var) =
(0.045, 0.005). For problem q, policy π ∈ Π_q: Y_t(π) := Σ_c a_c(π*_q, π)(μ_c − μ̂_c(t)), gap Δ_π := J(π*_q) − J(π) ≥ 0,
F_{k,π}(λ) := log E[e^{λY_{t_k}(π)} | 𝒜] (true MGF), and the ideal width t*_{k,π}(β) := inf_{λ>0}(β + F_{k,π}(λ))/λ.
FDC-BF's implemented width w_k(π', π; β) satisfies w_k ≥ t*_k(β) on E_var (fdc_hg_theorem (iv), Bennett-FPC branch).

Two standard facts used below.
- (C) Chernoff: P(Y_{t_k}(π) > t*_{k,π}(β) | 𝒜) ≤ e^{−β} (fdc_hg_theorem proof (i)).
- (T) t*(β) is an infimum of affine functions of β, hence concave and increasing; t*(0) = F'(0) = E Y = 0 (μ̂ is unbiased
  under WoR). So t*(β)/β is nonincreasing, i.e. t*(β(1+x/t*(β))) ≤ t*(β) + x for x ≥ 0.

## (a) FDC-LOC: gap-localised union with a free screen

**Construction.** At checkpoint k, on the same boxes FDC-BF already computes, form the pairwise rectangle lower bound
L(π', π) = Σ_{s: π'(s)≠π(s)} w_s(lo[s,π'(s)] − hi[s,π(s)]) and the screened gap Δ^lo_{q,π} = max_{π'∈Π_q} L(π', π). Survivors
S_q = {π' ∈ Π_q : Δ^lo_{q,π'} ≤ 0}. Let V(π', π) = Σ_{s∈D} w_s²(v_{s0} + v_{s1}) and b(π', π) = max_{c∈D} w_s/n_c (box-sup
variance proxy v_c of μ̂_c, live cells), and W̄_{q,π}(β) = max_{π'∈S_q}[√(2βV(π',π)) + b(π',π)β/3] (+∞ if a touched cell
has n = 0). Define

  Ĝ_k(β) = Σ_q Σ_{π∈Π_q} exp(−β(1 + (Δ^lo_{q,π} − ε)_+ / W̄_{q,π}(β))),   β̂_k = min{β ≤ β_J : Ĝ_k(β) ≤ δ_main/K}

(bisection; any returned β̂ with Ĝ_k(β̂) ≤ δ_main/K is admissible). [r1] Refinement: let
Δ^up_{q,π} = max_{π''∈S_q}[−L(π, π'')] (an upper bound on Δ_π on E_var); policies with Δ^up_{q,π} ≤ ε are surely
ε-good and are dropped from Ĝ_k. [r1] If Ĝ_k(0) ≤ δ_main/K (all remaining terms vanish), β̂_k = 10⁻³ (any β > 0 is then
admissible; β = 0 would give a degenerate λ grid). FDC-LOC is FDC-BF with β_J replaced by β̂_k in
every direction width at checkpoint k. Because Ĝ_k(β_J) ≤ M e^{−β_J} = δ_main/K, β̂_k ≤ β_J: FDC-LOC's widths are never
larger than FDC-BF's on the same schedule and boxes (constructive dominance). No δ is spent on the screen: it reuses E_var.

**Theorem LOC-1.** P(some certification of FDC-LOC is ε-wrong) ≤ δ_main + δ_var.

*Proof.* (1) Deterministic comparator: G_k(β) := Σ_q Σ_{π∈Π_q: Δ_π>ε, Y≢0, no touched cell empty at t_k} exp(−β(1 + (Δ_π − ε)/t*_{k,q,π}(β))),
indexed by (k, q, π) (π*_q differs across q; no union factor is lost). [r1] Directions with Y ≡ 0 (including when all touched
cells are exhausted) cannot produce a false certification and are excluded; directions touching an empty cell have infinite
width and cannot certify. G_k(0) is defined by its right limit. [r2] If the index set is empty, no false certification is possible on
E_var and the proof ends here; otherwise G_k(0) ≥ 1 > δ_main/K, so β_k^det > 0 and step (3) never divides by t*(0) = 0. G_k depends only on 𝒜 and the truth, is continuous and
nonincreasing (β/t*(β) is nondecreasing by (T)); let β_k^det = inf{β : G_k(β) ≤ δ_main/K}. (2) On E_var: [r1] for EVERY π' ∈ Π_q, L(π', π) ≤ J(π') − J(π) ≤ Δ_π, so Δ^lo_{q,π} ≤ Δ_π and π*_q ∈ S_q
(Δ^lo_{π*} ≤ 0); Δ_π ≤ Δ^up_{q,π} (π*_q ∈ S_q), so dropped policies are not in G's index set; t*_{k,π}(β) ≤ √(2βV) + bβ/3 ≤ W̄_{q,π}(β) (width_proposition Prop. 1 step 2 applied to the
Bennett-FPC sup, which dominates the true MGF on E_var, with (π*_q, π) one of the maximised pairs). Hence termwise
Ĝ_k(β) ≥ G_k(β) for every β, so Ĝ_k(β̂_k) ≤ δ_main/K ⇒ G_k(β̂_k) ≤ δ_main/K ⇒ β̂_k ≥ β_k^det. (3) A false certification
of q at k with answer π̂ means Δ_π̂ > ε and Δ̂(π*_q, π̂) + w_k(π*_q, π̂; β̂_k) ≤ U_q(k) ≤ ε, i.e. Y_{t_k}(π̂) ≥ w_k(β̂_k) +
x with x = Δ_π̂ − ε > 0. On E_var, w_k(β̂_k) ≥ t*(β̂_k) ≥ t*(β_k^det). By (T), for every β'' < β_k^det(1 + x/t*(β_k^det)),
t*(β'') < t*(β_k^det) + x ([r1] for β'' ≥ β_k^det use t*(β'') ≤ (β''/β_k^det)t*(β_k^det); for β'' < β_k^det monotonicity), so by (C) P(Y ≥ t*(β_k^det) + x) ≤ e^{−β''}; letting β'' increase,
P ≤ exp(−β_k^det(1 + x/t*(β_k^det))). (4) Union over (k, q, π with Δ_π > ε): P(false ∩ E_var) ≤ Σ_k G_k(β_k^det) ≤
δ_main. Add P(E_var^c) ≤ δ_var. The data-dependent β̂_k never enters a probability statement: it only has to dominate
the deterministic β_k^det on E_var. ∎

Remarks. The gap term is what makes the localisation legal: far-from-optimal centres are not removed from the union,
their events are charged at exp(−β(1+x/W̄)) ≈ 0, which they "earn" because a false certification of them needs a deviation
larger than the width by their excess gap. A naive survivor-count union (β = ln(|S|K/δ) with S data-dependent and no
gap charge) is NOT covered by this argument and is not used. The λ grid, sticky answers, stop rule and both observation
windows are inherited unchanged (fdc_theorem §5-6).

## (b) TU-FDC: time-uniform extension at no union cost (reverse-martingale maximal inequality)

**Lemma TU.** Conditional on 𝒜, for any fixed coefficient vector a and any checkpoint t_k such that n_c(t_k) ≥ 1 for
every c with a_c ≠ 0 ([r1]: with an empty cell the placeholder μ̂ = 1/2 breaks the reverse-martingale property; such
directions have infinite frozen width in TU-1 and never certify), the process
Y_t = Σ_c a_c(μ_c − μ̂_c(n_c(t))), t ≥ t_k, satisfies P(sup_{t_k ≤ t ≤ T} Y_t ≥ y | 𝒜) ≤ inf_{λ>0} e^{−λy} E[e^{λY_{t_k}} | 𝒜].

*Proof.* For one WoR cell, the sample means μ̂_c(n), n = N_c, N_c − 1, …, 1, form a reverse martingale with respect to the
decreasing filtration G^c_n = σ(multiset of the first n draws, X_{n+1}, …, X_{N_c}): by exchangeability
E[μ̂_c(n−1) | G^c_n] = μ̂_c(n), and μ̂_c(N_c) = μ_c (Serfling 1974; Bardenet & Maillard 2015, §2). Given 𝒜 the cells are
independent and n_c(t) is deterministic and nondecreasing in t (a cell whose count does not change contributes an
identity step; an exhausted cell contributes 0), so H_t = ∨_c G^c_{n_c(t)} is decreasing in t and
E[Y_t | H_{t+1}] = Y_{t+1}: (Y_t)_{t ≥ t_k} is a reverse martingale, e^{λY_t} a nonnegative reverse submartingale. In
reversed time it is a forward submartingale whose terminal element is e^{λY_{t_k}}, and Doob's maximal inequality gives
P(max_{t ≥ t_k} e^{λY_t} ≥ e^{λy}) ≤ e^{−λy}E e^{λY_{t_k}}. Infimum over λ. ∎

**Theorem TU-1.** Fix the K block points t_1 < … < t_K (the frozen checkpoint grid). TU-FDC certifies at ANY time
t ≥ t_1 (any data-dependent monitoring or stopping) with U_q(t) = max_{π'∈Π_q}[Δ̂_t(π', π̂_t) + w_{k(t)}(π', π̂_t)], where
k(t) = max{k : t_k ≤ t} and w_{k(t)} is FDC-BF's width computed from the counts and box at t_{k(t)} (β_J, ledger unchanged).
Then P(∃ t ≥ t_1, q: an ε-wrong certification) ≤ δ.

*Proof.* Replace (C) in FDC-BF's proof by Lemma TU: with y = t*_{k,π}(β_J) (and the λ-limit of fdc_hg_theorem (i)),
P(sup_{t ≥ t_k} Y_t(π) > t*_{k,π}(β_J)) ≤ e^{−β_J}, one event per (q, π, k) as before, so E_main^TU = ∩_{q,π,k}
{sup_{t≥t_k} Y_t(π) ≤ t*_{k,π}} ([r2] restricted to directions whose touched cells are non-empty at t_k, with H_t
defined over the touched cells only) has P ≥ 1 − δ_main. E_var is needed only at the block points (the box enters only
through w_k ≥ t*_k). On E_main^TU ∩ E_var, at any t with k = k(t): J(π*_q) − J(π̂_t) = Δ̂_t + Y_t(π̂_t) ≤ Δ̂_t + w_k ≤ U_q(t). ∎

Consequences. (i) Evaluated only at the K block points, TU-FDC is numerically identical to FDC-BF: the time-uniform
guarantee costs nothing there; FDC-BF's guarantee was already time-uniform on [t_1, T] with stale widths. (ii) The same
lemma upgrades every Chernoff/MGF certificate (RECT-ck-BF, FDC-LOC: replace (C) by Lemma TU in step 3) but NOT the exact-HG
quantile rectangle RECT-ck-HG: [r1, external reviewer counterexample] for N = 4, M = 2, n₀ = 2, P(Y₂ > 0) = 1/6 but
P(max_{2≤n≤4} Y_n > 0) = 1/2, so a fixed-time quantile radius need not survive monitoring. (iii) The guarantee is uniform over time under the frozen (outcome-free) schedule; HC-WoR's
is uniform under any predictable sampling rule. The comparison here is at a fixed frozen schedule, which both use.
(iv) A finer block grid K' costs ln(K'/K) in β; any K' is valid.

## Pre-stated go criteria and measurement protocol (written before running)

Dev only: CR9 dev half ε = 0.001; X5 (X9 dev half) ε = 0.02 (decision), 0.015 / 0.03 descriptive; report seeds 950-999
(50 streams), stop_k = 15, metric N80_pen, paired geometric-mean ratio with the v6 bootstrap (B = 10⁴, seed 42), one-sided
UB95. Rival configs = frozen v8 dev configs (RECT-ck-HG*, HC-WoR*, PJC-local*). No new tuned hyperparameter: FDC-LOC
inherits FDC-BF's split, λ grid and box; the bisection tolerance (1e-6 in β) is fixed in code. Toy: r4 near-tie toy, 200
streams, with the invalid NAIVE-joint power control; a single-cell Monte Carlo test of Lemma TU with an invalid
"fresh width at every n without union" control.

- **(a) GO** iff FDC-LOC/FDC-BF UB95 < 0.90 on at least one of {CR9, X5 ε=0.02} and ≤ 1.00 on the other, AND 0 false
  streams on dev and toy.
- **(b) GO** iff TU-FDC/HC-WoR* UB95 < 0.80 on both CR9 and X5 ε = 0.02 (primary: evaluation on the frozen K = 20 grid,
  where TU-FDC ≡ FDC-BF), AND 0 false streams on dev and toy (dense monitoring). Secondary (descriptive): a 4×-dense
  evaluation grid (3 geometric interpolants per checkpoint interval) on which TU-FDC is run with stale widths
  (K = 20 blocks) and with fresh widths (K' = 77 blocks), against HC-WoR* evaluated on the same dense grid.
