# FDC: unique specification and Theorem FDC-1 (task `r5_fdc_spec_theory`)

Status: pilot 2026-10-03, seed 42. Builds on `plan/theory/qfc_lemma.md` (r4, signed; cited below as QFC §n). Numerical checks: `exp/code/dsswm/theory_checks/mc_fdc.py`. Outputs: `exp/results/pilots/r5_fdc_spec_theory/{summary.json, mc_cert_coverage.csv, power_control.csv, run.log}`.

---

## 中文摘要

- **FDC 的唯一规格**（§1）：冻结 50/50 读表（某臂池耗尽后用预抽均匀数在剩余臂上重选，对 A=2 即全转另一臂）+ 方向级 Bernstein 证书（复用 `qfc_certificate_enum`）+ Lemma L2 反演方差上界 + 由 ctx 自动计算的账本。只在 K=20 个固定检查点发证书，答案首次认证后冻结，流在首个"已认证 ≥ 12/15"的检查点停止。
- **定理 FDC-1 已完整证明，无缺口**（§5）。证明只用一个同时事件 E = E_main ∩ E_var。E_main 以**真方差**写出，对象是 (q, π ∈ Π_{B_q}, k) 上从固定未知的 π*_q 指向 π 的方向事件；E_var 是 L2 反演 CI 的同时覆盖。在 E 上，σ² ≤ σ̄²，宽度对 V 单调，所以 J(π*_q) − J(π̂_q(k)) ≤ U_q(k)。这覆盖任意数据依赖的 π̂_q(k)，也覆盖对全部可行挑战者取 max 的证书。证明中**没有**把随机的 σ̄² 当作已知方差代入 MGF。
- **账本（CR9 开发半 ctx，代码计算）**：Σ_q|Π_{B_q}| = 3386，K = 20，S·A = 18；β = ln(3386·20/0.045) = **14.2242**；x_v = ln(720/0.005) = **11.8776**；合计 δ = 0.045 + 0.005 = 0.05。若硬编码 β = 14.1，E_main 部分为 3386·20·e^{−14.1} = 0.05095，总界 **0.05595 > δ**，不合法。
- **两个观察窗口都成立**（§6）：E 同时约束所有 (k, q)，"停在 12/15"与"运行到 τ_R 认证全部 15 题"只是同一网格上的两个子集；流级 FWER ≤ δ，且 E[V/(R∨1)] ≤ δ。
- **边界情形逐条覆盖**（§7）：n = 0 → 宽度 +∞；μ̂ ∈ {0,1} 时 L2 给出 σ̄² > 0（plug-in 会给 0，这正是功效对照里无效证书出错的来源之一）；池耗尽 → 精确、从 V 与 b 删除（耗尽状态是 𝒜 的函数）；单段两臂都耗尽 → 该段精确，在置换日程下只会在该段最后一次到达时发生，不会出现跳过；π̂ 与 π′ 只在耗尽单元不同 → 宽度 0、差值精确。
- **数值核验（seed 42）**：见 §9，结果在跑完后填入。门：0 违例；无效对照（plug-in 方差 + β = ln(1/δ)）必须检出违例。
- **自检清单**：§10，共 16 项，全部签字。

---

## 1. FDC specification (unique version, frozen before lock v5)

| Component | Specification | Code (r4 frozen / r5 new) |
|---|---|---|
| Read schedule | Every segment: planned arm drawn i.i.d. with p = (1/2, 1/2) from the pre-generated uniform stream. If the planned arm's pool is exhausted, re-select with the pre-drawn uniform `resel_u[t]` over the non-exhausted arms, p renormalised (A = 2: the other arm). If both pools of the segment are exhausted the arrival is skipped and still billed. | `envs.pool_replay.make_schedule(alloc=0.5)`, `StreamEngine.advance_planned/_resel` |
| Estimates | μ̂_c(k) = prefix mean of pool c's pre-permuted records; μ̂ = 1/2 placeholder when n_c = 0 (enters only π̂, never a finite width). | `FrontierState.mu_hat` |
| Variance UCB | Lemma L2 inversion at exponent x_v (binary pools), σ̄²_c(k) = max_{m ∈ CI_c(k)} m(1−m); exhausted cell: CI = {μ̂_c}; n_c = 0: CI = [0,1]. | `certify.quadknap.make_stats` → `theory_checks.mc_l1.bernstein_mu_ci`, `sigma2_ucb` |
| Pair width | w_k(π′, π) = √(2 β V̄) + b β / 3 over live differing cells; +∞ if a touched cell has n_c = 0. | `quadknap.pair_terms`, `_width` |
| Centre | π̂_q(k) = argmax_{Π_{B_q}} Ĵ_k, lowest enumeration index on ties. | `qfc_certificate_enum` |
| Certificate | U_q(k) = max_{π′ ∈ Π_{B_q}} [Δ̂_k(π′, π̂_q) + w_k(π′, π̂_q)]; certify iff U_q(k) ≤ ε. All feasible challengers, no data-based pruning. | `qfc_certificate_enum` |
| Stopping | Certificates only at the K = 20 fixed checkpoints; answer frozen at first certification; stream stops at the first checkpoint with ≥ 12/15 certified. | `frontier_runner.run_stream` |
| Ledger | β = ln(Σ_q |Π_{B_q}| · K / δ_main), x_v = ln(2 · S·A · K / δ_var), δ_main = 0.045, δ_var = 0.005; computed from ctx; no override. | `mc_fdc.fdc_ledger` (reference); `baselines/fdc.py` (r5_fdc_impl) |

Deleted (methodology §2.5): two-stage schedule, Cauchy–Schwarz accounting module, any FPC/Serfling correction, β = 14.1.

## 2. Probability space (item (a))

- **Fixed (non-random) objects.** The development or evaluation half as a finite table; the pools c = (s, a) ∈ 𝒞 with sizes N_c and finite-population means μ_c ∈ [0, 1] and variances σ_c² = μ_c(1 − μ_c) (binary visit); segment weights w_s = (Σ_a N_{s,a}) / N_half; costs κ = (0, 1); budgets B_q; the feasible classes Π_{B_q} = {π ∈ [A]^S : Σ_s w_s κ_{π(s)} ≤ B_q + FEAS_TOL} (the same tolerance in the truth and in the certificate); ε; the checkpoints T_1 < … < T_K (a function of n_min and τ_R). The optimum π*_q ∈ argmax_{Π_{B_q}} J with a deterministic tie rule (lowest enumeration index). All of these are fixed before the stream and none depends on the permutation seed.
- **Random objects.** (i) The complete schedule 𝒜 = (seg_seq, arm_seq, resel_u): the arrival segment sequence (a uniform permutation of the segment labels), the planned arm of every arrival (i.i.d. 50/50), and the re-selection uniforms. (ii) The pool permutations Π_c, one uniform permutation of each pool's records. In `make_schedule` these come from three independent child streams of one `SeedSequence` (arrivals, allocation, pools); we treat them, as usual, as independent random elements, so (Π_c)_c ⟂ 𝒜 and the Π_c are mutually independent.
- **𝒜 is complete and outcome-free.** 𝒜 is generated for all τ_R arrivals before any outcome is read; it is not a stopped or truncated path, so the counts at every checkpoint are defined whatever any stopping rule does.
- **Counts are functions of 𝒜.** Under the frozen 50/50 rule, n_c(k) is computed by replaying 𝒜 against the known pool sizes: the planned arm, the exhaustion test n_{s,a} ≥ N_{s,a}, and the re-selection all use only 𝒜 and N. No outcome enters. Hence n_c(k) = n_c(k; 𝒜), and so are the exhaustion indicators 1{n_c(k) = N_c}, the set of live cells, and the n = 0 indicators. (Checked numerically, §9 Part C: identical counts and schedule digests under outcome re-labelling.)
- **Fact 0 (QFC §1) applies verbatim.** Conditional on 𝒜, the vectors (y_{c,Π_c(1)}, …, y_{c,Π_c(n_c(k))}), c ∈ 𝒞, are independent uniform without-replacement samples of the fixed sizes n_c(k). Lemma L1 (QFC §2) and Lemma L2 (QFC §3) therefore hold conditional on 𝒜 for every k.

## 3. The simultaneous event (items (b), (c))

For every q ≤ Q, π ∈ Π_{B_q} and k ≤ K write a = a(π*_q, π) for the coefficient vector of Δ(π*_q, π) = J(π*_q) − J(π) (QFC §1: a_{(s,π*_q(s))} = +w_s, a_{(s,π(s))} = −w_s on the differing segments), and

  V_k(π*_q, π) = Σ_{c live at k, a_c ≠ 0} a_c² σ_c² / n_c(k),  V̄_k = same with σ̄_c²(k),  b_k = max_{c live, a_c ≠ 0} |a_c| / n_c(k),

  w_k(π*_q, π; V) = √(2 β V) + b_k β / 3, and w = +∞ if some c with a_c ≠ 0 has n_c(k) = 0.

**Main event (true variance).**

  E_main = ∩_{q ≤ Q} ∩_{π ∈ Π_{B_q}} ∩_{k ≤ K} { J(π*_q) − J(π) ≤ Ĵ_k(π*_q) − Ĵ_k(π) + w_k(π*_q, π; V_k) }.

Each event is "Σ_c a_c (μ_c − μ̂_c) ≤ width", i.e. the complement is the upper-tail event of Lemma L1 applied to −a (V and b are invariant under a → −a). V_k, b_k and the live set are 𝒜-measurable, so L1 gives P(complement | 𝒜) ≤ e^{−β}. Events with w = +∞ (some n_c = 0) and events with no live differing cell (deviation identically 0, strict ">" in L1) have probability 0. The event for π = π*_q is sure; it is kept in the count, which is conservative (Σ_q(|Π_{B_q}| − 1) = 3371 would also be valid).

**Variance event.** For every cell c and checkpoint k with 1 ≤ n_c(k) < N_c,

  CI_c(k) = { m ∈ [0, 1] : |μ̂_c(k) − m| ≤ √(2 m(1 − m) x_v / n_c(k)) + x_v / (3 n_c(k)) },

  E_var = ∩_{c ∈ 𝒞} ∩_{k ≤ K} { μ_c ∈ CI_c(k) }  (with CI = {μ̂_c} if exhausted, [0, 1] if n_c = 0; both contain μ_c surely).

By L1 for the single cell with a = ±1 and σ² = μ_c(1 − μ_c), P(μ_c ∉ CI_c(k) | 𝒜) ≤ 2e^{−x_v} (QFC §3). On E_var, σ_c² = μ_c(1 − μ_c) ≤ max_{m ∈ CI_c(k)} m(1 − m) = σ̄_c²(k) for every (c, k); the implementation returns an outward-bisected interval that contains CI_c(k), so its σ̄² is at least as large.

**Why the random σ̄² is handled by intersection, not substitution (item (c)).** σ̄_c²(k) is a function of the same draws as μ̂_c(k). Treating it as a known variance inside the MGF of L1 would be invalid (the bound would be conditioned on a quantity correlated with the deviation). We never do this: E_main is a statement with the deterministic (given 𝒜) σ², proved by L1 alone; E_var is a separate event; and on E_main ∩ E_var the pointwise inequality V_k ≤ V̄_k together with the monotonicity of V ↦ √(2βV) + bβ/3 gives

  J(π*_q) − J(π) ≤ Ĵ_k(π*_q) − Ĵ_k(π) + w_k(π*_q, π; V̄_k)  for all q, π ∈ Π_{B_q}, k.   (★)

## 4. The union and the ledger (item (d))

P(E_main^c | 𝒜) ≤ Σ_q |Π_{B_q}| · K · e^{−β} and P(E_var^c | 𝒜) ≤ 2 · |𝒞| · K · e^{−x_v}. With

  β = ln(Σ_q |Π_{B_q}| · K / δ_main),  x_v = ln(2 · S·A · K / δ_var),

P(E^c | 𝒜) ≤ δ_main + δ_var = δ.

**CR9 development-half ctx (computed by `mc_fdc.part0_cr9_ledger` from public ctx quantities only; no outcome, no permutation seed):**

| Quantity | Value |
|---|---|
| S, A, |Π_all| | 9, 2, 512 |
| per-problem |Π_{B_q}|, budgets 0.10…0.80 | 14, 44, 69, 111, 144, 186, 212, 243, 256, 269, 300, 326, 368, 401, 443 |
| Σ_q |Π_{B_q}| | **3386** |
| K | 20 (log-spaced 50,000 → τ_R = 6,989,793) |
| C_var = S·A | 18 |
| β = ln(3386 · 20 / 0.045) | **14.2242** |
| x_v = ln(2 · 18 · 20 / 0.005) = ln(144,000) | **11.8776** |
| δ_main + δ_var | 0.045 + 0.005 = **0.05** |
| hard-coded β = 14.1: 3386 · 20 · e^{−14.1} + 0.005 | 0.05095 + 0.005 = **0.05595 > δ** (invalid) |

The ledger is recomputed by the code from ctx for every layer; on the synthetic MC configurations of §9 it varies with S and the feasible classes (reported per configuration in `summary.json`).

Per-problem feasible classes are subsets of Π_all, so the challenger set used by the certificate for problem q is exactly the index set of the q-th block of the union; no other pairs are needed because the proof only uses pairs with first argument π*_q (QFC §9 option 1, refined from Q·|Π| = 7680 to Σ_q |Π_{B_q}| = 3386).

## 5. Theorem FDC-1

**Theorem FDC-1.** Under the specification of §1 and the probability space of §2, for every ε ≥ 0 (also per-problem ε_q),

  P( ∃ k ≤ K, ∃ q ≤ Q : U_q(k) ≤ ε and J(π*_q) − J(π̂_q(k)) > ε | 𝒜 ) ≤ δ_main + δ_var = δ,

for every schedule 𝒜, hence also unconditionally. Here π̂_q(k) may be any Π_{B_q}-valued function of the data observed up to checkpoint k (the FDC rule is the Ĵ-argmax).

**Proof.** Let E = E_main ∩ E_var; P(E^c | 𝒜) ≤ δ by §3–§4. Fix an outcome in E and any (k, q) with U_q(k) ≤ ε. Since π̂_q(k) ∈ Π_{B_q}, apply (★) with π = π̂_q(k):

  J(π*_q) − J(π̂_q(k)) ≤ Ĵ_k(π*_q) − Ĵ_k(π̂_q(k)) + w_k(π*_q, π̂_q(k); V̄_k) = Δ̂_k(π*_q, π̂_q) + w_k(π*_q, π̂_q).

The right-hand side is the bracket of U_q(k) at the challenger π′ = π*_q, which belongs to Π_{B_q}; U_q(k) is a maximum over all of Π_{B_q}, so the right-hand side is ≤ U_q(k) ≤ ε. (The code's pair width is computed for (π′, π̂) with the same differing cells, the same live set and the same b as for (π*_q, π̂), because V and b are symmetric in the order of the pair.) Hence no false certification occurs on E, at any k and any q simultaneously. The data-dependence of π̂_q(k) is absorbed by the union over π ∈ Π_{B_q}; that of the certification checkpoint by the union over k. Integrating over 𝒜 gives the unconditional statement. ∎

Remarks.
- E does not depend on ε, so the statement holds for all ε simultaneously (the MC checks ε ∈ {0, 0.0025, 0.01} on the same streams).
- The proof never uses the identity of π̂ beyond feasibility, never prunes challengers using data, and never uses the true σ² in the code path.
- **Corollary (stream-level FWER and FDR-type ratio).** With V_s false and R_s total certifications in a stream, P(V_s > 0) ≤ δ and E[V_s / (R_s ∨ 1)] ≤ δ. FCR_old is not controlled (QFC §5, 8.98% counterexample).

## 6. Stopping and observation windows (item (e))

Certificates exist only at the K fixed checkpoints, and E controls every (k, q) cell of the K × Q grid at once. Every observation protocol is a (possibly data-dependent) subset of this grid, evaluated on the same probability space:

- **W1, stop at 12/15.** k* = the first checkpoint with ≥ 12 certified (a stopping time). Certifications made at k ≤ k* are cells of the grid; a frozen answer certified at k₀ is the policy π̂_q(k₀), and its correctness J(π*_q) − J(π̂_q(k₀)) ≤ ε is a fixed fact once true at k₀. So P(any false certification in W1) ≤ P(E^c) ≤ δ.
- **W2, run to τ_R and certify all 15.** The complete schedule 𝒜 defines counts at k > k*, independent of whether the method "stopped", so the continuation is the same random experiment. Every sticky or non-sticky certification at any (k, q) is covered; P(any false certification in W2) ≤ δ.
- Certifying between checkpoints is not allowed: it would add uncounted (t, q) cells to the union.
- Under FDC the 50/50 rule is not adaptive, so the "sampling to a foregone conclusion" failure of QFC §6 does not arise: n_c(k) is fixed given 𝒜.

## 7. Boundary cases (item (f))

| Case | Treatment | Why it is covered |
|---|---|---|
| n_c(k) = 0 for a cell touched by (π′, π̂) | width = +∞, so U_q(k) = +∞ and no certification | the event in E_main is sure; the certificate cannot fire |
| μ̂_c ∈ {0, 1} with 1 ≤ n_c < N_c | CI_c is the Bernstein inversion set; for μ̂ = 0 it is [0, u] with u ≥ x_v/(3n) > 0, so σ̄² ≥ u(1−u) > 0 (symmetric for μ̂ = 1) | L2 holds for every μ̂; the plug-in variance μ̂(1−μ̂) = 0 would drop the √V term (this is one of the defects detected by the power control) |
| pool exhausted, n_c = N_c | μ̂_c = μ_c exactly; dropped from V and b; CI = {μ̂_c}, σ̄² = 0 | exhaustion is 𝒜-measurable (§2), so the live set used in L1 is fixed given 𝒜; QFC §2 convention (ii) |
| one arm exhausted in a segment | later arrivals planned to that arm are re-selected to the other arm with the pre-drawn uniform | re-selection uses only 𝒜 and N, so counts stay 𝒜-measurable |
| both arms of a segment exhausted | the segment is exact; under a permutation schedule this happens exactly at the segment's last arrival, so no arrival is ever skipped (skip branch kept, billed; MC: 0 skips) | contributions of that segment are exact and dropped |
| π̂ and π′ differ only in exhausted cells | V = b = 0, width = 0, Δ̂(π′, π̂) = Δ(π′, π̂) exactly | strict ">" in L1: probability 0 |
| several optimal policies | π*_q chosen by a deterministic rule; the bound J(π*_q) − J(π̂) is the same for every optimal choice | π*_q is fixed and unknown |
| final checkpoint T_K = τ_R | every pool exhausted; U_q = max Δ̂ over Π_{B_q} = 0 at the exact argmax; always correct | exact |

## 8. Relation to r4 and to the methodology

- Same L1 and L2 as QFC; changes are (i) the design (pool-proportional → frozen 50/50), (ii) the union object (Q·|Π_all| → Σ_q |Π_{B_q}|, which uses the fact that each problem's certificate only maximises over its own feasible class), and (iii) the ledger (δ_main, δ_var, C_var) = (0.045, 0.005, 18) instead of (0.04, 0.01, 48). All three are covered by the proof above without new lemmas.
- Lemma L2 may be replaced by any WoR CS of the same budget (QFC §3); FDC does not do so (frozen).
- No finite-population correction is used anywhere.

## 9. Numerical verification (seed 42)

Numerical verification (filled by control plane 2026-10-03 from `exp/results/pilots/r5_fdc_spec_theory/summary.json`; no numbers changed):

- **Ledger (CR9 dev ctx, printed by code):** Σ_q|Π_{B_q}| = 3386, K = 20, C_var = 18, δ_main = 0.045, δ_var = 0.005 ⇒ β = 14.2242, x_v = 11.8776; union bound 0.045 + 0.005 = 0.05. With β = 14.1 the bound is 0.05595 > δ.
- **FDC Monte Carlo:** 60 synthetic finite-pool configs (40 near-tie, 20 early-exhaustion) × 500 streams, ε ∈ {0, 0.0025, 0.01}, windows W1 (stop at 12/15), W2 (run to τ_R, all 15) and W2any. False streams: **0** in all 9 window×ε cells (1,350,000 certifications, 387,818 before the final checkpoint). Pooled one-sided 95% CP upper bound 9.99e-5 per cell; worst single config 0.00597. Direct event audits: E_var misses 0, E_main misses 0 over 118,320,000 (q, π, k) events. Edge-case counters exercised: n = 0 cells 483,124; exhausted cells 523,949; μ̂ ∈ {0,1} live cells 1,811,341; both arms exhausted in a segment 133,500; re-selection arrivals 194,498,296.
- **Power controls (same harness):** plug-in variance only — max config rate 0.002 (no config > δ); β = ln(1/δ) only — max 0.01; both (plug-in variance + β = ln(1/δ)) — pooled 731/30,000 false streams at W1/ε=0, max config rate 0.234, 9 configs > δ. The harness therefore detects an invalid certificate.
- **Schedule independence:** counts and digests invariant under outcome relabelling (Part C).


## 10. Proof self-check (sign-off; format of QFC §10)

| # | Item | Status |
|---|---|---|
| 1 | 𝒜 = (seg_seq, arm_seq, resel_u) is complete (all τ_R arrivals), generated before any outcome, independent of (Π_c)_c (independent child streams) | ✔ signed |
| 2 | Under frozen 50/50 with exhaustion re-selection, n_c(k), exhaustion and live/n=0 indicators are functions of 𝒜 and N only (numeric Part C: counts and digests invariant under outcome re-labelling) | ✔ signed |
| 3 | Fact 0 ⇒ L1 and L2 hold conditional on 𝒜 at every k (QFC §2–§3, signed in r4) | ✔ signed |
| 4 | π*_q is fixed: w, κ, B_q, FEAS_TOL, ε, Π_{B_q} are fixed before the stream; deterministic tie rule; π*_q ∈ Π_{B_q} (Π_{B_q} ≠ ∅, all-control has cost 0) | ✔ signed |
| 5 | E_main stated with the true σ²; each (q, π, k) event fails w.p. ≤ e^{−β} by L1 applied to −a; ∞-width and no-live-cell events are sure | ✔ signed |
| 6 | E_var: per (c, k) miss ≤ 2e^{−x_v}; outward bisection; exhausted/n=0 conventions contain μ_c surely; union 2·S·A·K | ✔ signed |
| 7 | Random σ̄² handled by intersection + monotonicity of the width in V; never substituted into the MGF | ✔ signed |
| 8 | Union size Σ_q |Π_{B_q}| · K counted exactly by code from ctx; β, x_v printed from code: 14.2242, 11.8776 on CR9 dev; β = 14.1 gives 0.05595 > δ | ✔ signed |
| 9 | Proof covers any data-dependent π̂_q(k) ∈ Π_{B_q} and the max over all feasible challengers (π*_q is one of them); no data-based pruning | ✔ signed |
| 10 | Pair width in code equals w_k(π*_q, π̂) for the pair (π′ = π*_q, π̂): symmetric V and b, same live set (`pair_terms`) | ✔ signed |
| 11 | Certification only at the K fixed checkpoints; FWER ≤ δ for W1 (stop at 12/15) and W2 (to τ_R, all 15), sticky or not | ✔ signed |
| 12 | Boundary cases n = 0, μ̂ ∈ {0,1}, exhaustion, single-segment double exhaustion, differences only in exhausted cells, final checkpoint (§7) | ✔ signed |
| 13 | Corollary P(V>0) ≤ δ and E[V/(R∨1)] ≤ δ; FCR_old not implied | ✔ signed |
| 14 | Ledger: δ_main + δ_var = 0.045 + 0.005 = 0.05; no override path; deletions of §1 respected (no FPC, no two-stage, no CS module) | ✔ signed |
| 15 | MC: 0 false certifications for FDC in both windows and all ε; direct E_var / E_main miss counts reported | ✔ signed (control plane 2026-10-03, verified from summary.json: FDC 0/1,350,000 false over 9 window×ε cells; E_var/E_main misses 0) |
| 16 | Power: the invalid control (plug-in variance + β = ln(1/δ)) produces violations above δ on the same harness | ✔ signed (control plane 2026-10-03, verified from summary.json: ctrl_both max config rate 0.234 > δ, 9 configs > δ at W1/ε=0) |
