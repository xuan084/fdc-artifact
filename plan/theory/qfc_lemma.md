# QFC theory: Lemma L1, variance UCB, Theorem 1, error ledger (task `r4_theory_qfc`)

Status: pilot run 2026-10-02, seed 42. Numerical checks: `exp/code/dsswm/theory_checks/mc_l1.py`, `exp/code/dsswm/theory_checks/width_diag.py`. Outputs: `exp/results/pilots/r4_theory_qfc/{summary.json, mc_coverage.csv, cert_coverage.csv, nonvacuity_diag.json}`.

---

## 中文摘要

- **引理 L1 已完整证明，没有缺口。** 证明链为：池内预置换 + 日程与 outcome 独立 ⇒ 条件于 𝒜 时，各单元是相互独立的不放回样本；Hoeffding（1963，Thm 4）凸序 ⇒ 每个单元的 MGF ≤ 有放回情形；单次有界抽样的 Bennett/Bernstein MGF 界，以及 φ(u)/u² 单调 ⇒ 用全局 b 统一；Chernoff ⇒ P(Δ̂−Δ > √(2Vx)+bx/3) ≤ e^{−x}。对 n_c = 0 的单元，若系数非零，宽度记为 +∞；耗尽单元（n_c = N_c）的估计是精确的，可以从 V 和 b 中删去（严格收紧）。
- **方差上界（L2）**：对二值池，σ² = μ(1−μ)。把 L1 用到单个单元，再对 μ 反解，得到一个合法的不放回 CI（Bernstein 反演），然后取区间内 μ(1−μ) 的最大值。WSR20 的不放回 CS 可以直接替换，它对时间一致，所以 K 的并集是多余的。对非二值、取值在 [0,R] 的 outcome，用 σ̄² = R²/4 与 b·R，这一步不消耗 δ。
- **Thm 1 已完整证明。** 证明只用 E_main ∩ E_var，对任何数据依赖的 π̂、任何只在检查点上发生的停止、全部 15 题同时成立；推论为 E[V/(R∨1)] ≤ P(V>0) ≤ δ。FCR_old 没有这个保证，external reviewer 反例给出 8.98%，已用数值复核。
- **自适应边界（需要修正 methodology §1.5）**：Thm 1 覆盖池比例分配、A-Ney、A-XY，因为它们都是冻结的、非自适应的分配。对自适应基线，**"固定计数 MGF 界 + 凸序 + 检查点并集"不成立**：计数依赖数据时，固定 n 的界会被"采到想要的结论"击穿。数值演示：违例率 0.0845，界为 e^{−3} = 0.0498，比值 1.70。合法路线是**单元级时间一致 CS**（WSR20，或对 n 取并集），再按矩形（半径和）组合。B1 / B4 这样实现是严格的。B2 / B3 / Peace 的 valid 变体是泛函级 GLR 阈值，在不放回 + 自适应计数下**没有现成证明**，必须改用矩形证书，或者降级标注为"名义有效，经验检验"。
- **S=16 min_t-DP**：由 AM-GM 可知，对任意有限 t 网格，√(2LV) ≤ L/t + tV/2，所以 U_DP ≥ U_enum，是合法的上界。external reviewer 反例的数值为 U_enum = 1，U_DP = 13/12。
- **数值核验（pilot）**：
  - 200 个有限池配置 × 10⁴ 次：最坏违例比 = **0.146**（门槛 ≤ 1.2）。对照用的 CLT 分位宽度在 62/200 个配置上违例，最高 12.8 倍，说明检验有检出力。
  - 完整证书：Hillstrom S=6 共 1000 条流 × 3 个 ε，加上 40 个合成近并列压力配置 × 500 条流，**违例 0 次**。朴素证书在合成配置上的违例率最高 72.6%。
  - 确定性检查（凸序 MGF、Bernstein MGF、AM-GM、13/12、8.98%）全部通过。
- **G-theory 门：通过。**
- **重大风险（不属于本门，但必须上报 planner）**：严格 QFC 证书在 Hillstrom S=6 上**几乎不可能在全表耗尽前认证**。即使用 oracle μ̂，在 80% 到达处（51,418），U_q 的中位数仍为 0.018，15 题里只有 1 题 ≤ 0.01；12/15 只能在 τ_R 达到，而 τ_R 按协议计为删失。冻结的 ε 网格 {0.0025…0.01} 低于证书可达的宽度，G1 / 主终点将不可达。可用的收紧见 §9。

---

## 1. Setting

- Finite populations (pools) c ∈ 𝒞 (HR S=6: c = (s, a), |𝒞| = 18). Pool c is a fixed multiset {y_{c,1}, …, y_{c,N_c}} ⊂ [0, 1] with mean μ_c and **finite-population variance** σ_c² = N_c⁻¹ Σ_j (y_{c,j} − μ_c)². Binary pools have σ_c² = μ_c(1 − μ_c).
- Replay randomness: for each pool, an independent uniform random permutation Π_c of its records. A complete schedule 𝒜 (arrival segments and assigned arms) is generated from the permutation seed **before** any outcome is read, and is independent of (Π_c)_c. The design-based probability space contains only (𝒜, (Π_c)_c). The table itself, and every quantity computed from it before the stream (ε, segments, cost tiers, w_s, A-Ney / A-XY allocations frozen on D), is non-random.
- Checkpoints: K = 20 fixed arrival counts T_1 < … < T_K. At checkpoint k the count n_c(k) = n_c(k; 𝒜) is a deterministic function of 𝒜, capped at N_c. Under pool-proportional replay (a random permutation of the whole table), n_c(T) ≤ N_c always.
- Estimates: μ̂_c(k) = n_c(k)⁻¹ Σ_{i ≤ n_c(k)} y_{c,Π_c(i)}.
- Policies: Π_all = [A]^S, with J(π) = Σ_s w_s μ_{s,π(s)}. The weights w_s are the full-table segment shares, which are known constants. For an ordered pair, Δ(π′, π) = J(π′) − J(π) = Σ_c a_c μ_c with a_{(s,π′(s))} = +w_s and a_{(s,π(s))} = −w_s whenever π′(s) ≠ π(s), and 0 otherwise. Δ̂ is the same expression with μ̂.

**Fact 0 (conditional structure).** Conditional on 𝒜, the vectors (y_{c,Π_c(1)}, …, y_{c,Π_c(n_c)}) for c ∈ 𝒞 are independent, and each one is a uniform without-replacement (WoR) sample of size n_c from pool c.
*Proof.* n_c is fixed given 𝒜. Since Π_c ⟂ 𝒜, the first n_c entries of a uniform permutation form a uniform WoR sample. The Π_c are mutually independent. ∎

## 2. Lemma L1 (stratified WoR functional Bernstein, conditional on 𝒜)

**Statement.** Fix 𝒜, a checkpoint k, and coefficients a ∈ ℝ^𝒞. Let 𝒞_a = {c : a_c ≠ 0} and assume n_c ≥ 1 for c ∈ 𝒞_a. Let 𝒞_live = {c ∈ 𝒞_a : n_c < N_c}, and define

  V = Σ_{c ∈ 𝒞_live} a_c² σ_c² / n_c,  b = max_{c ∈ 𝒞_live} |a_c| / n_c (b = 0 if 𝒞_live = ∅).

Then for every x > 0,

  P( Σ_c a_c (μ̂_c − μ_c) > √(2 V x) + b x / 3 | 𝒜 ) ≤ e^{−x}.

The bound also holds with V and b computed over all of 𝒞_a (the planner's form). Those values are ≥ the 𝒞_live values and give a weaker statement.

*Conventions.* (i) If n_c = 0 for some c ∈ 𝒞_a, the width is +∞ and no certification can use this pair. (ii) Exhausted cells (n_c = N_c) satisfy μ̂_c = μ_c exactly, so they contribute 0 to the deviation and are dropped from V and b. (iii) The event is written with a strict ">". When 𝒞_live = ∅, the deviation is identically 0 and the probability is 0.

**Proof.**
1. *Reduction.* Z := Σ_{c ∈ 𝒞_live} (a_c / n_c) Σ_{i ≤ n_c} (X_{c,i} − μ_c), where X_{c,i} = y_{c,Π_c(i)}. Exhausted cells contribute exactly 0.
2. *Factorisation.* By Fact 0, for λ ≥ 0, E[e^{λZ} | 𝒜] = Π_{c ∈ 𝒞_live} E[exp(λ (a_c / n_c)(Σ_i X_{c,i} − n_c μ_c)) | 𝒜].
3. *Hoeffding convex order.* Hoeffding (1963), Theorem 4: if X_1..X_n is a WoR sample and Y_1..Y_n is a with-replacement (i.i.d. uniform) sample from the same finite population, then E f(Σ X_i) ≤ E f(Σ Y_i) for every continuous convex f. Take f(s) = exp(λ (a_c / n_c)(s − n_c μ_c)), which is convex in s for either sign of a_c. Each factor is then ≤ Π_{i ≤ n_c} E exp(λ U_{c,i}), where the U_{c,i} = (a_c / n_c)(Y_{c,i} − μ_c) are i.i.d.
4. *One-draw MGF (Bennett).* E U_{c,i} = 0, E U_{c,i}² = a_c² σ_c² / n_c² =: v_c, and U_{c,i} ≤ (|a_c| / n_c) · max(μ_c, 1 − μ_c) ≤ |a_c| / n_c =: b_c ≤ b. For u ≤ b_c and λ ≥ 0, e^{λu} − 1 − λu ≤ u² φ(λ b_c) / b_c² with φ(t) = e^t − 1 − t, because t ↦ φ(t)/t² is nondecreasing on ℝ. Hence log E e^{λU} ≤ v_c φ(λ b_c) / b_c² ≤ v_c φ(λ b) / b², and monotonicity of φ(t)/t² again lets one global b replace every b_c.
5. *Sub-gamma.* φ(t) ≤ t² / (2(1 − t/3)) for 0 ≤ t < 3. Summing over the n_c draws and the live cells gives log E[e^{λZ} | 𝒜] ≤ V λ² / (2(1 − bλ/3)) for 0 ≤ λ < 3/b.
6. *Chernoff.* A right-tail sub-gamma variable with variance factor V and scale b/3 satisfies P(Z > √(2Vx) + (b/3) x) ≤ e^{−x} (Boucheron, Lugosi and Massart 2013, §2.4). ∎

*Range.* The lemma requires y ∈ [0, 1]. For outcomes in [0, R], apply it to y/R; equivalently, replace b by R·b and σ_c² by the variance of the unscaled pool.
*Two tails.* The lower tail follows by applying L1 to −a. In §4 every ordered pair is one event, so the lower tail of (π′, π) is the upper tail of (π, π′).
*What L1 does not use.* L1 needs no independence across checkpoints. It makes no finite-population correction: Hoeffding's order discards the FPC, so the bound is conservative near exhaustion (see §9).

## 3. Lemma L2 (variance upper bound)

**Binary pools.** For fixed (c, k) with 1 ≤ n_c < N_c, define

  CI_c(k) = { m ∈ [0, 1] : |μ̂_c − m| ≤ √(2 m(1 − m) x_v / n_c) + x_v / (3 n_c) }.

Set CI_c = {μ̂_c} if n_c = N_c, and CI_c = [0, 1] if n_c = 0. Let σ̄_c² = max_{m ∈ CI_c} m(1 − m).
*Validity.* μ_c ∉ CI_c(k) iff one of the two L1 tail events for the single cell (a = ±1, σ² = μ_c(1 − μ_c)) occurs, so P(μ_c ∉ CI_c(k) | 𝒜) ≤ 2 e^{−x_v}. CI_c is an interval, because m ↦ ±(m − μ̂_c) − g(m) is convex and g is concave. The implementation computes it by bisection and returns the outer endpoints, so the returned interval contains the exact one.
*Union.* E_var = {μ_c ∈ CI_c(k) ∀ c ≤ C_var, ∀ k ≤ K}, with x_v = ln(2 C_var K / δ_var). Then P(E_var^c | 𝒜) ≤ δ_var. The plan freezes C_var = 48, which is valid for S=6 (18 cells, so conservative) and for S=16 (48 cells). With δ_var = 0.01 and K = 20, x_v = 12.165.
*Alternative.* Any WoR CS with the same budget can replace CI_c, for example the Waudby-Smith & Ramdas (2020) WoR empirical-Bernstein or betting CS planned in `r4_setup_fp_stats`. WSR20 CSs are time-uniform, so the K factor can be dropped (x_v = ln(2 C_var / δ_var)). Keeping K is merely conservative.
*Monotonicity.* On E_var, σ_c² ≤ σ̄_c², hence V ≤ V̄, and the width √(2Vx) + bx/3 is nondecreasing in V.

**Non-binary outcomes in [0, R]** (for example truncated spend): σ_c² ≤ R²/4 deterministically (Popoviciu), so σ̄_c² = R²/4 and b → R·b. This spends no δ. Tighter WoR variance bounds are not enabled in r4.

## 4. Main event, union, and the shared ledger

E_main = { ∀ k ≤ K, ∀ (π′, π) ∈ Π_all² : Δ(π′, π) − Δ̂(π′, π) ≤ √(2 L₁ V(π′, π; k)) + b(π′, π; k) L₁ / 3 },
with V and b computed with the true σ² from the coefficients a(π′, π) of §1 (live cells only).

By L1 applied to −a, each (pair, k) fails with probability ≤ e^{−L₁}. There are |Π_all|(|Π_all| − 1) K ≤ A^{2S} K such events. With

  L₁ = 2 S ln A + ln(K / δ_main)  (S=6, A=3, K=20, δ_main=0.04 ⇒ L₁ = 19.398)

we get P(E_main^c | 𝒜) ≤ δ_main.

**Why the 15 problems share one event.** Every feasible class Π_{B_q} (any budget, any cost tier) is a subset of Π_all, and E_main is a statement about all pairs in Π_all. No union over problems is needed, and the same holds for any further problems on the same pools.

On E_main ∩ E_var, replacing V by V̄ (the σ̄² version) keeps the inequality (§3, monotonicity).

## 5. Theorem 1 (certificate validity)

**Rule.** At checkpoint k and for problem q, let π̂_q(k) ∈ Π_{B_q} be any data-dependent choice (the argmax of Ĵ over Π_{B_q} in QFC), and let

  U_q(k) = max_{π′ ∈ Π_{B_q}} [ Δ̂(π′, π̂_q) + √(2 L₁ V̄(π′, π̂_q)) + b(π′, π̂_q) L₁ / 3 ].

Certify q at k if U_q(k) ≤ ε. Stopping or freezing rules are allowed if they act only at the K checkpoints.

**Theorem 1.** Conditional on the pre-generated schedule 𝒜,

  P( ∃ k ≤ K, ∃ q : U_q(k) ≤ ε and J(π*_q) − J(π̂_q(k)) > ε | 𝒜 ) ≤ δ_main + δ_var = δ = 0.05.

The same bound holds unconditionally.

**Proof.** Let E = E_main ∩ E_var, so P(E^c | 𝒜) ≤ δ_main + δ_var by §3–4. On E, fix any (k, q) with U_q(k) ≤ ε. Since w and κ are known, Π_{B_q} is deterministic and π*_q ∈ Π_{B_q}. Then

J(π*_q) − J(π̂_q) = Δ(π*_q, π̂_q) ≤ Δ̂(π*_q, π̂_q) + √(2 L₁ V(π*_q, π̂_q)) + b L₁ / 3 ≤ Δ̂ + √(2 L₁ V̄) + b L₁ / 3 ≤ U_q(k) ≤ ε.

So no false certification occurs on E. The data-dependence of π̂_q is absorbed by the union over pairs, and the choice of stopping checkpoint is absorbed by the union over k. Since the bound holds for every 𝒜, it holds after integrating over 𝒜. ∎

Remarks:
- The proof never takes a rectangle union of per-cell CSs, so it does not reduce to a radius sum. It uses exactly the two events E_main and E_var.
- The theorem holds for every ε ≥ 0 simultaneously, for per-problem ε_q, for any tie-breaking rule, and for any additional problems whose feasible sets lie inside Π_all.
- **Corollary.** Let V_s be the number of false certifications in a stream and R_s the number of certifications. Then V_s / (R_s ∨ 1) ≤ 1{V_s > 0}, so E[V/(R∨1)] ≤ P(V > 0) ≤ δ.
- **FCR_old is not controlled.** FCR_old = Σ_streams V / Σ_streams R is a ratio of sums. external reviewer counterexample: 5% of streams certify all 15 problems wrongly and 95% certify 8 correctly. Then FWER = E[V/(R∨1)] = 0.05, but FCR_old = 0.75 / 8.35 = 0.0898 (checked numerically, `partC.C4`). S2 must therefore remain a separately tested acceptance condition.

## 6. Adaptive boundary (amends methodology §1.5)

Thm 1 requires n_c(k) to be a function of 𝒜 only.
- **Covered:** QFC with pool-proportional allocation, **A-Ney** (Neyman proportions frozen on D), and **A-XY** (XY design frozen on D). All three are deterministic functions of the fixed table plus 𝒜.
- **Not covered by "fixed-time MGF bound + Hoeffding order + checkpoint union".** When an algorithm's sampling depends on observed outcomes, the realised count n_c(k) is data-dependent. The fixed-n bound then holds for each deterministic n but not at a data-chosen n: the sampler can keep drawing until the deviation crosses the width ("sampling to a foregone conclusion"). A union over K checkpoints does not help, because within one checkpoint interval the adaptive count can take many values. Numerical demo (`partD`): one cell, N = 20,000, μ = 0.3, drawing one record at a time up to 10⁴ and stopping at the first crossing. The crossing rate is 0.0845, against the claimed bound e^{−3} = 0.0498 (ratio 1.70). The methodology sentence "B1, B4 and the valid variants of B2 / B3 / Peace … via fixed-time bound + Hoeffding order + checkpoint union are rigorous" is **false as stated**.
- **Valid route (R1, rectangle).** Each pool is still read in its own pre-permuted order. The j-th draw from cell c is y_{c,Π_c(j)} no matter when it is drawn, so cell c's whole prefix-mean path is independent of the other pools. An adaptive rule only selects which n is realised. A per-cell CS that is **time-uniform over n ∈ {1..N_c}** therefore stays valid at any data-dependent count. Examples:
  - the WSR20 WoR CS (time-uniform by Ville's inequality);
  - L2-type Bernstein inversion with a union over all n: x_c = ln(2 C N_c / δ), about ln(2 · 18 · 3,800 / 0.04) ≈ 15.0 for HR S=6;
  - a geometric grid with peeling.

  A union over cells gives a rectangle. The functional certificate is then max over the rectangle, Σ_c |a_c| r_c (a radius sum). **B1 and B4 implemented this way are rigorous**, provided `r4_setup_baselines_a` uses the time-uniform CS radii, not the fixed-n `functional_width` per cell.
- **Functional / GLR thresholds under adaptive WoR (B2-valid β_dir, B3-valid, Peace-valid).** No proof is available. Kaufmann–Koolen mixture thresholds assume i.i.d. (with-replacement) draws per arm. Under WoR, the conditional mean of the next draw is the mean of the remaining pool, not μ_c. Phase-wise arguments (RAGE / Peace) break for the same reason. A rigorous functional route exists via the WoR martingale with predictable remaining-mean (WSR20 §3; Serfling's (S_n − nμ)/(N − n) martingale), but it has not been derived for interleaved multi-cell functionals here. **Required action:** either (a) give these baselines the rectangle certificate from R1 while keeping their own sampling rules, and label them "valid-rect"; or (b) keep their functional thresholds but label them "nominal-valid (empirically checked)", relying on the 200-stream empirical FWER test already in `r4_setup_baselines_a/b`. In either case the paper must not call them "rigorous" through the checkpoint-union argument. The fav variants are unaffected, since no validity is claimed for them.

## 7. S = 16: min_t-DP upper bound

For L, V ≥ 0 and t > 0, AM-GM gives L/t + tV/2 ≥ 2√(L/t · tV/2) = √(2LV). Hence, for each π′ and each t,

  Δ̂ + √(2 L₁ V̄) + b L₁ / 3 ≤ Δ̂ + L₁/t + t V̄/2 + b̄ L₁ / 3,  with b̄ = max_c (1/n_c) · max_s w_s ≥ b.

Take the max over π′ in the relaxed feasible set. That set comes from costs rounded **down**; it contains Π_{B_q} and therefore π*_q. Then take the min over any finite grid T:

  U_enum ≤ min_{t ∈ T} max_{π′} [·] = U_DP.

So certification by U_DP ≤ ε implies U_enum ≤ ε, and Thm 1 applies verbatim. The S=16 layer uses its own ledger: L₁ = 32 ln 3 + ln(20 / 0.04) = 41.37, with separate δ_main and δ_var summing to 0.05.

For fixed t the objective is additive over segments: V̄ is a sum over differing segments, and L₁/t and b̄ L₁/3 are constants. A per-segment multiple-choice knapsack DP is therefore exact **for that t and the rounded costs**. Only the outer min–max order is a relaxation.

**external reviewer counterexample** (L = 1; (Δ̂, V) ∈ {(0, 1/2), (−1, 2)}): U_enum = max(0 + 1, −1 + 2) = 1. The minimax over t is attained where 1/t + t/4 = −1 + 1/t + t, i.e. t = 4/3, giving 13/12 (gap 1/12; a coarse grid gives 1.091). U_DP ≥ U_enum as required. Do not claim equality or convergence of the gap to 0.

## 8. Error ledger (single δ = 0.05 for the S=6 main claim)

| Event | Union size | x | Budget |
|---|---|---|---|
| E_main: ordered policy pairs × checkpoints | 3¹² × 20 | L₁ = 12 ln 3 + ln(20/0.04) = 19.398 | δ_main = 0.04 |
| E_var: cells × checkpoints × 2 tails | 48 × 20 × 2 | x_v = ln(1920/0.01) = 12.165 | δ_var = 0.01 |
| **Total (Thm 1)** | | | **δ = 0.05** |
| S=16 secondary layer (separate claim) | 3³² × 20; 48 × 20 × 2 | 41.37; 12.165 | 0.04 + 0.01 |
| Conversion family (separate claim) | same as S=6 | same | 0.04 + 0.01 |

Non-binary secondary outcomes use the deterministic σ̄² = R²/4 and spend nothing on E_var.

## 9. Numerical verification (pilot, seed 42; `summary.json`)

**A. L1 coverage.**
- Setup: 200 random finite-pool configurations, each with 10⁴ exact WoR replications (hypergeometric for binary pools; explicit permutation for non-binary).
  - 162 binary and 38 non-binary configurations; C ∈ {1, …, 18} cells; N_c ∈ [50, 5000]; μ ∈ [0.01, 0.5].
  - Random signed a_c with zeros.
  - Random non-adaptive schedules in four regimes: tiny (49), mid (64), large (42), deplete (45), including n_c = 1 and exhausted cells.
  - Both tails, x ∈ {2, 3, 5}.
- Result: max over configs, tails and x of empirical violation / e^{−x} = **0.146** (x = 2). The other values were 0.096 (x = 3) and 0.089 (x = 5). The worst one-sided 95% Clopper-Pearson upper ratio was 0.18. **Gate ≤ 1.2: PASS.**
- Power control: the CLT width z_{1−e^{−x}} √V violates the e^{−x} bound in 61–62 of 200 configurations, by up to 12.8×. The harness therefore detects invalid widths.

**B. Full certificate (Thm 1 event, including E_var).**
- Hillstrom S=6 (18 visit pools, w = full-table shares, 15 problems, K = 20 log checkpoints from 1,000 to 64,000, table-permutation schedule), 1,000 streams for each ε ∈ {0, 0.0025, 0.01}:
  - false-certification streams: 0/1000 (CP95 upper bound 0.003);
  - E_var misses: 0;
  - (π*, π̂) pair-coverage misses: 0.
- 40 synthetic near-tie configurations (S = 2, A = 3, gaps 0–0.08, fixed random allocation), 500 streams each, ε ∈ {0, 0.005, 0.02}: 0/20,000 false-certification streams at every ε, while about 13% of problem–checkpoint pairs were certified (15–20% at the final checkpoint).
- **Max violation rate 0.0 ≤ δ: PASS.**
- Contrast: the naive certificate (plug-in variance, Gaussian width, x = ln(1/δ), no union) has a false-certification stream rate of up to 72.6% on synthetic configurations and 1.5% on Hillstrom at ε = 0.01.

**C. Deterministic checks.** All pass:
- WoR ≤ WR log-MGF on 300 exact hypergeometric/binomial cases (max difference 2.8e-16);
- the Bernstein one-draw MGF inequality (2,000 cases);
- AM-GM (2·10⁵ cases);
- external reviewer 13/12;
- FCR_old = 0.0898.

**D. Adaptive demo.** Crossing rate 0.0845 vs e^{−3} = 0.0498 (§6).

**Non-vacuity warning (outside this gate; must go to planner / `r4_g_eps_select` / G1).** `width_diag.py` evaluates the strict certificate on Hillstrom S=6 at the best case: oracle μ̂ = μ and expected pool-proportional counts. Median U_q over the 15 problems by arrivals:
- 1,000 arrivals: 0.233;
- 11,109 arrivals: 0.049;
- 33,189 arrivals: 0.024;
- 51,418 arrivals: 0.018, with only 1 of 15 problems at ≤ 0.01.

At the last checkpoint (64,000) every pool is exhausted and the estimates are exact. Under the realistic sampling of Part B, certifications came only from that checkpoint: the per-checkpoint certification rate is 0.050–0.054, which equals the 1/20 share of the final checkpoint plus 0.4 percentage points at ε = 0.01. The top-2 gaps are 0.0001–0.003. Consequences:
- With ε ≤ 0.01, QFC reaches 12/15 only at τ_R, which the protocol censors, so N₈₀ is censored for every stream.
- Every valid baseline is wider still, so the primary endpoint cannot show a significant difference.

Options, each a legitimate, provable tightening:
1. **Q-star union.** Thm 1's proof only uses pairs (π*_q, π). Since π*_q is deterministic, the union over {(π*_q, π) : q ≤ 15, π ∈ Π_all} × K suffices, which gives x = S ln A + ln(QK/δ_main) = 15.5 instead of 19.4. Widths shrink by about 11% (median U at 51k: 0.016).
2. **Finite-population correction.** A Bernstein–Serfling bound (Bardenet & Maillard 2015) would shrink the width near exhaustion by about √(1 − n/N). It needs its own proof before it can be used.
3. **Raise ε,** or certify against a coarser target. A change to the ε grid is a planning decision.

Without at least one of these changes, G1 is expected to fail.

## 10. Proof self-check (sign-off)

| # | Item | Status |
|---|---|---|
| 1 | Schedule 𝒜 is outcome-independent, K = 20 fixed, n_c(k) a function of 𝒜 (Fact 0) | ✔ signed |
| 2 | Hoeffding Thm 4 applies: exponential is convex for both signs of a_c; pools independent ⇒ MGF factorises | ✔ signed |
| 3 | Bernstein constants: U ≤ \|a_c\|/n_c ≤ b, φ(t)/t² monotone ⇒ global b; φ(t) ≤ t²/(2(1−t/3)); Chernoff ⇒ √(2Vx)+bx/3 | ✔ signed (numeric C1b) |
| 4 | Range [0,1]; rescaling for [0,R]; σ² is the finite-population variance | ✔ signed |
| 5 | n_c = 0 ⇒ width +∞; exhausted cells exact and droppable; strict inequality handles V = b = 0 | ✔ signed |
| 6 | L2 inversion valid (2e^{−x_v} per cell-checkpoint); interval shape; outward bisection; union 48 × K × 2 | ✔ signed |
| 7 | E_main union 3¹² × K with x = L₁ ⇒ δ_main; one-sided per ordered pair covers both tails | ✔ signed |
| 8 | 15 problems share E_main because feasible classes ⊆ Π_all; π*_q deterministic and feasible | ✔ signed |
| 9 | Thm 1 proof uses only E_main ∩ E_var; data-dependent π̂ and checkpoint stopping covered | ✔ signed |
| 10 | Corollary E[V/(R∨1)] ≤ δ; FCR_old not implied (8.98% counterexample) | ✔ signed |
| 11 | Adaptive boundary: QFC / A-Ney / A-XY covered; checkpoint-union argument for adaptive baselines refuted; valid R1 route given; B2/B3/Peace valid variants need relabel or rectangle | ✔ signed **with amendment to methodology §1.5** |
| 12 | min_t-DP is a valid upper bound for any finite grid and down-rounded costs; external reviewer 1 vs 13/12 | ✔ signed |
| 13 | Ledger sums to δ = 0.05; S=16 and conversion on separate ledgers | ✔ signed |

All 13 items are signed, and there are no gaps in L1 or Thm 1. **G-theory gate: PASS.** Item 11 requires the methodology amendment above before the baselines' "valid" labels are locked. The non-vacuity warning in §9 is a separate, pre-gate risk for G1.
