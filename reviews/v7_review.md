# external reviewer review of lock v7 (FDC-BF confirmatory) — 2026-10-04

external reviewer MCP (config model, approval-policy never, read-only sandbox), thread `01a10325-e633-7ff0-b01e-c32b6aec86fe`. Two rounds, as planned.

## Round 1

### (a) Validity of FDC-BF (Theorem FDC-bet-1): no validity bug found

- **Poisson-binomial representation.** It holds for every admissible hypergeometric law, allowing Bernoulli parameters 0 and 1. Zero roots correspond to deterministic successes; padding with deterministic zeros gives n summands. For N > 1 the variance is exactly n m(1−m)(N−n)/(N−1). N = 1 is deterministic; in the code it is an exhausted cell with Ψ = 0.
- **Bennett for both signs of u.** Valid: apply the lemma to B_i − p_i or to its negative. The coefficients −w_s (challenger) and +w_s (centre) correctly give Y = Δ − Δ̂.
- **HG box.** Exact test inversion covers the true discrete pool mean, and the sup over the continuous interval is conservative. Bennett's sup is attained at clip(1/2). The running intersection preserves simultaneous coverage; an empty intersection already implies that the coverage event failed.
- **Data-dependent λ̂.** Legitimate. For each fixed (q, π, k), t(μ) is deterministic given the schedule, and E_var ∩ {Y > w} ⊆ {Y > t(μ)}. No union over λ is needed. The union over all feasible centres covers the data-dependent π̂; only the fixed true optimum is needed as challenger.
- **Ledger.** K Σ_q|Π_{B_q}| e^{−β} = 0.045 and 2SAK α_side = 0.005. Cell independence is conditional on the outcome-free schedule, not on the boxes.
- **Implementation.**
  - Exhausted cells contribute 0.
  - Touched empty cells give a width of ∞.
  - A non-finite width is mapped to ∞, which is conservative.
  - The λ grid affects only tightness.
  - No deviation affects FDC-BF.

### (b) Lock gaps

1. **LOCK-GAP: existing results could be resealed.** Counterexample: edit N80_pen for seed 33010 (outside the R1 seeds), rerun the completed task, and regenerate the replica report. The replacement seal would pass without any history rewrite.
   - **Fixed.**
     - `v7_seal.write_and_commit_seal` refuses if a seal or ref exists.
     - `run_r5s_v7.py` refuses to rerun or resume a sealed eval task.
     - `v7_seal.verify_seal` adds a history anchor: the seal file and the ref file must each have exactly one commit, the seal's commit must equal the ref's, and the ref must equal its commit.
     - Tests: `test_v7_seal_honest_passes_and_reseal_refused`, `test_v7_seal_recommitted_after_edit_refused`.
2. **LOCK-GAP: the lock lacked a historical anchor.** Editing the lock and recomputing its hash could pass the gate.
   - **Fixed.** `prereg_v7.lock_history_check` requires the lock file to be committed exactly once and to equal that commit. It is called by `load_locked_addendum`, and therefore by the gate, the Lenta v7 env and the analysis.
   - Test: `test_lock_history_anchor`.
3. **LOCK-GAP: disclosure inconsistency.** The draft said "blocks A/B confirmatory", but B is descriptive. It also did not state that v7 relaxes the threshold from v6's 0.70 to 0.80.
   - **Fixed** in the builder statement and in md §0:
     - only A is confirmatory, over fresh replay randomisation conditional on the already-used CR9 eval table;
     - the design was informed by the published v6 results, and only the variant comparison used dev data;
     - the 0.80 threshold is relaxed relative to v6 and does not reverse v6's downgrade.

### Judged acceptable

- Seed freshness: no saved hits, which supports "no saved run" but not more.
- The Lenta v7 gate is correctly placed.
- Rivals: frozen configs, and HC-WoR keeps its v6 tuning.
- The IUT needs no multiplicity correction. The percentile bootstrap is approximate.
- The 0/200 gate is legitimate. The CP UB for 0/200 is about 1.49%.
- stop_k = 15 is sound: no method uses stop_k before the stop, and N80_pen uses the errors at the 12/15 crossing.

### MINOR

- §2.4 of the exploration doc overclaimed domination of arbitrary fixed-λ betting. It is now restricted to optimising the same MGF bound.
- The exploration summary called FDC-BF "Bennett ∧ KL" (that is FDC-KLF). Erratum added; the code was always kind = bennett.

## Round 2

external reviewer checked the diffs and replied: **"OK to lock"**.
