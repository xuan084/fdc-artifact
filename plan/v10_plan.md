# v10 plan (authors, 2026-10-05): significance push after the r4 reviews

## Starting point

- **r4 reviews.** The critic scored 5 (`writing/review_r4_critic.md`); external reviewer scored 6 (`reviews/paper_r4_review.md`).
  - Both agree that rigour, numbers and preregistration are now at main-track level.
  - The bar of 7 needs a stronger source of significance:
    - the confirmatory logs have only 9 segments × 2 arms, with enumerable policy classes (≤ 4,096);
    - both tables are reused;
    - the novelty is a synthesis of known tools (Hoeffding 1963 convex order already transfers fixed-design bounds to WoR).
- **Decision point 10-15 (user rule):** submit to WWW'27 only if every mandatory item is done and both reviews are ≥ 7. Otherwise retarget to the next same-tier venue; never submit a thin version.

## Workstream 1 (main lever): certification over exponentially large segment-policy classes, FDC-DP

**Idea.**
- The direction-level Chernoff objective for a challenger π against the centre π̂ is separable across segments for fixed λ: Σ_s ψ_s(λ; π̂(s), π(s)) − λ Δ̂_s(π̂(s), π(s)).
- So for each λ on the grid, the worst feasible challenger under the knapsack budget is a multiple-choice knapsack DP over segments, with no enumeration of the A^S policies.
- The union ledger becomes log(Σ_q |Π_q|) ≤ S log A + log Q. This is linear in S, so S = 32–64 segments is affordable.
- The open question is the order of quantifiers. The certificate needs ∀π ∃λ; a single λ for all π is valid but conservative. Remedies: branch-and-bound or a per-λ-cell partition of the challenger space, or a λ-grid union in the ledger. Theory decides; validity first.

**Data.** Uplift-score segmentations with S ∈ {16, 32, 64}: quantiles of a score model fit on a split disjoint from the replay rows.
- Lenta: 687k rows; eval half used descriptively once, in v7 B.
- X5: eval half used in v8 and v9.
- Disclose the table reuse. The new segmentation is a new problem on old rows.

**Rivals.** Rectangles scale trivially: per-cell widths with a union over S·A cells. That is the comparison that should get stronger as S grows, because m_eff grows (Proposition 1).

**Go criteria** (written before dev runs, in `plan/fdc_dp_theory.md`):
1. A proof of validity.
2. Exact agreement with enumeration on small S.
3. Under 10 s per checkpoint at S = 64.
4. Dev UB95 < 0.80 against the best valid rectangle at S ≥ 32 on at least one table.

If go: external reviewer review, then lock v10 (fresh seeds), then a confirmatory block.

## Workstream 2: fair adaptive comparison (external reviewer r4 fix 2)

Rerun posthoc-F with the pilot's 1,000 draws per cell charged to N80, and give the adaptive PJC variants the same pilot information. Report amortisation sensitivity. This block is descriptive.

## Workstream 3: paper r5 (after workstreams 1–2)

Apply the following:
- **external reviewer r4 fix 1 (theory and scope corrections):**
  - TU sign;
  - strict tail;
  - filtration;
  - continuous crossings;
  - no "adaptive designs give up TU";
  - no "ledger not slack".
- **external reviewer r4 fix 4 (alignment):**
  - +box parent;
  - within-v9 monitoring comparison;
  - "paired geometric-mean N80_pen";
  - scope on standalone headlines.
- **external reviewer r4 fix 5:** stale "every number unchanged", plus the API guard.
- **Critic r4 P0s:**
  - novelty de-overclaim (cite Hoeffding 1963 convex order);
  - replace C3;
  - density.

Then run critic and external reviewer, two rounds each.

## Workstream 4: anonymous artifact

This is blocked on a user account; there is no remote and no gh. The user has been told.

## ε-selection rule for lock v10 (written 2026-10-05 before the full dev runs at ε ≥ 0.03)

- **Why a rule is needed.** In the smoke test, the X5 rectangles at ε 0.02 never reached 12/15 certificates before τ. A censored rival gives an uninformative ratio.
- **The rule.** Lock v10's primary ε per (table, S) is the **smallest ε on the grid** {X5: 0.02, 0.03, 0.04, 0.05; Lenta: 0.003, 0.004, 0.006, 0.008} at which the best valid rectangle (RECT-BF-DP or HC-WoR-DP) reaches N80 < τ on ≥ 80% of dev streams 950–999.
- **Fallback.** If no grid value qualifies, that (table, S) is reported as "rivals censored" (descriptive) and is not a confirmatory cell.
- **The rule does not look at FDC-DP's speed.**
