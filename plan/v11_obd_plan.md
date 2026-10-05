# v11 plan: confirmatory FDC-DP block on the Open Bandit Dataset (never-replayed real randomized log)

Written 2026-10-05 by the experimenter/methodologist, **before any v11 dev stream** (the only earlier OBD streams are
the feasibility pilot, `plan/obd_feasibility.md`, K = 20, seeds 950–979). Nothing below may be changed after the first
v11 dev stream except by an explicit, dated deviation note appended at the end of this file. The eval half
(`eval_outcome.npy`) is not read by anything until the v11 addendum is locked, and then only by the gated eval runner.

## 1. Claim under test

On a real uniform-random recommendation log that no earlier lock has replayed, TU-FDC-DP(b) certifies 12 of 15
exposure-budgeted segment-targeting problems after reading fewer rows than the matched, parameter-free time-uniform
rectangle RECT-BF-DP-TU, at a pre-registered ε, without any false certificate.

## 2. Data and blinding

- **Table.** Open Bandit Dataset (ZOZOTOWN), `random/all/all.csv`, uniform-random logging policy (propensity 1/80 per
  position). Blinded salted row-level split (salt `dsswm-obd-2026-10-05`, dev_frac 0.5, `PROVENANCE.json`). Dev:
  686,168 rows; eval: 688,159 rows. The eval half is the replay population of the confirmatory block (τ = eval N).
- **Status.** First use of the eval half. No eval-outcome statistic has been computed. `random/men`, `random/women`
  and `bts/*` stay unread (reserved for a possible later replication).
- **Data binding.** `dev.pkl`, `eval_labels.pkl`, `eval_outcome.npy` and `PROVENANCE.json` are bound by sha256 in the
  addendum; the eval reader re-checks them against PROVENANCE before reading.

## 3. Design (feasibility's recommended design, frozen from dev)

- **Segments (S = 9).** user_feature_0 × user_feature_3 cells; cells with < 2 % of dev rows are merged into their
  user_feature_0 "other" cell; any residual cell with < 1 % is merged into the largest cell
  (`obd_v11.segmentation(df, 'uf0x3')`). Frozen as an explicit map (uf0 value, uf3 value) → segment from dev.
  Eval rows whose (uf0, uf3) pair was not seen in dev go to their uf0 "other" segment if it exists, else to the
  largest dev segment (the same merge rule); the number of such rows is reported.
- **Arms (A = 2).** Arm 1 ("treatment") = the 40 items with the highest dev click rate (ties by item id), arm 0
  ("control") = the other 40. Frozen as an explicit item list. Arm a means "show a uniformly random item of group a".
  Each cell mean is the finite-population CTR of that group policy in that segment.
- **Costs and budgets.** Exposure (traffic-share) cost κ[s, 0] = 0, κ[s, 1] = max(1, round(8 · S · w_s)) with w_s the
  **dev** segment shares; budgets B_q = floor(b_q · Σ_s κ[s, 1]) for b_q = 0.10, 0.15, …, 0.80 (15 problems). Frozen
  from dev, so the policy classes are identical on dev and eval (as v10). Policy values use the eval segment weights.
  (Feasibility §4 suggested eval label shares for κ; freezing from dev is stricter and avoids any eval read before
  the lock. Disclosed.)
- **Design.** Frozen outcome-free 50/50 schedule (`balanced_alloc(S, A, 0.5)`), δ = 0.05, n_min = 5,000.
- **Checkpoint grid K = 40** (`fr.checkpoints(5000, τ, 40)`), **changed from K = 20** used in v7–v10 and in the
  feasibility pilot: with K = 20 the log grid quantises paired ratios to steps of about 0.77 / 0.60 / 0.46, so the
  effect size would be a grid artefact. K = 40 halves the log step. Both the ledger of FDC-DP and the 2 S A K
  rectangle events pay for the larger K. Disclosed as a design change.
- **Horizon.** Every stream runs to 15/15 certified or τ.

## 4. Methods

| role | name | construction |
|---|---|---|
| primary | TU-FDC-DP(b) | `FDCDPTimeUniform(block_points = K = 40 grid, scheme='b')`, defaults of v10 (node limit 200,000, grid ratio 2000^(1/160), split 0.045 / 0.005) |
| governing rival | RECT-BF-DP-TU | `RectBFDPTU(block_points = K = 40 grid, box=True)`, parameter-free |
| descriptive rival | HC-WoR-DP[tuned] | `HCWoRDP('nstar', c, tf)` tuned on dev by the block-G3 rule (below) |
| descriptive rival | RECT-HG-DP (CP) | `RectHGDP()`: exact hypergeometric cells at δ/(2 S A K), checkpoint-strength guarantee (valid at the K grid points only), labelled **CP** |

HC tuning rule (block G3 of `run_v10_posthoc_G.py`, applied per ε): over the declared grid HC_GRID = {c ∈ {0.5,
0.75}} × {tf ∈ {0.2, 0.4, 0.6, 0.8}} (schedule 'nstar', grid order as in run_v10_posthoc_G), at each ε of the ε grid,
the configuration with the lowest dev geometric-mean N80_pen over dev seeds 950–999; ties → grid order. The tuned
configuration of the selected ε is frozen before any eval row. HC is tuned on the same dev streams it is evaluated on,
so dev numbers are selection-optimistic for HC (conservative for the FDC-DP dev ratio).

## 5. ε selection (rival-success rule, v10 rule; never reads FDC-DP)

- Grid {3e-4, 5e-4, 7.5e-4, 1e-3, 1.5e-3} (click-rate units). 50 dev streams, seeds 950–999, dev half as replay
  population, K = 40.
- At each ε, the "dev-best rectangle" is the one of {RECT-BF-DP-TU, HC-WoR-DP[tuned at that ε]} with the lower dev
  geometric-mean N80_pen (tie → RECT-BF-DP-TU). The selected ε is the **smallest** grid ε at which the dev-best
  rectangle has strict N80_pen < τ on ≥ 40/50 dev streams.
- If no grid ε qualifies, there is no confirmatory cell; the block is descriptive only and no positive claim is made.
- The dev ε cell is the cell (ε) selected by the rule. FDC-DP rows are produced on dev only at the selected ε (for
  THRESH and the dev report) and never enter the rule; the rule is implemented in code that does not receive FDC-DP
  rows.
- Reporting: ε in absolute CTR units, relative to the dev base CTR (0.350 %), and the share of feasible policies that
  are ε-optimal (dev truth now; eval truth descriptively after the eval run).

## 6. Endpoint, statistic and decision rule

- **Endpoint.** N80_pen = the K = 40 checkpoint at which 12/15 problems are first certified if there is no false
  certificate by then, else τ (v9/v10 rule). A **false stream** is any false certificate over the whole run to 15/15.
- **Statistic.** Paired geometric-mean ratio N80_pen(TU-FDC-DP(b)) / N80_pen(RECT-BF-DP-TU) over 200 fresh eval
  streams (seeds **39000–39199**, never used), paired percentile bootstrap, B = 10^4, seed 42, one resample matrix
  for all comparators; one-sided UB95; two-sided CI reported.
- **THRESH (rule fixed now, value computed after the dev run, recorded in the addendum before the lock).**
  THRESH = min(0.90, UB95_dev + 0.10), where UB95_dev is the one-sided UB95 of the same statistic (B = 10^4, seed 42)
  on the 50 dev streams of the dev ε cell. Rounded up to 3 decimals.
- **Decision.** `positive_result_achieved` iff (1) UB95 < THRESH, (2) TU-FDC-DP(b) has 0/200 false streams over the
  whole run to 15/15, and (3) the replica check passes. Otherwise `positive_result_not_achieved` with the failing
  components (rival_superiority_failed with sub-label faster_below_1 / not_faster; validity_failure; replica_fail).
- **Descriptive.** Ratios against HC-WoR-DP[tuned] and RECT-HG-DP (CP), RECT-BF-DP-TU / HC; per-method false streams
  with Clopper–Pearson UB; exhaustion fractions at N80; share of streams reaching 15/15; ε-optimal policy share.
- **Replica.** R1 (independent re-run of the first 10 eval seeds, canonical rows identical), R2 (schedule digest
  identical across all four methods per seed), R2b (arrival digest identical) — v10 rules.

## 7. Gating and sealing (mirrors v10)

- A new gated reader `dsswm/envs/obd_v11_eval.py`: eval rows are loaded only after `prereg_v11.addendum_gate` passes
  for a registered task whose registered layer matches, the caller supplies the locked addendum's sha256, and an
  access-log line has been appended (`exp/results/full/v11_obd/eval_access_log.jsonl`). The dev env never touches
  eval files.
- `prereg_v11` chains to the live v10 addendum (and through it v9–v5), binds code, inputs (plan, dev report, frozen
  gates) and data by sha256, requires single-commit lock history, and refuses the DRAFT.
- Eval tasks are sealed at completion (v10 seal mechanism, v11 tag); a sealed task is never re-run.

## 8. Disclosures (to be carried into the addendum and the paper)

1. The arm grouping (top-40 vs bottom-40 items by dev CTR) and the choice of the uf0 × uf3 segmentation (picked for
   visible heterogeneity) used dev outcomes. Eval is untouched, which makes them legitimate, but the eval uplift gap
   will shrink by selection (feasibility holdout check: ratio 0.78 vs 0.72 at ε = 5e-4).
2. The split is row-level: 3.3 % of dev rows share an exact timestamp with an eval row (multi-position impressions),
   and user ids are absent, so repeat users can appear in both halves. Dev and eval are weakly dependent.
3. First use of the eval half; inference is conditional on this finite table (fresh replay randomness only).
4. K changed from 20 to 40 (reason in §3). Costs frozen from dev shares instead of eval label shares.
5. HC-WoR-DP is tuned on the dev streams that are also used for the ε rule (selection-optimistic for HC).
6. RECT-HG-DP has a checkpoint-strength (grid-only) guarantee, not a time-uniform one, and is descriptive.
7. FDC-DP's statistical certificate is FDC-BF's (Theorem DP-1); the contribution is computational and the block tests
   speed against named rectangle implementations, not "the best valid rectangle".
8. The headline claim is restricted to methods with proven finite-sample / anytime-valid guarantees (user ruling
   2026-10-03); plug-in methods are not run in this block.

## 9. Work order

1. This plan (committed alone). 2. Code + tests (new files only; no locked file modified). 3. Dev block: HC tuning and
ε rule (seeds 950–999, 5 ε × {RECT-BF-DP-TU, 8 HC configs}), then the dev ε cell with all four methods; THRESH;
dev report; freeze gates with sha256. 4. Draft addendum by builder script. 5. external reviewer lock review (rounds until lock).
6. Lock (main session), then eval tasks, replica, analysis.

---
**Dated note (2026-10-05, after external reviewer v11 lock review r1; append-only, design unchanged).**
- Disclosure 1's "holdout check" froze only the item groups on half of dev; segments were recomputed on the replay half (feasibility erratum, F4).
- §7's eval gate was tightened (F1). The gate never opens an eval file; eval-file hashes come from the hash-bound PROVENANCE.json; eval bytes are hashed only inside the reader after the access-log line.
- The dev block was regenerated on the fixed code (no stream computation changed).
