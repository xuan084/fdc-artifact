# Open Bandit Dataset: blinded feasibility for a confirmatory block (dev only)

**Verdict: GO-with-changes.** The OBD uniform-random "All" log supports a confirmatory FDC-DP vs RECT-BF-DP block. Use a 9-segment user-feature segmentation and two item-group arms with traffic-share costs. The block must take its ε from the frozen rival-success rule (expected ε = 5e-4 click-rate units). Three changes from the v10 template are required: a general-A env (cell = s·A + a), item-group arms frozen from dev, and a finer checkpoint grid or a ratio endpoint that is not quantised to a few grid steps. Multi-arm (A = 3) works in code but exhausts the control pool, so it should be descriptive only.

## 1. Blinding record

- Only `random/all/all.csv`, the README and `item_context.csv` were read. The BTS logs and the `random/men` and `random/women` logs stayed unread, so both campaigns remain available for a later independent replication.
- The split was made before any click was looked at. Salt `dsswm-obd-2026-10-05`, dev_frac 0.5, row_key = the CSV's unnamed index column (verified to equal 0..1,374,326). Files are `dev.pkl`, `eval_labels.pkl` and `eval_outcome.npy` (chmod 444), with their sha256 values in `PROVENANCE.json`.
- dev has 686,168 rows and eval has 688,159. No eval-outcome statistic was computed. All design and pilot work below uses dev only.
- The split is at row level. A few multi-position impressions share a timestamp across the two halves (3.3% of dev rows). User ids are absent, so the same user can appear in both halves. This makes dev and eval weakly dependent, which is acceptable because eval is used only once, but it must be disclosed.

## 2. Structure (dev)

- **Rows.** Each row is one (impression, position) recommendation. The item is uniform over 80, the propensity is 0.0125 on every row, and positions 1–3 are balanced.
- **Clicks.** The click rate is 0.350% (2,405 clicks). By position it is 0.36%, 0.34% and 0.34%, so positions are pooled.
- **Per-item rates.** Item click rates run from 0.07% to 0.82%, with 6–67 clicks per item and about 8.6k rows per item.
- **Item heterogeneity is real.** The across-item SD is 0.00176 against a Poisson-noise SD of about 0.00064.
- **User features.** There are four hashed categoricals, `user_feature_0..3`, with 4, 6, 10 and 10 levels; the README says "0-4", but the file has four columns.
  - One joint pattern covers 35.4% of rows; it looks like an "unknown user" pattern.
  - The user–item affinity columns are 99.96% zero, so they are useless for segmentation.
- **Item features.** `item_feature_0` is the only numeric item feature (41 standardised values). Saito et al. list "price, fashion brand, item categories" as item features, so this column is plausibly price, but the column-to-meaning map is not documented. It should not be the headline cost.
- **Mapping to the paper.** A segment is a frozen user-feature cell (covariate-only value lists). Arm a means "show a uniformly random item from group a", so each cell mean is a finite-population CTR of a randomised group policy.
- **Costs.** The only documented heterogeneous cost is exposure, i.e. segment traffic. The template is κ[s,1] = round(8·S·w_s), as in seg_v10. Segment sizes are very unequal (w from 0.03 to 0.37), so this is a genuine knapsack rather than a cardinality constraint. The budgets are 0.10–0.80 of total exposure, giving 15 problems.

## 3. Candidate designs and power (dev, whole dev half as replay pool, τ = 686,168, K = 20, δ = 0.05)

Arms for A2: the 40 items with the highest dev CTR are "treatment" and the other 40 are "control". The arm CTRs are 0.494% and 0.207%.

ε-rule: the v10 rival-success rule, i.e. the smallest ε where RECT-BF-DP-TU has N80_pen < τ on at least 80% of streams. It was applied to 30 streams with seeds 950–979.

| design | S | min cell clicks | J*−J(all-control) over q | ε by rule | RECT success | FDC success | geo-mean N80 ratio FDC/RECT | FDC wins | share of feasible policies ε-optimal |
|---|---|---|---|---|---|---|---|---|---|
| **uf0×uf3, A2** | 9 | 10 | 4.1e-4 … 2.6e-3 | **5e-4** | 0.90 | 1.00 | **0.72** (0.55 at 7.5e-4) | 30/30 | 0.30 |
| combo8 (top-7 uf tuples + other), A2 | 8 | 8 | 4.3e-4 … 2.1e-3 | 7.5e-4 | 0.97 | 1.00 | 0.55 | 30/30 | 0.69 |
| combo16, A2 | 16 | 2 | 5.7e-4 … 2.2e-3 | 1e-3 | 1.00 | 1.00 | 0.59 | 30/30 | 0.83 (near-trivial) |
| uf0×uf3, A3 (CTR tertiles, κ = a·size) | 9 | 5 | 3.3e-4 … 2.3e-3 | 1e-3 | 1.00 | 1.00 | 0.77 | 30/30 | 0.71 |

Facts that hold across every cell in the table:
- There were no false certificates (FWER events = 0 everywhere) and no node-limit hits.
- FDC-DP(a) equals TU-FDC-DP(b) everywhere.
- A whole cell runs in seconds.

In uf0×uf3 A2 the dev uplifts per segment run from +0.0022 to +0.0048, and three segments (w ≈ 0.11 in total) are negative: −0.0002, −0.0019 and −0.0007. This mix of positive and negative uplifts is what makes the frontier non-trivial.

**Smallest useful ε.**
- The frontier is non-trivial when all-control is ε-optimal in at most 2/15 problems and at most 30% of policies are ε-optimal. That holds for ε ≤ 5e-4.
- At ε = 1e-3, 60% of policies are ε-optimal and 3/15 problems are trivial.
- At ε = 3e-4 RECT never certifies, while FDC succeeds 30/30 (N80 ≈ 530k).
- ε = 5e-4 is about 14% of the base CTR, or 0.05 percentage points.

**Shrinkage / winner's-curse check.**
- The arm groups and segment lists were rebuilt on the even-row_id half of dev and replayed on the odd half (τ = 343k, half the eval size).
- The uplift pattern was stable, including the same three negative segments.
- The geo-mean ratio FDC/RECT was 0.78 at ε = 5e-4 (RECT 0/30 there), 0.78 at 7.5e-4 (RECT 29/30), 0.64 at 1e-3 and 0.53 at 1.5e-3.
- FWER was 0. The eval pool is twice this size, so the effect should sit between the holdout numbers and the dev numbers.

## 4. Recommended confirmatory design (eval half, single use)

- **Segments.** Use the 9 uf0×uf3 cells, with cells under 2% merged into their uf0 "other" cell. Freeze them as explicit hashed-value lists from dev.
- **Arms.** Use A = 2 item groups (top-40 vs bottom-40 by dev item CTR) and freeze the item list. Disclose that the grouping is dev-outcome-informed and cite the holdout check.
- **Costs and budgets.** Exposure cost κ[s,1] = round(8·9·w_s) uses the eval segment shares, which are label-only and legal. Run the 15 budgets 0.10–0.80.
- **ε.** Take ε from the frozen rival-success rule on a declared grid {3e-4, 5e-4, 7.5e-4, 1e-3, 1.5e-3}, evaluated on dev with 50 streams. Expected ε is 5e-4; use 7.5e-4 if RECT drops below 40/50.
- **Rivals.** RECT-BF-DP-TU (primary) and HC-WoR-DP (untuned transfer; not run here and needed for the rule). FDC-DP(a) and plug-in methods are descriptive only.
- **Endpoint.** N80_pen at 12/15 with fresh eval seeds. The K = 20 log grid quantises ratios to steps of about 0.77/0.60/0.46. Pre-register K = 40, or report the time-uniform evaluation on a dense grid, so that the effect size is not a grid artefact.
- **Secondary cell.** uf0×uf3 with A = 3 is descriptive only, as a multi-arm demonstration. The code handles A > 2 (`balanced_alloc`, general cells). With a 0.5 control share and 1/3 pool shares the control pool exhausts (exhaustion at k80 is up to 0.6 at ε = 5e-4), so it is not a clean confirmatory cell.
- **Replication option.** `random/men` (453k rows) and `random/women` (865k rows, CTR about 0.5%) are separate, unread campaigns. Split them with the same protocol before any use; they could provide a second confirmatory replication.

## 5. Risks

- **Low rates.** Cells hold 10–700 clicks, and the eval half has about 2.4k clicks in total. The certifiable ε is absolute (about 5e-4 to 1e-3), which is large relative to the 0.35% base. The paper must report ε in CTR units and relative to the base.
- **Near-ties.** At ε ≥ 1e-3 most policies are ε-optimal, which weakens a "frontier" claim. Keep ε ≤ 7.5e-4 or report the share of ε-optimal policies with each result.
- **Selection.** Both the arm grouping and the choice of uf0×uf3 (picked for visible heterogeneity) were made on dev outcomes. They are legitimate only because eval is untouched, and this must be stated.
- **Dependence.** Row-level independence is assumed. Impressions with several positions and repeat users are not modelled. The finite-population estimand (the log itself) limits the damage.
- **Exhaustion.** None for A = 2 at the recommended ε. A = 3 does exhaust (see §4).
- **Code.** `seg_v10.SegEnvV10` hard-codes A = 2. The new `envs/obd_v11.py` is general and dev-only, but it needs an eval twin plus sealing and lock plumbing analogous to `seg_v10_eval`.

Artifacts (new files only): `exp/code/dsswm/envs/obd_v11.py`, `exp/code/run_obd_dev_pilot.py`, `exp/results/pilots/obd_dev/{describe.json, describe_holdout.json, streams_*.jsonl}`. Total pilot CPU was a few minutes.

---
**Erratum (2026-10-05, external reviewer v11 lock review r1 F4; append-only).** §3's shrinkage check says the arm groups and the segment lists were rebuilt on the even-row_id half of dev. In fact only the item groups were built there: `obd_v11.OBDEnv(holdout=True)` recomputes the uf0×uf3 segmentation, and hence the costs, on the odd replay half. The 0.78 vs 0.72 comparison is therefore an item-group holdout, not a rehearsal of the full frozen-design transfer. The confirmatory v11 design freezes segments, items and costs from the whole dev half.
