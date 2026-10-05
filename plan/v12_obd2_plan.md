# v12 plan: second locked replication of lock v11 on two unread Open Bandit campaigns (random/women, random/men)

File name: `plan/v12_obd2_plan.md`. First committed as `plan/v12_obd_plan.md` (317619b4); renamed and extended with the
campaign-selection rationale (§1.1) and the timetable (§11) on user instruction, 2026-10-05, in a separate commit,
still before any split, any dev click read or any dev stream. No rule was changed by the rename.

Written 2026-10-05 by the experimenter/methodologist, **before any row of `random/women` or `random/men` was split or
any of their clicks was read** (only the zip directory listing, the README, the two CSV header lines and the item
counts of the two `item_context.csv` files were looked at: women 46 items, men 34 items). User decision of 2026-10-05
17:40 (the project notes, W2). Nothing below may change after the first v12 dev stream except by an explicit,
dated deviation note appended at the end of this file. No eval half is read by anything until the v12 addendum is
locked, and then only by the gated eval reader.

## 1. Claim under test

The lock-v11 result replicates on two further uniform-random campaigns of the same platform that no earlier step has
read: under the **identical written protocol** of lock v11 (design rules re-derived on each campaign's own dev half),
TU-FDC-DP(b) reaches 12 of 15 certified exposure-budgeted segment-targeting problems after reading fewer rows than the
matched time-uniform rectangle RECT-BF-DP-TU, at the rule-selected ε, without any false certificate.

- **Block A = `random/women`** (primary block of v12).
- **Block B = `random/men`** (its own verdict).
- **No pooling** across blocks, and no pooled or meta-analytic claim. Each block is reported with its own verdict
  whatever the outcome. Block B cannot rescue a failed block A, and vice versa.

### 1.1 Campaign-selection rationale

- **Block A = women (primary).** 865k rows (~432k per half, about 63 % of the v11 random/all half), 46 items. With
  roughly twice the rows of men it has about twice men's power, so the frozen rules are more likely to give a
  non-trivial tolerance there; it is therefore the primary block.
- **Block B = men.** 453k rows (~226k per half), 34 items.
- Both are separate, unread uniform-random campaigns of the same platform (separate item sets). Doing both turns the
  replication from "one further log" into "two independent campaigns" at low compute cost (a dev block is CPU minutes).

## 2. Data and blinding

- **Source.** `open_bandit_dataset/random/{women,men}/{women,men}.csv` inside
  `$DATA_DIR/open_bandit/raw/open_bandit_dataset.zip` (sha256 in the top-level
  `PROVENANCE.json`), read with Python `zipfile` (no unzip binary). Uniform-random logging policy.
- **Split (before any click is read; one script, one run per campaign).**
  - Salts `dsswm-obd-women-2026-10-05` and `dsswm-obd-men-2026-10-05`; dev_frac 0.5.
  - Rule as in the v11 PROVENANCE: dev iff `int(sha256(f'{salt}|{name}|{row_key}')[:8], 16) / 2**32 < 0.5`, with
    `name = 'open_bandit_women'` / `'open_bandit_men'` and `row_key` = the CSV's unnamed first column (checked to equal
    the 0-based row index; if it does not, the script stops and the rule is not silently changed).
  - Output per campaign in `$DATA_DIR/open_bandit/{women,men}/`: `dev.pkl` (all
    columns, `row_id` renamed from the unnamed index), `eval_labels.pkl` (every column except `click`),
    `eval_outcome.npy` (int8 click aligned with the label rows), all three `chmod 444`, and a `PROVENANCE.json` with
    the three sha256s, the rule, the salt, `split_created_utc` (written **once, at split creation, never rewritten**;
    a v11 nit) and **dev-only** statistics. The only eval quantities recorded are the eval row count and the
    non-outcome overlap counts of §2.1. No eval-outcome statistic is computed by anything before the lock.
  - The existing `random/all` files (`dev.pkl`, `eval_labels.pkl`, `eval_outcome.npy`, the top-level
    `PROVENANCE.json`) are **not modified**. The top-level `PROVENANCE.json` is hash-bound by lock v11
    (`data_sha256.obd_provenance`), so appending a pointer to it would break the v11 chain that v12 inherits; the
    pointer goes into a new file `open_bandit/V12_SPLITS.json` instead.
- **2.1 Overlap disclosure (non-outcome fields only).** The split script reports, for each campaign:
  - the timestamp range of the campaign and of the `random/all` eval half;
  - the number and share of campaign rows whose exact `timestamp` also occurs in the `random/all` eval half (read from
    the `timestamp` column of `random/all/eval_labels.pkl`, never its outcome file), and in the other new campaign;
  - the within-campaign dev/eval exact-timestamp sharing (as v11's 3.3 % note).
  User ids are absent, so user overlap cannot be measured; this is disclosed. The campaigns are separate campaigns
  with separate item sets (80 / 46 / 34 items), so a shared timestamp means a concurrent impression, not a duplicated
  row.

## 3. Design (v11 written rules, re-derived on each campaign's dev half)

Identical to `plan/v11_obd_plan.md` §3 except where a campaign constant must differ:

- **Segments.** user_feature_0 × user_feature_3 cells; cells with < 2 % of dev rows merged into their user_feature_0
  "other" cell; any residual cell with < 1 % merged into the largest cell (`obd_v11.segmentation(df, 'uf0x3')` and
  `obd_v11_eval._dev_rule`, imported unchanged). Value lists come from covariates only. Frozen as an explicit map
  from dev; eval rows with an unseen pair go to their uf0 "other" segment if it exists, else to the largest dev segment;
  fallback counts are reported. S is whatever the rule gives on the campaign (not forced to 9).
- **Arms (A = 2).** Arm 1 = the top half of the campaign's items by dev click rate (ties by item id;
  `obd_v11.item_groups(df, 2)`), arm 0 = the rest: women 46 items → 23 / 23, men 34 items → 17 / 17. Frozen as explicit
  item lists. An eval row with an item outside the lists stops the run (no silent remap).
- **Costs and budgets.** κ[s, 0] = 0, κ[s, 1] = max(1, round(8 · S · w_s)) with the **dev** segment shares;
  B_q = floor(b_q · Σ_s κ[s, 1]) for b_q = 0.10, 0.15, …, 0.80 (15 problems). Policy values use the replay half's own
  segment weights.
- **Design.** Frozen outcome-free 50/50 schedule `balanced_alloc(S, A, 0.5)`, δ = 0.05, n_min = 5,000.
- **Checkpoint grid K = 40** (`fr.checkpoints(5000, τ, 40)`), τ = the replay half's row count. Every stream runs to
  15/15 certified or τ.

## 4. Methods (v11, unchanged)

| role | name | construction |
|---|---|---|
| primary | TU-FDC-DP(b) | `FDCDPTimeUniform(K = 40 grid, scheme='b')`, v10/v11 defaults (node limit 200,000, grid ratio 2000^(1/160)) |
| governing rival | RECT-BF-DP-TU | `RectBFDPTU(K = 40 grid, box=True)`, parameter-free |
| descriptive | HC-WoR-DP[tuned] | `HCWoRDP('nstar', c, tf)`, tuned on the campaign's dev streams by the block-G3 rule over the 8-config grid {0.5, 0.75} × {0.2, 0.4, 0.6, 0.8} |
| descriptive | RECT-HG-DP (CP) | exact hypergeometric cells at δ/(2 S A K); **checkpoint-strength (CP)** guarantee; labelled CP, never "rigorous" or "time-uniform" in any v12 output |

## 5. ε selection (v11 rival-success rule; never reads FDC-DP), per campaign

- Grid {3e-4, 5e-4, 7.5e-4, 1e-3, 1.5e-3} (absolute click-rate units). 50 dev streams, seeds 950–999, the whole
  campaign dev half as replay population, K = 40.
- At each ε, the dev-best rectangle is the one of {RECT-BF-DP-TU, HC-WoR-DP[tuned at that ε]} with the lower dev
  geometric-mean N80_pen (tie → RECT-BF-DP-TU). The selected ε is the **smallest** grid ε at which the dev-best
  rectangle has strict N80_pen < τ on ≥ 40/50 dev streams. Implemented by `v11_analysis.select_hc / select_eps`
  (imported; `select_eps` refuses FDC-DP rows).
- If no grid ε qualifies, the block's ε is the **largest** grid ε (1.5e-3) and the block is descriptive (gate (i)).

## 6. Block-status gate (pre-stated here, before any dev click is read)

On each campaign's dev half, at the rule-selected ε, and computed from the dev truth (no FDC-DP row is read), a
campaign's block is **confirmatory** iff all three hold:

1. the rule selects some ε on the grid;
2. all-control is **not** ε-optimal in at least one of the 15 budget problems (i.e. the number of problems in which the
   all-control policy is ε-optimal is ≤ 14);
3. the mean over the 15 problems of the share of feasible policies that are ε-optimal is ≤ 0.5
   (`obd_v11_eval.eps_opt_share`, imported).

Otherwise the block is **descriptive**: still registered in the lock, run once on its eval half, sealed, replicated
and analysed with the same statistic, THRESH and decision function, but its output is labelled descriptive and supports
no confirmatory claim. The status is written to `exp/results/v12_gates/v12_{campaign}_block_status.json` by the same
step that freezes ε (before any FDC-DP dev row exists) and is bound by sha256 in the lock. Block A remains the primary
block whatever its status; if block A is descriptive and block B confirmatory, block B's verdict is the only
confirmatory v12 verdict and the paper says so.

## 7. Endpoint, statistic and decision rule (v11, per confirmatory block)

- **Endpoint.** N80_pen = the K = 40 checkpoint at which 12/15 problems are first certified if no false certificate by
  then, else τ. A **false stream** is any false certificate over the whole run to 15/15 or τ.
- **Statistic.** Paired geometric-mean ratio N80_pen(TU-FDC-DP(b)) / N80_pen(RECT-BF-DP-TU) over 200 fresh eval
  streams, paired percentile bootstrap, B = 10^4, seed 42, one resample matrix for all comparators of a block;
  one-sided UB95; two-sided CI reported.
  - **Eval seeds:** block A (women) **39200–39399**; block B (men) **39400–39599**. Never used before (v11 used
    39000–39199; earlier locks ≤ 38999).
- **THRESH (per campaign).** THRESH = min(0.90, UB95_dev + 0.10), rounded up to 3 decimals, UB95_dev = the one-sided
  UB95 of the same statistic on the campaign's 50 dev streams at its selected ε (B = 10^4, seed 42). Frozen before the
  lock.
- **Decision (per block).** `positive_result_achieved` iff (1) UB95 < THRESH, (2) TU-FDC-DP(b) has 0/200 false streams
  over the whole run, and (3) the block's replica check passes; otherwise `positive_result_not_achieved` with failing
  components (rival_superiority_failed with sub-label faster_below_1 / not_faster; validity_failure; replica_fail).
  `v11_analysis.decide` is imported unchanged.
- **Descriptive (pre-stated, every block).**
  - ratios vs HC-WoR-DP[tuned] and vs RECT-HG-DP (CP), and RECT-BF-DP-TU vs each;
  - per-method false streams with Clopper–Pearson UB; exhaustion at N80; share reaching 15/15;
  - **full-frontier ratio**: paired geometric-mean ratio of N_stop_pen (rows to 15/15, τ if never or if a false
    certificate), same bootstrap; horizon-sensitive, never the endpoint;
  - **uncensored-subset ratio**: the primary ratio restricted to streams on which neither TU-FDC-DP(b) nor
    RECT-BF-DP-TU had an exhausted pool at N80 (`exhaustion_at_k80.all == 0` for both), bootstrap B = 10^4 seed 42 on
    that subset; the subset size is reported; descriptive only (in v11 it was post hoc; here it is pre-stated);
  - eval base CTR, ε relative to it, the eval ε-optimal share and all-control count (eval truth, after the run);
  - fallback row counts.
- **Replica (per block).** R1 (independent re-run of the first 10 eval seeds; canonical rows identical except timing
  fields), R2 (schedule digest identical across all four methods per seed), R2b (arrival digest identical);
  `v11_replica` rules imported unchanged.

## 8. Gating and sealing (mirrors v11, with the v11 r2 P2 nits fixed in the new code)

- New gated reader `dsswm/envs/obd_v12_eval.py`, campaign-parameterised. Order: caller arguments → v12 gate (locked
  addendum, canonical hash, inherited v11 → v5 chain, code / input / non-eval data drift, task registered,
  single-commit history; no eval byte touched) → task registered for exactly this campaign's layer → caller's lock
  sha256 equals the lock → one access-log line appended and fsync'ed → eval bytes hashed and checked against the lock
  and the campaign PROVENANCE → deserialisation. The byte-verification helper is **private** (`_verify_eval_bytes`, not
  in `__all__`), and the module docstring states this order (v11 r2 P2-1, P2-2).
- Eval-file hashes used before the lock come from the hash-bound campaign `PROVENANCE.json` (no eval file opened).
- `prereg_v12` chains to the live v11 addendum (and through it v10–v5), binds code, inputs (this plan, the dev report,
  frozen gates, dev rows) and the eight campaign data files by sha256, requires single-commit lock history, refuses the
  DRAFT.
- Eval tasks `v12_women_full` and `v12_men_full`, each sealed alone at completion (v11 seal mechanism, v12 tag), never
  re-run. Access logs per block under `exp/results/full/v12_obd/`.
- New files only (`*v12*`); no locked or lock-hashed file (including all v11 code) is modified; v11 modules are reused
  by import.

## 9. Disclosures (to be carried into the addendum and the paper)

1. Arm grouping (top half of items by dev CTR) and the uf0 × uf3 rule used dev outcomes / were chosen on random/all
   dev; here they are applied mechanically, but each campaign's eval uplift gap will shrink by selection.
2. Row-level split; multi-position impressions can straddle halves; user ids are absent, so repeat users can occur in
   both halves and across campaigns (§2.1 counts).
3. First use of both eval halves; inference conditional on each finite table.
4. HC-WoR-DP is tuned on the same dev streams used by the ε rule (selection-optimistic for HC; descriptive).
5. RECT-HG-DP has a checkpoint-strength (CP) guarantee only.
6. The block-status gate is new in v12 (pre-stated here); it can only demote a block to descriptive.
7. Same dev seeds 950–999 as v11 (different tables; dev only).
8. Claims are restricted to the named rectangle constructions and to methods with proven finite-sample /
   anytime-valid guarantees (user ruling 2026-10-03); plug-in methods are not run.
9. The top-level OBD PROVENANCE is not amended (v11 hash binding); the pointer lives in `V12_SPLITS.json`.

## 10. Work order

1. This plan, committed alone. 2. Splits (one script, both campaigns; overlap counts). 3. Code + tests (`*v12*`).
4. Dev blocks per campaign: freeze design → 5 rule tasks (RECT-BF-DP-TU + 8 HC configs) → select ε + HC + block status
   → dev cell (all four methods) → dev replica → dev analysis → THRESH → MANIFEST; `plan/v12_dev_report.md`.
5. Builder → `plan/prereg_lock_v12_addendum_DRAFT.{md,json}`. 6. external reviewer lock review, at most two rounds. 7. Lock and
eval are **not** part of this step.

## 11. Timetable

- **2026-10-05 evening:** this plan; splits; dev blocks (both campaigns).
- **2026-10-06:** gated code and tests; draft addendum; external reviewer lock review (≤ 2 rounds).
- **2026-10-07:** lock → a single eval read per campaign → seal → replica → analysis.
- **2026-10-07/08:** paper r8.
- **2026-10-08/09:** critic and external reviewer reviews.
- **2026-10-10 .. 10-14:** buffer (decision point 10-15).

---
**Dated note (2026-10-05, after external reviewer v12 lock review r1; append-only, no rule changed).**
- **Descriptive-block output.** After the dev gate made both blocks descriptive, the coordinator instructed that a
  descriptive block issue no verdict. Commit 39dace28 implements this. §6's "same statistic, THRESH and decision
  function" now reads as follows.
  - The statistic and THRESH rule are unchanged.
  - The v11 decision function is called only to validate inputs. Its THRESH comparison and verdict are discarded.
  - The record carries `verdict = 'descriptive'`, THRESH is marked not applicable, and a replica failure is reported
    as `replica_status = 'fail'`.
  - The dev block was regenerated on that code, and its 4,980 rows are identical to run 1 (d23b7aaa).
- **Smoke-test exception.** §6 says the status file is written "before any FDC-DP dev row exists". An unsaved pre-rule
  smoke test (seed 950, ε 5e-4, all four methods, both campaigns) and the test suite's one-stream check (women, ε
  1.5e-3) ran FDC-DP on dev outside the registered sequence. Neither fed any rule. The status files precede the
  registered, saved dev cells in both runs.
