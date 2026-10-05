# v11 dev report: Open Bandit Dataset block (dev half only; eval untouched)

Written 2026-10-05 after the v11 dev block. Plan: `plan/v11_obd_plan.md` (commit 40ee83d8, before any v11 dev stream).
Code: `exp/code/run_v11.py`, `dsswm/envs/obd_v11_eval.py`, `dsswm/stats/{prereg_v11,v11_analysis,v11_replica,v11_seal}.py`
(first version commit 45435f70; gate fix after external reviewer lock review r1). Rows are bound to the code hash.

**Eval-file access.**
- No eval outcome byte was opened or hashed by v11 code.
- The first dev run (commit 8ed70164) hashed the bytes of `eval_labels.pkl` for row binding, without deserialising
  it. Since r1, both eval-file hashes come from the hash-bound `PROVENANCE.json`.
- The split script, before v11, recorded eval label-only shares in PROVENANCE.

**Regeneration after r1.** The dev block was re-run once on the fixed code. The fix touched gating and data binding
only, not any stream computation. All 2,450 rows reproduce the first run's N80_pen, cert_k, decided policies and
schedule/arrival digests exactly. The selection (ε = 3e-4, HC (0.5, 0.6)), the dev ratio and UB95, and THRESH are
unchanged. The gate files `v11_eps_hc.json`, `v11_thresh.json` and `MANIFEST.json` were rewritten once for this reason
only.

**Bottom line.** The pre-stated rule selected **ε = 3e-4** (8.6 % of the dev base CTR 0.350 %). On the 50 dev streams
TU-FDC-DP(b) needs **0.8255×** the rows of RECT-BF-DP-TU (paired geometric mean; one-sided UB95 **0.840**, 95 % CI
0.807–0.842), faster on 50/50 streams, with 0 false streams for every method. The THRESH rule gives
min(0.90, 0.840 + 0.10) = **THRESH = 0.90**. The effect is real but small and near the end of the table, so the eval
verdict is at genuine risk.

## 1. Frozen design (`exp/results/v11_gates/obd_v11_frozen.json`)

- **Segments.** 9 uf0 × uf3 cells, written as an explicit map of 23 dev-observed (uf0, uf3) pairs plus 4 "uf0|other"
  routes. The map reproduces `obd_v11.segmentation(dev, 'uf0x3')` row for row (unit test). Dev shares are 0.367,
  0.193, 0.193, 0.057, 0.048, 0.043, 0.035, 0.032, 0.031.
- **Arms.** Top-40 vs bottom-40 items by dev CTR, frozen as item lists.
- **Costs.** κ[s,1] = 26, 14, 14, 4, 3, 3, 3, 2, 2 (from the dev shares). There are 15 budgets: 7, 10, 14, 17, 21,
  24, 28, 31, 35, 39, 42, 46, 49, 53, 56.
- **Grid.** K = 40 checkpoints from 5,000 to τ. On dev τ = 686,168, and one log step is a factor of about 1.134.

## 2. ε rule and HC tuning (`exp/results/v11_gates/v11_eps_hc.json`)

The rule and tuning used 50 dev streams (seeds 950–999). FDC-DP rows were never passed to the rule, and `select_eps`
refuses them.

| ε | RECT-BF-DP-TU: successes / geo N80 | HC tuned (config): successes / geo N80 | dev-best | qualifies |
|---|---|---|---|---|
| 3e-4 | 48 / 607,878 | (c 0.5, tf 0.6) 47 / 606,345 | HC | **yes → selected** |

- **Grid stop.** The rule stops at the smallest qualifying ε, so cells above 3e-4 played no part in the selection. HC
  tuning ran at every grid ε; the full traces are in the gate file.
- **HC tuned configurations.** At 5e-4 the tuned config was (0.5, 0.6). At 7.5e-4, 1e-3 and 1.5e-3 it was (0.5, 0.4).
- **Tie-break.** At 3e-4, (0.5, 0.6), (0.5, 0.8) and (0.75, 0.8) tie exactly. Grid order picks (0.5, 0.6).
- **No false streams.** None of the 2,250 rule streams had a false certificate.
- **Difference from the feasibility pilot.**
  - The pilot (K = 20, 30 streams) found that RECT "never certifies" at 3e-4. With K = 40, RECT reaches 12/15 at
    checkpoint 38 of 40, i.e. N80 = 0.886 τ, strictly before τ, on 48/50 streams.
  - The finer grid therefore moved the rule's selection from the expected 5e-4 down to 3e-4. This is a consequence of
    the pre-stated K change, not a choice.
  - The rule was applied as written.

## 3. Dev ε cell (`v11_obd_pilot`, all four methods, ε = 3e-4; `exp/results/pilots/v11_dev/v11_analysis_dev.json`)

| comparison | geo-mean ratio | one-sided UB95 | faster / tied / slower |
|---|---|---|---|
| **TU-FDC-DP(b) / RECT-BF-DP-TU** (governing) | **0.8255** | **0.840** | 1.00 / 0 / 0 |
| TU-FDC-DP(b) / HC-WoR-DP[tuned] | 0.828 | 0.840 | 1.00 / 0 / 0 |
| TU-FDC-DP(b) / RECT-HG-DP (CP) | 0.840 | 0.851 | 1.00 / 0 / 0 |
| RECT-BF-DP-TU / HC-WoR-DP[tuned] | 1.003 | 1.010 | 0.02 / 0.94 / 0.04 |
| RECT-BF-DP-TU / RECT-HG-DP | 1.018 | 1.028 | 0 / 0.86 / 0.14 |

Per-method results on the 50 dev streams:

| method | N80 / τ (geo) | N80 < τ | reached 15/15 | false streams | exhausted pools at N80 |
|---|---|---|---|---|---|
| TU-FDC-DP(b) | 0.731 | 50/50 | 50/50 | 0 | 0 |
| RECT-BF-DP-TU | 0.886 | 48/50 | 50/50 | 0 | 0.04 |
| HC-WoR-DP[tuned] | 0.884 | 47/50 | 50/50 | 0 | 0.06 |
| RECT-HG-DP (CP) | 0.870 | 50/50 | 50/50 | 0 | 0 |

- **Node limit.** TU-FDC-DP(b) hit the branch-and-bound node limit 0 times.
- **Where each method certifies.** TU-FDC-DP(b) reaches 12/15 at checkpoint index 35, 36 or 37 (2, 20 and 28
  streams). RECT-BF-DP-TU reaches it at index 38 on 48 streams and at τ on 2 streams.
- **Quantisation.** The ratio is therefore a gain of 1–3 grid steps, and grid quantisation is still visible at K = 40.
  The two-sided CI is narrow (0.81–0.84) because the outcome is close to discrete.
- **ε units and triviality.**
  - ε = 3e-4 equals 0.03 percentage points of CTR, or 8.6 % of the dev base CTR.
  - Only 14.2 % of feasible policies are ε-optimal on dev (mean over problems), and no problem has all-control
    ε-optimal, so the frontier is non-trivial at this ε.
- **Replica.** The dev runner check passed:
  - R1: seeds 950–959 were re-run in an independent process, and the canonical rows are identical.
  - R2 and R2b pass on all 50 streams.
  - The pilot's HC-WoR-DP[tuned] rows reproduce the rule-task rows of config (0.5, 0.6) exactly.

## 4. THRESH (`exp/results/v11_gates/v11_thresh.json`)

THRESH = min(0.90, UB95_dev + 0.10) = min(0.90, 0.9402) = **0.90**, with UB95_dev = 0.8402. The dev decision, if the
rule were applied to dev, is positive. The gate files are hashed in `exp/results/v11_gates/MANIFEST.json`.

## 5. Risks for the eval run (stated before the lock)

- **Shrinkage.** The arm groups were picked on dev outcomes, so the eval uplifts shrink. In the feasibility holdout at
  5e-4 the ratio rose from 0.72 to 0.78. That holdout froze only the item groups and recomputed segments and costs on the replay half (feasibility erratum). The dev margin to THRESH is 0.90 − 0.84 = 0.06 on the UB, about half a grid
  step, which is thin.
- **Discreteness.** Both methods certify in the last ~12 % of the table (indices 35–38 of 40). A one-step shift
  changes the ratio by a factor of about 1.13.
- **τ_eval ≈ 688k.** τ is about 0.3 % larger than on dev, which shifts the grid slightly. Eval segment shares differ
  from dev by sampling only. Costs are frozen from dev.
- **Rule-boundary event.** On eval the rectangle may fail to reach 12/15 before τ. Its N80_pen is then τ (penalised),
  which favours FDC-DP. This is the pre-registered endpoint and is reported with the share of N80 < τ.
- **Expected outcome.** A positive result is plausible but not assured. A not-achieved verdict at UB in (0.90, 1)
  would carry the sub-label faster_below_1.
