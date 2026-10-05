# Lock v11 addendum (DRAFT): FDC-DP on the Open Bandit Dataset eval half

Status: **DRAFT, not locked.** The machine-readable draft is `plan/prereg_lock_v11_addendum_DRAFT.json`, built by
`exp/code/build_v11_addendum_draft.py`, which computes every hash. A draft never authorises an eval read
(`prereg_v11.load_locked_addendum` refuses the draft path and any status other than `locked`).

- **Plan.** `plan/v11_obd_plan.md`, committed alone before any v11 dev stream (40ee83d8).
- **Dev evidence.** `plan/v11_dev_report.md`.
- **Lock procedure.**
  1. external reviewer lock review.
  2. Rebuild the draft.
  3. Code-freeze commit.
  4. `prereg_v11.finalize_addendum(<freeze commit>)`.
  5. Commit the lock file alone.
  6. Run `run_v11.py --task v11_obd_full`, then `--task v11_obd_full --replica`, then `--analyse`.

## 1. What is registered

There is one confirmatory block (one verdict) on the Open Bandit Dataset `random/all` eval half. It has 688,159 rows,
this is its first use, and the whole half is the replay population. The block uses fresh seeds 39000–39199.

**Design** (frozen from dev in `exp/results/v11_gates/obd_v11_frozen.json`):
- **Segments.** S = 9 user_feature_0 × user_feature_3 segments.
- **Arms.** A = 2 item groups: the top-40 versus bottom-40 items by dev CTR.
- **Problems.** Exposure-cost knapsack problems with κ[s,1] = 26, 14, 14, 4, 3, 3, 3, 2, 2 and 15 budgets (0.10–0.80
  of total cost).
- **Sampling.** Outcome-free 50/50 design, δ = 0.05.
- **Checkpoints.** K = 40, changed from 20.
- **Stopping.** Streams run to 15/15.

**Methods:**
- **Primary.** TU-FDC-DP(b).
- **Governing rival.** RECT-BF-DP-TU.
- **Descriptive.** HC-WoR-DP[tuned], with c = 0.5 and tf = 0.6 tuned on dev by the block-G3 rule, and RECT-HG-DP,
  labelled CP because its guarantee is checkpoint-strength only.

**ε = 3e-4** (CTR units; 8.6 % of the dev base CTR). The pre-stated rival-success rule chose it on 50 dev streams.
The dev-best rectangle at 3e-4 was HC-WoR-DP[tuned], with 47/50 successes; RECT-BF-DP-TU had 48/50. 14 % of feasible
policies are ε-optimal on dev.

**Endpoint and decision:**
- **Endpoint.** N80_pen at 12/15.
- **Statistic.** Paired geometric-mean ratio TU-FDC-DP(b) / RECT-BF-DP-TU over 200 streams, with a paired percentile
  bootstrap (B = 10⁴, seed 42) and a one-sided UB95.
- **Decision.** `positive_result_achieved` iff:
  1. UB95 < **THRESH = 0.90**;
  2. TU-FDC-DP(b) has 0/200 false streams over the whole run;
  3. the replica check (R1/R2/R2b) passes.
- **THRESH.** THRESH = min(0.90, UB95_dev + 0.10). The rule was stated in the plan; UB95_dev was 0.840, so the cap
  applies.

## 2. Dev evidence

| | ratio | UB95 | false streams |
|---|---|---|---|
| FDC / RECT-BF-DP-TU | 0.8255 | 0.840 | 0 (all methods) |
| FDC / HC tuned | 0.828 | 0.840 | |
| FDC / RECT-HG-DP (CP) | 0.840 | 0.851 | |

The gain is 1–3 steps of the K = 40 grid near the end of the table. FDC certifies at 0.73 τ and the rectangles at
about 0.87–0.89 τ.

## 3. Eval access and blinding

- **Eval reader** (order fixed after external reviewer lock review r1, F1). `dsswm.envs.obd_v11_eval.read_eval_rows` runs:
  1. Check the caller's layer and that a sha256 was passed.
  2. The v11 gate checks status, schema, task membership, the v10→v5 chain, the canonical hash, code and input
     drift, drift of the non-eval data files, and single-commit history. **It never opens an eval file**: the
     eval-file hashes it checks are the ones recorded in the hash-bound `PROVENANCE.json`.
  3. The task must be registered for exactly `OBD-UF0X3-A2-S9`.
  4. The caller's lock sha256 must equal the locked addendum's.
  5. An access-log line is appended and fsync'ed (`exp/results/full/v11_obd/eval_access_log.jsonl`).
  6. Only then are the eval bytes hashed and compared with the lock and PROVENANCE, and the rows deserialised.

  A refusal at any step before 5 touches no eval file, and a failed log write refuses as well. Tests run under a
  suite-wide guard that fails on any open of a real eval file.
- **Pre-lock eval-file access.**
  - The split script (before v11) wrote and hashed the eval files. It also recorded eval label-only shares (position,
    item) in PROVENANCE, but no outcome statistic.
  - No v11 code has opened or hashed `eval_outcome.npy`.
  - The first v11 dev runs and the first draft build hashed the bytes of `eval_labels.pkl` without deserialising it.
    This happened before r1. Since r1, both eval hashes come from PROVENANCE.
- **Eval seal.** The eval task is sealed on completion (`v11_seal`) and is never re-run.

## 4. Disclosures

1. **Dev-outcome-informed choices.** The arm grouping and the choice of segmentation used dev outcomes, so selection
   shrinkage is expected. The feasibility holdout at 5e-4 (0.78 vs 0.72) froze only the item groups; it recomputed
   segments and costs on the replay half, so it is an item-group holdout only (feasibility erratum, r1 F4).
2. **Weak dependence between halves.** The split is at row level. 3.3 % of dev rows share a timestamp with an eval
   row, and repeat users are possible.
3. **Scope of inference.** This is the first use of the eval half, and inference is conditional on that finite table.
4. **K changed from 20 to 40.** One consequence is that RECT now succeeds at 3e-4 on dev, so the rule selected 3e-4
   rather than the 5e-4 the feasibility study expected. The rule was applied as written.
5. **Costs were frozen from dev shares.** Eval label shares were not used, so no eval read was needed.
6. **HC selection bias.** HC was tuned on the rule's dev streams, which makes it selection-optimistic for HC.
7. **RECT-HG-DP has a CP guarantee only.**
8. **THRESH is dev-calibrated.** The rule was pre-stated and its value is capped at 0.90.
9. **Scope of the result.**
   - FDC-DP's certificate is FDC-BF's, so the contribution is computational.
   - Claims are restricted to the named implementations and to methods with finite-sample guarantees.
10. **Not registered.** A = 3 is not registered. random/men, random/women and bts/* remain unread.
11. **Quantisation.** The dev ratio is quantised to about 1.134 per grid step, and the dev margin to THRESH is about
    half a step.
12. **Dev block regenerated once** after the r1 gate fix, which changed no stream computation. The gate files were
    rewritten for that reason only, and the values are compared in the dev report.
