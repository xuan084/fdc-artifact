# Lock v12: results summary (Open Bandit women and men campaigns; both blocks DESCRIPTIVE by the pre-stated gate)

- **Plan, lock and eval.**
  - Plan: `plan/v12_obd2_plan.md`, committed alone (317619b4, then 5b3eb048) before any split, click read or dev stream.
  - Lock: `plan/prereg_lock_v12_addendum.json`, sha256 9a99ef28…075c189; code freeze 20a5e314; lock commit be886f1c (2026-10-05).
  - external reviewer lock review, 2 rounds (`reviews/v12_lock_review_r{1,2}.md`). Round 2 was fix-then-lock with no P0/P1. Its two P2 wording residues were fixed before the lock, which is disclosed in the lock record.
  - Eval: 2026-10-05, 18:3x–18:46.
  - Seals committed alone: women 730f1d21 (ref 594a2470), men d994784b (ref 1e1aa9df). Run records: c161bec1. Replica: pass for both.
- **Protocol.** Identical to lock v11, with the same written rules re-derived on each campaign's own dev half after a fresh salted 50/50 split. Each eval half was read once.
- **Block status.** The gate fixed in the plan before any data was read makes a block confirmatory only if the mean ε-optimal policy share on dev at the rule-selected ε is at most 0.5.
  - Dev shares were 0.661 (women) and 0.858 (men), so both blocks are **descriptive**. Gates (i) and (ii) passed.
  - Neither block has a positive or negative verdict. Lock v11 remains the only confirmatory Open Bandit block.

## Results (200 fresh eval streams per campaign; paired geometric-mean N80_pen; bootstrap B = 10⁴, seed 42)

| Campaign (ε, ε / base CTR) | TU-FDC-DP / RECT-BF-DP-TU [95% CI] (UB95) | / HC-WoR-DP tuned | / RECT-HG-DP (CP) | Streams faster | ε-optimal share (eval truth) |
|---|---|---|---|---|---|
| women (7.5e-4, 15.5%) | 0.677 [0.670, 0.684] (0.683) | 0.715 | 0.731 | 100% | 0.585 |
| men (1e-3, 19.9%) | 0.739 [0.734, 0.745] (0.744) | 0.751 | 0.781 | 100% | 0.766 |

| Campaign | N80/τ: TU-FDC-DP / RECT-BF-DP-TU / HC tuned / RECT-HG-DP | False streams (all methods) | Exhausted fraction at N80 (all methods) |
|---|---|---|---|
| women | 0.597 / 0.883 / 0.835 / 0.818 | 0 | 0 |
| men | 0.670 / 0.906 / 0.892 / 0.858 | 0 | 0 |

## Reading

**The ordering replicates on two independent campaigns.**
- The joint certificate read 0.68× (women) and 0.74× (men) the matched rectangle's rows, and stopped earlier on every stream.
- Every rival stopped strictly before the end of the table (share N80 < τ_R = 1.0), and no method exhausted any pool.
- These readings meet the lock-v10 pre-exhaustion condition (both fractions zero) in both campaigns. Unlike v11, they also have no horizon penalty.

**Why the blocks are descriptive.** On these smaller campaigns the frozen ε rule moves ε to 7.5e-4 and 1e-3, because both rectangles fail at smaller ε. At those tolerances 59–77% of policies are ε-optimal on eval, so the frontier is less demanding, and the gate pre-stated this as non-confirmatory.

**What the results support.**
- They are descriptive replication evidence of the direction and size of the saving.
- They are not a second confirmatory test, and their ratios must not be pooled with v11's.

## Disclosures (from the lock)

- **Smoke tests.** An unsaved smoke test (one stream per method, both campaigns) and one test-suite dev stream ran before the rule. Neither fed any rule.
- **Dev regeneration.** The dev block was regenerated once after an output-relabelling change; all 4,980 rows were identical.
- **Overlap with other halves.** Exact-timestamp overlap with the exposed random/all eval half is 35 rows (women) and 29 (men). Women and men share 27 rows.
- **Data provenance.** Split preparation wrote and hashed the eval files. Post-split checks take the eval hashes from the hash-bound provenance files.
- **Same design-related disclosures as v11.** Arm groups and segmentation were derived with dev outcomes; costs are traffic shares, not prices; K = 40; HC was tuned on dev; RECT-HG-DP has checkpoint strength only.
- **Thresholds.** THRESH was computed (women 0.83, men 0.90) but is not applicable to a descriptive block.
