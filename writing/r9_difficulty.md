# r9 decision-difficulty summary (external review r9, P1-05)

Generated from existing data only by `writing/scripts/difficulty_r9.py`; the machine-readable record is
`writing/r9_difficulty.json` (per-problem counts, loaders, sources). No new stream was run, no registered verdict,
threshold or preregistered number changes, and no statistic was computed on an evaluation half's outcomes.
LaTeX: `writing/supplement/r9_tables/difficulty.tex` (full table, supplement) and
`writing/latex_acm/r9_difficulty_main.tex` (9 rows x 6 columns, main paper, tabular only).

## Definitions

Every difficulty number is labelled **DEV** (development half) or **EVAL** (sealed evaluation rows).

| Column | Half | Definition |
|---|---|---|
| eps | none | The registered tolerance of the block. |
| eps/base | DEV | eps divided by the mean outcome over all rows of the development half (Criteo visit 4.70%, X5 purchase 62.0%, Lenta visit 10.9%, OBD click 0.350% / 0.484% / 0.501%). |
| eps/head | DEV | eps divided by the development headroom J*_15 - J(all-control), the largest gain over all-control at the loosest budget. A value above 1 means that eps exceeds the largest gain any feasible policy can make. |
| Pooled share (a) | DEV | sum over q of #{feasible eps-optimal policies} divided by sum over q of #{feasible policies}. This is the S12 census definition. eps-optimal means J*_q - J(pi) <= eps on the development truth, with J*_q from the exact DP. |
| Mean share (b) | DEV | Mean over the 15 problems of the per-problem share of feasible policies that are eps-optimal. This is the lock-v12 block-status gate definition (`obd_v11_eval.eps_opt_share`, `v12_analysis.block_status`), reused unchanged. |
| Ctrl (c) | DEV | The number of the 15 problems in which the all-control policy is eps-optimal (v12 gate criterion ii). |
| Hor. P/R | EVAL | Out of 200 sealed streams, the number on which 12 of 15 was **not** reached strictly before tau_R (N80_pen >= tau_R; field `n80_lt_tau == False` from lock v10 on), for P = primary method and R = governing rival. |
| Exh. P/R | EVAL | Out of 200 sealed streams, the number with any (segment, arm) pool exhausted at the N80 checkpoint (`exhaustion_at_k80['all'] > 0`). The field was recorded from lock v10 on, so locks v7-v9 show n.r. (not recorded). |

The governing rival is the rival that the sealed analysis file names as governing. For lock v7 this is
`governing_rival` (HC-WoR). Lock v8 names RECT-ck-HG for the rectangle test (F) and PJC-local for the PJC test (G).
Lock v9 names HC-WoR@D for superiority and TU-PJC@D for non-inferiority on X5. From lock v10 on, the only registered
rival is RECT-BF-DP-TU. The non-inferiority (NI) rivals of X5 are also counted in the supplement table and the json.

**Development-half loaders** are each lock's own frozen design. Each is sha-checked against its lock where the
loader supports it.
- CR9 and X9 use `run_r5s_v8.init_env(layer, 'dev', (eps,))`, the loader of the S12 census.
- Lock v10 uses `SegEnvV10Frozen(data, S, 'dev', frozen_sha256=<lock v10>)`.
- Lock v11 uses `OBDEnvV11Frozen('dev', ...)`.
- Lock v12 uses `OBDEnvV12Frozen(campaign, 'dev', ...)`.

**Exact counting for 2^S classes.** All lock-v10 treatment costs are 8 units, so problem q means "treat at most
floor(B_q/8) segments".
- S = 16 is counted by full enumeration.
- S = 32 is counted exactly by meet-in-the-middle over two 2^16 halves.
- S = 64 is counted by exact branch-and-bound over the cardinality structure. For X5 it closes in 15 nodes.
- For Lenta S = 64, problems 6-15 exceed the 2·10^7-node limit. They are reported as a rigorous bracket from an
  exact integer DP. The DP uses uplifts rounded down (lower bound) and up (upper bound) to a grid of 10^-5·eps, with
  exact uint64 counts. The bracket is [0.9997660, 0.9997680] for the mean share and [0.9994704, 0.9994750] for the
  pooled share. It is marked `*`, and it equals 1.00 / 0.999 at the printed precision.
- The DP bracket contained the exact count on all 15 problems of every S <= 32 cardinality cell (cross-check in the
  json).
- Pairs within 1e-9 of the eps boundary number 0 everywhere except Lenta S = 32 (at most 1,372 of ~1.9·10^9 per
  problem), so the boundary convention cannot move any printed digit.

## Reproduction of published numbers (all matched exactly before any new number was used)

| Block | Quantity | Published | Computed | Source |
|---|---|---|---|---|
| Criteo CR9 | (problem, policy) pairs / eps-optimal | 3,386 / 437 | 3,386 / 437 | S12; `verify_r4_numbers.py` census |
| X5 X9 | pairs / eps-optimal | 3,347 / 3,076 | 3,347 / 3,076 | S12; `verify_r4_numbers.py` census |
| OBD v11 | mean share / all-control | 0.14165 / 0 | 0.14165 / 0 | `exp/results/pilots/v11_dev/v11_analysis_dev.json` (dev) |
| OBD v12 women | mean share / all-control | 0.661 / 3 | 0.6611021 / 3 | `exp/results/v12_gates/v12_women_block_status.json`; S23 |
| OBD v12 men | mean share / all-control | 0.858 / 5 | 0.8575500 / 5 | `exp/results/v12_gates/v12_men_block_status.json`; S23 |

No published number failed to reproduce.

## Full table

| Block (lock) | eps | eps/base DEV | eps/head DEV | Pooled share DEV | Mean share DEV | Ctrl DEV | Hor. P/R EVAL | Exh. P/R EVAL |
|---|---|---|---|---|---|---|---|---|
| Criteo S=9 (v7) | 0.001 | 2.1% | 0.12 | 0.129 (437/3,386) | 0.139 | 0 | 0/0 | n.r. |
| Criteo S=9 (v9) | 0.001 | | | | | | 0/0 | n.r. |
| X5 S=9 (v8) | 0.02 | 3.2% | 0.68 | 0.919 (3,076/3,347) | 0.959 | 8 | 0/0 (NI PJC-local 0) | n.r. |
| X5 S=9 (v9) | 0.02 | | | | | | 0/0 (NI TU-PJC 0) | n.r. |
| X5 S=16 (v10) | 0.03 | 4.8% | 1.01 | 0.99999 (434,387/434,393) | 0.99999 | 15 | 0/0 | 0/0 |
| X5 S=32 (v10) | 0.04 | 6.4% | 1.23 | 1 (all 28,228,392,857) | 1 | 15 | 0/0 | 0/200 |
| X5 S=64 (v10, descr.) | 0.05 | 8.1% | 1.49 | 1 (all 1.208·10^20) | 1 | 15 | 0/18 | 0/200 |
| Lenta S=16 (v10) | 0.004 | 3.7% | 0.51 | 0.719 (312,493/434,393) | 0.848 | 3 | 0/3 | 191/200 |
| Lenta S=32 (v10) | 0.006 | 5.5% | 0.68 | 0.952 (26,861,126,237/28,228,392,857) | 0.978 | 5 | 0/0 | 5/200 |
| Lenta S=64 (v10) | 0.008 | 7.3% | 0.84 | 0.99947* | 0.99977* | 7 | 0/0 | 0/200 |
| OBD random/all S=9 (v11) | 3e-4 | 8.6% | 0.12 | 0.100 (340/3,398) | 0.142 | 0 | 0/12 | 0/12 |
| OBD random/women S=8 (v12, descr.) | 7.5e-4 | 15.5% | 0.34 | 0.529 (900/1,702) | 0.661 | 3 | 0/0 | 0/0 |
| OBD random/men S=10 (v12, descr.) | 1e-3 | 19.9% | 0.51 | 0.763 (5,153/6,757) | 0.858 | 5 | 0/0 | 0/0 |

`*` marks a rigorous bracket that equals the printed value to the digits shown (see Definitions).

**EVAL sources** (all under `exp/results/full/`). Each count is over 200 sealed streams. The rival named in each
analysis file is the one used.

| Block | Rows file | P | R | Analysis file |
|---|---|---|---|---|
| Criteo (v7) | `v7a_full_b/results.jsonl` | FDC-BF | HC-WoR | `v7_analysis_A.json` |
| Criteo (v9) | `v9a_full_d/results.jsonl` | TU-FDC | HC-WoR | `v9_analysis_A.json` |
| X5 S=9 (v8) | `v8a_full_a/results.jsonl` | FDC-BF | RECT-ck-HG; NI PJC-local | `v8_analysis_A.json` |
| X5 S=9 (v9) | `v9b_full_d/results.jsonl` | TU-FDC | HC-WoR; NI TU-PJC | `v9_analysis_B.json` |
| X5 S=16 (v10) | `v10a_full_s16/results.jsonl` | TU-FDC-DP(b) | RECT-BF-DP-TU | `v10_analysis_A.json` |
| X5 S=32 (v10) | `v10a_full_s32/results.jsonl` | TU-FDC-DP(b) | RECT-BF-DP-TU | `v10_analysis_A.json` |
| X5 S=64 (v10, descr.) | `v10d_full_x5s64/results.jsonl` | TU-FDC-DP(b) | RECT-BF-DP-TU | `v10_analysis_D.json` |
| Lenta S=16 (v10) | `v10b_full_s16/results.jsonl` | TU-FDC-DP(b) | RECT-BF-DP-TU | `v10_analysis_B.json` |
| Lenta S=32 (v10) | `v10b_full_s32/results.jsonl` | TU-FDC-DP(b) | RECT-BF-DP-TU | `v10_analysis_B.json` |
| Lenta S=64 (v10) | `v10b_full_s64/results.jsonl` | TU-FDC-DP(b) | RECT-BF-DP-TU | `v10_analysis_B.json` |
| OBD random/all (v11) | `v11_obd/v11_obd_full/results.jsonl` | TU-FDC-DP(b) | RECT-BF-DP-TU | `v11_obd/v11_analysis.json` |
| OBD random/women (v12) | `v12_obd/v12_women_full/results.jsonl` | TU-FDC-DP(b) | RECT-BF-DP-TU | `v12_obd/v12_women_analysis.json` |
| OBD random/men (v12) | `v12_obd/v12_men_full/results.jsonl` | TU-FDC-DP(b) | RECT-BF-DP-TU | `v12_obd/v12_men_analysis.json` |

The new EVAL counts agree with sentences already in the paper. The lock-v10 rectangle reaches 12 of 15 before tau_R
on 197 of 200 Lenta S = 16 streams and on all streams of the other four confirmatory cells. On OBD v11 the rectangle
reaches the horizon on 12 streams, and neither method empties a pool on the other 188. In lock v12 no pool is
exhausted at N80. Evaluation-truth shares were not recomputed. The ones the paper already quotes, which were computed
at analysis time inside the sealed analysis, are 0.151 / 0 all-control for v11 and 0.585 / 0.766 with 3 / 5
all-control for v12. They remain labelled EVAL.

## Plain reading (no registered verdict changes)

On the development halves, the X5 decisions are near-trivial at their tolerances:
- At S = 9, 0.92 of (problem, policy) pairs are eps-optimal, and all-control is eps-optimal in 8 of 15 problems.
- In every lock-v10 X5 cell (S = 16, 32, 64), eps exceeds the largest achievable gain over all-control. Essentially
  every feasible policy is eps-optimal (share 1.00), and all-control is eps-optimal in all 15 problems.
- Lenta's lock-v10 cells and the two lock-v12 campaigns are also undemanding: mean shares are 0.85-1.00 and 0.66-0.86,
  and all-control is eps-optimal in 3-7 problems.

The demanding blocks are Criteo (mean share 0.14, all-control never eps-optimal) and the never-replayed Open Bandit
lock-v11 block (mean share 0.14, all-control never eps-optimal; the governing rectangle needed tau_R on 12 of 200
evaluation streams). These are the blocks where a speed-up reflects selecting among genuinely different policies,
not certifying a broad near-tie set. The lock-v10 and X5 savings therefore measure how fast a certificate confirms a
frontier that most policies already meet. This does not alter any lock-v7-v11 verdict. Applying v12's gate to earlier
locks retrospectively is neither done nor implied.

## Numbers the main paper should quote

- **X5 (S12 census, DEV):** 3,076 of 3,347 (0.919) pairs are eps-optimal at eps = 0.02. The mean per-problem share is
  0.96, and all-control is eps-optimal in 8 of 15 problems.
- **Criteo (DEV):** 437 of 3,386 (0.129) pairs are eps-optimal at eps = 0.001. The mean share is 0.14, and
  all-control is eps-optimal in 0 of 15 problems.
- **Lock v10 X5 cells (DEV):** the mean share is 1.00 at S = 16, 32 and 64. All-control is eps-optimal in 15 of 15
  problems, and eps is 1.01-1.49 times the development headroom over all-control.
- **Lock v10 Lenta cells (DEV):** the mean share is 0.85, 0.98 and 1.00 at S = 16, 32 and 64. All-control is
  eps-optimal in 3, 5 and 7 of 15 problems.
- **Open Bandit lock v11 (DEV):** the mean share is 0.14 at eps = 3·10^-4 (8.6% of base), and all-control is
  eps-optimal in 0 of 15 problems. **EVAL:** the rectangle reached tau_R on 12 of 200 streams and the primary on 0;
  12 rectangle streams and 0 primary streams had an exhausted pool at N80.
- **Lock v12 (DEV, unchanged):** the mean share is 0.661 / 0.858, and all-control is eps-optimal in 3 / 5 of 15
  problems.
- **Lock v10 EVAL:**
  - Primary: 0 of 200 horizon streams in every cell.
  - Rectangle: 3 horizon streams at Lenta S = 16, 18 at the descriptive X5 S = 64 cell, and 0 elsewhere.
  - Exhausted pool at N80: rectangle on 200 of 200 streams in every Lenta cell and at X5 S = 32 and 64.
  - Exhausted pool at N80: primary on 191, 5 and 0 of 200 at Lenta S = 16, 32 and 64.

**Suggested caption for `r9_difficulty_main.tex`:**
"Decision difficulty. eps (% base): tolerance and its share of the development outcome rate. Share: mean over the
15 problems of the fraction of feasible policies that are eps-optimal on the development truth (the lock-v12 gate
definition). Ctrl: problems (of 15) whose all-control policy is eps-optimal (development). Hor. and Exh.: sealed
evaluation streams (of 200) on which 12 of 15 was not reached before tau_R, or some pool was exhausted at N80, for
the primary method / governing rival; n.r. = not recorded. ‡ ranges over S = 16, 32, 64; the 18 horizon streams are
in the descriptive S = 64 cell. † descriptive. * exact bracket, equal at the printed precision. Difficulty is
reported, not used to revise verdicts."
