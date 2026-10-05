# v8-posthoc-E: untested sensitivities (POST-HOC, DESCRIPTIVE, DEV ONLY)

**Post-hoc disclosure.** Run on 2026-10-04 AFTER the v8 confirmatory analysis and the r3 paper revision, in response to reviewer requests (supplement S9, critic review_v8 P1-7) for the sensitivities the paper lists as untested.  Knob values were fixed before any row of this block was computed (the requested values plus the total-delta values that Proposition 1's S9 prediction names).  DEV halves only (X5 X9 dev and CR9 dev, seeds 950-999, not used for any tuning); no evaluation row was accessed.  Rival configs are the frozen lock-v8 configs, not re-tuned per setting.  50 streams per cell: CIs are wider than in the confirmatory tables.  Nothing here changes a locked verdict.

Status: POST-HOC, DESCRIPTIVE -- outside any preregistration lock (v5-v8); not confirmatory.  Ratio: FDC-BF / rival paired geomean of N80_pen at the same setting (< 1: FDC-BF faster).  Bootstrap: v6 rule: B = 10^4 resamples of streams, seed 42, two-sided percentile 95% CI on the geomean.  Rival columns for knobs a rival does not have (split, lambda grid for RECT-ck-HG / HC-WoR; replanning for frozen designs) use that rival's baseline rows, which the knob cannot change.  PJC members are run at the row's split, total delta, K and replanning setting (same ledger as FDC-BF); in the lambda-grid rows they keep the 161-point grid, which changed no FDC-BF or RECT-ck-BF row.  At K = 10 / 40 the PJC phase boundaries are mapped to the checkpoint nearest in arrivals to the registered K = 20 boundary.

## X5 RetailHero X9 dev, seeds 950-999, eps 0.02 (tau_R = 99,646)

Integrity: baseline rows reproduce the lock-v8 dev runner check (v8a_pilot, eps 0.02) on all six shared methods x 50 seeds: **yes** (FDC-BF 0 diff, RECT-ck-HG 0 diff, HC-WoR 0 diff, RECT-ck-BF 0 diff, PJC-local 0 diff, PJC-menu 0 diff).

| setting | FDC-BF N80/tau | FDC-BF vs own baseline | / RECT-ck-HG* | / HC-WoR* | / RECT-ck-BF | / PJC-local* | / PJC-menu* | / PJC reset | false streams (FDC-BF; any rival) |
|---|---|---|---|---|---|---|---|---|---|
| baseline (K 20, delta 0.05 = 0.045/0.005, 161 lambdas) | 0.2489 | 1.000 | 0.336 [0.323, 0.350] | 0.329 [0.316, 0.345] | 0.306 [0.293, 0.317] | 0.988 [0.968, 1.008] | 0.929 [0.902, 0.952] | 0.910 [0.880, 0.940] | 0; 0 |
| delta_var 0.0025 | 0.2479 | 0.996 | 0.335 [0.321, 0.349] | 0.328 [0.313, 0.343] | 0.305 [0.293, 0.316] | 0.992 [0.968, 1.017] | 0.932 [0.910, 0.956] | 0.910 [0.880, 0.940] | 0; 0 |
| delta_var 0.01 | 0.2531 | 1.017 | 0.342 [0.328, 0.356] | 0.335 [0.321, 0.350] | 0.311 [0.300, 0.323] | 1.004 [0.988, 1.021] | 0.944 [0.921, 0.968] | 0.917 [0.891, 0.944] | 0; 0 |
| delta 0.01 (0.009/0.001) | 0.2816 | 1.131 | 0.349 [0.339, 0.359] | 0.346 [0.338, 0.355] | 0.346 [0.338, 0.355] | 1.004 [0.980, 1.029] | 0.968 [0.948, 0.988] | 0.952 [0.925, 0.980] | 0; 0 |
| delta 0.10 (0.09/0.01) | 0.2399 | 0.964 | 0.345 [0.332, 0.358] | 0.340 [0.328, 0.353] | 0.295 [0.284, 0.306] | 1.004 [0.984, 1.025] | 0.964 [0.940, 0.988] | 0.917 [0.888, 0.948] | 0; 0 |
| K = 10 | 0.2694 | 1.082 | 0.344 [0.321, 0.365] | 0.286 [0.274, 0.302] | 0.269 [0.265, 0.272] | 0.991 [0.974, 1.000] | 0.983 [0.958, 1.000] | 0.933 [0.893, 0.974] | 0; 0 |
| K = 40 | 0.2523 | 1.014 | 0.352 [0.341, 0.363] | 0.358 [0.347, 0.369] | 0.311 [0.302, 0.320] | 1.002 [0.988, 1.018] | 0.966 [0.953, 0.980] | 0.936 [0.916, 0.957] | 0; 0 |
| lambda grid 81 | 0.2489 | 1.000 | 0.336 [0.323, 0.350] | 0.329 [0.316, 0.345] | 0.306 [0.293, 0.317] | 0.988 [0.968, 1.008] | 0.929 [0.902, 0.952] | 0.910 [0.880, 0.940] | 0; 0 |
| lambda grid 321 | 0.2489 | 1.000 | 0.336 [0.323, 0.350] | 0.329 [0.316, 0.345] | 0.306 [0.293, 0.317] | 0.988 [0.968, 1.008] | 0.929 [0.902, 0.952] | 0.910 [0.880, 0.940] | 0; 0 |
| replan x0.5 | 0.2489 | 1.000 | 0.336 [0.323, 0.350] | 0.329 [0.316, 0.345] | 0.306 [0.293, 0.317] | 0.976 [0.952, 0.996] | 0.932 [0.910, 0.956] | 0.913 [0.884, 0.944] | 0; 0 |
| replan x2 | 0.2489 | 1.000 | 0.336 [0.323, 0.350] | 0.329 [0.316, 0.345] | 0.306 [0.293, 0.317] | 0.980 [0.956, 1.004] | 0.948 [0.925, 0.972] | 0.913 [0.880, 0.944] | 0; 0 |

## CR9 Criteo dev, seeds 950-999, eps 0.001 (tau_R = 6,989,793)

| setting | FDC-BF N80/tau | FDC-BF vs own baseline | / RECT-ck-HG* | / HC-WoR* | / RECT-ck-BF | / PJC-local* | / PJC-menu* | / PJC reset | false streams (FDC-BF; any rival) |
|---|---|---|---|---|---|---|---|---|---|
| baseline (K 20, delta 0.05 = 0.045/0.005, 161 lambdas) | 0.1120 | 1.000 | 0.649 [0.617, 0.684] | 0.663 [0.626, 0.699] | 0.525 [0.493, 0.559] | 1.000 [0.985, 1.016] | 0.974 [0.954, 0.995] | 0.915 [0.878, 0.954] | 0; 0 |
| delta_var 0.0025 | 0.1120 | 1.000 | 0.649 [0.617, 0.684] | 0.663 [0.626, 0.699] | 0.525 [0.493, 0.559] | 1.000 [0.985, 1.016] | 0.974 [0.954, 0.995] | 0.915 [0.878, 0.954] | 0; 0 |
| delta_var 0.01 | 0.1120 | 1.000 | 0.649 [0.617, 0.684] | 0.663 [0.626, 0.699] | 0.525 [0.493, 0.559] | 0.990 [0.974, 1.000] | 0.969 [0.944, 0.990] | 0.915 [0.878, 0.954] | 0; 0 |
| delta 0.01 (0.009/0.001) | 0.1217 | 1.087 | 0.613 [0.582, 0.646] | 0.617 [0.591, 0.643] | 0.496 [0.473, 0.522] | 1.010 [0.990, 1.037] | 0.979 [0.959, 0.995] | 0.935 [0.906, 0.964] | 0; 0 |
| delta 0.10 (0.09/0.01) | 0.1047 | 0.935 | 0.646 [0.607, 0.684] | 0.646 [0.607, 0.688] | 0.501 [0.470, 0.533] | 1.005 [0.985, 1.026] | 0.944 [0.915, 0.969] | 0.887 [0.847, 0.930] | 0; 0 |
| K = 10 | 0.1176 | 1.050 | 0.624 [0.578, 0.674] | 0.610 [0.565, 0.666] | 0.518 [0.474, 0.559] | 0.978 [0.936, 1.022] | 0.926 [0.877, 0.968] | 0.896 [0.839, 0.957] | 0; 0 |
| K = 40 | 0.1073 | 0.958 | 0.634 [0.599, 0.667] | 0.686 [0.647, 0.723] | 0.503 [0.474, 0.532] | 1.003 [0.995, 1.013] | 0.975 [0.955, 0.990] | 0.920 [0.883, 0.955] | 0; 0 |
| lambda grid 81 | 0.1120 | 1.000 | 0.649 [0.617, 0.684] | 0.663 [0.626, 0.699] | 0.525 [0.493, 0.559] | 1.000 [0.985, 1.016] | 0.974 [0.954, 0.995] | 0.915 [0.878, 0.954] | 0; 0 |
| lambda grid 321 | 0.1120 | 1.000 | 0.649 [0.617, 0.684] | 0.663 [0.626, 0.699] | 0.525 [0.493, 0.559] | 1.000 [0.985, 1.016] | 0.974 [0.954, 0.995] | 0.915 [0.878, 0.954] | 0; 0 |
| replan x0.5 | 0.1120 | 1.000 | 0.649 [0.617, 0.684] | 0.663 [0.626, 0.699] | 0.525 [0.493, 0.559] | 1.010 [1.000, 1.026] | 0.974 [0.954, 0.995] | 0.920 [0.883, 0.964] | 0; 0 |
| replan x2 | 0.1120 | 1.000 | 0.649 [0.617, 0.684] | 0.663 [0.626, 0.699] | 0.525 [0.493, 0.559] | 1.016 [1.000, 1.037] | 0.959 [0.930, 0.985] | 0.925 [0.887, 0.959] | 0; 0 |

## Robustness

- **X9.** Rectangles: every FDC-BF/rectangle CI upper bound below 1 at every setting: yes; worst 0.358 (CI upper 0.369, HC-WoR at K=40).  Joint certificates (PJC): largest CI upper bound 1.029 (PJC-local at dtot=0.01, ratio 1.004); smallest ratio 0.910 (PJC-reset at base); all PJC CI upper bounds below 1.05: yes.  False streams over all method-setting cells: 0.
- **CR9.** Rectangles: every FDC-BF/rectangle CI upper bound below 1 at every setting: yes; worst 0.686 (CI upper 0.723, HC-WoR at K=40).  Joint certificates (PJC): largest CI upper bound 1.037 (PJC-local at dtot=0.01, ratio 1.010); smallest ratio 0.887 (PJC-reset at dtot=0.1); all PJC CI upper bounds below 1.05: yes.  False streams over all method-setting cells: 0.

## Interpretation (descriptive)

**The conclusions are robust on these dev streams.**  (1) *Rectangles (C2 direction):* FDC-BF/rectangle stays between 0.27 and 0.36 on X5 and between 0.50 and 0.69 on CR9 at every setting, with every CI upper bound below 1 (worst 0.369 on X5 and 0.723 on CR9, both HC-WoR* at K = 40).  The matched Bennett rectangle (RECT-ck-BF, the C4 contrast) moves least (X5 0.27-0.35, CR9 0.50-0.53).  (2) *Joint certificates (C3 direction):* when PJC is given the same split, delta, K and replanning batch, FDC-BF/PJC-local* stays within 0.976-1.016 (largest CI upper bound 1.037, below the registered 1.05 margin), and FDC-BF stays ahead of PJC-menu* and of the post-hoc reset pick at every setting (point estimates 0.89-0.98; CI upper bound reaches 1.000 only for PJC-menu* at K = 10 on X5).  So the tie with the no-reset joint twin and the small lead over the adaptive members do not depend on the registered knob values.  (3) *Knob by knob:* the lambda-grid density changes no N80 row of FDC-BF or RECT-ck-BF on either layer (x12 changes by at most 1.1%), so the 161-point grid is not a tuned choice.  Moving delta_var between 0.0025 and 0.01 changes FDC-BF by at most 2%.  K = 10 coarsens N80 for everyone, and K = 40 leaves the ratios within about 0.02 (X5) and 0.04 (CR9) of baseline.  Halving or doubling the replanning batch changes FDC-BF/PJC by at most 0.02; adaptive members gain nothing from finer batches.  On X5 the x0.5 batch (100 arrivals) is below the registered 200-arrival floor.  (4) *Proposition 1's delta prediction (S9) is borne out on CR9 and not on X5.*  On CR9, delta = 0.01 favours the joint width against RECT-ck-HG* (0.613 against 0.649 at baseline; x12 0.611 against 0.636), and delta = 0.10 leaves the ratio about unchanged or slightly toward the rectangle (x12 0.647).  On X5 both delta changes move the ratio slightly toward the rectangle (0.349 / 0.345 against 0.336).  These shifts are within about 0.04, i.e. second-order.  The total delta changes are not paper settings: the guarantee level moves with them (FWER <= delta).  (5) *Validity:* 0 false streams in all 5,500 method-setting-stream runs (2 layers x 55 method-setting cells x 50 streams), including delta = 0.10.

x12 (interpolated time to rank 12, finer than the checkpoint grid), FDC-BF/RECT-ck-HG*: X9 base 0.344, dtot=0.01 0.358, dtot=0.1 0.340; CR9 base 0.636, dtot=0.01 0.611, dtot=0.1 0.647.

See summary.json for every cell (CI, UB95, fast/tied/slow fractions, rival geomeans, betas).
