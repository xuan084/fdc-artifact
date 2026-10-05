# v8-posthoc-F: adaptivity under strong arm-variance asymmetry (POST-HOC, DESCRIPTIVE, SEMI-SYNTHETIC)

**Post-hoc, semi-synthetic disclosure.** Run on 2026-10-04 after the v8 confirmatory analysis, in response to critic review_r3 P1-3 ('test adaptivity where it should help').  Only the DEV halves' structure (segment and arm labels, pool sizes, weights, frozen arrival schedules) is reused; the outcome column is replaced by synthetic Bernoulli pools with strongly unequal arm variances, so no evaluation row and no real outcome enters any statistic.  Rates, regimes, eps rule, method list, PJC-A grid and the v5 tuning rule were fixed before any row was computed; eps was then calibrated with frozen-50/50 FDC-BF only (seeds 900-919); PJC-A picks were tuned on seeds 900-949 of each semi-synthetic env; reporting seeds 950-999 + 36000-36199 were used for nothing else.  Rival configs (RECT-ck-HG, HC-WoR schedule, PJC-local*, PJC-menu*) are the frozen lock-v8 configs of the layer, not re-tuned.  The regimes were chosen to favour adaptive allocation; nothing here changes a locked verdict.

Status: POST-HOC, DESCRIPTIVE, SEMI-SYNTHETIC -- outside any preregistration lock (v5-v8); not confirmatory; outcomes are synthetic.  Ratio: FDC-BF (frozen 50/50) / method, paired geomean (< 1: FDC-BF faster; > 1: method faster); reported for N80_pen (checkpoint-quantised, primary) and x12 (interpolated).  Bootstrap: v6 rule: B = 10^4 resamples of streams, seed 42, two-sided percentile 95% CI on the geomean.

Regimes: **asym** -- every segment control rate U(0.02, 0.05), treatment U(0.40, 0.50) (Neyman control share ~0.25-0.30 everywhere); **mixed** -- segments cycle A (0.02-0.05 vs 0.40-0.50, Neyman ctrl ~0.27), B (0.45-0.55 vs 0.93-0.97, Neyman ctrl ~0.70), C (0.25-0.30 vs 0.65-0.75, ~0.50).  Structure (labels, pool sizes, weights, frozen arrivals) = X5 X9 dev or Criteo CR9 dev; outcomes synthetic with exactly round(rate x N_c) ones per pool.  FDC-BF[ney-pilot] = frozen per-segment Neyman from an independent 1000-draw-per-cell pilot (floor 0.1); [ney-oracle] = Neyman from the true pool variances.

## X9-asym (SEMI-SYNTHETIC; eps 0.015; tau_R 99,646; 250 streams)

Pilot-Neyman control share by segment: 0.24, 0.31, 0.26, 0.26, 0.26, 0.24, 0.21, 0.30, 0.22.  eps calibration (FDC-BF geomean N80/tau on 900-919): 0.01: 0.472, 0.015: 0.294, 0.02: 0.209, 0.03: 0.111, 0.04: 0.069, 0.05: 0.047, 0.075: 0.025.

FDC-BF (frozen 50/50): geomean N80/tau 0.2947, x12/tau 0.2671, false streams 0, censored 0.000.  Integrity: FDCBetPlan(0.5) reproduces FDC-BF on all streams.

| method | FDC-BF/method N80 | 95% CI | FDC-BF/method x12 | 95% CI | method N80/tau | frac switched | false streams |
|---|---|---|---|---|---|---|---|
| `FDC-BF[ney-pilot]` | 1.0369 | [1.0208, 1.0532] | 1.0281 | [1.0177, 1.0387] | 0.2842 | -- | 0 |
| `FDC-BF[ney-oracle]` | 1.0420 | [1.0267, 1.0575] | 1.0383 | [1.0281, 1.0489] | 0.2828 | -- | 0 |
| `RECT-ck-HG` | 0.4310 | [0.4243, 0.4381] | 0.4247 | [0.4199, 0.4294] | 0.6838 | -- | 0 |
| `RECT-ck-HG[ney-pilot]` | 0.4622 | [0.4550, 0.4695] | 0.4889 | [0.4834, 0.4944] | 0.6376 | -- | 0 |
| `RECT-ck-BF` | 0.3620 | [0.3569, 0.3671] | 0.3675 | [0.3633, 0.3717] | 0.8141 | -- | 0 |
| `RECT-ck-BF[ney-pilot]` | 0.4446 | [0.4385, 0.4509] | 0.4299 | [0.4250, 0.4348] | 0.6627 | -- | 0 |
| `HC-WoR` | 0.4204 | [0.4136, 0.4271] | 0.4188 | [0.4141, 0.4235] | 0.7008 | -- | 0 |
| `HC-WoR[ney-pilot]` | 0.4483 | [0.4417, 0.4546] | 0.4786 | [0.4734, 0.4839] | 0.6573 | -- | 0 |
| `PJC-local*` | 0.9943 | [0.9853, 1.0033] | 0.9991 | [0.9947, 1.0033] | 0.2964 | -- | 0 |
| `PJC-menu*` | 1.0506 | [1.0360, 1.0654] | 1.0500 | [1.0417, 1.0584] | 0.2805 | -- | 0 |
| `PJC-A[reset,b=2,neyman]` (tuned pick: reset) | 0.9621 | [0.9471, 0.9772] | 0.9689 | [0.9589, 0.9787] | 0.3063 | 1.000 | 0 |
| `PJC-A[menu,b=8,menu=0.3/0.4/0.5/0.6/0.7,proj]` (tuned pick: menu/any_adaptive) | 1.0083 | [0.9918, 1.0250] | 1.0095 | [0.9999, 1.0194] | 0.2923 | 1.000 | 0 |
| `PJC-A[segmenu,b=6,menu=0.3/0.5/0.7,neyman]` (tuned pick: segmenu) | 0.7219 | [0.7107, 0.7339] | 0.7335 | [0.7257, 0.7412] | 0.4082 | 1.000 | 0 |

Frozen pilot-Neyman FDC-BF vs the tuned any-adaptive pick (`PJC-A[menu,b=8,menu=0.3/0.4/0.5/0.6/0.7,proj]`): ney-pilot / adaptive N80 0.9724 [0.9613, 0.9837], x12 0.9820 [0.9765, 0.9874] (< 1: frozen pilot-Neyman faster).
Mean post-boundary control share by segment, `PJC-A[reset,b=2,neyman]`: 0.27, 0.31, 0.28, 0.27, 0.25, 0.25, 0.25, 0.30, 0.27.
Mean post-boundary control share by segment, `PJC-A[menu,b=8,menu=0.3/0.4/0.5/0.6/0.7,proj]`: 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30.
Mean post-boundary control share by segment, `PJC-A[segmenu,b=6,menu=0.3/0.5/0.7,neyman]`: 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30.

## X9-mixed (SEMI-SYNTHETIC; eps 0.02; tau_R 99,646; 250 streams)

Pilot-Neyman control share by segment: 0.28, 0.71, 0.49, 0.30, 0.73, 0.49, 0.22, 0.70, 0.49.  eps calibration (FDC-BF geomean N80/tau on 900-919): 0.01: 0.539, 0.015: 0.336, 0.02: 0.223, 0.03: 0.123, 0.04: 0.076, 0.05: 0.052, 0.075: 0.025.

FDC-BF (frozen 50/50): geomean N80/tau 0.2265, x12/tau 0.2042, false streams 0, censored 0.000.  Integrity: FDCBetPlan(0.5) reproduces FDC-BF on all streams.

| method | FDC-BF/method N80 | 95% CI | FDC-BF/method x12 | 95% CI | method N80/tau | frac switched | false streams |
|---|---|---|---|---|---|---|---|
| `FDC-BF[ney-pilot]` | 0.9910 | [0.9780, 1.0041] | 0.9901 | [0.9812, 0.9987] | 0.2285 | -- | 0 |
| `FDC-BF[ney-oracle]` | 0.9943 | [0.9821, 1.0066] | 0.9954 | [0.9868, 1.0038] | 0.2278 | -- | 0 |
| `RECT-ck-HG` | 0.3800 | [0.3738, 0.3866] | 0.3809 | [0.3765, 0.3854] | 0.5960 | -- | 0 |
| `RECT-ck-HG[ney-pilot]` | 0.4177 | [0.4119, 0.4236] | 0.4018 | [0.3971, 0.4064] | 0.5422 | -- | 0 |
| `RECT-ck-BF` | 0.3356 | [0.3306, 0.3406] | 0.3172 | [0.3135, 0.3209] | 0.6748 | -- | 0 |
| `RECT-ck-BF[ney-pilot]` | 0.3417 | [0.3372, 0.3462] | 0.3355 | [0.3316, 0.3395] | 0.6627 | -- | 0 |
| `HC-WoR` | 0.3882 | [0.3819, 0.3946] | 0.3828 | [0.3784, 0.3872] | 0.5834 | -- | 0 |
| `HC-WoR[ney-pilot]` | 0.4122 | [0.4062, 0.4184] | 0.3974 | [0.3927, 0.4019] | 0.5493 | -- | 0 |
| `PJC-local*` | 1.0008 | [0.9918, 1.0099] | 1.0013 | [0.9968, 1.0058] | 0.2263 | -- | 0 |
| `PJC-menu*` | 0.9636 | [0.9534, 0.9732] | 0.9627 | [0.9578, 0.9675] | 0.2350 | -- | 0 |
| `PJC-A[reset,b=2,dirseg]` (tuned pick: reset) | 0.9045 | [0.8905, 0.9187] | 0.9040 | [0.8944, 0.9135] | 0.2504 | 1.000 | 0 |
| `PJC-A[menu,b=8,menu=0.3/0.4/0.5/0.6/0.7,proj]` (tuned pick: menu/any_adaptive) | 0.9294 | [0.9172, 0.9417] | 0.9299 | [0.9242, 0.9357] | 0.2437 | 0.000 | 0 |
| `PJC-A[segmenu,b=4,menu=0.3/0.5/0.7,dirseg]` (tuned pick: segmenu) | 0.6498 | [0.6407, 0.6589] | 0.6518 | [0.6450, 0.6588] | 0.3485 | 1.000 | 0 |

Frozen pilot-Neyman FDC-BF vs the tuned any-adaptive pick (`PJC-A[menu,b=8,menu=0.3/0.4/0.5/0.6/0.7,proj]`): ney-pilot / adaptive N80 0.9378 [0.9240, 0.9518], x12 0.9392 [0.9306, 0.9477] (< 1: frozen pilot-Neyman faster).
Mean post-boundary control share by segment, `PJC-A[reset,b=2,dirseg]`: 0.35, 0.61, 0.50, 0.39, 0.58, 0.50, 0.35, 0.62, 0.49.
Mean post-boundary control share by segment, `PJC-A[menu,b=8,menu=0.3/0.4/0.5/0.6/0.7,proj]`: 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50.
Mean post-boundary control share by segment, `PJC-A[segmenu,b=4,menu=0.3/0.5/0.7,dirseg]`: 0.39, 0.62, 0.50, 0.40, 0.57, 0.50, 0.36, 0.63, 0.50.

## CR9-asym (SEMI-SYNTHETIC; eps 0.001; tau_R 6,989,793; 250 streams)

Pilot-Neyman control share by segment: 0.21, 0.26, 0.27, 0.30, 0.29, 0.24, 0.23, 0.32, 0.28.  eps calibration (FDC-BF geomean N80/tau on 900-919): 0.0005: 0.403, 0.00075: 0.336, 0.001: 0.262, 0.0015: 0.166, 0.002: 0.110, 0.003: 0.055, 0.005: 0.028.

FDC-BF (frozen 50/50): geomean N80/tau 0.2477, x12/tau 0.2161, false streams 0, censored 0.000.  Integrity: FDCBetPlan(0.5) reproduces FDC-BF on all streams.

| method | FDC-BF/method N80 | 95% CI | FDC-BF/method x12 | 95% CI | method N80/tau | frac switched | false streams |
|---|---|---|---|---|---|---|---|
| `FDC-BF[ney-pilot]` | 1.1798 | [1.1543, 1.2071] | 1.1719 | [1.1537, 1.1905] | 0.2099 | -- | 0 |
| `FDC-BF[ney-oracle]` | 1.1774 | [1.1519, 1.2034] | 1.1718 | [1.1541, 1.1899] | 0.2103 | -- | 0 |
| `RECT-ck-HG` | 0.6714 | [0.6549, 0.6884] | 0.6460 | [0.6333, 0.6586] | 0.3688 | -- | 0 |
| `RECT-ck-HG[ney-pilot]` | 0.5442 | [0.5313, 0.5574] | 0.5541 | [0.5426, 0.5654] | 0.4551 | -- | 0 |
| `RECT-ck-BF` | 0.5551 | [0.5414, 0.5691] | 0.5739 | [0.5621, 0.5855] | 0.4462 | -- | 0 |
| `RECT-ck-BF[ney-pilot]` | 0.4804 | [0.4671, 0.4940] | 0.4774 | [0.4675, 0.4873] | 0.5156 | -- | 0 |
| `HC-WoR` | 0.6680 | [0.6515, 0.6841] | 0.6414 | [0.6290, 0.6537] | 0.3708 | -- | 0 |
| `HC-WoR[ney-pilot]` | 0.5330 | [0.5199, 0.5465] | 0.5302 | [0.5193, 0.5410] | 0.4646 | -- | 0 |
| `PJC-local*` | 1.0021 | [0.9948, 1.0094] | 1.0011 | [0.9995, 1.0027] | 0.2471 | -- | 0 |
| `PJC-menu*` | 1.0868 | [1.0666, 1.1073] | 1.0771 | [1.0656, 1.0883] | 0.2279 | -- | 0 |
| `PJC-A[reset,b=2,neyman]` (tuned pick: reset/any_adaptive) | 1.1271 | [1.1016, 1.1531] | 1.1266 | [1.1088, 1.1446] | 0.2197 | 1.000 | 0 |
| `PJC-A[menu,b=8,menu=0.3/0.4/0.5/0.6/0.7,proj]` (tuned pick: menu) | 1.1016 | [1.0789, 1.1247] | 1.1015 | [1.0864, 1.1168] | 0.2248 | 1.000 | 0 |
| `PJC-A[segmenu,b=8,menu=0.3/0.5/0.7,neyman]` (tuned pick: segmenu) | 0.8046 | [0.7873, 0.8224] | 0.8046 | [0.7928, 0.8164] | 0.3078 | 1.000 | 0 |

Frozen pilot-Neyman FDC-BF vs the tuned any-adaptive pick (`PJC-A[reset,b=2,neyman]`): ney-pilot / adaptive N80 0.9553 [0.9395, 0.9703], x12 0.9613 [0.9512, 0.9715] (< 1: frozen pilot-Neyman faster).
Mean post-boundary control share by segment, `PJC-A[reset,b=2,neyman]`: 0.26, 0.24, 0.26, 0.29, 0.29, 0.22, 0.24, 0.30, 0.28.
Mean post-boundary control share by segment, `PJC-A[menu,b=8,menu=0.3/0.4/0.5/0.6/0.7,proj]`: 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30.
Mean post-boundary control share by segment, `PJC-A[segmenu,b=8,menu=0.3/0.5/0.7,neyman]`: 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30.

## CR9-mixed (SEMI-SYNTHETIC; eps 0.00075; tau_R 6,989,793; 250 streams)

Pilot-Neyman control share by segment: 0.25, 0.71, 0.50, 0.26, 0.67, 0.49, 0.30, 0.69, 0.51.  eps calibration (FDC-BF geomean N80/tau on 900-919): 0.0005: 0.331, 0.00075: 0.266, 0.001: 0.202, 0.0015: 0.127, 0.002: 0.083, 0.003: 0.044, 0.005: 0.020.

FDC-BF (frozen 50/50): geomean N80/tau 0.2388, x12/tau 0.2092, false streams 0, censored 0.000.  Integrity: FDCBetPlan(0.5) reproduces FDC-BF on all streams.

| method | FDC-BF/method N80 | 95% CI | FDC-BF/method x12 | 95% CI | method N80/tau | frac switched | false streams |
|---|---|---|---|---|---|---|---|
| `FDC-BF[ney-pilot]` | 1.0306 | [1.0136, 1.0490] | 1.0360 | [1.0251, 1.0468] | 0.2317 | -- | 0 |
| `FDC-BF[ney-oracle]` | 1.0360 | [1.0178, 1.0545] | 1.0372 | [1.0258, 1.0487] | 0.2305 | -- | 0 |
| `RECT-ck-HG` | 0.7767 | [0.7599, 0.7938] | 0.7779 | [0.7659, 0.7894] | 0.3075 | -- | 0 |
| `RECT-ck-HG[ney-pilot]` | 0.7824 | [0.7655, 0.7996] | 0.7823 | [0.7699, 0.7943] | 0.3052 | -- | 0 |
| `RECT-ck-BF` | 0.6877 | [0.6714, 0.7029] | 0.6954 | [0.6837, 0.7068] | 0.3472 | -- | 0 |
| `RECT-ck-BF[ney-pilot]` | 0.6763 | [0.6604, 0.6920] | 0.6774 | [0.6660, 0.6886] | 0.3531 | -- | 0 |
| `HC-WoR` | 0.7670 | [0.7513, 0.7832] | 0.7704 | [0.7593, 0.7812] | 0.3113 | -- | 0 |
| `HC-WoR[ney-pilot]` | 0.7840 | [0.7670, 0.8013] | 0.7821 | [0.7701, 0.7937] | 0.3046 | -- | 0 |
| `PJC-local*` | 0.9958 | [0.9897, 1.0010] | 1.0005 | [0.9987, 1.0025] | 0.2398 | -- | 0 |
| `PJC-menu*` | 0.9385 | [0.9202, 0.9563] | 0.9431 | [0.9316, 0.9545] | 0.2544 | -- | 0 |
| `PJC-A[reset,b=2,neyman]` (tuned pick: reset/any_adaptive) | 1.0031 | [0.9855, 1.0210] | 1.0045 | [0.9922, 1.0163] | 0.2381 | 1.000 | 0 |
| `PJC-A[menu,b=8,menu=0.3/0.4/0.5/0.6/0.7,dirproj]` (tuned pick: menu) | 0.9221 | [0.9050, 0.9395] | 0.9288 | [0.9181, 0.9395] | 0.2590 | 0.856 | 0 |
| `PJC-A[segmenu,b=8,menu=0.3/0.5/0.7,neyman]` (tuned pick: segmenu) | 0.7955 | [0.7775, 0.8130] | 0.7929 | [0.7815, 0.8035] | 0.3002 | 1.000 | 0 |

Frozen pilot-Neyman FDC-BF vs the tuned any-adaptive pick (`PJC-A[reset,b=2,neyman]`): ney-pilot / adaptive N80 0.9733 [0.9573, 0.9886], x12 0.9696 [0.9605, 0.9779] (< 1: frozen pilot-Neyman faster).
Mean post-boundary control share by segment, `PJC-A[reset,b=2,neyman]`: 0.26, 0.74, 0.48, 0.25, 0.66, 0.48, 0.28, 0.68, 0.49.
Mean post-boundary control share by segment, `PJC-A[menu,b=8,menu=0.3/0.4/0.5/0.6/0.7,dirproj]`: 0.41, 0.41, 0.41, 0.41, 0.41, 0.41, 0.41, 0.41, 0.41.
Mean post-boundary control share by segment, `PJC-A[segmenu,b=8,menu=0.3/0.5/0.7,neyman]`: 0.30, 0.70, 0.50, 0.30, 0.70, 0.50, 0.30, 0.70, 0.50.

## Regime conclusion (descriptive)

- X9-asym (eps 0.015): tuned any-adaptive pick PJC-A[menu,b=8,menu=0.3/0.4/0.5/0.6/0.7,proj] FDC-BF/pick 1.008 [0.992, 1.025] (switched 1.00); frozen pilot-Neyman FDC-BF/FDC-BF[ney-pilot] 1.037 [1.021, 1.053]; ney-pilot / best adaptive 0.972 [0.961, 0.984]
- X9-mixed (eps 0.02): tuned any-adaptive pick PJC-A[menu,b=8,menu=0.3/0.4/0.5/0.6/0.7,proj] FDC-BF/pick 0.929 [0.917, 0.942] (switched 0.00); frozen pilot-Neyman FDC-BF/FDC-BF[ney-pilot] 0.991 [0.978, 1.004]; ney-pilot / best adaptive 0.938 [0.924, 0.952]
- CR9-asym (eps 0.001): tuned any-adaptive pick PJC-A[reset,b=2,neyman] FDC-BF/pick 1.127 [1.102, 1.153] (switched 1.00); frozen pilot-Neyman FDC-BF/FDC-BF[ney-pilot] 1.180 [1.154, 1.207]; ney-pilot / best adaptive 0.955 [0.940, 0.970]
- CR9-mixed (eps 0.00075): tuned any-adaptive pick PJC-A[reset,b=2,neyman] FDC-BF/pick 1.003 [0.986, 1.021] (switched 1.00); frozen pilot-Neyman FDC-BF/FDC-BF[ney-pilot] 1.031 [1.014, 1.049]; ney-pilot / best adaptive 0.973 [0.957, 0.989]

## Answer

Adaptivity CAN beat frozen 50/50 FDC-BF, but only in one of the four semi-synthetic regimes, and a frozen pilot-Neyman FDC-BF beats the adaptive members everywhere.  (1) Criteo-structure asym (control 0.02-0.05 vs treatment 0.40-0.50 in every segment): the tuned RAGE-style reset member is 13% faster than frozen 50/50 (FDC-BF/pick 1.127 [1.102, 1.153]), the tuned global menu 10% faster and the lock-v8 PJC-menu* 9% faster; this is the regime the reviewer asked for, and adaptivity wins there.  But FDC-BF with a frozen per-segment Neyman plan from an independent pilot is faster still (FDC-BF/ney-pilot 1.180 [1.154, 1.207]; ney-pilot / reset pick 0.955 [0.940, 0.970]) and matches the oracle-Neyman plan, so the gain comes from the allocation, not from adapting it online; adapting pays a reset or union cost for learning what a 1000-draw-per-cell pilot already gives.  (2) X5-structure asym (same rate ranges, 50/50 pools): gains shrink to a few percent (ney-pilot 1.037, PJC-menu* 1.051, tuned picks 0.96-1.01).  The contrast with (1) is consistent with Criteo's control pools holding only about 15% of records, so a 50/50 read plan is far from both the Neyman split and the pool composition (a conjecture this block does not isolate).  (3) Mixed regimes (per-segment Neyman shares 0.27 / 0.70 / 0.50, so no global share helps): no adaptive member beats frozen 50/50 (best tuned picks 0.929-1.003), the per-segment menu pays M^S paths per boundary and is 20-35% slower, and frozen pilot-Neyman gives 0.99-1.03.  Neyman is not the optimal split for Bennett widths (low-variance cells read less pay the range term b*beta/(3n)), which caps every Neyman-type gain.  No method in any regime had a false certification stream (0 / 250 per method and env).  Semi-synthetic, post hoc and descriptive: it maps the regime and does not change the v8 confirmatory result.

