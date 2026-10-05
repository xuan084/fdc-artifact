# v8-posthoc-D: genuinely adaptive PJC members vs FDC-BF (POST-HOC, DESCRIPTIVE)

**Post-hoc disclosure.** Run on 2026-10-04 AFTER the v8 confirmatory analysis (lock v8 sealed: v8a_full_a, v8c_full) in response to a reviewer point that 'no gain from adaptivity' was tested where nothing adapted.  Members, grid and tuning rule were fixed before any eval row of this block was computed, and picks were tuned on DEV seeds 900-949 only (v5 rule), but the block as a whole is post hoc: its existence, members and grid were chosen knowing the confirmatory outcome.  Eval access used the documented post-hoc path dsswm/envs/posthoc_access_v8.py (logged in eval_access_log.jsonl).  The full-grid eval table is eval-selected and optimistic for the variants.

Status: POST-HOC, DESCRIPTIVE -- outside any preregistration lock (v5-v8); not confirmatory.  Ratio: FDC-BF / variant paired geomean of N80_pen (< 1: FDC-BF faster; > 1: variant faster).  Bootstrap: v6 rule: B = 10^4 resamples of streams, seed 42, two-sided percentile 95% CI on the geomean.

Families (dsswm/baselines/pjc_adapt.py; all in FDC-BF's guarantee class, FWER <= 0.05 at the K checkpoints):
- **reset** (RAGE-style phase reset, phase-local estimator, ledger = FDC-BF's): per-segment Neyman from past data (`neyman`) or per-segment direction-optimal G-design over shares 0.10-0.90 (`dirseg`).
- **menu** (keep-data, global control share from {0.3,...,0.7}; union over menu paths): projected-variance (`proj`) or direction-optimal (`dirproj`) choice.
- **segmenu** (keep-data, PER-SEGMENT share from a menu; union charges M^S paths per boundary): rounded per-segment Neyman (`neyman`) or per-segment direction-optimal (`dirseg`).
Boundaries (checkpoint indices): [2], [4], [6], [8], [3, 6], [4, 8], [2, 5, 8], [3, 6, 9] (2, 3 and 4 phases).

## Toy validity (r4 near-tie toy, 200 streams each, eps 0.02)

| member | false streams | CP 95% upper | switched streams |
|---|---|---|---|
| PJC-A[reset,b=2,5,8,neyman] | 0/200 | 0.015 | 200 |
| PJC-A[reset,b=2,5,8,dirseg] | 0/200 | 0.015 | 62 |
| PJC-A[menu,b=2,5,8,menu=0.3/0.4/0.5/0.6/0.7,proj] | 0/200 | 0.015 | 108 |
| PJC-A[menu,b=2,5,8,menu=0.3/0.4/0.5/0.6/0.7,dirproj] | 0/200 | 0.015 | 133 |
| PJC-A[segmenu,b=2,5,8,menu=0.3/0.4/0.5/0.6/0.7,neyman] | 0/200 | 0.015 | 127 |
| PJC-A[segmenu,b=2,5,8,menu=0.3/0.5/0.7,neyman] | 0/200 | 0.015 | 74 |
| PJC-A[segmenu,b=2,5,8,menu=0.3/0.4/0.5/0.6/0.7,dirseg] | 0/200 | 0.015 | 167 |
| PJC-A[segmenu,b=2,5,8,menu=0.3/0.5/0.7,dirseg] | 0/200 | 0.015 | 51 |

## X5 RetailHero X9 eval, seeds 35000-35199, eps 0.02

Sealed FDC-BF rows: v8a_full_a (seal commit 737566cb9d3c); FDC-BF geomean N80/tau = 0.2429, false streams 0.  Runner re-run reproduces the sealed FDC-BF rows: yes (0 differing streams).

### Dev-tuned picks (the descriptive answer)

| family | pick (tuned on dev 900-949) | FDC-BF/variant | 95% CI | variant N80/tau | frac switched | frac switched before k80 | false streams |
|---|---|---|---|---|---|---|---|
| reset | `PJC-A[reset,b=2,neyman]` | 0.9237 | [0.9105, 0.9371] | 0.2630 | 1.000 | 1.000 | 0 |
| menu | `PJC-A[menu,b=8,menu=0.3/0.4/0.5/0.6/0.7,proj]` | 0.9429 | [0.9304, 0.9547] | 0.2576 | 0.000 | 0.000 | 0 |
| segmenu | `PJC-A[segmenu,b=8,menu=0.3/0.5/0.7,neyman]` | 0.6858 | [0.6746, 0.6972] | 0.3543 | 0.000 | 0.000 | 0 |
| any_adaptive | `PJC-A[reset,b=2,neyman]` | 0.9237 | [0.9105, 0.9371] | 0.2630 | 1.000 | 1.000 | 0 |

Eval-selected best of all 64 grid members (optimistic for the variants): `PJC-A[menu,b=8,menu=0.3/0.4/0.5/0.6/0.7,proj]` FDC-BF/variant 0.9429 [0.9304, 0.9547], switched 0.000.  Grid members with point ratio > 1: 0; with CI lower bound > 1: 0.

<details><summary>Full grid (eval, descriptive)</summary>

| member | FDC-BF/variant | 95% CI | frac switched | false |
|---|---|---|---|---|
| `PJC-A[menu,b=8,menu=0.3/0.4/0.5/0.6/0.7,proj]` | 0.9429 | [0.9304, 0.9547] | 0.000 | 0 |
| `PJC-A[menu,b=8,menu=0.3/0.4/0.5/0.6/0.7,dirproj]` | 0.9429 | [0.9304, 0.9547] | 0.000 | 0 |
| `PJC-A[menu,b=6,menu=0.3/0.4/0.5/0.6/0.7,proj]` | 0.9362 | [0.9237, 0.9488] | 0.000 | 0 |
| `PJC-A[menu,b=6,menu=0.3/0.4/0.5/0.6/0.7,dirproj]` | 0.9362 | [0.9237, 0.9488] | 0.000 | 0 |
| `PJC-A[menu,b=4,menu=0.3/0.4/0.5/0.6/0.7,proj]` | 0.9285 | [0.9152, 0.9420] | 0.000 | 0 |
| `PJC-A[menu,b=4,menu=0.3/0.4/0.5/0.6/0.7,dirproj]` | 0.9285 | [0.9152, 0.9420] | 0.000 | 0 |
| `PJC-A[menu,b=2,menu=0.3/0.4/0.5/0.6/0.7,proj]` | 0.9247 | [0.9114, 0.9381] | 0.000 | 0 |
| `PJC-A[menu,b=2,menu=0.3/0.4/0.5/0.6/0.7,dirproj]` | 0.9247 | [0.9114, 0.9381] | 0.000 | 0 |
| `PJC-A[reset,b=2,neyman]` | 0.9237 | [0.9105, 0.9371] | 1.000 | 0 |
| `PJC-A[reset,b=2,dirseg]` | 0.9199 | [0.9058, 0.9333] | 0.050 | 0 |
| `PJC-A[reset,b=4,dirseg]` | 0.8873 | [0.8728, 0.9021] | 0.020 | 0 |
| `PJC-A[reset,b=4,neyman]` | 0.8864 | [0.8728, 0.9002] | 1.000 | 0 |
| `PJC-A[menu,b=4,8,menu=0.3/0.4/0.5/0.6/0.7,proj]` | 0.8710 | [0.8576, 0.8846] | 0.000 | 0 |
| `PJC-A[menu,b=4,8,menu=0.3/0.4/0.5/0.6/0.7,dirproj]` | 0.8710 | [0.8576, 0.8846] | 0.000 | 0 |
| `PJC-A[menu,b=3,6,menu=0.3/0.4/0.5/0.6/0.7,proj]` | 0.8656 | [0.8524, 0.8791] | 0.000 | 0 |
| `PJC-A[menu,b=3,6,menu=0.3/0.4/0.5/0.6/0.7,dirproj]` | 0.8656 | [0.8524, 0.8791] | 0.000 | 0 |
| `PJC-A[reset,b=6,dirseg]` | 0.8410 | [0.8273, 0.8550] | 0.015 | 0 |
| `PJC-A[reset,b=3,6,dirseg]` | 0.8367 | [0.8230, 0.8506] | 0.060 | 0 |
| `PJC-A[reset,b=6,neyman]` | 0.8350 | [0.8213, 0.8489] | 1.000 | 0 |
| `PJC-A[reset,b=3,6,neyman]` | 0.8350 | [0.8222, 0.8480] | 1.000 | 0 |
| `PJC-A[menu,b=2,5,8,menu=0.3/0.4/0.5/0.6/0.7,proj]` | 0.8196 | [0.8079, 0.8315] | 0.000 | 0 |
| `PJC-A[menu,b=3,6,9,menu=0.3/0.4/0.5/0.6/0.7,proj]` | 0.8196 | [0.8079, 0.8315] | 0.000 | 0 |
| `PJC-A[menu,b=2,5,8,menu=0.3/0.4/0.5/0.6/0.7,dirproj]` | 0.8196 | [0.8079, 0.8315] | 0.000 | 0 |
| `PJC-A[menu,b=3,6,9,menu=0.3/0.4/0.5/0.6/0.7,dirproj]` | 0.8196 | [0.8079, 0.8315] | 0.000 | 0 |
| `PJC-A[reset,b=2,5,8,neyman]` | 0.7825 | [0.7681, 0.7980] | 1.000 | 0 |
| `PJC-A[reset,b=4,8,neyman]` | 0.7817 | [0.7673, 0.7963] | 1.000 | 0 |
| `PJC-A[reset,b=4,8,dirseg]` | 0.7809 | [0.7657, 0.7955] | 0.030 | 0 |
| `PJC-A[reset,b=8,dirseg]` | 0.7809 | [0.7665, 0.7955] | 0.010 | 0 |
| `PJC-A[reset,b=8,neyman]` | 0.7801 | [0.7657, 0.7947] | 1.000 | 0 |
| `PJC-A[reset,b=2,5,8,dirseg]` | 0.7777 | [0.7634, 0.7922] | 0.070 | 0 |
| `PJC-A[reset,b=3,6,9,neyman]` | 0.7265 | [0.7124, 0.7409] | 1.000 | 0 |
| `PJC-A[reset,b=3,6,9,dirseg]` | 0.7213 | [0.7080, 0.7356] | 0.060 | 0 |
| `PJC-A[segmenu,b=8,menu=0.3/0.5/0.7,neyman]` | 0.6858 | [0.6746, 0.6972] | 0.000 | 0 |
| `PJC-A[segmenu,b=8,menu=0.3/0.5/0.7,dirseg]` | 0.6858 | [0.6746, 0.6972] | 0.000 | 0 |
| `PJC-A[segmenu,b=6,menu=0.3/0.5/0.7,neyman]` | 0.6788 | [0.6677, 0.6900] | 0.000 | 0 |
| `PJC-A[segmenu,b=6,menu=0.3/0.5/0.7,dirseg]` | 0.6788 | [0.6677, 0.6900] | 0.000 | 0 |
| `PJC-A[segmenu,b=4,menu=0.3/0.5/0.7,neyman]` | 0.6746 | [0.6635, 0.6858] | 0.000 | 0 |
| `PJC-A[segmenu,b=4,menu=0.3/0.5/0.7,dirseg]` | 0.6746 | [0.6635, 0.6858] | 0.000 | 0 |
| `PJC-A[segmenu,b=2,menu=0.3/0.5/0.7,neyman]` | 0.6718 | [0.6608, 0.6830] | 0.000 | 0 |
| `PJC-A[segmenu,b=2,menu=0.3/0.5/0.7,dirseg]` | 0.6718 | [0.6608, 0.6830] | 0.000 | 0 |
| `PJC-A[segmenu,b=8,menu=0.3/0.4/0.5/0.6/0.7,neyman]` | 0.5992 | [0.5882, 0.6104] | 0.000 | 0 |
| `PJC-A[segmenu,b=8,menu=0.3/0.4/0.5/0.6/0.7,dirseg]` | 0.5992 | [0.5882, 0.6104] | 0.010 | 0 |
| `PJC-A[segmenu,b=6,menu=0.3/0.4/0.5/0.6/0.7,neyman]` | 0.5955 | [0.5845, 0.6066] | 0.000 | 0 |
| `PJC-A[segmenu,b=6,menu=0.3/0.4/0.5/0.6/0.7,dirseg]` | 0.5955 | [0.5845, 0.6066] | 0.020 | 0 |
| `PJC-A[segmenu,b=4,menu=0.3/0.4/0.5/0.6/0.7,dirseg]` | 0.5924 | [0.5815, 0.6035] | 0.025 | 0 |
| `PJC-A[segmenu,b=4,menu=0.3/0.4/0.5/0.6/0.7,neyman]` | 0.5918 | [0.5809, 0.6029] | 0.000 | 0 |
| `PJC-A[segmenu,b=2,menu=0.3/0.4/0.5/0.6/0.7,neyman]` | 0.5900 | [0.5792, 0.6004] | 0.015 | 0 |
| `PJC-A[segmenu,b=2,menu=0.3/0.4/0.5/0.6/0.7,dirseg]` | 0.5900 | [0.5792, 0.6004] | 0.160 | 0 |
| `PJC-A[segmenu,b=4,8,menu=0.3/0.5/0.7,neyman]` | 0.5439 | [0.5350, 0.5529] | 0.000 | 0 |
| `PJC-A[segmenu,b=4,8,menu=0.3/0.5/0.7,dirseg]` | 0.5439 | [0.5350, 0.5529] | 0.000 | 0 |
| `PJC-A[segmenu,b=3,6,menu=0.3/0.5/0.7,neyman]` | 0.5427 | [0.5339, 0.5518] | 0.000 | 0 |
| `PJC-A[segmenu,b=3,6,menu=0.3/0.5/0.7,dirseg]` | 0.5427 | [0.5339, 0.5518] | 0.000 | 0 |
| `PJC-A[segmenu,b=4,8,menu=0.3/0.4/0.5/0.6/0.7,neyman]` | 0.4598 | [0.4518, 0.4679] | 0.000 | 0 |
| `PJC-A[segmenu,b=4,8,menu=0.3/0.4/0.5/0.6/0.7,dirseg]` | 0.4598 | [0.4518, 0.4679] | 0.035 | 0 |
| `PJC-A[segmenu,b=3,6,menu=0.3/0.4/0.5/0.6/0.7,neyman]` | 0.4584 | [0.4509, 0.4660] | 0.005 | 0 |
| `PJC-A[segmenu,b=3,6,menu=0.3/0.4/0.5/0.6/0.7,dirseg]` | 0.4584 | [0.4509, 0.4660] | 0.065 | 0 |
| `PJC-A[segmenu,b=2,5,8,menu=0.3/0.5/0.7,neyman]` | 0.4569 | [0.4495, 0.4645] | 0.000 | 0 |
| `PJC-A[segmenu,b=3,6,9,menu=0.3/0.5/0.7,neyman]` | 0.4569 | [0.4495, 0.4645] | 0.000 | 0 |
| `PJC-A[segmenu,b=2,5,8,menu=0.3/0.5/0.7,dirseg]` | 0.4569 | [0.4495, 0.4645] | 0.000 | 0 |
| `PJC-A[segmenu,b=3,6,9,menu=0.3/0.5/0.7,dirseg]` | 0.4569 | [0.4495, 0.4645] | 0.000 | 0 |
| `PJC-A[segmenu,b=2,5,8,menu=0.3/0.4/0.5/0.6/0.7,neyman]` | 0.4038 | [0.3964, 0.4114] | 0.015 | 0 |
| `PJC-A[segmenu,b=3,6,9,menu=0.3/0.4/0.5/0.6/0.7,neyman]` | 0.4038 | [0.3964, 0.4114] | 0.005 | 0 |
| `PJC-A[segmenu,b=2,5,8,menu=0.3/0.4/0.5/0.6/0.7,dirseg]` | 0.4038 | [0.3968, 0.4114] | 0.185 | 0 |
| `PJC-A[segmenu,b=3,6,9,menu=0.3/0.4/0.5/0.6/0.7,dirseg]` | 0.4030 | [0.3956, 0.4105] | 0.065 | 0 |

</details>

### Did the variants adapt?

Mean fraction of streams whose design moved off 50/50, by family: reset 0.520, menu 0.000, segmenu 0.019.  Members that switched on at least half the streams: 8; the best of them is `PJC-A[reset,b=2,neyman]` with FDC-BF/variant 0.9237 [0.9105, 0.9371] (mean max |share - 0.5| = 0.021).  False streams over all grid members: 0.

## CR9 Criteo eval, v7 streams 33000-33199, eps 0.001

Sealed FDC-BF rows: v8c_full (seal commit 339531049a2f); FDC-BF geomean N80/tau = 0.1301, false streams 0.  Runner re-run reproduces the sealed FDC-BF rows: yes (0 differing streams).

### Dev-tuned picks (the descriptive answer)

| family | pick (tuned on dev 900-949) | FDC-BF/variant | 95% CI | variant N80/tau | frac switched | frac switched before k80 | false streams |
|---|---|---|---|---|---|---|---|
| reset | `PJC-A[reset,b=2,neyman]` | 0.9506 | [0.9298, 0.9718] | 0.1368 | 1.000 | 1.000 | 0 |
| menu | `PJC-A[menu,b=8,menu=0.3/0.4/0.5/0.6/0.7,proj]` | 0.9334 | [0.9166, 0.9493] | 0.1393 | 0.000 | 0.000 | 0 |
| segmenu | `PJC-A[segmenu,b=8,menu=0.3/0.5/0.7,neyman]` | 0.6700 | [0.6537, 0.6859] | 0.1941 | 0.150 | 0.150 | 0 |
| any_adaptive | `PJC-A[menu,b=8,menu=0.3/0.4/0.5/0.6/0.7,proj]` | 0.9334 | [0.9166, 0.9493] | 0.1393 | 0.000 | 0.000 | 0 |

Eval-selected best of all 64 grid members (optimistic for the variants): `PJC-A[reset,b=2,neyman]` FDC-BF/variant 0.9506 [0.9298, 0.9718], switched 1.000.  Grid members with point ratio > 1: 0; with CI lower bound > 1: 0.

<details><summary>Full grid (eval, descriptive)</summary>

| member | FDC-BF/variant | 95% CI | frac switched | false |
|---|---|---|---|---|
| `PJC-A[reset,b=2,neyman]` | 0.9506 | [0.9298, 0.9718] | 1.000 | 0 |
| `PJC-A[reset,b=2,dirseg]` | 0.9432 | [0.9226, 0.9643] | 0.770 | 0 |
| `PJC-A[menu,b=8,menu=0.3/0.4/0.5/0.6/0.7,proj]` | 0.9334 | [0.9166, 0.9493] | 0.000 | 0 |
| `PJC-A[menu,b=8,menu=0.3/0.4/0.5/0.6/0.7,dirproj]` | 0.9334 | [0.9166, 0.9493] | 0.000 | 0 |
| `PJC-A[menu,b=6,menu=0.3/0.4/0.5/0.6/0.7,proj]` | 0.9298 | [0.9130, 0.9456] | 0.000 | 0 |
| `PJC-A[menu,b=6,menu=0.3/0.4/0.5/0.6/0.7,dirproj]` | 0.9298 | [0.9130, 0.9456] | 0.000 | 0 |
| `PJC-A[menu,b=4,menu=0.3/0.4/0.5/0.6/0.7,proj]` | 0.9238 | [0.9059, 0.9407] | 0.000 | 0 |
| `PJC-A[menu,b=4,menu=0.3/0.4/0.5/0.6/0.7,dirproj]` | 0.9238 | [0.9059, 0.9407] | 0.000 | 0 |
| `PJC-A[menu,b=2,menu=0.3/0.4/0.5/0.6/0.7,proj]` | 0.9214 | [0.9036, 0.9383] | 0.010 | 0 |
| `PJC-A[menu,b=2,menu=0.3/0.4/0.5/0.6/0.7,dirproj]` | 0.9214 | [0.9036, 0.9383] | 0.005 | 0 |
| `PJC-A[reset,b=4,neyman]` | 0.9071 | [0.8861, 0.9286] | 1.000 | 0 |
| `PJC-A[reset,b=4,dirseg]` | 0.9024 | [0.8815, 0.9238] | 0.795 | 0 |
| `PJC-A[menu,b=4,8,menu=0.3/0.4/0.5/0.6/0.7,proj]` | 0.8770 | [0.8578, 0.8954] | 0.000 | 0 |
| `PJC-A[menu,b=4,8,menu=0.3/0.4/0.5/0.6/0.7,dirproj]` | 0.8770 | [0.8578, 0.8954] | 0.000 | 0 |
| `PJC-A[menu,b=3,6,menu=0.3/0.4/0.5/0.6/0.7,proj]` | 0.8747 | [0.8556, 0.8931] | 0.005 | 0 |
| `PJC-A[menu,b=3,6,menu=0.3/0.4/0.5/0.6/0.7,dirproj]` | 0.8747 | [0.8556, 0.8931] | 0.000 | 0 |
| `PJC-A[reset,b=6,neyman]` | 0.8522 | [0.8293, 0.8758] | 1.000 | 0 |
| `PJC-A[reset,b=3,6,dirseg]` | 0.8478 | [0.8250, 0.8701] | 0.990 | 0 |
| `PJC-A[reset,b=3,6,neyman]` | 0.8456 | [0.8239, 0.8690] | 1.000 | 0 |
| `PJC-A[reset,b=6,dirseg]` | 0.8412 | [0.8186, 0.8645] | 0.915 | 0 |
| `PJC-A[menu,b=3,6,9,menu=0.3/0.4/0.5/0.6/0.7,proj]` | 0.8143 | [0.7955, 0.8336] | 0.005 | 0 |
| `PJC-A[menu,b=3,6,9,menu=0.3/0.4/0.5/0.6/0.7,dirproj]` | 0.8143 | [0.7955, 0.8336] | 0.000 | 0 |
| `PJC-A[menu,b=2,5,8,menu=0.3/0.4/0.5/0.6/0.7,dirproj]` | 0.8080 | [0.7893, 0.8260] | 0.005 | 0 |
| `PJC-A[menu,b=2,5,8,menu=0.3/0.4/0.5/0.6/0.7,proj]` | 0.8059 | [0.7873, 0.8250] | 0.010 | 0 |
| `PJC-A[reset,b=8,dirseg]` | 0.7522 | [0.7291, 0.7751] | 1.000 | 0 |
| `PJC-A[reset,b=8,neyman]` | 0.7493 | [0.7263, 0.7720] | 1.000 | 0 |
| `PJC-A[reset,b=4,8,dirseg]` | 0.7444 | [0.7206, 0.7690] | 1.000 | 0 |
| `PJC-A[reset,b=4,8,neyman]` | 0.7425 | [0.7178, 0.7670] | 1.000 | 0 |
| `PJC-A[reset,b=2,5,8,neyman]` | 0.7425 | [0.7188, 0.7670] | 1.000 | 0 |
| `PJC-A[reset,b=2,5,8,dirseg]` | 0.7425 | [0.7188, 0.7660] | 1.000 | 0 |
| `PJC-A[reset,b=3,6,9,neyman]` | 0.6868 | [0.6648, 0.7086] | 1.000 | 0 |
| `PJC-A[reset,b=3,6,9,dirseg]` | 0.6823 | [0.6614, 0.7040] | 1.000 | 0 |
| `PJC-A[segmenu,b=8,menu=0.3/0.5/0.7,neyman]` | 0.6700 | [0.6537, 0.6859] | 0.150 | 0 |
| `PJC-A[segmenu,b=8,menu=0.3/0.5/0.7,dirseg]` | 0.6700 | [0.6537, 0.6859] | 0.365 | 0 |
| `PJC-A[segmenu,b=4,menu=0.3/0.5/0.7,dirseg]` | 0.6683 | [0.6511, 0.6850] | 0.450 | 0 |
| `PJC-A[segmenu,b=4,menu=0.3/0.5/0.7,neyman]` | 0.6674 | [0.6503, 0.6841] | 0.335 | 0 |
| `PJC-A[segmenu,b=6,menu=0.3/0.5/0.7,neyman]` | 0.6674 | [0.6503, 0.6841] | 0.240 | 0 |
| `PJC-A[segmenu,b=2,menu=0.3/0.5/0.7,neyman]` | 0.6666 | [0.6495, 0.6832] | 0.390 | 0 |
| `PJC-A[segmenu,b=2,menu=0.3/0.5/0.7,dirseg]` | 0.6666 | [0.6495, 0.6832] | 0.435 | 0 |
| `PJC-A[segmenu,b=6,menu=0.3/0.5/0.7,dirseg]` | 0.6666 | [0.6495, 0.6832] | 0.455 | 0 |
| `PJC-A[segmenu,b=8,menu=0.3/0.4/0.5/0.6/0.7,neyman]` | 0.5930 | [0.5762, 0.6102] | 1.000 | 0 |
| `PJC-A[segmenu,b=8,menu=0.3/0.4/0.5/0.6/0.7,dirseg]` | 0.5914 | [0.5747, 0.6086] | 0.625 | 0 |
| `PJC-A[segmenu,b=6,menu=0.3/0.4/0.5/0.6/0.7,neyman]` | 0.5907 | [0.5740, 0.6078] | 0.995 | 0 |
| `PJC-A[segmenu,b=4,menu=0.3/0.4/0.5/0.6/0.7,neyman]` | 0.5891 | [0.5725, 0.6062] | 0.975 | 0 |
| `PJC-A[segmenu,b=6,menu=0.3/0.4/0.5/0.6/0.7,dirseg]` | 0.5891 | [0.5725, 0.6062] | 0.665 | 0 |
| `PJC-A[segmenu,b=2,menu=0.3/0.4/0.5/0.6/0.7,neyman]` | 0.5884 | [0.5710, 0.6054] | 0.950 | 0 |
| `PJC-A[segmenu,b=2,menu=0.3/0.4/0.5/0.6/0.7,dirseg]` | 0.5861 | [0.5695, 0.6031] | 0.710 | 0 |
| `PJC-A[segmenu,b=4,menu=0.3/0.4/0.5/0.6/0.7,dirseg]` | 0.5861 | [0.5695, 0.6031] | 0.680 | 0 |
| `PJC-A[segmenu,b=3,6,menu=0.3/0.5/0.7,neyman]` | 0.5180 | [0.5040, 0.5323] | 0.435 | 0 |
| `PJC-A[segmenu,b=4,8,menu=0.3/0.5/0.7,neyman]` | 0.5180 | [0.5040, 0.5323] | 0.385 | 0 |
| `PJC-A[segmenu,b=4,8,menu=0.3/0.5/0.7,dirseg]` | 0.5180 | [0.5040, 0.5323] | 0.610 | 0 |
| `PJC-A[segmenu,b=3,6,menu=0.3/0.5/0.7,dirseg]` | 0.5173 | [0.5034, 0.5316] | 0.655 | 0 |
| `PJC-A[segmenu,b=4,8,menu=0.3/0.4/0.5/0.6/0.7,dirseg]` | 0.4692 | [0.4554, 0.4835] | 0.840 | 0 |
| `PJC-A[segmenu,b=4,8,menu=0.3/0.4/0.5/0.6/0.7,neyman]` | 0.4686 | [0.4548, 0.4829] | 1.000 | 0 |
| `PJC-A[segmenu,b=3,6,menu=0.3/0.4/0.5/0.6/0.7,neyman]` | 0.4674 | [0.4536, 0.4816] | 0.995 | 0 |
| `PJC-A[segmenu,b=3,6,menu=0.3/0.4/0.5/0.6/0.7,dirseg]` | 0.4650 | [0.4507, 0.4791] | 0.875 | 0 |
| `PJC-A[segmenu,b=3,6,9,menu=0.3/0.5/0.7,dirseg]` | 0.4608 | [0.4466, 0.4748] | 0.775 | 0 |
| `PJC-A[segmenu,b=3,6,9,menu=0.3/0.5/0.7,neyman]` | 0.4602 | [0.4460, 0.4742] | 0.475 | 0 |
| `PJC-A[segmenu,b=2,5,8,menu=0.3/0.5/0.7,neyman]` | 0.4596 | [0.4455, 0.4735] | 0.500 | 0 |
| `PJC-A[segmenu,b=2,5,8,menu=0.3/0.5/0.7,dirseg]` | 0.4596 | [0.4455, 0.4735] | 0.730 | 0 |
| `PJC-A[segmenu,b=2,5,8,menu=0.3/0.4/0.5/0.6/0.7,neyman]` | 0.3831 | [0.3718, 0.3947] | 1.000 | 0 |
| `PJC-A[segmenu,b=2,5,8,menu=0.3/0.4/0.5/0.6/0.7,dirseg]` | 0.3831 | [0.3718, 0.3947] | 0.935 | 0 |
| `PJC-A[segmenu,b=3,6,9,menu=0.3/0.4/0.5/0.6/0.7,dirseg]` | 0.3831 | [0.3718, 0.3947] | 0.965 | 0 |
| `PJC-A[segmenu,b=3,6,9,menu=0.3/0.4/0.5/0.6/0.7,neyman]` | 0.3826 | [0.3713, 0.3942] | 1.000 | 0 |

</details>

### Did the variants adapt?

Mean fraction of streams whose design moved off 50/50, by family: reset 0.967, menu 0.003, segmenu 0.675.  Members that switched on at least half the streams: 37; the best of them is `PJC-A[reset,b=2,neyman]` with FDC-BF/variant 0.9506 [0.9298, 0.9718] (mean max |share - 0.5| = 0.107).  False streams over all grid members: 0.

## Answer

No adaptive variant beats FDC-BF: every dev-tuned pick and every one of the 64 grid members has FDC-BF/variant < 1 with the 95% CI upper bound below 1 on both layers.  Two reasons, both visible in the table.  (i) When adaptation is free (keep-data, global menu), the data-chosen share is 0.5 on essentially every stream, because the two arms of a segment have nearly equal Bernoulli variances, so 50/50 is already the variance-optimal within-segment split; the menu cost (a larger union) is then paid for nothing.  (ii) Members that do move the design pay for it: the RAGE-style reset discards earlier-phase samples, and the per-segment keep-data menu charges M^S design paths per boundary (log L grows by S ln M per boundary), which costs more than the re-allocation gains.  Post hoc and descriptive: it does not change the v8 confirmatory result.
