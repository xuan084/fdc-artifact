# v8-posthoc-F2: pilot versus adaptive at equal information and charged pilot cost (POST-HOC, DESCRIPTIVE, SEMI-SYNTHETIC)

**Post-hoc, semi-synthetic disclosure.** Run on 2026-10-05 in response to external reviewer paper_r4_review S1 / fix 2, after block F's results were known. Environments, eps, reporting seeds (950-999 + 36000-36199) and the adaptive picks are block F's (picks tuned on 900-949 without a pilot and NOT re-tuned with one); pilot sizes {0, 250, 1000}, the pilot construction, the 'init' / 'init+warm' rules and the amortisation grid J in {1, 4, 16, inf} were fixed before any F2 row was computed.  No real outcome and no evaluation half enters any statistic.

Status: POST-HOC, DESCRIPTIVE, SEMI-SYNTHETIC -- outside any preregistration lock (v5-v8); not confirmatory; outcomes are synthetic.

- Pilot: independent with-replacement draws from the same finite pools (Bernoulli(exact pool mean) per cell), nested n in {250, 1000} per cell, drawn per stream (seeded by env and stream seed) and shared by every method on that stream; per-instance cost S*A*n = 4,500 / 18,000 reads.  A WoR prefix of the replay was NOT used: a plan computed from replay reads is not independent of the replay permutation, so the frozen-design theorem would not cover FDC-BF's full-pool estimator (valid only as a charged phase with a phase-local estimator, i.e. the PJC-A reset construction).
- Charging: N80_charged = N80_pen + S*A*n / J; J = certification instances sharing one pilot (J = inf: uncharged, block F's convention).  Same for x12.
- Equal information: adaptive picks get the same per-stream pilot: 'init' = phase-0 plan is the pilot per-segment Neyman plan (global menu re-centred on it: share logistic(logit p_s + logit c)); 'init+warm' also pools pilot counts into every design-rule plug-in variance.  Certificates use replay data only; ledgers unchanged.
- Phase-wise TU joint rival: not implemented (not cheap); adaptive variants remain checkpoint-strength.
- Ratios: A / B paired geomean over 250 streams; for 'FDC-BF / method' > 1 means the method is faster than frozen-50/50 FDC-BF.
- Bootstrap: v6 rule: B = 10^4 resamples of streams, seed 42, two-sided percentile 95% CI.
- Environment names follow block F: X9-* = X5-structure, CR9-* = Criteo-structure.

## X9-asym (SEMI-SYNTHETIC; eps 0.015; tau_R 99,646; 250 streams)

Adaptive picks (block F, tuned without a pilot): reset `PJC-A[reset,b=2,neyman]`, menu `PJC-A[menu,b=8,menu=0.3/0.4/0.5/0.6/0.7,proj]`.  Reproduction of F's pilot-0 rows on seeds 950-959: 30/30 identical.

FDC-BF (frozen 50/50, no pilot) / method, paired geomean N80 [95% CI]; method charged pilot/J:

| method | pilot reads (/tau) | J=1 (full charge) | J=4 | J=16 | uncharged | break-even J | frac switched | false streams |
|---|---|---|---|---|---|---|---|---|
| `FDC-BF[ney-pilot,F fixed env pilot n=1000]` | 18,000 (0.181) | 0.633 [0.624, 0.642] | 0.894 [0.881, 0.908] | 0.997 [0.982, 1.013] | 1.037 [1.021, 1.053] | 17.33 | -- | 0 |
| `PJC-A-reset[no pilot]` | 0 | 0.962 [0.947, 0.977] | 0.962 [0.947, 0.977] | 0.962 [0.947, 0.977] | 0.962 [0.947, 0.977] | -- | 1.00 | 0 |
| `PJC-A-menu[no pilot]` | 0 | 1.008 [0.992, 1.025] | 1.008 [0.992, 1.025] | 1.008 [0.992, 1.025] | 1.008 [0.992, 1.025] | -- | 1.00 | 0 |
| `FDC-BF[ney-pilot,n=250]` | 4,500 (0.045) | 0.893 [0.879, 0.907] | 0.995 [0.980, 1.012] | 1.025 [1.008, 1.042] | 1.035 [1.018, 1.052] | 4.54 | -- | 0 |
| `PJC-A-reset[pilot n=250,init]` | 4,500 (0.045) | 0.843 [0.830, 0.855] | 0.934 [0.919, 0.947] | 0.959 [0.945, 0.974] | 0.968 [0.953, 0.983] | never | 1.00 | 0 |
| `PJC-A-reset[pilot n=250,init+warm]` | 4,500 (0.045) | 0.840 [0.827, 0.853] | 0.930 [0.915, 0.945] | 0.956 [0.940, 0.971] | 0.964 [0.949, 0.980] | never | 1.00 | 0 |
| `PJC-A-menu[pilot n=250,init]` | 4,500 (0.045) | 0.836 [0.822, 0.850] | 0.925 [0.909, 0.942] | 0.951 [0.934, 0.968] | 0.960 [0.942, 0.977] | never | 0.24 | 0 |
| `PJC-A-menu[pilot n=250,init+warm]` | 4,500 (0.045) | 0.828 [0.815, 0.842] | 0.916 [0.900, 0.932] | 0.941 [0.925, 0.957] | 0.949 [0.933, 0.966] | never | 0.09 | 0 |
| `FDC-BF[ney-pilot,n=1000]` | 18,000 (0.181) | 0.632 [0.623, 0.641] | 0.892 [0.878, 0.905] | 0.994 [0.978, 1.010] | 1.033 [1.017, 1.051] | 19.04 | -- | 0 |
| `PJC-A-reset[pilot n=1000,init]` | 18,000 (0.181) | 0.608 [0.599, 0.616] | 0.845 [0.832, 0.857] | 0.936 [0.921, 0.951] | 0.971 [0.956, 0.986] | never | 1.00 | 0 |
| `PJC-A-reset[pilot n=1000,init+warm]` | 18,000 (0.181) | 0.605 [0.597, 0.614] | 0.840 [0.827, 0.852] | 0.930 [0.916, 0.944] | 0.964 [0.949, 0.980] | never | 1.00 | 0 |
| `PJC-A-menu[pilot n=1000,init]` | 18,000 (0.181) | 0.603 [0.595, 0.611] | 0.835 [0.823, 0.847] | 0.924 [0.910, 0.938] | 0.958 [0.943, 0.973] | never | 0.11 | 0 |
| `PJC-A-menu[pilot n=1000,init+warm]` | 18,000 (0.181) | 0.601 [0.593, 0.609] | 0.832 [0.820, 0.844] | 0.920 [0.907, 0.934] | 0.954 [0.940, 0.968] | never | 0.00 | 0 |

Head-to-head, frozen pilot-Neyman FDC-BF / adaptive (< 1: frozen pilot-Neyman faster).  'Equal information' rows charge both sides the same pilot; 'vs no pilot' rows charge only the frozen side:

| comparison | J=1 | J=4 | J=16 | uncharged | break-even J |
|---|---|---|---|---|---|
| `FDC-BF[ney-pilot,n=250] / PJC-A-reset[pilot n=250,init]` | 0.944 [0.932, 0.955] | 0.938 [0.925, 0.951] | 0.936 [0.923, 0.949] | 0.936 [0.922, 0.949] | same charge |
| `FDC-BF[ney-pilot,n=250] / PJC-A-reset[pilot n=250,init+warm]` | 0.941 [0.930, 0.951] | 0.934 [0.922, 0.946] | 0.932 [0.920, 0.945] | 0.932 [0.919, 0.944] | same charge |
| `FDC-BF[ney-pilot,n=250] / PJC-A-reset[no pilot]` | 1.078 [1.063, 1.092] | 0.966 [0.952, 0.980] | 0.939 [0.925, 0.953] | 0.929 [0.916, 0.943] | 2.10 |
| `FDC-BF[ney-pilot,n=250] / PJC-A-menu[pilot n=250,init]` | 0.936 [0.925, 0.948] | 0.930 [0.917, 0.942] | 0.928 [0.914, 0.941] | 0.927 [0.913, 0.940] | same charge |
| `FDC-BF[ney-pilot,n=250] / PJC-A-menu[pilot n=250,init+warm]` | 0.928 [0.917, 0.938] | 0.920 [0.908, 0.932] | 0.918 [0.906, 0.930] | 0.917 [0.905, 0.929] | same charge |
| `FDC-BF[ney-pilot,n=250] / PJC-A-menu[no pilot]` | 1.129 [1.116, 1.143] | 1.013 [1.000, 1.026] | 0.984 [0.971, 0.997] | 0.974 [0.961, 0.987] | 5.98 |
| `FDC-BF[ney-pilot,n=1000] / PJC-A-reset[pilot n=1000,init]` | 0.962 [0.954, 0.970] | 0.947 [0.937, 0.958] | 0.942 [0.930, 0.954] | 0.939 [0.927, 0.952] | same charge |
| `FDC-BF[ney-pilot,n=1000] / PJC-A-reset[pilot n=1000,init+warm]` | 0.958 [0.950, 0.966] | 0.942 [0.931, 0.953] | 0.936 [0.923, 0.948] | 0.933 [0.920, 0.946] | same charge |
| `FDC-BF[ney-pilot,n=1000] / PJC-A-reset[no pilot]` | 1.523 [1.504, 1.542] | 1.079 [1.064, 1.094] | 0.968 [0.954, 0.982] | 0.931 [0.917, 0.945] | 8.58 |
| `FDC-BF[ney-pilot,n=1000] / PJC-A-menu[pilot n=1000,init]` | 0.954 [0.946, 0.962] | 0.936 [0.926, 0.947] | 0.930 [0.918, 0.942] | 0.927 [0.915, 0.939] | same charge |
| `FDC-BF[ney-pilot,n=1000] / PJC-A-menu[pilot n=1000,init+warm]` | 0.951 [0.944, 0.959] | 0.933 [0.923, 0.943] | 0.926 [0.915, 0.937] | 0.923 [0.912, 0.935] | same charge |
| `FDC-BF[ney-pilot,n=1000] / PJC-A-menu[no pilot]` | 1.596 [1.579, 1.613] | 1.131 [1.118, 1.143] | 1.014 [1.003, 1.026] | 0.976 [0.964, 0.987] | 25.50 |

## X9-mixed (SEMI-SYNTHETIC; eps 0.02; tau_R 99,646; 250 streams)

Adaptive picks (block F, tuned without a pilot): reset `PJC-A[reset,b=2,dirseg]`, menu `PJC-A[menu,b=8,menu=0.3/0.4/0.5/0.6/0.7,proj]`.  Reproduction of F's pilot-0 rows on seeds 950-959: 30/30 identical.

FDC-BF (frozen 50/50, no pilot) / method, paired geomean N80 [95% CI]; method charged pilot/J:

| method | pilot reads (/tau) | J=1 (full charge) | J=4 | J=16 | uncharged | break-even J | frac switched | false streams |
|---|---|---|---|---|---|---|---|---|
| `FDC-BF[ney-pilot,F fixed env pilot n=1000]` | 18,000 (0.181) | 0.553 [0.546, 0.559] | 0.827 [0.817, 0.837] | 0.944 [0.932, 0.956] | 0.991 [0.978, 1.004] | never | -- | 0 |
| `PJC-A-reset[no pilot]` | 0 | 0.904 [0.890, 0.919] | 0.904 [0.890, 0.919] | 0.904 [0.890, 0.919] | 0.904 [0.890, 0.919] | -- | 1.00 | 0 |
| `PJC-A-menu[no pilot]` | 0 | 0.929 [0.917, 0.942] | 0.929 [0.917, 0.942] | 0.929 [0.917, 0.942] | 0.929 [0.917, 0.942] | -- | 0.00 | 0 |
| `FDC-BF[ney-pilot,n=250]` | 4,500 (0.045) | 0.826 [0.816, 0.836] | 0.943 [0.931, 0.955] | 0.978 [0.965, 0.991] | 0.990 [0.977, 1.003] | never | -- | 0 |
| `PJC-A-reset[pilot n=250,init]` | 4,500 (0.045) | 0.762 [0.751, 0.774] | 0.861 [0.847, 0.875] | 0.890 [0.876, 0.905] | 0.900 [0.885, 0.915] | never | 1.00 | 0 |
| `PJC-A-reset[pilot n=250,init+warm]` | 4,500 (0.045) | 0.765 [0.753, 0.776] | 0.864 [0.851, 0.877] | 0.893 [0.879, 0.907] | 0.903 [0.889, 0.917] | never | 1.00 | 0 |
| `PJC-A-menu[pilot n=250,init]` | 4,500 (0.045) | 0.785 [0.774, 0.796] | 0.889 [0.876, 0.902] | 0.920 [0.907, 0.934] | 0.931 [0.917, 0.945] | never | 0.02 | 0 |
| `PJC-A-menu[pilot n=250,init+warm]` | 4,500 (0.045) | 0.785 [0.774, 0.795] | 0.889 [0.876, 0.902] | 0.920 [0.907, 0.934] | 0.931 [0.917, 0.945] | never | 0.00 | 0 |
| `FDC-BF[ney-pilot,n=1000]` | 18,000 (0.181) | 0.555 [0.549, 0.562] | 0.832 [0.822, 0.842] | 0.951 [0.938, 0.963] | 0.998 [0.985, 1.012] | never | -- | 0 |
| `PJC-A-reset[pilot n=1000,init]` | 18,000 (0.181) | 0.524 [0.517, 0.531] | 0.764 [0.753, 0.776] | 0.863 [0.849, 0.877] | 0.902 [0.888, 0.917] | never | 1.00 | 0 |
| `PJC-A-reset[pilot n=1000,init+warm]` | 18,000 (0.181) | 0.524 [0.516, 0.531] | 0.764 [0.752, 0.776] | 0.863 [0.848, 0.877] | 0.902 [0.886, 0.917] | never | 1.00 | 0 |
| `PJC-A-menu[pilot n=1000,init]` | 18,000 (0.181) | 0.529 [0.523, 0.536] | 0.776 [0.765, 0.787] | 0.878 [0.865, 0.892] | 0.919 [0.905, 0.932] | never | 0.00 | 0 |
| `PJC-A-menu[pilot n=1000,init+warm]` | 18,000 (0.181) | 0.530 [0.523, 0.537] | 0.776 [0.766, 0.788] | 0.879 [0.866, 0.892] | 0.919 [0.905, 0.933] | never | 0.00 | 0 |

Head-to-head, frozen pilot-Neyman FDC-BF / adaptive (< 1: frozen pilot-Neyman faster).  'Equal information' rows charge both sides the same pilot; 'vs no pilot' rows charge only the frozen side:

| comparison | J=1 | J=4 | J=16 | uncharged | break-even J |
|---|---|---|---|---|---|
| `FDC-BF[ney-pilot,n=250] / PJC-A-reset[pilot n=250,init]` | 0.923 [0.911, 0.934] | 0.913 [0.900, 0.926] | 0.910 [0.897, 0.923] | 0.909 [0.896, 0.923] | same charge |
| `FDC-BF[ney-pilot,n=250] / PJC-A-reset[pilot n=250,init+warm]` | 0.925 [0.914, 0.937] | 0.916 [0.904, 0.929] | 0.913 [0.900, 0.926] | 0.912 [0.899, 0.926] | same charge |
| `FDC-BF[ney-pilot,n=250] / PJC-A-reset[no pilot]` | 1.095 [1.080, 1.109] | 0.959 [0.946, 0.972] | 0.925 [0.912, 0.938] | 0.913 [0.901, 0.926] | 2.09 |
| `FDC-BF[ney-pilot,n=250] / PJC-A-menu[pilot n=250,init]` | 0.949 [0.940, 0.959] | 0.943 [0.932, 0.953] | 0.941 [0.929, 0.952] | 0.940 [0.929, 0.951] | same charge |
| `FDC-BF[ney-pilot,n=250] / PJC-A-menu[pilot n=250,init+warm]` | 0.949 [0.940, 0.959] | 0.943 [0.932, 0.953] | 0.941 [0.930, 0.952] | 0.940 [0.929, 0.951] | same charge |
| `FDC-BF[ney-pilot,n=250] / PJC-A-menu[no pilot]` | 1.125 [1.110, 1.140] | 0.985 [0.971, 0.999] | 0.950 [0.937, 0.964] | 0.939 [0.926, 0.952] | 3.03 |
| `FDC-BF[ney-pilot,n=1000] / PJC-A-reset[pilot n=1000,init]` | 0.944 [0.936, 0.952] | 0.918 [0.907, 0.930] | 0.908 [0.895, 0.921] | 0.904 [0.890, 0.917] | same charge |
| `FDC-BF[ney-pilot,n=1000] / PJC-A-reset[pilot n=1000,init+warm]` | 0.943 [0.935, 0.951] | 0.918 [0.906, 0.929] | 0.907 [0.894, 0.920] | 0.903 [0.890, 0.916] | same charge |
| `FDC-BF[ney-pilot,n=1000] / PJC-A-reset[no pilot]` | 1.629 [1.609, 1.650] | 1.087 [1.072, 1.103] | 0.951 [0.938, 0.966] | 0.906 [0.893, 0.919] | 7.71 |
| `FDC-BF[ney-pilot,n=1000] / PJC-A-menu[pilot n=1000,init]` | 0.954 [0.947, 0.961] | 0.933 [0.922, 0.943] | 0.924 [0.912, 0.935] | 0.920 [0.908, 0.932] | same charge |
| `FDC-BF[ney-pilot,n=1000] / PJC-A-menu[pilot n=1000,init+warm]` | 0.954 [0.947, 0.961] | 0.933 [0.923, 0.944] | 0.924 [0.913, 0.936] | 0.921 [0.909, 0.933] | same charge |
| `FDC-BF[ney-pilot,n=1000] / PJC-A-menu[no pilot]` | 1.674 [1.653, 1.696] | 1.117 [1.102, 1.133] | 0.977 [0.963, 0.992] | 0.931 [0.917, 0.945] | 10.78 |

## CR9-asym (SEMI-SYNTHETIC; eps 0.001; tau_R 6,989,793; 250 streams)

Adaptive picks (block F, tuned without a pilot): reset `PJC-A[reset,b=2,neyman]`, menu `PJC-A[menu,b=8,menu=0.3/0.4/0.5/0.6/0.7,proj]`.  Reproduction of F's pilot-0 rows on seeds 950-959: 30/30 identical.

FDC-BF (frozen 50/50, no pilot) / method, paired geomean N80 [95% CI]; method charged pilot/J:

| method | pilot reads (/tau) | J=1 (full charge) | J=4 | J=16 | uncharged | break-even J | frac switched | false streams |
|---|---|---|---|---|---|---|---|---|
| `FDC-BF[ney-pilot,F fixed env pilot n=1000]` | 18,000 (0.003) | 1.165 [1.140, 1.192] | 1.176 [1.151, 1.203] | 1.179 [1.153, 1.206] | 1.180 [1.154, 1.207] | 1.00 | -- | 0 |
| `PJC-A-reset[no pilot]` | 0 | 1.127 [1.102, 1.153] | 1.127 [1.102, 1.153] | 1.127 [1.102, 1.153] | 1.127 [1.102, 1.153] | -- | 1.00 | 0 |
| `PJC-A-menu[no pilot]` | 0 | 1.102 [1.079, 1.125] | 1.102 [1.079, 1.125] | 1.102 [1.079, 1.125] | 1.102 [1.079, 1.125] | -- | 1.00 | 0 |
| `FDC-BF[ney-pilot,n=250]` | 4,500 (0.001) | 1.156 [1.133, 1.180] | 1.158 [1.136, 1.183] | 1.159 [1.136, 1.183] | 1.159 [1.136, 1.184] | 1.00 | -- | 0 |
| `PJC-A-reset[pilot n=250,init]` | 4,500 (0.001) | 1.121 [1.096, 1.147] | 1.124 [1.098, 1.150] | 1.125 [1.099, 1.151] | 1.125 [1.099, 1.151] | 1.00 | 1.00 | 0 |
| `PJC-A-reset[pilot n=250,init+warm]` | 4,500 (0.001) | 1.123 [1.097, 1.148] | 1.125 [1.100, 1.151] | 1.126 [1.100, 1.152] | 1.126 [1.100, 1.152] | 1.00 | 1.00 | 0 |
| `PJC-A-menu[pilot n=250,init]` | 4,500 (0.001) | 1.090 [1.068, 1.113] | 1.093 [1.070, 1.116] | 1.093 [1.071, 1.116] | 1.094 [1.071, 1.117] | 1.00 | 0.19 | 0 |
| `PJC-A-menu[pilot n=250,init+warm]` | 4,500 (0.001) | 1.089 [1.067, 1.112] | 1.092 [1.069, 1.115] | 1.092 [1.070, 1.115] | 1.092 [1.070, 1.115] | 1.00 | 0.19 | 0 |
| `FDC-BF[ney-pilot,n=1000]` | 18,000 (0.003) | 1.156 [1.131, 1.181] | 1.166 [1.141, 1.192] | 1.169 [1.144, 1.195] | 1.170 [1.145, 1.196] | 1.00 | -- | 0 |
| `PJC-A-reset[pilot n=1000,init]` | 18,000 (0.003) | 1.112 [1.087, 1.137] | 1.121 [1.096, 1.147] | 1.124 [1.098, 1.150] | 1.125 [1.099, 1.151] | 1.00 | 1.00 | 0 |
| `PJC-A-reset[pilot n=1000,init+warm]` | 18,000 (0.003) | 1.111 [1.086, 1.137] | 1.121 [1.096, 1.147] | 1.124 [1.098, 1.150] | 1.125 [1.099, 1.151] | 1.00 | 1.00 | 0 |
| `PJC-A-menu[pilot n=1000,init]` | 18,000 (0.003) | 1.083 [1.060, 1.107] | 1.093 [1.069, 1.117] | 1.095 [1.071, 1.119] | 1.096 [1.072, 1.120] | 1.00 | 0.02 | 0 |
| `PJC-A-menu[pilot n=1000,init+warm]` | 18,000 (0.003) | 1.084 [1.061, 1.108] | 1.094 [1.070, 1.118] | 1.096 [1.073, 1.120] | 1.097 [1.073, 1.121] | 1.00 | 0.01 | 0 |

Head-to-head, frozen pilot-Neyman FDC-BF / adaptive (< 1: frozen pilot-Neyman faster).  'Equal information' rows charge both sides the same pilot; 'vs no pilot' rows charge only the frozen side:

| comparison | J=1 | J=4 | J=16 | uncharged | break-even J |
|---|---|---|---|---|---|
| `FDC-BF[ney-pilot,n=250] / PJC-A-reset[pilot n=250,init]` | 0.970 [0.953, 0.988] | 0.970 [0.953, 0.988] | 0.970 [0.953, 0.988] | 0.970 [0.953, 0.988] | same charge |
| `FDC-BF[ney-pilot,n=250] / PJC-A-reset[pilot n=250,init+warm]` | 0.971 [0.954, 0.989] | 0.971 [0.954, 0.989] | 0.971 [0.954, 0.989] | 0.971 [0.954, 0.989] | same charge |
| `FDC-BF[ney-pilot,n=250] / PJC-A-reset[no pilot]` | 0.975 [0.958, 0.993] | 0.973 [0.956, 0.990] | 0.972 [0.955, 0.990] | 0.972 [0.955, 0.990] | 1.00 |
| `FDC-BF[ney-pilot,n=250] / PJC-A-menu[pilot n=250,init]` | 0.944 [0.930, 0.956] | 0.943 [0.930, 0.956] | 0.943 [0.930, 0.956] | 0.943 [0.930, 0.956] | same charge |
| `FDC-BF[ney-pilot,n=250] / PJC-A-menu[pilot n=250,init+warm]` | 0.943 [0.928, 0.955] | 0.942 [0.928, 0.955] | 0.942 [0.928, 0.955] | 0.942 [0.928, 0.955] | same charge |
| `FDC-BF[ney-pilot,n=250] / PJC-A-menu[no pilot]` | 0.953 [0.937, 0.968] | 0.951 [0.935, 0.966] | 0.950 [0.935, 0.965] | 0.950 [0.935, 0.965] | 1.00 |
| `FDC-BF[ney-pilot,n=1000] / PJC-A-reset[pilot n=1000,init]` | 0.962 [0.946, 0.978] | 0.961 [0.946, 0.977] | 0.961 [0.945, 0.977] | 0.961 [0.945, 0.977] | same charge |
| `FDC-BF[ney-pilot,n=1000] / PJC-A-reset[pilot n=1000,init+warm]` | 0.962 [0.946, 0.978] | 0.961 [0.946, 0.977] | 0.961 [0.945, 0.977] | 0.961 [0.945, 0.977] | same charge |
| `FDC-BF[ney-pilot,n=1000] / PJC-A-reset[no pilot]` | 0.975 [0.959, 0.990] | 0.966 [0.950, 0.981] | 0.964 [0.948, 0.979] | 0.963 [0.947, 0.978] | 1.00 |
| `FDC-BF[ney-pilot,n=1000] / PJC-A-menu[pilot n=1000,init]` | 0.937 [0.924, 0.950] | 0.937 [0.923, 0.949] | 0.937 [0.923, 0.949] | 0.937 [0.923, 0.949] | same charge |
| `FDC-BF[ney-pilot,n=1000] / PJC-A-menu[pilot n=1000,init+warm]` | 0.938 [0.925, 0.951] | 0.938 [0.924, 0.950] | 0.938 [0.924, 0.950] | 0.938 [0.924, 0.950] | same charge |
| `FDC-BF[ney-pilot,n=1000] / PJC-A-menu[no pilot]` | 0.953 [0.938, 0.967] | 0.944 [0.930, 0.958] | 0.942 [0.928, 0.956] | 0.941 [0.927, 0.955] | 1.00 |

## CR9-mixed (SEMI-SYNTHETIC; eps 0.00075; tau_R 6,989,793; 250 streams)

Adaptive picks (block F, tuned without a pilot): reset `PJC-A[reset,b=2,neyman]`, menu `PJC-A[menu,b=8,menu=0.3/0.4/0.5/0.6/0.7,dirproj]`.  Reproduction of F's pilot-0 rows on seeds 950-959: 30/30 identical.

FDC-BF (frozen 50/50, no pilot) / method, paired geomean N80 [95% CI]; method charged pilot/J:

| method | pilot reads (/tau) | J=1 (full charge) | J=4 | J=16 | uncharged | break-even J | frac switched | false streams |
|---|---|---|---|---|---|---|---|---|
| `FDC-BF[ney-pilot,F fixed env pilot n=1000]` | 18,000 (0.003) | 1.019 [1.002, 1.037] | 1.028 [1.011, 1.046] | 1.030 [1.013, 1.048] | 1.031 [1.014, 1.049] | 1.00 | -- | 0 |
| `PJC-A-reset[no pilot]` | 0 | 1.003 [0.986, 1.021] | 1.003 [0.986, 1.021] | 1.003 [0.986, 1.021] | 1.003 [0.986, 1.021] | -- | 1.00 | 0 |
| `PJC-A-menu[no pilot]` | 0 | 0.922 [0.905, 0.940] | 0.922 [0.905, 0.940] | 0.922 [0.905, 0.940] | 0.922 [0.905, 0.940] | -- | 0.86 | 0 |
| `FDC-BF[ney-pilot,n=250]` | 4,500 (0.001) | 1.031 [1.014, 1.048] | 1.033 [1.016, 1.050] | 1.034 [1.017, 1.051] | 1.034 [1.017, 1.051] | 1.00 | -- | 0 |
| `PJC-A-reset[pilot n=250,init]` | 4,500 (0.001) | 1.000 [0.982, 1.019] | 1.002 [0.984, 1.021] | 1.003 [0.984, 1.022] | 1.003 [0.985, 1.022] | 1.00 | 1.00 | 0 |
| `PJC-A-reset[pilot n=250,init+warm]` | 4,500 (0.001) | 1.001 [0.983, 1.020] | 1.003 [0.985, 1.022] | 1.004 [0.985, 1.023] | 1.004 [0.986, 1.023] | 1.00 | 1.00 | 0 |
| `PJC-A-menu[pilot n=250,init]` | 4,500 (0.001) | 0.989 [0.972, 1.006] | 0.991 [0.974, 1.008] | 0.992 [0.974, 1.008] | 0.992 [0.974, 1.008] | never | 0.08 | 0 |
| `PJC-A-menu[pilot n=250,init+warm]` | 4,500 (0.001) | 0.989 [0.972, 1.006] | 0.991 [0.974, 1.008] | 0.992 [0.974, 1.008] | 0.992 [0.974, 1.008] | never | 0.08 | 0 |
| `FDC-BF[ney-pilot,n=1000]` | 18,000 (0.003) | 1.013 [0.996, 1.030] | 1.021 [1.004, 1.038] | 1.023 [1.007, 1.041] | 1.024 [1.007, 1.041] | 1.00 | -- | 0 |
| `PJC-A-reset[pilot n=1000,init]` | 18,000 (0.003) | 0.992 [0.974, 1.011] | 1.000 [0.982, 1.019] | 1.002 [0.984, 1.021] | 1.003 [0.985, 1.022] | 3.52 | 1.00 | 0 |
| `PJC-A-reset[pilot n=1000,init+warm]` | 18,000 (0.003) | 0.989 [0.970, 1.008] | 0.997 [0.978, 1.016] | 0.999 [0.980, 1.018] | 1.000 [0.980, 1.019] | never | 1.00 | 0 |
| `PJC-A-menu[pilot n=1000,init]` | 18,000 (0.003) | 0.973 [0.956, 0.989] | 0.981 [0.964, 0.997] | 0.983 [0.966, 0.999] | 0.983 [0.966, 1.000] | never | 0.00 | 0 |
| `PJC-A-menu[pilot n=1000,init+warm]` | 18,000 (0.003) | 0.973 [0.956, 0.989] | 0.981 [0.964, 0.997] | 0.983 [0.966, 0.999] | 0.983 [0.966, 1.000] | never | 0.00 | 0 |

Head-to-head, frozen pilot-Neyman FDC-BF / adaptive (< 1: frozen pilot-Neyman faster).  'Equal information' rows charge both sides the same pilot; 'vs no pilot' rows charge only the frozen side:

| comparison | J=1 | J=4 | J=16 | uncharged | break-even J |
|---|---|---|---|---|---|
| `FDC-BF[ney-pilot,n=250] / PJC-A-reset[pilot n=250,init]` | 0.970 [0.956, 0.985] | 0.970 [0.956, 0.985] | 0.970 [0.956, 0.985] | 0.970 [0.956, 0.985] | same charge |
| `FDC-BF[ney-pilot,n=250] / PJC-A-reset[pilot n=250,init+warm]` | 0.971 [0.956, 0.986] | 0.971 [0.956, 0.986] | 0.971 [0.956, 0.986] | 0.971 [0.956, 0.986] | same charge |
| `FDC-BF[ney-pilot,n=250] / PJC-A-reset[no pilot]` | 0.973 [0.958, 0.987] | 0.971 [0.956, 0.985] | 0.970 [0.955, 0.985] | 0.970 [0.955, 0.985] | 1.00 |
| `FDC-BF[ney-pilot,n=250] / PJC-A-menu[pilot n=250,init]` | 0.959 [0.947, 0.972] | 0.959 [0.946, 0.972] | 0.959 [0.946, 0.972] | 0.959 [0.946, 0.972] | same charge |
| `FDC-BF[ney-pilot,n=250] / PJC-A-menu[pilot n=250,init+warm]` | 0.959 [0.947, 0.972] | 0.959 [0.946, 0.972] | 0.959 [0.946, 0.972] | 0.959 [0.946, 0.972] | same charge |
| `FDC-BF[ney-pilot,n=250] / PJC-A-menu[no pilot]` | 0.894 [0.878, 0.912] | 0.893 [0.876, 0.910] | 0.892 [0.876, 0.910] | 0.892 [0.875, 0.910] | 1.00 |
| `FDC-BF[ney-pilot,n=1000] / PJC-A-reset[pilot n=1000,init]` | 0.980 [0.964, 0.995] | 0.979 [0.963, 0.995] | 0.979 [0.963, 0.995] | 0.979 [0.963, 0.995] | same charge |
| `FDC-BF[ney-pilot,n=1000] / PJC-A-reset[pilot n=1000,init+warm]` | 0.977 [0.961, 0.992] | 0.976 [0.960, 0.992] | 0.976 [0.960, 0.992] | 0.976 [0.960, 0.992] | same charge |
| `FDC-BF[ney-pilot,n=1000] / PJC-A-reset[no pilot]` | 0.990 [0.975, 1.005] | 0.982 [0.967, 0.997] | 0.980 [0.965, 0.994] | 0.979 [0.964, 0.994] | 1.00 |
| `FDC-BF[ney-pilot,n=1000] / PJC-A-menu[pilot n=1000,init]` | 0.961 [0.949, 0.972] | 0.960 [0.948, 0.971] | 0.960 [0.948, 0.971] | 0.960 [0.948, 0.971] | same charge |
| `FDC-BF[ney-pilot,n=1000] / PJC-A-menu[pilot n=1000,init+warm]` | 0.961 [0.949, 0.972] | 0.960 [0.948, 0.971] | 0.960 [0.948, 0.971] | 0.960 [0.948, 0.971] | same charge |
| `FDC-BF[ney-pilot,n=1000] / PJC-A-menu[no pilot]` | 0.910 [0.893, 0.929] | 0.903 [0.885, 0.921] | 0.901 [0.883, 0.919] | 0.900 [0.883, 0.918] | 1.00 |

## Answer

Does 'the gain comes from the allocation, not from online adaptation' survive?  Only in a qualified form (semi-synthetic, post hoc, descriptive).  (1) EQUAL INFORMATION: when the adaptive picks get the same per-stream pilot (phase-0 pilot-Neyman plan, plus variance warm start) and both sides pay the same pilot, frozen pilot-Neyman FDC-BF is still faster than every adaptive variant in all four environments and both pilot sizes (ney-pilot / adaptive 0.90-0.98, every 95% CI below 1; e.g. Criteo-asym n=1000: 0.962 [0.946, 0.978] vs reset, 0.937 [0.924, 0.950] vs menu).  The pilot hardly helps the adaptive members (Criteo-asym reset 1.127 without vs 1.125 with the 1000-pilot): the reset member discards pre-boundary data and the menu member pays its path-union ledger even when it rarely leaves the pilot plan (menu init+warm switches on 0-19% of streams yet trails frozen pilot-Neyman by 4-8%).  So, at equal information, the advantage is the frozen allocation plus the absence of reset / union costs, not online learning.  (2) CHARGED PILOT, adaptive learns online for free: on Criteo structure (tau_R 7.0M; the pilot is 0.06-0.26% of tau_R) charging changes nothing material -- ney-pilot at full charge (J = 1) vs the no-pilot adaptive picks 0.89-0.99 (one CI touches 1: Criteo-mixed n=1000 vs reset 0.990 [0.975, 1.005]) and FDC-BF / ney-pilot 1.16 (asym) and 1.01-1.03 (mixed).  On X5 structure (tau_R 99,646; the pilot is 4.5% or 18% of tau_R) the conclusion REVERSES at full charge: ney-pilot is slower than frozen 50/50 (X5-asym 0.893 [0.879, 0.907] at n=250, 0.632 at n=1000) and slower than the no-pilot adaptive picks (1.08-1.67 against it); the pilot pays for itself only when shared across J >= 2.1-25.5 certification instances (break-even vs the adaptive picks), and against frozen 50/50 only at J >= 4.5 (n=250) or 19 (n=1000) on X5-asym and never on X5-mixed.  Wording for the paper: an informed frozen plan beats these reset / menu implementations at equal information in the four tested regimes; on small tables this holds only when the pilot is pre-existing or amortised over many certification runs.  No method had a false certification stream (0 / 250 per method and environment).  Phase-wise TU joint rival not implemented.

