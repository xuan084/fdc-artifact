"""Post-hoc (descriptive) cell-bound x aggregation 2x2 on the sealed v7 block-A rows.

All four cells share the frozen 50/50 schedule (identical schedule digests, R2 check of lock v7):
  Bernstein cell bound (no FPC) : rectangle = RECT-ck-Bern, joint = FDC (v5)
  finite-population cell bound  : rectangle = RECT-ck-HG,   joint = FDC-BF
Paired geometric-mean ratios over the 200 eval streams (seeds 33000-33199), endpoint N80_pen,
paired percentile bootstrap B = 10^4, seed 42 (same convention as the lock). Not part of any verdict.
Output: exp/results/full/v7_posthoc_2x2/summary.json
"""
import json, hashlib
from pathlib import Path
import numpy as np

FULL = Path(__file__).resolve().parents[1] / 'results' / 'full'
SRC = FULL / 'v7a_full_b' / 'results.jsonl'
M = ['RECT-ck-Bern', 'FDC', 'RECT-ck-HG', 'FDC-BF', 'HC-WoR', 'B4-bal']
rows = {}
for line in open(SRC):
    r = json.loads(line)
    if r['method'] in M:
        rows.setdefault(r['method'], {})[r['seed']] = r
seeds = sorted(rows['FDC-BF'])
assert len(seeds) == 200 and all(sorted(rows[m]) == seeds for m in M)
for m in M:  # same frozen schedule except HC-WoR? (HC-WoR also 50/50 frozen in lock v6/v7 group A)
    pass
dig = {m: [rows[m][s]['schedule_digest'] for s in seeds] for m in M}
same_sched = {m: all(a == b for a, b in zip(dig[m], dig['FDC-BF'])) for m in M}
L = {m: np.log([rows[m][s]['N80_pen'] for s in seeds]) for m in M}
rng = np.random.default_rng(42)
idx = rng.integers(0, 200, size=(10_000, 200))

def contrast(vec):
    est = float(np.exp(vec.mean()))
    bs = np.exp(vec[idx].mean(axis=1))
    return {'ratio': est, 'ci95': [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
            'ub95_one_sided': float(np.quantile(bs, .95))}

out = {
    'note': 'post-hoc descriptive re-analysis of sealed lock-v7 block A rows; not part of any verdict',
    'source': str(SRC.relative_to(FULL.parents[1])),
    'source_sha256': hashlib.sha256(SRC.read_bytes()).hexdigest(),
    'same_schedule_digest_as_FDC_BF': same_sched,
    'geomean_N80_pen': {m: float(np.exp(L[m].mean())) for m in M},
    'aggregation_effect_bernstein_cells (FDC / RECT-ck-Bern)': contrast(L['FDC'] - L['RECT-ck-Bern']),
    'aggregation_effect_fpc_cells (FDC-BF / RECT-ck-HG)': contrast(L['FDC-BF'] - L['RECT-ck-HG']),
    'cell_effect_rectangle (RECT-ck-HG / RECT-ck-Bern)': contrast(L['RECT-ck-HG'] - L['RECT-ck-Bern']),
    'cell_effect_joint (FDC-BF / FDC)': contrast(L['FDC-BF'] - L['FDC']),
    'both (FDC-BF / RECT-ck-Bern)': contrast(L['FDC-BF'] - L['RECT-ck-Bern']),
    'interaction_I_log': None,
}
I = (L['FDC-BF'] - L['RECT-ck-HG']) - (L['FDC'] - L['RECT-ck-Bern'])
bs = I[idx].mean(axis=1)
out['interaction_I_log'] = {'I': float(I.mean()), 'ci95': [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]}
(FULL / 'v7_posthoc_2x2').mkdir(exist_ok=True)
json.dump(out, open(FULL / 'v7_posthoc_2x2' / 'summary.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
