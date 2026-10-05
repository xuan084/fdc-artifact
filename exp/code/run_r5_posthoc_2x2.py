"""Post-hoc (zero-cost, exploratory) re-analysis of lock-v5 confirmatory rows.

Computes the design x certificate 2x2 with paired bootstrap CIs and the log
interaction I, plus the conservative N100 bound FDC_N100 / B4-bal_N80.
No new runs; reads only r5_cr_main_{a,b}, r5_cr_factorial, r5_cr_fwer_audit.
Labelled post-hoc: not part of the lock v5 analysis plan.
"""
import json
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parents[2]
FULL = WS / "exp/results/full"
OUT = FULL / "r5_posthoc_2x2"
OUT.mkdir(exist_ok=True)


def load(path, key="N80_pen", methods=None):
    rows = {}
    for line in open(path):
        r = json.loads(line)
        m = r.get("method")
        if methods and m not in methods:
            continue
        rows.setdefault(m, {})[r["seed"]] = r[key]
    return rows


main = {}
for blk in ("r5_cr_main_a", "r5_cr_main_b"):
    for m, d in load(FULL / blk / "results.jsonl").items():
        main.setdefault(m, {}).update(d)
fac = load(FULL / "r5_cr_factorial/results.jsonl")
aud = {}
for line in open(FULL / "r5_cr_fwer_audit/results.jsonl"):
    r = json.loads(line)
    if r.get("method") == "FDC" and r.get("window") == "full_horizon":
        aud[r["seed"]] = r["N100_pen"]

seeds = sorted(main["FDC"].keys())
assert len(seeds) == 200
cells = {
    "pool_rect (B4)": np.log([main["B4"][s] for s in seeds]),
    "pool_joint (QFC-pool)": np.log([main["QFC-pool"][s] for s in seeds]),
    "half_rect (B4-bal)": np.log([main["B4-bal"][s] for s in seeds]),
    "half_joint (F[half,qstar,r4])": np.log([fac["F[half,qstar,r4]"][s] for s in seeds]),
    "FDC": np.log([main["FDC"][s] for s in seeds]),
}
pr, pj, hr, hj, fdc = (cells[k] for k in cells)
contrasts = {
    "design_effect_rect (B4-bal/B4)": hr - pr,
    "design_effect_joint (half_joint/QFC-pool)": hj - pj,
    "cert_effect_pool (QFC-pool/B4)": pj - pr,
    "cert_effect_half (half_joint/B4-bal)": hj - hr,
    "both (half_joint/B4)": hj - pr,
    "interaction_I_log": hj - pj - hr + pr,
}
n100 = np.log([aud[s] for s in seeds])
contrasts["FDC_N100_over_B4bal_N80 (conservative bound)"] = n100 - hr

rng = np.random.default_rng(42)
B = 10000
idx = rng.integers(0, 200, size=(B, 200))
res = {"task_id": "r5_posthoc_2x2", "status": "posthoc_exploratory",
       "note": "Zero-cost re-analysis of lock-v5 confirmatory rows; not in the lock v5 analysis plan. Paired percentile bootstrap over streams, B=10000, seed 42.",
       "n_streams": 200, "geomean_N80_pen": {k: float(np.exp(v.mean())) for k, v in cells.items()},
       "contrasts": {}}
for k, v in contrasts.items():
    bs = v[idx].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    is_log = k.startswith("interaction")
    res["contrasts"][k] = {
        "point": float(v.mean()) if is_log else float(np.exp(v.mean())),
        "ci95": [float(lo), float(hi)] if is_log else [float(np.exp(lo)), float(np.exp(hi))],
        "scale": "log" if is_log else "ratio",
    }
res["contrasts"]["interaction_I_log"]["ratio_scale_point"] = float(np.exp(contrasts["interaction_I_log"].mean()))
json.dump(res, open(OUT / "summary.json", "w"), indent=1)
print(json.dumps(res, indent=1))
