"""Recompute the numbers added in revision r3 from the sealed result rows (read-only).

- matched mechanism contrast (Table A2): RECT-ck-BF / RECT-ck-Bern, FDC-BF / FDC, interaction I, paired bootstrap
  (B = 10^4, seed 42, two-sided percentile 95% CI), Criteo-test streams 33000-33199, eps 0.001;
- geometric-mean N80_pen rows and reads saved vs the exact rectangle on both logs (Table 2);
- trajectory medians (median certified-problem curve of FDC-BF vs every Criteo-test method).
Usage (repo root): .venv/bin/python3 iter_001/writing/scripts/verify_r3_numbers.py
"""
import collections
import json
from pathlib import Path

import numpy as np

FULL = Path(__file__).resolve().parents[2] / "exp/results/full"


def load(paths, eps):
    d = collections.defaultdict(dict)
    for p in paths:
        for line in open(FULL / p):
            r = json.loads(line)
            if abs(r["eps"] - eps) < 1e-12:
                d[r["method"]][r["seed"]] = r
    return d


def gm(x):
    return float(np.exp(np.mean(np.log(x))))


def main():
    cr = load(["v7a_full_b/results.jsonl", "v8c_full/results.jsonl"], 0.001)
    seeds = sorted(cr["FDC-BF"])
    a = {m: np.array([cr[m][s]["N80_pen"] for s in seeds], float)
         for m in ["FDC-BF", "FDC", "RECT-ck-BF", "RECT-ck-Bern", "RECT-ck-HG", "HC-WoR"]}
    idx = np.random.default_rng(42).integers(0, len(seeds), (10_000, len(seeds)))

    def boot(f):
        bs = np.array([f(i) for i in idx])
        return f(np.arange(len(seeds))), np.percentile(bs, [2.5, 97.5])

    def ratio(x, y):
        return lambda i: float(np.exp(np.mean(np.log(a[x][i] / a[y][i]))))

    print("Criteo geomean rows:", {m: round(gm(v)) for m, v in a.items()})
    print("RECT-ck-BF/RECT-ck-Bern", boot(ratio("RECT-ck-BF", "RECT-ck-Bern")))
    print("FDC-BF/FDC", boot(ratio("FDC-BF", "FDC")))
    print("FDC/RECT-ck-Bern", boot(ratio("FDC", "RECT-ck-Bern")))
    print("FDC-BF/RECT-ck-BF", boot(ratio("FDC-BF", "RECT-ck-BF")))
    print("I", boot(lambda i: float(np.mean(np.log(a["FDC-BF"][i] / a["FDC"][i]))
                                    - np.mean(np.log(a["RECT-ck-BF"][i] / a["RECT-ck-Bern"][i])))))
    tau_c = cr["FDC-BF"][seeds[0]]["tau_R"]
    sv_c = gm(a["RECT-ck-HG"]) - gm(a["FDC-BF"])
    print(f"Criteo reads saved vs RECT-ck-HG: {sv_c:.0f} ({sv_c / tau_c:.4f} of tau_R), share {1 - gm(a['FDC-BF']) / gm(a['RECT-ck-HG']):.3f}")

    x5 = load(["v8a_full_a/results.jsonl", "v8a_full_b/results.jsonl"], 0.02)
    xs = sorted(x5["FDC-BF"])
    g = {m: gm([x5[m][s]["N80_pen"] for s in xs]) for m in ["FDC-BF", "RECT-ck-HG", "HC-WoR", "RECT-ck-BF"]}
    tau_x = x5["FDC-BF"][xs[0]]["tau_R"]
    sv_x = g["RECT-ck-HG"] - g["FDC-BF"]
    print("X5 geomean rows:", {m: round(v) for m, v in g.items()})
    print(f"X5 reads saved vs RECT-ck-HG: {sv_x:.0f} ({sv_x / tau_x:.4f} of tau_R), share {1 - g['FDC-BF'] / g['RECT-ck-HG']:.3f}")
    for price, name in [(0.002, "clean-room $2/1k"), (0.10, "label $0.10")]:
        print(f"  {name}: Criteo ${sv_c * price:,.0f}  X5 ${sv_x * price:,.0f}")

    curves = collections.defaultdict(list)
    for line in open(FULL / "v7a_full_b/results.jsonl"):
        r = json.loads(line)
        curves[r["method"]].append(r["n_cert_curve"])
    med = {m: np.median(np.array(v), 0) for m, v in curves.items()}
    bad = {m: np.flatnonzero(med[m] > med["FDC-BF"]).tolist() for m in med if m != "FDC-BF"}
    print("checkpoints where a method's median exceeds FDC-BF's:", {m: v for m, v in bad.items() if v} or "none")
    print("HC-WoR minus RECT-ck-HG median, nonzero at ck idx:", np.flatnonzero(med["HC-WoR"] != med["RECT-ck-HG"]).tolist())


if __name__ == "__main__":
    main()
