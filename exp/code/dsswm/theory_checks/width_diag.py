"""Non-vacuity diagnostic for the strict QFC certificate on Hillstrom S=6.

Best case: muhat = mu (oracle), counts = pool-proportional expected counts.
Reports U_q per checkpoint under (a) the pre-registered union x = 2 S ln A +
ln(K/delta_main) and (b) the tighter, equally valid union over
{(pi*_q, pi) : q <= Q, pi in Pi_all} x checkpoints: x = S ln A + ln(Q K / delta_main).
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

from dsswm.theory_checks import mc_l1 as m


def main(out):
    p = m.certificate_problem_hillstrom()
    S, A, w, Np, ones = p["S"], p["A"], p["w"], p["Npool"], p["ones"]
    Pol = m._policies(S, A)
    fc = np.arange(S)[None, :] * A + Pol
    mu = (ones / Np).flatten()
    Nf = Np.flatten()
    J = (w[None, :] * mu[fc]).sum(1)
    Q, K, dm, dv = 15, 20, 0.04, 0.01
    xs = {"pairs_all": 2 * S * math.log(A) + math.log(K / dm),
          "pairs_qstar": S * math.log(A) + math.log(Q * K / dm)}
    xvar = math.log(2 * 48 * K / dv)
    rows = []
    gaps = []
    for T in p["ckpts"]:
        n = np.round(Nf * T / Nf.sum()).astype(int)
        lo, hi = m.bernstein_mu_ci(mu, n, Nf, xvar)
        s2 = m.sigma2_ucb(lo, hi)
        live = n < Nf
        vt = np.where(live, s2 / n, 0)
        bi = np.where(live, 1 / n, 0)
        rec = {"T": int(T)}
        for name, x in xs.items():
            Us = []
            for _, cv, B in p["problems"]:
                f = (w[None, :] * cv[Pol]).sum(1) <= B + 1e-12
                ih = int(np.argmax(np.where(f, J, -np.inf)))
                diff = Pol != Pol[ih][None, :]
                V = np.where(diff, (w ** 2)[None, :] * (vt[fc] + vt[fc[ih]][None, :]), 0).sum(1)
                b = np.where(diff, w[None, :] * np.maximum(bi[fc], bi[fc[ih]][None, :]), 0).max(1)
                Us.append(float(np.where(f, J - J[ih] + np.sqrt(2 * x * V) + b * x / 3, -np.inf).max()))
                if name == "pairs_all" and T == p["ckpts"][0]:
                    js = np.sort(J[f])[::-1]
                    gaps.append(float(js[0] - js[1]))
            Us = np.array(Us)
            rec[name] = {"median_U": float(np.median(Us)), "min_U": float(Us.min()),
                         "n_q_U_le_0.01": int((Us <= 0.01).sum()),
                         "n_q_U_le_0.0025": int((Us <= 0.0025).sum())}
        rows.append(rec)
    res = {"x": xs, "xvar": xvar, "top2_gap_per_problem": gaps, "by_checkpoint": rows,
           "note": "oracle muhat (best case); last checkpoint = full table, all cells exact"}
    Path(out).write_text(json.dumps(res, indent=1))
    for r in rows:
        print(r["T"], {k: (round(v["median_U"], 4), v["n_q_U_le_0.01"]) for k, v in r.items() if k != "T"})


if __name__ == "__main__":
    main(sys.argv[1])
