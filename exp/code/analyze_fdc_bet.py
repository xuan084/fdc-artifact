"""Analysis of the FDC-bet exploration runs (dev only).  Paired geomean ratios with the v6 bootstrap (B = 10^4, seed 42,
same resample matrix for every comparator), one-sided UB95, two-sided CI; endpoint N80_pen (primary) and x12
(interpolated crossing, diagnostic).  Usage: analyze_fdc_bet.py --task dev|lenta|tune [--ref NAME ...]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
WS = CODE.parents[1]
OUT = WS / "exp/results/pilots/fdc_bet"

from dsswm.stats.v6_analysis import boot_idx, paired  # noqa: E402


def load(task):
    rows = [json.loads(l) for l in (OUT / task / "results.jsonl").read_text().splitlines()]
    key = {}
    for r in rows:
        key[(r["method"], r["seed"], r["eps"])] = r          # last write wins (no duplicates expected)
    return list(key.values())


def geo(v):
    return float(np.exp(np.mean(np.log(np.asarray(v, dtype=float)))))


def analyse(task, refs, cands=None):
    rows = load(task)
    out = {"task": task, "per_eps": {}}
    for e in sorted({r["eps"] for r in rows}):
        R = [r for r in rows if r["eps"] == e]
        methods = sorted({r["method"] for r in R})
        seeds = sorted({r["seed"] for r in R})
        idx = boot_idx(len(seeds))
        by = {m: {r["seed"]: r for r in R if r["method"] == m} for m in methods}
        complete = [m for m in methods if set(by[m]) == set(seeds)]
        desc = {}
        for m in complete:
            rr = [by[m][s] for s in seeds]
            tau = rr[0]["tau_R"]
            digs = {r["schedule_digest"] for r in rr}
            desc[m] = {"n": len(rr), "geomean_N80_over_tau": geo([r["N80_pen"] / tau for r in rr]),
                       "geomean_x12_over_tau": geo([r["x12"] / tau for r in rr]),
                       "share_at_tau": float(np.mean([r["N80_pen"] >= tau for r in rr])),
                       "false_streams": int(sum(r["fwer_event"] for r in rr)),
                       "billing_ok_all": all(r["billing_ok"] for r in rr),
                       "sec_mean": float(np.mean([r["sec"] for r in rr]))}
        # schedule identity (R2-style): all frozen-50/50 methods share the digest per seed
        same = all(len({by[m][s]["schedule_digest"] for m in complete}) == 1 for s in seeds)
        ratios = {}
        for ref in refs:
            if ref not in complete:
                continue
            for m in (cands or complete):
                if m == ref or m not in complete:
                    continue
                for k in ("N80_pen", "x12"):
                    a = [by[m][s][k] for s in seeds]
                    b = [by[ref][s][k] for s in seeds]
                    ratios[f"{m}/{ref}|{k}"] = paired(a, b, idx)
        out["per_eps"][str(e)] = {"seeds": [seeds[0], seeds[-1]], "n_seeds": len(seeds), "describe": desc,
                                  "schedule_identity_all_methods": same, "ratios": ratios}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--ref", nargs="*", default=["RECT-ck-HG", "RECT-ck-HG-live", "HC-WoR", "FDC"])
    ap.add_argument("--print", action="store_true")
    a = ap.parse_args()
    res = analyse(a.task, a.ref)
    (OUT / f"analysis_{a.task}.json").write_text(json.dumps(res, indent=1))
    for e, d in res["per_eps"].items():
        print(f"== {a.task} eps={e} seeds {d['seeds']} n={d['n_seeds']} schedule_identity={d['schedule_identity_all_methods']}")
        for m, x in sorted(d["describe"].items(), key=lambda kv: kv[1]["geomean_N80_over_tau"]):
            print(f"  {m:28s} N80/tau={x['geomean_N80_over_tau']:.4f} x12/tau={x['geomean_x12_over_tau']:.4f} "
                  f"atTau={x['share_at_tau']:.2f} false={x['false_streams']} sec={x['sec_mean']:.1f}")
        for k, r in d["ratios"].items():
            if a.print or "|N80_pen" in k:
                print(f"  {k:48s} {r['geomean_ratio']:.3f} [{r['ci95_two_sided'][0]:.3f},{r['ci95_two_sided'][1]:.3f}] "
                      f"UB95={r['ub95_one_sided']:.3f} faster/tied/slower={r['frac_faster']:.2f}/{r['frac_tied']:.2f}/"
                      f"{r['frac_slower']:.2f}")


if __name__ == "__main__":
    main()
