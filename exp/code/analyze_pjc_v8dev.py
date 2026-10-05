"""Analysis of the round-8 feasibility runs (dev only).  Tuning selection per family on the tune task (seeds 900-949) by
the v5 rule (argmin geomean N80_pen subject to 0 false streams; tie-break geomean x12), then paired geomean ratios
FDC-BF / r (and r / FDC-BF) on the report task with the v6 bootstrap (B = 10^4, seed 42, one resample matrix).
Usage: analyze_pjc_v8dev.py --select       (writes selection.json from tune)
       analyze_pjc_v8dev.py --task dev|lenta|cr12|x5 --primary FDC-BF"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
WS = CODE.parents[1]
OUT = WS / "exp/results/pilots/pjc_v8"

from dsswm.stats.v6_analysis import boot_idx, paired  # noqa: E402
import run_pjc_v8dev as R  # noqa: E402

FAMILIES = {
    "PJC-local": list(R.PJC_LOCAL),
    "PJC-menu": list(R.PJC_MENU),
    "RECT-ck-HG-plan": ["RECT-ck-HG"] + R.RECT_PLANS,
    "HC-WoR-plan": ["HC-WoR"] + R.HC_PLANS,
    "RECT-ck-BF": list(R.MATCHED),
}


def load(task):
    key = {}
    for l in (OUT / task / "results.jsonl").read_text().splitlines():
        r = json.loads(l)
        key[(r["method"], r["seed"], r["eps"])] = r
    return list(key.values())


def geo(v):
    return float(np.exp(np.mean(np.log(np.asarray(v, dtype=float)))))


def describe(R_, m, seeds):
    rr = [R_[m][s] for s in seeds]
    tau = rr[0]["tau_R"]
    return {"geomean_N80_over_tau": geo([r["N80_pen"] / tau for r in rr]),
            "geomean_x12_over_tau": geo([r["x12"] / tau for r in rr]),
            "share_at_tau": float(np.mean([r["N80_pen"] >= tau for r in rr])),
            "false_streams": int(sum(r["fwer_event"] for r in rr)), "billing_ok_all": all(r["billing_ok"] for r in rr),
            "sec_mean": float(np.mean([r["sec"] for r in rr]))}


def select():
    rows = load("tune")
    seeds = sorted({r["seed"] for r in rows})
    by = {}
    for r in rows:
        by.setdefault(r["method"], {})[r["seed"]] = r
    out = {"rule": "argmin geomean N80_pen on tune seeds 900-949 s.t. 0 false streams; tie-break geomean x12",
           "families": {}}
    for fam, names in FAMILIES.items():
        cand = {m: describe(by, m, seeds) for m in names if m in by and set(by[m]) == set(seeds)}
        ok = {m: d for m, d in cand.items() if d["false_streams"] == 0}
        pick = min(ok, key=lambda m: (ok[m]["geomean_N80_over_tau"], ok[m]["geomean_x12_over_tau"])) if ok else None
        out["families"][fam] = {"pick": pick, "n_candidates": len(cand), "missing": [m for m in names if m not in cand],
                                "table": cand}
    ref = describe(by, "FDC-BF", seeds) if "FDC-BF" in by else None
    out["FDC-BF_tune"] = ref
    (OUT / "selection.json").write_text(json.dumps(out, indent=1))
    for fam, d in out["families"].items():
        print(f"== {fam}: pick {d['pick']} ({d['n_candidates']} candidates)")
        for m, x in sorted(d["table"].items(), key=lambda kv: kv[1]["geomean_N80_over_tau"]):
            print(f"   {m:48s} N80/tau={x['geomean_N80_over_tau']:.4f} x12/tau={x['geomean_x12_over_tau']:.4f} "
                  f"false={x['false_streams']}")
    if ref:
        print(f"   FDC-BF (tune) N80/tau={ref['geomean_N80_over_tau']:.4f} x12/tau={ref['geomean_x12_over_tau']:.4f}")
    return out


def analyse(task, primary):
    rows = load(task)
    sel = json.loads((OUT / "selection.json").read_text()) if (OUT / "selection.json").exists() else {"families": {}}
    picks = {d["pick"]: fam for fam, d in sel["families"].items() if d.get("pick")}
    out = {"task": task, "primary": primary, "tuned_picks": picks, "per_eps": {}}
    for e in sorted({r["eps"] for r in rows}):
        Re = [r for r in rows if r["eps"] == e]
        seeds = sorted({r["seed"] for r in Re})
        by = {}
        for r in Re:
            by.setdefault(r["method"], {})[r["seed"]] = r
        complete = [m for m in by if set(by[m]) == set(seeds)]
        idx = boot_idx(len(seeds))
        desc = {m: describe(by, m, seeds) for m in complete}
        ratios = {}
        if primary in complete:
            for m in complete:
                if m == primary:
                    continue
                for k in ("N80_pen", "x12"):
                    a = [by[primary][s][k] for s in seeds]
                    b = [by[m][s][k] for s in seeds]
                    ratios[f"{primary}/{m}|{k}"] = paired(a, b, idx)
        out["per_eps"][str(e)] = {"n_seeds": len(seeds), "seeds": [seeds[0], seeds[-1]], "describe": desc,
                                  "ratios": ratios}
    (OUT / f"analysis_{task}.json").write_text(json.dumps(out, indent=1))
    for e, d in out["per_eps"].items():
        print(f"== {task} eps={e} n={d['n_seeds']}")
        for m, x in sorted(d["describe"].items(), key=lambda kv: kv[1]["geomean_N80_over_tau"]):
            r = d["ratios"].get(f"{primary}/{m}|N80_pen")
            rs = (f"  {primary}/m={r['geomean_ratio']:.3f} [{r['ci95_two_sided'][0]:.3f},{r['ci95_two_sided'][1]:.3f}]"
                  f" UB={r['ub95_one_sided']:.3f} f/t/s={r['frac_faster']:.2f}/{r['frac_tied']:.2f}/{r['frac_slower']:.2f}"
                  ) if r else ""
            rx = d["ratios"].get(f"{primary}/{m}|x12")
            rxs = f" x12ratio={rx['geomean_ratio']:.3f} UB={rx['ub95_one_sided']:.3f}" if rx else ""
            tag = " *" + picks[m] if m in picks else ""
            print(f"  {m:46s} N80={x['geomean_N80_over_tau']:.4f} x12={x['geomean_x12_over_tau']:.4f} "
                  f"atTau={x['share_at_tau']:.2f} F={x['false_streams']}{rs}{rxs}{tag}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--task")
    ap.add_argument("--primary", default="FDC-BF")
    a = ap.parse_args()
    if a.select:
        select()
    if a.task:
        analyse(a.task, a.primary)
