"""Analysis of the FDC-HG dev runs (run_fdc_hg_dev.py): paired geomean N80_pen ratios with the v6 bootstrap
(B = 10^4, seed 42), false streams, timing, and the v9_plan s2 go criteria evaluated literally.  Dev only."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
WS = CODE.parents[1]
OUT = WS / "exp/results/pilots/fdc_hg"
from dsswm.stats.v6_analysis import boot_idx, paired  # noqa: E402

METHODS = ["FDC-HG", "FDC-BF", "RECT-ck-HG", "HC-WoR", "RECT-ck-BF", "PJC-local"]
COMPARATORS = ["FDC-BF", "RECT-ck-HG", "HC-WoR", "RECT-ck-BF", "PJC-local"]


def load(task):
    key = {}
    for line in (OUT / task / "results.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("error") is None:
            key[(r["method"], r["seed"], r["eps"])] = r
    return list(key.values())


def block(rows, eps):
    R = [r for r in rows if abs(r["eps"] - eps) < 1e-12]
    seeds = sorted({r["seed"] for r in R})
    assert all(950 <= s <= 999 for s in seeds)
    by = {m: {r["seed"]: r for r in R if r["method"] == m} for m in METHODS}
    complete = [m for m in METHODS if set(by[m]) == set(seeds)]
    idx = boot_idx(len(seeds))
    out = {"eps": eps, "n_seeds": len(seeds), "complete_methods": complete, "per_method": {}, "ratios": {}}
    for m in complete:
        rr = [by[m][s] for s in seeds]
        out["per_method"][m] = {
            "geomean_N80_pen": float(np.exp(np.mean(np.log([r["N80_pen"] for r in rr])))),
            "false_streams": int(sum(r["fwer_event"] for r in rr)), "false_certs": int(sum(r["n_false"] for r in rr)),
            "sec_per_ck_mean": float(np.mean([r["sec_per_ck"] for r in rr])),
            "sec_per_ck_max": float(np.max([r["sec_per_ck"] for r in rr])),
            "cert_sec_max": float(np.max([r.get("cert_sec_max", np.nan) for r in rr])),
            "validity": rr[0]["validity"]}
    if "FDC-HG" in complete:
        a = [by["FDC-HG"][s]["N80_pen"] for s in seeds]
        for c in COMPARATORS:
            if c in complete:
                out["ratios"][f"FDC-HG/{c}"] = paired(a, [by[c][s]["N80_pen"] for s in seeds], idx)
        if "FDC-BF" in complete:
            out["x12_ratio_HG_BF_descriptive"] = paired([by["FDC-HG"][s]["x12"] for s in seeds],
                                                        [by["FDC-BF"][s]["x12"] for s in seeds], idx)
            out["schedule_identity_HG_BF"] = all(by["FDC-HG"][s]["schedule_digest"] == by["FDC-BF"][s]["schedule_digest"]
                                                 for s in seeds)
            out["per_stream_HG_le_BF"] = all(by["FDC-HG"][s]["N80_pen"] <= by["FDC-BF"][s]["N80_pen"] for s in seeds)
    if "FDC-BF" in complete:
        b = [by["FDC-BF"][s]["N80_pen"] for s in seeds]
        for c in COMPARATORS[1:]:
            if c in complete:
                out["ratios"][f"FDC-BF/{c}"] = paired(b, [by[c][s]["N80_pen"] for s in seeds], idx)
    return out


def main():
    res = {"cr9": [block(load("cr9"), 0.001)], "x5": [block(load("x5"), e) for e in (0.015, 0.02, 0.03)]}
    toy = json.loads((OUT / "toy" / "summary.json").read_text()) if (OUT / "toy" / "summary.json").exists() else None
    res["toy"] = toy
    # go criteria (v9_plan s2), primary eps: CR9 0.001, X5 0.02
    cr = res["cr9"][0]["ratios"].get("FDC-HG/FDC-BF")
    x5 = next(b for b in res["x5"] if b["eps"] == 0.02)["ratios"].get("FDC-HG/FDC-BF")
    c1 = None
    if cr and x5:
        u1, u2 = cr["ub95_one_sided"], x5["ub95_one_sided"]
        c1 = bool((u1 < 0.90 and u2 <= 1.00) or (u2 < 0.90 and u1 <= 1.00))
    blocks = res["cr9"] + res["x5"]
    hg_false = sum(b["per_method"].get("FDC-HG", {}).get("false_streams", 0) for b in blocks)
    toy_false = toy["table"]["FDC-HG"]["false_streams"] if toy else None
    c2 = bool(hg_false == 0 and toy_false == 0)          # + unit test (test_fdc_hg.py) passes, recorded in the report
    tmax = max(b["per_method"].get("FDC-HG", {}).get("cert_sec_max", np.inf) for b in blocks)
    c3 = bool(tmax <= 10.0)
    res["go"] = {"c1_ratio": c1, "c1_detail": {"CR9_ub95": cr and cr["ub95_one_sided"],
                                               "X5_eps0.02_ub95": x5 and x5["ub95_one_sided"]},
                 "c2_zero_false": c2, "c2_detail": {"dev_false_streams_FDC-HG": hg_false, "toy_false_streams": toy_false},
                 "c3_time": c3, "c3_detail": {"max_cert_sec_per_checkpoint": tmax},
                 "GO": bool(c1 and c2 and c3)}
    (OUT / "analysis_dev.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["go"], indent=1))
    for b in blocks:
        print(f"-- eps {b['eps']} n={b['n_seeds']} sched_id={b.get('schedule_identity_HG_BF')} "
              f"HG<=BF all={b.get('per_stream_HG_le_BF')}")
        if "x12_ratio_HG_BF_descriptive" in b:
            v = b["x12_ratio_HG_BF_descriptive"]
            print(f"   x12 FDC-HG/FDC-BF (descriptive) {v['geomean_ratio']:.3f} UB95 {v['ub95_one_sided']:.3f}")
        for k, v in b["ratios"].items():
            print(f"   {k:24s} {v['geomean_ratio']:.3f}  UB95 {v['ub95_one_sided']:.3f}  "
                  f"CI [{v['ci95_two_sided'][0]:.3f}, {v['ci95_two_sided'][1]:.3f}]  faster {v['frac_faster']:.2f} "
                  f"tied {v['frac_tied']:.2f}")
        for m, v in b["per_method"].items():
            print(f"   {m:12s} gm N80 {v['geomean_N80_pen']:.0f} false {v['false_streams']} "
                  f"s/ck {v['sec_per_ck_mean']:.2f} (max {v['sec_per_ck_max']:.2f}) cert_max {v['cert_sec_max']:.2f}")


if __name__ == "__main__":
    main()
