"""Analysis of the v9 candidate dev runs (run_v9_candidates_dev.py) + reused FDC-HG dev comparator rows (same frozen code,
configs, seeds).  Paired geomean N80_pen ratios, v6 bootstrap (B = 10^4, seed 42), one-sided UB95; go criteria of
plan/v9_candidates_theory.md evaluated literally.  Dev only."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
WS = CODE.parents[1]
P9 = WS / "exp/results/pilots/v9_candidates"
PHG = WS / "exp/results/pilots/fdc_hg"
from dsswm.stats.v6_analysis import boot_idx, paired  # noqa: E402


def load(path, methods=None):
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r.get("error") is None and (methods is None or r["method"] in methods):
            out[(r["method"], r["seed"], r["eps"])] = r
    return out


def block(rows, eps, pairs):
    R = {k: v for k, v in rows.items() if abs(k[2] - eps) < 1e-12}
    seeds = sorted({k[1] for k in R})
    assert seeds and all(950 <= s <= 999 for s in seeds)
    meths = sorted({k[0] for k in R})
    full = [m for m in meths if all((m, s, eps) in R for s in seeds)]
    idx = boot_idx(len(seeds))
    res = {"eps": eps, "n_seeds": len(seeds), "methods": {}, "ratios": {}}
    for m in full:
        rr = [R[(m, s, eps)] for s in seeds]
        res["methods"][m] = {"geomean_N80_pen": float(np.exp(np.mean(np.log([r["N80_pen"] for r in rr])))),
                             "false_streams": int(sum(r["fwer_event"] for r in rr)),
                             "validity": rr[0]["validity"], "K_eval": rr[0].get("K_eval", 20)}
    for a, b in pairs:
        if a in full and b in full:
            res["ratios"][f"{a}/{b}"] = paired([R[(a, s, eps)]["N80_pen"] for s in seeds],
                                               [R[(b, s, eps)]["N80_pen"] for s in seeds], idx)
    # beta diagnostics at the stopping checkpoint (N80) for the localised methods
    for m in ("FDC-LOC", "ORACLE-LOC", "TU-LOC[K20]"):
        if m in full:
            rat, nun = [], []
            for s in seeds:
                r = R[(m, s, eps)]
                tr = r.get("beta_trace") or []
                hit = [b for b in tr if b["t"] <= r["N80_pen"]]
                if hit:
                    rat.append(hit[-1]["beta"] / hit[-1]["beta_J"])
                    nun.append(hit[-1].get("n_unresolved_e") or np.nan)
            res["methods"][m]["beta_ratio_at_stop_mean"] = float(np.mean(rat))
            res["methods"][m]["beta_ratio_at_stop_range"] = [float(np.min(rat)), float(np.max(rat))]
            res["methods"][m]["eff_union_at_stop_median"] = float(np.nanmedian(nun))
    if "FDC-LOC" in full and "FDC-BF" in full:
        res["per_stream_LOC_le_BF"] = all(R[("FDC-LOC", s, eps)]["N80_pen"] <= R[("FDC-BF", s, eps)]["N80_pen"]
                                          for s in seeds)
    return res


K20_PAIRS = [("FDC-LOC", "FDC-BF"), ("FDC-LOC", "RECT-ck-HG"), ("FDC-LOC", "HC-WoR"), ("FDC-LOC", "PJC-local"),
             ("FDC-LOC", "RECT-ck-BF"), ("ORACLE-LOC", "FDC-BF"), ("FDC-BF", "HC-WoR"), ("FDC-BF", "RECT-ck-HG"),
             ("FDC-BF", "PJC-local")]
D_PAIRS = [("TU-FDC[K20]", "HC-WoR"), ("TU-LOC[K20]", "HC-WoR"), ("FDC-BF[K77]", "HC-WoR"),
           ("TU-FDC[K20]", "RECT-ck-HG"), ("TU-FDC[K20]", "FDC-BF[K77]"), ("TU-LOC[K20]", "TU-FDC[K20]"),
           ("HC-WoR", "HC-WoR@K20"), ("TU-FDC[K20]", "FDC-BF@K20")]


def main():
    out = {}
    comp = ["FDC-BF", "RECT-ck-HG", "HC-WoR", "RECT-ck-BF", "PJC-local"]
    for task, epss in (("cr9", (0.001,)), ("x5", (0.015, 0.02, 0.03))):
        rows = load(PHG / task / "results.jsonl", comp)
        rows.update(load(P9 / task / "results.jsonl"))
        out[task] = [block(rows, e, K20_PAIRS) for e in epss]
    for task, base, epss in (("cr9d", "cr9", (0.001,)), ("x5d", "x5", (0.02,))):
        rows = load(P9 / task / "results.jsonl")
        if not rows:
            continue
        for k, r in load(PHG / base / "results.jsonl", ["HC-WoR", "FDC-BF"]).items():   # K20-grid references
            rows[(k[0] + "@K20", k[1], k[2])] = r
        out[task] = [block(rows, e, D_PAIRS) for e in epss if any(abs(k[2] - e) < 1e-12 for k in rows)]
    out["toy"] = {t: json.loads((P9 / t / "summary.json").read_text())["table"]
                  for t in ("toy", "toyd") if (P9 / t / "summary.json").exists()}

    def ub(task, eps, key):
        b = next((b for b in out.get(task, []) if abs(b["eps"] - eps) < 1e-12), None)
        v = b and b["ratios"].get(key)
        return v and v["ub95_one_sided"]

    def false_total(names):
        tot = 0
        for t in ("cr9", "x5", "cr9d", "x5d"):
            for b in out.get(t, []):
                tot += sum(b["methods"].get(n, {}).get("false_streams", 0) for n in names)
        return tot
    u_cr, u_x = ub("cr9", 0.001, "FDC-LOC/FDC-BF"), ub("x5", 0.02, "FDC-LOC/FDC-BF")
    toy_a = out["toy"].get("toy", {}).get("FDC-LOC", {}).get("false_streams")
    ga = bool(u_cr is not None and u_x is not None and ((u_cr < 0.90 and u_x <= 1.0) or (u_x < 0.90 and u_cr <= 1.0))
              and false_total(["FDC-LOC"]) == 0 and toy_a == 0)
    v_cr, v_x = ub("cr9", 0.001, "FDC-BF/HC-WoR"), ub("x5", 0.02, "FDC-BF/HC-WoR")
    toy_b = [out["toy"].get("toyd", {}).get(n, {}).get("false_streams") for n in ("TU-FDC[K20]", "TU-LOC[K20]")]
    gb = bool(v_cr is not None and v_x is not None and v_cr < 0.80 and v_x < 0.80
              and false_total(["FDC-BF", "TU-FDC[K20]", "TU-LOC[K20]", "FDC-BF[K77]"]) == 0 and toy_b == [0, 0])
    out["go"] = {"a": {"CR9_ub95": u_cr, "X5_0.02_ub95": u_x, "toy_false": toy_a,
                       "dev_false": false_total(["FDC-LOC"]), "GO": ga},
                 "b": {"primary": "TU-FDC = FDC-BF on the frozen K = 20 grid (Theorem TU-1, consequence (i))",
                       "CR9_ub95": v_cr, "X5_0.02_ub95": v_x, "toy_dense_false": toy_b,
                       "secondary_dense_TU-FDC[K20]/HC-WoR_ub95": {"CR9": ub("cr9d", 0.001, "TU-FDC[K20]/HC-WoR"),
                                                                    "X5": ub("x5d", 0.02, "TU-FDC[K20]/HC-WoR")},
                       "GO": gb}}
    (P9 / "analysis_dev.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["go"], indent=1))
    for t in ("cr9", "x5", "cr9d", "x5d"):
        for b in out.get(t, []):
            print(f"== {t} eps {b['eps']} n={b['n_seeds']} LOC<=BF all: {b.get('per_stream_LOC_le_BF')}")
            for k, v in b["ratios"].items():
                print(f"   {k:28s} {v['geomean_ratio']:.3f} UB95 {v['ub95_one_sided']:.3f} "
                      f"CI [{v['ci95_two_sided'][0]:.3f},{v['ci95_two_sided'][1]:.3f}] faster {v['frac_faster']:.2f}"
                      f" tied {v['frac_tied']:.2f}")
            for m, v in b["methods"].items():
                extra = ""
                if "beta_ratio_at_stop_mean" in v:
                    extra = (f" beta/beta_J@stop {v['beta_ratio_at_stop_mean']:.3f} "
                             f"{[round(x, 3) for x in v['beta_ratio_at_stop_range']]} "
                             f"eff_union {v['eff_union_at_stop_median']:.0f}")
                print(f"   {m:14s} gm N80 {v['geomean_N80_pen']:.0f} false {v['false_streams']} K_eval "
                      f"{v['K_eval']}{extra}")


if __name__ == "__main__":
    main()
