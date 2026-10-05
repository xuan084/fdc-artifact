"""Lock-v8 rival selection (dev data only; new file).  Writes the frozen configs of every tuned rival:

  exp/results/v8_gates/x9_configs.json   block A (X5 X9): every family tuned on X5 dev seeds 900-949 at eps 0.02
                                         (task pjc_v8/x5tune_v8), v5 rule: argmin geomean N80_pen subject to 0 false
                                         streams, tie-break geomean x12, then pre-declared grid order.
  exp/results/v8_gates/cr_configs.json   blocks B / C (CR12, CR9): PJC picks from the CR9 tune task (pjc_v8/tune,
                                         seeds 900-949, selection.json), v6 CR9 HC-WoR, own-plan rivals.
Neyman / sd^(2/3) read plans are frozen from the DEV half's finite-population cell variances of the same table (X5
dev, CR9 dev) -- outcome-free with respect to every eval half.  The v5 frozen configs (B4-bal, Hait-SW, B2-rect) are
copied unchanged for the descriptive literature rivals.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
WS = CODE.parents[1]
PJ = WS / "exp/results/pilots/pjc_v8"
OUT = WS / "exp/results/v8_gates"

import run_pjc_v8dev as R  # noqa: E402
from dsswm.baselines import pjc_bf as pj  # noqa: E402

RULE = ("argmin geomean N80_pen over the tuning seeds subject to 0 false streams; tie-break geomean x12; then the "
        "pre-declared grid order")


def geo(v):
    return float(np.exp(np.mean(np.log(np.asarray(v, dtype=float)))))


def load(task):
    key = {}
    for l in (PJ / task / "results.jsonl").read_text().splitlines():
        r = json.loads(l)
        key[(r["method"], r["seed"], r["eps"])] = r
    return list(key.values())


def pick(rows, names, seeds, eps):
    by = {}
    for r in rows:
        if abs(r["eps"] - eps) < 1e-15:
            by.setdefault(r["method"], {})[r["seed"]] = r
    table = {}
    for n in names:
        if n not in by or set(by[n]) != set(seeds):
            raise SystemExit(f"tuning incomplete for {n}: {len(by.get(n, {}))}/{len(seeds)}")
        rr = [by[n][s] for s in seeds]
        tau = rr[0]["tau_R"]
        table[n] = {"geomean_N80_over_tau": geo([r["N80_pen"] / tau for r in rr]),
                    "geomean_x12_over_tau": geo([r["x12"] / tau for r in rr]),
                    "false_streams": int(sum(r["fwer_event"] for r in rr)),
                    "share_at_tau": float(np.mean([r["N80_pen"] >= tau for r in rr]))}
    ok = [n for n in names if table[n]["false_streams"] == 0]
    best = min(ok, key=lambda n: (table[n]["geomean_N80_over_tau"], table[n]["geomean_x12_over_tau"], names.index(n)))
    return best, table


def plans(env):
    mu = env.true_mu("visit")
    s2 = mu * (1 - mu)
    return {"ney": pj.neyman_matrix(s2, None, 0.1).round(12).tolist(),
            "ney23": pj.neyman_matrix(np.maximum(s2, 0.0) ** (2.0 / 3.0), None, 0.1).round(12).tolist()}


def pjc_cfg(name):
    kw = dict(R.PJC_LOCAL.get(name) or R.PJC_MENU[name])
    kw["boundaries"] = list(kw["boundaries"])
    if "menu" in kw:
        kw["menu"] = list(kw["menu"])
    kw["pick"] = name
    return kw


def v5_frozen():
    return {k: dict(v) for k, v in json.loads((WS / "plan/prereg_lock.json").read_text())["rival_configs"]
            ["frozen_configs"].items()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # ------------------------------------------------------------------------------------------ X9 (block A)
    from dsswm.envs.x5_v8 import X5LayerEnv
    x5p = plans(X5LayerEnv("dev"))
    rows = load("x5tune_v8")
    seeds = list(range(900, 950))
    fams = {"RECT-ck-HG": ["RECT-ck-HG"] + R.RECT_PLANS, "HC-WoR": list(R.HC_GRID_V8),
            "PJC-local": list(R.PJC_LOCAL), "PJC-menu": list(R.PJC_MENU)}
    sel, tables = {}, {}
    for fam, names in fams.items():
        best, tab = pick(rows, names, seeds, 0.02)
        tables[fam] = {"pick": best, "table": tab}
        if fam == "RECT-ck-HG":
            tag = "0.5" if best == "RECT-ck-HG" else best[len("RECT-ck-HG["):-1]
            sel[fam] = {"plan": tag, "pick": best}
            if tag in ("ney", "ney23"):
                sel[fam]["alloc"] = x5p[tag]
        elif fam == "HC-WoR":
            cfg, tag = best[len("HC-WoR{"):-1].split("}[")
            sch, c, tf = cfg.split(",")
            sel[fam] = {"schedule": sch, "c": float(c), "target_frac": None if tf == "-" else float(tf), "plan": tag,
                        "pick": best}
            if tag in ("ney", "ney23"):
                sel[fam]["alloc"] = x5p[tag]
        else:
            sel[fam] = pjc_cfg(best)
    sel.update({k: v for k, v in v5_frozen().items() if k in ("B4-bal", "Hait-SW", "B2-rect")})
    descr = {m: tables[m]["table"][tables[m]["pick"]] for m in fams}
    ref = {}
    for m in ("FDC-BF", "RECT-ck-HG-live", "RECT-ck-BF", "RECT-ck-BF+box"):
        _, t = pick(rows, [m], seeds, 0.02)
        ref[m] = t[m]
    (OUT / "x9_configs.json").write_text(json.dumps({
        "layer": "X9", "table": "X5 RetailHero dev half (blinded hashed split)", "tuning_task": "pjc_v8/x5tune_v8",
        "seeds": [900, 949], "eps": 0.02, "rule": RULE, "selected": sel, "picked_summary": descr,
        "untuned_reference_on_tuning_seeds": ref, "dev_plans": x5p, "families": tables,
        "written_at": datetime.now().isoformat()}, indent=1))
    # ------------------------------------------------------------------------------------------ CR (blocks B, C)
    from dsswm.envs.pool_replay import PoolReplayEnv
    crp = plans(PoolReplayEnv("CR9", "dev"))
    s9 = json.loads((PJ / "selection.json").read_text())
    hc = json.loads((WS / "exp/results/v6_gates/hc_config_cr9.json").read_text())["selected"]
    if s9["families"]["RECT-ck-HG-plan"]["pick"] != "RECT-ck-HG" or s9["families"]["HC-WoR-plan"]["pick"] != "HC-WoR":
        raise SystemExit("CR9 own-plan picks are not 50/50: update the block-B/C design")
    cr = {"RECT-ck-HG": {"plan": "0.5", "pick": "RECT-ck-HG"},
          "HC-WoR": {"schedule": hc["schedule"], "c": hc["c"], "target_frac": hc["target_frac"], "plan": "0.5",
                     "pick": "HC-WoR (v6 CR9 config)"},
          "PJC-local": pjc_cfg(s9["families"]["PJC-local"]["pick"]),
          "PJC-menu": pjc_cfg(s9["families"]["PJC-menu"]["pick"]),
          "RECT-ck-HG[ney]": {"plan": "ney", "alloc": crp["ney"]},
          "HC-WoR[ney]": {"schedule": hc["schedule"], "c": hc["c"], "target_frac": hc["target_frac"], "plan": "ney",
                          "alloc": crp["ney"]}}
    cr.update({k: v for k, v in v5_frozen().items() if k in ("B4-bal", "Hait-SW", "B2-rect")})
    (OUT / "cr_configs.json").write_text(json.dumps({
        "layer": "CR9 / CR12 (CR9-tuned)", "tuning_task": "pjc_v8/tune (CR9 dev seeds 900-949, eps 0.001)",
        "rule": RULE, "selected": cr, "cr9_dev_plans": crp,
        "note": "CR9 own-plan tuning picked 50/50 for both RECT-ck-HG and HC-WoR (pjc_v8/selection.json); the Neyman "
                "variants are carried as descriptive block-C rivals only",
        "written_at": datetime.now().isoformat()}, indent=1))
    for fam, d in tables.items():
        t = d["table"][d["pick"]]
        print(f"X9 {fam:11s} pick {d['pick']:48s} N80/tau={t['geomean_N80_over_tau']:.4f} "
              f"x12/tau={t['geomean_x12_over_tau']:.4f}")
    for m, t in ref.items():
        print(f"X9 ref {m:30s} N80/tau={t['geomean_N80_over_tau']:.4f} x12/tau={t['geomean_x12_over_tau']:.4f}")
    print("CR PJC picks", cr["PJC-local"]["pick"], cr["PJC-menu"]["pick"])


if __name__ == "__main__":
    main()
