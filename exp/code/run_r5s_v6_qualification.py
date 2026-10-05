"""v6 addendum: baseline qualification of the new rivals (same procedure as r5_baseline_qualification, criteria (i)-(v)).

Toy FWER: r4 near-tie toy populations (test_baselines_r4a._toy_population), population seeds 900-919 x permutation
0-9 = 200 streams, eps = 0.02, 5 problems, stop at 4/5, adaptive exhaustion included.  Gate (iv): 0/200 false streams
(one-sided CP upper bound 0.0149) for every new rigorous configuration (RECT-ck-HG, RECT-ck-HG-live, RECT-ck-Bern, every HC-WoR grid
point); the power control NAIVE-control (plug-in GLR, beta = ln(1/delta)) must show false streams on the same harness.
Also runs the v6 unit tests.  Output: exp/results/pilots/v6_baseline_qualification/summary.json + toy_fwer.csv.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import csv  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from datetime import datetime  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402

from scipy import stats  # noqa: E402

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
WS = CODE.parents[1]
OUT = WS / "exp/results/pilots/v6_baseline_qualification"


def configs():
    from dsswm.baselines.wor_betting_v6 import HC_GRID
    c = [("RECT-ck-HG", "RECT-ck-HG", {}), ("RECT-ck-HG-live", "RECT-ck-HG-live", {}),
         ("RECT-ck-Bern", "RECT-ck-Bern", {})]
    for g in HC_GRID:
        lab = f"HC-WoR[{g['schedule']},c={g['c']},t*={g['target_frac']}]"
        c.append((lab, "HC-WoR", dict(g)))
    c.append(("B4-bal (v5 reference)", "B4-bal", {"share": 0.5}))
    c.append(("NAIVE-control", None, {}))
    return c


def _mk(base, kw):
    if base is None:
        from dsswm.baselines.combgame_joint import CombGameJoint
        return CombGameJoint("fav", beta_override=math.log(1 / 0.05), name="NAIVE-control")
    if base == "RECT-ck-HG":
        from dsswm.baselines.rect_v6 import RectCkHG
        return RectCkHG()
    if base == "RECT-ck-HG-live":
        from dsswm.baselines.rect_v6 import RectCkHGLive
        return RectCkHGLive()
    if base == "RECT-ck-Bern":
        from dsswm.baselines.rect_v6 import RectCkBern
        return RectCkBern()
    if base == "HC-WoR":
        from dsswm.baselines.wor_betting_v6 import HCWoRRect
        return HCWoRRect(kw["schedule"], kw["c"], kw["target_frac"])
    from dsswm.streams import r5_registry as reg
    return reg.make_method(base, **kw)


def _toy_job(args):
    label, base, kw, sd = args
    from dsswm.streams.frontier_runner_v6 import run_stream_v6
    from dsswm.tests.test_baselines_r4a import PROBS, TOY_EPS, _toy_population
    env = _toy_population(sd)
    out = []
    for p in range(10):
        m = _mk(base, kw)
        s, _ = run_stream_v6(env, m, 1000 * sd + p, PROBS, TOY_EPS, keep_U=False)
        out.append({"config": label, "method": m.name, "validity": m.validity, "population_seed": sd,
                    "perm_seed": s["perm_seed"], "fwer_event": s["fwer_event"], "n_cert": s["n_cert"],
                    "n_false": s["n_false"], "reached_stop": s["reached_stop"], "N80_over_tau": s["N80"] / env.tau_R,
                    "billing_ok": s["billing_ok"], "skipped": s["skipped"], "reselected": s["reselected"],
                    "sec": s["sec_total"]})
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = datetime.now()
    jobs = [(lab, b, kw, sd) for lab, b, kw in configs() for sd in range(900, 920)]
    with get_context("fork").Pool(4) as pool:
        rows = [r for chunk in pool.imap_unordered(_toy_job, jobs) for r in chunk]
    with open(OUT / "toy_fwer.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["config"], r["population_seed"], r["perm_seed"])))
    table = {}
    for lab, b, kw in configs():
        rr = [r for r in rows if r["config"] == lab]
        ev = sum(r["fwer_event"] for r in rr)
        n = len(rr)
        table[lab] = {"n_streams": n, "false_streams": ev, "false_certs": sum(r["n_false"] for r in rr),
                      "cp_upper": 1.0 if ev >= n else float(stats.beta.ppf(0.95, ev + 1, n - ev)),
                      "reached_stop": sum(r["reached_stop"] for r in rr),
                      "early_stop_lt_half_tau": sum(r["reached_stop"] and r["N80_over_tau"] < 0.5 for r in rr),
                      "with_reselection": sum(r["reselected"] > 0 for r in rr),
                      "billing_ok_all": all(r["billing_ok"] for r in rr), "validity": rr[0]["validity"]}
    rig = {k: v for k, v in table.items() if k != "NAIVE-control"}
    gate = all(v["false_streams"] == 0 and v["billing_ok_all"] for v in rig.values())
    power = table["NAIVE-control"]["false_streams"] > 0   # detects a broken certificate; NOT evidence of FWER > .05
    pt = subprocess.run([sys.executable, "-m", "pytest", "-q", "dsswm/tests/test_v6_addendum.py"], cwd=str(CODE),
                        capture_output=True, text=True, timeout=3000)
    summary = {"task_id": "v6_baseline_qualification", "written_at": datetime.now().isoformat(),
               "started_at": t0.isoformat(), "toy": "r4 near-tie toy, population seeds 900-919 x perm 0-9, eps 0.02",
               "table": table, "gate_iv_all_new_rigorous_0_of_200": gate, "power_control_detects": power,
               "pytest_tail": pt.stdout.strip().splitlines()[-1:], "pytest_returncode": pt.returncode}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({"gate": gate, "power": power, "pytest": summary["pytest_tail"]}))


if __name__ == "__main__":
    main()
