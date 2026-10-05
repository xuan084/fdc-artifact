"""v10 workstream 1: FDC-DP dev runner (plan/fdc_dp_theory.md).

v2 (post external reviewer FDC-DP r1, 2026-10-05): fixed code (weighted centre B1, contract B3, grid B4); full declared eps grid
(X5 0.02 / 0.03 / 0.04 / 0.05, Lenta 0.003 / 0.004 / 0.006 / 0.008); streams run to stop_k = 15 with N80 recorded at
12/15 from the same trajectory (B2); results in exp/results/pilots/fdc_dp_v2 (the v1 results in pilots/fdc_dp are
kept unchanged and are superseded); --analyse applies the pre-stated eps-selection rule mechanically (B2) and the
automated go gate including runtime and node-cap statistics.
  DEV ONLY: dev halves (seg_v10 has no eval access),
report seeds 950-999 (asserted).  NEW file.

Tasks  lenta16 | lenta32 | lenta64 | x5_16 | x5_32 | x5_64   (uplift-score segmentations, SR budget family, K = 20)
Methods FDC-DP(a), FDC-DP(b), RECT-BF-DP (matched Bennett rectangle, +box), HC-WoR-DP (hedged-capital WoR CS rectangle;
        transferred frozen v8 config at 50/50: Lenta <- CR config, X5 <- X9 config without its S = 9 Neyman matrix),
        optional FDC-DP(b)+R (hybrid, descriptive).
eps    pre-stated (plan s8): X5 0.02 primary, 0.015 secondary; Lenta 0.003 primary, 0.004 secondary.
Output exp/results/pilots/fdc_dp/<task>/results.jsonl (resumable, keyed by method/seed/eps/code sha);
       exp/results/fdcdp_<task>_PROGRESS.json / _DONE; --analyse writes exp/results/pilots/fdc_dp/analysis.json.

Usage (cwd = exp/code)
  run_fdc_dp_dev.py --task x5_64 [--workers 4] [--seeds 950-999] [--eps 0.02,0.015] [--methods ...]
  run_fdc_dp_dev.py --task all --workers 4
  run_fdc_dp_dev.py --analyse
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from datetime import datetime  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
WS = CODE.parents[1]
RES = WS / "exp/results"
OUT = RES / "pilots/fdc_dp_v2"
GATES = RES / "v8_gates"
REPORT_SEEDS = tuple(range(950, 1000))
TASKS = {f"{d}{'_' if d == 'x5' else ''}{S}": dict(data=d, S=S) for d in ("lenta", "x5") for S in (16, 32, 64)}
EPS = {"lenta": (0.003, 0.004, 0.006, 0.008), "x5": (0.02, 0.03, 0.04, 0.05)}   # declared rule grid (v10_plan)
RULE_GRID = EPS
RULE_MIN_SUCCESS = 40                     # >= 80 % of the 50 dev streams with N80_pen < tau (strict)
TAG = "fdcdp_v2"
PRIMARY_EPS = {"lenta": 0.003, "x5": 0.02}
METHODS = ("FDC-DP(b)", "FDC-DP(a)", "RECT-BF-DP", "HC-WoR-DP")
RECTS = ("RECT-BF-DP", "HC-WoR-DP")
CODE_FILES = ("run_fdc_dp_dev.py", "dsswm/baselines/fdc_dp.py", "dsswm/baselines/rect_dp.py", "dsswm/envs/seg_v10.py")
_ENV: dict = {}


def code_sha():
    per = {p: hashlib.sha256((CODE / p).read_bytes()).hexdigest() for p in CODE_FILES}
    return hashlib.sha256(json.dumps(per, sort_keys=True).encode()).hexdigest()[:16]


def hc_config(data):
    """Frozen v8 HC-WoR schedule parameters (dev-tuned on CR9 / X9), applied at the 50/50 plan."""
    key = "cr" if data == "lenta" else "x9"
    c = json.loads((GATES / f"{key}_configs.json").read_text())["selected"]["HC-WoR"]
    return {"schedule": c["schedule"], "c": float(c["c"]), "target_frac": c.get("target_frac"),
            "source": f"v8_gates/{key}_configs.json (plan forced to 0.5)"}


def make(name, data):
    from dsswm.baselines.fdc_dp import FDCDP
    from dsswm.baselines.rect_dp import HCWoRDP, RectBFDP
    if name == "FDC-DP(a)":
        return FDCDP("a")
    if name == "FDC-DP(b)":
        return FDCDP("b")
    if name == "FDC-DP(b)+R":
        return FDCDP("b", hybrid=True)
    if name == "RECT-BF-DP":
        return RectBFDP(box=True)
    if name == "HC-WoR-DP":
        c = hc_config(data)
        return HCWoRDP(c["schedule"], c["c"], c["target_frac"])
    raise KeyError(name)


def init_env(task, eps_list):
    from dsswm.baselines.fdc_dp import make_seg_ctx
    from dsswm.envs.seg_v10 import SegEnvV10, env_digest, true_opt
    T = TASKS[task]
    env = SegEnvV10(T["data"], T["S"], "dev")
    assert env.half == "dev"
    Js, mu = true_opt(env)
    ctxs = {e: make_seg_ctx(env.w, env.pool_sizes, env.problems, e, 0.05, env.checkpoints(20), env.tau_R,
                            env.replan_interval, stop_frac=1.0) for e in eps_list}
    _ENV.clear()
    _ENV.update(env=env, Js=Js, mu=mu, ctxs=ctxs, task=task, data=T["data"], digest=env_digest(env))


def job(a):
    name, seed, eps, csha = a
    from dsswm.envs.seg_v10 import run_stream_seg
    try:
        m = make(name, _ENV["data"])
        ctx = _ENV["ctxs"][eps]
        t0 = time.perf_counter()
        s, rows = run_stream_seg(_ENV["env"], m, seed, ctx, _ENV["Js"], _ENV["mu"])
        out = {"task": _ENV["task"], "method": name, "seed": int(seed), "eps": float(eps), **s,
               "sec": round(time.perf_counter() - t0, 3), "env_digest": _ENV["digest"], "code_sha": csha,
               "error": None}
        cs = getattr(m, "cert_stats", None)
        if cs:
            out.update({"bnb_nodes": int(sum(x["nodes"] for x in cs)), "bnb_calls": int(sum(x["bnb_calls"] for x in cs)),
                        "a_certs": int(sum(x["a_certified"] for x in cs)),
                        "node_limit_hits": int(sum(x["node_limit_hits"] for x in cs)),
                        "b_only_certs": int(sum(x["b_certified"] for x in cs)),
                        "n_cols_max": int(max(x["n_cols"] for x in cs)),
                        "beta": float(m.ledger["beta"]), "union_size_log": float(m.ledger["union_size_log"])})
        return out
    except Exception:  # noqa: BLE001
        return {"task": _ENV.get("task"), "method": name, "seed": int(seed), "eps": float(eps),
                "error": traceback.format_exc()}


def progress(task, done, total, metric=None):
    (RES / f"{TAG}_{task}_PROGRESS.json").write_text(json.dumps({
        "task_id": f"{TAG}_{task}", "epoch": done, "total_epochs": total, "step": done, "total_steps": total,
        "loss": None, "metric": metric or {}, "updated_at": datetime.now().isoformat()}))


def mark_done(task, status, txt):
    pf = RES / f"{TAG}_{task}_PROGRESS.json"
    fp = json.loads(pf.read_text()) if pf.exists() else {}
    (RES / f"{TAG}_{task}_DONE").write_text(json.dumps({"task_id": f"{TAG}_{task}", "status": status, "summary": txt,
                                                         "final_progress": fp,
                                                         "timestamp": datetime.now().isoformat()}))


def parse_seeds(txt):
    if txt is None:
        return list(REPORT_SEEDS)
    out = []
    for part in txt.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def run_task(task, methods, seeds, eps_list, workers):
    t_start = datetime.now()
    d = OUT / task
    d.mkdir(parents=True, exist_ok=True)
    csha = code_sha()
    rfile = d / "results.jsonl"
    done = set()
    if rfile.exists():
        keep = []
        for line in rfile.read_text().splitlines():
            x = json.loads(line)
            k = (x["method"], x["seed"], x["eps"])
            if x.get("error") is None and x.get("code_sha") == csha and k not in done:
                done.add(k)
                keep.append(line)
        rfile.write_text("".join(k + "\n" for k in keep))
    jobs = [(m, s, e, csha) for m in methods for e in eps_list for s in seeds if (m, s, e) not in done]
    jobs.sort(key=lambda j: j[0] != "HC-WoR-DP")
    total = len(jobs) + len(done)
    print(f"[{task}] {len(jobs)} jobs to run ({len(done)} done), code {csha}", flush=True)
    (RES / f"{TAG}_{task}.pid").write_text(str(os.getpid()))
    init_env(task, eps_list)
    (d / "env.json").write_text(json.dumps({**_ENV["env"].describe(), "digest": _ENV["digest"],
                                            "ledger_beta": None}, indent=1, default=str))
    errs = 0
    n_done = len(done)
    with get_context("fork").Pool(workers) as pool, open(rfile, "a") as f:
        for r in pool.imap_unordered(job, jobs, chunksize=1):
            if r.get("error"):
                errs += 1
                with open(d / "errors.log", "a") as fe:
                    fe.write(f"{r['method']} {r['seed']} {r['eps']}\n{r['error']}\n")
                continue
            f.write(json.dumps(r) + "\n")
            f.flush()
            n_done += 1
            if n_done % 5 == 0 or n_done == total:
                progress(task, n_done, total, {"runs_done": n_done, "errors": errs})
    progress(task, n_done, total, {"runs_done": n_done, "errors": errs})
    pid = RES / f"{TAG}_{task}.pid"
    if pid.exists():
        pid.unlink()
    wall = round((datetime.now() - t_start).total_seconds() / 60, 2)
    mark_done(task, "success" if errs == 0 else "errors", f"{n_done} rows, {errs} errors, wall {wall} min")
    print(f"[{task}] done: {n_done} rows, {errs} errors, wall {wall} min", flush=True)


# =============================================================================================== analysis
def _geo(v):
    return float(np.exp(np.mean(np.log(np.asarray(v, dtype=float)))))


def select_eps(R, data, seeds, grid=None, min_success=RULE_MIN_SUCCESS):
    """Pre-stated eps-selection rule (plan/v10_plan.md, written before the eps >= 0.03 runs; external reviewer r1 B2 form):
    the smallest eps on the FULL declared grid at which the dev-best rectangle (lower geometric-mean N80_pen of
    RECT-BF-DP / HC-WoR-DP at that eps; tie -> RECT-BF-DP, the parameter-free one) has strict N80_pen < tau on
    >= 40 of the 50 dev streams.  Every rectangle needs exactly the 50 unique seeds at every grid eps.  No qualifying
    eps -> 'rivals_censored' (descriptive cell).  FDC-DP's speed is never looked at."""
    grid = RULE_GRID[data] if grid is None else grid
    trace = []
    for e in sorted(grid):
        cnt = {}
        for m in RECTS:
            rr = [R.get((m, s, e)) for s in seeds]
            if any(r is None for r in rr) or len(set(seeds)) != len(seeds):
                raise ValueError(f"eps rule: {m} at eps {e} lacks the full set of {len(seeds)} unique seeds")
            cnt[m] = {"successes": int(sum(r["N80_pen"] < r["tau_R"] for r in rr)),
                      "geomean_N80_pen": _geo([r["N80_pen"] for r in rr])}
        best = min(RECTS, key=lambda m: (cnt[m]["geomean_N80_pen"], RECTS.index(m)))
        ok = cnt[best]["successes"] >= min_success
        trace.append({"eps": e, "best_rect": best, "per_rect": cnt, "qualifies": ok})
        if ok:
            return {"selected_eps": e, "rival": best, "successes": cnt[best]["successes"], "n": len(seeds),
                    "fallback": None, "trace": trace}
    return {"selected_eps": None, "rival": None, "successes": None, "n": len(seeds),
            "fallback": "rivals_censored (descriptive cell)", "trace": trace}


def analyse(tasks=None, seeds=None):
    from dsswm.stats.v6_analysis import boot_idx, paired
    csha = code_sha()
    seeds = list(REPORT_SEEDS) if seeds is None else list(seeds)
    res = {"written_at": datetime.now().isoformat(), "code_sha": csha, "version": "v2 (post external reviewer FDC-DP r1)",
           "endpoint": "N80_pen at 12/15 certified (same trajectory, run to 15/15 or tau); false streams over the "
                       "whole run", "tasks": {}, "eps_rule": {}}
    idx = boot_idx(len(seeds))
    for task in (tasks or sorted(TASKS)):
        f = OUT / task / "results.jsonl"
        if not f.exists():
            continue
        R = {}
        for line in f.read_text().splitlines():
            x = json.loads(line)
            if x.get("error") is None and x.get("code_sha") == csha:
                R[(x["method"], x["seed"], x["eps"])] = x
        data = TASKS[task]["data"]
        for eps in sorted({k[2] for k in R}):
            meths = sorted({k[0] for k in R if k[2] == eps})
            full = [m for m in meths if all((m, s, eps) in R for s in seeds)]
            out = {"eps": eps, "primary_v1": eps == PRIMARY_EPS[data], "n_seeds": len(seeds), "methods": {},
                   "ratios": {}}
            for m in full:
                rr = [R[(m, s, eps)] for s in seeds]
                cks = [v for r in rr for v in r["sec_cert_by_k"]]
                e = {"geomean_N80_pen": _geo([r["N80_pen"] for r in rr]),
                     "false_streams": int(sum(r["fwer_event"] for r in rr)),
                     "false_streams_by_k80": int(sum(r["false_by_k80"] for r in rr)),
                     "n80_lt_tau": int(sum(r["n80_lt_tau"] for r in rr)),
                     "reached_15": int(sum(r["reached_stop"] for r in rr)),
                     "validity": rr[0]["validity"], "tau_R": rr[0]["tau_R"],
                     "sec_cert_per_ck_median": float(np.median(cks)), "sec_cert_per_ck_max": float(np.max(cks)),
                     "exhaustion_at_k80_mean": {k: float(np.mean([r["exhaustion_at_k80"][k] for r in rr
                                                                  if r["exhaustion_at_k80"]]))
                                                if any(r["exhaustion_at_k80"] for r in rr) else None
                                                for k in ("all", "ctrl", "treat")}}
                if "bnb_nodes" in rr[0]:
                    calls = int(sum(r["bnb_calls"] for r in rr))
                    hits = int(sum(r["node_limit_hits"] for r in rr))
                    e.update({"node_limit_hits": hits, "bnb_calls": calls,
                              "node_hit_frac": (hits / calls) if calls else 0.0,
                              "b_only_certs": int(sum(r["b_only_certs"] for r in rr)),
                              "a_certs": int(sum(r["a_certs"] for r in rr)),
                              "bnb_nodes_total": int(sum(r["bnb_nodes"] for r in rr)), "beta": rr[0]["beta"]})
                out["methods"][m] = e
            vec = {m: [R[(m, s, eps)]["N80_pen"] for s in seeds] for m in full}
            for a, b in [("FDC-DP(b)", r) for r in RECTS] + [("FDC-DP(a)", r) for r in RECTS] + \
                        [("FDC-DP(b)", "FDC-DP(a)"), ("FDC-DP(b)+R", "FDC-DP(b)"), ("RECT-BF-DP", "HC-WoR-DP")]:
                if a in vec and b in vec:
                    out["ratios"][f"{a}/{b}"] = paired(vec[a], vec[b], idx)
            res["tasks"].setdefault(task, []).append(out)
        try:
            res["eps_rule"][task] = select_eps(R, data, seeds)
        except ValueError as ex:
            res["eps_rule"][task] = {"error": str(ex)}
    # rule-selected cells (the lock-v10 candidates)
    sel = {}
    for task, rule in res["eps_rule"].items():
        e = rule.get("selected_eps")
        if e is None:
            sel[task] = {"cell": "descriptive (rivals censored)", "rule": rule.get("fallback") or rule.get("error")}
            continue
        b = next(x for x in res["tasks"][task] if x["eps"] == e)
        r = b["ratios"].get(f"FDC-DP(b)/{rule['rival']}")
        nofalse = all(v["false_streams"] == 0 for v in b["methods"].values() if v["validity"] == "rigorous")
        sel[task] = {"eps": e, "rival": rule["rival"], "rival_successes": rule["successes"],
                     "ratio": r["geomean_ratio"], "ub95": r["ub95_one_sided"], "no_false_rigorous": nofalse}
    res["rule_selected_cells"] = sel
    # go criterion 4 as pre-stated (v1 primary eps) -- kept separate from the rule-selected analysis
    hits = []
    for task, blocks in res["tasks"].items():
        if TASKS[task]["S"] < 32:
            continue
        for b in blocks:
            if not b["primary_v1"]:
                continue
            rr = {m: b["ratios"].get(f"FDC-DP(b)/{m}") for m in RECTS}
            rects = [m for m in RECTS if rr[m] is not None]
            if not rects:
                continue
            best = min(rects, key=lambda m: (b["methods"][m]["geomean_N80_pen"], RECTS.index(m)))
            nofalse = all(v["false_streams"] == 0 for v in b["methods"].values() if v["validity"] == "rigorous")
            hits.append({"task": task, "eps": b["eps"], "rect": best, "ub95": rr[best]["ub95_one_sided"],
                         "no_false": nofalse, "pass": bool(rr[best]["ub95_one_sided"] < 0.80 and nofalse)})
    res["go_criterion_4_v1_primary_eps"] = {
        "rule": "UB95(FDC-DP(b)/dev-best of the two named rectangles) < 0.80 at S >= 32 on >= 1 table at the v1 "
                "primary eps (X5 0.02, Lenta 0.003), 0 false streams", "cells": hits,
        "pass": any(h["pass"] for h in hits) if hits else None}
    res["go_criterion_4_rule_selected"] = {
        "rule": "same criterion at the rule-selected eps (S >= 32)",
        "cells": {t: v for t, v in sel.items() if TASKS[t]["S"] >= 32},
        "pass": any(v.get("ub95", 9) < 0.80 and v.get("no_false_rigorous") for t, v in sel.items()
                    if TASKS[t]["S"] >= 32)}
    # go criterion 3 (automated, plan/fdc_dp_theory.md s10.3): S = 64, both tables
    g3 = []
    for task, bl in res["tasks"].items():
        if TASKS[task]["S"] != 64:
            continue
        for b in bl:
            mb, ma = b["methods"].get("FDC-DP(b)"), b["methods"].get("FDC-DP(a)")
            if mb is None:
                continue
            g3.append({"task": task, "eps": b["eps"], "b_median": mb["sec_cert_per_ck_median"],
                       "b_max": mb["sec_cert_per_ck_max"], "a_median": ma["sec_cert_per_ck_median"] if ma else None,
                       "node_hit_frac": mb.get("node_hit_frac"), "node_limit_hits": mb.get("node_limit_hits"),
                       "bnb_calls": mb.get("bnb_calls"),
                       "pass": bool(mb["sec_cert_per_ck_median"] < 10 and mb["sec_cert_per_ck_max"] < 30
                                    and (ma is None or ma["sec_cert_per_ck_median"] < 2)
                                    and (mb.get("node_hit_frac") or 0.0) <= 0.01)})
    res["go_criterion_3"] = {"rule": "S = 64, every table and eps: FDC-DP(b) median < 10 s and max < 30 s per "
                                     "checkpoint (all 15 problems), FDC-DP(a) median < 2 s, node-limit hits <= 1 % "
                                     "of B&B calls", "cells": g3, "pass": all(x["pass"] for x in g3) if g3 else None}
    res["b_vs_a"] = {t: {str(b["eps"]): {"b_only_certs": b["methods"].get("FDC-DP(b)", {}).get("b_only_certs"),
                                         "ratio_b_over_a": (b["ratios"].get("FDC-DP(b)/FDC-DP(a)") or {}).get(
                                             "geomean_ratio")} for b in bl} for t, bl in res["tasks"].items()}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "analysis.json").write_text(json.dumps(res, indent=1))
    for task, bl in res["tasks"].items():
        for b in bl:
            g = {m: (round(v["geomean_N80_pen"]), v["n80_lt_tau"], v["false_streams"]) for m, v in b["methods"].items()}
            r = {k: (round(v["geomean_ratio"], 3), round(v["ub95_one_sided"], 3)) for k, v in b["ratios"].items()
                 if k.startswith("FDC-DP(b)/")}
            print(task, b["eps"], g, r)
    print("rule", json.dumps(sel, default=str))
    print("go3", res["go_criterion_3"]["pass"], "go4(v1 eps)", res["go_criterion_4_v1_primary_eps"]["pass"],
          "go4(rule)", res["go_criterion_4_rule_selected"]["pass"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", help="task name, comma list, or 'all'")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seeds", default=None, help="e.g. 950-999 or 950,951 (dev report seeds only)")
    ap.add_argument("--eps", default=None, help="comma list (default: pre-stated per data set)")
    ap.add_argument("--methods", default=None)
    ap.add_argument("--analyse", action="store_true")
    args = ap.parse_args()
    seeds = parse_seeds(args.seeds)
    if not all(950 <= s <= 999 for s in seeds):
        raise SystemExit("dev report seeds 950-999 only")
    if args.analyse:
        analyse(seeds=None if args.seeds is None else seeds)
        return
    workers = max(1, min(4, args.workers))
    tasks = sorted(TASKS) if args.task == "all" else args.task.split(",")
    for task in tasks:
        if task not in TASKS:
            raise SystemExit(f"unknown task {task}; choose from {sorted(TASKS)}")
        data = TASKS[task]["data"]
        eps_list = tuple(float(x) for x in args.eps.split(",")) if args.eps else EPS[data]
        methods = args.methods.split(",") if args.methods else list(METHODS)
        run_task(task, methods, seeds, eps_list, workers)


if __name__ == "__main__":
    main()
