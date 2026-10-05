"""Round-8 feasibility runner (methodologist, 2026-10-04): PJC-BF phased adaptive joint rival, matched Bennett-rectangle,
own-plan rectangles.  DEV SEEDS ONLY (900-999, asserted); dev halves only (CR9 / CR12 dev, LR9 dev); no eval seed, no
eval half, no locked file is touched.  New file; uses the frozen v5 runner / v6 harness / v7 methods unchanged.

Tasks
  toy    FWER on the r4 near-tie toy (population seeds 900-919 x perm 0-9 = 200 streams, eps 0.02) for every new rigorous
         method + the invalid NAIVE-joint power control.
  tune   CR9 dev, seeds 900-949, eps 0.001: the full pre-declared grids (PJC local, PJC menu, rectangle / HC-WoR plans,
         RECT-ck-BF).  Selection per family by the v5 rule: argmin geomean N80_pen subject to 0 false streams.
  dev    CR9 dev, seeds 950-999, eps 0.001: references + every grid member (all reported; tuned picks flagged by
         analyze_pjc_v8dev.py from the tune task only).
  lenta  LR9 dev half, seeds 950-999, eps 0.002 / 0.003 / 0.004: references + tuned picks (descriptive).
  cr12   CR12 dev half, seeds 950-969, eps 0.001: scale feasibility (FDC-BF, FDC, RECT-ck-HG, HC-WoR, PJC picks).
  x5     X5 RetailHero X9 DEV half (blinded hashed split), seeds 900-919, eps 0.005-0.015: fresh-table feasibility.
Rows as in run_fdc_bet.job (N80_pen primary, x12 diagnostic, schedule digest, timing) plus the PJC menu path / plans.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
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
OUT = WS / "exp/results/pilots/pjc_v8"
DEV = set(range(900, 1000))
N_WORKERS = int(os.environ.get("PJC_WORKERS", "8"))

import run_fdc_bet as rfb  # noqa: E402
from dsswm.baselines import pjc_bf as pj  # noqa: E402
from dsswm.streams import frontier as fr  # noqa: E402
from dsswm.streams.frontier_runner import build_ctx, true_policy_values  # noqa: E402
from dsswm.streams.frontier_runner_v6 import run_stream_v6  # noqa: E402

REFS = ["FDC-BF", "FDC-MR[front3]", "FDC", "RECT-ck-HG", "RECT-ck-HG-live", "HC-WoR"]

# ---------------------------------------------------------------- pre-declared grids (fixed before any tune run)
PJC_LOCAL = {}
for b in ((0,), (2,), (4,), (2, 5, 8, 11)):
    for rule in ("neyman", "half"):
        for rect in (False, True):
            m = pj.PJCBF(mode="local", boundaries=b, rule=rule, rect=rect)
            PJC_LOCAL[m.name] = dict(mode="local", boundaries=b, rule=rule, rect=rect)
# external reviewer v8 review item 3: the no-reset member (boundaries = (), deterministic 50/50 tracking, one phase) is a valid
# member of the local family and is added to the grid so that the rival is not handicapped by a forced reset.
_m0 = pj.PJCBF(mode="local", boundaries=(), rule="half", rect=False)
PJC_LOCAL[_m0.name] = dict(mode="local", boundaries=(), rule="half", rect=False)
PJC_MENU = {}
for b in ((0,), (2,), (2, 6)):
    for menu in ((0.4, 0.5), (0.4, 0.45, 0.5), (0.35, 0.5, 0.65)):
        m = pj.PJCBF(mode="menu", boundaries=b, rule="proj", menu=menu)
        PJC_MENU[m.name] = dict(mode="menu", boundaries=b, rule="proj", menu=menu)
_m = pj.PJCBF(mode="menu", boundaries=(2,), rule="proj", menu=(0.4, 0.45, 0.5), rect=True)
PJC_MENU[_m.name] = dict(mode="menu", boundaries=(2,), rule="proj", menu=(0.4, 0.45, 0.5), rect=True)
RECT_PLANS = [f"RECT-ck-HG[{s:g}]" for s in (0.3, 0.4, 0.6, 0.7)] + ["RECT-ck-HG[ney]", "RECT-ck-HG[ney23]"]
HC_PLANS = [f"HC-WoR[{s:g}]" for s in (0.4, 0.6)] + ["HC-WoR[ney]", "HC-WoR[ney23]"]
MATCHED = ["RECT-ck-BF", "RECT-ck-BF+box"]
DESCR = ["FDC-BF[ney]"]
HC_X5 = ["HC-WoR{prpl,0.75,-}"] + [f"HC-WoR{{nstar,0.75,{tf:g}}}" for tf in (0.2, 0.4, 0.6, 0.8, 1.0)]
GRID = list(PJC_LOCAL) + list(PJC_MENU) + RECT_PLANS + HC_PLANS + MATCHED + DESCR
# v8 X5 dev tuning grid (authors approval 2026-10-04): HC-WoR v6 schedule grid extended by the x5tune target
# fractions, crossed with the rectangle read-plan grid {0.3, 0.4, 0.5, 0.6, 0.7, ney, ney23}.
PLAN_TAGS = ("0.5", "0.3", "0.4", "0.6", "0.7", "ney", "ney23")
HC_SCHED_V8 = ["prpl,0.5,-", "prpl,0.75,-"] + [f"nstar,0.75,{tf:g}" for tf in (0.1, 0.2, 0.35, 0.5, 0.6, 0.8, 1.0)]
HC_GRID_V8 = [f"HC-WoR{{{c}}}[{t}]" for c in HC_SCHED_V8 for t in PLAN_TAGS]
X5_TUNE_V8 = (["FDC-BF", "RECT-ck-HG", "RECT-ck-HG-live"] + RECT_PLANS + list(PJC_LOCAL) + list(PJC_MENU) + MATCHED
              + HC_GRID_V8)


def _hc_cfg(layer):
    g = WS / "exp/results/v6_gates" / ("rival_configs_lr9.json" if layer == "LR9" else "hc_config_cr9.json")
    sel = json.loads(g.read_text())["selected"]
    return sel["HC-WoR"] if layer == "LR9" else sel


def _plan_matrix(tag):
    """Frozen per-cell plans from the DEV half's finite-population variances (truth of the dev half; for an eval block
    these would come from the dev half, i.e. outcome-free w.r.t. eval)."""
    s2 = _ENV["sigma2"]
    if tag == "ney":
        return pj.neyman_matrix(s2, None, 0.1)
    if tag == "ney23":
        return pj.neyman_matrix(np.maximum(s2, 0.0) ** (2.0 / 3.0), None, 0.1)   # p ~ sd^(2/3) (radius-sum optimum)
    raise ValueError(tag)


def make(name, layer):
    if name in PJC_LOCAL or name in PJC_MENU:
        kw = PJC_LOCAL.get(name) or PJC_MENU[name]
        return pj.PJCBF(**kw, name=name)
    if name in ("RECT-ck-BF", "RECT-ck-BF+box"):
        return pj.RectCkBF(box=name.endswith("+box"))
    if name.startswith("RECT-ck-HG[") and name not in ("RECT-ck-HG",):
        tag = name[len("RECT-ck-HG["):-1]
        if tag in ("ney", "ney23"):
            return pj.RectCkHGPlan(alloc=_plan_matrix(tag), name=name)
        return pj.RectCkHGPlan(share=float(tag), name=name)
    if name.startswith("HC-WoR{") and name.endswith("]"):   # HC-WoR{schedule,c,target_frac}[plan] (v8 X5 grid)
        cfg, tag = name[len("HC-WoR{"):-1].split("}[")
        sch, c, tf = cfg.split(",")
        tf = None if tf == "-" else float(tf)
        if tag in ("ney", "ney23"):
            return pj.HCWoRPlan(sch, float(c), tf, alloc=_plan_matrix(tag), name=name)
        return pj.HCWoRPlan(sch, float(c), tf, share=float(tag), name=name)
    if name.startswith("HC-WoR{"):                          # explicit HC-WoR config: HC-WoR{schedule,c,target_frac}
        sch, c, tf = name[len("HC-WoR{"):-1].split(",")
        from dsswm.baselines.wor_betting_v6 import HCWoRRect
        return HCWoRRect(sch, float(c), None if tf == "-" else float(tf), name=name)
    if name.startswith("HC-WoR["):
        tag = name[len("HC-WoR["):-1]
        p = _hc_cfg(layer)
        if tag in ("ney", "ney23"):
            return pj.HCWoRPlan(p["schedule"], p["c"], p["target_frac"], alloc=_plan_matrix(tag), name=name)
        return pj.HCWoRPlan(p["schedule"], p["c"], p["target_frac"], share=float(tag), name=name)
    if name.startswith("FDC-BF[") :
        tag = name[len("FDC-BF["):-1]
        return pj.FDCBetPlan(alloc=_plan_matrix(tag), name=name) if tag.startswith("ney") else \
            pj.FDCBetPlan(share=float(tag), name=name)
    if name == "HC-WoR" and layer in ("CR12", "X9"):
        from dsswm.baselines.wor_betting_v6 import HCWoRRect
        p = _hc_cfg("CR9")
        return HCWoRRect(p["schedule"], p["c"], p["target_frac"])
    return rfb.make(name, "LR9" if layer == "LR9" else "CR9")


TASKS = {
    "tune": dict(layer="CR9", seeds=list(range(900, 950)), eps=(0.001,), methods=["FDC-BF", "RECT-ck-HG", "HC-WoR"] + GRID),
    "dev": dict(layer="CR9", seeds=list(range(950, 1000)), eps=(0.001,), methods=REFS + GRID),
    "lenta": dict(layer="LR9", seeds=list(range(950, 1000)), eps=(0.002, 0.003, 0.004), methods=None),
    "cr12": dict(layer="CR12", seeds=list(range(950, 970)), eps=(0.001,), methods=None),
    "x5": dict(layer="X9", seeds=list(range(900, 920)), eps=(0.005, 0.0075, 0.01, 0.015), methods=None),
    # X5 wide-eps feasibility: HC-WoR re-tuned on X5 dev 900-929 (grid HC_X5), report 930-949
    "x5tune": dict(layer="X9", seeds=list(range(900, 930)), eps=(0.015, 0.02, 0.03), methods=None),
    "x5rep": dict(layer="X9", seeds=list(range(930, 950)), eps=(0.015, 0.02, 0.03), methods=None),
    # v8: full X5 dev tuning of every rival family on dev seeds 900-949 at the primary eps 0.02 (v5 selection rule)
    "x5tune_v8": dict(layer="X9", seeds=list(range(900, 950)), eps=(0.02,), methods=X5_TUNE_V8),
}
_ENV: dict = {}


def init_env(layer, eps_list):
    if layer in ("CR9", "CR12"):
        from dsswm.envs.pool_replay import PoolReplayEnv
        env, probs, outcome = PoolReplayEnv(layer, "dev"), fr.cr_problems("visit"), "visit"
    elif layer == "X9":
        from dsswm.envs.x5_v8 import X5LayerEnv
        env, probs, outcome = X5LayerEnv("dev"), fr.cr_problems("visit"), "visit"
    else:
        from dsswm.envs.lenta_v6 import LentaLayerEnv, lr9_problems
        env, probs, outcome = LentaLayerEnv("dev"), lr9_problems(), "response_att"
    assert getattr(env, "half", "dev") == "dev"
    ctx0 = build_ctx(env, probs, eps_list[0])
    J = true_policy_values(ctx0.pols, env.w, env.true_mu(outcome))
    Js = np.array([J[ctx0.feas[q]].max() for q in range(ctx0.Q)])
    mu = env.true_mu(outcome)
    _ENV.update(env=env, probs=probs, outcome=outcome, J=J, Js=Js, layer=layer, sigma2=mu * (1 - mu),
                ctxs={e: build_ctx(env, probs, e) for e in eps_list})


def job(a):
    name, seed, eps = a
    assert seed in DEV
    ctx = _ENV["ctxs"][eps]
    try:
        m = make(name, _ENV["layer"])
        t0 = time.perf_counter()
        s, rows = run_stream_v6(_ENV["env"], m, seed, ctx.problems, ctx.eps, outcome=_ENV["outcome"], ctx=ctx,
                                J_true=_ENV["J"], J_star=_ENV["Js"], keep_U=True)
        ck = ctx.checkpoints
        tau = int(ctx.tau_R)
        curve = [r["n_cert"] for r in rows]
        nfalse = [r["n_false"] for r in rows]
        k12 = next((k for k, c in enumerate(curve) if c >= ctx.stop_k), None)
        n80_pen = int(ck[k12]) if (k12 is not None and nfalse[k12] == 0) else tau
        xx = rfb.x12(rows, ck, ctx.eps, ctx.stop_k)
        out = {"method": name, "validity": m.validity, "seed": int(seed), "eps": float(eps), "N80_pen": n80_pen,
               "k80": k12, "x12": xx if xx is not None else float(tau), "fwer_event": bool(s["fwer_event"]),
               "n_false": int(s["n_false"]), "n_cert": int(s["n_cert"]), "cert_k": s["cert_k"],
               "billing_ok": bool(s["billing_ok"]), "schedule_digest": s["schedule_digest"], "tau_R": tau,
               "alloc_kind": m.alloc_kind, "sec": round(time.perf_counter() - t0, 3), "error": None}
        if isinstance(m, pj.PJCBF):
            out["path"] = list(m.path)
            out["beta"] = m.ledger["beta"]
            out["ctrl_share_plans"] = [round(float(np.mean(p[:, 0])), 4) for p in m.plans]
            last = rows[-1]["n_cells"] if rows else None
            out["n_cells_end"] = last
        return out
    except Exception:  # noqa: BLE001
        return {"method": name, "seed": int(seed), "eps": float(eps), "error": traceback.format_exc()}


def toy_job(a):
    name, sd = a
    from dsswm.tests.test_baselines_r4a import PROBS, TOY_EPS, _toy_population
    env = _toy_population(sd)
    mu = env.true_mu("visit")
    _ENV["sigma2"] = mu * (1 - mu)
    out = []
    for p in range(10):
        m = rfb.make(name, "toy") if name == "NAIVE-joint" else make(name, "toy")
        s, _ = run_stream_v6(env, m, 1000 * sd + p, PROBS, TOY_EPS, keep_U=False)
        out.append({"method": name, "validity": m.validity, "population_seed": sd, "perm_seed": s["perm_seed"],
                    "fwer_event": bool(s["fwer_event"]), "n_cert": s["n_cert"], "n_false": s["n_false"],
                    "reached_stop": s["reached_stop"], "N80_over_tau": s["N80"] / env.tau_R,
                    "billing_ok": bool(s["billing_ok"]), "reselected": s["reselected"], "sec": s["sec_total"],
                    "path": list(getattr(m, "path", []))})
    return out


def run_pool(fn, jobs):
    with get_context("fork").Pool(N_WORKERS) as pool:
        for r in pool.imap_unordered(fn, jobs, chunksize=1):
            yield r


def geo(v):
    return float(np.exp(np.mean(np.log(np.asarray(v, dtype=float)))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["toy"] + sorted(TASKS))
    ap.add_argument("--methods", default=None, help="semicolon-separated list (default: task list)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = datetime.now()
    if args.task == "toy":
        from scipy import stats
        default = ["PJC-BF[local,b=0,neyman]", "PJC-BF[local,b=0,neyman,R]", "PJC-BF[local,b=2,5,8,11,neyman]",
                   "PJC-BF-M[b=0,menu=0.35/0.5/0.65,proj]", "PJC-BF-M[b=2,6,menu=0.35/0.5/0.65,proj]",
                   "RECT-ck-BF", "RECT-ck-BF+box", "NAIVE-joint"]
        names = args.methods.split(";") if args.methods else default
        # toy: K = 20 checkpoints from n_min = 200; boundaries are checkpoint indices, all valid
        rows = [r for ch in run_pool(toy_job, [(n, sd) for n in names for sd in range(900, 920)]) for r in ch]
        d = OUT / "toy"
        d.mkdir(exist_ok=True)
        with open(d / "results.jsonl", "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        table = {}
        for n in names:
            rr = [r for r in rows if r["method"] == n]
            ev = sum(r["fwer_event"] for r in rr)
            table[n] = {"n_streams": len(rr), "false_streams": ev, "false_certs": sum(r["n_false"] for r in rr),
                        "cp_upper": 1.0 if ev >= len(rr) else float(stats.beta.ppf(0.95, ev + 1, len(rr) - ev)),
                        "reached_stop": sum(r["reached_stop"] for r in rr),
                        "early_stop_lt_half_tau": sum(r["reached_stop"] and r["N80_over_tau"] < 0.5 for r in rr),
                        "geomean_N80_over_tau": geo([r["N80_over_tau"] for r in rr]),
                        "nondefault_path_streams": sum(any(x != 0 for x in r["path"]) for r in rr),
                        "billing_ok_all": all(r["billing_ok"] for r in rr), "validity": rr[0]["validity"]}
        prev = json.loads((d / "summary.json").read_text()) if (d / "summary.json").exists() else {"table": {}}
        prev["table"].update(table)
        prev.update({"task": "toy", "written_at": datetime.now().isoformat(),
                     "toy": "r4 near-tie toy, population seeds 900-919 x perm 0-9, eps 0.02"})
        (d / "summary.json").write_text(json.dumps(prev, indent=1))
        print(json.dumps({k: (v["false_streams"], round(v["geomean_N80_over_tau"], 4)) for k, v in table.items()}))
        return
    T = TASKS[args.task]
    assert all(s in DEV for s in T["seeds"])
    names = args.methods.split(";") if args.methods else T["methods"]
    if names is None:
        raise SystemExit("--methods required for this task")
    d = OUT / args.task
    d.mkdir(exist_ok=True)
    rfile = d / "results.jsonl"
    done = set()
    if rfile.exists():
        for l in rfile.read_text().splitlines():
            x = json.loads(l)
            done.add((x["method"], x["seed"], x["eps"]))
    jobs = [(n, s, e) for n in names for e in T["eps"] for s in T["seeds"] if (n, s, e) not in done]
    jobs.sort(key=lambda j: not j[0].startswith(("HC-WoR", "PJC")))
    print(f"[{args.task}] {len(jobs)} jobs", flush=True)
    errs = 0
    init_env(T["layer"], tuple(T["eps"]))
    with open(rfile, "a") as f:
        for r in run_pool(job, jobs):
            if r.get("error"):
                errs += 1
                with (d / "errors.log").open("a") as g:
                    g.write(f"{r['method']} {r['seed']} {r['eps']}\n{r['error']}\n")
                continue
            f.write(json.dumps(r) + "\n")
            f.flush()
    print(f"[{args.task}] done, {errs} errors, wall {(datetime.now() - t0).total_seconds() / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
