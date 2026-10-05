"""FDC-bet exploration runner (R2, round 5 follow-up).  DEV SEEDS ONLY (900-999); eval seeds and Lenta eval outcomes
are never touched (asserted).  New file; uses the frozen v5 runner and the v6 harness (HC-WoR) unchanged.

Tasks
  toy        FWER qualification: r4 near-tie toy (population seeds 900-919 x perm 0-9 = 200 streams, eps 0.02),
             every FDC-bet variant + FDC + a power control (invalid certificate) -> exp/results/pilots/fdc_bet/toy
  tune       CR9 dev, tuning seeds 900-949, eps 0.001: every variant (selection of the MR weight profile rho and the
             delta split happens HERE, by the lock-v5 rule: argmin geomean N80_pen, 0 false streams)
  dev        CR9 dev, seeds 950-999 (= v6 block-A dev pilot seeds), eps 0.001: FDC, RECT-ck-HG, RECT-ck-HG-live,
             HC-WoR (frozen v6 config) + every FDC-bet variant (ALL reported, no silent selection)
  lenta      LR9 DEV half, seeds 950-999, eps 0.002 / 0.003 / 0.004, same method list (HC-WoR: frozen LR9 config)
Each row: N80_pen (primary, as in run_r5s_v6.job), schedule digest, plus the diagnostic 'x12' = log-linear interpolated
arrival count at which the 12th problem's sticky U_q crosses eps (computed from the U trajectory up to the stop).
Workers: 6 (host limit), BLAS threads 1.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
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
OUT = WS / "exp/results/pilots/fdc_bet"
N_WORKERS = 6
DEV = set(range(900, 1000))

from dsswm.streams import frontier as fr  # noqa: E402
from dsswm.streams.frontier_runner import build_ctx, true_policy_values  # noqa: E402
from dsswm.streams.frontier_runner_v6 import run_stream_v6  # noqa: E402

RIVALS = ["FDC", "RECT-ck-HG", "RECT-ck-HG-live", "HC-WoR"]
BET_VARIANTS = ["FDC-re", "FDC-FPC", "FDC-BF-L2", "FDC-BF", "FDC-KLF", "FDC-KLF+R"]
MR_VARIANTS = [f"FDC-MR[{r}]" for r in ("uniform", "geo2", "inv_d2", "front3")] + \
              ["FDC-MR[uniform,nopart]", "FDC-MR[uniform,norect]", "FDC-MR[uniform,split.025]"] + \
              [f"FDC-MR[{r}]" for r in ("front2", "front3x", "mixF50", "mixF30", "mixF50x")]   # tuning round 2


def make(name, layer):
    if name == "FDC":
        from dsswm.baselines.fdc import FDCMethod
        return FDCMethod()
    if name == "RECT-ck-HG":
        from dsswm.baselines.rect_v6 import RectCkHG
        return RectCkHG()
    if name == "RECT-ck-HG-live":
        from dsswm.baselines.rect_v6 import RectCkHGLive
        return RectCkHGLive()
    if name == "HC-WoR":
        from dsswm.baselines.wor_betting_v6 import HCWoRRect
        g = WS / "exp/results/v6_gates" / ("hc_config_cr9.json" if layer == "CR9" else "rival_configs_lr9.json")
        sel = json.loads(g.read_text())["selected"]
        p = sel if layer == "CR9" else sel["HC-WoR"]
        return HCWoRRect(p["schedule"], p["c"], p["target_frac"])
    if name == "NAIVE-joint":                      # power control: invalid (no union, plug-in point box)
        return _naive()
    if name in BET_VARIANTS:
        from dsswm.baselines.fdc_bet import make_variant
        return make_variant(name)
    if name.startswith("FDC-MR["):
        from dsswm.baselines.fdc_mr import FDCMR
        opts = name[len("FDC-MR["):-1].split(",")
        kw = {"rho": opts[0]}
        for o in opts[1:]:
            if o == "nopart":
                kw["partition"] = False
            elif o == "norect":
                kw["rect"] = False
            elif o == "split.025":
                kw["split"] = (0.025, 0.025)
            else:
                raise ValueError(o)
        return FDCMR(name=name, **kw)
    raise ValueError(name)


def _naive():
    from dsswm.baselines.fdc_bet import FDCBet

    class Naive(FDCBet):
        validity = "none"

        def setup(self, ctx):
            super().setup(ctx)
            self.ledger["beta"] = math.log(1.0 / 0.05)       # no union over directions / checkpoints

        def _box(self, ctx, st):
            mu = st.mu_hat
            return mu.copy(), mu.copy()                    # plug-in variance (no variance event)
    return Naive(kind="min", var_box="HG", fpc=True, rect=False, name="NAIVE-joint")


TASKS = {
    "tune": dict(layer="CR9", seeds=list(range(900, 950)), eps=(0.001,), methods=BET_VARIANTS + MR_VARIANTS),
    "dev": dict(layer="CR9", seeds=list(range(950, 1000)), eps=(0.001,), methods=RIVALS + BET_VARIANTS + MR_VARIANTS),
    "lenta": dict(layer="LR9", seeds=list(range(950, 1000)), eps=(0.002, 0.003, 0.004),
                  methods=RIVALS + BET_VARIANTS + MR_VARIANTS),
}
_ENV: dict = {}


def init_env(layer, eps_list):
    if layer == "CR9":
        from dsswm.envs.pool_replay import PoolReplayEnv
        env, probs, outcome = PoolReplayEnv("CR9", "dev"), fr.cr_problems("visit"), "visit"
    else:
        from dsswm.envs.lenta_v6 import LentaLayerEnv, lr9_problems
        env, probs, outcome = LentaLayerEnv("dev"), lr9_problems(), "response_att"
    assert getattr(env, "half", "dev") == "dev"
    ctx0 = build_ctx(env, probs, eps_list[0])
    J = true_policy_values(ctx0.pols, env.w, env.true_mu(outcome))
    Js = np.array([J[ctx0.feas[q]].max() for q in range(ctx0.Q)])
    _ENV.update(env=env, probs=probs, outcome=outcome, J=J, Js=Js, ctxs={e: build_ctx(env, probs, e) for e in eps_list},
                layer=layer)


def x12(rows, ck, eps, stop_k):
    """Interpolated arrival count at which the stop_k-th problem's sticky U crosses eps (log-log interpolation)."""
    if not rows:
        return None
    U = np.minimum.accumulate(np.array([r["U"] for r in rows], dtype=float), 0)
    lc = np.log(ck[:len(rows)].astype(float))
    cr = []
    for q in range(U.shape[1]):
        u = U[:, q]
        idx = np.flatnonzero(u <= eps)
        if len(idx) == 0:
            cr.append(np.inf)
            continue
        k = idx[0]
        if k == 0 or not np.isfinite(u[k - 1]) or u[k] <= 0 or eps <= 0:
            cr.append(lc[k])
            continue
        a, b = math.log(u[k - 1]), math.log(max(u[k], 1e-300))
        t = (a - math.log(eps)) / (a - b) if a > b else 1.0
        cr.append(lc[k - 1] + min(max(t, 0.0), 1.0) * (lc[k] - lc[k - 1]))
    v = np.sort(cr)[stop_k - 1]
    return float(math.exp(v)) if np.isfinite(v) else None


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
        k12 = next((k for k, c in enumerate(curve) if c >= 12), None)
        n80_pen = int(ck[k12]) if (k12 is not None and nfalse[k12] == 0) else tau
        xx = x12(rows, ck, ctx.eps, ctx.stop_k)
        return {"method": name, "validity": m.validity, "seed": int(seed), "eps": float(eps), "N80_pen": n80_pen,
                "k80": k12, "x12": xx if xx is not None else float(tau), "fwer_event": bool(s["fwer_event"]),
                "n_false": int(s["n_false"]), "n_cert": int(s["n_cert"]), "cert_k": s["cert_k"],
                "billing_ok": bool(s["billing_ok"]), "schedule_digest": s["schedule_digest"], "tau_R": tau,
                "sec": round(time.perf_counter() - t0, 3), "error": None}
    except Exception:  # noqa: BLE001
        return {"method": name, "seed": int(seed), "eps": float(eps), "error": traceback.format_exc()}


def toy_job(a):
    name, sd = a
    from dsswm.tests.test_baselines_r4a import PROBS, TOY_EPS, _toy_population
    env = _toy_population(sd)
    out = []
    for p in range(10):
        m = make(name, "toy")
        s, _ = run_stream_v6(env, m, 1000 * sd + p, PROBS, TOY_EPS, keep_U=False)
        out.append({"method": name, "validity": m.validity, "population_seed": sd, "perm_seed": s["perm_seed"],
                    "fwer_event": bool(s["fwer_event"]), "n_cert": s["n_cert"], "n_false": s["n_false"],
                    "reached_stop": s["reached_stop"], "N80_over_tau": s["N80"] / env.tau_R,
                    "billing_ok": bool(s["billing_ok"]), "reselected": s["reselected"], "sec": s["sec_total"]})
    return out


def run_pool(fn, jobs, init=None, initargs=()):
    with get_context("fork").Pool(N_WORKERS, initializer=init, initargs=initargs) as pool:
        for r in pool.imap_unordered(fn, jobs, chunksize=1):
            yield r


def geo(v):
    return float(np.exp(np.mean(np.log(np.asarray(v, dtype=float)))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["toy"] + sorted(TASKS))
    ap.add_argument("--methods", default=None, help="comma list (default: task list)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = datetime.now()
    if args.task == "toy":
        from scipy import stats
        names = args.methods.split(",") if args.methods else ["FDC"] + BET_VARIANTS + MR_VARIANTS + ["NAIVE-joint"]
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
                        "with_reselection": sum(r["reselected"] > 0 for r in rr),
                        "billing_ok_all": all(r["billing_ok"] for r in rr), "validity": rr[0]["validity"]}
        summ = {"task": "toy", "written_at": datetime.now().isoformat(), "started_at": t0.isoformat(),
                "toy": "r4 near-tie toy, population seeds 900-919 x perm 0-9, eps 0.02", "table": table}
        prev = json.loads((d / "summary.json").read_text()) if (d / "summary.json").exists() else {"table": {}}
        prev["table"].update(table)
        summ["table"] = prev["table"]
        (d / "summary.json").write_text(json.dumps(summ, indent=1))
        print(json.dumps({k: (v["false_streams"], v["geomean_N80_over_tau"]) for k, v in table.items()}))
        return
    T = TASKS[args.task]
    assert all(s in DEV for s in T["seeds"])
    names = args.methods.split(",") if args.methods else T["methods"]
    d = OUT / args.task
    d.mkdir(exist_ok=True)
    rfile = d / "results.jsonl"
    done = set()
    if rfile.exists():
        for l in rfile.read_text().splitlines():
            x = json.loads(l)
            done.add((x["method"], x["seed"], x["eps"]))
    jobs = [(n, s, e) for n in names for e in T["eps"] for s in T["seeds"] if (n, s, e) not in done]
    jobs.sort(key=lambda j: j[0] != "HC-WoR")
    print(f"[{args.task}] {len(jobs)} jobs", flush=True)
    errs = 0
    with open(rfile, "a") as f:
        init_env(T["layer"], tuple(T["eps"]))
        for r in run_pool(job, jobs):
            if r.get("error"):
                errs += 1
                (d / "errors.log").open("a").write(f"{r['method']} {r['seed']} {r['eps']}\n{r['error']}\n")
                continue
            f.write(json.dumps(r) + "\n")
            f.flush()
    print(f"[{args.task}] done, {errs} errors, wall {(datetime.now() - t0).total_seconds() / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
