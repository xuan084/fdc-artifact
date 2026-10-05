"""v9 candidates dev runner (plan/v9_candidates_theory.md).  DEV ONLY: dev halves, report seeds 950-999 (asserted),
toy population seeds 900-919.  New file; reuses the frozen v8 job (run_r5s_v8.job) and frozen v8 rival configs.

Tasks
  toy    K = 20 grid: FDC-LOC, FDC-BF, NAIVE-joint (invalid power control)
  toyd   4x-dense evaluation grid: TU-FDC[K20] (stale widths), TU-LOC[K20], FDC-BF[K77] (fresh, K' = 77 ledger),
         NAIVE-joint (invalid) -- time-uniform FWER under dense monitoring
  cr9 / x5    K = 20 grid: FDC-LOC (comparators reused from exp/results/pilots/fdc_hg, same code / configs / seeds)
  cr9d / x5d  dense grid: TU-FDC[K20], TU-LOC[K20], FDC-BF[K77], HC-WoR (frozen v8 config), RECT-ck-HG (K77 ledger)
Output: exp/results/pilots/v9_candidates/<task>/results.jsonl (resumable).
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import copy  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from datetime import datetime  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
WS = CODE.parents[1]
OUT = WS / "exp/results/pilots/v9_candidates"
REPORT_SEEDS = list(range(950, 1000))
N_WORKERS = int(os.environ.get("V9C_WORKERS", "6"))

import run_fdc_bet as rfb  # noqa: E402
import run_r5s_v8 as R8  # noqa: E402
from dsswm.baselines.fdc_loc import FDCLoc, FDCTimeUniform, dense_grid  # noqa: E402

TASKS = {"cr9": dict(layer="CR9", eps=(0.001,), dense=False, methods=["FDC-LOC", "ORACLE-LOC"]),
         "x5": dict(layer="X9", eps=(0.015, 0.02, 0.03), dense=False, methods=["FDC-LOC", "ORACLE-LOC"]),
         "cr9d": dict(layer="CR9", eps=(0.001,), dense=True,
                      methods=["TU-FDC[K20]", "TU-LOC[K20]", "FDC-BF[K77]", "HC-WoR", "RECT-ck-HG"]),
         "x5d": dict(layer="X9", eps=(0.02,), dense=True,
                     methods=["TU-FDC[K20]", "TU-LOC[K20]", "FDC-BF[K77]", "HC-WoR", "RECT-ck-HG"])}
_LAST = {}
_BLOCKS = {}
_orig_make = R8.make


class _OracleLoc(FDCLoc):
    """INVALID diagnostic (validity 'none'): G_hat built from the TRUE gaps (x = (Delta - eps)_+, Delta <= eps dropped),
    same Wbar as FDC-LOC (survivors = pi*_q only).  Upper bound on what any gap localisation of this form can gain."""
    validity = "none"

    def _beta(self, ctx, st, lo, hi):
        import math as _m
        from dsswm.baselines.fdc_loc import pair_tables
        J, Js = R8._ENV["J"], R8._ENV["Js"]
        L, V, b, Z = pair_tables(ctx, st.n, st.N, lo, hi)
        xs, Vs, bs, zs = [], [], [], []
        for q in range(ctx.Q):
            f = np.asarray(ctx.feas[q], dtype=bool)
            istar = int(np.flatnonzero(f)[np.argmax(J[f])])
            gap = Js[q] - J
            sel = f & (gap > ctx.eps + 1e-12)
            xs.append(gap[sel] - ctx.eps)
            Vs.append(V[istar, sel])
            bs.append(b[istar, sel])
            zs.append(Z[istar, sel])

        def G(beta):
            tot = 0.0
            for x, Vq, bq, zq in zip(xs, Vs, bs, zs):
                wq = np.sqrt(2 * beta * Vq) + bq * beta / 3
                with np.errstate(divide="ignore", invalid="ignore"):
                    r = np.where(wq > 0, x / np.where(wq > 0, wq, 1), np.inf)
                r = np.where(zq, 0.0, r)
                tot += float(np.sum(np.exp(-beta * (1 + r)))) if beta > 0 else float(x.size)
            return tot
        bJ = self.ledger["beta"]
        tgt = self.split[0] / self.ledger["K"]
        a, c = 0.0, bJ
        while c - a > 1e-6:
            mid = 0.5 * (a + c)
            if G(mid) <= tgt:
                c = mid
            else:
                a = mid
        return c, {"n_unresolved_e": G(c) * _m.exp(c)}


def make_new(name, blocks=None):
    if name == "ORACLE-LOC":
        return _OracleLoc(name="ORACLE-LOC")
    if name == "FDC-LOC":
        return FDCLoc(name="FDC-LOC")
    if name == "TU-FDC[K20]":
        return FDCTimeUniform(blocks, name=name, localise=False)
    if name == "TU-LOC[K20]":
        return FDCTimeUniform(blocks, name=name, localise=True)
    if name == "FDC-BF[K77]":
        from dsswm.baselines.fdc_bet import make_variant
        m = make_variant("FDC-BF")
        m.name = name
        return m
    return None


def _make(name, kw):
    m = make_new(name, _BLOCKS.get("ck"))
    if m is None:
        m = _orig_make(name, kw)
    _LAST["m"] = m
    return m


R8.make = _make


def job(a):
    out = R8.job(a)
    if out.get("error") is None:
        m = _LAST.get("m")
        if getattr(m, "beta_trace", None):
            out["beta_trace"] = m.beta_trace
        if hasattr(m, "n_stale_evals"):
            out["n_stale_evals"] = m.n_stale_evals
        out["K_eval"] = int(len(R8._ENV["ctxs"][out["eps"]].checkpoints))
    for k in [k for k in out if k.startswith("rs_")]:
        out.pop(k)
    return out


# ------------------------------------------------------------------------------------------------ toy
def toy_job(a):
    name, sd, dense = a
    from dsswm.streams.frontier_runner import build_ctx
    from dsswm.streams.frontier_runner_v6 import run_stream_v6
    from dsswm.tests.test_baselines_r4a import PROBS, TOY_EPS, _toy_population
    env = _toy_population(sd)
    ctx = build_ctx(env, PROBS, TOY_EPS)
    ck = ctx.checkpoints.copy()
    if dense:
        ctx = copy.copy(ctx)
        ctx.checkpoints = dense_grid(ck, 4)
    out = []
    for p in range(10):
        m = make_new(name, ck)
        if m is None:
            m = rfb.make(name, "toy")
        s, _ = run_stream_v6(env, m, 1000 * sd + p, PROBS, TOY_EPS, ctx=ctx, keep_U=False)
        out.append({"method": name, "dense": dense, "K_eval": int(len(ctx.checkpoints)), "validity": m.validity,
                    "population_seed": sd, "perm_seed": s["perm_seed"], "fwer_event": bool(s["fwer_event"]),
                    "n_cert": s["n_cert"], "n_false": s["n_false"], "reached_stop": s["reached_stop"],
                    "N80_over_tau": s["N80"] / env.tau_R, "billing_ok": bool(s["billing_ok"])})
    return out


def run_toy(task):
    from scipy import stats
    dense = task == "toyd"
    names = (["TU-FDC[K20]", "TU-LOC[K20]", "FDC-BF[K77]", "NAIVE-joint"] if dense
             else ["FDC-LOC", "FDC-BF", "NAIVE-joint"])
    d = OUT / task
    d.mkdir(parents=True, exist_ok=True)
    with get_context("fork").Pool(N_WORKERS) as pool:
        rows = [r for ch in pool.imap_unordered(toy_job, [(n, sd, dense) for n in names for sd in range(900, 920)])
                for r in ch]
    with open(d / "results.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    table = {}
    for n in names:
        rr = [r for r in rows if r["method"] == n]
        ev = sum(r["fwer_event"] for r in rr)
        table[n] = {"n_streams": len(rr), "false_streams": ev, "false_certs": sum(r["n_false"] for r in rr),
                    "cp_upper95": 1.0 if ev >= len(rr) else float(stats.beta.ppf(0.95, ev + 1, len(rr) - ev)),
                    "geomean_N80_over_tau": rfb.geo([r["N80_over_tau"] for r in rr]),
                    "K_eval": rr[0]["K_eval"], "validity": rr[0]["validity"]}
    (d / "summary.json").write_text(json.dumps({"task": task, "written_at": datetime.now().isoformat(),
                                                "toy": "r4 near-tie toy, pop seeds 900-919 x perm 0-9, eps 0.02",
                                                "table": table}, indent=1))
    print(json.dumps({k: (v["false_streams"], round(v["geomean_N80_over_tau"], 4)) for k, v in table.items()}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["toy", "toyd"] + sorted(TASKS))
    ap.add_argument("--methods", default=None)
    args = ap.parse_args()
    if args.task in ("toy", "toyd"):
        run_toy(args.task)
        return
    t0 = datetime.now()
    T = TASKS[args.task]
    d = OUT / args.task
    d.mkdir(parents=True, exist_ok=True)
    names = args.methods.split(",") if args.methods else T["methods"]
    cfg = R8.load_configs(T["layer"])
    R8.init_env(T["layer"], "dev", tuple(T["eps"]))
    assert R8._ENV["env"].half == "dev"
    assert all(950 <= s <= 999 for s in REPORT_SEEDS)
    ck = next(iter(R8._ENV["ctxs"].values())).checkpoints.copy()
    _BLOCKS["ck"] = ck
    if T["dense"]:
        g = dense_grid(ck, 4)
        for e in list(R8._ENV["ctxs"]):
            c2 = copy.copy(R8._ENV["ctxs"][e])
            c2.checkpoints = g
            R8._ENV["ctxs"][e] = c2
        print(f"[{args.task}] dense grid K_eval = {len(g)}", flush=True)
    rfile = d / "results.jsonl"
    done = set()
    if rfile.exists():
        for line in rfile.read_text().splitlines():
            x = json.loads(line)
            if x.get("error") is None:
                done.add((x["method"], x["seed"], x["eps"]))
    jobs = [(n, R8.method_kw(n, cfg), s, e, 15, "dev", "dev", "dev")
            for n in names for e in T["eps"] for s in REPORT_SEEDS if (n, s, e) not in done]
    jobs.sort(key=lambda j: j[0] != "HC-WoR")
    print(f"[{args.task}] {len(jobs)} jobs", flush=True)
    errs = 0
    with get_context("fork").Pool(N_WORKERS) as pool, open(rfile, "a") as f:
        for r in pool.imap_unordered(job, jobs, chunksize=1):
            if r.get("error"):
                errs += 1
                with open(d / "errors.log", "a") as fe:
                    fe.write(f"{r['method']} {r['seed']} {r['eps']}\n{r['error']}\n")
                continue
            f.write(json.dumps(r) + "\n")
            f.flush()
    print(f"[{args.task}] done, {errs} errors, wall {(datetime.now() - t0).total_seconds() / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
