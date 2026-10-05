"""r5_t0_alloc_audit (PILOT, diagnostic): why does the r5 pragmatist's QFC-ney (oracle Neyman shares) sit at
N80 = tau_R on 40/40 development streams?

Read-only on the r4 frozen modules. Development half of CR9, dev seeds 900-909 only.

Arms compared (all QFCMethod with a fixed allocation, i.e. the same rigorous certificate, Thm 1 non-adaptive):
  ney_raw     : the offline script's matrix  sd / sd.sum(1), sd = sqrt(mu (1 - mu)) of the TRUE cell means (no floor)
  ney_floor01 : r4 A-Ney recipe, frozen_alloc.neyman_alloc(true_sigma2) with p_min = 0.01 (what r4 reported)
  ney_floor05 : same with p_min = 0.05 (mechanism dose-response)
  half        : 50/50 within every segment (QFC-half)
  p040        : 0.40 control / 0.60 treatment in every segment (count-conservation reference path)

Per stream and checkpoint we record every cell's n, remaining pool and allocation share (cell_trace.csv) and, per
problem, the QFC U decomposition of the worst challenger (Delta_hat, V, b, width terms, the segment that dominates V,
whether a touched cell has n = 0) in problem_trace.csv.
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import numpy as np  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
from dsswm.baselines.frontier_common import QFCMethod  # noqa: E402
from dsswm.baselines.frozen_alloc import neyman_alloc  # noqa: E402
from dsswm.certify.quadknap import make_stats, qfc_certificate_enum, pair_terms, cell_terms  # noqa: E402
from dsswm.envs.pool_replay import PoolReplayEnv, make_schedule  # noqa: E402
from dsswm.streams import frontier as fr  # noqa: E402
from dsswm.streams.frontier_runner import StreamEngine, build_ctx, run_stream, true_policy_values  # noqa: E402

TASK = "r5_t0_alloc_audit"
WS = CODE.parent.parent
RES = WS / "exp" / "results"
OUT = RES / "pilots" / TASK
SEEDS = list(range(900, 910))
EPS = 0.001
G = {}


def init():
    env = PoolReplayEnv("CR9", "dev")
    ctx = build_ctx(env, fr.cr_problems("visit"), EPS)
    Jt = true_policy_values(ctx.pols, env.w, env.true_mu("visit"))
    G.update(env=env, ctx=ctx, Jt=Jt, Js=np.array([Jt[ctx.feas[q]].max() for q in range(ctx.Q)]))


def allocs():
    env = G["env"]
    mu = env.true_mu("visit")
    sd = np.sqrt(mu * (1 - mu))
    s2 = env.true_sigma2("visit")
    return {"ney_raw": sd / sd.sum(1, keepdims=True),
            "ney_floor01": neyman_alloc(s2, p_min=0.01),
            "ney_floor05": neyman_alloc(s2, p_min=0.05),
            "half": np.full((env.S, env.A), 0.5),
            "p040": np.tile([0.40, 0.60], (env.S, 1))}


def job(arg):
    kind, seed = arg
    env, ctx = G["env"], G["ctx"]
    P = allocs()[kind]
    m = QFCMethod(name="QFC-" + kind, alloc_p=P)
    # (a) the official runner (N80 as the pragmatist computed it)
    summ, _ = run_stream(env, m, seed, ctx.problems, eps=ctx.eps, ctx=ctx, J_true=G["Jt"], J_star=G["Js"],
                         keep_U=False)
    # (b) a full-length trace over all K checkpoints (no early stop), same schedule / engine / certificate
    sch = make_schedule(env, seed, alloc=P)
    eng = StreamEngine(env, sch, "visit", delta_cell=ctx.delta / (env.S * env.A), R=ctx.R)
    L1, x_v = m.params["L1"], m.params["x_v"]
    cells, probs = [], []
    first_cert_k = np.full(ctx.Q, -1)
    for k, t in enumerate(ctx.checkpoints):
        eng.advance_planned(int(t))
        n = eng.n.copy()
        mu_hat = np.where(n > 0, eng.sum / np.maximum(n, 1), 0.5)
        stats = make_stats(ctx.w, mu_hat, n, N=eng.N, x_v=x_v, binary=True, R=1.0)
        v, bc, live = cell_terms(stats)
        res = qfc_certificate_enum(stats, ctx.problems, L1, ctx.eps, pols=ctx.pols)
        for s in range(env.S):
            for a in range(env.A):
                cells.append({"method": kind, "seed": seed, "k": k, "t": int(t), "s": s, "a": a,
                              "p_alloc": round(float(P[s, a]), 6), "n": int(n[s, a]),
                              "remaining": int(eng.N[s, a] - n[s, a]), "exhausted": int(n[s, a] >= eng.N[s, a]),
                              "mu_hat": float(mu_hat[s, a]), "var_ucb": float(stats.var_ucb[s, a]),
                              "v_term": float(v[s, a]), "b_term": float(bc[s, a])})
        for q, r in enumerate(res):
            ph = np.array(r["pi_hat"])
            wk = np.array(r["worst"])
            diff = wk != ph
            segs = np.flatnonzero(diff)
            vs = np.array([v[s, wk[s]] + v[s, ph[s]] for s in segs]) if segs.size else np.zeros(0)
            zero_touch = [int(s) for s in segs if n[s, wk[s]] == 0 or n[s, ph[s]] == 0]
            dom = int(segs[int(np.argmax(vs))]) if segs.size else -1
            V, b = r["worst_V"], r["worst_b"]
            if r["certified"] and first_cert_k[q] < 0:
                first_cert_k[q] = k
            probs.append({"method": kind, "seed": seed, "k": k, "t": int(t), "q": q, "qid": r["qid"],
                          "U": r["U"], "certified": int(r["certified"]), "pi_hat": "".join(map(str, ph)),
                          "worst": "".join(map(str, wk)), "dhat": r["worst_dhat"], "V": V, "b": b,
                          "sqrt_2L1V": math.sqrt(2 * L1 * V) if math.isfinite(V) else float("inf"),
                          "bL1_3": b * L1 / 3 if math.isfinite(b) else float("inf"),
                          "dom_seg_V": dom,
                          "dom_seg_V_share": float(vs.max() / vs.sum()) if segs.size and np.isfinite(vs).all()
                          and vs.sum() > 0 else (1.0 if segs.size else 0.0),
                          "zero_n_touched_segs": ";".join(map(str, zero_touch)),
                          "n_cert_so_far": int((first_cert_k >= 0).sum())})
    return {"kind": kind, "seed": seed, "N80": summ["N80"], "censored": summ["censored"], "k_stop": summ["k_stop"],
            "n_false": summ["n_false"], "reselected": summ["reselected"], "billing_ok": summ["billing_ok"],
            "digest": summ["schedule_digest"], "first_cert_k": first_cert_k.tolist()}, cells, probs


def progress(done, total, extra=None):
    (RES / f"{TASK}_PROGRESS.json").write_text(json.dumps({
        "task_id": TASK, "epoch": done, "total_epochs": total, "step": done, "total_steps": total,
        "loss": None, "metric": extra or {}, "updated_at": datetime.now().isoformat()}))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (RES / f"{TASK}.pid").write_text(str(os.getpid()))
    t0 = time.time()
    init()
    env, ctx = G["env"], G["ctx"]
    kinds = ["ney_raw", "ney_floor01", "ney_floor05", "half", "p040"]
    jobs = [(k, s) for k in kinds for s in SEEDS]
    summ, cells, probs = [], [], []
    with ProcessPoolExecutor(4, initializer=init) as ex:
        for i, (sm, c, p) in enumerate(ex.map(job, jobs)):
            summ.append(sm)
            cells += c
            probs += p
            progress(i + 1, len(jobs), {"last": f"{sm['kind']}/{sm['seed']}", "N80": sm["N80"]})
    for name, rows in (("cell_trace.csv", cells), ("problem_trace.csv", probs)):
        with open(OUT / name, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
    with open(OUT / "streams.jsonl", "w") as f:
        for r in summ:
            f.write(json.dumps(r) + "\n")
    A = allocs()
    agg = {}
    for k in kinds:
        rr = [r for r in summ if r["kind"] == k]
        agg[k] = {"alloc_p": np.round(A[k], 5).tolist(),
                  "N80": [r["N80"] for r in rr], "median_N80": float(np.median([r["N80"] for r in rr])),
                  "n_at_tauR": int(sum(r["N80"] == env.tau_R for r in rr)),
                  "censored": int(sum(r["censored"] for r in rr)), "false_streams": int(sum(r["n_false"] > 0 for r in rr)),
                  "billing_ok_all": bool(all(r["billing_ok"] for r in rr))}
    out = {"task_id": TASK, "tau_R": int(env.tau_R), "checkpoints": ctx.checkpoints.tolist(), "eps": EPS,
           "seeds": SEEDS, "per_alloc": agg, "sec": round(time.time() - t0, 1)}
    json.dump(out, open(OUT / "run_summary.json", "w"), indent=1)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "alloc_p"} for k, v in agg.items()}, indent=1))
    pid = RES / f"{TASK}.pid"
    if pid.exists():
        pid.unlink()


if __name__ == "__main__":
    main()
