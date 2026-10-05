"""OBD dev-only feasibility pilot (NEW file, v11 retarget).  Descriptive scan + a few streams per method/eps.
Usage (cwd = exp/code): run_obd_dev_pilot.py --describe | --streams SEG A EPS1,EPS2 --seeds 950-959"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
import argparse, itertools, json, sys, time
from multiprocessing import get_context
from pathlib import Path
import numpy as np
CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
OUT = CODE.parents[1] / "exp/results/pilots/obd_dev"
_E = {}
HOLD = "--holdout" in sys.argv


def describe(seg, A):
    from dsswm.envs.obd_v11 import OBDEnv
    from dsswm.envs.seg_v10 import true_opt
    env = OBDEnv(seg, A, holdout=HOLD)
    Js, mu = true_opt(env)
    sp = env.problems
    w = env.w
    up = mu[:, 1:] - mu[:, [0]]
    clicks = np.bincount(env.cell, weights=env.outcomes_view("visit"), minlength=env.S * env.A).reshape(env.S, env.A)
    J0 = float((w * mu[:, 0]).sum())
    out = {"layer": env.layer, "S": env.S, "A": env.A, "N": env.N, "w": w.round(4).tolist(),
           "pool_min": int(env.pool_sizes.min()), "clicks_min": int(clicks.min()),
           "clicks_by_cell": clicks.astype(int).tolist(), "mu": mu.round(5).tolist(), "uplift": up.round(5).tolist(),
           "cost": sp.cost[:, 1].tolist(), "budgets": sp.budgets.tolist(), "J0_all_control": round(J0, 6),
           "Jstar_minus_J0": (Js - J0).round(6).tolist()}
    if env.A ** env.S <= 200000:
        pols = np.array(list(itertools.product(range(env.A), repeat=env.S)))
        val = (w[None, :] * mu[np.arange(env.S)[None, :], pols]).sum(1)
        cst = sp.cost[np.arange(env.S)[None, :], pols].sum(1)
        frac = {}
        for eps in (0.0002, 0.0003, 0.0005, 0.00075, 0.001, 0.0015, 0.002):
            fr_ = []
            for q in range(sp.Q):
                f = cst <= sp.budgets[q]
                fr_.append(float((val[f] >= Js[q] - eps).mean()))
            frac[str(eps)] = {"mean_frac_eps_opt": round(float(np.mean(fr_)), 4),
                              "n_q_all_control_eps_opt": int(sum(J0 >= Js[q] - eps for q in range(sp.Q)))}
        out["eps_opt_fraction"] = frac
        out["n_policies"] = int(len(pols))
    return out


def init(seg, A, eps):
    from dsswm.baselines.fdc_dp import make_seg_ctx
    from dsswm.envs.obd_v11 import OBDEnv
    from dsswm.envs.seg_v10 import true_opt
    env = OBDEnv(seg, A, holdout=HOLD)
    Js, mu = true_opt(env)
    _E.update(env=env, Js=Js, mu=mu, ck=env.checkpoints(20), seg=seg, A=A)


def job(a):
    name, seed, eps = a
    import traceback
    from dsswm.baselines.fdc_dp import FDCDP, FDCDPTimeUniform, make_seg_ctx
    from dsswm.baselines.rect_dp import RectBFDPTU
    from dsswm.envs.seg_v10 import run_stream_seg
    env = _E["env"]
    ck = _E["ck"]
    try:
        ctx = make_seg_ctx(env.w, env.pool_sizes, env.problems, eps, 0.05, ck, env.tau_R, env.replan_interval,
                           stop_frac=1.0)
        m = {"TU-FDC-DP(b)": lambda: FDCDPTimeUniform(ck.copy(), scheme="b"),
             "RECT-BF-DP-TU": lambda: RectBFDPTU(ck.copy(), box=True),
             "FDC-DP(a)": lambda: FDCDP("a")}[name]()
        t0 = time.perf_counter()
        s, _ = run_stream_seg(env, m, seed, ctx, _E["Js"], _E["mu"], n80_k=12)
        r = {k: s[k] for k in ("N80_pen", "n80_lt_tau", "n_cert", "n_false", "fwer_event", "cert_k",
                               "exhaustion_at_k80", "tau_R")}
        r.update(method=name, seed=seed, eps=eps, layer=env.layer, sec=round(time.perf_counter() - t0, 2),
                 node_limit_hits=int(sum(x["node_limit_hits"] for x in getattr(m, "cert_stats", []))))
        return r
    except Exception:
        return {"method": name, "seed": seed, "eps": eps, "error": traceback.format_exc()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--describe", action="store_true")
    ap.add_argument("--streams", nargs=3)
    ap.add_argument("--seeds", default="950-959")
    ap.add_argument("--methods", default="TU-FDC-DP(b),RECT-BF-DP-TU")
    ap.add_argument("--holdout", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.describe:
        res = [describe(s, A) for s, A in ((("uf0x3", 2), ("combo8", 2)) if HOLD else
                                           (("uf0", 2), ("uf0x3", 2), ("combo8", 2), ("combo16", 2), ("uf0x3", 3)))]
        (OUT / ("describe_holdout.json" if HOLD else "describe.json")).write_text(json.dumps(res, indent=1))
        for r in res:
            print(json.dumps({k: r[k] for k in r if k not in ("clicks_by_cell",)}))
    if a.streams:
        seg, A, eps_s = a.streams
        A = int(A)
        lo, hi = map(int, a.seeds.split("-"))
        init(seg, A, None)
        jobs = [(m, s, float(e)) for e in eps_s.split(",") for m in a.methods.split(",") for s in range(lo, hi + 1)]
        f = open(OUT / f"streams_{seg}_A{A}{'_holdout' if HOLD else ''}.jsonl", "a")
        with get_context("fork").Pool(a.workers) as pool:
            for r in pool.imap_unordered(job, jobs):
                f.write(json.dumps(r) + "\n"); f.flush()
                if "error" in r:
                    print(r["error"]); continue
                print(r["method"], r["eps"], r["seed"], r["N80_pen"], r["n_cert"], r["n_false"], r["sec"], flush=True)
