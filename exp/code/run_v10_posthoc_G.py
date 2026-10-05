"""Block v10-posthoc-G (POST HOC, DESCRIPTIVE; response to critic r5 R5-P0-2 / R5-P0-3).  NEW file: every lock-bound
module (fdc_dp.py, rect_dp.py, seg_v10*.py, run_v10.py, ...) is imported unchanged.

  G1  reads-versus-eps curves: N80/tau against eps for TU-FDC-DP(b), RECT-BF-DP-TU and HC-WoR-DP (untuned v10
      transfer), X5 eps {0.02, 0.03, 0.04, 0.05, 0.06, 0.08}, Lenta eps {0.003, 0.004, 0.006, 0.008, 0.010, 0.012},
      S in {16, 32, 64}; dev seeds 950-999 and eval halves with fresh seeds 38400-38599 (post-hoc access path).
  G2  heterogeneous-cost knapsack budgets (dev only, seeds 950-999): pre-stated cost rule (plan/v10_posthoc_G_plan.md),
      15 budgets = fractions 0.10..0.80 of total cost; FDC-DP(b), FDC-DP(a), RECT-BF-DP, HC-WoR-DP; eps by the lock-v10
      rival-success rule on the G1 grid.
  G3  stronger rivals at the lock-v10 cells: RECT-HG-DP (exact hypergeometric cells, delta / (2 S A K), worst-gap DP)
      and HC-WoR-DP tuned on dev over a declared 8-config grid (chosen by dev geomean N80_pen); dev seeds 950-999 and
      eval with fresh seeds 38600-38799.
  G4  union-cost trend table (beta_J / beta_C against S) next to the observed ratios.

Eval access only through dsswm.envs.posthoc_access_v10 (lock verified, confirmatory tasks sealed, task id
'v10-posthoc-G*', seeds 38400-38999, every read logged).  Rows carry posthoc = true and label 'post hoc, descriptive'.

Usage (cwd = exp/code)
  run_v10_posthoc_G.py --cells G1-dev-x5-16,G2-dev-lenta-64 [--workers 4]     # run cells sequentially (resumable)
  run_v10_posthoc_G.py --list                                                   # cell ids
  run_v10_posthoc_G.py --select-hc                                              # G3 dev HC selection -> hc_tuned.json
  run_v10_posthoc_G.py --analyse                                                # summary.{md,json}, table.tex, curves
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import collections  # noqa: E402
import hashlib  # noqa: E402
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
RES = WS / "exp/results"
OUT = RES / "full" / "v10_posthoc_G"
ROWS = OUT / "rows"
K_GRID = 20
STOP_FRAC = 1.0
N80_K = 12
LABEL = "post hoc, descriptive"
COMPUTE_VERSION = "g-1"       # bump when the compute part of this file changes (rows are keyed by it + library hashes)

DEV_SEEDS = tuple(range(950, 1000))
G1_EVAL_SEEDS = tuple(range(38400, 38600))
G3_EVAL_SEEDS = tuple(range(38600, 38800))
EPS_GRID = {"x5": (0.02, 0.03, 0.04, 0.05, 0.06, 0.08), "lenta": (0.003, 0.004, 0.006, 0.008, 0.010, 0.012)}
LOCK_EPS = {("x5", 16): 0.03, ("x5", 32): 0.04, ("x5", 64): 0.05,
            ("lenta", 16): 0.004, ("lenta", 32): 0.006, ("lenta", 64): 0.008}
LOCK_TASK = {("x5", 16): "v10a_full_s16", ("x5", 32): "v10a_full_s32", ("x5", 64): "v10d_full_x5s64",
             ("lenta", 16): "v10b_full_s16", ("lenta", 32): "v10b_full_s32", ("lenta", 64): "v10b_full_s64"}
SGRID = (16, 32, 64)
DATAS = ("x5", "lenta")
G1_METHODS = ("TU-FDC-DP(b)", "RECT-BF-DP-TU", "HC-WoR-DP")
G2_METHODS = ("FDC-DP(b)", "FDC-DP(a)", "RECT-BF-DP", "HC-WoR-DP")
HC_GRID = tuple((c, tf) for c in (0.5, 0.75) for tf in (0.2, 0.4, 0.6, 0.8))      # declared, 8 configs, grid order
HC_NAMES = tuple(f"HC-WoR-DP[c={c},tf={tf}]" for c, tf in HC_GRID)
G3_DEV_METHODS = ("RECT-HG-DP",) + HC_NAMES
G3_EVAL_METHODS = ("TU-FDC-DP(b)", "RECT-BF-DP-TU", "RECT-HG-DP", "HC-WoR-DP[tuned]")
RULE_MIN_SUCCESS = 40
# G2 cost rule (pre-stated in plan/v10_posthoc_G_plan.md before any G2 stream)
COST_G = 8
X5_COST_SEED = 41000
X5_COST_SIGMA = 0.5
LENTA_COST_FEATURE = "x_mean_discount_depth_15d"

LIB_FILES = ["dsswm/baselines/fdc_dp.py", "dsswm/baselines/rect_dp.py", "dsswm/baselines/rect_hg_dp.py",
             "dsswm/baselines/rect_v6.py", "dsswm/baselines/rect_tu_v9.py", "dsswm/baselines/pjc_bf.py",
             "dsswm/baselines/wor_betting_v6.py", "dsswm/baselines/fdc_bet.py", "dsswm/baselines/b4_bal.py",
             "dsswm/baselines/frontier_common.py", "dsswm/envs/seg_v10.py", "dsswm/envs/seg_v10_eval.py",
             "dsswm/envs/posthoc_access_v10.py", "dsswm/envs/pool_replay.py", "dsswm/streams/frontier_runner.py",
             "dsswm/streams/frontier_runner_v6.py", "dsswm/streams/frontier.py"]
_ENV: dict = {}


# =============================================================================================== cells
def _cells():
    C = {}
    for d in DATAS:
        for S in SGRID:
            C[f"G1-dev-{d}-{S}"] = dict(part="G1", data=d, S=S, half="dev", eps=EPS_GRID[d], methods=G1_METHODS,
                                        seeds=DEV_SEEDS, cost="frozen")
            C[f"G1-eval-{d}-{S}"] = dict(part="G1", data=d, S=S, half="eval", eps=EPS_GRID[d], methods=G1_METHODS,
                                         seeds=G1_EVAL_SEEDS, cost="frozen")
            C[f"G2-dev-{d}-{S}"] = dict(part="G2", data=d, S=S, half="dev", eps=EPS_GRID[d], methods=G2_METHODS,
                                        seeds=DEV_SEEDS, cost="hetero")
            C[f"G3-dev-{d}-{S}"] = dict(part="G3", data=d, S=S, half="dev", eps=(LOCK_EPS[(d, S)],),
                                        methods=G3_DEV_METHODS, seeds=DEV_SEEDS, cost="frozen")
            C[f"G3-eval-{d}-{S}"] = dict(part="G3", data=d, S=S, half="eval", eps=(LOCK_EPS[(d, S)],),
                                         methods=G3_EVAL_METHODS, seeds=G3_EVAL_SEEDS, cost="frozen")
    return C


CELLS = _cells()


def code_sha():
    per = {p: hashlib.sha256((CODE / p).read_bytes()).hexdigest() for p in LIB_FILES}
    per["COMPUTE_VERSION"] = COMPUTE_VERSION
    return hashlib.sha256(json.dumps(per, sort_keys=True).encode()).hexdigest()[:16], per


# =============================================================================================== G2 costs
def hetero_costs(data, S):
    """Pre-stated G2 cost rule (covariates / fixed seed only, dev-frozen; never outcomes).
    X5 (no cost covariate): c_s = exp(0.5 z_s), z ~ N(0, 1) from default_rng(41000 + S).
    Lenta: c_s = dev replay-pool mean of x_mean_discount_depth_15d in frozen segment s (promotion-cost proxy).
    kappa[s, 1] = max(1, rint(8 c_s / mean_s c)), kappa[s, 0] = 0; budgets floor(b_q sum_s kappa[s, 1])."""
    from dsswm.envs import seg_v10_eval as E
    if data == "x5":
        c = np.exp(X5_COST_SIGMA * np.random.default_rng(X5_COST_SEED + S).normal(size=S))
        src = f"seeded lognormal: exp({X5_COST_SIGMA} z), z ~ N(0,1), rng {X5_COST_SEED}+S"
    else:
        fz = E.load_frozen_seg()
        X, arm, y, rows, msk = E._dev_replay("lenta")
        del y                                                # outcomes are never used by the cost rule
        rp = ~msk
        sc = E._score(E._load_models(fz, "lenta"), X[rp])
        cuts = [float.fromhex(v) for v in fz["data"]["lenta"]["S"][str(S)]["cuts_hex"]]
        seg = E.assign_segments(sc, cuts)
        j = list(E.SV.LENTA_FEATURES).index(LENTA_COST_FEATURE)
        v = X[rp][:, j]
        c = np.array([np.nanmean(v[seg == s]) for s in range(S)])
        src = f"dev replay-pool segment mean of {LENTA_COST_FEATURE} (frozen segmentation)"
    k1 = np.maximum(1, np.rint(COST_G * c / c.mean())).astype(np.int64)
    return k1, src


def hetero_problems(data, S):
    from dsswm.baselines.fdc_dp import SegProblems
    from dsswm.envs.seg_v10 import SR_BUDGETS
    k1, src = hetero_costs(data, S)
    cost = np.stack([np.zeros(S, dtype=np.int64), k1], 1)
    tot = int(k1.sum())
    B = np.array([int(math.floor(b * tot + 1e-9)) for b in SR_BUDGETS], dtype=np.int64)
    sp = SegProblems(cost=cost, budgets=B, qids=[f"HK|c1|B{b:.2f}" for b in SR_BUDGETS], budget_frac=list(SR_BUDGETS))
    return sp, src


# =============================================================================================== methods
def hc_v10_config(data):
    key = "cr" if data == "lenta" else "x9"
    c = json.loads((RES / "v8_gates" / f"{key}_configs.json").read_text())["selected"]["HC-WoR"]
    return c["schedule"], float(c["c"]), c.get("target_frac")


def hc_tuned(data, S):
    f = OUT / "hc_tuned.json"
    sel = json.loads(f.read_text())["selected"][f"{data}-{S}"]
    return float(sel["c"]), float(sel["tf"])


def make(name, data, S, ck):
    from dsswm.baselines.fdc_dp import FDCDP, FDCDPTimeUniform
    from dsswm.baselines.rect_dp import HCWoRDP, RectBFDP, RectBFDPTU
    from dsswm.baselines.rect_hg_dp import RectHGDP
    if name == "TU-FDC-DP(b)":
        return FDCDPTimeUniform(ck, scheme="b")
    if name == "FDC-DP(b)":
        return FDCDP("b")
    if name == "FDC-DP(a)":
        return FDCDP("a")
    if name == "RECT-BF-DP-TU":
        return RectBFDPTU(ck, box=True)
    if name == "RECT-BF-DP":
        return RectBFDP(box=True)
    if name == "RECT-HG-DP":
        return RectHGDP()
    if name == "HC-WoR-DP":
        sch, c, tf = hc_v10_config(data)
        return HCWoRDP(sch, c, tf)
    if name == "HC-WoR-DP[tuned]":
        c, tf = hc_tuned(data, S)
        return HCWoRDP("nstar", c, tf, name=name)
    if name.startswith("HC-WoR-DP[c="):
        inner = name[len("HC-WoR-DP["):-1]
        kv = dict(x.split("=") for x in inner.split(","))
        return HCWoRDP("nstar", float(kv["c"]), float(kv["tf"]), name=name)
    raise KeyError(name)


# =============================================================================================== env / jobs
def init_env(cid):
    from dsswm.baselines.fdc_dp import make_seg_ctx
    from dsswm.envs.seg_v10 import env_digest, true_opt
    C = CELLS[cid]
    access = None
    if C["half"] == "eval":
        from dsswm.envs.posthoc_access_v10 import SegEnvV10PosthocEval, check_digest_against_seal
        env = SegEnvV10PosthocEval(C["data"], C["S"], f"v10-posthoc-{cid}", C["seeds"],
                                   reason=f"block v10-posthoc-G {C['part']}: {LABEL} (critic r5 R5-P0-2/3)")
        access = {"log": env.posthoc_access, "seal_check": check_digest_against_seal(env)}
    else:
        from dsswm.envs.seg_v10_eval import SegEnvV10Frozen
        env = SegEnvV10Frozen(C["data"], C["S"], "dev")
    cost_src = "frozen v10 problems (kappa = 8, cardinality budgets)"
    if C["cost"] == "hetero":
        env.problems, cost_src = hetero_problems(C["data"], C["S"])
    Js, mu = true_opt(env)
    ctxs = {e: make_seg_ctx(env.w, env.pool_sizes, env.problems, e, 0.05, env.checkpoints(K_GRID), env.tau_R,
                            env.replan_interval, stop_frac=STOP_FRAC) for e in C["eps"]}
    _ENV.clear()
    _ENV.update(env=env, Js=Js, mu=mu, ctxs=ctxs, cid=cid, C=C, digest=env_digest(env), access=access,
                cost_src=cost_src)


def job(a):
    name, seed, eps, csha = a
    from dsswm.envs.seg_v10 import run_stream_seg
    C = _ENV["C"]
    ctx = _ENV["ctxs"][eps]
    try:
        m = make(name, C["data"], C["S"], ctx.checkpoints.copy())
        t0 = time.perf_counter()
        s, rows = run_stream_seg(_ENV["env"], m, seed, ctx, _ENV["Js"], _ENV["mu"], n80_k=N80_K)
        out = {"cell": _ENV["cid"], "part": C["part"], "half": C["half"], "data": C["data"], "S": C["S"],
               "method": name, "seed": int(seed), "eps": float(eps), "posthoc": True, "label": LABEL}
        out.update({k: s[k] for k in ("validity", "schedule_digest", "reached_stop", "stop_k", "n80_k", "k_stop",
                                      "k80", "N80_pen", "N80_raw", "n80_lt_tau", "false_by_k80",
                                      "exhaustion_at_k80", "N_stop_pen", "n_cert", "n_false", "fwer_event",
                                      "cert_k", "decided_pi", "billing_ok", "n_cert_curve", "tau_R")})
        out.update({"sec": round(time.perf_counter() - t0, 3), "sec_cert_by_k": s["sec_cert_by_k"],
                    "env_digest": _ENV["digest"], "code_sha": csha, "error": None})
        cs = getattr(m, "cert_stats", None)
        if cs:
            nodes = [x["nodes"] for x in cs]
            out.update({"bnb_nodes": int(sum(nodes)), "bnb_nodes_max_ck": int(max(nodes)),
                        "bnb_calls": int(sum(x["bnb_calls"] for x in cs)),
                        "node_limit_hits": int(sum(x["node_limit_hits"] for x in cs)),
                        "b_only_certs": int(sum(x["b_certified"] for x in cs)),
                        "a_certs": int(sum(x["a_certified"] for x in cs)),
                        "beta": float(m.ledger["beta"]), "union_size_log": float(m.ledger["union_size_log"])})
        if name.startswith("HC-WoR-DP"):
            out["hc_config"] = {"schedule": m.schedule, "c": m.c, "target_frac": m.target_frac}
        return out
    except Exception:  # noqa: BLE001
        return {"cell": _ENV.get("cid"), "method": name, "seed": int(seed), "eps": float(eps),
                "error": traceback.format_exc()}


def load_rows(cid, csha=None):
    f = ROWS / f"{cid}.jsonl"
    if not f.exists():
        return []
    csha = csha or code_sha()[0]
    out, seen = [], set()
    for line in f.read_text().splitlines():
        try:
            x = json.loads(line)
        except ValueError:
            continue
        k = (x["method"], x["seed"], x["eps"])
        if x.get("error") is None and x.get("code_sha") == csha and k not in seen:
            seen.add(k)
            out.append(x)
    return out


def progress(cid, done, total, errs):
    (RES / f"v10pG_{cid}_PROGRESS.json").write_text(json.dumps({
        "task_id": f"v10pG_{cid}", "epoch": done, "total_epochs": total, "step": done, "total_steps": total,
        "loss": None, "metric": {"runs_done": done, "errors": errs}, "updated_at": datetime.now().isoformat()}))


def env_describe(env):
    """Metadata only (not hashed into rows): SegEnvV10PosthocEval has no describe(), so build the same dict."""
    if hasattr(env, "describe"):
        return env.describe()
    return {"layer": env.layer, "half": env.half, "S": env.S, "N": env.N,
            "pool_sizes_min": int(env.pool_sizes.min()), "w_range": [float(env.w.min()), float(env.w.max())],
            "cost": env.problems.cost[:, 1].tolist(), "budgets": env.problems.budgets.tolist()}


def run_cell(cid, workers):
    C = CELLS[cid]
    t_start = datetime.now()
    ROWS.mkdir(parents=True, exist_ok=True)
    csha, per = code_sha()
    rfile = ROWS / f"{cid}.jsonl"
    keep = load_rows(cid, csha)
    rfile.write_text("".join(json.dumps(x) + "\n" for x in keep))
    done = {(x["method"], x["seed"], x["eps"]) for x in keep}
    jobs = [(m, s, float(e), csha) for m in C["methods"] for e in C["eps"] for s in C["seeds"]
            if (m, s, float(e)) not in done]
    jobs.sort(key=lambda j: not j[0].startswith("HC"))
    total = len(jobs) + len(done)
    print(f"[{cid}] {len(jobs)} jobs to run ({len(done)} done), code {csha}", flush=True)
    if not jobs:
        return
    init_env(cid)
    meta = {"cell": cid, **{k: (list(v) if isinstance(v, tuple) else v) for k, v in C.items() if k != "seeds"},
            "seeds": [C["seeds"][0], C["seeds"][-1]], "label": LABEL, "code_sha": csha, "code_files": per,
            "env": env_describe(_ENV["env"]), "env_digest": _ENV["digest"], "cost_source": _ENV["cost_src"],
            "costs": _ENV["env"].problems.cost[:, 1].tolist(), "budgets": _ENV["env"].problems.budgets.tolist(),
            "posthoc_access": _ENV["access"], "started_at": t_start.isoformat()}
    (ROWS / f"{cid}.meta.json").write_text(json.dumps(meta, indent=1, default=str))
    errs, n_done = 0, len(done)
    with get_context("fork").Pool(workers) as pool, open(rfile, "a") as f:
        for r in pool.imap_unordered(job, jobs, chunksize=1):
            if r.get("error"):
                errs += 1
                with open(ROWS / f"{cid}.errors.log", "a") as fe:
                    fe.write(f"{r['method']} {r['seed']} {r['eps']}\n{r['error']}\n")
                continue
            f.write(json.dumps(r) + "\n")
            f.flush()
            n_done += 1
            if n_done % 20 == 0 or n_done == total:
                progress(cid, n_done, total, errs)
    progress(cid, n_done, total, errs)
    meta["ended_at"] = datetime.now().isoformat()
    meta["wall_min"] = round((datetime.now() - t_start).total_seconds() / 60, 2)
    meta["errors"] = errs
    (ROWS / f"{cid}.meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print(f"[{cid}] done: {n_done}/{total} rows, {errs} errors, wall {meta['wall_min']} min", flush=True)


# =============================================================================================== analysis helpers
def _geo(v):
    return float(np.exp(np.mean(np.log(np.asarray(v, dtype=float)))))


def by_key(rows):
    return {(r["method"], int(r["seed"]), float(r["eps"])): r for r in rows}


def vec(R, m, seeds, eps, key="N80_pen"):
    return [R[(m, int(s), float(eps))][key] for s in seeds]


def complete(R, m, seeds, eps):
    return all((m, int(s), float(eps)) in R for s in seeds)


def mean_exh(rr):
    v = [r["exhaustion_at_k80"] for r in rr if r.get("exhaustion_at_k80")]
    if not v:
        return None
    return {k: round(float(np.mean([x[k] for x in v])), 4) for k in ("all", "ctrl", "treat")}


def method_stats(R, m, seeds, eps):
    rr = [R[(m, int(s), float(eps))] for s in seeds]
    tau = rr[0]["tau_R"]
    k80 = collections.Counter(-1 if r["k80"] is None else int(r["k80"]) for r in rr)
    mode_k, mode_n = k80.most_common(1)[0]
    d = {"geomean_N80_pen": _geo([r["N80_pen"] for r in rr]),
         "N80_over_tau": _geo([r["N80_pen"] / tau for r in rr]), "tau_R": tau,
         "n80_lt_tau": int(sum(bool(r["n80_lt_tau"]) for r in rr)), "n": len(rr),
         "false_streams": int(sum(bool(r["fwer_event"]) for r in rr)),
         "k80_dist": {str(k): v for k, v in sorted(k80.items())}, "k80_mode": mode_k,
         "k80_mode_share": mode_n / len(rr),
         "sd_log_N80": float(np.std(np.log([r["N80_pen"] for r in rr]), ddof=1)),
         "exhaustion_at_k80": mean_exh(rr),
         "sec_per_ck_median": float(np.median([v for r in rr for v in r["sec_cert_by_k"]])),
         "sec_per_ck_max": float(np.max([v for r in rr for v in r["sec_cert_by_k"]]))}
    d["pinned"] = bool(d["k80_mode_share"] >= 0.95)
    if "bnb_calls" in rr[0]:
        calls = int(sum(r["bnb_calls"] for r in rr))
        d.update({"bnb_calls": calls, "bnb_nodes": int(sum(r["bnb_nodes"] for r in rr)),
                  "bnb_nodes_per_call": (sum(r["bnb_nodes"] for r in rr) / calls) if calls else 0.0,
                  "bnb_nodes_max_ck": int(max(r["bnb_nodes_max_ck"] for r in rr)),
                  "node_limit_hits": int(sum(r["node_limit_hits"] for r in rr)),
                  "b_only_certs": int(sum(r["b_only_certs"] for r in rr)),
                  "a_certs": int(sum(r["a_certs"] for r in rr)), "beta": rr[0]["beta"]})
    return d


def ratio(R, a, b, seeds, eps, idx):
    from dsswm.stats.v6_analysis import paired
    r = paired(vec(R, a, seeds, eps), vec(R, b, seeds, eps), idx)
    return {k: r[k] for k in ("geomean_ratio", "ub95_one_sided", "ci95_two_sided", "frac_faster", "frac_tied",
                              "frac_slower", "sd_log_ratio", "n")}


def beta_c(S, A=2, K=K_GRID, dm=0.045):
    return math.log(2 * S * A * K / dm)


def rule_select(R, seeds, grid, rects=("RECT-BF-DP-TU", "HC-WoR-DP")):
    """Lock-v10 rival-success rule: smallest grid eps where the dev-best of the two named rectangles (lower geomean
    N80_pen; tie -> the first) has strict N80_pen < tau on >= 40/50 dev streams.  Never looks at FDC-DP."""
    trace = []
    for e in sorted(grid):
        cnt = {}
        for m in rects:
            if not complete(R, m, seeds, e):
                raise ValueError(f"rule: {m} incomplete at {e}")
            rr = [R[(m, int(s), float(e))] for s in seeds]
            cnt[m] = {"successes": int(sum(r["N80_pen"] < r["tau_R"] for r in rr)),
                      "geomean": _geo([r["N80_pen"] for r in rr])}
        best = min(rects, key=lambda m: (cnt[m]["geomean"], rects.index(m)))
        ok = cnt[best]["successes"] >= RULE_MIN_SUCCESS
        trace.append({"eps": e, "best": best, "per_rect": cnt, "qualifies": ok})
        if ok:
            return {"selected_eps": e, "rival": best, "trace": trace}
    return {"selected_eps": None, "rival": None, "trace": trace}


# =============================================================================================== G3 HC selection
def select_hc():
    """Declared rule: per (table, S), the HC config of HC_GRID with the lowest dev geomean N80_pen at the lock eps
    (dev seeds 950-999); ties -> grid order.  Written before any G3 eval row."""
    sel, detail = {}, {}
    for d in DATAS:
        for S in SGRID:
            R = by_key(load_rows(f"G3-dev-{d}-{S}"))
            e = LOCK_EPS[(d, S)]
            g = {}
            for name, (c, tf) in zip(HC_NAMES, HC_GRID):
                if not complete(R, name, DEV_SEEDS, e):
                    raise SystemExit(f"G3-dev-{d}-{S}: {name} incomplete")
                rr = [R[(name, s, e)] for s in DEV_SEEDS]
                g[name] = {"c": c, "tf": tf, "geomean_N80_pen": _geo([r["N80_pen"] for r in rr]),
                           "n80_lt_tau": int(sum(bool(r["n80_lt_tau"]) for r in rr)),
                           "false_streams": int(sum(bool(r["fwer_event"]) for r in rr))}
            best = min(HC_NAMES, key=lambda n: (g[n]["geomean_N80_pen"], HC_NAMES.index(n)))
            sel[f"{d}-{S}"] = {"name": best, "c": g[best]["c"], "tf": g[best]["tf"], "eps": e}
            detail[f"{d}-{S}"] = g
    out = {"rule": "lowest dev geomean N80_pen at the lock-v10 eps over the declared 8-config grid "
                   "(nstar schedule, c in {0.5, 0.75}, target_frac in {0.2, 0.4, 0.6, 0.8}); ties -> grid order; "
                   "dev seeds 950-999; selected before any G3 eval row", "selected": sel, "grid": detail,
           "written_at": datetime.now().isoformat(), "code_sha": code_sha()[0]}
    (OUT / "hc_tuned.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(sel, indent=1))


# =============================================================================================== analysis
def analyse():
    from dsswm.stats.v6_analysis import boot_idx
    from dsswm.baselines.fdc_dp import dp_ledger
    from dsswm.envs.seg_v10_eval import frozen_problems, load_frozen_seg
    csha = code_sha()[0]
    res = {"block": "v10-posthoc-G", "label": LABEL, "code_sha": csha, "written_at": datetime.now().isoformat(),
           "endpoint": "N80_pen (12/15 certified, run to 15/15 or tau); paired geomean ratio; paired percentile "
                       "bootstrap B = 1e4 seed 42, one-sided UB95 and two-sided 95% CI",
           "G1": {}, "G2": {}, "G3": {}, "G4": {}, "checks": {}}
    idx50, idx200 = boot_idx(50), boot_idx(200)
    # ---------------------------------------------------------------------------------------- G1
    for d in DATAS:
        for S in SGRID:
            for half, seeds, idx in (("dev", DEV_SEEDS, idx50), ("eval", G1_EVAL_SEEDS, idx200)):
                cid = f"G1-{half}-{d}-{S}"
                R = by_key(load_rows(cid, csha))
                cell = {}
                for e in EPS_GRID[d]:
                    if not all(complete(R, m, seeds, e) for m in G1_METHODS):
                        continue
                    ms = {m: method_stats(R, m, seeds, e) for m in G1_METHODS}
                    cell[str(e)] = {"eps": e, "lock_eps": e == LOCK_EPS[(d, S)], "methods": ms,
                                    "ratios": {"FDC/RECT-BF-DP-TU": ratio(R, "TU-FDC-DP(b)", "RECT-BF-DP-TU", seeds, e,
                                                                          idx),
                                               "FDC/HC-WoR-DP": ratio(R, "TU-FDC-DP(b)", "HC-WoR-DP", seeds, e, idx)}}
                rule = None
                if half == "dev" and cell:
                    try:
                        rule = rule_select(R, seeds, EPS_GRID[d])
                    except ValueError as ex:
                        rule = {"error": str(ex)}
                res["G1"][cid] = {"data": d, "S": S, "half": half, "n": len(seeds), "by_eps": cell, "rule": rule}
    # dev reproduction check against the lock-v10 pilot rows at the lock eps
    rep = {}
    for (d, S), t in LOCK_TASK.items():
        pt = RES / "pilots" / t.replace("_full_", "_pilot_") / "results.jsonl"
        R = by_key(load_rows(f"G1-dev-{d}-{S}", csha))
        if not pt.exists() or not R:
            continue
        e = LOCK_EPS[(d, S)]
        n_cmp = n_eq = 0
        for line in pt.read_text().splitlines():
            x = json.loads(line)
            k = (x["method"], int(x["seed"]), float(x["eps"]))
            if k in R:
                n_cmp += 1
                y = R[k]
                n_eq += int(all(x[f] == y[f] for f in ("N80_pen", "cert_k", "decided_pi", "schedule_digest")))
        rep[f"{d}-{S}"] = {"eps": e, "compared": n_cmp, "identical": n_eq}
    res["checks"]["G1_dev_reproduces_lock_pilot_rows"] = rep
    # ---------------------------------------------------------------------------------------- G2
    for d in DATAS:
        for S in SGRID:
            cid = f"G2-dev-{d}-{S}"
            R = by_key(load_rows(cid, csha))
            meta_f = ROWS / f"{cid}.meta.json"
            if not R or not meta_f.exists():
                continue
            meta = json.loads(meta_f.read_text())
            cell = {}
            for e in EPS_GRID[d]:
                if not all(complete(R, m, DEV_SEEDS, e) for m in G2_METHODS):
                    continue
                ms = {m: method_stats(R, m, DEV_SEEDS, e) for m in G2_METHODS}
                va = vec(R, "FDC-DP(a)", DEV_SEEDS, e)
                vb = vec(R, "FDC-DP(b)", DEV_SEEDS, e)
                cell[str(e)] = {"eps": e, "methods": ms,
                                "ratios": {"FDC(b)/RECT-BF-DP": ratio(R, "FDC-DP(b)", "RECT-BF-DP", DEV_SEEDS, e, idx50),
                                           "FDC(a)/RECT-BF-DP": ratio(R, "FDC-DP(a)", "RECT-BF-DP", DEV_SEEDS, e, idx50),
                                           "FDC(b)/HC-WoR-DP": ratio(R, "FDC-DP(b)", "HC-WoR-DP", DEV_SEEDS, e, idx50),
                                           "FDC(b)/FDC(a)": ratio(R, "FDC-DP(b)", "FDC-DP(a)", DEV_SEEDS, e, idx50)},
                                "streams_b_earlier_N80": int(sum(b < a for a, b in zip(va, vb))),
                                "streams_b_later_N80": int(sum(b > a for a, b in zip(va, vb)))}
            try:
                rule = rule_select(R, DEV_SEEDS, EPS_GRID[d], rects=("RECT-BF-DP", "HC-WoR-DP"))
            except ValueError as ex:
                rule = {"error": str(ex)}
            fb = next(iter(cell.values()))["methods"]["FDC-DP(b)"]["beta"] if cell else None
            res["G2"][cid] = {"data": d, "S": S, "costs": meta["costs"], "budgets": meta["budgets"],
                              "cost_source": meta["cost_source"], "beta_J": fb, "beta_C": beta_c(S),
                              "beta_ratio": (fb / beta_c(S)) if fb else None, "by_eps": cell, "rule": rule}
    # ---------------------------------------------------------------------------------------- G3
    hcf = OUT / "hc_tuned.json"
    hc = json.loads(hcf.read_text()) if hcf.exists() else None
    res["G3"]["hc_selection"] = hc
    for d in DATAS:
        for S in SGRID:
            e = LOCK_EPS[(d, S)]
            Rd = by_key(load_rows(f"G3-dev-{d}-{S}", csha) + load_rows(f"G1-dev-{d}-{S}", csha))
            out = {"eps": e, "dev": None, "eval": None}
            if hc and all(complete(Rd, m, DEV_SEEDS, e) for m in ("TU-FDC-DP(b)", "RECT-BF-DP-TU", "RECT-HG-DP")):
                tn = hc["selected"][f"{d}-{S}"]["name"]
                ms = ("TU-FDC-DP(b)", "RECT-BF-DP-TU", "RECT-HG-DP", tn, "HC-WoR-DP")
                out["dev"] = {"hc_tuned": tn, "methods": {m: method_stats(Rd, m, DEV_SEEDS, e) for m in ms},
                              "ratios": {f"FDC/{m}": ratio(Rd, "TU-FDC-DP(b)", m, DEV_SEEDS, e, idx50)
                                         for m in ms[1:]} |
                              {"RECT-HG-DP/RECT-BF-DP-TU": ratio(Rd, "RECT-HG-DP", "RECT-BF-DP-TU", DEV_SEEDS, e,
                                                                 idx50)},
                              "note": "dev HC-tuned ratio is selection-optimistic for HC (tuned on these streams)"}
            Re = by_key(load_rows(f"G3-eval-{d}-{S}", csha))
            if all(complete(Re, m, G3_EVAL_SEEDS, e) for m in G3_EVAL_METHODS):
                out["eval"] = {"methods": {m: method_stats(Re, m, G3_EVAL_SEEDS, e) for m in G3_EVAL_METHODS},
                               "ratios": {f"FDC/{m}": ratio(Re, "TU-FDC-DP(b)", m, G3_EVAL_SEEDS, e, idx200)
                                          for m in G3_EVAL_METHODS[1:]} |
                               {"RECT-HG-DP/RECT-BF-DP-TU": ratio(Re, "RECT-HG-DP", "RECT-BF-DP-TU", G3_EVAL_SEEDS,
                                                                  e, idx200)}}
            res["G3"][f"{d}-{S}"] = out
    # ---------------------------------------------------------------------------------------- G4
    fz = load_frozen_seg()
    g4 = {"S9_reference": {"beta_J": 14.21, "beta_C": 9.68, "ratio": 1.47, "source": "plan/fdc_dp_theory.md s2"}}
    for S in SGRID:
        sp = frozen_problems(fz, "x5", S)
        led = dp_ledger(sp, S, 2, K_GRID)
        g4[str(S)] = {"beta_J": led["beta"], "beta_C": beta_c(S), "ratio": led["beta"] / beta_c(S),
                      "union_size_log10": led["union_size_log"] / math.log(10)}
    lock_obs = {}
    for blk in ("A", "B", "D"):
        f = RES / "full" / f"v10_analysis_{blk}.json"
        if f.exists():
            a = json.loads(f.read_text())
            for t, c in a["cells"].items():
                r = c["comparisons"]["TU-FDC-DP(b)/RECT-BF-DP-TU"]
                lock_obs[t] = {"eps": c["eps"], "ratio": r["geomean_ratio"], "ub95": r["ub95_one_sided"],
                               "vs_HC": c["comparisons"]["TU-FDC-DP(b)/HC-WoR-DP"]["geomean_ratio"]}
    g4["lock_v10_observed"] = lock_obs
    g4["v9_S9_observed"] = "0.53-0.71 rows ratio vs time-uniform rivals (v9; v10_summary.md)"
    res["G4"] = g4
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(res, indent=1, default=str))
    plot_curves(res)
    write_tex(res)
    print("analysis written", OUT)
    return res


def plot_curves(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = {"TU-FDC-DP(b)": "#2a78d6", "RECT-BF-DP-TU": "#eb6834", "HC-WoR-DP": "#1baf7a"}
    lab = {"TU-FDC-DP(b)": "TU-FDC-DP(b)", "RECT-BF-DP-TU": "RECT-BF-DP-TU", "HC-WoR-DP": "HC-WoR-DP (untuned)"}
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2), sharey="row")
    for i, d in enumerate(DATAS):
        for j, S in enumerate(SGRID):
            ax = axes[i, j]
            for half, ls, mk, alpha in (("eval", "-", "o", 1.0), ("dev", "--", None, 0.55)):
                g = res["G1"].get(f"G1-{half}-{d}-{S}", {}).get("by_eps", {})
                if not g:
                    continue
                es = sorted(float(k) for k in g)
                for m in G1_METHODS:
                    y = [g[str(e)]["methods"][m]["N80_over_tau"] for e in es]
                    ax.plot(es, y, ls=ls, marker=mk, ms=4, color=cols[m], alpha=alpha, lw=1.6 if half == "eval" else 1,
                            label=f"{lab[m]} ({half})" if (i, j) == (0, 0) else None)
            ax.axvline(LOCK_EPS[(d, S)], color="#999999", lw=0.8, ls=":")
            ax.set_title(f"{'X5' if d == 'x5' else 'Lenta'}, S = {S}", fontsize=10)
            ax.set_ylim(0, 1.05)
            ax.grid(alpha=0.25, lw=0.5)
            ax.spines[["top", "right"]].set_visible(False)
            if i == 1:
                ax.set_xlabel("ε")
            if j == 0:
                ax.set_ylabel("N80 / τ (geomean)")
    fig.legend(loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle("Reads to certify 12/15 problems against ε (post hoc, descriptive; dotted = lock-v10 ε)", fontsize=10)
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    fig.savefig(OUT / "curves.pdf")
    fig.savefig(OUT / "curves.png", dpi=150)
    plt.close(fig)


def write_tex(res):
    L = [r"% v10-posthoc-G (post hoc, descriptive): G1 eval, TU-FDC-DP(b) / rival N80 ratio by eps",
         r"\begin{tabular}{llrrrrrl}", r"\toprule",
         r"Table & $S$ & $\varepsilon$ & FDC $N_{80}/\tau$ & RECT $N_{80}/\tau$ & ratio vs RECT [95\% CI] & "
         r"ratio vs HC & RECT pinned \\", r"\midrule"]
    for d in DATAS:
        for S in SGRID:
            g = res["G1"].get(f"G1-eval-{d}-{S}", {}).get("by_eps", {})
            for k in sorted(g, key=float):
                c = g[k]
                r = c["ratios"]["FDC/RECT-BF-DP-TU"]
                L.append(f"{'X5' if d == 'x5' else 'Lenta'} & {S} & {float(k):g} & "
                         f"{c['methods']['TU-FDC-DP(b)']['N80_over_tau']:.3f} & "
                         f"{c['methods']['RECT-BF-DP-TU']['N80_over_tau']:.3f} & {r['geomean_ratio']:.3f} "
                         f"[{r['ci95_two_sided'][0]:.3f}, {r['ci95_two_sided'][1]:.3f}] & "
                         f"{c['ratios']['FDC/HC-WoR-DP']['geomean_ratio']:.3f} & "
                         f"{'yes' if c['methods']['RECT-BF-DP-TU']['pinned'] else 'no'} \\\\")
            L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}", ""]
    L += [r"% G3: stronger rivals at the lock-v10 cells (eval, seeds 38600-38799)", r"\begin{tabular}{llrrrr}",
          r"\toprule", r"Table & $S$ & $\varepsilon$ & vs RECT-BF-DP-TU & vs RECT-HG-DP & vs HC-tuned \\",
          r"\midrule"]
    for d in DATAS:
        for S in SGRID:
            ev = res["G3"].get(f"{d}-{S}", {}).get("eval")
            if not ev:
                continue
            rr = ev["ratios"]
            f = lambda k: f"{rr[k]['geomean_ratio']:.3f} ({rr[k]['ub95_one_sided']:.3f})"  # noqa: E731
            L.append(f"{'X5' if d == 'x5' else 'Lenta'} & {S} & {LOCK_EPS[(d, S)]:g} & {f('FDC/RECT-BF-DP-TU')} & "
                     f"{f('FDC/RECT-HG-DP')} & {f('FDC/HC-WoR-DP[tuned]')} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", ""]
    L += [r"% G4: union cost against S", r"\begin{tabular}{rrrr}", r"\toprule",
          r"$S$ & $\beta_J$ & $\beta_C$ & $\beta_J/\beta_C$ \\", r"\midrule",
          r"9 & 14.21 & 9.68 & 1.47 \\"]
    for S in SGRID:
        g = res["G4"][str(S)]
        L.append(f"{S} & {g['beta_J']:.2f} & {g['beta_C']:.2f} & {g['ratio']:.2f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", ""]
    (OUT / "table.tex").write_text("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--select-hc", action="store_true")
    ap.add_argument("--analyse", action="store_true")
    a = ap.parse_args()
    if a.list:
        print("\n".join(CELLS))
        return
    if a.select_hc:
        select_hc()
        return
    if a.analyse:
        analyse()
        return
    workers = max(1, min(4, a.workers))
    for cid in a.cells.split(","):
        if cid not in CELLS:
            raise SystemExit(f"unknown cell {cid}")
        if cid.startswith("G3-eval") and not (OUT / "hc_tuned.json").exists():
            raise SystemExit("G3 eval needs hc_tuned.json (run --select-hc after the G3 dev cells)")
        run_cell(cid, workers)


if __name__ == "__main__":
    main()
