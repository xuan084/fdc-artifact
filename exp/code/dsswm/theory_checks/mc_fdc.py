"""Monte Carlo checks for Theorem FDC-1 (task r5_fdc_spec_theory; plan/theory/fdc_theorem.md).

FDC = frozen 50/50 read schedule (exhausted arm -> re-selection with the pre-drawn uniform, StreamEngine._resel)
      + direction-level Bernstein certificate (certify.quadknap.qfc_certificate_enum, Lemma L1 width)
      + Lemma L2 inversion variance UCB (certify.quadknap.make_stats)
      + ledger computed from ctx only:  beta = ln(sum_q |Pi_{B_q}| * K / delta_main),
                                         x_v  = ln(2 * S * A * K / delta_var).

Parts
-----
0  Ledger on the CR9 development-half ctx (public quantities only: w, pool sizes, problems, checkpoints; no outcome
   and no permutation seed is touched): sum_q |Pi_{B_q}|, K, S*A, beta, x_v, and the size of the bound obtained with
   the hard-coded beta = 14.1.
A  Full-certificate MC on synthetic finite pools (seed 42):
     40 near-tie configurations + 20 early-exhaustion configurations, 500 streams each, eps in {0, 0.0025, 0.01},
     two observation windows per stream:
       W1  stop at the first checkpoint with >= 12/15 certified problems (sticky answers);
       W2  run to tau_R and certify all 15 problems (sticky answers), plus W2-any: any (k, q) with U_q(k) <= eps.
     Direct event checks with the truth: E_var misses (mu_c outside CI_c(k)), E_main misses (true-variance
     direction events for (pi*_q, pi), pi in Pi_{B_q}), and edge-case counters.
B  Power controls on the same streams: plug-in variance (no E_var), beta_0 = ln(1/delta) (no union), and both.
C  Schedule independence: counts n_c(k) and schedule digests are identical under outcome re-labelling.

CPU only; <= 4 worker processes; BLAS threads pinned to 1. Never reads any CR evaluation seed or CR stream.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np

_CODE = Path(__file__).resolve().parents[2]
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from dsswm.baselines.frontier_common import make_ctx  # noqa: E402
from dsswm.certify.quadknap import QFCStats, make_stats, pair_terms, qfc_certificate_enum, _width  # noqa: E402
from dsswm.envs.pool_replay import make_schedule  # noqa: E402
from dsswm.streams import frontier as fr  # noqa: E402
from dsswm.streams.frontier_runner import StreamEngine, SyntheticPoolEnv  # noqa: E402
from dsswm.theory_checks.mc_l1 import bernstein_mu_ci, cp_upper  # noqa: E402

TASK_ID = "r5_fdc_spec_theory"
DELTA, DELTA_MAIN, DELTA_VAR = 0.05, 0.045, 0.005
K = 20
EPS = (0.0, 0.0025, 0.01)
CERTS = ("FDC", "ctrl_plugin", "ctrl_beta0", "ctrl_both")
STOP_FRAC = 0.8


# =============================================================================================== ledger
def fdc_ledger(ctx, delta_main=DELTA_MAIN, delta_var=DELTA_VAR):
    """FDC ledger from the ctx only (no overrides). Union sizes are counted exactly."""
    sigma = int(ctx.feas.sum())
    Kc = int(len(ctx.checkpoints))
    C = int(ctx.S * ctx.A)
    beta = math.log(sigma * Kc / delta_main)
    x_v = math.log(2 * C * Kc / delta_var)
    return {"sigma_Pi_Bq": sigma, "K": Kc, "C_var": C, "delta_main": delta_main, "delta_var": delta_var,
            "beta": beta, "x_v": x_v, "per_q_feasible": ctx.feas.sum(1).astype(int).tolist(),
            "bound_main": sigma * Kc * math.exp(-beta), "bound_var": 2 * C * Kc * math.exp(-x_v)}


def part0_cr9_ledger():
    from dsswm.envs.pool_replay import PoolReplayEnv
    env = PoolReplayEnv("CR9", "dev")
    ctx = make_ctx(env.w, env.pool_sizes, fr.cr_problems(), 0.001, DELTA, env.checkpoints(K), env.replan_interval,
                   env.tau_R)
    led = fdc_ledger(ctx)
    led.update({"layer": "CR9", "half": "dev", "S": env.S, "A": env.A, "P": int(ctx.P), "tau_R": int(env.tau_R),
                "checkpoints": ctx.checkpoints.tolist(), "pool_sizes": env.pool_sizes.tolist(),
                "beta_14p1_main_bound": led["sigma_Pi_Bq"] * K * math.exp(-14.1),
                "beta_14p1_total_bound": led["sigma_Pi_Bq"] * K * math.exp(-14.1) + DELTA_VAR,
                "beta_4dp": round(led["beta"], 4), "x_v_4dp": round(led["x_v"], 4),
                "matches_spec": bool(round(led["beta"], 4) == 14.2242 and round(led["x_v"], 4) == 11.8776
                                     and led["sigma_Pi_Bq"] == 3386)})
    return led


# =============================================================================================== synthetic configs
def make_config(rng, cid):
    """Near-tie (cid < 40) or early-exhaustion (cid >= 40) synthetic finite-pool configuration (A = 2, binary)."""
    early = cid >= 40
    S = int(rng.integers(3, 7))
    A = 2
    seg_n = np.exp(rng.uniform(np.log(300), np.log(30000), S)).astype(int)
    if early:
        ctrl_share = rng.uniform(0.03, 0.2, S)                   # 50/50 exhausts control after ~2*share of arrivals
        tiny = rng.integers(S)
        seg_n[tiny] = int(rng.integers(20, 60))                 # a tiny segment (n = 0 cells at early checkpoints)
    else:
        ctrl_share = rng.uniform(0.15, 0.5, S)
    sizes = np.zeros((S, A), dtype=np.int64)
    sizes[:, 0] = np.maximum(1, np.round(seg_n * ctrl_share)).astype(int)
    sizes[:, 1] = np.maximum(1, seg_n - sizes[:, 0])
    mu = np.zeros((S, A))
    base_kind = rng.choice(["zero", "small", "mid"], size=S, p=[0.2, 0.4, 0.4])
    for s in range(S):
        b = {"zero": 0.0, "small": rng.uniform(0.002, 0.05), "mid": rng.uniform(0.1, 0.4)}[base_kind[s]]
        # near-tie uplifts: a common value plus small perturbations (exact ties with prob 0.3)
        mu[s, 0] = b
        mu[s, 1] = b
    common = rng.uniform(-0.01, 0.04)
    for s in range(S):
        if base_kind[s] == "zero" and rng.random() < 0.5:
            continue                                            # all-zero pools on both arms (mu_hat in {0})
        tau = common if rng.random() < 0.3 else common + rng.normal(0, 0.015)
        mu[s, 1] = float(np.clip(mu[s, 0] + tau, 0.0, 1.0))
    if early and rng.random() < 0.5:
        s1 = int(rng.integers(S))
        mu[s1, 1] = 1.0                                         # an all-ones pool (mu_hat = 1)
    n_min = int(rng.integers(5, 15)) if early else int(rng.integers(20, 60))
    return {"cid": cid, "kind": "early_exhaust" if early else "near_tie", "S": S, "A": A,
            "sizes": sizes.tolist(), "mu": mu.tolist(), "n_min": n_min}


def build(cfg, seed=0):
    env = SyntheticPoolEnv(np.array(cfg["sizes"]), cfg["mu"], seed=seed, n_min=cfg["n_min"])
    problems = fr.cr_problems("visit")
    ctx = make_ctx(env.w, env.pool_sizes, problems, 0.0, DELTA, env.checkpoints(K), env.replan_interval, env.tau_R,
                   stop_frac=STOP_FRAC)
    return env, ctx


# =============================================================================================== one stream
def _stats(ctx, mu_hat, n, N, var):
    return QFCStats(w=ctx.w, mu_hat=mu_hat, n=n, var_ucb=var, N=N, R=1.0)


def run_config(args):
    cfg, n_streams, seed_base = args
    t0 = time.perf_counter()
    env, ctx = build(cfg)
    led = fdc_ledger(ctx)
    beta, x_v = led["beta"], led["x_v"]
    beta0 = math.log(1.0 / DELTA)
    S, A, Q, P = ctx.S, ctx.A, ctx.Q, ctx.P
    pols = ctx.pols
    idx = {tuple(int(x) for x in r): i for i, r in enumerate(pols)}
    mu = env.true_mu("visit")
    sig2 = env.true_sigma2("visit")
    seg = np.arange(S)
    J = (ctx.w[None, :] * mu[seg[None, :], pols]).sum(1)
    pistar = np.array([int(np.argmax(np.where(ctx.feas[q], J, -np.inf))) for q in range(Q)])   # lowest index on ties
    Jstar = J[pistar]
    gaps = np.array([np.sort(Jstar[q] - J[ctx.feas[q]])[1] if ctx.feas[q].sum() > 1 else np.inf for q in range(Q)])
    N = env.pool_sizes.astype(float)
    alloc = np.full((S, A), 1.0 / A)                             # frozen 50/50
    tallies = {(c, w, e): 0 for c in CERTS for w in ("W1", "W2", "W2any") for e in EPS}
    certs_made = {(c, e): 0 for c in CERTS for e in EPS}
    stop_reached = {(c, e): 0 for c in CERTS for e in EPS}
    pre_final_cert = {e: 0 for e in EPS}                       # FDC certifications at k < K-1 (non-vacuity)
    evar_miss = emain_miss = emain_events = 0
    edge = {"n0_cell_ck": 0, "exhausted_cell_ck": 0, "mu_hat_01_live_cell_ck": 0, "both_arms_exhausted_seg_ck": 0,
            "pairs_diff_only_exhausted": 0, "inf_width_pairs": 0, "resel_arrivals": 0, "skipped_arrivals": 0}
    for j in range(n_streams):
        perm_seed = seed_base + j
        sch = make_schedule(env, perm_seed, alloc=alloc)
        eng = StreamEngine(env, sch, "visit", delta_cell=DELTA / (S * A))
        state = {(c, e): {"und": np.ones(Q, bool), "dec": np.full(Q, -1), "false": np.zeros(Q, bool),
                          "stopped": False, "w1_false": False, "any_false": False} for c in CERTS for e in EPS}
        for k, t in enumerate(ctx.checkpoints):
            eng.advance_planned(int(t))
            n = eng.n.astype(float)
            mu_hat = np.where(n > 0, eng.sum / np.maximum(n, 1), 0.5)
            st_l2 = make_stats(ctx.w, mu_hat, n, N=N, x_v=x_v)
            st_pl = _stats(ctx, mu_hat, n, N, mu_hat * (1 - mu_hat))
            res = {"FDC": qfc_certificate_enum(st_l2, ctx.problems, beta, 0.0, pols=pols),
                   "ctrl_plugin": qfc_certificate_enum(st_pl, ctx.problems, beta, 0.0, pols=pols),
                   "ctrl_beta0": qfc_certificate_enum(st_l2, ctx.problems, beta0, 0.0, pols=pols),
                   "ctrl_both": qfc_certificate_enum(st_pl, ctx.problems, beta0, 0.0, pols=pols)}
            # ---- direct event checks (truth side)
            lo, hi = bernstein_mu_ci(mu_hat, n, N, x_v)
            live = (n > 0) & (n < N)
            if np.any(live & ((mu < lo - 1e-12) | (mu > hi + 1e-12))):
                evar_miss += 1
            st_true = _stats(ctx, mu_hat, n, N, sig2)
            for q in range(Q):
                dh, V, b = pair_terms(st_true, pols, pols[pistar[q]])      # dh = Jhat(pi) - Jhat(pi*)
                wdt = _width(V, b, beta)
                viol = (Jstar[q] - J) + dh > wdt + 1e-12                    # J(pi*)-J(pi) > Jhat(pi*)-Jhat(pi)+w
                emain_events += int(ctx.feas[q].sum())
                if np.any(viol & ctx.feas[q]):
                    emain_miss += 1
            # ---- edge counters
            edge["n0_cell_ck"] += int((n <= 0).sum())
            exh = n >= N
            edge["exhausted_cell_ck"] += int(exh.sum())
            edge["mu_hat_01_live_cell_ck"] += int((live & ((mu_hat == 0) | (mu_hat == 1))).sum())
            edge["both_arms_exhausted_seg_ck"] += int(exh.all(1).sum())
            for r in res["FDC"]:
                ih = idx[r["pi_hat"]]
                d = pols != pols[ih][None, :]
                both = exh[seg[None, :], pols] & exh[seg, pols[ih]][None, :]
                only_exh = d.any(1) & ~(d & ~both).any(1)
                edge["pairs_diff_only_exhausted"] += int(only_exh.sum())
                if not np.isfinite(r["U"]):
                    edge["inf_width_pairs"] += 1
            # ---- certification bookkeeping
            for c in CERTS:
                for e in EPS:
                    sd = state[(c, e)]
                    for q, r in enumerate(res[c]):
                        if r["U"] <= e:
                            ih = idx[r["pi_hat"]]
                            wrong = bool(Jstar[q] - J[ih] > e + 1e-12)
                            if wrong:
                                sd["any_false"] = True
                            if sd["und"][q]:
                                sd["und"][q] = False
                                sd["dec"][q] = ih
                                sd["false"][q] = wrong
                                certs_made[(c, e)] += 1
                                if c == "FDC" and k < len(ctx.checkpoints) - 1:
                                    pre_final_cert[e] += 1
                    if not sd["stopped"]:
                        sd["w1_false"] = bool(sd["false"].any())
                        if (~sd["und"]).sum() >= ctx.stop_k:
                            sd["stopped"] = True
                            stop_reached[(c, e)] += 1
        edge["resel_arrivals"] += int(eng.n_reselected)
        edge["skipped_arrivals"] += int(eng.n_skipped)
        assert eng.billing_ok()
        for c in CERTS:
            for e in EPS:
                sd = state[(c, e)]
                tallies[(c, "W1", e)] += int(sd["w1_false"])
                tallies[(c, "W2", e)] += int(sd["false"].any())
                tallies[(c, "W2any", e)] += int(sd["any_false"])
    return {"cfg": cfg, "ledger": {k: v for k, v in led.items() if k != "per_q_feasible"},
            "n_streams": n_streams, "min_gap": float(np.min(gaps)), "median_gap": float(np.median(gaps)),
            "tallies": {f"{c}|{w}|{e}": v for (c, w, e), v in tallies.items()},
            "certs_made": {f"{c}|{e}": v for (c, e), v in certs_made.items()},
            "stop_reached": {f"{c}|{e}": v for (c, e), v in stop_reached.items()},
            "fdc_pre_final_certs": {str(e): v for e, v in pre_final_cert.items()},
            "evar_miss_ck": evar_miss, "emain_miss_qck": emain_miss, "emain_events": emain_events,
            "edge": edge, "sec": round(time.perf_counter() - t0, 1)}


# =============================================================================================== part C
def schedule_independence(cfgs, seeds=(1, 2, 3)):
    out = []
    for cfg in cfgs:
        envs = [build(cfg, seed=s)[0] for s in (0, 7)]           # same labels, different outcome placement
        _, ctx = build(cfg)
        alloc = np.full((cfg["S"], cfg["A"]), 0.5)
        for ps in seeds:
            counts, dig = [], []
            for env in envs:
                sch = make_schedule(env, 10_000_000 + ps, alloc=alloc)
                eng = StreamEngine(env, sch, "visit", delta_cell=0.01)
                cs = []
                for t in ctx.checkpoints:
                    eng.advance_planned(int(t))
                    cs.append(eng.n.copy())
                counts.append(np.array(cs))
                dig.append(sch.digest())
            out.append({"cid": cfg["cid"], "seed": ps, "counts_equal": bool(np.array_equal(*counts)),
                        "digest_equal": dig[0] == dig[1],
                        "outcomes_differ": bool(np.any(envs[0].outcomes_view("visit") != envs[1].outcomes_view("visit")))})
    return out


# =============================================================================================== io
def write_progress(rd, epoch, total, metric=None):
    (rd / f"{TASK_ID}_PROGRESS.json").write_text(json.dumps({
        "task_id": TASK_ID, "epoch": epoch, "total_epochs": total, "step": epoch, "total_steps": total,
        "loss": None, "metric": metric or {}, "updated_at": datetime.now().isoformat()}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--streams", type=int, default=500)
    ap.add_argument("--n_near", type=int, default=40)
    ap.add_argument("--n_early", type=int, default=20)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="exp/results/pilots/r5_fdc_spec_theory")
    ap.add_argument("--results_dir", default="exp/results")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rd = Path(a.results_dir)
    t_start = time.time()
    led0 = part0_cr9_ledger()
    print(f"[ledger] CR9 dev: sum|Pi_Bq|={led0['sigma_Pi_Bq']} K={led0['K']} C_var={led0['C_var']} "
          f"beta={led0['beta']:.4f} x_v={led0['x_v']:.4f} beta14.1 total={led0['beta_14p1_total_bound']:.5f}",
          flush=True)
    rng = np.random.default_rng(a.seed)
    cids = list(range(a.n_near)) + list(range(40, 40 + a.n_early))
    cfgs = [make_config(rng, c) for c in cids]
    jobs = [(cfg, a.streams, 1_000_000 + 10_000 * cfg["cid"]) for cfg in cfgs]
    results = []
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(run_config, jobs)):
            results.append(r)
            write_progress(rd, i + 1, len(jobs), {"configs_done": i + 1})
            print(f"[mc] cfg {r['cfg']['cid']:2d} ({r['cfg']['kind']}) S={r['cfg']['S']} "
                  f"FDC W2any@0={r['tallies']['FDC|W2any|0.0']} both W2any@0.01={r['tallies']['ctrl_both|W2any|0.01']} "
                  f"Evar={r['evar_miss_ck']} Emain={r['emain_miss_qck']} {r['sec']}s", flush=True)
    results.sort(key=lambda r: r["cfg"]["cid"])
    indep = schedule_independence(cfgs[:3] + cfgs[-3:])

    # ---- csvs
    with open(out / "mc_cert_coverage.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["cid", "kind", "S", "min_gap", "beta", "x_v", "eps", "window", "n_streams", "n_false_streams",
                     "cp95_upper", "fdc_certs", "fdc_pre_final_certs", "stop_reached", "evar_miss_ck", "emain_miss_qck"])
        for r in results:
            for e in EPS:
                for w in ("W1", "W2", "W2any"):
                    nf = r["tallies"][f"FDC|{w}|{e}"]
                    wr.writerow([r["cfg"]["cid"], r["cfg"]["kind"], r["cfg"]["S"], f"{r['min_gap']:.6g}",
                                 f"{r['ledger']['beta']:.4f}", f"{r['ledger']['x_v']:.4f}", e, w, r["n_streams"], nf,
                                 f"{cp_upper(nf, r['n_streams']):.5f}", r["certs_made"][f"FDC|{e}"],
                                 r["fdc_pre_final_certs"][str(e)], r["stop_reached"][f"FDC|{e}"],
                                 r["evar_miss_ck"], r["emain_miss_qck"]])
    with open(out / "power_control.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["cid", "kind", "certificate", "eps", "window", "n_streams", "n_false_streams", "rate",
                     "exceeds_delta"])
        for r in results:
            for c in CERTS:
                for e in EPS:
                    for w in ("W1", "W2", "W2any"):
                        nf = r["tallies"][f"{c}|{w}|{e}"]
                        wr.writerow([r["cfg"]["cid"], r["cfg"]["kind"], c, e, w, r["n_streams"], nf,
                                     f"{nf / r['n_streams']:.4f}", int(nf / r["n_streams"] > DELTA)])

    # ---- aggregate
    def agg(cert, w, e, kind=None):
        rs = [r for r in results if kind is None or r["cfg"]["kind"] == kind]
        nf = sum(r["tallies"][f"{cert}|{w}|{e}"] for r in rs)
        ns = sum(r["n_streams"] for r in rs)
        mx = max(r["tallies"][f"{cert}|{w}|{e}"] / r["n_streams"] for r in rs)
        n_exceed = sum(r["tallies"][f"{cert}|{w}|{e}"] / r["n_streams"] > DELTA for r in rs)
        return {"false_streams": nf, "streams": ns, "pooled_rate": nf / ns, "cp95_upper_pooled": cp_upper(nf, ns),
                "max_config_rate": mx, "configs_rate_gt_delta": int(n_exceed), "configs_with_any_false":
                sum(r["tallies"][f"{cert}|{w}|{e}"] > 0 for r in rs)}

    fdc = {f"{w}|{e}": agg("FDC", w, e) for w in ("W1", "W2", "W2any") for e in EPS}
    ctrl = {c: {f"{w}|{e}": agg(c, w, e) for w in ("W1", "W2", "W2any") for e in EPS} for c in CERTS[1:]}
    fdc_total_false = sum(r["tallies"][f"FDC|W2any|{e}"] for r in results for e in EPS)
    worst_cp = max(cp_upper(r["tallies"][f"FDC|W2any|{e}"], r["n_streams"]) for r in results for e in EPS)
    cells = len(results) * 2
    edge = {k: sum(r["edge"][k] for r in results) for k in results[0]["edge"]}
    both_ctrl_power = {f"{w}|{e}": ctrl["ctrl_both"][f"{w}|{e}"]["configs_with_any_false"]
                       for w in ("W1", "W2") for e in EPS}
    power_ok = any(ctrl["ctrl_both"][f"{w}|{e}"]["configs_rate_gt_delta"] > 0 for w in ("W1", "W2") for e in EPS)
    certs_total = sum(r["certs_made"][f"FDC|{e}"] for r in results for e in EPS)
    pre_final = sum(r["fdc_pre_final_certs"][str(e)] for r in results for e in EPS)
    summary = {
        "task_id": TASK_ID, "seed": a.seed, "mode": "PILOT", "timestamp": datetime.now().isoformat(),
        "cr9_dev_ledger": led0,
        "mc": {"n_configs": len(results), "n_near_tie": a.n_near, "n_early_exhaust": a.n_early,
               "streams_per_config": a.streams, "eps_grid": list(EPS), "windows": ["W1", "W2", "W2any"],
               "cells_config_x_window": cells, "fdc": fdc, "controls": ctrl,
               "fdc_false_streams_all_windows_eps": fdc_total_false, "fdc_worst_config_cp95_upper": worst_cp,
               "fdc_certifications_total": certs_total, "fdc_certifications_before_final_checkpoint": pre_final,
               "evar_miss_checkpoints": sum(r["evar_miss_ck"] for r in results),
               "emain_miss_problem_checkpoints": sum(r["emain_miss_qck"] for r in results),
               "emain_events_checked": sum(r["emain_events"] for r in results),
               "edge_case_counters": edge, "ctrl_both_configs_with_false": both_ctrl_power,
               "min_gap_quantiles": np.quantile([r["min_gap"] for r in results], [0, .25, .5, .75, 1]).tolist()},
        "schedule_independence": {"checks": len(indep), "all_counts_equal": all(x["counts_equal"] for x in indep),
                                  "all_digests_equal": all(x["digest_equal"] for x in indep),
                                  "outcomes_differ": all(x["outcomes_differ"] for x in indep)},
        "metrics": {"n_false_mc": int(fdc_total_false), "cp_upper": worst_cp, "beta": round(led0["beta"], 4),
                    "x_v": round(led0["x_v"], 4)},
        "configs": [{"cid": r["cfg"]["cid"], "kind": r["cfg"]["kind"], "S": r["cfg"]["S"], "sizes": r["cfg"]["sizes"],
                     "mu": r["cfg"]["mu"], "n_min": r["cfg"]["n_min"], "beta": r["ledger"]["beta"],
                     "x_v": r["ledger"]["x_v"], "min_gap": r["min_gap"], "sec": r["sec"]} for r in results],
        "sec_total": round(time.time() - t_start, 1),
    }
    summary["gate"] = {
        "zero_false_certifications": fdc_total_false == 0,
        "invalid_control_detects_violations": bool(power_ok),
        "ledger_matches_spec": led0["matches_spec"],
        "schedule_independent": summary["schedule_independence"]["all_counts_equal"],
        "selfcheck_signed": None,     # filled after the proof self-check (plan/theory/fdc_theorem.md s10)
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({"gate": summary["gate"], "metrics": summary["metrics"], "sec": summary["sec_total"]}))
    return summary


if __name__ == "__main__":
    main()
