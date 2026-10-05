"""r5_t1_reconcile (PILOT == FULL): T1 version reconciliation + C4 dev factorial + frozen C4 prediction model.

Setting: CR9 dev half, eps* = 0.001, 15 problems, stop at 12/15, delta = 0.05, dev seeds 900-999 (100 paired streams).
No evaluation seed (30000+) is touched.

(A) Reproduce the three exploratory numbers (each within +-2 %):
      0.856 = FIX0.40-rig-b14.1 / B2-fav   (contrarian RB class, idea/r5_offline/contrarian/offline_ctr.py, K = 20, 100 str)
      0.887 = QFC-half / B2-fav            (pragmatist offline_k60.py: r4 ledger, K = 60, seeds 900-929)
      0.97  = FIX0.50-fav (beta = ln(P^2 K / delta) = 18.47) / B2-fav  (text-only number, regenerated here, 100 str)
    The contrarian RB class is imported from the original file (sha256 recorded), not copied.
(B) Factor-swap ladder on the same streams (each step one factor), geometric mean N80_pen, ratio to B2-fav (matched K)
    with paired bootstrap CI, per-step log change; plus one-factor-at-a-time (OFAT) effects from the FDC base and the
    log-additivity residual  log(target / FDC) - sum OFAT  for each of the three numbers.
(C) C4 dev factorial: FDCAblation 2 x 2 x 2 (design x union x ledger) x 100 streams: main effects, interactions,
    Shapley shares (bootstrap CI), and FDC vs B4-bal (share frozen by r5_rival_tuning).
(D) C4 prediction model: deterministic-equivalent model (theoretical offline_proj.py; true cell means, expected counts
    of the design, the actual certificate code with each cell's beta / x_v) calibrated on seeds 900-949 ONLY, frozen
    to exp/results/r5_gates/c4_model.json (sha256) BEFORE seeds 950-999 are run; then per-cell log errors with
    bootstrap CI on 950-999.
All beta = 14.1 configurations and every '-explore' / plug-in configuration are exploratory (validity != rigorous).
CPU only, 4 worker processes, BLAS threads pinned to 1 (concurrent run: timings are inflated).
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import csv
import hashlib
import importlib.util
import itertools
import json
import math
import sys
import time
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
WS = CODE.parents[1]
TASK = "r5_t1_reconcile"
RES = WS / "exp/results"
OUT = RES / "pilots" / TASK
GATES = RES / "r5_gates"
EPS = 0.001
CAL_SEEDS = list(range(900, 950))
VAL_SEEDS = list(range(950, 1000))
SEEDS = CAL_SEEDS + VAL_SEEDS
K60_REPRO_SEEDS = list(range(900, 930))
IDENTITY_SEEDS = (900, 901, 902)
B_BOOT = 10_000
CONTRARIAN = WS / "idea/r5_offline/contrarian/offline_ctr.py"
PRAGMATIST = WS / "idea/r5_offline/pragmatist/offline_k60.py"
THEORETICAL = WS / "idea/r5_offline/theoretical/offline_proj.py"
TARGETS = {"0.856": 0.856, "0.887": 0.887, "0.97": 0.97}
LOG_ERR_LIMIT = 0.095

from dsswm.baselines.b4_bal import B4Bal  # noqa: E402
from dsswm.baselines.combgame_joint import CombGameJoint  # noqa: E402
from dsswm.baselines.fdc import FACTORIAL_CELLS, FDCAblation, FDCMethod, fdc_ledger  # noqa: E402
from dsswm.baselines.frontier_common import QFCMethod  # noqa: E402
from dsswm.baselines.plugin_r5 import FixBalFav  # noqa: E402
from dsswm.streams import frontier as fr  # noqa: E402
from dsswm.streams.frontier_runner import build_ctx, run_stream, true_policy_values  # noqa: E402

_G = {}


def _load_contrarian():
    spec = importlib.util.spec_from_file_location("offline_ctr_r5", CONTRARIAN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha_file(p, n=16):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:n]


def init_env():
    from dsswm.envs.pool_replay import PoolReplayEnv
    env = PoolReplayEnv("CR9", "dev")
    probs = fr.cr_problems("visit")
    ctx = {K: build_ctx(env, probs, EPS, K=K) for K in (20, 60)}
    J = true_policy_values(ctx[20].pols, env.w, env.true_mu("visit"))
    Js = np.array([J[ctx[20].feas[q]].max() for q in range(ctx[20].Q)])
    _G.update(env=env, ctx=ctx, J=J, Js=Js, ctr=_load_contrarian())


def beta_dirP(ctx):
    return math.log(ctx.P ** 2 * len(ctx.checkpoints) / ctx.delta)


def b_r4(ctx):
    return fdc_ledger(ctx, "qstar", "r4")["beta"]


def b_fdc(ctx):
    return fdc_ledger(ctx, "feas", "r5")["beta"]


# ------------------------------------------------------------------------------------------------- configurations
# name -> (K, factory(ctx) -> method, meta). meta: role / step label / exploratory flag.
def _rb(spec):
    return lambda ctx: _G["ctr"].mk(spec)


def _cell(d, u, l):
    return lambda ctx: FDCAblation(d, u, l)


CONFIGS = {
    # references
    "B2-fav": (20, lambda ctx: CombGameJoint("fav"), {"role": "reference (plug-in, published config)"}),
    "B2-fav@K60": (60, lambda ctx: CombGameJoint("fav"), {"role": "reference at K=60"}),
    # (A)/(B) ladder, main path 0.856 -> FDC -> QFC-half K20 -> QFC-half K60 (0.887)
    "L0 FIX0.40-rig-b14.1 [0.856]": (20, _rb("FIX0.40-rig-b14.1"), {"role": "target 0.856; contrarian RB cert"}),
    "L1 FIX0.50-rig-b14.1": (20, _rb("FIX0.50-rig-b14.1"), {"role": "share 0.40 -> 0.50"}),
    "L2 FDCx beta=14.1": (20, lambda ctx: FDCAblation(beta_override=14.1),
                          {"role": "certificate: contrarian per-checkpoint EB var + max-cell linear term "
                                   "(x_v=ln(S A K/.005)) -> r4 qfc_certificate_enum + Lemma L2 inversion (x_v FDC)"}),
    "FDC": (20, lambda ctx: FDCMethod(), {"role": "beta 14.1 -> 14.2242 (FDC, rigorous)"}),
    "L4a FDCx beta=15.161": (20, lambda ctx: FDCAblation(beta_override=b_r4(ctx)),
                             {"role": "beta 14.2242 -> 15.161 (x_v kept at FDC 11.8776)"}),
    "L4b QFC-half@K20 = FDCx[half,qstar,r4]": (20, _cell("half", "qstar", "r4"),
                                              {"role": "x_v 11.8776 -> 12.1653 (r4 ledger complete)"}),
    "L5 QFC-half@K60 [0.887]": (60, lambda ctx: QFCMethod(name="QFC-half", alloc_p=np.full((ctx.S, ctx.A), 0.5)),
                                {"role": "K 20 -> 60 (r4 ledger at K=60); target 0.887 (pragmatist object)"}),
    # plug-in branch FDC -> 0.97
    "P1 FIX-bal-fav beta=14.2242": (20, lambda ctx: FixBalFav(share=0.5, beta=b_fdc(ctx), name="FIX-bal-fav-bFDC"),
                                    {"role": "rigorous (L1 width + L2 var UCB) -> plug-in var GLR at FDC beta"}),
    "P2 FIX-bal-fav (beta_tight 14.12)": (20, lambda ctx: FixBalFav(share=0.5),
                                          {"role": "beta 14.2242 -> 14.12 (registered FIX-bal-fav)"}),
    "P3 FIX0.50-fav b=18.47 [0.97]": (20, _rb("FIX0.50-fav"), {"role": "beta 14.12 -> ln(P^2 K/delta)=18.47; "
                                                                        "target 0.97 (contrarian RB fav)"}),
    # OFAT from FDC base
    "O-share FDCx share=0.40": (20, lambda ctx: FDCAblation(share=0.4), {"role": "OFAT: share 0.50 -> 0.40"}),
    "O-cert FIX0.50-rig (contrarian cert, beta=14.2242)": (20, _rb("FIX0.50-rig"),
                                                           {"role": "OFAT: certificate -> contrarian RB rig"}),
    "O-beta18 FDCx beta=18.47": (20, lambda ctx: FDCAblation(beta_override=beta_dirP(ctx)),
                                 {"role": "OFAT: beta 14.2242 -> 18.47 (rigorous cert)"}),
    "O-K60 FDC@K60": (60, lambda ctx: FDCMethod(), {"role": "OFAT: K 20 -> 60 (FDC ledger recomputed at K=60)"}),
    # C4 rival with the same design
    "B4-bal[share=0.50]": (20, lambda ctx: B4Bal(share=0.5), {"role": "C4: same design, rectangular certificate"}),
}
for _d, _u, _l in FACTORIAL_CELLS:
    _n = f"F[{_d},{_u},{_l}]"
    if (_d, _u, _l) == ("half", "feas", "r5"):
        continue  # == FDC (identity verified in r5_fdc_impl and below)
    if (_d, _u, _l) == ("half", "qstar", "r4"):
        continue  # == L4b
    CONFIGS[_n] = (20, _cell(_d, _u, _l), {"role": "C4 factorial cell"})
CELL_NAME = {c: f"F[{c[0]},{c[1]},{c[2]}]" for c in FACTORIAL_CELLS}
CELL_NAME[("half", "feas", "r5")] = "FDC"
CELL_NAME[("half", "qstar", "r4")] = "L4b QFC-half@K20 = FDCx[half,qstar,r4]"

RB_CONFIGS = {"L0 FIX0.40-rig-b14.1 [0.856]", "L1 FIX0.50-rig-b14.1", "P3 FIX0.50-fav b=18.47 [0.97]",
              "O-cert FIX0.50-rig (contrarian cert, beta=14.2242)"}
LADDER_MAIN = ["L0 FIX0.40-rig-b14.1 [0.856]", "L1 FIX0.50-rig-b14.1", "L2 FDCx beta=14.1", "FDC",
               "L4a FDCx beta=15.161", "L4b QFC-half@K20 = FDCx[half,qstar,r4]", "L5 QFC-half@K60 [0.887]"]
LADDER_PLUGIN = ["FDC", "P1 FIX-bal-fav beta=14.2242", "P2 FIX-bal-fav (beta_tight 14.12)",
                 "P3 FIX0.50-fav b=18.47 [0.97]"]


def ref_of(name):
    return "B2-fav@K60" if CONFIGS[name][0] == 60 else "B2-fav"


# ------------------------------------------------------------------------------------------------- stream jobs
def job(a):
    name, seed = a
    K, fac, _ = CONFIGS[name]
    ctx = _G["ctx"][K]
    m = fac(ctx)
    t0 = time.perf_counter()
    s, _ = run_stream(_G["env"], m, seed, ctx.problems, EPS, ctx=ctx, K=K, J_true=_G["J"], J_star=_G["Js"],
                      keep_U=False)
    validity = getattr(m, "validity", "none")
    if name in RB_CONFIGS:  # contrarian offline RB: its own 'rigorous' label is unproven -> exploratory
        validity = "none (exploratory: contrarian RB, unproven)"
    return {"config": name, "K": K, "method_name": m.name, "validity": validity,
            "seed": seed, "N80_pen": int(s["N80"]), "reached_stop": bool(s["reached_stop"]),
            "fwer_event": bool(s["fwer_event"]), "censored": bool(s["censored"]), "n_cert": s["n_cert"],
            "n_false": s["n_false"], "k_stop": s["k_stop"], "billing_ok": bool(s["billing_ok"]),
            "schedule_digest": s["schedule_digest"], "sec": round(time.perf_counter() - t0, 3)}


def identity_job(seed):
    """Code-path identities: registry objects vs the offline objects used for the three numbers."""
    from dsswm.streams import r5_registry as reg
    ctx = _G["ctx"][20]
    pairs = [("registry QFC-half", reg.make_method("QFC-half"), "FDCx[half,qstar,r4]", FDCAblation("half", "qstar", "r4")),
             ("FixBalFav(beta=18.47)", FixBalFav(share=0.5, beta=beta_dirP(ctx), name="x"),
              "RB FIX0.50-fav", _G["ctr"].mk("FIX0.50-fav")),
             ("FDCMethod", FDCMethod(), "FDCx[half,feas,r5]", FDCAblation("half", "feas", "r5"))]
    out = []
    for la, ma, lb, mb in pairs:
        sa, ra = run_stream(_G["env"], ma, seed, ctx.problems, EPS, ctx=ctx, J_true=_G["J"], J_star=_G["Js"],
                            keep_U=True)
        sb, rb = run_stream(_G["env"], mb, seed, ctx.problems, EPS, ctx=ctx, J_true=_G["J"], J_star=_G["Js"],
                            keep_U=True)
        same_cert = sa["cert_k"] == sb["cert_k"] and sa["N80"] == sb["N80"]
        maxdu = 0.0
        for x, y in zip(ra, rb):
            ux, uy = np.asarray(x["U"], float), np.asarray(y["U"], float)
            fin = np.isfinite(ux) & np.isfinite(uy)
            if fin.any():
                maxdu = max(maxdu, float(np.max(np.abs(ux[fin] - uy[fin]))))
            if not np.array_equal(np.isfinite(ux), np.isfinite(uy)):
                maxdu = float("inf")
        out.append({"seed": seed, "a": la, "b": lb, "same_cert_k_and_N80": bool(same_cert), "max_abs_dU": maxdu})
    return out


def progress(done, total, metric=None):
    (RES / f"{TASK}_PROGRESS.json").write_text(json.dumps({
        "task_id": TASK, "epoch": done, "total_epochs": total, "step": done, "total_steps": total, "loss": None,
        "metric": metric or {}, "updated_at": datetime.now().isoformat()}))


# ------------------------------------------------------------------------------------------------- statistics
def logs_by(rows, name, seeds):
    d = {r["seed"]: r["N80_pen"] for r in rows if r["config"] == name}
    return np.log(np.array([d[s] for s in seeds], float))


def boot_mean(x, B=B_BOOT, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(B, len(x)))
    return x[idx].mean(1)


def paired_ratio(rows, a, b, seeds, B=B_BOOT, seed=42):
    d = logs_by(rows, a, seeds) - logs_by(rows, b, seeds)
    bs = boot_mean(d, B, seed)
    return {"ratio": float(np.exp(d.mean())), "ci95": [float(np.exp(np.quantile(bs, .025))),
                                                      float(np.exp(np.quantile(bs, .975)))],
            "upper95_one_sided": float(np.exp(np.quantile(bs, .95))), "log_mean": float(d.mean()),
            "frac_faster": float((d < 0).mean()), "frac_tie": float((d == 0).mean()), "n": len(seeds)}


def ci_py_ratio(rows, a, b, seeds):
    """Exactly the bootstrap of idea/r5_offline/contrarian/ci.py (rng 1, 4000 draws, 100-sample resampling)."""
    lr = logs_by(rows, a, seeds) - logs_by(rows, b, seeds)
    rng = np.random.default_rng(1)
    bs = [lr[rng.integers(0, 100, 100)].mean() for _ in range(4000)]
    return {"ratio": float(np.exp(lr.mean())), "ci95": [float(x) for x in np.exp(np.quantile(bs, [.025, .975]))]}


def k60_py_ratio(rows, a, b, seeds):
    """Exactly the bootstrap of idea/r5_offline/pragmatist/offline_k60.py (rng 0, 2000 draws)."""
    lg = logs_by(rows, a, seeds) - logs_by(rows, b, seeds)
    rng = np.random.default_rng(0)
    bs = [np.exp(rng.choice(lg, len(lg)).mean()) for _ in range(2000)]
    return {"ratio": float(np.exp(lg.mean())), "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]}


# ------------------------------------------------------------------------------------------------- C4 model
def cell_params(cell, ctx):
    d, u, l = cell
    led = fdc_ledger(ctx, u, l)
    return {"design": d, "beta": led["beta"], "x_v": led["x_v"]}


def expected_counts(env, t, design):
    N = env.pool_sizes.astype(float)
    ns = env.w * t
    if design == "pool":
        n = ns[:, None] * N / N.sum(1, keepdims=True)
    else:
        n0 = np.minimum(0.5 * ns, N[:, 0])
        n1 = np.minimum(ns - n0, N[:, 1])
        n0 = np.minimum(ns - n1, N[:, 0])
        n = np.stack([n0, n1], 1)
    return n


def n_cert_det(env, ctx, t, prm, mu):
    from dsswm.certify.quadknap import make_stats, qfc_certificate_enum
    n = expected_counts(env, t, prm["design"])
    n = np.where(n >= env.pool_sizes - 1e-6, env.pool_sizes.astype(float), n)   # exhausted exactly -> exact cell
    st = make_stats(ctx.w, mu, n, N=env.pool_sizes.astype(float), x_v=prm["x_v"], binary=True, R=ctx.R)
    res = qfc_certificate_enum(st, ctx.problems, prm["beta"], ctx.eps, pols=ctx.pols)
    return sum(r["certified"] for r in res)


def t_det(env, ctx, prm, mu, need=12):
    lo, hi = 2e4, float(env.tau_R)
    if n_cert_det(env, ctx, hi, prm, mu) < need:
        return hi
    for _ in range(50):
        mid = math.sqrt(lo * hi)
        if n_cert_det(env, ctx, mid, prm, mu) >= need:
            hi = mid
        else:
            lo = mid
    return hi


def snap_up(t, ck, tau):
    i = np.searchsorted(ck, t, side="left")
    return float(ck[i]) if i < len(ck) else float(tau)


def fit_model(rows, env, ctx):
    """Calibrate on CAL_SEEDS only. Two pre-declared one-parameter variants; primary = lower calibration RMSE."""
    mu = env.true_mu("visit")
    ck = np.asarray(ctx.checkpoints, float)
    cells = {}
    for c in FACTORIAL_CELLS:
        prm = cell_params(c, ctx)
        td = t_det(env, ctx, prm, mu)
        obs = float(logs_by(rows, CELL_NAME[c], CAL_SEEDS).mean())
        cells[c] = {"params": prm, "t_det": td, "log_obs_cal": obs}
    # M1: log g = log t_det + c  (c = mean residual, least squares)
    c1 = float(np.mean([v["log_obs_cal"] - math.log(v["t_det"]) for v in cells.values()]))
    rm1 = float(np.sqrt(np.mean([(v["log_obs_cal"] - math.log(v["t_det"]) - c1) ** 2 for v in cells.values()])))
    # M2: log g = log snap_up(t_det * e^c) (c on a grid, least squares on the calibration half)
    grid = np.linspace(-0.6, 0.6, 1201)

    def m2err(c):
        return float(np.sqrt(np.mean([(v["log_obs_cal"] - math.log(snap_up(v["t_det"] * math.exp(c), ck, ctx.tau_R)))
                                      ** 2 for v in cells.values()])))
    errs = [m2err(c) for c in grid]
    c2 = float(grid[int(np.argmin(errs))])
    rm2 = float(min(errs))
    primary = "M1" if rm1 <= rm2 else "M2"
    pred = {}
    for c, v in cells.items():
        p1 = math.log(v["t_det"]) + c1
        p2 = math.log(snap_up(v["t_det"] * math.exp(c2), ck, ctx.tau_R))
        pred[CELL_NAME[c]] = {"cell": list(c), "beta": v["params"]["beta"], "x_v": v["params"]["x_v"],
                              "design": v["params"]["design"], "t_det": v["t_det"],
                              "t_det_over_tau": v["t_det"] / ctx.tau_R, "log_obs_cal": v["log_obs_cal"],
                              "pred_log_M1": p1, "pred_log_M2": p2,
                              "pred_geomean_primary": math.exp(p1 if primary == "M1" else p2)}
    return {"variants": {"M1": {"form": "log g = log t_det + c", "c": c1, "cal_rmse_log": rm1},
                         "M2": {"form": "log g = log snap_up_checkpoint(t_det * exp(c))", "c": c2, "cal_rmse_log": rm2,
                                "c_grid": "linspace(-0.6, 0.6, 1201)"}},
            "primary": primary, "primary_rule": "variant with lower RMSE of log error on calibration seeds 900-949 "
                                                "(declared before calibration)", "cells": pred}


def freeze_model(model, ctx):
    body = {"task_id": TASK, "frozen_at": datetime.now().isoformat(), "layer": "CR9", "half": "dev", "eps": EPS,
            "K": len(ctx.checkpoints), "calibration_seeds": [CAL_SEEDS[0], CAL_SEEDS[-1]],
            "validation_seeds": [VAL_SEEDS[0], VAL_SEEDS[-1]], "validation_streams_run_before_freeze": False,
            "source_model": "idea/r5_offline/theoretical/offline_proj.py (deterministic-equivalent N80: smallest t with "
                            ">= 12/15 problems certified at true cell means and expected design counts), with the "
                            "projection's sqrt(2 beta V) at true sigma^2 replaced by the actual certificate code "
                            "(quadknap.qfc_certificate_enum: Lemma L1 width incl. b beta/3, Lemma L2 variance UCB at "
                            "the cell's x_v evaluated at the true means) and each cell's actual beta / x_v",
            "inputs": "true cell means of the CR9 dev half (truth, dev only), pool sizes, design expected counts; "
                      "observed N80_pen of seeds 900-949 for the one calibration constant",
            "error_limit_abs_log": LOG_ERR_LIMIT, "model": model,
            "code_sha256_16": {"run_r5_t1_reconcile.py": sha_file(__file__),
                               "offline_proj.py": sha_file(THEORETICAL),
                               "dsswm/certify/quadknap.py": sha_file(CODE / "dsswm/certify/quadknap.py"),
                               "dsswm/baselines/fdc.py": sha_file(CODE / "dsswm/baselines/fdc.py")}}
    blob = json.dumps(body, sort_keys=True, indent=1).encode()
    body["sha256"] = hashlib.sha256(blob).hexdigest()
    body["sha256_scope"] = "sha256 of json.dumps(file minus the two sha256 fields, sort_keys=True, indent=1)"
    (GATES / "c4_model.json").write_text(json.dumps(body, indent=1, sort_keys=True))
    return body["sha256"]


# ------------------------------------------------------------------------------------------------- factorial
FACT_LEVEL = {"design": ("pool", "half"), "union": ("qstar", "feas"), "ledger": ("r4", "r5")}


def factorial_analysis(rows, seeds):
    """Per-stream log N80_pen for the 8 cells; base = (pool,qstar,r4) = QFC-pool, target = (half,feas,r5) = FDC."""
    L = {c: logs_by(rows, CELL_NAME[c], seeds) for c in FACTORIAL_CELLS}

    def stats_from(Lm):
        g = {c: Lm[c].mean() for c in FACTORIAL_CELLS}

        def v(sub):  # sub = set of factor indices switched to FDC level
            c = tuple(FACT_LEVEL[f][1 if i in sub else 0] for i, f in enumerate(("design", "union", "ledger")))
            return g[c]
        base = v(set())
        tot = v({0, 1, 2}) - base
        phi = []
        for i in range(3):
            others = [j for j in range(3) if j != i]
            s = 0.0
            for r in range(3):
                for sub in itertools.combinations(others, r):
                    wgt = math.factorial(r) * math.factorial(3 - r - 1) / math.factorial(3)
                    s += wgt * (v(set(sub) | {i}) - v(set(sub)))
            phi.append(s)
        # +-1 coded contrasts (effect = mean(high) - mean(low)), high = FDC level
        x = {c: [1 if c[i] == FACT_LEVEL[f][1] else -1 for i, f in enumerate(("design", "union", "ledger"))]
             for c in FACTORIAL_CELLS}
        eff = {}
        labels = {(0,): "D", (1,): "U", (2,): "L", (0, 1): "DxU", (0, 2): "DxL", (1, 2): "UxL", (0, 1, 2): "DxUxL"}
        for idx, lab in labels.items():
            eff[lab] = float(np.mean([g[c] * np.prod([x[c][i] for i in idx]) for c in FACTORIAL_CELLS]) * 2)
        # simple effects of D (switch design only) at the base and at FDC levels of U, L
        simple = {"D_at_base": v({0}) - base, "D_at_U,L=FDC": v({0, 1, 2}) - v({1, 2}),
                  "U_at_base": v({1}) - base, "L_at_base": v({2}) - base}
        return {"total_log": tot, "phi": phi, "effects": eff, "simple": simple, "g": g}
    pt = stats_from(L)
    rng = np.random.default_rng(42)
    n = len(seeds)
    bs_share, bs_phi = [], []
    for _ in range(2000):
        idx = rng.integers(0, n, n)
        st = stats_from({c: L[c][idx] for c in FACTORIAL_CELLS})
        bs_phi.append(st["phi"])
        bs_share.append(st["phi"][0] / st["total_log"] if st["total_log"] != 0 else np.nan)
    bs_phi = np.array(bs_phi)
    bs_share = np.array(bs_share)
    shares = [p / pt["total_log"] for p in pt["phi"]]
    return {"seeds": [seeds[0], seeds[-1]], "n": n,
            "cells": {f"{c[0]},{c[1]},{c[2]}": {"name": CELL_NAME[c], "geomean_N80_pen": float(np.exp(pt["g"][c])),
                                                "mean_log": float(pt["g"][c])} for c in FACTORIAL_CELLS},
            "total_log_speedup_FDC_vs_QFCpool": float(pt["total_log"]),
            "total_ratio_FDC_over_QFCpool": float(math.exp(pt["total_log"])),
            "shapley_log": {"design": pt["phi"][0], "union": pt["phi"][1], "ledger": pt["phi"][2]},
            "shapley_share": {"design": shares[0], "union": shares[1], "ledger": shares[2]},
            "shapley_ci95": {f: [float(np.quantile(bs_phi[:, i], .025)), float(np.quantile(bs_phi[:, i], .975))]
                             for i, f in enumerate(("design", "union", "ledger"))},
            "design_share_ci95": [float(np.nanquantile(bs_share, .025)), float(np.nanquantile(bs_share, .975))],
            "design_share_lower95_one_sided": float(np.nanquantile(bs_share, .05)),
            "effects_coded_log": pt["effects"], "simple_effects_log": {k: float(v) for k, v in pt["simple"].items()},
            "interaction_note": "Shapley distributes interactions equally among involved factors; coded effects "
                                "(high = FDC level) reported separately",
            "c4_dev_design_share_ge_2_3": bool(shares[0] >= 2 / 3)}


# ------------------------------------------------------------------------------------------------- main
def run_jobs(pool, names, seeds, rows, rfile, done0, total):
    jobs = [(n, s) for n in names for s in seeds]
    # put slow configs first for load balance
    jobs.sort(key=lambda a: (0 if (CONFIGS[a[0]][0] == 60 or "pool" in a[0]) else 1))
    done = done0
    for r in pool.imap_unordered(job, jobs, chunksize=1):
        rows.append(r)
        with open(rfile, "a") as f:
            f.write(json.dumps(r) + "\n")
        done += 1
        if done % 25 == 0:
            progress(done, total, {"runs_done": done})
            print(f"[{done}/{total}] {r['config']} seed {r['seed']} N80={r['N80_pen']} sec={r['sec']}", flush=True)
    return done


def geo_row(rows, name, seeds):
    rr = [r for r in rows if r["config"] == name and r["seed"] in set(seeds)]
    lg = np.log([r["N80_pen"] for r in rr])
    return {"config": name, "K": CONFIGS[name][0], "method_name": rr[0]["method_name"], "validity": rr[0]["validity"],
            "streams": len(rr), "geomean_N80_pen": float(np.exp(lg.mean())), "mean_log": float(lg.mean()),
            "sd_log": float(lg.std(ddof=1)), "median_N80_pen": float(np.median([r["N80_pen"] for r in rr])),
            "false_streams": int(sum(r["fwer_event"] for r in rr)), "censored": int(sum(r["censored"] for r in rr)),
            "billing_ok_all": all(r["billing_ok"] for r in rr), "sec_mean": float(np.mean([r["sec"] for r in rr]))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    GATES.mkdir(parents=True, exist_ok=True)
    (RES / f"{TASK}.pid").write_text(str(os.getpid()))
    t_start = datetime.now()
    (OUT / "start_time.txt").write_text(t_start.isoformat())
    init_env()
    env, ctx20, ctx60 = _G["env"], _G["ctx"][20], _G["ctx"][60]
    names = list(CONFIGS)
    total = len(names) * len(SEEDS) + len(IDENTITY_SEEDS)
    rows = []
    rfile = OUT / "results.jsonl"
    rfile.write_text("")
    progress(0, total)
    ledgers = {"FDC@K20": fdc_ledger(ctx20, "feas", "r5"), "r4@K20": fdc_ledger(ctx20, "qstar", "r4"),
               "FDC@K60": fdc_ledger(ctx60, "feas", "r5"), "r4@K60": fdc_ledger(ctx60, "qstar", "r4"),
               "beta_dir(P^2K/delta)@K20": beta_dirP(ctx20), "beta_dir@K60": beta_dirP(ctx60)}
    for v in ledgers.values():
        if isinstance(v, dict):
            v.pop("per_q_feasible", None)
    print("ledgers", json.dumps(ledgers, default=float), flush=True)
    with get_context("fork").Pool(4) as pool:
        ident = [x for lst in pool.map(identity_job, IDENTITY_SEEDS) for x in lst]
        print("identity:", ident, flush=True)
        # phase 1: calibration seeds only, then freeze the C4 model
        done = run_jobs(pool, names, CAL_SEEDS, rows, rfile, len(IDENTITY_SEEDS), total)
        model = fit_model(rows, env, ctx20)
        model_sha = freeze_model(model, ctx20)
        print("C4 model frozen", model_sha, model["primary"], {k: v["c"] for k, v in model["variants"].items()},
              flush=True)
        # phase 2: validation seeds
        run_jobs(pool, names, VAL_SEEDS, rows, rfile, done, total)
    # ------------------------------------------------------------------ (A) reproduction
    repro = {}
    r856 = ci_py_ratio(rows, "L0 FIX0.40-rig-b14.1 [0.856]", "B2-fav", SEEDS)
    r887 = k60_py_ratio(rows, "L5 QFC-half@K60 [0.887]", "B2-fav@K60", K60_REPRO_SEEDS)
    r97 = ci_py_ratio(rows, "P3 FIX0.50-fav b=18.47 [0.97]", "B2-fav", SEEDS)
    for key, r, how in (("0.856", r856, "contrarian ci.py bootstrap, seeds 900-999, K=20"),
                        ("0.887", r887, "pragmatist offline_k60.py bootstrap, seeds 900-929, K=60"),
                        ("0.97", r97, "contrarian ci.py bootstrap, seeds 900-999, K=20 (text-only original)")):
        rel = r["ratio"] / TARGETS[key] - 1
        repro[key] = {**r, "target": TARGETS[key], "rel_error": rel, "within_2pct": abs(rel) <= 0.02, "how": how}
    # extra: original raw stream records of run3 (0.856) - per-stream identity
    run3 = json.load(open(WS / "idea/r5_offline/contrarian/run3.json"))
    raw = {(x["method"], x["seed"]): x["N80"] for x in run3}
    mine = {(r["config"], r["seed"]): r["N80_pen"] for r in rows}
    id856 = sum(raw[("FIX0.40-rig-b14.1", s)] == mine[("L0 FIX0.40-rig-b14.1 [0.856]", s)] for s in SEEDS)
    idb2 = sum(raw[("B2-fav", s)] == mine[("B2-fav", s)] for s in SEEDS)
    repro["0.856"]["per_stream_identical_to_run3"] = f"{id856}/100"
    repro["0.856"]["B2fav_per_stream_identical_to_run3"] = f"{idb2}/100"
    repro["0.887"]["also_100_streams"] = paired_ratio(rows, "L5 QFC-half@K60 [0.887]", "B2-fav@K60", SEEDS)
    repro["0.887"]["K20_40streams_offline_qfc_alloc(0.8896)"] = k60_py_ratio(
        rows, "L4b QFC-half@K20 = FDCx[half,qstar,r4]", "B2-fav", list(range(900, 940)))
    # ------------------------------------------------------------------ (B) ladder + OFAT
    table = {n: geo_row(rows, n, SEEDS) for n in names}
    for n in names:
        if n.startswith("B2-fav"):
            continue
        table[n]["vs_B2fav_matchedK"] = paired_ratio(rows, n, ref_of(n), SEEDS)
    fdc_vs = table["FDC"]["vs_B2fav_matchedK"]["log_mean"]

    def ladder(seq):
        out = []
        prev = None
        for n in seq:
            lr = table[n]["vs_B2fav_matchedK"]["log_mean"]
            step = None
            if prev is not None:
                d = logs_by(rows, n, SEEDS) - logs_by(rows, prev, SEEDS)
                bs = boot_mean(d)
                step = {"dlog_geomean_N80": float(d.mean()), "ci95": [float(np.quantile(bs, .025)),
                                                                      float(np.quantile(bs, .975))],
                        "dlog_ratio_vs_B2fav": lr - table[prev]["vs_B2fav_matchedK"]["log_mean"]}
            out.append({"config": n, "role": CONFIGS[n][2]["role"], "validity": table[n]["validity"],
                        "K": CONFIGS[n][0], "geomean_N80_pen": table[n]["geomean_N80_pen"],
                        "ratio_vs_B2fav": table[n]["vs_B2fav_matchedK"]["ratio"],
                        "ci95": table[n]["vs_B2fav_matchedK"]["ci95"], "step": step})
            prev = n
        return out
    lad_main, lad_plug = ladder(LADDER_MAIN), ladder(LADDER_PLUGIN)
    ofat = {"share 0.50->0.40": "O-share FDCx share=0.40",
            "cert r4+L2 -> contrarian RB": "O-cert FIX0.50-rig (contrarian cert, beta=14.2242)",
            "beta 14.2242->14.1": "L2 FDCx beta=14.1",
            "beta 14.2242->15.161": "L4a FDCx beta=15.161",
            "ledger FDC -> r4 (union qstar + r4 deltas/C_var)": "L4b QFC-half@K20 = FDCx[half,qstar,r4]",
            "K 20->60": "O-K60 FDC@K60",
            "rigorous -> plug-in (beta FDC)": "P1 FIX-bal-fav beta=14.2242",
            "beta 14.2242->18.47 (rigorous)": "O-beta18 FDCx beta=18.47"}
    ofat_eff = {k: table[v]["vs_B2fav_matchedK"]["log_mean"] - fdc_vs for k, v in ofat.items()}
    targets = {"0.856": ("L0 FIX0.40-rig-b14.1 [0.856]",
                         ["share 0.50->0.40", "cert r4+L2 -> contrarian RB", "beta 14.2242->14.1"]),
               "0.887(K60,100str)": ("L5 QFC-half@K60 [0.887]",
                                     ["ledger FDC -> r4 (union qstar + r4 deltas/C_var)", "K 20->60"]),
               "0.97": ("P3 FIX0.50-fav b=18.47 [0.97]", ["rigorous -> plug-in (beta FDC)",
                                                         "beta 14.2242->18.47 (rigorous)"])}
    additivity = {}
    for k, (n, fs) in targets.items():
        joint = table[n]["vs_B2fav_matchedK"]["log_mean"] - fdc_vs
        s = sum(ofat_eff[f] for f in fs)
        additivity[k] = {"target_config": n, "joint_dlog_vs_FDC": joint, "ofat_factors": fs,
                         "ofat_dlog": {f: ofat_eff[f] for f in fs}, "sum_ofat": s, "residual_log": joint - s,
                         "residual_rel": math.exp(joint - s) - 1}
    # ------------------------------------------------------------------ (C) factorial + B4-bal
    fac = factorial_analysis(rows, SEEDS)
    fac["FDC_vs_B4bal"] = paired_ratio(rows, "FDC", "B4-bal[share=0.50]", SEEDS)
    fac["FDC_vs_B4bal"]["c4_dev_upper_lt_1"] = fac["FDC_vs_B4bal"]["upper95_one_sided"] < 1.0
    fac["FDC_vs_QFCpool"] = paired_ratio(rows, "FDC", CELL_NAME[("pool", "qstar", "r4")], SEEDS)
    fac["all_cells_rigorous"] = all(table[CELL_NAME[c]]["validity"] == "rigorous" for c in FACTORIAL_CELLS)
    fac["false_streams_per_cell"] = {CELL_NAME[c]: table[CELL_NAME[c]]["false_streams"] for c in FACTORIAL_CELLS}
    json.dump(fac, open(OUT / "shapley_dev.json", "w"), indent=1)
    with open(OUT / "factorial_dev.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["design", "union", "ledger", "config", "validity", "beta", "x_v", "streams", "geomean_N80_pen",
                    "mean_log", "sd_log", "ratio_vs_QFCpool", "ratio_vs_B2fav", "false_streams", "censored"])
        for c in FACTORIAL_CELLS:
            n = CELL_NAME[c]
            prm = cell_params(c, ctx20)
            rq = paired_ratio(rows, n, CELL_NAME[("pool", "qstar", "r4")], SEEDS)["ratio"]
            w.writerow([*c, n, table[n]["validity"], round(prm["beta"], 4), round(prm["x_v"], 4), table[n]["streams"],
                        round(table[n]["geomean_N80_pen"], 1), round(table[n]["mean_log"], 5),
                        round(table[n]["sd_log"], 5), round(rq, 4),
                        round(table[n]["vs_B2fav_matchedK"]["ratio"], 4), table[n]["false_streams"],
                        table[n]["censored"]])
    # ------------------------------------------------------------------ (D) model validation (950-999)
    frozen = json.loads((GATES / "c4_model.json").read_text())
    chk = dict(frozen)
    sha_stored = chk.pop("sha256")
    chk.pop("sha256_scope")
    sha_ok = hashlib.sha256(json.dumps(chk, sort_keys=True, indent=1).encode()).hexdigest() == sha_stored
    prim = frozen["model"]["primary"]
    val = {}
    for cname, cm in frozen["model"]["cells"].items():
        lv = logs_by(rows, cname, VAL_SEEDS)
        bs = boot_mean(lv)
        out = {}
        for var in ("M1", "M2"):
            p = cm[f"pred_log_{var}"]
            out[var] = {"log_error": float(lv.mean() - p), "ci95": [float(np.quantile(bs, .025) - p),
                                                                     float(np.quantile(bs, .975) - p)]}
        val[cname] = {"cell": cm["cell"], "obs_geomean_val": float(np.exp(lv.mean())),
                      "pred_geomean_primary": cm["pred_geomean_primary"], **out,
                      "primary_abs_log_error_le_0.095": abs(out[prim]["log_error"]) <= LOG_ERR_LIMIT}
    mval = {"primary": prim, "frozen_sha256": sha_stored, "sha_verified": sha_ok, "cells": val,
            "max_abs_log_error_primary": max(abs(v[prim]["log_error"]) for v in val.values()),
            "all_cells_within_0.095": all(v["primary_abs_log_error_le_0.095"] for v in val.values()),
            "consequence_if_fail": "C4 'mechanism predictable' wording withdrawn; descriptive decomposition only "
                                   "(methodology s6.3); C1 unaffected"}
    # ------------------------------------------------------------------ outputs
    write_table_md(repro, lad_main, lad_plug, additivity, ofat_eff, table, fac, mval, ledgers)
    t_end = datetime.now()
    ident_ok = all(x["same_cert_k_and_N80"] and x["max_abs_dU"] <= 1e-9 for x in ident)
    passed = {"three_numbers_within_2pct": all(v["within_2pct"] for v in repro.values()),
              "reconcile_table_complete": True,
              "factorial_8_cells_x_100": all(table[CELL_NAME[c]]["streams"] == 100 for c in FACTORIAL_CELLS),
              "c4_model_frozen_before_validation": bool(sha_ok), "code_path_identities": ident_ok,
              "billing_ok_all": all(t["billing_ok_all"] for t in table.values())}
    summary = {"task_id": TASK, "mode": "pilot (== full)", "started_at": t_start.isoformat(),
               "ended_at": t_end.isoformat(), "wall_min": round((t_end - t_start).total_seconds() / 60, 1),
               "timing_note": "并发运行（4 worker，与其他 r5 任务共享 20 核），计时偏高",
               "setting": {"layer": "CR9", "half": "dev", "eps": EPS, "seeds": [900, 999], "Q": ctx20.Q,
                           "stop_k": int(ctx20.stop_k), "tau_R": int(ctx20.tau_R), "K_main": 20},
               "ledgers": ledgers, "pass_criteria": passed, "go_no_go": "GO" if all(passed.values()) else "NO_GO",
               "reproduction": repro, "ladder_main": lad_main, "ladder_plugin": lad_plug, "ofat_dlog_vs_FDC": ofat_eff,
               "log_additivity": additivity, "configs": table, "factorial": fac, "c4_model_validation": mval,
               "identity_checks": ident, "exploratory_note": "all beta=14.1, beta-override, share!=0.5 and plug-in "
                                                             "configs are exploratory (validity none); FDC and the 8 "
                                                             "factorial cells and B4-bal are rigorous",
               "eval_seeds_touched": False,
               "code_sha256_16": {"run_r5_t1_reconcile.py": sha_file(__file__), "offline_ctr.py": sha_file(CONTRARIAN),
                                  "offline_k60.py": sha_file(PRAGMATIST), "offline_proj.py": sha_file(THEORETICAL),
                                  "dsswm/baselines/fdc.py": sha_file(CODE / "dsswm/baselines/fdc.py"),
                                  "dsswm/baselines/plugin_r5.py": sha_file(CODE / "dsswm/baselines/plugin_r5.py"),
                                  "dsswm/streams/frontier_runner.py":
                                      sha_file(CODE / "dsswm/streams/frontier_runner.py")}}
    json.dump(summary, open(OUT / "summary.json", "w"), indent=1, default=float)
    return summary


def write_table_md(repro, lad_main, lad_plug, additivity, ofat_eff, table, fac, mval, ledgers):
    L = ["# T1 reconciliation table (CR9 dev, seeds 900-999, eps* = 0.001; exploratory except rigorous rows)", ""]
    L.append("## (A) Reproduction of the three numbers")
    L.append("")
    L.append("| target | reproduced | 95% CI | rel. error | within +-2% | protocol |")
    L.append("|---|---|---|---|---|---|")
    for k, v in repro.items():
        L.append(f"| {k} | {v['ratio']:.4f} | [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}] | {v['rel_error']:+.2%} | "
                 f"{v['within_2pct']} | {v['how']} |")
    L.append("")
    L.append(f"0.856: per-stream N80 identical to original run3.json: {repro['0.856']['per_stream_identical_to_run3']}"
             f" (B2-fav {repro['0.856']['B2fav_per_stream_identical_to_run3']}). 0.887 on all 100 streams: "
             f"{repro['0.887']['also_100_streams']['ratio']:.4f}; QFC-half K=20 on 900-939: "
             f"{repro['0.887']['K20_40streams_offline_qfc_alloc(0.8896)']['ratio']:.4f} (original 0.8896).")
    L.append("")
    for title, lad in (("(B1) Main ladder: 0.856 -> FDC -> QFC-half K20 -> QFC-half K60 (0.887)", lad_main),
                       ("(B2) Plug-in branch: FDC -> 0.97", lad_plug)):
        L.append(f"## {title}")
        L.append("")
        L.append("| step | config | change | validity | K | geomean N80_pen | ratio vs B2-fav [95% CI] | "
                 "dlog N80 vs prev [95% CI] | dlog ratio |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for i, r in enumerate(lad):
            st = r["step"]
            s1 = "-" if st is None else f"{st['dlog_geomean_N80']:+.4f} [{st['ci95'][0]:+.4f}, {st['ci95'][1]:+.4f}]"
            s2 = "-" if st is None else f"{st['dlog_ratio_vs_B2fav']:+.4f}"
            L.append(f"| {i} | {r['config']} | {r['role']} | {r['validity']} | {r['K']} | {r['geomean_N80_pen']:,.0f} | "
                     f"{r['ratio_vs_B2fav']:.4f} [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}] | {s1} | {s2} |")
        L.append("")
    L.append("Ladder steps telescope exactly (sum of dlog ratio = endpoint difference); additivity is tested by OFAT.")
    L.append("")
    L.append("## (B3) One-factor-at-a-time effects from FDC (dlog ratio vs B2-fav, matched K) and log-additivity")
    L.append("")
    L.append("| factor | dlog |")
    L.append("|---|---|")
    for k, v in ofat_eff.items():
        L.append(f"| {k} | {v:+.4f} |")
    L.append("")
    L.append("| target | joint dlog vs FDC | sum OFAT | residual (log) | residual (rel) |")
    L.append("|---|---|---|---|---|")
    for k, v in additivity.items():
        L.append(f"| {k} | {v['joint_dlog_vs_FDC']:+.4f} | {v['sum_ofat']:+.4f} | {v['residual_log']:+.4f} | "
                 f"{v['residual_rel']:+.2%} |")
    L.append("")
    L.append("## (C) C4 dev factorial (all 8 cells rigorous) and FDC vs B4-bal")
    L.append("")
    L.append("| cell (design,union,ledger) | config | geomean N80_pen |")
    L.append("|---|---|---|")
    for k, v in fac["cells"].items():
        L.append(f"| {k} | {v['name']} | {v['geomean_N80_pen']:,.0f} |")
    L.append("")
    sh = fac["shapley_share"]
    L.append(f"Total log speed-up FDC vs QFC-pool: {fac['total_log_speedup_FDC_vs_QFCpool']:+.4f} (ratio "
             f"{fac['total_ratio_FDC_over_QFCpool']:.4f}). Shapley shares: design {sh['design']:.3f} (95% CI "
             f"[{fac['design_share_ci95'][0]:.3f}, {fac['design_share_ci95'][1]:.3f}]), union {sh['union']:.3f}, "
             f"ledger {sh['ledger']:.3f}. Coded effects: "
             + ", ".join(f"{k} {v:+.4f}" for k, v in fac["effects_coded_log"].items()) + ".")
    L.append("")
    b = fac["FDC_vs_B4bal"]
    L.append(f"FDC / B4-bal[share=0.50]: {b['ratio']:.4f} [95% CI {b['ci95'][0]:.3f}, {b['ci95'][1]:.3f}], one-sided "
             f"upper {b['upper95_one_sided']:.4f}, faster share {b['frac_faster']:.2f}.")
    L.append("")
    L.append(f"## (D) C4 prediction model (frozen sha256 {mval['frozen_sha256'][:16]}..., primary {mval['primary']}) "
             "on validation seeds 950-999")
    L.append("")
    L.append("| cell | obs geomean | pred (primary) | log err M1 [95% CI] | log err M2 [95% CI] | primary within 0.095 |")
    L.append("|---|---|---|---|---|---|")
    for k, v in mval["cells"].items():
        L.append(f"| {k} | {v['obs_geomean_val']:,.0f} | {v['pred_geomean_primary']:,.0f} | "
                 f"{v['M1']['log_error']:+.4f} [{v['M1']['ci95'][0]:+.3f}, {v['M1']['ci95'][1]:+.3f}] | "
                 f"{v['M2']['log_error']:+.4f} [{v['M2']['ci95'][0]:+.3f}, {v['M2']['ci95'][1]:+.3f}] | "
                 f"{v['primary_abs_log_error_le_0.095']} |")
    L.append("")
    L.append(f"Max |log error| (primary): {mval['max_abs_log_error_primary']:.4f}; all within 0.095: "
             f"{mval['all_cells_within_0.095']}.")
    L.append("")
    L.append("## Ledgers")
    L.append("")
    for k, v in ledgers.items():
        if isinstance(v, dict):
            L.append(f"- {k}: beta = {v['beta']:.4f}, x_v = {v['x_v']:.4f}, union = {v['union_size']}, "
                     f"C_var = {v['C_var']}, deltas = ({v['delta_main']}, {v['delta_var']})")
        else:
            L.append(f"- {k}: {v:.4f}")
    (OUT / "reconcile_table.md").write_text("\n".join(L) + "\n")


def mark_done(status, summary_txt):
    pid = RES / f"{TASK}.pid"
    if pid.exists():
        pid.unlink()
    pf = RES / f"{TASK}_PROGRESS.json"
    fp = {}
    if pf.exists():
        try:
            fp = json.loads(pf.read_text())
        except ValueError:
            pass
    (RES / f"{TASK}_DONE").write_text(json.dumps({"task_id": TASK, "status": status, "summary": summary_txt,
                                                  "final_progress": fp, "timestamp": datetime.now().isoformat()}))


if __name__ == "__main__":
    try:
        s = main()
        rp = "; ".join(f"{k}->{v['ratio']:.4f}" for k, v in s["reproduction"].items())
        mark_done("success" if s["go_no_go"] == "GO" else "failed", f"{s['go_no_go']} | {rp}")
        print(json.dumps({k: s[k] for k in ("go_no_go", "pass_criteria", "wall_min")}, indent=1))
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        mark_done("failed", f"exception: {e!r}")
        raise
