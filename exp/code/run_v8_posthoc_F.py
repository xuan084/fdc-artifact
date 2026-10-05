"""v8-posthoc-F: POST-HOC, DESCRIPTIVE, SEMI-SYNTHETIC block (outside any lock) for critic review_r3 P1-3: test
adaptivity where it SHOULD help -- cells whose two arms have very unequal Bernoulli variances, so the within-segment
Neyman split is far from 50/50.

New file; locked v5-v8 files are imported, never modified.  No evaluation half is touched: the structure (segment
labels, arm labels -> finite pool sizes, segment weights, frozen arrival schedules) is the DEV half of X5 X9 and of
Criteo CR9; ONLY the outcome column is replaced by synthetic Bernoulli pools.

Semi-synthetic environments (labelled SEMI-SYNTHETIC everywhere):
  regime 'asym'   every segment: control rate c_s ~ U(0.02, 0.05), treatment rate t_s ~ U(0.40, 0.50)
                  (Neyman control share ~ 0.25-0.30 in every segment; one global share captures it).
  regime 'mixed'  segments cycle through three types (s mod 3):
                    A  c ~ U(0.02, 0.05), t ~ U(0.40, 0.50)   Neyman control share ~ 0.27  (treatment noisy)
                    B  c ~ U(0.45, 0.55), t ~ U(0.93, 0.97)   Neyman control share ~ 0.70  (control noisy)
                    C  c ~ U(0.25, 0.30), t ~ U(0.65, 0.75)   Neyman control share ~ 0.50  (symmetric)
                  uplifts overlap (0.35-0.52) across types, so every segment matters for the budget frontier; no
                  single global share is variance-optimal.
  Rates are drawn once per (layer, regime) from a fixed seed.  Each finite pool gets EXACTLY round(rate * N_c) ones at
  seed-fixed rows (finite-population truth known exactly); arrivals and within-pool orders are the frozen schedules
  of the stream engine (labels only).

eps per environment (fixed before any tuning / evaluation row): from a grid, the eps at which frozen-50/50 FDC-BF's
geomean N80/tau on calibration seeds 900-919 is closest (in log) to 0.25 (the real-data FDC-BF level); FDC-BF only.

Methods (paired comparator: FDC-BF frozen 50/50)
  frozen-design, guarantee by Theorem 1 (plan fixed before outcomes):
    FDC-BF                 frozen 50/50
    FDC-BF[ney-pilot]      frozen per-segment Neyman from an independent PILOT sample (1000 Bernoulli draws per cell
                           from the same rates, separate seed; Laplace plug-in; floor 0.1) -- the 'dev pilot'
    FDC-BF[ney-oracle]     frozen Neyman from the true pool variances (reference; still valid: depends on pool contents
                           only, not on the within-pool permutation)
    RECT-ck-HG, RECT-ck-HG[ney-pilot], RECT-ck-BF, RECT-ck-BF[ney-pilot], HC-WoR, HC-WoR[ney-pilot]
                           (HC-WoR: lock-v8 schedule / c / target_frac of the layer; plan 0.5 or the pilot Neyman)
  adaptive, same guarantee class as FDC-BF:
    PJC-local*, PJC-menu*  lock-v8 frozen configs of the layer (not re-tuned)
    PJC-A grid (64 members, run_v8_posthoc_D.GRID): tuned on THIS env's tuning seeds 900-949 by the v5 rule per family
                           (reset / menu / segmenu / any); picks evaluated.
Reporting seeds 950-999 and 36000-36199 (250 streams per env).  Ratios FDC-BF / method paired geomean (N80_pen and
x12), bootstrap B = 10^4 seed 42 (v6 rule).

Tasks (cwd = exp/code):  --task calib | tune | select | eval | analyse   [--env X9-asym ...]
Outputs: exp/results/full/v8_posthoc_F/{calib,tune,eval}/<env>.jsonl, envs.json, selection.json, summary.{json,md},
table.tex
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
RES = WS / "exp/results"
OUT = RES / "full/v8_posthoc_F"
N_WORKERS = int(os.environ.get("POSTHOC_WORKERS", "8"))
TASK_ID = "v8-posthoc-F"

from dsswm.baselines.fdc_bet import make_variant  # noqa: E402
from dsswm.baselines.pjc_adapt import PJCAdapt  # noqa: E402
from dsswm.baselines.pjc_bf import FDCBetPlan, RectCkBF, neyman_matrix  # noqa: E402
from dsswm.streams import frontier as fr  # noqa: E402
from dsswm.streams.frontier_runner import build_ctx, true_policy_values  # noqa: E402
from dsswm.streams.frontier_runner_v6 import ctx_with_stop, run_stream_v6  # noqa: E402
from run_r5s_v7 import x_rank  # noqa: E402
from run_v8_posthoc_D import FAMILIES, GRID  # noqa: E402

STOP_K = 15
ENVS = ["X9-asym", "X9-mixed", "CR9-asym", "CR9-mixed"]
EPS_GRID = {"X9": [0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.075],
            "CR9": [0.0005, 0.00075, 0.001, 0.0015, 0.002, 0.003, 0.005]}
CALIB_SEEDS = list(range(900, 920))
TUNE_SEEDS = list(range(900, 950))
EVAL_SEEDS = list(range(950, 1000)) + list(range(36000, 36200))
TARGET_N80 = 0.25
PILOT_N = 1000
P_FLOOR = 0.1
TYPES = {"A": ((0.02, 0.05), (0.40, 0.50)), "B": ((0.45, 0.55), (0.93, 0.97)), "C": ((0.25, 0.30), (0.65, 0.75))}
FROZEN = ["FDC-BF", "FDC-BF[ney-pilot]", "FDC-BF[ney-oracle]", "RECT-ck-HG", "RECT-ck-HG[ney-pilot]", "RECT-ck-BF",
          "RECT-ck-BF[ney-pilot]", "HC-WoR", "HC-WoR[ney-pilot]", "PJC-local*", "PJC-menu*"]
_ENV: dict = {}


# =============================================================================================== environments
def seg_types(regime, S):
    return ["A"] * S if regime == "asym" else [("A", "B", "C")[s % 3] for s in range(S)]


def build_env(name):
    layer, regime = name.split("-")
    if layer == "X9":
        from dsswm.envs.x5_v8 import X5LayerEnv
        base = X5LayerEnv("dev")
    else:
        from dsswm.envs.pool_replay import PoolReplayEnv
        base = PoolReplayEnv("CR9", "dev")
    assert base.half == "dev"
    S, A = base.S, base.A
    code = {"X9": 9, "CR9": 39}[layer] * 10 + {"asym": 1, "mixed": 2}[regime]
    rng = np.random.default_rng([2026, 10, 4, code])
    types = seg_types(regime, S)
    rates = np.zeros((S, A))
    for s, ty in enumerate(types):
        (c0, c1), (t0, t1) = TYPES[ty]
        rates[s] = [rng.uniform(c0, c1), rng.uniform(t0, t1)]
    y = np.zeros(base.N)
    for c in range(S * A):
        rows = base.pool_rows[c]
        k = int(round(rates.ravel()[c] * len(rows)))
        y[rng.choice(rows, size=k, replace=False)] = 1.0
    y.setflags(write=False)
    env = object.__new__(type(base))
    env.__dict__.update(base.__dict__)
    env._y = {"visit": y}
    env.layer = f"{layer}-semisynth-{regime}"
    # independent pilot ('dev pilot' of the synthetic world): PILOT_N Bernoulli draws per cell, separate seed
    prng = np.random.default_rng([2026, 10, 4, code, 7])
    ps = prng.binomial(PILOT_N, rates)
    mh = (ps + 1.0) / (PILOT_N + 2.0)
    ney_pilot = neyman_matrix(mh * (1 - mh), floor=P_FLOOR)
    ney_oracle = neyman_matrix(env.true_sigma2("visit"), floor=P_FLOOR)
    spec = {"layer": layer, "regime": regime, "segment_types": types, "rates_drawn": rates.tolist(),
            "true_mu": env.true_mu("visit").tolist(), "pool_sizes": env.pool_sizes.tolist(), "w": env.w.tolist(),
            "tau_R": int(env.tau_R), "pilot_n_per_cell": PILOT_N, "pilot_successes": ps.tolist(),
            "ney_pilot_ctrl_share": ney_pilot[:, 0].tolist(), "ney_oracle_ctrl_share": ney_oracle[:, 0].tolist()}
    return env, ney_pilot, ney_oracle, spec


def load_cfg(layer):
    key = "x9" if layer == "X9" else "cr"
    return json.loads((RES / "v8_gates" / f"{key}_configs.json").read_text())["selected"]


def make(name):
    import run_r5s_v8 as RV8
    from dsswm.baselines.pjc_bf import HCWoRPlan, RectCkHGPlan
    from dsswm.baselines.rect_v6 import RectCkHG
    from dsswm.baselines.wor_betting_v6 import HCWoRRect
    npil, nora, cfg = _ENV["ney_pilot"], _ENV["ney_oracle"], _ENV["cfg"]
    if name == "FDC-BF":
        return make_variant("FDC-BF")
    if name == "FDC-BF[ney-pilot]":
        return FDCBetPlan(alloc=npil, name=name)
    if name == "FDC-BF[ney-oracle]":
        return FDCBetPlan(alloc=nora, name=name)
    if name == "FDC-BF[0.5-plan]":                                   # integrity: must equal FDC-BF
        return FDCBetPlan(share=0.5, name=name)
    if name == "RECT-ck-HG":
        return RectCkHG()
    if name == "RECT-ck-HG[ney-pilot]":
        return RectCkHGPlan(alloc=npil, name=name)
    if name == "RECT-ck-BF":
        return RectCkBF(box=False)
    if name == "RECT-ck-BF[ney-pilot]":
        m = RectCkBF(box=False, name=name)
        base_setup = m.setup

        def setup(ctx, _b=base_setup, _p=npil):
            _b(ctx)
            m.alloc_p = _p / _p.sum(1, keepdims=True)
        m.setup = setup
        return m
    if name in ("HC-WoR", "HC-WoR[ney-pilot]"):
        k = cfg["HC-WoR"]
        sch, c, tf = k["schedule"], float(k["c"]), k.get("target_frac")
        if name == "HC-WoR":
            return HCWoRRect(sch, c, tf)
        return HCWoRPlan(sch, c, tf, alloc=npil, name=name)
    if name in ("PJC-local*", "PJC-menu*"):
        base = name[:-1]
        m = RV8.make(base, RV8.method_kw(base, cfg))
        m.name = name
        return m
    if name in GRID:
        return PJCAdapt(**GRID[name], name=name)
    raise ValueError(name)


def init_env(name, eps):
    env, npil, nora, spec = build_env(name)
    probs = fr.cr_problems("visit")
    ctx = ctx_with_stop(build_ctx(env, probs, eps), STOP_K)
    J = true_policy_values(ctx.pols, env.w, env.true_mu("visit"))
    Js = np.array([J[ctx.feas[q]].max() for q in range(ctx.Q)])
    _ENV.update(env=env, ctx=ctx, J=J, Js=Js, name=name, ney_pilot=npil, ney_oracle=nora, spec=spec,
                cfg=load_cfg(name.split("-")[0]), eps=eps)


def job(a):
    name, seed = a
    ctx = _ENV["ctx"]
    try:
        m = make(name)
        t0 = time.perf_counter()
        s, rows = run_stream_v6(_ENV["env"], m, seed, ctx.problems, ctx.eps, outcome="visit", ctx=ctx,
                                J_true=_ENV["J"], J_star=_ENV["Js"], keep_U=True)
        ck = ctx.checkpoints
        tau = int(ctx.tau_R)
        curve = [r["n_cert"] for r in rows]
        nfalse = [r["n_false"] for r in rows]
        k12 = next((k for k, c in enumerate(curve) if c >= 12), None)
        n80 = int(ck[k12]) if (k12 is not None and nfalse[k12] == 0) else tau
        xx = x_rank([r["U"] for r in rows], ck, ctx.eps, 12)
        out = {"env": _ENV["name"], "method": name, "seed": int(seed), "eps": float(ctx.eps), "posthoc": True,
               "semisynthetic": True, "block": TASK_ID, "N80_pen": n80, "k80": k12,
               "x12": float(xx) if xx is not None else float(tau), "fwer_event": bool(s["fwer_event"]),
               "n_false": int(s["n_false"]), "n_cert": int(s["n_cert"]), "cert_k": s["cert_k"],
               "billing_ok": bool(s["billing_ok"]), "reselected": int(s["reselected"]),
               "schedule_digest": s["schedule_digest"], "tau_R": tau, "sec": round(time.perf_counter() - t0, 3),
               "error": None}
        if isinstance(m, PJCAdapt):
            p0 = m.plans[0]
            out.update({"switched": bool(m.switched),
                        "switched_before_k80": bool(any(float(np.max(np.abs(p - p0))) > 1e-9
                                                        for b, p in zip(m.boundaries, m.plans[1:])
                                                        if k12 is None or b < k12)),
                        "max_abs_share_dev": float(max([float(np.max(np.abs(p[:, 0] - 0.5))) for p in m.plans]
                                                       or [0.0])),
                        "ctrl_share_plans": [[round(float(x), 4) for x in p[:, 0]] for p in m.plans[1:]],
                        "beta": float(m.ledger["beta"]), "log_L": float(m.ledger.get("log_L", 0.0))})
        return out
    except Exception:  # noqa: BLE001
        return {"env": _ENV["name"], "method": name, "seed": int(seed), "error": traceback.format_exc()}


def calib_job(a):
    eps, seed = a
    env, ctx0 = _ENV["env"], _ENV["ctxs"][eps]
    try:
        m = make_variant("FDC-BF")
        s, rows = run_stream_v6(env, m, seed, ctx0.problems, eps, outcome="visit", ctx=ctx0, J_true=_ENV["Jc"][eps],
                                J_star=_ENV["Jsc"][eps], keep_U=False)
        ck, tau = ctx0.checkpoints, int(ctx0.tau_R)
        curve = [r["n_cert"] for r in rows]
        nfalse = [r["n_false"] for r in rows]
        k12 = next((k for k, c in enumerate(curve) if c >= 12), None)
        n80 = int(ck[k12]) if (k12 is not None and nfalse[k12] == 0) else tau
        return {"env": _ENV["name"], "eps": eps, "seed": seed, "N80_pen": n80, "tau_R": tau,
                "reached": k12 is not None, "fwer_event": bool(s["fwer_event"]), "error": None}
    except Exception:  # noqa: BLE001
        return {"env": _ENV["name"], "eps": eps, "seed": seed, "error": traceback.format_exc()}


def pool_map(fn, jobs):
    with get_context("fork").Pool(N_WORKERS) as pool:
        for r in pool.imap_unordered(fn, jobs, chunksize=1):
            yield r


def geo(v):
    return float(np.exp(np.mean(np.log(np.asarray(v, dtype=float)))))


def jl_load(f, keyf):
    key = {}
    if f.exists():
        for line in f.read_text().splitlines():
            r = json.loads(line)
            key[keyf(r)] = r
    return key


def progress(stage, n, tot):
    (RES / f"v8_posthoc_F_{stage}_PROGRESS.json").write_text(json.dumps(
        {"task_id": f"v8_posthoc_F_{stage}", "step": n, "total_steps": tot, "updated_at": datetime.now().isoformat()}))


# =============================================================================================== tasks
def run_calib(envs):
    d = OUT / "calib"
    d.mkdir(parents=True, exist_ok=True)
    envspec = json.loads((OUT / "envs.json").read_text()) if (OUT / "envs.json").exists() else {}
    for name in envs:
        f = d / f"{name}.jsonl"
        done = jl_load(f, lambda r: (r["eps"], r["seed"]))
        layer = name.split("-")[0]
        env, npil, nora, spec = build_env(name)
        probs = fr.cr_problems("visit")
        ctxs = {e: ctx_with_stop(build_ctx(env, probs, e), STOP_K) for e in EPS_GRID[layer]}
        Jc = {e: true_policy_values(ctxs[e].pols, env.w, env.true_mu("visit")) for e in ctxs}
        Jsc = {e: np.array([Jc[e][ctxs[e].feas[q]].max() for q in range(ctxs[e].Q)]) for e in ctxs}
        _ENV.update(env=env, ctxs=ctxs, Jc=Jc, Jsc=Jsc, name=name)
        jobs = [(e, s) for e in EPS_GRID[layer] for s in CALIB_SEEDS if (e, s) not in done]
        print(f"[calib {name}] {len(jobs)} jobs", flush=True)
        with open(f, "a") as g:
            for r in pool_map(calib_job, jobs):
                if r.get("error"):
                    print(r["error"], flush=True)
                    continue
                g.write(json.dumps(r) + "\n")
                g.flush()
        R = jl_load(f, lambda r: (r["eps"], r["seed"]))
        tab = {}
        for e in EPS_GRID[layer]:
            rr = [R[(e, s)] for s in CALIB_SEEDS]
            tab[str(e)] = {"geomean_N80_over_tau": geo([r["N80_pen"] / r["tau_R"] for r in rr]),
                           "frac_reached": float(np.mean([r["reached"] for r in rr])),
                           "false_streams": int(sum(r["fwer_event"] for r in rr))}
        pick = min(EPS_GRID[layer], key=lambda e: abs(np.log(tab[str(e)]["geomean_N80_over_tau"] / TARGET_N80)))
        spec.update({"eps": pick, "eps_rule": f"eps in {EPS_GRID[layer]} with FDC-BF (frozen 50/50) geomean N80/tau "
                     f"on seeds 900-919 closest in log to {TARGET_N80}", "calibration": tab})
        envspec[name] = spec
        print(f"[calib {name}] " + ", ".join(f"{e}:{v['geomean_N80_over_tau']:.3f}" for e, v in tab.items())
              + f" -> eps {pick}", flush=True)
        (OUT / "envs.json").write_text(json.dumps(envspec, indent=1))


def run_rows(stage, envs, names_fn, seeds):
    d = OUT / stage
    d.mkdir(parents=True, exist_ok=True)
    spec = json.loads((OUT / "envs.json").read_text())
    for name in envs:
        f = d / f"{name}.jsonl"
        done = set(jl_load(f, lambda r: (r["method"], r["seed"])))
        names = names_fn(name)
        jobs = [(n, s) for n in names for s in seeds if (n, s) not in done]
        jobs.sort(key=lambda j: not j[0].startswith("HC-WoR"))           # heavy first
        print(f"[{stage} {name}] {len(jobs)} jobs, eps {spec[name]['eps']}", flush=True)
        if not jobs:
            continue
        init_env(name, spec[name]["eps"])
        n = errs = 0
        t0 = time.time()
        with open(f, "a") as g:
            for r in pool_map(job, jobs):
                if r.get("error"):
                    errs += 1
                    with open(d / "errors.log", "a") as h:
                        h.write(f"{name} {r['method']} {r['seed']}\n{r['error']}\n")
                    continue
                g.write(json.dumps(r) + "\n")
                g.flush()
                n += 1
                if n % 100 == 0:
                    print(f"[{stage} {name}] {n}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
                    progress(f"{stage}_{name}", n, len(jobs))
        print(f"[{stage} {name}] done, {errs} errors, {time.time() - t0:.0f}s", flush=True)


def describe(rows, seeds):
    rr = [rows[s] for s in seeds]
    tau = rr[0]["tau_R"]
    d = {"geomean_N80_over_tau": geo([r["N80_pen"] / tau for r in rr]),
         "geomean_x12_over_tau": geo([r["x12"] / tau for r in rr]),
         "false_streams": int(sum(r["fwer_event"] for r in rr)), "billing_ok_all": all(r["billing_ok"] for r in rr),
         "frac_censored": float(np.mean([r["N80_pen"] >= tau for r in rr]))}
    if "switched" in rr[0]:
        d["frac_switched"] = float(np.mean([r["switched"] for r in rr]))
        d["frac_switched_before_k80"] = float(np.mean([r["switched_before_k80"] for r in rr]))
        d["mean_max_abs_share_dev"] = float(np.mean([r["max_abs_share_dev"] for r in rr]))
        sh = [np.array(p) for r in rr for p in r["ctrl_share_plans"]]
        d["mean_ctrl_share_after_switch_by_seg"] = (np.mean(sh, 0).round(3).tolist() if sh else None)
    return d


def by_method(R):
    by = {}
    for (m, s), r in R.items():
        by.setdefault(m, {})[s] = r
    return by


def select():
    out = {"rule": "v5 rule per family: argmin geomean N80_pen on the env's tuning seeds 900-949 s.t. 0 false streams; "
                   "tie-break geomean x12", "envs": {}}
    for name in ENVS:
        f = OUT / "tune" / f"{name}.jsonl"
        if not f.exists():
            continue
        by = by_method(jl_load(f, lambda r: (r["method"], r["seed"])))
        e = {"FDC-BF_tune": describe(by["FDC-BF"], TUNE_SEEDS), "families": {}}
        for fam, names in list(FAMILIES.items()) + [("any_adaptive", list(GRID))]:
            cand = {m: describe(by[m], TUNE_SEEDS) for m in names if m in by and set(by[m]) >= set(TUNE_SEEDS)}
            ok = {m: v for m, v in cand.items() if v["false_streams"] == 0}
            pick = min(ok, key=lambda m: (ok[m]["geomean_N80_over_tau"], ok[m]["geomean_x12_over_tau"])) if ok else None
            e["families"][fam] = {"pick": pick, "n_candidates": len(cand), "missing": len(names) - len(cand)}
        # full tuning-grid table, paired vs FDC-BF on the tuning seeds (descriptive, dev-selected by construction)
        from dsswm.stats.v6_analysis import boot_idx, paired
        idx = boot_idx(len(TUNE_SEEDS))
        a = np.array([by["FDC-BF"][s]["N80_pen"] for s in TUNE_SEEDS], float)
        grid = {}
        for m in GRID:
            if m in by and set(by[m]) >= set(TUNE_SEEDS):
                b = np.array([by[m][s]["N80_pen"] for s in TUNE_SEEDS], float)
                p = paired(a, b, idx)
                grid[m] = {"FDCBF_over_variant": p["geomean_ratio"], "ci95": p["ci95_two_sided"],
                           **describe(by[m], TUNE_SEEDS)}
        e["tune_grid"] = grid
        out["envs"][name] = e
        print(f"== {name}: FDC-BF tune N80/tau {e['FDC-BF_tune']['geomean_N80_over_tau']:.4f}")
        for fam, x in e["families"].items():
            v = grid.get(x["pick"], {})
            print(f"   {fam:13s} {x['pick']}  ratio {v.get('FDCBF_over_variant', float('nan')):.4f} "
                  f"sw {v.get('frac_switched', float('nan')):.2f}")
    (OUT / "selection.json").write_text(json.dumps(out, indent=1))
    return out


def eval_names(name):
    sel = json.loads((OUT / "selection.json").read_text())["envs"][name]["families"]
    picks = []
    for fam in ("reset", "menu", "segmenu", "any_adaptive"):
        p = sel[fam]["pick"]
        if p and p not in picks:
            picks.append(p)
    return FROZEN + ["FDC-BF[0.5-plan]"] + picks


# =============================================================================================== analysis
def analyse():
    from dsswm.stats.v6_analysis import boot_idx, paired
    spec = json.loads((OUT / "envs.json").read_text())
    sel = json.loads((OUT / "selection.json").read_text())
    idx = boot_idx(len(EVAL_SEEDS))
    summ = {"block": TASK_ID,
            "status": "POST-HOC, DESCRIPTIVE, SEMI-SYNTHETIC -- outside any preregistration lock (v5-v8); not "
                      "confirmatory; outcomes are synthetic",
            "disclosure": (
                "Run on 2026-10-04 after the v8 confirmatory analysis, in response to critic review_r3 P1-3 ('test "
                "adaptivity where it should help').  Only the DEV halves' structure (segment and arm labels, pool sizes, "
                "weights, frozen arrival schedules) is reused; the outcome column is replaced by synthetic Bernoulli "
                "pools with strongly unequal arm variances, so no evaluation row and no real outcome enters any "
                "statistic.  Rates, regimes, eps rule, method list, PJC-A grid and the v5 tuning rule were fixed before "
                "any row was computed; eps was then calibrated with frozen-50/50 FDC-BF only (seeds 900-919); PJC-A "
                "picks were tuned on seeds 900-949 of each semi-synthetic env; reporting seeds 950-999 + "
                "36000-36199 were used for nothing else.  Rival configs (RECT-ck-HG, HC-WoR schedule, PJC-local*, "
                "PJC-menu*) are the frozen lock-v8 configs of the layer, not re-tuned.  The regimes were chosen to "
                "favour adaptive allocation; nothing here changes a locked verdict."),
            "ratio_convention": "FDC-BF (frozen 50/50) / method, paired geomean (< 1: FDC-BF faster; > 1: method "
                                "faster); reported for N80_pen (checkpoint-quantised, primary) and x12 (interpolated)",
            "bootstrap": "v6 rule: B = 10^4 resamples of streams, seed 42, two-sided percentile 95% CI on the geomean",
            "seeds": {"calib": [900, 919], "tune": [900, 949], "eval": "950-999 + 36000-36199 (250 streams)"},
            "envs": {}}
    for name in ENVS:
        f = OUT / "eval" / f"{name}.jsonl"
        if not f.exists():
            continue
        by = by_method(jl_load(f, lambda r: (r["method"], r["seed"])))
        names = eval_names(name)
        miss = [m for m in names if m not in by or set(by[m]) < set(EVAL_SEEDS)]
        if miss:
            print(f"{name}: incomplete {miss}")
            continue
        ref = by["FDC-BF"]
        integ = by["FDC-BF[0.5-plan]"]
        integrity = int(sum(integ[s]["N80_pen"] != ref[s]["N80_pen"] or integ[s]["cert_k"] != ref[s]["cert_k"]
                            for s in EVAL_SEEDS))
        a = np.array([ref[s]["N80_pen"] for s in EVAL_SEEDS], float)
        ax = np.array([ref[s]["x12"] for s in EVAL_SEEDS], float)
        fam_of = {v["pick"]: fam for fam, v in sel["envs"][name]["families"].items() if v["pick"]}
        E = {"spec": spec[name], "eps": spec[name]["eps"],
             "fdc_bf_0.5plan_equals_fdc_bf": {"n_different": integrity, "pass": integrity == 0},
             "methods": {}}
        for m in names:
            if m == "FDC-BF[0.5-plan]":
                continue
            rows = by[m]
            b = np.array([rows[s]["N80_pen"] for s in EVAL_SEEDS], float)
            bx = np.array([rows[s]["x12"] for s in EVAL_SEEDS], float)
            ent = describe(rows, EVAL_SEEDS)
            if m != "FDC-BF":
                p = paired(a, b, idx)
                px = paired(ax, bx, idx)
                ent.update({"FDCBF_over_method_N80": p["geomean_ratio"], "ci95_N80": p["ci95_two_sided"],
                            "frac_method_faster": p["frac_slower"], "frac_tied": p["frac_tied"],
                            "FDCBF_over_method_x12": px["geomean_ratio"], "ci95_x12": px["ci95_two_sided"]})
            if m in fam_of:
                ent["tuned_pick_of"] = [f for f, v in sel["envs"][name]["families"].items() if v["pick"] == m]
            E["methods"][m] = ent
        # does a frozen pilot-Neyman FDC-BF recover the adaptive gain?  paired FDC-BF[ney-pilot] / best adaptive pick
        best = sel["envs"][name]["families"]["any_adaptive"]["pick"]       # tuned on 900-949, not eval-selected
        if best:
            np_ = np.array([by["FDC-BF[ney-pilot]"][s]["N80_pen"] for s in EVAL_SEEDS], float)
            npx = np.array([by["FDC-BF[ney-pilot]"][s]["x12"] for s in EVAL_SEEDS], float)
            b = np.array([by[best][s]["N80_pen"] for s in EVAL_SEEDS], float)
            bx = np.array([by[best][s]["x12"] for s in EVAL_SEEDS], float)
            p, px = paired(np_, b, idx), paired(npx, bx, idx)
            E["neypilot_vs_best_adaptive_pick"] = {
                "best_adaptive_pick": best, "neypilot_over_adaptive_N80": p["geomean_ratio"],
                "ci95_N80": p["ci95_two_sided"], "neypilot_over_adaptive_x12": px["geomean_ratio"],
                "ci95_x12": px["ci95_two_sided"],
                "note": "< 1: frozen pilot-Neyman FDC-BF faster than the best tuned adaptive pick"}
        summ["envs"][name] = E
    summ["regime_conclusion"] = conclude(summ)
    summ["written_at"] = datetime.now().isoformat()
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1))
    write_md(summ)
    write_tex(summ)


def conclude(summ):
    lines = []
    for name, E in summ["envs"].items():
        M = E["methods"]
        ad = {m: v for m, v in M.items() if m in GRID}
        best = E["neypilot_vs_best_adaptive_pick"]["best_adaptive_pick"]
        nep = M["FDC-BF[ney-pilot]"]
        s = (f"{name} (eps {E['eps']:g}): tuned any-adaptive pick {best} FDC-BF/pick "
             f"{ad[best]['FDCBF_over_method_N80']:.3f} [{ad[best]['ci95_N80'][0]:.3f}, {ad[best]['ci95_N80'][1]:.3f}] "
             f"(switched {ad[best]['frac_switched']:.2f}); frozen pilot-Neyman FDC-BF/FDC-BF[ney-pilot] "
             f"{nep['FDCBF_over_method_N80']:.3f} [{nep['ci95_N80'][0]:.3f}, {nep['ci95_N80'][1]:.3f}]; "
             f"ney-pilot / best adaptive {E['neypilot_vs_best_adaptive_pick']['neypilot_over_adaptive_N80']:.3f} "
             f"[{E['neypilot_vs_best_adaptive_pick']['ci95_N80'][0]:.3f}, "
             f"{E['neypilot_vs_best_adaptive_pick']['ci95_N80'][1]:.3f}]") if best else name
        lines.append(s)
    return lines


ANSWER = (
    "Adaptivity CAN beat frozen 50/50 FDC-BF, but only in one of the four semi-synthetic regimes, and a frozen "
    "pilot-Neyman FDC-BF beats the adaptive members everywhere.  (1) Criteo-structure asym (control 0.02-0.05 vs "
    "treatment 0.40-0.50 in every segment): the tuned RAGE-style reset member is 13% faster than frozen 50/50 "
    "(FDC-BF/pick 1.127 [1.102, 1.153]), the tuned global menu 10% faster and the lock-v8 PJC-menu* 9% faster; this "
    "is the regime the reviewer asked for, and adaptivity wins there.  But FDC-BF with a frozen per-segment Neyman "
    "plan from an independent pilot is faster still (FDC-BF/ney-pilot 1.180 [1.154, 1.207]; ney-pilot / reset pick "
    "0.955 [0.940, 0.970]) and matches the oracle-Neyman plan, so the gain comes from the allocation, not from "
    "adapting it online; adapting pays a reset or union cost for learning what a 1000-draw-per-cell pilot already "
    "gives.  (2) X5-structure asym (same rate ranges, 50/50 pools): gains shrink to a few percent (ney-pilot 1.037, "
    "PJC-menu* 1.051, tuned picks 0.96-1.01).  The contrast with (1) is consistent with Criteo's control pools "
    "holding only about 15% of records, so a 50/50 read plan is far from both the Neyman split and the pool "
    "composition (a conjecture this block does not isolate).  (3) Mixed regimes (per-segment Neyman shares "
    "0.27 / 0.70 / 0.50, so no global share helps): no adaptive member beats frozen 50/50 (best tuned picks "
    "0.929-1.003), the per-segment menu pays M^S paths per boundary and is 20-35% slower, and frozen pilot-Neyman "
    "gives 0.99-1.03.  Neyman is not the optimal split for Bennett widths (low-variance cells read less pay the range "
    "term b*beta/(3n)), which caps every Neyman-type gain.  No method in any regime had a false certification stream "
    "(0 / 250 per method and env).  Semi-synthetic, post hoc and descriptive: it maps the regime and does not change "
    "the v8 confirmatory result.")

ORDER = ["FDC-BF[ney-pilot]", "FDC-BF[ney-oracle]", "RECT-ck-HG", "RECT-ck-HG[ney-pilot]", "RECT-ck-BF",
         "RECT-ck-BF[ney-pilot]", "HC-WoR", "HC-WoR[ney-pilot]", "PJC-local*", "PJC-menu*"]


def ordered(E):
    rest = [m for m in E["methods"] if m in GRID]
    return [m for m in ORDER if m in E["methods"]] + rest


def write_md(s):
    L = ["# v8-posthoc-F: adaptivity under strong arm-variance asymmetry (POST-HOC, DESCRIPTIVE, SEMI-SYNTHETIC)", "",
         "**Post-hoc, semi-synthetic disclosure.** " + s["disclosure"], "",
         f"Status: {s['status']}.  Ratio: {s['ratio_convention']}.  Bootstrap: {s['bootstrap']}.", "",
         "Regimes: **asym** -- every segment control rate U(0.02, 0.05), treatment U(0.40, 0.50) (Neyman control share "
         "~0.25-0.30 everywhere); **mixed** -- segments cycle A (0.02-0.05 vs 0.40-0.50, Neyman ctrl ~0.27), B "
         "(0.45-0.55 vs 0.93-0.97, Neyman ctrl ~0.70), C (0.25-0.30 vs 0.65-0.75, ~0.50).  Structure (labels, pool "
         "sizes, weights, frozen arrivals) = X5 X9 dev or Criteo CR9 dev; outcomes synthetic with exactly "
         "round(rate x N_c) ones per pool.  FDC-BF[ney-pilot] = frozen per-segment Neyman from an independent "
         f"{PILOT_N}-draw-per-cell pilot (floor {P_FLOOR}); [ney-oracle] = Neyman from the true pool variances.", ""]
    for name, E in s["envs"].items():
        sp = E["spec"]
        M = E["methods"]
        f = M["FDC-BF"]
        L += [f"## {name} (SEMI-SYNTHETIC; eps {E['eps']:g}; tau_R {sp['tau_R']:,}; 250 streams)", "",
              "Pilot-Neyman control share by segment: " + ", ".join(f"{x:.2f}" for x in sp["ney_pilot_ctrl_share"])
              + f".  eps calibration (FDC-BF geomean N80/tau on 900-919): "
              + ", ".join(f"{e}: {v['geomean_N80_over_tau']:.3f}" for e, v in sp["calibration"].items()) + ".", "",
              f"FDC-BF (frozen 50/50): geomean N80/tau {f['geomean_N80_over_tau']:.4f}, x12/tau "
              f"{f['geomean_x12_over_tau']:.4f}, false streams {f['false_streams']}, censored "
              f"{f['frac_censored']:.3f}.  Integrity: FDCBetPlan(0.5) reproduces FDC-BF on "
              f"{'all' if E['fdc_bf_0.5plan_equals_fdc_bf']['pass'] else 'NOT all'} streams.", "",
              "| method | FDC-BF/method N80 | 95% CI | FDC-BF/method x12 | 95% CI | method N80/tau | frac switched | "
              "false streams |", "|---|---|---|---|---|---|---|---|"]
        for m in ordered(E):
            v = M[m]
            tag = f" (tuned pick: {'/'.join(v['tuned_pick_of'])})" if "tuned_pick_of" in v else ""
            sw = f"{v['frac_switched']:.3f}" if "frac_switched" in v else "--"
            L.append(f"| `{m}`{tag} | {v['FDCBF_over_method_N80']:.4f} | [{v['ci95_N80'][0]:.4f}, "
                     f"{v['ci95_N80'][1]:.4f}] | {v['FDCBF_over_method_x12']:.4f} | [{v['ci95_x12'][0]:.4f}, "
                     f"{v['ci95_x12'][1]:.4f}] | {v['geomean_N80_over_tau']:.4f} | {sw} | {v['false_streams']} |")
        nb = E.get("neypilot_vs_best_adaptive_pick")
        if nb:
            L += ["", f"Frozen pilot-Neyman FDC-BF vs the tuned any-adaptive pick (`{nb['best_adaptive_pick']}`): "
                  f"ney-pilot / adaptive N80 {nb['neypilot_over_adaptive_N80']:.4f} [{nb['ci95_N80'][0]:.4f}, "
                  f"{nb['ci95_N80'][1]:.4f}], x12 {nb['neypilot_over_adaptive_x12']:.4f} [{nb['ci95_x12'][0]:.4f}, "
                  f"{nb['ci95_x12'][1]:.4f}] (< 1: frozen pilot-Neyman faster)."]
        for m in [m for m in ordered(E) if m in GRID]:
            v = M[m]
            if v.get("mean_ctrl_share_after_switch_by_seg"):
                L.append(f"Mean post-boundary control share by segment, `{m}`: "
                         + ", ".join(f"{x:.2f}" for x in v["mean_ctrl_share_after_switch_by_seg"]) + ".")
        L.append("")
    L += ["## Regime conclusion (descriptive)", ""] + [f"- {x}" for x in s["regime_conclusion"]] + [""]
    L += ["## Answer", "", ANSWER, ""]
    (OUT / "summary.md").write_text("\n".join(L) + "\n")


def write_tex(s):
    short = {"FDC-BF[ney-pilot]": "FDC-BF, frozen pilot-Neyman", "FDC-BF[ney-oracle]": "FDC-BF, frozen oracle-Neyman",
             "RECT-ck-HG": "RECT-ck-HG", "RECT-ck-HG[ney-pilot]": "RECT-ck-HG, pilot-Neyman",
             "RECT-ck-BF": "RECT-ck-BF", "RECT-ck-BF[ney-pilot]": "RECT-ck-BF, pilot-Neyman", "HC-WoR": "HC-WoR",
             "HC-WoR[ney-pilot]": "HC-WoR, pilot-Neyman", "PJC-local*": "PJC-local$^*$", "PJC-menu*": "PJC-menu$^*$"}
    envs = list(s["envs"])
    L = ["% v8-posthoc-F (POST-HOC, DESCRIPTIVE, SEMI-SYNTHETIC); generated by exp/code/run_v8_posthoc_F.py",
         "\\begin{table}[t]", "\\centering\\small", "\\setlength{\\tabcolsep}{3pt}",
         "\\caption{Semi-synthetic, post-hoc, descriptive: arm-variance asymmetry. Paired geometric-mean ratio "
         "FDC-BF (frozen 50/50) / method of $N_{80}$ with 95\\% bootstrap CI over 250 streams "
         "($<1$: FDC-BF faster). Structure from the X5 and Criteo development halves; outcomes synthetic "
         "(asym: control 0.02--0.05 vs treatment 0.40--0.50 in every segment; mixed: three segment types with Neyman "
         "control shares $\\approx$0.27/0.70/0.50). Adaptive PJC-A picks tuned on separate seeds. "
         "No method had a false certification stream.}",
         "\\label{tab:posthocF}", "\\begin{tabular}{l" + "c" * len(envs) + "}", "\\toprule",
         "Method & " + " & ".join(n.replace("X9", "X5").replace("CR9", "Criteo") for n in envs) + " \\\\",
         " & " + " & ".join(f"$\\varepsilon={s['envs'][n]['eps']:g}$" for n in envs) + " \\\\", "\\midrule"]

    def cell(v):
        return f"{v['FDCBF_over_method_N80']:.2f} [{v['ci95_N80'][0]:.2f}, {v['ci95_N80'][1]:.2f}]"
    for m in ORDER:
        L.append(short[m] + " & " + " & ".join(cell(s["envs"][n]["methods"][m]) for n in envs) + " \\\\")
    L.append("\\midrule")
    for fam, lab in (("reset", "PJC-A reset (tuned)"), ("menu", "PJC-A global menu (tuned)"),
                     ("segmenu", "PJC-A per-segment menu (tuned)")):
        cells = []
        for n in envs:
            M = s["envs"][n]["methods"]
            m = next((k for k, v in M.items() if fam in v.get("tuned_pick_of", [])), None)
            cells.append(cell(M[m]) + f" ({M[m]['frac_switched']:.2f})" if m else "--")
        L.append(lab + " & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}",
          "\\\\[2pt]{\\footnotesize Parentheses after adaptive rows: fraction of streams whose design moved off "
          "50/50.}", "\\end{table}"]
    (OUT / "table.tex").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["calib", "tune", "select", "eval", "analyse"])
    ap.add_argument("--env", nargs="*", default=ENVS)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.task == "calib":
        run_calib(args.env)
    elif args.task == "tune":
        run_rows("tune", args.env, lambda n: ["FDC-BF"] + list(GRID), TUNE_SEEDS)
    elif args.task == "select":
        select()
    elif args.task == "eval":
        if not (OUT / "selection.json").exists():
            raise SystemExit("eval refused: tune + select first (picks fixed before evaluation)")
        run_rows("eval", args.env, eval_names, EVAL_SEEDS)
    else:
        analyse()


if __name__ == "__main__":
    main()
