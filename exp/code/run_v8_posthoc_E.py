"""v8-posthoc-E: POST-HOC, DESCRIPTIVE sensitivity block (outside any lock) for the sensitivities the paper lists as
untested (supplement S9; critic review_v8 P1-7): delta split, total delta, checkpoint count K, lambda-grid density,
and the replanning batch of the adaptive joint certificates.

New file; locked v5-v8 files are imported, never modified.  DEV halves only (no eval access):
  X5 RetailHero X9 dev, seeds 950-999, eps 0.02 (decision eps of lock v8)
  CR9 Criteo dev,       seeds 950-999, eps 0.001
stop_k = 15 (N80_pen read at the 12/15 crossing of the same trajectory, as in lock v8).  Seeds 950-999 were not used
for any tuning (rival configs and the post-hoc-D picks were tuned on 900-949).

Knobs (baseline = the registered setting: K = 20, delta = 0.05 split 0.045 / 0.005, 161-point lambda grid on
[1/200, 10] x lambda_0, replan = max(200, ceil(tau_R / 2000))):
  dvar    delta_var in {0.0025, 0.01} (delta_main = 0.05 - delta_var)       FDC-BF, RECT-ck-BF, PJC x3 (methods with a split)
  dtot    total delta in {0.01, 0.10}, split 9:1 (Proposition 1's delta_main 0.009 / 0.09 prediction)
                                                                             all seven
  K       K in {10, 40}, geometric grid n_min..tau_R (env.checkpoints(K))   all seven; PJC phase boundaries are mapped to
          the K-grid checkpoint nearest (in arrivals) to the registered K = 20 boundary
  grid    lambda-grid density 81 / 321 points (same range)                  FDC-BF, RECT-ck-BF (PJC keeps 161)
  replan  replanning interval x {0.5, 2}                                    PJC-local*, PJC-menu*, PJC-A reset pick
Methods without a knob are unaffected by it (RECT-ck-HG and HC-WoR have no split and no lambda grid; the frozen-design
methods do not use replanning batches), so their baseline rows are the paired comparators there.

Usage (cwd = exp/code):  run_v8_posthoc_E.py --task run_x5 | run_cr9 | analyse
Outputs: exp/results/full/v8_posthoc_E/{run_x5,run_cr9}/results.jsonl, summary.json, summary.md, table.tex
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import copy  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import types  # noqa: E402
from datetime import datetime  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
WS = CODE.parents[1]
RES = WS / "exp/results"
OUT = RES / "full/v8_posthoc_E"
N_WORKERS = int(os.environ.get("POSTHOC_WORKERS", "14"))
TASK_ID = "v8-posthoc-E"

from dsswm.baselines.fdc_bet import _LAMBDA_GRID, FDCBet, psi_bar  # noqa: E402,F401
from dsswm.baselines.pjc_bf import RectCkBF, bennett_cell_radius  # noqa: E402
from dsswm.baselines.rect_v6 import RectCkHG, hg_mean_interval  # noqa: E402
from dsswm.streams import frontier as fr  # noqa: E402
from dsswm.streams.frontier_runner import build_ctx, true_policy_values  # noqa: E402
from dsswm.streams.frontier_runner_v6 import ctx_with_stop, run_stream_v6  # noqa: E402
from run_r5s_v7 import x_rank  # noqa: E402

SEEDS = list(range(950, 1000))
STOP_K = 15
LAYERS = {"run_x5": dict(layer="X9", eps=0.02, cfg="x9"), "run_cr9": dict(layer="CR9", eps=0.001, cfg="cr")}
RESET_PICK = "PJC-A[reset,b=2,neyman]"          # post-hoc-D dev-tuned reset pick (same on X9 and CR9)
BASE = dict(K=20, dv=0.005, dt=0.05, G=161, rm=1.0)
PJC3 = ["PJC-local", "PJC-menu", "PJC-reset"]
ALL7 = ["FDC-BF", "RECT-ck-HG", "HC-WoR", "RECT-ck-BF"] + PJC3
FOUR = ["FDC-BF", "RECT-ck-BF", "RECT-ck-HG", "HC-WoR"]
SETTINGS = {"base": (dict(), ALL7)}
for _dv in (0.0025, 0.01):
    SETTINGS[f"dvar={_dv:g}"] = (dict(dv=_dv), ["FDC-BF", "RECT-ck-BF"] + PJC3)
for _dt in (0.01, 0.10):
    SETTINGS[f"dtot={_dt:g}"] = (dict(dt=_dt, dv=_dt / 10.0), FOUR + PJC3)
for _K in (10, 40):
    SETTINGS[f"K={_K}"] = (dict(K=_K), FOUR + PJC3)
for _G in (81, 321):
    SETTINGS[f"grid={_G}"] = (dict(G=_G), ["FDC-BF", "RECT-ck-BF"])
for _rm in (0.5, 2.0):
    SETTINGS[f"replan=x{_rm:g}"] = (dict(rm=_rm), ["PJC-local", "PJC-menu", "PJC-reset"])
_ENV: dict = {}


def knobs(setting):
    k = dict(BASE)
    k.update(SETTINGS[setting][0])
    return k


def lam_grid(G):
    if G == 161:
        return _LAMBDA_GRID
    return np.exp(np.linspace(math.log(1.0 / 200.0), math.log(10.0), int(G)))


# =============================================================================================== knob-aware methods
class FDCBFE(FDCBet):
    """FDC-BF with a configurable (delta_main, delta_var) and lambda grid; ctx.delta stays 0.05 (only the ledger moves)."""

    def __init__(self, dm, dv, grid):
        super().__init__(kind="bennett", var_box="HG", fpc=True, rect=False, split=(0.045, 0.005), name="FDC-BF",
                         grid=grid)
        self.split = (float(dm), float(dv))


class RectCkBFE(RectCkBF):
    """RECT-ck-BF (matched Bennett-FPC rectangle) with configurable split and lambda grid."""

    def __init__(self, dm, dv, grid):
        super().__init__(box=False)
        self._split, self._grid = (float(dm), float(dv)), grid

    def _ledger(self, ctx):
        K = len(ctx.checkpoints)
        SA = ctx.S * ctx.A
        d_main, d_var = self._split
        return {"beta_c": math.log(2 * SA * K / d_main), "alpha_side_var": d_var / (2.0 * SA * K),
                "delta_main": d_main, "delta_var": d_var}

    def _bounds(self, ctx, st):                                 # = RectCkBF._bounds with the grid passed through
        n = np.asarray(st.n, float)
        N = np.asarray(st.N, float)
        s = np.asarray(st.sum, float)
        vlo, vhi = hg_mean_interval(st.N, st.n, st.sum, self.ledger["alpha_side_var"])
        self._vlo = np.maximum(self._vlo, vlo)
        self._vhi = np.maximum(np.minimum(self._vhi, vhi), self._vlo)
        r = bennett_cell_radius(n, N, self._vlo, self._vhi, self.ledger["beta_c"], grid=self._grid)
        mu = np.where(n > 0, s / np.maximum(n, 1.0), 0.5)
        lo = np.maximum(mu - r, s / np.maximum(N, 1.0))
        hi = np.minimum(mu + r, (s + (N - n)) / np.maximum(N, 1.0))
        exact = n >= N
        lo = np.where(exact, mu, lo)
        hi = np.where(exact, mu, hi)
        empty = n <= 0
        lo = np.where(empty, 0.0, lo)
        hi = np.where(empty, 1.0, hi)
        return np.clip(lo, 0.0, 1.0), np.clip(np.maximum(hi, lo), 0.0, 1.0)


class RectCkHGE(RectCkHG):
    """RECT-ck-HG (frozen config plan 0.5 on both layers) with total delta dt."""

    def __init__(self, dt):
        super().__init__()
        self._dt = float(dt)

    def _ledger(self, ctx):
        K = len(ctx.checkpoints)
        a = self._dt / (2.0 * ctx.S * ctx.A * K)
        return {"alpha_side": a, "delta": self._dt, "K": K}


def make(name, k, cfg):
    import run_r5s_v8 as RV8
    if name == "FDC-BF":
        return FDCBFE(k["dt"] - k["dv"], k["dv"], lam_grid(k["G"]))
    if name == "RECT-ck-BF":
        return RectCkBFE(k["dt"] - k["dv"], k["dv"], lam_grid(k["G"]))
    if name == "RECT-ck-HG":
        assert str(cfg["RECT-ck-HG"].get("plan")) == "0.5" and cfg["RECT-ck-HG"].get("alloc") is None
        return RectCkHGE(k["dt"])
    if name == "HC-WoR":
        m = RV8.make("HC-WoR", RV8.method_kw("HC-WoR", cfg))
        dt = float(k["dt"])
        if abs(dt - 0.05) > 1e-15:
            orig = m.cs_params

            def cs_params(self, ctx, _o=orig, _dt=dt):
                p = _o(ctx)
                p["alpha"] = _dt / (ctx.S * ctx.A)
                return p
            m.cs_params = types.MethodType(cs_params, m)
        return m
    if name in ("PJC-local", "PJC-menu", "PJC-reset"):
        if name == "PJC-reset":
            from dsswm.baselines.pjc_adapt import PJCAdapt
            from run_v8_posthoc_D import GRID
            kw = dict(GRID[RESET_PICK])
            kw["boundaries"] = remap_boundaries(kw["boundaries"], k["K"])
            m = PJCAdapt(**kw, name=RESET_PICK)
        else:
            kw = RV8.method_kw(name, cfg)
            kw["boundaries"] = list(remap_boundaries(kw["boundaries"], k["K"]))
            m = RV8.make(name, kw)
        split = (k["dt"] - k["dv"], k["dv"])
        if abs(split[0] - 0.045) > 1e-15 or abs(split[1] - 0.005) > 1e-15:
            from dsswm.baselines.pjc_adapt import PJCAdapt, adapt_ledger
            from dsswm.baselines.pjc_bf import pjc_ledger
            orig = m.setup

            def setup(self, ctx, _o=orig, _sp=split):
                _o(ctx)
                M = len(self.menu) if self.menu else 1
                if isinstance(self, PJCAdapt):
                    self.ledger = adapt_ledger(ctx, self.family, self.boundaries, M, split=_sp)
                else:
                    self.ledger = pjc_ledger(ctx, self.mode, self.boundaries, M, split=_sp)
            m.setup = types.MethodType(setup, m)
        return m
    raise ValueError(name)


def remap_boundaries(bnd, K):
    """Registered boundaries are K = 20 checkpoint indices; map each to the K-grid index nearest in arrivals."""
    if int(K) == 20 or not bnd:
        return tuple(int(b) for b in bnd)
    c20 = _ENV["ctxs"][20].checkpoints
    cK = _ENV["ctxs"][int(K)].checkpoints
    out = []
    for b in bnd:
        j = int(np.argmin(np.abs(np.log(cK[:-1].astype(float)) - math.log(float(c20[int(b)])))))
        out.append(min(j, int(K) - 2))
    return tuple(sorted(set(out)))


# =============================================================================================== runner
def init_env(layer, eps, cfg_key):
    if layer == "X9":
        from dsswm.envs.x5_v8 import X5LayerEnv
        env = X5LayerEnv("dev")
    else:
        from dsswm.envs.pool_replay import PoolReplayEnv
        env = PoolReplayEnv("CR9", "dev")
    assert env.half == "dev"
    probs = fr.cr_problems("visit")
    ctxs = {}
    for K in (10, 20, 40):
        ctxs[K] = ctx_with_stop(build_ctx(env, probs, eps, K=K), STOP_K)
    c20 = ctxs[20]
    J = true_policy_values(c20.pols, env.w, env.true_mu("visit"))
    Js = np.array([J[c20.feas[q]].max() for q in range(c20.Q)])
    cfg = json.loads((RES / "v8_gates" / f"{cfg_key}_configs.json").read_text())["selected"]
    _ENV.update(env=env, ctxs=ctxs, J=J, Js=Js, layer=layer, cfg=cfg, base_replan=int(c20.replan))


def ctx_for(k):
    ctx = _ENV["ctxs"][int(k["K"])]
    if abs(k["rm"] - 1.0) > 1e-12:
        ctx = copy.copy(ctx)
        ctx.replan = max(1, int(round(_ENV["base_replan"] * k["rm"])))
    return ctx


def job(a):
    setting, name, seed = a
    k = knobs(setting)
    ctx = ctx_for(k)
    try:
        m = make(name, k, _ENV["cfg"])
        t0 = time.perf_counter()
        s, rows = run_stream_v6(_ENV["env"], m, seed, ctx.problems, ctx.eps, outcome="visit", ctx=ctx,
                                J_true=_ENV["J"], J_star=_ENV["Js"], keep_U=True)
        ck = ctx.checkpoints
        tau = int(ctx.tau_R)
        curve = [r["n_cert"] for r in rows]
        nfalse = [r["n_false"] for r in rows]
        k12 = next((i for i, c in enumerate(curve) if c >= 12), None)
        n80 = int(ck[k12]) if (k12 is not None and nfalse[k12] == 0) else tau
        xx = x_rank([r["U"] for r in rows], ck, ctx.eps, 12)
        led = getattr(m, "ledger", None) or {}
        return {"setting": setting, "method": name, "seed": int(seed), "layer": _ENV["layer"], "half": "dev",
                "eps": float(ctx.eps), "K": int(len(ck)), "replan": int(ctx.replan), "knobs": k, "posthoc": True,
                "block": TASK_ID, "N80_pen": n80, "k80": k12, "x12": float(xx) if xx is not None else float(tau),
                "fwer_event": bool(s["fwer_event"]), "n_false": int(s["n_false"]), "n_cert": int(s["n_cert"]),
                "cert_k": s["cert_k"], "billing_ok": bool(s["billing_ok"]), "schedule_digest": s["schedule_digest"],
                "tau_R": tau, "beta": float(led["beta"]) if "beta" in led else
                (float(led["beta_c"]) if "beta_c" in led else None),
                "sec": round(time.perf_counter() - t0, 3), "error": None}
    except Exception:  # noqa: BLE001
        return {"setting": setting, "method": name, "seed": int(seed), "error": traceback.format_exc()}


def load(task):
    f = OUT / task / "results.jsonl"
    key = {}
    if f.exists():
        for line in f.read_text().splitlines():
            r = json.loads(line)
            key[(r["setting"], r["method"], r["seed"])] = r
    return key


def run_task(task):
    T = LAYERS[task]
    d = OUT / task
    d.mkdir(parents=True, exist_ok=True)
    done = set(load(task))
    jobs = [(st, m, s) for st, (_, ms) in SETTINGS.items() for m in ms for s in SEEDS if (st, m, s) not in done]
    # heavy first (HC-WoR, then K = 40) for better packing
    jobs.sort(key=lambda j: (j[1] != "HC-WoR", j[0] != "K=40"))
    print(f"[{task}] {len(jobs)} jobs", flush=True)
    if not jobs:
        return
    init_env(T["layer"], T["eps"], T["cfg"])
    n = errs = 0
    t0 = time.time()
    with get_context("fork").Pool(N_WORKERS) as pool, open(d / "results.jsonl", "a") as f:
        for r in pool.imap_unordered(job, jobs, chunksize=1):
            if r.get("error"):
                errs += 1
                with open(d / "errors.log", "a") as g:
                    g.write(f"{r['setting']} {r['method']} {r['seed']}\n{r['error']}\n")
                continue
            f.write(json.dumps(r) + "\n")
            f.flush()
            n += 1
            if n % 100 == 0:
                print(f"[{task}] {n}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
                (RES / f"v8_posthoc_E_{task}_PROGRESS.json").write_text(json.dumps(
                    {"task_id": f"v8_posthoc_E_{task}", "step": n, "total_steps": len(jobs),
                     "updated_at": datetime.now().isoformat()}))
    print(f"[{task}] done, {errs} errors, {time.time() - t0:.0f}s", flush=True)


# =============================================================================================== analysis
def geo(v):
    return float(np.exp(np.mean(np.log(np.asarray(v, dtype=float)))))


def reproduce_pilot_x9(R):
    """Integrity: the base-setting rows on X9 dev 950-999 must equal the lock-v8 dev runner check (v8a_pilot, eps 0.02)."""
    f = RES / "pilots/v8a_pilot/results.jsonl"
    if not f.exists():
        return {"available": False}
    want = {"FDC-BF": "FDC-BF", "RECT-ck-HG": "RECT-ck-HG", "HC-WoR": "HC-WoR", "RECT-ck-BF": "RECT-ck-BF",
            "PJC-local": "PJC-local", "PJC-menu": "PJC-menu"}
    P = {}
    for line in f.read_text().splitlines():
        r = json.loads(line)
        if r.get("error") is None and abs(float(r["eps"]) - 0.02) < 1e-12 and r["method"] in want:
            P[(r["method"], r["seed"])] = r
    out = {}
    for m in want:
        diff = [s for s in SEEDS if (m, s) not in P or ("base", m, s) not in R
                or P[(m, s)]["N80_pen"] != R[("base", m, s)]["N80_pen"]
                or P[(m, s)]["schedule_digest"] != R[("base", m, s)]["schedule_digest"]]
        out[m] = {"n_compared": len(SEEDS), "n_different": len(diff), "different_seeds": diff[:10]}
    return {"available": True, "source": "exp/results/pilots/v8a_pilot/results.jsonl (eps 0.02)", "per_method": out,
            "pass": all(v["n_different"] == 0 for v in out.values())}


def analyse():
    from dsswm.stats.v6_analysis import boot_idx, paired
    idx = boot_idx(len(SEEDS))
    summ = {"block": TASK_ID,
            "status": "POST-HOC, DESCRIPTIVE -- outside any preregistration lock (v5-v8); not confirmatory",
            "disclosure": ("Run on 2026-10-04 AFTER the v8 confirmatory analysis and the r3 paper revision, in response "
                           "to reviewer requests (supplement S9, critic review_v8 P1-7) for the sensitivities the paper "
                           "lists as untested.  Knob values were fixed before any row of this block was computed (the "
                           "requested values plus the total-delta values that Proposition 1's S9 prediction names).  "
                           "DEV halves only (X5 X9 dev and CR9 dev, seeds 950-999, not used for any tuning); no "
                           "evaluation row was accessed.  Rival configs are the frozen lock-v8 configs, not re-tuned "
                           "per setting.  50 streams per cell: CIs are wider than in the confirmatory tables.  Nothing "
                           "here changes a locked verdict."),
            "ratio_convention": "FDC-BF / rival paired geomean of N80_pen at the same setting (< 1: FDC-BF faster)",
            "bootstrap": "v6 rule: B = 10^4 resamples of streams, seed 42, two-sided percentile 95% CI on the geomean",
            "seeds": [SEEDS[0], SEEDS[-1], len(SEEDS)], "stop_k": STOP_K, "baseline_knobs": BASE,
            "settings": {s: {"knobs": knobs(s), "methods_run": ms} for s, (_, ms) in SETTINGS.items()}, "layers": {}}
    rivals = ["RECT-ck-HG", "HC-WoR", "RECT-ck-BF", "PJC-local", "PJC-menu", "PJC-reset"]
    for task, T in LAYERS.items():
        R = load(task)
        lay = T["layer"]

        def rows(setting, m):
            st = setting if m in SETTINGS[setting][1] else "base"
            got = [R.get((st, m, s)) for s in SEEDS]
            if any(g is None for g in got):
                raise RuntimeError(f"missing rows {lay} {st} {m}")
            return st, got

        L = {"eps": T["eps"], "tau_R": None, "settings": {}}
        if lay == "X9":
            L["reproduces_v8a_pilot"] = reproduce_pilot_x9(R)
        for setting, (_, ms) in SETTINGS.items():
            _, F = rows(setting, "FDC-BF")
            a = np.array([r["N80_pen"] for r in F], float)
            tau = F[0]["tau_R"]
            L["tau_R"] = tau
            _, F0 = rows("base", "FDC-BF")
            a0 = np.array([r["N80_pen"] for r in F0], float)
            own = paired(a, a0, idx)
            ent = {"knobs": knobs(setting), "varied": [m for m in ms],
                   "FDC-BF": {"geomean_N80_over_tau": geo(a / tau), "false_streams": int(sum(r["fwer_event"] for r in F)),
                              "beta": F[0]["beta"], "billing_ok_all": all(r["billing_ok"] for r in F),
                              "vs_FDC-BF_base": {"ratio": own["geomean_ratio"], "ci95": own["ci95_two_sided"]}},
                   "rivals": {}}
            for m in rivals:
                st, B = rows(setting, m)
                b = np.array([r["N80_pen"] for r in B], float)
                p = paired(a, b, idx)
                ent["rivals"][m] = {"rows_from_setting": st, "FDCBF_over_rival": p["geomean_ratio"],
                                    "ci95": p["ci95_two_sided"], "ub95": p["ub95_one_sided"],
                                    "frac_fdc_faster": p["frac_faster"], "frac_tied": p["frac_tied"],
                                    "frac_fdc_slower": p["frac_slower"], "rival_geomean_N80_over_tau": geo(b / tau),
                                    "x12_ratio_geomean": float(np.exp(np.mean(np.log(
                                        np.array([r["x12"] for r in F]) / np.array([r["x12"] for r in B]))))),
                                    "rival_false_streams": int(sum(r["fwer_event"] for r in B)),
                                    "rival_billing_ok_all": all(r["billing_ok"] for r in B),
                                    "same_schedule_frac": float(np.mean([x["schedule_digest"] == y["schedule_digest"]
                                                                         for x, y in zip(F, B)]))}
            L["settings"][setting] = ent
        summ["layers"][lay] = L
    summ["robustness"] = robustness(summ)
    summ["written_at"] = datetime.now().isoformat()
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1))
    write_md(summ)
    write_tex(summ)


def robustness(s):
    out = {}
    for lay, L in s["layers"].items():
        rect = {st: {m: (e["rivals"][m]["FDCBF_over_rival"], e["rivals"][m]["ci95"][1])
                     for m in ("RECT-ck-HG", "HC-WoR", "RECT-ck-BF")} for st, e in L["settings"].items()}
        pjc = {st: {m: (e["rivals"][m]["FDCBF_over_rival"], e["rivals"][m]["ci95"])
                    for m in ("PJC-local", "PJC-menu", "PJC-reset")} for st, e in L["settings"].items()}
        worst_rect = max((v[0], v[1], st, m) for st, d in rect.items() for m, v in d.items())
        pj_hi = max((v[1][1], v[0], st, m) for st, d in pjc.items() for m, v in d.items())
        pj_lo = min((v[0], v[1][0], st, m) for st, d in pjc.items() for m, v in d.items())
        fs = {st: {"FDC-BF": e["FDC-BF"]["false_streams"],
                   **{m: e["rivals"][m]["rival_false_streams"] for m in e["rivals"]}} for st, e in L["settings"].items()}
        out[lay] = {"rectangles_all_ci_upper_below_1": all(v[1] < 1 for d in rect.values() for v in d.values()),
                    "rect_worst": {"ratio": worst_rect[0], "ci_upper": worst_rect[1], "setting": worst_rect[2],
                                   "rival": worst_rect[3]},
                    "pjc_max_ci_upper": {"ci_upper": pj_hi[0], "ratio": pj_hi[1], "setting": pj_hi[2], "rival": pj_hi[3]},
                    "pjc_min_ratio": {"ratio": pj_lo[0], "ci_lower": pj_lo[1], "setting": pj_lo[2], "rival": pj_lo[3]},
                    "pjc_all_ci_upper_below_1.05": pj_hi[0] < 1.05,
                    "false_streams_total": int(sum(sum(d.values()) for d in fs.values())),
                    "false_streams_by_setting": fs}
    return out


def _fmt(e, m):
    r = e["rivals"][m]
    return f"{r['FDCBF_over_rival']:.3f} [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]"


ORDER = ["base", "dvar=0.0025", "dvar=0.01", "dtot=0.01", "dtot=0.1", "K=10", "K=40", "grid=81", "grid=321",
         "replan=x0.5", "replan=x2"]
LABEL_MD = {"base": "baseline (K 20, delta 0.05 = 0.045/0.005, 161 lambdas)", "dvar=0.0025": "delta_var 0.0025",
            "dvar=0.01": "delta_var 0.01", "dtot=0.01": "delta 0.01 (0.009/0.001)", "dtot=0.1": "delta 0.10 (0.09/0.01)",
            "K=10": "K = 10", "K=40": "K = 40", "grid=81": "lambda grid 81", "grid=321": "lambda grid 321",
            "replan=x0.5": "replan x0.5", "replan=x2": "replan x2"}


def write_md(s):
    L = ["# v8-posthoc-E: untested sensitivities (POST-HOC, DESCRIPTIVE, DEV ONLY)", "",
         "**Post-hoc disclosure.** " + s["disclosure"], "",
         f"Status: {s['status']}.  Ratio: {s['ratio_convention']}.  Bootstrap: {s['bootstrap']}.  "
         "Rival columns for knobs a rival does not have (split, lambda grid for RECT-ck-HG / HC-WoR; replanning for "
         "frozen designs) use that rival's baseline rows, which the knob cannot change.  PJC members are run at the "
         "row's split, total delta, K and replanning setting (same ledger as FDC-BF); in the lambda-grid rows they keep "
         "the 161-point grid, which changed no FDC-BF or RECT-ck-BF row.  At K = 10 / 40 the PJC phase boundaries are "
         "mapped to the checkpoint nearest in arrivals to the registered K = 20 boundary.", ""]
    for lay, Ld in s["layers"].items():
        nm = "X5 RetailHero X9 dev, seeds 950-999, eps 0.02" if lay == "X9" else "CR9 Criteo dev, seeds 950-999, eps 0.001"
        L += [f"## {nm} (tau_R = {Ld['tau_R']:,})", ""]
        if "reproduces_v8a_pilot" in Ld:
            rp = Ld["reproduces_v8a_pilot"]
            L += [f"Integrity: baseline rows reproduce the lock-v8 dev runner check (v8a_pilot, eps 0.02) on all six "
                  f"shared methods x 50 seeds: **{'yes' if rp.get('pass') else 'NO'}** "
                  f"({', '.join(f'{m} {v['n_different']} diff' for m, v in rp.get('per_method', {}).items())}).", ""]
        L += ["| setting | FDC-BF N80/tau | FDC-BF vs own baseline | / RECT-ck-HG* | / HC-WoR* | / RECT-ck-BF | "
              "/ PJC-local* | / PJC-menu* | / PJC reset | false streams (FDC-BF; any rival) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for st in ORDER:
            e = Ld["settings"][st]
            f = e["FDC-BF"]
            fr_ = sum(r["rival_false_streams"] for m, r in e["rivals"].items() if r["rows_from_setting"] == st
                      or st == "base")
            L.append(f"| {LABEL_MD[st]} | {f['geomean_N80_over_tau']:.4f} | {f['vs_FDC-BF_base']['ratio']:.3f} | " +
                     " | ".join(_fmt(e, m) for m in ("RECT-ck-HG", "HC-WoR", "RECT-ck-BF", "PJC-local", "PJC-menu",
                                                     "PJC-reset")) + f" | {f['false_streams']}; {fr_} |")
        L.append("")
    L += ["## Robustness", ""]
    for lay, r in s["robustness"].items():
        w, ph, pl = r["rect_worst"], r["pjc_max_ci_upper"], r["pjc_min_ratio"]
        L.append(f"- **{lay}.** Rectangles: every FDC-BF/rectangle CI upper bound below 1 at every setting: "
                 f"{'yes' if r['rectangles_all_ci_upper_below_1'] else 'NO'}; worst {w['ratio']:.3f} (CI upper "
                 f"{w['ci_upper']:.3f}, {w['rival']} at {w['setting']}).  Joint certificates (PJC): largest CI upper "
                 f"bound {ph['ci_upper']:.3f} ({ph['rival']} at {ph['setting']}, ratio {ph['ratio']:.3f}); smallest "
                 f"ratio {pl['ratio']:.3f} ({pl['rival']} at {pl['setting']}); all PJC CI upper bounds below 1.05: "
                 f"{'yes' if r['pjc_all_ci_upper_below_1.05'] else 'NO'}.  False streams over all method-setting cells: "
                 f"{r['false_streams_total']}.")
    L += ["", "## Interpretation (descriptive)", "", INTERP, "",
          "x12 (interpolated time to rank 12, finer than the checkpoint grid), FDC-BF/RECT-ck-HG*: " +
          "; ".join(f"{lay} " + ", ".join(f"{st} {Ld['settings'][st]['rivals']['RECT-ck-HG']['x12_ratio_geomean']:.3f}"
                                          for st in ("base", "dtot=0.01", "dtot=0.1"))
                    for lay, Ld in s["layers"].items()) + "."]
    L += ["", "See summary.json for every cell (CI, UB95, fast/tied/slow fractions, rival geomeans, betas)."]
    (OUT / "summary.md").write_text("\n".join(L) + "\n")


INTERP = (
    "**The conclusions are robust on these dev streams.**  (1) *Rectangles (C2 direction):* FDC-BF/rectangle stays "
    "between 0.27 and 0.36 on X5 and between 0.50 and 0.69 on CR9 at every setting, with every CI upper bound below 1 "
    "(worst 0.369 on X5 and 0.723 on CR9, both HC-WoR* at K = 40).  The matched Bennett rectangle (RECT-ck-BF, the C4 "
    "contrast) moves least (X5 0.27-0.35, CR9 0.50-0.53).  (2) *Joint certificates (C3 direction):* when PJC is given "
    "the same split, delta, K and replanning batch, FDC-BF/PJC-local* stays within 0.976-1.016 (largest CI upper bound "
    "1.037, below the registered 1.05 margin), and FDC-BF stays ahead of PJC-menu* and of the post-hoc reset pick at "
    "every setting (point estimates 0.89-0.98; CI upper bound reaches 1.000 only for PJC-menu* at K = 10 on X5).  So the tie with the no-reset joint twin and the small lead over the adaptive members "
    "do not depend on the registered knob values.  (3) *Knob by knob:* the lambda-grid density changes no N80 row "
    "of FDC-BF or RECT-ck-BF on either layer (x12 changes by at most 1.1%), so the 161-point grid is not a tuned "
    "choice.  Moving delta_var between 0.0025 and 0.01 changes FDC-BF by at most 2%.  K = 10 coarsens N80 for "
    "everyone, and K = 40 leaves the ratios within about 0.02 (X5) and 0.04 (CR9) of baseline.  Halving or doubling "
    "the replanning batch changes FDC-BF/PJC by at most 0.02; adaptive members gain nothing from finer batches.  On "
    "X5 the x0.5 batch (100 arrivals) is below the registered 200-arrival floor.  (4) *Proposition 1's delta "
    "prediction (S9) is borne out on CR9 and not on X5.*  On CR9, delta = 0.01 favours the joint width against "
    "RECT-ck-HG* (0.613 against 0.649 at baseline; x12 0.611 against 0.636), and delta = 0.10 leaves the ratio about "
    "unchanged or slightly toward the rectangle (x12 0.647).  On X5 both delta changes move the ratio slightly toward "
    "the rectangle (0.349 / 0.345 against 0.336).  These shifts are within about 0.04, i.e. second-order.  The total "
    "delta changes are not paper settings: the guarantee level moves with them (FWER <= delta).  (5) *Validity:* 0 "
    "false streams in all 5,500 method-setting-stream runs (2 layers x 55 method-setting cells x 50 streams), including delta = 0.10.")


LABEL_TEX = {"base": r"baseline", "dvar=0.0025": r"$\delta_{\mathrm{var}}=0.0025$",
             "dvar=0.01": r"$\delta_{\mathrm{var}}=0.01$", "dtot=0.01": r"$\delta=0.01$",
             "dtot=0.1": r"$\delta=0.10$", "K=10": r"$K=10$", "K=40": r"$K=40$", "grid=81": r"$\lambda$ grid 81",
             "grid=321": r"$\lambda$ grid 321", "replan=x0.5": r"replan $\times 0.5$", "replan=x2": r"replan $\times 2$"}


def write_tex(s):
    def c(e, m):
        r = e["rivals"][m]
        return f"{r['FDCBF_over_rival']:.2f} [{r['ci95'][0]:.2f}, {r['ci95'][1]:.2f}]"

    L = [r"% v8-posthoc-E (post hoc, descriptive, dev halves only; generated by exp/code/run_v8_posthoc_E.py)",
         r"\begin{table*}[t]",
         r"\caption{\textbf{Sensitivity (post hoc, descriptive, development halves, 50 streams per cell, seeds "
         r"950--999).} Paired geometric-mean ratio FDC-BF/rival of $\Npen$ with 95\% bootstrap CI ($<1$: FDC-BF reads "
         r"fewer rows); X5 at $\eps=0.02$, CR9 at $\eps=0.001$. Baseline: $K=20$, $\delta=0.05$ split 0.045/0.005, "
         r"161-point $\lambda$ grid, registered replanning batch. Rivals without the varied knob keep their baseline "
         r"rows; rival configurations are the frozen lock-v8 ones. FS: false streams of FDC-BF / of all methods run at "
         r"that setting.}",
         r"\label{tab:sens}", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{@{}llllllll@{}}", r"\toprule",
         r"Setting & RECT-ck-HG* & HC-WoR* & RECT-ck-BF & PJC-local* & PJC-menu* & PJC reset & FS \\"]
    for lay, Ld in s["layers"].items():
        L += [r"\midrule",
              (r"\multicolumn{8}{@{}l}{\emph{X5 RetailHero (development half), $\tau_R=" + f"{Ld['tau_R']:,}".replace(",", "{,}")
               + r"$}} \\") if lay == "X9" else
              (r"\multicolumn{8}{@{}l}{\emph{Criteo CR9 (development half), $\tau_R=" + f"{Ld['tau_R']:,}".replace(",", "{,}")
               + r"$}} \\")]
        for st in ORDER:
            e = Ld["settings"][st]
            vary = set(SETTINGS[st][1])
            cells = []
            for m in ("RECT-ck-HG", "HC-WoR", "RECT-ck-BF", "PJC-local", "PJC-menu", "PJC-reset"):
                cells.append(c(e, m) if (st == "base" or m in vary) else "--")
            nf_r = sum(e["rivals"][m]["rival_false_streams"] for m in e["rivals"] if st == "base" or m in vary)
            L.append(f"{LABEL_TEX[st]} & " + " & ".join(cells) + f" & {e['FDC-BF']['false_streams']}/{nf_r} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (OUT / "table.tex").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["analyse"] + sorted(LAYERS))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.task == "analyse":
        analyse()
    else:
        run_task(a.task)


if __name__ == "__main__":
    main()
