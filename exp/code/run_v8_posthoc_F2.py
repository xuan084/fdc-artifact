"""v8-posthoc-F2: POST-HOC, DESCRIPTIVE, SEMI-SYNTHETIC block (outside any lock) for external reviewer paper_r4_review S1 / fix 2:
a fair pilot-versus-adaptive comparison.

Problem with block F: FDC-BF[ney-pilot] used 1000 independent pilot draws per cell (18,000 outcomes per 9-segment,
2-arm instance) that were NOT charged to N80, while the adaptive PJC-A members got no pilot at all.

New file; block F (run_v8_posthoc_F.py, its envs, selection and eval rows) and the locked v5-v8 modules are imported /
read, never modified.  Same four semi-synthetic environments as F (structure = X5 X9 dev and Criteo CR9 dev halves,
synthetic Bernoulli outcome pools; eps from F's envs.json), same reporting seeds 950-999 + 36000-36199 (250 streams).
F's names are kept: 'X9-*' = X5-structure environments, 'CR9-*' = Criteo-structure environments.

Pilot (choice documented): INDEPENDENT EXTRA DRAWS, with replacement, from the same finite pools, i.e. per cell
Bernoulli(mu_c) with mu_c the exact pool mean; nested sizes n in {250, 1000} per cell (the 250-pilot is the first 250
of the 1000 draws); drawn per STREAM from rng([2026, 10, 5, env_code, seed]) so that every method on a stream sees the
same pilot, and the pilot varies across streams (F used one fixed pilot per environment).  Why not a WoR prefix
removed from the replay: a plan computed from the first reads of the replay is not independent of the replay
permutation, so the FDC-BF frozen-design theorem would no longer apply to the full-pool estimator; making that valid
means treating the pilot as a charged phase with a phase-local estimator, which is exactly the PJC-A 'reset'
construction already in the table.  Independent draws keep every method in its guarantee class: conditional on the
pool contents, the pilot is independent of the within-pool permutation (the same argument that keeps F's
ney-oracle valid).

(i) Charged pilot.  N80_charged = N80_pen + S*A*n / J, where S*A*n is the per-instance pilot cost (18n reads; 4,500 or
    18,000) and J is the number of certification instances (frontier runs on the same pools) that share one pilot:
    J = 1 (full charge), 4, 16, inf (F's uncharged convention).  Also the break-even J* at which the charged pilot
    method ties the comparator (point estimate).  Pilot size 0 = no pilot (FDC-BF frozen 50/50 / original PJC-A).
(ii) Equal information.  The adaptive picks of F (tuned on 900-949 without a pilot; NOT re-tuned here) get the same
    per-stream pilot:
      'init'       phase-0 plan = the pilot per-segment Neyman plan (floor 0.1) instead of 50/50.  For the global
                   menu family the M menu designs are re-centred on the pilot plan: design(c) has per-segment control
                   share logistic(logit(p_pilot,s) + logit(c)) (c = 0.5 -> the pilot plan), so M and the path ledger
                   are unchanged and the menu stays data-free given the pilot.
      'init+warm'  'init' plus a variance warm start: every design-rule plug-in variance uses the pooled
                   (pilot + replay) Laplace mean (S_c + s_pilot + 1) / (n_c + n_pilot + 2).  The certificate itself
                   uses replay data only.
    Validity: the pilot is independent of the replay permutation, so every plan stays F_{T_j} v sigma(pilot)-
    measurable and the per-phase / per-path FDC-bet-1 argument of PJC-A applies unchanged (same ledger).
(iii) Phase-wise TU joint rival: NOT implemented (not cheap: needs a per-phase maximal inequality with a phase-local
    checkpoint ledger and a new variance-box ledger); the evaluated adaptive variants remain checkpoint-strength.

Pilot size 0 rows (FDC-BF frozen 50/50 and the two PJC-A picks without a pilot) are READ from block F's eval rows
(deterministic code, same seeds); a reproduction check reruns them on seeds 950-959 under this file and requires
identical N80_pen and cert_k.

Tasks (cwd = exp/code):  --task repro | eval [--shard i --nshards n] | analyse
Outputs: exp/results/full/v8_posthoc_F2/{repro/<env>.jsonl, eval/<env>.s<i>.jsonl, summary.{md,json}, table.tex}
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
RES = WS / "exp/results"
FDIR = RES / "full/v8_posthoc_F"
OUT = RES / "full/v8_posthoc_F2"
N_WORKERS = int(os.environ.get("POSTHOC_WORKERS", "4"))
TASK_ID = "v8-posthoc-F2"

import run_v8_posthoc_F as F  # noqa: E402
from dsswm.baselines.b4_bal import balanced_alloc  # noqa: E402
from dsswm.baselines.fdc_bet import make_variant  # noqa: E402
from dsswm.baselines.pjc_adapt import SHARE_GRID, PJCAdapt  # noqa: E402
from dsswm.baselines.pjc_bf import FDCBetPlan, neyman_matrix  # noqa: E402
from dsswm.streams import frontier as fr  # noqa: E402
from dsswm.streams.frontier_runner import build_ctx, true_policy_values  # noqa: E402
from dsswm.streams.frontier_runner_v6 import ctx_with_stop, run_stream_v6  # noqa: E402
from run_r5s_v7 import x_rank  # noqa: E402
from run_v8_posthoc_D import GRID  # noqa: E402

ENVS = F.ENVS
EVAL_SEEDS = F.EVAL_SEEDS
REPRO_SEEDS = list(range(950, 960))
PILOTS = (250, 1000)
PILOT_MAX = max(PILOTS)
P_FLOOR = 0.1
J_AMORT = (1, 4, 16, None)                      # None = uncharged (F's convention)
VARIANTS = ("init", "init+warm")
ENV_CODE = {"X9-asym": 91, "X9-mixed": 92, "CR9-asym": 391, "CR9-mixed": 392}
_ENV: dict = {}


# =============================================================================================== pilot
def pilot_counts(name, seed, mu):
    """Per-stream independent pilot: PILOT_MAX with-replacement draws per cell from the finite pool (Bernoulli(mu_c));
    returns {n: successes (S, A)} for the nested sizes."""
    rng = np.random.default_rng([2026, 10, 5, ENV_CODE[name], int(seed)])
    x = rng.random(mu.shape + (PILOT_MAX,)) < mu[..., None]
    cs = np.cumsum(x, axis=-1)
    return {n: cs[..., n - 1].astype(float) for n in PILOTS}


def pilot_neyman(succ, n):
    mh = (succ + 1.0) / (n + 2.0)
    return neyman_matrix(mh * (1.0 - mh), floor=P_FLOOR)


# =============================================================================================== PJC-A with a pilot
def _logit(x):
    return np.log(x) - np.log1p(-x)


class PJCAPilot(PJCAdapt):
    """PJC-A member given an independent pilot ('init' plan and optional variance warm start); see module doc."""

    def __init__(self, base_kw, pilot_succ, pilot_n, warm, name):
        super().__init__(**base_kw, name=name)
        if self.family == "segmenu":
            raise ValueError("segmenu not used in F2")
        self.pilot_s = np.asarray(pilot_succ, float)
        self.pilot_n = int(pilot_n)
        self.warm = bool(warm)
        self.p_pilot = pilot_neyman(self.pilot_s, self.pilot_n)

    def setup(self, ctx):
        super().setup(ctx)
        self._p = self.p_pilot.copy()
        self._p0 = self._p.copy()
        self.plans = [self._p.copy()]

    def _design(self, c):
        """Menu design c re-centred on the pilot plan (c = 0.5 -> pilot plan); rows sum to 1, floor p_floor."""
        if self.family == "reset":
            raise RuntimeError
        sh = 1.0 / (1.0 + np.exp(-(_logit(self.p_pilot[:, 0]) + _logit(float(c)))))
        sh = np.clip(sh, self.p_floor, 1.0 - self.p_floor)
        return np.stack([sh, 1.0 - sh], axis=1)

    def _v(self, st):
        n, s = np.asarray(st.n, float), np.asarray(st.sum, float)
        if self.warm:
            n, s = n + self.pilot_n, s + self.pilot_s
        mh = (s + 1.0) / (n + 2.0)
        return mh * (1.0 - mh)

    def _cell_var_fn(self, ctx, st):
        n = np.asarray(st.n, float)
        N = np.asarray(st.N, float)
        w = np.asarray(ctx.w, float)
        v = self._v(st)
        H = min(float(ctx.tau_R), 2.0 * float(st.t))
        arr = max(H - float(st.t), 0.0) * w
        if self.mode == "local":
            Np = N - n
            coef2 = (w[:, None] * np.where(N > 0, Np / np.maximum(N, 1.0), 0.0)) ** 2

            def f(p):
                npj = np.minimum(Np, p * arr[:, None])
                with np.errstate(divide="ignore", invalid="ignore"):
                    t = np.where(Np > 0, v * (1.0 / np.maximum(npj, 1.0) - 1.0 / np.maximum(Np, 1.0)), 0.0)
                return coef2 * np.maximum(t, 0.0)
        else:
            coef2 = (w ** 2)[:, None] * np.ones_like(n)

            def f(p):
                npj = np.minimum(N, n + p * arr[:, None])
                with np.errstate(divide="ignore", invalid="ignore"):
                    t = np.where(N > 0, v * (1.0 / np.maximum(npj, 1.0) - 1.0 / np.maximum(N, 1.0)), 0.0)
                return coef2 * np.maximum(t, 0.0)
        return f

    def _adaptive_plan(self, ctx, st, J, ihs, cache, out):
        S, A = ctx.S, ctx.A
        n, N = np.asarray(st.n, float), np.asarray(st.N, float)
        v = self._v(st)
        if self.arule == "neyman":                                         # reset only here
            r = np.where(N > 0, (N - n) / np.maximum(N, 1.0), 0.0)
            return neyman_matrix(v, r, self.p_floor), None
        if self.arule == "proj":                                           # menu, projected global variance
            H = min(float(ctx.tau_R), 2.0 * float(st.t))
            arr = (H - st.t) * np.asarray(ctx.w, float)
            best, bj = np.inf, 0
            for j, c in enumerate(self.menu):
                p = self._design(c)
                npj = np.minimum(N, n + p * arr[:, None])
                with np.errstate(divide="ignore", invalid="ignore"):
                    term = np.where(npj > 0, v * (1.0 / np.maximum(npj, 1.0) - 1.0 / np.maximum(N, 1.0)), 0.0)
                obj = float(((np.asarray(ctx.w, float) ** 2)[:, None] * np.maximum(term, 0.0)).sum())
                if obj < best - 1e-18:
                    best, bj = obj, j
            return self._design(self.menu[bj]), bj
        viol = self._violators(ctx, J, ihs, cache, out)
        if not viol:                                                       # all certified: keep the phase-0 plan
            if self.family == "menu":
                j = self.menu.index(0.5) if 0.5 in self.menu else 0
                return self._design(self.menu[j]), j
            return self._p0.copy(), None
        f = self._cell_var_fn(ctx, st)
        if self.family == "menu":
            objs = [self._gobj(ctx, viol, f(self._design(c))) for c in self.menu]
            j = int(np.argmin(objs))
            self.objective_trace.append([round(o, 6) for o in objs])
            return self._design(self.menu[j]), j
        sh, o = self._coord_descent(ctx, f, viol, SHARE_GRID, 0.5)          # reset dirseg
        self.objective_trace.append(round(o, 6))
        return np.stack([sh, 1.0 - sh], axis=1), None


# =============================================================================================== methods
def picks(name):
    sel = json.loads((FDIR / "selection.json").read_text())["envs"][name]["families"]
    return {"reset": sel["reset"]["pick"], "menu": sel["menu"]["pick"]}


def ney_name(n):
    return f"FDC-BF[ney-pilot,n={n}]"


def ad_name(fam, n, var):
    return f"PJC-A-{fam}[pilot n={n},{var}]"


def eval_names(name):
    out = []
    for n in PILOTS:
        out.append(ney_name(n))
        for fam in ("reset", "menu"):
            for var in VARIANTS:
                out.append(ad_name(fam, n, var))
    return out


def parse(m):
    if m.startswith("FDC-BF[ney-pilot"):
        return "ney", None, int(m.split("n=")[1].rstrip("]")), None
    fam = m.split("[")[0].split("-")[-1]
    n = int(m.split("n=")[1].split(",")[0])
    var = m.split(",")[-1].rstrip("]")
    return "adapt", fam, n, var


def make(m, seed):
    if m == "FDC-BF":
        return make_variant("FDC-BF")
    if m in GRID:
        return PJCAdapt(**GRID[m], name=m)
    kind, fam, n, var = parse(m)
    succ = pilot_counts(_ENV["name"], seed, _ENV["mu"])[n]
    if kind == "ney":
        return FDCBetPlan(alloc=pilot_neyman(succ, n), name=m)
    base = GRID[_ENV["picks"][fam]]
    return PJCAPilot(base, succ, n, warm=(var == "init+warm"), name=m)


def init_env(name):
    spec = json.loads((FDIR / "envs.json").read_text())[name]
    env, _npil, _nora, _spec = F.build_env(name)
    ctx = ctx_with_stop(build_ctx(env, fr.cr_problems("visit"), spec["eps"]), F.STOP_K)
    J = true_policy_values(ctx.pols, env.w, env.true_mu("visit"))
    Js = np.array([J[ctx.feas[q]].max() for q in range(ctx.Q)])
    _ENV.update(env=env, ctx=ctx, J=J, Js=Js, name=name, mu=np.asarray(env.true_mu("visit"), float),
                picks=picks(name), eps=spec["eps"])


def job(a):
    m, seed = a
    ctx = _ENV["ctx"]
    try:
        meth = make(m, seed)
        t0 = time.perf_counter()
        s, rows = run_stream_v6(_ENV["env"], meth, seed, ctx.problems, ctx.eps, outcome="visit", ctx=ctx,
                                J_true=_ENV["J"], J_star=_ENV["Js"], keep_U=True)
        ck, tau = ctx.checkpoints, int(ctx.tau_R)
        curve = [r["n_cert"] for r in rows]
        nfalse = [r["n_false"] for r in rows]
        k12 = next((k for k, c in enumerate(curve) if c >= 12), None)
        n80 = int(ck[k12]) if (k12 is not None and nfalse[k12] == 0) else tau
        xx = x_rank([r["U"] for r in rows], ck, ctx.eps, 12)
        pn = 0 if m == "FDC-BF" or m in GRID else parse(m)[2]
        out = {"env": _ENV["name"], "method": m, "seed": int(seed), "eps": float(ctx.eps), "posthoc": True,
               "semisynthetic": True, "block": TASK_ID, "pilot_n_per_cell": pn,
               "pilot_reads": int(pn * ctx.S * ctx.A), "N80_pen": n80, "k80": k12,
               "x12": float(xx) if xx is not None else float(tau), "fwer_event": bool(s["fwer_event"]),
               "n_false": int(s["n_false"]), "n_cert": int(s["n_cert"]), "cert_k": s["cert_k"],
               "billing_ok": bool(s["billing_ok"]), "tau_R": tau, "sec": round(time.perf_counter() - t0, 3),
               "error": None}
        if isinstance(meth, PJCAdapt):
            p0 = meth.plans[0]
            out.update({"switched": bool(meth.switched),
                        "ctrl_share_phase0": [round(float(x), 4) for x in p0[:, 0]],
                        "ctrl_share_plans": [[round(float(x), 4) for x in p[:, 0]] for p in meth.plans[1:]],
                        "beta": float(meth.ledger["beta"])})
        return out
    except Exception:  # noqa: BLE001
        return {"env": _ENV["name"], "method": m, "seed": int(seed), "error": traceback.format_exc()}


def pool_map(fn, jobs):
    with get_context("fork").Pool(N_WORKERS) as pool:
        yield from pool.imap_unordered(fn, jobs, chunksize=1)


def jl_load(files):
    key = {}
    for f in files:
        if f.exists():
            for line in f.read_text().splitlines():
                r = json.loads(line)
                key[(r["method"], r["seed"])] = r
    return key


# =============================================================================================== tasks
def run_jobs(name, jobs, f, tag):
    if not jobs:
        print(f"[{tag} {name}] nothing to do", flush=True)
        return
    init_env(name)
    n = errs = 0
    t0 = time.time()
    with open(f, "a") as g:
        for r in pool_map(job, jobs):
            if r.get("error"):
                errs += 1
                with open(OUT / "errors.log", "a") as h:
                    h.write(f"{name} {r['method']} {r['seed']}\n{r['error']}\n")
                continue
            g.write(json.dumps(r) + "\n")
            g.flush()
            n += 1
            if n % 50 == 0:
                print(f"[{tag} {name}] {n}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
                (RES / f"v8_posthoc_F2_{tag}_PROGRESS.json").write_text(json.dumps(
                    {"task_id": f"v8_posthoc_F2_{tag}", "env": name, "step": n, "total_steps": len(jobs),
                     "updated_at": datetime.now().isoformat()}))
    print(f"[{tag} {name}] done {n}, {errs} errors, {time.time() - t0:.0f}s", flush=True)


def run_repro(envs):
    d = OUT / "repro"
    d.mkdir(parents=True, exist_ok=True)
    for name in envs:
        f = d / f"{name}.jsonl"
        done = set(jl_load([f]))
        pk = picks(name)
        jobs = [(m, s) for m in ["FDC-BF", pk["reset"], pk["menu"]] for s in REPRO_SEEDS if (m, s) not in done]
        run_jobs(name, jobs, f, "repro")


def run_eval(envs, shard, nshards):
    d = OUT / "eval"
    d.mkdir(parents=True, exist_ok=True)
    for name in envs:
        f = d / f"{name}.s{shard}.jsonl"
        done = set(jl_load(sorted(d.glob(f"{name}.s*.jsonl"))))
        allj = [(m, s) for m in eval_names(name) for s in EVAL_SEEDS]
        jobs = [j for i, j in enumerate(allj) if i % nshards == shard and j not in done]
        run_jobs(name, jobs, f, f"eval_s{shard}")


# =============================================================================================== analysis
def geo(v):
    return float(np.exp(np.mean(np.log(np.asarray(v, dtype=float)))))


def charged(rows, J, key="N80_pen"):
    return np.array([r[key] + (0.0 if J is None else r["pilot_reads"] / J) for r in rows], float)


def breakeven_J(a_rows, b_rows):
    """Smallest J >= 1 at which the pilot method a (charged pilot_reads / J) ties b (b charged at its own J, same J)
    in paired geomean N80; None if a is slower even uncharged; 1.0 if a is faster at the full charge."""
    def r(J):
        return geo(charged(a_rows, J)) / geo(charged(b_rows, J))
    if r(None) >= 1.0:
        return None
    if r(1) <= 1.0:
        return 1.0
    lo, hi = 1.0, 2.0
    while r(hi) > 1.0 and hi < 1e9:
        lo, hi = hi, hi * 2
    for _ in range(60):
        mid = math.sqrt(lo * hi)
        lo, hi = (mid, hi) if r(mid) > 1.0 else (lo, mid)
    return hi


def analyse():
    from dsswm.stats.v6_analysis import boot_idx, paired
    idx = boot_idx(len(EVAL_SEEDS))
    spec = json.loads((FDIR / "envs.json").read_text())
    summ = {"block": TASK_ID,
            "status": "POST-HOC, DESCRIPTIVE, SEMI-SYNTHETIC -- outside any preregistration lock (v5-v8); not "
                      "confirmatory; outcomes are synthetic",
            "disclosure": (
                "Run on 2026-10-05 in response to external reviewer paper_r4_review S1 / fix 2, after block F's results were known. "
                "Environments, eps, reporting seeds (950-999 + 36000-36199) and the adaptive picks are block F's "
                "(picks tuned on 900-949 without a pilot and NOT re-tuned with one); pilot sizes {0, 250, 1000}, "
                "the pilot construction, the 'init' / 'init+warm' rules and the amortisation grid J in {1, 4, 16, "
                "inf} were fixed before any F2 row was computed.  No real outcome and no evaluation half enters any "
                "statistic."),
            "pilot": ("independent with-replacement draws from the same finite pools (Bernoulli(exact pool mean) per "
                      "cell), nested n in {250, 1000} per cell, drawn per stream (seeded by env and stream seed) and "
                      "shared by every method on that stream; per-instance cost S*A*n = 4,500 / 18,000 reads.  A WoR "
                      "prefix of the replay was NOT used: a plan computed from replay reads is not independent of the "
                      "replay permutation, so the frozen-design theorem would not cover FDC-BF's full-pool estimator "
                      "(valid only as a charged phase with a phase-local estimator, i.e. the PJC-A reset "
                      "construction)."),
            "charging": "N80_charged = N80_pen + S*A*n / J; J = certification instances sharing one pilot "
                        "(J = inf: uncharged, block F's convention).  Same for x12.",
            "equal_information": ("adaptive picks get the same per-stream pilot: 'init' = phase-0 plan is the pilot "
                                  "per-segment Neyman plan (global menu re-centred on it: share logistic(logit p_s + "
                                  "logit c)); 'init+warm' also pools pilot counts into every design-rule plug-in "
                                  "variance.  Certificates use replay data only; ledgers unchanged."),
            "phasewise_TU_rival": "not implemented (not cheap); adaptive variants remain checkpoint-strength",
            "ratio_convention": "A / B paired geomean over 250 streams; for 'FDC-BF / method' > 1 means the method is "
                                "faster than frozen-50/50 FDC-BF",
            "bootstrap": "v6 rule: B = 10^4 resamples of streams, seed 42, two-sided percentile 95% CI",
            "envs": {}}
    for name in ENVS:
        R = jl_load(sorted((OUT / "eval").glob(f"{name}.s*.jsonl")))
        miss = [(m, s) for m in eval_names(name) for s in EVAL_SEEDS if (m, s) not in R]
        if miss:
            print(f"{name}: {len(miss)} missing rows")
            continue
        Fr = F.jl_load(FDIR / "eval" / f"{name}.jsonl", lambda r: (r["method"], r["seed"]))
        pk = picks(name)
        SA = len(spec[name]["pool_sizes"]) * len(spec[name]["pool_sizes"][0])
        Fr = {k: dict(r, pilot_reads=(SA * F.PILOT_N if k[0] == "FDC-BF[ney-pilot]" else 0))   # in-memory copies
              for k, r in Fr.items()}
        rows = {m: [R[(m, s)] for s in EVAL_SEEDS] for m in eval_names(name)}
        rows["FDC-BF"] = [Fr[("FDC-BF", s)] for s in EVAL_SEEDS]
        rows["FDC-BF[ney-pilot,F fixed env pilot n=1000]"] = [Fr[("FDC-BF[ney-pilot]", s)] for s in EVAL_SEEDS]
        for fam in ("reset", "menu"):
            rows[f"PJC-A-{fam}[no pilot]"] = [Fr[(pk[fam], s)] for s in EVAL_SEEDS]
        tau = rows["FDC-BF"][0]["tau_R"]

        def cmp(A, B, J, key="N80_pen", JB="same"):
            a = charged(rows[A], J, key)
            b = charged(rows[B], J if JB == "same" else JB, key)
            p = paired(a, b, idx)
            return {"ratio": p["geomean_ratio"], "ci95": p["ci95_two_sided"]}

        E = {"eps": spec[name]["eps"], "tau_R": tau, "picks_from_F": pk, "methods": {}, "head_to_head": {}}
        for m in rows:
            if m == "FDC-BF":
                continue
            ent = {"false_streams": int(sum(r["fwer_event"] for r in rows[m])),
                   "pilot_reads": int(rows[m][0]["pilot_reads"]),
                   "pilot_over_tau": rows[m][0]["pilot_reads"] / tau,
                   "geomean_N80_over_tau_uncharged": geo(charged(rows[m], None) / tau),
                   "geomean_N80_over_tau_charged_J1": geo(charged(rows[m], 1) / tau)}
            if "switched" in rows[m][0]:
                ent["frac_switched"] = float(np.mean([r["switched"] for r in rows[m]]))
            for J in J_AMORT:
                tag = "inf" if J is None else str(J)
                p = paired(charged(rows["FDC-BF"], None), charged(rows[m], J), idx)
                ent[f"FDCBF_over_method_N80_J{tag}"] = {"ratio": p["geomean_ratio"], "ci95": p["ci95_two_sided"]}
                px = paired(charged(rows["FDC-BF"], None, "x12"), charged(rows[m], J, "x12"), idx)
                ent[f"FDCBF_over_method_x12_J{tag}"] = {"ratio": px["geomean_ratio"], "ci95": px["ci95_two_sided"]}
            be = breakeven_J(rows[m], rows["FDC-BF"]) if rows[m][0]["pilot_reads"] else None
            ent["breakeven_J_vs_FDCBF"] = be
            E["methods"][m] = ent
        # head-to-head: frozen pilot-Neyman vs adaptive, equal information and equal charge
        H = {}
        for n in PILOTS:
            for fam in ("reset", "menu"):
                for var in VARIANTS:
                    a, b = ney_name(n), ad_name(fam, n, var)
                    H[f"{a} / {b}"] = {f"J{'inf' if J is None else J}": cmp(a, b, J) for J in J_AMORT}
                # charged pilot-Neyman vs the adaptive pick that pays nothing (learns online)
                a, b = ney_name(n), f"PJC-A-{fam}[no pilot]"
                H[f"{a} / {b}"] = {f"J{'inf' if J is None else J}": {
                    "ratio": paired(charged(rows[a], J), charged(rows[b], None), idx)["geomean_ratio"],
                    "ci95": paired(charged(rows[a], J), charged(rows[b], None), idx)["ci95_two_sided"]}
                    for J in J_AMORT}
                H[f"{a} / {b}"]["breakeven_J"] = breakeven_J(rows[a], rows[b])
        E["head_to_head"] = H
        # repro check
        rp = jl_load([OUT / "repro" / f"{name}.jsonl"])
        nd = nt = 0
        for (m, s), r in rp.items():
            nt += 1
            o = Fr[(m, s)]
            nd += int(o["N80_pen"] != r["N80_pen"] or o["cert_k"] != r["cert_k"])
        E["repro_vs_F"] = {"n_checked": nt, "n_different": nd, "pass": nt > 0 and nd == 0}
        summ["envs"][name] = E
    summ["answer"] = ANSWER
    summ["written_at"] = datetime.now().isoformat()
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1))
    write_md(summ)
    write_tex(summ)


ANSWER = (
    "Does 'the gain comes from the allocation, not from online adaptation' survive?  Only in a qualified form "
    "(semi-synthetic, post hoc, descriptive).  (1) EQUAL INFORMATION: when the adaptive picks get the same per-stream "
    "pilot (phase-0 pilot-Neyman plan, plus variance warm start) and both sides pay the same pilot, frozen "
    "pilot-Neyman FDC-BF is still faster than every adaptive variant in all four environments and both pilot sizes "
    "(ney-pilot / adaptive 0.90-0.98, every 95% CI below 1; e.g. Criteo-asym n=1000: 0.962 [0.946, 0.978] vs reset, "
    "0.937 [0.924, 0.950] vs menu).  The pilot hardly helps the adaptive members (Criteo-asym reset 1.127 without vs "
    "1.125 with the 1000-pilot): the reset member discards pre-boundary data and the menu member pays its path-union "
    "ledger even when it rarely leaves the pilot plan (menu init+warm switches on 0-19% of streams yet trails frozen "
    "pilot-Neyman by 4-8%).  So, at equal information, the advantage is the frozen allocation plus the absence of "
    "reset / union costs, not online learning.  (2) CHARGED PILOT, adaptive learns online for free: on Criteo "
    "structure (tau_R 7.0M; the pilot is 0.06-0.26% of tau_R) charging changes nothing material -- ney-pilot at "
    "full charge (J = 1) vs the no-pilot adaptive picks 0.89-0.99 (one CI touches 1: Criteo-mixed n=1000 vs reset "
    "0.990 [0.975, 1.005]) and FDC-BF / ney-pilot 1.16 (asym) and 1.01-1.03 (mixed).  On X5 structure (tau_R "
    "99,646; the pilot is 4.5% or 18% of tau_R) the conclusion REVERSES at full charge: ney-pilot is slower than "
    "frozen 50/50 (X5-asym 0.893 [0.879, 0.907] at n=250, 0.632 at n=1000) and slower than the no-pilot adaptive "
    "picks (1.08-1.67 against it); the pilot pays for itself only when shared across J >= 2.1-25.5 certification "
    "instances (break-even vs the adaptive picks), and against frozen 50/50 only at J >= 4.5 (n=250) or 19 "
    "(n=1000) on X5-asym and never on X5-mixed.  Wording for the paper: an informed frozen plan beats these "
    "reset / menu implementations at equal information in the four tested regimes; on small tables this holds "
    "only when the pilot is pre-existing or amortised over many certification runs.  No method had a false "
    "certification stream (0 / 250 per method and environment).  Phase-wise TU joint rival not implemented.")


def fmt(c):
    return f"{c['ratio']:.3f} [{c['ci95'][0]:.3f}, {c['ci95'][1]:.3f}]"


def row_order():
    out = ["FDC-BF[ney-pilot,F fixed env pilot n=1000]", "PJC-A-reset[no pilot]", "PJC-A-menu[no pilot]"]
    for n in PILOTS:
        out.append(ney_name(n))
        for fam in ("reset", "menu"):
            for var in VARIANTS:
                out.append(ad_name(fam, n, var))
    return out


def write_md(s):
    L = ["# v8-posthoc-F2: pilot versus adaptive at equal information and charged pilot cost "
         "(POST-HOC, DESCRIPTIVE, SEMI-SYNTHETIC)", "",
         "**Post-hoc, semi-synthetic disclosure.** " + s["disclosure"], "",
         f"Status: {s['status']}.", "",
         f"- Pilot: {s['pilot']}", f"- Charging: {s['charging']}", f"- Equal information: {s['equal_information']}",
         f"- Phase-wise TU joint rival: {s['phasewise_TU_rival']}.", f"- Ratios: {s['ratio_convention']}.",
         f"- Bootstrap: {s['bootstrap']}.",
         "- Environment names follow block F: X9-* = X5-structure, CR9-* = Criteo-structure.", ""]
    for name, E in s["envs"].items():
        M = E["methods"]
        L += [f"## {name} (SEMI-SYNTHETIC; eps {E['eps']:g}; tau_R {E['tau_R']:,}; 250 streams)", "",
              f"Adaptive picks (block F, tuned without a pilot): reset `{E['picks_from_F']['reset']}`, menu "
              f"`{E['picks_from_F']['menu']}`.  Reproduction of F's pilot-0 rows on seeds 950-959: "
              f"{E['repro_vs_F']['n_checked'] - E['repro_vs_F']['n_different']}/{E['repro_vs_F']['n_checked']} "
              f"identical.", "",
              "FDC-BF (frozen 50/50, no pilot) / method, paired geomean N80 [95% CI]; method charged pilot/J:", "",
              "| method | pilot reads (/tau) | J=1 (full charge) | J=4 | J=16 | uncharged | break-even J | "
              "frac switched | false streams |", "|---|---|---|---|---|---|---|---|---|"]
        for m in row_order():
            v = M[m]
            pr = f"{v['pilot_reads']:,} ({v['pilot_over_tau']:.3f})" if v["pilot_reads"] else "0"
            be = v["breakeven_J_vs_FDCBF"]
            bes = "--" if not v["pilot_reads"] else ("never" if be is None else f"{be:.2f}")
            sw = f"{v['frac_switched']:.2f}" if "frac_switched" in v else "--"
            L.append(f"| `{m}` | {pr} | {fmt(v['FDCBF_over_method_N80_J1'])} | {fmt(v['FDCBF_over_method_N80_J4'])} | "
                     f"{fmt(v['FDCBF_over_method_N80_J16'])} | {fmt(v['FDCBF_over_method_N80_Jinf'])} | {bes} | {sw} | "
                     f"{v['false_streams']} |")
        L += ["", "Head-to-head, frozen pilot-Neyman FDC-BF / adaptive (< 1: frozen pilot-Neyman faster).  'Equal "
              "information' rows charge both sides the same pilot; 'vs no pilot' rows charge only the frozen side:",
              "", "| comparison | J=1 | J=4 | J=16 | uncharged | break-even J |", "|---|---|---|---|---|---|"]
        for k, v in E["head_to_head"].items():
            be = v.get("breakeven_J", "n/a")
            bes = "same charge" if be == "n/a" else ("never" if be is None else f"{be:.2f}")
            L.append(f"| `{k}` | {fmt(v['J1'])} | {fmt(v['J4'])} | {fmt(v['J16'])} | {fmt(v['Jinf'])} | {bes} |")
        L.append("")
    L += ["## Answer", "", s.get("answer", "(see report)"), ""]
    (OUT / "summary.md").write_text("\n".join(L) + "\n")


def write_tex(s):
    envs = list(s["envs"])
    L = ["% v8-posthoc-F2 (POST-HOC, DESCRIPTIVE, SEMI-SYNTHETIC); generated by exp/code/run_v8_posthoc_F2.py",
         "\\begin{table}[t]", "\\centering\\small", "\\setlength{\\tabcolsep}{2.5pt}",
         "\\caption{Semi-synthetic, post-hoc, descriptive: pilot versus adaptive allocation with the pilot charged. "
         "Paired geometric-mean ratio FDC-BF (frozen 50/50, no pilot) / method of $N_{80}$ with 95\\% bootstrap CI "
         "over 250 streams ($>1$: method faster). Pilot reads ($18n$ per instance, $n$ per cell, independent draws "
         "from the same pools) are added to the method's $N_{80}$ in full ($J{=}1$) or amortised over $J{=}16$ "
         "instances. Adaptive PJC-A picks are block F's (tuned without a pilot); with a pilot they start from the "
         "pilot Neyman plan and warm-start their variance plug-ins. No method had a false certification stream.}",
         "\\label{tab:posthocF2}", "\\begin{tabular}{ll" + "c" * len(envs) + "}", "\\toprule",
         "Method & $J$ & " + " & ".join(n.replace("X9", "X5").replace("CR9", "Criteo") for n in envs) + " \\\\",
         "\\midrule"]

    def cell(n, m, J):
        return fmt(s["envs"][n]["methods"][m][f"FDCBF_over_method_N80_J{J}"]).replace("[", "{\\scriptsize[").replace(
            "]", "]}")
    lab = {"PJC-A-reset[no pilot]": "PJC-A reset, no pilot", "PJC-A-menu[no pilot]": "PJC-A menu, no pilot"}
    for m in ("PJC-A-reset[no pilot]", "PJC-A-menu[no pilot]"):
        L.append(f"{lab[m]} & -- & " + " & ".join(cell(n, m, "inf") for n in envs) + " \\\\")
    for n_ in PILOTS:
        L.append("\\midrule")
        for m, lb in ((ney_name(n_), f"FDC-BF pilot-Neyman, $n{{=}}{n_}$"),
                      (ad_name("reset", n_, "init+warm"), f"PJC-A reset + pilot, $n{{=}}{n_}$"),
                      (ad_name("menu", n_, "init+warm"), f"PJC-A menu + pilot, $n{{=}}{n_}$")):
            for J in ("1", "16"):
                L.append(f"{lb} & {J} & " + " & ".join(cell(e, m, J) for e in envs) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (OUT / "table.tex").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["repro", "eval", "analyse"])
    ap.add_argument("--env", nargs="*", default=ENVS)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.task == "repro":
        run_repro(args.env)
    elif args.task == "eval":
        run_eval(args.env, args.shard, args.nshards)
    else:
        analyse()


if __name__ == "__main__":
    main()
