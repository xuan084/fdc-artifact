"""v9 block C preparation: Hillstrom 3-arm layer, DEV HALF ONLY (new file; no locked file, no eval half touched).

Tasks (cwd = exp/code)
  design   guard tables on the hashed DEV half for every candidate segmentation (CN6 / CR6 / CC6 / CZ6) x outcome
           (visit / conversion); pre-declared selection rule (below); writes exp/results/pilots/hillstrom_v9/design.json
  toy      A = 3 near-tie toy FWER (population seeds 900-919 x perm 0-9 = 200 streams, eps 0.02) for every rigorous
           method + an invalid plug-in control (NAIVE-GLR, beta = 1) that shows the toy can detect false certificates
  tune     selected layer, dev seeds 900-949, eps = eps_dev: FDC-BF + every rival grid member
  report   selected layer, dev seeds 950-999, eps_dev and its two grid neighbours: FDC-BF, tuned picks, matched
           rectangles, untuned references
  analyse  rival selection (v5 rule on tune rows) + paired ratios / CIs / false streams on report rows -> summary.json

Pre-declared design rule v1 (written before any certification run on Hillstrom; superseded, kept for the record):
  non-trivial(q, eps)  iff  min over the simple policies {pooled-greedy, mens-to-largest-segments, womens-to-largest-
                            segments} of the true dev regret > eps (r4 rule R4 + the third arm);
  eps_hi(seg, outcome) = largest eps in EPS_GRID with 10 <= #non-trivial <= 13 (None if no grid value qualifies);
  outcome = visit unless no segmentation has eps_hi(visit) (then conversion); segmentation = argmax eps_hi, ties ->
  order CN6, CR6, CC6, CZ6; problems F1 only.
  v1 picked CN6 / visit / F1 / eps 0.003 (results/pilots/hillstrom_v9/design_rule_v1.json).  A dev feasibility look
  (seed 900) showed FDC-BF and RECT-ck-HG both certify 12/15 only at tau_R (FDC-BF 12th U = 0.008 at 0.83 tau_R):
  every comparison would be a terminal pile-up.  Conversion had no qualifying eps except CC6 at 0.001.

Design rule v2 (2026-10-04, dev only, adopted after the v1 feasibility failure; disclosed as a revision):
  candidates = {CN6, CR6, CC6, CZ6} x cost family {F1, F3, F2} (hillstrom_v9.COST_FAMILIES), outcome visit;
  eps_hi as in v1 (same non-trivial definition, same 10-13 window);
  feasible(candidate) iff at eps_hi at least one of the two untuned references {FDC-BF, RECT-ck-HG[bal]} reaches
      12/15 before tau_R (with no false certification) on >= 8 of the 10 dev seeds 900-909;
  family = first family in precedence F1 (equal costs) > F3 (inherited r4 tier C2) > F2 (no precedent) that has a
      feasible candidate;  segmentation = argmax eps_hi within that family, tie -> more non-trivial problems at
      eps_hi, then order CN6, CR6, CC6, CZ6;  eps_dev = its eps_hi.
Design rule B (2026-10-04, authors request after design A's rivals all piled up at tau_R; stated before any
design-B run; dev only):
  candidates = hillstrom_v9.SEGMENTATIONS_B (S = 6: CN6 CR6 CC6 CZ6; S = 4: DN4 DR4 DC4 DZ4; S = 3: C3) x cost family
      {F1, F3, F2}, outcome visit, eps grid EPS_GRID_B = 0.0025 ... 0.03;
  nontriv(eps) as in v1/v2;  c_rect(eps, seed) = number of problems whose running-min U of the untuned reference
      RECT-ck-HG[bal] is <= eps at checkpoint index 18 (the last checkpoint before tau_R), dev seeds 900-909;
  admissible eps: 10 <= nontriv(eps) <= 13 AND median_seed c_rect(eps) >= 10;
  eps_B(candidate) = smallest admissible eps (finest target that rivals can still certify);
  selection = largest S; then family precedence F1 > F3 > F2; then smaller eps_B; then order of SEGMENTATIONS_B.
  Rivals are then re-tuned on dev seeds 900-949 at eps_B with the same grids and v5 rule (task tune_b), and reported
  on 950-999 (task report_b).

Rival selection (lock v5 rule): argmin geomean N80_pen over tune seeds subject to 0 false streams; tie-break geomean
x12; then grid order.  FDC-BF (uniform 1/3 design) is not tuned.
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
OUT = RES / "pilots/hillstrom_v9"
TASK = "hillstrom_v9_dev"
STOP12 = 12            # N80 = first checkpoint with >= 12 of 15 problems certified
DEV = set(range(900, 1000))
N_WORKERS = int(os.environ.get("HV9_WORKERS", "6"))
EPS_GRID_B = (0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02, 0.025, 0.03)
EPS_GRID = (0.001, 0.0015, 0.002, 0.0025, 0.003, 0.004, 0.005, 0.006, 0.0075, 0.01, 0.0125, 0.015, 0.02)

from dsswm.baselines import multiarm_v9 as ma  # noqa: E402
from dsswm.baselines.frontier_common import Method, glr_certificate, plugin_var  # noqa: E402
from dsswm.envs import hillstrom_v9 as hv  # noqa: E402
from dsswm.streams import frontier as fr  # noqa: E402
from dsswm.streams.frontier_runner import SyntheticPoolEnv, build_ctx, true_policy_values  # noqa: E402
from dsswm.streams.frontier_runner_v6 import run_stream_v6  # noqa: E402

# ---------------------------------------------------------------- pre-declared grids (fixed before any tune run)
PLAN_TAGS = ma.PLAN_TAGS_A3
RECT_GRID = [f"RECT-ck-HG[{t}]" for t in PLAN_TAGS]
HC_SCHED = ["prpl,0.5,-", "prpl,0.75,-"] + [f"nstar,0.75,{tf:g}" for tf in (0.1, 0.2, 0.35, 0.5, 0.6, 0.8, 1.0)]
HC_GRID = [f"HC-WoR{{{c}}}[{t}]" for c in HC_SCHED for t in PLAN_TAGS]
PJC_GRID = {}
for _b in ((0,), (2,), (4,), (2, 5, 8, 11)):
    for _rule in ("neyman", "bal"):
        for _rect in (False, True):
            _m = ma.PJCBFA(boundaries=_b, rule=_rule, rect=_rect)
            PJC_GRID[_m.name] = dict(boundaries=_b, rule=_rule, rect=_rect)
_m = ma.PJCBFA(boundaries=(), rule="bal", rect=False)              # no-reset member (external reviewer v8 item 3)
PJC_GRID[_m.name] = dict(boundaries=(), rule="bal", rect=False)
MATCHED = ["RECT-ck-BF", "RECT-ck-BF+box"]
DESCR = ["FDC-BF[0.5]", "FDC-BF[ney]"]
FAMILIES = {"RECT-ck-HG": RECT_GRID, "HC-WoR": HC_GRID, "PJC-local": list(PJC_GRID)}
TUNE_METHODS = ["FDC-BF"] + RECT_GRID + list(PJC_GRID) + MATCHED + HC_GRID
_ENV: dict = {}


# =============================================================================================== methods
class NaiveGLR(Method):
    """INVALID power control: plug-in variance, beta = 1 (no union, no variance event)."""
    alloc_kind = "fixed"
    validity = "none"
    name = "NAIVE-GLR"

    def setup(self, ctx):
        self.alloc_p = ma.uniform_alloc(ctx.S, ctx.A)

    def certify(self, ctx, st):
        return glr_certificate(ctx, st, plugin_var(st), 1.0)


def make(name, S=6, A=3, sigma2=None):
    if name == "FDC-BF":
        return ma.fdc_bf_a()
    if name.startswith("FDC-BF["):
        tag = name[len("FDC-BF["):-1]
        return ma.fdc_bf_a(alloc=ma.plan_matrix(tag, S, A, sigma2), name=name)
    if name in ("RECT-ck-BF", "RECT-ck-BF+box"):
        return ma.RectCkBFA(box=name.endswith("+box"))
    if name.startswith("RECT-ck-HG["):
        return ma.make_plan_rect(name[len("RECT-ck-HG["):-1], S, A, sigma2, name=name)
    if name.startswith("HC-WoR{"):
        cfg, tag = name[len("HC-WoR{"):-1].split("}[")
        sch, c, tf = cfg.split(",")
        return ma.make_plan_hc(sch, float(c), None if tf == "-" else float(tf), tag, S, A, sigma2, name=name)
    if name in PJC_GRID:
        return ma.PJCBFA(**PJC_GRID[name], name=name)
    if name == "NAIVE-GLR":
        return NaiveGLR()
    raise KeyError(name)


def x12(rows, ck, eps, stop_k):
    import run_fdc_bet as rfb
    return rfb.x12(rows, ck, eps, stop_k)


# =============================================================================================== design
def guard(env, outcome, family="F1"):
    probs = hv.hv9_problems(outcome, family)
    mu = env.true_mu(outcome)
    pooled = env.true_pooled_mu(outcome)
    pols = fr.enumerate_policies(env.S, env.A).astype(np.int64)
    J = fr.policy_values(pols, env.w, mu)
    rows = []
    for p in probs:
        cost = fr.policy_costs(pols, env.w, p.kappa)
        feas = cost <= p.budget + fr.FEAS_TOL
        Jf = np.where(feas, J, -np.inf)
        i = int(np.argmax(Jf))
        triv = {"pooled_greedy": fr.trivial_pooled_greedy(env.w, pooled, p), "all_mens": fr.trivial_all_arm(env.w, p, 1),
                "all_womens": fr.trivial_all_arm(env.w, p, 2)}
        reg = {k: float(Jf[i] - fr.policy_values(np.array([v]), env.w, mu)[0]) for k, v in triv.items()}
        srt = np.sort(Jf[np.isfinite(Jf)])[::-1]
        rows.append({"qid": p.qid, "budget": p.budget, "J_star": float(Jf[i]), "pi_star": [int(x) for x in pols[i]],
                     "n_feasible": int(feas.sum()), "second_gap": float(srt[0] - srt[1]) if len(srt) > 1 else None,
                     "regret": reg, "regret_best_trivial": min(reg.values()),
                     "eps_set_size": {str(e): int((Jf >= Jf[i] - e).sum()) for e in EPS_GRID}})
    nt = {str(e): int(sum(r["regret_best_trivial"] > e for r in rows)) for e in EPS_GRID}
    qual = [e for e in EPS_GRID if 10 <= nt[str(e)] <= 13]
    return rows, nt, (max(qual) if qual else None)


def _feasibility(seg, family, eps, seeds=range(900, 910)):
    init_env(seg, "visit", (eps,), family)
    jobs = [(m, sd, eps) for m in ("FDC-BF", "RECT-ck-HG[bal]") for sd in seeds]
    res = list(run_pool(job, jobs))
    out = {}
    for m in ("FDC-BF", "RECT-ck-HG[bal]"):
        rr = [r for r in res if r["method"] == m and not r.get("error")]
        tau = rr[0]["tau_R"]
        out[m] = {"n": len(rr), "before_tau": int(sum(r["N80_pen"] < tau for r in rr)),
                  "false_streams": int(sum(r["fwer_event"] for r in rr)),
                  "geomean_N80_over_tau": geo([r["N80_pen"] / tau for r in rr])}
    out["feasible"] = any(out[m]["before_tau"] >= 8 for m in ("FDC-BF", "RECT-ck-HG[bal]"))
    return out


def task_design():
    OUT.mkdir(parents=True, exist_ok=True)
    out = {"rule": __doc__.split("Design rule v2")[1].split("Rival selection")[0].strip(),
           "split": hv.split_summary(), "eps_grid": list(EPS_GRID), "candidates": {}}
    for seg in hv.SEGMENTATIONS:
        env = hv.HillstromV9Env(seg, "dev", outcomes=("visit", "conversion"))
        c = {"seg_desc": env.seg_desc, "w": env.w.round(5).tolist(), "pool_sizes": env.pool_sizes.tolist(),
             "tau_R": env.tau_R, "dev_cell_means": {o: env.true_mu(o).round(5).tolist() for o in ("visit", "conversion")}}
        for fam in hv.FAMILY_PRECEDENCE:
            for o in ("visit", "conversion"):
                rows, nt, ehi = guard(env, o, fam)
                c[f"{o}|{fam}"] = {"nontrivial_by_eps": nt, "eps_hi": ehi, "problems": rows}
        out["candidates"][seg] = c
        print(seg, {k: v["eps_hi"] for k, v in c.items() if isinstance(v, dict) and "eps_hi" in v}, flush=True)
    feas = {}
    for fam in hv.FAMILY_PRECEDENCE:
        for seg in hv.SEGMENTATIONS:
            ehi = out["candidates"][seg][f"visit|{fam}"]["eps_hi"]
            if ehi is None:
                continue
            feas[f"{seg}|{fam}"] = dict(_feasibility(seg, fam, ehi), eps_hi=ehi)
            print(seg, fam, ehi, feas[f"{seg}|{fam}"], flush=True)
    out["feasibility"] = feas
    chosen = None
    for fam in hv.FAMILY_PRECEDENCE:
        ok = [s for s in hv.SEGMENTATIONS if feas.get(f"{s}|{fam}", {}).get("feasible")]
        if ok:
            def key(s):
                c = out["candidates"][s][f"visit|{fam}"]
                return (c["eps_hi"], c["nontrivial_by_eps"][str(c["eps_hi"])], -hv.SEGMENTATIONS.index(s))
            chosen = (max(ok, key=key), fam)
            break
    if chosen is None:
        raise SystemExit("no feasible Hillstrom design under rule v2")
    seg, fam = chosen
    eps = out["candidates"][seg][f"visit|{fam}"]["eps_hi"]
    out["selected"] = {"segmentation": seg, "family": fam, "kappa": list(hv.COST_FAMILIES[fam]), "outcome": "visit",
                       "eps_dev": eps, "eps_report": list(_neighbours(eps)),
                       "n_nontrivial_at_eps": out["candidates"][seg][f"visit|{fam}"]["nontrivial_by_eps"][str(eps)],
                       "rule": "v2"}
    out["written_at"] = datetime.now().isoformat()
    (OUT / "design.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["selected"]))


def rect_traj_job(a):
    seg, fam, sd = a
    from dsswm.streams.frontier_runner_v6 import ctx_with_stop
    env = _ENV["envs"][seg]
    probs = hv.hv9_problems("visit", fam)
    ctx = ctx_with_stop(build_ctx(env, probs, 1e-9), 15)
    m = make("RECT-ck-HG[bal]", env.S, env.A)
    s, rows = run_stream_v6(env, m, sd, probs, 1e-9, outcome="visit", ctx=ctx, keep_U=True)
    U = np.minimum.accumulate(np.array([r["U"] for r in rows]), 0)
    return seg, fam, sd, U[18].tolist(), len(rows)


def task_design_b():
    out = {"rule": __doc__.split("Design rule B")[1].split("Rival selection")[0].strip(), "eps_grid": list(EPS_GRID_B),
           "candidates": {}}
    _ENV["envs"] = {sg: hv.HillstromV9Env(sg, "dev", outcomes=("visit",)) for sg in hv.SEGMENTATIONS_B}
    jobs = [(sg, f, sd) for sg in hv.SEGMENTATIONS_B for f in hv.FAMILY_PRECEDENCE for sd in range(900, 910)]
    traj = {}
    for sg, f, sd, u18, nr in run_pool(rect_traj_job, jobs):
        assert nr == 20
        traj.setdefault((sg, f), []).append(u18)
    for sg in hv.SEGMENTATIONS_B:
        env = _ENV["envs"][sg]
        for f in hv.FAMILY_PRECEDENCE:
            global EPS_GRID
            save, EPS_GRID = EPS_GRID, EPS_GRID_B
            rows, nt, _ = guard(env, "visit", f)
            EPS_GRID = save
            U = np.array(traj[(sg, f)])                         # (seeds, Q)
            crect = {str(e): float(np.median((U <= e).sum(1))) for e in EPS_GRID_B}
            adm = [e for e in EPS_GRID_B if 10 <= nt[str(e)] <= 13 and crect[str(e)] >= 10]
            out["candidates"][f"{sg}|{f}"] = {"S": env.S, "nontrivial_by_eps": nt, "median_rect_cert_at_k18": crect,
                                              "admissible": adm, "eps_B": min(adm) if adm else None,
                                              "pool_min": int(env.pool_sizes.min())}
            print(sg, f, "S", env.S, "eps_B", out["candidates"][f"{sg}|{f}"]["eps_B"],
                  {e: (nt[str(e)], crect[str(e)]) for e in EPS_GRID_B}, flush=True)
    el = [(k, v) for k, v in out["candidates"].items() if v["eps_B"] is not None]
    if not el:
        out["selected"] = None
        (OUT / "design_b.json").write_text(json.dumps(out, indent=1))
        raise SystemExit("design B: no admissible candidate")
    def key(kv):
        k, v = kv
        sg, f = k.split("|")
        return (-v["S"], hv.FAMILY_PRECEDENCE.index(f), v["eps_B"], hv.SEGMENTATIONS_B.index(sg))
    k, v = min(el, key=key)
    sg, f = k.split("|")
    i = EPS_GRID_B.index(v["eps_B"])
    out["selected"] = {"segmentation": sg, "family": f, "kappa": list(hv.COST_FAMILIES[f]), "outcome": "visit",
                       "eps_dev": v["eps_B"], "eps_report": list(EPS_GRID_B[max(0, i - 1): i + 2]),
                       "n_nontrivial_at_eps": v["nontrivial_by_eps"][str(v["eps_B"])], "rule": "B"}
    out["written_at"] = datetime.now().isoformat()
    (OUT / "design_b.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["selected"]))


def _neighbours(eps):
    i = EPS_GRID.index(eps)
    return EPS_GRID[max(0, i - 1): i + 2]


DESIGN_FILE = "design.json"
SFX = ""                 # "" = design A (rule v2), "_b" = design B


def load_design():
    return json.loads((OUT / DESIGN_FILE).read_text())["selected"]


# =============================================================================================== toy
TOY_EPS = 0.02
TOY_PROBS = [fr.Problem(f"toy3|B{b:.2f}", "visit", (0.0, 1.0, 1.0), b) for b in (0.2, 0.35, 0.5, 0.65, 0.8)]
TOY_METHODS = ["FDC-BF", "RECT-ck-BF", "RECT-ck-BF+box", "RECT-ck-HG[bal]", "RECT-ck-HG[0.5]",
               "HC-WoR{nstar,0.75,0.2}[bal]", "PJC-BF[local,b=2,neyman]", "PJC-BF[local,b=,bal]",
               "PJC-BF[local,b=2,5,8,11,neyman,R]", "NAIVE-GLR"]


def toy3_population(seed):
    """A = 3 near-tie toy: S = 3 segments, pools of 2k-8k records, base rate U(0.15, 0.6), two e-mail arms with
    independent N(0, 0.1) uplifts -> eps = 0.02 certificates are reached before exhaustion and eps-wrong answers
    (regret up to ~0.1) exist in every class (also between the two treatment arms)."""
    rng = np.random.default_rng(seed)
    S = 3
    sizes = rng.integers(2000, 8000, size=(S, 3))
    base = rng.uniform(0.15, 0.6, size=S)
    up = rng.normal(0.0, 0.10, size=(S, 2))
    mu = np.column_stack([base, np.clip(base[:, None] + up, 0.01, 0.99)])
    return SyntheticPoolEnv(sizes, mu, seed=seed, n_min=200)


def toy_job(a):
    name, sd = a
    env = toy3_population(sd)
    mu = env.true_mu("visit")
    out = []
    for p in range(10):
        try:
            m = make(name, env.S, env.A, mu * (1 - mu))
            s, _ = run_stream_v6(env, m, 1000 * sd + p, TOY_PROBS, TOY_EPS, keep_U=False)
            out.append({"method": name, "validity": m.validity, "population_seed": sd, "perm_seed": s["perm_seed"],
                        "fwer_event": bool(s["fwer_event"]), "n_cert": s["n_cert"], "n_false": s["n_false"],
                        "reached_stop": s["reached_stop"], "N80_over_tau": s["N80"] / env.tau_R,
                        "billing_ok": bool(s["billing_ok"]), "sec": s["sec_total"]})
        except Exception:  # noqa: BLE001
            out.append({"method": name, "population_seed": sd, "perm": p, "error": traceback.format_exc()})
    return out


def task_toy(names):
    from scipy import stats
    d = OUT / "toy"
    d.mkdir(parents=True, exist_ok=True)
    rows = [r for ch in run_pool(toy_job, [(n, sd) for n in names for sd in range(900, 920)]) for r in ch]
    errs = [r for r in rows if r.get("error")]
    if errs:
        (d / "errors.log").write_text("\n".join(r["error"] for r in errs[:5]))
    rows = [r for r in rows if not r.get("error")]
    with open(d / "results.jsonl", "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    table = {}
    for n in names:
        rr = [r for r in rows if r["method"] == n]
        if not rr:
            continue
        ev = sum(r["fwer_event"] for r in rr)
        table[n] = {"n_streams": len(rr), "false_streams": ev, "false_certs": sum(r["n_false"] for r in rr),
                    "cp_upper95": 1.0 if ev >= len(rr) else float(stats.beta.ppf(0.95, ev + 1, len(rr) - ev)),
                    "reached_stop": sum(r["reached_stop"] for r in rr),
                    "early_stop_lt_half_tau": sum(r["reached_stop"] and r["N80_over_tau"] < 0.5 for r in rr),
                    "geomean_N80_over_tau": geo([r["N80_over_tau"] for r in rr]),
                    "billing_ok_all": all(r["billing_ok"] for r in rr), "validity": rr[0]["validity"]}
    prev = json.loads((d / "summary.json").read_text()) if (d / "summary.json").exists() else {"table": {}}
    prev["table"].update(table)
    prev.update({"task": "toy3", "written_at": datetime.now().isoformat(), "n_errors": len(errs),
                 "toy": "A = 3 near-tie toy (toy3_population), population seeds 900-919 x perm 0-9, eps 0.02, "
                        "kappa (0, 1, 1), budgets 0.2-0.8"})
    (d / "summary.json").write_text(json.dumps(prev, indent=1))
    print(json.dumps({k: (v["false_streams"], v["n_streams"], round(v["geomean_N80_over_tau"], 4))
                      for k, v in table.items()}))


# =============================================================================================== streams
def init_env(seg, outcome, eps_list, family="F1"):
    env = hv.HillstromV9Env(seg, "dev", outcomes=(outcome,))
    assert env.half == "dev"
    probs = hv.hv9_problems(outcome, family)
    ctx0 = build_ctx(env, probs, eps_list[0])
    mu = env.true_mu(outcome)
    J = true_policy_values(ctx0.pols, env.w, mu)
    Js = np.array([J[ctx0.feas[q]].max() for q in range(ctx0.Q)])
    _ENV.update(env=env, probs=probs, outcome=outcome, J=J, Js=Js, sigma2=mu * (1 - mu),
                ctxs={e: build_ctx(env, probs, e) for e in eps_list})


def job(a):
    name, seed, eps = a
    assert seed in DEV
    ctx = _ENV["ctxs"][eps]
    if _ENV.get("stop_k_run"):
        from dsswm.streams.frontier_runner_v6 import ctx_with_stop
        ctx = ctx_with_stop(ctx, _ENV["stop_k_run"])          # continue the trajectory; N80 still read at 12/15
    env = _ENV["env"]
    try:
        m = make(name, env.S, env.A, _ENV["sigma2"])
        t0 = time.perf_counter()
        s, rows = run_stream_v6(env, m, seed, ctx.problems, ctx.eps, outcome=_ENV["outcome"], ctx=ctx,
                                J_true=_ENV["J"], J_star=_ENV["Js"], keep_U=True)
        ck = ctx.checkpoints
        tau = int(ctx.tau_R)
        curve = [r["n_cert"] for r in rows]
        nfalse = [r["n_false"] for r in rows]
        k12 = next((k for k, c in enumerate(curve) if c >= STOP12), None)
        n80_pen = int(ck[k12]) if (k12 is not None and nfalse[k12] == 0) else tau
        xx = x12(rows, ck, ctx.eps, STOP12)
        return {"method": name, "validity": m.validity, "seed": int(seed), "eps": float(eps), "N80_pen": n80_pen,
                "k80": k12, "x12": xx if xx is not None else float(tau), "fwer_event": bool(s["fwer_event"]),
                "n_false": int(s["n_false"]), "n_cert": int(s["n_cert"]), "cert_k": s["cert_k"],
                "billing_ok": bool(s["billing_ok"]), "schedule_digest": s["schedule_digest"], "tau_R": tau,
                "alloc_kind": m.alloc_kind, "sec": round(time.perf_counter() - t0, 3),
                "sec_cert": s["sec_cert"], "n_ck_visited": len(rows),
                # uncensored width diagnostic: 12th smallest running-min U_q at each visited checkpoint
                "u12_curve": [float(np.sort(np.minimum.accumulate(np.array([r["U"] for r in rows]), 0)[k])
                                        [STOP12 - 1]) for k in range(len(rows))],
                "error": None}
    except Exception:  # noqa: BLE001
        return {"method": name, "seed": int(seed), "eps": float(eps), "error": traceback.format_exc()}


def run_pool(fn, jobs):
    with get_context("fork").Pool(N_WORKERS) as pool:
        for r in pool.imap_unordered(fn, jobs, chunksize=1):
            yield r


def geo(v):
    return float(np.exp(np.mean(np.log(np.asarray(v, dtype=float)))))


def task_streams(task, names, seeds, eps_list):
    des = load_design()
    d = OUT / task
    d.mkdir(parents=True, exist_ok=True)
    rfile = d / "results.jsonl"
    done = set()
    if rfile.exists():
        for l in rfile.read_text().splitlines():
            x = json.loads(l)
            done.add((x["method"], x["seed"], x["eps"]))
    jobs = [(n, s, e) for n in names for e in eps_list for s in seeds if (n, s, e) not in done]
    jobs.sort(key=lambda j: not j[0].startswith(("HC-WoR", "PJC")))
    print(f"[{task}] {len(jobs)} jobs ({des['segmentation']}, {des['outcome']}, eps {eps_list})", flush=True)
    init_env(des["segmentation"], des["outcome"], tuple(eps_list), des["family"])
    if task.startswith("report"):
        _ENV["stop_k_run"] = 15                                # full trajectory (uncensored width diagnostic)
    errs, n_done, t0 = 0, 0, time.time()
    with open(rfile, "a") as f:
        for r in run_pool(job, jobs):
            n_done += 1
            if r.get("error"):
                errs += 1
                with (d / "errors.log").open("a") as g:
                    g.write(f"{r['method']} {r['seed']} {r['eps']}\n{r['error']}\n")
                continue
            f.write(json.dumps(r) + "\n")
            f.flush()
            if n_done % 50 == 0:
                (RES / f"{TASK}_PROGRESS.json").write_text(json.dumps({
                    "task_id": TASK, "phase": task, "step": n_done, "total_steps": len(jobs),
                    "updated_at": datetime.now().isoformat()}))
    print(f"[{task}] done, {errs} errors, wall {(time.time() - t0) / 60:.1f} min", flush=True)


def load_rows(task):
    key = {}
    p = OUT / task / "results.jsonl"
    for l in p.read_text().splitlines():
        r = json.loads(l)
        key[(r["method"], r["seed"], r["eps"])] = r
    return list(key.values())


def pick(rows, names, seeds, eps):
    by = {}
    for r in rows:
        if abs(r["eps"] - eps) < 1e-15:
            by.setdefault(r["method"], {})[r["seed"]] = r
    table = {}
    for n in names:
        if n not in by or set(by[n]) != set(seeds):
            raise SystemExit(f"tuning incomplete for {n}: {len(by.get(n, {}))}/{len(seeds)}")
        rr = [by[n][s] for s in seeds]
        tau = rr[0]["tau_R"]
        table[n] = {"geomean_N80_over_tau": geo([r["N80_pen"] / tau for r in rr]),
                    "geomean_x12_over_tau": geo([r["x12"] / tau for r in rr]),
                    "false_streams": int(sum(r["fwer_event"] for r in rr)),
                    "share_at_tau": float(np.mean([r["N80_pen"] >= tau for r in rr])),
                    "sec_mean": float(np.mean([r["sec"] for r in rr]))}
    ok = [n for n in names if table[n]["false_streams"] == 0]
    best = min(ok, key=lambda n: (table[n]["geomean_N80_over_tau"], table[n]["geomean_x12_over_tau"], names.index(n)))
    return best, table


def task_select():
    des = load_design()
    rows = load_rows("tune" + SFX)
    seeds = list(range(900, 950))
    sel = {}
    for fam, names in FAMILIES.items():
        best, tab = pick(rows, names, seeds, des["eps_dev"])
        sel[fam] = {"pick": best, "table": tab}
    ref = {}
    for m in ["FDC-BF"] + MATCHED:
        _, t = pick(rows, [m], seeds, des["eps_dev"])
        ref[m] = t[m]
    out = {"layer": f"H{des['segmentation']}", "outcome": des["outcome"], "eps": des["eps_dev"], "seeds": [900, 949],
           "rule": "argmin geomean N80_pen subject to 0 false streams; tie-break geomean x12; then grid order",
           "picks": {f: v["pick"] for f, v in sel.items()}, "families": sel, "untuned_reference": ref,
           "written_at": datetime.now().isoformat()}
    (OUT / f"selection{SFX}.json").write_text(json.dumps(out, indent=1))
    for f, v in sel.items():
        t = v["table"][v["pick"]]
        print(f"{f:11s} pick {v['pick']:40s} N80/tau={t['geomean_N80_over_tau']:.4f} x12/tau={t['geomean_x12_over_tau']:.4f}")
    for m, t in ref.items():
        print(f"ref {m:20s} N80/tau={t['geomean_N80_over_tau']:.4f} false={t['false_streams']}")
    return out


def report_methods():
    sel = json.loads((OUT / f"selection{SFX}.json").read_text())["picks"]
    base = ["FDC-BF", sel["RECT-ck-HG"], sel["HC-WoR"], sel["PJC-local"]] + MATCHED + DESCR
    for extra in ("RECT-ck-HG[bal]", "RECT-ck-HG[ney]", "HC-WoR{nstar,0.75,0.2}[bal]", "HC-WoR{nstar,0.75,0.6}[ney]",
                  "PJC-BF[local,b=,bal]", "PJC-BF[local,b=2,neyman]"):
        if extra not in base:
            base.append(extra)
    return base


def boot_ratio(lf, lr, B=10_000, seed=42):
    d = np.log(lf) - np.log(lr)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(B, len(d)))
    m = d[idx].mean(1)
    return {"ratio": float(np.exp(d.mean())), "lo95": float(np.exp(np.quantile(m, 0.025))),
            "hi95": float(np.exp(np.quantile(m, 0.975))), "ub95_one_sided": float(np.exp(np.quantile(m, 0.95))),
            "faster_streams": int((d < 0).sum()), "slower_streams": int((d > 0).sum()), "n": int(len(d))}


K_DIAG = 18          # checkpoint index 18 = last checkpoint before tau_R (K = 20)


def _u12(r, k):
    c = r.get("u12_curve") or []
    return c[k] if k < len(c) else np.nan


def task_analyse():
    from scipy import stats
    des = load_design()
    sel = json.loads((OUT / f"selection{SFX}.json").read_text())
    rows = load_rows("report" + SFX)
    seeds = list(range(950, 1000))
    names = report_methods()
    out = {"layer": f"H{des['segmentation']}", "outcome": des["outcome"], "eps_dev": des["eps_dev"],
           "seeds": [950, 999], "picks": sel["picks"], "by_eps": {}}
    for eps in des["eps_report"]:
        by = {}
        for r in rows:
            if abs(r["eps"] - eps) < 1e-15:
                by.setdefault(r["method"], {})[r["seed"]] = r
        res = {"methods": {}, "ratios_FDC_BF_over": {}}
        for n in names:
            if n not in by or set(by[n]) != set(seeds):
                continue
            rr = [by[n][s] for s in seeds]
            tau = rr[0]["tau_R"]
            ev = int(sum(r["fwer_event"] for r in rr))
            res["methods"][n] = {"geomean_N80_over_tau": geo([r["N80_pen"] / tau for r in rr]),
                                 "geomean_x12_over_tau": geo([r["x12"] / tau for r in rr]),
                                 "share_at_tau": float(np.mean([r["N80_pen"] >= tau for r in rr])),
                                 "false_streams": ev, "false_certs": int(sum(r["n_false"] for r in rr)),
                                 "cp_upper95": float(stats.beta.ppf(0.95, ev + 1, len(rr) - ev)) if ev < len(rr) else 1.0,
                                 "sec_mean": float(np.mean([r["sec"] for r in rr])),
                                 "sec_cert_per_ck_max": float(max(r["sec_cert"] / max(r["n_ck_visited"], 1)
                                                                  for r in rr))}
        if "FDC-BF" in res["methods"]:
            lf = np.array([by["FDC-BF"][s]["N80_pen"] for s in seeds], float)
            for n in res["methods"]:
                if n == "FDC-BF":
                    continue
                lr = np.array([by[n][s]["N80_pen"] for s in seeds], float)
                res["ratios_FDC_BF_over"][n] = boot_ratio(lf, lr)
                kk = K_DIAG
                uf = np.array([_u12(by["FDC-BF"][s], kk) for s in seeds])
                ur = np.array([_u12(by[n][s], kk) for s in seeds])
                ok = np.isfinite(uf) & np.isfinite(ur) & (uf > 0) & (ur > 0)
                if ok.sum() >= 10:
                    res.setdefault("u12_ratio_FDC_BF_over", {})[n] = dict(boot_ratio(uf[ok], ur[ok]), checkpoint_index=kk)
        out["by_eps"][str(eps)] = res
    out["written_at"] = datetime.now().isoformat()
    (OUT / f"summary{SFX}.json").write_text(json.dumps(out, indent=1))
    for e, res in out["by_eps"].items():
        print(f"--- eps {e}")
        for n, m in res["methods"].items():
            r = res["ratios_FDC_BF_over"].get(n)
            rs = f"FDC-BF/r={r['ratio']:.3f} [{r['lo95']:.3f},{r['hi95']:.3f}] UB={r['ub95_one_sided']:.3f}" if r else ""
            print(f"{n:40s} N80/tau={m['geomean_N80_over_tau']:.4f} at_tau={m['share_at_tau']:.2f} "
                  f"false={m['false_streams']} {rs}")


def main():
    global SFX, DESIGN_FILE
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True,
                    choices=["design", "design_b", "toy", "tune", "select", "report", "analyse"])
    ap.add_argument("--design", choices=["a", "b"], default="a")
    ap.add_argument("--methods", default=None)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.design == "b":
        SFX, DESIGN_FILE = "_b", "design_b.json"
    names = a.methods.split(";") if a.methods else None
    if a.task == "design":
        task_design()
    elif a.task == "design_b":
        task_design_b()
    elif a.task == "toy":
        task_toy(names or TOY_METHODS)
    elif a.task == "tune":
        des = load_design()
        task_streams("tune" + SFX, names or TUNE_METHODS, list(range(900, 950)), [des["eps_dev"]])
    elif a.task == "select":
        task_select()
    elif a.task == "report":
        des = load_design()
        task_streams("report" + SFX, names or report_methods(), list(range(950, 1000)), list(des["eps_report"]))
    else:
        task_analyse()


if __name__ == "__main__":
    main()
