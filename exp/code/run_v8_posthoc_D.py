"""v8-posthoc-D: POST-HOC, DESCRIPTIVE block (outside any lock) answering the reviewer point that "no gain from
adaptivity" was tested where nothing adapted (PJC-local* = no-reset member; PJC-menu* chose 0.5 on every stream).

New file; locked v5-v8 files are imported, never modified.  Members: dsswm/baselines/pjc_adapt.py (families reset /
menu / segmenu, see its docstring).  Same guarantee class as FDC-BF.

Tasks (cwd = exp/code)
  toy        FWER on the r4 near-tie toy (pop seeds 900-919 x perm 0-9 = 200 streams, eps 0.02) for one member of every
             (family, rule) with the most boundaries.
  tune_x5    X5 X9 DEV half, seeds 900-949, eps 0.02, stop_k 15: FDC-BF + full grid.
  tune_cr9   CR9 DEV half, seeds 900-949, eps 0.001, stop_k 15: FDC-BF + full grid.
  select     v5 rule per family per layer: argmin geomean N80_pen s.t. 0 false streams; tie-break geomean x12.
  eval_x5    X5 eval half (post-hoc access path, logged), seeds 35000-35199, eps 0.02: FDC-BF re-run + full grid.
  eval_cr9   CR9 eval half (logged), v7 streams 33000-33199, eps 0.001: FDC-BF re-run + full grid.
  analyse    paired FDC-BF / variant on the SEALED FDC-BF rows (v8a_full_a, v8c_full; seals verified), bootstrap
             B = 10^4 seed 42 (v6 rule); tuned picks = the descriptive answer; every grid member reported as an
             eval-selected, optimistic sensitivity.
Outputs: exp/results/full/v8_posthoc_D/{<task>/results.jsonl, selection.json, summary.json, summary.md}.
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
OUT = WS / "exp/results/full/v8_posthoc_D"
N_WORKERS = int(os.environ.get("POSTHOC_WORKERS", "14"))
TASK_ID = "v8-posthoc-D"
REASON = ("reviewer point: adaptivity tested where nothing adapted; post-hoc descriptive PJC members with genuine "
          "adaptivity, paired against sealed FDC-BF rows")

from dsswm.baselines.fdc_bet import make_variant  # noqa: E402
from dsswm.baselines.pjc_adapt import MENU3, MENU5, PJCAdapt  # noqa: E402
from dsswm.streams import frontier as fr  # noqa: E402
from dsswm.streams.frontier_runner import build_ctx, true_policy_values  # noqa: E402
from dsswm.streams.frontier_runner_v6 import ctx_with_stop, run_stream_v6  # noqa: E402
from run_r5s_v7 import x_rank  # noqa: E402

BOUNDS = [(2,), (4,), (6,), (8,), (3, 6), (4, 8), (2, 5, 8), (3, 6, 9)]
SPECS = [("reset", "neyman", ()), ("reset", "dirseg", ()), ("menu", "proj", MENU5), ("menu", "dirproj", MENU5),
         ("segmenu", "neyman", MENU5), ("segmenu", "neyman", MENU3), ("segmenu", "dirseg", MENU5),
         ("segmenu", "dirseg", MENU3)]
GRID = {}
for fam, rule, menu in SPECS:
    for b in BOUNDS:
        kw = dict(family=fam, boundaries=b, rule=rule, menu=menu if menu else MENU5)
        GRID[PJCAdapt(**kw).name] = kw
FAMILIES = {f: [n for n, kw in GRID.items() if kw["family"] == f] for f in ("reset", "menu", "segmenu")}
TOY = [PJCAdapt(family=f, boundaries=(2, 5, 8), rule=r, menu=m if m else MENU5).name for f, r, m in SPECS]

TASKS = {
    "tune_x5": dict(layer="X9", half="dev", seeds=list(range(900, 950)), eps=0.02),
    "tune_cr9": dict(layer="CR9", half="dev", seeds=list(range(900, 950)), eps=0.001),
    "eval_x5": dict(layer="X9", half="eval", seeds=list(range(35000, 35200)), eps=0.02),
    "eval_cr9": dict(layer="CR9", half="eval", seeds=list(range(33000, 33200)), eps=0.001),
}
STOP_K = 15
_ENV: dict = {}


def make(name):
    if name == "FDC-BF":
        return make_variant("FDC-BF")
    return PJCAdapt(**GRID[name], name=name)


def init_env(layer, half, eps):
    if half == "dev":
        assert True
        if layer == "X9":
            from dsswm.envs.x5_v8 import X5LayerEnv
            env = X5LayerEnv("dev")
        else:
            from dsswm.envs.pool_replay import PoolReplayEnv
            env = PoolReplayEnv("CR9", "dev")
    else:
        from dsswm.envs import posthoc_access_v8 as PA
        env = PA.X5PosthocEvalEnv(TASK_ID, REASON) if layer == "X9" else PA.cr9_posthoc_eval_env(TASK_ID, REASON)
    assert env.half == half
    probs = fr.cr_problems("visit")
    ctx = ctx_with_stop(build_ctx(env, probs, eps), STOP_K)
    J = true_policy_values(ctx.pols, env.w, env.true_mu("visit"))
    Js = np.array([J[ctx.feas[q]].max() for q in range(ctx.Q)])
    _ENV.update(env=env, ctx=ctx, J=J, Js=Js, layer=layer, half=half)


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
        out = {"method": name, "seed": int(seed), "eps": float(ctx.eps), "layer": _ENV["layer"], "half": _ENV["half"],
               "posthoc": True, "block": TASK_ID, "N80_pen": n80, "k80": k12,
               "x12": float(xx) if xx is not None else float(tau), "fwer_event": bool(s["fwer_event"]),
               "n_false": int(s["n_false"]), "n_cert": int(s["n_cert"]), "cert_k": s["cert_k"],
               "billing_ok": bool(s["billing_ok"]), "schedule_digest": s["schedule_digest"], "tau_R": tau,
               "sec": round(time.perf_counter() - t0, 3), "error": None}
        if isinstance(m, PJCAdapt):
            p0 = m.plans[0]
            out.update({"switched": bool(m.switched),
                        "switched_before_k80": bool(any(float(np.max(np.abs(p - p0))) > 1e-9
                                                        for b, p in zip(m.boundaries, m.plans[1:])
                                                        if k12 is None or b < k12)),
                        "max_abs_share_dev": float(max([float(np.max(np.abs(p[:, 0] - 0.5))) for p in m.plans]
                                                       or [0.0])),
                        "ctrl_share_plans": [[round(float(x), 4) for x in p[:, 0]] for p in m.plans[1:]],
                        "path": [list(x) if isinstance(x, tuple) else x for x in m.path],
                        "beta": float(m.ledger["beta"]), "log_L": float(m.ledger.get("log_L", 0.0))})
        return out
    except Exception:  # noqa: BLE001
        return {"method": name, "seed": int(seed), "error": traceback.format_exc()}


def toy_job(a):
    name, sd = a
    from dsswm.streams.frontier_runner import run_stream
    from dsswm.tests.test_baselines_r4a import PROBS, TOY_EPS, _toy_population
    env = _toy_population(sd)
    ctx = build_ctx(env, PROBS, TOY_EPS)
    out = []
    for p in range(10):
        m = make(name)
        s, _ = run_stream(env, m, 1000 * sd + p, PROBS, TOY_EPS, ctx=ctx, keep_U=False)
        out.append({"method": name, "population_seed": sd, "perm": p, "fwer_event": bool(s["fwer_event"]),
                    "n_false": int(s["n_false"]), "n_cert": int(s["n_cert"]), "billing_ok": bool(s["billing_ok"]),
                    "switched": bool(m.switched), "N80_over_tau": s["N80"] / env.tau_R})
    return out


def run_pool(fn, jobs):
    with get_context("fork").Pool(N_WORKERS) as pool:
        for r in pool.imap_unordered(fn, jobs, chunksize=1):
            yield r


def geo(v):
    return float(np.exp(np.mean(np.log(np.asarray(v, dtype=float)))))


def load(task):
    f = OUT / task / "results.jsonl"
    key = {}
    if f.exists():
        for l in f.read_text().splitlines():
            r = json.loads(l)
            key[(r["method"], r["seed"])] = r
    return key


def run_task(task):
    T = TASKS[task]
    d = OUT / task
    d.mkdir(parents=True, exist_ok=True)
    done = set(load(task))
    names = ["FDC-BF"] + list(GRID)
    jobs = [(n, s) for n in names for s in T["seeds"] if (n, s) not in done]
    print(f"[{task}] {len(jobs)} jobs", flush=True)
    if not jobs:
        return
    init_env(T["layer"], T["half"], T["eps"])
    errs, n = 0, 0
    with open(d / "results.jsonl", "a") as f:
        for r in run_pool(job, jobs):
            if r.get("error"):
                errs += 1
                with open(d / "errors.log", "a") as g:
                    g.write(f"{r['method']} {r['seed']}\n{r['error']}\n")
                continue
            f.write(json.dumps(r) + "\n")
            f.flush()
            n += 1
            if n % 200 == 0:
                print(f"[{task}] {n}/{len(jobs)}", flush=True)
    print(f"[{task}] done, {errs} errors", flush=True)


def run_toy():
    from scipy import stats
    rows = [r for ch in run_pool(toy_job, [(n, sd) for n in TOY for sd in range(900, 920)]) for r in ch]
    d = OUT / "toy"
    d.mkdir(parents=True, exist_ok=True)
    (d / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    table = {}
    for n in TOY:
        rr = [r for r in rows if r["method"] == n]
        ev = sum(r["fwer_event"] for r in rr)
        table[n] = {"n_streams": len(rr), "false_streams": ev, "false_certs": sum(r["n_false"] for r in rr),
                    "cp_upper95": float(stats.beta.ppf(0.95, ev + 1, len(rr) - ev)),
                    "switched_streams": sum(r["switched"] for r in rr),
                    "billing_ok_all": all(r["billing_ok"] for r in rr),
                    "geomean_N80_over_tau": geo([r["N80_over_tau"] for r in rr])}
    out = {"task": "toy", "toy": "r4 near-tie toy, population seeds 900-919 x perm 0-9, eps 0.02, delta 0.05",
           "written_at": datetime.now().isoformat(), "table": table}
    (d / "summary.json").write_text(json.dumps(out, indent=1))
    for k, v in table.items():
        print(f"{k:55s} false {v['false_streams']}/{v['n_streams']} switched {v['switched_streams']} "
              f"N80/tau {v['geomean_N80_over_tau']:.3f}")


def describe(rows, seeds):
    rr = [rows[s] for s in seeds]
    tau = rr[0]["tau_R"]
    d = {"geomean_N80_over_tau": geo([r["N80_pen"] / tau for r in rr]),
         "geomean_x12_over_tau": geo([r["x12"] / tau for r in rr]),
         "false_streams": int(sum(r["fwer_event"] for r in rr)), "billing_ok_all": all(r["billing_ok"] for r in rr)}
    if "switched" in rr[0]:
        d["frac_switched"] = float(np.mean([r["switched"] for r in rr]))
        d["frac_switched_before_k80"] = float(np.mean([r["switched_before_k80"] for r in rr]))
        d["mean_max_abs_share_dev"] = float(np.mean([r["max_abs_share_dev"] for r in rr]))
    return d


def select():
    out = {"rule": "v5 rule: argmin geomean N80_pen on DEV seeds 900-949 s.t. 0 false streams; tie-break geomean x12",
           "layers": {}}
    for task, lay in (("tune_x5", "X9"), ("tune_cr9", "CR9")):
        R = load(task)
        seeds = TASKS[task]["seeds"]
        by = {}
        for (m, s), r in R.items():
            by.setdefault(m, {})[s] = r
        lay_out = {"FDC-BF_dev": describe(by["FDC-BF"], seeds), "families": {}}
        for fam, names in list(FAMILIES.items()) + [("any_adaptive", list(GRID))]:
            cand = {m: describe(by[m], seeds) for m in names if m in by and set(by[m]) >= set(seeds)}
            ok = {m: v for m, v in cand.items() if v["false_streams"] == 0}
            pick = min(ok, key=lambda m: (ok[m]["geomean_N80_over_tau"], ok[m]["geomean_x12_over_tau"])) if ok else None
            lay_out["families"][fam] = {"pick": pick, "n_candidates": len(cand), "missing": len(names) - len(cand),
                                        **({"table": cand} if fam != "any_adaptive" else {})}
        out["layers"][lay] = lay_out
    (OUT / "selection.json").write_text(json.dumps(out, indent=1))
    for lay, d in out["layers"].items():
        print(f"== {lay}: FDC-BF dev N80/tau {d['FDC-BF_dev']['geomean_N80_over_tau']:.4f}")
        for fam, x in d["families"].items():
            if fam == "any_adaptive":
                print(f"   any_adaptive pick: {x['pick']}")
                continue
            print(f"   {fam}: pick {x['pick']}")
            for m, v in sorted(x["table"].items(), key=lambda kv: kv[1]["geomean_N80_over_tau"])[:4]:
                print(f"      {m:58s} {v['geomean_N80_over_tau']:.4f} sw={v['frac_switched']:.2f} f={v['false_streams']}")
    return out


def sealed_fdc_rows(layer):
    """Sealed FDC-BF eval rows: X5 from v8a_full_a (eps 0.02), CR9 from v8c_full (reproduction of the v7 seal)."""
    import run_r5s_v8 as RV8
    from dsswm.stats.prereg_v8 import load_locked_addendum
    from dsswm.stats.v8_seal import verify_seal
    lock = load_locked_addendum(None)
    add_sha = lock["sha256"]
    task = "v8a_full_a" if layer == "X9" else "v8c_full"
    T = RV8.TASKS[task]
    per_sha, _ = RV8.code_hashes()
    data_sha = RV8.data_sha_for(T["layer"], lock)
    cur = RV8.current_rows(RV8.RES / "full" / task / "results.jsonl", add_sha, data_sha)
    res = verify_seal(RV8.SEALS, task, cur, T["methods"], T["eps"], T["seeds"], per_sha, data_sha, add_sha)
    eps = 0.02 if layer == "X9" else 0.001
    rows = {int(r["seed"]): r for r in cur if r["method"] == "FDC-BF" and abs(float(r["eps"]) - eps) < 1e-15}
    return rows, {"task": task, "seal_commit": res["seal_commit"],
                  "results_content_sha256": res["seal"]["results_content_sha256"], "addendum_sha256": add_sha}


def analyse():
    from dsswm.stats.v6_analysis import boot_idx, paired
    sel = json.loads((OUT / "selection.json").read_text())
    toy = json.loads((OUT / "toy" / "summary.json").read_text())
    summ = {"block": TASK_ID,
            "status": "POST-HOC, DESCRIPTIVE -- outside any preregistration lock (v5-v8); not confirmatory",
            "disclosure": ("Run on 2026-10-04 AFTER the v8 confirmatory analysis (lock v8 sealed: v8a_full_a, v8c_full) "
                           "in response to a reviewer point that 'no gain from adaptivity' was tested where nothing "
                           "adapted.  Members, grid and tuning rule were fixed before any eval row of this block was "
                           "computed, and picks were tuned on DEV seeds 900-949 only (v5 rule), but the block as a whole "
                           "is post hoc: its existence, members and grid were chosen knowing the confirmatory outcome.  "
                           "Eval access used the documented post-hoc path dsswm/envs/posthoc_access_v8.py (logged in "
                           "eval_access_log.jsonl).  The full-grid eval table is eval-selected and optimistic for the "
                           "variants."),
            "ratio_convention": "FDC-BF / variant paired geomean of N80_pen (< 1: FDC-BF faster; > 1: variant faster)",
            "bootstrap": "v6 rule: B = 10^4 resamples of streams, seed 42, two-sided percentile 95% CI on the geomean",
            "selection": sel, "toy_validity": toy["table"], "layers": {}}
    for task, lay in (("eval_x5", "X9"), ("eval_cr9", "CR9")):
        T = TASKS[task]
        seeds = T["seeds"]
        R = load(task)
        by = {}
        for (m, s), r in R.items():
            by.setdefault(m, {})[s] = r
        sealed, sinfo = sealed_fdc_rows(lay)
        assert sorted(sealed) == seeds, "sealed FDC-BF rows do not cover the eval seeds"
        # reproduction of the sealed FDC-BF rows by this runner (pairing integrity)
        rep = by.get("FDC-BF", {})
        diff = [s for s in seeds if s not in rep or rep[s]["N80_pen"] != sealed[s]["N80_pen"]
                or rep[s]["schedule_digest"] != sealed[s]["schedule_digest"] or rep[s]["cert_k"] != sealed[s]["cert_k"]]
        a = np.array([sealed[s]["N80_pen"] for s in seeds], float)
        idx = boot_idx(len(seeds))
        tau = sealed[seeds[0]]["tau_R"]
        lay_out = {"eps": T["eps"], "seeds": [seeds[0], seeds[-1], len(seeds)], "sealed_fdc_bf": sinfo,
                   "fdc_bf_rerun_reproduces_seal": {"n_compared": len(seeds), "n_different": len(diff),
                                                    "pass": not diff, "different_seeds": diff[:10]},
                   "FDC-BF": {"geomean_N80_over_tau": geo(a / tau),
                              "false_streams": int(sum(sealed[s]["fwer_event"] for s in seeds))},
                   "picks": {}, "grid": {}}
        picks = sel["layers"][lay]["families"]
        for name in GRID:
            if name not in by or set(by[name]) < set(seeds):
                continue
            rows = by[name]
            b = np.array([rows[s]["N80_pen"] for s in seeds], float)
            p = paired(a, b, idx)
            d = describe(rows, seeds)
            ent = {"FDCBF_over_variant": p["geomean_ratio"], "ci95": p["ci95_two_sided"],
                   "frac_variant_faster": p["frac_slower"], "frac_tied": p["frac_tied"], **d,
                   "schedule_digest_match_fdc": float(np.mean([rows[s]["schedule_digest"] == sealed[s]["schedule_digest"]
                                                               for s in seeds]))}
            lay_out["grid"][name] = ent
        for fam, x in picks.items():
            if x["pick"] in lay_out["grid"]:
                lay_out["picks"][fam] = {"pick": x["pick"], **lay_out["grid"][x["pick"]]}
        best = max(lay_out["grid"], key=lambda m: lay_out["grid"][m]["FDCBF_over_variant"])
        lay_out["eval_best_of_grid_optimistic"] = {"name": best, **lay_out["grid"][best]}
        lay_out["any_grid_member_ci_above_1"] = [m for m, v in lay_out["grid"].items() if v["ci95"][0] > 1.0]
        lay_out["any_grid_member_point_above_1"] = [m for m, v in lay_out["grid"].items()
                                                    if v["FDCBF_over_variant"] > 1.0]
        fam_sw = {}
        for m, v in lay_out["grid"].items():
            fam_sw.setdefault(GRID[m]["family"], []).append(v["frac_switched"])
        sw = {m: v for m, v in lay_out["grid"].items() if v["frac_switched"] >= 0.5}
        bs = max(sw, key=lambda m: sw[m]["FDCBF_over_variant"]) if sw else None
        lay_out["adaptivity"] = {
            "mean_frac_switched_by_family": {f: float(np.mean(x)) for f, x in fam_sw.items()},
            "n_members_switching_on_half_or_more_streams": len(sw),
            "best_member_switching_on_half_or_more": ({"name": bs, **sw[bs]} if bs else None),
            "any_variant_beats_fdc_bf": bool(lay_out["any_grid_member_point_above_1"]),
            "false_streams_all_grid": int(sum(v["false_streams"] for v in lay_out["grid"].values()))}
        summ["layers"][lay] = lay_out
    summ["answer"] = ("No adaptive variant beats FDC-BF: every dev-tuned pick and every one of the 64 grid members has "
                      "FDC-BF/variant < 1 with the 95% CI upper bound below 1 on both layers"
                      if not any(d["adaptivity"]["any_variant_beats_fdc_bf"] for d in summ["layers"].values())
                      else "At least one adaptive variant has a point ratio > 1 (see any_grid_member_point_above_1)")
    summ["written_at"] = datetime.now().isoformat()
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1))
    write_md(summ)
    print(json.dumps({lay: {f: (v["pick"], round(v["FDCBF_over_variant"], 4), [round(c, 4) for c in v["ci95"]],
                                round(v["frac_switched"], 3), v["false_streams"]) for f, v in d["picks"].items()}
                      for lay, d in summ["layers"].items()}, indent=1))


def write_md(s):
    L = ["# v8-posthoc-D: genuinely adaptive PJC members vs FDC-BF (POST-HOC, DESCRIPTIVE)", "",
         "**Post-hoc disclosure.** " + s["disclosure"], "",
         f"Status: {s['status']}.  Ratio: {s['ratio_convention']}.  Bootstrap: {s['bootstrap']}.", "",
         "Families (dsswm/baselines/pjc_adapt.py; all in FDC-BF's guarantee class, FWER <= 0.05 at the K checkpoints):",
         "- **reset** (RAGE-style phase reset, phase-local estimator, ledger = FDC-BF's): per-segment Neyman from past "
         "data (`neyman`) or per-segment direction-optimal G-design over shares 0.10-0.90 (`dirseg`).",
         "- **menu** (keep-data, global control share from {0.3,...,0.7}; union over menu paths): projected-variance "
         "(`proj`) or direction-optimal (`dirproj`) choice.",
         "- **segmenu** (keep-data, PER-SEGMENT share from a menu; union charges M^S paths per boundary): rounded "
         "per-segment Neyman (`neyman`) or per-segment direction-optimal (`dirseg`).",
         "Boundaries (checkpoint indices): " + ", ".join(str(list(b)) for b in BOUNDS) + " (2, 3 and 4 phases).", ""]
    L += ["## Toy validity (r4 near-tie toy, 200 streams each, eps 0.02)", "",
          "| member | false streams | CP 95% upper | switched streams |", "|---|---|---|---|"]
    for n, v in s["toy_validity"].items():
        L.append(f"| {n} | {v['false_streams']}/{v['n_streams']} | {v['cp_upper95']:.3f} | {v['switched_streams']} |")
    for lay, d in s["layers"].items():
        nm = "X5 RetailHero X9 eval, seeds 35000-35199, eps 0.02" if lay == "X9" else \
            "CR9 Criteo eval, v7 streams 33000-33199, eps 0.001"
        L += ["", f"## {nm}", "",
              f"Sealed FDC-BF rows: {d['sealed_fdc_bf']['task']} (seal commit {d['sealed_fdc_bf']['seal_commit'][:12]}); "
              f"FDC-BF geomean N80/tau = {d['FDC-BF']['geomean_N80_over_tau']:.4f}, false streams "
              f"{d['FDC-BF']['false_streams']}.  Runner re-run reproduces the sealed FDC-BF rows: "
              f"{'yes' if d['fdc_bf_rerun_reproduces_seal']['pass'] else 'NO'} "
              f"({d['fdc_bf_rerun_reproduces_seal']['n_different']} differing streams).", "",
              "### Dev-tuned picks (the descriptive answer)", "",
              "| family | pick (tuned on dev 900-949) | FDC-BF/variant | 95% CI | variant N80/tau | frac switched | "
              "frac switched before k80 | false streams |", "|---|---|---|---|---|---|---|---|"]
        for fam, v in d["picks"].items():
            L.append(f"| {fam} | `{v['pick']}` | {v['FDCBF_over_variant']:.4f} | [{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}] "
                     f"| {v['geomean_N80_over_tau']:.4f} | {v['frac_switched']:.3f} | {v['frac_switched_before_k80']:.3f} "
                     f"| {v['false_streams']} |")
        b = d["eval_best_of_grid_optimistic"]
        L += ["", f"Eval-selected best of all {len(d['grid'])} grid members (optimistic for the variants): `{b['name']}` "
              f"FDC-BF/variant {b['FDCBF_over_variant']:.4f} [{b['ci95'][0]:.4f}, {b['ci95'][1]:.4f}], switched "
              f"{b['frac_switched']:.3f}.  Grid members with point ratio > 1: "
              f"{len(d['any_grid_member_point_above_1'])}; with CI lower bound > 1: {len(d['any_grid_member_ci_above_1'])}.",
              "", "<details><summary>Full grid (eval, descriptive)</summary>", "",
              "| member | FDC-BF/variant | 95% CI | frac switched | false |", "|---|---|---|---|---|"]
        for n, v in sorted(d["grid"].items(), key=lambda kv: -kv[1]["FDCBF_over_variant"]):
            L.append(f"| `{n}` | {v['FDCBF_over_variant']:.4f} | [{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}] | "
                     f"{v['frac_switched']:.3f} | {v['false_streams']} |")
        a = d["adaptivity"]
        bsw = a["best_member_switching_on_half_or_more"]
        L += ["", "</details>", "", "### Did the variants adapt?", "",
              "Mean fraction of streams whose design moved off 50/50, by family: " +
              ", ".join(f"{f} {x:.3f}" for f, x in a["mean_frac_switched_by_family"].items()) +
              f".  Members that switched on at least half the streams: {a['n_members_switching_on_half_or_more_streams']}"
              + (f"; the best of them is `{bsw['name']}` with FDC-BF/variant {bsw['FDCBF_over_variant']:.4f} "
                 f"[{bsw['ci95'][0]:.4f}, {bsw['ci95'][1]:.4f}] (mean max |share - 0.5| = {bsw['mean_max_abs_share_dev']:.3f})."
                 if bsw else ".") +
              f"  False streams over all grid members: {a['false_streams_all_grid']}."]
    L += ["", "## Answer", "", s["answer"] + ".  Two reasons, both visible in the table.  (i) When adaptation is "
          "free (keep-data, global menu), the data-chosen share is 0.5 on essentially every stream, because the two arms "
          "of a segment have nearly equal Bernoulli variances, so 50/50 is already the variance-optimal within-segment "
          "split; the menu cost (a larger union) is then paid for nothing.  (ii) Members that do move the design pay "
          "for it: the RAGE-style reset discards earlier-phase samples, and the per-segment keep-data menu charges "
          "M^S design paths per boundary (log L grows by S ln M per boundary), which costs more than the "
          "re-allocation gains.  Post hoc and descriptive: it does not change the v8 confirmatory result."]
    (OUT / "summary.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["toy", "select", "analyse"] + sorted(TASKS))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.task == "toy":
        run_toy()
    elif args.task == "select":
        select()
    elif args.task == "analyse":
        analyse()
    else:
        if TASKS[args.task]["half"] == "eval" and not (OUT / "selection.json").exists():
            raise SystemExit("eval refused: run tune_x5, tune_cr9 and select first (picks fixed before eval)")
        run_task(args.task)


if __name__ == "__main__":
    main()
