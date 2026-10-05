"""r3_p2_lin_stream_smoke: P2 E1-Lin stream + five-level switch smoke (dev seeds only) + Lin power study.

Uses dsswm/streams/lin_stream.py (r3_setup_lin_stream) unchanged. Dev instances = seeds 740-749 after gap-quota
replacement (746 -> 750, 749 -> 751, see r3_setup_lin_stream pilot).
  pilot : 10 dev instances x stream 0 x 15 problems x {JPC-Lin off/full, RAGE off/full, XY-static off, G-opt full}
          (900 runs) + 3 instances x 15 x {JPC-Lin, RAGE} x {vol, ev1} (180 runs); T_max_Lin = 20000.
          Pass: median single run <= 5 s AND evidence-boundary mismatch 0 AND completion >= 0.9 AND
          dynamic-class Lambda_hat_perp <= 1e-8.
  full  : 10 dev instances x stream 0 x 15 x all Lin arms ({JPC-Lin, RAGE} x {off, vol, ev1, full, orth},
          XY-static off, G-opt full, B1eb full, Lin-Static JPC-Lin x {off, ev1, full}); re-estimate instance-level
          log-ratio SD / A2 MDE at n = 120, T_max_Lin and the stream count for r3_prereg_lock.
Learner side (write-ahead, before any truth scoring): predictors.jsonl with the closed-form Lin leverage predictors
(Lambda_perp under the dynamic class = model class, rho*, A_k) at the certification / stop time of every set-method
problem. results.jsonl = harness scoring (run_lin_cell rows).
"Single run" = one (instance, stream, problem, method, arm) solve; wall time excludes the predictor computation.
Usage: run_r3_p2_lin_stream_smoke.py --mode {pilot,full} [--workers 4]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
WS = HERE.parents[1]
RES_ROOT = WS / "exp" / "results"
TASK = "r3_p2_lin_stream_smoke"
PLANNED_MIN = 30

DEV_SEEDS = [740, 741, 742, 743, 744, 745, 750, 747, 748, 751]   # 740-749 after quota replacement
EXTRA_SEEDS = DEV_SEEDS[:3]
T_MAX = 20000
N_PROB = 15
SET_METHODS = ("JPC-Lin", "RAGE", "XY-static", "G-opt")
PILOT_CORE = [("JPC-Lin", ("off", "full")), ("RAGE", ("off", "full")), ("XY-static", ("off",)), ("G-opt", ("full",))]
PILOT_EXTRA = [("JPC-Lin", ("vol", "ev1")), ("RAGE", ("vol", "ev1"))]
FULL_CELLS = [("JPC-Lin", ("off", "vol", "ev1", "full", "orth")), ("RAGE", ("off", "vol", "ev1", "full", "orth")),
              ("XY-static", ("off",)), ("G-opt", ("full",)), ("B1eb", ("full",))]
FULL_STATIC = [("JPC-Lin", ("off", "ev1", "full"))]
CODE_FILES = ["dsswm/streams/lin_stream.py", "dsswm/baselines/lin_rage.py", "dsswm/mechanism/leverage.py",
              "dsswm/evidence/reuse_switch.py", "dsswm/evidence/orth_design.py", "run_r3_lin.py",
              "run_r3_p2_lin_stream_smoke.py"]
LAMBDA_TOL = 1e-8
Z80 = 1.959964 + 0.841621      # two-sided alpha = 0.05, power 0.8


def progress(step, total, phase, metric=None):
    (RES_ROOT / f"{TASK}_PROGRESS.json").write_text(json.dumps({
        "task_id": TASK, "epoch": step, "total_epochs": total, "step": step, "total_steps": total, "loss": None,
        "metric": {"phase": phase, **(metric or {})}, "updated_at": datetime.now().isoformat()}))


def mark_done(status, summary):
    pid = RES_ROOT / f"{TASK}.pid"
    if pid.exists():
        pid.unlink()
    pf = RES_ROOT / f"{TASK}_PROGRESS.json"
    fp = {}
    if pf.exists():
        try:
            fp = json.loads(pf.read_text())
        except ValueError:
            pass
    (RES_ROOT / f"{TASK}_DONE").write_text(json.dumps({"task_id": TASK, "status": status, "summary": summary,
                                                        "final_progress": fp, "timestamp": datetime.now().isoformat()}))


def code_sha():
    out = {f: hashlib.sha256((HERE / f).read_bytes()).hexdigest() for f in CODE_FILES}
    out["_combined"] = hashlib.sha256("".join(out[f] for f in CODE_FILES).encode()).hexdigest()
    return out


# ============================================================================================ learner-side hook
# The per-job context lives on the lin_stream module (not in this script's globals): loky/cloudpickle ships functions
# defined in __main__ by value, so a global dict here would be a fresh copy per call and the hook would keep the first.
def _ctx():
    import dsswm.streams.lin_stream as ls
    if not hasattr(ls, "_p2_ctx"):
        ls._p2_ctx = {}
    return ls._p2_ctx


def _install_predictor_hook():
    """Wrap lin_stream.solve: after the learner stops on problem k (and BEFORE run_lin_cell scores anything), log the
    closed-form Lin predictors computed from the learner's own evidence set. Sibling-platform solves are skipped."""
    import dsswm.streams.lin_stream as ls
    from dsswm.baselines import lin_rage
    from dsswm.certify.lin_closed import certify_lin
    from dsswm.mechanism.leverage import LinLeverage
    if getattr(ls, "_p2_hooked", False):
        return
    orig = lin_rage.solve

    def solve(pub, method, k, sw, lr, handle, rng, tmax):
        res = orig(pub, method, k, sw, lr, handle, rng, tmax)
        c = _ctx()
        if not c or sw.arm != c["arm"] or method not in SET_METHODS or c.get("static") or lr is None:
            return res
        t0 = time.perf_counter()
        try:
            lev = c["lev"].get(id(pub.lc))
            if lev is None:
                lev = c["lev"][id(pub.lc)] = LinLeverage(pub.lc, ref=pub.lc, sigma=pub.sigma)
            ell = lr.inner
            th = ell.theta_hat()
            Z = pub.Z[k]
            cert = certify_lin(Z, ell, pub.eps)
            vals = Z @ th
            vals[cert["pi_hat"]] = -np.inf
            ru = int(np.argmax(vals))
            mc = int(cert["binding"]) if int(cert["binding"]) != int(cert["pi_hat"]) else ru
            pairs = [("theta_hat_runner_up", int(cert["pi_hat"]), ru), ("minimax_challenger", int(cert["pi_hat"]), mc)]
            q = pub.problems[k]
            pd = lev.predictors(q, lr.obs, th, pairs, cert["r_bar"], pub.eps, pub.aspace)
            c["plog"].write({"instance": c["seed"], "stream": c["stream"], "method": method, "arm": c["arm"],
                             "problem": q.pid, "problem_index": int(k), "layer": "E1-Lin", "status": res["status"],
                             "new_env_steps": int(res["steps"]), "n_obs_set": len(lr.obs),
                             "prov_counts": lr.provenance_counts(), "dynamic": pd})
            res["extra"]["lambda_perp_dyn"] = pd["lambda_perp_max"]
        except Exception as e:  # noqa: BLE001 - logged, run continues
            c["pred_errs"].append({"seed": c["seed"], "method": method, "arm": c["arm"], "k": int(k), "error": repr(e),
                                   "tb": traceback.format_exc()[-2000:]})
        res["extra"]["pred_sec"] = time.perf_counter() - t0
        return res

    ls.solve = solve
    ls._p2_hooked = True


def run_job(seed, static, method, arms, n_problems, tmax, parts_dir, stream=0):
    from dsswm.mechanism.predictor_log import PredictorLog
    from dsswm.streams.lin_stream import LinStreamBuilder, run_lin_cell
    _install_predictor_hook()
    tag = f"i{seed}_s{stream}_{'static' if static else 'dyn'}_{method}_{'-'.join(arms)}"
    done = parts_dir / f"{tag}.json"
    if done.exists():                                         # resumable
        d = json.loads(done.read_text())
        return d["rows"], d["samples"], d["errs"], d["pred_errs"]
    ppath = parts_dir / f"{tag}_predictors.jsonl"
    if ppath.exists():
        ppath.unlink()
    plog = PredictorLog(ppath)
    b = LinStreamBuilder(seed)
    rows, samples, errs, pred_errs, cache = [], [], [], [], {}
    lev_cache: dict = {}
    for arm in arms:
        t0 = time.perf_counter()
        prov = None
        if arm == "orth":                       # lock definition of the Lin orth arm (run_r3_lin.LinOrthProvider)
            try:
                prov = _make_orth_provider(b, seed, stream, static, method, n_problems, tmax, cache, parts_dir, tag)
            except Exception as e:  # noqa: BLE001
                errs.append({"seed": seed, "static": static, "method": method, "arm": arm, "stage": "orth_provider",
                             "error": repr(e), "tb": traceback.format_exc()[-3000:]})
                continue
        ctx = _ctx()
        ctx.clear()
        ctx.update({"seed": seed, "stream": stream, "arm": arm, "static": static, "plog": plog, "lev": lev_cache,
                     "pred_errs": pred_errs})
        try:
            rr = run_lin_cell(b, stream, method, arm, tmax=tmax, static=static, n_problems=n_problems,
                              sibling_cache=cache, samples=samples, orth_provider=prov)
            cell_sec = time.perf_counter() - t0
            if prov is not None:
                n_rep = sum(int(r["replay_steps"]) for r in rr)
                if int(prov.env_o.n_steps) != n_rep or prov.n_rows_out != n_rep:
                    errs.append({"seed": seed, "static": static, "method": method, "arm": arm, "stage": "orth_audit",
                                 "error": f"orth sibling steps {prov.env_o.n_steps} / rows out {prov.n_rows_out} "
                                          f"!= replay rows {n_rep}"})
            for r in rr:
                rec = prov.records.get(r["problem_index"], {}) if prov is not None else {}
                r.update({"orth_status": rec.get("status"), "orth_trivial": rec.get("trivial"),
                          "N_k": rec.get("N_k"), "orth_provider_sec": rec.get("provider_sec")})
                r["cell_sec"] = cell_sec
                r["run_s"] = r["wall_clock_s"] - float(r.get("x_pred_sec") or 0.0)
            rows += rr
        except Exception as e:  # noqa: BLE001 - crashes are counted, not hidden
            errs.append({"seed": seed, "static": static, "method": method, "arm": arm, "error": repr(e),
                         "tb": traceback.format_exc()[-3000:]})
        finally:
            _ctx().clear()
    done.write_text(json.dumps({"rows": rows, "samples": samples, "errs": errs, "pred_errs": pred_errs}, default=str))
    return rows, samples, errs, pred_errs


def _make_orth_provider(b, seed, stream, static, method, n_problems, tmax, cache, parts_dir, tag):
    """Sibling `full` run (noise + 7919) -> shadow ledger -> run_r3_lin.LinOrthProvider (second sibling platform)."""
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import run_r3_lin as R
    from dsswm.evidence.reuse_switch import ShadowLedger
    from dsswm.mechanism.predictor_log import PredictorLog
    from dsswm.streams.lin_stream import _public, sibling_run
    rows_s, env_s, pad_fn, sib_info = sibling_run(b, stream, method, tmax, static, n_problems)
    st0 = b.stream(stream, static=static, n_problems=n_problems)
    shadow = ShadowLedger(rows_s, [q.pid for q in st0.problems], pad_fn=pad_fn)
    cache[(b.seed, stream, static, method, n_problems, tmax)] = (shadow, env_s, sib_info)
    op = parts_dir / f"{tag}_orth.jsonl"
    op.unlink(missing_ok=True)
    st_o = b.stream(stream, static=static, n_problems=n_problems)
    return R.LinOrthProvider(b, stream, _public(b, st_o), st_o.init_obs, rows_s, shadow, PredictorLog(op),
                             {"instance": seed, "stream": stream, "method": method, "arm": "orth"}, st_o.noise_seed)


# ============================================================================================ analysis
def lr_(a, b):
    return float(np.log((a + 1.0) / (b + 1.0)))


def S_of(rows, inst, method, arm, tau=T_MAX, layer="E1-Lin"):
    rr = [r for r in rows if r["instance"] == inst and r["method"] == method and r["arm"] == arm
          and r["layer"] == layer and r["gap_layer"] != "tie"]
    return sum(r[f"cost_tau{tau}"] for r in rr) if rr else None


def analyse(rows, errs, pred_errs, preds, mode):
    out = {}
    cells = defaultdict(list)
    for r in rows:
        cells[(r["layer"], r["method"], r["arm"])].append(r)
    by_cell = {}
    for (lay, m, a), rr in sorted(cells.items()):
        stream_steps = defaultdict(int)
        for r in rr:
            stream_steps[r["instance"]] += r["new_env_steps"]
        by_cell[f"{lay}|{m}|{a}"] = {
            "n": len(rr), "status": dict(Counter(r["status"] for r in rr)),
            "completion": float(np.mean([r["status"] == "CERTIFIED" for r in rr])),
            "completion_nontie": float(np.mean([r["status"] == "CERTIFIED" for r in rr if r["gap_layer"] != "tie"]
                                               or [np.nan])),
            "completion_by_gap_layer": {g: float(np.mean([r["status"] == "CERTIFIED" for r in rr
                                                          if r["gap_layer"] == g])) for g in ("tie", "near", "clear")
                                        if any(r["gap_layer"] == g for r in rr)},
            "median_steps": float(np.median([r["new_env_steps"] for r in rr])),
            "p90_steps": float(np.percentile([r["new_env_steps"] for r in rr], 90)),
            "max_steps": int(max(r["new_env_steps"] for r in rr)),
            "stream_steps_median": float(np.median(list(stream_steps.values()))),
            "stream_steps_by_instance": dict(stream_steps),
            "zero_cost": int(sum(r["zero_cost"] for r in rr)), "false_cert": int(sum(r["false_cert"] for r in rr)),
            "truncated_at_tmax": int(sum(r["truncated_at_tmax"] for r in rr)),
            "replay_steps_total": int(sum(r["replay_steps"] for r in rr)),
            "pad_steps_total": int(sum(r["replay_pad_steps"] for r in rr)),
            "median_run_s": float(np.median([r["run_s"] for r in rr])),
            "p90_run_s": float(np.percentile([r["run_s"] for r in rr], 90)),
            "max_run_s": float(max(r["run_s"] for r in rr)),
            "median_cell_s": float(np.median(list({(r["instance"], r["stream"]): r["cell_sec"] for r in rr}.values()))),
            "sec_per_1000_steps": float(1000 * sum(r["run_s"] for r in rr)
                                        / max(1, sum(r["new_env_steps"] for r in rr))),
            "median_pred_s": float(np.median([float(r.get("x_pred_sec") or 0.0) for r in rr]))}
        if m == "B1eb":
            by_cell[f"{lay}|{m}|{a}"]["reused_trials_shared"] = [r.get("x_n_reused_trials") for r in rr if r["shared"]]
    out["by_cell"] = by_cell

    # evidence boundary: billing check (raised inside end_problem), replay accounting per arm, replay never billed
    eb = []
    for r in rows:
        base = r["arm"]
        bad = []
        if not r["billing_ok"]:
            bad.append("billing_error")
        rep = r["replay_steps"]
        if base in ("off", "ev1", "full") and rep != 0:
            bad.append("replay_in_non_replay_arm")
        if base == "vol" and rep != r["n_rounds_billed"] - 20 - r["new_env_steps"]:
            bad.append("vol_replay_ne_Nk")
        if base == "orth" and r["status"] != "ORTH_INFEASIBLE" and rep != r["n_rounds_billed"] - 20 - r["new_env_steps"]:
            bad.append("orth_replay_ne_Nk")
        if base == "orth" and r["status"] == "ORTH_INFEASIBLE" and (rep != 0 or r["new_env_steps"] != 0):
            bad.append("orth_infeasible_but_billed")
        pc = r.get("prov_counts") or {}
        if base in ("off", "vol") and pc.get("real", r["new_env_steps"]) != r["new_env_steps"] and "real" in pc:
            bad.append("real_rows_ne_new_steps")
        if bad:
            eb.append({"instance": r["instance"], "method": r["method"], "arm": r["arm"], "pid": r["pid"], "bad": bad,
                       "prov_counts": pc})
    out["evidence_boundary"] = {"mismatch": len(eb), "examples": eb[:10], "crashes": len(errs),
                                "billing_mismatch": int(sum(not r["billing_ok"] for r in rows))}

    dyn = [r for r in rows if r["layer"] == "E1-Lin"]
    out["completion_rate_all_rows"] = float(np.mean([r["status"] == "CERTIFIED" for r in dyn]))
    dyn_run = [r for r in dyn if r["status"] != "ORTH_INFEASIBLE"]
    out["completion_rate"] = float(np.mean([r["status"] == "CERTIFIED" for r in dyn_run]))
    orth = [r for r in rows if r["arm"] == "orth"]
    if orth:
        nt = [r for r in orth if not r.get("orth_trivial")]
        out["orth"] = {"n_rows": len(orth), "n_trivial_N0": len(orth) - len(nt),
                       "feasible_rate_nontrivial": float(np.mean([r["status"] != "ORTH_INFEASIBLE" for r in nt]))
                       if nt else None,
                       "by_method": {m: float(np.mean([r["status"] != "ORTH_INFEASIBLE" for r in nt
                                                       if r["method"] == m])) for m in sorted({r["method"] for r in nt})},
                       "completion_given_feasible": float(np.mean([r["status"] == "CERTIFIED" for r in orth
                                                                   if r["status"] != "ORTH_INFEASIBLE"]))}
    out["completion_rate_nontie"] = float(np.mean([r["status"] == "CERTIFIED" for r in dyn_run
                                                   if r["gap_layer"] != "tie"]))
    out["truncated_at_tmax"] = int(sum(r["truncated_at_tmax"] for r in dyn))
    out["false_cert_total"] = int(sum(r["false_cert"] for r in rows))
    out["sec_per_run_median"] = float(np.median([r["run_s"] for r in rows]))
    out["sec_per_run_p90"] = float(np.percentile([r["run_s"] for r in rows], 90))
    out["sec_per_run_max"] = float(max(r["run_s"] for r in rows))
    out["sec_per_stream_cell_median"] = float(np.median(list({(r["instance"], r["layer"], r["method"], r["arm"]):
                                                              r["cell_sec"] for r in rows}.values())))
    out["sum_run_s"] = float(sum(r["run_s"] for r in rows))

    # Lambda_perp (learner-side predictors, dynamic class = model class)
    lam = [p["dynamic"]["lambda_perp_max"] for p in preds if p.get("dynamic", {}).get("lambda_perp_max") is not None]
    lam = [float(x) if not isinstance(x, str) else float("inf") for x in lam]
    lnp = [float(q["lambda_perp_nonparam"]) for p in preds for q in p["dynamic"]["pairs"]
           if not isinstance(q["lambda_perp_nonparam"], str)]
    out["lin_lambda_perp"] = {"n_predictor_records": len(preds), "n_values": len(lam),
                              "max": float(max(lam)) if lam else None,
                              "median": float(np.median(lam)) if lam else None,
                              "n_gt_tol": int(sum(x > LAMBDA_TOL for x in lam)), "tol": LAMBDA_TOL,
                              "nonparam_median_diagnostic": float(np.median(lnp)) if lnp else None,
                              "predictor_errors": len(pred_errs),
                              "median_pred_sec": float(np.median([p["dynamic"]["sec_per_call"] for p in preds]))
                              if preds else None}

    # instance-level descriptive log ratios (non-tie cost, tau = T_max), SD -> MDE at n = 120 instances
    insts = sorted({r["instance"] for r in dyn})
    lrs = defaultdict(list)
    for i in insts:
        S = {(m, a): S_of(rows, i, m, a) for m in ("JPC-Lin", "RAGE", "XY-static", "G-opt", "B1eb")
             for a in ("off", "vol", "ev1", "full", "orth")}
        def add(name, m1, a1, m2, a2):
            if S[(m1, a1)] is not None and S[(m2, a2)] is not None:
                lrs[name].append(lr_(S[(m1, a1)], S[(m2, a2)]))
        add("JPC_full_off", "JPC-Lin", "full", "JPC-Lin", "off")
        add("RAGE_full_off", "RAGE", "full", "RAGE", "off")
        add("JPC_vs_RAGE_off", "JPC-Lin", "off", "RAGE", "off")
        add("JPC_vs_XYstatic_off", "JPC-Lin", "off", "XY-static", "off")
        add("JPC_full_vs_Gopt_full", "JPC-Lin", "full", "G-opt", "full")
        add("JPC_vol_off", "JPC-Lin", "vol", "JPC-Lin", "off")
        add("JPC_ev1_off", "JPC-Lin", "ev1", "JPC-Lin", "off")
        add("JPC_full_vol", "JPC-Lin", "full", "JPC-Lin", "vol")
        add("RAGE_vol_off", "RAGE", "vol", "RAGE", "off")
        add("RAGE_ev1_off", "RAGE", "ev1", "RAGE", "off")
        add("RAGE_full_vol", "RAGE", "full", "RAGE", "vol")
        add("B1eb_full_vs_JPC_full", "B1eb", "full", "JPC-Lin", "full")
        Ss = {a: S_of(rows, i, "JPC-Lin", a, layer="Lin-Static") for a in ("off", "ev1", "full")}
        if Ss["full"] is not None and Ss["off"] is not None:
            lrs["LinStatic_JPC_full_off"].append(lr_(Ss["full"], Ss["off"]))
            if S[("JPC-Lin", "full")] is not None and S[("JPC-Lin", "off")] is not None:
                lrs["Dyn_minus_Static_full_off"].append(lr_(S[("JPC-Lin", "full")], S[("JPC-Lin", "off")])
                                                        - lr_(Ss["full"], Ss["off"]))
        if len(lrs["JPC_full_off"]) == len(lrs["RAGE_full_off"]) and S[("JPC-Lin", "full")] is not None \
                and S[("RAGE", "full")] is not None:
            lrs["I_full_off_A2"].append(lr_(S[("JPC-Lin", "full")], S[("JPC-Lin", "off")])
                                        - lr_(S[("RAGE", "full")], S[("RAGE", "off")]))
    power = {}
    for name, v in lrs.items():
        v = np.asarray(v)
        sd = float(v.std(ddof=1)) if len(v) > 1 else None
        power[name] = {"n": len(v), "mean": float(v.mean()), "sd": sd, "values": [round(float(x), 4) for x in v],
                       "se_n120": None if sd is None else sd / math.sqrt(120),
                       "mde80_n120": None if sd is None else Z80 * sd / math.sqrt(120),
                       "n_for_mde_0.4": None if sd is None else int(math.ceil((Z80 * sd / 0.4) ** 2))}
    out["instance_log_ratios"] = power
    out["note_power"] = ("pilot：仅 10 个开发实例 × 流 0，SD 为粗估；I_full_off_A2 = log(S_JPC_full/S_JPC_off) - "
                         "log(S_RAGE_full/S_RAGE_off)，非平局成本按 tau = T_max 截断。")

    # T_max_Lin: steps distribution of certified non-tie problems; censoring at smaller tau
    cert_nt = [r["new_env_steps"] for r in dyn if r["status"] == "CERTIFIED" and r["gap_layer"] != "tie"]
    tau_cens = {}
    for tau in (5000, 10000, 20000):
        tau_cens[str(tau)] = float(np.mean([r[f"censored_tau{tau}"] for r in dyn if r["gap_layer"] != "tie"]))
    out["tmax_study"] = {"certified_nontie_steps_p50": float(np.median(cert_nt)) if cert_nt else None,
                         "p90": float(np.percentile(cert_nt, 90)) if cert_nt else None,
                         "p99": float(np.percentile(cert_nt, 99)) if cert_nt else None,
                         "max": int(max(cert_nt)) if cert_nt else None, "censored_frac_nontie_by_tau": tau_cens}

    # timing budget per stream for the full / eval design
    per_stream = defaultdict(float)
    for r in rows:
        per_stream[(r["instance"], r["stream"], r["layer"])] += r["run_s"]
    cell_stream = {k: v["median_cell_s"] for k, v in by_cell.items()}
    out["budget"] = {"median_cell_s_by_cell": cell_stream,
                     "core_cells_sec_per_instance_stream": float(sum(
                         v for k, v in cell_stream.items() if k.split("|")[2] in ("off", "full"))),
                     "all_cells_sec_per_instance_stream_observed": float(sum(cell_stream.values()))}
    return out


def recommend(an, mode):
    """Lock recommendations (descriptive; the lock task makes the final choice)."""
    ts = an["tmax_study"]
    cens = ts["censored_frac_nontie_by_tau"]
    tmax = T_MAX
    for tau in (5000, 10000):
        if cens[str(tau)] <= cens[str(T_MAX)] + 0.02:
            tmax = tau
            break
    # stream count: per instance, all cells x 3 streams x 4 workers within ~1 h for 120 instances?
    per_inst_stream = an["budget"]["all_cells_sec_per_instance_stream_observed"]
    est_3 = per_inst_stream * 3 * 120 / 4 / 60
    est_1 = per_inst_stream * 120 / 4 / 60
    return {"T_max_Lin": tmax, "T_max_Lin_rule": "最小的 tau∈{5000,10000,20000}，使非平局删失率不超过 tau=20000 时 +0.02",
            "est_minutes_120inst_3streams_observed_cells_4workers": est_3,
            "est_minutes_120inst_1stream_observed_cells_4workers": est_1,
            "streams_recommendation": 3 if est_3 <= 120 else "1 (+ 流 1-2 只跑 {JPC-Lin, RAGE} x {off, full}，降规模预案)",
            "basis": "pilot 计时（并发运行，偏高）；仅覆盖本模式实际跑过的 cell" if mode == "pilot" else "full 计时（并发运行）"}


def write_table(an, out_dir):
    lines = ["| Method | Arm | Completion | Median steps | sec/run |", "|---|---|---|---|---|"]
    for k, v in an["by_cell"].items():
        lay, m, a = k.split("|")
        lines.append(f"| {m}{' (Lin-Static)' if lay != 'E1-Lin' else ''} | {a} | {v['completion']:.3f} | "
                     f"{v['median_steps']:.0f} | {v['median_run_s']:.3f} |")
    (out_dir / "table_lin_smoke.md").write_text("\n".join(lines) + "\n")
    return lines


def update_gpu_progress(status, start_iso, wall_min, snapshot):
    import fcntl
    gp = WS / "exp" / "gpu_progress.json"
    lock = WS / "exp" / "gpu_progress.lock"
    with open(lock, "a+") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        d = json.loads(gp.read_text()) if gp.exists() else {}
        for k, v in (("completed", []), ("failed", []), ("running", {}), ("timings", {})):
            d.setdefault(k, v)
        key = "completed" if status == "success" else "failed"
        other = "failed" if key == "completed" else "completed"
        if TASK not in d[key]:
            d[key].append(TASK)
        if TASK in d[other]:
            d[other].remove(TASK)
        d["running"].pop(TASK, None)
        d["timings"][TASK] = {"planned_min": PLANNED_MIN, "actual_min": int(round(wall_min)), "start_time": start_iso,
                              "end_time": datetime.now().isoformat(), "config_snapshot": snapshot}
        tmp = gp.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(d, indent=2))
        os.replace(tmp, gp)
        fcntl.flock(lf, fcntl.LOCK_UN)


def main():
    from joblib import Parallel, delayed
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("pilot", "full"), default="pilot")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-gpu-progress", action="store_true")
    a = ap.parse_args()
    out_dir = RES_ROOT / ("pilots" if a.mode == "pilot" else "full") / TASK
    parts = out_dir / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    (out_dir / "samples").mkdir(exist_ok=True)
    (RES_ROOT / f"{TASK}.pid").write_text(str(os.getpid()))
    (RES_ROOT / f"{TASK}_gpu_profile.json").write_text(json.dumps({
        "gpu_name": "none (CPU only)", "vram_total_mb": 0, "max_batch_size": "n/a", "vram_used_mb": 0,
        "utilization_pct": None, "note": "E1-Lin learner loop is CPU-only numpy (11-dim ellipsoid); gpu slot = token"}))
    start_iso = datetime.now().isoformat()
    t0 = time.perf_counter()
    status = "failed"
    try:
        if a.mode == "pilot":
            jobs = [(s, False, m, arms, N_PROB, T_MAX, parts) for s in DEV_SEEDS for m, arms in PILOT_CORE]
            jobs += [(s, False, m, arms, N_PROB, T_MAX, parts) for s in EXTRA_SEEDS for m, arms in PILOT_EXTRA]
        else:
            jobs = [(s, False, m, (arm,), N_PROB, T_MAX, parts) for s in DEV_SEEDS for m, arms in FULL_CELLS
                    for arm in arms]
            jobs += [(s, True, m, arms, N_PROB, T_MAX, parts) for s in DEV_SEEDS for m, arms in FULL_STATIC]
        # longest first (vol includes a sibling run; RAGE / XY-static tend to be slower)
        jobs.sort(key=lambda j: -(len(j[3]) + 2 * ("vol" in j[3]) + (j[2] in ("RAGE", "XY-static"))))
        progress(0, len(jobs), "running", {"n_jobs": len(jobs)})
        res = []
        CH = max(1, a.workers * 2)
        for i in range(0, len(jobs), CH):
            res += Parallel(n_jobs=a.workers, backend="loky")(delayed(run_job)(*j) for j in jobs[i:i + CH])
            progress(min(i + CH, len(jobs)), len(jobs), "running", {"n_jobs": len(jobs)})
            print(f"[{datetime.now():%H:%M:%S}] {min(i + CH, len(jobs))}/{len(jobs)} jobs", flush=True)
        run_sec = time.perf_counter() - t0
        rows = [r for rr, _, _, _ in res for r in rr]
        samples = [x for _, ss, _, _ in res for x in ss]
        errs = [e for _, _, ee, _ in res for e in ee]
        pred_errs = [e for _, _, _, pe in res for e in pe]
        # merge learner predictor parts (in job order), then write harness results
        from dsswm.mechanism.predictor_log import load_jsonl
        preds = []
        with open(out_dir / "predictors.jsonl", "w") as f:
            for p in sorted(parts.glob("*_predictors.jsonl")):
                for rec in load_jsonl(p):
                    preds.append(rec)
                    f.write(json.dumps(rec, sort_keys=True) + "\n")
        with open(out_dir / "orth_designs.jsonl", "w") as f:
            for p in sorted(parts.glob("*_orth.jsonl")):
                for rec in load_jsonl(p):
                    f.write(json.dumps(rec, sort_keys=True, default=str) + "\n")
        scored_at = time.time()
        with open(out_dir / "results.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps({**r, "problem": r["pid"], "scored_at": scored_at}, default=str) + "\n")
        wa_ok = all(p["logged_at"] <= scored_at for p in preds)
        if errs:
            (out_dir / "errors.json").write_text(json.dumps(errs, indent=1))
        if pred_errs:
            (out_dir / "predictor_errors.json").write_text(json.dumps(pred_errs[:50], indent=1))
        (out_dir / "samples" / "cells.json").write_text(json.dumps(samples[:60], indent=1, default=str))
        an = analyse(rows, errs, pred_errs, preds, a.mode)
        an["predictor_write_ahead_ok"] = wa_ok
        # qualitative samples: 8 rows spread across cells / layers
        pick, seen = [], set()
        for r in rows:
            key = (r["method"], r["arm"], r["gap_layer"])
            if key not in seen and r["instance"] == DEV_SEEDS[0]:
                seen.add(key)
                pick.append({k: r[k] for k in ("instance", "pid", "method", "arm", "gap_layer", "true_gap", "shared",
                                               "status", "new_env_steps", "replay_steps", "certified_policy",
                                               "true_regret", "run_s", "prov_counts")}
                            | {"lambda_perp_dyn": r.get("x_lambda_perp_dyn"), "r_bar_end": r.get("x_r_bar_end")})
        (out_dir / "samples" / "qualitative_rows.json").write_text(json.dumps(pick, indent=1, default=str))
        table = write_table(an, out_dir)
        summary = {"task": TASK, "mode": a.mode, "started_at": start_iso, "dev_seeds": DEV_SEEDS,
                   "extra_seeds": EXTRA_SEEDS if a.mode == "pilot" else None, "tmax": T_MAX, "n_problems": N_PROB,
                   "stream": 0, "n_jobs": len(jobs), "n_runs": len(rows), "run_sec": run_sec,
                   "note": "并发运行（与其他 r3 任务共享 20 核 CPU，4 worker）；计时偏高。CPU only。种子 746/749 因配额失败"
                           "按 setup 记录替换为 750/751。",
                   **an, "lock_recommendation": recommend(an, a.mode), "table": table}
        if a.mode == "pilot":
            crit = {"median_single_run_le_5s": an["sec_per_run_median"] <= 5.0,
                    "evidence_boundary_mismatch_zero": an["evidence_boundary"]["mismatch"] == 0
                    and an["evidence_boundary"]["crashes"] == 0,
                    "completion_ge_0.9": an["completion_rate"] >= 0.9,
                    "lin_dynamic_lambda_perp_le_1e-8": an["lin_lambda_perp"]["max"] is not None
                    and an["lin_lambda_perp"]["max"] <= LAMBDA_TOL and an["lin_lambda_perp"]["predictor_errors"] == 0}
        else:
            crit = {"no_crash": an["evidence_boundary"]["crashes"] == 0,
                    "evidence_boundary_mismatch_zero": an["evidence_boundary"]["mismatch"] == 0,
                    "power_written": "I_full_off_A2" in an["instance_log_ratios"]}
        summary["pass_criteria"] = crit
        summary["passed"] = all(crit.values())
        summary["go_no_go"] = "GO" if summary["passed"] else "NO_GO"
        summary["code_sha256"] = code_sha()
        summary["wall_clock_s"] = time.perf_counter() - t0
        summary["finished_at"] = datetime.now().isoformat()
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
        print(json.dumps(crit), summary["go_no_go"], flush=True)
        status = "success" if summary["passed"] else "failed"
        mark_done(status, f"{summary['go_no_go']} {crit}")
    except Exception as e:  # noqa: BLE001
        (out_dir / "crash.txt").write_text(traceback.format_exc())
        mark_done("failed", repr(e))
        raise
    finally:
        if not a.no_gpu_progress:
            update_gpu_progress(status, start_iso, (time.perf_counter() - t0) / 60,
                                {"mode": a.mode, "layer": "E1-Lin 3x3 sigma=1.5 eps=0.05", "dev_seeds": DEV_SEEDS,
                                 "n_problems": N_PROB, "tmax": T_MAX, "stream": 0, "cpu_workers": a.workers,
                                 "gpu_model": "none (CPU only)", "gpu_count": 0, "concurrent": "other r3 tasks"})


if __name__ == "__main__":
    main()
