"""r3_lin_[a-c]: E1-Lin continuous tier, reuse x method (A1, A2, A3, A3o on Lin; Lambda_perp = 0 negative control).

Design  E1-Lin 3x3, sigma = 1.5, eps_Lin = 0.05, gap-quota streams (tie / near / clear x 5, 30% shared (s0, Pi)),
        T_max_Lin = 20000 (lock frozen_items.T_max_Lin). CRN: every arm of one (instance, stream, method) runs on a fresh
        copy of the same platform (lin_stream.run_lin_cell, unchanged harness).
  cells (stream 0): {JPC-Lin, RAGE} x {off, vol, orth, ev1, full} + XY-static off + G-opt full + B1eb full
                    + Lin-Static (own static quota) JPC-Lin x {off, ev1, full}  = 18 cells.
  streams 1-2     : as frozen in the lock (frozen_items.lin_streams.streams_1_2): "all cells" -> the same 18 cells as
                    stream 0 (full design, v3 lock); "core ..." -> {JPC-Lin, RAGE} x {off, full} (downscale preset).
  Lin-Static control, lock option (b): on every E1-Lin dynamic set-method certification the predictor hook ALSO logs
        Lambda_perp of the STATIC model class (d = 11, ref = dynamic class) on the dynamic platform's own observations
        (`static_model_on_dynamic_rows`); frozen-load Lin-Static rows give Lambda_perp == 0 by construction.
  vol   sibling platform (noise + 7919) run by the same method in the `full` arm (lin_stream.sibling_run); the shadow
        ledger is built once per (instance, stream, method) and shared by vol and orth.
  orth  learner-side provider, lock definition (gates.orth_confirmatory.definitions.lin_N_k): xi_perp on the
        deterministic load-state polytope (LinOrthModel, A1 exact dynamics, theta-free Fisher), start state = sibling
        state at problem start, one contrast pair (theta_hat top-2, ridge on n0 + vol rows), Fisher-trace matched to the
        vol rows shadow.take(k, N_k); executed on a SECOND sibling platform (noise + 2 x 7919); infeasible ->
        ORTH_INFEASIBLE (never downgraded). orth is descriptive (orth_confirmatory = false in the lock).
Logging (write separation)
  predictors.jsonl  learner process, fsync'ed, inside solve() right after the learner stops on problem k and BEFORE
                    run_lin_cell scores problem k: set methods -> closed-form Lin predictors (Lambda_perp under the
                    model class with the dynamic class as reference, rho*, A_k); B1eb -> light record.
                    Lin dynamic: model = ref = dynamic class (Lambda_perp == 0 by construction, negative control);
                    Lin-Static: model = static class (d = 11), ref = dynamic class (Lambda_perp > 0 expected).
  orth_designs.jsonl learner side, written by the provider before the orth problem is solved.
  results.jsonl     harness rows, written after the cell. JOIN KEY (instance, stream, method, arm, problem=k);
                    Lin-Static rows use method "LinStatic:JPC-Lin" so the key never collides with E1-Lin JPC-Lin.
Pilot  smoke + timing only (NO scientific readout): dev seeds 742-743, stream 0, all 18 cells x 15 problems (540 runs).
       Pass: 0 crashes AND billing 0 mismatch AND predictors before results for every certification AND projected
       full-chunk wall clock <= 55 min.
Full   chunk a/b/c = eval 10200-10215 / 10216-10231 / 10232-10247; requires the v3 lock (assert_locked()).
       Resumable: one part file per (instance, stream, layer, method) job.
Usage: run_r3_lin.py --chunk a --mode {pilot,full} [--workers 4]
Concurrent run (shares 20 cores with 3 other round-3 tasks): timings are "concurrent". CPU only.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import fcntl  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
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
N_PROB = 15
TAU_PRIMARY = 20000                      # Lin: T_max_Lin, LIN_TAUS scoring
ORTH_SIB_OFFSET = 2 * 7919
SET_METHODS = ("JPC-Lin", "RAGE", "XY-static", "G-opt")
DYN_CELLS = [("JPC-Lin", ("off", "vol", "orth", "ev1", "full")), ("RAGE", ("off", "vol", "orth", "ev1", "full")),
             ("XY-static", ("off",)), ("G-opt", ("full",)), ("B1eb", ("full",))]
STATIC_CELLS = [("JPC-Lin", ("off", "ev1", "full"))]
CORE_CELLS = [("JPC-Lin", ("off", "full")), ("RAGE", ("off", "full"))]
CHUNKS = {c: (10200 + 16 * i, 10216 + 16 * i) for i, c in enumerate("abc")}
RESERVE = {c: 11000 + 25 * i for i, c in enumerate("abc")}     # quota_fail replacement; unallocated block (>= 11000)
PILOT = {"seeds": [742, 743], "streams": [0]}
DEV_RESERVE = 761
LAMBDA_TOL = 1e-8
CODE_FILES = ["dsswm/streams/lin_stream.py", "dsswm/baselines/lin_rage.py", "dsswm/mechanism/leverage.py",
              "dsswm/evidence/reuse_switch.py", "dsswm/evidence/orth_design.py", "dsswm/mechanism/predictor_log.py",
              "run_r3_lin.py"]
TASK = "r3_lin_a"                                              # set in main()
PLANNED_MIN = 45


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


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


def mlabel(method, static):
    return f"LinStatic:{method}" if static else method


# ============================================================================================ learner-side hook
def _ctx():
    import dsswm.streams.lin_stream as ls
    if not hasattr(ls, "_r3lin_ctx"):
        ls._r3lin_ctx = {}
    return ls._r3lin_ctx


def _install_predictor_hook():
    """Wrap lin_stream.solve: after the learner stops on problem k (and BEFORE run_lin_cell scores problem k), log
    the learner-side record. Sibling-platform solves run with an empty context and are skipped."""
    import dsswm.streams.lin_stream as ls
    from dsswm.baselines import lin_rage
    from dsswm.certify.lin_closed import certify_lin
    if getattr(ls, "_r3lin_hooked", False):
        return
    orig = lin_rage.solve

    def solve(pub, method, k, sw, lr, handle, rng, tmax):
        res = orig(pub, method, k, sw, lr, handle, rng, tmax)
        c = _ctx()
        if not c or sw.arm != c["arm"] or lr is None:
            return res
        t0 = time.perf_counter()
        q = pub.problems[k]
        base = {"instance": c["seed"], "stream": c["stream"], "method": mlabel(method, c["static"]),
                "method_base": method, "arm": c["arm"], "problem": int(k), "pid": q.pid,
                "layer": "Lin-Static" if c["static"] else "E1-Lin", "status": res["status"],
                "new_env_steps": int(res["steps"]), "pi_hat": res["pi"]}
        try:
            if method not in SET_METHODS:
                c["plog"].write({**base, "record": "light", "n_rows_in_set": int(lr.n_rounds_total)})
            else:
                lev = c["lev"]
                ell = lr.inner
                th = ell.theta_hat()
                Z = pub.Z[k]
                cert = certify_lin(Z, ell, pub.eps)
                vals = Z @ th
                vals[cert["pi_hat"]] = -np.inf
                ru = int(np.argmax(vals))
                mc = int(cert["binding"]) if int(cert["binding"]) != int(cert["pi_hat"]) else ru
                pairs = [("theta_hat_runner_up", int(cert["pi_hat"]), ru),
                         ("minimax_challenger", int(cert["pi_hat"]), mc)]
                pd = lev.predictors(q, lr.obs, th, pairs, cert["r_bar"], pub.eps, pub.aspace)
                pd["pairs"] = [{kk: v for kk, v in p.items() if kk != "need_data"}
                               | {"n_need_data": int(np.size(p["need_data"]))} for p in pd["pairs"]]
                rec = {**base, "record": "lin_predictors", "n_obs_set": len(lr.obs),
                       "prov_counts": lr.provenance_counts(), "r_bar": float(cert["r_bar"]),
                       "lambda_perp_max": pd["lambda_perp_max"], "dynamic": pd}
                lev_s = c.get("lev_s")
                if lev_s is not None:                    # lock option (b): static model on dynamic-platform rows
                    ps = lev_s.predictors(q, lr.obs, np.zeros(lev_s.Phi.shape[1]), pairs, cert["r_bar"], pub.eps,
                                          pub.aspace)
                    rec["static_model_on_dynamic_rows"] = {
                        "lambda_perp_max": ps["lambda_perp_max"],
                        "pairs": [{kk: p.get(kk) for kk in ("label", "k1", "k2", "lambda_perp", "phi_perp",
                                                            "norm_h_over_xi", "rho_star", "S2", "A_k")}
                                  for p in ps["pairs"]],
                        "note": "model = static LinClass (d=11), ref = dynamic class, xi = this cell's own rows; "
                                "dJ_hat omitted (theta-free leverage)"}
                    rec["lambda_perp_static_model_max"] = ps["lambda_perp_max"]
                c["plog"].write(rec)
                res["extra"]["lambda_perp"] = pd["lambda_perp_max"]
        except Exception as e:  # noqa: BLE001 - logged; a light record keeps the write-ahead line
            c["pred_errs"].append({**base, "error": repr(e), "tb": traceback.format_exc()[-2000:]})
            try:
                c["plog"].write({**base, "record": "light_after_error", "error": repr(e)})
            except Exception:  # noqa: BLE001
                pass
        res["extra"]["pred_sec"] = time.perf_counter() - t0
        return res

    ls.solve = solve
    ls._r3lin_hooked = True


# ============================================================================================ orth provider (learner)
def design_record(d):
    """JSON-safe compact design record (identical to run_r3_p3_orth_replay_feasibility.design_record)."""
    from dsswm.evidence.orth_design import strip
    out = {k: v for k, v in d.items() if k not in ("perp", "par", "min")}
    for m in ("perp", "par", "min"):
        if d.get(m) is not None:
            out[m] = strip(d[m])
    return out


class LinOrthProvider:
    """Lock definition of the Lin orth arm (see module doc). Returns N_k rows executed on the second sibling platform,
    [] for N_k = 0 (trivial), None if xi_perp is not constructible (-> ORTH_INFEASIBLE)."""

    def __init__(self, b, stream, pub, init_obs, rows_s, shadow, olog, key_base, noise_seed):
        from dsswm.evidence.orth_design import LinOrthModel
        from dsswm.streams.lin_stream import make_env
        self.b, self.pub, self.init_obs, self.rows_s, self.shadow = b, pub, list(init_obs), rows_s, shadow
        self.olog, self.key_base, self.noise = olog, key_base, noise_seed
        self.env_o = make_env(b.seed, b.truth, noise_seed + ORTH_SIB_OFFSET)     # harness: second sibling platform
        self.ho = self.env_o.handle()
        self.codec = self.env_o.codec
        self.model = LinOrthModel(pub.geom, self.codec, b.lc.nmax, pub.sigma)
        self.records = {}
        self.n_rows_out = 0

    def __call__(self, k, pid, n_k, context):
        from dsswm.evidence.orth_design import alignment, lin_design_for_problem, run_mixture
        from dsswm.models.lin_class import ridge_estimate
        t0 = time.perf_counter()
        q = self.pub.problems[k]
        rec = {**self.key_base, "problem": int(k), "pid": pid, "N_k": int(n_k), "H": int(q.H),
               "n_policies": len(q.policies)}
        if n_k == 0:
            rec.update({"status": "trivial_N0", "trivial": True, "feasible": True, "provider_sec": 0.0})
            self.olog.write(rec)
            self.records[k] = rec
            return []
        vol = [o for o, _, _ in self.shadow.take(k, n_k)]
        X, y = [], []
        for o in self.init_obs + vol:
            Xi, yi = self.b.lc.obs_rows(o)
            X.append(Xi)
            y.append(yi)
        th, _ = ridge_estimate(np.concatenate(X), np.concatenate(y))
        Z = self.pub.Z[k]
        o_ = np.argsort(-(Z @ th))
        dirs = [Z[o_[0]] - Z[o_[1]]]
        first = [o for o, p in self.rows_s if p == pid]
        s_loads = np.asarray(first[0].loads if first else q.loads0, np.int64)
        s0 = self.codec.encode(s_loads)
        d = lin_design_for_problem(self.model, dirs, s0, q.H, vol)
        rec.update({"trivial": False, "pi_hat": int(o_[0]), "runner_up": int(o_[1]), **design_record(d)})
        rows = None
        if d["feasible"]:
            rng = np.random.default_rng([int(self.b.seed), int(self.noise), 7919, 2, int(k)])
            rows, n_ep = run_mixture(self.ho, s_loads, np.ones(len(s_loads), np.int64), d["perp"]["policies"],
                                     d["perp"]["weights"], q.H, n_k, self.codec, rng, action_map=self.model.legal)
            I = self.model.fisher_rows(rows)
            tr = float(np.trace(I)) / len(rows)
            rec["realized"] = {"trace_per_row": tr, "trace_err": tr / d["T_vol_per_row"] - 1.0,
                               "A": [alignment(I, u) for u in dirs]}
            rec["n_episodes"] = int(n_ep)
            self.n_rows_out += len(rows)
        rec["provider_sec"] = time.perf_counter() - t0
        self.olog.write(rec)
        self.records[k] = rec
        return rows


# ============================================================================================ one job
def jtag(j):
    seed, stream, static, method, arms = j
    return f"i{seed}_s{stream}_{'static' if static else 'dyn'}_{method}"


def job(seed, stream, static, method, arms, out_dir):
    from dsswm.evidence.reuse_switch import ShadowLedger
    from dsswm.mechanism.leverage import LinLeverage
    from dsswm.mechanism.predictor_log import PredictorLog, ResultLog
    from dsswm.streams.lin_stream import LinStreamBuilder, T_MAX_LIN, _public, run_lin_cell, sibling_run
    _install_predictor_hook()
    tag = jtag((seed, stream, static, method, arms))
    parts = out_dir / "parts"
    pp, rp, op, jp = (parts / f"{tag}_pred.jsonl", parts / f"{tag}_res.jsonl", parts / f"{tag}_orth.jsonl",
                      parts / f"{tag}.json")
    for p in (pp, rp, op):
        p.unlink(missing_ok=True)                        # an unfinished job is rerun from scratch
    plog, rlog, olog = PredictorLog(pp, fsync=True), ResultLog(rp, fsync=False), PredictorLog(op, fsync=True)
    t_job = time.perf_counter()
    b = LinStreamBuilder(seed)
    t_sel = time.perf_counter()
    b.select(static)
    sel_sec = time.perf_counter() - t_sel
    tmax = T_MAX_LIN
    lev = LinLeverage(b.lcs if static else b.lc, ref=b.lc, sigma=b.cfg["sigma"])
    lev_s = None if static else LinLeverage(b.lcs, ref=b.lc, sigma=b.cfg["sigma"])   # lock option (b)
    errs, pred_errs, samples, cells = [], [], [], {}
    cache, shadow, rows_s, sib_sec, sib_info = {}, None, None, 0.0, {}
    if any(a in ("vol", "orth") for a in arms):
        ts = time.perf_counter()
        rows_s, env_s, pad_fn, sib_info = sibling_run(b, stream, method, tmax, static, N_PROB)
        st0 = b.stream(stream, static=static, n_problems=N_PROB)
        shadow = ShadowLedger(rows_s, [q.pid for q in st0.problems], pad_fn=pad_fn)
        cache[(b.seed, stream, static, method, N_PROB, tmax)] = (shadow, env_s, sib_info)
        sib_sec = time.perf_counter() - ts
    for arm in arms:
        t0 = time.perf_counter()
        prov = None
        try:
            if arm == "orth":
                st_o = b.stream(stream, static=static, n_problems=N_PROB)
                prov = LinOrthProvider(b, stream, _public(b, st_o), st_o.init_obs, rows_s, shadow, olog,
                                       {"instance": seed, "stream": stream, "method": mlabel(method, static),
                                        "arm": "orth"}, st_o.noise_seed)
            ctx = _ctx()
            ctx.clear()
            ctx.update({"seed": seed, "stream": stream, "arm": arm, "static": static, "plog": plog, "lev": lev,
                        "lev_s": lev_s, "pred_errs": pred_errs})
            try:
                rr = run_lin_cell(b, stream, method, arm, tmax=tmax, static=static, n_problems=N_PROB,
                                  sibling_cache=cache, samples=samples, orth_provider=prov)
            finally:
                _ctx().clear()
            cell_sec = time.perf_counter() - t0
            if prov is not None:
                n_rep = sum(int(r["replay_steps"]) for r in rr)
                if int(prov.env_o.n_steps) != n_rep or prov.n_rows_out != n_rep:
                    errs.append({"tag": tag, "arm": arm, "stage": "orth_audit",
                                 "error": f"orth sibling steps {prov.env_o.n_steps} / rows out {prov.n_rows_out} "
                                          f"!= replay rows {n_rep}"})
            for r in rr:
                k = r["problem_index"]
                rec = prov.records.get(k, {}) if prov is not None else {}
                r.update({"method": mlabel(method, static), "method_base": method, "problem": int(k),
                          "cell_sec": cell_sec, "run_s": r["wall_clock_s"] - float(r.get("x_pred_sec") or 0.0),
                          "predictor_sec": float(r.get("x_pred_sec") or 0.0),
                          "orth_status": rec.get("status"), "orth_trivial": rec.get("trivial"),
                          "N_k": rec.get("N_k"), "orth_provider_sec": rec.get("provider_sec"),
                          "first_cert_step": r.get("x_first_cert_step")})
                rlog.write(r)                             # harness: after the cell (predictors already on disk)
            cells[arm] = {"cell_sec": cell_sec, "n": len(rr)}
        except Exception as e:  # noqa: BLE001 - a crashed cell is counted, never hidden
            errs.append({"tag": tag, "arm": arm, "stage": "cell", "error": repr(e),
                         "tb": traceback.format_exc()[-3000:]})
    out = {"tag": tag, "seed": seed, "stream": stream, "static": static, "method": method, "arms": list(arms),
           "errors": errs, "pred_errors": pred_errs, "samples": samples, "cells": cells, "sibling": sib_info,
           "sibling_sec": sib_sec, "select_sec": sel_sec, "job_sec": time.perf_counter() - t_job,
           "worker_pid": os.getpid()}
    tmp = jp.with_name(jp.name + ".tmp")
    tmp.write_text(json.dumps(out, default=str))
    os.replace(tmp, jp)
    return {"tag": tag, "n_err": len(errs), "n_pred_err": len(pred_errs), "sec": round(out["job_sec"], 2)}


# ============================================================================================ main-process stages
def check_instances(seeds, reserve):
    from dsswm.streams.gap_quota import fill_instances
    from dsswm.streams.lin_stream import LinStreamBuilder
    info = {}

    def ok(s):
        if s not in info:
            b = LinStreamBuilder(s)
            d, st = b.select(False), b.select(True)
            info[s] = {"quota_fail_dyn": d["quota_fail"], "quota_fail_static": st["quota_fail"],
                       "n_draws_dyn": d["n_draws"], "n_draws_static": st["n_draws"]}
        return not (info[s]["quota_fail_dyn"] or info[s]["quota_fail_static"])

    used, repl = fill_instances(seeds, ok, reserve_start=reserve)
    return used, repl, info


def lock_streams(ls_cfg):
    """frozen_items.lin_streams.streams_1_2 -> (extra streams, 'all' | 'core' | None)."""
    v = str((ls_cfg or {}).get("streams_1_2", ""))
    if "all" in v:
        return [1, 2], "all"
    if "core" in v:
        return [1, 2], "core"
    return [], None


def build_jobs(seeds, streams_core, extra_mode="core"):
    jobs = []
    for s in seeds:
        for m, arms in DYN_CELLS:
            jobs.append((s, 0, False, m, arms))
        for m, arms in STATIC_CELLS:
            jobs.append((s, 0, True, m, arms))
        for st in streams_core:
            if extra_mode == "all":
                for m, arms in DYN_CELLS:
                    jobs.append((s, st, False, m, arms))
                for m, arms in STATIC_CELLS:
                    jobs.append((s, st, True, m, arms))
                continue
            for m, arms in CORE_CELLS:
                jobs.append((s, st, False, m, arms))
    jobs.sort(key=lambda j: -(len(j[4]) + 3 * ("vol" in j[4]) + (j[3] in ("RAGE", "B1eb"))))   # longest first
    return jobs


def run_pool(jobs, out_dir, workers):
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    todo = [j for j in jobs if not (out_dir / "parts" / f"{jtag(j)}.json").exists()]
    done = len(jobs) - len(todo)
    progress(done, len(jobs), "runs", {"resumed_jobs": done})
    outs = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(job, *j, out_dir): j for j in todo}
        for f in as_completed(futs):
            try:
                r = f.result()
            except Exception as e:  # noqa: BLE001
                r = {"tag": jtag(futs[f]), "fatal": repr(e), "tb": traceback.format_exc()[-2000:]}
            outs.append(r)
            done += 1
            progress(done, len(jobs), "runs", {"last": r.get("tag")})
            log(f"job {done}/{len(jobs)} {r}")
    return outs


def merge(jobs, out_dir):
    from dsswm.mechanism.predictor_log import join, load_jsonl
    parts = out_dir / "parts"
    P, R, O, errs, perrs, meta, samples, missing = [], [], [], [], [], [], [], []
    for j in jobs:
        t = jtag(j)
        if not (parts / f"{t}.json").exists():
            missing.append(t)
            continue
        d = json.loads((parts / f"{t}.json").read_text())
        errs += d["errors"]
        perrs += d["pred_errors"]
        samples += d["samples"]
        meta.append({kk: v for kk, v in d.items() if kk not in ("errors", "samples", "pred_errors")})
        P += load_jsonl(parts / f"{t}_pred.jsonl")
        R += load_jsonl(parts / f"{t}_res.jsonl")
        O += load_jsonl(parts / f"{t}_orth.jsonl")
    for name, rows in (("predictors.jsonl", P), ("results.jsonl", R), ("orth_designs.jsonl", O)):
        with open(out_dir / name, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r, sort_keys=True, default=str) + "\n")
    try:
        Jn = join(out_dir / "predictors.jsonl", out_dir / "results.jsonl", check_order=True)
        wa = {"write_ahead_ok": True, "n_joined": len(Jn)}
    except RuntimeError as e:
        wa = {"write_ahead_ok": False, "error": str(e)}
    kf = ("instance", "stream", "method", "arm", "problem")
    pk, ok_ = {}, {}
    for r in P:
        pk.setdefault(tuple(r[x] for x in kf), r["logged_at"])
    for r in O:
        ok_.setdefault(tuple(r[x] for x in kf), r["logged_at"])
    cert = [r for r in R if r.get("status") == "CERTIFIED"]
    miss = [r for r in cert if tuple(r[x] for x in kf) not in pk]
    late = [r for r in cert if (k := tuple(r[x] for x in kf)) in pk and pk[k] > r["scored_at"]]
    orth_rows = [r for r in R if r["arm"] == "orth"]
    orth_miss = [r for r in orth_rows if tuple(r[x] for x in kf) not in ok_]
    orth_late = [r for r in orth_rows if (k := tuple(r[x] for x in kf)) in ok_ and ok_[k] > r["scored_at"]]
    wa.update({"n_certified": len(cert), "n_cert_without_predictor": len(miss), "n_cert_predictor_late": len(late),
               "n_orth_rows": len(orth_rows), "n_orth_design_records": len(O),
               "n_orth_rows_without_design": len(orth_miss), "n_orth_design_late": len(orth_late),
               "n_predictor_errors": len(perrs)})
    (out_dir / "samples").mkdir(exist_ok=True)
    (out_dir / "samples" / "cells.json").write_text(json.dumps(samples[:60], indent=1, default=str))
    if errs or missing or perrs:
        (out_dir / "errors.json").write_text(json.dumps({"errors": errs, "predictor_errors": perrs[:50],
                                                         "missing_parts": missing}, indent=1, default=str))
    return R, P, O, meta, errs, perrs, missing, wa


# ============================================================================================ analysis
def audit(rows, n0=20):
    """Billing + provenance audit per row (same invariants as tests/test_lin_stream + r3_nl_main)."""
    bad = []
    for r in rows:
        pc = r.get("prov_counts") or {}
        rep = sum(v for kk, v in pc.items() if kk.startswith("replay"))
        n_k = r["n_rounds_billed"] - n0 - r["new_env_steps"]
        a = r["arm"]
        conds = {"billing_ok": r["billing_ok"], "env_eq_billed": r["env_n_steps"] == r["n_rounds_billed"]}
        if r["status"] != "ORTH_INFEASIBLE":
            conds.update({"prov_sum_eq_total": sum(pc.values()) == r["n_rounds_total"],
                          "prov_replay_eq_replay_steps": rep == r["replay_steps"],
                          "pad_eq": pc.get("replay_vol_pad", 0) == r["replay_pad_steps"]})
            if a in ("vol", "orth"):
                conds["replay_eq_Nk"] = r["replay_steps"] == n_k
                conds["real_eq_n0_plus_new"] = pc.get("initial", 0) + pc.get("real", 0) == n0 + r["new_env_steps"]
            elif a == "off":
                conds["off_no_replay"] = rep == 0
                conds["off_drops_old"] = pc.get("initial", 0) + pc.get("real", 0) == n0 + r["new_env_steps"]
            else:
                conds["persist_no_replay"] = rep == 0
                conds["persist_sees_all_billed"] = pc.get("initial", 0) + pc.get("real", 0) == r["n_rounds_billed"]
            if a == "ev1" and r["status"] == "CERTIFIED":
                conds["ev1_min_one_new_step"] = r["new_env_steps"] >= 1
        else:
            conds["infeasible_zero_new_steps"] = r["new_env_steps"] == 0
        f = [kk for kk, v in conds.items() if not v]
        if f:
            bad.append({"instance": r["instance"], "stream": r["stream"], "method": r["method"], "arm": a,
                        "k": r["problem"], "failed": f})
    arms = sorted({r["arm"] for r in rows})
    return {"n_rows": len(rows), "n_bad": len(bad), "examples": bad[:20],
            "bad_by_arm": {a: sum(1 for b in bad if b["arm"] == a) for a in arms},
            "billing_mismatch_by_arm": {a: sum(not r["billing_ok"] for r in rows if r["arm"] == a) for a in arms},
            "replay_billed_total": sum(r["replay_steps"] for r in rows if r["arm"] not in ("vol", "orth"))}


def _f(x):
    return float("inf") if isinstance(x, str) else float(x)


def analyse(R, P, meta, seeds):
    by = defaultdict(list)
    for r in R:
        by[(r["layer"], r["method"], r["arm"])].append(r)
    sib = defaultdict(list)
    for m in meta:
        sib[(m["static"], m["method"], m["stream"])].append(m["sibling_sec"])
    cells = {}
    for (lay, m, a), rr in sorted(by.items()):
        cert = [r for r in rr if r["status"] == "CERTIFIED"]
        nt = [r for r in rr if r["gap_layer"] != "tie"]
        cells[f"{lay}|{m}|{a}"] = {
            "n": len(rr), "n_nontie": len(nt), "status_counts": dict(Counter(r["status"] for r in rr)),
            "completion": float(np.mean([r["status"] == "CERTIFIED" for r in rr])),
            "completion_nontie": float(np.mean([r["status"] == "CERTIFIED" for r in nt])) if nt else None,
            "n_zero_cost": int(sum(r["zero_cost"] for r in rr)), "n_false_cert": int(sum(r["false_cert"] for r in rr)),
            "fcr_descriptive": (sum(r["false_cert"] for r in cert) / len(cert)) if cert else None,
            "nontie_steps_total": int(sum(r[f"cost_tau{TAU_PRIMARY}"] for r in nt)),
            "median_new_steps": float(np.median([r["new_env_steps"] for r in rr])),
            "truncated_at_tmax": int(sum(r["truncated_at_tmax"] for r in rr)),
            "replay_steps_total": int(sum(r["replay_steps"] for r in rr)),
            "pad_steps_total": int(sum(r["replay_pad_steps"] for r in rr)),
            "orth_feasible_rate_nontrivial": (float(np.mean([r["status"] != "ORTH_INFEASIBLE" for r in rr
                                                              if not r.get("orth_trivial")]))
                                              if a == "orth" and any(not r.get("orth_trivial") for r in rr) else None),
            "billing_mismatch": int(sum(not r["billing_ok"] for r in rr)),
            "sec_per_problem_run": float(np.mean([r["run_s"] for r in rr])),
            "sec_per_problem_predictor": float(np.mean([r["predictor_sec"] for r in rr])),
            "sec_per_problem": float(np.mean([r["wall_clock_s"] for r in rr]))}
    for (static, m, stream), v in sib.items():          # sibling run once per job, charged to the vol cell
        c = cells.get(f"{'Lin-Static' if static else 'E1-Lin'}|{mlabel(m, static)}|vol")
        if c is not None and stream == 0 and v:
            c["sibling_sec_per_problem"] = float(np.mean(v)) / N_PROB
            c["sec_per_problem"] += c["sibling_sec_per_problem"]
    # Lambda_perp: negative control (dynamic class) vs Lin-Static
    lam = defaultdict(list)
    for p in P:
        if p.get("record") == "lin_predictors" and p.get("lambda_perp_max") is not None:
            lam[p["layer"]].append(_f(p["lambda_perp_max"]))
            if p.get("lambda_perp_static_model_max") is not None:
                lam["E1-Lin|static_model_option_b"].append(_f(p["lambda_perp_static_model_max"]))
    lam_s = {}
    for lay, v in lam.items():
        v = np.asarray(v)
        lam_s[lay] = {"n": int(len(v)), "max": float(v.max()), "median": float(np.median(v)),
                      "frac_gt_tol": float(np.mean(v > LAMBDA_TOL)), "tol": LAMBDA_TOL}
    # descriptive per-instance contrasts (pilot: pipeline check only, NO readout); non-tie cost at tau = T_max_Lin
    S = {}
    for (lay, m, a), rr in by.items():
        for i in seeds:
            sel = [r for r in rr if r["instance"] == i and r["gap_layer"] != "tie" and r["stream"] == 0]
            S[(i, lay, m, a)] = sum(r[f"cost_tau{TAU_PRIMARY}"] for r in sel) if sel else None

    def lr(i, m, a, b, lay="E1-Lin", m2=None):
        x, y = S.get((i, lay, m, a)), S.get((i, lay, m2 or m, b))
        return None if x is None or y is None else float(np.log((x + 1) / (y + 1)))
    per_inst = {}
    for i in seeds:
        d = {"A1_JPC_full_off": lr(i, "JPC-Lin", "full", "off"), "RAGE_full_off": lr(i, "RAGE", "full", "off"),
             "A3_JPC_ev1_vol": lr(i, "JPC-Lin", "ev1", "vol"), "JPC_vs_RAGE_off": lr(i, "JPC-Lin", "off", "off",
                                                                                    m2="RAGE"),
             "LinStatic_full_off": lr(i, "LinStatic:JPC-Lin", "full", "off", lay="Lin-Static")}
        d["A2_I_lin"] = (None if d["A1_JPC_full_off"] is None or d["RAGE_full_off"] is None
                         else d["A1_JPC_full_off"] - d["RAGE_full_off"])
        feas = {r["problem"] for r in by.get(("E1-Lin", "JPC-Lin", "orth"), [])
                if r["instance"] == i and r["status"] != "ORTH_INFEASIBLE" and r["gap_layer"] != "tie"}
        so = sum(r[f"cost_tau{TAU_PRIMARY}"] for r in by.get(("E1-Lin", "JPC-Lin", "orth"), [])
                 if r["instance"] == i and r["problem"] in feas)
        sv = sum(r[f"cost_tau{TAU_PRIMARY}"] for r in by.get(("E1-Lin", "JPC-Lin", "vol"), [])
                 if r["instance"] == i and r["problem"] in feas)
        d["A3o_JPC_orth_vol_constructible"] = float(np.log((so + 1) / (sv + 1))) if feas else None
        d["A3o_n_constructible_nontie"] = len(feas)
        per_inst[str(i)] = d
    return cells, lam_s, per_inst


def project(cells, meta, n_inst_full, streams_core, extra_mode="core"):
    """Lock rule: sum(sec/problem x problem slots x safety 1.2) / (4 workers x 60) + allowance."""
    slots0 = n_inst_full * N_PROB
    core = ({f"E1-Lin|{m}|{a}" for m, arms in CORE_CELLS for a in arms} if extra_mode != "all" else set(cells))
    out = []
    for c, v in cells.items():
        slots = slots0 + (len(streams_core) * slots0 if c in core else 0)
        out.append({"cell": c, "sec_per_problem": v["sec_per_problem"], "problem_slots": slots, "safety": 1.2,
                    "cpu_s": v["sec_per_problem"] * slots * 1.2, "source": "this pilot (dev 742-743, concurrent)"})
    # per-job fixed overhead (builder + quota selection + stream construction), measured
    ovh = [m["job_sec"] - sum(c["cell_sec"] for c in m["cells"].values()) - m["sibling_sec"] for m in meta]
    n_jobs_full = n_inst_full * (len(DYN_CELLS) + len(STATIC_CELLS) + len(streams_core) * (
        len(DYN_CELLS) + len(STATIC_CELLS) if extra_mode == "all" else len(CORE_CELLS)))
    ovh_s = float(np.mean(ovh)) * n_jobs_full * 1.2 if ovh else 0.0
    cpu_s = sum(c["cpu_s"] for c in out) + ovh_s
    allowance = 1.0
    return {"rule": "sum(sec/problem x slots x 1.2) / (4 x 60) + per-job overhead + 1 min allowance; vol cells carry "
                    "the sibling run; orth cells carry the provider (FW + execution)",
            "cells": out, "per_job_overhead_s_mean": float(np.mean(ovh)) if ovh else None,
            "n_jobs_full": n_jobs_full, "cpu_s": cpu_s, "allowance_min": allowance,
            "projected_min_full_chunk": cpu_s / (4 * 60) + allowance}


def update_shared_summary(entry, md):
    lockp = RES_ROOT / "pilot_summary.lock"
    with open(lockp, "a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            pj = RES_ROOT / "pilot_summary.json"
            d = json.loads(pj.read_text()) if pj.exists() else {"tasks": {}}
            d.setdefault("tasks", {})[TASK] = entry
            tmp = pj.with_name(pj.name + f".tmp{os.getpid()}")
            tmp.write_text(json.dumps(d, indent=1, ensure_ascii=False, default=str))
            os.replace(tmp, pj)
            pm = RES_ROOT / "pilot_summary.md"
            txt = pm.read_text() if pm.exists() else ""
            head = f"## {TASK} "
            if head in txt:
                s = txt.index(head)
                e = txt.find("\n## ", s + 1)
                txt = txt[:s] + md.strip() + "\n" + (txt[e:] if e >= 0 else "")
            else:
                txt = txt.rstrip() + "\n\n" + md.strip() + "\n"
            pm.write_text(txt)
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)


def update_gpu_progress(status, start_iso, wall_min, snapshot):
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
    global TASK
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", default="a", choices=sorted(CHUNKS))
    ap.add_argument("--mode", default="pilot", choices=["pilot", "full"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-gpu-progress", action="store_true")
    a = ap.parse_args()
    TASK = f"r3_lin_{a.chunk}"
    (RES_ROOT / f"{TASK}.pid").write_text(str(os.getpid()))
    (RES_ROOT / f"{TASK}_gpu_profile.json").write_text(json.dumps({
        "gpu_name": "none (CPU only)", "vram_total_mb": 0, "max_batch_size": "n/a", "vram_used_mb": 0,
        "utilization_pct": None, "note": "E1-Lin learner loop is CPU-only numpy (ellipsoid d<=12); gpu slot 3 = "
                                         "virtual CPU slot; 4 workers, OMP=1"}))
    t_all = time.perf_counter()
    start = datetime.now()
    out_dir = RES_ROOT / ("pilots" if a.mode == "pilot" else "full") / TASK
    (out_dir / "parts").mkdir(parents=True, exist_ok=True)
    status, summary = "failed", None
    try:
        lock = json.loads((WS / "plan" / "prereg_lock.json").read_text())
        lock_core, lock_mode = lock_streams(lock.get("frozen_items", {}).get("lin_streams", {}))
        extra_mode = lock_mode
        if a.mode == "full":
            from dsswm.stats.prereg import assert_locked
            lock = assert_locked()
            streams_core, extra_mode = lock_streams(lock.get("frozen_items", {}).get("lin_streams", {}))
            lo, hi = CHUNKS[a.chunk]
            seeds, reserve = list(range(lo, hi)), RESERVE[a.chunk]
        else:
            seeds, reserve, streams_core = PILOT["seeds"], DEV_RESERVE, []
        progress(0, 1, "quota_check")
        used, repl, qinfo = check_instances(seeds, reserve)
        log(f"instances {used} replacements {repl}")
        jobs = build_jobs(used, streams_core, extra_mode)
        t_runs = time.perf_counter()
        outs = run_pool(jobs, out_dir, a.workers)
        runs_sec = time.perf_counter() - t_runs
        R, P, O, meta, errs, perrs, missing, wa = merge(jobs, out_dir)
        fatal = [o for o in outs if "fatal" in o]
        au = audit(R)
        cells, lam_s, per_inst = analyse(R, P, meta, used)
        n_full = CHUNKS[a.chunk][1] - CHUNKS[a.chunk][0]
        proj = project(cells, meta, n_full, lock_core, lock_mode)
        proj["lock_projection_min"] = (lock.get("timing_projection", {}).get("per_task", {})
                                       .get("r3_lin_[a-c]", {}).get("projected_min_chosen"))
        proj["streams_core_assumed"] = lock_core
        proj["streams_1_2_mode_assumed"] = lock_mode
        proj_min = proj["projected_min_full_chunk"]
        expected = sum(len(j[4]) * N_PROB for j in jobs)
        crashes = len(fatal) + len(missing) + len(errs)
        bill_mm = sum(not r["billing_ok"] for r in R)
        lam_dyn = lam_s.get("E1-Lin", {})
        lam_sta = lam_s.get("Lin-Static", {})
        pass_flags = {"zero_crashes": crashes == 0 and len(R) == expected,
                      "billing_zero_mismatch": bill_mm == 0 and au["n_bad"] == 0 and au["replay_billed_total"] == 0,
                      "predictors_before_results_every_certification": bool(
                          wa.get("write_ahead_ok") and wa["n_cert_without_predictor"] == 0
                          and wa["n_cert_predictor_late"] == 0 and wa["n_orth_rows_without_design"] == 0
                          and wa["n_orth_design_late"] == 0),
                      "projected_full_le_55min": proj_min <= 55.0, "n_runs_ge_100": len(R) >= 100}
        diag = {"predictor_errors_zero": len(perrs) == 0,
                "lambda_perp_dynamic_le_tol": lam_dyn.get("max") is not None and lam_dyn["max"] <= LAMBDA_TOL,
                "lambda_perp_static_positive": lam_sta.get("median") is not None and lam_sta["median"] > LAMBDA_TOL,
                "lambda_perp_static_model_option_b_positive": (
                    lam_s.get("E1-Lin|static_model_option_b", {}).get("median") is not None
                    and lam_s["E1-Lin|static_model_option_b"]["median"] > LAMBDA_TOL)}
        go = all(pass_flags.values())
        # qualitative samples: one row per (layer, method, arm, gap_layer) on the first instance
        pick, seen = [], set()
        for r in R:
            key = (r["layer"], r["method"], r["arm"], r["gap_layer"])
            if key not in seen and r["instance"] == used[0] and r["stream"] == 0:
                seen.add(key)
                pick.append({k: r.get(k) for k in ("instance", "problem", "pid", "layer", "method", "arm", "gap_layer",
                                                   "true_gap", "shared", "status", "new_env_steps", "replay_steps",
                                                   "certified_policy", "true_regret", "orth_status", "N_k", "run_s",
                                                   "x_lambda_perp", "prov_counts")})
        (out_dir / "samples" / "qualitative_rows.json").write_text(json.dumps(pick[:80], indent=1, default=str))
        summary = {
            "task": TASK, "mode": a.mode, "seeds": used, "replacements": repl, "reserve_start": reserve,
            "quota_info": qinfo, "streams_core": streams_core,
            "streams_1_2_mode": extra_mode, "n_problems": N_PROB, "tmax": TAU_PRIMARY,
            "eps_lin": 0.05, "n_jobs": len(jobs), "n_runs": len(R), "n_expected_runs": expected,
            "n_predictor_lines": len(P), "n_orth_design_records": len(O),
            "crashes": {"fatal_jobs": fatal, "missing_parts": missing, "n_errors": len(errs), "errors_head": errs[:5]},
            "predictor_errors": {"n": len(perrs), "head": perrs[:3]},
            "billing_mismatch": bill_mm, "audit": au, "write_ahead": wa, "cells": cells, "lambda_perp": lam_s,
            "per_instance_descriptive": per_inst, "runs_wall_sec": runs_sec,
            "timing_projection": proj, "pass_criteria": pass_flags, "diagnostics": diag,
            "go_no_go": "GO" if go else "NO_GO",
            "scientific_readout": "none (pilot = smoke + timing only; dev seeds; contrasts are descriptive)",
            "lock_status_at_run": lock.get("status"), "code_sha256": code_sha(),
            "note": "并发运行（与其他 r3 任务共享 20 核，4 worker，OMP=1），计时偏高；CPU only",
            "start_time": start.isoformat(), "end_time": datetime.now().isoformat(),
            "total_wall_sec": time.perf_counter() - t_all}
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
        log(f"summary: go={go} flags={pass_flags} diag={diag} proj={proj_min:.1f} min runs={len(R)}")
        if a.mode == "pilot":
            entry = {"candidate_id": "cand_f", "go_no_go": "GO" if go else "NO_GO", "pass_criteria": pass_flags,
                     "diagnostics": diag, "projected_full_min": proj_min, "n_runs": len(R),
                     "lambda_perp": lam_s, "scientific_readout": "none (smoke + timing)"}
            md = (f"## {TASK} (pilot, dev 742-743, stream 0, 18 cells x 15 problems)\n"
                  f"- 结论: **{'GO' if go else 'NO_GO'}**；runs={len(R)}/{expected}，崩溃={crashes}，"
                  f"计费不一致={bill_mm}，审计异常={au['n_bad']}\n"
                  f"- write-ahead: 认证 {wa.get('n_certified')} 次，缺预测记录 {wa.get('n_cert_without_predictor')}，"
                  f"晚于评分 {wa.get('n_cert_predictor_late')}；orth 设计记录 {len(O)}\n"
                  f"- Λ̂⊥: Lin 动态类 max={lam_dyn.get('max')}，Lin-Static median={lam_sta.get('median')}\n"
                  f"- 单分块 full 预计 {proj_min:.1f} min（lock 预估 {proj['lock_projection_min']}）；并发运行，计时偏高\n")
            update_shared_summary(entry, md)
        status = "success" if go else "failed"
        mark_done(status, f"{TASK} {a.mode}: GO={go} runs={len(R)} proj={proj_min:.1f}min")
        return summary
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        log(tb)
        (out_dir / "fatal.txt").write_text(tb)
        mark_done("failed", f"{TASK} {a.mode} crashed: {e!r}")
        raise
    finally:
        if not a.no_gpu_progress:
            update_gpu_progress(status, start.isoformat(), (time.perf_counter() - t_all) / 60,
                                {"mode": a.mode, "layer": "E1-Lin 3x3 sigma=1.5 eps=0.05 + Lin-Static",
                                 "seeds": PILOT["seeds"] if a.mode == "pilot" else list(range(*CHUNKS[a.chunk])),
                                 "n_problems": N_PROB, "tmax": TAU_PRIMARY, "n_cells_stream0": 18,
                                 "cpu_workers": a.workers, "gpu_model": "none (CPU only)", "gpu_count": 0,
                                 "concurrent": "other r3 tasks"})


if __name__ == "__main__":
    main()
