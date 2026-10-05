"""r3_nl_main_[a-h]: NL-R0 main reuse x method contrast (A1, A2, A3, A3o, Q2, H0, billing gate). Methodology 2.1 / 5.1.

Design  NL-R0 (E1-NL-S G_1, realisable), gap-quota streams (tie / near / clear x 5), {JPC, B3} x
        {off, vol, orth, ev1, full}, CRN: every arm of one (instance, stream, method) runs on a fresh copy of the same
        platform (same noise seed, same n0 rows, same problem order, acquisition rng [seed, noise, method_code]).
  vol   sibling platform (noise + 7919, same theta*) run by the SAME method in the `full` arm (r3_harness.sibling_run);
        the shadow ledger is built once per (instance, stream, method) and shared by vol and orth (deterministic prefix
        function of (k, N_k), pad pool drawn sequentially on the sibling platform).
  orth  learner-side provider (identical to r3_p3): xi_perp on the reachable occupancy polytope at theta_hat of the
        real ledger, Fisher-trace matched to the vol rows shadow.take(k, N_k), A <= 0.05 for both pre-registered
        pairs; executed on a SECOND sibling platform (noise + 2 x 7919); infeasible -> ORTH_INFEASIBLE (never
        downgraded). Design records -> orth_designs.jsonl (learner side, write-ahead).
Logging (methodology s.1 write separation)
  predictors.jsonl  learner process, fsync'ed, at the end of every problem BEFORE the harness scores the stream:
                    full / ev1 arms: Lambda_hat_perp (S1), S2 = 1/rho*, A_k, placebo A_k, phi_perp, Rem bar and the
                    trivial predictors (gap_hat/eps, log tr I^-1, eta_hat_gof); off / vol / orth arms: a light
                    certification record (status, theta_hat, set size, gap_hat/eps) so that every certification has a
                    write-ahead line.
  results.jsonl     harness, after the stream (truth scoring). JOIN KEY (instance, stream, method, arm, problem).
Billing  ReuseSwitch.end_problem asserts the billing invariants after every problem (mismatches recorded, not hidden);
         a per-row provenance audit is repeated in the merge.

Pilot  smoke + timing only (NO scientific readout): dev seeds 734-735, stream 0, 6 problems, {JPC,B3} x 5 arms
       (120 runs); projects the full wall-clock of one chunk (lock rule) and lists the downscale option if > 55 min.
Full   chunk a..h = eval 10000-10014, ..., 10105-10119 x 3 streams x 15 problems x 10 cells (6750 runs per chunk;
       orth on `orth_streams` streams as chosen by the lock); requires the v3 lock (assert_locked()).
       Resumable: one part file per (instance, stream, method) job.

Usage: run_r3_nl_main.py --chunk a --mode {pilot,full} [--workers 4]
Concurrent run (shares 20 cores + the RTX 4090 with 3 other round-3 tasks): timings are "concurrent".
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import fcntl  # noqa: E402
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
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
WS = HERE.parents[1]
RES_ROOT = WS / "exp" / "results"
CACHE = WS / "exp" / "cache" / "jtables_r3"
EPS, DELTA = 0.02, 0.05
TMAX = 6000
N0 = 20
TAU = 3000
METHODS = ("JPC", "B3")
ARMS = ("off", "vol", "orth", "ev1", "full")
FULL_PRED_ARMS = ("full", "ev1")
ORTH_SIB_OFFSET = 2 * 7919
CHUNKS = {c: (10000 + 15 * i, 10015 + 15 * i) for i, c in enumerate("abcdefgh")}
RESERVE = {c: 10800 + 25 * i for i, c in enumerate("abcdefgh")}   # quota_fail replacement, unallocated, disjoint
DEV_RESERVE = 761
PILOT = {"seeds": [734, 735], "streams": [0], "n_problems": 6}
CODE_FILES = ["dsswm/evidence/reuse_switch.py", "dsswm/evidence/orth_design.py", "dsswm/baselines/switched_nl.py",
              "dsswm/streams/gap_quota.py", "dsswm/streams/r3_harness.py", "dsswm/mechanism/leverage.py",
              "dsswm/mechanism/predictor_log.py", "run_r3_p3_orth_replay_feasibility.py",
              "run_r3_p4_t0_mechanism_gate.py", "run_r3_nl_main.py"]
TASK = "r3_nl_main_a"                          # set in main()


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


# ============================================================================================ worker context
_W: dict = {}


def _ctx():
    if "pub" not in _W:
        torch.set_num_threads(1)
        from dsswm.baselines.switched_nl import PublicNL
        from dsswm.evidence.orth_design import NLOrthModel
        from dsswm.exact.nl_propagate import NLPropagator
        from dsswm.mechanism.leverage import NLLeverage
        from dsswm.models.nl_class import NLClass
        from dsswm.streams.gap_quota import QuotaBuilder
        from dsswm.streams.generator import DEFAULT_NL_S_GRID
        import run_r3_p4_t0_mechanism_gate as P4
        P4.EPS = EPS
        ncl = NLClass(DEFAULT_NL_S_GRID, device="cpu")
        b = QuotaBuilder(PILOT["seeds"][0], ncl, None, CACHE)
        prop = NLPropagator(2, 2, 2, b.aspace, 1.0, 0.3, device="cpu")
        TH = NLLeverage(prop).theta_matrix(ncl.np_params)
        lev = NLLeverage(prop, theta_grid=TH, eps=EPS)
        _W.update(ncl=ncl, prop=prop, TH=TH, lev=lev, model=NLOrthModel(prop, lev),
                  pub=PublicNL(prop, ncl, [], [], 1.0, 2, EPS, DELTA), builders={})
    return _W


def _builder(seed):
    from dsswm.streams.gap_quota import QuotaBuilder
    W = _ctx()
    if seed not in W["builders"]:
        W["builders"] = {seed: QuotaBuilder(seed, W["ncl"], None, CACHE, prop_cpu=W["prop"])}
    return W["builders"][seed]


# ============================================================================================ one (instance, stream, method)
def job(seed, stream, method, n_problems, arms, out_dir):
    from dsswm.evidence.reuse_switch import ShadowLedger
    from dsswm.mechanism.predictor_log import PredictorLog, ResultLog
    from dsswm.streams.r3_harness import sibling_run
    from run_r3_p3_orth_replay_feasibility import NLOrthProvider  # noqa: F401  (imported in run_arm)
    W = _ctx()
    tag = f"i{seed}_s{stream}_{method}"
    parts = out_dir / "parts"
    pp, rp, op, jp = (parts / f"{tag}_pred.jsonl", parts / f"{tag}_res.jsonl", parts / f"{tag}_orth.jsonl",
                      parts / f"{tag}.json")
    for p in (pp, rp, op):
        p.unlink(missing_ok=True)                       # an unfinished job is rerun from scratch
    plog, rlog, olog = PredictorLog(pp, fsync=True), ResultLog(rp, fsync=False), PredictorLog(op, fsync=True)
    t_job = time.perf_counter()
    b = _builder(seed)
    sel = b.select("quota")
    order = np.random.default_rng([seed, 45, stream]).permutation(len(sel["base"]))
    ov = [o["cos_max"] for o in b.overlap([sel["base"][i] for i in order])][:n_problems]
    # ---- sibling platform (same method, `full` arm): shadow ledger shared by vol and orth
    shadow, sib_info, sib_sec = None, {}, 0.0
    if any(a in ("vol", "orth") for a in arms):
        ts = time.perf_counter()
        st0 = b.stream(stream, "quota", n_problems=n_problems)
        pub0 = W["pub"].with_problems(st0.problems, st0.J)
        rows_s, _env_s, pad_fn, sib_info = sibling_run(b, pub0, stream, "quota", method, TMAX, n_problems)
        shadow = ShadowLedger(rows_s, [q.pid for q in st0.problems], pad_fn=pad_fn)
        sib_sec = time.perf_counter() - ts
    errs, samples, cell_meta = [], [], {}
    for arm in arms:
        try:
            cell_meta[arm] = run_arm(b, seed, stream, method, arm, n_problems, shadow, ov, plog, rlog, olog,
                                     samples, errs)
        except Exception as e:  # noqa: BLE001 - a crashed cell is counted, never hidden
            errs.append({"tag": tag, "arm": arm, "stage": "cell", "error": repr(e),
                         "tb": traceback.format_exc()[-3000:]})
    out = {"tag": tag, "seed": seed, "stream": stream, "method": method, "arms": list(arms), "errors": errs,
           "samples": samples, "cells": cell_meta, "sibling": sib_info, "sibling_sec": sib_sec,
           "vol_pad_pool": None if shadow is None else len(shadow.pad_pool),
           "job_sec": time.perf_counter() - t_job, "worker_pid": os.getpid()}
    tmp = jp.with_name(jp.name + ".tmp")
    tmp.write_text(json.dumps(out, default=str))
    os.replace(tmp, jp)
    return {"tag": tag, "n_err": len(errs), "sec": out["job_sec"]}


def run_arm(b, seed, stream, method, arm, n_problems, shadow, ov, plog, rlog, olog, samples, errs):
    from dsswm.baselines.switched_nl import solve
    from dsswm.certify.minimax_enum import certify_minimax
    from dsswm.evidence.reuse_switch import BillingError, ReuseSwitch, score_truncations, TAUS
    from dsswm.mechanism.leverage import theta_hat_index
    from dsswm.streams.gap_quota import make_env
    from dsswm.streams.r3_harness import _rng
    from run_r3_p3_orth_replay_feasibility import NLOrthProvider
    from run_r3_p4_t0_mechanism_gate import learner_record
    W = _ctx()
    t_cell = time.perf_counter()
    st = b.stream(stream, "quota", n_problems=n_problems)            # fresh platform copy (CRN)
    pub = W["pub"].with_problems(st.problems, st.J)
    env = st.env
    h = env.handle()
    prov, env_o = None, None
    if arm == "orth":
        env_o = make_env(seed, b.truth, st.noise_seed + ORTH_SIB_OFFSET)     # harness: second sibling platform
        prov = NLOrthProvider(W, pub, shadow, env_o.handle(), h, olog,
                              {"instance": seed, "stream": stream, "method": method, "arm": "orth"}, seed,
                              st.noise_seed)
    sw = ReuseSwitch(arm, pub.make_set_factory(method), st.init_obs, shadow=shadow if arm == "vol" else None,
                     orth_provider=prov)
    if prov is not None:
        prov.bind(sw)
    rng = _rng(seed, st.noise_seed, method)
    keep = []
    for k, q in enumerate(st.problems):
        key = {"instance": seed, "stream": stream, "method": method, "arm": arm, "problem": k}
        t0 = time.perf_counter()
        try:
            lr = sw.begin_problem(k, q.pid, context={"stream": stream, "kind": "quota"})
        except Exception as e:  # noqa: BLE001 - provider crash: recorded, problem counted infeasible
            errs.append({**key, "stage": "begin", "error": repr(e), "tb": traceback.format_exc()[-2500:]})
            sw.accounts[-1].orth_feasible = False
            lr = None
        if lr is None:
            res = {"status": "ORTH_INFEASIBLE", "pi": None, "steps": 0, "extra": {}}
        else:
            res = solve(pub, method, k, sw, lr, h, rng, TMAX)
        wall = time.perf_counter() - t0
        # ---------------------------------------------------- learner side, write-ahead (before any truth scoring)
        t1 = time.perf_counter()
        pred_ok, k_hat, set_size = False, None, None
        if lr is not None:
            try:
                mask = lr.mask().numpy().astype(bool)
                set_size = int(mask.sum())
                k_hat = theta_hat_index(lr.inner.cum.cpu().numpy(), mask)
                Jl = pub.J[k]
                Jh = np.sort(Jl[k_hat])[::-1]
                base = {**key, "pid": q.pid, "status": res["status"], "new_steps": int(res["steps"]),
                        "theta_hat": int(k_hat), "set_size": set_size, "n_rows_in_set": len(lr.obs),
                        "n_replay_in_set": int(lr.n_replay),
                        "gap_hat_over_eps": float((Jh[0] - Jh[1]) / EPS) if len(Jh) > 1 else None}
                if arm in FULL_PRED_ARMS:
                    pi_hat, pi_src = res["pi"], "certificate"
                    if pi_hat is None and mask.any():
                        pi_hat, pi_src = certify_minimax(pub.Reg[k], mask, EPS, pub.top_m)["pi"], "minimax_on_mask"
                    if pi_hat is None:
                        pi_hat, pi_src = int(np.argmax(Jl[k_hat])), "argmax_theta_hat"
                    r_bar = float(res["extra"].get("r_bar_end", float("nan")))
                    pr = learner_record(W["lev"], W["TH"], q, Jl, list(lr.obs), mask, k_hat, int(pi_hat), r_bar,
                                        with_rem=True)
                    plog.write({**base, "record": "full_predictors", "pi_hat": int(pi_hat), "pi_hat_source": pi_src,
                                **{kk: v for kk, v in pr.items() if kk != "pairs"},
                                "pairs": [{kk: v for kk, v in p.items() if kk != "need_data"} |
                                          {"n_need_data": len(p["need_data"])} for p in pr["pairs"]]})
                else:
                    plog.write({**base, "record": "light", "pi_hat": res["pi"]})
                pred_ok = True
            except Exception as e:  # noqa: BLE001
                errs.append({**key, "stage": "predictor", "error": repr(e), "tb": traceback.format_exc()[-2500:]})
        t_pred = time.perf_counter() - t1
        billing_ok, berr = True, None
        try:
            acc = sw.end_problem(env.n_steps)
        except BillingError as e:                               # recorded, never silently dropped
            billing_ok, berr = False, str(e)
            acc = sw.accounts[-1]
            sw._k = sw._pid = None
            sw._new = 0
        keep.append({"k": k, "res": res, "wall": wall, "t_pred": t_pred, "pred_ok": pred_ok, "k_hat": k_hat,
                     "set_size": set_size, "billing_ok": billing_ok, "billing_error": berr, "acc": acc,
                     "env_n_steps": int(env.n_steps)})
    orth_sib_steps = None
    if env_o is not None:
        orth_sib_steps = int(env_o.n_steps)
        n_rep = sum(int(kr["acc"].replay_steps) for kr in keep)
        if orth_sib_steps != n_rep:
            errs.append({"instance": seed, "stream": stream, "method": method, "arm": arm, "stage": "orth_audit",
                         "error": f"orth sibling steps {orth_sib_steps} != replay rows handed out {n_rep}"})
    # ---------------------------------------------------- harness scoring (after the stream)
    ti = st.theta_index
    for kr in keep:
        k, res, acc = kr["k"], kr["res"], kr["acc"]
        q = st.problems[k]
        hm = st.harness[k]
        Jt = st.J[k][ti]
        pi = res["pi"]
        cert = res["status"] == "CERTIFIED"
        regret = float(Jt.max() - Jt[pi]) if pi is not None else None
        rec = prov.records.get(k, {}) if prov is not None else {}
        row = {"instance": seed, "stream": stream, "method": method, "arm": arm, "problem": k, "problem_index": k,
               "kind": "quota", "noise_seed": st.noise_seed, "pid": q.pid, "layer": "NL-R0",
               "gap_layer": hm["gap_layer"], "true_gap": hm["true_gap"], "true_gap_orig": hm["true_gap_orig"],
               "status": res["status"], "new_env_steps": int(res["steps"]), "replay_steps": int(acc.replay_steps),
               "replay_pad_steps": int(acc.replay_pad_steps), "n_rounds_total": int(acc.n_rounds_total),
               "n_rounds_billed": int(acc.n_rounds_billed), "env_n_steps": kr["env_n_steps"],
               "billing_ok": kr["billing_ok"], "billing_error": kr["billing_error"], "censored": not cert,
               "zero_cost": bool(cert and res["steps"] == 0), "certified_policy": pi, "true_regret": regret,
               "false_cert": bool(cert and regret is not None and regret > EPS), "set_size": kr["set_size"],
               "orth_feasible": acc.orth_feasible, "orth_status": rec.get("status"), "orth_trivial": rec.get("trivial"),
               "N_k": rec.get("N_k"), "prov_counts": dict(acc.prov_counts), "cos_overlap": ov[k],
               "mu_flip": None,   # NL-R0 is realisable (theta* on the grid): the hazard-zone mu_flip is not defined
               "predictor_logged": kr["pred_ok"], "wall_clock_s": kr["wall"], "predictor_sec": kr["t_pred"],
               "orth_provider_sec": rec.get("provider_sec"), "eps": EPS, "delta": DELTA, "tmax": TMAX,
               "first_cert_step": res["extra"].get("first_cert_step")}
        if kr["k_hat"] is not None:
            Jh = st.J[k][kr["k_hat"]]
            row["eta_arg"] = float(Jt.max() - Jt[int(np.argmax(Jh))])
            row["eta_dec"] = float(np.max(np.abs(Jh - Jt)))
            row["theta_hat_is_truth"] = bool(kr["k_hat"] == ti)
        else:
            row["eta_arg"] = row["eta_dec"] = row["theta_hat_is_truth"] = None
        row.update(score_truncations(int(res["steps"]), cert, TAUS))
        rlog.write(row)
        if k < 2 and len(samples) < 40:
            samples.append({"instance": seed, "stream": stream, "method": method, "arm": arm, "problem": k,
                            "problem_public": q.public_dict(), "J_true": [round(float(x), 6) for x in Jt],
                            "row": {kk: row.get(kk) for kk in ("status", "new_env_steps", "replay_steps",
                                                               "certified_policy", "true_regret", "gap_layer",
                                                               "set_size", "orth_status", "prov_counts")}})
    return {"cell_sec": time.perf_counter() - t_cell, "n": len(keep), "orth_sibling_steps": orth_sib_steps}


# ============================================================================================ main-process stages
def tables(seeds, reserve):
    """GPU: class J tables of the candidate pools (cached), quota selection, deterministic replacement."""
    from dsswm.exact.nl_propagate import NLPropagator
    from dsswm.models.nl_class import NLClass
    from dsswm.streams.gap_quota import QuotaBuilder, fill_instances
    from dsswm.streams.generator import DEFAULT_NL_S_GRID
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    ncl = NLClass(DEFAULT_NL_S_GRID, device=dev)
    b0 = QuotaBuilder(seeds[0], ncl, None, CACHE)
    propg = NLPropagator(2, 2, 2, b0.aspace, 1.0, 0.3, device=dev)
    propc = NLPropagator(2, 2, 2, b0.aspace, 1.0, 0.3, device="cpu")
    t0 = time.perf_counter()
    info, ok = {}, {}

    def check(s):
        if s not in ok:
            b = QuotaBuilder(s, ncl, propg, CACHE, prop_cpu=propc)
            sel = b.select("quota")
            ok[s] = not sel["quota_fail"]
            info[s] = {"quota_fail": sel["quota_fail"], "n_draws": sel["n_draws"], "layer_counts": sel["layer_counts"],
                       "tables_computed": b.n_table_computed, "table_sec": b.sec_tables}
        return ok[s]

    used, repl = fill_instances(seeds, check, reserve_start=reserve)
    vram = torch.cuda.max_memory_allocated() / 2 ** 20 if dev == "cuda" else 0.0
    tot = torch.cuda.get_device_properties(0).total_memory / 2 ** 20 if dev == "cuda" else 0
    prof = {"gpu_name": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu", "vram_total_mb": tot,
            "max_batch_size": "n/a (class J tables: one problem x |Theta|=13824 per call; no batch dimension)",
            "vram_used_mb": vram, "utilization_pct": (100.0 * vram / tot) if tot else None,
            "note": "GPU only for class J tables of candidate problems (cached); JPC/B3 loops, FW/DP orth designs, "
                    "predictors and harness scoring on CPU (4 workers, OMP=1); 并发运行（与其他 3 个任务共享 4090 与 20 核）"}
    (RES_ROOT / f"{TASK}_gpu_profile.json").write_text(json.dumps(prof))
    del ncl, propg
    if dev == "cuda":
        torch.cuda.empty_cache()
    return used, repl, info, {"sec": time.perf_counter() - t0, "vram_peak_mb": vram, "device": dev}


def jtag(j):
    return f"i{j[0]}_s{j[1]}_{j[2]}"


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
    P, R, O, errs, meta, samples, missing = [], [], [], [], [], [], []
    for j in jobs:
        t = jtag(j)
        if not (parts / f"{t}.json").exists():
            missing.append(t)
            continue
        d = json.loads((parts / f"{t}.json").read_text())
        errs += d["errors"]
        samples += d["samples"]
        meta.append({kk: v for kk, v in d.items() if kk not in ("errors", "samples")})
        P += load_jsonl(parts / f"{t}_pred.jsonl")
        R += load_jsonl(parts / f"{t}_res.jsonl")
        O += load_jsonl(parts / f"{t}_orth.jsonl")
    for name, rows in (("predictors.jsonl", P), ("results.jsonl", R), ("orth_designs.jsonl", O)):
        with open(out_dir / name, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r, sort_keys=True) + "\n")
    try:
        Jn = join(out_dir / "predictors.jsonl", out_dir / "results.jsonl", check_order=True)
        wa = {"write_ahead_ok": True, "n_joined": len(Jn)}
    except RuntimeError as e:
        wa = {"write_ahead_ok": False, "error": str(e)}
    kf = ("instance", "stream", "method", "arm", "problem")
    pk = {}
    for r in P:
        pk.setdefault(tuple(r[x] for x in kf), r["logged_at"])
    ok_ = {}
    for r in O:
        ok_.setdefault(tuple(r[x] for x in kf), r["logged_at"])
    cert = [r for r in R if r.get("status") == "CERTIFIED"]
    miss = [r for r in cert if tuple(r[x] for x in kf) not in pk]
    late = [r for r in cert if (k := tuple(r[x] for x in kf)) in pk and pk[k] > r["scored_at"]]
    full_cert = [r for r in cert if r["arm"] in FULL_PRED_ARMS]
    full_miss = [r for r in full_cert if tuple(r[x] for x in kf) not in pk]
    orth_rows = [r for r in R if r["arm"] == "orth"]
    orth_late = [r for r in orth_rows if (k := tuple(r[x] for x in kf)) in ok_ and ok_[k] > r["scored_at"]]
    wa.update({"n_certified": len(cert), "n_cert_without_predictor": len(miss), "n_cert_predictor_late": len(late),
               "n_certified_full_ev1": len(full_cert), "n_full_ev1_cert_without_full_predictors": len(full_miss),
               "n_orth_rows": len(orth_rows), "n_orth_design_records": len(O),
               "n_orth_design_late": len(orth_late)})
    (out_dir / "samples").mkdir(exist_ok=True)
    (out_dir / "samples" / "cells.json").write_text(json.dumps(samples[:60], indent=1, default=str))
    if errs or missing:
        (out_dir / "errors.json").write_text(json.dumps({"errors": errs, "missing_parts": missing}, indent=1,
                                                        default=str))
    return R, P, O, meta, errs, missing, wa


# ============================================================================================ analysis
def audit(rows):
    """Billing + provenance audit per row (same invariants as r3_p1, plus orth)."""
    bad = []
    for r in rows:
        pc = r["prov_counts"] or {}
        rep = sum(v for kk, v in pc.items() if kk.startswith("replay"))
        n_k = r["n_rounds_billed"] - N0 - r["new_env_steps"]
        conds = {"billing_ok": r["billing_ok"], "env_eq_billed": r["env_n_steps"] == r["n_rounds_billed"]}
        if r["status"] != "ORTH_INFEASIBLE":
            conds.update({"prov_sum_eq_total": sum(pc.values()) == r["n_rounds_total"],
                          "prov_replay_eq_replay_steps": rep == r["replay_steps"],
                          "pad_eq": pc.get("replay_vol_pad", 0) == r["replay_pad_steps"]})
            if r["arm"] in ("vol", "orth"):
                conds["replay_eq_Nk"] = r["replay_steps"] == n_k
                conds["real_eq_n0_plus_new"] = pc.get("initial", 0) + pc.get("real", 0) == N0 + r["new_env_steps"]
            elif r["arm"] == "off":
                conds["off_no_replay"] = rep == 0
                conds["off_drops_old"] = pc.get("initial", 0) + pc.get("real", 0) == N0 + r["new_env_steps"]
            else:
                conds["persist_no_replay"] = rep == 0
                conds["persist_sees_all_billed"] = pc.get("initial", 0) + pc.get("real", 0) == r["n_rounds_billed"]
            if r["arm"] == "ev1" and r["status"] == "CERTIFIED":
                conds["ev1_min_one_new_step"] = r["new_env_steps"] >= 1
        else:
            conds["infeasible_zero_new_steps"] = r["new_env_steps"] == 0
        f = [kk for kk, v in conds.items() if not v]
        if f:
            bad.append({"instance": r["instance"], "stream": r["stream"], "method": r["method"], "arm": r["arm"],
                        "k": r["problem"], "failed": f})
    return {"n_rows": len(rows), "n_bad": len(bad), "examples": bad[:20],
            "bad_by_arm": {a: sum(1 for b in bad if b["arm"] == a) for a in ARMS},
            "billing_mismatch_by_arm": {a: sum(not r["billing_ok"] for r in rows if r["arm"] == a) for a in ARMS},
            "replay_billed_total": sum(r["replay_steps"] for r in rows if r["arm"] not in ("vol", "orth"))}


def analyse(R, meta, seeds):
    by = defaultdict(list)
    for r in R:
        by[(r["method"], r["arm"])].append(r)
    sib = defaultdict(list)          # sibling run seconds per (method), attributed once per job
    for m in meta:
        sib[m["method"]].append(m["sibling_sec"])
    cells = {}
    for (m, a), rr in sorted(by.items()):
        cert = [r for r in rr if r["status"] == "CERTIFIED"]
        nt = [r for r in rr if r["gap_layer"] != "tie"]
        sec = [r["wall_clock_s"] + r["predictor_sec"] for r in rr]
        cells[f"{m}|{a}"] = {
            "n": len(rr), "n_nontie": len(nt), "status_counts": dict(Counter(r["status"] for r in rr)),
            "n_certified": len(cert), "n_zero_cost": sum(r["zero_cost"] for r in rr),
            "n_false_cert": sum(r["false_cert"] for r in rr),
            "fcr_descriptive": (sum(r["false_cert"] for r in cert) / len(cert)) if cert else None,
            "nontie_steps_tau3000_total": int(sum(r["cost_tau3000"] for r in nt)),
            "new_steps_total": int(sum(r["new_env_steps"] for r in rr)),
            "median_new_steps": float(np.median([r["new_env_steps"] for r in rr])),
            "replay_steps_total": int(sum(r["replay_steps"] for r in rr)),
            "pad_steps_total": int(sum(r["replay_pad_steps"] for r in rr)),
            "orth_feasible_rate_nontrivial": (float(np.mean([bool(r["orth_feasible"]) for r in rr
                                                              if not r.get("orth_trivial")]))
                                              if a == "orth" and any(not r.get("orth_trivial") for r in rr) else None),
            "billing_mismatch": sum(not r["billing_ok"] for r in rr),
            "sec_per_problem_method": float(np.mean([r["wall_clock_s"] for r in rr])),
            "sec_per_problem_predictor": float(np.mean([r["predictor_sec"] for r in rr])),
            "sec_per_problem": float(np.mean(sec))}
    n_prob = max(1, len({(r["instance"], r["stream"], r["problem"]) for r in R}))
    sib_per_problem = {m: (sum(v) / n_prob) for m, v in sib.items()}
    for m in METHODS:                # the sibling run is executed once per job and is charged to the vol cell
        c = cells.get(f"{m}|vol")
        if c is not None:
            c["sibling_sec_per_problem"] = sib_per_problem.get(m, 0.0)
            c["sec_per_problem"] += sib_per_problem.get(m, 0.0)
    # descriptive per-instance contrasts (pilot: pipeline check only, NO readout)
    S = {}
    for (m, a), rr in by.items():
        for i in seeds:
            sel = [r for r in rr if r["instance"] == i and r["gap_layer"] != "tie"]
            S[(i, m, a)] = sum(r["cost_tau3000"] for r in sel) if sel else None

    def lr(i, m, a, b):
        x, y = S.get((i, m, a)), S.get((i, m, b))
        return None if x is None or y is None else float(np.log((x + 1) / (y + 1)))
    per_inst = {}
    for i in seeds:
        d = {"A1_JPC_full_off": lr(i, "JPC", "full", "off"), "B3_full_off": lr(i, "B3", "full", "off"),
             "A3_JPC_ev1_vol": lr(i, "JPC", "ev1", "vol"), "A3o_JPC_orth_vol": lr(i, "JPC", "orth", "vol")}
        d["A2_I"] = (None if d["A1_JPC_full_off"] is None or d["B3_full_off"] is None
                     else d["A1_JPC_full_off"] - d["B3_full_off"])
        # A3o is defined on orth-constructible problems only (ORTH_INFEASIBLE rows are excluded from both arms)
        feas = {(r["stream"], r["problem"]) for r in by.get(("JPC", "orth"), [])
                if r["instance"] == i and r["status"] != "ORTH_INFEASIBLE" and r["gap_layer"] != "tie"}
        so = sum(r["cost_tau3000"] for r in by.get(("JPC", "orth"), []) if r["instance"] == i
                 and (r["stream"], r["problem"]) in feas)
        sv = sum(r["cost_tau3000"] for r in by.get(("JPC", "vol"), []) if r["instance"] == i
                 and (r["stream"], r["problem"]) in feas)
        d["A3o_JPC_orth_vol"] = float(np.log((so + 1) / (sv + 1))) if feas else None
        d["A3o_n_constructible_nontie"] = len(feas)
        per_inst[str(i)] = d
    return cells, per_inst


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


def p1_table_sec_per_seed():
    """Fresh-pool GPU table time per instance measured by r3_p1 pilot (dev 721-727; 720 was cached)."""
    try:
        s = json.loads((RES_ROOT / "pilots" / "r3_p1_reuse_ablation_nl" / "summary.json").read_text())
        v = [x["table_sec"] for x in s["quota"].values() if x.get("table_sec")]
        return float(np.mean(v)) if v else 2.0
    except Exception:  # noqa: BLE001
        return 2.0


def main():
    global TASK
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", default="a", choices=sorted(CHUNKS))
    ap.add_argument("--mode", default="pilot", choices=["pilot", "full"])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    TASK = f"r3_nl_main_{a.chunk}"
    (RES_ROOT / f"{TASK}.pid").write_text(str(os.getpid()))
    t_all = time.perf_counter()
    start = datetime.now()
    out_dir = RES_ROOT / ("pilots" if a.mode == "pilot" else "full") / TASK
    (out_dir / "parts").mkdir(parents=True, exist_ok=True)
    try:
        lock = json.loads((WS / "plan" / "prereg_lock.json").read_text())
        orth_streams = 3
        if a.mode == "full":
            from dsswm.stats.prereg import assert_locked
            lock = assert_locked()
            ds = lock.get("frozen_items", {}).get("downscale_decisions", {}).get("r3_nl_main_[a-h]", [])
            if any("orth" in str(x) and "stream 0" in str(x) for x in ds):
                orth_streams = 1
            lo, hi = CHUNKS[a.chunk]
            seeds, streams, n_problems, reserve = list(range(lo, hi)), [0, 1, 2], 15, RESERVE[a.chunk]
        else:
            seeds, streams, n_problems, reserve = PILOT["seeds"], PILOT["streams"], PILOT["n_problems"], DEV_RESERVE
            ds = []
        progress(0, 1, "tables")
        used, repl, qinfo, tinfo = tables(seeds, reserve)
        log(f"tables {tinfo} replacements {repl}")
        jobs = []
        for s in used:
            for st in streams:
                for m in METHODS:
                    arms = tuple(x for x in ARMS if x != "orth" or st < orth_streams)
                    jobs.append((s, st, m, n_problems, arms))
        jobs.sort(key=lambda j: (j[2] != "B3", j[0], j[1]))           # B3 jobs are slower: start them first
        t_runs = time.perf_counter()
        outs = run_pool(jobs, out_dir, a.workers)
        runs_sec = time.perf_counter() - t_runs
        R, P, O, meta, errs, missing, wa = merge(jobs, out_dir)
        fatal = [o for o in outs if "fatal" in o]
        au = audit(R)
        cells, per_inst = analyse(R, meta, used)
        # ---- projection of one full chunk (lock rule): sum(sec/problem x slots x 1.2) / (4 x 60) + allowance
        slots = 15 * 3 * 15
        proj_cells = [{"cell": f"quota|{c}", "sec_per_problem": v["sec_per_problem"],
                       "problem_slots": slots if not c.endswith("|orth") else 15 * orth_streams * 15, "safety": 1.2,
                       "cpu_s": v["sec_per_problem"] * (slots if not c.endswith("|orth") else 15 * orth_streams * 15)
                       * 1.2, "source": "this pilot (6 problems/stream, concurrent)"} for c, v in cells.items()]
        cpu_s = sum(c["cpu_s"] for c in proj_cells)
        table_min = 15 * p1_table_sec_per_seed() / 60
        allowance = 1.0 + table_min
        proj_min = cpu_s / (4 * 60) + allowance
        downscale = None
        if proj_min > 55.0:
            orth_cpu = sum(c["cpu_s"] for c in proj_cells if c["cell"].endswith("|orth"))
            downscale = {"option": "methodology-6 preset: NL main orth arm on stream 0 only",
                         "projected_min": (cpu_s - orth_cpu * 2 / 3) / 240 + allowance}
        expected = sum(len(j[4]) * n_problems for j in jobs)
        crashes = len(fatal) + len(missing) + len(errs)
        bill_mm = sum(not r["billing_ok"] for r in R)
        pass_flags = {"zero_crashes": crashes == 0 and len(R) == expected,
                      "billing_zero_mismatch": bill_mm == 0 and au["n_bad"] == 0 and au["replay_billed_total"] == 0,
                      "predictors_before_results_every_certification": bool(
                          wa.get("write_ahead_ok") and wa["n_cert_without_predictor"] == 0
                          and wa["n_cert_predictor_late"] == 0 and wa["n_full_ev1_cert_without_full_predictors"] == 0
                          and wa["n_orth_design_late"] == 0),
                      "projected_full_le_55min": proj_min <= 55.0, "n_runs_ge_100": len(R) >= 100}
        go = all(pass_flags.values())
        summary = {
            "task": TASK, "mode": a.mode, "seeds": used, "streams": streams, "n_problems": n_problems,
            "methods": list(METHODS), "arms": list(ARMS), "orth_streams": orth_streams, "replacements": repl,
            "reserve_start": reserve, "quota_info": qinfo, "tables": tinfo, "n_jobs": len(jobs), "n_runs": len(R),
            "n_expected_runs": expected, "n_predictor_lines": len(P), "n_orth_design_records": len(O),
            "crashes": {"fatal_jobs": fatal, "missing_parts": missing, "n_errors": len(errs),
                        "errors_head": errs[:5]},
            "billing_mismatch": bill_mm, "audit": au, "write_ahead": wa, "cells": cells,
            "per_instance_descriptive": per_inst, "runs_wall_sec": runs_sec,
            "total_wall_sec": time.perf_counter() - t_all,
            "timing_projection": {"rule": "sum(sec/problem x 675 slots x 1.2) / (4 workers x 60) + allowance "
                                          "(1 min + GPU pool tables of 15 fresh instances at the r3_p1 rate)",
                                  "cells": proj_cells, "cpu_s": cpu_s, "allowance_min": allowance,
                                  "projected_min_full_chunk": proj_min,
                                  "lock_projection_min": lock.get("timing_projection", {}).get("per_task", {})
                                  .get("r3_nl_main_[a-h]", {}).get("projected_min_chosen"),
                                  "downscale_option": downscale,
                                  "caveat": "pilot = 6 problems/stream (full 15: vol/orth N_k and off ledgers grow "
                                            "with k); the sibling run is charged once to the vol cell; 并发运行"},
            "pass_criteria": pass_flags, "go_no_go": "GO" if go else "NO_GO",
            "scientific_readout": "none (pilot = smoke + timing only; dev seeds; contrasts are descriptive)",
            "lock_status_at_run": lock.get("status"), "lock_downscale_decisions": ds, "code_sha256": code_sha(),
            "start_time": start.isoformat(), "end_time": datetime.now().isoformat()}
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
        log(f"summary: go={go} flags={pass_flags} proj={proj_min:.1f} min runs={len(R)}")
        if a.mode == "pilot":
            entry = {"candidate_id": "cand_f", "go_no_go": "GO" if go else "NO_GO",
                     "pass_criteria": pass_flags, "projected_full_min": proj_min, "n_runs": len(R),
                     "downscale_option": downscale, "scientific_readout": "none (smoke + timing)"}
            md = (f"## {TASK} (pilot, dev 734-735, stream 0, 6 problems)\n"
                  f"- 结论: **{'GO' if go else 'NO_GO'}**；runs={len(R)}/{expected}，崩溃={crashes}，"
                  f"计费不一致={bill_mm}，审计异常={au['n_bad']}\n"
                  f"- write-ahead: 认证 {wa.get('n_certified')} 次，缺预测记录 {wa.get('n_cert_without_predictor')}，"
                  f"晚于评分 {wa.get('n_cert_predictor_late')}\n"
                  f"- 单分块 full 预计 {proj_min:.1f} min（lock 预估 "
                  f"{summary['timing_projection']['lock_projection_min']}）；并发运行，计时偏高\n")
            update_shared_summary(entry, md)
        mark_done("success" if go else "partial", f"{TASK} {a.mode}: GO={go} runs={len(R)} proj={proj_min:.1f}min")
        return summary
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        log(tb)
        (out_dir / "fatal.txt").write_text(tb)
        mark_done("failed", f"{TASK} {a.mode} crashed: {e!r}")
        raise


if __name__ == "__main__":
    main()
