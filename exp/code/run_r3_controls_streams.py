"""r3_controls_streams: S-stream audit (descriptive) -- no-tie stream and low-overlap stream. Methodology 2.5 / 5.1.

Design  NL-R0 (E1-NL-S G_1, realisable), same instances as the NL main contrast (eval 10000-10047; quota_fail
        replacement identical to r3_nl_main chunk rules so that theta* is shared), 3 streams (noise 42/123/456),
        two stream kinds from dsswm.streams.gap_quota:
          no_tie       15 clear problems (Delta >= 2 eps), stream order = rng [s, 45, t] permutation of the base order
          low_overlap  quota composition (5 tie / 5 near / 5 clear) chosen greedily by the PUBLIC decision direction
                       (min over candidates of max_prev |cos|); the greedy order IS the stream order
        {JPC, B3} x {off, ev1, full}; CRN: every arm of one (instance, kind, stream, method) runs on a fresh copy of
        the same platform (same noise seed, same n0 rows, same problem order, acquisition rng [seed, noise, method]).
Readouts (all descriptive; S-stream is a mandatory audit, not a confirmatory test)
  stream ratio   sum cost_tau3000(full) / sum cost_tau3000(off) on non-tie problems, per (kind, method)
  reuse factor   inverse of the stream ratio (off / full)
  K curve        cumulative new env steps incl. n0 (paid once per stream), K = 1..15, and the payback point K*
  dividend       log((off+1)/(full+1)) per problem, stratified by the public cos-overlap tercile of the problem
Logging (write separation)  predictors.jsonl (learner, fsync'ed, at the end of every problem BEFORE harness scoring;
  full/ev1: full predictor record; off: light record), results.jsonl (harness, after the stream). JOIN KEY
  (kind, instance, stream, method, arm, problem) -- `kind` is an extra key because the two stream kinds share
  instances and stream indices.
Pilot  smoke + timing only (NO scientific readout): dev seeds 734-739, stream 0, 15 problems, 2 kinds x 2 methods x
       3 arms (up to 1080 runs); projects the full wall-clock (lock rule) and lists a downscale option if > 55 min.
Full   eval 10000-10047 x 3 streams x 15 problems x 2 kinds x 6 cells; requires the v3 lock (assert_locked()).
       Resumable: one part file per (kind, instance, stream, method) job.

Usage: run_r3_controls_streams.py --mode {pilot,full} [--workers 4]
Concurrent run (shares 20 cores + the RTX 4090 with other round-3 tasks): timings are "concurrent" (并发运行).
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
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
METHODS = ("JPC", "B3")
ARMS = ("off", "ev1", "full")
KINDS = ("no_tie", "low_overlap")
FULL_PRED_ARMS = ("full", "ev1")
TASK = "r3_controls_streams"
# same chunk -> reserve rule as r3_nl_main (so a replaced instance gets the same replacement seed / theta*)
NM_CHUNKS = {c: (10000 + 15 * i, 10015 + 15 * i) for i, c in enumerate("abcdefgh")}
NM_RESERVE = {c: 10800 + 25 * i for i, c in enumerate("abcdefgh")}
FULL_SEEDS = list(range(10000, 10048))
DEV_RESERVE = 761
PILOT = {"seeds": [734, 735, 736, 737, 738, 739], "streams": [0], "n_problems": 15}
CODE_FILES = ["dsswm/evidence/reuse_switch.py", "dsswm/baselines/switched_nl.py", "dsswm/streams/gap_quota.py",
              "dsswm/streams/r3_harness.py", "dsswm/mechanism/leverage.py", "dsswm/mechanism/predictor_log.py",
              "run_r3_p4_t0_mechanism_gate.py", "run_r3_nl_main.py", "run_r3_controls_streams.py"]
JKEY = ("kind", "instance", "stream", "method", "arm", "problem")


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
_B: dict = {}


def _builder(seed):
    from dsswm.streams.gap_quota import QuotaBuilder
    import run_r3_nl_main as NM
    W = NM._ctx()
    if seed not in _B:
        _B.clear()
        _B[seed] = QuotaBuilder(seed, W["ncl"], None, CACHE, prop_cpu=W["prop"])
    return _B[seed]


def stream_overlap(b, st):
    """Public cos overlap of each stream problem with the previous ones, in stream order."""
    cands = [b.candidate(m["cand_index"]) for m in st.harness]
    return [o["cos_max"] for o in b.overlap(cands)]


# ============================================================================================ one job
def job(kind, seed, stream, method, n_problems, arms, out_dir):
    from dsswm.mechanism.predictor_log import PredictorLog, ResultLog
    tag = f"{kind}_i{seed}_s{stream}_{method}"
    parts = out_dir / "parts"
    pp, rp, jp = parts / f"{tag}_pred.jsonl", parts / f"{tag}_res.jsonl", parts / f"{tag}.json"
    for p in (pp, rp):
        p.unlink(missing_ok=True)                       # an unfinished job is rerun from scratch
    plog, rlog = PredictorLog(pp, fsync=True), ResultLog(rp, fsync=False)
    t_job = time.perf_counter()
    b = _builder(seed)
    st0 = b.stream(stream, kind, n_problems=n_problems)
    ov = stream_overlap(b, st0)
    errs, samples, cell_meta = [], [], {}
    for arm in arms:
        try:
            cell_meta[arm] = run_arm(b, kind, seed, stream, method, arm, n_problems, ov, plog, rlog, samples, errs)
        except Exception as e:  # noqa: BLE001 - a crashed cell is counted, never hidden
            errs.append({"tag": tag, "arm": arm, "stage": "cell", "error": repr(e),
                         "tb": traceback.format_exc()[-3000:]})
    out = {"tag": tag, "kind": kind, "seed": seed, "stream": stream, "method": method, "arms": list(arms),
           "errors": errs, "samples": samples, "cells": cell_meta, "cos_overlap": ov,
           "job_sec": time.perf_counter() - t_job, "worker_pid": os.getpid()}
    tmp = jp.with_name(jp.name + ".tmp")
    tmp.write_text(json.dumps(out, default=str))
    os.replace(tmp, jp)
    return {"tag": tag, "n_err": len(errs), "sec": out["job_sec"]}


def run_arm(b, kind, seed, stream, method, arm, n_problems, ov, plog, rlog, samples, errs):
    import run_r3_nl_main as NM
    from dsswm.baselines.switched_nl import solve
    from dsswm.certify.minimax_enum import certify_minimax
    from dsswm.evidence.reuse_switch import BillingError, ReuseSwitch, score_truncations, TAUS
    from dsswm.mechanism.leverage import theta_hat_index
    from dsswm.streams.r3_harness import _rng
    from run_r3_p4_t0_mechanism_gate import learner_record
    W = NM._ctx()
    t_cell = time.perf_counter()
    st = b.stream(stream, kind, n_problems=n_problems)               # fresh platform copy (CRN)
    pub = W["pub"].with_problems(st.problems, st.J)
    env = st.env
    h = env.handle()
    sw = ReuseSwitch(arm, pub.make_set_factory(method), st.init_obs)
    rng = _rng(seed, st.noise_seed, method)
    keep = []
    for k, q in enumerate(st.problems):
        key = {"kind": kind, "instance": seed, "stream": stream, "method": method, "arm": arm, "problem": k}
        t0 = time.perf_counter()
        lr = sw.begin_problem(k, q.pid, context={"stream": stream, "kind": kind})
        res = solve(pub, method, k, sw, lr, h, rng, TMAX)
        wall = time.perf_counter() - t0
        # ---------------------------------------------------- learner side, write-ahead (before any truth scoring)
        t1 = time.perf_counter()
        pred_ok, k_hat, set_size = False, None, None
        try:
            mask = lr.mask().numpy().astype(bool)
            set_size = int(mask.sum())
            k_hat = theta_hat_index(lr.inner.cum.cpu().numpy(), mask)
            Jl = pub.J[k]
            Jh = np.sort(Jl[k_hat])[::-1]
            base = {**key, "pid": q.pid, "status": res["status"], "new_steps": int(res["steps"]),
                    "theta_hat": int(k_hat), "set_size": set_size, "n_rows_in_set": len(lr.obs),
                    "n_replay_in_set": int(lr.n_replay), "cos_overlap_public": ov[k],
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
        row = {"kind": kind, "instance": seed, "stream": stream, "method": method, "arm": arm, "problem": k,
               "problem_index": k, "noise_seed": st.noise_seed, "pid": q.pid, "cand_index": hm["cand_index"],
               "layer": "NL-R0", "gap_layer": hm["gap_layer"], "true_gap": hm["true_gap"],
               "status": res["status"], "new_env_steps": int(res["steps"]), "replay_steps": int(acc.replay_steps),
               "replay_pad_steps": int(acc.replay_pad_steps), "n_rounds_total": int(acc.n_rounds_total),
               "n_rounds_billed": int(acc.n_rounds_billed), "env_n_steps": kr["env_n_steps"],
               "billing_ok": kr["billing_ok"], "billing_error": kr["billing_error"], "censored": not cert,
               "zero_cost": bool(cert and res["steps"] == 0), "certified_policy": pi, "true_regret": regret,
               "false_cert": bool(cert and regret is not None and regret > EPS), "set_size": kr["set_size"],
               "prov_counts": dict(acc.prov_counts), "cos_overlap": ov[k], "mu_flip": None,
               "predictor_logged": kr["pred_ok"], "wall_clock_s": kr["wall"], "predictor_sec": kr["t_pred"],
               "eps": EPS, "delta": DELTA, "tmax": TMAX, "first_cert_step": res["extra"].get("first_cert_step")}
        if kr["k_hat"] is not None:
            Jh = st.J[k][kr["k_hat"]]
            row["eta_arg"] = float(Jt.max() - Jt[int(np.argmax(Jh))])
            row["eta_dec"] = float(np.max(np.abs(Jh - Jt)))
            row["theta_hat_is_truth"] = bool(kr["k_hat"] == ti)
        else:
            row["eta_arg"] = row["eta_dec"] = row["theta_hat_is_truth"] = None
        row.update(score_truncations(int(res["steps"]), cert, TAUS))
        rlog.write(row)
        if k < 2 and len(samples) < 24:
            samples.append({"kind": kind, "instance": seed, "stream": stream, "method": method, "arm": arm,
                            "problem": k, "problem_public": q.public_dict(),
                            "J_true": [round(float(x), 6) for x in Jt],
                            "row": {kk: row.get(kk) for kk in ("status", "new_env_steps", "certified_policy",
                                                               "true_regret", "gap_layer", "set_size",
                                                               "cos_overlap", "prov_counts")}})
    return {"cell_sec": time.perf_counter() - t_cell, "n": len(keep)}


# ============================================================================================ main-process stages
def tables(groups):
    """GPU: class J tables of the candidate pools (cached); quota / no_tie / low_overlap selection; deterministic
    quota_fail replacement with the r3_nl_main rule (only `quota` failure triggers a replacement, so theta* matches
    the main contrast). A no_tie failure of a kept instance is recorded and that kind is skipped (descriptive)."""
    from dsswm.exact.nl_propagate import NLPropagator
    from dsswm.models.nl_class import NLClass
    from dsswm.streams.gap_quota import QuotaBuilder, fill_instances
    from dsswm.streams.generator import DEFAULT_NL_S_GRID
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    ncl = NLClass(DEFAULT_NL_S_GRID, device=dev)
    b0 = QuotaBuilder(groups[0][0][0], ncl, None, CACHE)
    propg = NLPropagator(2, 2, 2, b0.aspace, 1.0, 0.3, device=dev)
    propc = NLPropagator(2, 2, 2, b0.aspace, 1.0, 0.3, device="cpu")
    t0 = time.perf_counter()
    info, ok, builders = {}, {}, {}

    def check(s):
        if s not in ok:
            b = QuotaBuilder(s, ncl, propg, CACHE, prop_cpu=propc)
            sel = b.select("quota")
            ok[s] = not sel["quota_fail"]
            info[s] = {"quota": {"quota_fail": sel["quota_fail"], "n_draws": sel["n_draws"],
                                 "layer_counts": sel["layer_counts"]}}
            builders[s] = b
        return ok[s]

    used, repl = [], []
    for seeds, reserve in groups:
        u, r = fill_instances(seeds, check, reserve_start=reserve)
        used += u
        repl += r
    kinds_ok = {}
    for s in used:
        b = builders[s]
        kinds_ok[s] = []
        for kind in KINDS:
            ts = time.perf_counter()
            sel = b.select(kind)
            info[s][kind] = {"quota_fail": sel["quota_fail"], "n_draws": sel["n_draws"],
                             "layer_counts": sel["layer_counts"], "select_sec": time.perf_counter() - ts}
            if not sel["quota_fail"]:
                kinds_ok[s].append(kind)
        info[s]["tables_computed"] = b.n_table_computed
        info[s]["table_sec"] = b.sec_tables
    vram = torch.cuda.max_memory_allocated() / 2 ** 20 if dev == "cuda" else 0.0
    tot = torch.cuda.get_device_properties(0).total_memory / 2 ** 20 if dev == "cuda" else 0
    prof = {"gpu_name": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu", "vram_total_mb": tot,
            "max_batch_size": "n/a (class J tables: one problem x |Theta|=13824 per call; no batch dimension)",
            "vram_used_mb": vram, "utilization_pct": (100.0 * vram / tot) if tot else None,
            "note": "GPU only for class J tables of candidate problems (cached); JPC/B3 loops, decision directions, "
                    "predictors and harness scoring on CPU (4 workers, OMP=1); 并发运行（与其他任务共享 4090 与 20 核）"}
    (RES_ROOT / f"{TASK}_gpu_profile.json").write_text(json.dumps(prof))
    del ncl, propg
    if dev == "cuda":
        torch.cuda.empty_cache()
    return used, repl, kinds_ok, info, {"sec": time.perf_counter() - t0, "vram_peak_mb": vram, "device": dev}


def jtag(j):
    return f"{j[0]}_i{j[1]}_s{j[2]}_{j[3]}"


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
    from dsswm.mechanism.predictor_log import load_jsonl
    parts = out_dir / "parts"
    P, R, errs, meta, samples, missing = [], [], [], [], [], []
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
    for name, rows in (("predictors.jsonl", P), ("results.jsonl", R)):
        with open(out_dir / name, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r, sort_keys=True) + "\n")
    pk = {}
    for r in P:
        pk.setdefault(tuple(r[x] for x in JKEY), r["logged_at"])
    keys_r = [tuple(r[x] for x in JKEY) for r in R]
    cert = [r for r in R if r.get("status") == "CERTIFIED"]
    miss_all = [k for k in keys_r if k not in pk]
    late_all = [k for k, r in zip(keys_r, R) if k in pk and pk[k] > r["scored_at"]]
    miss = [r for r in cert if tuple(r[x] for x in JKEY) not in pk]
    late = [r for r in cert if (k := tuple(r[x] for x in JKEY)) in pk and pk[k] > r["scored_at"]]
    full_cert = [r for r in cert if r["arm"] in FULL_PRED_ARMS]
    full_keys = {tuple(r[x] for x in JKEY) for r in P if r.get("record") == "full_predictors"}
    full_miss = [r for r in full_cert if tuple(r[x] for x in JKEY) not in full_keys]
    wa = {"join_key": list(JKEY), "write_ahead_ok": not late_all, "n_results": len(R), "n_predictors": len(P),
          "n_results_without_predictor": len(miss_all), "n_predictor_late_any": len(late_all),
          "n_certified": len(cert), "n_cert_without_predictor": len(miss), "n_cert_predictor_late": len(late),
          "n_certified_full_ev1": len(full_cert), "n_full_ev1_cert_without_full_predictors": len(full_miss)}
    (out_dir / "samples").mkdir(exist_ok=True)
    (out_dir / "samples" / "cells.json").write_text(json.dumps(samples[:80], indent=1, default=str))
    if errs or missing:
        (out_dir / "errors.json").write_text(json.dumps({"errors": errs, "missing_parts": missing}, indent=1,
                                                        default=str))
    return R, P, meta, errs, missing, wa


# ============================================================================================ analysis
def audit(rows):
    """Billing + provenance audit per row (r3_nl_main invariants restricted to off / ev1 / full)."""
    bad = []
    for r in rows:
        pc = r["prov_counts"] or {}
        rep = sum(v for kk, v in pc.items() if kk.startswith("replay"))
        conds = {"billing_ok": r["billing_ok"], "env_eq_billed": r["env_n_steps"] == r["n_rounds_billed"],
                 "prov_sum_eq_total": sum(pc.values()) == r["n_rounds_total"],
                 "prov_replay_eq_replay_steps": rep == r["replay_steps"], "no_replay": rep == 0}
        if r["arm"] == "off":
            conds["off_drops_old"] = pc.get("initial", 0) + pc.get("real", 0) == N0 + r["new_env_steps"]
        else:
            conds["persist_sees_all_billed"] = pc.get("initial", 0) + pc.get("real", 0) == r["n_rounds_billed"]
        if r["arm"] == "ev1" and r["status"] == "CERTIFIED":
            conds["ev1_min_one_new_step"] = r["new_env_steps"] >= 1
        f = [kk for kk, v in conds.items() if not v]
        if f:
            bad.append({"kind": r["kind"], "instance": r["instance"], "stream": r["stream"], "method": r["method"],
                        "arm": r["arm"], "k": r["problem"], "failed": f})
    return {"n_rows": len(rows), "n_bad": len(bad), "examples": bad[:20],
            "bad_by_arm": {a: sum(1 for b in bad if b["arm"] == a) for a in ARMS},
            "billing_mismatch_by_arm": {a: sum(not r["billing_ok"] for r in rows if r["arm"] == a) for a in ARMS},
            "replay_billed_total": sum(r["replay_steps"] for r in rows)}


def k_curves(R, n_problems):
    """Cumulative new env steps incl. n0 (paid once per stream), K = 1..n; K* = first K from which
    cum_full <= cum_off for every later K (per kind, method; per instance-stream pair)."""
    idx = defaultdict(dict)
    for r in R:
        idx[(r["kind"], r["method"], r["arm"], r["instance"], r["stream"])][r["problem"]] = r["new_env_steps"]
    cur, kstar = {}, {}
    for kind in KINDS:
        for m in METHODS:
            for a in ARMS:
                cv = [N0 + np.cumsum([d[k] for k in range(n_problems)])
                      for (kk, mm, aa, i, s), d in idx.items() if (kk, mm, aa) == (kind, m, a) and len(d) == n_problems]
                if cv:
                    cur[f"{kind}|{m}|{a}"] = {"n_streams": len(cv), "mean_cum": np.mean(cv, 0).round(2).tolist()}
            ks = []
            for (kk, mm, aa, i, s), d in idx.items():
                if (kk, mm, aa) != (kind, m, "full"):
                    continue
                o = idx.get((kind, m, "off", i, s))
                if o is None or len(d) != n_problems or len(o) != n_problems:
                    continue
                cf = N0 + np.cumsum([d[k] for k in range(n_problems)])
                co = N0 + np.cumsum([o[k] for k in range(n_problems)])
                okk = cf <= co
                ks.append(next((K + 1 for K in range(n_problems) if okk[K:].all()), None))
            if ks:
                kstar[f"{kind}|{m}"] = {"n_pairs": len(ks), "K_star": ks, "n_never": sum(k is None for k in ks),
                                        "median_K_star": (float(np.median([k for k in ks if k is not None]))
                                                          if any(k is not None for k in ks) else None)}
    return cur, kstar


def analyse(R, n_problems):
    by = defaultdict(list)
    for r in R:
        by[(r["kind"], r["method"], r["arm"])].append(r)
    cells = {}
    for (kd, m, a), rr in sorted(by.items()):
        cert = [r for r in rr if r["status"] == "CERTIFIED"]
        nt = [r for r in rr if r["gap_layer"] != "tie"]
        cells[f"{kd}|{m}|{a}"] = {
            "n": len(rr), "n_nontie": len(nt), "status_counts": dict(Counter(r["status"] for r in rr)),
            "layer_counts": dict(Counter(r["gap_layer"] for r in rr)),
            "n_certified": len(cert), "n_zero_cost": sum(r["zero_cost"] for r in rr),
            "n_false_cert": sum(r["false_cert"] for r in rr),
            "fcr_descriptive": (sum(r["false_cert"] for r in cert) / len(cert)) if cert else None,
            "nontie_steps_tau3000_total": int(sum(r["cost_tau3000"] for r in nt)),
            "new_steps_total": int(sum(r["new_env_steps"] for r in rr)),
            "median_new_steps": float(np.median([r["new_env_steps"] for r in rr])),
            "billing_mismatch": sum(not r["billing_ok"] for r in rr),
            "sec_per_problem_method": float(np.mean([r["wall_clock_s"] for r in rr])),
            "sec_per_problem_predictor": float(np.mean([r["predictor_sec"] for r in rr])),
            "sec_per_problem": float(np.mean([r["wall_clock_s"] + r["predictor_sec"] for r in rr]))}
    # ---- stream ratio / reuse factor (non-tie, tau = 3000), pooled and per instance
    ratios = {}
    for kd in KINDS:
        for m in METHODS:
            for a in ("full", "ev1"):
                num = by.get((kd, m, a), [])
                den = by.get((kd, m, "off"), [])
                if not num or not den:
                    continue
                sn = sum(r["cost_tau3000"] for r in num if r["gap_layer"] != "tie")
                sd = sum(r["cost_tau3000"] for r in den if r["gap_layer"] != "tie")
                per = defaultdict(lambda: [0, 0])
                for r in num:
                    if r["gap_layer"] != "tie":
                        per[r["instance"]][0] += r["cost_tau3000"]
                for r in den:
                    if r["gap_layer"] != "tie":
                        per[r["instance"]][1] += r["cost_tau3000"]
                lr_i = [float(np.log((x + 1) / (y + 1))) for x, y in per.values()]
                ratios[f"{kd}|{m}|{a}/off"] = {
                    "stream_ratio_pooled": (sn + 1) / (sd + 1), "reuse_factor_pooled": (sd + 1) / (sn + 1),
                    "mean_log_ratio_per_instance": float(np.mean(lr_i)) if lr_i else None,
                    "n_instances": len(lr_i)}
    # ---- reuse dividend by public cos-overlap tercile (problem 0 has no overlap -> excluded)
    pair = defaultdict(dict)
    for r in R:
        pair[(r["kind"], r["method"], r["instance"], r["stream"], r["problem"])][r["arm"]] = r
    dividend = {}
    cos_all = [r["cos_overlap"] for r in R if r["cos_overlap"] is not None and r["arm"] == "off"]
    edges = np.quantile(cos_all, [1 / 3, 2 / 3]).tolist() if len(cos_all) >= 3 else None
    for kd in KINDS:
        cos_k = [r["cos_overlap"] for r in R if r["kind"] == kd and r["cos_overlap"] is not None
                 and r["arm"] == "off"]
        ek = np.quantile(cos_k, [1 / 3, 2 / 3]).tolist() if len(cos_k) >= 3 else None
        for m in METHODS:
            terc = defaultdict(list)
            for (kk, mm, i, s, k), d in pair.items():
                if (kk, mm) != (kd, m) or "off" not in d or "full" not in d or ek is None:
                    continue
                c = d["off"]["cos_overlap"]
                if c is None or d["off"]["gap_layer"] == "tie":
                    continue
                t = int(np.searchsorted(ek, c, side="right"))
                terc[t].append(float(np.log((d["off"]["cost_tau3000"] + 1) / (d["full"]["cost_tau3000"] + 1))))
            dividend[f"{kd}|{m}"] = {"tercile_edges_cos": ek,
                                     "by_tercile": {str(t): {"n": len(v), "mean_log_off_over_full": float(np.mean(v)),
                                                             "median": float(np.median(v))}
                                                    for t, v in sorted(terc.items())}}
    overlap_desc = {kd: {"cos_mean": float(np.mean(v)) if v else None, "cos_median": float(np.median(v)) if v else None,
                         "n": len(v)}
                    for kd in KINDS
                    for v in [[r["cos_overlap"] for r in R if r["kind"] == kd and r["cos_overlap"] is not None
                               and r["arm"] == "off" and r["method"] == "JPC"]]}
    kc, ks = k_curves(R, n_problems)
    return cells, ratios, dividend, {"pooled_tercile_edges": edges, "by_kind": overlap_desc}, kc, ks


def plot_dividend(dividend, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        for key, d in dividend.items():
            bt = d["by_tercile"]
            if not bt:
                continue
            xs = sorted(int(t) for t in bt)
            ax.plot([x + 1 for x in xs], [bt[str(x)]["mean_log_off_over_full"] for x in xs], marker="o", label=key)
        ax.set_xticks([1, 2, 3])
        ax.set_xlabel("public cos-overlap tercile (1 = lowest)")
        ax.set_ylabel("mean log((off+1)/(full+1)), tau=3000")
        ax.set_title("Reuse dividend vs overlap (non-tie problems)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / "dividend_by_overlap.png", dpi=130)
        plt.close(fig)
        return True
    except Exception:  # noqa: BLE001
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="pilot", choices=["pilot", "full"])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    (RES_ROOT / f"{TASK}.pid").write_text(str(os.getpid()))
    t_all = time.perf_counter()
    start = datetime.now()
    out_dir = RES_ROOT / ("pilots" if a.mode == "pilot" else "full") / TASK
    (out_dir / "parts").mkdir(parents=True, exist_ok=True)
    try:
        lock = json.loads((WS / "plan" / "prereg_lock.json").read_text())
        if a.mode == "full":
            from dsswm.stats.prereg import assert_locked
            lock = assert_locked()
            ds = lock.get("frozen_items", {}).get("downscale_decisions", {}).get(TASK, [])
            groups = []
            for c, (lo, hi) in NM_CHUNKS.items():
                sl = [s for s in FULL_SEEDS if lo <= s < hi]
                if sl:
                    groups.append((sl, NM_RESERVE[c]))
            streams, n_problems = [0, 1, 2], 15
        else:
            groups = [(PILOT["seeds"], DEV_RESERVE)]
            streams, n_problems, ds = PILOT["streams"], PILOT["n_problems"], []
        progress(0, 1, "tables")
        used, repl, kinds_ok, qinfo, tinfo = tables(groups)
        log(f"tables {tinfo} replacements {repl} kinds_ok {kinds_ok}")
        jobs = []
        for kind in KINDS:
            for s in used:
                if kind not in kinds_ok[s]:
                    continue
                for st in streams:
                    for m in METHODS:
                        jobs.append((kind, s, st, m, n_problems, ARMS))
        jobs.sort(key=lambda j: (j[3] != "B3", j[0] != "low_overlap", j[1], j[2]))   # slower jobs first
        t_runs = time.perf_counter()
        outs = run_pool(jobs, out_dir, a.workers)
        runs_sec = time.perf_counter() - t_runs
        R, P, meta, errs, missing, wa = merge(jobs, out_dir)
        fatal = [o for o in outs if "fatal" in o]
        au = audit(R)
        cells, ratios, dividend, overlap_desc, kc, ks = analyse(R, n_problems)
        plotted = plot_dividend(dividend, out_dir)
        # ---- projection of the full run (lock rule): sum(sec/problem x slots x 1.2) / (4 workers x 60) + allowance
        slots = 48 * 3 * 15
        proj_cells = [{"cell": c, "sec_per_problem": v["sec_per_problem"], "problem_slots": slots, "safety": 1.2,
                       "cpu_s": v["sec_per_problem"] * slots * 1.2,
                       "source": "this pilot (dev 734-739, stream 0, 15 problems, concurrent)"}
                      for c, v in cells.items()]
        cpu_s = sum(c["cpu_s"] for c in proj_cells)
        n_used = max(1, len(used))
        table_sec_per_inst = float(np.mean([qinfo[s].get("table_sec", 0.0) for s in used])) if used else 0.0
        sel_sec_per_inst = float(np.mean([qinfo[s]["low_overlap"]["select_sec"] for s in used])) if used else 0.0
        allowance = 1.0 + 48 * (table_sec_per_inst + sel_sec_per_inst) / 60
        proj_min = cpu_s / (4 * 60) + allowance
        downscale = None
        if proj_min > 55.0:
            downscale = {"option": "split into 2 chunks of 24 instances (each <= 55 min) or streams {0,1} only",
                         "projected_min_2_chunks_each": cpu_s / 2 / 240 + allowance / 2,
                         "projected_min_2_streams": cpu_s * 2 / 3 / 240 + allowance}
        expected = len(jobs) * len(ARMS) * n_problems
        crashes = len(fatal) + len(missing) + len(errs)
        bill_mm = sum(not r["billing_ok"] for r in R)
        pass_flags = {"zero_crashes": crashes == 0 and len(R) == expected,
                      "billing_zero_mismatch": bill_mm == 0 and au["n_bad"] == 0 and au["replay_billed_total"] == 0,
                      "predictors_before_results_every_certification": bool(
                          wa["write_ahead_ok"] and wa["n_cert_without_predictor"] == 0
                          and wa["n_cert_predictor_late"] == 0 and wa["n_full_ev1_cert_without_full_predictors"] == 0),
                      "projected_full_le_55min": proj_min <= 55.0, "n_runs_ge_100": len(R) >= 100}
        # full pass_criteria = "rows complete" (+ integrity); the <=55 min projection flag is a pilot-only lock rule
        go = all(v for k, v in pass_flags.items() if a.mode == "pilot" or k != "projected_full_le_55min")
        summary = {
            "task": TASK, "mode": a.mode, "seeds": used, "kinds": list(KINDS), "kinds_ok": kinds_ok,
            "streams": streams, "n_problems": n_problems, "methods": list(METHODS), "arms": list(ARMS),
            "replacements": repl, "quota_info": qinfo, "tables": tinfo, "n_jobs": len(jobs), "n_runs": len(R),
            "n_expected_runs": expected, "n_predictor_lines": len(P),
            "crashes": {"fatal_jobs": fatal, "missing_parts": missing, "n_errors": len(errs),
                        "errors_head": errs[:5]},
            "billing_mismatch": bill_mm, "audit": au, "write_ahead": wa, "cells": cells,
            "stream_ratio_reuse_factor_descriptive": ratios, "dividend_by_overlap_descriptive": dividend,
            "overlap_descriptive": overlap_desc, "K_curve_descriptive": kc, "K_star_descriptive": ks,
            "dividend_plot": "dividend_by_overlap.png" if plotted else None,
            "runs_wall_sec": runs_sec, "total_wall_sec": time.perf_counter() - t_all,
            "timing_projection": {"rule": "sum(sec/problem x 2160 slots x 1.2) / (4 workers x 60) + allowance "
                                          "(1 min + 48 x (GPU tables + low_overlap selection) per instance)",
                                  "cells": proj_cells, "cpu_s": cpu_s, "allowance_min": allowance,
                                  "projected_min_full": proj_min,
                                  "lock_projection_min": lock.get("timing_projection", {}).get("per_task", {})
                                  .get(TASK, {}).get("projected_min_chosen"),
                                  "downscale_option": downscale,
                                  "caveat": "pilot table/selection time per instance measured on dev seeds (cache "
                                            "state may differ for eval seeds); 并发运行"},
            "pass_criteria": pass_flags, "go_no_go": "GO" if go else "NO_GO",
            "scientific_readout": "none (pilot = smoke + timing only; dev seeds; ratios/dividends are descriptive)",
            "lock_status_at_run": lock.get("status"), "lock_downscale_decisions": ds, "code_sha256": code_sha(),
            "start_time": start.isoformat(), "end_time": datetime.now().isoformat()}
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
        log(f"summary: go={go} flags={pass_flags} proj={proj_min:.1f} min runs={len(R)}/{expected}")
        if a.mode == "pilot":
            import run_r3_nl_main as NM
            NM.TASK = TASK
            entry = {"candidate_id": "cand_f", "go_no_go": "GO" if go else "NO_GO",
                     "pass_criteria": pass_flags, "projected_full_min": proj_min, "n_runs": len(R),
                     "downscale_option": downscale, "scientific_readout": "none (smoke + timing)"}
            md = (f"## {TASK} (pilot, dev 734-739, stream 0, 15 problems, no_tie + low_overlap)\n"
                  f"- 结论: **{'GO' if go else 'NO_GO'}**；runs={len(R)}/{expected}，崩溃={crashes}，"
                  f"计费不一致={bill_mm}，审计异常={au['n_bad']}\n"
                  f"- write-ahead: 认证 {wa['n_certified']} 次，缺预测记录 {wa['n_cert_without_predictor']}，"
                  f"晚于评分 {wa['n_cert_predictor_late']}\n"
                  f"- full 预计 {proj_min:.1f} min（lock 预估 {summary['timing_projection']['lock_projection_min']}）；"
                  f"并发运行，计时偏高\n")
            NM.update_shared_summary(entry, md)
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
