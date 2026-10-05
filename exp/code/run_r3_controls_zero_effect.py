"""r3_controls_zero_effect: negative control -- offset-null zero-effect environment. Methodology 2.5.

Design  NL-R0 (E1-NL-S G_1, realisable), same instances as the NL main contrast (eval 10000-10047; quota_fail
        replacement identical to r3_nl_main chunk rules, so theta* and the quota problems are shared), 3 streams
        (noise 42/123/456), 15 quota problems per stream turned into offset-null problems by dsswm.streams.gap_quota:
          U'_q(traj; pi) = (U_q(traj) + b_pi) / (1 + max b),   b_pi = max_pi' J*(pi') - J*(pi) >= 0
        (harness-computed, handed to the learner as a PUBLIC utility term -- disclosed in the paper). Every candidate
        has the same true value (all problems are ties), but other class members still disagree, so the learner must
        collect data to certify an eps-tie. Any "saving" only means "declared the eps-tie earlier".
        {JPC, B3} x {off, ev1, full}; CRN: every arm of one (instance, stream, method) runs on a fresh copy of the same
        platform (same noise seed, same n0 rows, same problem order, acquisition rng [seed, noise, method]).
Readouts (descriptive in the pilot; full feeds the tier-C switch of the lock):
  stream_steps_by_arm     sum / median new env steps (and cost_tau3000) per (method, arm)
  jpc_over_b3_null        log((JPC + 1) / (B3 + 1)) of cost_tau3000 per arm, pooled and per (instance, stream)
  tie_declaration_rate    fraction of problems certified (= eps-tie declared) within T_max, per (method, arm)
  tier-C switch           zero-effect saving (JPC vs B3) >= main-environment non-tie saving -> tier C (evaluated by
                          the aggregate task against r3_nl_main; this script only produces the zero-effect side)
Logging (write separation)  predictors.jsonl (learner, fsync'ed, at the end of every problem BEFORE harness scoring),
  results.jsonl (harness, after the stream). JOIN KEY (kind, instance, stream, method, arm, problem).
Pilot  smoke + timing only (NO scientific readout): dev seeds 734-739, stream 0, 15 problems, 2 methods x 3 arms
       (540 runs); projects the full wall-clock for the full design and for the lock downscale
       (split_2_chunks_24_inst + B3_stream0) and checks <= 55 min.
Full   requires the v3 lock (assert_locked()). If the lock downscale `split_2_chunks_24_inst+B3_stream0` is listed,
       pass --chunk a (eval 10000-10023) or --chunk b (10024-10047) and B3 runs on stream 0 only.
       Resumable: one part file per (instance, stream, method) job.

Usage: run_r3_controls_zero_effect.py --mode {pilot,full} [--chunk {a,b}] [--workers 4]
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
KIND = "offset_null"
FULL_PRED_ARMS = ("full", "ev1")
TASK = "r3_controls_zero_effect"
LOCK_KEY = "r3_controls_zero_effect"      # lock / task_plan key (TASK gets a chunk suffix in chunked full runs)
NM_CHUNKS = {c: (10000 + 15 * i, 10015 + 15 * i) for i, c in enumerate("abcdefgh")}
NM_RESERVE = {c: 10800 + 25 * i for i, c in enumerate("abcdefgh")}
FULL_SEEDS = list(range(10000, 10048))
ZE_CHUNKS = {"a": (10000, 10024), "b": (10024, 10048)}
DOWNSCALE_KEY = "split_2_chunks_24_inst+B3_stream0"
DEV_RESERVE = 761
PILOT = {"seeds": [734, 735, 736, 737, 738, 739], "streams": [0], "n_problems": 15}
SAFETY = 1.35            # same factor the v3 lock used for this task
CODE_FILES = ["dsswm/evidence/reuse_switch.py", "dsswm/baselines/switched_nl.py", "dsswm/streams/gap_quota.py",
              "dsswm/streams/r3_harness.py", "dsswm/mechanism/leverage.py", "dsswm/mechanism/predictor_log.py",
              "run_r3_p4_t0_mechanism_gate.py", "run_r3_nl_main.py", "run_r3_controls_streams.py",
              "run_r3_controls_zero_effect.py"]
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
               "layer": "NL-R0", "gap_layer": hm["gap_layer"], "orig_gap_layer": hm["gap_layer"],
               "true_gap": hm["true_gap"], "true_gap_orig": hm["true_gap_orig"], "env_kind": KIND,
               "tie_declared": bool(cert),
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
                                                               "true_regret", "gap_layer", "true_gap",
                                                               "true_gap_orig", "set_size",
                                                               "cos_overlap", "prov_counts")}})
    return {"cell_sec": time.perf_counter() - t_cell, "n": len(keep)}


# ============================================================================================ main-process stages
def tables(groups, keep_range=None):
    """GPU: class J tables of the candidate pools (cached); quota selection with the r3_nl_main replacement rule.
    groups = [(requested seeds of one NM chunk in order, reserve start)]. The fill is run over the WHOLE NM chunk so
    reserve consumption (hence theta*) matches the main contrast; keep_range=(lo, hi) then keeps only the positions
    whose requested seed lies in [lo, hi)."""
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
    info, ok = {}, {}

    def check(s):
        if s not in ok:
            ts = time.perf_counter()
            b = QuotaBuilder(s, ncl, propg, CACHE, prop_cpu=propc)
            sel = b.select("quota")
            ok[s] = not sel["quota_fail"]
            info[s] = {"quota_fail": sel["quota_fail"], "n_draws": sel["n_draws"], "layer_counts": sel["layer_counts"],
                       "tables_computed": b.n_table_computed, "table_sec": b.sec_tables,
                       "check_sec": time.perf_counter() - ts}
        return ok[s]

    used, repl = [], []
    for seeds, reserve in groups:
        u, r = fill_instances(seeds, check, reserve_start=reserve)
        rmap = {x["failed"]: x["replacement"] for x in r}
        for req, got in zip(seeds, u):
            if keep_range is None or keep_range[0] <= req < keep_range[1]:
                used.append(got)
                if req in rmap:
                    repl.append({"failed": req, "replacement": got})
    vram = torch.cuda.max_memory_allocated() / 2 ** 20 if dev == "cuda" else 0.0
    tot = torch.cuda.get_device_properties(0).total_memory / 2 ** 20 if dev == "cuda" else 0
    prof = {"gpu_name": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu", "vram_total_mb": tot,
            "max_batch_size": "n/a (class J tables: one problem x |Theta|=13824 per call; no batch dimension)",
            "vram_used_mb": vram, "utilization_pct": (100.0 * vram / tot) if tot else None,
            "note": "GPU only for class J tables of candidate problems (cached); JPC/B3 loops, predictors and harness "
                    "scoring on CPU (4 workers, OMP=1); 并发运行（与其他任务共享 4090 与 20 核）"}
    (RES_ROOT / f"{TASK}_gpu_profile.json").write_text(json.dumps(prof))
    del ncl, propg
    if dev == "cuda":
        torch.cuda.empty_cache()
    return used, repl, info, {"sec": time.perf_counter() - t0, "vram_peak_mb": vram, "device": dev}


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
    """Billing + provenance audit per row (r3_nl_main invariants restricted to off / ev1 / full) plus the
    offset-null invariants (true gap 0 => no false certificate possible, regret 0)."""
    bad = []
    for r in rows:
        pc = r["prov_counts"] or {}
        rep = sum(v for kk, v in pc.items() if kk.startswith("replay"))
        conds = {"billing_ok": r["billing_ok"], "env_eq_billed": r["env_n_steps"] == r["n_rounds_billed"],
                 "prov_sum_eq_total": sum(pc.values()) == r["n_rounds_total"],
                 "prov_replay_eq_replay_steps": rep == r["replay_steps"], "no_replay": rep == 0,
                 "null_true_gap_zero": abs(r["true_gap"]) <= 1e-12,
                 "null_regret_zero": r["true_regret"] is None or abs(r["true_regret"]) <= 1e-12}
        if r["arm"] == "off":
            conds["off_drops_old"] = pc.get("initial", 0) + pc.get("real", 0) == N0 + r["new_env_steps"]
        else:
            conds["persist_sees_all_billed"] = pc.get("initial", 0) + pc.get("real", 0) == r["n_rounds_billed"]
        if r["arm"] == "ev1" and r["status"] == "CERTIFIED":
            conds["ev1_min_one_new_step"] = r["new_env_steps"] >= 1
        f = [kk for kk, v in conds.items() if not v]
        if f:
            bad.append({"instance": r["instance"], "stream": r["stream"], "method": r["method"], "arm": r["arm"],
                        "k": r["problem"], "failed": f})
    return {"n_rows": len(rows), "n_bad": len(bad), "examples": bad[:20],
            "bad_by_arm": {a: sum(1 for b in bad if b["arm"] == a) for a in ARMS},
            "billing_mismatch_by_arm": {a: sum(not r["billing_ok"] for r in rows if r["arm"] == a) for a in ARMS},
            "replay_billed_total": sum(r["replay_steps"] for r in rows)}


def analyse(R):
    by = defaultdict(list)
    for r in R:
        by[(r["method"], r["arm"])].append(r)
    cells = {}
    for (m, a), rr in sorted(by.items()):
        cert = [r for r in rr if r["status"] == "CERTIFIED"]
        cells[f"{KIND}|{m}|{a}"] = {
            "n": len(rr), "status_counts": dict(Counter(r["status"] for r in rr)),
            "orig_layer_counts": dict(Counter(r["orig_gap_layer"] for r in rr)),
            "n_certified": len(cert), "tie_declaration_rate": len(cert) / len(rr) if rr else None,
            "tie_declaration_rate_tau3000": (sum(not r["censored_tau3000"] for r in rr) / len(rr)) if rr else None,
            "n_zero_cost": sum(r["zero_cost"] for r in rr), "n_false_cert": sum(r["false_cert"] for r in rr),
            "steps_tau3000_total": int(sum(r["cost_tau3000"] for r in rr)),
            "new_steps_total": int(sum(r["new_env_steps"] for r in rr)),
            "median_new_steps": float(np.median([r["new_env_steps"] for r in rr])),
            "median_new_steps_certified": float(np.median([r["new_env_steps"] for r in cert])) if cert else None,
            "billing_mismatch": sum(not r["billing_ok"] for r in rr),
            "sec_per_problem_method": float(np.mean([r["wall_clock_s"] for r in rr])),
            "sec_per_problem_predictor": float(np.mean([r["predictor_sec"] for r in rr])),
            "sec_per_problem": float(np.mean([r["wall_clock_s"] + r["predictor_sec"] for r in rr])),
            "sec_per_problem_max": float(np.max([r["wall_clock_s"] + r["predictor_sec"] for r in rr]))}
    # ---- JPC / B3 on the null (all problems; tau = 3000), per arm, pooled and per (instance, stream)
    jb = {}
    for a in ARMS:
        J_, B_ = by.get(("JPC", a), []), by.get(("B3", a), [])
        if not J_ or not B_:
            continue
        per = defaultdict(lambda: [0, 0])
        for r in J_:
            per[(r["instance"], r["stream"])][0] += r["cost_tau3000"]
        bkeys = set()
        for r in B_:
            per[(r["instance"], r["stream"])][1] += r["cost_tau3000"]
            bkeys.add((r["instance"], r["stream"]))
        lr_ = [float(np.log((x + 1) / (y + 1))) for kk, (x, y) in per.items() if kk in bkeys]
        sj = sum(r["cost_tau3000"] for r in J_ if (r["instance"], r["stream"]) in bkeys)
        sb = sum(r["cost_tau3000"] for r in B_)
        jb[a] = {"log_ratio_pooled": float(np.log((sj + 1) / (sb + 1))), "ratio_pooled": (sj + 1) / (sb + 1),
                 "mean_log_ratio_per_instance_stream": float(np.mean(lr_)) if lr_ else None,
                 "n_pairs": len(lr_)}
    # ---- within-method reuse ratio on the null (full or ev1 vs off)
    reuse = {}
    for m in METHODS:
        off = by.get((m, "off"), [])
        for a in ("ev1", "full"):
            num = by.get((m, a), [])
            if num and off:
                sn, sd = sum(r["cost_tau3000"] for r in num), sum(r["cost_tau3000"] for r in off)
                reuse[f"{m}|{a}/off"] = {"stream_ratio_pooled": (sn + 1) / (sd + 1),
                                         "log_ratio_pooled": float(np.log((sn + 1) / (sd + 1)))}
    return cells, jb, reuse


def plot_null(jb, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4.6, 3.2))
        xs = [a for a in ARMS if a in jb]
        ax.bar(xs, [jb[a]["log_ratio_pooled"] for a in xs], color="#4C72B0")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel("arm")
        ax.set_ylabel("log((JPC+1)/(B3+1)), tau=3000")
        ax.set_title("Zero-effect (offset-null) environment")
        fig.tight_layout()
        fig.savefig(out_dir / "null_jpc_over_b3.png", dpi=130)
        plt.close(fig)
        return True
    except Exception:  # noqa: BLE001
        return False


def project(cells, n_inst, jpc_streams, b3_streams, table_sec_per_inst):
    rows_ = []
    for c, v in cells.items():
        m = c.split("|")[1]
        slots = n_inst * (jpc_streams if m == "JPC" else b3_streams) * 15
        rows_.append({"cell": c, "sec_per_problem": v["sec_per_problem"], "problem_slots": slots, "safety": SAFETY,
                      "cpu_s": v["sec_per_problem"] * slots * SAFETY})
    cpu_s = sum(r["cpu_s"] for r in rows_)
    allowance = 0.5 + n_inst * table_sec_per_inst / 60
    return {"n_inst": n_inst, "jpc_streams": jpc_streams, "b3_streams": b3_streams, "cells": rows_, "cpu_s": cpu_s,
            "allowance_min": allowance, "projected_min": cpu_s / (4 * 60) + allowance}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="pilot", choices=["pilot", "full"])
    ap.add_argument("--chunk", default=None, choices=[None, "a", "b"])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    global TASK
    task_id = TASK = LOCK_KEY if a.chunk is None else f"{LOCK_KEY}_{a.chunk}"
    (RES_ROOT / f"{task_id}.pid").write_text(str(os.getpid()))
    t_all = time.perf_counter()
    start = datetime.now()
    out_dir = RES_ROOT / ("pilots" if a.mode == "pilot" else "full") / task_id
    (out_dir / "parts").mkdir(parents=True, exist_ok=True)
    try:
        lock = json.loads((WS / "plan" / "prereg_lock.json").read_text())
        keep_range = None
        if a.mode == "full":
            from dsswm.stats.prereg import assert_locked
            lock = assert_locked()
            ds = lock.get("frozen_items", {}).get("downscale_decisions", {}).get(LOCK_KEY, [])
            downscaled = any(x.startswith(DOWNSCALE_KEY) for x in ds)
            if downscaled and a.chunk is None:
                raise SystemExit(f"lock downscale {DOWNSCALE_KEY} requires --chunk a|b")
            seeds_req = FULL_SEEDS
            if a.chunk is not None:
                keep_range = ZE_CHUNKS[a.chunk]
                seeds_req = [s for s in FULL_SEEDS if keep_range[0] <= s < keep_range[1]]
            groups = []
            for c, (lo, hi) in NM_CHUNKS.items():
                if any(lo <= s < hi for s in seeds_req):
                    groups.append((list(range(lo, hi)), NM_RESERVE[c]))   # whole NM chunk -> same reserve walk
            streams, n_problems = [0, 1, 2], 15
            b3_streams = [0] if downscaled else streams
        else:
            groups = [(PILOT["seeds"], DEV_RESERVE)]
            streams, n_problems, ds = PILOT["streams"], PILOT["n_problems"], []
            b3_streams = streams
        progress(0, 1, "tables")
        used, repl, qinfo, tinfo = tables(groups, keep_range)
        log(f"tables {tinfo} used {used} replacements {repl}")
        jobs = [(KIND, s, st, m, n_problems, ARMS) for s in used for m in METHODS
                for st in (b3_streams if m == "B3" else streams)]
        jobs.sort(key=lambda j: (j[3] != "B3", j[1], j[2]))   # slower B3 jobs first
        t_runs = time.perf_counter()
        outs = run_pool(jobs, out_dir, a.workers)
        runs_sec = time.perf_counter() - t_runs
        R, P, meta, errs, missing, wa = merge(jobs, out_dir)
        fatal = [o for o in outs if "fatal" in o]
        au = audit(R)
        cells, jb, reuse = analyse(R)
        plotted = plot_null(jb, out_dir)
        tsec = float(np.mean([qinfo[s]["check_sec"] for s in used if s in qinfo])) if used else 0.0
        proj_design = project(cells, 48, 3, 3, tsec)
        proj_ds = project(cells, 24, 3, 1, tsec)
        alt = {"split_3_chunks_16_inst+B3_stream0": project(cells, 16, 3, 1, tsec)["projected_min"],
               "split_2_chunks_24_inst": project(cells, 24, 3, 3, tsec)["projected_min"],
               "B3_stream0_only": project(cells, 48, 3, 1, tsec)["projected_min"]}
        chosen = proj_design if proj_design["projected_min"] <= 55.0 else proj_ds
        expected = len(jobs) * len(ARMS) * n_problems
        crashes = len(fatal) + len(missing) + len(errs)
        bill_mm = sum(not r["billing_ok"] for r in R)
        pass_flags = {"zero_crashes": crashes == 0 and len(R) == expected,
                      "billing_zero_mismatch": bill_mm == 0 and au["n_bad"] == 0 and au["replay_billed_total"] == 0,
                      "predictors_before_results_every_certification": bool(
                          wa["write_ahead_ok"] and wa["n_cert_without_predictor"] == 0
                          and wa["n_cert_predictor_late"] == 0 and wa["n_full_ev1_cert_without_full_predictors"] == 0),
                      "projected_full_le_55min": chosen["projected_min"] <= 55.0, "n_runs_ge_100": len(R) >= 100}
        go = all(pass_flags.values())
        summary = {
            "task": task_id, "mode": a.mode, "env": KIND, "seeds": used, "streams": streams, "b3_streams": b3_streams,
            "n_problems": n_problems, "methods": list(METHODS), "arms": list(ARMS), "replacements": repl,
            "quota_info": qinfo, "tables": tinfo, "n_jobs": len(jobs), "n_runs": len(R), "n_expected_runs": expected,
            "n_predictor_lines": len(P),
            "crashes": {"fatal_jobs": fatal, "missing_parts": missing, "n_errors": len(errs), "errors_head": errs[:5]},
            "billing_mismatch": bill_mm, "audit": au, "write_ahead": wa, "cells": cells,
            "stream_steps_by_arm": {c: {kk: v[kk] for kk in ("new_steps_total", "steps_tau3000_total",
                                                             "median_new_steps")} for c, v in cells.items()},
            "tie_declaration_rate": {c: v["tie_declaration_rate"] for c, v in cells.items()},
            "jpc_over_b3_null_descriptive": jb, "reuse_ratio_null_descriptive": reuse,
            "plot": "null_jpc_over_b3.png" if plotted else None,
            "runs_wall_sec": runs_sec, "total_wall_sec": time.perf_counter() - t_all,
            "timing_projection": {
                "rule": "sum(sec/problem x slots x 1.35) / (4 workers x 60) + 0.5 min + n_inst x table/check sec",
                "full_design_48inst_3streams": proj_design,
                "lock_downscale_per_chunk_24inst_B3_stream0": proj_ds, "other_options_min": alt,
                "chosen": ("full_design" if chosen is proj_design else DOWNSCALE_KEY),
                "projected_min_chosen": chosen["projected_min"],
                "lock_projection": {kk: lock.get("timing_projection", {}).get("per_task", {}).get(LOCK_KEY, {}).get(kk)
                                    for kk in ("projected_min_full_design", "projected_min_chosen")},
                "caveat": "pilot on dev seeds; eval-seed J tables may differ in cache state; 并发运行（计时偏高）"},
            "pass_criteria": pass_flags, "go_no_go": "GO" if go else "NO_GO",
            "scientific_readout": "none (pilot = smoke + timing only; dev seeds; ratios descriptive)",
            "lock_status_at_run": lock.get("status"), "lock_downscale_decisions": ds, "code_sha256": code_sha(),
            "start_time": start.isoformat(), "end_time": datetime.now().isoformat()}
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
        log(f"summary: go={go} flags={pass_flags} proj_design={proj_design['projected_min']:.1f} "
            f"proj_ds={proj_ds['projected_min']:.1f} runs={len(R)}/{expected}")
        if a.mode == "pilot":
            import run_r3_nl_main as NM
            NM.TASK = LOCK_KEY
            entry = {"candidate_id": "cand_f", "go_no_go": "GO" if go else "NO_GO", "pass_criteria": pass_flags,
                     "projected_full_min_design": proj_design["projected_min"],
                     "projected_full_min_downscaled_per_chunk": proj_ds["projected_min"],
                     "n_runs": len(R), "scientific_readout": "none (smoke + timing)"}
            md = (f"## {LOCK_KEY} (pilot, dev 734-739, stream 0, 15 offset-null problems)\n"
                  f"- 结论: **{'GO' if go else 'NO_GO'}**；runs={len(R)}/{expected}，崩溃={crashes}，"
                  f"计费不一致={bill_mm}，审计异常={au['n_bad']}\n"
                  f"- write-ahead: 认证 {wa['n_certified']} 次，缺预测记录 {wa['n_cert_without_predictor']}，"
                  f"晚于评分 {wa['n_cert_predictor_late']}\n"
                  f"- full 预计：完整设计 {proj_design['projected_min']:.1f} min；lock 降规模（2 块×24 实例 + B3 仅流 0）"
                  f"每块 {proj_ds['projected_min']:.1f} min；并发运行，计时偏高\n")
            NM.update_shared_summary(entry, md)
        mark_done("success" if go else "partial",
                    f"{task_id} {a.mode}: GO={go} runs={len(R)} proj_ds={proj_ds['projected_min']:.1f}min")
        return summary
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        log(tb)
        (out_dir / "fatal.txt").write_text(tb)
        mark_done("failed", f"{task_id} {a.mode} crashed: {e!r}")
        raise


if __name__ == "__main__":
    main()
