"""r3_p3_orth_replay_feasibility: P3 orth-replay constructibility on the reachable occupancy polytope (dev seeds only).

NL-R0 (E1-NL-S G_1, realisable), gap-quota streams. Per instance (stream 0, 15 problems):
  1. sibling platform (noise + 7919) runs JPC `full` -> shadow ledger of the `vol` arm (exactly as r3_harness);
  2. JPC runs the `orth` arm on the real platform; at begin_problem(k) the orth provider (learner side,
     dsswm.evidence.orth_design) builds xi_perp / xi_par at theta_hat (MLE of the real ledger n0 + old rows) on the
     reachable occupancy polytope of problem k (horizon H_k, current public state), Fisher-trace matched to the vol
     rows shadow.take(k, N_k) (+-5%), A(xi_perp; u_p) <= 0.05 for both pre-registered contrast directions; when
     constructed, it executes the FW-weighted mixture on a SECOND sibling platform (noise + 2 * 7919, same theta*,
     reset to the current state for every episode) and hands the N_k authentic rows to the switch
     (provenance replay_orth, never billed); otherwise the problem is ORTH_INFEASIBLE (never downgraded).
  3. every problem of the orth arm passes the reuse-switch billing assertions (env.n_steps == billed, replay = N_k).
E1-Lin (seed 740, stream 0, first 10 problems): sibling JPC-Lin `full` run -> vol rows; xi_perp / xi_par on the
deterministic load-state polytope (exact dynamics A1, theta-free Fisher), executed on a second Lin sibling.
Outputs: results.jsonl (harness scoring of the JPC orth runs), predictors.jsonl (learner-side design records,
write-ahead), lin_designs.jsonl, summary.json, samples/, figures/.
Usage: run_r3_p3_orth_replay_feasibility.py --mode {pilot,full} [--workers 4]
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
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
WS = HERE.parents[1]
RES_ROOT = WS / "exp" / "results"
CACHE = WS / "exp" / "cache" / "jtables_r3"
TASK = "r3_p3_orth_replay_feasibility"
EPS, DELTA = 0.02, 0.05
TMAX = 6000
RESERVE_START = 761
ORTH_SIB_OFFSET = 2 * 7919
MODES = {
    "pilot": {"seeds": list(range(728, 734)), "streams": [0], "n_problems": 15, "supp_streams": [1],
              "lin_seeds": [740], "lin_streams": [0], "lin_n_problems": 10},
    "full": {"seeds": list(range(720, 740)), "streams": [0], "n_problems": 15, "supp_streams": [],
             # Lin dev 740-749 after the r3_setup_lin_stream quota replacement (746 -> 750, 749 -> 751)
             "lin_seeds": [740, 741, 742, 743, 744, 745, 750, 747, 748, 751], "lin_streams": [0],
             "lin_n_problems": 15},
}
CODE_FILES = ["dsswm/evidence/orth_design.py", "dsswm/evidence/reuse_switch.py", "dsswm/baselines/switched_nl.py",
              "dsswm/streams/gap_quota.py", "dsswm/streams/r3_harness.py", "dsswm/mechanism/leverage.py",
              "run_r3_p3_orth_replay_feasibility.py"]


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


def _f(x):
    return None if x is None else float(x)


def design_record(d):
    """JSON-safe compact design record."""
    from dsswm.evidence.orth_design import strip
    out = {k: v for k, v in d.items() if k not in ("perp", "par", "min")}
    for m in ("perp", "par", "min"):
        if d.get(m) is not None:
            out[m] = strip(d[m])
    return out


# ============================================================================================ NL worker
_W: dict = {}


def _ctx():
    if "pub" not in _W:
        torch.set_num_threads(1)
        from dsswm.baselines.switched_nl import PublicNL
        from dsswm.evidence.orth_design import NLOrthModel
        from dsswm.exact.nl_propagate import NLPropagator
        from dsswm.mechanism.leverage import NLLeverage
        from dsswm.models.nl_class import NLClass
        from dsswm.streams.generator import DEFAULT_NL_S_GRID
        from dsswm.streams.gap_quota import QuotaBuilder
        ncl = NLClass(DEFAULT_NL_S_GRID, device="cpu")
        b = QuotaBuilder(MODES["pilot"]["seeds"][0], ncl, None, CACHE)
        prop = NLPropagator(2, 2, 2, b.aspace, 1.0, 0.3, device="cpu")
        TH = NLLeverage(prop).theta_matrix(ncl.np_params)
        lev = NLLeverage(prop, theta_grid=TH, eps=EPS)
        _W.update(ncl=ncl, prop=prop, TH=TH, lev=lev, model=NLOrthModel(prop, lev),
                  pub=PublicNL(prop, ncl, [], [], 1.0, 2, EPS, DELTA))
    return _W


class NLOrthProvider:
    """Learner-side orth provider bound to one switch (real ledger -> theta_hat) and one shadow ledger (vol rows).
    The sibling handle is a truth-isolated EnvHandle of the second sibling platform."""

    def __init__(self, W, pub, shadow, sib_handle, real_handle, plog, key_base, seed, noise):
        self.W, self.pub, self.shadow = W, pub, shadow
        self.sib, self.real = sib_handle, real_handle
        self.plog, self.key_base = plog, key_base
        self.seed, self.noise = seed, noise
        self.sw = None
        self.lr = None
        self.n_fed = 0
        self.records = {}

    def bind(self, sw):
        from dsswm.evidence.lr_set import SeqLRSet
        self.sw = sw
        self.lr = SeqLRSet(self.pub.prop, self.pub.LT, self.pub.delta)

    def __call__(self, k, pid, n_k, context):
        from dsswm.evidence.orth_design import nl_design_for_problem, realized_stats, run_mixture
        from dsswm.mechanism.leverage import select_pairs, theta_hat_index
        W, pub = self.W, self.pub
        t0 = time.perf_counter()
        for o, _ in self.sw.real_ledger[self.n_fed:]:          # learner's own real data (n0 + old problems)
            self.lr.update(o)
        self.n_fed = len(self.sw.real_ledger)
        mask = self.lr.mask().numpy().astype(bool)
        kh = theta_hat_index(self.lr.cum.cpu().numpy(), mask)
        J = pub.J[k]
        pi_hat = int(np.argmax(J[kh]))
        pairs = select_pairs(J, mask, kh, pi_hat)
        q = pub.problems[k]
        loads, eng = self.real.observable_state()
        s0 = pub.prop.codec.encode(loads, eng)
        rec = {**self.key_base, "problem": int(k), "pid": pid, "N_k": int(n_k), "theta_hat": int(kh),
               "set_size": int(mask.sum()), "pi_hat": pi_hat, "s0_code": int(s0), "H": int(q.H),
               "n_rows_theta_hat": int(self.n_fed)}
        if n_k == 0:
            rec.update({"status": "trivial_N0", "feasible": True, "trivial": True, "fw_sec": 0.0})
            self.plog.write(rec)
            self.records[k] = rec
            return []
        vol = [o for o, _, _ in self.shadow.take(k, n_k)]
        n_pad = sum(1 for _, p, _ in self.shadow.take(k, n_k) if p == "replay_vol_pad")
        d = nl_design_for_problem(W["model"], W["lev"], pub.LT_np[kh], W["TH"][kh], q, pairs, s0, vol,
                                  fallback_order=[int(x) for x in np.argsort(-J[kh], kind="stable")])
        rec.update({"trivial": False, "vol_pad_rows": int(n_pad), **design_record(d)})
        rows = None
        if d["feasible"]:
            rng = np.random.default_rng([int(self.seed), int(self.noise), 7919, 2, int(k)])
            te = time.perf_counter()
            r0 = self.sib.n_resets
            rows, n_ep = run_mixture(self.sib, loads, eng, d["perp"]["policies"], d["perp"]["weights"], q.H, n_k,
                                     pub.prop.codec, rng)
            dirs = [W["lev"].theta_gradient(W["lev"].contrast_leverage(W["lev"].mu_from_theta(W["TH"][kh]), q,
                                                                       p["k1"], p["k2"])["h"],
                                            W["lev"].mu_from_theta(W["TH"][kh]))
                    for p in d["pairs"] if "u_norm" in p]
            rz = realized_stats(W["lev"], rows, W["TH"][kh], dirs, d["T_vol_per_row"])
            rec.update({"realized": rz, "n_episodes": int(n_ep), "sib_resets": int(self.sib.n_resets - r0),
                        "exec_sec": time.perf_counter() - te})
        rec["provider_sec"] = time.perf_counter() - t0
        self.plog.write(rec)
        self.records[k] = rec
        return rows


def run_job(seed, stream, n_problems, out_dir):
    from dsswm.baselines.switched_nl import solve
    from dsswm.evidence.reuse_switch import BillingError, ReuseSwitch, ShadowLedger, score_truncations, TAUS
    from dsswm.mechanism.predictor_log import PredictorLog
    from dsswm.streams.gap_quota import QuotaBuilder, make_env
    from dsswm.streams.r3_harness import _rng, sibling_run
    W = _ctx()
    tag = f"i{seed}_s{stream}"
    parts = out_dir / "parts"
    ppath = parts / f"{tag}_predictors.jsonl"
    if ppath.exists():
        ppath.unlink()
    t_job = time.perf_counter()
    b = QuotaBuilder(seed, W["ncl"], None, CACHE, prop_cpu=W["prop"])
    st = b.stream(stream, "quota", n_problems=n_problems)
    pub = W["pub"].with_problems(st.problems, st.J)
    method = "JPC"
    ts = time.perf_counter()
    rows_s, env_s, pad_fn, sib_info = sibling_run(b, pub, stream, "quota", method, TMAX, n_problems)
    shadow = ShadowLedger(rows_s, [q.pid for q in st.problems], pad_fn=pad_fn)
    sib_sec = time.perf_counter() - ts
    env = st.env
    h = env.handle()
    env_o = make_env(seed, b.truth, st.noise_seed + ORTH_SIB_OFFSET)          # harness: second sibling platform
    plog = PredictorLog(ppath, fsync=False)
    prov = NLOrthProvider(W, pub, shadow, env_o.handle(), h, plog,
                          {"instance": seed, "stream": stream, "method": method, "arm": "orth"}, seed,
                          st.noise_seed)
    sw = ReuseSwitch("orth", pub.make_set_factory(method), st.init_obs, orth_provider=prov)
    prov.bind(sw)
    rng = _rng(seed, st.noise_seed, method)
    out, errs = [], []
    for k, q in enumerate(st.problems):
        t0 = time.perf_counter()
        hm = st.harness[k]
        billing_ok, err = True, None
        try:
            lr = sw.begin_problem(k, q.pid, context={"stream": stream, "kind": "quota"})
        except Exception as e:  # noqa: BLE001 - provider crash is recorded, problem counted infeasible
            errs.append({"k": k, "error": repr(e), "tb": traceback.format_exc()[-2500:]})
            sw.accounts[-1].orth_feasible = False
            lr = None
        if lr is None:
            res = {"status": "ORTH_INFEASIBLE", "pi": None, "steps": 0, "extra": {}}
        else:
            res = solve(pub, method, k, sw, lr, h, rng, TMAX)
        try:
            acc = sw.end_problem(env.n_steps)
        except BillingError as e:
            billing_ok, err = False, str(e)
            acc = sw.accounts[-1]
            sw._k = sw._pid = None
            sw._new = 0
        Jt = st.J[k][st.theta_index]
        pi = res["pi"]
        cert = res["status"] == "CERTIFIED"
        regret = float(Jt.max() - Jt[pi]) if pi is not None else None
        rec = prov.records.get(k, {})
        row = {"instance": seed, "stream": stream, "kind": "quota", "noise_seed": st.noise_seed, "problem": k,
               "problem_index": k, "pid": q.pid, "method": method, "arm": "orth", "layer": "NL-R0",
               "gap_layer": hm["gap_layer"], "true_gap": hm["true_gap"], "status": res["status"],
               "new_env_steps": int(res["steps"]), "replay_steps": int(acc.replay_steps),
               "n_rounds_total": int(acc.n_rounds_total), "n_rounds_billed": int(acc.n_rounds_billed),
               "env_n_steps": int(env.n_steps), "billing_ok": billing_ok, "billing_error": err,
               "orth_feasible": acc.orth_feasible, "orth_status": rec.get("status"), "N_k": rec.get("N_k"),
               "trivial": rec.get("trivial"), "prov_counts": acc.prov_counts, "censored": not cert,
               "zero_cost": bool(cert and res["steps"] == 0), "certified_policy": pi, "true_regret": regret,
               "false_cert": bool(cert and regret is not None and regret > pub.eps), "eps": pub.eps, "tmax": TMAX,
               "fw_sec": rec.get("fw_sec"), "provider_sec": rec.get("provider_sec"),
               "wall_clock_s": time.perf_counter() - t0}
        row.update(score_truncations(int(res["steps"]), cert, TAUS))
        out.append(row)
    assert env_o.n_steps == sum(r["replay_steps"] for r in out), "orth sibling steps != replay rows handed out"
    part = {"tag": tag, "rows": out, "errors": errs, "sibling_sec": sib_sec, "sibling": sib_info,
            "orth_sibling_steps": int(env_o.n_steps), "orth_sibling_resets": int(env_o.n_resets),
            "vol_pad_pool": len(shadow.pad_pool), "job_sec": time.perf_counter() - t_job, "worker_pid": os.getpid()}
    tmp = parts / f"{tag}.json.tmp"
    tmp.write_text(json.dumps(part, default=str))
    os.replace(tmp, parts / f"{tag}.json")
    return {"tag": tag, "n_rows": len(out), "n_err": len(errs), "sec": part["job_sec"]}


# ============================================================================================ Lin job
def run_lin_job(seed, stream, n_problems, out_dir):
    from dsswm.evidence.orth_design import LinOrthModel, alignment, lin_design_for_problem, run_mixture
    from dsswm.evidence.reuse_switch import ShadowLedger
    from dsswm.models.lin_class import ridge_estimate
    from dsswm.streams.lin_stream import LinStreamBuilder, _public, make_env, sibling_run
    t_job = time.perf_counter()
    b = LinStreamBuilder(seed)
    st = b.stream(stream, n_problems=n_problems)
    pub = _public(b, st)
    rows_s, env_s, pad_fn, sib_info = sibling_run(b, stream, "JPC-Lin", 20000, False, n_problems)
    shadow = ShadowLedger(rows_s, [q.pid for q in st.problems], pad_fn=pad_fn)
    env = st.env
    sigma = env.handle().known_constants()["sigma"]
    model = LinOrthModel(pub.geom, env.codec, b.lc.nmax, sigma)
    env_o = make_env(seed, b.truth, st.noise_seed + ORTH_SIB_OFFSET)
    ho = env_o.handle()
    recs = []
    for k, q in enumerate(st.problems):
        t0 = time.perf_counter()
        pre = shadow.prefix(k)
        n_k = len(pre)
        first = [o for o, pid in rows_s if pid == q.pid]
        s_loads = np.asarray(first[0].loads if first else q.loads0, np.int64)
        s0 = env.codec.encode(s_loads)
        rec = {"instance": seed, "stream": stream, "method": "JPC-Lin", "arm": "orth", "problem": k, "pid": q.pid,
               "N_k": n_k, "H": int(q.H), "n_policies": len(q.policies), "gap_layer_harness": st.harness[k]["gap_layer"],
               "shared": st.harness[k]["shared"]}
        if n_k == 0:
            rec.update({"status": "trivial_N0", "trivial": True, "feasible": True})
            recs.append(rec)
            continue
        vol = [o for o, _, _ in shadow.take(k, n_k)]
        X, y = [], []
        for o in list(st.init_obs) + vol:
            Xi, yi = b.lc.obs_rows(o)
            X.append(Xi)
            y.append(yi)
        th, _ = ridge_estimate(np.concatenate(X), np.concatenate(y))
        Z = pub.Z[k]
        v = Z @ th
        o_ = np.argsort(-v)
        dirs = [Z[o_[0]] - Z[o_[1]]]
        d = lin_design_for_problem(model, dirs, s0, q.H, vol)
        rec.update({"trivial": False, "pi_hat": int(o_[0]), "runner_up": int(o_[1]), **design_record(d)})
        if d["feasible"]:
            rng = np.random.default_rng([int(seed), int(st.noise_seed), 7919, 2, int(k)])
            rows, n_ep = run_mixture(ho, s_loads, np.ones(len(s_loads), np.int64), d["perp"]["policies"],
                                     d["perp"]["weights"], q.H, n_k, env.codec, rng, action_map=model.legal)
            I = model.fisher_rows(rows)
            tr = float(np.trace(I)) / len(rows)
            rec["realized"] = {"trace_per_row": tr, "trace_err": tr / d["T_vol_per_row"] - 1.0,
                               "A": [alignment(I, u) for u in dirs]}
            rec["n_episodes"] = int(n_ep)
        rec["provider_sec"] = time.perf_counter() - t0
        recs.append(rec)
    return {"tag": f"lin{seed}_s{stream}", "records": recs, "sibling": sib_info,
            "job_sec": time.perf_counter() - t_job}


# ============================================================================================ main-process stages
def build_tables(seeds):
    from dsswm.exact.nl_propagate import NLPropagator
    from dsswm.models.nl_class import NLClass
    from dsswm.streams.generator import DEFAULT_NL_S_GRID
    from dsswm.streams.gap_quota import QuotaBuilder, fill_instances
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    ncl = NLClass(DEFAULT_NL_S_GRID, device=dev)
    b0 = QuotaBuilder(seeds[0], ncl, None, CACHE)
    propg = NLPropagator(2, 2, 2, b0.aspace, 1.0, 0.3, device=dev)
    t0 = time.perf_counter()
    info, ok = {}, {}

    def check(s):
        if s not in ok:
            b = QuotaBuilder(s, ncl, propg, CACHE)
            sel = b.select("quota")
            ok[s] = not sel["quota_fail"]
            info[s] = {"quota_fail": sel["quota_fail"], "n_draws": sel["n_draws"], "layer_counts": sel["layer_counts"],
                       "tables_computed": b.n_table_computed, "table_sec": b.sec_tables}
        return ok[s]

    used, repl = fill_instances(seeds, check, reserve_start=RESERVE_START)
    vram = torch.cuda.max_memory_allocated() / 2 ** 20 if dev == "cuda" else 0.0
    prof = {"gpu_name": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
            "vram_total_mb": torch.cuda.get_device_properties(0).total_memory / 2 ** 20 if dev == "cuda" else 0,
            "max_batch_size": "n/a (class J tables: one problem x |Theta|=13824 per call; no batch dimension)",
            "vram_used_mb": vram, "utilization_pct": None,
            "note": "GPU only for class J tables of new candidate problems (cached); FW / DP / LP and JPC on CPU, "
                    "4 workers, OMP=1 each; 并发运行（与其他任务共享 4090 与 CPU）"}
    (RES_ROOT / f"{TASK}_gpu_profile.json").write_text(json.dumps(prof))
    del ncl, propg
    if dev == "cuda":
        torch.cuda.empty_cache()
    return used, repl, info, {"sec": time.perf_counter() - t0, "vram_peak_mb": vram, "device": dev}


def run_all(jobs, lin_jobs, out_dir, workers):
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    todo = [j for j in jobs if not (out_dir / "parts" / f"i{j[0]}_s{j[1]}.json").exists()]
    total = len(jobs) + len(lin_jobs)
    done = len(jobs) - len(todo)
    progress(done, total, "runs", {"resumed_jobs": done})
    log, lin_out = [], []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(run_job, *j, out_dir): ("nl", j) for j in todo}
        futs.update({ex.submit(run_lin_job, *j, out_dir): ("lin", j) for j in lin_jobs})
        for f in as_completed(futs):
            kind, j = futs[f]
            try:
                r = f.result()
            except Exception as e:  # noqa: BLE001
                r = {"tag": f"{kind}{j}", "fatal": repr(e), "tb": traceback.format_exc()[-2000:]}
            if kind == "lin" and "records" in r:
                lin_out.append(r)
                r = {"tag": r["tag"], "n_records": len(r["records"]), "sec": r["job_sec"]}
            log.append(r)
            done += 1
            progress(done, total, "runs", {"last": r.get("tag")})
            print("job", done, "/", total, r, flush=True)
    return log, lin_out


def _q(v, qs=(0.1, 0.5, 0.9)):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    if len(v) == 0:
        return None
    return {"n": int(len(v)), "mean": float(v.mean()), "min": float(v.min()), "max": float(v.max()),
            **{f"p{int(q * 100)}": float(np.quantile(v, q)) for q in qs}}


def analyse(rows, preds, lin_recs):
    by_key = {(p["instance"], p["problem"]): p for p in preds}
    nt = [p for p in preds if not p.get("trivial")]
    feas = [p for p in nt if p.get("feasible")]
    status_counts = {}
    for p in preds:
        status_counts[p.get("status")] = status_counts.get(p.get("status"), 0) + 1
    perp_tr = [abs(p["perp"]["trace_err"]) for p in feas]
    perp_A = [p["perp"]["A_max_pairs"] for p in feas]
    par_A = [p["par"]["A_max_pairs"] for p in nt if p.get("par") and p["par"]["feasible"]]
    par_tr = [abs(p["par"]["trace_err"]) for p in nt if p.get("par")]
    amin = [(p["perp"]["A_max_pairs"] if p.get("feasible") else (p["min"]["A_max_pairs"] if p.get("min") else None))
            for p in nt]
    A_vol = [max(p["A_vol"]) for p in nt if p.get("A_vol")]
    real_tr = [abs(p["realized"]["trace_err"]) for p in feas if p.get("realized")]
    real_A = [max(p["realized"]["A"]) for p in feas if p.get("realized")]
    fw = [p.get("fw_sec") for p in nt]
    prov = [p.get("provider_sec") for p in nt]
    iters = [p["perp"]["n_oracle"] for p in nt if p.get("perp")]
    cert_inf = sum(1 for p in nt if p.get("status") == "certified_infeasible")
    # gap-layer split (harness) -- descriptive only
    by_layer = {}
    for r in rows:
        p = by_key.get((r["instance"], r["problem"]))
        if p is None or p.get("trivial"):
            continue
        d = by_layer.setdefault(r["gap_layer"], [0, 0])
        d[0] += 1
        d[1] += int(bool(p.get("feasible")))
    bill_mis = sum(1 for r in rows if not r["billing_ok"])
    executed = [r for r in rows if r["orth_feasible"] and not r.get("trivial")]
    exec_bill_mis = sum(1 for r in executed if not r["billing_ok"])
    replay_ok = all((r["replay_steps"] == (r["N_k"] or 0)) for r in rows if r["orth_feasible"])
    # alternative thresholds (descriptive sensitivity, NOT the pre-registered gate)
    sens = {}
    for amax in (0.05, 0.1, 0.2):
        sens[str(amax)] = float(np.mean([a is not None and a <= amax + 1e-12 for a in amin])) if amin else None
    lin_nt = [r for r in lin_recs if not r.get("trivial")]
    lin = {"n_records": len(lin_recs), "n_nontrivial": len(lin_nt),
           "n_constructed": sum(1 for r in lin_nt if r.get("feasible")),
           "constructible_rate": (sum(1 for r in lin_nt if r.get("feasible")) / len(lin_nt)) if lin_nt else None,
           "status_counts": {s: sum(1 for r in lin_recs if r.get("status") == s)
                             for s in {r.get("status") for r in lin_recs}},
           "A_vol": _q([max(r["A_vol"]) for r in lin_nt]),
           "A_perp_constructed": _q([r["perp"]["A_max_pairs"] for r in lin_nt if r.get("feasible")]),
           "A_min_all": _q([(r["perp"]["A_max_pairs"] if r.get("feasible") else r["min"]["A_max_pairs"])
                            for r in lin_nt if r.get("perp")]),
           "A_par": _q([r["par"]["A_max_pairs"] for r in lin_nt if r.get("par")]),
           "trace_err_perp_constructed": _q([abs(r["perp"]["trace_err"]) for r in lin_nt if r.get("feasible")]),
           "realized_trace_err": _q([abs(r["realized"]["trace_err"]) for r in lin_nt if r.get("realized")]),
           "realized_A": _q([max(r["realized"]["A"]) for r in lin_nt if r.get("realized")]),
           "fw_sec": _q([r.get("fw_sec") for r in lin_nt])}
    return {
        "nl": {"n_problems": len(preds), "n_trivial_N0": len(preds) - len(nt), "n_nontrivial": len(nt),
               "n_constructed": len(feas), "n_certified_infeasible": cert_inf,
               "constructible_rate_nontrivial": len(feas) / len(nt) if nt else None,
               "constructible_rate_incl_trivial": (len(feas) + len(preds) - len(nt)) / len(preds) if preds else None,
               "status_counts": status_counts,
               "trace_match_err_perp_constructed": _q(perp_tr), "A_perp_constructed": _q(perp_A),
               "A_min_achievable_all_nontrivial": _q(amin), "A_vol": _q(A_vol), "A_par": _q(par_A),
               "trace_match_err_par": _q(par_tr), "realized_trace_err": _q(real_tr), "realized_A": _q(real_A),
               "fw_sec_per_problem": _q(fw), "provider_sec_per_problem": _q(prov), "fw_oracle_calls_perp": _q(iters),
               "constructible_by_gap_layer_harness": {k: {"n": v[0], "constructed": v[1],
                                                          "rate": v[1] / v[0] if v[0] else None}
                                                      for k, v in by_layer.items()},
               "rate_if_A_threshold_were (descriptive)": sens},
        "jpc_runs": {"n_rows": len(rows), "billing_mismatch_all": bill_mis, "n_executed_orth_data": len(executed),
                     "billing_mismatch_executed": exec_bill_mis, "replay_rows_equal_N_k": replay_ok,
                     "status_counts": {s: sum(1 for r in rows if r["status"] == s) for s in {r["status"] for r in rows}},
                     "orth_steps_mean_executed": float(np.mean([r["new_env_steps"] for r in executed]))
                     if executed else None,
                     "false_cert": int(sum(r["false_cert"] for r in rows))},
        "lin": lin,
    }


def make_plot(preds, lin_recs, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nt = [p for p in preds if not p.get("trivial")]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    bins = np.linspace(0, 0.8, 33)
    ax[0].hist([max(p["A_vol"]) for p in nt if p.get("A_vol")], bins=bins, alpha=0.5, label="vol (empirical)")
    ax[0].hist([p["perp"]["A_max_pairs"] if p.get("feasible") else p["min"]["A_max_pairs"] for p in nt
                if p.get("perp")], bins=bins, alpha=0.5, label="min-A design (xi_perp if constructed)")
    ax[0].hist([p["par"]["A_max_pairs"] for p in nt if p.get("par")], bins=bins, alpha=0.5, label="xi_par")
    ax[0].axvline(0.05, color="k", ls="--", lw=1)
    ax[0].set_xlabel("alignment A (max over contrast pairs)")
    ax[0].set_ylabel("count")
    ax[0].set_title("NL: A of vol / min-A / xi_par (trace band 5%)")
    ax[0].legend(fontsize=7)
    te = [p["perp"]["trace_err"] for p in nt if p.get("feasible")]
    tr = [p["realized"]["trace_err"] for p in nt if p.get("realized")]
    b2 = np.linspace(-0.3, 0.3, 31)
    ax[1].hist(te, bins=b2, alpha=0.6, label="planned xi_perp")
    ax[1].hist(tr, bins=b2, alpha=0.6, label="realised orth rows")
    ax[1].axvline(-0.05, color="k", ls="--", lw=1)
    ax[1].axvline(0.05, color="k", ls="--", lw=1)
    ax[1].set_xlabel("trace-match error tr I(orth)/tr I(vol) - 1")
    ax[1].set_title("NL: trace-match error (constructed)")
    ax[1].legend(fontsize=7)
    ln = [r for r in lin_recs if not r.get("trivial")]
    ax[2].hist([max(r["A_vol"]) for r in ln], bins=bins, alpha=0.5, label="vol")
    ax[2].hist([r["perp"]["A_max_pairs"] if r.get("feasible") else r["min"]["A_max_pairs"] for r in ln],
               bins=bins, alpha=0.5, label="min-A design")
    ax[2].hist([r["par"]["A_max_pairs"] for r in ln], bins=bins, alpha=0.5, label="xi_par")
    ax[2].axvline(0.05, color="k", ls="--", lw=1)
    ax[2].set_title("E1-Lin")
    ax[2].set_xlabel("alignment A")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    (out_dir / "figures").mkdir(exist_ok=True)
    fig.savefig(out_dir / "figures" / "orth_alignment_trace_hist.png", dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("pilot", "full"), default="pilot")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    cfg = MODES[a.mode]
    out_dir = RES_ROOT / ("pilots" if a.mode == "pilot" else "full") / TASK
    for d in (out_dir, out_dir / "parts", out_dir / "samples"):
        d.mkdir(parents=True, exist_ok=True)
    (RES_ROOT / f"{TASK}.pid").write_text(str(os.getpid()))
    started = datetime.now().isoformat()
    t0 = time.perf_counter()
    try:
        progress(0, 1, "jtables")
        seeds, repl, qinfo, tinfo = build_tables(cfg["seeds"])
        print("tables", tinfo, "replacements", repl, flush=True)
        jobs = [(s, st, cfg["n_problems"]) for s in seeds for st in cfg["streams"] + cfg["supp_streams"]]
        lin_jobs = [(s, st, cfg["lin_n_problems"]) for s in cfg["lin_seeds"] for st in cfg["lin_streams"]]
        t_runs = time.perf_counter()
        joblog, lin_out = run_all(jobs, lin_jobs, out_dir, a.workers)
        runs_sec = time.perf_counter() - t_runs
        from dsswm.mechanism.predictor_log import ResultLog, join, load_jsonl
        rows, preds, errs, parts_meta = [], [], [], []
        pred_path, res_path = out_dir / "predictors.jsonl", out_dir / "results.jsonl"
        for p in (pred_path, res_path):
            if p.exists():
                p.unlink()
        with open(pred_path, "w") as fp:
            for j in jobs:
                tag = f"i{j[0]}_s{j[1]}"
                pp = out_dir / "parts" / f"{tag}.json"
                if not pp.exists():
                    errs.append({"tag": tag, "error": "job part missing"})
                    continue
                part = json.loads(pp.read_text())
                rows += part["rows"]
                errs += part["errors"]
                parts_meta.append({k: v for k, v in part.items() if k not in ("rows", "errors")})
                for rec in load_jsonl(out_dir / "parts" / f"{tag}_predictors.jsonl"):
                    preds.append(rec)
                    fp.write(json.dumps(rec, sort_keys=True) + "\n")
        rlog = ResultLog(res_path, fsync=False)
        for r in rows:
            rlog.write(r)
        try:
            wa = {"n_joined": len(join(pred_path, res_path, check_order=True)), "write_ahead_ok": True}
        except RuntimeError as e:
            wa = {"write_ahead_ok": False, "error": str(e)}
        lin_recs = [r for lo in lin_out for r in lo["records"]]
        with open(out_dir / "lin_designs.jsonl", "w") as fh:
            for r in lin_recs:
                fh.write(json.dumps(r, default=str, sort_keys=True) + "\n")
        prim = set(cfg["streams"])
        an = analyse([r for r in rows if r["stream"] in prim], [p for p in preds if p["stream"] in prim], lin_recs)
        an["jpc_runs_primary_streams_only"] = an.pop("jpc_runs")
        an_all = analyse(rows, preds, lin_recs)
        an["jpc_runs"] = an_all["jpc_runs"]
        if cfg["supp_streams"]:
            sp = set(cfg["supp_streams"])
            an_s = analyse([r for r in rows if r["stream"] in sp], [p for p in preds if p["stream"] in sp], [])
            an["nl_supplement_streams"] = {"streams": sorted(sp), "purpose": "billing validation (>= 30 executed "
                                           "orth-data JPC runs); rate reported separately, NOT the pilot reading",
                                           **{k: an_s["nl"][k] for k in ("n_nontrivial", "n_constructed",
                                                                          "constructible_rate_nontrivial",
                                                                          "status_counts")}}
        try:
            make_plot([p for p in preds if p["stream"] in prim], lin_recs, out_dir)
        except Exception as e:  # noqa: BLE001
            print("plot failed", e, flush=True)
        # samples: representative constructed / infeasible designs + the JPC rows
        samp = {"constructed": [p for p in preds if p.get("status") == "constructed"][:5],
                "certified_infeasible": [p for p in preds if p.get("status") == "certified_infeasible"][:3],
                "other": [p for p in preds if p.get("status") not in ("constructed", "certified_infeasible",
                                                                      "trivial_N0")][:3],
                "lin": lin_recs[:5],
                "jpc_rows_executed": [r for r in rows if r["orth_feasible"] and not r.get("trivial")][:10]}
        (out_dir / "samples" / "designs.json").write_text(json.dumps(samp, indent=1, default=str))
        nl = an["nl"]
        n_exec = an["jpc_runs"]["n_executed_orth_data"]
        crit = {
            "constructible_rate_reported": nl["constructible_rate_nontrivial"] is not None
            and an["lin"]["constructible_rate"] is not None,
            "trace_err_le_5pct_constructed": nl["n_constructed"] == 0
            or nl["trace_match_err_perp_constructed"]["max"] <= 0.05 + 1e-9,
            "A_le_0.05_constructed": nl["n_constructed"] == 0 or nl["A_perp_constructed"]["max"] <= 0.05 + 1e-9,
            "billing_mismatch_zero_executed": an["jpc_runs"]["billing_mismatch_all"] == 0
            and an["jpc_runs"]["replay_rows_equal_N_k"],
            "executed_orth_runs_ge_30": n_exec >= 30,
            "le_3s_per_problem": nl["fw_sec_per_problem"] is not None and nl["fw_sec_per_problem"]["p90"] <= 3.0,
            "zero_crashes": len(errs) == 0 and not any("fatal" in j for j in joblog),
        }
        rate = nl["constructible_rate_nontrivial"]
        crit_info = {}
        if a.mode == "full":
            # full pass criterion = rate written to the lock; the <= 3 s/problem budget is a pilot criterion
            crit_info["le_3s_per_problem (pilot-only, informational)"] = crit.pop("le_3s_per_problem")
        summary = {"task": TASK, "mode": a.mode, "started_at": started, "finished_at": datetime.now().isoformat(),
                   "note": "并发运行（与其他任务共享 CPU 与 4090），计时偏高。可构造率本身是发现（任何值都可）。",
                   "definitions": {
                       "polytope": "conv{per-step occupancy of deterministic Markov policies, horizon H_k, start = "
                                   "current public state, transitions at theta_hat}",
                       "A": "A(xi;u) = u^T I(xi) u / (||u||^2 tr I(xi)), u = d(J(pi_hat)-J(pi'))/dtheta at theta_hat; "
                            "both pre-registered pairs (theta_hat runner-up, minimax challenger), max over pairs",
                       "trace_match": "per-row expected Fisher trace of xi vs per-row empirical Fisher trace of the "
                                      "vol rows shadow.take(k, N_k), both at theta_hat",
                       "solver": "fully-corrective Frank-Wolfe = column generation, LMO = exact finite-horizon DP, "
                                 "<= 200 oracle calls; inner band (3%, A 0.04) first, then pre-registered (5%, 0.05)",
                       "constructible": "planned xi_perp satisfies |trace err| <= 0.05 and A <= 0.05 (all pairs)",
                       "trivial_N0": "N_k = 0 (first problem of a stream): no replay rows, excluded from the rate",
                       "lin_N_k": "Lin: N_k = sibling full-arm prefix size; start state = sibling state at problem "
                                  "start; one contrast pair (theta_hat top-2, ridge on n0 + vol rows)"},
                   "design": {"seeds": seeds, "seed_replacements": repl, "streams": cfg["streams"],
                              "n_problems": cfg["n_problems"], "lin_seeds": cfg["lin_seeds"],
                              "lin_n_problems": cfg["lin_n_problems"], "eps": EPS, "delta": DELTA, "tmax": TMAX},
                   "quota": {str(s): v for s, v in qinfo.items()}, "jtables": tinfo, "runs_sec": runs_sec,
                   "parts": parts_meta, "write_ahead": wa, "errors": errs[:10], **an,
                   "pass_criteria": crit, "informational_criteria": crit_info, "passed": all(crit.values()),
                   "go_no_go": "GO" if all(crit.values()) else "NO_GO",
                   "A3o_status_dev_pilot": ("confirmatory-eligible (rate >= 0.5)" if rate is not None and rate >= 0.5
                                            else "descriptive (rate < 0.5)") + " -- pilot reading only; the lock "
                                                                                "uses the full dev study 720-739",
                   "code_sha256": code_sha(), "wall_clock_s": time.perf_counter() - t0}
        summary["metrics"] = {"constructible_rate": rate, "constructible_rate_lin": an["lin"]["constructible_rate"],
                              "trace_match_err": nl["trace_match_err_perp_constructed"],
                              "alignment_A": {"perp": nl["A_perp_constructed"], "vol": nl["A_vol"],
                                              "par": nl["A_par"], "A_min": nl["A_min_achievable_all_nontrivial"]},
                              "fw_sec_per_problem": nl["fw_sec_per_problem"]}
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
        mark_done("success", f"NL constructible {nl['n_constructed']}/{nl['n_nontrivial']} "
                             f"(rate {rate}), Lin {an['lin']['n_constructed']}/{an['lin']['n_nontrivial']}; "
                             f"go={summary['go_no_go']}")
        print(json.dumps(summary["metrics"], indent=1, default=str))
        print("criteria", crit)
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        print(tb, flush=True)
        (out_dir / "fatal.txt").write_text(tb)
        mark_done("failed", repr(e))
        raise


if __name__ == "__main__":
    main()
