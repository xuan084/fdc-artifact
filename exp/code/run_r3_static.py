"""r3_static_[a-d]: NL static-twin risk layer -- reuse dose ev(m) and the multi-regime probe (C1, C2, HD1 replication,
Q1/Q2 pooled samples). Methodology 2.3 / 5.3.

Setting
  truth     NL-R0 theta* of the gap-quota builder (uniform on G_1) with gamma_i and lam multiplied by the dose
            g in {0, 0.5, 1} (g = 1: NL-R0 itself; g = 0: the exact static twin). Same problems (5/5/5 quota selected
            at g = 1), same initial-data actions, same noise seed per stream across doses (CRN).
  static    learner = parameter-count-matched static class (alpha' = alpha - gamma, tau' = tau - lam, gamma' = lam' = 0;
            |Theta| = 13824), JPC x {off, ev1, ev20, ev200, full, probe20}. probe20 = ev(20) whose 20 required new steps
            are forced multi-regime probes (acquire.multi_regime_probe: drive / keep a participant at load >= 2; billed).
  dyn_ref   JPC full on the dynamic class: G_1 at g in {0, 1}, G_ext (gamma in {0,.375,.75}, lam in {0,.25,.5},
            |Theta| = 46656) at g = 0.5 (pre-registered dynamic reference, HD1 replication).
  c_q       public class-max normalisation of the dynamic G_1 class (QuotaBuilder), identical for every class / dose.
Logging (methodology s.1 write separation): the learner writes predictors.jsonl at every certification time
  (Lambda_hat_perp = S1, S2 = 1/rho*, A_k, placebo A_k, trivial baselines) and -- static `full` arm -- at the opening of
  every problem k >= 1 (Q2 opportunity, arm 'full@open'), fsync'ed, BEFORE the harness scores the stream; results.jsonl
  is written by the harness after the stream. JOIN KEY: (instance, stream, method, arm, problem) with
  method = JPC_static_g{g} / JPC_dynref_g{g} (the dose is part of the method label).
Billing: ReuseSwitch.end_problem asserts the billing invariants after every problem (mismatches are recorded).

Pilot (smoke + timing only, NO scientific readout): dev seeds 736-737, stream 0, 8 problems, g = 1 static x 6 arms +
  dyn_ref; plus a dose-path smoke on dev 736 (g in {0, 0.5}: static full + dyn_ref, incl. the G_ext class); projects
  the full wall-clock of one chunk for the lock.
Full: chunk a/b/c/d = eval 10300-10311 / 10312-10323 / 10324-10335 / 10336-10347 x 3 streams x 15 problems x 3 doses x
  7 cells (requires the v3 lock: dsswm.stats.prereg.assert_locked()).

Usage: run_r3_static.py --chunk a --mode {pilot,full} [--workers 4]
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
from collections import defaultdict  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
WS = HERE.parents[1]
RES_ROOT = WS / "exp" / "results"
CACHE = WS / "exp" / "cache"
CACHE_DYN = CACHE / "jtables_r3"            # dynamic G_1 class tables of the quota pools (QuotaBuilder)
CACHE_STA = CACHE / "jtables_r3_static"     # static-class tables of r3 quota problems (shared with r3_p4)
CACHE_EXT = CACHE / "jtables_ext_hd1"       # G_ext class tables (r3 pids do not collide with the hd1 pids)
EPS, DELTA = 0.02, 0.05
TOP_M = 5
TMAX = 6000
N0 = 20
STATIC_ARMS = ("off", "ev1", "ev20", "ev200", "full", "probe20")
DOSES = (0.0, 0.5, 1.0)
PROBE_M = 20
CHUNKS = {"a": (10300, 10312), "b": (10312, 10324), "c": (10324, 10336), "d": (10336, 10348)}
RESERVE = {"a": 10700, "b": 10725, "c": 10750, "d": 10775}   # quota_fail replacement (unallocated, disjoint per chunk)
CODE_FILES = ["dsswm/evidence/reuse_switch.py", "dsswm/baselines/switched_nl.py", "dsswm/streams/gap_quota.py",
              "dsswm/mechanism/leverage.py", "dsswm/mechanism/predictor_log.py",
              "dsswm/acquire/multi_regime_probe.py", "run_r3_static.py"]
TASK = "r3_static_a"                        # set in main()


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


def gtag(g):
    return f"g{g:g}"


def method_label(cls, g):
    return f"JPC_{'static' if cls == 'static' else 'dynref'}_{gtag(g)}"


def dynref_class(g):
    return "dyn_Gext" if abs(g - 0.5) < 1e-12 else "dyn_G1"


def dosed_truth(truth: dict, g: float) -> dict:
    t = dict(truth)
    t["gamma"] = [float(x) * g for x in truth["gamma"]]
    t["lam"] = float(truth["lam"]) * g
    return t


def vstar_of(truth: dict) -> np.ndarray:
    return np.concatenate([truth["alpha"], truth["gamma"], truth["tauL"], truth["beta"], truth["tauR"],
                           [truth["psi"]], [truth["lam"]]]).astype(float)


# ============================================================================================ worker context
_W: dict = {}


def _ctx():
    if "prop" not in _W:
        torch.set_num_threads(1)
        from dsswm.exact.nl_propagate import NLPropagator
        from dsswm.models.nl_class import NLClass
        from dsswm.streams.gap_quota import QuotaBuilder
        from dsswm.streams.generator import DEFAULT_NL_S_GRID
        ncl = NLClass(DEFAULT_NL_S_GRID, device="cpu")
        b = QuotaBuilder(736, ncl, None, CACHE_DYN)
        prop = NLPropagator(2, 2, 2, b.aspace, 1.0, 0.3, device="cpu")
        _W.update(ncl=ncl, prop=prop, cls={}, builders={})
    return _W


def _class(cls):
    """(ncl_like, NLLeverage, TH, PublicNL base) of a learner class -- public, truth-free; cached per worker."""
    W = _ctx()
    if cls not in W["cls"]:
        from dsswm.baselines.switched_nl import PublicNL
        from dsswm.mechanism.leverage import NLLeverage
        from dsswm.models.nl_class import NLClass
        from run_r3_p4_t0_mechanism_gate import StaticNLClass
        prop = W["prop"]
        if cls == "static":
            nc = StaticNLClass(W["ncl"])
            TH = NLLeverage(prop, static=True).theta_matrix(nc.np_params)
            lev = NLLeverage(prop, static=True, theta_grid=TH, eps=EPS)
        else:
            nc = W["ncl"] if cls == "dyn_G1" else NLClass(gext_grid(), device="cpu")
            TH = NLLeverage(prop).theta_matrix(nc.np_params)
            lev = NLLeverage(prop, theta_grid=TH, eps=EPS)
        pub = PublicNL(prop, nc, [], [], 1.0, 2, EPS, DELTA, top_m=TOP_M)
        W["cls"][cls] = (nc, lev, TH, pub)
    return W["cls"][cls]


def gext_grid():
    from dsswm.models.nl_class import NLGrid
    from dsswm.streams.generator import DEFAULT_NL_S_GRID as G1
    return NLGrid(L=2, R=2, alpha=G1.alpha, gamma=(0.0, 0.375, 0.75), tauL=G1.tauL, beta=G1.beta, tauR=G1.tauR,
                  psi=G1.psi, lam=(0.0, 0.25, 0.5))


def _builder(seed):
    from dsswm.streams.gap_quota import QuotaBuilder
    W = _ctx()
    if seed not in W["builders"]:
        W["builders"] = {seed: QuotaBuilder(seed, W["ncl"], None, CACHE_DYN, prop_cpu=W["prop"])}
    return W["builders"][seed]


def learner_tables(cls, st):
    """Public class J tables of the stream problems (class-max c_q of the dynamic G_1 class)."""
    if cls == "dyn_G1":
        return list(st.J)
    d = CACHE_STA if cls == "static" else CACHE_EXT
    return [np.load(d / f"{q.pid}_raw.npy") * q.utility.c_q for q in st.problems]


PRED_KEYS_OPEN = ("S1", "S2", "A_k_max", "A_k_placebo_block_max", "A_k_placebo_global_max", "phi_perp_max",
                  "log_tr_Iinv", "neg_log_gap_hat", "eta_hat_gof", "rho_star_min", "sec_per_call")


# ============================================================================================ one cell (worker)
def job(seed, stream, g, cls, arm, n_problems, out_dir):
    from dsswm.baselines.switched_nl import solve
    from dsswm.acquire.multi_regime_probe import MultiRegimeProbe
    from dsswm.certify.eta_loc import build_plans, j_values
    from dsswm.certify.minimax_enum import certify_minimax
    from dsswm.evidence.reuse_switch import BillingError, ReuseSwitch, score_truncations, TAUS
    from dsswm.mechanism.leverage import theta_hat_index
    from dsswm.mechanism.predictor_log import PredictorLog, ResultLog
    from dsswm.streams.gap_quota import gap_layer, make_env, top2_gap
    from dsswm.streams.generator import _initial_data
    from run_r3_p4_t0_mechanism_gate import learner_record
    import run_r3_p4_t0_mechanism_gate as P4
    P4.EPS = EPS
    W = _ctx()
    prop = W["prop"]
    method = method_label(cls if cls == "static" else "dyn", g)
    tag = f"i{seed}_s{stream}_{gtag(g)}_{cls}_{arm}"
    parts = out_dir / "parts"
    pp, rp, jp = parts / f"{tag}_pred.jsonl", parts / f"{tag}_res.jsonl", parts / f"{tag}.json"
    for p in (pp, rp):
        p.unlink(missing_ok=True)
    plog, rlog = PredictorLog(pp, fsync=True), ResultLog(rp, fsync=False)
    t_job = time.perf_counter()
    b = _builder(seed)
    st = b.stream(stream, "quota", n_problems=n_problems)
    # ------------------------------------------------------------------ harness: dosed platform (CRN)
    truth_g = dosed_truth(st.truth, g)
    env = make_env(seed, truth_g, st.noise_seed)
    init = _initial_data(env, N0, np.random.default_rng([seed, 12]))
    h = env.handle()
    # ------------------------------------------------------------------ learner (public objects only)
    nc, lev, TH, pub0 = _class(cls)
    t_tab = time.perf_counter()
    J_learn = learner_tables(cls, st)
    t_tab = time.perf_counter() - t_tab
    pub = pub0.with_problems(st.problems, J_learn)
    sw_arm = f"ev{PROBE_M}" if arm == "probe20" else arm
    sw = ReuseSwitch(sw_arm, pub.make_set_factory("JPC"), init)
    rng = np.random.default_rng([int(seed), int(st.noise_seed), 101])
    prng = np.random.default_rng([int(seed), int(st.noise_seed), 101, 2020])
    probe = MultiRegimeProbe(env.aspace, 2) if arm == "probe20" else None
    keep, errs = [], []
    for k, q in enumerate(st.problems):
        key = {"instance": seed, "stream": stream, "method": method, "problem": k}
        lr = sw.begin_problem(k, q.pid)
        t_open = 0.0
        if cls == "static" and arm == "full" and k >= 1:          # Q2 opportunity (evidence at the opening of k)
            t0 = time.perf_counter()
            try:
                mask = lr.mask().numpy().astype(bool)
                k_hat = theta_hat_index(lr.inner.cum.cpu().numpy(), mask)
                c0 = certify_minimax(pub.Reg[k], mask, EPS, TOP_M)
                pi0 = int(c0["pi"]) if c0["pi"] is not None else int(np.argmax(J_learn[k][k_hat]))
                pr = learner_record(lev, TH, q, J_learn[k], list(lr.obs), mask, k_hat, pi0, float(c0["r_bar"]),
                                    with_rem=False)
                plog.write({**key, "arm": "full@open", "source": "static_open", "dose": g, "learner_class": cls,
                            "theta_hat": k_hat, "pi_hat": pi0, "cert_status_at_open": c0["status"].value,
                            **{kk: pr[kk] for kk in PRED_KEYS_OPEN if kk in pr}})
            except Exception as e:  # noqa: BLE001
                errs.append({**key, "stage": "open", "error": repr(e), "tb": traceback.format_exc()[-2000:]})
            t_open = time.perf_counter() - t0
        t0 = time.perf_counter()
        pstat = probe.run(PROBE_M, sw, h, prng) if probe is not None else {}
        res = solve(pub, "JPC", k, sw, lr, h, rng, TMAX)
        wall = time.perf_counter() - t0
        t1 = time.perf_counter()
        pred_ok, k_hat_c, pi_hat = False, None, None
        try:                                                      # certification-time predictors (write-ahead)
            mask = lr.mask().numpy().astype(bool)
            k_hat_c = theta_hat_index(lr.inner.cum.cpu().numpy(), mask)
            pi_hat, pi_src = res["pi"], "certificate"
            if pi_hat is None and mask.any():
                pi_hat, pi_src = certify_minimax(pub.Reg[k], mask, EPS, TOP_M)["pi"], "minimax_on_mask"
            if pi_hat is None:
                pi_hat, pi_src = int(np.argmax(J_learn[k][k_hat_c])), "argmax_theta_hat"
            r_bar = float(res["extra"].get("r_bar_end", float("nan")))
            pr = learner_record(lev, TH, q, J_learn[k], list(lr.obs), mask, k_hat_c, int(pi_hat), r_bar,
                                with_rem=True)
            plog.write({**key, "arm": arm, "source": f"{cls}_cert", "dose": g, "learner_class": cls, "pid": q.pid,
                        "status": res["status"], "new_steps": int(res["steps"]), "theta_hat": k_hat_c,
                        "pi_hat": int(pi_hat), "pi_hat_source": pi_src, "set_size": int(mask.sum()),
                        "n_rows_in_set": len(lr.obs), **{kk: v for kk, v in pr.items() if kk != "pairs"},
                        "pairs": [{kk: v for kk, v in p.items() if kk != "need_data"} | {"n_need_data":
                                  len(p["need_data"])} for p in pr["pairs"]]})
            pred_ok = True
        except Exception as e:  # noqa: BLE001
            errs.append({**key, "stage": "cert_pred", "error": repr(e), "tb": traceback.format_exc()[-2000:]})
        t_pred = time.perf_counter() - t1
        billing_ok, berr = True, None
        try:
            acc = sw.end_problem(env.n_steps)
        except BillingError as e:
            billing_ok, berr = False, str(e)
            acc = sw.accounts[-1]
            sw._k = sw._pid = None
            sw._new = 0
        keep.append({"k": k, "res": res, "pstat": pstat, "wall": wall, "t_pred": t_pred, "t_open": t_open,
                     "pred_ok": pred_ok, "k_hat": k_hat_c, "billing_ok": billing_ok, "billing_error": berr,
                     "billed": int(acc.n_rounds_billed), "prov": dict(acc.prov_counts)})
    # ------------------------------------------------------------------ harness scoring (after the stream)
    t_h = time.perf_counter()
    ti = st.theta_index
    V = vstar_of(truth_g)[None]
    jt_check = []
    samples = []
    for kr in keep:
        k, res = kr["k"], kr["res"]
        q = st.problems[k]
        Jt = j_values(prop, build_plans(prop, q), V, q.utility.w, q.utility.w_ret)[0] * q.utility.c_q
        if abs(g - 1.0) < 1e-12:
            jt_check.append(float(np.abs(Jt - st.J[k][ti]).max()))
        gap = top2_gap(Jt)
        pi = res["pi"]
        cert = res["status"] == "CERTIFIED"
        regret = float(Jt.max() - Jt[pi]) if pi is not None else None
        row = {"instance": seed, "stream": stream, "method": method, "arm": arm, "problem": k, "pid": q.pid,
               "dose": g, "learner_class": cls, "noise_seed": st.noise_seed, "status": res["status"],
               "new_env_steps": int(res["steps"]), "certified_policy": pi, "zero_cost": bool(cert and res["steps"] == 0),
               "censored": not cert, "true_gap": gap, "gap_layer": gap_layer(gap, EPS),
               "gap_layer_quota": st.harness[k]["gap_layer"], "problem_block": k // 5, "true_regret": regret,
               "false_cert": bool(cert and regret is not None and regret > EPS),
               "billing_ok": kr["billing_ok"], "billing_error": kr["billing_error"], "n_rounds_billed": kr["billed"],
               "prov_counts": kr["prov"], "predictor_logged": kr["pred_ok"], "wall_clock_s": kr["wall"],
               "predictor_sec": kr["t_pred"], "open_predictor_sec": kr["t_open"],
               "first_cert_step": res["extra"].get("first_cert_step"), "forced_steps": res["extra"].get("forced_steps"),
               "set_size_end": res["extra"].get("set_size"), "set_size_start": res["extra"].get("set_size_start"),
               "eps": EPS, "delta": DELTA, "tmax": TMAX, **kr["pstat"]}
        row.update(score_truncations(int(res["steps"]), cert, TAUS))
        if kr["k_hat"] is not None:
            Jh = J_learn[k][kr["k_hat"]]
            row["eta_arg"] = float(Jt.max() - Jt[int(np.argmax(Jh))])
            row["eta_dec"] = float(np.max(np.abs(Jh - Jt)))
        row["eta_best"] = float(np.min(np.max(np.abs(J_learn[k] - Jt[None]), 1)))   # best-in-class misspec
        rlog.write(row)
        if cls == "static" and arm == "full" and k >= 1:
            rlog.write({**{kk: row[kk] for kk in ("instance", "stream", "method", "problem", "dose", "learner_class",
                                                   "status", "zero_cost", "gap_layer", "gap_layer_quota",
                                                   "false_cert")}, "arm": "full@open"})
        if k < 2:
            samples.append({"tag": tag, "problem": q.public_dict(), "J_true": [round(float(x), 6) for x in Jt],
                            "J_theta_hat": None if kr["k_hat"] is None else
                            [round(float(x), 6) for x in J_learn[k][kr["k_hat"]]],
                            "row": {kk: row.get(kk) for kk in ("status", "new_env_steps", "certified_policy",
                                                               "true_regret", "false_cert", "gap_layer",
                                                               "probe_frac_load2", "prov_counts")}})
    out = {"tag": tag, "seed": seed, "stream": stream, "dose": g, "cls": cls, "arm": arm, "n_rows": len(keep),
           "errors": errs, "samples": samples, "job_sec": time.perf_counter() - t_job,
           "harness_sec": time.perf_counter() - t_h, "table_load_sec": t_tab, "env_steps": int(env.n_steps),
           "jtrue_g1_check_maxdiff": max(jt_check) if jt_check else None,
           "truth_in_learner_class": truth_in_class(truth_g, cls), "worker_pid": os.getpid()}
    tmp = jp.with_name(jp.name + ".tmp")
    tmp.write_text(json.dumps(out, default=str))
    os.replace(tmp, jp)
    return {"tag": tag, "n_rows": len(keep), "n_err": len(errs), "sec": out["job_sec"]}


def truth_in_class(truth, cls):
    from dsswm.streams.generator import DEFAULT_NL_S_GRID as G1
    if cls == "static":
        return bool(np.allclose(truth["gamma"], 0) and abs(truth["lam"]) < 1e-12)
    gr = G1 if cls == "dyn_G1" else gext_grid()

    def on(x, vals):
        return bool(np.all(np.min(np.abs(np.asarray(x, float)[..., None] - np.asarray(vals, float)), -1) < 1e-9))
    return (on(truth["gamma"], gr.gamma) and on([truth["lam"]], gr.lam))


# ============================================================================================ main-process stages
def tables(seeds, n_problems, need_ext, reserve):
    """GPU: quota selection (G_1 tables, cached), static and G_ext class tables of the selected problems (cached)."""
    from dsswm.exact.nl_propagate import NLPropagator
    from dsswm.models.nl_class import NLClass
    from dsswm.streams.gap_quota import QuotaBuilder, fill_instances
    from dsswm.streams.generator import DEFAULT_NL_S_GRID
    from dsswm.streams.utilities import Utility
    from run_r3_p4_t0_mechanism_gate import static_np_params
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for d in (CACHE_STA, CACHE_EXT):
        d.mkdir(parents=True, exist_ok=True)
    ncl = NLClass(DEFAULT_NL_S_GRID, device=dev)
    b0 = QuotaBuilder(seeds[0], ncl, None, CACHE_DYN)
    propg = NLPropagator(2, 2, 2, b0.aspace, 1.0, 0.3, device=dev)
    propc = NLPropagator(2, 2, 2, b0.aspace, 1.0, 0.3, device="cpu")
    Ps = {k: torch.as_tensor(v, device=dev, dtype=torch.float64) for k, v in static_np_params(ncl.np_params).items()}
    Px = NLClass(gext_grid(), device=dev).torch_params() if need_ext else None
    t0 = time.perf_counter()
    info, ok, sec = {}, {}, {"static": [], "ext": [], "dyn_pool": 0.0}

    def check(s):
        if s not in ok:
            b = QuotaBuilder(s, ncl, propg, CACHE_DYN, prop_cpu=propc)
            sel = b.select("quota")
            ok[s] = not sel["quota_fail"]
            sec["dyn_pool"] += b.sec_tables
            info[s] = {"quota_fail": sel["quota_fail"], "n_draws": sel["n_draws"], "layer_counts": sel["layer_counts"],
                       "dyn_tables_computed": b.n_table_computed}
        return ok[s]

    used, repl = fill_instances(seeds, check, reserve_start=reserve)
    with open(CACHE / ".r3_static_jt.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        for s in used:
            b = QuotaBuilder(s, ncl, propg, CACHE_DYN, prop_cpu=propc)
            for q in b.stream(0, "quota").problems[: 15]:
                u1 = Utility(w=q.utility.w, w_ret=q.utility.w_ret, c_q=1.0)
                for name, d, P in (("static", CACHE_STA, Ps), ("ext", CACHE_EXT, Px)):
                    if P is None:
                        continue
                    path = d / f"{q.pid}_raw.npy"
                    if path.exists():
                        continue
                    t1 = time.perf_counter()
                    raw = propg.j_table(P, q.policies, q.loads0, q.engaged0, q.H, u1)
                    tmp = path.with_name(path.stem + f".tmp{os.getpid()}.npy")
                    np.save(tmp, raw)
                    os.replace(tmp, path)
                    sec[name].append(time.perf_counter() - t1)
        fcntl.flock(lk, fcntl.LOCK_UN)
    vram = torch.cuda.max_memory_allocated() / 2 ** 20 if dev == "cuda" else 0.0
    prof = {"gpu_name": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
            "vram_total_mb": torch.cuda.get_device_properties(0).total_memory / 2 ** 20 if dev == "cuda" else 0,
            "max_batch_size": "n/a (class J tables: one problem x |Theta| (13824 static / 46656 G_ext) per call; "
                              "no batch dimension)",
            "vram_used_mb": vram, "utilization_pct": (100.0 * vram / (torch.cuda.get_device_properties(0).total_memory
                                                                       / 2 ** 20)) if dev == "cuda" else None,
            "note": "GPU only for class J tables (cached); JPC loop, predictors, harness scoring on CPU (4 workers, "
                    "OMP=1); 并发运行（与其他 3 个任务共享 4090 与 20 核）"}
    (RES_ROOT / f"{TASK}_gpu_profile.json").write_text(json.dumps(prof))
    del ncl, propg, Ps, Px
    if dev == "cuda":
        torch.cuda.empty_cache()
    return used, repl, info, {"sec": time.perf_counter() - t0, "vram_peak_mb": vram, "device": dev,
                              "n_static_new": len(sec["static"]), "n_ext_new": len(sec["ext"]),
                              "sec_per_static_table": float(np.mean(sec["static"])) if sec["static"] else None,
                              "sec_per_ext_table": float(np.mean(sec["ext"])) if sec["ext"] else None,
                              "dyn_pool_sec": sec["dyn_pool"]}


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


def jtag(j):
    seed, stream, g, cls, arm, _n = j
    return f"i{seed}_s{stream}_{gtag(g)}_{cls}_{arm}"


def merge(jobs, out_dir):
    from dsswm.mechanism.predictor_log import join, load_jsonl
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
    try:
        Jn = join(out_dir / "predictors.jsonl", out_dir / "results.jsonl", check_order=True)
        wa = {"write_ahead_ok": True, "n_joined": len(Jn)}
    except RuntimeError as e:
        wa = {"write_ahead_ok": False, "error": str(e)}
    # every certification must have a predictor line logged before its result line
    pk = {}
    for r in P:
        pk.setdefault((r["instance"], r["stream"], r["method"], r["arm"], r["problem"]), r["logged_at"])
    cert = [r for r in R if r.get("status") == "CERTIFIED" and r["arm"] != "full@open"]
    miss = [r for r in cert if (r["instance"], r["stream"], r["method"], r["arm"], r["problem"]) not in pk]
    late = [r for r in cert if (k := (r["instance"], r["stream"], r["method"], r["arm"], r["problem"])) in pk
            and pk[k] > r["scored_at"]]
    wa.update({"n_certified": len(cert), "n_cert_without_predictor": len(miss), "n_cert_predictor_late": len(late)})
    (out_dir / "samples").mkdir(exist_ok=True)
    (out_dir / "samples" / "cells.json").write_text(json.dumps(samples[:60], indent=1, default=str))
    if errs or missing:
        (out_dir / "errors.json").write_text(json.dumps({"errors": errs, "missing_parts": missing}, indent=1))
    return R, P, meta, errs, missing, wa


# ============================================================================================ analysis
def cell_key(r):
    return (r["dose"], r["learner_class"], r["arm"])


def analyse(R, meta, n_problems, chunk_slots):
    rows = [r for r in R if r["arm"] != "full@open"]
    by = defaultdict(list)
    for r in rows:
        by[cell_key(r)].append(r)
    cells = {}
    for (g, cls, arm), rr in sorted(by.items()):
        sec = [r["wall_clock_s"] + r["predictor_sec"] + r.get("open_predictor_sec", 0.0) for r in rr]
        cert = [r for r in rr if r["status"] == "CERTIFIED"]
        cells[f"{gtag(g)}|{cls}|{arm}"] = {
            "n": len(rr), "n_certified": len(cert), "n_zero_cost": sum(r["zero_cost"] for r in rr),
            "n_false_cert": sum(r["false_cert"] for r in rr),
            "fcr_descriptive": (sum(r["false_cert"] for r in rr) / len(cert)) if cert else None,
            "status_counts": dict(sorted({s: sum(r["status"] == s for r in rr) for s in {r["status"] for r in rr}}
                                         .items())),
            "mean_new_steps": float(np.mean([r["new_env_steps"] for r in rr])),
            "median_new_steps": float(np.median([r["new_env_steps"] for r in rr])),
            "sec_per_problem": float(np.mean(sec)), "sec_per_problem_method": float(np.mean([r["wall_clock_s"]
                                                                                             for r in rr])),
            "sec_per_problem_predictor": float(np.mean([r["predictor_sec"] for r in rr])),
            "billing_mismatch": sum(not r["billing_ok"] for r in rr),
            "probe_frac_load2_mean": (float(np.mean([r["probe_frac_load2"] for r in rr]))
                                      if arm == "probe20" else None),
            "probe_frac_left_load2_active_mean": (float(np.mean([r["probe_frac_left_load2_active"] for r in rr]))
                                                  if arm == "probe20" else None),
            "eta_best_median": float(np.median([r["eta_best"] for r in rr]))}
    # ---- projection of one full chunk (lock rule: sum(sec/problem x slots x safety 1.2) / (4 x 60) + allowance)
    proj_cells = []
    for g in DOSES:
        for cls, arm in [("static", a) for a in STATIC_ARMS] + [(dynref_class(g), "full")]:
            c = cells.get(f"{gtag(g)}|{cls}|{arm}") or cells.get(f"g1|{cls}|{arm}")
            src = "measured" if f"{gtag(g)}|{cls}|{arm}" in cells else "g=1 measurement reused"
            if c is None and cls == "dyn_G1":
                c, src = cells.get("g1|dyn_G1|full"), "g=1 dyn_G1 reused"
            if c is None:
                continue
            proj_cells.append({"cell": f"{gtag(g)}|{cls}|{arm}", "sec_per_problem": c["sec_per_problem"],
                               "problem_slots": chunk_slots, "safety": 1.2,
                               "cpu_s": c["sec_per_problem"] * chunk_slots * 1.2, "source": src})
    return cells, proj_cells


def update_shared_summary(entry, md):
    """Append this task's pilot entry to exp/results/pilot_summary.{json,md} under the shared lock."""
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
            if head in txt:                                   # replace an older section of this task
                s = txt.index(head)
                e = txt.find("\n## ", s + 1)
                txt = txt[:s] + md.strip() + "\n" + (txt[e:] if e >= 0 else "")
            else:
                txt = txt.rstrip() + "\n\n" + md.strip() + "\n"
            pm.write_text(txt)
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)


def main():
    global TASK
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", default="a", choices=sorted(CHUNKS))
    ap.add_argument("--mode", default="pilot", choices=["pilot", "full"])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    TASK = f"r3_static_{a.chunk}"
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
            lo, hi = CHUNKS[a.chunk]
            seeds, streams, n_problems = list(range(lo, hi)), [0, 1, 2], 15
            jobs = [(s, st, g, cls, arm, n_problems) for s in seeds for st in streams for g in DOSES
                    for cls, arm in [("static", x) for x in STATIC_ARMS] + [(dynref_class(g), "full")]]
            need_ext = True
        else:
            seeds, streams, n_problems = [736, 737], [0], 8
            jobs = [(s, 0, 1.0, cls, arm, n_problems) for s in seeds
                    for cls, arm in [("static", x) for x in STATIC_ARMS] + [("dyn_G1", "full")]]
            jobs += [(736, 0, g, cls, "full", n_problems) for g in (0.0, 0.5) for cls in ("static", dynref_class(g))]
            need_ext = True
        progress(0, len(jobs), "tables")
        used, repl, qinfo, tinfo = tables(seeds, n_problems, need_ext, RESERVE[a.chunk])
        log(f"tables {tinfo} replacements {repl}")
        if repl:
            rmap = {r["failed"]: r["replacement"] for r in repl}
            jobs = [(rmap.get(j[0], j[0]),) + tuple(j[1:]) for j in jobs]
        jobs.sort(key=lambda j: (j[0], j[2], j[3] != "dyn_Gext"))
        t_runs = time.perf_counter()
        outs = run_pool(jobs, out_dir, a.workers)
        runs_sec = time.perf_counter() - t_runs
        R, P, meta, errs, missing, wa = merge(jobs, out_dir)
        fatal = [o for o in outs if "fatal" in o]
        chunk_slots = 12 * 3 * 15
        cells, proj_cells = analyse(R, meta, n_problems, chunk_slots)
        n_ext_tables = 12 * 15
        ext_sec = tinfo["sec_per_ext_table"] or 0.0
        sta_sec = tinfo["sec_per_static_table"] or 0.0
        allowance_min = (n_ext_tables * (ext_sec + sta_sec)) / 60 + 2.0
        cpu_s = sum(c["cpu_s"] for c in proj_cells)
        proj_min = cpu_s / (4 * 60) + allowance_min
        n_runs = len([r for r in R if r["arm"] != "full@open"])
        bill_mm = sum(not r["billing_ok"] for r in R if r["arm"] != "full@open")
        jt_chk = [m["jtrue_g1_check_maxdiff"] for m in meta if m.get("jtrue_g1_check_maxdiff") is not None]
        crashes = len(fatal) + len(missing) + len(errs)
        pass_flags = {"zero_crashes": crashes == 0, "billing_zero_mismatch": bill_mm == 0,
                      "write_ahead_every_certification": bool(wa.get("write_ahead_ok") and
                                                               wa["n_cert_without_predictor"] == 0 and
                                                               wa["n_cert_predictor_late"] == 0),
                      "projected_full_le_55min": proj_min <= 55.0, "n_runs_ge_100": n_runs >= 100}
        go = all(pass_flags.values())
        downscale = None
        if proj_min > 55.0:
            ev200 = sum(c["cpu_s"] for c in proj_cells if c["cell"].endswith("|ev200"))
            downscale = {"option": "ev200 on stream 0 only (lock contingency, not a methodology preset)",
                         "projected_min": (cpu_s - ev200 * 2 / 3) / 240 + allowance_min}
        summary = {
            "task": TASK, "mode": a.mode, "seeds": used, "streams": streams, "n_problems": n_problems,
            "replacements": repl, "quota_info": qinfo, "tables": tinfo, "n_jobs": len(jobs), "n_runs": n_runs,
            "n_open_records": len([r for r in R if r["arm"] == "full@open"]), "n_predictor_lines": len(P),
            "crashes": {"fatal_jobs": fatal, "missing_parts": missing, "n_errors": len(errs)},
            "billing_mismatch": bill_mm, "write_ahead": wa, "jtrue_g1_check_maxdiff": max(jt_chk) if jt_chk else None,
            "truth_in_learner_class": {m["tag"]: m["truth_in_learner_class"] for m in meta},
            "cells": cells, "runs_wall_sec": runs_sec, "total_wall_sec": time.perf_counter() - t_all,
            "timing_projection": {"rule": "sum(sec/problem x 540 slots x 1.2) / (4 workers x 60) + allowance "
                                          "(G_ext + static tables of 12 instances x 15 problems on GPU + 2 min)",
                                  "cells": proj_cells, "cpu_s": cpu_s, "allowance_min": allowance_min,
                                  "projected_min_full_chunk": proj_min, "lock_projection_min": 50.26317374602148,
                                  "downscale_option": downscale,
                                  "caveat": "pilot = 8 problems/stream (full 15: later problems see longer ledgers); "
                                            "g=0/0.5 static arms other than full reuse g=1 timings; 并发运行"},
            "pass_criteria": pass_flags, "go_no_go": "GO" if go else "NO_GO",
            "scientific_readout": "none (pilot = smoke + timing only; dev seeds; FCR counts are descriptive)",
            "lock_status_at_run": lock.get("status"), "code_sha256": code_sha(),
            "start_time": start.isoformat(), "end_time": datetime.now().isoformat()}
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
        log(f"summary: go={go} flags={pass_flags} proj={proj_min:.1f} min runs={n_runs}")
        mark_done("success" if go else "partial", f"{TASK} {a.mode}: GO={go} runs={n_runs} proj={proj_min:.1f}min")
        return summary
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        log(tb)
        (out_dir / "fatal.txt").write_text(tb)
        mark_done("failed", f"{TASK} {a.mode} crashed: {e!r}")
        raise


if __name__ == "__main__":
    main()
