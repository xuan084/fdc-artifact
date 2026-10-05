"""r3_heldout_misspec: held-out misspecification families for Q1 (methodology 2.6, 5.4).

Families (truth outside the learner's dynamic class G_1, |Theta| = 13824; learner = JPC on G_1, arms {full, ev1}):
  m1r_eps / m1r_2eps  pair synergy logit += s * S_ij with the pattern S generated on the REPLICA side
                      (replica.model.replica_synergy_pattern, rng [seed, 9071]; independent of the main package's m1
                      pattern rng [seed, 61]); s calibrated so that eta_min = min_theta max_{q,pi} |J_true - J_theta|
                      equals eps / 2 eps over the 15 stream problems
  alias               (psi, lam) moved off the grid jointly: (psi*, lam*) = (psi, lam) + s * (u_psi, u_lam), direction
                      angle phi ~ U(0.2, 1.37) rad (mixes the incentive response and the load term of retention), signs
                      point into the grid hull; target eta_min ~ U[1, 2] eps (rng [seed, 73]); theta* not on G_1
  m3                  negative control (outside the theory): hidden two-type left participants, psi_left[i, 1] =
                      psi +/- s by latent type (types = permutation of {0, 1}, rng [seed, 74]); eta_min = 2 eps;
                      16 instances x 1 stream, descriptive only (Lambda_hat_perp is predicted to fail)
Problems: the instance's round-3 gap-quota stream (class J tables truth-free, cached); gap layers are re-derived from
the TRUE J (the in-class quota layer is kept as gap_layer_r0). Platform = dsswm NLEnv carrying the misspecified truth
(CRN across arms: same noise seed, same n0 data). True J for scoring is recomputed INDEPENDENTLY by the replica
(exp/code/replica, never imports dsswm); the main propagator's J at the same truth is used only for the calibration
bisection and for the replica cross-check (|diff| <= 1e-6).
Write separation: the learner writes predictors.jsonl (S1 = Lambda_hat_perp, S2 = 1/rho*, A_k, trivial baselines
-log gap_hat/eps, log tr I^-1, eta_hat) at certification time of every problem, BEFORE any scoring; the harness
writes results.jsonl after all arms of the stream ended.

pilot: dev seeds 738-739, stream 0, all four families x {full, ev1} -> smoke + timing only (no scientific readout),
       projected full wall-clock for r3_prereg_lock.
full : assert_locked() (version 3); eval 10500-10547 x streams {0, 1} for m1r_eps, m1r_2eps, alias; m3 on
       10500-10515 x stream 0. quota_fail seeds are replaced deterministically from the reserve 10548-10599.

Usage: run_r3_heldout_misspec.py --mode {pilot,full} [--workers 4]
Concurrent run (shares 20 cores and the RTX 4090 with other round-3 tasks): timings are "concurrent".
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
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
CACHE_DYN = WS / "exp" / "cache" / "jtables_r3"
TASK = "r3_heldout_misspec"
EPS, DELTA = 0.02, 0.05
TMAX = 6000
TOP_M = 5
N_PROBLEMS = 15
ARMS = ("full", "ev1")
FAMILIES = ("m1r_eps", "m1r_2eps", "alias", "m3")
REPLICA_TOL = 1e-6
FULL_BUDGET_MIN = 55.0
MODES = {
    "pilot": {"seeds": [738, 739], "streams": [0], "m3_seeds": [738, 739], "m3_streams": [0]},
    "full": {"seeds": list(range(10500, 10548)), "streams": [0, 1], "m3_n": 16, "m3_streams": [0],
             "reserve_start": 10548, "reserve_max": 52},
}
CODE_FILES = ["run_r3_heldout_misspec.py", "replica/model.py", "dsswm/mechanism/leverage.py",
              "dsswm/mechanism/predictor_log.py", "dsswm/evidence/reuse_switch.py", "dsswm/baselines/switched_nl.py",
              "dsswm/streams/gap_quota.py", "run_r3_p4_t0_mechanism_gate.py"]


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
    if "ncl" not in _W:
        torch.set_num_threads(1)
        from dsswm.exact.nl_propagate import NLPropagator
        from dsswm.mechanism.leverage import NLLeverage
        from dsswm.models.nl_class import NLClass
        from dsswm.streams.gap_quota import QuotaBuilder
        from dsswm.streams.generator import DEFAULT_NL_S_GRID
        ncl = NLClass(DEFAULT_NL_S_GRID, device="cpu")
        b = QuotaBuilder(720, ncl, None, CACHE_DYN)
        prop = NLPropagator(2, 2, 2, b.aspace, 1.0, 0.3, device="cpu")
        TH = NLLeverage(prop).theta_matrix(ncl.np_params)
        lev = NLLeverage(prop, theta_grid=TH, eps=EPS)
        _W.update(ncl=ncl, prop=prop, TH=TH, lev=lev)
    return _W


def policy_callable(pol, aspace):
    """Bridge: opaque policy -> replica callable (t, loads, eng) -> (pairs, incentives). Data only."""
    cache = {}
    acts = aspace.actions

    def f(t, loads, eng):
        key = (t, loads, eng)
        r = cache.get(key)
        if r is None:
            a = acts[int(pol.act(t, np.array(loads), np.array(eng)))]
            r = (tuple(tuple(int(x) for x in p) for p in a.pairs),
                 {(int(i), int(j)): int(lv) for (i, j, lv) in a.incentives})
            cache[key] = r
        return r
    return f


# ============================================================================================ truths (harness)
def base_params(truth):
    """In-class R0 truth dict (gap_quota.nl_r0_truth) -> propagator parameter dict (numpy)."""
    psi = float(truth["psi"])
    return {"alpha": np.array(truth["alpha"], float), "beta": np.array(truth["beta"], float),
            "gamma": np.array(truth["gamma"], float), "tauL": np.array(truth["tauL"], float),
            "tauR": np.array(truth["tauR"], float), "psi_left": np.array([[0.0, psi], [0.0, psi]]),
            "lam": float(truth["lam"])}


def family_spec(family, seed, truth):
    """Returns (truth_fn(s) -> param dict, target eta_min, s bracket start, s max, meta)."""
    from replica.model import replica_synergy_pattern
    tp0 = base_params(truth)
    psi0, lam0 = float(truth["psi"]), float(truth["lam"])
    if family.startswith("m1r"):
        S = replica_synergy_pattern(seed)
        target = (1.0 if family == "m1r_eps" else 2.0) * EPS

        def fn(s):
            return {**tp0, "syn": s * S}
        return fn, target, 0.05, 20.0, {"pattern": S.tolist(), "pattern_rng": [int(seed), 9071]}
    if family == "alias":
        rng = np.random.default_rng([int(seed), 73])
        target = float(rng.uniform(1.0, 2.0)) * EPS
        phi = float(rng.uniform(0.2, 1.37))
        sp = 1.0 if psi0 < 0.75 else -1.0
        if 0.25 < psi0 < 0.75:
            sp = 1.0 if rng.random() < 0.5 else -1.0
        sl = 1.0 if lam0 < 0.25 else -1.0
        u = np.array([sp * math.cos(phi), sl * math.sin(phi)])

        def fn(s):
            p = {**tp0}
            ps = psi0 + s * u[0]
            p["psi_left"] = np.array([[0.0, ps], [0.0, ps]])
            p["lam"] = lam0 + s * u[1]
            return p
        return fn, target, 0.05, 1.0, {"direction": u.tolist(), "phi": phi, "psi0": psi0, "lam0": lam0}
    if family == "m3":
        types = np.random.default_rng([int(seed), 74]).permutation(2)

        def fn(s):
            p = {**tp0}
            pl = np.zeros((2, 2))
            pl[:, 1] = np.where(types == 1, psi0 + s, max(psi0 - s, -2.0))
            p["psi_left"] = pl
            return p
        return fn, 2.0 * EPS, 0.05, 4.0, {"types": types.tolist(), "psi0": psi0}
    raise ValueError(family)


def true_J_main(prop, problems, plans, tp):
    from dsswm.exact.nl_propagate import params_to_torch
    LT, EY = prop.tables(params_to_torch(tp, device="cpu"))
    out = []
    for q, pl in zip(problems, plans):
        u = q.utility
        out.append(np.array([float(prop._run_plan(a, e, LT, EY, u.w, u.w_ret, u.c_q)[0]) for a, e in pl]))
    return out


def calibrate(prop, st, fn, target, hi0, s_max):
    """Bisection on s so that eta_min(s) = min_theta max_{q,pi} |J_true(s) - J_theta| = target (main propagator)."""
    plans = [[prop.build_policy_plan(p, q.loads0, q.engaged0, q.H) for p in q.policies] for q in st.problems]
    Jall = np.concatenate(st.J, 1)

    def eta_min(s):
        jt = np.concatenate(true_J_main(prop, st.problems, plans, fn(s)))
        dev = np.abs(Jall - jt[None]).max(1)
        k = int(np.argmin(dev))
        return float(dev[k]), k

    lo, hi, it = 0.0, hi0, 0
    while eta_min(hi)[0] < target and hi < s_max:
        lo, hi = hi, min(hi * 2, s_max)
        it += 1
    reached_bracket = eta_min(hi)[0] >= target
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        v = eta_min(mid)[0]
        it += 1
        if v < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-6 or abs(v - target) < 1e-4 * target:
            break
    s = 0.5 * (lo + hi)
    v, k = eta_min(s)
    return {"strength": s, "eta_min": v, "eta_min_over_eps": v / EPS, "target_over_eps": target / EPS,
            "theta_min": k, "iters": it, "bracket_reached": bool(reached_bracket),
            "calibrated": bool(abs(v - target) <= 0.02 * target)}, plans


def replica_J(tp, problems, aspace):
    """Independent truth J by the replica (no dsswm import inside replica)."""
    from replica.model import ReplicaNL
    syn = tp.get("syn")
    pl = np.asarray(tp["psi_left"], float)
    rep = ReplicaNL(tp["alpha"], tp["beta"], tp["gamma"], tp["tauL"], tp["tauR"], [float(pl[0, 1])], float(tp["lam"]),
                    c=1.0, rho_ret=0.3, L=2, R=2, nmax=2, y_mode="both_engaged", ret_load="pre",
                    syn=syn, psi_left=pl)
    out = []
    for q in problems:
        u = q.utility
        w, w_ret, c_q = np.asarray(u.w, float), float(u.w_ret), float(u.c_q)
        out.append(np.array([rep.exact_J(policy_callable(p, aspace), q.loads0, q.engaged0, q.H, w, w_ret, c_q)
                             for p in q.policies]))
    return out


def make_platform(seed, stream_noise, tp):
    from dsswm.envs.nl import NLEnv
    from dsswm.streams.generator import NL_DEFAULTS
    pl = np.asarray(tp["psi_left"], float)
    return NLEnv(tp["alpha"], tp["beta"], tp["gamma"], tp["tauL"], tp["tauR"], [float(pl[0, 1])], tp["lam"],
                 c=NL_DEFAULTS["c"], rho_ret=NL_DEFAULTS["rho_ret"], L=2, R=2, nmax=NL_DEFAULTS["nmax"],
                 budget=NL_DEFAULTS["budget"], incentive_levels=(1,), seed=int(seed) * 1000 + int(stream_noise),
                 syn=tp.get("syn"), psi_left=pl)


# ============================================================================================ job
def job(seed, family, stream, out_dir):
    """One (instance, family, stream): calibrate truth, run JPC x {full, ev1} (write-ahead predictors), then score."""
    import run_r3_p4_t0_mechanism_gate as p4
    from dsswm.baselines.switched_nl import PublicNL, solve
    from dsswm.certify.minimax_enum import certify_minimax
    from dsswm.evidence.reuse_switch import BillingError, ReuseSwitch
    from dsswm.mechanism.leverage import theta_hat_index
    from dsswm.mechanism.predictor_log import PredictorLog, ResultLog
    from dsswm.streams.gap_quota import STREAM_NOISE, QuotaBuilder, gap_layer, top2_gap
    from dsswm.streams.generator import NL_DEFAULTS, _initial_data
    W = _ctx()
    prop, ncl, lev, TH = W["prop"], W["ncl"], W["lev"], W["TH"]
    t_job = time.perf_counter()
    tag = f"i{seed}_{family}_s{stream}"
    parts = out_dir / "parts"
    pp, rp = parts / f"{tag}_pred.jsonl", parts / f"{tag}_res.jsonl"
    for p in (pp, rp):
        p.unlink(missing_ok=True)
    plog, rlog = PredictorLog(pp, fsync=False), ResultLog(rp, fsync=False)
    b = QuotaBuilder(seed, ncl, None, CACHE_DYN, prop_cpu=prop)
    st = b.stream(stream, "quota", n_problems=N_PROBLEMS)
    timing = {}
    # ------------------------------------------------------------------ harness: truth + calibration + replica J
    t0 = time.perf_counter()
    fn, target, hi0, s_max, fmeta = family_spec(family, seed, st.truth)
    calib, _plans = calibrate(prop, st, fn, target, hi0, s_max)
    tp = fn(calib["strength"])
    timing["calibration_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    J_main = true_J_main(prop, st.problems, _plans, tp)
    timing["main_J_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    J_rep = replica_J(tp, st.problems, b.aspace)
    timing["replica_J_s"] = time.perf_counter() - t0
    rep_diff = [float(np.max(np.abs(a - c))) for a, c in zip(J_main, J_rep)]
    calib.update(fmeta)
    calib["replica_vs_main_max_abs_diff"] = max(rep_diff)
    calib["theta_star_on_grid_index_in_class_R0"] = int(st.theta_index)
    calib["theta_min_is_R0_theta"] = bool(calib["theta_min"] == st.theta_index)
    # ------------------------------------------------------------------ learner (public objects only)
    pub = PublicNL(prop, ncl, st.problems, list(st.J), 1.0, 2, EPS, DELTA, top_m=TOP_M)
    method = "JPC"
    errs, runs = [], []
    for arm in ARMS:
        t_arm = time.perf_counter()
        env = make_platform(seed, STREAM_NOISE[stream], tp)
        init = _initial_data(env, NL_DEFAULTS["n0"], np.random.default_rng([seed, 12]))
        h = env.handle()
        sw = ReuseSwitch(arm, pub.make_set_factory(method), init)
        rng = np.random.default_rng([int(seed), int(STREAM_NOISE[stream]), 101])
        for k, q in enumerate(st.problems):
            key = {"instance": seed, "stream": stream, "method": f"{method}_{family}", "arm": arm, "problem": k}
            lr = sw.begin_problem(k, q.pid)
            t0 = time.perf_counter()
            try:
                res = solve(pub, method, k, sw, lr, h, rng, TMAX)
            except Exception as e:  # noqa: BLE001
                errs.append({**key, "stage": "solve", "error": repr(e), "tb": traceback.format_exc()[-2000:]})
                res = {"status": "CRASH", "pi": None, "steps": sw.new_steps, "extra": {}}
            wall = time.perf_counter() - t0
            pred_ok, t0p = False, time.perf_counter()
            try:                                              # certification-time predictors (write-ahead)
                mask = lr.mask().numpy().astype(bool)
                k_hat = theta_hat_index(lr.inner.cum.cpu().numpy(), mask)
                pi_hat = res["pi"]
                if pi_hat is None and mask.any():
                    pi_hat = certify_minimax(pub.Reg[k], mask, EPS, TOP_M)["pi"]
                if pi_hat is None:
                    pi_hat = int(np.argmax(st.J[k][k_hat]))
                r_bar = float(res["extra"].get("r_bar_end", float("nan")))
                pr = p4.learner_record(lev, TH, q, st.J[k], list(lr.obs), mask, k_hat, int(pi_hat), r_bar,
                                       with_rem=True)
                plog.write({**key, "family": family, "source": f"heldout_{family}", "status": res["status"],
                            "new_steps": int(res["steps"]), "theta_hat": k_hat, "pi_hat": int(pi_hat),
                            "set_size": int(mask.sum()), **{kk: v for kk, v in pr.items() if kk != "pairs"},
                            "pairs": [{kk: v for kk, v in p.items() if kk != "need_data"} | {"n_need_data":
                                      len(p["need_data"])} for p in pr["pairs"]]})
                pred_ok = True
            except Exception as e:  # noqa: BLE001
                errs.append({**key, "stage": "predictors", "error": repr(e), "tb": traceback.format_exc()[-2000:]})
                k_hat = None
            pred_s = time.perf_counter() - t0p
            billing_ok, berr = True, None
            try:
                acc = sw.end_problem(env.n_steps)
                billed = int(acc.n_rounds_billed)
            except BillingError as e:
                billing_ok, berr, billed = False, str(e), None
                sw._k = sw._pid = None
                sw._new = 0
            runs.append({"key": key, "res": res, "k_hat": k_hat, "pred_ok": pred_ok, "wall": wall, "pred_s": pred_s,
                         "billing_ok": billing_ok, "billing_error": berr, "billed": billed,
                         "env_n_steps": int(env.n_steps)})
        timing[f"arm_{arm}_s"] = time.perf_counter() - t_arm
    # ------------------------------------------------------------------ harness scoring (after the whole stream)
    for r in runs:
        key, res = r["key"], r["res"]
        k = key["problem"]
        Jt = np.asarray(J_rep[k], float)
        gap = top2_gap(Jt)
        pi = res["pi"]
        cert = res["status"] == "CERTIFIED"
        regret = float(Jt.max() - Jt[pi]) if pi is not None else None
        row = {**key, "family": family, "source": f"heldout_{family}", "pid": st.problems[k].pid,
               "status": res["status"], "new_env_steps": int(res["steps"]), "certified_policy": pi,
               "zero_cost": bool(cert and res["steps"] == 0), "censored": not cert,
               "true_gap": gap, "gap_layer": gap_layer(gap, EPS), "gap_layer_r0": st.harness[k]["gap_layer"],
               "true_regret": regret, "false_cert": bool(cert and regret is not None and regret > EPS),
               "truth_source": "replica", "replica_main_abs_diff": rep_diff[k],
               "billing_ok": r["billing_ok"], "billing_error": r["billing_error"], "n_rounds_billed": r["billed"],
               "env_n_steps": r["env_n_steps"], "predictor_logged": r["pred_ok"], "wall_clock_s": r["wall"],
               "predictor_s": r["pred_s"], "strength": calib["strength"], "eta_min_over_eps": calib["eta_min_over_eps"]}
        if r["k_hat"] is not None:
            Jh = st.J[k][r["k_hat"]]
            row["eta_arg"] = float(Jt.max() - Jt[int(np.argmax(Jh))])
            row["eta_dec"] = float(np.max(np.abs(Jh - Jt)))
        rlog.write(row)
    out = {"tag": tag, "seed": seed, "family": family, "stream": stream, "n_rows": len(runs), "errors": errs,
           "calibration": calib, "timing": timing, "sec": time.perf_counter() - t_job,
           "samples": [{"problem": st.problems[k].public_dict(), "J_true_replica": [round(float(x), 6) for x in J_rep[k]],
                        "J_true_main": [round(float(x), 6) for x in J_main[k]],
                        "J_theta_R0": [round(float(x), 6) for x in st.J[k][st.theta_index]]} for k in range(2)]}
    (parts / f"{tag}.json").write_text(json.dumps(out, default=str))
    return {"tag": tag, "n_rows": len(runs), "n_err": len(errs), "sec": out["sec"]}


# ============================================================================================ main-process stages
def prepare_tables(seeds):
    """Class J tables of the quota candidates on CUDA (cached; truth-free). Returns timing + quota_fail seeds."""
    from dsswm.exact.nl_propagate import NLPropagator
    from dsswm.models.nl_class import NLClass
    from dsswm.streams.gap_quota import QuotaBuilder
    from dsswm.streams.generator import DEFAULT_NL_S_GRID
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    ncl = NLClass(DEFAULT_NL_S_GRID, device=dev)
    b0 = QuotaBuilder(seeds[0], ncl, None, CACHE_DYN)
    propg = NLPropagator(2, 2, 2, b0.aspace, 1.0, 0.3, device=dev)
    t0 = time.perf_counter()
    fails, n_new, sec_tab = [], 0, 0.0
    for s in seeds:
        b = QuotaBuilder(s, ncl, propg, CACHE_DYN)
        if b.select("quota")["quota_fail"]:
            fails.append(s)
        n_new += b.n_table_computed
        sec_tab += b.sec_tables
    vram = torch.cuda.max_memory_allocated() / 2 ** 20 if dev == "cuda" else 0.0
    return {"device": dev, "sec": time.perf_counter() - t0, "n_new_tables": n_new, "sec_new_tables": sec_tab,
            "vram_peak_mb": vram, "quota_fail_seeds": fails}


def time_fresh_tables(seed, n=24):
    """Timing probe for the full stage: n class J tables recomputed on CUDA into a scratch cache (dev seed)."""
    from dsswm.exact.nl_propagate import NLPropagator
    from dsswm.models.nl_class import NLClass
    from dsswm.streams.gap_quota import QuotaBuilder
    from dsswm.streams.generator import DEFAULT_NL_S_GRID
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tmp = Path(tempfile.mkdtemp(prefix="r3_heldout_tab_"))
    try:
        ncl = NLClass(DEFAULT_NL_S_GRID, device=dev)
        b0 = QuotaBuilder(seed, ncl, None, tmp)
        propg = NLPropagator(2, 2, 2, b0.aspace, 1.0, 0.3, device=dev)
        b = QuotaBuilder(seed, ncl, propg, tmp)
        b.candidate(0)                                   # warm-up (CUDA init, params upload)
        t0 = time.perf_counter()
        for j in range(1, n + 1):
            b.candidate(j)
        per = (time.perf_counter() - t0) / n
        prof = {"gpu_name": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
                "vram_total_mb": torch.cuda.get_device_properties(0).total_memory / 2 ** 20 if dev == "cuda" else 0,
                "max_batch_size": "n/a (class J table = one problem x |Theta|=13824 per call; no batch dimension)",
                "vram_used_mb": torch.cuda.max_memory_allocated() / 2 ** 20 if dev == "cuda" else 0.0,
                "utilization_pct": None,
                "note": "GPU only for class J tables of new quota candidates (cached); calibration, replica J, JPC runs "
                        "and predictors on CPU (4 workers, OMP=1); 并发运行"}
        n_pool = len(list(CACHE_DYN.glob(f"r3q{seed}_c*_raw.npy")))
        return {"sec_per_table": per, "n_timed": n, "device": dev, "pool_size_cached_seed": n_pool}, prof
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run_pool(jobs, workers):
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    out = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(job, *a): a[:3] for a in jobs}
        for i, f in enumerate(as_completed(futs)):
            try:
                r = f.result()
            except Exception as e:  # noqa: BLE001
                r = {"tag": str(futs[f]), "fatal": repr(e), "tb": traceback.format_exc()[-2000:]}
            out.append(r)
            progress(i + 1, len(jobs), "jobs", {"last": r.get("tag")})
            log(f"job {i + 1}/{len(jobs)} {r}")
    return out


def merge(out_dir):
    from dsswm.mechanism.predictor_log import join, load_jsonl
    parts = out_dir / "parts"
    P, R, metas = [], [], []
    for f in sorted(parts.glob("i*_s[0-9].json")):
        d = json.loads(f.read_text())
        metas.append(d)
        P += load_jsonl(parts / f"{d['tag']}_pred.jsonl")
        R += load_jsonl(parts / f"{d['tag']}_res.jsonl")
    for name, rows in (("predictors.jsonl", P), ("results.jsonl", R)):
        with open(out_dir / name, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r, sort_keys=True) + "\n")
    try:
        J = join(out_dir / "predictors.jsonl", out_dir / "results.jsonl", check_order=True)
        wa = {"write_ahead_ok": True, "n_joined": len(J)}
    except RuntimeError as e:
        J = join(out_dir / "predictors.jsonl", out_dir / "results.jsonl", check_order=False)
        wa = {"write_ahead_ok": False, "error": str(e), "n_joined": len(J)}
    return J, R, metas, wa


def analyse(J, R, metas, joblog, tinfo, tab_probe, mode, workers):
    from collections import defaultdict
    errs = [e for m in metas for e in m["errors"]]
    fatal = [j for j in joblog if "fatal" in j]
    billing_bad = [r for r in R if not r["billing_ok"]]
    cert_rows = [r for r in R if r["status"] == "CERTIFIED"]
    joined = {tuple(j["key"]) for j in J}
    cert_missing_pred = [r for r in cert_rows if (r["instance"], r["stream"], r["method"], r["arm"], r["problem"])
                         not in joined]
    rep_max = max(m["calibration"]["replica_vs_main_max_abs_diff"] for m in metas) if metas else None
    # ---- timing projection for the full design
    per_family = defaultdict(list)
    for m in metas:
        per_family[m["family"]].append(m["sec"])
    fam_sec = {f: float(np.mean(v)) for f, v in per_family.items()}
    full = MODES["full"]
    n_jobs_full = {f: len(full["seeds"]) * len(full["streams"]) for f in ("m1r_eps", "m1r_2eps", "alias")}
    n_jobs_full["m3"] = full["m3_n"] * len(full["m3_streams"])
    cpu_s = sum(fam_sec.get(f, 0.0) * n for f, n in n_jobs_full.items())
    safety = 1.2
    pool = tab_probe.get("pool_size_cached_seed") or 160
    gpu_s = tab_probe["sec_per_table"] * pool * (len(full["seeds"]) + 2)    # + ~2 reserve seeds
    proj_min = (cpu_s * safety / workers + gpu_s) / 60.0
    # downscale option (if over budget): 1 stream for m1r_eps only
    opt = None
    if proj_min > FULL_BUDGET_MIN:
        cpu2 = cpu_s - fam_sec.get("m1r_eps", 0.0) * len(full["seeds"])
        opt = {"option": "m1r_eps on stream 0 only (m1r_2eps, alias keep 2 streams)",
               "projected_min": (cpu2 * safety / workers + gpu_s) / 60.0}
    # ---- descriptive smoke readout (NOT evidence; dev seeds)
    desc = {}
    for f in FAMILIES:
        for arm in ARMS:
            rr = [r for r in R if r["family"] == f and r["arm"] == arm]
            if not rr:
                continue
            c = [r for r in rr if r["status"] == "CERTIFIED"]
            zc = [r for r in c if r["zero_cost"]]
            desc[f"{f}|{arm}"] = {
                "n_runs": len(rr), "n_certified": len(c), "n_zero_cost": len(zc),
                "n_false_cert": sum(r["false_cert"] for r in c), "fcr": (sum(r["false_cert"] for r in c) / len(c)
                                                                         if c else None),
                "n_zero_cost_false": sum(r["false_cert"] for r in zc),
                "status_counts": dict(sorted({s: sum(r["status"] == s for r in rr)
                                              for s in {r["status"] for r in rr}}.items())),
                "median_new_steps": float(np.median([r["new_env_steps"] for r in rr])),
                "gap_layers_true": {l: sum(r["gap_layer"] == l for r in rr) for l in ("tie", "near", "clear")},
                "median_wall_s_per_problem": float(np.median([r["wall_clock_s"] + r["predictor_s"] for r in rr]))}
    calib = {m["tag"]: {k: m["calibration"][k] for k in ("strength", "eta_min_over_eps", "target_over_eps",
                                                         "calibrated", "bracket_reached", "theta_min_is_R0_theta",
                                                         "replica_vs_main_max_abs_diff")} for m in metas}
    # predictor sanity on the joined rows (finite S1/S2 share; not an AUC readout)
    s1 = [j["pred"].get("S1") for j in J]
    s2 = [j["pred"].get("S2") for j in J]
    pred_sanity = {"n": len(J), "S1_finite": int(sum(isinstance(x, (int, float)) and math.isfinite(x) for x in s1)),
                   "S2_finite": int(sum(isinstance(x, (int, float)) and math.isfinite(x) for x in s2)),
                   "S2_inf": int(sum(x == "inf" for x in s2)),
                   "S1_median": float(np.median([x for x in s1 if isinstance(x, (int, float))])) if s1 else None}
    gates = {"zero_crashes": not errs and not fatal and not any(r["status"] == "CRASH" for r in R),
             "billing_0_mismatch": len(billing_bad) == 0,
             "predictors_before_results_every_cert": (not cert_missing_pred) and True,
             "projected_full_le_55min": proj_min <= FULL_BUDGET_MIN,
             "replica_vs_main_le_1e-6": rep_max is not None and rep_max <= REPLICA_TOL,
             "n_runs_ge_100": len(R) >= 100}
    return {"gates": gates, "n_runs": len(R), "n_errors": len(errs), "errors_head": errs[:5], "n_fatal": len(fatal),
            "n_billing_mismatch": len(billing_bad), "n_certified": len(cert_rows),
            "n_cert_without_predictor": len(cert_missing_pred), "replica_vs_main_max_abs_diff": rep_max,
            "timing": {"mean_job_sec_by_family": fam_sec, "n_jobs_full": n_jobs_full, "cpu_s_full": cpu_s,
                       "safety": safety, "workers": workers, "gpu_table_probe": tab_probe, "gpu_s_full_tables": gpu_s,
                       "projected_full_min": proj_min, "downscale_option": opt,
                       "pilot_table_stage": tinfo},
            "calibration": calib, "descriptive_smoke_not_evidence": desc, "predictor_sanity": pred_sanity}


def write_samples(J, metas, out_dir):
    sd = out_dir / "samples"
    sd.mkdir(exist_ok=True)
    ex = {}
    for m in metas:
        ex.setdefault(m["family"], {"tag": m["tag"], "calibration": m["calibration"], "problems": m["samples"]})
    (sd / "truth_examples.json").write_text(json.dumps(ex, indent=1, default=str))
    pick = []
    for f in FAMILIES:
        fj = [j for j in J if j["res"]["family"] == f and j["res"]["status"] == "CERTIFIED"]
        fj = sorted(fj, key=lambda j: (not j["res"]["false_cert"], not j["res"]["zero_cost"]))[:3]
        pick += [{"key": j["key"], "pred": {k: j["pred"].get(k) for k in ("S1", "S2", "A_k_max", "neg_log_gap_hat",
                                                                          "log_tr_Iinv", "eta_hat_gof", "new_steps",
                                                                          "set_size", "logged_at")},
                  "res": {k: j["res"].get(k) for k in ("status", "zero_cost", "false_cert", "true_regret", "true_gap",
                                                       "gap_layer", "gap_layer_r0", "eta_arg", "eta_dec", "scored_at")}}
                 for j in fj]
    (sd / "certification_examples.json").write_text(json.dumps(pick, indent=1, default=str))


def update_gpu_progress(status, start_iso, wall_min, snapshot, planned):
    import fcntl
    p = WS / "exp" / "gpu_progress.json"
    lk = WS / "exp" / "gpu_progress.lock"
    with open(lk, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        d = json.loads(p.read_text()) if p.exists() else {}
        for k in ("completed", "failed"):
            d.setdefault(k, [])
        d.setdefault("running", {})
        d.setdefault("timings", {})
        lst = d["completed"] if status == "success" else d["failed"]
        if TASK not in lst:
            lst.append(TASK)
        d["running"].pop(TASK, None)
        d["timings"][TASK] = {"planned_min": planned, "actual_min": int(round(wall_min)), "start_time": start_iso,
                              "end_time": datetime.now().isoformat(), "config_snapshot": snapshot}
        tmp = p.with_name(p.name + f".tmp{os.getpid()}")
        tmp.write_text(json.dumps(d, indent=1))
        os.replace(tmp, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("pilot", "full"), default="pilot")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-gpu-progress", action="store_true")
    a = ap.parse_args()
    cfg = MODES[a.mode]
    out_dir = RES_ROOT / ("pilots" if a.mode == "pilot" else "full") / TASK
    (out_dir / "parts").mkdir(parents=True, exist_ok=True)
    (RES_ROOT / f"{TASK}.pid").write_text(str(os.getpid()))
    started = datetime.now().isoformat()
    t0 = time.perf_counter()
    try:
        lock_info = None
        if a.mode == "full":
            from dsswm.stats.prereg import assert_locked
            lock = assert_locked()
            lock_info = {"version": lock["version"], "sha256": lock["sha256"]}
        progress(0, 1, "tables")
        if a.mode == "full":
            from dsswm.streams.gap_quota import fill_instances
            seeds_all = cfg["seeds"] + list(range(cfg["reserve_start"], cfg["reserve_start"] + cfg["reserve_max"]))
            tinfo = prepare_tables(cfg["seeds"])
            fails = set(tinfo["quota_fail_seeds"])
            if fails:
                tinfo["reserve"] = prepare_tables(seeds_all[len(cfg["seeds"]):][: 2 * len(fails) + 2])
                fails |= set(tinfo["reserve"]["quota_fail_seeds"])
            seeds, repl = fill_instances(cfg["seeds"], lambda s: s not in fails, cfg["reserve_start"],
                                         cfg["reserve_max"])
            tinfo["replacements"] = repl
            m3_seeds = seeds[: cfg["m3_n"]]
        else:
            tinfo = prepare_tables(sorted(set(cfg["seeds"]) | set(cfg["m3_seeds"])))
            seeds = [s for s in cfg["seeds"] if s not in tinfo["quota_fail_seeds"]]
            m3_seeds = [s for s in cfg["m3_seeds"] if s not in tinfo["quota_fail_seeds"]]
        log(f"tables {tinfo}")
        tab_probe, prof = time_fresh_tables(cfg["seeds"][0] if a.mode == "pilot" else 738)
        (RES_ROOT / f"{TASK}_gpu_profile.json").write_text(json.dumps(prof))
        log(f"table probe {tab_probe}")
        jobs = [(s, f, stv, out_dir) for s in seeds for f in ("m1r_eps", "m1r_2eps", "alias") for stv in cfg["streams"]]
        jobs += [(s, "m3", stv, out_dir) for s in m3_seeds for stv in cfg["m3_streams"]]
        joblog = run_pool(jobs, min(a.workers, 4))
        J, R, metas, wa = merge(out_dir)
        an = analyse(J, R, metas, joblog, tinfo, tab_probe, a.mode, min(a.workers, 4))
        an["gates"]["write_ahead_order_ok"] = wa["write_ahead_ok"]
        write_samples(J, metas, out_dir)
        if a.mode == "pilot":
            passed = all(an["gates"].values())
        else:
            passed = an["gates"]["zero_crashes"] and an["gates"]["billing_0_mismatch"] and wa["write_ahead_ok"]
        summary = {
            "task": TASK, "mode": a.mode, "started_at": started, "finished_at": datetime.now().isoformat(),
            "note": ("pilot = smoke + timing only on dev seeds (no scientific readout); 并发运行（与 r3_static_a / "
                     "r3_nl_main_a / r3_hazard_a 共享 CPU 与 4090），计时偏高" if a.mode == "pilot" else "并发运行"),
            "design": {"seeds": seeds, "streams": cfg["streams"], "m3_seeds": m3_seeds, "m3_streams": cfg["m3_streams"],
                       "families": list(FAMILIES), "arms": list(ARMS), "method": "JPC", "n_problems": N_PROBLEMS,
                       "eps": EPS, "delta": DELTA, "tmax": TMAX, "truth_for_scoring": "replica exact J",
                       "replica_tol": REPLICA_TOL},
            "lock": lock_info, "write_ahead": wa, **an, "passed": passed, "go_no_go": "GO" if passed else "NO_GO",
            "code_sha256": code_sha(), "wall_clock_s": time.perf_counter() - t0, "jobs": joblog}
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
        wall_min = (time.perf_counter() - t0) / 60
        if not a.no_gpu_progress:
            update_gpu_progress("success" if passed else "failed", started, wall_min,
                                {"mode": a.mode, "n_jobs": len(jobs), "n_runs": an["n_runs"], "families": list(FAMILIES),
                                 "arms": list(ARMS), "workers": a.workers, "gpu_model": prof["gpu_name"],
                                 "gpu_count": 1, "projected_full_min": an["timing"]["projected_full_min"],
                                 "note": "CPU-dominated; 并发运行"}, 10 if a.mode == "pilot" else 50)
        progress(1, 1, "done", {"go_no_go": summary["go_no_go"]})
        mark_done("success" if passed else "failed",
                  f"{summary['go_no_go']} gates={an['gates']}; runs={an['n_runs']}; "
                  f"projected_full={an['timing']['projected_full_min']:.1f} min; "
                  f"replica_diff={an['replica_vs_main_max_abs_diff']:.1e}")
        log(f"done {summary['go_no_go']} in {wall_min:.1f} min; gates={an['gates']}")
    except Exception as e:  # noqa: BLE001
        (out_dir / "crash.txt").write_text(traceback.format_exc())
        mark_done("failed", repr(e))
        raise


if __name__ == "__main__":
    main()
