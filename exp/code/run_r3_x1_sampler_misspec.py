"""r3_x1_sampler_misspec: X1 (exploratory) DDA vs Neyman-on-leverage sampler under realizable vs misspecified truths,
plus X2 (exploratory) rho* versus the oracle ||r||_xi.

Design (methodology s.1 seed block x1 = 10670-10693; not pre-registered thresholds, descriptive):
  truths     R0         learner = dynamic G_1 class, truth = NL-R0 theta* (on G_1, realizable)
             static_g1  learner = parameter-count-matched static twin (gamma = lam = 0), truth = NL-R0 theta* (dose g=1)
             m1_2eps    learner = dynamic G_1, truth = theta* + pair synergy s * S_ij (S double-centred, rng [seed, 61]),
                        s calibrated so that eta_min = 2 eps (identical construction to r3_p4 job_dev)
  samplers   DDA        JPC's locked KL decision-directed design (dsswm.baselines.switched_nl.solve, untouched)
             NEY        Neyman-on-leverage: target cell allocation n_x  propto  |(h_hat/xi_hat)^perp(x)| * sigma_hat(x),
                        sigma_hat(x) = sqrt(mu_hat (1 - mu_hat)) at theta_hat = LR-set MLE, h_hat = contrast leverage of
                        the primary (max Lambda_perp) pair from select_pairs(J, mask, theta_hat, pi_hat), perp = residual
                        of the xi-weighted projection onto span Phi (leverage.orth_leverage) on the smoothed design
                        xi_tilde = (n + 1/2) / (N + C/2) (so NEED_DATA cells carry target mass). Action rule: tracking,
                        argmax over 1-step probes and 2-step steer-then-observe sequences (expectation under theta_hat,
                        exactly the DDA column set) of E_a[cell draws] . (tau - xi_hat); explore 0.05 (= DDA); zero
                        target mass -> DDA fallback (counted).
             Everything else identical: SeqLRSet (UI threshold log 1/delta), exact minimax certificate, reuse arm
             `full` (method JPC), CRN (same platform noise seed + n0 data per (instance, stream, truth)), T_max = 6000.
  readouts   X1: per (truth) instance-level log((S_NEY + 1)/(S_DDA + 1)), S = stream total new env steps (15 problems);
             90% instance-bootstrap CI (B = 10^4, seed 42) and TOST on R0 with margin [0.8, 1.25]; under static / m1
             the stream-cost ratio and the FCR difference (NEY - DDA) with bootstrap CI.
             X2: learner-side rho*_min (written ahead) vs harness-side oracle ||r||_xi (r = mu* - mu_hat on the
             certification design), coloured by false certification (scatter, appendix).
Write separation: predictors.jsonl (learner, at certification time, before scoring) / results.jsonl (harness, after
all samplers of the (instance, truth) stream ended).

pilot: dev seeds 734-735, stream 0, 3 truths x 2 samplers x 15 problems = 180 runs -> smoke + timing only (no
       scientific readout); also checks that the local JPC loop with sampler=DDA reproduces the package solver
       step-for-step on one stream (the NEY loop differs only in the chooser), projects full wall-clock.
full : assert_locked() (version 3); eval 10670-10693 x stream 0. quota_fail seeds are dropped and listed (the lock
       defines no reserve for x1; exploratory).

Usage: run_r3_x1_sampler_misspec.py --mode {pilot,full} [--workers 4]
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
CACHE_DYN = WS / "exp" / "cache" / "jtables_r3"
CACHE_STA = WS / "exp" / "cache" / "jtables_r3_static"
TASK = "r3_x1_sampler_misspec"
EPS, DELTA = 0.02, 0.05
TMAX = 6000
TOP_M = 5
N_PROBLEMS = 15
TRUTHS = ("R0", "static_g1", "m1_2eps")
SAMPLERS = ("DDA", "NEY")
EXPLORE = 0.05
TOST = (0.8, 1.25)
FULL_BUDGET_MIN = 55.0
MODES = {"pilot": {"seeds": [734, 735], "streams": [0]},
         "full": {"seeds": list(range(10670, 10694)), "streams": [0]}}
CODE_FILES = ["run_r3_x1_sampler_misspec.py", "run_r3_p4_t0_mechanism_gate.py", "dsswm/mechanism/leverage.py",
              "dsswm/mechanism/oracle_check.py", "dsswm/mechanism/predictor_log.py", "dsswm/baselines/switched_nl.py",
              "dsswm/acquire/nl_kl_dda.py", "dsswm/evidence/reuse_switch.py", "dsswm/streams/gap_quota.py"]


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


# ============================================================================================ Neyman-on-leverage
class NeymanLeverageSampler:
    """Learner-side (truth-free) Neyman tracking sampler on the design-orthogonal leverage residual."""

    def __init__(self, lev, TH, prop, aspace):
        self.lev, self.TH, self.prop = lev, TH, prop
        self.L, self.R, self.P, self.N, self.nb = lev.L, lev.R, lev.P, lev.N, lev.nb
        self.C, self.npair = lev.C, lev.n_pair
        self.nA, self.nS = aspace.n, prop.codec.size
        self.sigma_cache, self.E_cache, self.h_cache = {}, {}, {}
        self._build(prop.codec, aspace)

    def _ret(self, p, y, n):
        return self.npair + (p * 2 + y) * self.N + n

    def _build(self, codec, aspace):
        L, R, N, nb = self.L, self.R, self.N, self.nb
        D = np.zeros((self.nS * self.nA, self.C))
        rows, xp, ri0, ri1, rj0, rj1 = [], [], [], [], [], []
        for code in range(self.nS):
            loads, eng = codec.decode(code)
            for a_idx, a in enumerate(aspace.actions):
                row = code * self.nA + a_idx
                inc = {(i, j): l for i, j, l in a.incentives}
                grouped = np.zeros(self.P, bool)
                for i, j in a.pairs:
                    if eng[i] and eng[L + j]:
                        b = inc.get((i, j), 0)
                        x = ((i * R + j) * N + loads[i]) * nb + b
                        D[row, x] += 1.0
                        rows.append(row)
                        xp.append(x)
                        ri0.append(self._ret(i, 0, loads[i]))
                        ri1.append(self._ret(i, 1, loads[i]))
                        rj0.append(self._ret(L + j, 0, loads[L + j]))
                        rj1.append(self._ret(L + j, 1, loads[L + j]))
                        grouped[i] = grouped[L + j] = True
                for p in range(self.P):
                    if eng[p] and not grouped[p]:
                        D[row, self._ret(p, 0, loads[p])] += 1.0
        self.D = D
        self.pairs_idx = tuple(np.asarray(v, int) for v in (rows, xp, ri0, ri1, rj0, rj1))

    def expected_draws(self, kh):
        """(nS * nA, C) expected cell draws of every (state, action) under theta_hat (retention y ~ mu_pair)."""
        E = self.E_cache.get(kh)
        if E is None:
            mu = self.mu(kh)
            E = self.D.copy()
            rows, xp, ri0, ri1, rj0, rj1 = self.pairs_idx
            m = mu[xp]
            for col, w in ((ri0, 1 - m), (ri1, m), (rj0, 1 - m), (rj1, m)):
                np.add.at(E, (rows, col), w)
            if len(self.E_cache) > 64:
                self.E_cache.clear()
            self.E_cache[kh] = E
        return E

    def mu(self, kh):
        m = self.sigma_cache.get(kh)
        if m is None:
            m = self.lev.mu_from_theta(self.TH[kh])
            self.sigma_cache[kh] = m
        return m

    def target(self, problem, J, mask, kh, pi_hat, counts):
        from dsswm.mechanism.leverage import orth_leverage, select_pairs
        pairs = select_pairs(J, mask, kh, int(pi_hat))
        if not pairs:
            return None, {"reason": "no_pairs"}
        tot = counts.sum()
        xi_t = (counts + 0.5) / (tot + 0.5 * self.C)
        best = None
        for label, k1, k2 in pairs:
            key = (problem.pid, kh, int(k1), int(k2))
            h = self.h_cache.get(key)
            if h is None:
                h = self.lev.contrast_leverage(self.mu(kh), problem, k1, k2)["h"]
                if len(self.h_cache) > 512:
                    self.h_cache.clear()
                self.h_cache[key] = h
            ol = orth_leverage(h, xi_t, self.lev.Phi)
            if best is None or ol["lambda_perp"] > best[0]:
                best = (ol["lambda_perp"], ol["resid"], label)
        mu = self.mu(kh)
        w = np.abs(best[1]) * np.sqrt(mu * (1.0 - mu))
        s = w.sum()
        if not np.isfinite(s) or s <= 0:
            return None, {"reason": "zero_target"}
        return w / s, {"lambda_perp_tilde": best[0], "pair": best[2]}

    def choose(self, code, kh, tau, counts, LT_row, legal, rng):
        from dsswm.acquire.nl_kl_dda import next_state_dist
        if rng.random() < EXPLORE:
            return int(rng.choice(legal)), "explore"
        tot = counts.sum()
        xi = counts / tot if tot > 0 else counts
        d = tau - xi
        E = self.expected_draws(kh)
        s_all = (E @ d).reshape(self.nS, self.nA)[:, legal]           # one-step tracking score at every state
        s1 = s_all[code]
        best_next = s_all.max(1)                                      # best 1-step score from each next state
        two = np.empty(len(legal))
        for ia, a in enumerate(legal):
            nxt, p = next_state_dist(self.prop, LT_row, code, int(a))
            two[ia] = 0.5 * (s1[ia] + float(p @ best_next[nxt]))
        cand = np.concatenate([s1, two])
        top = np.flatnonzero(cand >= cand.max() - 1e-12)
        j = int(rng.choice(top))
        return int(legal[j % len(legal)]), ("ney1" if j < len(legal) else "ney2")


def solve_jpc_local(pub, k, sw, lr, handle, rng, tmax, sampler, ney=None):
    """JPC on problem k with a pluggable chooser. Mirrors switched_nl.solve_set_method(method='JPC') line by line;
    sampler='DDA' must reproduce the package solver step for step (checked in the pilot)."""
    from dsswm.acquire.nl_kl_dda import dda_choose
    from dsswm.baselines.switched_nl import _forced_blocking, _status_value
    from dsswm.certify.minimax_enum import certify_minimax, trichotomy
    Reg, J = pub.Reg[k], pub.J[k]
    eps, top_m = pub.eps, pub.top_m
    q = pub.problems[k]
    t0 = time.perf_counter()
    status, cert, n_forced, n_explore = None, None, 0, 0
    modes = {}
    size0 = lr.size()
    first_cert_step = None
    counts = ney.lev.cell_counts(list(lr.obs)) if ney is not None else None
    n_seen = len(lr.obs) if ney is not None else 0
    while True:
        mask = lr.mask().numpy()
        if not mask.any():
            status = "MODEL_CONFLICT"
            cert = certify_minimax(Reg, mask, eps, top_m)
            break
        cert = certify_minimax(Reg, mask, eps, top_m)
        ok = _status_value(cert) == "CERTIFIED"
        if ok and first_cert_step is None:
            first_cert_step = sw.new_steps
        if ok and sw.may_certify():
            status = "CERTIFIED"
            break
        if not ok and not pub.all_single and sw.new_steps % 10 == 0:
            amb, _, _ = trichotomy(Reg, mask, pub.gid, eps)
            if amb:
                status = "OUT_OF_SCOPE"
                break
        if sw.new_steps >= tmax:
            status = "NEED_DATA"
            break
        kh = lr.mle()
        if ok:
            blk = _forced_blocking(Reg, mask, cert["pi"], top_m)
            n_forced += 1
        else:
            blk = cert["blocking"]
        code = pub.prop.codec.encode(*handle.observable_state())
        a, mode = None, None
        if sampler == "NEY":
            obs = lr.obs
            if len(obs) > n_seen:                                    # incremental ledger counts
                counts = counts + ney.lev.cell_counts(list(obs)[n_seen:])
                n_seen = len(obs)
            pi_hat = cert["pi"] if cert["pi"] is not None else int(np.argmax(J[kh]))
            tau, _ = ney.target(q, J, mask.astype(bool), kh, pi_hat, counts)
            if tau is not None:
                a, mode = ney.choose(code, kh, tau, counts, pub.LT_np[kh], pub.legal, rng)
            else:
                mode = "dda_fallback"
        if a is None:
            margins = pub.log_thr - lr.log_ratio().numpy()[blk]
            a, info = dda_choose(code, pub.py[kh:kh + 1], pub.pe[kh:kh + 1], pub.LT_np[kh], pub.py[blk], pub.pe[blk],
                                 margins, pub.inc, pub.prop, pub.legal, rng)
            n_explore += int(info["mode"] != "dda")
            mode = mode or info["mode"]
        if mode == "explore":
            n_explore += int(sampler == "NEY")
        modes[mode] = modes.get(mode, 0) + 1
        sw.record(handle.step(a))
    mask = lr.mask().numpy()
    pi = cert["pi"] if status == "CERTIFIED" else None
    return {"status": status, "pi": pi, "steps": sw.new_steps,
            "extra": {"set_size_start": int(size0), "set_size": int(mask.sum()), "r_bar_end": float(cert["r_bar"]),
                      "first_cert_step": first_cert_step, "forced_steps": n_forced,
                      "wall_clock_s": time.perf_counter() - t0, "cutoff": float(lr.cutoff()),
                      "explore_or_zero_info_steps": n_explore, "chooser_modes": modes}}


# ============================================================================================ job
def job(seed, truth, stream, out_dir, check_local=False):
    """One (instance, truth, stream): both samplers (CRN), write-ahead predictors, then harness scoring."""
    import run_r3_p4_t0_mechanism_gate as p4
    from dsswm.baselines.switched_nl import PublicNL, solve
    from dsswm.certify.minimax_enum import certify_minimax
    from dsswm.envs.nl import NLEnv
    from dsswm.evidence.reuse_switch import BillingError, ReuseSwitch
    from dsswm.mechanism import oracle_check as oc
    from dsswm.mechanism.leverage import theta_hat_index
    from dsswm.mechanism.predictor_log import PredictorLog, ResultLog
    from dsswm.streams.gap_quota import STREAM_NOISE, gap_layer, top2_gap
    from dsswm.streams.generator import NL_DEFAULTS, _initial_data
    W = p4._ctx()
    prop = W["prop"]
    t_job = time.perf_counter()
    tag = f"i{seed}_{truth}_s{stream}"
    parts = out_dir / "parts"
    pp, rp = parts / f"{tag}_pred.jsonl", parts / f"{tag}_res.jsonl"
    for p in (pp, rp):
        p.unlink(missing_ok=True)
    plog, rlog = PredictorLog(pp, fsync=False), ResultLog(rp, fsync=False)
    b = p4._builder(seed)
    st0 = b.stream(stream, "quota", n_problems=N_PROBLEMS)
    ti = st0.theta_index
    calib, timing, errs = None, {}, []
    # ------------------------------------------------------------------ harness: truth + platform factory
    t0 = time.perf_counter()
    if truth == "m1_2eps":
        S = p4.synergy_pattern(seed)
        s, eta_min, k_min, it, J_true = p4.calibrate_m1(prop, st0, S, 2 * EPS)
        calib = {"strength": s, "eta_min": eta_min, "eta_min_over_eps": eta_min / EPS, "theta_min": k_min,
                 "theta_min_is_theta_star": bool(k_min == ti), "iters": it}
        tt = st0.truth

        def platform():
            env = NLEnv(np.array(tt["alpha"]), np.array(tt["beta"]), np.array(tt["gamma"]), np.array(tt["tauL"]),
                        np.array(tt["tauR"]), [tt["psi"]], tt["lam"], c=NL_DEFAULTS["c"],
                        rho_ret=NL_DEFAULTS["rho_ret"], L=2, R=2, nmax=NL_DEFAULTS["nmax"],
                        budget=NL_DEFAULTS["budget"], incentive_levels=(1,),
                        seed=int(seed) * 1000 + int(STREAM_NOISE[stream]), syn=s * S)
            return env, _initial_data(env, NL_DEFAULTS["n0"], np.random.default_rng([seed, 12]))
        ncl_l, lev, TH = W["ncl"], W["lev"], W["TH"]
        J_learn = list(st0.J)
    else:
        J_true = [J[ti] for J in st0.J]

        def platform():
            st = b.stream(stream, "quota", n_problems=N_PROBLEMS)      # fresh platform copy (CRN)
            return st.env, st.init_obs
        if truth == "static_g1":
            ncl_l, lev, TH = W["scl"], W["levs"], W["THs"]
            J_learn = [np.load(CACHE_STA / f"{q.pid}_raw.npy") * q.utility.c_q for q in st0.problems]
        else:
            ncl_l, lev, TH = W["ncl"], W["lev"], W["TH"]
            J_learn = list(st0.J)
    timing["truth_s"] = time.perf_counter() - t0
    # ------------------------------------------------------------------ learner (public objects only)
    pub = PublicNL(prop, ncl_l, st0.problems, J_learn, 1.0, 2, EPS, DELTA, top_m=TOP_M)
    ney = NeymanLeverageSampler(lev, TH, prop, pub.aspace)
    runs, truth_params = [], None
    local_check = None
    for sampler in SAMPLERS:
        t_arm = time.perf_counter()
        env, init = platform()
        truth_params = env.true_params()
        h = env.handle()
        sw = ReuseSwitch("full", pub.make_set_factory("JPC"), init)
        rng = np.random.default_rng([int(seed), int(STREAM_NOISE[stream]), 101])
        for k, q in enumerate(st0.problems):
            key = {"instance": seed, "stream": stream, "method": f"JPC_{sampler}_{truth}", "arm": "full",
                   "problem": k}
            lr = sw.begin_problem(k, q.pid)
            t0 = time.perf_counter()
            try:
                if sampler == "DDA":
                    res = solve(pub, "JPC", k, sw, lr, h, rng, TMAX)
                else:
                    res = solve_jpc_local(pub, k, sw, lr, h, rng, TMAX, "NEY", ney=ney)
            except Exception as e:  # noqa: BLE001
                errs.append({**key, "stage": "solve", "error": repr(e), "tb": traceback.format_exc()[-2000:]})
                res = {"status": "CRASH", "pi": None, "steps": sw.new_steps, "extra": {}}
            wall = time.perf_counter() - t0
            rec, pred_ok, t0p = {"k_hat": None}, False, time.perf_counter()
            try:                                                  # certification-time predictors (write-ahead)
                mask = lr.mask().numpy().astype(bool)
                k_hat = theta_hat_index(lr.inner.cum.cpu().numpy(), mask)
                pi_hat = res["pi"]
                if pi_hat is None and mask.any():
                    pi_hat = certify_minimax(pub.Reg[k], mask, EPS, TOP_M)["pi"]
                if pi_hat is None:
                    pi_hat = int(np.argmax(J_learn[k][k_hat]))
                r_bar = float(res["extra"].get("r_bar_end", float("nan")))
                pr = p4.learner_record(lev, TH, q, J_learn[k], list(lr.obs), mask, k_hat, int(pi_hat), r_bar,
                                       with_rem=True)
                plog.write({**key, "setting": truth, "sampler": sampler, "source": f"x1_{truth}",
                            "status": res["status"], "new_steps": int(res["steps"]), "theta_hat": k_hat,
                            "pi_hat": int(pi_hat), "set_size": int(mask.sum()),
                            **{kk: v for kk, v in pr.items() if kk != "pairs"},
                            "pairs": [{kk: v for kk, v in p.items() if kk != "need_data"} | {"n_need_data":
                                      len(p["need_data"])} for p in pr["pairs"]]})
                pred_ok = True
                prim = max(pr["pairs"], key=lambda p: p["lambda_perp"])
                rec = {"k_hat": k_hat, "pairs": [(p["k1"], p["k2"], p["label"]) for p in pr["pairs"]],
                       "prim": (prim["k1"], prim["k2"], prim["label"]), "counts": lev.cell_counts(list(lr.obs)),
                       "rho_star_min": pr["rho_star_min"], "rho_star_prim": prim["rho_star"],
                       "lambda_perp": pr["S1"], "S2": pr["S2"], "r_bar": r_bar}
            except Exception as e:  # noqa: BLE001
                errs.append({**key, "stage": "predictors", "error": repr(e), "tb": traceback.format_exc()[-2000:]})
            pred_s = time.perf_counter() - t0p
            billing_ok, berr = True, None
            try:
                acc = sw.end_problem(env.n_steps)
                billed = int(acc.n_rounds_billed)
            except BillingError as e:
                billing_ok, berr, billed = False, str(e), None
                sw._k = sw._pid = None
                sw._new = 0
            runs.append({"key": key, "sampler": sampler, "res": res, "rec": rec, "pred_ok": pred_ok, "wall": wall,
                         "pred_s": pred_s, "billing_ok": billing_ok, "billing_error": berr, "billed": billed,
                         "env_n_steps": int(env.n_steps)})
        timing[f"sampler_{sampler}_s"] = time.perf_counter() - t_arm
    # ------------------------------------------------------------------ consistency: local loop (DDA) == package
    if check_local:
        t0 = time.perf_counter()
        env, init = platform()
        h = env.handle()
        sw = ReuseSwitch("full", pub.make_set_factory("JPC"), init)
        rng = np.random.default_rng([int(seed), int(STREAM_NOISE[stream]), 101])
        ref = [r for r in runs if r["sampler"] == "DDA"]
        mism = []
        for k, q in enumerate(st0.problems):
            lr = sw.begin_problem(k, q.pid)
            res = solve_jpc_local(pub, k, sw, lr, h, rng, TMAX, "DDA")
            sw.end_problem(env.n_steps)
            r0 = ref[k]["res"]
            if (res["status"], res["pi"], res["steps"]) != (r0["status"], r0["pi"], r0["steps"]):
                mism.append({"problem": k, "local": [res["status"], res["pi"], res["steps"]],
                             "package": [r0["status"], r0["pi"], r0["steps"]]})
        local_check = {"n_problems": N_PROBLEMS, "n_mismatch": len(mism), "mismatches": mism,
                       "sec": time.perf_counter() - t0}
    # ------------------------------------------------------------------ harness scoring (after the whole stream)
    t0 = time.perf_counter()
    for r in runs:
        key, res, rec = r["key"], r["res"], r["rec"]
        k = key["problem"]
        Jt = np.asarray(J_true[k], float)
        gap = top2_gap(Jt)
        pi = res["pi"]
        cert = res["status"] == "CERTIFIED"
        regret = float(Jt.max() - Jt[pi]) if pi is not None else None
        row = {**key, "truth": truth, "sampler": r["sampler"], "source": f"x1_{truth}", "pid": st0.problems[k].pid,
               "status": res["status"], "new_env_steps": int(res["steps"]), "certified_policy": pi,
               "zero_cost": bool(cert and res["steps"] == 0), "censored": not cert, "true_gap": gap,
               "gap_layer": gap_layer(gap, EPS), "gap_layer_r0": st0.harness[k]["gap_layer"], "true_regret": regret,
               "false_cert": bool(cert and regret is not None and regret > EPS),
               "billing_ok": r["billing_ok"], "billing_error": r["billing_error"], "n_rounds_billed": r["billed"],
               "env_n_steps": r["env_n_steps"], "predictor_logged": r["pred_ok"], "wall_clock_s": r["wall"],
               "predictor_s": r["pred_s"], "chooser_modes": res["extra"].get("chooser_modes"),
               "first_cert_step": res["extra"].get("first_cert_step"), "calibration": calib}
        if rec["k_hat"] is not None:
            Jh = J_learn[k][rec["k_hat"]]
            row["eta_arg"] = float(Jt.max() - Jt[int(np.argmax(Jh))])
            row["eta_dec"] = float(np.max(np.abs(Jh - Jt)))
            try:                                                  # X2: oracle ||r||_xi on the certification design
                xi = rec["counts"] / rec["counts"].sum()
                k1, k2, lab = rec["prim"]
                chk = oc.nl_identity_check(lev, st0.problems[k], k1, k2, ncl_l.params_at(rec["k_hat"]), truth_params,
                                           xi=xi, eps=EPS)
                row.update({"x2_rho_star_min": rec["rho_star_min"], "x2_rho_star_prim": rec["rho_star_prim"],
                            "x2_lambda_perp": rec["lambda_perp"], "x2_S2": rec["S2"], "x2_r_bar": rec["r_bar"],
                            "x2_oracle_r_norm_xi": chk["r_norm_xi"], "x2_oracle_lambda_delta": chk["lambda_delta"],
                            "x2_identity_err": chk["identity_err"], "x2_rem_exact_over_eps": chk["rem_exact"] / EPS,
                            "x2_pair": lab})
            except Exception as e:  # noqa: BLE001
                errs.append({**key, "stage": "x2_oracle", "error": repr(e), "tb": traceback.format_exc()[-2000:]})
        rlog.write(row)
    timing["scoring_s"] = time.perf_counter() - t0
    out = {"tag": tag, "seed": seed, "truth": truth, "stream": stream, "n_rows": len(runs), "errors": errs,
           "calibration": calib, "timing": timing, "local_check": local_check, "sec": time.perf_counter() - t_job,
           "samples": [{"problem": st0.problems[k].public_dict(), "J_true": [round(float(x), 6) for x in J_true[k]],
                        "J_learn_theta_star_row": [round(float(x), 6) for x in J_learn[k][ti]]} for k in range(2)]}
    (parts / f"{tag}.json").write_text(json.dumps(out, default=str))
    return {"tag": tag, "n_rows": len(runs), "n_err": len(errs), "sec": out["sec"],
            "local_mismatch": None if local_check is None else local_check["n_mismatch"]}


# ============================================================================================ main-process stages
def prepare_tables(seeds):
    """Dynamic class J tables of the quota candidates + static-class J tables of the stream problems (CUDA, cached)."""
    from dsswm.exact.nl_propagate import NLPropagator
    from dsswm.models.nl_class import NLClass
    from dsswm.streams.gap_quota import QuotaBuilder
    from dsswm.streams.generator import DEFAULT_NL_S_GRID
    from dsswm.streams.utilities import Utility
    import run_r3_p4_t0_mechanism_gate as p4
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    CACHE_STA.mkdir(parents=True, exist_ok=True)
    ncl = NLClass(DEFAULT_NL_S_GRID, device=dev)
    b0 = QuotaBuilder(seeds[0], ncl, None, CACHE_DYN)
    propg = NLPropagator(2, 2, 2, b0.aspace, 1.0, 0.3, device=dev)
    propc = NLPropagator(2, 2, 2, b0.aspace, 1.0, 0.3, device="cpu")
    Ps = {k: torch.as_tensor(v, device=dev, dtype=torch.float64) for k, v in p4.static_np_params(ncl.np_params).items()}
    t0 = time.perf_counter()
    fails, n_dyn, n_sta, sec_sta = [], 0, 0, 0.0
    for s in seeds:
        b = QuotaBuilder(s, ncl, propg, CACHE_DYN, prop_cpu=propc)
        sel = b.select("quota")
        n_dyn += b.n_table_computed
        if sel["quota_fail"]:
            fails.append(s)
            continue
        for q in b.stream(0, "quota", n_problems=N_PROBLEMS).problems:
            path = CACHE_STA / f"{q.pid}_raw.npy"
            if path.exists():
                continue
            ts = time.perf_counter()
            u1 = Utility(w=q.utility.w, w_ret=q.utility.w_ret, c_q=1.0)
            raw = propg.j_table(Ps, q.policies, q.loads0, q.engaged0, q.H, u1)
            tmp = path.with_name(path.stem + f".tmp{os.getpid()}.npy")
            np.save(tmp, raw)
            os.replace(tmp, path)
            n_sta += 1
            sec_sta += time.perf_counter() - ts
    vram = torch.cuda.max_memory_allocated() / 2 ** 20 if dev == "cuda" else 0.0
    prof = {"gpu_name": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
            "vram_total_mb": torch.cuda.get_device_properties(0).total_memory / 2 ** 20 if dev == "cuda" else 0,
            "max_batch_size": "n/a (class J table = one problem x |Theta|=13824 per call; no batch dimension)",
            "vram_used_mb": vram, "utilization_pct": None,
            "note": "GPU only for dynamic/static class J tables of new problems (cached); JPC runs (both samplers), "
                    "predictors and oracle X2 checks on CPU (4 workers, OMP=1); 并发运行"}
    (RES_ROOT / f"{TASK}_gpu_profile.json").write_text(json.dumps(prof))
    del ncl, propg, Ps
    if dev == "cuda":
        torch.cuda.empty_cache()
    return {"device": dev, "sec": time.perf_counter() - t0, "n_new_dyn_tables": n_dyn, "n_new_static_tables": n_sta,
            "sec_new_static_tables": sec_sta, "vram_peak_mb": vram, "quota_fail_seeds": fails}


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


def _boot_mean(x, B=10000, seed=42):
    x = np.asarray(x, float)
    if len(x) < 2:
        return {"n": int(len(x)), "mean": float(x.mean()) if len(x) else None, "ci90": None}
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return {"n": int(len(x)), "mean": float(x.mean()), "ci90": [float(np.quantile(m, .05)), float(np.quantile(m, .95))],
            "p_lo": float(np.mean(m <= math.log(TOST[0]))), "p_hi": float(np.mean(m >= math.log(TOST[1])))}


def x1_readout(R):
    """Instance-level stream-cost log ratio NEY/DDA and FCR difference per truth (descriptive in the pilot)."""
    out = {}
    for truth in TRUTHS:
        by = {}
        for r in R:
            if r["truth"] != truth:
                continue
            d = by.setdefault(r["instance"], {s: {"S": 0, "n_cert": 0, "n_false": 0, "n": 0} for s in SAMPLERS})
            c = d[r["sampler"]]
            c["S"] += r["new_env_steps"]
            c["n"] += 1
            c["n_cert"] += int(r["status"] == "CERTIFIED")
            c["n_false"] += int(r["false_cert"])
        inst = sorted(i for i, d in by.items() if all(d[s]["n"] == N_PROBLEMS for s in SAMPLERS))
        lr = [math.log((by[i]["NEY"]["S"] + 1) / (by[i]["DDA"]["S"] + 1)) for i in inst]
        fcr = {s: (sum(by[i][s]["n_false"] for i in inst) / max(sum(by[i][s]["n_cert"] for i in inst), 1))
               for s in SAMPLERS}
        dfcr = [(by[i]["NEY"]["n_false"] / max(by[i]["NEY"]["n_cert"], 1))
                - (by[i]["DDA"]["n_false"] / max(by[i]["DDA"]["n_cert"], 1)) for i in inst]
        bm = _boot_mean(lr)
        tost = None
        if bm["ci90"] is not None:
            tost = {"margin": list(TOST), "ratio_ci90": [math.exp(bm["ci90"][0]), math.exp(bm["ci90"][1])],
                    "equivalent": bool(math.log(TOST[0]) < bm["ci90"][0] and bm["ci90"][1] < math.log(TOST[1])),
                    "p_tost": max(bm["p_lo"], bm["p_hi"])}
        out[truth] = {"n_instances": len(inst), "stream_cost": {s: [by[i][s]["S"] for i in inst] for s in SAMPLERS},
                      "mean_log_ratio_ney_over_dda": bm["mean"], "geo_ratio": math.exp(bm["mean"]) if inst else None,
                      "log_ratio_boot": bm, "tost_R0": tost if truth == "R0" else None,
                      "fcr_pooled": fcr, "n_cert": {s: sum(by[i][s]["n_cert"] for i in inst) for s in SAMPLERS},
                      "n_false": {s: sum(by[i][s]["n_false"] for i in inst) for s in SAMPLERS},
                      "fcr_diff_ney_minus_dda": _boot_mean(dfcr) if dfcr else None}
    return out


def x2_readout(R):
    rows = [r for r in R if "x2_oracle_r_norm_xi" in r]
    out = {}
    for truth in TRUTHS:
        rr = [r for r in rows if r["truth"] == truth and r["status"] == "CERTIFIED"]
        rho = np.array([r["x2_rho_star_min"] if isinstance(r["x2_rho_star_min"], (int, float)) else np.inf
                        for r in rr], float)
        rn = np.array([r["x2_oracle_r_norm_xi"] for r in rr], float)
        fin = np.isfinite(rho)
        sp = None
        if fin.sum() >= 3:
            from scipy.stats import spearmanr
            sp = float(spearmanr(rho[fin], rn[fin]).correlation)
        out[truth] = {"n_cert": len(rr), "n_rho_finite": int(fin.sum()),
                      "median_rho_star": float(np.median(rho[fin])) if fin.any() else None,
                      "median_oracle_r_norm": float(np.median(rn)) if len(rn) else None,
                      "share_rnorm_gt_rho": float(np.mean(rn[fin] > rho[fin])) if fin.any() else None,
                      "spearman_rho_vs_rnorm": sp, "n_false_cert": int(sum(r["false_cert"] for r in rr)),
                      "max_identity_err": float(max((r["x2_identity_err"] for r in rr), default=0.0))}
    return out


def analyse(J, R, metas, joblog, tinfo, mode, workers):
    from collections import defaultdict
    errs = [e for m in metas for e in m["errors"]]
    fatal = [j for j in joblog if "fatal" in j]
    billing_bad = [r for r in R if not r["billing_ok"]]
    cert_rows = [r for r in R if r["status"] == "CERTIFIED"]
    joined = {tuple(j["key"]) for j in J}
    cert_missing = [r for r in cert_rows if (r["instance"], r["stream"], r["method"], r["arm"], r["problem"])
                    not in joined]
    lc = [m["local_check"] for m in metas if m.get("local_check")]
    per_truth = defaultdict(list)
    for m in metas:
        per_truth[m["truth"]].append(m["sec"] - (m["local_check"]["sec"] if m.get("local_check") else 0.0))
    t_sec = {t: float(np.mean(v)) for t, v in per_truth.items()}
    n_full = len(MODES["full"]["seeds"]) * len(MODES["full"]["streams"])
    cpu_s = sum(t_sec.get(t, 0.0) for t in TRUTHS) * n_full
    safety = 1.2
    n_new_static = n_full * N_PROBLEMS
    sec_tab = (tinfo["sec_new_static_tables"] / tinfo["n_new_static_tables"]) if tinfo["n_new_static_tables"] else 0.6
    gpu_s = sec_tab * n_new_static * 2 + 8.5 * n_full                # static tables + dynamic quota pool (~160 x 0.053 s, heldout probe)
    proj_min = (cpu_s * safety / workers + gpu_s) / 60.0
    opt = None
    if proj_min > FULL_BUDGET_MIN:
        opt = {"option": "first 12 instances only", "projected_min": proj_min / 2}
    desc = {}
    for t in TRUTHS:
        for s in SAMPLERS:
            rr = [r for r in R if r["truth"] == t and r["sampler"] == s]
            if not rr:
                continue
            c = [r for r in rr if r["status"] == "CERTIFIED"]
            modes = defaultdict(int)
            for r in rr:
                for kk, v in (r.get("chooser_modes") or {}).items():
                    modes[kk] += v
            desc[f"{t}|{s}"] = {
                "n_runs": len(rr), "n_certified": len(c), "n_zero_cost": sum(r["zero_cost"] for r in c),
                "n_false_cert": sum(r["false_cert"] for r in c),
                "fcr": (sum(r["false_cert"] for r in c) / len(c)) if c else None,
                "status_counts": {st: sum(r["status"] == st for r in rr) for st in sorted({r["status"] for r in rr})},
                "stream_steps_total": int(sum(r["new_env_steps"] for r in rr)),
                "median_new_steps_paid": float(np.median([r["new_env_steps"] for r in rr if r["new_env_steps"] > 0]
                                                         or [0])),
                "chooser_modes": dict(modes),
                "median_wall_s_per_problem": float(np.median([r["wall_clock_s"] + r["predictor_s"] for r in rr]))}
    calib = {m["tag"]: m["calibration"] for m in metas if m["calibration"]}
    gates = {"zero_crashes": not errs and not fatal and not any(r["status"] == "CRASH" for r in R),
             "billing_0_mismatch": len(billing_bad) == 0,
             "predictors_before_results_every_cert": not cert_missing,
             "projected_full_le_55min": proj_min <= FULL_BUDGET_MIN,
             "n_runs_ge_100": len(R) >= (100 if mode == "pilot" else 0),
             "local_dda_loop_reproduces_package": bool(lc) and all(x["n_mismatch"] == 0 for x in lc)}
    if mode == "full":
        gates.pop("local_dda_loop_reproduces_package")
    return {"gates": gates, "n_runs": len(R), "n_errors": len(errs), "errors_head": errs[:5], "n_fatal": len(fatal),
            "fatal_head": fatal[:3], "n_billing_mismatch": len(billing_bad), "n_certified": len(cert_rows),
            "n_cert_without_predictor": len(cert_missing), "local_dda_check": lc,
            "timing": {"mean_job_sec_by_truth": t_sec, "n_jobs_full_per_truth": n_full, "cpu_s_full": cpu_s,
                       "safety": safety, "workers": workers, "gpu_s_full_tables_est": gpu_s,
                       "projected_full_min": proj_min, "downscale_option": opt, "table_stage": tinfo},
            "calibration_m1": calib, "descriptive_smoke_not_evidence": desc,
            "x1_readout": x1_readout(R), "x2_readout": x2_readout(R)}


def write_samples(J, R, metas, out_dir):
    sd = out_dir / "samples"
    sd.mkdir(exist_ok=True)
    ex = {}
    for m in metas:
        ex.setdefault(m["truth"], {"tag": m["tag"], "calibration": m["calibration"], "problems": m["samples"]})
    (sd / "truth_examples.json").write_text(json.dumps(ex, indent=1, default=str))
    pick = []
    for t in TRUTHS:
        for s in SAMPLERS:
            fj = [j for j in J if j["res"]["truth"] == t and j["res"]["sampler"] == s]
            fj = sorted(fj, key=lambda j: (not j["res"]["false_cert"], j["res"]["zero_cost"],
                                           -j["res"]["new_env_steps"]))[:3]
            pick += [{"key": j["key"], "pred": {k: j["pred"].get(k) for k in ("S1", "S2", "rho_star_min", "A_k_max",
                                                                              "new_steps", "set_size", "logged_at")},
                      "res": {k: j["res"].get(k) for k in ("status", "zero_cost", "false_cert", "true_regret",
                                                           "true_gap", "gap_layer", "new_env_steps", "chooser_modes",
                                                           "x2_oracle_r_norm_xi", "x2_rho_star_min", "scored_at")}}
                     for j in fj]
    (sd / "certification_examples.json").write_text(json.dumps(pick, indent=1, default=str))
    x2 = [{k: r.get(k) for k in ("instance", "truth", "sampler", "problem", "status", "false_cert", "gap_layer",
                                 "x2_rho_star_min", "x2_oracle_r_norm_xi", "x2_lambda_perp", "x2_r_bar")}
          for r in R if "x2_oracle_r_norm_xi" in r]
    (sd / "x2_points.json").write_text(json.dumps(x2, default=str))
    return x2


def plot_x2(x2, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return None
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for a, t in zip(ax, TRUTHS):
        pts = [p for p in x2 if p["truth"] == t and p["status"] == "CERTIFIED"
               and isinstance(p["x2_rho_star_min"], (int, float)) and math.isfinite(p["x2_rho_star_min"])
               and p["x2_rho_star_min"] > 0]
        for fc, col, lab in ((False, "#4a7fb5", "correct"), (True, "#c8473a", "false cert")):
            q = [p for p in pts if bool(p["false_cert"]) == fc]
            a.scatter([max(p["x2_oracle_r_norm_xi"], 1e-6) for p in q], [p["x2_rho_star_min"] for p in q], s=12,
                      c=col, alpha=.7, label=f"{lab} (n={len(q)})")
        a.plot([1e-4, 1], [1e-4, 1], ls="--", c="gray", lw=.8)
        a.set_xscale("log")
        a.set_yscale("log")
        a.set_title(t)
        a.set_xlabel(r"oracle $\|r\|_{\hat\xi}$")
        a.legend(fontsize=7)
    ax[0].set_ylabel(r"$\rho^*$ (learner, write-ahead)")
    fig.tight_layout()
    p = out_dir / "x2_rho_star_vs_oracle_r.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return str(p)


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
    ap.add_argument("--seeds", type=str, default=None, help="debug override, comma separated")
    ap.add_argument("--truths", type=str, default=None, help="debug override, comma separated")
    a = ap.parse_args()
    cfg = MODES[a.mode]
    out_dir = RES_ROOT / ("pilots" if a.mode == "pilot" else "full") / TASK
    (out_dir / "parts").mkdir(parents=True, exist_ok=True)
    (RES_ROOT / f"{TASK}.pid").write_text(str(os.getpid()))
    started = datetime.now().isoformat()
    t0 = time.perf_counter()
    workers = min(a.workers, 4)
    try:
        lock_info = None
        if a.mode == "full":
            from dsswm.stats.prereg import assert_locked
            lock = assert_locked()
            lock_info = {"version": lock["version"], "sha256": lock["sha256"]}
        seeds_req = [int(x) for x in a.seeds.split(",")] if a.seeds else cfg["seeds"]
        truths = tuple(a.truths.split(",")) if a.truths else TRUTHS
        progress(0, 1, "tables")
        tinfo = prepare_tables(seeds_req)
        log(f"tables {tinfo}")
        seeds = [s for s in seeds_req if s not in tinfo["quota_fail_seeds"]]
        jobs = [(s, t, stv, out_dir, a.mode == "pilot" and s == seeds[0] and t == "R0")
                for s in seeds for t in truths for stv in cfg["streams"]]
        joblog = run_pool(jobs, workers)
        J, R, metas, wa = merge(out_dir)
        an = analyse(J, R, metas, joblog, tinfo, a.mode, workers)
        an["gates"]["write_ahead_order_ok"] = wa["write_ahead_ok"]
        x2 = write_samples(J, R, metas, out_dir)
        fig = plot_x2(x2, out_dir)
        if a.mode == "pilot":
            passed = all(an["gates"].values())
        else:
            passed = an["gates"]["zero_crashes"] and an["gates"]["billing_0_mismatch"] and wa["write_ahead_ok"]
        summary = {
            "task": TASK, "mode": a.mode, "started_at": started, "finished_at": datetime.now().isoformat(),
            "note": ("pilot = smoke + timing only on dev seeds (no scientific readout); 并发运行，计时偏高"
                     if a.mode == "pilot" else "exploratory (X1/X2 not pre-registered thresholds); 并发运行"),
            "design": {"seeds": seeds, "quota_fail_dropped": tinfo["quota_fail_seeds"], "streams": cfg["streams"],
                       "truths": list(truths), "samplers": list(SAMPLERS), "method": "JPC", "arm": "full",
                       "n_problems": N_PROBLEMS, "eps": EPS, "delta": DELTA, "tmax": TMAX, "explore": EXPLORE,
                       "tost_margin": list(TOST),
                       "neyman_rule": "n_x propto |(h_hat/xi_tilde)^perp(x)| sqrt(mu_hat(1-mu_hat)); primary pair = "
                                      "max Lambda_perp; xi_tilde=(n+1/2)/(N+C/2); 1-/2-step tracking argmax"},
            "lock": lock_info, "write_ahead": wa, **an, "passed": passed, "go_no_go": "GO" if passed else "NO_GO",
            "x2_figure": fig, "code_sha256": code_sha(), "wall_clock_s": time.perf_counter() - t0, "jobs": joblog}
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
        wall_min = (time.perf_counter() - t0) / 60
        if not a.no_gpu_progress:
            update_gpu_progress("success" if passed else "failed", started, wall_min,
                                {"mode": a.mode, "n_jobs": len(jobs), "n_runs": an["n_runs"], "truths": list(truths),
                                 "samplers": list(SAMPLERS), "workers": workers, "gpu_model": "RTX 4090",
                                 "gpu_count": 1, "projected_full_min": an["timing"]["projected_full_min"],
                                 "note": "CPU-dominated; 并发运行"}, 8 if a.mode == "pilot" else 30)
        progress(1, 1, "done", {"go_no_go": summary["go_no_go"]})
        mark_done("success" if passed else "failed",
                  f"{summary['go_no_go']} gates={an['gates']}; runs={an['n_runs']}; "
                  f"projected_full={an['timing']['projected_full_min']:.1f} min")
        log(f"done {summary['go_no_go']} in {wall_min:.1f} min; gates={an['gates']}")
    except Exception as e:  # noqa: BLE001
        (out_dir / "crash.txt").write_text(traceback.format_exc())
        mark_done("failed", repr(e))
        raise


if __name__ == "__main__":
    main()
