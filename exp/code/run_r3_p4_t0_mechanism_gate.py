"""r3_p4_t0_mechanism_gate: T0 mechanism gate (zero new interaction on the P1 NL-R0 ledgers) + exploratory Q-family
AUCs on dev static-twin g=1 / m1 eta=2eps JPC-full runs + cand_o / cand_l gates + S1/S2 variant preview (pilot) or
freeze (full). Dev seeds only (P1 ledgers 720-727 / 720-739; dev runs 734-739 / 720-739); never touches eval seeds.

Part A  (R0, zero new interaction): every P1 ledger snapshot (instance x stream x method x arm x problem).
  learner side (from the snapshot only): recompute xi_hat, theta_hat (LR-set MLE re-derived by replaying the stored
      observation payloads through a fresh SeqLRSet), the contrast pairs and Lambda_perp; compare with P1's write-ahead
      predictor line (T0(iii) retrievability + reproducibility). For the `full` arm also the Q2 *opportunity*
      predictors of problem k from the snapshot of problem k-1 (= the evidence available at the opening of k).
  harness side: T0(i) identity  dJ* - dJ_theta_hat = <h, r> + Rem  (Gauss-Legendre exact remainder), exact Rem/eps,
      eta = max_pi |J_theta_hat - J*|, oracle Lambda_Delta; T0(ii) uses the logged computable bound Rbar_em.
Part B  (dev runs, risk layers, NEW interaction on dev seeds only): stream 0, 15 gap-quota problems, JPC `full` arm
  static_g1   learner = parameter-count-matched static class (gamma = lam = 0 reparametrisation, |Theta| = 13824),
              truth = NL-R0 theta* (dynamic, on G_1): the matched static twin at dose g = 1
  m1_2eps     learner = dynamic G_1 class, truth = NL-R0 theta* + pair synergy s * S_ij (S double-centred,
              rng [seed, 61]), s calibrated so that eta_min = min_theta max_{q,pi} |J_true - J_theta| = 2 eps
  learner writes (write-ahead, before scoring) at the opening of every problem k >= 1 (Q2 opportunity) and at the
  certification time (Q1): Lambda_perp (S1), S2 = 1/rho*, A_k, placebo A_k, trivial baselines
  (-log gap_hat/eps, log tr I^-1, eta_hat = per-draw KL of the saturated cell model vs theta_hat).
  harness scores: true gap / layer, false certification, eta_arg = true regret of argmax_pi J_theta_hat
  (ex-post label, theta_hat at certification), eta_dec = max_pi |J_true - J_theta_hat|, identity check.
Exploratory AUCs are NOT evidence (dev seeds, pre-registered as exploratory).

Usage: run_r3_p4_t0_mechanism_gate.py --mode {pilot,full} [--workers 4]
Concurrent run (shares the 20 cores and the RTX 4090 with other round-3 tasks): timings are "concurrent".
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import gzip  # noqa: E402
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
CACHE_DYN = WS / "exp" / "cache" / "jtables_r3"
CACHE_STA = WS / "exp" / "cache" / "jtables_r3_static"
TASK = "r3_p4_t0_mechanism_gate"
P1 = "r3_p1_reuse_ablation_nl"
EPS, DELTA = 0.02, 0.05
TMAX = 6000
TOP_M = 5
PLANNED_MIN = 15
MODES = {
    "pilot": {"p1_dir": RES_ROOT / "pilots" / P1, "dev_seeds": list(range(734, 740)), "n_problems": 15},
    "full": {"p1_dir": RES_ROOT / "full" / P1, "dev_seeds": list(range(720, 740)), "n_problems": 15},
}
SETTINGS = ("static_g1", "m1_2eps")
METHOD_OF = {"static_g1": "JPC_static_g1", "m1_2eps": "JPC_m1_2eps"}
CODE_FILES = ["dsswm/mechanism/leverage.py", "dsswm/mechanism/oracle_check.py", "dsswm/mechanism/predictor_log.py",
              "dsswm/stats/auc.py", "run_r3_p4_t0_mechanism_gate.py"]


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


# ============================================================================================ classes
def static_np_params(np_params: dict) -> dict:
    """g(n) = g_ret(n) = 1  <=>  alpha' = alpha - gamma, gamma' = 0, tau' = tau - lam, lam' = 0 (exact; hd1)."""
    p = {k: np.array(v, copy=True) for k, v in np_params.items()}
    lam = np.asarray(p["lam"], float).reshape(-1)
    p["alpha"] = p["alpha"] - p["gamma"]
    p["gamma"] = np.zeros_like(p["gamma"])
    p["tauL"] = p["tauL"] - lam[:, None]
    p["tauR"] = p["tauR"] - lam[:, None]
    p["lam"] = np.zeros_like(lam)
    return p


class StaticNLClass:
    """Static twin of an NLClass (same index set, load-free reparametrisation). Public, truth-free."""

    def __init__(self, ncl, device="cpu"):
        self.grid, self.L, self.R, self.B, self.nb = ncl.grid, ncl.L, ncl.R, ncl.B, ncl.nb
        self.device, self.dtype = device, torch.float64
        self.np_params = static_np_params(ncl.np_params)

    def torch_params(self, sl=None):
        sl = sl or slice(None)
        return {k: torch.as_tensor(v[sl], device=self.device, dtype=self.dtype) for k, v in self.np_params.items()}

    def params_at(self, k):
        return {key: v[k] for key, v in self.np_params.items()}


def synergy_pattern(seed, L=2, R=2):
    s = np.random.default_rng([seed, 61]).standard_normal((L, R))
    s = s - s.mean(1, keepdims=True)
    s = s - s.mean(0, keepdims=True)
    return s / max(np.abs(s).max(), 1e-12)


# ============================================================================================ learner-side helpers
def gof_eta_hat(lev, observations, mu_hat):
    """eta_hat: per-draw KL( saturated per-cell Bernoulli fit || theta_hat ) -- a learner-side goodness-of-fit level."""
    pr = lev.prop
    n1 = np.zeros(lev.C)
    n = np.zeros(lev.C)
    for o in observations:
        idx, _ = pr.obs_factors(o)
        for f in idx:
            f = int(f)
            c = lev.cell_of_factor(f)
            n[c] += 1
            if f < pr.U_pair:
                n1[c] += 1
            elif f >= pr.off_re:
                n1[c] += (f - pr.off_re) % 2
    m = n > 0
    if not m.any():
        return 0.0
    p = n1[m] / n[m]
    q = np.clip(mu_hat[m], 1e-12, 1 - 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = np.where(p > 0, p * np.log(p / q), 0.0)
        t0 = np.where(p < 1, (1 - p) * np.log((1 - p) / (1 - q)), 0.0)
    return float(np.sum(n[m] * (t1 + t0)) / n.sum())


def learner_record(lev, TH, q, J, obs, mask, k_hat, pi_hat, r_bar, with_rem=True):
    """All Q-family predictors + trivial baselines at one (opportunity or certification) time. Truth-free."""
    from dsswm.mechanism.leverage import select_pairs
    pairs = select_pairs(J, mask, k_hat, int(pi_hat))
    pred = lev.predictors(q, obs, TH[k_hat], pairs, r_bar if np.isfinite(r_bar) else 0.0, EPS, with_rem=with_rem)
    Jh = np.sort(J[k_hat])[::-1]
    gap_hat = float(Jh[0] - Jh[1]) if len(Jh) > 1 else 0.0
    mu = lev.mu_from_theta(TH[k_hat])
    pred["gap_hat_over_eps"] = gap_hat / EPS
    pred["neg_log_gap_hat"] = -math.log(max(gap_hat, 1e-12) / EPS)
    pred["log_tr_Iinv"] = math.log(max(pred["trace_I_pinv"], 1e-300))
    pred["eta_hat_gof"] = gof_eta_hat(lev, obs, mu)
    pred["phi_perp_max"] = max(p["phi_perp"] for p in pred["pairs"])
    pred["A_k_placebo_block_max"] = max(p["A_k_placebo_block"] for p in pred["pairs"])
    pred["A_k_placebo_global_max"] = max(p["A_k_placebo_global"] for p in pred["pairs"])
    if with_rem and "rem_bar" in pred["pairs"][0]:
        pred["rem_bar_over_eps_max"] = max(p["rem_bar"] for p in pred["pairs"]) / EPS
    pred["log_S1"] = math.log(max(pred["S1"], 1e-300))
    return pred


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
        scl = StaticNLClass(ncl)
        THs = NLLeverage(prop, static=True).theta_matrix(scl.np_params)
        levs = NLLeverage(prop, static=True, theta_grid=THs, eps=EPS)
        _W.update(ncl=ncl, prop=prop, TH=TH, lev=lev, scl=scl, THs=THs, levs=levs,
                  LT=prop.tables(ncl.torch_params())[0], builders={})
    return _W


def _builder(seed):
    from dsswm.streams.gap_quota import QuotaBuilder
    W = _ctx()
    if seed not in W["builders"]:
        W["builders"] = {seed: QuotaBuilder(seed, W["ncl"], None, CACHE_DYN, prop_cpu=W["prop"])}
    return W["builders"][seed]


def _obs_from_payload(d):
    from dsswm.core.provenance import Observation
    return Observation(env_kind=d["env_kind"], t=int(d["t"]), loads=tuple(d["loads"]), engaged=tuple(d["engaged"]),
                       action=int(d["action"]), outcomes=tuple(tuple(x) for x in d["outcomes"]),
                       next_loads=tuple(d["next_loads"]), next_engaged=tuple(d["next_engaged"]),
                       serial=int(d["serial"]))


# ============================================================================================ Part A (R0 ledgers)
def _replay_lr(prop, LT, obs):
    """Harness-side re-derivation of the snapshot's LR set from the STORED payloads of observations that were
    authenticated when P1 collected them (env.step()). The reconstructed objects are not re-registered, so the
    provenance gate of SeqLRSet.update is bypassed for this offline recomputation only (never in a learner loop)."""
    import dsswm.evidence.lr_set as L
    from dsswm.evidence.lr_set import SeqLRSet
    orig = L.require_authentic
    L.require_authentic = lambda o: None
    try:
        lr = SeqLRSet(prop, LT, DELTA)
        for o in obs:
            lr.update(o)
    finally:
        L.require_authentic = orig
    return lr


def job_r0(seed, stream, method, n_problems, p1_dir, out_dir):
    """All P1 ledger snapshots of one (instance, stream, method): T0 (i)-(iv) + Q2 opportunity predictors."""
    from dsswm.certify.minimax_enum import certify_minimax, regret_matrix
    from dsswm.mechanism import oracle_check as oc
    from dsswm.mechanism.leverage import theta_hat_index
    from dsswm.mechanism.predictor_log import PredictorLog, ResultLog, load_jsonl
    W = _ctx()
    lev, TH, ncl, prop = W["lev"], W["TH"], W["ncl"], W["prop"]
    t_job = time.perf_counter()
    tag = f"A_i{seed}_s{stream}_{method}"
    parts = out_dir / "parts"
    pp, rp = parts / f"{tag}_pred.jsonl", parts / f"{tag}_res.jsonl"
    for p in (pp, rp):
        p.unlink(missing_ok=True)
    plog, rlog = PredictorLog(pp, fsync=False), ResultLog(rp, fsync=False)
    b = _builder(seed)
    st = b.stream(stream, "quota", n_problems=n_problems)
    truth = st.env.true_params()                         # harness
    ti = st.theta_index
    ledir = p1_dir / "samples" / "ledgers"
    payload = {}
    with gzip.open(ledir / f"i{seed}_s{stream}_{method}_obs.jsonl.gz", "rt") as fh:
        for line in fh:
            d = json.loads(line)
            payload[int(d["serial"])] = d
    p1pred = {(r["arm"], int(r["problem"])): r for r in load_jsonl(p1_dir / "predictors.jsonl")
              if r["instance"] == seed and r["stream"] == stream and r["method"] == method}
    p1res = {(r["arm"], int(r["problem"])): r for r in load_jsonl(p1_dir / "results.jsonl")
             if r["instance"] == seed and r["stream"] == stream and r["method"] == method}
    rows, errs = [], []
    snaps = {}
    for arm in ("off", "vol", "ev1", "full"):
        for k in range(n_problems):
            key = {"instance": seed, "stream": stream, "method": method, "arm": arm, "problem": k}
            try:
                fn = ledir / f"i{seed}_s{stream}_{method}_{arm}_p{k:02d}.npz"
                z = dict(np.load(fn))
                q = st.problems[k]
                J = st.J[k]
                ser = z["serials"]
                missing = [int(s) for s in ser if int(s) not in payload]
                obs = [_obs_from_payload(payload[int(s)]) for s in ser if int(s) in payload]
                # ---- (iii) retrievability: per-cell counts, per-x counts, theta_hat, mask from the snapshot alone
                cnt_store = z["cell_counts_real"] + z["cell_counts_replay"]
                cnt_re = lev.cell_counts(obs)
                lr = _replay_lr(prop, W["LT"], obs)
                mask_re = lr.mask().numpy().astype(bool)
                k_hat_re = theta_hat_index(lr.cum.cpu().numpy(), mask_re)
                mask_store = np.unpackbits(z["mask_packed"])[: int(z["mask_len"])].astype(bool)
                k_hat = int(z["theta_hat"])
                pred_p1 = p1pred.get((arm, k))
                x_total = int(z["x_n_real"].sum() + z["x_n_replay"].sum())
                ret = {"npz_ok": True, "payload_missing": len(missing),
                       "cell_counts_match": bool(np.array_equal(cnt_store, cnt_re)),
                       "x_counts_total_eq_n_obs": x_total == len(ser),
                       "theta_hat_recomputed_eq_stored": k_hat_re == k_hat,
                       "mask_recomputed_eq_stored": bool(np.array_equal(mask_re, mask_store)),
                       "theta_hat_eq_p1_predictor": pred_p1 is not None and int(pred_p1["theta_hat"]) == k_hat}
                # ---- learner-side recomputation from the snapshot (xi_hat, theta_hat, pairs, Lambda_perp)
                pi_hat = int(z["pi_hat"])
                r_bar = float(pred_p1["r_bar_method"]) if pred_p1 is not None else float("nan")
                t0 = time.perf_counter()
                pred = learner_record(lev, TH, q, J, obs, mask_store, k_hat, pi_hat, r_bar, with_rem=False)
                sec = time.perf_counter() - t0
                lam_p1 = pred_p1["lambda_perp_max"] if pred_p1 is not None else None
                ret["lambda_reproduced_relerr"] = (abs(pred["S1"] - lam_p1) / max(abs(lam_p1), 1e-300)
                                                   if lam_p1 is not None else None)
                ret["pairs_match_p1"] = (pred_p1 is not None and [(p["k1"], p["k2"]) for p in pred["pairs"]]
                                         == [(p["k1"], p["k2"]) for p in pred_p1["pairs"]])
                plog.write({**key, "source": "R0_ledger_recomputed", "pid": q.pid, "theta_hat": k_hat,
                            "pi_hat": pi_hat, "status": pred_p1["status"] if pred_p1 else None,
                            "new_steps": int(z["new_steps"]), "recompute_sec": sec,
                            "S1_p1_logged": lam_p1, "S2_p1_logged": pred_p1["S2"] if pred_p1 else None,
                            "rem_bar_over_eps_p1": (max(p["rem_bar"] for p in pred_p1["pairs"]) / EPS
                                                    if pred_p1 else None),
                            "sec_per_call_p1": pred_p1["sec_per_call"] if pred_p1 else None,
                            **{kk: pred[kk] for kk in ("S1", "S2", "A_k_max", "phi_perp_max", "trace_I_pinv",
                                                       "log_tr_Iinv", "neg_log_gap_hat", "gap_hat_over_eps",
                                                       "eta_hat_gof", "n_obs", "n_factor_draws", "rho_star_min")},
                            "pairs": [{kk: v for kk, v in p.items() if kk != "need_data"} | {"n_need_data":
                                      len(p["need_data"])} for p in pred["pairs"]]})
                snaps[(arm, k)] = {"obs": obs, "mask": mask_store, "k_hat": k_hat, "cnt": cnt_store}
                # ---- harness: T0(i) identity, exact Rem, eta, Lambda_Delta (truth)
                xi = cnt_store / cnt_store.sum()
                ids = []
                for p in pred["pairs"]:
                    chk = oc.nl_identity_check(lev, q, p["k1"], p["k2"], ncl.params_at(k_hat), truth, xi=xi, eps=EPS)
                    ids.append({"label": p["label"], "identity_err": chk["identity_err"],
                                "rem_exact_over_eps": chk["rem_exact"] / EPS, "first_order": chk["first_order"],
                                "lambda_delta": chk["lambda_delta"], "r_norm_xi": chk["r_norm_xi"],
                                "plugin_consistency_err": chk["plugin_consistency_err"]})
                Jt = J[ti]
                res1 = p1res.get((arm, k), {})
                eta = float(np.max(np.abs(J[k_hat] - Jt)))
                eta_arg = float(Jt.max() - Jt[int(np.argmax(J[k_hat]))])
                rr = {**key, "source": "R0_ledger", "pairs": ids, "identity_err_max": max(x["identity_err"] for x in ids),
                      "rem_exact_over_eps_max": max(abs(x["rem_exact_over_eps"]) for x in ids),
                      "eta": eta, "eta_arg": eta_arg, "theta_hat_is_truth": k_hat == ti,
                      "gap_layer": res1.get("gap_layer"), "true_gap": res1.get("true_gap"),
                      "status": res1.get("status"), "zero_cost": res1.get("zero_cost"),
                      "false_cert": res1.get("false_cert"), "new_env_steps": res1.get("new_env_steps"),
                      "retrieval": ret}
                rlog.write(rr)
                rows.append(rr)
            except Exception as e:  # noqa: BLE001
                errs.append({**key, "error": repr(e), "tb": traceback.format_exc()[-2000:]})
    # ---- Q2 opportunity predictors for the full arm: problem k opens with the evidence of snapshot k-1
    for k in range(1, n_problems):
        key = {"instance": seed, "stream": stream, "method": method, "arm": "full@open", "problem": k}
        try:
            s = snaps[("full", k - 1)]
            q, J = st.problems[k], st.J[k]
            cert = certify_minimax(regret_matrix(J), s["mask"], EPS, TOP_M)
            pi0 = int(cert["pi"]) if cert["pi"] is not None else int(np.argmax(J[s["k_hat"]]))
            pred = learner_record(lev, TH, q, J, s["obs"], s["mask"], s["k_hat"], pi0, float(cert["r_bar"]),
                                  with_rem=False)
            plog.write({**key, "source": "R0_ledger_open", "theta_hat": s["k_hat"], "pi_hat": pi0,
                        "cert_status_at_open": cert["status"].value,
                        **{kk: pred[kk] for kk in ("S1", "S2", "A_k_max", "A_k_placebo_block_max",
                                                   "A_k_placebo_global_max", "phi_perp_max", "log_tr_Iinv",
                                                   "neg_log_gap_hat", "eta_hat_gof", "rho_star_min")}})
            res1 = p1res.get(("full", k), {})
            rlog.write({**key, "source": "R0_ledger_open", "zero_cost": res1.get("zero_cost"),
                        "status": res1.get("status"), "gap_layer": res1.get("gap_layer"),
                        "false_cert": res1.get("false_cert")})
        except Exception as e:  # noqa: BLE001
            errs.append({**key, "error": repr(e), "tb": traceback.format_exc()[-2000:]})
    out = {"tag": tag, "n_rows": len(rows), "errors": errs, "sec": time.perf_counter() - t_job}
    (parts / f"{tag}.json").write_text(json.dumps(out, default=str))
    return {"tag": tag, "n_rows": len(rows), "n_err": len(errs), "sec": out["sec"]}


# ============================================================================================ Part B (dev runs)
def calibrate_m1(prop, st, S, target):
    """Bisection on the synergy strength s so that eta_min = min_theta max_{q,pi} |J_true(s) - J_theta| = target."""
    from dsswm.exact.nl_propagate import params_to_torch
    base = st.env.true_params()
    plans = [[prop.build_policy_plan(p, q.loads0, q.engaged0, q.H) for p in q.policies] for q in st.problems]
    Jall = np.concatenate(st.J, 1)

    def jtrue(s):
        tp = dict(base)
        tp["syn"] = s * S
        LT, EY = prop.tables(params_to_torch(tp, device="cpu"))
        out = []
        for q, pl in zip(st.problems, plans):
            u = q.utility
            out.append(np.array([float(prop._run_plan(a, e, LT, EY, u.w, u.w_ret, u.c_q)[0]) for a, e in pl]))
        return out

    def eta_min(s):
        jt = np.concatenate(jtrue(s))
        dev = np.abs(Jall - jt[None]).max(1)
        k = int(np.argmin(dev))
        return float(dev[k]), k

    lo, hi, it = 0.0, 0.05, 0
    while eta_min(hi)[0] < target and hi < 20:
        lo, hi = hi, hi * 2
        it += 1
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
    return s, v, k, it, jtrue(s)


def job_dev(seed, setting, n_problems, out_dir):
    """One dev stream (stream 0, 15 problems) of JPC `full` on the static twin g=1 or m1 eta=2eps."""
    from dsswm.baselines.switched_nl import PublicNL, solve
    from dsswm.certify.minimax_enum import certify_minimax
    from dsswm.envs.nl import NLEnv
    from dsswm.evidence.reuse_switch import ReuseSwitch
    from dsswm.mechanism import oracle_check as oc
    from dsswm.mechanism.leverage import theta_hat_index
    from dsswm.mechanism.predictor_log import PredictorLog, ResultLog
    from dsswm.streams.gap_quota import STREAM_NOISE, gap_layer, top2_gap
    from dsswm.streams.generator import NL_DEFAULTS, _initial_data
    W = _ctx()
    prop = W["prop"]
    t_job = time.perf_counter()
    method = METHOD_OF[setting]
    tag = f"B_i{seed}_{setting}"
    parts = out_dir / "parts"
    pp, rp = parts / f"{tag}_pred.jsonl", parts / f"{tag}_res.jsonl"
    for p in (pp, rp):
        p.unlink(missing_ok=True)
    plog, rlog = PredictorLog(pp, fsync=False), ResultLog(rp, fsync=False)
    b = _builder(seed)
    stream = 0
    st = b.stream(stream, "quota", n_problems=n_problems)
    ti = st.theta_index
    calib = None
    # ------------------------------------------------------------------ harness: platform + truth tables
    if setting == "static_g1":
        env = st.env
        init = st.init_obs
        J_true = [J[ti] for J in st.J]
        truth = env.true_params()
        ncl_l, lev, TH = W["scl"], W["levs"], W["THs"]
        J_learn = []
        for q in st.problems:
            raw = np.load(CACHE_STA / f"{q.pid}_raw.npy")
            J_learn.append(raw * q.utility.c_q)              # same public c_q as the dynamic class (class-max)
    else:
        S = synergy_pattern(seed)
        s, eta_min, k_min, it, J_true = calibrate_m1(prop, st, S, 2 * EPS)
        calib = {"strength": s, "eta_min": eta_min, "eta_min_over_eps": eta_min / EPS, "theta_min": k_min,
                 "theta_min_is_theta_star": bool(k_min == ti), "iters": it}
        t = st.truth
        env = NLEnv(np.array(t["alpha"]), np.array(t["beta"]), np.array(t["gamma"]), np.array(t["tauL"]),
                    np.array(t["tauR"]), [t["psi"]], t["lam"], c=NL_DEFAULTS["c"], rho_ret=NL_DEFAULTS["rho_ret"],
                    L=2, R=2, nmax=NL_DEFAULTS["nmax"], budget=NL_DEFAULTS["budget"], incentive_levels=(1,),
                    seed=int(seed) * 1000 + int(STREAM_NOISE[stream]), syn=s * S)
        init = _initial_data(env, NL_DEFAULTS["n0"], np.random.default_rng([seed, 12]))
        truth = env.true_params()
        ncl_l, lev, TH = W["ncl"], W["lev"], W["TH"]
        J_learn = list(st.J)
    # ------------------------------------------------------------------ learner (public objects only)
    pub = PublicNL(prop, ncl_l, st.problems, J_learn, 1.0, 2, EPS, DELTA, top_m=TOP_M)
    h = env.handle()
    sw = ReuseSwitch("full", pub.make_set_factory("JPC"), init)
    rng = np.random.default_rng([int(seed), int(st.noise_seed), 101])
    log_rows, errs = [], []
    for k, q in enumerate(st.problems):
        key = {"instance": seed, "stream": stream, "method": method, "problem": k}
        lr = sw.begin_problem(k, q.pid)
        if k >= 1:                                            # Q2 opportunity: evidence at the opening of problem k
            try:
                mask = lr.mask().numpy().astype(bool)
                k_hat = theta_hat_index(lr.inner.cum.cpu().numpy(), mask)
                c0 = certify_minimax(pub.Reg[k], mask, EPS, TOP_M)
                pi0 = int(c0["pi"]) if c0["pi"] is not None else int(np.argmax(J_learn[k][k_hat]))
                pr = learner_record(lev, TH, q, J_learn[k], list(lr.obs), mask, k_hat, pi0, float(c0["r_bar"]),
                                    with_rem=False)
                plog.write({**key, "arm": "full@open", "source": f"dev_{setting}_open", "theta_hat": k_hat,
                            "pi_hat": pi0, "cert_status_at_open": c0["status"].value,
                            **{kk: pr[kk] for kk in ("S1", "S2", "A_k_max", "A_k_placebo_block_max",
                                                     "A_k_placebo_global_max", "phi_perp_max", "log_tr_Iinv",
                                                     "neg_log_gap_hat", "eta_hat_gof", "rho_star_min",
                                                     "sec_per_call")}})
            except Exception as e:  # noqa: BLE001
                errs.append({**key, "stage": "open", "error": repr(e), "tb": traceback.format_exc()[-2000:]})
        t0 = time.perf_counter()
        res = solve(pub, "JPC", k, sw, lr, h, rng, TMAX)
        wall = time.perf_counter() - t0
        rec = {"k_hat": None, "pairs": None}
        try:                                                  # certification-time predictors (write-ahead)
            mask = lr.mask().numpy().astype(bool)
            k_hat = theta_hat_index(lr.inner.cum.cpu().numpy(), mask)
            pi_hat = res["pi"]
            if pi_hat is None and mask.any():
                pi_hat = certify_minimax(pub.Reg[k], mask, EPS, TOP_M)["pi"]
            if pi_hat is None:
                pi_hat = int(np.argmax(J_learn[k][k_hat]))
            r_bar = float(res["extra"].get("r_bar_end", float("nan")))
            pr = learner_record(lev, TH, q, J_learn[k], list(lr.obs), mask, k_hat, int(pi_hat), r_bar, with_rem=True)
            plog.write({**key, "arm": "full", "source": f"dev_{setting}", "status": res["status"],
                        "new_steps": int(res["steps"]), "theta_hat": k_hat, "pi_hat": int(pi_hat),
                        "set_size": int(mask.sum()), **{kk: v for kk, v in pr.items() if kk != "pairs"},
                        "pairs": [{kk: v for kk, v in p.items() if kk != "need_data"} | {"n_need_data":
                                  len(p["need_data"])} for p in pr["pairs"]]})
            rec = {"k_hat": k_hat, "pairs": [(p["k1"], p["k2"], p["label"]) for p in pr["pairs"]],
                   "xi": lev.cell_counts(list(lr.obs))}
        except Exception as e:  # noqa: BLE001
            errs.append({**key, "stage": "cert", "error": repr(e), "tb": traceback.format_exc()[-2000:]})
        acc = sw.end_problem(env.n_steps)
        log_rows.append({"k": k, "res": res, "rec": rec, "wall": wall, "billed": int(acc.n_rounds_billed)})
    # ------------------------------------------------------------------ harness scoring (after the stream)
    n_rows = 0
    for lrw in log_rows:
        k, res, rec = lrw["k"], lrw["res"], lrw["rec"]
        key = {"instance": seed, "stream": stream, "method": method, "arm": "full", "problem": k}
        Jt = np.asarray(J_true[k], float)
        gap = top2_gap(Jt)
        pi = res["pi"]
        cert = res["status"] == "CERTIFIED"
        regret = float(Jt.max() - Jt[pi]) if pi is not None else None
        row = {**key, "source": f"dev_{setting}", "setting": setting, "pid": st.problems[k].pid,
               "status": res["status"], "new_env_steps": int(res["steps"]), "certified_policy": pi,
               "zero_cost": bool(cert and res["steps"] == 0), "true_gap": gap, "gap_layer": gap_layer(gap, EPS),
               "gap_layer_r0": st.harness[k]["gap_layer"], "true_regret": regret,
               "false_cert": bool(cert and regret is not None and regret > EPS), "wall_clock_s": lrw["wall"],
               "set_size_end": res["extra"].get("set_size"), "calibration": calib}
        if rec["k_hat"] is not None:
            Jh = J_learn[k][rec["k_hat"]]
            row["eta_arg"] = float(Jt.max() - Jt[int(np.argmax(Jh))])
            row["eta_dec"] = float(np.max(np.abs(Jh - Jt)))
            row["eta_arg_certified"] = regret
            try:
                xi = rec["xi"] / rec["xi"].sum()
                ids = []
                for k1, k2, lab in rec["pairs"]:
                    chk = oc.nl_identity_check(lev, st.problems[k], k1, k2, ncl_l.params_at(rec["k_hat"]), truth,
                                               xi=xi, eps=EPS)
                    ids.append({"label": lab, "identity_err": chk["identity_err"],
                                "rem_exact_over_eps": chk["rem_exact"] / EPS, "lambda_delta": chk["lambda_delta"],
                                "r_norm_xi": chk["r_norm_xi"]})
                row["pairs"] = ids
                row["identity_err_max"] = max(x["identity_err"] for x in ids)
            except Exception as e:  # noqa: BLE001
                errs.append({**key, "stage": "identity", "error": repr(e), "tb": traceback.format_exc()[-2000:]})
        rlog.write(row)
        if k >= 1:
            rlog.write({**key, "arm": "full@open", "source": f"dev_{setting}_open", "zero_cost": row["zero_cost"],
                        "status": row["status"], "gap_layer": row["gap_layer"], "false_cert": row["false_cert"]})
        n_rows += 1
    out = {"tag": tag, "n_rows": n_rows, "errors": errs, "calibration": calib, "sec": time.perf_counter() - t_job,
           "env_steps": int(env.n_steps)}
    (parts / f"{tag}.json").write_text(json.dumps(out, default=str))
    return {"tag": tag, "n_rows": n_rows, "n_err": len(errs), "sec": out["sec"]}


# ============================================================================================ main-process stages
def static_tables(seeds, n_problems):
    """Static-class J tables of the dev quota problems on CUDA (cached)."""
    from dsswm.exact.nl_propagate import NLPropagator
    from dsswm.models.nl_class import NLClass
    from dsswm.streams.gap_quota import QuotaBuilder
    from dsswm.streams.generator import DEFAULT_NL_S_GRID
    from dsswm.streams.utilities import Utility
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    CACHE_STA.mkdir(parents=True, exist_ok=True)
    ncl = NLClass(DEFAULT_NL_S_GRID, device=dev)
    b0 = QuotaBuilder(seeds[0], ncl, None, CACHE_DYN)
    propg = NLPropagator(2, 2, 2, b0.aspace, 1.0, 0.3, device=dev)
    propc = NLPropagator(2, 2, 2, b0.aspace, 1.0, 0.3, device="cpu")
    Ps = {k: torch.as_tensor(v, device=dev, dtype=torch.float64) for k, v in static_np_params(ncl.np_params).items()}
    t0 = time.perf_counter()
    n_new, fails = 0, []
    for s in seeds:
        b = QuotaBuilder(s, ncl, propg, CACHE_DYN, prop_cpu=propc)
        sel = b.select("quota")
        if sel["quota_fail"]:
            fails.append(s)
            continue
        for q in b.stream(0, "quota", n_problems=n_problems).problems:
            path = CACHE_STA / f"{q.pid}_raw.npy"
            if path.exists():
                continue
            u1 = Utility(w=q.utility.w, w_ret=q.utility.w_ret, c_q=1.0)
            raw = propg.j_table(Ps, q.policies, q.loads0, q.engaged0, q.H, u1)
            tmp = path.with_name(path.stem + f".tmp{os.getpid()}.npy")
            np.save(tmp, raw)
            os.replace(tmp, path)
            n_new += 1
    vram = torch.cuda.max_memory_allocated() / 2 ** 20 if dev == "cuda" else 0.0
    prof = {"gpu_name": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
            "vram_total_mb": torch.cuda.get_device_properties(0).total_memory / 2 ** 20 if dev == "cuda" else 0,
            "max_batch_size": "n/a (class J tables: one problem x |Theta|=13824 per call; no batch dimension)",
            "vram_used_mb": vram, "utilization_pct": None,
            "note": "GPU only for static-class J tables of new dev problems (cached); ledger recomputation, identity "
                    "checks, JPC runs and predictors on CPU (4 workers, OMP=1); 并发运行"}
    (RES_ROOT / f"{TASK}_gpu_profile.json").write_text(json.dumps(prof))
    del ncl, propg, Ps
    if dev == "cuda":
        torch.cuda.empty_cache()
    return {"sec": time.perf_counter() - t0, "n_new": n_new, "vram_peak_mb": vram, "device": dev,
            "quota_fail_seeds": fails}


def run_pool(jobs, workers):
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    out = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(fn, *args): (fn.__name__, args[:3]) for fn, args in jobs}
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
    P, R, errs, calib = [], [], [], {}
    for f in sorted(parts.glob("*.json")):
        d = json.loads(f.read_text())
        errs += d.get("errors", [])
        if d.get("calibration"):
            calib[d["tag"]] = d["calibration"]
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
    return J, errs, calib, wa


# ============================================================================================ analysis
def _q(x, p):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    return float(np.quantile(x, p)) if len(x) else None


def _num(v):
    if isinstance(v, str):
        return math.inf if v == "inf" else (-math.inf if v == "-inf" else math.nan)
    return math.nan if v is None else float(v)


def auc_block(rows, score_keys, label_fn, cluster_fn, B=2000):
    from dsswm.stats.auc import auc_ci, delta_auc_ci, pr_auc
    y = np.array([bool(label_fn(r)) for r in rows])
    cl = np.array([cluster_fn(r) for r in rows])
    out = {"n": int(len(rows)), "n_pos": int(y.sum()), "n_clusters": int(len(set(cl.tolist())))}
    if len(rows) == 0 or y.sum() == 0 or (~y).sum() == 0:
        out["note"] = "AUC not evaluable (one outcome class empty)"
        return out
    sc = {k: np.array([_num(fn(r)) for r in rows]) for k, fn in score_keys.items()}
    for k, s in sc.items():
        s = np.where(np.isnan(s), -np.inf, s)
        sc[k] = s
        ci = auc_ci(s, y, cl, B=B, seed=42)
        out[k] = {"auc": ci["est"], "ci95": [ci["lo"], ci["hi"]], "pr_auc": pr_auc(s, y)}
    if "eta_arg_oracle" in sc:
        for k in ("S1", "S2"):
            if k in sc:
                d = delta_auc_ci(sc[k], sc["eta_arg_oracle"], y, cl, B=B, seed=43)
                out[f"delta_auc_{k}_minus_eta_arg"] = {"est": d["est"], "ci95": [d["lo"], d["hi"]]}
    return out


def incremental_logistic(rows, label_fn, base_key, add_key):
    """In-sample AUC of logistic(base) vs logistic(base + add) (descriptive; optimistic by construction)."""
    from dsswm.stats.auc import auc, logistic_fit
    y = np.array([bool(label_fn(r)) for r in rows], float)
    if y.sum() == 0 or y.sum() == len(y):
        return None
    xb = np.array([_num(r["res"].get(base_key)) for r in rows])
    xa = np.array([math.log(max(_num(r["pred"].get(add_key)), 1e-300)) for r in rows])
    ok = np.isfinite(xb) & np.isfinite(xa)
    if ok.sum() < 5:
        return None
    Xb, Xf = xb[ok][:, None], np.stack([xb[ok], xa[ok]], 1)
    try:
        fb = logistic_fit(Xb, y[ok], ridge=1e-6)
        ff = logistic_fit(Xf, y[ok], ridge=1e-6)
        pb = np.c_[np.ones(ok.sum()), Xb] @ np.asarray(fb["beta"])
        pf = np.c_[np.ones(ok.sum()), Xf] @ np.asarray(ff["beta"])
        return {"auc_base": auc(pb, y[ok] > 0), "auc_full": auc(pf, y[ok] > 0),
                "increment": auc(pf, y[ok] > 0) - auc(pb, y[ok] > 0), "n": int(ok.sum())}
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}


def analyse(J, mode):
    A = [j for j in J if j["res"].get("source") == "R0_ledger"]
    Aopen = [j for j in J if j["res"].get("source") == "R0_ledger_open"]
    Bc = [j for j in J if str(j["res"].get("source", "")).startswith("dev_") and j["key"][3] == "full"]
    Bopen = [j for j in J if str(j["res"].get("source", "")).endswith("_open") and j["res"]["source"] != "R0_ledger_open"]
    out = {}
    # ------------------------------------------------------------------ T0 on R0 ledgers
    idm = [j["res"]["identity_err_max"] for j in A]
    ret = [j["res"]["retrieval"] for j in A]
    rem = [j["pred"]["rem_bar_over_eps_p1"] for j in A]
    eta = [j["res"]["eta"] for j in A]
    rem_sub = [r for r, e in zip(rem, eta) if r is not None and e <= 2 * EPS]
    rem_sub1 = [r for r, e in zip(rem, eta) if r is not None and e <= EPS]
    phi = [j["pred"]["phi_perp_max"] for j in A]
    phi_prim = [max(j["pred"]["pairs"], key=lambda p: p["lambda_perp"])["phi_perp"] for j in A]
    rex = [j["res"]["rem_exact_over_eps_max"] for j in A]
    rex_sub = [r for r, e in zip(rex, eta) if e <= 2 * EPS]
    relerr = [r["lambda_reproduced_relerr"] for r in ret if r["lambda_reproduced_relerr"] is not None]
    sec_p1 = [j["pred"]["sec_per_call_p1"] for j in A if j["pred"].get("sec_per_call_p1") is not None]
    sec_re = [j["pred"]["recompute_sec"] for j in A]
    keys_iii = ("npz_ok", "cell_counts_match", "x_counts_total_eq_n_obs", "theta_hat_recomputed_eq_stored",
                "mask_recomputed_eq_stored", "theta_hat_eq_p1_predictor", "pairs_match_p1")
    iii_fail = {k: sum(1 for r in ret if not r.get(k)) for k in keys_iii}
    iii_fail["payload_missing_total"] = sum(r["payload_missing"] for r in ret)
    t0 = {
        "n_ledger_snapshots": len(A),
        "i_identity_err_max": max(idm) if idm else None, "i_identity_err_median": _q(idm, .5),
        "i_pass": bool(idm) and max(idm) <= 1e-9,
        "ii_rem_bar_over_eps": {"n_eta_le_2eps": len(rem_sub), "p90_eta_le_2eps": _q(rem_sub, .9),
                                "median_eta_le_2eps": _q(rem_sub, .5), "n_eta_le_eps": len(rem_sub1),
                                "p90_eta_le_eps": _q(rem_sub1, .9), "p90_all": _q(rem, .9), "median_all": _q(rem, .5),
                                "min_all": _q(rem, 0.0)},
        "ii_pass": (_q(rem_sub, .9) is not None and _q(rem_sub, .9) <= 0.5),
        "ii_exact_rem_over_eps_oracle": {"median_eta_le_2eps": _q(rex_sub, .5), "p90_eta_le_2eps": _q(rex_sub, .9),
                                         "max_eta_le_2eps": _q(rex_sub, 1.0),
                                         "note": "harness-only exact |Rem| at theta*; shows how loose the box bound is"},
        "iii_fail_counts": iii_fail,
        "iii_pass": all(v == 0 for k, v in iii_fail.items() if k != "pairs_match_p1") and len(A) > 0,
        "iii_lambda_reproduced_relerr_max": max(relerr) if relerr else None,
        "iv_phi_perp": {"median_max_over_pairs": _q(phi, .5), "median_primary_pair": _q(phi_prim, .5),
                        "p10": _q(phi, .1), "p90": _q(phi, .9),
                        "median_gt_0.3": (_q(phi, .5) or 0) > 0.3},
        "eta_share_le_2eps": float(np.mean([e <= 2 * EPS for e in eta])) if eta else None,
        "theta_hat_is_truth_share": float(np.mean([j["res"]["theta_hat_is_truth"] for j in A])) if A else None,
        "sec_per_call": {"p1_logged_median": _q(sec_p1, .5), "p1_logged_p90": _q(sec_p1, .9),
                         "recompute_no_rem_median": _q(sec_re, .5), "note": "CPU 1 thread, 并发运行"},
    }
    t0["scope"] = "eta<=2eps" if t0["ii_pass"] else "eta<=eps (T0(ii) failed: Thm 3 scope narrows)"
    t0["by_arm"] = {}
    for arm in ("off", "vol", "ev1", "full"):
        aa = [j for j in A if j["key"][3] == arm]
        t0["by_arm"][arm] = {"n": len(aa), "rem_bar_over_eps_median": _q([j["pred"]["rem_bar_over_eps_p1"] for j in aa], .5),
                             "phi_perp_median": _q([j["pred"]["phi_perp_max"] for j in aa], .5),
                             "S1_median": _q([j["pred"]["S1"] for j in aa], .5),
                             "A_k_median": _q([j["pred"]["A_k_max"] for j in aa], .5)}
    out["T0"] = t0
    # ------------------------------------------------------------------ cand_l G-R gate (zero interaction, R0 ledgers)
    gr = {"rbar2_over_eps_median": _q(rem, .5), "rbar2_over_eps_p90": _q(rem, .9),
          "rbar2_over_eps_median_eta_le_2eps": _q(rem_sub, .5), "rbar2_over_eps_p90_eta_le_2eps": _q(rem_sub, .9),
          "definition": "Rbar2 = computable hybrid-telescoping remainder bound (leverage.rem_bound) on the theta_hat "
                        "half-grid cell box, summed over the contrast pair, max over the two pre-registered pairs; "
                        "all P1 R0 ledger snapshots"}
    gr["pass"] = (gr["rbar2_over_eps_median"] is not None and gr["rbar2_over_eps_median"] <= 0.25
                  and gr["rbar2_over_eps_p90"] <= 0.5)
    out["cand_l_GR_gate"] = gr
    # ------------------------------------------------------------------ dev risk-layer runs
    dev = {}
    for setting in SETTINGS:
        rr = [j for j in Bc if j["res"].get("setting") == setting]
        cert = [j for j in rr if j["res"]["status"] == "CERTIFIED"]
        zc = [j for j in cert if j["res"]["zero_cost"]]
        dev[setting] = {"n_runs": len(rr), "status": dict(Counter(j["res"]["status"] for j in rr)),
                        "layers": dict(Counter(j["res"]["gap_layer"] for j in rr)),
                        "certified": len(cert), "false_cert": sum(j["res"]["false_cert"] for j in cert),
                        "zero_cost": len(zc), "zero_cost_false": sum(j["res"]["false_cert"] for j in zc),
                        "paid_false": sum(j["res"]["false_cert"] for j in cert if not j["res"]["zero_cost"]),
                        "zero_cost_nontie": sum(1 for j in zc if j["res"]["gap_layer"] != "tie"),
                        "zero_cost_nontie_false": sum(j["res"]["false_cert"] for j in zc
                                                      if j["res"]["gap_layer"] != "tie"),
                        "identity_err_max": max([j["res"].get("identity_err_max", 0.0) for j in rr] or [None]),
                        "S1_median_cert": _q([j["pred"]["S1"] for j in cert], .5),
                        "rem_bar_over_eps_median_cert": _q([j["pred"].get("rem_bar_over_eps_max") for j in cert], .5),
                        "eta_arg_over_eps_median_cert": _q([j["res"].get("eta_arg", np.nan) / EPS for j in cert], .5),
                        "sec_per_call_median": _q([j["pred"]["sec_per_call"] for j in rr], .5),
                        "new_steps_median": _q([j["res"]["new_env_steps"] for j in rr], .5)}
    out["dev_runs"] = dev
    # ------------------------------------------------------------------ exploratory Q1 AUC (zero-cost reuse certs)
    def is_q1(j):
        r = j["res"]
        return (r["status"] == "CERTIFIED" and r["zero_cost"] and j["key"][4] >= 1 and r["gap_layer"] != "tie")

    q1 = [j for j in Bc if is_q1(j)]
    scores = {"S1": lambda j: j["pred"]["S1"], "S2": lambda j: j["pred"]["S2"],
              "A_k": lambda j: j["pred"]["A_k_max"], "neg_log_gap_hat": lambda j: j["pred"]["neg_log_gap_hat"],
              "log_tr_Iinv": lambda j: j["pred"]["log_tr_Iinv"], "eta_hat_gof": lambda j: j["pred"]["eta_hat_gof"],
              "eta_arg_oracle": lambda j: j["res"].get("eta_arg"), "eta_dec_oracle": lambda j: j["res"].get("eta_dec")}
    lab = lambda j: j["res"]["false_cert"]  # noqa: E731
    clu = lambda j: j["key"][0]  # noqa: E731
    q1b = {"pooled_static_m1": auc_block(q1, scores, lab, clu)}
    for setting in SETTINGS:
        q1b[setting] = auc_block([j for j in q1 if j["res"]["setting"] == setting], scores, lab, clu)
    # pooled with the R0 zero-cost events (realisable: all negatives expected)
    r0zc = [j for j in A if j["key"][3] == "full" and j["res"]["zero_cost"] and j["key"][4] >= 1
            and j["res"]["gap_layer"] != "tie"]
    r0scores = {k: v for k, v in scores.items() if k not in ("eta_arg_oracle", "eta_dec_oracle")}
    r0scores["eta_arg_oracle"] = lambda j: j["res"].get("eta_arg")
    pooled_all = [{"pred": j["pred"], "res": j["res"], "key": j["key"]} for j in q1 + r0zc]
    q1b["pooled_static_m1_R0"] = auc_block(pooled_all, r0scores, lab, clu)
    q1b["n_R0_zero_cost_nontie"] = len(r0zc)
    q1b["R0_zero_cost_nontie_false"] = sum(bool(j["res"]["false_cert"]) for j in r0zc)
    q1b["incremental_logistic_eta_arg_plus_S1"] = incremental_logistic(q1, lab, "eta_arg", "S1")
    q1b["note"] = "exploratory only (dev seeds); never used as evidence"
    out["Q1_exploratory"] = q1b
    # ------------------------------------------------------------------ exploratory Q2 (zero-cost occurrence at k>=1)
    op = [j for j in Aopen + Bopen]
    s2 = {"A_k": lambda j: j["pred"]["A_k_max"], "A_k_placebo_block": lambda j: j["pred"]["A_k_placebo_block_max"],
          "A_k_placebo_global": lambda j: j["pred"]["A_k_placebo_global_max"],
          "S1": lambda j: j["pred"]["S1"], "neg_log_gap_hat": lambda j: -j["pred"]["neg_log_gap_hat"],
          "log_tr_Iinv_neg": lambda j: -j["pred"]["log_tr_Iinv"]}
    labz = lambda j: bool(j["res"]["zero_cost"])  # noqa: E731
    q2 = {"pooled": auc_block(op, s2, labz, clu)}
    q2["R0_P1_full"] = auc_block(Aopen, s2, labz, clu)
    q2["dev_static_m1"] = auc_block(Bopen, s2, labz, clu)
    try:
        from dsswm.stats.auc import lr_test
        y = np.array([labz(j) for j in op], float)
        base = np.array([[j["pred"]["neg_log_gap_hat"], j["pred"]["log_tr_Iinv"]] for j in op])
        ak = np.array([[math.log(max(j["pred"]["A_k_max"], 1e-12))] for j in op])
        okr = np.all(np.isfinite(base), 1) & np.isfinite(ak[:, 0])
        q2["lr_test_log_Ak"] = lr_test(base[okr], np.c_[base[okr], ak[okr]], y[okr])
    except Exception as e:  # noqa: BLE001
        q2["lr_test_log_Ak"] = {"error": repr(e)}
    q2["note"] = "exploratory only; positive score direction: larger = more likely zero-cost (baselines sign-flipped)"
    out["Q2_exploratory"] = q2
    # ------------------------------------------------------------------ cand_o promotion gate
    pb = q1b["pooled_static_m1"]
    auc_s1 = pb.get("S1", {}).get("auc") if isinstance(pb.get("S1"), dict) else None
    d_eta = pb.get("delta_auc_S1_minus_eta_arg", {}).get("est") if isinstance(pb.get("delta_auc_S1_minus_eta_arg"), dict) else None
    co = {"nontie_auc_S1": auc_s1, "delta_auc_S1_minus_eta_arg": d_eta, "T0_ii_pass": t0["ii_pass"],
          "evaluable": auc_s1 is not None and np.isfinite(auc_s1),
          "rule": "promote iff non-tie AUC(Lambda_hat_perp) >= 0.80 AND AUC(S1) - AUC(eta_arg) > 0 AND T0(ii) passes"}
    co["pass"] = bool(co["evaluable"] and auc_s1 >= 0.80 and d_eta is not None and d_eta > 0 and t0["ii_pass"])
    out["cand_o_gate"] = co
    # ------------------------------------------------------------------ S1 / S2 variant preview (full: freeze)
    a1 = pb.get("S1", {}).get("auc") if isinstance(pb.get("S1"), dict) else None
    a2 = pb.get("S2", {}).get("auc") if isinstance(pb.get("S2"), dict) else None
    if a1 is None or a2 is None or not (np.isfinite(a1) and np.isfinite(a2)):
        choice, why = "S2", "AUC not evaluable -> tie rule (S2)"
    elif abs(a1 - a2) < 1e-12:
        choice, why = "S2", "tie -> S2"
    else:
        choice, why = ("S1", "higher non-tie AUC") if a1 > a2 else ("S2", "higher non-tie AUC")
    out["variant"] = {"auc_S1": a1, "auc_S2": a2, "choice": choice, "reason": why,
                      "binding": mode == "full", "scope": t0["scope"],
                      "note": "pilot: provisional preview only; the binding choice is made in full mode on 720-739"}
    return out


def make_plots(J, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    A = [j for j in J if j["res"].get("source") == "R0_ledger"]
    if not A:
        return
    rem = np.array([j["pred"]["rem_bar_over_eps_p1"] for j in A], float)
    eta = np.array([j["res"]["eta"] for j in A], float)
    rex = np.array([max(j["res"]["rem_exact_over_eps_max"], 1e-18) for j in A], float)
    phi = np.array([j["pred"]["phi_perp_max"] for j in A], float)
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.6))
    bins = np.logspace(np.log10(max(min(rex.min(), rem.min()), 1e-12)), np.log10(rem.max() * 1.2), 40)
    ax[0].hist(rem[eta <= 2 * EPS], bins=bins, alpha=.75, label=r"$\bar R_{em}/\epsilon$ (bound), $\eta\leq2\epsilon$",
               color="#3b6ea8")
    ax[0].hist(rex[eta <= 2 * EPS], bins=bins, alpha=.6, label=r"exact $|Rem|/\epsilon$ (oracle)", color="#c8793a")
    ax[0].axvline(0.5, color="k", ls="--", lw=.8, label="T0(ii) threshold 0.5")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("Rem / eps")
    ax[0].set_ylabel("count (ledger snapshots)")
    ax[0].legend(frameon=False, fontsize=7)
    ax[1].hist(phi, bins=30, color="#4a8c5c")
    ax[1].axvline(0.3, color="k", ls="--", lw=.8, label="T0(iv) 0.3")
    ax[1].set_xlabel(r"$\hat\phi_\perp$ (max over the two pairs)")
    ax[1].set_ylabel("count")
    ax[1].legend(frameon=False, fontsize=7)
    fig.suptitle(f"T0 on P1 NL-R0 ledgers (n={len(A)} snapshots, dev seeds)", fontsize=9)
    fig.tight_layout()
    (out_dir / "figures").mkdir(exist_ok=True)
    fig.savefig(out_dir / "figures" / "fig_t0_rem_phi_hist.png", dpi=150)
    plt.close(fig)


def write_samples(J, out_dir):
    sd = out_dir / "samples"
    sd.mkdir(exist_ok=True)
    A = [j for j in J if j["res"].get("source") == "R0_ledger"]
    A = sorted(A, key=lambda j: j["pred"]["S1"])
    pick = [A[i] for i in np.linspace(0, len(A) - 1, min(8, len(A))).astype(int)] if A else []
    (sd / "r0_ledger_examples.json").write_text(json.dumps(pick, indent=1, default=str))
    zc = [j for j in J if str(j["res"].get("source", "")).startswith("dev_") and j["key"][3] == "full"
          and j["res"].get("zero_cost")]
    fz = [j for j in zc if j["res"]["false_cert"]][:6] + [j for j in zc if not j["res"]["false_cert"]][:4]
    (sd / "dev_zero_cost_events.json").write_text(json.dumps(fz, indent=1, default=str))


def update_gpu_progress(status, start_iso, wall_min, snapshot):
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
        d["timings"][TASK] = {"planned_min": PLANNED_MIN, "actual_min": int(round(wall_min)), "start_time": start_iso,
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
    global PLANNED_MIN
    PLANNED_MIN = 15 if a.mode == "pilot" else 40
    out_dir = RES_ROOT / ("pilots" if a.mode == "pilot" else "full") / TASK
    (out_dir / "parts").mkdir(parents=True, exist_ok=True)
    (RES_ROOT / f"{TASK}.pid").write_text(str(os.getpid()))
    started = datetime.now().isoformat()
    t0 = time.perf_counter()
    try:
        p1 = json.loads((cfg["p1_dir"] / "summary.json").read_text())
        p1_seeds, p1_streams, p1_np = p1["design"]["seeds"], p1["design"]["streams"], p1["design"]["n_problems"]
        progress(0, 1, "static_jtables")
        tinfo = static_tables(cfg["dev_seeds"], cfg["n_problems"])
        log(f"static tables {tinfo}")
        dev_seeds = [s for s in cfg["dev_seeds"] if s not in tinfo["quota_fail_seeds"]]
        jobs = [(job_dev, (s, st, cfg["n_problems"], out_dir)) for s in dev_seeds for st in SETTINGS]
        jobs += [(job_r0, (s, stv, m, p1_np, cfg["p1_dir"], out_dir)) for s in p1_seeds for stv in p1_streams
                 for m in ("JPC", "B3")]
        joblog = run_pool(jobs, min(a.workers, 4))
        J, errs, calib, wa = merge(out_dir)
        an = analyse(J, a.mode)
        try:
            make_plots(J, out_dir)
        except Exception as e:  # noqa: BLE001
            log(f"plot failed {e!r}")
        write_samples(J, out_dir)
        t0g = an["T0"]
        hard = {"T0_i_identity_le_1e-9": t0g["i_pass"], "T0_iii_counts_theta_hat_retrievable": t0g["iii_pass"],
                "no_fatal_jobs": not any("fatal" in j for j in joblog), "write_ahead_ok": wa["write_ahead_ok"]}
        passed = all(hard.values())
        summary = {
            "task": TASK, "mode": a.mode, "started_at": started, "finished_at": datetime.now().isoformat(),
            "note": "并发运行（与 r3_p2 / r3_p5 共享 CPU 与 4090），计时偏高；探索性 AUC 只来自开发种子，不计入证据",
            "design": {"p1_ledgers": {"dir": str(cfg["p1_dir"].relative_to(WS)), "seeds": p1_seeds,
                                      "streams": p1_streams, "n_problems": p1_np, "methods": ["JPC", "B3"],
                                      "arms": ["off", "vol", "ev1", "full"]},
                       "dev_runs": {"seeds": dev_seeds, "stream": 0, "n_problems": cfg["n_problems"],
                                    "settings": list(SETTINGS), "method": "JPC", "arm": "full", "tmax": TMAX},
                       "eps": EPS, "delta": DELTA, "new_interaction_on_R0": 0},
            "static_tables": tinfo, "m1_calibration": calib, "write_ahead": wa,
            "n_errors": len(errs), "errors_head": errs[:10], "jobs": joblog, **an,
            "hard_gates": hard, "passed": passed, "go_no_go": "GO" if passed else "NO_GO",
            "soft_gates": {"T0_ii_rem_p90_le_0.5": t0g["ii_pass"], "T0_iv_phi_median_le_0.3": not t0g["iv_phi_perp"]["median_gt_0.3"],
                           "consequence": ("T0(ii) fails -> Thm 3 / Q family scope narrows to eta <= eps; "
                                           if not t0g["ii_pass"] else "") +
                                          ("T0(iv) median phi_perp > 0.3 -> write 'targeted evidence has no cost "
                                           "advantage'" if t0g["iv_phi_perp"]["median_gt_0.3"] else "")},
            "code_sha256": code_sha(), "wall_clock_s": time.perf_counter() - t0}
        dv = an["dev_runs"]
        summary["metrics"] = {
            "identity_err_max": max([t0g["i_identity_err_max"] or 0.0] + [dv[s]["identity_err_max"] or 0.0 for s in SETTINGS]),
            "rem_over_eps_p90": t0g["ii_rem_bar_over_eps"]["p90_eta_le_2eps"],
            "phi_perp_median": t0g["iv_phi_perp"]["median_max_over_pairs"],
            "dev_auc_lambda": an["Q1_exploratory"]["pooled_static_m1"].get("S1"),
            "dev_auc_Ak": an["Q2_exploratory"]["pooled"].get("A_k"),
            "sec_per_call": t0g["sec_per_call"]["p1_logged_median"],
            "cand_o_gate": an["cand_o_gate"]["pass"], "cand_l_GR_gate": an["cand_l_GR_gate"]["pass"],
            "variant_preview": an["variant"]["choice"], "T0_scope": t0g["scope"]}
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
        wall_min = (time.perf_counter() - t0) / 60
        if not a.no_gpu_progress:
            update_gpu_progress("success" if passed else "failed", started, wall_min,
                                {"n_ledgers": t0g["n_ledger_snapshots"], "n_dev_runs": sum(dv[s]["n_runs"] for s in SETTINGS),
                                 "workers": a.workers, "gpu_model": tinfo.get("device"), "gpu_count": 1,
                                 "note": "CPU-dominated"})
        progress(1, 1, "done", {"go_no_go": summary["go_no_go"]})
        mark_done("success" if passed else "failed",
                  f"{summary['go_no_go']} hard={hard}; T0(i) max={t0g['i_identity_err_max']:.2e}; "
                  f"T0(ii) p90={t0g['ii_rem_bar_over_eps']['p90_eta_le_2eps']}; scope={t0g['scope']}; "
                  f"cand_o={an['cand_o_gate']['pass']}; cand_l={an['cand_l_GR_gate']['pass']}")
        log(f"done {summary['go_no_go']} in {wall_min:.1f} min")
    except Exception as e:  # noqa: BLE001
        (out_dir / "crash.txt").write_text(traceback.format_exc())
        mark_done("failed", repr(e))
        raise


if __name__ == "__main__":
    main()
