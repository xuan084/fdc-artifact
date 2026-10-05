"""hf2_static_falsify: HF2 zero-interaction falsification of the parameter-matched static model (cand_a / cand_g).

Setting (methodology 2.3 / 4.1 / 4.2): truth = R0 (kappa_high natural generator snapped to G_1; dynamic: load enters
the outcome logit and retention) or its kappa = 0 matched twin (every gamma_i and lam set to 0, all other coordinates
identical, CRN). Learner = static class on the SAME grid G_1 (13824 points, same parameters) in which the load
enters neither the outcome logit nor retention: g(n) = g_ret(n) = 1 (round-0 nl_component_ablations definition).
Implemented by the exact reparametrisation alpha' = alpha - gamma, gamma' = 0, tau' = tau - lam, lam' = 0 (checked
against the g = g_ret = 1 propagator tables at start-up).

Arms (each on its own fresh platform copy, CRN on the instance; each arm re-uses its OWN ledger across the stream):
  plugin        in-class plug-in SeqLRSet + exact minimax certificate + DDA (JPC, uncorrected) -- the "original" run.
                SHADOW mixed sets are updated on exactly the same rounds (no influence on acquisition):
                  main (pool+reg[load bucket]+ext, 1/3 each), pid (reg = problem id), placebo (reg = placebo labels),
                  half ({pool, ext}), reg_lb / reg_pid / reg_placebo ({pool, reg}, 1/2 each: isolates the regime
                  contrast), ext_only ({pool, ext}, 1/2 each).
                Conversion of an original false certificate = the shadow set had become empty (MODEL_CONFLICT) at or
                before the certification round, on the identical data.
  mixed_lb      JPC driven by the mixed set {pool, reg(load bucket), ext} (primary arm), replay check before every
                zero-cost ledger-reuse certificate, MODEL_CONFLICT -> B2 fallback (whole-trial residual + EB LUCB,
                steps charged to the problem), system keeps running.
  mixed_pid     same, reg = problem id (secondary arm)
  mixed_placebo same, reg = placebo labels (fixed per instance before data; only the reg component is replaced)
Twin (kappa = 0): plugin (+ shadows) and mixed_lb: the twin truth is IN the static class, so every MODEL_CONFLICT on
the twin is a false alarm.

Pilot: dev instances 640-649 x 10 problems x 1 stream (seed 42).  Full: lock range (10000-10047) x 3 streams x 15
problems (requires dsswm.stats.prereg.assert_locked()). Incremental: finished (instance, stream, truth, arm) units are
skipped on restart.

Usage: run_hf2_static_falsify.py --mode {pilot,full} [--workers 4] [--smoke]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import fcntl  # noqa: E402
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
CACHE = WS / "exp" / "cache"
JT1 = CACHE / "jtables_f1"
JTS = CACHE / "jtables_static_f1"
LOCK = WS / "plan" / "prereg_lock.json"
TASK = "hf2_static_falsify"
EPS, DELTA = 0.02, 0.05
TOP_M = 5
LOG_THR = math.log(1.0 / DELTA)
TMAX_STEP = 3000
TMAX_TRIAL = int(2.31e6)
PILOT_SEEDS = list(range(640, 650))
PILOT_NPROB = 10
PLACEBO_FREQS = (0.45, 0.35, 0.20)

from dsswm.acquire.nl_kl_dda import IncidenceIndex, class_prob_tables, dda_choose  # noqa: E402
from dsswm.baselines.whole_trial_bai import lucb  # noqa: E402
from dsswm.certify.eta_loc import build_plans, j_values  # noqa: E402
from dsswm.certify.minimax_enum import certify_minimax, group_ids, observable_signature, regret_matrix, trichotomy  # noqa: E402
from dsswm.evidence.lr_set import SeqLRSet  # noqa: E402
from dsswm.evidence.mixed_lr import MixedLRSet  # noqa: E402
from dsswm.evidence.regimes import PlaceboRegime, load_bucket, problem_id_regime  # noqa: E402
from dsswm.evidence.replay import replay_check  # noqa: E402
from dsswm.exact.nl_propagate import NLPropagator  # noqa: E402
from dsswm.models.nl_class import NLClass  # noqa: E402
from dsswm.stats.cp import clopper_pearson  # noqa: E402
from dsswm.streams.generator import DEFAULT_NL_S_GRID, NL_DEFAULTS, generator_hash  # noqa: E402
from dsswm.streams.offgrid import STREAMS, make_offgrid_instance  # noqa: E402

C_KNOWN, RHO_RET, NMAX = NL_DEFAULTS["c"], NL_DEFAULTS["rho_ret"], NL_DEFAULTS["nmax"]
SHADOWS = ("main", "pid", "placebo", "half", "reg_lb", "reg_pid", "reg_placebo", "ext_only")
MIXED_ARMS = {"mixed_lb": "main", "mixed_pid": "pid", "mixed_placebo": "placebo"}
ARMS_BY_TRUTH = {"R0": ["plugin", "mixed_lb", "mixed_pid", "mixed_placebo"], "twin": ["plugin", "mixed_lb"]}


# ------------------------------------------------------------------------------------ static class
def static_np_params(np_params: dict) -> dict:
    """g(n) = g_ret(n) = 1  <=>  alpha' = alpha - gamma, gamma' = 0, tau' = tau - lam, lam' = 0 (exact)."""
    p = {k: np.array(v, copy=True) for k, v in np_params.items()}
    lam = np.asarray(p["lam"], float).reshape(-1)
    p["alpha"] = p["alpha"] - p["gamma"]
    p["gamma"] = np.zeros_like(p["gamma"])
    p["tauL"] = p["tauL"] - lam[:, None]
    p["tauR"] = p["tauR"] - lam[:, None]
    p["lam"] = np.zeros_like(lam)
    return p


def static_torch_params(ncl: NLClass, device, dtype=torch.float64) -> dict:
    return {k: torch.as_tensor(v, device=device, dtype=dtype) for k, v in static_np_params(ncl.np_params).items()}


def check_static_equivalence(prop, ncl, device):
    """Shifted parametrisation == round-0 g = g_ret = 1 parametrisation (log-prob tables)."""
    P = ncl.torch_params()
    P = {k: v.to(device) for k, v in P.items()}
    N = NMAX + 1
    P["g"] = torch.ones(N, dtype=torch.float64, device=device)
    P["g_ret"] = torch.ones(N, dtype=torch.float64, device=device)
    a = prop.tables(P)[0]
    b = prop.tables(static_torch_params(ncl, device))[0]
    return float((a - b).abs().max())


def vstar_of(truth: dict) -> np.ndarray:
    return np.concatenate([truth["alpha"], truth["gamma"], truth["tauL"], truth["beta"], truth["tauR"],
                           [truth["psi"]], [truth["lam"]]]).astype(float)


def U1(q):
    return type("U", (), {"w": q.utility.w, "w_ret": q.utility.w_ret, "c_q": 1.0})()


def atomic_save(path: Path, arr):
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "wb") as fh:
        np.save(fh, arr)
    os.replace(tmp, path)


def make_inst(seed, truth, stream):
    return make_offgrid_instance(seed, "R0", stream=stream, twin=(truth == "twin"))


# ------------------------------------------------------------------------------------ phase 1: tables (GPU)
def precompute(seeds, n_prob, streams, dev, log):
    """Static-class J tables on G_1 (truth-free, cached), public c_q from the dynamic G_1 class max, and harness
    J_true at the R0 / twin truth by the exact propagator."""
    t0 = time.perf_counter()
    inst0 = make_inst(seeds[0], "R0", 0)
    prop = NLPropagator(2, 2, NMAX, inst0.env.aspace, C_KNOWN, RHO_RET, device=dev)
    ncl = NLClass(DEFAULT_NL_S_GRID, device=dev)
    meta = {"cq": {}, "jtrue": {"R0": {}, "twin": {}}, "sec": {}, "static_equiv_maxdiff": None}
    meta["static_equiv_maxdiff"] = check_static_equivalence(prop, ncl, dev)
    log(f"static reparametrisation vs g=g_ret=1 tables: max |diff| = {meta['static_equiv_maxdiff']:.2e}")
    Ps = static_torch_params(ncl, dev)
    JTS.mkdir(parents=True, exist_ok=True)
    JT1.mkdir(parents=True, exist_ok=True)
    P1 = None
    t_s, t_d = [], []
    pids_needed = {}
    for s in seeds:
        inst = make_inst(s, "R0", 0)                       # stream 0 = identity order
        probs = inst.problems[: (n_prob if streams == [0] else len(inst.problems))]
        tw = make_inst(s, "twin", 0)
        for q in probs:
            pids_needed[q.pid] = (q, vstar_of(inst.truth), vstar_of(tw.truth))
    with open(JTS / ".hf2.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        for pid, (q, _, _) in pids_needed.items():
            ps = JTS / f"{pid}_raw.npy"
            if not ps.exists():
                t1 = time.perf_counter()
                atomic_save(ps, prop.j_table(Ps, q.policies, q.loads0, q.engaged0, q.H, U1(q)))
                t_s.append(time.perf_counter() - t1)
            pd = JT1 / f"{pid}_raw.npy"
            if not pd.exists():
                P1 = P1 or ncl.torch_params()
                t1 = time.perf_counter()
                atomic_save(pd, prop.j_table(P1, q.policies, q.loads0, q.engaged0, q.H, U1(q)))
                t_d.append(time.perf_counter() - t1)
        fcntl.flock(lk, fcntl.LOCK_UN)
    for pid, (q, v0, vt) in pids_needed.items():
        c = 1.0 / float(np.load(JT1 / f"{pid}_raw.npy").max())       # public dynamic-class max (round-0 scale)
        meta["cq"][pid] = c
        plans = build_plans(prop, q)
        jj = j_values(prop, plans, np.stack([v0, vt]), q.utility.w, q.utility.w_ret) * c
        meta["jtrue"]["R0"][pid] = jj[0]
        meta["jtrue"]["twin"][pid] = jj[1]
    meta["sec"] = {"static_jtable_new_mean": float(np.mean(t_s)) if t_s else None, "n_static_new": len(t_s),
                   "dyn_jtable_new_mean": float(np.mean(t_d)) if t_d else None, "n_dyn_new": len(t_d),
                   "total_s": time.perf_counter() - t0}
    if torch.cuda.is_available():
        meta["gpu_peak_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20
        meta["gpu_name"] = torch.cuda.get_device_name(0)
        meta["vram_total_mb"] = torch.cuda.get_device_properties(0).total_memory / 2 ** 20
    del prop, ncl, Ps, P1
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return meta


# ------------------------------------------------------------------------------------ phase 2: learners (CPU)
_CTX: dict = {}


def ctx(aspace):
    if "LT" not in _CTX:
        torch.set_num_threads(1)
        ncl = NLClass(DEFAULT_NL_S_GRID, device="cpu")
        prop = NLPropagator(2, 2, NMAX, aspace, C_KNOWN, RHO_RET, device="cpu")
        LT = prop.tables(static_torch_params(ncl, "cpu"))[0]
        npp = static_np_params(ncl.np_params)
        py, pe = class_prob_tables(npp, C_KNOWN, NMAX, aspace.nb)
        gid = group_ids(observable_signature(py, pe, max_level=1))
        _CTX.update(ncl=ncl, prop=prop, LT=LT, LT_np=LT.numpy(), py=py, pe=pe, gid=gid,
                    all_single=len(np.unique(gid)) == ncl.B, inc=IncidenceIndex(prop.codec, aspace, NMAX),
                    legal=np.arange(aspace.n))
    return _CTX


def make_set(kind, C, seed):
    prop, LT = C["prop"], C["LT"]
    plc = PlaceboRegime(seed, PLACEBO_FREQS)
    if kind == "pool":
        return SeqLRSet(prop, LT, DELTA)
    if kind == "main":
        return MixedLRSet(prop, LT, DELTA, regime_fn=load_bucket, name="main")
    if kind == "pid":
        return MixedLRSet(prop, LT, DELTA, regime_fn=problem_id_regime, name="pid")
    if kind == "placebo":
        return MixedLRSet(prop, LT, DELTA, regime_fn=plc, name="placebo")
    if kind == "half":
        return MixedLRSet(prop, LT, DELTA, components=("pool", "ext"), name="half")
    if kind == "reg_lb":
        return MixedLRSet(prop, LT, DELTA, components=("pool", "reg"), regime_fn=load_bucket, name="reg_lb")
    if kind == "reg_pid":
        return MixedLRSet(prop, LT, DELTA, components=("pool", "reg"), regime_fn=problem_id_regime, name="reg_pid")
    if kind == "reg_placebo":
        return MixedLRSet(prop, LT, DELTA, components=("pool", "reg"), regime_fn=plc, name="reg_placebo")
    if kind == "ext_only":
        return MixedLRSet(prop, LT, DELTA, components=("pool", "ext"), weights=(0.5, 0.5), name="ext_only")
    raise ValueError(kind)


def replay_kwargs(kind, seed):
    if kind == "main":
        return {"regime_fn": load_bucket}
    if kind == "pid":
        return {"regime_fn": problem_id_regime}
    if kind == "placebo":
        return {"regime_fn": PlaceboRegime(seed, PLACEBO_FREQS)}
    raise ValueError(kind)


def _u_max(q, c):
    return c * (2 * float(q.utility.w.sum()) + q.utility.w_ret * 4)


def run_b2(env, q, c, J_proxy_row, seed, noise, k):
    counters = {}

    def sampler(a, n):
        cc = counters.get(a, 0)
        counters[a] = cc + 1
        r = np.random.default_rng([seed, noise, k, a, cc, 57])
        ys, es = env.simulate_batch(q.policies[a], q.loads0, q.engaged0, q.H, n, r)
        return c * (ys @ q.utility.w + q.utility.w_ret * es)
    return lucb(sampler, len(q.policies), q.H, EPS, DELTA, _u_max(q, c), TMAX_TRIAL, kind="eb", proxy=J_proxy_row)


def run_unit(seed, stream, truth, arm, n_prob, meta, want_samples):
    """One (instance, stream, truth, arm). Returns rows, samples, error, unit stats."""
    torch.set_num_threads(1)
    t_unit = time.time()
    noise = STREAMS[stream][0]
    inst = make_inst(seed, truth, stream)
    C = ctx(inst.env.aspace)
    prop = C["prop"]
    problems = inst.problems[:n_prob] if stream == 0 else [q for q in inst.problems if q.pid in meta["cq"]][:n_prob]
    env = inst.env
    h = env.handle()
    lead_kind = "pool" if arm == "plugin" else MIXED_ARMS[arm]
    lead = make_set(lead_kind, C, seed)
    shadows = {k: make_set(k, C, seed) for k in SHADOWS} if arm == "plugin" else {}
    ledger = []                                    # (obs, problem_id) in collection order

    def feed(obs, pid):
        if lead_kind == "pool":
            lead.update(obs)
        else:
            lead.update(obs, problem_id=pid)
        for ms in shadows.values():
            ms.update(obs, problem_id=pid)
        ledger.append((obs, pid))

    for o in inst.init_obs:
        feed(o, None)
    rng = np.random.default_rng([seed, noise, 202, list(MIXED_ARMS).index(arm) + 1 if arm != "plugin" else 0])
    rows, samples = [], []
    conflict_seen = False
    for k, q in enumerate(problems):
        t0 = time.perf_counter()
        c = meta["cq"][q.pid]
        J = np.load(JTS / f"{q.pid}_raw.npy").astype(float) * c
        Reg = regret_matrix(J)
        Jt = np.asarray(meta["jtrue"][truth][q.pid])
        steps, status, traj, replay_info = 0, None, [], None
        while True:
            mask = lead.mask().numpy()
            cert = certify_minimax(Reg, mask, EPS, TOP_M)
            sv = cert["status"].value
            if sv == "CERTIFIED" and steps == 0 and lead_kind != "pool" and len(ledger) > len(inst.init_obs):
                # zero-cost certificate from ledger reuse: replay check on the full ledger (no interaction)
                n_before = h.n_steps
                rc = replay_check(prop, C["LT"], ledger, q.pid, DELTA, env_handle=h, **replay_kwargs(lead_kind, seed))
                assert h.n_steps == n_before
                replay_info = {"replay_conflict": bool(rc["conflict"]), "replay_size": int(rc["size"]),
                               "online_size": int(mask.sum()), "n_replayed": int(rc["n_replayed"]),
                               "replay_eq_online": bool(np.array_equal(rc["mask"].numpy(), mask))}
                if rc["conflict"]:
                    sv = "MODEL_CONFLICT"
            if sv in ("CERTIFIED", "MODEL_CONFLICT"):
                status = sv
                break
            if not C["all_single"] and steps % 10 == 0:
                amb, _, _ = trichotomy(Reg, mask, C["gid"], EPS)
                if amb:
                    status = "OUT_OF_SCOPE"
                    break
            if steps >= TMAX_STEP:
                status = "NEED_DATA"
                break
            kh = lead.mle()
            blk = cert["blocking"] or [int(i) for i in np.flatnonzero(mask)[:TOP_M]]
            margins = LOG_THR - lead.log_ratio().numpy()[blk]
            code = prop.codec.encode(*h.observable_state())
            a, info = dda_choose(code, C["py"][kh:kh + 1], C["pe"][kh:kh + 1], C["LT_np"][kh], C["py"][blk],
                                 C["pe"][blk], margins, C["inc"], prop, C["legal"], rng)
            obs = h.step(a)
            feed(obs, q.pid)
            steps += 1
            if want_samples and len(traj) < 30:
                traj.append({"step": steps, "a": int(a), "set": int(mask.sum()), "r_bar": round(float(cert["r_bar"]), 4),
                             "mle": int(kh), "mode": info["mode"]})
        assert env.n_steps == len(ledger) == lead.n_rounds, "step accounting mismatch"
        mask = lead.mask().numpy()
        pi = cert["pi"] if status == "CERTIFIED" else None
        jpc_status = status
        fb = None
        charged = steps
        if status == "MODEL_CONFLICT":
            conflict_seen = True
            kh = lead.mle()
            fb = run_b2(env, q, c, J[kh], seed, noise, k)
            charged = steps + int(fb["steps"])
            status = "CERTIFIED_B2" if fb["status"] == "CERTIFIED" else "NEED_DATA_B2"
            pi = fb["pi"] if fb["status"] == "CERTIFIED" else None
        certified = status in ("CERTIFIED", "CERTIFIED_B2")
        tr = float(Jt.max() - Jt[pi]) if pi is not None else None
        if not certified:
            charged = max(charged, TMAX_TRIAL if fb is not None else TMAX_STEP)
        r = {"kind": "hf2", "instance": seed, "stream": stream, "noise_seed": noise, "truth": truth, "arm": arm,
             "method": f"{arm}@{truth}", "problem": q.pid, "gen_index": int(q.meta.get("gen_index", -1)), "k": k,
             "jpc_status": jpc_status, "status": status, "certified": certified,
             "steps_consumed_stepwise": int(steps), "fallback_steps": int(fb["steps"]) if fb else 0,
             "new_env_steps": int(charged), "censored": not certified, "certified_policy": pi, "true_regret": tr,
             "false_cert": bool(certified and tr is not None and tr > EPS),
             "false_cert_jpc": bool(jpc_status == "CERTIFIED" and tr is not None and tr > EPS),
             "zero_cost": bool(jpc_status == "CERTIFIED" and steps == 0),
             "set_size": int(mask.sum()), "conflict_round": getattr(lead, "conflict_round", None),
             "ledger_rounds": len(ledger), "env_n_steps": int(env.n_steps), "lr_n_rounds": int(lead.n_rounds),
             "true_gap_top2": float(np.sort(Jt)[-1] - np.sort(Jt)[-2]), "H": q.H, "n_policies": len(q.policies),
             "eps": EPS, "delta": DELTA, "wall_clock_s": time.perf_counter() - t0, "rollouts": 0,
             "falsification_scope": lead.falsification_scope() if lead_kind != "pool" else {"components": []}}
        if replay_info:
            r["replay"] = replay_info
        if lead_kind != "pool":
            r["component_adv"] = {kk: round(v, 3) for kk, v in lead.component_advantage().items()}
            r["conflict_before_problem"] = bool(conflict_seen and jpc_status == "MODEL_CONFLICT"
                                                and lead.conflict_round is not None
                                                and lead.conflict_round <= len(ledger) - steps)
        if shadows:
            r["shadow"] = {kk: {"conflict": ms.conflict_round is not None, "conflict_round": ms.conflict_round,
                                "size": ms.size(), "margin": round(ms.evidence_margin(), 3),
                                "adv": {cc: round(v, 2) for cc, v in ms.component_advantage().items()}}
                           for kk, ms in shadows.items()}
        rows.append(r)
        if want_samples and k < 3:
            samples.append({**r, "J_true": Jt.round(4).tolist(), "trajectory_head": traj,
                            "truth_params": inst.truth})
    return rows, samples, None, {"instance": seed, "stream": stream, "truth": truth, "arm": arm,
                                 "sec": time.time() - t_unit, "rounds": len(ledger)}


def run_unit_safe(*a):
    try:
        return run_unit(*a)
    except Exception as e:  # noqa: BLE001
        seed, stream, truth, arm = a[:4]
        return [], [], {"instance": seed, "stream": stream, "truth": truth, "arm": arm, "error": repr(e),
                        "tb": traceback.format_exc()}, {"instance": seed, "stream": stream, "truth": truth,
                                                        "arm": arm, "sec": 0.0, "rounds": 0}


# ------------------------------------------------------------------------------------ analysis
def cp(k, n):
    if n == 0:
        return [None, None]
    lo, hi = clopper_pearson(k, n, 0.05)
    return [float(lo), float(hi)]


def arm_summary(rows):
    n = len(rows)
    cert = [r for r in rows if r["certified"]]
    fc = sum(r["false_cert"] for r in cert)
    jc = [r for r in rows if r["jpc_status"] == "CERTIFIED"]
    fcj = sum(r["false_cert_jpc"] for r in jc)
    sc = {}
    for r in rows:
        sc[r["status"]] = sc.get(r["status"], 0) + 1
    streams = {}
    for r in rows:
        streams.setdefault((r["instance"], r["stream"]), []).append(r)
    conf_streams = sum(any(x["jpc_status"] == "MODEL_CONFLICT" for x in v) for v in streams.values())
    return {"n": n, "status_counts": sc, "completion": len(cert) / n if n else None,
            "certs": len(cert), "false_certs": fc, "fcr": fc / len(cert) if cert else None, "fcr_cp95": cp(fc, len(cert)),
            "jpc_certs": len(jc), "jpc_false_certs": fcj, "fcr_jpc": fcj / len(jc) if jc else None,
            "fcr_jpc_cp95": cp(fcj, len(jc)),
            "model_conflict_problems": sum(r["jpc_status"] == "MODEL_CONFLICT" for r in rows),
            "model_conflict_problem_rate": sum(r["jpc_status"] == "MODEL_CONFLICT" for r in rows) / n if n else None,
            "n_streams": len(streams), "streams_with_conflict": conf_streams,
            "stream_conflict_rate": conf_streams / len(streams) if streams else None,
            "stream_conflict_cp95": cp(conf_streams, len(streams)),
            "zero_cost_rate": sum(r["zero_cost"] for r in rows) / n if n else None,
            "stream_steps_charged_mean": float(np.mean([sum(x["new_env_steps"] for x in v) for v in streams.values()])),
            "stepwise_steps_mean_per_problem": float(np.mean([r["steps_consumed_stepwise"] for r in rows])),
            "fallback_steps_mean_per_problem": float(np.mean([r["fallback_steps"] for r in rows])),
            "replay_checks": sum("replay" in r for r in rows),
            "replay_triggers": sum(r.get("replay", {}).get("replay_conflict", False) for r in rows),
            "replay_eq_online_all": all(r["replay"]["replay_eq_online"] for r in rows if "replay" in r),
            "wall_clock_per_problem_s": float(np.mean([r["wall_clock_s"] for r in rows]))}


def analyse(rows):
    out = {"by_arm": {}}
    for key in sorted({r["method"] for r in rows}):
        out["by_arm"][key] = arm_summary([r for r in rows if r["method"] == key])
    idx = {(r["instance"], r["stream"], r["truth"], r["arm"], r["k"]): r for r in rows}
    # ---- shadow conversion on the identical data (original false certificates of the static plug-in)
    orig = [r for r in rows if r["arm"] == "plugin" and r["truth"] == "R0" and r["false_cert_jpc"]]
    orig_cert = [r for r in rows if r["arm"] == "plugin" and r["truth"] == "R0" and r["jpc_status"] == "CERTIFIED"]
    conv = {}
    for s in SHADOWS:
        kk = sum(r["shadow"][s]["conflict"] for r in orig)
        true_c = sum(r["shadow"][s]["conflict"] for r in orig_cert if not r["false_cert_jpc"])
        conv[s] = {"converted": kk, "n_false": len(orig), "rate": kk / len(orig) if orig else None,
                   "cp95": cp(kk, len(orig)),
                   "also_flagged_correct_certs": true_c, "n_correct_certs": len(orig_cert) - len(orig)}
    out["shadow_conversion"] = conv
    # residual FCR if every shadow-flagged certificate is withheld (converted to MODEL_CONFLICT)
    out["shadow_residual_fcr"] = {}
    for s in SHADOWS:
        keep = [r for r in orig_cert if not r["shadow"][s]["conflict"]]
        fc = sum(r["false_cert_jpc"] for r in keep)
        out["shadow_residual_fcr"][s] = {"certs_kept": len(keep), "false": fc, "fcr": fc / len(keep) if keep else None,
                                         "cp95": cp(fc, len(keep))}
    # ---- arm-level conversion: same (instance, stream, problem) under the mixed arm (own trajectory)
    arm_conv = {}
    for a in MIXED_ARMS:
        hit, n, fc_same = 0, 0, 0
        for r in orig:
            m = idx.get((r["instance"], r["stream"], "R0", a, r["k"]))
            if m is None:
                continue
            n += 1
            hit += m["jpc_status"] == "MODEL_CONFLICT"
            fc_same += m["false_cert"]
        arm_conv[a] = {"converted": hit, "n_false": n, "rate": hit / n if n else None, "cp95": cp(hit, n),
                       "still_false_on_same_problem": fc_same}
    out["arm_conversion"] = arm_conv
    # ---- placebo / true-label detection ratios
    def ratio(a, b):
        return (a / b) if (a is not None and b not in (None, 0)) else None
    sc = out["shadow_conversion"]
    out["placebo_over_true"] = {
        "shadow_full_mixed": ratio(sc["placebo"]["rate"], sc["main"]["rate"]),
        "shadow_reg_only": ratio(sc["reg_placebo"]["rate"], sc["reg_lb"]["rate"]),
        "shadow_reg_only_pid_over_lb": ratio(sc["reg_pid"]["rate"], sc["reg_lb"]["rate"]),
        "arm_full_mixed": ratio(arm_conv["mixed_placebo"]["rate"], arm_conv["mixed_lb"]["rate"]),
        "arm_pid_over_lb": ratio(arm_conv["mixed_pid"]["rate"], arm_conv["mixed_lb"]["rate"]),
        "note": "pre-registered placebo replaces only the reg component; with ext present the placebo arm inherits "
                "ext's detections (ext registers the free g(n) direction, i.e. exactly the static model's missing "
                "direction). reg-only rows isolate the regime contrast (Proposition C')."}
    # ---- twin false alarms (twin truth is in the static class)
    tw = {}
    for a in ARMS_BY_TRUTH["twin"]:
        rr = [r for r in rows if r["truth"] == "twin" and r["arm"] == a]
        if rr:
            s = arm_summary(rr)
            tw[a] = {"stream_conflict": s["streams_with_conflict"], "n_streams": s["n_streams"],
                     "stream_rate": s["stream_conflict_rate"], "stream_cp95": s["stream_conflict_cp95"],
                     "problem_conflict": s["model_conflict_problems"], "n_problems": s["n"], "fcr": s["fcr"]}
    twp = [r for r in rows if r["truth"] == "twin" and r["arm"] == "plugin"]
    if twp:
        last = {}
        for r in twp:
            if (r["instance"], r["stream"]) not in last or r["k"] > last[(r["instance"], r["stream"])]["k"]:
                last[(r["instance"], r["stream"])] = r
        for s in SHADOWS:
            kk = sum(v["shadow"][s]["conflict"] for v in last.values())
            tw[f"shadow_{s}"] = {"stream_conflict": kk, "n_streams": len(last), "stream_rate": kk / len(last),
                                 "stream_cp95": cp(kk, len(last))}
    out["twin_false_alarm"] = tw
    # ---- replay share of detections (mixed arms)
    rep = {}
    for a in MIXED_ARMS:
        rr = [r for r in rows if r["arm"] == a and r["truth"] == "R0"]
        det = [r for r in rr if r["jpc_status"] == "MODEL_CONFLICT"]
        via = sum(r.get("replay", {}).get("replay_conflict", False) for r in det)
        rep[a] = {"detections": len(det), "via_replay": via, "share": via / len(det) if det else None,
                  "replay_checks": sum("replay" in r for r in rr)}
    out["replay_share_of_detections"] = rep
    # ---- conflict timing (problem index of first MODEL_CONFLICT) vs first original false certificate
    timing = []
    for (inst, st) in sorted({(r["instance"], r["stream"]) for r in orig}):
        first_fc = min(r["k"] for r in orig if r["instance"] == inst and r["stream"] == st)
        e = {"instance": inst, "stream": st, "first_false_cert_k": first_fc}
        for a in MIXED_ARMS:
            ks = [r["k"] for r in rows if r["instance"] == inst and r["stream"] == st and r["arm"] == a
                  and r["truth"] == "R0" and r["jpc_status"] == "MODEL_CONFLICT"]
            e[f"{a}_first_conflict_k"] = min(ks) if ks else None
        pl = [r for r in rows if r["instance"] == inst and r["stream"] == st and r["arm"] == "plugin" and r["truth"] == "R0"]
        cr = [r["shadow"]["main"]["conflict_round"] for r in pl if r["shadow"]["main"]["conflict_round"]]
        e["shadow_main_conflict_round"] = min(cr) if cr else None
        fr = [r["ledger_rounds"] for r in pl if r["k"] == first_fc]
        e["first_false_cert_round"] = fr[0] if fr else None
        timing.append(e)
    out["conflict_timing"] = timing
    return out


# ------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resummarize", action="store_true")
    ap.add_argument("--tag", default="", help="supplementary dev run: output suffix; never touches PID/DONE")
    ap.add_argument("--dev-seeds", default="", help="supplementary dev seeds a-b (pilot only)")
    ap.add_argument("--streams", default="", help="supplementary streams, e.g. 0,1,2 (pilot only)")
    args = ap.parse_args()
    pilot = args.mode == "pilot"
    lock = json.loads(LOCK.read_text())
    assert lock.get("version") == 2, "prereg lock v2 required"
    if not pilot:
        from dsswm.stats.prereg import assert_locked
        lock = assert_locked()
    quiet = args.smoke or bool(args.tag)
    out_dir = RES_ROOT / ("pilots" if pilot else "full") / (TASK + ("_smoke" if args.smoke else "") + args.tag)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    res_path = out_dir / "results.jsonl"

    def progress(step, total, phase, metric=None):
        if quiet:
            return
        (RES_ROOT / f"{TASK}_PROGRESS.json").write_text(json.dumps({
            "task_id": TASK, "epoch": step, "total_epochs": total, "step": step, "total_steps": total, "loss": None,
            "metric": {"phase": phase, **(metric or {})}, "updated_at": datetime.now().isoformat()}))

    def mark_done(status, summary):
        if quiet:
            return
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
                                                            "final_progress": fp,
                                                            "timestamp": datetime.now().isoformat()}))

    if not (quiet or args.resummarize):
        (RES_ROOT / f"{TASK}.pid").write_text(str(os.getpid()))
    logf = open(out_dir / "run.log", "a")

    def log(msg):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    if pilot:
        seeds, n_prob, streams = PILOT_SEEDS, PILOT_NPROB, [0]
        if args.dev_seeds:
            a_, b_ = (int(x) for x in args.dev_seeds.split("-"))
            seeds = list(range(a_, b_ + 1))
        if args.streams:
            streams = [int(x) for x in args.streams.split(",")]
            n_prob = 15 if streams != [0] else n_prob
        assert max(seeds) < 10000, "dev seeds only in pilot"
    else:
        rng_ = lock["eval_manifest"]["per_task_ranges"][TASK]
        seeds = [s for a, b in rng_ for s in range(a, b + 1)]
        n_prob, streams = int(lock["eval_manifest"].get("K_problems", 15)), [0, 1, 2]
    if args.smoke:
        seeds, n_prob = seeds[:1], 3
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)
    T0 = time.time()
    summary = {"task_id": TASK, "mode": args.mode, "seed": 42, "eps": EPS, "delta": DELTA, "truth_level": "R0",
               "learner": "static class on G_1 (g = g_ret = 1; |Theta| = 13824)", "instances": [seeds[0], seeds[-1]],
               "n_instances": len(seeds), "n_problems_per_stream": n_prob, "streams": streams,
               "eval_seeds_touched": not pilot, "generator_hash": generator_hash(), "concurrent_run": True,
               "note": "并发运行（4 槽并行，每任务 4 worker）；计时偏高", "tmax_stepwise": TMAX_STEP,
               "tmax_whole_trial": TMAX_TRIAL, "lock_status": lock.get("status"),
               "lock_sha256": lock.get("sha256") or lock.get("sha256_provisional"),
               "arms": ARMS_BY_TRUTH, "shadows": list(SHADOWS), "placebo_freqs": list(PLACEBO_FREQS)}
    try:
        if args.resummarize:
            rows = [json.loads(x) for x in open(res_path)]
            old = json.loads((out_dir / "summary.json").read_text())
            old.update(analyse(rows))
            (out_dir / "summary.json").write_text(json.dumps(old, indent=1, default=float))
            return
        log(f"start {TASK} mode={args.mode} device={dev} seeds {seeds[0]}-{seeds[-1]} ({len(seeds)}) x {n_prob} "
            f"problems x streams {streams}; lock={lock.get('status')}")
        progress(1, 4, "gpu: static J tables + J_true")
        meta = precompute(seeds, n_prob, streams, dev, log)
        summary["table_sec"] = meta["sec"]
        summary["static_equiv_maxdiff"] = meta["static_equiv_maxdiff"]
        summary["gpu_peak_mb"] = meta.get("gpu_peak_mb")
        log(f"tables: {meta['sec']}; gpu peak {meta.get('gpu_peak_mb')} MB")
        if not quiet and meta.get("gpu_name"):
            (RES_ROOT / f"{TASK}_gpu_profile.json").write_text(json.dumps({
                "gpu_name": meta["gpu_name"], "vram_total_mb": meta["vram_total_mb"], "max_batch_size": None,
                "vram_used_mb": meta["gpu_peak_mb"], "utilization_pct": 100 * meta["gpu_peak_mb"] / meta["vram_total_mb"],
                "note": "GPU only for static-class J tables (13824 x K per problem, one batch); learners are CPU. "
                        "Batch probing not applicable (exact propagator, single full-class batch fits); shared 4090, "
                        "<= 5 GB budget per task"}))
        assert meta["static_equiv_maxdiff"] < 1e-9, "static reparametrisation mismatch"
        done = set()
        if res_path.exists():
            cnt = {}
            for x in open(res_path):
                r = json.loads(x)
                u = (r["instance"], r["stream"], r["truth"], r["arm"])
                cnt[u] = cnt.get(u, 0) + 1
            done = {u for u, c in cnt.items() if c >= n_prob}
            log(f"resume: {len(done)} finished units")
        jobs = [(s, st, tr, a) for s in seeds for st in streams for tr, arms in ARMS_BY_TRUTH.items() for a in arms
                if (s, st, tr, a) not in done]
        jobs.sort(key=lambda j: (j[3] != "plugin", j[2] != "R0", j[0]))
        progress(2, 4, "cpu: arms", {"jobs": len(jobs)})
        from joblib import Parallel, delayed
        errors, samples, unit_sec = [], [], []
        gen = Parallel(n_jobs=args.workers, backend="loky", return_as="generator_unordered")(
            delayed(run_unit_safe)(s, st, tr, a, n_prob, meta, s == seeds[0] and st == 0) for (s, st, tr, a) in jobs)
        n_done = 0
        with open(res_path, "a") as fh:
            for rr, ss, ee, us in gen:
                n_done += 1
                if ee:
                    errors.append(ee)
                    log(f"ERROR {ee['instance']} s{ee['stream']} {ee['truth']} {ee['arm']}: {ee['error']}\n{ee['tb']}")
                else:
                    for r in rr:
                        fh.write(json.dumps(r, default=float) + "\n")
                    fh.flush()
                    samples += ss
                unit_sec.append(us)
                st_ = {}
                for r in rr:
                    st_[r["status"]] = st_.get(r["status"], 0) + 1
                fcs = sum(r["false_cert"] for r in rr)
                log(f"unit {n_done}/{len(jobs)} inst {us['instance']} {us['truth']} {us['arm']}: {us['sec']:.1f}s "
                    f"rounds {us['rounds']} {st_} false {fcs}")
                progress(2, 4, "cpu: arms", {"jobs_done": n_done, "jobs": len(jobs)})
        rows = [json.loads(x) for x in open(res_path)]
        (out_dir / "samples" / "samples.json").write_text(json.dumps(samples, indent=1, default=float))
        (out_dir / "errors.json").write_text(json.dumps(errors, indent=1))
        progress(3, 4, "summary")
        an = analyse(rows)
        summary.update(an)
        summary["crashes"] = len(errors)
        summary["n_rows"] = len(rows)
        summary["evidence_boundary_ok"] = all(r["env_n_steps"] == r["lr_n_rounds"] for r in rows)
        per = {}
        for us in unit_sec:
            per.setdefault(us["arm"] + "@" + us["truth"], []).append(us["sec"])
        cpu_full = sum(float(np.mean(v)) / n_prob * 15 * 48 * 3 for v in per.values()) if pilot else None
        summary["timing_projection"] = {"unit_sec_mean": {k: float(np.mean(v)) for k, v in per.items()},
                                        "projected_full_cpu_sec": cpu_full,
                                        "projected_full_wall_min_at_4_workers": cpu_full / 4 / 60 if cpu_full else None,
                                        "note": "linear in problems x instances x streams; concurrent run"}
        bp = summary["by_arm"]
        pl = bp.get("plugin@R0", {})
        fcr_pl = pl.get("fcr_jpc")
        pc = {"zero_crashes": len(errors) == 0, "evidence_boundary_ok": summary["evidence_boundary_ok"],
              "static_plugin_fcr_gt_0.10": bool(fcr_pl is not None and fcr_pl > 0.10),
              "conversion_rate_computable": an["shadow_conversion"]["main"]["rate"] is not None,
              "placebo_ratio_computable": an["shadow_conversion"]["placebo"]["rate"] is not None
              and an["shadow_conversion"]["reg_placebo"]["rate"] is not None}
        summary["pass_criteria"] = pc
        mlb = bp.get("mixed_lb@R0", {})
        summary["hf2_gates_preview"] = {
            "detect_ge_0.6_shadow_main": (an["shadow_conversion"]["main"]["rate"] or 0) >= 0.6,
            "detect_ge_0.6_arm_mixed_lb": (an["arm_conversion"]["mixed_lb"]["rate"] or 0) >= 0.6,
            "fcr_mixed_le_2delta": (mlb.get("fcr") is not None and mlb["fcr"] <= 2 * DELTA),
            "fcr_mixed_cp_upper_le_2delta": (mlb.get("fcr_cp95") or [None, None])[1] is not None
            and mlb["fcr_cp95"][1] <= 2 * DELTA,
            "placebo_le_third_full_mixed": (an["placebo_over_true"]["shadow_full_mixed"] is not None
                                            and an["placebo_over_true"]["shadow_full_mixed"] <= 1 / 3),
            "placebo_le_third_reg_only": (an["placebo_over_true"]["shadow_reg_only"] is not None
                                          and an["placebo_over_true"]["shadow_reg_only"] <= 1 / 3),
            "twin_false_alarm_le_delta_mixed_lb": (an["twin_false_alarm"].get("mixed_lb", {}).get("stream_rate")
                                                   is not None and an["twin_false_alarm"]["mixed_lb"]["stream_rate"] <= DELTA),
            "note": "pilot preview on 10 dev instances x 1 stream; gates are evaluated in r2_analysis_aggregate on full"}
        summary["go_no_go"] = "GO" if all(pc.values()) else "NO_GO"
        summary["wall_clock_s"] = time.time() - T0
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
        for k_, v in bp.items():
            log(f"{k_:22s} {v['status_counts']} FCR {v['false_certs']}/{v['certs']} (jpc {v['jpc_false_certs']}/"
                f"{v['jpc_certs']}) conflicts {v['model_conflict_problems']} streams {v['streams_with_conflict']}/"
                f"{v['n_streams']} steps/stream {v['stream_steps_charged_mean']:.0f}")
        log(f"shadow conversion: { {k: (v['converted'], v['n_false']) for k, v in an['shadow_conversion'].items()} }")
        log(f"arm conversion: { {k: (v['converted'], v['n_false']) for k, v in an['arm_conversion'].items()} }")
        log(f"placebo/true: {an['placebo_over_true']}")
        log(f"twin false alarm: { {k: (v['stream_conflict'], v['n_streams']) for k, v in an['twin_false_alarm'].items()} }")
        log(f"done in {time.time() - T0:.0f}s: {summary['go_no_go']} {pc}")
        progress(4, 4, "done", {"go_no_go": summary["go_no_go"]})
        mark_done("success", f"{args.mode}: {summary['go_no_go']} {pc}")
    except Exception as e:  # noqa: BLE001
        log(traceback.format_exc())
        summary["error"] = repr(e)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
        mark_done("failed", repr(e))
        raise


if __name__ == "__main__":
    main()
