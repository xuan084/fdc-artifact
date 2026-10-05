"""hd1_matched_twin_static: HD1 dynamics validity -- static vs dynamic class on CRN-matched twins with dynamics dose g.

Setting (methodology 2.3): truth = R0 kappa_high truth (round-1 generator snapped to G_1) with gamma_i and lam
multiplied by the dose g in {0, 0.5, 1}; g = 0 is exactly the matched twin (gamma = lam = 0, all other coordinates
identical), g = 1 is R0. All doses share the instance seed (CRN: initial data actions, env noise seed, problem stream).

Learner classes (each arm on its own fresh platform copy; each arm re-uses its OWN ledger across the stream):
  static     parameter-count-matched static class on G_1 (13824): load enters neither the outcome logit nor
             retention (g(n) = g_ret(n) = 1), by the exact reparametrisation of run_hf2_static_falsify
             (alpha' = alpha - gamma, tau' = tau - lam). Misspecified for every g > 0; contains the g = 0 twin.
  dyn_G1     dynamic class G_1 (13824). Contains the truth at g in {0, 1}; NOT at g = 0.5 (diagnostic only there).
  dyn_Gext   dynamic class on the extended grid gamma in {0, .375, .75}, lam in {0, .25, .5} (46656). Contains the
             truth at every dose (pre-registered dynamic reference arm for g = 0.5).
Pre-registered dynamic reference: dyn_ref(g) = dyn_Gext at g = 0.5, dyn_G1 at g in {0, 1}.
Learner per arm = JPC (in-class plug-in SeqLRSet, UI threshold log 1/delta, exact minimax certificate, DDA
acquisition, cross-problem ledger reuse). No falsification layer (HF2 covers that); rollouts never enter Theta_t.
c_q = 1 / max J over the dynamic G_1 class (public, data-free, identical for every class and dose).

Readouts: static / dynamic FCR by g (Clopper-Pearson), Jonckheere trend test over g (instance-stream units; null by
within-unit permutation of dose labels, respecting the CRN pairing; plus the unpaired normal approximation),
share of false certificates that are zero-cost ledger re-use, harness-only diagnostic eta_best = min_theta max_pi
|J_theta - J_true| (best-in-class decision misspecification; never used by the learner).

Pilot: dev instances 680-683 x 10 problems x 1 stream (seed 42) x 3 doses. Full: lock range (10000-10047) x 3 streams
x 15 problems (requires dsswm.stats.prereg.assert_locked()). Incremental: finished (instance, stream, dose, class)
units are skipped on restart.

Usage: run_hd1_matched_twin_static.py --mode {pilot,full} [--workers 4] [--smoke] [--tag _x --dev-seeds a-b]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import fcntl  # noqa: E402
import itertools  # noqa: E402
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
LOCK = WS / "plan" / "prereg_lock.json"
TASK = "hd1_matched_twin_static"
EPS, DELTA = 0.02, 0.05
TOP_M = 5
LOG_THR = math.log(1.0 / DELTA)
TMAX_STEP = 3000
PILOT_SEEDS = [680, 681, 682, 683]
PILOT_NPROB = 10
DOSES = (0.0, 0.5, 1.0)
CLASSES = ("static", "dyn_G1", "dyn_Gext")

from dsswm.acquire.nl_kl_dda import IncidenceIndex, class_prob_tables, dda_choose  # noqa: E402
from dsswm.certify.eta_loc import build_plans, j_values  # noqa: E402
from dsswm.certify.minimax_enum import certify_minimax, group_ids, observable_signature, regret_matrix, trichotomy  # noqa: E402
from dsswm.evidence.lr_set import SeqLRSet  # noqa: E402
from dsswm.exact.nl_propagate import NLPropagator  # noqa: E402
from dsswm.models.nl_class import NLClass, NLGrid  # noqa: E402
from dsswm.stats.cp import clopper_pearson  # noqa: E402
from dsswm.streams.generator import DEFAULT_NL_S_GRID, NL_DEFAULTS, generator_hash  # noqa: E402
from dsswm.streams.offgrid import STREAMS, make_offgrid_instance  # noqa: E402

C_KNOWN, RHO_RET, NMAX = NL_DEFAULTS["c"], NL_DEFAULTS["rho_ret"], NL_DEFAULTS["nmax"]
G1 = DEFAULT_NL_S_GRID
GEXT = NLGrid(L=2, R=2, alpha=G1.alpha, gamma=(0.0, 0.375, 0.75), tauL=G1.tauL, beta=G1.beta, tauR=G1.tauR,
              psi=G1.psi, lam=(0.0, 0.25, 0.5))
JT_DIR = {"static": CACHE / "jtables_static_f1", "dyn_G1": CACHE / "jtables_f1", "dyn_Gext": CACHE / "jtables_ext_hd1"}
GRID_OF = {"static": G1, "dyn_G1": G1, "dyn_Gext": GEXT}


def dyn_ref_class(g: float) -> str:
    return "dyn_Gext" if abs(g - 0.5) < 1e-12 else "dyn_G1"


# ------------------------------------------------------------------------------------ class params
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


def class_np_params(cls: str, ncl: NLClass) -> dict:
    return static_np_params(ncl.np_params) if cls == "static" else ncl.np_params


def class_torch_params(cls: str, ncl: NLClass, device, dtype=torch.float64) -> dict:
    return {k: torch.as_tensor(v, device=device, dtype=dtype) for k, v in class_np_params(cls, ncl).items()}


def check_static_equivalence(prop, ncl, device):
    P = {k: v.to(device) for k, v in ncl.torch_params().items()}
    P["g"] = torch.ones(NMAX + 1, dtype=torch.float64, device=device)
    P["g_ret"] = torch.ones(NMAX + 1, dtype=torch.float64, device=device)
    a = prop.tables(P)[0]
    b = prop.tables(class_torch_params("static", ncl, device))[0]
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


def make_inst(seed, g, stream):
    return make_offgrid_instance(seed, "R0", stream=stream, dose=float(g))


def truth_in_class(truth: dict, cls: str) -> bool:
    """Harness-side membership check (exact grid membership of the true parameter vector)."""
    if cls == "static":
        return bool(np.allclose(truth["gamma"], 0) and abs(truth["lam"]) < 1e-12)
    gr = GRID_OF[cls]

    def on(x, vals):
        return bool(np.all(np.min(np.abs(np.asarray(x, float)[..., None] - np.asarray(vals, float)), -1) < 1e-9))
    return (on(truth["alpha"], gr.alpha) and on(truth["gamma"], gr.gamma) and on(truth["tauL"], gr.tauL)
            and on(truth["beta"], gr.beta) and on(truth["tauR"], gr.tauR) and on([truth["psi"]], gr.psi)
            and on([truth["lam"]], gr.lam))


# ------------------------------------------------------------------------------------ phase 1: tables (GPU)
def precompute(seeds, n_prob, streams, dev, log):
    """Class J tables (truth-free, cached), public c_q (dynamic G_1 max), harness J_true per dose (exact propagator)
    and harness-only best-in-class misspecification eta_best per (class, dose, problem)."""
    t0 = time.perf_counter()
    inst0 = make_inst(seeds[0], 1.0, 0)
    prop = NLPropagator(2, 2, NMAX, inst0.env.aspace, C_KNOWN, RHO_RET, device=dev)
    ncl1 = NLClass(G1, device=dev)
    nclx = NLClass(GEXT, device=dev)
    assert ncl1.B == 13824 and nclx.B == 46656, (ncl1.B, nclx.B)
    meta = {"cq": {}, "jtrue": {str(g): {} for g in DOSES}, "eta_best": {}, "sec": {},
            "static_equiv_maxdiff": check_static_equivalence(prop, ncl1, dev), "truth_in_class": {}}
    log(f"static reparametrisation vs g=g_ret=1 tables: max |diff| = {meta['static_equiv_maxdiff']:.2e}")
    Pc = {"static": class_torch_params("static", ncl1, dev), "dyn_G1": ncl1.torch_params(),
          "dyn_Gext": nclx.torch_params()}
    for d in JT_DIR.values():
        d.mkdir(parents=True, exist_ok=True)
    tnew = {c: [] for c in CLASSES}
    probs_needed, truths = {}, {}
    for s in seeds:
        for g in DOSES:
            inst = make_inst(s, g, 0)                        # stream 0 = identity order
            truths[(s, g)] = inst.truth
            for cls in CLASSES:
                meta["truth_in_class"][f"{s}|{g}|{cls}"] = truth_in_class(inst.truth, cls)
            probs = inst.problems[: (n_prob if streams == [0] else len(inst.problems))]
            for q in probs:
                probs_needed.setdefault(q.pid, (q, s))
    with open(CACHE / ".hd1_jt.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        for pid, (q, _) in probs_needed.items():
            for cls in CLASSES:
                p = JT_DIR[cls] / f"{pid}_raw.npy"
                if not p.exists():
                    t1 = time.perf_counter()
                    atomic_save(p, prop.j_table(Pc[cls], q.policies, q.loads0, q.engaged0, q.H, U1(q)))
                    tnew[cls].append(time.perf_counter() - t1)
        fcntl.flock(lk, fcntl.LOCK_UN)
    for pid, (q, s) in probs_needed.items():
        c = 1.0 / float(np.load(JT_DIR["dyn_G1"] / f"{pid}_raw.npy").max())
        meta["cq"][pid] = c
        plans = build_plans(prop, q)
        V = np.stack([vstar_of(truths[(s, g)]) for g in DOSES])
        jj = j_values(prop, plans, V, q.utility.w, q.utility.w_ret) * c
        for gi, g in enumerate(DOSES):
            meta["jtrue"][str(g)][pid] = jj[gi]
        for cls in CLASSES:
            J = np.load(JT_DIR[cls] / f"{pid}_raw.npy") * c
            for gi, g in enumerate(DOSES):
                meta["eta_best"][f"{pid}|{g}|{cls}"] = float(np.abs(J - jj[gi][None]).max(1).min())
    meta["sec"] = {"jtable_new_mean": {c: (float(np.mean(v)) if v else None) for c, v in tnew.items()},
                   "n_new": {c: len(v) for c, v in tnew.items()}, "total_s": time.perf_counter() - t0}
    if torch.cuda.is_available():
        meta["gpu_peak_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20
        meta["gpu_name"] = torch.cuda.get_device_name(0)
        meta["vram_total_mb"] = torch.cuda.get_device_properties(0).total_memory / 2 ** 20
    del prop, ncl1, nclx, Pc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return meta


# ------------------------------------------------------------------------------------ phase 2: learners (CPU)
_CTX: dict = {}


def ctx(cls, aspace):
    if cls not in _CTX:
        torch.set_num_threads(1)
        ncl = NLClass(GRID_OF[cls], device="cpu")
        prop = NLPropagator(2, 2, NMAX, aspace, C_KNOWN, RHO_RET, device="cpu")
        LT = prop.tables(class_torch_params(cls, ncl, "cpu"))[0]
        npp = class_np_params(cls, ncl)
        py, pe = class_prob_tables(npp, C_KNOWN, NMAX, aspace.nb)
        gid = group_ids(observable_signature(py, pe, max_level=1))
        _CTX[cls] = dict(B=ncl.B, prop=prop, LT=LT, LT_np=LT.numpy(), py=py, pe=pe, gid=gid,
                         all_single=len(np.unique(gid)) == ncl.B, inc=IncidenceIndex(prop.codec, aspace, NMAX),
                         legal=np.arange(aspace.n))
    return _CTX[cls]


def run_unit(seed, stream, g, cls, n_prob, meta, want_samples):
    """One (instance, stream, dose, class): JPC with ledger reuse along the problem stream."""
    torch.set_num_threads(1)
    t_unit = time.time()
    noise = STREAMS[stream][0]
    inst = make_inst(seed, g, stream)
    C = ctx(cls, inst.env.aspace)
    prop = C["prop"]
    problems = inst.problems[:n_prob] if stream == 0 else [q for q in inst.problems if q.pid in meta["cq"]][:n_prob]
    env = inst.env
    h = env.handle()
    lr = SeqLRSet(prop, C["LT"], DELTA)
    n_led = 0
    for o in inst.init_obs:
        lr.update(o)
        n_led += 1
    rng = np.random.default_rng([seed, noise, 303, CLASSES.index(cls), int(round(g * 10))])
    in_class = truth_in_class(inst.truth, cls)
    rows, samples = [], []
    for k, q in enumerate(problems):
        t0 = time.perf_counter()
        c = meta["cq"][q.pid]
        J = np.load(JT_DIR[cls] / f"{q.pid}_raw.npy").astype(float) * c
        Reg = regret_matrix(J)
        Jt = np.asarray(meta["jtrue"][str(g)][q.pid])
        steps, status, traj = 0, None, []
        while True:
            mask = lr.mask().numpy()
            cert = certify_minimax(Reg, mask, EPS, TOP_M)
            sv = cert["status"].value
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
            kh = lr.mle()
            blk = cert["blocking"] or [int(i) for i in np.flatnonzero(mask)[:TOP_M]]
            margins = LOG_THR - lr.log_ratio().numpy()[blk]
            code = prop.codec.encode(*h.observable_state())
            a, info = dda_choose(code, C["py"][kh:kh + 1], C["pe"][kh:kh + 1], C["LT_np"][kh], C["py"][blk],
                                 C["pe"][blk], margins, C["inc"], prop, C["legal"], rng)
            obs = h.step(a)
            lr.update(obs)
            n_led += 1
            steps += 1
            if want_samples and len(traj) < 30:
                traj.append({"step": steps, "a": int(a), "set": int(mask.sum()), "r_bar": round(float(cert["r_bar"]), 4),
                             "mle": int(kh), "mode": info["mode"]})
        assert env.n_steps == n_led == lr.n_rounds, "step accounting mismatch (rollout/evidence boundary)"
        mask = lr.mask().numpy()
        pi = cert["pi"] if status == "CERTIFIED" else None
        certified = status == "CERTIFIED"
        tr = float(Jt.max() - Jt[pi]) if pi is not None else None
        charged = steps if certified else max(steps, TMAX_STEP)
        r = {"kind": "hd1", "instance": seed, "stream": stream, "noise_seed": noise, "dose": float(g), "cls": cls,
             "method": f"{cls}@g{g}", "truth_in_class": bool(in_class), "problem": q.pid,
             "gen_index": int(q.meta.get("gen_index", -1)), "k": k, "status": status, "certified": certified,
             "steps_consumed_stepwise": int(steps), "new_env_steps": int(charged), "censored": not certified,
             "certified_policy": pi, "true_regret": tr, "false_cert": bool(certified and tr is not None and tr > EPS),
             "zero_cost": bool(certified and steps == 0), "set_size": int(mask.sum()), "class_size": int(C["B"]),
             "r_bar": float(cert["r_bar"]) if np.isfinite(cert["r_bar"]) else None,
             "eta_best": meta["eta_best"][f"{q.pid}|{g}|{cls}"], "ledger_rounds": int(n_led),
             "env_n_steps": int(env.n_steps), "lr_n_rounds": int(lr.n_rounds),
             "true_gap_top2": float(np.sort(Jt)[-1] - np.sort(Jt)[-2]), "H": q.H, "n_policies": len(q.policies),
             "eps": EPS, "delta": DELTA, "wall_clock_s": time.perf_counter() - t0, "rollouts": 0}
        rows.append(r)
        if want_samples and (k < 2 or r["false_cert"]) and len(samples) < 4:
            samples.append({**r, "J_true": Jt.round(4).tolist(), "J_certified_class_max_regret": None,
                            "trajectory_head": traj, "truth_params": inst.truth})
    return rows, samples, None, {"instance": seed, "stream": stream, "dose": g, "cls": cls,
                                 "sec": time.time() - t_unit, "rounds": n_led}


def run_unit_safe(*a):
    try:
        return run_unit(*a)
    except Exception as e:  # noqa: BLE001
        seed, stream, g, cls = a[:4]
        return [], [], {"instance": seed, "stream": stream, "dose": g, "cls": cls, "error": repr(e),
                        "tb": traceback.format_exc()}, {"instance": seed, "stream": stream, "dose": g, "cls": cls,
                                                        "sec": 0.0, "rounds": 0}


# ------------------------------------------------------------------------------------ analysis
def cp(k, n):
    if n == 0:
        return [None, None]
    lo, hi = clopper_pearson(k, n, 0.05)
    return [float(lo), float(hi)]


def cell_summary(rows):
    n = len(rows)
    cert = [r for r in rows if r["certified"]]
    fc = [r for r in cert if r["false_cert"]]
    sc = {}
    for r in rows:
        sc[r["status"]] = sc.get(r["status"], 0) + 1
    streams = {}
    for r in rows:
        streams.setdefault((r["instance"], r["stream"]), []).append(r)
    fc_zero = sum(r["zero_cost"] for r in fc)
    return {"n": n, "status_counts": sc, "completion": len(cert) / n if n else None, "certs": len(cert),
            "false_certs": len(fc), "fcr": len(fc) / len(cert) if cert else None, "fcr_cp95": cp(len(fc), len(cert)),
            "zero_cost_rate": sum(r["zero_cost"] for r in rows) / n if n else None,
            "false_certs_zero_cost": fc_zero, "reuse_share_of_false_certs": fc_zero / len(fc) if fc else None,
            "reuse_share_cp95": cp(fc_zero, len(fc)),
            "fcr_among_zero_cost": (fc_zero / max(1, sum(r["zero_cost"] for r in cert))) if cert else None,
            "fcr_among_paid": ((len(fc) - fc_zero) / max(1, sum(not r["zero_cost"] for r in cert))) if cert else None,
            "true_regret_mean_false": float(np.mean([r["true_regret"] for r in fc])) if fc else None,
            "eta_best_mean": float(np.mean([r["eta_best"] for r in rows])) if rows else None,
            "eta_best_frac_gt_eps": float(np.mean([r["eta_best"] > EPS for r in rows])) if rows else None,
            "truth_in_class_all": all(r["truth_in_class"] for r in rows),
            "stream_steps_mean": float(np.mean([sum(x["new_env_steps"] for x in v) for v in streams.values()])),
            "stepwise_steps_mean_per_problem": float(np.mean([r["steps_consumed_stepwise"] for r in rows])),
            "n_streams": len(streams), "wall_clock_per_problem_s": float(np.mean([r["wall_clock_s"] for r in rows]))}


def jt_stat(groups):
    """Jonckheere-Terpstra statistic for groups ordered by increasing dose (ties count 1/2)."""
    s = 0.0
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            a = np.asarray(groups[i], float)[:, None]
            b = np.asarray(groups[j], float)[None, :]
            s += float((a < b).sum() + 0.5 * (a == b).sum())
    return s


def jonckheere(unit_vals: dict, n_perm=10000, seed=42):
    """unit_vals[unit] = [v(g0), v(g.5), v(g1)] (CRN-paired units). Returns JT statistic, one-sided p (increasing)
    by within-unit permutation of dose labels (paired-aware; exact if <= 2e5 permutations), and the unpaired normal
    approximation p (ties-uncorrected)."""
    units = [u for u, v in unit_vals.items() if all(x is not None for x in v)]
    if len(units) < 2:
        return {"n_units": len(units), "jt": None, "p_perm_within_unit": None, "p_normal_unpaired": None}
    M = np.array([unit_vals[u] for u in units], float)
    k = M.shape[1]
    obs = jt_stat([M[:, j] for j in range(k)])
    perms = list(itertools.permutations(range(k)))
    n_exact = len(perms) ** len(units)
    if n_exact <= 200000:
        ge, tot = 0, 0
        for combo in itertools.product(range(len(perms)), repeat=len(units)):
            P = np.stack([M[i, list(perms[c])] for i, c in enumerate(combo)])
            ge += jt_stat([P[:, j] for j in range(k)]) >= obs - 1e-12
            tot += 1
        p_perm, how = ge / tot, "exact"
    else:
        rng = np.random.default_rng(seed)
        ge = 0
        for _ in range(n_perm):
            P = np.stack([row[rng.permutation(k)] for row in M])
            ge += jt_stat([P[:, j] for j in range(k)]) >= obs - 1e-12
        p_perm, how = (ge + 1) / (n_perm + 1), f"mc{n_perm}"
    n = [M.shape[0]] * k
    N = sum(n)
    mu = (N ** 2 - sum(x ** 2 for x in n)) / 4
    var = (N ** 2 * (2 * N + 3) - sum(x ** 2 * (2 * x + 3) for x in n)) / 72
    from scipy.stats import norm
    p_norm = float(1 - norm.cdf((obs - mu) / math.sqrt(var))) if var > 0 else None
    return {"n_units": len(units), "jt": obs, "jt_null_mean": mu, "p_perm_within_unit": float(p_perm),
            "perm_method": how, "p_normal_unpaired": p_norm,
            "unit_means_by_g": [float(x) for x in M.mean(0)]}


def analyse(rows):
    out = {"by_cell": {}}
    for cls in CLASSES:
        for g in DOSES:
            rr = [r for r in rows if r["cls"] == cls and r["dose"] == g]
            if rr:
                out["by_cell"][f"{cls}@g{g}"] = cell_summary(rr)
    dref = {}
    for g in DOSES:
        rr = [r for r in rows if r["cls"] == dyn_ref_class(g) and r["dose"] == g]
        if rr:
            dref[f"g{g}"] = {"class": dyn_ref_class(g), **cell_summary(rr)}
    out["dyn_ref_by_g"] = dref
    out["static_fcr_by_g"] = {f"g{g}": out["by_cell"].get(f"static@g{g}", {}).get("fcr") for g in DOSES}
    out["static_fcr_cp95_by_g"] = {f"g{g}": out["by_cell"].get(f"static@g{g}", {}).get("fcr_cp95") for g in DOSES}
    out["dynamic_fcr_by_g"] = {f"g{g}": dref.get(f"g{g}", {}).get("fcr") for g in DOSES}
    out["dynamic_fcr_cp95_by_g"] = {f"g{g}": dref.get(f"g{g}", {}).get("fcr_cp95") for g in DOSES}
    # Jonckheere on (instance, stream) units: per-unit FCR at each dose
    jt = {}
    for name, picker in {"static": lambda g: "static", "dyn_ref": dyn_ref_class, "dyn_Gext": lambda g: "dyn_Gext",
                         "dyn_G1": lambda g: "dyn_G1"}.items():
        uv = {}
        for (inst, st) in sorted({(r["instance"], r["stream"]) for r in rows}):
            vals = []
            for g in DOSES:
                cc = [r for r in rows if r["instance"] == inst and r["stream"] == st and r["dose"] == g
                      and r["cls"] == picker(g) and r["certified"]]
                vals.append(sum(r["false_cert"] for r in cc) / len(cc) if cc else None)
            uv[f"{inst}|{st}"] = vals
        jt[name] = {**jonckheere(uv), "unit_fcr": uv}
    out["jonckheere"] = jt
    out["jonckheere_p"] = jt["static"]["p_perm_within_unit"]
    # zero-cost ledger reuse share of false certificates (static, pooled over g > 0)
    fc = [r for r in rows if r["cls"] == "static" and r["false_cert"]]
    out["reuse_share_of_false_certs"] = {
        "static_pooled": sum(r["zero_cost"] for r in fc) / len(fc) if fc else None,
        "static_pooled_n": len(fc), "static_pooled_cp95": cp(sum(r["zero_cost"] for r in fc), len(fc)),
        "by_g": {f"g{g}": out["by_cell"].get(f"static@g{g}", {}).get("reuse_share_of_false_certs") for g in DOSES}}
    # false certificates by problem index k (reuse grows along the stream)
    byk = {}
    for r in rows:
        if r["cls"] == "static" and r["certified"] and r["dose"] > 0:
            b = byk.setdefault(r["k"], [0, 0])
            b[0] += r["false_cert"]
            b[1] += 1
    out["static_fcr_by_k_gpos"] = {str(k): {"false": v[0], "certs": v[1]} for k, v in sorted(byk.items())}
    # eta_best (harness diagnostic) vs false certification, static class
    bins = [(0, EPS / 2), (EPS / 2, EPS), (EPS, 2 * EPS), (2 * EPS, 1e9)]
    eb = []
    for lo, hi in bins:
        cc = [r for r in rows if r["cls"] == "static" and r["certified"] and lo <= r["eta_best"] < hi]
        k_ = sum(r["false_cert"] for r in cc)
        eb.append({"bin": [lo, hi if hi < 1e8 else None], "certs": len(cc), "false": k_,
                   "fcr": k_ / len(cc) if cc else None, "cp95": cp(k_, len(cc))})
    out["static_fcr_by_eta_best"] = eb
    # paired cost: static vs dyn_ref stream steps by dose (per unit)
    cost = {}
    for g in DOSES:
        rat = []
        for (inst, st) in sorted({(r["instance"], r["stream"]) for r in rows}):
            a = sum(r["new_env_steps"] for r in rows if r["instance"] == inst and r["stream"] == st and r["dose"] == g
                    and r["cls"] == "static")
            b = sum(r["new_env_steps"] for r in rows if r["instance"] == inst and r["stream"] == st and r["dose"] == g
                    and r["cls"] == dyn_ref_class(g))
            if a > 0 and b > 0:
                rat.append(math.log(a / b))
        cost[f"g{g}"] = {"mean_log_static_over_dynref": float(np.mean(rat)) if rat else None, "n_units": len(rat)}
    out["stream_cost_static_vs_dynref"] = cost
    return out


# ------------------------------------------------------------------------------------ plotting
def plot(summary, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return None
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    xs = list(DOSES)
    series = [("static (G_1, load-free)", "static", lambda g: "static", "#c0392b", "o"),
              ("dynamic reference", "dyn_ref", dyn_ref_class, "#2c7fb8", "s"),
              ("dynamic G_ext (all g)", "dyn_Gext", lambda g: "dyn_Gext", "#7fcdbb", "^")]
    for lab, _, pick, col, mk in series:
        ys, lo, hi = [], [], []
        for g in xs:
            c = summary["by_cell"].get(f"{pick(g)}@g{g}")
            if not c or c["fcr"] is None:
                ys.append(np.nan); lo.append(0); hi.append(0)  # noqa: E702
                continue
            ys.append(c["fcr"]); lo.append(c["fcr"] - c["fcr_cp95"][0]); hi.append(c["fcr_cp95"][1] - c["fcr"])  # noqa: E702
        ax.errorbar(xs, ys, yerr=[lo, hi], color=col, marker=mk, capsize=3, label=lab, lw=1.6)
    ax.axhline(DELTA, color="grey", ls="--", lw=1, label=r"$\delta$ = 0.05")
    ax.set_xlabel("dynamics dose g (gamma, lambda multiplied by g)")
    ax.set_ylabel("false certification rate (95% CP)")
    ax.set_xticks(xs)
    ax.set_ylim(-0.02, 1.0)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(f"HD1 ({summary['mode']}): FCR vs dynamics dose", fontsize=10)
    fig.tight_layout()
    p = out_dir / "fcr_vs_dose.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return str(p)


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
    summary = {"task_id": TASK, "mode": args.mode, "seed": 42, "eps": EPS, "delta": DELTA,
               "truth": "R0 kappa_high truth with gamma, lam x g; g=0 == matched twin", "doses": list(DOSES),
               "classes": {"static": "G_1 load-free (g = g_ret = 1), |Theta| = 13824",
                           "dyn_G1": "dynamic G_1, |Theta| = 13824", "dyn_Gext": "dynamic extended grid, |Theta| = 46656"},
               "dyn_ref_rule": "dyn_Gext at g = 0.5, dyn_G1 at g in {0, 1} (methodology 2.3)",
               "instances": [seeds[0], seeds[-1]], "n_instances": len(seeds), "n_problems_per_stream": n_prob,
               "streams": streams, "eval_seeds_touched": not pilot, "generator_hash": generator_hash(),
               "concurrent_run": True, "note": "并发运行（4 槽并行，每任务 4 worker）；计时偏高",
               "tmax_stepwise": TMAX_STEP, "lock_status": lock.get("status"),
               "lock_sha256": lock.get("sha256") or lock.get("sha256_provisional")}
    try:
        if args.resummarize:
            rows = [json.loads(x) for x in open(res_path)]
            old = json.loads((out_dir / "summary.json").read_text())
            old.update(analyse(rows))
            old["figure"] = plot(old, out_dir)
            (out_dir / "summary.json").write_text(json.dumps(old, indent=1, default=float))
            return
        log(f"start {TASK} mode={args.mode} device={dev} seeds {seeds[0]}-{seeds[-1]} ({len(seeds)}) x {n_prob} "
            f"problems x streams {streams} x doses {DOSES} x classes {CLASSES}; lock={lock.get('status')}")
        progress(1, 4, "gpu: class J tables + J_true")
        meta = precompute(seeds, n_prob, streams, dev, log)
        summary["table_sec"] = meta["sec"]
        summary["static_equiv_maxdiff"] = meta["static_equiv_maxdiff"]
        summary["gpu_peak_mb"] = meta.get("gpu_peak_mb")
        summary["truth_in_class_by_cell"] = {
            f"{cls}@g{g}": all(meta["truth_in_class"][f"{s}|{g}|{cls}"] for s in seeds) for cls in CLASSES
            for g in DOSES}
        log(f"tables: {meta['sec']}; gpu peak {meta.get('gpu_peak_mb')} MB; truth-in-class {summary['truth_in_class_by_cell']}")
        if not quiet and meta.get("gpu_name"):
            (RES_ROOT / f"{TASK}_gpu_profile.json").write_text(json.dumps({
                "gpu_name": meta["gpu_name"], "vram_total_mb": meta["vram_total_mb"], "max_batch_size": None,
                "vram_used_mb": meta["gpu_peak_mb"], "utilization_pct": 100 * meta["gpu_peak_mb"] / meta["vram_total_mb"],
                "note": "GPU only for truth-free class J tables (<= 46656 x K per problem, one full-class batch) and "
                        "J_true; learners are CPU. Batch probing not applicable (exact propagator, the full class fits "
                        "in one batch far below the 5 GB per-task budget); shared 4090"}))
        assert meta["static_equiv_maxdiff"] < 1e-9, "static reparametrisation mismatch"
        tic = summary["truth_in_class_by_cell"]
        assert tic["static@g0.0"] and tic["dyn_G1@g0.0"] and tic["dyn_G1@g1.0"] and all(
            tic[f"dyn_Gext@g{g}"] for g in DOSES), f"truth-in-class precondition failed: {tic}"
        assert not tic["static@g0.5"] and not tic["static@g1.0"] and not tic["dyn_G1@g0.5"]
        done = set()
        if res_path.exists():
            cnt = {}
            for x in open(res_path):
                r = json.loads(x)
                u = (r["instance"], r["stream"], r["dose"], r["cls"])
                cnt[u] = cnt.get(u, 0) + 1
            done = {u for u, c in cnt.items() if c >= n_prob}
            log(f"resume: {len(done)} finished units")
        jobs = [(s, st, g, cls) for s in seeds for st in streams for g in DOSES for cls in CLASSES
                if (s, st, g, cls) not in done]
        jobs.sort(key=lambda j: (j[3] != "dyn_Gext", j[0], j[2]))     # big class first for load balance
        progress(2, 4, "cpu: arms", {"jobs": len(jobs)})
        from joblib import Parallel, delayed
        errors, samples, unit_sec = [], [], []
        gen = Parallel(n_jobs=args.workers, backend="loky", return_as="generator_unordered")(
            delayed(run_unit_safe)(s, st, g, cls, n_prob, meta, s == seeds[0] and st == 0) for (s, st, g, cls) in jobs)
        n_done = 0
        with open(res_path, "a") as fh:
            for rr, ss, ee, us in gen:
                n_done += 1
                if ee:
                    errors.append(ee)
                    log(f"ERROR {ee['instance']} s{ee['stream']} g{ee['dose']} {ee['cls']}: {ee['error']}\n{ee['tb']}")
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
                log(f"unit {n_done}/{len(jobs)} inst {us['instance']} g{us['dose']} {us['cls']}: {us['sec']:.1f}s "
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
            per.setdefault(f"{us['cls']}@g{us['dose']}", []).append(us["sec"])
        cpu_full = sum(float(np.mean(v)) / n_prob * 15 * 48 * 3 for v in per.values()) if pilot else None
        summary["timing_projection"] = {"unit_sec_mean": {k: float(np.mean(v)) for k, v in per.items()},
                                        "projected_full_cpu_sec": cpu_full,
                                        "projected_full_wall_min_at_4_workers": cpu_full / 4 / 60 if cpu_full else None,
                                        "note": "linear in problems x instances x streams; concurrent run; ledger "
                                                "re-use makes later problems cheaper, so linear scaling is conservative"}
        sf, df = an["static_fcr_by_g"], an["dynamic_fcr_by_g"]
        dref_fc = sum(v["false_certs"] for v in an["dyn_ref_by_g"].values())
        pc = {"zero_crashes": len(errors) == 0, "evidence_boundary_ok": summary["evidence_boundary_ok"],
              "static_fcr_g1_gt_g0": bool(sf["g1.0"] is not None and sf["g0.0"] is not None and sf["g1.0"] > sf["g0.0"]),
              "dynamic_ref_fcr_zero": dref_fc == 0}
        summary["pass_criteria"] = pc
        s1 = an["by_cell"].get("static@g1.0", {})
        s0 = an["by_cell"].get("static@g0.0", {})
        summary["hd1_gates_preview"] = {
            "kappa_high_static_cp_lower_gt_delta": bool((s1.get("fcr_cp95") or [None])[0] is not None
                                                        and s1["fcr_cp95"][0] > DELTA),
            "twin_static_fcr_le_delta": bool(s0.get("fcr") is not None and s0["fcr"] <= DELTA),
            "jonckheere_p_lt_0.05": bool(an["jonckheere_p"] is not None and an["jonckheere_p"] < 0.05),
            "note": "pilot preview on few dev units: the exact within-unit permutation p is coarse (with tied zero FCRs "
                    "at g in {0, 0.5} its floor is far above 1/6^n_units); gates are evaluated in r2_analysis_aggregate "
                    "on full"}
        summary["go_no_go"] = "GO" if all(pc.values()) else "NO_GO"
        summary["wall_clock_s"] = time.time() - T0
        summary["figure"] = plot(summary, out_dir)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
        for k_, v in an["by_cell"].items():
            log(f"{k_:16s} {v['status_counts']} FCR {v['false_certs']}/{v['certs']} cp95 {v['fcr_cp95']} zero-cost "
                f"{v['zero_cost_rate']:.2f} reuse-share-of-false {v['reuse_share_of_false_certs']} steps/stream "
                f"{v['stream_steps_mean']:.0f} eta_best {v['eta_best_mean']:.4f}")
        log(f"static FCR by g {sf}; dynamic-ref FCR by g {df}; JT p(static) {an['jonckheere_p']}")
        log(f"done in {time.time() - T0:.0f}s: {summary['go_no_go']} {pc}")
        progress(4, 4, "done", {"go_no_go": summary["go_no_go"]})
        mark_done("success", f"{args.mode}: {summary['go_no_go']} {pc}; static FCR {sf}; dyn-ref FCR {df}")
    except Exception as e:  # noqa: BLE001
        log(traceback.format_exc())
        summary["error"] = repr(e)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
        mark_done("failed", repr(e))
        raise


if __name__ == "__main__":
    main()
