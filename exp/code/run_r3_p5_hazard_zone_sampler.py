"""r3_p5_hazard_zone_sampler: P5 R1-adv rejection sampler + ex-ante bin-edge freeze (dev seeds 750-760 only).

Harness-side task (creates off-grid ground truth). Methodology 2.4 / 5.3 (B-adv).

Draw unit = (theta*_d, q_j) pair of one dev instance s:
  theta*_d  ~ uniform on the G_1 hull (class box BOX of models.grid_ladder; methodology 2.4 "continuous uniform in the
              convex hull of G_1"), rng [s, 51, d]; the round-1 R1 sub-box (streams.offgrid.R1_BOX) is reported as a
              subset indicator only
  q_j       = gap_quota.candidate_problem(s, j) (round-3 pool generator; class J tables cached in exp/cache/jtables_r3)
Ex-ante quantities (computed BEFORE any learner run, used only for placement):
  (i)  mu_flip (harness): volume fraction of the clipped Voronoi cell C(g) of theta*'s nearest grid point g in which
       the grid-point decision argmax_pi J_g(pi) is eps-wrong (true regret > eps); Monte Carlo with M uniform points
       in C(g), exact propagator at every point. Diagnostics: mu_flip_half (regret > eps/2), mu_flip_arg (argmax
       differs). The g°-caliber of mu_flip needs a design and is therefore NOT ex-ante; the g° caliber is reported on
       the ex-post side (eta_arg at the stopping-time KL projection).
  (ii) Lambda_hat_perp on n0 (learner-side quantity, called by the harness for placement only): LR set of the n0 = 20
       initial rounds (env with theta*_d, noise seed s*1000+42, initial-action rng [s, 12]), theta_hat = LR-set MLE,
       pi_hat = minimax policy on Theta_t, pre-registered pairs (theta_hat runner-up, minimax challenger),
       S1 = max-pair Lambda_hat_perp, S2 = 1/rho*.
Hazard-zone acceptance (pre-registered here for the pilot): pair accepted iff mu_flip >= 1/M (the cell of theta*
  intersects the eps-flip region of the grid decision). Secondary acceptance readouts: point flip
  eta_arg_near(theta*) > eps, mu_flip >= 0.25, R1 sub-box subset.
Ex-ante bins: tertiles of mu_flip (caliber i) and of S1 (caliber ii) over the ACCEPTED pairs (pilot: provisional
  edges; full: frozen edges for r3_prereg_lock, from the extended draws).
Ex-post bins (scoring only): eta_arg / eps in [0, .5), [.5, 1), [1, inf), two flip calibers:
  eta_arg_gcirc = true regret of argmax J_{g°}, g° = KL projection of theta* onto G_1 over the realised design
                  (all ledger rows at the stopping time; exact per-round KL as in hr3 / g0_resolution_gate)
  eta_arg_near  = true regret of argmax J_{nearest grid point} (design-free)
Pilot runs: per instance the first theta* draw (draw order) whose pool supplies >= Q_PER_BIN accepted problems in
  every mu_flip bin; the first Q_PER_BIN per bin (pool order) form one stream (stream 0, noise 42, order rng
  [s, 45, 0]); uncorrected JPC full (G_1, SeqLRSet log 1/delta, minimax certificate, DDA, no inflation, no fallback)
  on all instances, B3 full (reference) on the first N_B3_INST instances. Predictors are written (write-ahead,
  learner side) at the certification time before truth scoring.
Projection for the eval lock: 3x3 occupancy (ex-ante bin x ex-post eta_arg_near bin) over all accepted pairs ->
  instances needed per cell for >= 30 problems at 5 placed problems per ex-ante bin per instance.
Usage: run_r3_p5_hazard_zone_sampler.py --mode {pilot,full} [--workers 4]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import fcntl  # noqa: E402
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
CACHE = WS / "exp" / "cache" / "jtables_r3"
TASK = "r3_p5_hazard_zone_sampler"
EPS, DELTA = 0.02, 0.05
TMAX = 6000
TAU = 3000
N0 = 20
NOISE = 42
POST_EDGES = (0.0, 0.5, 1.0, math.inf)            # eta_arg / eps
POST_LABELS = ("[0,0.5)", "[0.5,1)", "[1,inf)")
ACC_MIN_RATE = 1e-3
MODES = {
    "pilot": {"seeds": list(range(750, 761)), "n_draws": 60, "n_pool": 30, "mc": 128, "q_per_bin": 3,
              "run": True, "n_b3_inst": 4, "planned_min": 12},
    "full": {"seeds": list(range(750, 761)), "n_draws": 200, "n_pool": 60, "mc": 128, "q_per_bin": 5,
             "run": False, "n_b3_inst": 0, "planned_min": 25},
}
MODES["smoke"] = {"seeds": [750, 751], "n_draws": 8, "n_pool": 6, "mc": 32, "q_per_bin": 1, "run": True,
                  "n_b3_inst": 1, "planned_min": 2}
EVAL = {"n_instances": 64, "per_bin_per_instance": 5, "streams": 3, "cell_target": 30}
CODE_FILES = ["run_r3_p5_hazard_zone_sampler.py", "dsswm/mechanism/leverage.py", "dsswm/baselines/switched_nl.py",
              "dsswm/evidence/reuse_switch.py", "dsswm/streams/gap_quota.py", "dsswm/certify/eta_loc.py"]


# ============================================================================================ scheduler protocol
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


def update_gpu_progress(status, start_iso, wall_min, snapshot, planned):
    p = WS / "exp" / "gpu_progress.json"
    lock = WS / "exp" / "gpu_progress.lock"
    with open(lock, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        d = json.loads(p.read_text()) if p.exists() else {"completed": [], "failed": [], "running": {}, "timings": {}}
        key = "completed" if status == "success" else "failed"
        if TASK not in d.setdefault(key, []):
            d[key].append(TASK)
        other = "failed" if key == "completed" else "completed"
        if TASK in d.get(other, []):
            d[other].remove(TASK)
        d.setdefault("running", {}).pop(TASK, None)
        d.setdefault("timings", {})[TASK] = {"planned_min": planned, "actual_min": int(round(wall_min)),
                                             "start_time": start_iso, "end_time": datetime.now().isoformat(),
                                             "config_snapshot": snapshot}
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(d, indent=2))
        tmp.replace(p)
        fcntl.flock(lf, fcntl.LOCK_UN)


def code_sha():
    out = {f: hashlib.sha256((HERE / f).read_bytes()).hexdigest() for f in CODE_FILES}
    out["_combined"] = hashlib.sha256("".join(out[f] for f in CODE_FILES).encode()).hexdigest()
    return out


# ============================================================================================ truth helpers (harness)
def box_axes():
    from dsswm.models.grid_ladder import _box_axes
    from dsswm.streams.generator import DEFAULT_NL_S_GRID
    return np.array(_box_axes(DEFAULT_NL_S_GRID), float)              # (12, 2), vector order of grid_ladder


def draw_theta(seed: int, d: int) -> np.ndarray:
    bx = box_axes()
    rng = np.random.default_rng([int(seed), 51, int(d)])
    return bx[:, 0] + (bx[:, 1] - bx[:, 0]) * rng.random(bx.shape[0])


def in_r1_box(v: np.ndarray) -> bool:
    from dsswm.streams.offgrid import R1_BOX
    L, R = 2, 2
    sl = {"alpha": v[0:L], "gamma": v[L:2 * L], "tau": np.concatenate([v[2 * L:3 * L], v[3 * L + R:3 * L + 2 * R]]),
          "beta": v[3 * L:3 * L + R], "psi": v[-2:-1], "lam": v[-1:]}
    return all(np.all((sl[k] >= lo) & (sl[k] <= hi)) for k, (lo, hi) in R1_BOX.items())


def truth_dict(v: np.ndarray) -> dict:
    return {"alpha": [float(x) for x in v[0:2]], "gamma": [float(x) for x in v[2:4]],
            "tauL": [float(x) for x in v[4:6]], "beta": [float(x) for x in v[6:8]],
            "tauR": [float(x) for x in v[8:10]], "psi": float(v[10]), "lam": float(v[11])}


def pool_problem(seed: int, j: int, aspace):
    """Round-3 pool problem j with class-max normalisation (truth-free, identical to QuotaBuilder.candidate)."""
    from dsswm.streams.gap_quota import candidate_problem
    q = candidate_problem(seed, j, aspace)
    raw = np.load(CACHE / f"{q.pid}_raw.npy")
    c_q = 1.0 / float(raw.max())
    q.utility.c_q = c_q
    q.meta["u_max"] = c_q * (2 * float(q.utility.w.sum()) + q.utility.w_ret * 4)
    q.meta["normalisation"] = "class_max"
    return q, raw * c_q, c_q


def _bin_of(x, edges):
    """Bins [e0,e1), [e1,e2), [e2,inf); edges = (e1, e2) interior tertile edges (values equal to an edge go up)."""
    if x is None or not np.isfinite(x):
        return None
    return int(np.searchsorted(np.asarray(edges, float), x, side="right"))


def post_bin(eta_over_eps):
    for i in range(3):
        if POST_EDGES[i] <= eta_over_eps < POST_EDGES[i + 1]:
            return i
    return 2


# ============================================================================================ phase A (GPU, main)
def phase_a_gpu(cfg, out_dir, log):
    """Class J tables of the pools (cached), true J at every theta*_d, mu_flip by MC in the nearest cell."""
    from dsswm.certify.eta_loc import build_plans, j_values
    from dsswm.exact.nl_propagate import NLPropagator
    from dsswm.models.grid_ladder import cell_boxes, cell_index_of
    from dsswm.models.nl_class import NLClass
    from dsswm.streams.gap_quota import candidate_problem, make_env, nl_r0_truth
    from dsswm.streams.generator import DEFAULT_NL_S_GRID
    from dsswm.streams.utilities import Utility
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    ncl = NLClass(DEFAULT_NL_S_GRID, device=dev)
    ncl_cpu = NLClass(DEFAULT_NL_S_GRID, device="cpu")
    aspace = make_env(cfg["seeds"][0], nl_r0_truth(cfg["seeds"][0]), NOISE).aspace
    prop = NLPropagator(2, 2, 2, aspace, 1.0, 0.3, device=dev)
    params = None
    lo_all, hi_all = cell_boxes(ncl_cpu)
    CACHE.mkdir(parents=True, exist_ok=True)
    M, D, P = cfg["mc"], cfg["n_draws"], cfg["n_pool"]
    t_tab, n_tab, t_mc = 0.0, 0, 0.0
    for s in cfg["seeds"]:
        fn = out_dir / "parts" / f"phaseA_i{s}.npz"
        if fn.exists():
            continue
        V = np.stack([draw_theta(s, d) for d in range(D)])                   # (D, 12)
        star = cell_index_of(V, ncl_cpu)                                     # nearest grid point per draw
        mc_rng = np.random.default_rng([s, 52])
        lo, hi = lo_all[star], hi_all[star]
        Vmc = lo[:, None, :] + (hi - lo)[:, None, :] * mc_rng.random((D, M, V.shape[1]))
        Vmc = Vmc.reshape(D * M, -1)
        mu_flip = np.zeros((D, P)); mu_half = np.zeros((D, P)); mu_arg = np.zeros((D, P))
        eta_near = np.zeros((D, P)); eta_dec_near = np.zeros((D, P)); gap = np.zeros((D, P))
        npol = np.zeros(P, np.int64)
        Jstar_list = [[None] * P for _ in range(D)]
        for j in range(P):
            q = candidate_problem(s, j, aspace)
            path = CACHE / f"{q.pid}_raw.npy"
            if not path.exists():
                if params is None:
                    params = ncl.torch_params()
                t0 = time.perf_counter()
                raw = prop.j_table(params, q.policies, q.loads0, q.engaged0, q.H,
                                   Utility(w=q.utility.w, w_ret=q.utility.w_ret, c_q=1.0))
                t_tab += time.perf_counter() - t0
                n_tab += 1
                tmp = path.with_name(path.name + f".tmp{os.getpid()}.npy")
                np.save(tmp, raw)
                os.replace(tmp, path)
            else:
                raw = np.load(path)
            c = 1.0 / float(raw.max())
            Jc = raw * c
            npol[j] = Jc.shape[1]
            t0 = time.perf_counter()
            plans = build_plans(prop, q)
            Jt = j_values(prop, plans, V, q.utility.w, q.utility.w_ret) * c            # (D, K)
            Jm = (j_values(prop, plans, Vmc, q.utility.w, q.utility.w_ret) * c).reshape(D, M, -1)
            t_mc += time.perf_counter() - t0
            pig = np.argmax(Jc[star], 1)                                              # grid-point decision
            regm = Jm.max(2) - np.take_along_axis(Jm, pig[:, None, None].repeat(M, 1), 2)[..., 0]
            mu_flip[:, j] = (regm > EPS).mean(1)
            mu_half[:, j] = (regm > EPS / 2).mean(1)
            mu_arg[:, j] = (np.argmax(Jm, 2) != pig[:, None]).mean(1)
            eta_near[:, j] = Jt.max(1) - Jt[np.arange(D), pig]
            eta_dec_near[:, j] = np.abs(Jt - Jc[star]).max(1)
            srt = np.sort(Jt, 1)
            gap[:, j] = srt[:, -1] - srt[:, -2]
            for d in range(D):
                Jstar_list[d][j] = Jt[d]
        Kmax = int(npol.max())
        Jpad = np.full((D, P, Kmax), np.nan)
        for d in range(D):
            for j in range(P):
                Jpad[d, j, :npol[j]] = Jstar_list[d][j]
        r1 = np.array([in_r1_box(v) for v in V])
        tmp = fn.with_name(fn.stem + ".tmp.npz")
        np.savez_compressed(tmp, V=V, star=star, mu_flip=mu_flip, mu_half=mu_half, mu_arg=mu_arg, eta_near=eta_near,
                            eta_dec_near=eta_dec_near, gap=gap, npol=npol, Jstar=Jpad, in_r1_box=r1)
        os.replace(tmp, fn)
        log(f"phaseA i{s}: acc(mu_flip>0)={float((mu_flip > 0).mean()):.3f}")
    vram = torch.cuda.max_memory_allocated() / 2 ** 20 if dev == "cuda" else 0.0
    prof = {"gpu_name": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
            "vram_total_mb": torch.cuda.get_device_properties(0).total_memory / 2 ** 20 if dev == "cuda" else 0,
            "max_batch_size": f"j_values chunk 16384 points (MC batch per problem = {D * M} points, one call)",
            "vram_used_mb": vram, "utilization_pct": (100.0 * vram / (torch.cuda.get_device_properties(0).total_memory
                                                                     / 2 ** 20)) if dev == "cuda" else None,
            "note": "GPU only for class J tables + exact J at theta* draws / MC cell points (float64); shared 4090, "
                    "per-task VRAM cap 5 GB (project override) -> utilisation deliberately low; 并发运行"}
    (RES_ROOT / f"{TASK}_gpu_profile.json").write_text(json.dumps(prof))
    del ncl, prop, params
    if dev == "cuda":
        torch.cuda.empty_cache()
    return {"device": dev, "vram_peak_mb": vram, "tables_computed": n_tab, "table_sec": t_tab, "mc_sec": t_mc}


# ============================================================================================ workers
_W: dict = {}


def _ctx():
    if "pub" not in _W:
        torch.set_num_threads(1)
        from dsswm.acquire.nl_kl_dda import IncidenceIndex, class_prob_tables
        from dsswm.baselines.switched_nl import PublicNL
        from dsswm.exact.nl_propagate import NLPropagator
        from dsswm.mechanism.leverage import NLLeverage
        from dsswm.models.nl_class import NLClass
        from dsswm.streams.gap_quota import make_env, nl_r0_truth
        from dsswm.streams.generator import DEFAULT_NL_S_GRID
        ncl = NLClass(DEFAULT_NL_S_GRID, device="cpu")
        aspace = make_env(750, nl_r0_truth(750), NOISE).aspace
        prop = NLPropagator(2, 2, 2, aspace, 1.0, 0.3, device="cpu")
        TH = NLLeverage(prop).theta_matrix(ncl.np_params)
        lev = NLLeverage(prop, theta_grid=TH, eps=EPS)
        pub = PublicNL(prop, ncl, [], [], 1.0, 2, EPS, DELTA)
        py, pe = class_prob_tables(ncl.np_params, 1.0, 2, aspace.nb)
        _W.update(ncl=ncl, aspace=aspace, prop=prop, TH=TH, lev=lev, pub=pub, py=py, pe=pe,
                  inc=IncidenceIndex(prop.codec, aspace, 2))
    return _W


def _env_for(seed, v, noise=NOISE):
    from dsswm.streams.gap_quota import make_env
    from dsswm.streams.generator import _initial_data
    env = make_env(seed, truth_dict(v), noise)
    init = _initial_data(env, N0, np.random.default_rng([seed, 12]))
    return env, init


def job_lambda_n0(seed, n_draws, n_pool, out_dir):
    """Ex-ante caliber (ii): Lambda_hat_perp on the n0 data of every (theta*_d, q_j) pair of one instance."""
    from dsswm.certify.minimax_enum import certify_minimax, regret_matrix
    from dsswm.evidence.lr_set import SeqLRSet
    from dsswm.mechanism.leverage import select_pairs, theta_hat_index
    W = _ctx()
    fn = out_dir / "parts" / f"lambda_i{seed}.json"
    if fn.exists():
        return {"seed": seed, "cached": True}
    t0 = time.perf_counter()
    A = np.load(out_dir / "parts" / f"phaseA_i{seed}.npz")
    V = A["V"]
    pool = [pool_problem(seed, j, W["aspace"]) for j in range(n_pool)]
    Regs = [regret_matrix(J) for _, J, _ in pool]
    LT = W["pub"].LT
    S1 = np.full((n_draws, n_pool), np.nan)
    S2 = np.full((n_draws, n_pool), np.nan)
    rstar = np.full((n_draws, n_pool), np.nan)
    rbar = np.full((n_draws, n_pool), np.nan)
    pihat = np.zeros((n_draws, n_pool), np.int64)
    setsz = np.zeros(n_draws, np.int64)
    errs = []
    for d in range(n_draws):
        try:
            _, init = _env_for(seed, V[d])
            E = SeqLRSet(W["prop"], LT, DELTA)
            for o in init:
                E.update(o)
            mask = E.mask().numpy().astype(bool)
            setsz[d] = int(mask.sum())
            k_hat = theta_hat_index(E.cum.cpu().numpy(), mask)
            for j, (q, J, _) in enumerate(pool):
                cert = certify_minimax(Regs[j], mask, EPS, 5)
                pi = int(cert["pi"])
                pairs = select_pairs(J, mask, k_hat, pi)
                pr = W["lev"].predictors(q, init, W["TH"][k_hat], pairs, float(cert["r_bar"]), EPS, with_rem=False)
                S1[d, j] = pr["S1"]
                S2[d, j] = pr["S2"] if pr["S2"] is not None else np.nan
                rstar[d, j] = pr["rho_star_min"] if pr["rho_star_min"] is not None else np.nan
                rbar[d, j] = float(cert["r_bar"])
                pihat[d, j] = pi
        except Exception as e:  # noqa: BLE001
            errs.append({"d": d, "error": repr(e), "tb": traceback.format_exc()[-2000:]})
    out = {"seed": seed, "S1": S1.tolist(), "S2": np.where(np.isfinite(S2), S2, 1e300).tolist(),
           "rho_star_min": np.where(np.isfinite(rstar), rstar, 1e300).tolist(), "r_bar": rbar.tolist(),
           "pi_hat": pihat.tolist(), "set_size_n0": setsz.tolist(), "errors": errs,
           "sec": time.perf_counter() - t0, "sec_per_pair": (time.perf_counter() - t0) / (n_draws * n_pool)}
    tmp = fn.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out))
    os.replace(tmp, fn)
    return {"seed": seed, "sec": out["sec"], "n_err": len(errs)}


def job_run(seed, method, placement, out_dir):
    """One stream (stream 0) of the placed problems: uncorrected `method` full arm, write-ahead predictors, scoring."""
    from dsswm.baselines.switched_nl import solve
    from dsswm.certify.minimax_enum import certify_minimax
    from dsswm.evidence.reuse_switch import BillingError, ReuseSwitch, score_truncations
    from dsswm.mechanism.leverage import select_pairs, theta_hat_index
    from dsswm.mechanism.predictor_log import PredictorLog
    from dsswm.acquire.nl_kl_dda import kl_features, single_prob_tables
    from dsswm.streams.r3_harness import METHOD_CODE
    W = _ctx()
    tag = f"i{seed}_{method}"
    parts = out_dir / "parts"
    ppath = parts / f"run_{tag}_predictors.jsonl"
    if ppath.exists():
        ppath.unlink()
    plog = PredictorLog(ppath)
    t_job = time.perf_counter()
    v = np.asarray(placement["theta"], float)
    js = placement["problems"]                       # pool indices in stream order
    pool = {j: pool_problem(seed, j, W["aspace"]) for j in js}
    problems = [pool[j][0] for j in js]
    Js = [pool[j][1] for j in js]
    cqs = [pool[j][2] for j in js]
    A = np.load(parts / f"phaseA_i{seed}.npz")
    d = placement["draw"]
    star = int(A["star"][d])
    pub = W["pub"].with_problems(problems, Js)
    env, init = _env_for(seed, v)
    h = env.handle()
    tpy, tpe = single_prob_tables(env.true_params(), 1.0, 2, W["aspace"].nb)
    Fkl = kl_features(tpy, tpe, W["py"], W["pe"])                   # (nf, B) harness only
    sw = ReuseSwitch("full", pub.make_set_factory(method), init)
    rng = np.random.default_rng([int(seed), NOISE, METHOD_CODE[method]])
    rows, samples, errs = [], [], []
    for k, (q, j) in enumerate(zip(problems, js)):
        t0 = time.perf_counter()
        lr = sw.begin_problem(k, q.pid, context={"stream": 0, "kind": "r1adv"})
        billing_ok, berr = True, None
        try:
            res = solve(pub, method, k, sw, lr, h, rng, TMAX)
        except Exception as e:  # noqa: BLE001
            errs.append({"k": k, "error": repr(e), "tb": traceback.format_exc()[-2000:]})
            res = {"status": "CRASH", "pi": None, "steps": sw.new_steps, "extra": {}}
        # ---------------- learner side, before scoring: predictors at the certification time (write-ahead)
        tp = time.perf_counter()
        try:
            mask = lr.mask().numpy().astype(bool)
            k_hat = theta_hat_index(lr.inner.cum.cpu().numpy(), mask)
            pi_hat = res["pi"] if res["pi"] is not None else int(certify_minimax(pub.Reg[k], mask, EPS, 5)["pi"])
            r_bar = float(res["extra"].get("r_bar_end", 0.0))
            pairs = select_pairs(pub.J[k], mask, k_hat, int(pi_hat))
            pred = W["lev"].predictors(q, list(lr.obs), W["TH"][k_hat], pairs, r_bar if np.isfinite(r_bar) else 0.0,
                                       EPS, with_rem=True)
            Jh = np.sort(pub.J[k][k_hat])[::-1]
            plog.write({"instance": seed, "stream": 0, "method": method, "arm": "full", "problem": int(k),
                        "pid": q.pid, "status": res["status"], "pi_hat": int(pi_hat), "theta_hat": int(k_hat),
                        "set_size": int(mask.sum()), "new_steps": int(sw.new_steps), "n_rows_in_set": len(lr.obs),
                        "r_bar_method": r_bar, "gap_hat_over_eps": float((Jh[0] - Jh[1]) / EPS), **pred})
        except Exception as e:  # noqa: BLE001
            errs.append({"k": k, "where": "predictors", "error": repr(e), "tb": traceback.format_exc()[-2000:]})
            pred = {}
        pred_sec = time.perf_counter() - tp
        try:
            acc = sw.end_problem(env.n_steps)
        except BillingError as e:
            billing_ok, berr = False, str(e)
            acc = sw.accounts[-1]
            sw._k = sw._pid = None
            sw._new = 0
        # ---------------- harness scoring (truth)
        K = len(q.policies)
        Jt = A["Jstar"][d, j, :K]
        Jc = Js[k]
        cnt = np.zeros(W["inc"].nf)
        pairs_cnt = Counter((W["prop"].codec.encode(o.loads, o.engaged), int(o.action)) for o in lr.obs)
        for (code, a), n in pairs_cnt.items():
            cnt += n * W["inc"].S[code * W["inc"].nA + a]
        kl = cnt @ Fkl
        gc = int(np.argmin(kl))
        pi = res["pi"]
        cert = res["status"] == "CERTIFIED"
        regret = float(Jt.max() - Jt[pi]) if pi is not None else None
        eta_arg_g = float(Jt.max() - Jt[int(np.argmax(Jc[gc]))])
        eta_arg_n = float(Jt.max() - Jt[int(np.argmax(Jc[star]))])
        srt = np.sort(Jt)
        tg = float(srt[-1] - srt[-2])
        row = {"task": TASK, "layer": "R1-adv", "instance": seed, "stream": 0, "noise_seed": NOISE, "draw": d,
               "problem_index": k, "problem": k, "pool_j": int(j), "pid": q.pid, "method": method, "arm": "full",
               "gap_layer": "tie" if tg < EPS else ("near" if tg < 2 * EPS else "clear"), "true_gap": tg,
               "status": res["status"], "new_env_steps": int(res["steps"]), "replay_steps": int(acc.replay_steps),
               "n_rounds_billed": int(acc.n_rounds_billed), "env_n_steps": int(env.n_steps),
               "billing_ok": billing_ok, "billing_error": berr, "censored": not cert,
               "zero_cost": bool(cert and res["steps"] == 0), "certified_policy": pi, "true_regret": regret,
               "false_cert": bool(cert and regret is not None and regret > EPS),
               "set_size": int(res["extra"].get("set_size", -1)) if res["extra"] else None,
               "eta_dec": float(np.abs(Jt - Jc[gc]).max()), "eta_dec_near": float(np.abs(Jt - Jc[star]).max()),
               "eta_arg": eta_arg_g, "eta_arg_gcirc": eta_arg_g, "eta_arg_near": eta_arg_n,
               "eta_arg_over_eps": eta_arg_g / EPS, "eta_arg_near_over_eps": eta_arg_n / EPS,
               "post_bin_gcirc": post_bin(eta_arg_g / EPS), "post_bin_near": post_bin(eta_arg_n / EPS),
               "gcirc": gc, "nearest": star, "gcirc_eq_nearest": gc == star, "theta_cell_alive": None,
               "mu_flip": placement["mu_flip"][k], "ante_bin_muflip": placement["ante_bin_muflip"][k],
               "lambda_n0": placement["lambda_n0"][k], "ante_bin_lambda": placement["ante_bin_lambda"][k],
               "orth_feasible": None, "cos_overlap": None, "c_q": cqs[k], "eps": EPS, "delta": DELTA, "tmax": TMAX,
               "wall_clock_s": time.perf_counter() - t0, "predictor_sec": pred_sec,
               "lambda_cert": pred.get("S1"), "S2_cert": pred.get("S2"),
               "first_cert_step": res["extra"].get("first_cert_step") if res["extra"] else None}
        row["theta_cell_alive"] = bool(lr.mask().numpy()[star])
        row.update(score_truncations(int(res["steps"]), cert))
        rows.append(row)
        if len(samples) < 3 or (row["false_cert"] and len(samples) < 6):
            samples.append({"row": {kk: row[kk] for kk in ("instance", "method", "pid", "status", "new_env_steps",
                                                           "certified_policy", "true_regret", "false_cert",
                                                           "eta_arg_gcirc", "eta_arg_near", "mu_flip", "lambda_n0",
                                                           "lambda_cert", "gap_layer")},
                            "problem": q.public_dict() if hasattr(q, "public_dict") else q.pid,
                            "J_true": np.round(Jt, 5).tolist(), "J_nearest": np.round(Jc[star], 5).tolist(),
                            "J_gcirc": np.round(Jc[gc], 5).tolist()})
    part = {"tag": tag, "rows": rows, "samples": samples, "errors": errs, "sec": time.perf_counter() - t_job}
    tmp = parts / f"run_{tag}.json.tmp"
    tmp.write_text(json.dumps(part, default=str))
    os.replace(tmp, parts / f"run_{tag}.json")
    return {"tag": tag, "n_rows": len(rows), "n_err": len(errs), "sec": part["sec"]}


def run_pool(fn, args_list, workers, label):
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    out = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(fn, *a): a for a in args_list}
        for i, f in enumerate(as_completed(futs)):
            try:
                r = f.result()
            except Exception as e:  # noqa: BLE001
                r = {"args": str(futs[f][:2]), "fatal": repr(e), "tb": traceback.format_exc()[-2000:]}
            out.append(r)
            progress(i + 1, len(args_list), label, {"last": str(r)[:200]})
            print(label, i + 1, "/", len(args_list), str(r)[:300], flush=True)
    return out


# ============================================================================================ analysis
def cp(k, n, a=0.05):
    from dsswm.stats.cp import clopper_pearson
    return clopper_pearson(k, n, a) if n > 0 else (None, None)


def assemble_draws(cfg, out_dir):
    """All (instance, d, j) pairs with both ex-ante calibers."""
    recs = []
    for s in cfg["seeds"]:
        A = np.load(out_dir / "parts" / f"phaseA_i{s}.npz")
        Lm = json.loads((out_dir / "parts" / f"lambda_i{s}.json").read_text())
        S1 = np.asarray(Lm["S1"], float)
        S2 = np.asarray(Lm["S2"], float)
        D, P = A["mu_flip"].shape
        for d in range(D):
            for j in range(P):
                recs.append({"instance": s, "draw": d, "pool_j": j, "star": int(A["star"][d]),
                             "in_r1_box": bool(A["in_r1_box"][d]), "mu_flip": float(A["mu_flip"][d, j]),
                             "mu_flip_half": float(A["mu_half"][d, j]), "mu_flip_arg": float(A["mu_arg"][d, j]),
                             "eta_arg_near": float(A["eta_near"][d, j]), "eta_dec_near": float(A["eta_dec_near"][d, j]),
                             "true_gap": float(A["gap"][d, j]), "lambda_n0": float(S1[d, j]),
                             "S2_n0": float(S2[d, j]) if S2[d, j] < 1e299 else math.inf,
                             "set_size_n0": int(Lm["set_size_n0"][d]), "pi_hat_n0": int(Lm["pi_hat"][d][j])})
    return recs


def acceptance(recs, M):
    n = len(recs)
    acc = [r for r in recs if r["mu_flip"] >= 1.0 / M]
    def rate(sub, f):
        k = sum(1 for r in sub if f(r))
        lo, hi = cp(k, len(sub))
        return {"n": len(sub), "k": k, "rate": k / len(sub) if sub else None, "cp95": [lo, hi]}
    r1 = [r for r in recs if r["in_r1_box"]]
    out = {"primary_mu_flip_ge_1_over_M": rate(recs, lambda r: r["mu_flip"] >= 1.0 / M),
           "mu_flip_ge_0.25": rate(recs, lambda r: r["mu_flip"] >= 0.25),
           "point_flip_eta_near_gt_eps": rate(recs, lambda r: r["eta_arg_near"] > EPS),
           "point_flip_eta_near_gt_half_eps": rate(recs, lambda r: r["eta_arg_near"] > EPS / 2),
           "mu_flip_half_ge_1_over_M": rate(recs, lambda r: r["mu_flip_half"] >= 1.0 / M),
           "r1_subbox_share_of_draws": len(r1) / n if n else None,
           "r1_subbox_primary": rate(r1, lambda r: r["mu_flip"] >= 1.0 / M) if r1 else None,
           "point_flip_given_accepted": rate(acc, lambda r: r["eta_arg_near"] > EPS) if acc else None,
           "nontie_given_accepted": rate(acc, lambda r: r["true_gap"] >= EPS) if acc else None,
           "per_instance": {}}
    def layers(sub):
        c = Counter("tie" if r["true_gap"] < EPS else ("near" if r["true_gap"] < 2 * EPS else "clear") for r in sub)
        return {k: c.get(k, 0) / len(sub) for k in ("tie", "near", "clear")} if sub else None
    pf = [r for r in recs if r["eta_arg_near"] > EPS]
    out["gap_layer_share"] = {"all_draws": layers(recs), "accepted": layers(acc), "point_flip": layers(pf)}
    for s in sorted({r["instance"] for r in recs}):
        sub = [r for r in recs if r["instance"] == s]
        out["per_instance"][str(s)] = rate(sub, lambda r: r["mu_flip"] >= 1.0 / M)
    return out, acc


def tertile_edges(x):
    x = np.asarray([v for v in x if np.isfinite(v)], float)
    q = np.quantile(x, [1 / 3, 2 / 3]) if len(x) else np.array([np.nan, np.nan])
    return [float(q[0]), float(q[1])]


def occupancy(acc, ante_key, edges, post_key="eta_arg_near"):
    """3x3 counts ex-ante bin x ex-post (design-free nearest caliber) bin over accepted pairs."""
    tab = np.zeros((3, 3), int)
    for r in acc:
        a = _bin_of(r[ante_key], edges)
        if a is None:
            continue
        tab[a, post_bin(r[post_key] / EPS)] += 1
    return tab


def projection(tab):
    """Instances needed per cell for >= 30 problems with 5 placed problems per ex-ante bin per instance."""
    out = []
    for a in range(3):
        n_a = tab[a].sum()
        row = []
        for b in range(3):
            p = tab[a, b] / n_a if n_a else 0.0
            per_inst = EVAL["per_bin_per_instance"] * p
            need = math.ceil(EVAL["cell_target"] / per_inst) if per_inst > 0 else None
            row.append({"p_post_given_ante": p, "expected_problems_per_instance": per_inst,
                        "instances_needed_unique_problems": need,
                        "expected_cell_n_at_64_instances": per_inst * EVAL["n_instances"],
                        "expected_cell_rows_at_64x3_streams": per_inst * EVAL["n_instances"] * EVAL["streams"],
                        "streams_needed_if_rows_counted": (math.ceil(EVAL["cell_target"] / per_inst)
                                                           if per_inst > 0 else None)})
        out.append(row)
    return out


def place(cfg, recs, edges_mu, edges_lam):
    """Per instance: first draw whose pool fills q_per_bin accepted problems in every mu_flip bin."""
    M, Q = cfg["mc"], cfg["q_per_bin"]
    by = defaultdict(list)
    for r in recs:
        by[(r["instance"], r["draw"])].append(r)
    placements, log = {}, []
    for s in cfg["seeds"]:
        best = None
        for d in range(cfg["n_draws"]):
            rr = sorted(by[(s, d)], key=lambda r: r["pool_j"])
            picked = {0: [], 1: [], 2: []}
            for r in rr:
                if r["mu_flip"] < 1.0 / M:
                    continue
                b = _bin_of(r["mu_flip"], edges_mu)
                if len(picked[b]) < Q:
                    picked[b].append(r)
            fill = min(len(v) for v in picked.values())
            if best is None or fill > best[0]:
                best = (fill, d, picked)
            if fill >= Q:
                break
        fill, d, picked = best
        chosen = sorted([r for b in range(3) for r in picked[b]], key=lambda r: r["pool_j"])
        perm = np.random.default_rng([s, 45, 0]).permutation(len(chosen))
        chosen = [chosen[i] for i in perm]
        placements[s] = {"draw": d, "theta": draw_theta(s, d).tolist(), "problems": [r["pool_j"] for r in chosen],
                         "mu_flip": [r["mu_flip"] for r in chosen],
                         "ante_bin_muflip": [_bin_of(r["mu_flip"], edges_mu) for r in chosen],
                         "lambda_n0": [r["lambda_n0"] for r in chosen],
                         "ante_bin_lambda": [_bin_of(r["lambda_n0"], edges_lam) for r in chosen],
                         "eta_arg_near_exante_point": [r["eta_arg_near"] for r in chosen],
                         "fill_per_bin": fill, "draws_scanned": d + 1}
        log.append({"instance": s, "draw": d, "draws_scanned": d + 1, "fill_per_bin": fill,
                    "n_problems": len(chosen)})
    return placements, log


def cell_table(rows, ante_key, post_key, method):
    R = [r for r in rows if r["method"] == method]
    out = []
    for a in range(3):
        for b in range(3):
            rr = [r for r in R if r[ante_key] == a and r[post_key] == b]
            cc = [r for r in rr if r["status"] == "CERTIFIED"]
            k = sum(r["false_cert"] for r in cc)
            lo, hi = cp(k, len(cc))
            out.append({"ante_bin": a, "post_bin": POST_LABELS[b], "n_problems": len(rr), "n_cert": len(cc),
                        "n_false_cert": int(k), "fcr": k / len(cc) if cc else None, "cp95": [lo, hi],
                        "fcr_per_problem": k / len(rr) if rr else None})
    return out


def summarize_runs(rows):
    out = {}
    for m in sorted({r["method"] for r in rows}):
        R = [r for r in rows if r["method"] == m]
        cc = [r for r in R if r["status"] == "CERTIFIED"]
        k = sum(r["false_cert"] for r in cc)
        lo, hi = cp(k, len(cc))
        nt = [r for r in cc if r["gap_layer"] != "tie"]
        knt = sum(r["false_cert"] for r in nt)
        out[m] = {"n": len(R), "status": dict(Counter(r["status"] for r in R)), "n_cert": len(cc),
                  "false_cert": int(k), "fcr": k / len(cc) if cc else None, "cp95": [lo, hi],
                  "fcr_nontie": knt / len(nt) if nt else None, "n_cert_nontie": len(nt),
                  "zero_cost_rate": float(np.mean([r["zero_cost"] for r in R])),
                  "median_new_steps": float(np.median([r["new_env_steps"] for r in R])),
                  "gap_layers": dict(Counter(r["gap_layer"] for r in R)),
                  "post_bin_gcirc_counts": dict(Counter(POST_LABELS[r["post_bin_gcirc"]] for r in R)),
                  "post_bin_near_counts": dict(Counter(POST_LABELS[r["post_bin_near"]] for r in R)),
                  "eta_arg_gcirc_over_eps_quantiles": np.quantile([r["eta_arg_over_eps"] for r in R],
                                                                  [0, .25, .5, .75, .9, 1]).round(3).tolist(),
                  "gcirc_eq_nearest_rate": float(np.mean([r["gcirc_eq_nearest"] for r in R])),
                  "theta_cell_alive_rate": float(np.mean([r["theta_cell_alive"] for r in R])),
                  "median_wall_s": float(np.median([r["wall_clock_s"] for r in R])),
                  "billing_mismatch": int(sum(not r["billing_ok"] for r in R))}
    return out


def make_plot(tab_mu, tab_lam, run_tab, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figs = out_dir / "figures"
    figs.mkdir(exist_ok=True)
    panels = [("ex-ante mu_flip tertile (accepted draws)", tab_mu), ("ex-ante Lambda_hat_perp(n0) tertile", tab_lam)]
    if run_tab is not None:
        panels.append(("JPC runs: mu_flip bin x eta_arg(g°) bin", run_tab))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4))
    for ax, (title, t) in zip(np.atleast_1d(axes), panels):
        t = np.asarray(t)
        im = ax.imshow(t.T, origin="lower", cmap="Blues")
        for a in range(3):
            for b in range(3):
                ax.text(a, b, str(int(t[a, b])), ha="center", va="center",
                        color="white" if t[a, b] > t.max() / 2 else "black", fontsize=10)
        ax.set_xticks(range(3), ["low", "mid", "high"])
        ax.set_yticks(range(3), list(POST_LABELS))
        ax.set_xlabel("ex-ante bin")
        ax.set_ylabel("eta_arg / eps bin")
        ax.set_title(title, fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("P5 dev (seeds 750-760): ex-ante x ex-post bin occupancy", fontsize=10)
    fig.tight_layout()
    fig.savefig(figs / "occupancy_heatmap.png", dpi=130)
    plt.close(fig)


# ============================================================================================ main
def main():
    global RES_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("pilot", "full", "smoke"), default="pilot")
    ap.add_argument("--out", default=None, help="smoke only: output directory (no scheduler files)")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    cfg = MODES[a.mode]
    smoke = a.mode == "smoke"
    out_dir = Path(a.out) if smoke else RES_ROOT / ("pilots" if a.mode == "pilot" else "full") / TASK
    for d in (out_dir, out_dir / "parts", out_dir / "samples"):
        d.mkdir(parents=True, exist_ok=True)
    if smoke:
        RES_ROOT = out_dir
    (RES_ROOT / f"{TASK}.pid").write_text(str(os.getpid()))
    started = datetime.now().isoformat()
    t0 = time.perf_counter()
    timing = {}
    snapshot = {"seeds": f"{cfg['seeds'][0]}-{cfg['seeds'][-1]}", "n_draws_theta": cfg["n_draws"],
                "n_pool": cfg["n_pool"], "mc_per_cell": cfg["mc"], "grid": "G_1 |Theta|=13824", "workers": a.workers,
                "gpu_model": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "gpu_count": 1}
    status = "failed"
    try:
        progress(0, 1, "phaseA_gpu")
        tA = time.perf_counter()
        ga = phase_a_gpu(cfg, out_dir, lambda m: print(m, flush=True))
        timing["phaseA_gpu_sec"] = time.perf_counter() - tA
        tB = time.perf_counter()
        lam_log = run_pool(job_lambda_n0, [(s, cfg["n_draws"], cfg["n_pool"], out_dir) for s in cfg["seeds"]],
                           a.workers, "lambda_n0")
        timing["lambda_n0_sec"] = time.perf_counter() - tB
        recs = assemble_draws(cfg, out_dir)
        with gzip.open(out_dir / "draws.jsonl.gz", "wt") as fh:
            for r in recs:
                fh.write(json.dumps(r) + "\n")
        acc_stats, acc = acceptance(recs, cfg["mc"])
        edges_mu = tertile_edges([r["mu_flip"] for r in acc])
        edges_lam = tertile_edges([r["lambda_n0"] for r in acc])
        edges_s2 = tertile_edges([r["S2_n0"] for r in acc])
        tab_mu = occupancy(acc, "mu_flip", edges_mu)
        tab_lam = occupancy(acc, "lambda_n0", edges_lam)
        proj_mu, proj_lam = projection(tab_mu), projection(tab_lam)
        acc_nt = [r for r in acc if r["true_gap"] >= EPS]
        tab_mu_nt = occupancy(acc_nt, "mu_flip", edges_mu)
        tab_lam_nt = occupancy(acc_nt, "lambda_n0", edges_lam)
        nontie_by_bin = {c: [float(np.mean([r["true_gap"] >= EPS for r in acc if _bin_of(r[k], e) == b]))
                             for b in range(3)] for c, k, e in (("mu_flip", "mu_flip", edges_mu),
                                                                ("lambda_n0", "lambda_n0", edges_lam))}
        lam_errs = []
        for s in cfg["seeds"]:
            lam_errs += json.loads((out_dir / "parts" / f"lambda_i{s}.json").read_text())["errors"]
        # rank agreement of the two ex-ante calibers on accepted pairs
        from scipy.stats import spearmanr
        rho = spearmanr([r["mu_flip"] for r in acc], [r["lambda_n0"] for r in acc]).correlation if len(acc) > 2 else None
        cross = np.zeros((3, 3), int)
        for r in acc:
            cross[_bin_of(r["mu_flip"], edges_mu), _bin_of(r["lambda_n0"], edges_lam)] += 1
        rows, run_errs, run_log, placements, place_log = [], [], [], {}, []
        if cfg["run"]:
            placements, place_log = place(cfg, recs, edges_mu, edges_lam)
            (out_dir / "placements.json").write_text(json.dumps(placements, indent=1))
            jobs = [(s, "B3", placements[s], out_dir) for s in cfg["seeds"][:cfg["n_b3_inst"]]]
            jobs += [(s, "JPC", placements[s], out_dir) for s in cfg["seeds"]]
            tC = time.perf_counter()
            run_log = run_pool(job_run, jobs, a.workers, "runs")
            timing["runs_sec"] = time.perf_counter() - tC
            from dsswm.mechanism.predictor_log import ResultLog, join, load_jsonl
            pred_path, res_path = out_dir / "predictors.jsonl", out_dir / "results.jsonl"
            for p in (pred_path, res_path):
                if p.exists():
                    p.unlink()
            samples = []
            with open(pred_path, "w") as fp:
                for (s, m, _, _) in jobs:
                    pp = out_dir / "parts" / f"run_i{s}_{m}.json"
                    if not pp.exists():
                        run_errs.append({"tag": f"i{s}_{m}", "error": "part missing"})
                        continue
                    part = json.loads(pp.read_text())
                    rows += part["rows"]
                    samples += part["samples"]
                    run_errs += part["errors"]
                    for rec in load_jsonl(out_dir / "parts" / f"run_i{s}_{m}_predictors.jsonl"):
                        fp.write(json.dumps(rec, sort_keys=True) + "\n")
            rlog = ResultLog(res_path, fsync=False)
            for r in rows:
                rlog.write(r)
            try:
                wa = {"n_joined": len(join(pred_path, res_path, check_order=True)), "write_ahead_ok": True}
            except RuntimeError as e:
                wa = {"write_ahead_ok": False, "error": str(e)}
            (out_dir / "samples" / "trajectories.json").write_text(json.dumps(samples[:12], indent=1, default=str))
        else:
            wa = None
        run_summary = summarize_runs(rows) if rows else None
        jpc_rows = [r for r in rows if r["method"] == "JPC"]
        run_tab = None
        if jpc_rows:
            run_tab = np.zeros((3, 3), int)
            for r in jpc_rows:
                run_tab[r["ante_bin_muflip"], r["post_bin_gcirc"]] += 1
        try:
            make_plot(tab_mu, tab_lam, run_tab, out_dir)
        except Exception as e:  # noqa: BLE001
            print("plot failed", e, flush=True)
        n_draws_total = len(recs)
        acc_rate = acc_stats["primary_mu_flip_ge_1_over_M"]["rate"]
        crashes = (len(lam_errs) + len(run_errs) + sum("fatal" in r for r in lam_log + run_log)
                   + sum(r["status"] == "CRASH" for r in rows))
        crit = {"acceptance_ge_0.1pct": acc_rate is not None and acc_rate >= ACC_MIN_RATE,
                "rejection_draws_ge_1e4": n_draws_total >= 10_000,
                "zero_crashes": crashes == 0,
                "every_3x3_cell_nonempty_in_projection_mu_flip": bool((tab_mu > 0).all()),
                "every_3x3_cell_nonempty_in_projection_lambda": bool((tab_lam > 0).all())}
        if cfg["run"]:
            crit["jpc_runs_ge_90"] = len(jpc_rows) >= 90
            crit["b3_runs_ge_30"] = sum(r["method"] == "B3" for r in rows) >= 30
            crit["billing_mismatch_zero"] = all(r["billing_ok"] for r in rows)
            crit["predictor_write_ahead_ok"] = bool(wa and wa["write_ahead_ok"])
        passed = all(crit.values())
        directed = not crit["acceptance_ge_0.1pct"]
        summary = {
            "task": TASK, "mode": a.mode, "started_at": started, "finished_at": datetime.now().isoformat(),
            "note": "开发种子 750–760，pilot 读数不计入证据；分箱边界为 provisional（full 模式冻结）。并发运行，计时偏高。",
            "design": {"seeds": cfg["seeds"], "theta_proposal": "uniform on G_1 hull (grid_ladder.BOX)",
                       "n_theta_draws_per_instance": cfg["n_draws"], "n_pool_per_instance": cfg["n_pool"],
                       "rejection_draws_total": n_draws_total, "mc_points_per_cell": cfg["mc"],
                       "acceptance_rule": f"mu_flip >= 1/{cfg['mc']} (cell of theta* meets the eps-flip region of "
                                          "the nearest-grid-point decision)",
                       "ex_ante_calibers": {"i": "mu_flip (nearest-grid caliber, harness MC)",
                                            "ii": "Lambda_hat_perp S1 on n0 (learner-side, max over 2 pre-registered "
                                                  "pairs, with_rem=False)"},
                       "ex_post_bins_eta_arg_over_eps": list(POST_LABELS),
                       "ex_post_calibers": ["g° (KL projection over realised design at stopping time)",
                                            "nearest grid point (design-free)"],
                       "runs": {"stream": 0, "noise_seed": NOISE, "q_per_bin": cfg["q_per_bin"],
                                "methods": {"JPC": "uncorrected, full arm, all instances",
                                            "B3": f"full arm, first {cfg['n_b3_inst']} instances"},
                                "tmax": TMAX} if cfg["run"] else None,
                       "eps": EPS, "delta": DELTA},
            "phaseA": ga, "timing": timing,
            "acceptance": acc_stats,
            "bin_edges_provisional" if a.mode == "pilot" else "bin_edges_frozen": {
                "mu_flip_tertiles": edges_mu, "lambda_n0_S1_tertiles": edges_lam, "S2_n0_tertiles": edges_s2,
                "n_accepted": len(acc), "rule": "bins [min,e1), [e1,e2), [e2,max]; value == edge goes up"},
            "exante_caliber_agreement": {"spearman_mu_flip_vs_lambda_n0": rho, "cross_bins_mu_x_lambda": cross.tolist()},
            "occupancy_accepted_nearest_post": {"mu_flip": tab_mu.tolist(), "lambda_n0": tab_lam.tolist(),
                                                "rows": "ex-ante bin (low, mid, high)", "cols": list(POST_LABELS)},
            "occupancy_accepted_nontie_nearest_post": {"mu_flip": tab_mu_nt.tolist(), "lambda_n0": tab_lam_nt.tolist(),
                                                       "nontie_share_by_ante_bin": nontie_by_bin},
            "projection_for_lock": {"assumptions": EVAL, "mu_flip": proj_mu, "lambda_n0": proj_lam,
                                    "nontie_only_mu_flip": projection(tab_mu_nt),
                                    "nontie_only_lambda_n0": projection(tab_lam_nt),
                                    "note": "ex-post via design-free eta_arg_near at theta*; g° caliber only from "
                                            "runs. 'instances_needed' counts unique (instance, problem) units"},
            "placement": place_log, "runs": run_summary, "write_ahead": wa,
            "fcr_by_cell_dev": ({"JPC_muflip_x_gcirc": cell_table(rows, "ante_bin_muflip", "post_bin_gcirc", "JPC"),
                                 "JPC_muflip_x_near": cell_table(rows, "ante_bin_muflip", "post_bin_near", "JPC"),
                                 "JPC_lambda_x_gcirc": cell_table(rows, "ante_bin_lambda", "post_bin_gcirc", "JPC"),
                                 "B3_muflip_x_gcirc": cell_table(rows, "ante_bin_muflip", "post_bin_gcirc", "B3")}
                                if rows else None),
            "errors": {"lambda": lam_errs[:10], "runs": run_errs[:10], "n_lambda": len(lam_errs),
                       "n_runs": len(run_errs)},
            "directed_construction_triggered": directed,
            "pass_criteria": crit, "passed": passed, "go_no_go": "GO" if passed else "NO_GO",
            "code_sha256": code_sha(), "wall_clock_s": time.perf_counter() - t0}
        summary["metrics"] = {"acceptance_rate": acc_rate, "ex_post_bin_counts": tab_mu.tolist(),
                              "fcr_by_cell_dev": summary["fcr_by_cell_dev"]}
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
        print(json.dumps(crit), summary["go_no_go"], flush=True)
        progress(1, 1, "done", {"go_no_go": summary["go_no_go"]})
        status = "success" if crit["zero_crashes"] else "failed"      # NO-GO on a soft gate is a result
        mark_done(status, f"{summary['go_no_go']} acc={acc_rate} {crit}")
    except Exception as e:  # noqa: BLE001
        (out_dir / "crash.txt").write_text(traceback.format_exc())
        mark_done("failed", repr(e))
        raise
    finally:
        snapshot["mode"] = a.mode
        if not smoke:
            update_gpu_progress(status, started, (time.perf_counter() - t0) / 60, snapshot,
                                30 if a.mode == "full" else 25)


if __name__ == "__main__":
    main()
