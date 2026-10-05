"""hd3_matched_twin_rect: HD3 corrected H4 contrast -- rect(time) / joint evidence on CRN-matched twins.

Setting (methodology 2.3): kappa_high truth = R0 truth (round-1 generator snapped to G_1, gamma = 0.75, lam = 0.5);
matched twin = the same theta* with every gamma_i and lam set to 0 (all other coordinates identical, same instance
seed -> same initial-data actions, same env noise seed, same problem stream). Both truths lie in the dynamic class
G_1 (|Theta| = 13824), which is the learner class for every arm (asserted).

Fresh-start protocol (isolates the evidence factor): every (instance, twin, stream, problem, method) run starts from
the shared n0 = 20 initial rounds only (no cross-problem ledger reuse) on its own fresh platform copy, and acquires
real env.step() rounds with the pre-registered DDA sampler until its stopping rule fires or T_max = 3000 new rounds
(censored, charged T_max).

Evidence (same LR set Theta_t: plug-in SeqLRSet, UI threshold log 1/delta; rollouts never enter Theta_t):
  joint     exact minimax certificate  R_bar = min_pi max_{theta in Theta_t} [max_k J_theta(k) - J_theta(pi)] <= eps
  rect      B6-sup rect(time): per-step value blocks J_theta(pi) = sum_b J_theta,b(pi) (b = step 0..H-1, plus the
            terminal retention block), block-wise sup over Theta_t taken separately:
               UB(pi) = max_k sum_b max_{theta in Theta_t} [J_theta,b(k) - J_theta,b(pi)],  stop iff min_pi UB <= eps.
            UB >= R_bar always (temporal cancellation is lost); equality iff a single theta attains every block sup.
            DDA alternatives = the block-maximising models of the binding policy (by block contribution).
Sampler: DDA (acquire/nl_kl_dda.py, m = 5, explore 0.05, two-step lookahead), identical code for both arms.

Readouts (instance = cluster; per instance stream totals averaged over streams):
  L_twin = log(S_rect / S_joint)  for twin in {khigh, k0}
  paired D = L_khigh - L_k0, paired instance bootstrap (B = 10000) -> 95% CI; HD3 gate: CI lower > 0.
  Plus: per-problem log((rect+1)/(joint+1)), FCR / completion per arm, design-free temporal-cancellation diagnostic
  kappa_time(Theta) = UB_rect / R_bar evaluated on the same LR set (at n0 and at the joint stopping set).

Pilot: dev instances 684-688 x 10 problems x 1 stream (seed 42) x 2 twins. Full: lock range (10000-10047) x 3 streams
x 15 problems (requires dsswm.stats.prereg.assert_locked()). Incremental: finished (instance, stream, twin, method)
units are skipped on restart.

Usage: run_hd3_matched_twin_rect.py --mode {pilot,full} [--workers 4] [--smoke]
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
LOCK = WS / "plan" / "prereg_lock.json"
TASK = "hd3_matched_twin_rect"
EPS, DELTA = 0.02, 0.05
TOP_M = 5
LOG_THR = math.log(1.0 / DELTA)
TMAX_STEP = 3000
PILOT_SEEDS = [684, 685, 686, 687, 688]
PILOT_NPROB = 10
TWINS = ("khigh", "k0")          # khigh = R0 truth (g = 1); k0 = matched twin (gamma = lam = 0)
METHODS = ("joint", "rect")
CLS = "dyn_G1"
BOOT_B = 10000


from dsswm.acquire.nl_factor_acq import perstep_jtable  # noqa: E402
from dsswm.acquire.nl_kl_dda import dda_choose  # noqa: E402
from dsswm.certify.eta_loc import build_plans, j_values  # noqa: E402
from dsswm.certify.minimax_enum import certify_minimax, regret_matrix, trichotomy  # noqa: E402
from dsswm.evidence.lr_set import SeqLRSet  # noqa: E402
from dsswm.exact.nl_propagate import NLPropagator  # noqa: E402
from dsswm.models.nl_class import NLClass  # noqa: E402
from dsswm.stats.cp import clopper_pearson  # noqa: E402
from dsswm.streams.generator import generator_hash  # noqa: E402
from dsswm.streams.offgrid import STREAMS, make_offgrid_instance  # noqa: E402
import run_hd1_matched_twin_static as HD1  # noqa: E402  (shared class context, J-table cache, helpers)

C_KNOWN, RHO_RET, NMAX = HD1.C_KNOWN, HD1.RHO_RET, HD1.NMAX
G1 = HD1.G1
JT_DIR = HD1.JT_DIR[CLS]                      # dyn G_1 raw J tables (truth-free, shared with HD1/HR1)
JTS_DIR = CACHE / "jtables_perstep_f1"         # dyn G_1 per-step raw tables (B, K, H+1), float32


def make_inst(seed, twin, stream):
    return make_offgrid_instance(seed, "R0", stream=stream, twin=(twin == "k0"))


# ------------------------------------------------------------------------------------ phase 1: tables (GPU)
def precompute(seeds, n_prob, streams, dev, log):
    t0 = time.perf_counter()
    inst0 = make_inst(seeds[0], "khigh", 0)
    prop = NLPropagator(2, 2, NMAX, inst0.env.aspace, C_KNOWN, RHO_RET, device=dev)
    ncl = NLClass(G1, device=dev)
    assert ncl.B == 13824
    P = ncl.torch_params()
    JT_DIR.mkdir(parents=True, exist_ok=True)
    JTS_DIR.mkdir(parents=True, exist_ok=True)
    meta = {"cq": {}, "jtrue": {t: {} for t in TWINS}, "truth_in_class": {}, "truth_sha": {}, "sum_check_maxdiff": 0.0,
            "kappa_true": {}}
    probs_needed, truths = {}, {}
    for s in seeds:
        for tw in TWINS:
            inst = make_inst(s, tw, 0)
            truths[(s, tw)] = inst.truth
            meta["truth_in_class"][f"{s}|{tw}"] = HD1.truth_in_class(inst.truth, CLS)
            meta["truth_sha"][f"{s}|{tw}"] = inst.truth["sha256"]
            probs = inst.problems[: (n_prob if streams == [0] else len(inst.problems))]
            for q in probs:
                probs_needed.setdefault(q.pid, (q, s))
    tnew = []
    with open(CACHE / ".hd3_jt.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        for pid, (q, _) in probs_needed.items():
            p = JT_DIR / f"{pid}_raw.npy"
            pt = JTS_DIR / f"{pid}_rawt.npy"
            if not pt.exists():
                t1 = time.perf_counter()
                raw_t = perstep_jtable(prop, P, q.policies, q.loads0, q.engaged0, q.H, q.utility.w, q.utility.w_ret)
                HD1.atomic_save(pt, raw_t.astype(np.float32))
                if not p.exists():
                    HD1.atomic_save(p, prop.j_table(P, q.policies, q.loads0, q.engaged0, q.H, HD1.U1(q)))
                tnew.append(time.perf_counter() - t1)
        fcntl.flock(lk, fcntl.LOCK_UN)
    for pid, (q, s) in probs_needed.items():
        raw = np.load(JT_DIR / f"{pid}_raw.npy")
        rawt = np.load(JTS_DIR / f"{pid}_rawt.npy").astype(np.float64)
        meta["sum_check_maxdiff"] = max(meta["sum_check_maxdiff"], float(np.abs(rawt.sum(-1) - raw).max()))
        c = 1.0 / float(raw.max())
        meta["cq"][pid] = c
        plans = build_plans(prop, q)
        V = np.stack([HD1.vstar_of(truths[(s, tw)]) for tw in TWINS])
        jj = j_values(prop, plans, V, q.utility.w, q.utility.w_ret) * c
        for i, tw in enumerate(TWINS):
            meta["jtrue"][tw][pid] = jj[i]
    meta["sec"] = {"perstep_new_mean": float(np.mean(tnew)) if tnew else None, "n_new": len(tnew),
                   "total_s": time.perf_counter() - t0}
    if torch.cuda.is_available():
        meta["gpu_peak_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20
        meta["gpu_name"] = torch.cuda.get_device_name(0)
        meta["vram_total_mb"] = torch.cuda.get_device_properties(0).total_memory / 2 ** 20
    del prop, ncl, P
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return meta


# ------------------------------------------------------------------------------------ rect(time) certificate
def rect_cert(Jt: np.ndarray, mask: np.ndarray, kh: int):
    """B6-sup rect(time) on the LR set. Jt (B, K, H+1) normalised per-step blocks.
    UB(pi) = max_k sum_b max_{theta in mask} [Jt(theta,k,b) - Jt(theta,pi,b)];  pi_rect = argmin UB."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return {"status": "MODEL_CONFLICT", "pi": None, "ub": float("inf"), "blocking": [], "UB_all": None}
    X = Jt[idx]                                                  # (n, K, H+1)
    K = X.shape[1]
    UB = np.empty(K)
    best = None
    for j in range(K):
        D = X - X[:, j:j + 1, :]                                 # (n, K, H+1)
        sup = D.max(0)                                           # (K, H+1)
        tot = sup.sum(1)
        UB[j] = max(float(tot.max()), 0.0)
        if best is None or UB[j] < best[0] - 1e-15:
            best = (UB[j], j, D, sup, tot)
    ub, pi, D, sup, tot = best
    kb = int(np.argmax(tot))
    arg = D[:, kb, :].argmax(0)                                  # (H+1,) block-maximising models
    con = sup[kb]
    order = np.argsort(-con)
    blk = []
    for o in order:
        th = int(idx[arg[o]])
        if con[o] > 0 and th != kh and th not in blk:
            blk.append(th)
        if len(blk) >= TOP_M:
            break
    return {"status": "CERTIFIED" if ub <= EPS else "NEED_DATA", "pi": int(pi), "ub": float(ub), "blocking": blk,
            "UB_all": UB, "binding": kb}


def kappa_time(Jt, Reg, mask):
    """Design-free temporal-cancellation diagnostic on one LR set: min_pi UB_rect / R_bar (both at their own pi)."""
    if not mask.any():
        return None, None, None
    rb = float(max(Reg[mask].max(0).min(), 0.0))
    ub = rect_cert(Jt, mask, -1)["ub"]
    return (ub / rb if rb > 1e-9 else None), rb, ub


# ------------------------------------------------------------------------------------ phase 2: learners (CPU)
def run_unit(seed, stream, twin, method, n_prob, meta, want_samples):
    """One (instance, stream, twin, method): fresh-start certification of each problem from the shared n0 data."""
    torch.set_num_threads(1)
    t_unit = time.time()
    noise = STREAMS[stream][0]
    inst0 = make_inst(seed, twin, stream)
    C = HD1.ctx(CLS, inst0.env.aspace)
    prop = C["prop"]
    problems = inst0.problems[:n_prob] if stream == 0 else [q for q in inst0.problems if q.pid in meta["cq"]][:n_prob]
    in_class = HD1.truth_in_class(inst0.truth, CLS)
    rows, samples = [], []
    for k, q in enumerate(problems):
        t0 = time.perf_counter()
        inst = make_inst(seed, twin, stream)                     # fresh platform copy (CRN: same seeds)
        assert inst.problems[k].pid == q.pid
        env = inst.env
        h = env.handle()
        lr = SeqLRSet(prop, C["LT"], DELTA)
        for o in inst.init_obs:
            lr.update(o)
        n0 = env.n_steps
        rng = np.random.default_rng([seed, noise, 404, METHODS.index(method), TWINS.index(twin), k])
        c = meta["cq"][q.pid]
        J = np.load(JT_DIR / f"{q.pid}_raw.npy").astype(float) * c
        Jt = np.load(JTS_DIR / f"{q.pid}_rawt.npy").astype(np.float64) * c
        Reg = regret_matrix(J)
        Jtrue = np.asarray(meta["jtrue"][twin][q.pid])
        mask0 = lr.mask().numpy()
        k0_ratio, rb0, ub0 = kappa_time(Jt, Reg, mask0)
        steps, status, traj = 0, None, []
        while True:
            mask = lr.mask().numpy()
            jc = certify_minimax(Reg, mask, EPS, TOP_M)
            if method == "joint":
                cert = {"status": jc["status"].value, "pi": jc["pi"], "ub": jc["r_bar"], "blocking": jc["blocking"]}
            else:
                cert = rect_cert(Jt, mask, lr.mle())
            sv = cert["status"]
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
            blk = list(cert["blocking"])
            if method == "rect" and not blk:
                blk = [b for b in jc["blocking"] if b != kh]
            if not blk:
                blk = [int(i) for i in np.flatnonzero(mask) if int(i) != kh][:TOP_M] or [kh]
            margins = LOG_THR - lr.log_ratio().numpy()[blk]
            code = prop.codec.encode(*h.observable_state())
            a, info = dda_choose(code, C["py"][kh:kh + 1], C["pe"][kh:kh + 1], C["LT_np"][kh], C["py"][blk],
                                 C["pe"][blk], margins, C["inc"], prop, C["legal"], rng)
            obs = h.step(a)
            lr.update(obs)
            steps += 1
            if want_samples and len(traj) < 30:
                traj.append({"step": steps, "a": int(a), "set": int(mask.sum()), "bound": round(float(cert["ub"]), 4),
                             "r_bar_joint": round(float(jc["r_bar"]), 4), "mle": int(kh), "mode": info["mode"]})
        assert env.n_steps == lr.n_rounds == n0 + steps, "step accounting mismatch (rollout/evidence boundary)"
        mask = lr.mask().numpy()
        kf_ratio, rbf, ubf = kappa_time(Jt, Reg, mask)
        certified = status == "CERTIFIED"
        pi = cert["pi"] if certified else None
        tr = float(Jtrue.max() - Jtrue[pi]) if pi is not None else None
        charged = steps if certified else max(steps, TMAX_STEP)
        r = {"kind": "hd3", "instance": seed, "stream": stream, "noise_seed": noise, "twin": twin,
             "dose": 1.0 if twin == "khigh" else 0.0, "method": method, "cls": CLS, "truth_in_class": bool(in_class),
             "problem": q.pid, "k": k, "status": status, "certified": certified,
             "steps_consumed_stepwise": int(steps), "new_env_steps": int(charged), "censored": not certified,
             "certified_policy": pi, "true_regret": tr, "false_cert": bool(certified and tr is not None and tr > EPS),
             "zero_cost": bool(certified and steps == 0), "set_size": int(mask.sum()), "class_size": int(C["B"]),
             "final_bound": float(cert["ub"]) if np.isfinite(cert["ub"]) else None,
             "final_r_bar_joint": float(jc["r_bar"]) if np.isfinite(jc["r_bar"]) else None,
             "kappa_time_n0": k0_ratio, "r_bar_n0": rb0, "ub_rect_n0": ub0,
             "kappa_time_final": kf_ratio, "r_bar_final": rbf, "ub_rect_final": ubf,
             "theta_star_in_set": None, "env_n_steps": int(env.n_steps), "lr_n_rounds": int(lr.n_rounds),
             "true_gap_top2": float(np.sort(Jtrue)[-1] - np.sort(Jtrue)[-2]), "H": q.H, "n_policies": len(q.policies),
             "eps": EPS, "delta": DELTA, "wall_clock_s": time.perf_counter() - t0, "rollouts": 0,
             "eta_loc": 0.0, "eta_dec": 0.0, "theta_cell_alive": None}
        rows.append(r)
        if want_samples and len(samples) < 4 and (k < 2 or r["false_cert"]):
            samples.append({**r, "J_true": Jtrue.round(4).tolist(), "trajectory_head": traj,
                            "truth_params": {kk: inst.truth[kk] for kk in ("alpha", "gamma", "beta", "psi", "lam")}})
    return rows, samples, None, {"instance": seed, "stream": stream, "twin": twin, "method": method,
                                 "sec": time.time() - t_unit}


def run_unit_safe(*a):
    try:
        return run_unit(*a)
    except Exception as e:  # noqa: BLE001
        seed, stream, twin, method = a[:4]
        return [], [], {"instance": seed, "stream": stream, "twin": twin, "method": method, "error": repr(e),
                        "tb": traceback.format_exc()}, {"instance": seed, "stream": stream, "twin": twin,
                                                        "method": method, "sec": 0.0}


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
    st = [r["steps_consumed_stepwise"] for r in rows]
    kt = [r["kappa_time_n0"] for r in rows if r["kappa_time_n0"] is not None]
    return {"n": n, "status_counts": sc, "completion": len(cert) / n if n else None, "certs": len(cert),
            "false_certs": len(fc), "fcr": len(fc) / len(cert) if cert else None, "fcr_cp95": cp(len(fc), len(cert)),
            "zero_cost_rate": sum(r["zero_cost"] for r in rows) / n if n else None,
            "censored": sum(r["censored"] for r in rows),
            "steps_mean": float(np.mean([r["new_env_steps"] for r in rows])),
            "steps_median": float(np.median([r["new_env_steps"] for r in rows])),
            "stepwise_mean": float(np.mean(st)),
            "kappa_time_n0_median": float(np.median(kt)) if kt else None,
            "wall_clock_per_problem_s": float(np.mean([r["wall_clock_s"] for r in rows]))}


def per_instance_stats(rows):
    """Per instance & twin: S_m = mean over streams of the stream total (charged steps); L = log(S_rect/S_joint);
    secondary: mean over problems of log((rect+1)/(joint+1)) (problem-paired)."""
    out = {}
    insts = sorted({r["instance"] for r in rows})
    for i in insts:
        for tw in TWINS:
            S, pp = {}, []
            ok = True
            for m in METHODS:
                tots = {}
                for r in rows:
                    if r["instance"] == i and r["twin"] == tw and r["method"] == m:
                        tots[r["stream"]] = tots.get(r["stream"], 0) + r["new_env_steps"]
                if not tots:
                    ok = False
                    break
                S[m] = float(np.mean(list(tots.values())))
            if not ok:
                continue
            by = {}
            for r in rows:
                if r["instance"] == i and r["twin"] == tw:
                    by.setdefault((r["stream"], r["problem"]), {})[r["method"]] = r["new_env_steps"]
            for v in by.values():
                if len(v) == 2:
                    pp.append(math.log((v["rect"] + 1.0) / (v["joint"] + 1.0)))
            out[(i, tw)] = {"S_joint": S["joint"], "S_rect": S["rect"],
                            "L": math.log(max(S["rect"], 1.0) / max(S["joint"], 1.0)),
                            "L_pp": float(np.mean(pp)) if pp else None, "n_pairs": len(pp)}
    return out


def boot_ci(vals, B=BOOT_B, seed=42):
    v = np.asarray(vals, float)
    if v.size < 2:
        return {"estimate": float(v.mean()) if v.size else None, "ci95": [None, None], "n": int(v.size)}
    rng = np.random.default_rng(seed)
    bs = v[rng.integers(0, v.size, (B, v.size))].mean(1)
    lo, hi = np.quantile(bs, [0.025, 0.975])
    return {"estimate": float(v.mean()), "ci95": [float(lo), float(hi)], "n": int(v.size),
            "frac_boot_gt_0": float(np.mean(bs > 0)), "sign_pos": int((v > 0).sum()), "sign_neg": int((v < 0).sum())}


def analyse(rows):
    out = {"by_cell": {}}
    for tw in TWINS:
        for m in METHODS:
            rr = [r for r in rows if r["twin"] == tw and r["method"] == m]
            if rr:
                out["by_cell"][f"{m}@{tw}"] = cell_summary(rr)
    pi = per_instance_stats(rows)
    insts = sorted({i for i, _ in pi if (i, "khigh") in pi and (i, "k0") in pi})
    Lh = [pi[(i, "khigh")]["L"] for i in insts]
    L0 = [pi[(i, "k0")]["L"] for i in insts]
    D = [a - b for a, b in zip(Lh, L0)]
    Dpp = [pi[(i, "khigh")]["L_pp"] - pi[(i, "k0")]["L_pp"] for i in insts
           if pi[(i, "khigh")]["L_pp"] is not None and pi[(i, "k0")]["L_pp"] is not None]
    out["per_instance"] = [{"instance": i, "L_khigh": pi[(i, "khigh")]["L"], "L_k0": pi[(i, "k0")]["L"],
                            "D": pi[(i, "khigh")]["L"] - pi[(i, "k0")]["L"],
                            "S_joint_khigh": pi[(i, "khigh")]["S_joint"], "S_rect_khigh": pi[(i, "khigh")]["S_rect"],
                            "S_joint_k0": pi[(i, "k0")]["S_joint"], "S_rect_k0": pi[(i, "k0")]["S_rect"],
                            "Lpp_khigh": pi[(i, "khigh")]["L_pp"], "Lpp_k0": pi[(i, "k0")]["L_pp"]} for i in insts]
    out["log_rect_over_joint_by_kappa"] = {"khigh": boot_ci(Lh), "k0": boot_ci(L0)}
    out["paired_D"] = boot_ci(D)
    out["paired_D_ci"] = out["paired_D"]["ci95"]
    out["paired_D_problem_level_secondary"] = boot_ci(Dpp)
    # design-free kappa_time diagnostic at n0 (identical LR set for both methods; uses joint rows only)
    kt = {}
    for tw in TWINS:
        v = [r["kappa_time_n0"] for r in rows if r["twin"] == tw and r["method"] == "joint"
             and r["kappa_time_n0"] is not None]
        vf = [r["kappa_time_final"] for r in rows if r["twin"] == tw and r["method"] == "joint"
              and r["kappa_time_final"] is not None]
        kt[tw] = {"n0_median": float(np.median(v)) if v else None, "n0_mean_log": float(np.mean(np.log(v))) if v else None,
                  "n0_n": len(v), "final_median": float(np.median(vf)) if vf else None, "final_n": len(vf),
                  "share_exact_1_n0": float(np.mean(np.isclose(v, 1.0))) if v else None}
    out["kappa_time_diagnostic"] = kt
    out["hd3_gate"] = {"paired_D_ci_lower_gt_0": bool(out["paired_D"]["ci95"][0] is not None
                                                     and out["paired_D"]["ci95"][0] > 0),
                       "direction_D_positive": bool(out["paired_D"]["estimate"] is not None
                                                    and out["paired_D"]["estimate"] > 0)}
    return out


def plot(summary, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return None
    pi = summary.get("per_instance") or []
    if not pi:
        return None
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    for r in pi:
        col = "#2c7fb8" if r["D"] > 0 else "#c0392b"
        ax.plot([0, 1], [r["L_k0"], r["L_khigh"]], "-o", color=col, alpha=0.7, lw=1.2, ms=4)
    ax.axhline(0, color="grey", lw=0.8, ls=":")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([r"twin ($\gamma=\lambda=0$)", r"$\kappa_{high}$ (R0)"])
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylabel("log(N_rect(time) / N_joint), per instance")
    d = summary["paired_D"]
    ci = d["ci95"]
    ax.set_title(f"HD3 ({summary['mode']}): D = {d['estimate']:.3f}"
                 + (f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci[0] is not None else ""), fontsize=9)
    fig.tight_layout()
    p = out_dir / "paired_plot.png"
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
    args = ap.parse_args()
    pilot = args.mode == "pilot"
    lock = json.loads(LOCK.read_text())
    assert lock.get("version") == 2, "prereg lock v2 required"
    if not pilot:
        from dsswm.stats.prereg import assert_locked
        lock = assert_locked()
    quiet = args.smoke
    out_dir = RES_ROOT / ("pilots" if pilot else "full") / (TASK + ("_smoke" if args.smoke else ""))
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
        assert max(seeds) < 10000, "dev seeds only in pilot"
    else:
        rng_ = lock["eval_manifest"]["per_task_ranges"][TASK]
        seeds = [s for a, b in rng_ for s in range(a, b + 1)]
        n_prob, streams = int(lock["eval_manifest"].get("K_problems", 15)), [0, 1, 2]
    if args.smoke:
        seeds, n_prob = seeds[:1], 2
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)
    T0 = time.time()
    summary = {"task_id": TASK, "mode": args.mode, "seed": 42, "eps": EPS, "delta": DELTA,
               "truth": "khigh = R0 kappa_high truth (gamma=0.75, lam=0.5); k0 = matched twin (gamma=lam=0, rest identical)",
               "learner_class": "dynamic G_1, |Theta| = 13824 (contains both truths)",
               "protocol": "fresh-start per problem from shared n0=20 (no ledger reuse); DDA sampler for both arms; "
                           "T_max=3000 charged on censoring",
               "evidence": {"joint": "exact minimax R_bar <= eps",
                            "rect": "B6-sup rect(time): min_pi max_k sum_b sup_Theta [J_b(k)-J_b(pi)] <= eps "
                                    "(H step blocks + terminal retention block)"},
               "statistic": "L = log(S_rect/S_joint), S = stream total of charged steps averaged over streams; "
                            "D = L_khigh - L_k0 per instance; paired instance bootstrap B=10000",
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
            f"problems x streams {streams} x twins {TWINS} x methods {METHODS}; lock={lock.get('status')}")
        progress(1, 4, "gpu: per-step class J tables + J_true")
        meta = precompute(seeds, n_prob, streams, dev, log)
        summary["table_sec"] = meta["sec"]
        summary["perstep_sum_check_maxdiff"] = meta["sum_check_maxdiff"]
        summary["gpu_peak_mb"] = meta.get("gpu_peak_mb")
        summary["truth_in_class_all"] = all(meta["truth_in_class"].values())
        summary["twin_truths_distinct"] = all(meta["truth_sha"][f"{s}|khigh"] != meta["truth_sha"][f"{s}|k0"]
                                              for s in seeds)
        log(f"tables: {meta['sec']}; per-step sum check {meta['sum_check_maxdiff']:.2e}; gpu peak "
            f"{meta.get('gpu_peak_mb')} MB; truth in class {summary['truth_in_class_all']}")
        if not quiet and meta.get("gpu_name"):
            (RES_ROOT / f"{TASK}_gpu_profile.json").write_text(json.dumps({
                "gpu_name": meta["gpu_name"], "vram_total_mb": meta["vram_total_mb"], "max_batch_size": None,
                "vram_used_mb": meta["gpu_peak_mb"], "utilization_pct": 100 * meta["gpu_peak_mb"] / meta["vram_total_mb"],
                "note": "GPU only for truth-free per-step class J tables (13824 x K x (H+1), full class in one pass, "
                        "4096-chunk) and J_true; learners are CPU. Batch probing not applicable (exact propagator); "
                        "shared 4090"}))
        assert meta["sum_check_maxdiff"] < 1e-5, "per-step tables do not sum to the J table"
        assert summary["truth_in_class_all"], "truth-in-class precondition failed"
        assert summary["twin_truths_distinct"]
        done = set()
        if res_path.exists():
            cnt = {}
            for x in open(res_path):
                r = json.loads(x)
                u = (r["instance"], r["stream"], r["twin"], r["method"])
                cnt[u] = cnt.get(u, 0) + 1
            done = {u for u, c in cnt.items() if c >= n_prob}
            log(f"resume: {len(done)} finished units")
        jobs = [(s, st, tw, m) for s in seeds for st in streams for tw in TWINS for m in METHODS
                if (s, st, tw, m) not in done]
        jobs.sort(key=lambda j: (j[3] != "rect", j[0]))           # slower arm first for load balance
        progress(2, 4, "cpu: arms", {"jobs": len(jobs)})
        from joblib import Parallel, delayed
        errors, samples, unit_sec = [], [], []
        gen = Parallel(n_jobs=args.workers, backend="loky", return_as="generator_unordered")(
            delayed(run_unit_safe)(s, st, tw, m, n_prob, meta, s == seeds[0] and st == 0) for (s, st, tw, m) in jobs)
        n_done = 0
        with open(res_path, "a") as fh:
            for rr, ss, ee, us in gen:
                n_done += 1
                if ee:
                    errors.append(ee)
                    log(f"ERROR {ee['instance']} s{ee['stream']} {ee['twin']} {ee['method']}: {ee['error']}\n{ee['tb']}")
                else:
                    for r in rr:
                        fh.write(json.dumps(r, default=float) + "\n")
                    fh.flush()
                    samples += ss
                unit_sec.append(us)
                st_ = {}
                for r in rr:
                    st_[r["status"]] = st_.get(r["status"], 0) + 1
                log(f"unit {n_done}/{len(jobs)} inst {us['instance']} s{us['stream']} {us['twin']} {us['method']}: "
                    f"{us['sec']:.1f}s {st_} steps {sum(r['new_env_steps'] for r in rr)} "
                    f"false {sum(r['false_cert'] for r in rr)}")
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
            per.setdefault(f"{us['method']}@{us['twin']}", []).append(us["sec"])
        cpu_full = sum(float(np.mean(v)) / n_prob * 15 * 48 * 3 for v in per.values()) if pilot else None
        summary["timing_projection"] = {"unit_sec_mean": {k: float(np.mean(v)) for k, v in per.items()},
                                        "projected_full_cpu_sec": cpu_full,
                                        "projected_full_wall_min_at_4_workers": cpu_full / 4 / 60 if cpu_full else None,
                                        "note": "linear in problems x instances x streams; concurrent run"}
        fc = {k: v["false_certs"] for k, v in an["by_cell"].items()}
        pc = {"zero_crashes": len(errors) == 0, "evidence_boundary_ok": summary["evidence_boundary_ok"],
              "end_to_end": len(rows) == len(seeds) * len(streams) * n_prob * 4,
              "D_direction_reported": an["paired_D"]["estimate"] is not None,
              "rect_ge_joint_bound_sanity_tol1e-6_float32_blocks": all(r["ub_rect_final"] >= r["r_bar_final"] - 1e-6 for r in rows
                                                if r["ub_rect_final"] is not None and r["r_bar_final"] is not None)}
        summary["pass_criteria"] = pc
        summary["direction"] = "D>0" if (an["paired_D"]["estimate"] or 0) > 0 else "D<=0"
        summary["go_no_go"] = "GO" if all(pc.values()) else "NO_GO"
        summary["wall_clock_s"] = time.time() - T0
        summary["figure"] = plot(summary, out_dir)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
        for k_, v in an["by_cell"].items():
            log(f"{k_:12s} {v['status_counts']} FCR {v['false_certs']}/{v['certs']} cp95 {v['fcr_cp95']} "
                f"steps mean {v['steps_mean']:.0f} median {v['steps_median']:.0f} zero-cost {v['zero_cost_rate']:.2f}")
        log(f"L by kappa {an['log_rect_over_joint_by_kappa']}; paired D {an['paired_D']}; kappa_time "
            f"{an['kappa_time_diagnostic']}; false certs {fc}")
        log(f"done in {time.time() - T0:.0f}s: {summary['go_no_go']} {pc}")
        progress(4, 4, "done", {"go_no_go": summary["go_no_go"]})
        mark_done("success", f"{args.mode}: {summary['go_no_go']}; D = {an['paired_D']['estimate']} "
                             f"CI {an['paired_D']['ci95']}; L {an['log_rect_over_joint_by_kappa']['khigh']['estimate']} "
                             f"(khigh) vs {an['log_rect_over_joint_by_kappa']['k0']['estimate']} (k0)")
    except Exception as e:  # noqa: BLE001
        log(traceback.format_exc())
        summary["error"] = repr(e)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
        mark_done("failed", repr(e))
        raise


if __name__ == "__main__":
    main()
