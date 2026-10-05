"""r4_setup_fcc: FCC setup -- unit tests, bound validity on random small instances, contract table (analytic + 10^5-step
chi^2 simulation per truth family), parameter-repetition proof gate, ms_per_bound / BnB timing.

Harness side (creates ground truth for the contract simulations and the repetition MC); dsswm.certify.fcc itself is
learner side and never touches truth.

Usage: run_r4_setup_fcc.py --mode {pilot,full} [--workers 4]
pilot: 200 random instances (bound validity), contract sims dev seeds 800-809 x 10^4 steps per family, repetition MC
       on off-grid R1 dev instances 800-809 (15 problems each, 200 trajectories per policy).
full : 1000 random instances (+ sha256 of the per-instance table); same contract / repetition design.
Concurrent run (shares 20 cores with other round-4 tasks): timings are "concurrent" (并发运行). CPU only.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from datetime import datetime  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "dsswm" / "tests"))
WS = HERE.parents[1]
RES = WS / "exp" / "results"
TASK = "r4_setup_fcc"
DEV_SEEDS = list(range(800, 810))
STEPS_PER_SEED = 10_000
M4_S = 0.94            # hf3 registered m4 strength at eta = 2 eps (results.jsonl strength field)
M4_S_WEAK = 0.094
ALPHA_FAM = 0.01


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def progress(step, total, phase, metric=None):
    (RES / f"{TASK}_PROGRESS.json").write_text(json.dumps({
        "task_id": TASK, "epoch": step, "total_epochs": total, "step": step, "total_steps": total, "loss": None,
        "metric": {"phase": phase, **(metric or {})}, "updated_at": datetime.now().isoformat()}))


def mark_done(status, summary):
    pid = RES / f"{TASK}.pid"
    if pid.exists():
        pid.unlink()
    pf = RES / f"{TASK}_PROGRESS.json"
    fp = json.loads(pf.read_text()) if pf.exists() else {}
    (RES / f"{TASK}_DONE").write_text(json.dumps({"task_id": TASK, "status": status, "summary": summary,
                                                  "final_progress": fp, "timestamp": datetime.now().isoformat()}))


# =============================================================================================== 1 unit tests
def run_unit_tests(out_dir):
    t0 = time.time()
    junit = out_dir / "pytest_fcc.xml"
    cmd = [sys.executable, "-m", "pytest", "-q", "dsswm/tests/test_fcc.py", "dsswm/tests/test_no_truth_import.py",
           f"--junitxml={junit}"]
    r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    tail = (r.stdout + r.stderr).strip().splitlines()[-3:]
    import xml.etree.ElementTree as ET
    root = ET.parse(junit).getroot()
    ts = root if root.tag == "testsuite" else root.find("testsuite")
    n, f, e, s = (int(ts.get(k, 0)) for k in ("tests", "failures", "errors", "skipped"))
    return {"returncode": r.returncode, "tests": n, "failures": f, "errors": e, "skipped": s,
            "passed": n - f - e - s, "all_pass": r.returncode == 0 and f == 0 and e == 0, "tail": tail,
            "seconds": round(time.time() - t0, 1)}


# =============================================================================================== 2 bound validity
def _validity_job(seed):
    from test_fcc import check_random_instance
    r = check_random_instance(seed, n_points=10_000)
    rng_width = r["Jmax"] - r["Jmin"]
    r["slack_hi"] = r["jhi"] - r["Jmax"]
    r["slack_lo"] = r["Jmin"] - r["jlo"]
    r["amplification"] = (r["jhi"] - r["jlo"]) / rng_width if rng_width > 1e-12 else float("nan")
    return r


# =============================================================================================== 3 contract simulation
def family_env(fam, seed):
    """Ground-truth platform of one misspecification family (dev seed). Returns (env, meta)."""
    from dsswm.envs.mis import make_m4_env, make_mis_env
    from dsswm.envs.nl import NLEnv
    from dsswm.models.grid_ladder import BOX
    from dsswm.streams.offgrid import sample_r1_truth, truth_for
    from replica.model import replica_synergy_pattern
    kw = dict(c=1.0, rho_ret=0.3, L=2, R=2, nmax=2, budget=1, incentive_levels=(1,), seed=seed * 1000 + 42)
    r0 = truth_for(seed, "R0")
    base = {k: (np.asarray(v, float) if isinstance(v, np.ndarray) else v) for k, v in r0.items()}
    base_l = dict(base, psi=[base["psi"]])
    rng = np.random.default_rng([seed, 61])
    meta = {}
    if fam == "m1":
        env = make_mis_env("m1", base_l, 1.0, rng, **kw)
    elif fam == "m1r":
        S = replica_synergy_pattern(seed)
        env = NLEnv(base["alpha"], base["beta"], base["gamma"], base["tauL"], base["tauR"], [base["psi"]],
                    base["lam"], syn=1.0 * S, **kw)
    elif fam in ("m2", "m3"):
        env = make_mis_env(fam, base_l, 1.0, rng, **kw)
    elif fam == "alias":
        env = NLEnv(base["alpha"], base["beta"], base["gamma"], base["tauL"], base["tauR"], [base["psi"] + 0.37],
                    base["lam"] * 0.6 + 0.07, **kw)
    elif fam in ("static_twin_g1", "static_twin_g0"):
        g = 1.0 if fam.endswith("g1") else 0.0
        th = truth_for(seed, "R0", dose=g) if g > 0 else truth_for(seed, "R0", twin=True)
        env = NLEnv(th["alpha"], th["beta"], th["gamma"], th["tauL"], th["tauR"], [th["psi"]], th["lam"], **kw)
    elif fam == "R1":
        th = sample_r1_truth(seed)
        env = NLEnv(th["alpha"], th["beta"], th["gamma"], th["tauL"], th["tauR"], [th["psi"]], th["lam"], **kw)
    elif fam == "R1adv":
        r = np.random.default_rng([seed, 51, 0])
        u = lambda k, n=None: r.uniform(*BOX[k], n)  # noqa: E731
        env = NLEnv(u("alpha", 2), u("beta", 2), u("gamma", 2), u("tauL", 2), u("tauR", 2), [float(u("psi"))],
                    float(u("lam")), **kw)
    elif fam in ("m4", "m4_weak"):
        s = M4_S if fam == "m4" else M4_S_WEAK
        env = make_m4_env(base, drift_s=s, **kw)
        meta["drift_s"] = s
    else:
        raise ValueError(fam)
    return env, meta


def _contract_job(args):
    fam, seed, n_steps = args
    from dsswm.certify.fcc import CellIndex
    env, meta = family_env(fam, seed)
    C = CellIndex(2, 2, 2, env.aspace.nb)
    rng = np.random.default_rng([seed, 8801, FAMILIES.index(fam)])
    A = env.aspace
    L, P = 2, 4
    prev_y = np.zeros(P, dtype=int)
    prev_m = np.zeros(P, dtype=int)
    last_a, last_succ = 0, False
    ep_len, ep_pos = 6, 0
    rows = []      # (cell, x, f_time, f_prev_y, f_prev_m, f_other, f_cause, f_pos)
    for t in range(n_steps):
        if ep_pos == ep_len:
            env.reset_to(rng.integers(0, 3, P), (rng.random(P) < 0.85).astype(np.int64))
            ep_len, ep_pos = int(rng.choice([6, 8])), 0
            prev_y[:] = 0
            prev_m[:] = 0
        a = last_a if (last_succ and rng.random() < 0.5) else int(rng.integers(A.n))
        obs = env.step(a)
        ep_pos += 1
        tb = min(3, 4 * t // n_steps)
        matched = np.concatenate([A.matched_left(a), A.matched_right(a)]).astype(int)
        y_p = np.zeros(P, dtype=int)
        cause = np.zeros(P, dtype=int)          # 0 unmatched, 1 matched-active, 2 matched-inactive
        for (i, j, b, y, active) in obs.outcomes:
            for q in (i, L + j):
                cause[q] = 1 if active else 2
            if active:
                y_p[i] = y_p[L + j] = y
                rows.append((C.pair(i, j, obs.loads[i], b), y, tb, prev_y[i], prev_m[i], obs.loads[L + j],
                             int(sum(obs.engaged)) - 2, min(ep_pos - 1, 3)))
        for p in range(P):
            if obs.engaged[p]:
                other = int(sum(obs.engaged)) - 1
                rows.append((C.ret(p, obs.loads[p], y_p[p]), obs.next_engaged[p], tb, prev_y[p], prev_m[p], other,
                             cause[p] if y_p[p] == 0 else 3, min(ep_pos - 1, 3)))
        prev_y, prev_m = y_p, matched
        last_a, last_succ = a, bool(y_p.any())
    arr = np.array(rows, dtype=np.int64)
    return fam, seed, arr, meta


FEATURES = ("time_quartile", "prev_outcome", "prev_matched", "partner_load_or_n_engaged_others", "n_engaged_or_y0_cause",
            "episode_position")


def chi2_tests(arr, n_cells):
    from scipy.stats import chi2_contingency, fisher_exact
    out = []
    for c in np.unique(arr[:, 0]):
        sub = arr[arr[:, 0] == c]
        x = sub[:, 1]
        if len(sub) < 40 or x.min() == x.max():
            continue
        for fi, fname in enumerate(FEATURES):
            f = sub[:, 2 + fi]
            levels = [v for v in np.unique(f) if (f == v).sum() >= 20]
            if len(levels) < 2:
                continue
            tab = np.array([[np.sum((f == v) & (x == k)) for v in levels] for k in (0, 1)])
            if (tab.sum(1) == 0).any():
                continue
            exp = tab.sum(1, keepdims=True) * tab.sum(0, keepdims=True) / tab.sum()
            if exp.min() < 5:
                if tab.shape[1] == 2:
                    p = float(fisher_exact(tab)[1])
                    test = "fisher"
                else:
                    continue
            else:
                p = float(chi2_contingency(tab, correction=False)[1])
                test = "chi2"
            out.append({"cell": int(c), "feature": fname, "n": int(len(sub)), "levels": len(levels), "p": p,
                        "test": test})
    return out


ANALYTIC = {
    "m1": ("pair synergy logit += s * S_ij (double-centred; main package rng [seed, 61])",
           "S_ij is a constant of the pair (i, j): absorbed by the pair-cell index (i, j)", "satisfied"),
    "m1r": ("pair synergy logit += s * S_ij (replica pattern rng [seed, 9071])",
            "absorbed by the pair-cell index (i, j)", "satisfied"),
    "m2": ("saturating fatigue gamma_i * g(n_i), g concave", "g(n_i) is a function of n_i: absorbed by the pair-cell "
           "index n_i (retention keeps lam * n_p: cell index n_p)", "satisfied"),
    "m3": ("hidden two-type left participants: psi_left[i, b] by latent type (fixed per instance)",
           "type is a constant of left participant i: absorbed by the pair-cell index (i, b)", "satisfied"),
    "alias": ("(psi, lam) moved off the grid jointly", "only the values of cell probabilities move", "satisfied"),
    "static_twin_g1": ("truth = NL-R0 (gamma, lam at dose 1); learner = static class (learner-side misspecification)",
                       "truth is an NLEnv: cells (i,j,n_i,b) / (p,n_p,y_p) are sufficient; the learner class is "
                       "irrelevant to FCC (non-parametric cells)", "satisfied"),
    "static_twin_g0": ("exact static twin: gamma = lam = 0 (cells then do not depend on n)",
                       "cells remain sufficient (load index is redundant)", "satisfied"),
    "R1": ("off-grid continuous theta* (sample_r1_truth)", "theta* off the learner grid only changes cell values",
           "satisfied"),
    "R1adv": ("theta* uniform on the G_1 hull (grid_ladder.BOX; R1-adv sampler base draw rng [s, 51, d])",
              "only cell values change; the R1-adv acceptance step selects problems, not dynamics", "satisfied"),
    "m4": ("temporal drift alpha_i(t) = alpha_i + s * t / T_ref (global platform round)",
           "pair-cell probability depends on the global round t, which is not a cell index: NON-STATIONARY",
           "violated (FCC makes no claim)"),
    "m4_weak": ("m4 with s / 10 (power check of the chi^2 time split)", "non-stationary", "violated (FCC makes no "
                                                                                          "claim)"),
}
FAMILIES = ["m1", "m1r", "m2", "m3", "alias", "static_twin_g1", "static_twin_g0", "R1", "R1adv", "m4", "m4_weak"]


def summarise_contract(results, n_cells, out_dir):
    from scipy.stats import kstest
    per_fam = defaultdict(list)
    meta_f = {}
    nsteps = Counter()
    for fam, seed, arr, meta in results:
        for r in chi2_tests(arr, n_cells):
            r.update(fam=fam, seed=seed)
            per_fam[fam].append(r)
        meta_f[fam] = meta
        nsteps[fam] += STEPS_PER_SEED
    rows_csv = [r for fam in FAMILIES for r in per_fam[fam]]
    with open(out_dir / "contract_chi2_tests.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["fam", "seed", "cell", "feature", "n", "levels", "test", "p"])
        w.writeheader()
        for r in rows_csv:
            w.writerow({k: r[k] for k in w.fieldnames})
    verdicts = {}
    for fam in FAMILIES:
        ps = np.array([r["p"] for r in per_fam[fam]])
        m = len(ps)
        bonf = int((ps < ALPHA_FAM / max(m, 1)).sum())
        by_feat = {}
        for fname in FEATURES:
            pf = np.array([r["p"] for r in per_fam[fam] if r["feature"] == fname])
            by_feat[fname] = {"n_tests": int(len(pf)), "frac_p_lt_0.01": float((pf < 0.01).mean()) if len(pf) else None,
                              "bonf_rejections": int((pf < ALPHA_FAM / max(m, 1)).sum()),
                              "min_p": float(pf.min()) if len(pf) else None}
        ks = float(kstest(ps, "uniform").pvalue) if m > 5 else None
        sim = "violated" if bonf > 0 else "no evidence of violation"
        expected = ANALYTIC[fam][2]
        if expected.startswith("violated"):
            consistent = True              # violated families: FCC makes no claim; detection is a power statement
            sim = "violation detected" if bonf > 0 else "not detected (power limit of the chi^2 time split)"
        else:
            consistent = sim == "no evidence of violation"
        verdicts[fam] = {"n_steps": int(nsteps[fam]), "n_tests": m, "bonferroni_rejections": bonf,
                         "frac_p_lt_0.01": float((ps < 0.01).mean()) if m else None,
                         "min_p": float(ps.min()) if m else None, "ks_uniform_p": ks, "by_feature": by_feat,
                         "sim_verdict": sim, "analytic_verdict": expected, "consistent": bool(consistent),
                         **meta_f.get(fam, {})}
    # markdown table
    md = ["# FCC contract table (r4_setup_fcc)", "",
          "生成：`exp/code/run_r4_setup_fcc.py`（解析读码 `envs/nl.py`、`envs/mis.py`、`envs/static.py`、"
          "`streams/offgrid.py`、`evidence/mixed_lr.py:factorize`，加每族 10 个 dev 种子 800-809 × 10^4 步模拟）。"
          "本表在任何 FCC 认证结果出现之前写成，进入 r4_prereg_lock。", "",
          "## 因子格与更新条件（契约）", "",
          "| 因子格 | 索引 | 更新条件 | 结果变量 | 生成式（envs/nl.py） |",
          "|---|---|---|---|---|",
          "| pair | (i, j, n_i, b) | 匹配且双方在本轮开始时都 engaged（active=1） | y_ij | "
          "sigmoid(alpha_i + beta_j − gamma_i g(n_i) + psi_left[i, b] + syn_ij) |",
          "| retention | (p, n_p, y_p) | p 在本轮开始时 engaged | e_p' | "
          "sigmoid(tau_p + c·y_p − lam·n_p)，n_p 为本轮负载更新前的负载；y_p=0 当 p 未匹配或其 pair inactive |",
          "| 已知常数 | — | p 未 engaged | e_p' | Bern(rho_ret)，rho_ret = 0.3（`known_constants`），不进入未知集合 |",
          "| 已知规则 | — | 每轮 | loads' | matched +1（上限 nmax），unmatched −1（下限 0）（core/dynamics.py） |",
          "", f"格数：pair {2 * 2 * 3 * 2} + retention {4 * 3 * 2} = {n_cells}（L=R=2, nmax=2, nb=2）；"
              f"δ_cell = δ / {n_cells}。", "",
          "## 逐族判定", "",
          "χ² 检验：每个 (实例, 格, 未记录历史特征) 一个 2×k 列联表（结果 × 特征分箱；分箱 n<20 丢弃；期望<5 时 2×2 用 Fisher，"
          "否则跳过）。特征：time_quartile（全局时间四分位，平稳性）、prev_outcome（上一轮本人结果）、prev_matched、"
          "partner_load / 其他 engaged 人数、n_engaged / y=0 成因（未匹配、active 失败、inactive）、episode_position。"
          f"族内 Bonferroni α = {ALPHA_FAM}。行为策略：50% 重复上一轮成功动作、50% 均匀随机（含激励），每 6/8 步随机重置状态。", "",
          "| 族 | 真值生成式 | 契约解析 | 解析判定 | 步数 | 检验数 | Bonferroni 拒绝 | p<0.01 比例 | 最小 p | 模拟判定 | 一致 |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for fam in FAMILIES:
        v = verdicts[fam]
        gen, why, exp = ANALYTIC[fam]
        md.append(f"| {fam} | {gen} | {why} | {exp} | {v['n_steps']} | {v['n_tests']} | {v['bonferroni_rejections']} | "
                  f"{v['frac_p_lt_0.01']:.4f} | {v['min_p']:.2e} | {v['sim_verdict']} | {'是' if v['consistent'] else '否'} |")
    md += ["", "## m4 按特征的拒绝数", ""]
    for fam in ("m4", "m4_weak"):
        bf = verdicts[fam]["by_feature"]
        md.append(f"- {fam}（drift_s = {verdicts[fam].get('drift_s')}）：" + "；".join(
            f"{k} {bf[k]['bonf_rejections']}/{bf[k]['n_tests']}" for k in FEATURES))
    md += ["", "结论：FCC 的主张只覆盖“契约成立”的族（m1、m1r、m2、m3、alias、静态孪生 g∈{0,1}、R1、R1-adv）；"
               "m4 漂移违反平稳性，FCC 对 m4 **不作任何主张**。", ""]
    (out_dir / "contract_table.md").write_text("\n".join(md))
    return verdicts


# =============================================================================================== 4 repetition gate
def _repetition_job(args):
    seed, n_traj = args
    from dsswm.certify.fcc import DecoupledSolver, cell_probs
    from dsswm.streams.offgrid import make_offgrid_instance
    inst = make_offgrid_instance(seed, "R1", K=15)
    env = inst.env
    solver = DecoupledSolver(2, 2, 2, env.aspace, rho_ret=0.3)
    theta = cell_probs(env.true_params(), solver.cells, c=1.0)
    ext = np.concatenate([theta, [0.0, 0.3]])
    nC = solver.cells.n
    rng = np.random.default_rng([seed, 8802])
    rows = []
    for prob in inst.problems:
        for k, pi in enumerate(prob.policies):
            plan = solver.plan(pi, prob.loads0, prob.engaged0, prob.H)
            # ---- Monte Carlo realised repetition
            cnt = np.zeros((n_traj, nC), dtype=np.int64)
            sidx = np.zeros(n_traj, dtype=np.int64)          # index into plan.states[t]
            for t in range(prob.H):
                cid = plan.cellids[t][sidx]                   # (n, G, 5)
                pr = ext[cid]
                y = rng.random((n_traj, solver.G)) < pr[:, :, 0]
                r = np.where(y, pr[:, :, 1], pr[:, :, 2])
                ea = rng.random((n_traj, solver.G)) < r
                r = np.where(y, pr[:, :, 3], pr[:, :, 4])
                eb = rng.random((n_traj, solver.G)) < r
                for g in range(solver.G):
                    pc = cid[:, g, 0]
                    act = pc < nC
                    np.add.at(cnt, (np.flatnonzero(act), pc[act]), 1)
                    for col1, col0 in ((1, 2), (3, 4)):
                        rc = np.where(y[:, g], cid[:, g, col1], cid[:, g, col0])
                        ok = rc < nC
                        np.add.at(cnt, (np.flatnonzero(ok), rc[ok]), 1)
                combo = np.zeros(n_traj, dtype=np.int64)
                for g in range(solver.G):
                    combo = combo * 4 + (2 * ea[:, g] + eb[:, g])
                sidx = plan.nxt[t][sidx, combo]
            # ---- structural worst case (pair cells exact; retention cells counted for both y -> upper bound)
            M = np.zeros((len(plan.states[prob.H]) + 1, nC), dtype=np.int64)
            for t in range(prob.H - 1, -1, -1):
                cid = plan.cellids[t]
                nxt = plan.nxt[t]
                Mn = np.zeros((cid.shape[0], nC), dtype=np.int64)
                best = M[nxt].max(1)                          # (S, nC)
                pres = np.zeros((cid.shape[0], nC + 2), dtype=np.int64)
                np.put_along_axis(pres, cid.reshape(cid.shape[0], -1), 1, axis=1)
                Mn = best + pres[:, :nC]
                M = np.vstack([Mn, np.zeros((1, nC), dtype=np.int64)])
            worst = M[0]
            pair_c = cnt[:, :solver.cells.n_pair]
            ret_c = cnt[:, solver.cells.n_pair:]
            rows.append({"seed": seed, "pid": prob.pid, "policy": k, "H": prob.H,
                         "mc_max_pair": int(pair_c.max()), "mc_max_ret": int(ret_c.max()),
                         "mc_frac_traj_any_repeat": float((cnt.max(1) >= 2).mean()),
                         "mc_frac_traj_pair_repeat": float((pair_c.max(1) >= 2).mean()),
                         "mc_mean_max_rep": float(cnt.max(1).mean()),
                         "worst_pair": int(worst[:solver.cells.n_pair].max()),
                         "worst_ret_ub": int(worst[solver.cells.n_pair:].max()),
                         "hist_pair": np.bincount(pair_c[pair_c > 0], minlength=9)[:9].tolist(),
                         "hist_ret": np.bincount(ret_c[ret_c > 0], minlength=9)[:9].tolist()})
    return rows


# =============================================================================================== 5 timing / BnB
def timing_and_bnb(seed=800, n_steps=3000, n_bnb_problems=3):
    from dsswm.certify.fcc import FCCCertifier, cell_probs, value_bounds_bnb
    from dsswm.streams.offgrid import make_offgrid_instance
    inst = make_offgrid_instance(seed, "R1", K=15)
    env = inst.env
    cert = FCCCertifier(2, 2, 2, env.aspace, rho_ret=0.3, delta=0.05)
    t0 = time.perf_counter()
    for o in inst.init_obs:
        cert.observe(o)
    rng = np.random.default_rng([seed, 8803])
    for _ in range(n_steps):
        cert.observe(env.step(int(rng.integers(env.aspace.n))))
    cs_ms = (time.perf_counter() - t0) / (n_steps + len(inst.init_obs)) * 1000
    lo, hi = cert.cs.intervals()
    theta = cell_probs(env.true_params(), cert.cells, c=1.0)
    cover = bool(((theta >= lo) & (theta <= hi)).all())
    out = {"cs_ms_per_round": cs_ms, "cs_cover_truth": cover, "n_rounds": n_steps + len(inst.init_obs),
           "median_cs_width": float(np.median(hi - lo)), "max_cs_width": float((hi - lo).max()),
           "per_H": defaultdict(list), "plan_ms": [], "bnb": []}
    for prob in inst.problems:
        for pi in prob.policies:
            t1 = time.perf_counter()
            plan = cert.solver.plan(pi, prob.loads0, prob.engaged0, prob.H)
            t2 = time.perf_counter()
            jl, jh = cert.solver.solve(plan, prob.utility.w, prob.utility.w_ret, prob.utility.c_q, lo, hi)
            t3 = time.perf_counter()
            out["plan_ms"].append((t2 - t1) * 1000)
            out["per_H"][prob.H].append({"ms": (t3 - t2) * 1000, "width": jh - jl, "n_states": int(
                sum(len(s) for s in plan.states))})
    # decoupling amplification + BnB on the first problems (shared-parameter range estimated by box points)
    for prob in inst.problems[:n_bnb_problems]:
        pi = prob.policies[0]
        plan = cert.solver.plan(pi, prob.loads0, prob.engaged0, prob.H)
        u = prob.utility
        jl, jh = cert.solver.solve(plan, u.w, u.w_ret, u.c_q, lo, hi)
        used = sorted(cert.solver.used_cells(plan))
        pts = []
        for _ in range(300):
            th = lo + (hi - lo) * rng.random(len(lo))
            if rng.random() < 0.5:
                th = np.where(rng.random(len(lo)) < 0.5, lo, hi)
            a, _b = cert.solver.solve(plan, u.w, u.w_ret, u.c_q, th, th)
            pts.append(a)
        # vertex local search (coordinate flips over the used cells, 6 restarts) -> better shared-parameter range
        for sense in (1, -1):
            for _r in range(6):
                th = np.where(rng.random(len(lo)) < 0.5, lo, hi)
                cur = cert.solver.solve(plan, u.w, u.w_ret, u.c_q, th, th)[0]
                improved = True
                while improved:
                    improved = False
                    for c in used:
                        th2 = th.copy()
                        th2[c] = hi[c] if th[c] == lo[c] else lo[c]
                        v = cert.solver.solve(plan, u.w, u.w_ret, u.c_q, th2, th2)[0]
                        if sense * v > sense * cur + 1e-15:
                            th, cur, improved = th2, v, True
                pts.append(cur)
        tb = time.perf_counter()
        bh = value_bounds_bnb(cert.solver, plan, u, lo, hi, max_cells=4, max_nodes=48, time_budget_s=20, sense="hi")
        bl = value_bounds_bnb(cert.solver, plan, u, lo, hi, max_cells=4, max_nodes=48, time_budget_s=20, sense="lo")
        out["bnb"].append({"pid": prob.pid, "H": prob.H, "n_used_cells": len(used), "dec_lo": jl, "dec_hi": jh,
                           "shared_est_min": float(min(pts)), "shared_est_max": float(max(pts)),
                           "amplification_width": (jh - jl) / max(max(pts) - min(pts), 1e-12),
                           "bnb_lo": bl["bnb"], "bnb_hi": bh["bnb"], "bnb_nodes": bh["nodes"] + bl["nodes"],
                           "bnb_s": time.perf_counter() - tb,
                           "J_true": float(cert.solver.solve(plan, u.w, u.w_ret, u.c_q, theta, theta)[0])})
    per_H = {}
    for H, v in out["per_H"].items():
        ms = np.array([x["ms"] for x in v])
        per_H[str(H)] = {"n": len(v), "ms_median": float(np.median(ms)), "ms_p95": float(np.percentile(ms, 95)),
                         "width_median": float(np.median([x["width"] for x in v])),
                         "states_median": float(np.median([x["n_states"] for x in v]))}
    out["per_H"] = per_H
    out["plan_ms_median"] = float(np.median(out["plan_ms"]))
    del out["plan_ms"]
    return out


# =============================================================================================== 6 non-vacuity diagnostic
def _nonvac_job(args):
    seed, checkpoints, eps = args
    from dsswm.certify.fcc import FCCCertifier, cell_probs
    from dsswm.streams.offgrid import make_offgrid_instance
    inst = make_offgrid_instance(seed, "R1", K=15)
    env = inst.env
    cert = FCCCertifier(2, 2, 2, env.aspace, rho_ret=0.3, delta=0.05)
    theta = cell_probs(env.true_params(), cert.cells, c=1.0)
    for o in inst.init_obs:
        cert.observe(o)
    rng = np.random.default_rng([seed, 8804])
    plans = [[cert.solver.plan(pi, pr.loads0, pr.engaged0, pr.H) for pi in pr.policies] for pr in inst.problems]
    Jt = [np.array([cert.solver.solve(pl, pr.utility.w, pr.utility.w_ret, pr.utility.c_q, theta, theta)[0]
                    for pl in pls]) for pr, pls in zip(inst.problems, plans)]
    out, done = [], len(inst.init_obs)
    for T in checkpoints:
        for _ in range(T - done):
            cert.observe(env.step(int(rng.integers(env.aspace.n))))
        done = T
        lo, hi = cert.cs.intervals()
        widths, cert_n, wrong, gaps = [], 0, 0, []
        for pr, pls, jt in zip(inst.problems, plans, Jt):
            b = cert.bounds(pr, plans=pls)
            widths += list(b[:, 1] - b[:, 0])
            st, k = FCCCertifier.decide(b, eps)
            srt = np.sort(jt)
            gaps.append(float(srt[-1] - srt[-2]) if len(srt) > 1 else 1.0)
            if st == "CERTIFIED":
                cert_n += 1
                wrong += int(jt[k] < jt.max() - eps)
        out.append({"seed": seed, "rounds": T, "median_cell_width": float(np.median(hi - lo)),
                    "median_J_width": float(np.median(widths)), "frac_certified": cert_n / len(inst.problems),
                    "n_wrong": wrong, "median_true_top2_gap": float(np.median(gaps)),
                    "truth_covered": bool(((theta >= lo) & (theta <= hi)).all())})
    return out


# =============================================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    out_dir = RES / ("pilots" if args.mode == "pilot" else "full") / TASK
    out_dir.mkdir(parents=True, exist_ok=True)
    (RES / f"{TASK}.pid").write_text(str(os.getpid()))
    t_start = time.time()
    summary = {"task_id": TASK, "mode": args.mode, "seed": 42, "started": datetime.now().isoformat(),
               "timing_note": "并发运行 (concurrent with other round-4 tasks; <= 4 CPU workers, BLAS threads = 1)"}
    try:
        progress(0, 5, "unit_tests")
        log("unit tests")
        summary["unit_tests"] = run_unit_tests(out_dir)
        log(f"unit tests: {summary['unit_tests']['passed']}/{summary['unit_tests']['tests']}")

        progress(1, 5, "bound_validity")
        n_inst = 200 if args.mode == "pilot" else 1000
        with Pool(args.workers) as pool:
            vr = pool.map(_validity_job, range(n_inst))
        with open(out_dir / "bound_validity.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(vr[0].keys()))
            w.writeheader()
            w.writerows(vr)
        sha = hashlib.sha256((out_dir / "bound_validity.csv").read_bytes()).hexdigest()
        amp = np.array([r["amplification"] for r in vr])
        summary["bound_validity"] = {
            "n_instances": n_inst, "points_per_instance": 10_000, "violations": int(sum(not r["ok"] for r in vr)),
            "min_slack_hi": float(min(r["slack_hi"] for r in vr)), "min_slack_lo": float(min(r["slack_lo"] for r in vr)),
            "n_full_vertex_enum": int(sum(r["n_vertices"] > 0 for r in vr)),
            "amplification_width_median": float(np.nanmedian(amp)), "amplification_width_p90": float(np.nanpercentile(amp, 90)),
            "csv_sha256": sha}
        log(f"bound validity: {summary['bound_validity']}")

        progress(2, 5, "contract_sims")
        jobs = [(f, s, STEPS_PER_SEED) for f in FAMILIES for s in DEV_SEEDS]
        with Pool(args.workers) as pool:
            cres = pool.map(_contract_job, jobs, chunksize=2)
        verdicts = summarise_contract(cres, 48, out_dir)
        summary["contract"] = {f: {k: v for k, v in d.items() if k != "by_feature"} for f, d in verdicts.items()}
        summary["contract_by_feature_m4"] = verdicts["m4"]["by_feature"]
        summary["contract_complete"] = all(f in verdicts and verdicts[f]["n_tests"] > 0 for f in FAMILIES)
        summary["contract_all_consistent"] = all(v["consistent"] for v in verdicts.values())
        log("contract: " + ", ".join(f"{f}:{v['bonferroni_rejections']}/{v['n_tests']}" for f, v in verdicts.items()))

        progress(3, 5, "repetition_gate")
        with Pool(args.workers) as pool:
            rr = pool.map(_repetition_job, [(s, 200) for s in DEV_SEEDS])
        rows = [r for x in rr for r in x]
        with open(out_dir / "repetition_counts.csv", "w", newline="") as fh:
            keys = [k for k in rows[0] if not k.startswith("hist")] + [f"hist_pair_{k}" for k in range(1, 9)] + \
                [f"hist_ret_{k}" for k in range(1, 9)]
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for r in rows:
                d = {k: r[k] for k in r if not k.startswith("hist")}
                d.update({f"hist_pair_{k}": r["hist_pair"][k] for k in range(1, 9)})
                d.update({f"hist_ret_{k}": r["hist_ret"][k] for k in range(1, 9)})
                w.writerow(d)
        rep = {}
        for H in (6, 8):
            sub = [r for r in rows if r["H"] == H]
            if not sub:
                continue
            hp = np.sum([r["hist_pair"] for r in sub], 0)
            hr = np.sum([r["hist_ret"] for r in sub], 0)
            rep[f"H{H}"] = {"n_problem_policies": len(sub), "mc_max_pair": int(max(r["mc_max_pair"] for r in sub)),
                            "mc_max_ret": int(max(r["mc_max_ret"] for r in sub)),
                            "worst_pair_structural": int(max(r["worst_pair"] for r in sub)),
                            "worst_ret_structural_ub": int(max(r["worst_ret_ub"] for r in sub)),
                            "frac_traj_any_repeat_mean": float(np.mean([r["mc_frac_traj_any_repeat"] for r in sub])),
                            "frac_traj_pair_repeat_mean": float(np.mean([r["mc_frac_traj_pair_repeat"] for r in sub])),
                            "frac_policies_with_pair_repeat": float(np.mean([r["mc_max_pair"] >= 2 for r in sub])),
                            "hist_pair_counts_1to8": hp[1:].tolist(), "hist_ret_counts_1to8": hr[1:].tolist()}
        summary["repetition"] = rep
        summary["max_param_repetition"] = int(max(max(r["mc_max_pair"], r["mc_max_ret"]) for r in rows))
        summary["repetition_explanation"] = (
            "同一未知因子格在一条 H=6/8 轨迹里反复出现（见 hist：计数≥2 的格-轨迹占多数），所以轨迹概率是该格参数的高次多项式"
            "（p^2、p(1−p)、…），J(θ) 关于共享参数不是多线性的，箱上的最大值可以落在内部（p(1−p) 反例：顶点 0.16 < 内部 0.25）。"
            "因此“共享参数取顶点”的做法不是合法上界；FCC 改用解耦（矩形）区间 MDP：每个 (t, 状态) 内每个格最多出现一次，"
            "一步目标关于该步的参数是多线性的，逐步取顶点是该松弛下的精确极值，且松弛值 ≥ 共享参数真极值。")
        log(f"repetition: {rep}")

        progress(4, 5, "timing_bnb")
        summary["timing"] = timing_and_bnb()
        summary["ms_per_bound"] = {H: v["ms_median"] for H, v in summary["timing"]["per_H"].items()}
        log(f"timing: {summary['timing']['per_H']}  bnb: {summary['timing']['bnb']}")

        progress(5, 6, "nonvacuity_diag")
        cps = [1000, 3000, 6000, 20000, 60000]
        with Pool(args.workers) as pool:
            nv = [r for x in pool.map(_nonvac_job, [(s_, cps, 0.02) for s_ in DEV_SEEDS[:4]]) for r in x]
        with open(out_dir / "nonvacuity_diag.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(nv[0].keys()))
            w.writeheader()
            w.writerows(nv)
        summary["nonvacuity_diag"] = {str(T): {k: float(np.mean([r[k] for r in nv if r["rounds"] == T]))
                                               for k in ("median_cell_width", "median_J_width", "frac_certified",
                                                         "n_wrong", "median_true_top2_gap")} for T in cps}
        summary["nonvacuity_note"] = ("诊断（非门）：均匀随机行为策略、R1 dev 实例 800-803、eps=0.02、delta=0.05、48 格并集；"
                                      "未用 DDA/主动采集，也未用 BnB。用于提前评估 G-fac 的完成率风险。")
        log(f"nonvacuity: {summary['nonvacuity_diag']}")

        ut, bv = summary["unit_tests"], summary["bound_validity"]
        summary["gate"] = {"all_unit_tests_pass": ut["all_pass"], "bound_violations": bv["violations"],
                           "contract_table_complete": summary["contract_complete"],
                           "contract_all_consistent_with_analytic": summary["contract_all_consistent"],
                           "pass": bool(ut["all_pass"] and bv["violations"] == 0 and summary["contract_complete"])}
        summary["go_no_go"] = "GO" if summary["gate"]["pass"] else "NO_GO"
        summary["wall_s"] = round(time.time() - t_start, 1)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
        progress(6, 6, "done", {"gate_pass": summary["gate"]["pass"]})
        mark_done("success", f"{args.mode} gate_pass={summary['gate']['pass']}")
        log(f"done gate={summary['gate']}")
    except Exception:
        tb = traceback.format_exc()
        log(tb)
        summary["error"] = tb
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
        mark_done("failed", tb[-500:])
        raise


if __name__ == "__main__":
    main()
