"""tier2_e2_scale: E2 (6x6, H=8, continuous persistent parameters) Monte-Carlo scale-up. EMPIRICAL ONLY.

Usage: run_tier2_e2_scale.py --mode {pilot,full} [--workers 3] [--instances N] [--problems K]

Per instance (= statistical unit): theta* in R^32 drawn from the generator box, shared n0 initial rounds, a Type-1
stream of K problems processed in order with persistent evidence (JPC, B3) or trial history (B1).
Methods (each on its own copy of the platform, same seeds -> common random numbers):
  JPC   continuous LR set Theta_t = {v : NLL(v) - NLL(v_hat) <= chi2_{d,1-delta_model}/2}; certificate over a
        decision-directed sparse product grid inside Theta_t (5 super-blocks x {0, 1} x boundary step along the
        linearised worst direction of the top-3 challengers; boundary point + leave-one-block-out points); per-member MC values with CRN + empirical-Bernstein
        envelope (staged: screen all members, re-estimate only still-ambiguous ones with 4x more episodes per
        stage); certificate value = copy-dual + BnB
        bound on the MC point estimates + the largest MC radius. DDA probes: one-round KL from v_hat to the
        top-m binding members, per real step at the current platform state.
  B1    whole-trial LUCB with Hoeffding anytime CS (pre-registered B1); B1eb: same with empirical-Bernstein CS.
  B3    GLM / linearised transductive design: Wald ellipsoid (same chi2 level), MC values + score-function
        gradients at v_hat, MC envelope on the value differences, greedy transductive design per step.
Truth: J_theta*(pi) estimated with 1e6 rollouts per policy (SE reported; NOT exact). False certification is judged
against these estimates.
Caveats (reported in summary.json): the LR radius is the fixed-n Wilks value (not anytime-valid); the product grid
is an INNER approximation of the continuous Theta_t, so R_bar is not a guaranteed upper bound over Theta_t; the MC
envelope is valid per member conditional on the grid.
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse
import itertools
import json
import math
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from scipy.stats import chi2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
WS = HERE.parents[1]

from dsswm.baselines.whole_trial_bai import lucb  # noqa: E402
from dsswm.certify.dual_bnb import bnb_bound, dual_bound  # noqa: E402
from dsswm.models import e2_model as M  # noqa: E402
from dsswm.stats.cp import clopper_pearson  # noqa: E402
from dsswm.stats.km_rmst import kaplan_meier, rmst  # noqa: E402
from dsswm.streams.e2_generator import E2_DEFAULTS, e2_generator_hash, make_e2_instance, random_action  # noqa: E402

TASK = "tier2_e2_scale"
DEV = "cuda"
DELTA = 0.05
DELTA_MODEL = 0.04
DELTA_MC = 0.01
EPS = 0.005          # E2 epsilon on the [0,1] utility scale; calibrated on dev seeds 500-511 (see summary)
TOP_M = 5
N_GRAD = 16384       # MC episodes for v_hat values / gradients (JPC directions)
N_STAGES = (4096, 16384, 65536, 262144, 1 << 20)   # JPC staged MC: episodes per member at each stage
N_REFINE = (10**9, 10**9, 16, 8, 4)                 # max members (re)estimated at each stage (by current UB)
N1, N2 = N_STAGES[0], N_STAGES[-1]
N_B3 = 65536          # B3 MC episodes at v_hat
N_TRUE = 1_000_000
GRID_VALUES = (0.0, 1.0)
N_DIR = 3
SAMPLE_ALL = False   # wrappers (run_e2_anchor.py) set True to keep JPC traces for any instance seed
R_LR = 0.5 * chi2.ppf(1 - DELTA_MODEL, M.D)
BETA_WALD = math.sqrt(chi2.ppf(1 - DELTA_MODEL, M.D))


# ------------------------------------------------------------------ scheduler protocol
def progress(res_root, step, total, phase, metric=None):
    (res_root / f"{TASK}_PROGRESS.json").write_text(json.dumps({
        "task_id": TASK, "epoch": step, "total_epochs": total, "step": step, "total_steps": total, "loss": None,
        "metric": {"phase": phase, **(metric or {})}, "updated_at": datetime.now().isoformat()}))


def mark_done(res_root, status, summary):
    pid = res_root / f"{TASK}.pid"
    if pid.exists():
        pid.unlink()
    pf = res_root / f"{TASK}_PROGRESS.json"
    fp = {}
    if pf.exists():
        try:
            fp = json.loads(pf.read_text())
        except ValueError:
            pass
    (res_root / f"{TASK}_DONE").write_text(json.dumps({"task_id": TASK, "status": status, "summary": summary,
                                                        "final_progress": fp, "timestamp": datetime.now().isoformat()}))


# ------------------------------------------------------------------ helpers
def eb_radius(var, n, delta_row, u_range=2.0):
    """Maurer-Pontil empirical-Bernstein radius (two-sided) for a mean of n iid values in a range u_range."""
    lg = np.log(4.0 / delta_row)
    return np.sqrt(2.0 * np.maximum(var, 0.0) * lg / n) + 7.0 * u_range * lg / (3.0 * (n - 1))


def delta_call(delta_total, j):
    """Anytime allocation over certification calls j = 1, 2, ... within an instance stream."""
    return 6.0 * delta_total / (math.pi ** 2 * j * j)


def structured_pool(q):
    """Public, problem-specific structured probes: affinity-greedy matching and the 6 rotations, each with no
    incentive or an incentive on one matched pair."""
    from dsswm.streams.e2_policies import _greedy_np
    bases = [_greedy_np(q.aff, 6).astype(bool)]
    for off in range(6):
        mt = np.zeros((6, 6), bool)
        for i in range(6):
            mt[i, (i + off) % 6] = True
        bases.append(mt)
    acts = []
    for mt in bases:
        acts.append((mt, np.zeros((6, 6), np.int64)))
        for i, j in zip(*np.nonzero(mt)):
            inc = np.zeros((6, 6), np.int64); inc[i, j] = 1
            acts.append((mt, inc))
    return np.stack([a[0] for a in acts]), np.stack([a[1] for a in acts])


def candidate_actions(rng, pool, n_rand=48):
    acts = [random_action(rng) for _ in range(n_rand)]
    mats = np.concatenate([pool[0], np.stack([a[0] for a in acts])])
    incs = np.concatenate([pool[1], np.stack([a[1] for a in acts])])
    return mats, incs


class Clock:
    def __init__(self):
        self.t = {}

    def add(self, k, s):
        self.t[k] = self.t.get(k, 0.0) + s


# ------------------------------------------------------------------ JPC certificate
def nll_points(V: np.ndarray, data) -> np.ndarray:
    out = []
    for s in range(0, len(V), 512):
        out.append(M.nll_batch(torch.tensor(V[s:s + 512], device=DEV, dtype=torch.float64), data).cpu().numpy())
    return np.concatenate(out)


def boundary_step(v_hat, u, nll0, data):
    """Largest s in (0, 2] on a grid with NLL(clip(v_hat + s u)) - nll0 <= R_LR."""
    ss = np.linspace(0.0, 2.0, 41)[1:]
    pts = np.clip(v_hat[None] + ss[:, None] * u[None], M.LO, M.HI)
    ok = (nll_points(pts, data) - nll0) <= R_LR
    if not ok.any():
        return 0.025
    # first violation bounds the feasible prefix
    bad = np.flatnonzero(~ok)
    last = (bad[0] - 1) if len(bad) else len(ss) - 1
    return float(ss[max(last, 0)])


def build_grid(v_hat, nll0, data, G, Jh, k_hat, Vinv):
    """Decision-directed product grids (one per top challenger direction). Returns members (Mm, d), digits per grid."""
    lin = []
    for k in range(len(Jh)):
        if k == k_hat:
            lin.append(-np.inf); continue
        g = G[k] - G[k_hat]
        lin.append(Jh[k] - Jh[k_hat] + math.sqrt(2 * R_LR) * math.sqrt(max(float(g @ Vinv @ g), 0.0)))
    order = [int(k) for k in np.argsort(lin)[::-1] if k != k_hat][:N_DIR]
    members, grids = [v_hat[None]], []
    nb = len(M.SUPER_BLOCKS)
    # sparse product grid: the boundary point (all super-blocks moved) and the nb leave-one-block-out points
    combos = np.array([[1] * nb] + [[0 if b == c else 1 for b in range(nb)] for c in range(nb)])
    vals = np.array(GRID_VALUES)
    for k in order:
        g = G[k] - G[k_hat]
        nrm = math.sqrt(max(float(g @ Vinv @ g), 1e-18))
        u = math.sqrt(2 * R_LR) * (Vinv @ g) / nrm
        s = boundary_step(v_hat, u, nll0, data)
        step = s * u
        pts = np.repeat(v_hat[None], len(combos), 0)
        for bi, blk in enumerate(M.SUPER_BLOCKS):
            pts[:, blk] += vals[combos[:, bi]][:, None] * step[blk][None]
        pts = np.clip(pts, M.LO, M.HI)
        inside = (nll_points(pts, data) - nll0) <= R_LR
        start = sum(len(m) for m in members)
        members.append(pts[inside])
        grids.append({"challenger": k, "rows": np.arange(start, start + int(inside.sum())), "digits": combos[inside],
                      "step_scale": s, "lin_ub": float(lin[k])})
    return np.concatenate(members), grids, lin


def mc_members(Vm: np.ndarray, q, N: int, gen: torch.Generator):
    """Per-member per-policy per-participant mean contributions (Mm, K, P) and per-episode totals (Mm, K, N)."""
    U = torch.rand((N, q.H, M.L, M.R), device=DEV, generator=gen)
    E = torch.rand((N, q.H, M.P), device=DEV, generator=gen)
    Vt = torch.tensor(Vm, device=DEV, dtype=torch.float32)
    Cm, Tot = [], []
    for pol in q.policies:
        c = M.rollout_contrib(Vt, pol, q.loads0, q.engaged0, q.H, U, E, q.utility.w, q.utility.w_ret, q.utility.c_q)
        Cm.append(c.mean(1).double().cpu().numpy()); Tot.append(c.sum(2))
        del c
    return np.stack(Cm, 1), torch.stack(Tot, 1)


def diff_stats(Tot: torch.Tensor, k_hat: int):
    X = (Tot - Tot[:, k_hat:k_hat + 1]).double()
    n = X.shape[2]
    mu = X.mean(2).cpu().numpy()
    var = X.var(2, unbiased=True).cpu().numpy()
    return mu, var, n


def jpc_certify(q, data, v_hat, nll0, call_j, gen, clock, want_grid=True):
    t0 = time.perf_counter()
    Jh, G, _ = M.values_and_grads(v_hat, q.policies, q.loads0, q.engaged0, q.H, q.utility.w, q.utility.w_ret,
                                  q.utility.c_q, N_GRAD, gen, device=DEV)
    k_hat = int(np.argmax(Jh))
    V = M.fisher_total(v_hat, data)
    Vinv = np.linalg.inv(V)
    clock.add("jpc_grad", time.perf_counter() - t0)
    t0 = time.perf_counter()
    Vm, grids, lin = build_grid(v_hat, nll0, data, G, Jh, k_hat, Vinv)
    clock.add("jpc_grid", time.perf_counter() - t0)
    K = len(q.policies)
    Mm = len(Vm)
    dj = delta_call(DELTA_MC, call_j)
    t0 = time.perf_counter()
    rows_n = max(Mm * (K - 1), 1)
    d_stage = (dj / len(N_STAGES)) / rows_n           # union over stages x members x challengers
    mu = np.zeros((Mm, K)); rad = np.full((Mm, K), np.inf); Cmean = np.zeros((Mm, K, M.P))
    rad[:, k_hat] = 0.0
    n_used = np.zeros(Mm, dtype=np.int64)
    rollouts = K * N_GRAD
    active = np.arange(Mm)
    stage_log = []
    for st, (Ns, cap) in enumerate(zip(N_STAGES, N_REFINE)):
        if len(active) == 0:
            break
        active = active[:cap]
        Cs, Ts = mc_members(Vm[active], q, Ns, gen)
        mus, vars_, ns = diff_stats(Ts, k_hat)
        del Ts
        rads = eb_radius(vars_, ns, d_stage)
        rads[:, k_hat] = 0.0
        mu[active], rad[active], Cmean[active] = mus, rads, Cs
        n_used[active] = Ns
        rollouts += len(active) * K * Ns
        ub_s = mu + rad; ub_s[:, k_hat] = -np.inf
        lb_s = mu - rad; lb_s[:, k_hat] = -np.inf
        stage_log.append((Ns, int(len(active))))
        if lb_s.max() > EPS:          # a member confidently violates eps: certificate impossible on this data
            break
        mem_ub = ub_s.max(1)
        amb = [int(m) for m in np.argsort(-mem_ub) if mem_ub[m] > EPS]
        active = np.array(amb, dtype=np.int64)
    refined = n_used > N_STAGES[0]
    cand = np.flatnonzero(refined)
    clock.add("jpc_mc", time.perf_counter() - t0)
    t0 = time.perf_counter()
    ub = mu + rad
    ub[:, k_hat] = -np.inf
    lb = mu - rad
    lb[:, k_hat] = -np.inf
    r_exact = float(max(0.0, ub.max()))
    r_lb = float(max(0.0, lb.max()))
    bi = np.unravel_index(int(np.argmax(ub)), ub.shape)
    rad_bind_exact = float(rad[bi])
    # copy dual + BnB per (grid, challenger) on the MC point estimates; the MC radius is added on top
    r_dual, r_rect, n_lp, dual_rows = 0.0, 0.0, 0, []
    rad_bind_dual = 0.0
    radices = [len(GRID_VALUES)] * len(M.SUPER_BLOCKS)
    for gi in grids:
        rows = gi["rows"]
        if len(rows) == 0:
            continue
        dg = gi["digits"]
        for k in range(K):
            if k == k_hat:
                continue
            Dk = Cmean[rows, k, :] - Cmean[rows, k_hat, :]          # (n_rows, P) participant attribution
            # fold each row's own MC radius into one attribution column: row objective = mu_row + rad_row exactly,
            # so the copy dual bounds max_rows(mu + rad) directly (no "max rad of another row" slack)
            Dk = Dk.copy()
            Dk[:, 0] += rad[rows, k]
            d = dual_bound(Dk, dg, radices)
            bound, inc = d["bound"], d["incumbent"]
            n_lp += 1
            if bound > inc + 1e-12 and bound > EPS:
                width = [np.ptp([Dk[dg[:, b] == v].sum(1).max() for v in np.unique(dg[:, b])]) for b in range(dg.shape[1])]
                hubs = [int(b) for b in np.argsort(width)[::-1][:2] if len(np.unique(dg[:, b])) > 1]
                if hubs:
                    bb = bnb_bound(Dk, dg, radices, hubs, inc)
                    bound = min(bound, bb["bound"])
                    n_lp += bb["n_lp"]
            # sound for the grid by weak duality: max_rows(mu + rad) <= bound
            mrad = float(rad[rows, k][int(np.argmax(Dk.sum(1)))])     # radius of the row attaining the incumbent
            val = bound
            dual_rows.append({"grid_challenger": gi["challenger"], "k": k, "dual_bnb": bound, "rect": d["rect"],
                              "incumbent": inc, "max_rad": mrad})
            if val > r_dual:
                r_dual, rad_bind_dual = val, mrad
            r_rect = max(r_rect, d["rect"])
    # rows outside any grid (member 0 = v_hat) enter exactly
    k0 = int(np.argmax(ub[0]))
    if ub[0, k0] > r_dual:
        r_dual, rad_bind_dual = float(ub[0, k0]), float(rad[0, k0])
    clock.add("jpc_dual", time.perf_counter() - t0)
    if r_dual <= EPS:
        status = "CERTIFIED"
    elif r_exact > EPS or r_lb > EPS:
        status = "NEED_DATA"
    else:
        status = "COMPUTE_UNKNOWN"
    # binding members for DDA: top-m distinct members by excess over eps
    exc = (ub - EPS).max(1)
    bind = [int(m) for m in np.argsort(-exc) if exc[m] > 0][:TOP_M]
    return {"status": status, "pi": k_hat, "r_bar": r_dual, "r_bar_exact": r_exact, "r_bar_rect": r_rect,
            "r_lb": r_lb, "rad_bind": rad_bind_dual, "rad_bind_exact": rad_bind_exact,
            "mc_share": min(1.0, rad_bind_dual / r_dual) if r_dual > 0 else None,
            "mc_share_exact": min(1.0, rad_bind_exact / r_exact) if r_exact > 0 else None,
            "n_members": Mm, "n_refined": int(refined.sum()), "n_lp": n_lp, "bind_members": Vm[bind],
            "bind_w": exc[bind], "lin_ub_max": float(max(lin)), "Jhat": Jh.tolist(), "rollouts": int(rollouts),
            "grid_steps": [g["step_scale"] for g in grids], "exact_cert": r_exact <= EPS, "mc_stages": stage_log}


# ------------------------------------------------------------------ instance context / scoring
class Ctx:
    def __init__(self, seed, noise, n_problems):
        self.seed, self.noise, self.K = seed, noise, n_problems
        base = make_e2_instance(seed, noise_seed=noise, device=DEV, K=n_problems)
        self.problems = base.problems
        self.Jtrue, self.SEtrue = [], []
        for k, q in enumerate(self.problems):
            J, se = base.env.true_values(q.policies, q.loads0, q.engaged0, q.H, q.utility.w, q.utility.w_ret,
                                         q.utility.c_q, n=N_TRUE, seed=10_000 * seed + k)
            self.Jtrue.append(J); self.SEtrue.append(se)
        self.v_true = base.env.true_vector()                        # harness only (theta*-in-set diagnostic)

    def fresh(self):
        return make_e2_instance(self.seed, noise_seed=self.noise, device=DEV, K=self.K)


def score_row(ctx, k, method, status, pi, steps, censored, extra):
    J, se = ctx.Jtrue[k], ctx.SEtrue[k]
    reg = float(J.max() - J[pi]) if pi is not None else None
    reg_se = float(math.sqrt(se[pi] ** 2 + se[int(np.argmax(J))] ** 2)) if pi is not None else None
    cert = status == "CERTIFIED"
    row = {"instance": ctx.seed, "noise_seed": ctx.noise, "problem_index": k, "ptype": 1, "method": method,
           "status": status, "new_env_steps": int(steps), "censored": bool(censored), "certified_policy": pi,
           "true_regret": reg, "true_regret_se": reg_se, "false_cert": bool(cert and reg is not None and reg > EPS),
           "false_cert_2se": bool(cert and reg is not None and reg - 2 * (reg_se or 0) > EPS), "eps": EPS,
           "delta": DELTA, "J_true_gap12": float(np.sort(J)[-1] - np.sort(J)[-2])}
    row.update(extra)
    return row


def next_batch(steps):
    return max(8, int(math.ceil(0.25 * steps)))


def run_jpc(ctx, tmax, samples=None, log=None):
    inst = ctx.fresh()
    env = inst.env
    data = M.E2Data()
    for o in inst.init_obs:
        data.add(o)
    gen = torch.Generator(device=DEV); gen.manual_seed(1_000_003 * ctx.seed + ctx.noise)
    rng = np.random.default_rng([ctx.seed, ctx.noise, 101])
    v_hat, nll0 = M.fit_mle(data, device="cpu")
    call_j = 0
    rows = []
    for k, q in enumerate(ctx.problems):
        clock = Clock()
        t0 = time.perf_counter()
        steps, calls, unknown_calls, rollouts, n_explore = 0, 0, 0, 0, 0
        first_exact = None
        traj = []
        pool = structured_pool(q)
        while True:
            call_j += 1; calls += 1
            cert = jpc_certify(q, data, v_hat, nll0, call_j, gen, clock)
            rollouts += cert["rollouts"]
            unknown_calls += int(cert["status"] == "COMPUTE_UNKNOWN")
            if cert["exact_cert"] and first_exact is None:
                first_exact = steps
            theta_in = bool(float(nll_points(ctx.v_true[None], data)[0]) - nll0 <= R_LR)
            traj.append({"steps": steps, "status": cert["status"], "r_bar": cert["r_bar"],
                         "r_bar_exact": cert["r_bar_exact"], "r_lb": cert["r_lb"], "rad_bind": cert["rad_bind"],
                         "pi": cert["pi"], "n_members": cert["n_members"], "n_refined": cert["n_refined"],
                         "lin_ub_max": cert["lin_ub_max"], "theta_star_in_LR_set": theta_in,
                         "grid_steps": cert["grid_steps"], "mc_stages": cert["mc_stages"]})
            if cert["status"] == "CERTIFIED" or steps >= tmax:
                break
            nb = min(next_batch(steps), tmax - steps)
            t1 = time.perf_counter()
            Valt, wts = cert["bind_members"], np.maximum(cert["bind_w"], 1e-6)
            for _ in range(nb):
                loads, eng = env.observable_state()
                if rng.random() < 0.05 or len(Valt) == 0:
                    mt, inc = random_action(rng); n_explore += 1
                else:
                    mats, incs = candidate_actions(rng, pool)
                    kl = M.round_kl(v_hat, Valt, loads, eng, mats, incs)          # (A, m)
                    a = int(np.argmax(kl @ wts))
                    mt, inc = mats[a], incs[a]
                data.add(env.step(mt, inc))
                steps += 1
            clock.add("jpc_dda", time.perf_counter() - t1)
            t1 = time.perf_counter()
            v_hat, nll0 = M.fit_mle(data, device="cpu", init=v_hat)
            clock.add("mle", time.perf_counter() - t1)
        assert env.n_steps == data.n_rounds, "evidence must contain exactly the real platform rounds"
        status = cert["status"]
        extra = {"r_bar_end": cert["r_bar"], "r_bar_exact_end": cert["r_bar_exact"], "r_bar_rect_end": cert["r_bar_rect"],
                 "r_lb_end": cert["r_lb"], "mc_rad_bind_end": cert["rad_bind"], "mc_error_share": cert["mc_share"],
                 "mc_error_share_exact": cert["mc_share_exact"], "n_calls": calls, "compute_unknown_calls": unknown_calls,
                 "first_exact_cert_steps": first_exact, "n_members_end": cert["n_members"],
                 "n_refined_end": cert["n_refined"], "n_lp_end": cert["n_lp"], "lin_ub_end": cert["lin_ub_max"],
                 "theta_star_in_LR_set_end": traj[-1]["theta_star_in_LR_set"],
                 "zero_cost_cert": bool(steps == 0 and status == "CERTIFIED"), "explore_steps": n_explore,
                 "wall_clock_s": time.perf_counter() - t0, "time_split_s": {a: round(b, 3) for a, b in clock.t.items()},
                 "rollouts": int(rollouts), "H": q.H, "n_policies": len(q.policies), "total_rounds": data.n_rounds}
        rows.append(score_row(ctx, k, "JPC", status, cert["pi"], steps, status != "CERTIFIED", extra))
        if samples is not None and (ctx.seed < 3 or SAMPLE_ALL) and k < 3:
            samples.append({"instance": ctx.seed, "problem": q.pid, "policies": [p.name for p in q.policies],
                            "J_true": ctx.Jtrue[k].tolist(), "J_true_se": ctx.SEtrue[k].tolist(),
                            "Jhat_end": cert["Jhat"], "status": status, "steps": steps, "calls": traj})
        if log:
            log(f"  inst {ctx.seed} q{k} JPC {status} steps={steps} calls={calls} rbar={cert['r_bar']:.4f} "
                f"exact={cert['r_bar_exact']:.4f} share={cert['mc_share'] if cert['mc_share'] is None else round(cert['mc_share'], 3)} "
                f"t={time.perf_counter() - t0:.1f}s")
    return rows


def run_b3(ctx, tmax, log=None):
    inst = ctx.fresh()
    env = inst.env
    data = M.E2Data()
    for o in inst.init_obs:
        data.add(o)
    gen = torch.Generator(device=DEV); gen.manual_seed(2_000_003 * ctx.seed + ctx.noise)
    rng = np.random.default_rng([ctx.seed, ctx.noise, 303])
    v_hat, nll0 = M.fit_mle(data, device="cpu")
    call_j = 0
    rows = []
    for k, q in enumerate(ctx.problems):
        t0 = time.perf_counter()
        steps, calls, rollouts = 0, 0, 0
        K = len(q.policies)
        pool = structured_pool(q)
        while True:
            call_j += 1; calls += 1
            Jh, G, Us = M.values_and_grads(v_hat, q.policies, q.loads0, q.engaged0, q.H, q.utility.w,
                                           q.utility.w_ret, q.utility.c_q, N_B3, gen, device=DEV)
            rollouts += K * N_B3
            kh = int(np.argmax(Jh))
            X = (Us - Us[kh:kh + 1]).double()
            var = X.var(1).cpu().numpy()
            rad = eb_radius(var, X.shape[1], delta_call(DELTA_MC, call_j) / max(K - 1, 1))
            V = M.fisher_total(v_hat, data)
            Vinv = np.linalg.inv(V)
            Dg = G - G[kh][None]
            widths = np.sqrt(np.maximum(np.einsum("kd,de,ke->k", Dg, Vinv, Dg), 0.0))
            ub = Jh - Jh[kh] + BETA_WALD * widths + rad
            ub[kh] = -np.inf
            r_bar = float(max(0.0, ub.max()))
            if r_bar <= EPS or steps >= tmax:
                break
            blk = Dg[ub > EPS]
            nb = min(next_batch(steps), tmax - steps)
            Vc = V.copy()
            for _ in range(nb):
                loads, eng = env.observable_state()
                if rng.random() < 0.05:
                    mt, inc = random_action(rng)
                else:
                    mats, incs = candidate_actions(rng, pool)
                    best, bv = 0, np.inf
                    for a in range(len(mats)):
                        Fa = M.fisher_terms(v_hat, loads, eng, mats[a], incs[a])
                        Vi = np.linalg.inv(Vc + Fa)
                        val = float(np.max(np.einsum("kd,de,ke->k", blk, Vi, blk)))
                        if val < bv - 1e-15:
                            best, bv = a, val
                    mt, inc = mats[best], incs[best]
                Vc += M.fisher_terms(v_hat, loads, eng, mt, inc)
                data.add(env.step(mt, inc))
                steps += 1
            v_hat, nll0 = M.fit_mle(data, device="cpu", init=v_hat)
        status = "CERTIFIED" if r_bar <= EPS else "NEED_DATA"
        extra = {"r_bar_end": r_bar, "mc_rad_end": float(rad[np.argmax(ub)]) if K > 1 else 0.0,
                 "mc_error_share": (float(rad[np.argmax(ub)]) / r_bar) if r_bar > 0 else None, "n_calls": calls,
                 "zero_cost_cert": bool(steps == 0 and status == "CERTIFIED"), "wall_clock_s": time.perf_counter() - t0,
                 "rollouts": int(rollouts), "H": q.H, "n_policies": K, "beta_wald": BETA_WALD}
        rows.append(score_row(ctx, k, "B3", status, kh, steps, status != "CERTIFIED", extra))
        if log:
            log(f"  inst {ctx.seed} q{k} B3 {status} steps={steps} rbar={r_bar:.4f} t={time.perf_counter() - t0:.1f}s")
    return rows


def run_trials(ctx, tmax_trial, kind, log=None):
    inst = ctx.fresh()
    env = inst.env                                  # harness: the trial sampler IS the platform
    method = "B1" if kind == "hoeffding" else "B1eb"
    rows = []
    for k, q in enumerate(ctx.problems):
        t0 = time.perf_counter()

        def sampler(a, n, q=q):
            return env.true_utilities(q.policies[a], q.loads0, q.engaged0, q.H, int(n), q.utility.w,
                                      q.utility.w_ret, q.utility.c_q)
        res = lucb(sampler, len(q.policies), q.H, EPS, DELTA, 1.0, tmax_trial, kind=kind)
        extra = {"wall_clock_s": time.perf_counter() - t0, "rollouts": int(res["pulls"]), "trials": int(res["pulls"]),
                 "H": q.H, "n_policies": len(q.policies), "final_gap": res.get("gap"),
                 "zero_cost_cert": bool(res["steps"] == 0 and res["status"] == "CERTIFIED"),
                 "history_reuse": "none (s0 differs across problems in E2)"}
        rows.append(score_row(ctx, k, method, res["status"], res["pi"], res["steps"], res["censored"], extra))
        if log:
            log(f"  inst {ctx.seed} q{k} {method} {res['status']} steps={res['steps']} t={time.perf_counter() - t0:.1f}s")
    return rows


def run_instance(seed, noise, n_problems, tmax, tmax_trial, methods, out_dir):
    torch.set_num_threads(1)
    logf = open(out_dir / f"worker_{seed}_{noise}.log", "a")

    def log(msg):
        logf.write(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n"); logf.flush()

    t0 = time.time()
    rows, samples, err = [], [], None
    try:
        ctx = Ctx(seed, noise, n_problems)
        log(f"inst {seed} truth done {time.time() - t0:.1f}s")
        for m in methods:
            tm = time.time()
            if m == "JPC":
                rows += run_jpc(ctx, tmax, samples=samples, log=log)
            elif m == "B3":
                rows += run_b3(ctx, tmax, log=log)
            elif m == "B1":
                rows += run_trials(ctx, tmax_trial, "hoeffding", log=log)
            elif m == "B1eb":
                rows += run_trials(ctx, tmax_trial, "eb", log=log)
            log(f"inst {seed} {m} done {time.time() - tm:.1f}s")
    except Exception:
        err = traceback.format_exc()
        log(err)
    torch.cuda.empty_cache()
    return {"seed": seed, "noise": noise, "rows": rows, "samples": samples, "error": err,
            "wall_s": time.time() - t0, "max_mem_gb": torch.cuda.max_memory_allocated() / 1e9}


# ------------------------------------------------------------------ summary
def summarize(rows, methods, tmax_common, tmax_step):
    out = {}
    for m in methods:
        rs = [r for r in rows if r["method"] == m]
        if not rs:
            continue
        steps = np.array([min(r["new_env_steps"], tmax_common) for r in rs], float)
        cens = np.array([r["censored"] or r["new_env_steps"] > tmax_common for r in rs])
        n_cert = int(sum(r["status"] == "CERTIFIED" for r in rs))
        n_false = int(sum(r["false_cert"] for r in rs))
        lo, hi = clopper_pearson(n_false, n_cert) if n_cert else (0.0, 1.0)
        regs = [r["true_regret"] for r in rs if r["true_regret"] is not None]
        s_step = np.minimum(steps, tmax_step)
        c_step = cens | (steps > tmax_step)
        d = {"n": len(rs), "rmst_new_env_steps": rmst(steps, cens, tmax_common),
             "rmst_new_env_steps_tau_step": rmst(s_step, c_step, tmax_step),
             "completion_rate": float(np.mean(~cens)), "mean_steps": float(steps.mean()),
             "median_steps_uncensored": float(np.median(steps[~cens])) if (~cens).any() else None,
             "n_certified": n_cert, "n_false_cert": n_false,
             "n_false_cert_2se": int(sum(r["false_cert_2se"] for r in rs)),
             "fcr": n_false / n_cert if n_cert else None, "fcr_cp_upper": hi,
             "true_regret_mean": float(np.mean(regs)) if regs else None,
             "true_regret_max": float(np.max(regs)) if regs else None,
             "zero_cost_cert_frac": float(np.mean([bool(r.get("zero_cost_cert")) for r in rs])),
             "wall_clock_s_total": float(sum(r.get("wall_clock_s", 0.0) for r in rs)),
             "wall_clock_s_per_problem_median": float(np.median([r.get("wall_clock_s", 0.0) for r in rs])),
             "rollouts_total": int(sum(r.get("rollouts", 0) for r in rs)),
             "status_counts": {s: int(sum(r["status"] == s for r in rs)) for s in
                               ("CERTIFIED", "NEED_DATA", "COMPUTE_UNKNOWN", "OUT_OF_SCOPE", "MODEL_CONFLICT")}}
        d["compute_unknown_rate"] = d["status_counts"]["COMPUTE_UNKNOWN"] / len(rs)
        if m == "JPC":
            calls = sum(r["n_calls"] for r in rs)
            d["compute_unknown_call_rate"] = sum(r["compute_unknown_calls"] for r in rs) / max(calls, 1)
            d["n_calls_total"] = calls
            sh = [r["mc_error_share"] for r in rs if r.get("mc_error_share") is not None]
            shc = [r["mc_error_share"] for r in rs if r.get("mc_error_share") is not None and r["status"] == "CERTIFIED"]
            d["mc_error_share_mean"] = float(np.mean(sh)) if sh else None
            d["mc_error_share_median"] = float(np.median(sh)) if sh else None
            d["mc_error_share_certified_mean"] = float(np.mean(shc)) if shc else None
            d["mc_error_share_certified_max"] = float(np.max(shc)) if shc else None
            d["mc_error_share_certified_median"] = float(np.median(shc)) if shc else None
            d["mc_share_saturated_frac_certified"] = float(np.mean([x >= 0.999 for x in shc])) if shc else None
            alt = [r["mc_rad_bind_end"] / (abs(r["r_bar_end"] - r["mc_rad_bind_end"]) + r["mc_rad_bind_end"])
                   for r in rs if r["status"] == "CERTIFIED" and r["mc_rad_bind_end"] > 0]
            d["mc_share_alt_abs_certified_mean"] = float(np.mean(alt)) if alt else None   # rad / (|point| + rad)
            d["mc_rad_bind_certified_median"] = float(np.median([r["mc_rad_bind_end"] for r in rs
                                                                 if r["status"] == "CERTIFIED"])) if shc else None
            fe = [r["first_exact_cert_steps"] for r in rs]
            d["exact_grid_would_certify_earlier"] = int(sum(1 for r, f in zip(rs, fe) if f is not None and
                                                            (r["censored"] or f < r["new_env_steps"])))
            d["theta_star_in_LR_set_rate_end"] = float(np.mean([r["theta_star_in_LR_set_end"] for r in rs]))
            ts = {}
            for r in rs:
                for a, b in r["time_split_s"].items():
                    ts[a] = ts.get(a, 0.0) + b
            d["time_split_s"] = {a: round(b, 1) for a, b in ts.items()}
        if m == "B3":
            sh = [r["mc_error_share"] for r in rs if r.get("mc_error_share") is not None]
            d["mc_error_share_mean"] = float(np.mean(sh)) if sh else None
        out[m] = d
    return out


def paired(rows, add=1.0):
    idx = {}
    for r in rows:
        idx.setdefault((r["instance"], r["noise_seed"], r["problem_index"]), {})[r["method"]] = r
    out = {}
    for b in ("B1", "B1eb", "B3"):
        v = [(d["JPC"]["new_env_steps"] + add) / (d[b]["new_env_steps"] + add) for d in idx.values() if "JPC" in d and b in d]
        if v:
            v = np.array(v)
            out[f"JPC_over_{b}"] = {"median": float(np.median(v)), "q25": float(np.quantile(v, .25)),
                                    "q75": float(np.quantile(v, .75)), "mean_log": float(np.mean(np.log(v))),
                                    "n": len(v), "note": "(steps+1) ratio; censored baselines enter at their cap"}
    # per-instance cumulative steps
    inst = {}
    for (i, nz, k), d in idx.items():
        for m, r in d.items():
            inst.setdefault(m, {}).setdefault((i, nz), 0)
            inst[m][(i, nz)] += r["new_env_steps"]
    out["cum_steps_per_instance_median"] = {m: float(np.median(list(v.values()))) for m, v in inst.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="pilot", choices=["pilot", "full"])
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--instances", type=int, default=None)
    ap.add_argument("--problems", type=int, default=None)
    ap.add_argument("--tmax", type=int, default=3000, help="per-problem cap for per-step methods")
    ap.add_argument("--tmax-trial", type=int, default=20_000_000, help="per-problem cap for whole-trial methods")
    ap.add_argument("--methods", default="JPC,B3,B1,B1eb")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    res_root = WS / "exp" / "results"
    out_dir = res_root / ("pilots" if args.mode == "pilot" else "full") / (TASK + args.tag)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    (res_root / f"{TASK}.pid").write_text(str(os.getpid()))
    start_iso = datetime.now().isoformat()
    (out_dir / "start_time.txt").write_text(start_iso)
    logf = open(out_dir / "run.log", "a")

    def log(msg):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n"); logf.flush()

    n_inst = args.instances or (10 if args.mode == "pilot" else 24)
    n_prob = args.problems or 10
    seeds = list(range(n_inst)) if args.mode == "pilot" else list(range(10000, 10000 + n_inst))
    noises = [42]
    methods = args.methods.split(",")
    log(f"start mode={args.mode} instances={n_inst} problems={n_prob} methods={methods} eps={EPS} delta={DELTA} "
        f"R_LR={R_LR:.2f} beta_wald={BETA_WALD:.2f} N_stages={N_STAGES} tmax={args.tmax} tmax_trial={args.tmax_trial} "
        f"e2_generator_hash={e2_generator_hash()} workers={args.workers} (concurrent with other tasks)")
    jobs = [(s, nz) for nz in noises for s in seeds]
    total = len(jobs)
    progress(res_root, 0, total, "running")
    t_run = time.time()
    results = []
    import multiprocessing as mp
    ctxm = mp.get_context("spawn")
    with ctxm.Pool(args.workers) as pool:
        its = pool.imap_unordered(_job, [(s, nz, n_prob, args.tmax, args.tmax_trial, methods, str(out_dir))
                                         for s, nz in jobs])
        for res in its:
            results.append(res)
            log(f"instance {res['seed']} done wall={res['wall_s']:.0f}s rows={len(res['rows'])} "
                f"mem={res['max_mem_gb']:.2f}GB err={'yes' if res['error'] else 'no'}")
            progress(res_root, len(results), total, "running", {"elapsed_s": round(time.time() - t_run, 1)})
    rows = sorted([r for res in results for r in res["rows"]],
                  key=lambda r: (r["method"], r["instance"], r["noise_seed"], r["problem_index"]))
    errors = [{"seed": r["seed"], "error": r["error"]} for r in results if r["error"]]
    with open(out_dir / "results.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    samples = [s for res in results for s in res["samples"]]
    (out_dir / "samples" / "jpc_traces.json").write_text(json.dumps(samples, indent=1))
    (out_dir / "errors.json").write_text(json.dumps(errors, indent=1))
    # RMST horizon: smallest T with B1 completion >= 50% (as in E1-NL); else the trial cap
    b1 = sorted(r["new_env_steps"] for r in rows if r["method"] == "B1" and not r["censored"])
    n_b1 = sum(1 for r in rows if r["method"] == "B1")
    if n_b1 and len(b1) >= math.ceil(0.5 * n_b1):
        tmax_common, rule = int(b1[math.ceil(0.5 * n_b1) - 1]), "smallest T with B1 completion >= 50%"
    else:
        tmax_common, rule = args.tmax_trial, "B1 completion < 50% even at tmax_trial; T_max = tmax_trial"
    summ = summarize(rows, methods, tmax_common, args.tmax)
    pr = paired(rows)
    j = summ.get("JPC", {})
    share = j.get("mc_error_share_certified_mean")
    end_to_end = bool(not errors and j.get("n") == n_inst * n_prob)
    pass_share = bool(share is not None and share < 0.5)
    go = end_to_end and pass_share
    summary = {
        "task_id": TASK, "mode": args.mode, "env": "E2 (6x6, H=8, continuous theta in R^32)", "eps": EPS,
        "delta": DELTA, "delta_model": DELTA_MODEL, "delta_mc": DELTA_MC, "R_LR": R_LR, "beta_wald": BETA_WALD,
        "n_instances": n_inst, "problems_per_instance": n_prob, "instance_seeds": seeds, "noise_seeds": noises,
        "n0": E2_DEFAULTS["n0"], "e2_generator_hash": e2_generator_hash(), "tmax_step_methods": args.tmax,
        "tmax_trial_methods": args.tmax_trial, "T_max_common": tmax_common, "T_max_rule": rule,
        "mc": {"N_grad": N_GRAD, "N_stages": N_STAGES, "N_refine_caps": [min(c, 10**6) for c in N_REFINE], "N_B3": N_B3, "N_true": N_TRUE,
               "grid_values": GRID_VALUES, "n_directions": N_DIR, "super_blocks": len(M.SUPER_BLOCKS)},
        "methods": summ, "paired": pr,
        "pass_criteria": "pipeline runs end-to-end AND MC error share of R-bar < 50%",
        "pipeline_end_to_end": end_to_end, "mc_error_share_certified_mean_JPC": share, "mc_share_pass": pass_share,
        "mc_error_share_definition": "rad_MC(binding row) / R_bar at the final certificate call, clipped to [0,1]; "
                                     "pass criterion evaluated on CERTIFIED problems (R_bar is the certified bound); "
                                     "NEED_DATA ends stop MC refinement early once a member confidently violates eps, "
                                     "so their R_bar / share are not tight and are reported separately.",
        "go_no_go": "GO" if go else "NO_GO",
        "epsilon_calibration": "eps=0.005 on the [0,1] utility scale: on dev seeds 500-511 (60 problems, n0=40) the "
                               "point-estimate pick has true regret > eps in 17% of problems (10-90% non-triviality "
                               "window); eps=0.02 gives 1.7% (trivial). Policy-value spread median 0.030.",
        "caveats": ["EMPIRICAL ONLY: no exact truth; J_theta* estimated with 1e6 rollouts per policy (SE reported).",
                    "LR radius is the fixed-n Wilks value chi2_{32,0.96}/2 (not anytime-valid); B3 uses the same level.",
                    "JPC certificate is computed over a decision-directed product grid that is an INNER approximation "
                    "of the continuous LR set; R_bar is therefore not a guaranteed upper bound over Theta_t. "
                    "False certifications are checked against the MC truth.",
                    "MC envelope (empirical Bernstein, union over members x challengers, anytime over calls) is "
                    "valid conditional on the grid.",
                    "B3 ignores the linearisation error and gradient MC noise (favourable to B3).",
                    "B1 history re-use is inactive: s0 differs across problems in E2.",
                    "Wall-clock measured with 3 concurrent workers sharing one RTX 4090 with another task."],
        "wall_clock_total_s": time.time() - t_run, "errors": len(errors),
        "worker_max_mem_gb": max([r["max_mem_gb"] for r in results] + [0.0]),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    log(json.dumps({m: {k: v for k, v in d.items() if k in ("rmst_new_env_steps", "completion_rate",
                                                             "compute_unknown_rate", "mc_error_share_mean", "mc_error_share_certified_mean",
                                                             "n_false_cert", "wall_clock_s_total")}
                    for m, d in summ.items()}))
    log(f"GO/NO-GO={summary['go_no_go']} end_to_end={end_to_end} share={share}")
    mark_done(res_root, "success" if not errors else "failed",
              f"E2 pilot {summary['go_no_go']}: JPC completion={j.get('completion_rate')} mc_share={share}")


def _job(a):
    return run_instance(a[0], a[1], a[2], a[3], a[4], a[5].split(",") if isinstance(a[5], str) else a[5], Path(a[6]))


if __name__ == "__main__":
    main()
