"""h2_kappa_distribution: distribution of the cancellation coefficient kappa under the pre-registered generator.

kappa_P(d) = sum_p ||d_p||_{A^+} / ||sum_p d_p||_{A^+}     (A = information matrix of a design xi)

Partitions of the decision difference d = z_{pi*} - z_{pi'} (value-contribution decompositions, all in range(A)):
  time         d_t       = per-step contribution                        (H blocks)
  participant  d_i       = contribution of outcomes of left participant i (L blocks; E1-Lin outcomes are per pair,
                           each pair has exactly one left participant, who carries the dynamic gamma_i)
  participant_R          = same with right participants (secondary)
  policy       {z_{pi*}, -z_{pi'}}  (per-policy marginal intervals, CPE-style)
  both         d_{t,i}   (time x left participant)

Binding pair: argmax_k ||d_k||^2_{A(xi_J)^+} / max(Delta_k, eps)^2 under the joint-optimal oracle design xi_J
(theta* known). Primary kappa = kappa_P(d_bind; A(xi_J)). Envelopes: min / max over {uniform, joint-opt, rect_P-opt}.
E1-NL-S reference: linearisation at the grid MLE theta_hat (n0 initial rounds), Fisher-information designs.

Usage: run_h2_kappa_distribution.py --mode {pilot,full}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
WS = HERE.parents[1]

from dsswm.models.lin_class import LinClass  # noqa: E402
from dsswm.streams.generator import LIN_DEFAULTS, generator_hash, make_lin_instance  # noqa: E402
from dsswm.streams.policies import sample_policy  # noqa: E402

TASK = "h2_kappa_distribution"
PARTS = ("time", "participant", "participant_R", "policy", "both")
EPS_LIN, EPS_NL = 0.05, 0.02


# ----------------------------------------------------------------------------------------------- protocol files
def progress(res_dir, step, total, phase, metric=None):
    (res_dir / f"{TASK}_PROGRESS.json").write_text(json.dumps({
        "task_id": TASK, "epoch": step, "total_epochs": total, "step": step, "total_steps": total, "loss": None,
        "metric": {"phase": phase, **(metric or {})}, "updated_at": datetime.now().isoformat()}))


def mark_done(res_dir, status, summary):
    pid = res_dir / f"{TASK}.pid"
    if pid.exists():
        pid.unlink()
    pf = res_dir / f"{TASK}_PROGRESS.json"
    fp = {}
    if pf.exists():
        try:
            fp = json.loads(pf.read_text())
        except ValueError:
            pass
    (res_dir / f"{TASK}_DONE").write_text(json.dumps({"task_id": TASK, "status": status, "summary": summary,
                                                      "final_progress": fp, "timestamp": datetime.now().isoformat()}))


# ----------------------------------------------------------------------------------------------- design optimiser
class Library:
    """Design atoms. Atom j has information sum_{m in j} w_m g_m g_m^T (g in reduced coordinates)."""

    def __init__(self, G, w, aidx, n_atoms, Q):
        self.G, self.w, self.aidx, self.n, self.Q = G, w, aidx, n_atoms, Q
        self.r = G.shape[1]

    def A(self, xi):
        wx = self.w * xi[self.aidx]
        return (self.G * wx[:, None]).T @ self.G + 1e-12 * np.eye(self.r)


def reduced_basis(rows, tol=1e-9):
    U, s, _ = np.linalg.svd(np.asarray(rows).T, full_matrices=False)
    return U[:, s > tol * s.max()]


def block_norms(A, blocks):
    """blocks: list over challengers of (nb_k, r) arrays -> list of norm arrays and the A^{-1} B products."""
    allB = np.vstack(blocks)
    U = np.linalg.solve(A, allB.T)
    nrm = np.sqrt(np.maximum((allB.T * U).sum(0), 0.0))
    return allB, U, nrm


def objective(lib, xi, blocks, Delta):
    A = lib.A(xi)
    _, _, nrm = block_norms(A, blocks)
    out, s = [], 0
    for B in blocks:
        out.append(nrm[s:s + len(B)].sum())
        s += len(B)
    return (np.array(out) / Delta) ** 2


def optimise(lib, blocks, Delta, iters=1200, xi0=None):
    """min_xi max_k (sum_b ||B_kb||_{A(xi)^{-1}})^2 / Delta_k^2 by entropic mirror descent on a softmax-smoothed max."""
    n = lib.n
    xi = np.full(n, 1.0 / n) if xi0 is None else xi0.copy()
    sizes = [len(B) for B in blocks]
    owner = np.repeat(np.arange(len(blocks)), sizes)
    best_val, best_xi = np.inf, xi.copy()
    for it in range(iters):
        A = lib.A(xi)
        allB, U, nrm = block_norms(A, blocks)
        S = np.bincount(owner, weights=nrm, minlength=len(blocks))
        f = (S / Delta) ** 2
        fm = f.max()
        if fm < best_val:
            best_val, best_xi = fm, xi.copy()
        tau = 0.05 * (0.02 / 0.05) ** (it / iters)
        lam = np.exp((f - fm) / (tau * fm))
        lam /= lam.sum()
        # d f_k / d xi_j = (S_k / Delta_k^2) * sum_b -(sum_{m in j} w_m (g_m.u_b)^2) / ||B_b||
        coef = lam[owner] * (S[owner] / Delta[owner] ** 2) / np.maximum(nrm, 1e-300)
        proj = lib.G @ U                                  # (M, nb)
        gm = -(lib.w[:, None] * proj ** 2) @ coef
        grad = np.bincount(lib.aidx, weights=gm, minlength=n)
        g = grad / max(np.abs(grad).max(), 1e-300)
        eta = 2.0 / np.sqrt(it + 1.0)
        xi = xi * np.exp(-eta * g)
        xi = np.maximum(xi / xi.sum(), 1e-10)
        xi /= xi.sum()
    fm = objective(lib, xi, blocks, Delta).max()
    if fm < best_val:
        best_val, best_xi = fm, xi
    return best_xi, float(best_val)


def kappa_of(A, blocks):
    """kappa = sum_b ||B_b|| / ||sum_b B_b||, all A^{-1} norms."""
    _, _, nrm = block_norms(A, [blocks, blocks.sum(0, keepdims=True)])
    return float(nrm[:-1].sum() / nrm[-1]) if nrm[-1] > 0 else float("nan")


# ----------------------------------------------------------------------------------------------- per-problem core
def partition_blocks(Cs, Cc, part):
    """Cs, Cc: (H, P, r) per-(step, participant-block) contributions of pi* and challenger. Returns (nb, r)."""
    D = Cs - Cc
    L = D.shape[1]
    if part == "time":
        B = D.sum(1)
    elif part == "participant":
        B = D.sum(0)
    elif part == "both":
        B = D.reshape(-1, D.shape[-1])
    elif part == "policy":
        B = np.stack([Cs.sum((0, 1)), -Cc.sum((0, 1))])
    else:
        raise ValueError(part)
    keep = np.abs(B).max(1) > 1e-13
    return B[keep] if keep.any() else B[:1]


def analyse_problem(C_list, J, eps, lib, parts=PARTS, Cr_list=None, iters=1200, want_rect=True):
    """C_list[k]: (H, Pblk, r) contributions; J: true (or plug-in) values. Returns dict or None if degenerate."""
    n = len(C_list)
    k_star = int(np.argmax(J))
    z = [C.sum((0, 1)) for C in C_list]
    ch = [k for k in range(n) if k != k_star and np.abs(z[k_star] - z[k]).max() > 1e-12]
    if not ch:
        return None
    Delta_raw = np.array([J[k_star] - J[k] for k in ch])
    Delta = np.maximum(Delta_raw, eps)
    joint_blocks = [(z[k_star] - z[k])[None] for k in ch]
    xi_u = np.full(lib.n, 1.0 / lib.n)
    xi_J, rho_J = optimise(lib, joint_blocks, Delta, iters=iters)
    A_J, A_U = lib.A(xi_J), lib.A(xi_u)
    fJ = objective(lib, xi_J, joint_blocks, Delta)
    b = int(np.argmax(fJ))
    kb = ch[b]
    out = {"n_policies": n, "n_challengers": len(ch), "pi_star": k_star, "binding": kb,
           "Delta_bind": float(Delta_raw[b]), "Delta_bind_floored": bool(Delta_raw[b] < eps), "rho_joint": rho_J,
           "rho_joint_uniform": float(objective(lib, xi_u, joint_blocks, Delta).max()),
           "xi_J_entropy": float(-(xi_J * np.log(np.maximum(xi_J, 1e-300))).sum())}
    for part in parts:
        if part == "participant_R":
            if Cr_list is None:
                continue
            Cs, Cc = Cr_list[k_star], Cr_list[kb]
            pblocks = [partition_blocks(Cr_list[k_star], Cr_list[k], "participant") for k in ch]
            bb = partition_blocks(Cs, Cc, "participant")
        else:
            Cs, Cc = C_list[k_star], C_list[kb]
            pblocks = [partition_blocks(C_list[k_star], C_list[k], part) for k in ch]
            bb = partition_blocks(Cs, Cc, part)
        kJ, kU = kappa_of(A_J, bb), kappa_of(A_U, bb)
        ks = [kJ, kU]
        if want_rect:
            xi_R, rho_R = optimise(lib, pblocks, Delta, iters=iters, xi0=xi_J)
            kR = kappa_of(lib.A(xi_R), bb)
            ks.append(kR)
            out[f"kappa_{part}_rectopt"] = kR
            out[f"rho_rect_{part}"] = rho_R
            out[f"rho_ratio_{part}"] = rho_R / rho_J
            out[f"kappa_eff_{part}"] = float(np.sqrt(max(rho_R / rho_J, 0.0)))   # operational: sqrt(rho_P/rho_joint)
            # sandwich check over all challengers at their own designs is the H1 task; here the binding-pair form
        out[f"kappa_{part}"] = kJ
        out[f"kappa_{part}_uniform"] = kU
        out[f"kappa0_{part}"] = float(np.nanmin(ks))
        out[f"kappainf_{part}"] = float(np.nanmax(ks))
        out[f"envelope_ratio_{part}"] = float(np.nanmax(ks) / np.nanmin(ks))
        out[f"nblocks_{part}"] = int(len(bb))
    out["_A_J"] = A_J
    out["_A_U"] = A_U
    return out


# ----------------------------------------------------------------------------------------------- E1-Lin
def lin_library(lc, nb_used, static):
    rows, names = [], []
    nlev = [0] if static else range(lc.nmax + 1)
    for i in range(lc.L):
        for j in range(lc.R):
            for n in nlev:
                for b in range(nb_used):
                    rows.append(lc.feature(i, j, n, b))
                    names.append((i, j, n, b))
    return np.array(rows), names


def lin_contrib(lc, policy, loads0, H, util, aspace):
    """(H, L, d) contributions of left participants and (H, R, d) of right participants."""
    from dsswm.core.dynamics import next_loads
    loads = np.asarray(loads0, np.int64).copy()
    ones = np.ones(lc.L + lc.R, dtype=np.int64)
    CL = np.zeros((H, lc.L, lc.d))
    CR = np.zeros((H, lc.R, lc.d))
    for t in range(H):
        a_idx = policy.act(t, loads, ones)
        a = aspace.actions[a_idx]
        inc = {(i, j): l for i, j, l in a.incentives}
        for i, j in a.pairs:
            x = lc.feature(i, j, loads[i], inc.get((i, j), 0)) * util.c_q * util.w[t]
            CL[t, i] += x
            CR[t, j] += x
        loads = next_loads(loads, aspace, a_idx, lc.nmax, lc.static)
    return CL, CR


def contrast_policies(prng, aspace, H, aff, nmax):
    """Common contrasts built on the problem's affinity proxy (deterministic families)."""
    from dsswm.streams.policies import Policy
    mk = lambda fam, params: Policy(fam, params, aspace, H, aff, rng_seed=0)  # noqa: E731
    out = {}
    for base in ("aff", "rot"):
        for frac in (0.25, 0.5):
            out[f"front_vs_late[{base},{frac}]"] = (mk("front_incentive", {"base": base, "frac": frac, "offset": 0}),
                                                    mk("late_incentive", {"base": base, "frac": frac, "offset": 0}))
    for off in (0, 1):
        out[f"rotation_vs_fixed[off={off}]"] = (mk("rotation", {"offset": off, "inc": "none"}),
                                                mk("greedy_affinity", {"inc": "none"}))
    out["front_vs_uniform[aff,0.5]"] = (mk("front_incentive", {"base": "aff", "frac": 0.5, "offset": 0}),
                                        mk("uniform_incentive", {"base": "aff", "offset": 0}))
    return out


def run_lin_instance(args):
    seed, H, variant, K, iters = args
    t0 = time.perf_counter()
    static = variant == "static"
    prior = dict(LIN_DEFAULTS["prior"])
    if variant in ("prior_x1.5", "prior_/1.5"):
        f = 1.5 if variant == "prior_x1.5" else 1 / 1.5
        prior = {k: (lo, hi * f) for k, (lo, hi) in prior.items()}
    inst = make_lin_instance(seed, ptypes=(1,), H_choices=(H,), K=K, static=static, prior=prior)
    env = inst.env
    theta = env.true_theta_vector()
    lc = LinClass(3, 3, env.aspace.nb, nmax=3, static=static)
    rows, _ = lin_library(lc, 2, static)          # Type-1: incentive levels 0/1 only
    used = np.abs(rows).max(0) > 0                 # drop the never-used psi2 coordinate
    rows = rows[:, used]
    Q = reduced_basis(rows)
    lib = Library(rows @ Q, np.ones(len(rows)), np.arange(len(rows)), len(rows), Q)
    res_rows, contrast_rows, samples = [], [], []
    for q in inst.problems:
        CL, CR = [], []
        for p in q.policies:
            a, b = lin_contrib(lc, p, q.loads0, q.H, q.utility, env.aspace)
            CL.append(a)
            CR.append(b)
        J = np.array([C.sum((0, 1)) @ theta for C in CL])
        red = lambda C: C[..., used] @ Q  # noqa: E731
        # range check: every contribution must lie in span(library)
        resid = max(float(np.abs(C[..., used] - (C[..., used] @ Q) @ Q.T).max()) for C in CL)
        CLr, CRr = [red(C) for C in CL], [red(C) for C in CR]
        res = analyse_problem(CLr, J, EPS_LIN, lib, Cr_list=CRr, iters=iters)
        base = {"env": "E1-Static" if static else "E1-Lin", "variant": variant, "instance": seed, "problem": q.pid,
                "H": q.H, "range_resid": resid}
        if res is None:
            res_rows.append({**base, "degenerate": True})
            continue
        A_J, A_U = res.pop("_A_J"), res.pop("_A_U")
        res_rows.append({**base, "degenerate": False,
                         "pi_star_name": q.policies[res["pi_star"]].name,
                         "binding_name": q.policies[res["binding"]].name, **res})
        if len(samples) < 2:
            ks, kb = res["pi_star"], res["binding"]
            samples.append({"problem": q.pid, "H": q.H, "pi_star": q.policies[ks].name,
                            "binding": q.policies[kb].name, "J_true": J.tolist(),
                            "d_time_blocks_raw": (CL[ks] - CL[kb]).sum(1)[:, used].round(6).tolist(),
                            "d_participant_blocks_raw": (CL[ks] - CL[kb]).sum(0)[:, used].round(6).tolist(),
                            "kappa": {p: res.get(f"kappa_{p}") for p in PARTS}})
        if variant == "base":
            for name, (pa, pb) in contrast_policies(None, env.aspace, q.H, q.aff, 3).items():
                Ca, Cb = lin_contrib(lc, pa, q.loads0, q.H, q.utility, env.aspace), \
                    lin_contrib(lc, pb, q.loads0, q.H, q.utility, env.aspace)
                Da, Db = red(Ca[0]), red(Cb[0])
                if np.abs(Da.sum((0, 1)) - Db.sum((0, 1))).max() < 1e-12:
                    contrast_rows.append({"instance": seed, "problem": q.pid, "H": q.H, "contrast": name,
                                          "identical": True})
                    continue
                row = {"instance": seed, "problem": q.pid, "H": q.H, "contrast": name, "identical": False,
                       "abs_value_gap_true": float(abs((Ca[0].sum((0, 1)) - Cb[0].sum((0, 1))) @ theta))}
                for part in ("time", "participant", "both"):
                    bb = partition_blocks(Da, Db, part)
                    row[f"kappa_{part}_jointopt"] = kappa_of(A_J, bb)
                    row[f"kappa_{part}_uniform"] = kappa_of(A_U, bb)
                contrast_rows.append(row)
    return res_rows, contrast_rows, samples, time.perf_counter() - t0


# ----------------------------------------------------------------------------------------------- E1-NL-S
def run_nl(seeds, H_choices, K, iters, res_dir):
    import torch
    torch.set_num_threads(4)
    from dsswm.exact.nl_propagate import NLPropagator
    from dsswm.models.nl_class import NLClass
    from dsswm.streams.generator import DEFAULT_NL_S_GRID, NL_DEFAULTS, make_nl_instance, nl_class_max_jtable
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float64
    ncls = NLClass(DEFAULT_NL_S_GRID, device=dev)
    params_all = ncls.torch_params()
    rows_out, samples = [], []
    lib_cache = {}

    class StateCodecEmpty:
        size = 0
    names = ["alpha0", "alpha1", "beta0", "beta1", "gamma0", "gamma1", "tauL0", "tauL1", "tauR0", "tauR1", "psi", "lam"]

    def unpack(th, L=2, R=2):
        return {"alpha": th[0:2][None], "beta": th[2:4][None], "gamma": th[4:6][None], "tauL": th[6:8][None],
                "tauR": th[8:10][None], "psi_left": torch.stack([torch.zeros(2, dtype=dt), th[10].expand(2)], -1)[None],
                "lam": th[11:12]}

    for seed in seeds:
        for H in H_choices:
            inst = make_nl_instance(seed, grid=DEFAULT_NL_S_GRID, nl_class=ncls, H_choices=(H,), K=K)
            env = inst.env
            prop_g = NLPropagator(2, 2, 2, env.aspace, NL_DEFAULTS["c"], NL_DEFAULTS["rho_ret"], device=dev)
            prop = NLPropagator(2, 2, 2, env.aspace, NL_DEFAULTS["c"], NL_DEFAULTS["rho_ret"], device="cpu")
            LT, _ = prop_g.tables(params_all)
            ll = prop_g.loglik(LT, inst.init_obs).sum(1)
            k_hat = int(torch.argmax(ll))
            ph = ncls.params_at(k_hat)
            th_hat = np.concatenate([ph["alpha"], ph["beta"], ph["gamma"], ph["tauL"], ph["tauR"],
                                     [ph["psi_left"][0, 1]], [ph["lam"]]])
            th_t = torch.tensor(th_hat, dtype=dt)
            # Fisher library at theta_hat: every (state, action) one-round interaction (cached per theta_hat).
            if k_hat in lib_cache:
                lib = lib_cache[k_hat]
            pr = {k: v.detach().numpy()[0] for k, v in unpack(th_t).items()}
            G, W, AI = [], [], []
            codec = prop.codec if k_hat not in lib_cache else StateCodecEmpty()
            atom = 0
            for code in range(codec.size):
                ld, en = codec.decode(code)
                for a_idx, a in enumerate(env.aspace.actions):
                    inc = {(i, j): l for i, j, l in a.incentives}
                    yprob = np.zeros(4)
                    act = np.zeros(4, bool)
                    for i, j in a.pairs:
                        if en[i] and en[2 + j]:
                            b = inc.get((i, j), 0)
                            eta = pr["alpha"][i] + pr["beta"][j] - pr["gamma"][i] * ld[i] + pr["psi_left"][i, b]
                            p = 1 / (1 + np.exp(-eta))
                            g = np.zeros(12); g[i] = 1; g[2 + j] = 1; g[4 + i] = -ld[i]; g[10] = b
                            G.append(g); W.append(p * (1 - p)); AI.append(atom)
                            yprob[i] = yprob[2 + j] = p
                            act[i] = act[2 + j] = True
                    tau = np.concatenate([pr["tauL"], pr["tauR"]])
                    for pp in range(4):
                        if not en[pp]:
                            continue
                        g = np.zeros(12); g[6 + pp] = 1; g[11] = -ld[pp]
                        wq = 0.0
                        for yv, py in ((0, 1 - yprob[pp]), (1, yprob[pp])) if act[pp] else ((0, 1.0),):
                            qv = 1 / (1 + np.exp(-(tau[pp] + NL_DEFAULTS["c"] * yv - pr["lam"] * ld[pp])))
                            wq += py * qv * (1 - qv)
                        G.append(g); W.append(wq); AI.append(atom)
                    atom += 1
            if k_hat not in lib_cache:
                G, W, AI = np.array(G), np.array(W), np.array(AI)
                Q = reduced_basis(G)
                lib_cache[k_hat] = Library(G @ Q, W, AI, atom, Q)
            lib = lib_cache[k_hat]
            Q = lib.Q
            for q in inst.problems:
                nl_class_max_jtable(q, prop_g, params_all)
                util = q.utility
                C_list, J_list, CR_list = [], [], []
                for pol in q.policies:
                    plan, esum = prop.build_policy_plan(pol, q.loads0, q.engaged0, q.H)
                    # left participant of each eyidx entry (pad -> -1)
                    def contrib(th, plan=plan):
                        LTx, EYx = prop.tables(unpack(th))
                        Pt = torch.ones(1, 1, dtype=dt)
                        out = []
                        for t, st in enumerate(plan):
                            ei = st["eyidx"]
                            li = torch.where(ei < prop.U_pair, ei // (prop.R * prop.N * prop.nb), -1)
                            ey = EYx[:, ei]                                   # (1, S, F)
                            row = [util.w[t] * (Pt[..., None] * ey * (li == i)).sum((1, 2))[0] for i in range(2)]
                            row += [torch.zeros((), dtype=dt), torch.zeros((), dtype=dt)]
                            logp = LTx[:, st["fidx"]].sum(-1) + st["const"][None]
                            Wt = Pt[:, st["src"]] * torch.exp(logp)
                            Pn = torch.zeros(1, st["n_next"], dtype=dt)
                            Pn = Pn.index_add(1, st["dst"], Wt)
                            Pt = Pn
                            out.append(torch.stack(row))
                        out = torch.stack(out)                                # (H, 4)
                        # final engagement: retention utility attributed to the last step, per participant
                        states = None
                        return out, Pt
                    # per-participant final engagement probability needs codes of final states
                    s0 = prop.codec.encode(q.loads0, q.engaged0)
                    states = np.array([s0])
                    for t in range(q.H):
                        nxt = []
                        for code in states:
                            ldd, enn = prop.codec.decode(int(code))
                            a_idx = pol.act(t, ldd, enn)
                            nxt.append(prop._struct(int(code), a_idx)[2])
                        states = np.unique(np.concatenate(nxt))
                    efin = torch.tensor(np.array([prop.codec.decode(int(c))[1] for c in states]), dtype=dt)  # (S, P)

                    def full(th, contrib=contrib, efin=efin):
                        out, Pt = contrib(th)
                        ret = util.w_ret * (Pt[0][:, None] * efin).sum(0)    # (P,)
                        out = out.clone()
                        out[-1] = out[-1] + torch.stack([ret[0], ret[1], ret[2], ret[3]])
                        return out * util.c_q
                    val = full(th_t).detach().numpy()
                    Jac = torch.autograd.functional.jacobian(full, th_t).numpy()   # (H, 4, 12)
                    # participant partition: left blocks (pair outcomes + left retention); right retention blocks
                    C_list.append(Jac @ Q)
                    J_list.append(val.sum())
                J = np.array(J_list)
                res = analyse_problem(C_list, J, EPS_NL, lib, parts=("time", "participant", "policy", "both"),
                                      iters=iters, want_rect=True)
                base = {"env": "E1-NL-S(lin@theta_hat)", "variant": "base", "instance": seed, "problem": q.pid,
                        "H": q.H, "theta_hat_index": k_hat, "theta_hat_is_truth": bool(k_hat == inst.truth["theta_index"])}
                if res is None:
                    rows_out.append({**base, "degenerate": True})
                    continue
                res.pop("_A_J"); res.pop("_A_U")
                rows_out.append({**base, "degenerate": False, "pi_star_name": q.policies[res["pi_star"]].name,
                                 "binding_name": q.policies[res["binding"]].name, **res})
                if len(samples) < 2:
                    samples.append({"problem": q.pid, "H": q.H, "theta_hat": dict(zip(names, th_hat.tolist())),
                                    "J_plugin": J.tolist(), "kappa": {p: res.get(f"kappa_{p}") for p in
                                                                      ("time", "participant", "policy", "both")}})
            progress(res_dir, 0, 0, f"nl seed {seed} H {H} done")
    return rows_out, samples


# ----------------------------------------------------------------------------------------------- aggregation
def share(x, thr=1.5):
    x = np.asarray([v for v in x if v == v])
    return float((x >= thr).mean()) if len(x) else float("nan")


def quant(x):
    x = np.asarray([v for v in x if v == v])
    if not len(x):
        return {}
    return {k: float(np.quantile(x, p)) for k, p in (("q10", .1), ("q25", .25), ("median", .5), ("q75", .75),
                                                      ("q90", .9))} | {"max": float(x.max()), "n": int(len(x))}


def cp_interval(k, n, a=0.05):
    from scipy.stats import beta
    lo = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return [float(lo), float(hi)]


def summarise(df, parts):
    out = {}
    for H, g in df.groupby("H"):
        d = {"n": int(len(g))}
        for p in parts:
            c = f"kappa_{p}"
            if c in g:
                d[f"share_ge_1.5_{p}"] = share(g[c])
                d[f"quantiles_{p}"] = quant(g[c])
                d[f"envelope_ratio_median_{p}"] = float(np.nanmedian(g[f"envelope_ratio_{p}"]))
                d[f"share_kappa0_ge_1.5_{p}"] = share(g[f"kappa0_{p}"])
                d[f"share_kappainf_ge_1.5_{p}"] = share(g[f"kappainf_{p}"])
                if f"kappa_eff_{p}" in g:
                    d[f"share_kappa_eff_ge_1.5_{p}"] = share(g[f"kappa_eff_{p}"])
                    d[f"quantiles_kappa_eff_{p}"] = quant(g[f"kappa_eff_{p}"])
        if "kappa_time" in g and "kappa_participant" in g:
            for pre in ("kappa", "kappa0", "kappa_eff"):
                if f"{pre}_time" in g:
                    d[f"share_time_or_participant_ge_1.5[{pre}]"] = float(
                        ((g[f"{pre}_time"] >= 1.5) | (g[f"{pre}_participant"] >= 1.5)).mean())
        out[int(H)] = d
    return out


def make_figures(df, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Hs = sorted(df.H.unique())
    fig, axes = plt.subplots(1, len(Hs), figsize=(4 * len(Hs), 3.3), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, H in zip(axes, Hs):
        g = df[df.H == H]
        for p, col in (("time", "C0"), ("participant", "C1"), ("policy", "C2"), ("both", "C3")):
            x = np.sort(g[f"kappa_{p}"].dropna().values)
            if len(x):
                ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=col, label=p)
        ax.axvline(1.5, ls="--", color="k", lw=0.8)
        ax.set_xscale("log")
        ax.set_title(f"E1-Lin, H={H} (n={len(g)})")
        ax.set_xlabel("kappa (joint-opt design, binding pair)")
    axes[0].set_ylabel("ECDF")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "kappa_ecdf.png", dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    import pandas as pd
    from joblib import Parallel, delayed
    pilot = args.mode == "pilot"
    res_root = WS / "exp" / "results"
    out_dir = res_root / ("pilots" if pilot else "full") / TASK
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    (res_root / f"{TASK}.pid").write_text(str(os.getpid()))
    t_start = time.time()
    (out_dir / "start_time.txt").write_text(datetime.now().isoformat())
    iters = 800 if pilot else 1200
    try:
        if pilot:
            # 200 problems: H=1 -> 6 instances x 10, H=4 / H=8 -> 7 instances x 10 (dev seeds, disjoint per H)
            jobs = [(s, 1, "base", 10, iters) for s in range(0, 6)]
            jobs += [(s, 4, "base", 10, iters) for s in range(6, 13)]
            jobs += [(s, 8, "base", 10, iters) for s in range(13, 20)]
            sens = [(s, H, v, 10, iters) for v in ("prior_x1.5", "prior_/1.5") for H, ss in ((4, range(6, 10)),
                                                                                         (8, range(13, 17))) for s in ss]
            static = [(s, H, "static", 10, iters) for H, ss in ((4, range(6, 9)), (8, range(13, 16))) for s in ss]
            nl_seeds, nl_H, nl_K = list(range(0, 3)), (1, 6, 8), 5
        else:
            Hs = (1, 2, 4, 8, 16)
            jobs = [(10000 + 100 * hi + s, H, "base", 10, iters) for hi, H in enumerate(Hs) for s in range(100)]
            sens = [(10000 + 100 * hi + s, H, v, 10, iters) for v in ("prior_x1.5", "prior_/1.5")
                    for hi, H in enumerate((4, 8)) for s in range(50)]
            static = [(10000 + 100 * hi + s, H, "static", 10, iters) for hi, H in enumerate((4, 8)) for s in range(20)]
            nl_seeds, nl_H, nl_K = list(range(10000, 10010)), (1, 6, 8), 10
        all_jobs = jobs + sens + static
        progress(res_root, 0, len(all_jobs), "lin")
        results = Parallel(n_jobs=args.workers, verbose=0)(delayed(run_lin_instance)(j) for j in all_jobs)
        rows, crows, samples, secs = [], [], [], []
        for r, c, s, t in results:
            rows += r; crows += c; samples += s; secs.append(t)
        progress(res_root, len(all_jobs), len(all_jobs), "lin done; nl start")
        t_nl = time.perf_counter()
        nl_rows, nl_samples = run_nl(nl_seeds, nl_H, nl_K, 400, res_root)
        t_nl = time.perf_counter() - t_nl
        df = pd.DataFrame(rows)
        dfc = pd.DataFrame(crows)
        dfn = pd.DataFrame(nl_rows)
        df.to_csv(out_dir / "kappa.csv", index=False)
        dfc.to_csv(out_dir / "contrasts.csv", index=False)
        dfn.to_csv(out_dir / "kappa_nl_linearised.csv", index=False)
        with open(out_dir / "samples" / "binding_pairs.json", "w") as f:
            json.dump({"lin": samples[:8], "nl": nl_samples}, f, indent=1)
        ok = df[~df.degenerate]
        base = ok[ok.variant == "base"]
        make_figures(base, out_dir)
        lin_parts = ("time", "participant", "participant_R", "policy", "both")
        by_H = summarise(base, lin_parts)
        h48 = base[base.H.isin([4, 8])]
        n48 = len(h48)
        k48 = int(((h48.kappa_time >= 1.5) | (h48.kappa_participant >= 1.5)).sum())
        h1 = base[base.H == 1]
        h1_dev = float((h1.kappa_time - 1).abs().max()) if len(h1) else float("nan")
        crit_share = k48 / max(n48, 1)
        crit_h1 = bool(len(h1) and h1_dev < 1e-9)
        share_time_48 = share(h48.kappa_time)
        sens_out = {}
        for v in ("base", "prior_x1.5", "prior_/1.5"):
            g = ok[(ok.variant == v) & ok.H.isin([4, 8])]
            if v == "base":
                g = g[g.instance.isin(list(range(6, 10)) + list(range(13, 17)))] if pilot else g
            sens_out[v] = {"n": int(len(g)), "share_time": share(g.kappa_time),
                           "share_participant": share(g.kappa_participant),
                           "share_time_or_participant": float(((g.kappa_time >= 1.5) | (g.kappa_participant >= 1.5)).mean())
                           if len(g) else float("nan"),
                           "median_kappa_time": float(g.kappa_time.median()) if len(g) else float("nan")}
        st = ok[ok.variant == "static"]
        static_out = summarise(st, ("time", "participant")) if len(st) else {}
        contrasts_out = {}
        if len(dfc):
            cc = dfc[~dfc.identical]
            for (name, H), g in cc.groupby(["contrast", "H"]):
                contrasts_out.setdefault(name, {})[int(H)] = {
                    "n": int(len(g)), "n_identical": int(dfc[(dfc.contrast == name) & (dfc.H == H)].identical.sum()),
                    "kappa_time_jointopt": quant(g.kappa_time_jointopt),
                    "kappa_participant_jointopt": quant(g.kappa_participant_jointopt),
                    "kappa_time_uniform_median": float(g.kappa_time_uniform.median()),
                    "share_time_ge_1.5_jointopt": share(g.kappa_time_jointopt)}
        nl_ok = dfn[~dfn.degenerate] if len(dfn) else dfn
        nl_out = summarise(nl_ok, ("time", "participant", "policy", "both")) if len(nl_ok) else {}
        if len(nl_ok):
            nl_out["theta_hat_is_truth_rate"] = float(nl_ok.theta_hat_is_truth.mean())
            nl_h1 = nl_ok[nl_ok.H == 1]
            nl_out["H1_max_abs_kappa_time_minus_1"] = float((nl_h1.kappa_time - 1).abs().max()) if len(nl_h1) else None
        rho_check = {}
        for p in ("time", "participant", "policy", "both"):
            r = base[f"rho_ratio_{p}"]
            lo, hi = base[f"kappa0_{p}"] ** 2, base[f"kappainf_{p}"] ** 2
            rho_check[p] = {"median_rho_ratio": float(r.median()),
                            "frac_within_[kappa0^2, kappainf^2](1% tol)": float(((r >= lo * 0.99) & (r <= hi * 1.01)).mean()),
                            "spearman_rho_ratio_vs_kappa2": float(pd.Series(r).corr(base[f"kappa_{p}"] ** 2, method="spearman"))}
        robust = {}
        for pre in ("kappa", "kappa0", "kappa_eff"):
            robust[pre] = {"share_time_or_participant": float(((h48[f"{pre}_time"] >= 1.5) | (h48[f"{pre}_participant"] >= 1.5)).mean()),
                           "share_time": share(h48[f"{pre}_time"]), "share_participant": share(h48[f"{pre}_participant"])}
        go = bool(crit_share >= 0.30 and crit_h1)
        summary = {
            "task_id": TASK, "mode": args.mode, "seed": 42, "generator_hash": generator_hash(),
            "n_problems_base": int(len(df[df.variant == "base"])), "n_degenerate_base": int(df[df.variant == "base"].degenerate.sum()),
            "design_library": "E1-Lin: 72 single-observation atoms x(i,j,n_i,b), b in {0,1}, n_i in 0..3 (reduced to the 9-dim identifiable span; the alpha+c/beta-c gauge is removed). Binding pair under joint-opt (oracle, theta* known) design, Delta floored at eps=0.05.",
            "partition_definitions": {"time": "per-step value contribution", "participant": "contribution of outcomes of each left participant (carries the dynamic gamma_i)",
                                      "participant_R": "same, right participants", "policy": "per-policy marginal intervals {z_pi*, z_pi'}", "both": "time x left participant"},
            "metrics": {
                "share_kappa_ge_1.5_by_H": by_H,
                "pass_quantity_share_time_or_participant_H4_8": crit_share,
                "pass_quantity_ci_cp95": cp_interval(k48, n48),
                "share_time_only_H4_8": share_time_48,
                "share_participant_only_H4_8": share(h48.kappa_participant),
                "H1_max_abs_kappa_time_minus_1": h1_dev,
                "gate_robustness_H4_8 (kappa=joint-opt design literal; kappa0=min over 3 designs; kappa_eff=sqrt(rho_P/rho_joint))": robust,
                "H1_share_participant_ge_1.5": share(h1.kappa_participant),
                "generator_sensitivity_H4_8": sens_out,
                "static_control_E1_Static": static_out,
                "common_contrasts": contrasts_out,
                "nl_linearised_reference": nl_out,
                "rho_ratio_vs_kappa_binding_pair": rho_check,
            },
            "pass_criteria": {"share(kappa_time or kappa_participant >= 1.5) >= 0.30 over H in {4,8}": bool(crit_share >= 0.30),
                              "kappa_time == 1 exactly for all H=1 problems": crit_h1},
            "go_no_go": "GO" if go else "NO_GO",
            "timing": {"wall_clock_s": time.time() - t_start, "lin_instance_s_mean": float(np.mean(secs)),
                       "nl_s": t_nl, "workers": args.workers, "note": "concurrent run (other tasks on the machine)"},
        }
        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=1, default=float)
        progress(res_root, 1, 1, "done", {"go_no_go": summary["go_no_go"], "share": crit_share})
        mark_done(res_root, "success", f"{summary['go_no_go']}: share={crit_share:.3f}, H1 dev={h1_dev:.2e}")
        print(json.dumps({k: summary[k] for k in ("go_no_go", "pass_criteria")}, indent=1))
        print("share", crit_share, "time-only", share_time_48, "wall", time.time() - t_start)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        mark_done(res_root, "failed", f"{type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    main()
