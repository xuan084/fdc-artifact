"""E1-Lin methods wired into the five-level reuse switch (round 3). Learner side: no truth access.

All set-based methods share ONE evidence type and ONE certificate, so their cost ratios only reflect reuse and
sampling (methodology section 2.2):
  evidence     Abbasi-Yadkori time-uniform ellipsoid (evidence.ellipsoid.EllipsoidSet, sigma known, S = round-0 lock)
  certificate  closed form certify.lin_closed.certify_lin (exact sup over the ellipsoid), checked on a geometric
               schedule (every max(1, ceil(CERT_CHECK_FRAC * n_new)) new steps, plus n_new = 0 and n_new = m)
Methods differ only in which legal action is played next on the REAL platform trajectory (no resets; the platform
state carries over between the problems of a stream, as on the NL layer):
  JPC-Lin    DDA: phased (doubling) Frank-Wolfe transductive design over legal atoms (i, j, n, b<=1) on the undecided
             directions z_k - z_pihat (ub_k > eps), weighted 1 / (gap_hat_k + eps)^2, offset by the current V, tracked
             at the current loads (deficit tracking: the legal action with the largest summed signed deficit of its matched atoms; if
             no action has a positive score, a uniformly random legal round, which steers the loads towards target
             atoms that are unreachable from the current state; same tracker for every design method)
  RAGE       Fiez et al. 2019 XY-allocation (extracted from run_hd2_kappa_sandwich_lin.run_seq, design_mode='rage'):
             all pairs inside the surviving set {pihat} U {k : ub_k > eps}, unit weights, entropic mirror descent
             (`optimise`), re-designed at doubling phase ends, same stopping rule
  XY-static  fixed XY-optimal design over ALL pairs of the problem's candidates, computed once per problem from the
             geometry only (no data), tracked
  G-opt      fixed G-optimal coverage design over the legal atoms (min_xi max_x ||x||_{A(xi)^-1}), problem-independent
  B1eb       whole-trial LUCB (arm = candidate policy, pull = one full H-step trial from s0 after reset_to(s0), H billed
             steps). Gaussian outcomes are unbounded, so the round-0 bounded empirical-Bernstein radius is replaced by
             its known-variance (Bernstein / sub-Gaussian) limit with the EXACT public per-arm variance
             v_k = sigma^2 c_q^2 sum_t w_t^2 m_t(k) (m_t = matched pairs at step t, known by A1), union bound
             delta_n = 6 delta / (pi^2 K n^2) as in baselines.whole_trial_bai. (s0, Pi) reuse: every complete trial
             inside the active evidence set whose start state equals s0 and whose recorded action sequence has the
             candidate's action sequence as prefix is re-scored under the new utility (same (s0, Pi), new weights or a
             shorter horizon). Rows that are not complete trials (initial data, padding, other designs) are unusable
             for B1eb but still sit in its evidence set and are billed exactly like for every other method.

Every new platform round goes through ReuseSwitch.record (the only billed path). ev(m): certification is attempted
only when switch.may_certify(); when the certificate already holds but more new steps are required, the method keeps
acquiring with its own rule on all challengers (forced steps).
"""
from __future__ import annotations

import math
import time

import numpy as np

from ..certify.lin_closed import certify_lin
from ..core.dynamics import next_loads
from ..core.provenance import require_authentic
from ..evidence.ellipsoid import EllipsoidSet

METHODS = ("JPC-Lin", "RAGE", "XY-static", "G-opt", "B1eb")
METHOD_CODE = {"JPC-Lin": 1101, "RAGE": 1303, "XY-static": 1404, "G-opt": 1505, "B1eb": 1606}
S_LIN = 2.8071337695236402          # round-0 lock 'E1-Lin.ellipsoid_S' (prior box radius)
EPS_LIN, DELTA, LAM = 0.05, 0.05, 1.0
T_MAX_LIN = 20000
LIN_TAUS = (5000, 10000, 20000)
CERT_CHECK_FRAC = 0.01
PHASE0 = 32
EXPLORE = 0.02
FW_ITERS = 150
RAGE_ITERS = 300


# ------------------------------------------------------------------------------------------------ geometry
def reduced_basis(rows, tol=1e-9):
    U, s, _ = np.linalg.svd(np.asarray(rows, float).T, full_matrices=False)
    return U[:, s > tol * s.max()]


class LinGeom:
    """Public geometry: legal atoms (i, j, n, b) with b <= maxlev, their features, reduced coordinates Q of the
    legal library, and per-action atom indices at given loads. On the static class the load index is collapsed."""

    def __init__(self, lc, aspace, maxlev: int = 1):
        self.lc, self.aspace, self.maxlev = lc, aspace, int(maxlev)
        L, R, nb = lc.L, lc.R, lc.nb
        self.L, self.R, self.nb = L, R, nb
        self.N = 1 if lc.static else lc.nmax + 1
        n_atoms = L * R * self.N * nb
        self.Ftab = np.zeros((n_atoms, lc.d))
        self.atom_legal = np.zeros(n_atoms, bool)
        for i in range(L):
            for j in range(R):
                for n in range(self.N):
                    for b in range(nb):
                        k = self.fidx(i, j, n, b)
                        self.Ftab[k] = lc.feature(i, j, n, b)
                        self.atom_legal[k] = b <= self.maxlev
        self.Q = reduced_basis(self.Ftab[self.atom_legal])
        self.Fr = self.Ftab @ self.Q
        self.legal = np.array([a for a in range(aspace.n) if aspace.max_level(a) <= self.maxlev], np.int64)
        P = min(L, R)
        n = aspace.n
        self.ai = np.zeros((n, P), np.int64)
        self.aj = np.zeros((n, P), np.int64)
        self.ab = np.zeros((n, P), np.int64)
        self.am = np.zeros((n, P), bool)
        for a_idx, a in enumerate(aspace.actions):
            inc = {(i, j): l for i, j, l in a.incentives}
            for s_, (i, j) in enumerate(a.pairs):
                self.ai[a_idx, s_], self.aj[a_idx, s_], self.ab[a_idx, s_] = i, j, inc.get((i, j), 0)
                self.am[a_idx, s_] = True
        self._gopt = None

    def fidx(self, i, j, n, b):
        return ((i * self.R + j) * self.N + n) * self.nb + b

    def action_atoms(self, loads):
        n_i = np.zeros_like(self.ai) if self.N == 1 else np.asarray(loads)[self.ai]
        return ((self.ai * self.R + self.aj) * self.N + n_i) * self.nb + self.ab

    def reduce_V(self, V):
        return self.Q.T @ V @ self.Q


def fw_design(T, Vr, n_phase, geom: LinGeom, weights=None, iters=FW_ITERS):
    """Frank-Wolfe min-max transductive design over legal atoms in reduced coordinates:
    min_lam max_t w_t ||g_t||^2_{(Vr + n A(lam))^-1}. T: (m, r) targets. Returns lam over ALL atoms (0 if illegal)."""
    F = geom.Fr[geom.atom_legal]
    k = len(F)
    w = np.ones(len(T)) if weights is None else np.asarray(weights, float)
    lam = np.full(k, 1.0 / k)
    for it in range(iters):
        A = Vr + n_phase * (F.T * lam) @ F
        Ai = np.linalg.inv(A)
        AiT = Ai @ T.T
        vals = w * (T * AiT.T).sum(1)
        t = int(np.argmax(vals))
        grad = -n_phase * w[t] * (F @ AiT[:, t]) ** 2
        j = int(np.argmin(grad))
        g = 2.0 / (it + 3.0)
        lam *= (1 - g)
        lam[j] += g
    full = np.zeros(len(geom.Fr))
    full[geom.atom_legal] = lam
    return full


def optimise(G, blocks, Delta, iters=RAGE_ITERS, xi0=None):
    """RAGE XY-allocation (run_h2_kappa_distribution.optimise, atom weights 1): min_xi max_k
    (sum_b ||B_kb||_{A(xi)^-1})^2 / Delta_k^2 by entropic mirror descent on a softmax-smoothed max.
    G: (n_atoms, r) atom features in reduced coordinates; blocks: list of (nb_k, r)."""
    n, r = G.shape
    xi = np.full(n, 1.0 / n) if xi0 is None else xi0.copy()
    sizes = [len(B) for B in blocks]
    owner = np.repeat(np.arange(len(blocks)), sizes)
    allB = np.vstack(blocks)
    Delta = np.asarray(Delta, float)
    best_val, best_xi = np.inf, xi.copy()

    def norms(x):
        A = (G * x[:, None]).T @ G + 1e-12 * np.eye(r)
        U = np.linalg.solve(A, allB.T)
        return U, np.sqrt(np.maximum((allB.T * U).sum(0), 0.0))

    for it in range(iters):
        U, nrm = norms(xi)
        S = np.bincount(owner, weights=nrm, minlength=len(blocks))
        f = (S / Delta) ** 2
        fm = f.max()
        if fm < best_val:
            best_val, best_xi = fm, xi.copy()
        tau = 0.05 * (0.02 / 0.05) ** (it / iters)
        lam = np.exp((f - fm) / (tau * fm))
        lam /= lam.sum()
        coef = lam[owner] * (S[owner] / Delta[owner] ** 2) / np.maximum(nrm, 1e-300)
        grad = -((G @ U) ** 2) @ coef
        g = grad / max(np.abs(grad).max(), 1e-300)
        eta = 2.0 / np.sqrt(it + 1.0)
        xi = xi * np.exp(-eta * g)
        xi = np.maximum(xi / xi.sum(), 1e-10)
        xi /= xi.sum()
    _, nrm = norms(xi)
    fm = float(((np.bincount(owner, weights=nrm, minlength=len(blocks)) / Delta) ** 2).max())
    if fm < best_val:
        best_val, best_xi = fm, xi
    return best_xi, float(best_val)


def rage_design(geom: LinGeom, Zr, S_idx):
    """RAGE XY design over all pairs in the surviving set S_idx (unit weights). Returns lam over all atoms."""
    blocks = [(Zr[a] - Zr[b])[None] for ii, a in enumerate(S_idx) for b in S_idx[ii + 1:]
              if np.abs(Zr[a] - Zr[b]).max() > 1e-13]
    if not blocks:
        return None
    F = geom.Fr[geom.atom_legal]
    xs, _ = optimise(F, blocks, np.ones(len(blocks)))
    full = np.zeros(len(geom.Fr))
    full[geom.atom_legal] = xs
    return full


def gopt_design(geom: LinGeom):
    """G-optimal coverage of the legal atoms (cached on the geometry)."""
    if geom._gopt is None:
        F = geom.Fr[geom.atom_legal]
        geom._gopt = fw_design(F, 1e-9 * np.eye(F.shape[1]), 1.0, geom, iters=4 * FW_ITERS)
    return geom._gopt


# ------------------------------------------------------------------------------------------------ public problem data
def z_tables(lc, problems, aspace):
    """Public candidate features: Z (n_pol, d) and per-step Zt (n_pol, H, d) of every problem (A1: exact)."""
    Zs, Zts = [], []
    for q in problems:
        Zt = np.stack([lc.z_per_step(p, q.loads0, q.H, q.utility, aspace) for p in q.policies])
        Zts.append(Zt)
        Zs.append(Zt.sum(1))
    return Zs, Zts


def action_sequence(policy, loads0, H, aspace, nmax, static):
    loads = np.asarray(loads0, np.int64).copy()
    ones = np.ones(len(loads), dtype=np.int64)
    seq, m = [], []
    for t in range(H):
        a = policy.act(t, loads, ones)
        seq.append(int(a))
        m.append(len(aspace.actions[a].pairs))
        loads = next_loads(loads, aspace, a, nmax, static)
    return tuple(seq), np.array(m)


class PublicLin:
    """Truth-free per-stream objects shared by all methods / arms."""

    def __init__(self, lc, aspace, problems, sigma, eps=EPS_LIN, delta=DELTA, S=S_LIN, lam=LAM, maxlev=1,
                 geom=None):
        self.lc, self.aspace = lc, aspace
        self.problems = list(problems)
        self.sigma, self.eps, self.delta, self.S, self.lam = float(sigma), eps, delta, S, lam
        self.geom = geom or LinGeom(lc, aspace, maxlev)
        self.Z, self.Zt = z_tables(lc, self.problems, aspace)
        self.Zr = [Z @ self.geom.Q for Z in self.Z]
        self.seqs = [[action_sequence(p, q.loads0, q.H, aspace, lc.nmax, lc.static) for p in q.policies]
                     for q in self.problems]

    def make_set_factory(self, method: str):
        if method == "B1eb":
            return TrialStore
        return lambda: EllipsoidSet(self.lc, sigma=self.sigma, delta=self.delta, S=self.S, lam=self.lam)


# ------------------------------------------------------------------------------------------------ B1eb trial bookkeeping
# serial -> (trial_uid, position, s0 tuple, action sequence of the whole trial). Written only by solve_b1eb for the
# rounds it played itself (on this or a sibling platform); a row is usable as trial data only if its whole trial is
# inside the active evidence set.
TRIAL_TAGS: dict = {}
_TRIAL_UID = [0]


class TrialStore:
    """B1eb evidence: the authenticated rows (n_rounds / update API of the reuse switch)."""

    def __init__(self):
        self.n_rounds = 0
        self.obs: list = []

    def update(self, obs):
        require_authentic(obs)
        self.obs.append(obs)
        self.n_rounds += 1

    def trials(self):
        """Complete trials inside the store: list of (s0, action_seq, ysum per step)."""
        groups: dict = {}
        for o in self.obs:
            tag = TRIAL_TAGS.get(o.serial)
            if tag is None:
                continue
            uid, pos, s0, seq = tag
            g = groups.setdefault(uid, {"s0": s0, "seq": seq, "y": {}})
            g["y"][pos] = sum(val for _, _, _, val in o.outcomes)
        out = []
        for g in groups.values():
            H = len(g["seq"])
            if len(g["y"]) == H:
                out.append((g["s0"], g["seq"], np.array([g["y"][t] for t in range(H)])))
        return out


# ------------------------------------------------------------------------------------------------ set methods
def _check_due(n_new, next_check, m):
    return n_new >= next_check or n_new == m


def solve_set_method(pub: PublicLin, method: str, k: int, sw, lr, handle, rng, tmax: int):
    """JPC-Lin / RAGE / XY-static / G-opt on problem k with the ellipsoid inside `lr`."""
    t0 = time.perf_counter()
    ell = lr.inner
    geom = pub.geom
    Z, Zr = pub.Z[k], pub.Zr[k]
    eps = pub.eps
    m_req = sw.m if sw.base == "ev" else 0
    nP = len(Z)
    lam_t, cnt, phase_atoms, phase_end = None, None, 0, 0
    if method == "XY-static":
        lam_t = rage_design(geom, Zr, list(range(nP)))
    elif method == "G-opt":
        lam_t = gopt_design(geom)
    if lam_t is not None:
        cnt = np.zeros(len(geom.Fr))
    status, res, next_check, n_forced, n_designs, first_cert_step, n_checks = None, None, 0, 0, 0, None, 0
    r_bar_start = None
    while True:
        n_new = sw.new_steps
        if res is None or _check_due(n_new, next_check, m_req):
            res = certify_lin(Z, ell, eps)
            n_checks += 1
            next_check = n_new + max(1, int(math.ceil(CERT_CHECK_FRAC * n_new)))
            if r_bar_start is None:
                r_bar_start = res["r_bar"]
            ok = res["status"].value == "CERTIFIED"
            if ok and first_cert_step is None:
                first_cert_step = n_new
            if ok and sw.may_certify():
                status = "CERTIFIED"
                break
        if n_new >= tmax:
            status = "NEED_DATA"
            break
        ok = res["status"].value == "CERTIFIED"
        if method in ("JPC-Lin", "RAGE") and n_new >= phase_end:
            n_phase = max(PHASE0, n_new)
            phase_end = n_new + n_phase
            pi = res["pi_hat"]
            act = [j for j in range(nP) if j != pi and (ok or res["ub"][j] > eps)]
            n_forced += int(ok)
            if method == "JPC-Lin":
                T = np.array([Zr[j] - Zr[pi] for j in act]).reshape(-1, Zr.shape[1])
                keep = np.abs(T).max(1) > 1e-13 if len(T) else np.zeros(0, bool)
                if keep.any():
                    th = ell.theta_hat()
                    gap = np.array([max(float((Z[pi] - Z[j]) @ th), 0.0) for j in act])[keep] + eps
                    lam_t = fw_design(T[keep], geom.reduce_V(ell.V), n_phase, geom, weights=1.0 / gap ** 2)
                else:
                    lam_t = None
            else:
                lam_t = rage_design(geom, Zr, [pi] + act)
            cnt = np.zeros(len(geom.Fr))
            phase_atoms = 0
            n_designs += 1
        loads = handle.observable_state()[0]
        if lam_t is None or rng.random() < EXPLORE:
            a = int(rng.choice(geom.legal))
        else:
            at = geom.action_atoms(loads)
            tgt = lam_t * (phase_atoms + 3)
            sc = np.where(geom.am, (tgt - cnt)[at], 0.0).sum(1)[geom.legal]
            if sc.max() <= 1e-12:   # no reachable atom below target: random legal round (steers the loads)
                a = int(rng.choice(geom.legal))
            else:
                a = int(geom.legal[int(np.argmax(sc + 1e-9 * rng.random(len(geom.legal))))])
        obs = handle.step(a)
        sw.record(obs)
        if cnt is not None:
            at_a = geom.action_atoms(loads)[a][geom.am[a]]
            np.add.at(cnt, at_a, 1)
            phase_atoms += len(at_a)
    return {"status": status, "pi": res["pi_hat"] if status == "CERTIFIED" else None, "steps": sw.new_steps,
            "extra": {"r_bar_start": r_bar_start, "r_bar_end": float(res["r_bar"]),
                      "first_cert_step": first_cert_step, "forced_phases": n_forced, "n_designs": n_designs,
                      "n_checks": n_checks, "sqrt_beta_end": float(ell.sqrt_beta()), "n_obs_set": int(ell.n_obs),
                      "wall_clock_s": time.perf_counter() - t0}}


# ------------------------------------------------------------------------------------------------ B1eb
def _delta_n(delta, K, n):
    return 6.0 * delta / (math.pi ** 2 * K * np.maximum(n, 1) ** 2)


def known_var_rad(n, v, delta, K):
    n = np.maximum(n, 1)
    return np.sqrt(2.0 * v * np.log(2.0 / _delta_n(delta, K, n)) / n)


def solve_b1eb(pub: PublicLin, k: int, sw, lr, handle, rng, tmax: int, batch_frac=0.02):
    t0 = time.perf_counter()
    q = pub.problems[k]
    H, K = q.H, len(q.policies)
    s0 = tuple(int(x) for x in q.loads0)
    w, cq = np.asarray(q.utility.w, float), float(q.utility.c_q)
    seqs = pub.seqs[k]
    v = np.array([pub.sigma ** 2 * cq ** 2 * float((w ** 2 * m).sum()) for _, m in seqs])
    # (s0, Pi) reuse: complete trials in the evidence set that start at s0 and whose action path has the candidate's
    samples = [[] for _ in range(K)]
    n_reused = 0
    for ts0, tseq, ysum in lr.inner.trials():
        if ts0 != s0:
            continue
        for a, (seq, _) in enumerate(seqs):
            if len(tseq) >= H and tseq[:H] == seq:
                samples[a].append(cq * float(w @ ysum[:H]))
                n_reused += 1
    n = np.array([len(s) for s in samples], float)
    s1 = np.array([sum(s) for s in samples], float)

    def pull(a, b):
        nonlocal s1
        for _ in range(b):
            handle.reset_to(np.asarray(s0, np.int64))
            _TRIAL_UID[0] += 1
            uid = _TRIAL_UID[0]
            ys = np.zeros(H)
            for t, act in enumerate(seqs[a][0]):
                obs = handle.step(act)
                TRIAL_TAGS[obs.serial] = (uid, t, s0, seqs[a][0])
                sw.record(obs)
                ys[t] = sum(val for _, _, _, val in obs.outcomes)
            u = cq * float(w @ ys)
            samples[a].append(u)
            n[a] += 1
            s1[a] += u

    status, k_hat, gap, n_pulls, first_cert_step = None, None, np.inf, 0, None
    while True:
        if (n >= 1).all():
            mu = s1 / n
            rad = known_var_rad(n, v, pub.delta, K)
            lcb, ucb = mu - rad, mu + rad
            k_hat = int(np.argmax(mu))
            ucb_o = ucb.copy()
            ucb_o[k_hat] = -np.inf
            ch = int(np.argmax(ucb_o))
            gap = float(ucb_o[ch] - lcb[k_hat])
            if gap <= pub.eps and first_cert_step is None:
                first_cert_step = sw.new_steps
            if gap <= pub.eps and sw.may_certify():
                status = "CERTIFIED"
                break
            if gap <= pub.eps:                      # ev(m): certificate holds, more new steps required
                todo = [(k_hat, 1)]
            else:
                b = max(1, int(batch_frac * min(n[k_hat], n[ch])))
                todo = [(k_hat, b), (ch, b)]
        else:
            todo = [(int(np.argmin(n)), 1)]
        budget = (tmax - sw.new_steps) // H
        if budget < 1:
            status = "NEED_DATA"
            break
        for a, b in todo:
            b = int(min(b, (tmax - sw.new_steps) // H))
            if b >= 1:
                pull(a, b)
                n_pulls += b
    return {"status": status, "pi": k_hat if status == "CERTIFIED" else None, "steps": sw.new_steps,
            "extra": {"n_reused_trials": int(n_reused), "pulls": int(n_pulls), "gap_end": float(gap),
                      "n_per_arm": [int(x) for x in n], "first_cert_step": first_cert_step,
                      "wall_clock_s": time.perf_counter() - t0}}


def solve(pub: PublicLin, method: str, k: int, sw, lr, handle, rng, tmax: int):
    if method == "B1eb":
        return solve_b1eb(pub, k, sw, lr, handle, rng, tmax)
    if method in ("JPC-Lin", "RAGE", "XY-static", "G-opt"):
        return solve_set_method(pub, method, k, sw, lr, handle, rng, tmax)
    raise ValueError(method)
