"""QFC certificate over a per-segment policy class with a knapsack budget (round 4; methodology 1.3 / 1.4 / R2).

Objects
-------
stats   per-cell sufficient statistics of one stream at one checkpoint (``QFCStats``): segment weights w (S,), point
        estimates mu_hat (S, A), counts n (S, A), variance upper bounds var_ucb (S, A) (Lemma L2: Bernstein inversion
        of the binary pool mean, ``make_stats``), optional pool sizes N (S, A) (exhausted cells are exact and dropped
        from V and b), outcome range R.
problem anything with ``kappa`` (A,) and ``budget`` (e.g. ``streams.frontier.Problem``); feasibility of a
        deterministic per-segment policy pi is  sum_s w_s kappa[pi(s)] <= budget + FEAS_TOL.

Pair width (Lemma L1, live cells only). For pi' != pi on the segment set D:
    Delta_hat(pi', pi) = sum_{s in D} w_s (mu_hat[s, pi'(s)] - mu_hat[s, pi(s)])
    V_bar(pi', pi)     = sum_{s in D} w_s^2 (var_ucb[s, pi'(s)] / n[s, pi'(s)] + var_ucb[s, pi(s)] / n[s, pi(s)])
    b(pi', pi)         = R * max_{c touched in D} w_s / n_c
    width              = sqrt(2 L1 V_bar) + b L1 / 3          (+inf if a touched cell has n_c = 0)

Main certificate (all primary layers: CR9 |Pi| = 512, HR8 6561, CR12 4096; HR6 729)
    U_q = max_{pi' in Pi_{B_q}} [Delta_hat(pi', pi_hat_q) + width(pi', pi_hat_q)],  certified iff U_q <= eps,
computed by explicit enumeration of Pi_all = [A]^S (``qfc_certificate_enum``). L1 is the Q*-union exponent
S ln A + ln(Q K / delta_main) (qfc_lemma section 9, option 1; ``l1_qstar``). No finite-population (Serfling-type)
correction is used anywhere in this module.

Conservative upper bound for classes too large to enumerate (``qfc_upper_dp``, CR12 S-gap appendix / S=16).
For any t > 0, AM-GM gives sqrt(2 L V) <= L/t + t V / 2, and b <= b_bar := R * max_{live c} (1/n_c) * max_s w_s. Hence
    U_DP = min_{t in T} max_{pi' in Pi_B^down} [Delta_hat + L/t + t V_bar / 2] + b_bar L / 3  >=  U_q,
where Pi_B^down is the feasible set under costs rounded DOWN to an integer grid (a superset of Pi_B) and T is any
finite grid (default: 40 geometric points). For fixed t the bracket is additive over segments, so the inner max is a
multiple-choice knapsack DP on the rounded costs. U_DP is a computable conservative upper bound and nothing more:
four relaxation gaps separate it from U_q and none of them is claimed to vanish (``gap_report``):
    (iv) low-order globalisation   b(pi', pi_hat) -> b_bar
    (iii) min-max order            max_pi' min_t -> min_t max_pi'  (continuous t)
    (ii) t grid                    continuous t -> finite grid T
    (i) cost rounding              Pi_B -> Pi_B^down
external reviewer counterexample (L = 1; (Delta_hat, V) in {(0, 1/2), (-1, 2)}): enumeration gives 1, min_t max gives 13/12.

Proposal (``propose_dp``): argmax of J_hat under costs rounded UP (a subset of Pi_B), then every proposal is re-checked
against the original costs; a proposal that fails the re-check is replaced by the cheapest-arm policy (reported).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..streams.frontier import FEAS_TOL, enumerate_policies

__all__ = ["QFCStats", "make_stats", "l1_qstar", "l1_allpairs", "cell_terms", "pair_terms", "qfc_certificate_enum",
           "t_grid_default", "qfc_upper_dp", "propose_dp", "gap_report", "minmax_t_cont", "minmax_t_grid",
           "relaxed_feasible_mask", "DEFAULT_T_POINTS", "DEFAULT_UNITS"]

DEFAULT_T_POINTS = 40
DEFAULT_UNITS = 1024          # integer cost units per budget (cost grid resolution u = budget / DEFAULT_UNITS)
_ROUND_GUARD = 1e-9           # guard (in cost units) that keeps the round-down set a superset under float error


# =============================================================================================== statistics
@dataclass
class QFCStats:
    w: np.ndarray            # (S,)
    mu_hat: np.ndarray       # (S, A)
    n: np.ndarray            # (S, A) sample counts
    var_ucb: np.ndarray      # (S, A) sigma_bar^2 on the original outcome scale
    N: np.ndarray | None = None   # (S, A) pool sizes; n >= N => exhausted (exact, dropped)
    R: float = 1.0

    def __post_init__(self):
        self.w = np.asarray(self.w, dtype=float)
        self.mu_hat = np.asarray(self.mu_hat, dtype=float)
        self.n = np.asarray(self.n, dtype=float)
        self.var_ucb = np.broadcast_to(np.asarray(self.var_ucb, dtype=float), self.mu_hat.shape).copy()
        if self.N is not None:
            self.N = np.asarray(self.N, dtype=float)
        S, A = self.mu_hat.shape
        assert self.w.shape == (S,) and self.n.shape == (S, A)

    @property
    def S(self):
        return self.mu_hat.shape[0]

    @property
    def A(self):
        return self.mu_hat.shape[1]


def make_stats(w, mu_hat, n, N=None, x_v=None, var_ucb=None, binary=True, R=1.0) -> QFCStats:
    """Build ``QFCStats``. If var_ucb is None: binary pools use Lemma L2 (Bernstein inversion at exponent x_v, then
    max m(1-m) over the interval); non-binary outcomes in [0, R] use the deterministic Popoviciu bound R^2 / 4."""
    mu_hat = np.asarray(mu_hat, dtype=float)
    n = np.asarray(n, dtype=float)
    if var_ucb is None:
        if binary:
            from ..theory_checks.mc_l1 import bernstein_mu_ci, sigma2_ucb
            if x_v is None:
                raise ValueError("x_v (Lemma L2 exponent) is required for binary pools")
            Nn = np.full_like(n, np.inf) if N is None else np.asarray(N, dtype=float)
            lo, hi = bernstein_mu_ci(mu_hat, n, Nn, float(x_v))
            var_ucb = sigma2_ucb(lo, hi)
        else:
            var_ucb = np.full_like(mu_hat, R * R / 4.0)
    return QFCStats(w=w, mu_hat=mu_hat, n=n, var_ucb=var_ucb, N=N, R=R)


def l1_qstar(S, A, Q, K, delta_main):
    """Q*-union exponent (qfc_lemma section 9 option 1): x = S ln A + ln(Q K / delta_main)."""
    return S * math.log(A) + math.log(Q * K / delta_main)


def l1_allpairs(S, A, K, delta_main):
    """All-ordered-pairs union exponent (qfc_lemma section 4): x = 2 S ln A + ln(K / delta_main)."""
    return 2 * S * math.log(A) + math.log(K / delta_main)


def cell_terms(stats: QFCStats):
    """Per-cell variance term v = w_s^2 var_ucb / n and low-order term bc = R w_s / n.
    Exhausted cells (n >= N) -> 0 (exact); cells with n = 0 -> +inf."""
    w2 = (stats.w ** 2)[:, None]
    n = stats.n
    zero = n <= 0
    live = ~zero
    if stats.N is not None:
        live &= n < stats.N
    nn = np.where(zero, 1.0, n)
    v = np.where(zero, np.inf, np.where(live, w2 * stats.var_ucb / nn, 0.0))
    bc = np.where(zero, np.inf, np.where(live, stats.R * stats.w[:, None] / nn, 0.0))
    return v, bc, live


def _b_bar(stats: QFCStats):
    """Global low-order constant b_bar = R * max_{live c, n_c >= 1} (1 / n_c) * max_s w_s (>= b of every pair)."""
    _, _, live = cell_terms(stats)
    if not live.any():
        return 0.0
    return float(stats.R * np.max(1.0 / stats.n[live]) * np.max(stats.w))


def pair_terms(stats: QFCStats, pols: np.ndarray, pi_hat):
    """(Delta_hat, V_bar, b) of every row of pols against pi_hat (vectorised). inf where a touched cell has n = 0."""
    pi_hat = np.asarray(pi_hat, dtype=np.int64)
    S = stats.S
    v, bc, _ = cell_terms(stats)
    seg = np.arange(S)[None, :]
    diff = pols != pi_hat[None, :]
    mu_p = stats.mu_hat[seg, pols]
    mu_h = stats.mu_hat[np.arange(S), pi_hat][None, :]
    dhat = (stats.w[None, :] * np.where(diff, mu_p - mu_h, 0.0)).sum(1)
    vh = v[np.arange(S), pi_hat][None, :]
    bh = bc[np.arange(S), pi_hat][None, :]
    V = np.where(diff, v[seg, pols] + vh, 0.0).sum(1)
    b = np.where(diff, np.maximum(bc[seg, pols], bh), 0.0).max(1)
    return dhat, V, b


def _width(V, b, L1):
    with np.errstate(invalid="ignore"):
        out = np.sqrt(2.0 * L1 * V) + b * L1 / 3.0
    return np.where(np.isinf(V) | np.isinf(b), np.inf, out)


def _feasible(pols, w, problem):
    cost = (np.asarray(w)[None, :] * np.asarray(problem.kappa, dtype=float)[pols]).sum(1)
    return cost <= problem.budget + FEAS_TOL, cost


# =============================================================================================== main certificate
def qfc_certificate_enum(stats: QFCStats, problems, L1: float, eps, pols=None, pi_hat=None):
    """Exact-enumeration QFC certificate for every problem.

    eps: float or per-problem sequence. pi_hat: optional per-problem override (any policy in Pi_{B_q}; default the
    argmax of J_hat over Pi_{B_q}, lowest enumeration index on ties).
    Returns a list of dicts: qid, U, certified, pi_hat, J_hat, worst (challenger attaining U), worst_dhat, worst_V,
    worst_b, n_feasible.
    """
    pols = enumerate_policies(stats.S, stats.A) if pols is None else pols
    seg = np.arange(stats.S)[None, :]
    Jhat = (stats.w[None, :] * stats.mu_hat[seg, pols]).sum(1)
    eps_list = list(eps) if np.ndim(eps) else [float(eps)] * len(problems)
    out = []
    for qi, p in enumerate(problems):
        feas, _ = _feasible(pols, stats.w, p)
        if pi_hat is not None and pi_hat[qi] is not None:
            ph = np.asarray(pi_hat[qi], dtype=np.int64)
            match = np.flatnonzero((pols == ph[None, :]).all(1))
            if match.size == 0 or not feas[match[0]]:
                raise ValueError(f"pi_hat for problem {qi} is not in Pi_B")
            ih = int(match[0])
        else:
            ih = int(np.argmax(np.where(feas, Jhat, -np.inf)))
        dhat, V, b = pair_terms(stats, pols, pols[ih])
        val = np.where(feas, dhat + _width(V, b, L1), -np.inf)
        k = int(np.argmax(val))
        U = float(val[k])
        out.append({"qid": getattr(p, "qid", str(qi)), "U": U, "certified": bool(U <= eps_list[qi]),
                    "eps": float(eps_list[qi]), "pi_hat": tuple(int(x) for x in pols[ih]), "J_hat": float(Jhat[ih]),
                    "worst": tuple(int(x) for x in pols[k]), "worst_dhat": float(dhat[k]), "worst_V": float(V[k]),
                    "worst_b": float(b[k]), "n_feasible": int(feas.sum())})
    return out


# =============================================================================================== t-envelope helpers
def minmax_t_cont(dhat, V, L, iters=300):
    """inf_{t > 0} max_i [dhat_i + L/t + t V_i / 2] for finite candidate arrays (convex in log t; golden section).
    Returns (value, t*). If every V_i = 0 the infimum is max dhat (t -> inf)."""
    dhat = np.asarray(dhat, dtype=float)
    V = np.asarray(V, dtype=float)
    if np.any(np.isinf(V)):
        return float("inf"), float("nan")
    if not np.any(V > 0):
        return float(dhat.max()), float("inf")

    def h(u):
        t = math.exp(u)
        return float(np.max(dhat + L / t + t * V / 2.0))

    Vp = V[V > 0]
    lo = math.log(math.sqrt(2 * L / Vp.max())) - 12.0
    hi = math.log(math.sqrt(2 * L / Vp.min())) + 12.0
    g = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = h(c), h(d)
    for _ in range(iters):
        if fc <= fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = h(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = h(d)
    u = (a + b) / 2
    val = min(h(u), fc, fd)
    return val, math.exp(u)


def minmax_t_grid(dhat, V, L, grid):
    """min_{t in grid} max_i [dhat_i + L/t + t V_i / 2]. Returns (value, t*)."""
    dhat = np.asarray(dhat, dtype=float)[None, :]
    V = np.asarray(V, dtype=float)[None, :]
    t = np.asarray(grid, dtype=float)[:, None]
    vals = (dhat + L / t + t * V / 2.0).max(1)
    k = int(np.argmin(vals))
    return float(vals[k]), float(grid[k])


def t_grid_default(stats: QFCStats, L1: float, n_points=DEFAULT_T_POINTS):
    """Data-dependent geometric grid covering the per-pair optima t = sqrt(2 L / V) (any finite grid is valid)."""
    v, _, _ = cell_terms(stats)
    vf = v[np.isfinite(v) & (v > 0)]
    if vf.size == 0:
        return np.geomspace(1.0, 1e6, n_points)
    v_lo = float(vf.min())
    v_hi = float(2.0 * np.sort(vf)[::-1][: 2 * stats.S].sum())
    t_lo = math.sqrt(2 * L1 / v_hi) / 2.0
    t_hi = math.sqrt(2 * L1 / v_lo) * 2.0
    return np.geomspace(t_lo, t_hi, n_points)


# =============================================================================================== cost rounding
def _unit(problem, w):
    B = float(problem.budget)
    if B > 0:
        return B / DEFAULT_UNITS
    c = float(np.max(np.asarray(w)) * np.max(np.asarray(problem.kappa, dtype=float)))
    return (c if c > 0 else 1.0) / DEFAULT_UNITS


def _round_costs(stats_w, problem, mode, units=None):
    """Integer costs on the grid u. mode 'down': floor(c/u - guard) with capacity floor((B + tol)/u + guard), so every
    policy feasible under the original costs stays feasible (superset). mode 'up': ceil(c/u) with capacity floor(B/u)
    (subset up to float error; proposals are re-checked against the original costs)."""
    w = np.asarray(stats_w, dtype=float)
    kappa = np.asarray(problem.kappa, dtype=float)
    B = float(problem.budget)
    u = _unit(problem, w) if units is None else (B if B > 0 else 1.0) / units
    c = w[:, None] * kappa[None, :] / u
    if mode == "down":
        ic = np.maximum(np.floor(c - _ROUND_GUARD), 0).astype(np.int64)
        cap = int(math.floor((B + FEAS_TOL) / u + _ROUND_GUARD))
    elif mode == "up":
        ic = np.ceil(c).astype(np.int64)
        cap = int(math.floor(B / u))
    elif mode == "naive_up":          # control only: rounds UP but is used as if it were a relaxation (invalid)
        ic = np.ceil(c).astype(np.int64)
        cap = int(math.floor(B / u))
    else:
        raise ValueError(mode)
    return ic, max(cap, -1)


def relaxed_feasible_mask(pols, w, problem, mode="down", units=None):
    ic, cap = _round_costs(w, problem, mode, units)
    tot = ic[np.arange(pols.shape[1])[None, :], pols].sum(1)
    return tot <= cap


def _mck_max(vals, ic, cap, track=False):
    """Multiple-choice knapsack: max over policies of sum_s vals[..., s, pi(s)] with sum_s ic[s, pi(s)] <= cap.
    vals: (T, S, A) with -inf for excluded choices. Returns (T,) maxima (and per-segment choice tables if track)."""
    T, S, A = vals.shape
    if cap < 0:
        return np.full(T, -np.inf), None
    f = np.full((T, cap + 1), -np.inf)
    f[:, 0] = 0.0
    choices = np.zeros((S, T, cap + 1), dtype=np.int16) if track else None
    for s in range(S):
        g = np.full((T, cap + 1), -np.inf)
        ch = np.zeros((T, cap + 1), dtype=np.int16) if track else None
        for a in range(A):
            cst = int(ic[s, a])
            if cst > cap:
                continue
            col = vals[:, s, a]
            if np.all(np.isneginf(col)):
                continue
            cand = np.full((T, cap + 1), -np.inf)
            cand[:, cst:] = f[:, :cap + 1 - cst] + col[:, None]
            better = cand > g
            g = np.where(better, cand, g)
            if track:
                ch = np.where(better, a, ch)
        f = g
        if track:
            choices[s] = ch
    return f.max(1), (choices, f)


def _backtrack(choices, f_last, ic, ti):
    S = choices.shape[0]
    j = int(np.argmax(f_last[ti]))
    pi = [0] * S
    for s in range(S - 1, -1, -1):
        a = int(choices[s, ti, j])
        pi[s] = a
        j -= int(ic[s, a])
    return tuple(pi)


# =============================================================================================== proposal
def propose_dp(stats: QFCStats, problem, units=None):
    """argmax J_hat over the rounded-UP feasible set (DP), then re-check under the original costs.
    Returns dict(pi, J_hat, cost, verified, fallback)."""
    ic, cap = _round_costs(stats.w, problem, "up", units)
    vals = (stats.w[:, None] * stats.mu_hat)[None, :, :]
    best, aux = _mck_max(vals, ic, cap, track=True)
    kappa = np.asarray(problem.kappa, dtype=float)
    fallback = False
    if np.isfinite(best[0]):
        pi = _backtrack(aux[0], aux[1], ic, 0)
    else:
        pi = tuple(int(np.argmin(kappa)) for _ in range(stats.S))
        fallback = True
    cost = float((stats.w * kappa[list(pi)]).sum())
    verified = bool(cost <= problem.budget + FEAS_TOL)
    if not verified:
        pi = tuple(int(np.argmin(kappa)) for _ in range(stats.S))
        cost = float((stats.w * kappa[list(pi)]).sum())
        fallback = True
    J = float((stats.w * stats.mu_hat[np.arange(stats.S), list(pi)]).sum())
    return {"pi": pi, "J_hat": J, "cost": cost, "verified": verified,
            "feasible": bool(cost <= problem.budget + FEAS_TOL), "fallback": fallback}


# =============================================================================================== DP upper bound
def _dp_vals(stats: QFCStats, pi_hat, t_grid):
    """Per-(t, s, a) additive objective: w_s (mu_hat[s,a] - mu_hat[s,pi_hat(s)]) + [a != pi_hat(s)] t/2 (v[s,a] +
    v[s,pi_hat(s)]). Choices with an n = 0 cell are +inf."""
    S, A = stats.S, stats.A
    pi_hat = np.asarray(pi_hat, dtype=np.int64)
    v, _, _ = cell_terms(stats)
    base = stats.w[:, None] * (stats.mu_hat - stats.mu_hat[np.arange(S), pi_hat][:, None])
    vpair = v + v[np.arange(S), pi_hat][:, None]
    other = np.ones((S, A), dtype=bool)
    other[np.arange(S), pi_hat] = False
    vpair = np.where(other, vpair, 0.0)
    base = np.where(other, base, 0.0)
    t = np.asarray(t_grid, dtype=float)[:, None, None]
    with np.errstate(invalid="ignore"):
        vals = base[None] + t * vpair[None] / 2.0
    vals = np.where(np.isinf(vpair)[None], np.inf, vals)
    return vals


def qfc_upper_dp(stats: QFCStats, problem, L1: float, pi_hat=None, t_grid=None, units=None, track=True,
                 rounding="down"):
    """Conservative upper bound U_DP >= U_q (see module docstring). pi_hat defaults to ``propose_dp``.

    Returns dict(U_dp, t_star, worst (challenger at t_star in the relaxed set), b_bar, pi_hat, proposal, cap).
    ``rounding='naive_up'`` is a deliberately invalid control used only to check that the tests can detect violations.
    """
    prop = None
    if pi_hat is None:
        prop = propose_dp(stats, problem, units)
        pi_hat = prop["pi"]
    pi_hat = tuple(int(x) for x in pi_hat)
    if t_grid is None:
        t_grid = t_grid_default(stats, L1)
    t_grid = np.asarray(t_grid, dtype=float)
    ic, cap = _round_costs(stats.w, problem, rounding, units)
    vals = _dp_vals(stats, pi_hat, t_grid)
    bbar = _b_bar(stats)
    # +inf choices: reachable in the relaxed set => U_DP = +inf; unreachable => drop them (exact for the relaxed set)
    infm = np.isinf(vals[0])
    if infm.any():
        min_ic = ic.min(1)
        rest = min_ic.sum() - min_ic
        reach = (rest[:, None] + ic) <= cap
        if np.any(infm & reach):
            return {"U_dp": float("inf"), "t_star": float("nan"), "worst": None, "b_bar": bbar, "pi_hat": pi_hat,
                    "proposal": prop, "cap": cap}
        vals = np.where(infm[None], -np.inf, vals)
    inner, aux = _mck_max(vals, ic, cap, track=track)
    total = inner + L1 / t_grid + bbar * L1 / 3.0
    ti = int(np.argmin(total))
    worst = _backtrack(aux[0], aux[1], ic, ti) if (track and np.isfinite(inner[ti])) else None
    return {"U_dp": float(total[ti]), "t_star": float(t_grid[ti]), "worst": worst, "b_bar": bbar, "pi_hat": pi_hat,
            "proposal": prop, "cap": cap, "inner": inner, "t_grid": t_grid}


# =============================================================================================== gap decomposition
def gap_report(stats: QFCStats, problem, L1: float, pi_hat=None, t_grid=None, units=None, pols=None):
    """Telescoping decomposition U_DP - U_enum = gap_i + gap_ii + gap_iii + gap_iv on an enumerable instance,
    all computed against the same pi_hat (default: the DP proposal, i.e. what the DP pipeline would certify).

        U_enum = max_{Pi_B} [dhat + sqrt(2 L V) + b L/3]
        U_iv   = max_{Pi_B} [dhat + sqrt(2 L V)] + b_bar L/3                 gap_iv  = U_iv  - U_enum  (low-order)
        U_iii  = inf_t max_{Pi_B} [dhat + L/t + t V/2] + b_bar L/3           gap_iii = U_iii - U_iv    (min-max order)
        U_ii   = min_{t in T} max_{Pi_B} [...] + b_bar L/3                   gap_ii  = U_ii  - U_iii   (t grid)
        U_DP   = min_{t in T} max_{Pi_B^down} [...] + b_bar L/3 (DP)         gap_i   = U_DP  - U_ii    (cost rounding)
    Also reports U_enum at the enumeration argmax (proposal effect, outside the four gaps), the DP-vs-enumeration check
    on the relaxed set, and the proposal's J_hat shortfall.
    """
    pols = enumerate_policies(stats.S, stats.A) if pols is None else pols
    feas, _ = _feasible(pols, stats.w, problem)
    seg = np.arange(stats.S)[None, :]
    Jhat = (stats.w[None, :] * stats.mu_hat[seg, pols]).sum(1)
    i_enum = int(np.argmax(np.where(feas, Jhat, -np.inf)))
    prop = propose_dp(stats, problem, units)
    if pi_hat is None:
        pi_hat = prop["pi"]
    pi_hat = tuple(int(x) for x in pi_hat)
    if t_grid is None:
        t_grid = t_grid_default(stats, L1)
    t_grid = np.asarray(t_grid, dtype=float)
    dhat, V, b = pair_terms(stats, pols, pi_hat)
    bbar = _b_bar(stats)
    f_dh, f_V, f_b = dhat[feas], V[feas], b[feas]
    U_enum = float(np.max(f_dh + _width(f_V, f_b, L1)))
    U_iv = float(np.max(f_dh + _width(f_V, np.zeros_like(f_b), L1))) + bbar * L1 / 3.0
    U_iii, t_cont = minmax_t_cont(f_dh, f_V, L1)
    U_iii += bbar * L1 / 3.0
    U_ii, t_ii = minmax_t_grid(f_dh, f_V, L1, t_grid)
    U_ii += bbar * L1 / 3.0
    dp = qfc_upper_dp(stats, problem, L1, pi_hat=pi_hat, t_grid=t_grid, units=units, track=False)
    U_dp = dp["U_dp"]
    # DP (per t, rounded costs) vs explicit enumeration of the relaxed set
    rmask = relaxed_feasible_mask(pols, stats.w, problem, "down", units)
    superset = bool(np.all(rmask[feas]))
    if np.isfinite(U_dp):
        tt = t_grid[:, None]
        enum_inner = np.where(rmask[None, :], dhat[None, :] + tt * V[None, :] / 2.0, -np.inf).max(1)
        dp_check = float(np.max(np.abs(enum_inner - dp["inner"])))
    else:
        dp_check = float("nan")
    dh_e, V_e, b_e = pair_terms(stats, pols, pols[i_enum])
    U_enum_argmax = float(np.max(np.where(feas, dh_e + _width(V_e, b_e, L1), -np.inf)))
    gaps = {"gap_i_cost_rounding": U_dp - U_ii, "gap_ii_t_grid": U_ii - U_iii, "gap_iii_minmax": U_iii - U_iv,
            "gap_iv_low_order": U_iv - U_enum}
    return {"U_enum": U_enum, "U_iv": U_iv, "U_iii": U_iii, "U_ii": U_ii, "U_dp": U_dp, **gaps,
            "gap_total": U_dp - U_enum, "violation": bool(U_dp < U_enum - 1e-12), "t_cont": t_cont, "t_grid_star": t_ii,
            "t_dp_star": dp["t_star"], "b_bar": bbar, "pi_hat": pi_hat, "pi_hat_enum": tuple(int(x) for x in pols[i_enum]),
            "U_enum_at_enum_argmax": U_enum_argmax, "proposal_Jhat_shortfall": float(Jhat[i_enum] - prop["J_hat"]),
            "proposal_feasible": prop["feasible"], "proposal_fallback": prop["fallback"],
            "relaxed_superset": superset, "dp_vs_enum_relaxed_maxabs": dp_check,
            "n_feasible": int(feas.sum()), "n_relaxed": int(rmask.sum())}
