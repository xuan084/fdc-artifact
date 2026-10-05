"""FDC-DP: the FDC-BF joint certificate over exponentially large segment-policy classes (v10, NEW file).

Theory: plan/fdc_dp_theory.md (validity, the two computable schemes, exactness, ledger, TU).  Locked modules
(fdc_bet, rect_v6, b4_bal, ...) are imported and never modified.

Problem.  S segments, A arms, integer costs kappa[s, a] >= 0, Q budgets B_q; the policy class of problem q is
Pi_q = {pi in [A]^S : sum_s kappa[s, pi(s)] <= B_q} (never enumerated).  Ledger M = sum_q |Pi_q| by a counting DP
(exact Python integers), beta_J = ln(M K / delta_main).

Certificate (identical rule to FDC-BF, Bennett-FPC cells, exact-HG variance box, delta split 0.045 / 0.005):
    U_q = max_{pi in Pi_q, pi != pi_hat} min_{j in J} C_j(pi),   certified iff U_q <= eps,
with "columns" C_j(pi) = const_j + sum_s T[s, pi(s), j]  (T[s, pi_hat(s), j] = 0):
    joint column lambda in Lambda : const = beta / lambda,
        T[s, a] = w_s (mu_hat[s, a] - mu_hat[s, pi_hat(s)]) + (Psi_bar_{s,a}(lambda w_s) + Psi_bar_{s,pi_hat(s)}(lambda w_s)) / lambda
    exact column (lambda = inf)  : const = 0, T[s, a] = w_s (mu_hat[s,a] - mu_hat[s,pi_hat(s)]) if both cells are
        deterministic on E_var (exhausted, or zero box variance), else +inf  (FDC-BF's width-0 rule)
    rectangle column (hybrid only): const = 0, T[s, a] = w_s (hi_R[s, a] - lo_R[s, pi_hat(s)])
Cells with n = 0 give +inf in every column (FDC-BF's zero_touch rule).

Schemes
  'a'  single-column-per-check:  U_a = min_j [const_j + max_{pi != pi_hat} sum_s T[s, pi(s), j]] >= U_q
       (max-min <= min-max); the inner max is a multiple-choice knapsack DP with a "differs" flag.  Sound, conservative.
  'b'  exact decision when the search completes: depth-first branch-and-bound over segments; node bound = scheme (a)
       on the subtree (suffix DP tables), leaves exact.  Contract (external reviewer FDC-DP r1 B3): if the search completes it
       returns exactly [U_q <= eps] (decision mode) or U_q (max mode); if the node cap is reached it ABSTAINS ("not
       certified", sound, counted).  The capped algorithm is therefore not an unconditional iff implementation.
  Reported fields (B3): the third element of a certify() answer is U_a, the scheme-(a) UPPER BOUND on U_q, never an
  exact certificate value; ``last_detail`` holds per problem {U_a, U_lower (exact U_q >= U_lower when scheme (b) found
  a violator leaf), U_exact (only with exact_U=True: max-mode B&B), decision source}.
  Centre (external reviewer FDC-DP r1 B1): pi_hat_q = argmax_{pi in Pi_q} sum_s w_s mu_hat[s, pi(s)] -- the WEIGHTED empirical
  value, i.e. dp_argmax(w[:, None] * mu_hat, ...).
  Checkpoint interface (B4): fresh-width FDCDP.certify accepts only times on the predeclared grid ctx.checkpoints
  (validated at setup); denser monitoring goes only through FDCDPTimeUniform (frozen block widths).
  enumeration reference ('enum_U') for small S: identical columns, explicit policies (ground truth for the tests).

Lambda grid: global per (checkpoint, centre-independent): geometric with FDC-BF's density (ratio 2000^(1/160)), from
sqrt(2 beta / V_max) / 200 to max(10 sqrt(2 beta / V_min), beta / eps); any data-dependent lambda set is valid
(plan/fdc_dp_theory.md s3).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .b4_bal import balanced_alloc
from .fdc_bet import FDC_BET_DELTA, psi_bar
from .frontier_common import Method
from .rect_v6 import hg_mean_interval

__all__ = ["SegProblems", "SegCtx", "make_seg_ctx", "reduce_costs", "count_policies", "dp_ledger", "dp_argmax",
           "lambda_grid", "cell_tables", "joint_columns", "dp_max_cols", "suffix_tables", "bnb", "enum_U",
           "scheme_a_U", "FDCDP", "FDCDPTimeUniform", "GRID_RATIO", "DEFAULT_NODE_LIMIT"]

GRID_RATIO = math.exp(math.log(2000.0) / 160.0)      # FDC-BF's lambda grid density (161 points over a factor 2000)
DEFAULT_NODE_LIMIT = 200_000
_NEG = -np.inf


# =============================================================================================== problems / ctx
@dataclass
class SegProblems:
    """Integer-cost budget problems over an implicit per-segment policy class."""
    cost: np.ndarray                 # (S, A) int >= 0
    budgets: np.ndarray              # (Q,) int
    qids: list
    budget_frac: list = field(default_factory=list)

    def __post_init__(self):
        self.cost = np.asarray(self.cost, dtype=np.int64)
        self.budgets = np.asarray(self.budgets, dtype=np.int64)
        if (self.cost < 0).any():
            raise ValueError("costs must be >= 0")

    @property
    def Q(self):
        return int(len(self.budgets))

    def feasible(self, pi, q):
        pi = np.asarray(pi, dtype=np.int64)
        return int(self.cost[np.arange(len(pi)), pi].sum()) <= int(self.budgets[q])


@dataclass
class SegCtx:
    """Learner-side public context for implicit policy classes (FrontierCtx without ``pols`` / ``feas``)."""
    w: np.ndarray
    S: int
    A: int
    sp: SegProblems
    eps: float
    delta: float
    checkpoints: np.ndarray
    N: np.ndarray
    tau_R: int
    stop_k: int
    binary: bool = True
    R: float = 1.0
    replan: int = 200
    extra: dict = field(default_factory=dict)

    @property
    def Q(self):
        return self.sp.Q

    @property
    def problems(self):
        return list(self.sp.qids)


def make_seg_ctx(w, N, sp, eps, delta, checkpoints, tau_R, replan=200, stop_frac=0.8):
    N = np.asarray(N, dtype=np.int64)
    S, A = N.shape
    return SegCtx(w=np.asarray(w, dtype=float), S=S, A=A, sp=sp, eps=float(eps), delta=float(delta),
                  checkpoints=np.asarray(checkpoints, dtype=np.int64), N=N, tau_R=int(tau_R),
                  stop_k=int(math.ceil(stop_frac * sp.Q - 1e-9)), replan=int(replan))


def reduce_costs(cost, budgets):
    """Divide costs by their gcd g; budgets become floor(B / g) (sum of multiples of g <= B <=> <= floor(B/g))."""
    cost = np.asarray(cost, dtype=np.int64)
    budgets = np.asarray(budgets, dtype=np.int64)
    nz = cost[cost > 0]
    g = int(np.gcd.reduce(nz)) if nz.size else 1
    g = max(g, 1)
    return cost // g, np.floor_divide(budgets, g), g


# =============================================================================================== counting / ledger
def count_policies(cost, budgets):
    """Exact |Pi_q| for every budget (Python ints).  Counting DP over exact cost; budgets < 0 -> 0."""
    cost, budgets, _ = reduce_costs(cost, budgets)
    S, A = cost.shape
    cap = int(max(int(budgets.max()), 0))
    cnt = [0] * (cap + 1)
    cnt[0] = 1
    for s in range(S):
        new = [0] * (cap + 1)
        for a in range(A):
            k = int(cost[s, a])
            if k > cap:
                continue
            for c in range(cap - k + 1):
                if cnt[c]:
                    new[c + k] += cnt[c]
        cnt = new
    pref = [0] * (cap + 1)
    run = 0
    for c in range(cap + 1):
        run += cnt[c]
        pref[c] = run
    return [pref[int(b)] if b >= 0 else 0 for b in budgets]


def dp_ledger(sp: SegProblems, S, A, K, delta_main=0.045, delta_var=0.005):
    counts = count_policies(sp.cost, sp.budgets)
    M = int(sum(counts))
    beta = math.log(M) + math.log(K / delta_main)
    return {"union_size": M, "union_size_log": math.log(M), "counts": [str(c) for c in counts], "K": int(K),
            "S_A": int(S * A), "delta_main": delta_main, "delta_var": delta_var,
            "delta_total": delta_main + delta_var, "beta": beta,
            "beta_upper_SlnA": S * math.log(A) + math.log(sp.Q * K / delta_main),
            "alpha_side_var": delta_var / (2.0 * S * A * K), "var_box": "HG"}


# =============================================================================================== DP primitives
def dp_argmax(vals, cost, budget):
    """argmax_{pi: sum cost <= budget} sum_s vals[s, pi(s)] (exact multiple-choice knapsack DP with traceback).

    Ties: the lowest arm index wins at every segment (deterministic).  Returns (value, pi) or (-inf, None)."""
    vals = np.asarray(vals, dtype=float)
    cost, (budget,), _ = reduce_costs(cost, [budget])
    S, A = vals.shape
    cap = int(budget)
    if cap < 0:
        return _NEG, None
    # backward table: best[s][c] = max value of segments s.. with cost <= c
    best = np.full((S + 1, cap + 1), _NEG)
    best[S, :] = 0.0
    for s in range(S - 1, -1, -1):
        row = np.full(cap + 1, _NEG)
        for a in range(A):
            k = int(cost[s, a])
            if k > cap:
                continue
            cand = np.full(cap + 1, _NEG)
            cand[k:] = best[s + 1, :cap + 1 - k] + vals[s, a]
            row = np.maximum(row, cand)
        best[s] = row
    if not np.isfinite(best[0, cap]):
        return _NEG, None
    pi = np.zeros(S, dtype=np.int64)
    c = cap
    for s in range(S):
        tgt = best[s, c]
        for a in range(A):
            k = int(cost[s, a])
            if k <= c and best[s + 1, c - k] + vals[s, a] >= tgt - 1e-15 * max(1.0, abs(tgt)):
                pi[s] = a
                c -= k
                break
    return float(best[0, cap]), pi


def _shift_max(dst, src, t, k):
    """dst[:, k:] = max(dst[:, k:], src[:, :C+1-k] + t[:, None]) with -inf + inf -> -inf (infeasible)."""
    C1 = dst.shape[1]
    if k >= C1:
        return
    with np.errstate(invalid="ignore"):
        v = src[:, :C1 - k] + t[:, None]
    v = np.where(np.isnan(v), _NEG, v)
    np.maximum(dst[:, k:], v, out=dst[:, k:])


def dp_max_cols(T, cost, diff, cap):
    """For every column j: max over pi with sum cost <= c and >= 1 differing segment of sum_s T[s, pi(s), j], for every
    c = 0..cap.  Returns (J, cap+1) array (-inf where infeasible).  Forward DP with a 'differs' flag."""
    S, A, J = T.shape
    f0 = np.zeros((J, cap + 1))
    f1 = np.full((J, cap + 1), _NEG)
    for s in range(S):
        n0 = np.full((J, cap + 1), _NEG)
        n1 = np.full((J, cap + 1), _NEG)
        for a in range(A):
            k = int(cost[s, a])
            t = T[s, a]
            if diff[s, a]:
                _shift_max(n1, f0, t, k)
                _shift_max(n1, f1, t, k)
            else:
                _shift_max(n0, f0, t, k)
                _shift_max(n1, f1, t, k)
        f0, f1 = n0, n1
    return f1


def suffix_tables(T, cost, diff, order, cap):
    """free[i], need[i] (J, cap+1): best sum over segments order[i:] with cost <= c, any / with >= 1 difference."""
    S, A, J = T.shape
    L = len(order)
    free = np.empty((L + 1, J, cap + 1))
    need = np.empty((L + 1, J, cap + 1))
    free[L] = 0.0
    need[L] = _NEG
    for i in range(L - 1, -1, -1):
        s = order[i]
        fr = np.full((J, cap + 1), _NEG)
        nd = np.full((J, cap + 1), _NEG)
        for a in range(A):
            k = int(cost[s, a])
            t = T[s, a]
            _shift_max(fr, free[i + 1], t, k)
            _shift_max(nd, free[i + 1] if diff[s, a] else need[i + 1], t, k)
        free[i] = fr
        need[i] = nd
    return free, need


def scheme_a_U(const, T, cost, diff, caps):
    """Scheme (a): U_a(cap) = min_j [const_j + max_{pi != pi_hat, cost <= cap} sum_s T].  -inf: no challenger."""
    caps = [int(c) for c in caps]
    f1 = dp_max_cols(T, cost, diff, max(max(caps), 0))
    out = []
    for c in caps:
        if c < 0:
            out.append(_NEG)
            continue
        col = f1[:, c]
        if not np.isfinite(col).any() and np.all(col == _NEG):
            out.append(_NEG)
            continue
        with np.errstate(invalid="ignore"):
            v = const + col
        out.append(float(np.min(v)))
    return out


def bnb(const, T, cost, diff, cap, thr=None, node_limit=DEFAULT_NODE_LIMIT, prune_dominated=True, order=None):
    """Exact max_{pi != pi_hat, cost <= cap} min_j [const_j + sum_s T[s, pi(s), j]] by branch-and-bound.

    thr given (decision mode): returns dict(status in {'certified', 'violator', 'node_limit'}, ...): 'certified' iff the
    exact maximum is <= thr.  thr None (max mode): exact maximum in 'U' (status 'exact' or 'node_limit').
    Dominance pruning (decision mode only, plan s4.3): an option a != pi_hat(s) with T[s, a, j] <= 0 for every j and
    cost >= the centre's is dropped when min_j const_j <= thr (lossless for the decision)."""
    S, A, J = T.shape
    cost = np.asarray(cost, dtype=np.int64)
    ph_cost = np.array([int(cost[s][~diff[s]][0]) for s in range(S)], dtype=np.int64)
    keep = np.ones((S, A), dtype=bool)
    decision = thr is not None
    if decision and prune_dominated and float(np.min(const)) <= thr:
        dom = diff & (T <= 0).all(2) & (cost >= ph_cost[:, None])
        keep &= ~dom
    active = [s for s in range(S) if (keep[s] & diff[s]).any()]
    base = int(sum(ph_cost[s] for s in range(S) if s not in set(active)))
    capr = int(cap) - base
    info = {"nodes": 0, "n_active": len(active), "status": None, "U": None, "witness": None}
    if capr < 0 or not active:
        info.update(status="certified" if decision else "exact", U=_NEG)
        return info
    if order is None:
        # heuristic: the root's best column, segments by their largest flip value in it (most dangerous first)
        T0 = np.where(keep[..., None], T, _NEG)
        Tf = np.where(np.isinf(T0) & (T0 > 0), 1e300, T0)
        jstar = int(np.argmin(const))
        val = np.array([np.max(np.where(diff[s] & keep[s], Tf[s, :, jstar], _NEG)) for s in active])
        order = [active[i] for i in np.argsort(-val, kind="stable")]
    Tk = np.where(keep[..., None], T, np.inf)        # dropped options never chosen: cost them out below
    costk = np.where(keep, cost, capr + 1 + int(cost.max()) + 1)
    free, need = suffix_tables(Tk, costk, diff, order, capr)
    L = len(order)
    best = _NEG if not decision else None
    bound = thr if decision else _NEG
    stack = [(0, 0, False, np.zeros(J))]
    nodes = 0
    while stack:
        i, used, hd, part = stack.pop()
        s = order[i]
        kids = []
        for a in range(A):
            if not keep[s, a]:
                continue
            k = int(costk[s, a])
            if used + k > capr:
                continue
            hd2 = hd or bool(diff[s, a])
            tail = (free if hd2 else need)[i + 1][:, capr - used - k]
            if tail[0] == _NEG:
                continue
            with np.errstate(invalid="ignore"):
                p2 = part + Tk[s, a]
                ub = float(np.min(const + p2 + tail))
            if np.isnan(ub):
                ub = np.inf
            kids.append((ub, a, k, hd2, p2))
        nodes += 1
        for ub, a, k, hd2, p2 in sorted(kids, key=lambda x: x[0]):      # push lowest first -> highest popped first
            if i + 1 == L:
                # leaf: ub is the exact value of this policy
                if decision:
                    if ub > thr:
                        info.update(status="violator", nodes=nodes, U=ub, witness=None)
                        return info
                elif ub > bound:
                    bound = ub
                    best = ub
                continue
            if ub > bound:
                stack.append((i + 1, used + k, hd2, p2))
        if nodes >= node_limit and stack:
            info.update(status="node_limit", nodes=nodes, U=None if decision else best)
            return info
    info["nodes"] = nodes
    if decision:
        info["status"] = "certified"
    else:
        info.update(status="exact", U=best)
    return info


def enum_U(const, T, cost, cap, ph):
    """Ground truth for small S: explicit enumeration of [A]^S; max over feasible pi != pi_hat of min_j C_j(pi)."""
    S, A, J = T.shape
    import itertools
    best = _NEG
    arg = None
    ph = np.asarray(ph)
    for pi in itertools.product(range(A), repeat=S):
        pi = np.asarray(pi)
        if (pi == ph).all() or int(cost[np.arange(S), pi].sum()) > cap:
            continue
        with np.errstate(invalid="ignore"):
            v = float(np.min(const + T[np.arange(S), pi].sum(0)))
        if v > best:
            best, arg = v, pi
    return best, arg


# =============================================================================================== cell tables / columns
def _vcell(n, N, lo, hi):
    vmax = np.clip(0.5, lo, hi)
    vmax = vmax * (1.0 - vmax)
    live = (n > 0) & (n < N)
    fp = (N - n) / np.maximum(N - 1.0, 1.0)
    return np.where(live, vmax * fp / np.maximum(n, 1.0), 0.0), live


def lambda_grid(beta, eps, w, n, N, lo, hi, ratio=GRID_RATIO):
    """Global geometric lambda grid covering FDC-BF's per-direction ranges plus beta/eps (plan s3.3)."""
    n = np.asarray(n, float)
    N = np.asarray(N, float)
    v, live = _vcell(n, N, lo, hi)
    w2 = (np.asarray(w, float) ** 2)[:, None]
    pv = w2 * v
    pos = pv[pv > 0]
    top = max(beta / eps, 1e-12) if eps > 0 else 1.0
    if pos.size == 0:
        return np.array([top])
    vmax_tot = float(np.sum(2.0 * pv.max(1)))
    vmin = float(pos.min())
    lo_l = math.sqrt(2.0 * beta / vmax_tot) / 200.0
    hi_l = max(math.sqrt(2.0 * beta / vmin) * 10.0, top)
    m = int(math.ceil(math.log(hi_l / lo_l) / math.log(ratio)))
    return lo_l * ratio ** np.arange(m + 1)


def cell_tables(w, n, N, lo, hi, lam):
    """P[s, a, j] = sup_box Psi^Bennett-FPC_{s,a}(lambda_j w_s) (0 exhausted, +inf empty); det[s, a] = deterministic on
    E_var (exhausted, or live with zero box variance)."""
    n = np.asarray(n, float)
    N = np.asarray(N, float)
    w = np.asarray(w, float)
    S, A = n.shape
    v, live = _vcell(n, N, lo, hi)
    P = np.zeros((S, A, len(lam)))
    for s in range(S):
        for a in range(A):
            if n[s, a] <= 0:
                P[s, a] = np.inf
            elif live[s, a] and v[s, a] > 0:
                P[s, a] = psi_bar(lam * w[s], n[s, a], N[s, a], lo[s, a], hi[s, a], "bennett")
    P = np.where(np.isnan(P), np.inf, P)
    det = (n >= N) | (live & (v <= 0))
    return P, det


def joint_columns(w, mu_hat, ph, P, det, lam, beta, exact_col=True):
    """(const (J,), T (S, A, J), diff (S, A)) for centre ph; T[s, ph(s)] = 0."""
    w = np.asarray(w, float)
    mu = np.asarray(mu_hat, float)
    S, A = mu.shape
    seg = np.arange(S)
    dmu = w[:, None] * (mu - mu[seg, ph][:, None])                       # (S, A)
    Pc = P[seg, ph]                                                        # (S, J) centre cells
    with np.errstate(invalid="ignore", over="ignore"):
        T = dmu[:, :, None] + (P + Pc[:, None, :]) / lam[None, None, :]
    T = np.where(np.isnan(T), np.inf, T)
    const = beta / lam
    if exact_col:
        ex = np.where(det & det[seg, ph][:, None], dmu, np.inf)
        T = np.concatenate([T, ex[:, :, None]], axis=2)
        const = np.concatenate([const, [0.0]])
    diff = np.ones((S, A), dtype=bool)
    diff[seg, ph] = False
    T[seg, ph, :] = 0.0
    return const, T, diff


# =============================================================================================== method
class FDCDP(Method):
    """FDC-DP certificate (fixed 50/50 design, FDC-BF cells / box / split).  scheme 'a' | 'b'.

    hybrid=True: delta_main split equally between the joint family (beta = ln(2 M K / delta_main)) and a matched
    Bennett rectangle family (beta_C = ln(4 S A K / delta_main)); the rectangle enters as one more column."""

    alloc_kind = "fixed"
    validity = "rigorous"
    cs_kind = "none"

    def __init__(self, scheme="b", hybrid=False, node_limit=DEFAULT_NODE_LIMIT, split=(0.045, 0.005), name=None,
                 grid_ratio=GRID_RATIO, exact_U=False):
        if scheme not in ("a", "b"):
            raise ValueError(scheme)
        if abs(split[0] + split[1] - FDC_BET_DELTA) > 1e-12 or min(split) <= 0:
            raise ValueError("split must be positive and sum to 0.05")
        self.scheme, self.hybrid, self.node_limit = scheme, bool(hybrid), int(node_limit)
        self.split = (float(split[0]), float(split[1]))
        self.grid_ratio = float(grid_ratio)
        self.exact_U = bool(exact_U)
        self.last_detail = []
        self.name = name or f"FDC-DP({scheme}){'+R' if hybrid else ''}"
        self.alloc_p = None
        self.ledger = None

    # ------------------------------------------------------------------ setup / boxes
    def setup(self, ctx):
        if abs(float(ctx.delta) - FDC_BET_DELTA) > 1e-12:
            raise ValueError("FDC-DP: delta fixed at 0.05")
        if not ctx.binary:
            raise ValueError("FDC-DP: binary pools")
        self.alloc_p = balanced_alloc(ctx.S, ctx.A, 0.5)
        grid = np.asarray(self._block_grid(ctx), dtype=np.int64)
        if grid.ndim != 1 or len(grid) == 0 or (grid <= 0).any() or (np.diff(grid) <= 0).any():
            raise ValueError("FDC-DP: the checkpoint / block grid must be a non-empty strictly increasing positive grid")
        self._grid_set = set(int(x) for x in grid)
        K = int(len(grid))
        self.ledger = dp_ledger(ctx.sp, ctx.S, ctx.A, K, *self.split)
        if self.hybrid:
            dm = self.split[0]
            self.ledger["beta"] = math.log(self.ledger["union_size"]) + math.log(K / (dm / 2.0))
            self.ledger["beta_C"] = math.log(2 * ctx.S * ctx.A * K / (dm / 2.0))
            self.ledger["hybrid_split"] = (dm / 2.0, dm / 2.0)
        self._lo = np.zeros((ctx.S, ctx.A))
        self._hi = np.ones((ctx.S, ctx.A))
        self._rlo = np.zeros((ctx.S, ctx.A))
        self._rhi = np.ones((ctx.S, ctx.A))
        self._last_t = -1
        self.cert_stats = []

    def _block_grid(self, ctx):
        return ctx.checkpoints

    def _box(self, st):
        lo, hi = hg_mean_interval(st.N, st.n, st.sum, self.ledger["alpha_side_var"])
        self._lo = np.maximum(self._lo, lo)
        self._hi = np.maximum(np.minimum(self._hi, hi), self._lo)
        return self._lo.copy(), self._hi.copy()

    def _rect_bounds(self, st, lo_box, hi_box):
        """Matched Bennett rectangle (pjc_bf.RectCkBF(+box) formula at beta_C), running intersection."""
        from .pjc_bf import bennett_cell_radius
        n = np.asarray(st.n, float)
        N = np.asarray(st.N, float)
        s = np.asarray(st.sum, float)
        r = bennett_cell_radius(n, N, lo_box, hi_box, self.ledger["beta_C"])
        mu = np.where(n > 0, s / np.maximum(n, 1.0), 0.5)
        lo = np.maximum(mu - r, s / np.maximum(N, 1.0))
        hi = np.minimum(mu + r, (s + (N - n)) / np.maximum(N, 1.0))
        exact = n >= N
        lo = np.where(exact, mu, lo)
        hi = np.where(exact, mu, hi)
        lo = np.where(n <= 0, 0.0, lo)
        hi = np.where(n <= 0, 1.0, hi)
        lo = np.maximum(lo, lo_box)
        hi = np.minimum(hi, hi_box)
        self._rlo = np.maximum(self._rlo, np.clip(lo, 0, 1))
        self._rhi = np.maximum(np.minimum(self._rhi, np.clip(hi, 0, 1)), self._rlo)
        return self._rlo.copy(), self._rhi.copy()

    # ------------------------------------------------------------------ certificate
    def certify(self, ctx, st):
        if st.t <= self._last_t:
            raise RuntimeError("checkpoints must be visited in strictly increasing order")
        if int(st.t) not in self._grid_set:
            raise RuntimeError(f"FDC-DP: fresh-width certify at t={st.t} off the predeclared checkpoint grid (the K "
                               "ledger does not cover it; use FDCDPTimeUniform for denser monitoring)")
        self._last_t = st.t
        lo, hi = self._box(st)
        rect = self._rect_bounds(st, lo, hi) if self.hybrid else None
        return self.certify_with(ctx, st.mu_hat, st.n, st.N, lo, hi, getattr(st, "undecided", None), rect=rect,
                                 t=st.t)

    def certify_with(self, ctx, mu, n, N, lo, hi, undecided=None, rect=None, t=None, tables=None):
        import time
        t0 = time.perf_counter()
        beta = self.ledger["beta"]
        sp = ctx.sp
        cost, budgets, _ = reduce_costs(sp.cost, sp.budgets)
        if tables is None:
            lam = lambda_grid(beta, ctx.eps, ctx.w, n, N, lo, hi, self.grid_ratio)
            P, det = cell_tables(ctx.w, n, N, lo, hi, lam)
        else:
            lam, P, det = tables
        out = [None] * ctx.Q
        groups = {}
        vals = np.asarray(ctx.w, float)[:, None] * np.asarray(mu, float)       # weighted centre (B1)
        detail = [None] * ctx.Q
        for q in range(ctx.Q):
            _, ph = dp_argmax(vals, sp.cost, int(sp.budgets[q]))
            if ph is None:
                raise RuntimeError(f"problem {q}: empty policy class")
            if undecided is not None and not undecided[q]:
                out[q] = (False, tuple(int(x) for x in ph), float("nan"))
                continue
            groups.setdefault(tuple(int(x) for x in ph), []).append(q)
        st = {"t": t, "n_centres": len(groups), "n_cols": int(len(lam)) + 1, "nodes": 0, "bnb_calls": 0,
              "node_limit_hits": 0, "a_certified": 0, "b_certified": 0}
        for key, qs in groups.items():
            ph = np.asarray(key, dtype=np.int64)
            const, T, diff = joint_columns(ctx.w, mu, ph, P, det, lam, beta)
            if rect is not None:
                rlo, rhi = rect
                seg = np.arange(ctx.S)
                Tr = ctx.w[:, None] * (rhi - rlo[seg, ph][:, None])
                Tr[seg, ph] = 0.0
                T = np.concatenate([T, Tr[:, :, None]], axis=2)
                const = np.concatenate([const, [0.0]])
            caps = [int(budgets[q]) for q in qs]
            Ua = scheme_a_U(const, T, cost, diff, caps)
            for q, cap, ua in zip(qs, caps, Ua):
                U = max(0.0, ua)
                ok = U <= ctx.eps
                d = {"q": q, "U_a": float(U), "U_lower": None, "U_exact": None, "decision": None}
                if ok:
                    st["a_certified"] += 1
                    d["decision"] = "certified_a"
                elif self.scheme == "b":
                    r = bnb(const, T, cost, diff, cap, thr=ctx.eps, node_limit=self.node_limit)
                    st["nodes"] += r["nodes"]
                    st["bnb_calls"] += 1
                    if r["status"] == "node_limit":
                        st["node_limit_hits"] += 1
                        d["decision"] = "abstain_node_limit"
                    elif r["status"] == "certified":
                        ok = True
                        st["b_certified"] += 1
                        d["decision"] = "certified_b"
                    else:
                        d["decision"] = "violator_b"
                        d["U_lower"] = None if r["U"] is None else max(0.0, float(r["U"]))
                else:
                    d["decision"] = "not_certified_a"
                if self.exact_U:
                    rx = bnb(const, T, cost, diff, cap, thr=None, node_limit=self.node_limit)
                    d["U_exact"] = None if rx["status"] != "exact" else max(0.0, float(rx["U"]))
                detail[q] = d
                out[q] = (bool(ok), key, float(U))
        st["sec"] = round(time.perf_counter() - t0, 4)
        self.cert_stats.append(st)
        self.last_detail = detail
        return out

    def describe(self):
        d = super().describe()
        d.update({"scheme": self.scheme, "hybrid": self.hybrid, "node_limit": self.node_limit,
                  "ledger": self.ledger, "design": "frozen 50/50 (balanced_alloc 0.5)",
                  "guarantee": "FWER <= 0.05 at the K checkpoints (plan/fdc_dp_theory.md Theorem DP-1)"})
        return d


class FDCDPTimeUniform(FDCDP):
    """TU-FDC-DP: certify at any evaluation time with widths (counts, box, lambda grid, Psi tables, beta) frozen at the
    latest block point; Delta_hat from the current data (Lemma TU; plan s6).  Block grid must be a subset of the
    evaluation grid.  On the block grid it is identical to FDC-DP."""

    def __init__(self, block_points, **kw):
        kw.setdefault("name", None)
        super().__init__(**kw)
        self.block_points = np.asarray(block_points, dtype=np.int64)
        if kw.get("name") is None:
            self.name = f"TU-FDC-DP({self.scheme}){'+R' if self.hybrid else ''}"

    def _block_grid(self, ctx):
        return self.block_points

    def setup(self, ctx):
        super().setup(ctx)
        self._snap = None
        self._next_block = 0
        self.n_stale_evals = 0

    def certify(self, ctx, st):
        if st.t < self._last_t:
            raise RuntimeError("evaluation times must be increasing")
        self._last_t = st.t
        while self._next_block < len(self.block_points) and self.block_points[self._next_block] <= st.t:
            if int(self.block_points[self._next_block]) != st.t:
                raise RuntimeError("block grid must be a subset of the evaluation grid")
            lo, hi = self._box(st)
            rect = self._rect_bounds(st, lo, hi) if self.hybrid else None
            beta = self.ledger["beta"]
            lam = lambda_grid(beta, ctx.eps, ctx.w, st.n, st.N, lo, hi, self.grid_ratio)
            P, det = cell_tables(ctx.w, st.n, st.N, lo, hi, lam)
            self._snap = (np.asarray(st.n).copy(), np.asarray(st.N).copy(), lo, hi, rect, (lam, P, det))
            self._next_block += 1
        if self._snap is None:
            return [(False, None, float("inf")) for _ in range(ctx.Q)]
        n_k, N_k, lo, hi, rect, tables = self._snap
        if int(self.block_points[self._next_block - 1]) != st.t:
            self.n_stale_evals += 1
            if rect is not None:
                rect = self._rect_stale(st, rect)
        return self.certify_with(ctx, st.mu_hat, n_k, N_k, lo, hi, getattr(st, "undecided", None), rect=rect,
                                 t=st.t, tables=tables)

    def _rect_stale(self, st, rect):
        raise NotImplementedError("hybrid TU off the block grid is not implemented (not used in v10 dev)")
