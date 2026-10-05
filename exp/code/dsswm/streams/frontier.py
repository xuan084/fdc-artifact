"""Round-4 real-layer problem definitions (truth-free).

Everything here is a pure function of *public* objects: covariate-based segment labels, segment weights w_s,
cost tiers, budgets and (when an exact answer is requested) a cell-mean matrix ``mu`` passed in by the caller.
Ground-truth means live only in ``dsswm.envs.pool_replay``; this module never imports it.

Segmentations (methodology s1.1):
  * HR S=6  : hs3 x newbie, hs3 = x_history_segment in {0} / {1,2} / {3,4,5,6}  (0-based coding, external reviewer dev definition)
  * HR S=16 : S6 x purchase-category history (mens-only / womens-only / both); hs3=0 has no 'both' rows -> 16 non-empty
  * LR S=8  : age quartile (full-table quantiles, right-inclusive bins) x main_format
Problem sets:
  * HR : outcome x cost tier {C1 (0,1,1), C2 (0,1.5,1), C3 (0,2,1)} x budget {0.2,0.4,0.6,0.8,1.0} -> 15 problems
  * LR : response_att x treatment cost {0.5,1,2} x budget {0.25,0.5,0.75} -> 9 problems
Feasibility: sum_s w_s kappa_{pi(s)} <= B (deterministic per-segment policies).

r4 pre-lock revision (methodology "Pilot 后锁前修订" R1/R4) -- real Criteo Uplift v2.1 layers + Hillstrom HR8:
  * CR9  : "non-mode" indicators of f0, f2, f3, f9 (mode on the FULL table; f6 == f0 and f8 == f2 after binarisation,
           so they are dropped) -> 9 non-empty segments, A=2 (0=control, 1=treatment), |Pi| = 512
  * CR12 : f0, f2 in three levels (mode / non-mode <= median of non-mode values / above) x f9 non-mode -> 12 non-empty
           segments, |Pi| = 4096 (scale-sensitivity / DP comparison layer only)
  * HR8  : hs3 x purchase-category history (mens-only / womens-only / both); hs3=0 has no 'both' -> 8 segments, A=3
  * CR problems: visit x treatment cost 1 x budget {0.10, 0.15, ..., 0.80} -> 15 problems (N80 = 12/15)
Segment codes are built lexicographically (code = code*3 + level per feature) and re-indexed over the non-empty codes,
exactly as in plan/feasibility_r4/criteo_diag3.py.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

HR_ARMS = ("none", "mens", "womens")
LR_ARMS = ("control", "sms")
HR_COST_TIERS = {"C1": (0.0, 1.0, 1.0), "C2": (0.0, 1.5, 1.0), "C3": (0.0, 2.0, 1.0)}
HR_BUDGETS = (0.2, 0.4, 0.6, 0.8, 1.0)
LR_TREAT_COSTS = (0.5, 1.0, 2.0)
LR_BUDGETS = (0.25, 0.5, 0.75)
EPS_GRID = (0.0025, 0.005, 0.0075, 0.01)
FEAS_TOL = 1e-12
CR_ARMS = ("control", "treatment")
CR_BUDGETS = tuple(round(0.10 + 0.05 * i, 2) for i in range(15))          # 0.10, 0.15, ..., 0.80
CR_EPS_GRID = (0.001, 0.0015, 0.002, 0.003)                               # R1 eps grid (J units)
CR_EPS_DIAG = (0.0005, 0.00075, 0.001, 0.0015, 0.002, 0.003, 0.004)        # planner diagnostic grid (reporting only)
CR9_FEATURES = ("f0", "f2", "f3", "f9")
CR_DUPLICATES = {"f6": "f0", "f8": "f2"}                                  # identical non-mode indicators (dropped)


# --------------------------------------------------------------------------------------------- segmentations
def hs3(history_segment):
    h = np.asarray(history_segment).astype(np.int64)
    if h.min() < 0 or h.max() > 6:
        raise ValueError("x_history_segment must be 0-based in [0, 6]")
    return np.where(h == 0, 0, np.where(h <= 2, 1, 2)).astype(np.int64)


def seg_hr6(df):
    """S=6: hs3 x newbie -> label 2*hs3 + newbie."""
    return (2 * hs3(df["x_history_segment"].to_numpy()) + df["x_newbie"].to_numpy().astype(np.int64)).astype(np.int64)


HR16_CAT = {(1, 0): 0, (0, 1): 1, (1, 1): 2}   # (x_mens, x_womens) -> mens-only / womens-only / both


def seg_hr16(df):
    """S=16: (S6 label, category) pairs, re-indexed to 0..15 in lexicographic order of the non-empty pairs."""
    s6 = seg_hr6(df)
    m, w = df["x_mens"].to_numpy().astype(np.int64), df["x_womens"].to_numpy().astype(np.int64)
    if np.any((m == 0) & (w == 0)):
        raise ValueError("rows with neither mens nor womens history are not covered by the S=16 definition")
    cat = np.where((m == 1) & (w == 0), 0, np.where((m == 0) & (w == 1), 1, 2)).astype(np.int64)
    raw = 3 * s6 + cat
    uniq = np.unique(raw)
    return np.searchsorted(uniq, raw).astype(np.int64), [(int(r) // 3, int(r) % 3) for r in uniq]


def lenta_age_edges(age_full):
    """Quartile edges on the FULL table (methodology: '全表分位点'), so dev and full halves share bins."""
    return tuple(float(q) for q in np.quantile(np.asarray(age_full, dtype=float), [0.25, 0.5, 0.75]))


def seg_lr8(df, edges):
    """S=8: age quartile (bins (-inf,q1],(q1,q2],(q2,q3],(q3,inf)) x main_format -> 2*quartile + format."""
    age = df["x_age"].to_numpy().astype(float)
    q = np.searchsorted(np.asarray(edges), age, side="left").astype(np.int64)   # age<=q1 -> 0, ..., age>q3 -> 3
    fmt = df["x_main_format"].to_numpy().astype(np.int64)
    if fmt.min() < 0 or fmt.max() > 1:
        raise ValueError("x_main_format must be binary")
    return (2 * q + fmt).astype(np.int64)


def seg_hr8(df):
    """HR8: hs3 x category (mens-only / womens-only / both), re-indexed to 0..7 over the non-empty pairs (hs3=0 has no
    'both'). Returns (labels, [(hs3, cat), ...])."""
    h = hs3(df["x_history_segment"].to_numpy())
    m, w = df["x_mens"].to_numpy().astype(np.int64), df["x_womens"].to_numpy().astype(np.int64)
    if np.any((m == 0) & (w == 0)):
        raise ValueError("rows with neither mens nor womens history are not covered by the HR8 definition")
    cat = np.where((m == 1) & (w == 0), 0, np.where((m == 0) & (w == 1), 1, 2)).astype(np.int64)
    raw = 3 * h + cat
    uniq = np.unique(raw)
    return np.searchsorted(uniq, raw).astype(np.int64), [(int(r) // 3, int(r) % 3) for r in uniq]


def column_mode(v):
    """Most frequent value (ties -> smallest value); deterministic, covariate-only."""
    u, c = np.unique(np.asarray(v), return_counts=True)
    return u[int(np.argmax(c))]


def criteo_modes(df, features=("f0", "f2", "f3", "f9", "f6", "f8")):
    """Per-feature modes on the FULL table (so every half shares the same segment definition)."""
    return {f: column_mode(df[f].to_numpy()) for f in features}


def _nonmode(df, f, modes):
    return (df[f].to_numpy() != modes[f]).astype(np.int64)


def _three_level(df, f, modes):
    """0 = mode, 1 = non-mode <= median of the non-mode values (full table), 2 = above. Returns (codes, median)."""
    v = df[f].to_numpy()
    nm = v != modes[f]
    med = np.median(v[nm])
    return np.where(~nm, 0, np.where(v <= med, 1, 2)).astype(np.int64), float(med)


def _reindex(raw):
    uniq = np.unique(raw)
    return np.searchsorted(uniq, raw).astype(np.int64), [int(r) for r in uniq]


def _digits(code, n, base=3):
    out = []
    for _ in range(n):
        out.append(code % base)
        code //= base
    return out[::-1]


def seg_cr9(df, modes=None):
    """CR9: non-mode indicators of f0, f2, f3, f9 -> labels 0..8 (lexicographic over non-empty codes).
    Returns (labels, descriptions)."""
    modes = criteo_modes(df, CR9_FEATURES) if modes is None else modes
    raw = np.zeros(len(df), dtype=np.int64)
    for f in CR9_FEATURES:
        raw = raw * 3 + _nonmode(df, f, modes)
    lab, uniq = _reindex(raw)
    desc = ["|".join(f"{f}nm={d}" for f, d in zip(CR9_FEATURES, _digits(r, 4))) for r in uniq]
    return lab, desc


def seg_cr12(df, modes=None):
    """CR12: f0 (3 levels) x f2 (3 levels) x f9 non-mode -> labels 0..11. Returns (labels, descriptions, medians)."""
    modes = criteo_modes(df, ("f0", "f2", "f9")) if modes is None else modes
    t0, m0 = _three_level(df, "f0", modes)
    t2, m2 = _three_level(df, "f2", modes)
    raw = (t0 * 3 + t2) * 3 + _nonmode(df, "f9", modes)
    lab, uniq = _reindex(raw)
    desc = ["|".join(f"{f}={d}" for f, d in zip(("f0q", "f2q", "f9nm"), _digits(r, 3))) for r in uniq]
    return lab, desc, {"f0": m0, "f2": m2}


def replan_interval(tau_R):
    """Batch re-planning interval of the adaptive baselines (lock constant): max(200, ceil(tau_R / 2000)) arrivals."""
    return max(200, int(math.ceil(tau_R / 2000)))


# --------------------------------------------------------------------------------------------- problems
@dataclass(frozen=True)
class Problem:
    qid: str
    outcome: str
    kappa: tuple            # per-arm unit cost
    budget: float


def hr_problems(outcome="visit"):
    return [Problem(f"{outcome}|{t}|B{b:.1f}", outcome, HR_COST_TIERS[t], b) for t in HR_COST_TIERS for b in HR_BUDGETS]


def lr_problems(outcome="response_att"):
    return [Problem(f"{outcome}|c{c:g}|B{b:.2f}", outcome, (0.0, c), b) for c in LR_TREAT_COSTS for b in LR_BUDGETS]


def cr_problems(outcome="visit"):
    """CR: treatment cost 1, control cost 0, budget B = max treated population share; 15 problems."""
    return [Problem(f"{outcome}|c1|B{b:.2f}", outcome, (0.0, 1.0), b) for b in CR_BUDGETS]


# --------------------------------------------------------------------------------------------- policy class
def enumerate_policies(S, A):
    """All deterministic per-segment policies, shape (A**S, S); row i is the base-A digits of i (seg 0 most significant)."""
    if A ** S > 2_000_000:
        raise ValueError(f"|Pi| = {A}^{S} too large for explicit enumeration; use the DP / meet-in-middle solvers")
    return np.array(list(itertools.product(range(A), repeat=S)), dtype=np.int8)


def policy_costs(pols, w, kappa):
    return (np.asarray(w)[None, :] * np.asarray(kappa)[pols]).sum(1)


def policy_values(pols, w, mu):
    """J(pi) = sum_s w_s mu[s, pi(s)] for every row of pols."""
    S = pols.shape[1]
    return (np.asarray(w)[None, :] * np.asarray(mu)[np.arange(S)[None, :], pols]).sum(1)


@dataclass
class ExactAnswer:
    qid: str
    J_star: float
    pi_star: tuple
    cost_star: float
    n_feasible: int
    eps_set_size: dict = field(default_factory=dict)     # eps -> |{pi feasible: J(pi) >= J* - eps}|
    second_gap: float = float("nan")                     # J* - best feasible value among pi != pi* (tie diagnostic)


def solve_enum(w, mu, problem, pols=None, eps_grid=EPS_GRID):
    """Exact optimum and eps-correct answer sets by explicit enumeration (S=6: 729 policies; LR S=8: 256)."""
    S, A = np.asarray(mu).shape
    pols = enumerate_policies(S, A) if pols is None else pols
    c = policy_costs(pols, w, problem.kappa)
    v = policy_values(pols, w, mu)
    feas = c <= problem.budget + FEAS_TOL
    vf = np.where(feas, v, -np.inf)
    i = int(np.argmax(vf))
    J = float(vf[i])
    rest = np.delete(vf, i)
    return ExactAnswer(problem.qid, J, tuple(int(x) for x in pols[i]), float(c[i]), int(feas.sum()),
                       {e: int((vf >= J - e).sum()) for e in eps_grid}, float(J - rest.max()) if rest.size else np.inf)


def _int_costs(n_seg, kappa):
    """Exact integer costs: w_s kappa_a = n_s kappa_a / N; with kappa in half-integers, 2 n_s kappa_a is an integer."""
    k2 = np.asarray(kappa, dtype=float) * 2.0
    if not np.allclose(k2, np.round(k2)):
        raise ValueError("exact DP requires half-integer costs")
    return np.asarray(n_seg, dtype=np.int64)[:, None] * np.round(k2).astype(np.int64)[None, :]


def solve_dp(n_seg, mu, problem):
    """Exact multiple-choice knapsack DP on integer costs (units 1/(2N)); no rounding, no relaxation.

    n_seg: per-segment row counts of the finite population (w_s = n_s / N). Returns (J*, pi*, cost*)."""
    n_seg = np.asarray(n_seg, dtype=np.int64)
    N = int(n_seg.sum())
    w = n_seg / N
    mu = np.asarray(mu, dtype=float)
    S, A = mu.shape
    ic = _int_costs(n_seg, problem.kappa)
    cap = int(np.floor(problem.budget * 2 * N + 1e-9))
    NEG = -np.inf
    f = np.full(cap + 1, NEG)
    f[0] = 0.0
    choice = np.zeros((S, cap + 1), dtype=np.int8)
    for s in range(S):
        g = np.full(cap + 1, NEG)
        for a in range(A):
            cst = int(ic[s, a])
            if cst > cap:
                continue
            cand = np.full(cap + 1, NEG)
            cand[cst:] = f[:cap + 1 - cst] + w[s] * mu[s, a]
            better = cand > g
            g = np.where(better, cand, g)
            choice[s] = np.where(better, a, choice[s])
        f = g
    b = int(np.argmax(f))
    J = float(f[b])
    pi = [0] * S
    for s in range(S - 1, -1, -1):
        a = int(choice[s, b])
        pi[s] = a
        b -= int(ic[s, a])
    pi = tuple(pi)
    cost = float((w * np.asarray(problem.kappa)[list(pi)]).sum())
    return J, pi, cost


def solve_mitm(n_seg, mu, problem, eps_grid=EPS_GRID, block=512):
    """Exact enumeration of all A**S policies by meet-in-the-middle (used for S=16 to cross-check the DP and to count
    eps-correct answer sets). Costs are compared in exact integers."""
    n_seg = np.asarray(n_seg, dtype=np.int64)
    N = int(n_seg.sum())
    w = n_seg / N
    mu = np.asarray(mu, dtype=float)
    S, A = mu.shape
    h = S // 2
    ic = _int_costs(n_seg, problem.kappa)
    cap = int(np.floor(problem.budget * 2 * N + 1e-9))
    PL, PR = enumerate_policies(h, A), enumerate_policies(S - h, A)
    cL = ic[np.arange(h)[None, :], PL].sum(1)
    cR = ic[np.arange(h, S)[None, :], PR].sum(1)
    vL = (w[None, :h] * mu[np.arange(h)[None, :], PL]).sum(1)
    vR = (w[None, h:] * mu[np.arange(h, S)[None, :], PR]).sum(1)
    best, arg, nfeas = -np.inf, None, 0
    for i0 in range(0, len(PL), block):
        cc = cL[i0:i0 + block, None] + cR[None, :]
        vv = np.where(cc <= cap, vL[i0:i0 + block, None] + vR[None, :], -np.inf)
        nfeas += int(np.isfinite(vv).sum())
        k = int(np.argmax(vv))
        if vv.flat[k] > best:
            best, arg = float(vv.flat[k]), (i0 + k // vv.shape[1], k % vv.shape[1])
    counts = {e: 0 for e in eps_grid}
    second = -np.inf
    for i0 in range(0, len(PL), block):
        cc = cL[i0:i0 + block, None] + cR[None, :]
        vv = np.where(cc <= cap, vL[i0:i0 + block, None] + vR[None, :], -np.inf)
        for e in eps_grid:
            counts[e] += int((vv >= best - e).sum())
        if arg[0] >= i0 and arg[0] < i0 + block:
            vv = vv.copy()
            vv[arg[0] - i0, arg[1]] = -np.inf
        second = max(second, float(vv.max()))
    pi = tuple(int(x) for x in np.concatenate([PL[arg[0]], PR[arg[1]]]))
    cost = float((w * np.asarray(problem.kappa)[list(pi)]).sum())
    return ExactAnswer(problem.qid, best, pi, cost, nfeas, counts, best - second)


# --------------------------------------------------------------------------------------------- trivial policies
def _fill_in_order(w, kappa, budget, arm_order, seg_order):
    """Assign arms segment by segment (seg_order); each segment gets the first arm in arm_order that keeps the policy
    feasible (arm 0 / no-treatment has cost 0 in every tier, so the result is always feasible)."""
    S = len(w)
    pi = np.zeros(S, dtype=np.int64)
    spent = 0.0
    for s in seg_order:
        for a in arm_order:
            c = w[s] * kappa[a]
            if spent + c <= budget + FEAS_TOL:
                pi[s] = a
                spent += c
                break
    return tuple(int(x) for x in pi)


def trivial_pooled_greedy(w, mu_pooled, problem):
    """'不分段平均贪心': segment-blind. Arms ranked by pooled uplift per unit cost (mu_a - mu_0)/kappa_a over the whole
    population (ties / zero-cost arms first by raw uplift); segments filled in public order (descending w_s)."""
    kappa = np.asarray(problem.kappa)
    up = np.asarray(mu_pooled) - mu_pooled[0]
    ratio = np.array([np.inf if kappa[a] == 0 and up[a] > 0 else (up[a] / kappa[a] if kappa[a] > 0 else -np.inf)
                      for a in range(len(kappa))])
    arm_order = [int(a) for a in np.argsort(-ratio, kind="stable") if a != 0 and up[a] > 0] + [0]
    seg_order = list(np.argsort(-np.asarray(w), kind="stable"))
    return _fill_in_order(np.asarray(w), kappa, problem.budget, arm_order, seg_order)


def trivial_all_arm(w, problem, arm):
    """'Mens 全发' (HR arm=1) / 'treat everyone' (LR arm=1): give `arm` to segments in descending-w_s order while the
    budget allows, no-treatment elsewhere."""
    seg_order = list(np.argsort(-np.asarray(w), kind="stable"))
    return _fill_in_order(np.asarray(w), np.asarray(problem.kappa), problem.budget, [arm, 0], seg_order)


# --------------------------------------------------------------------------------------------- checkpoints
def checkpoints(n_min, tau_R, K=20):
    """K pre-fixed log-spaced arrival counts from n_min to tau_R (inclusive, strictly increasing integers)."""
    g = np.unique(np.round(np.exp(np.linspace(np.log(n_min), np.log(tau_R), K))).astype(np.int64))
    if len(g) != K:
        raise ValueError("log grid collapsed; widen [n_min, tau_R]")
    g[-1] = tau_R
    return g
