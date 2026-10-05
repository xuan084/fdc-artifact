"""Round-4 real layer: finite-population, without-replacement pool replay of Hillstrom / Lenta RCT logs.

Ground truth lives here (finite-population cell means, exact optima). Learners only receive a ``ReplayHandle``: the
public arrival segment, the public pool sizes / remaining counts, and the outcome of each record they pull.

Protocol (methodology s1.1-1.2):
  * tidy.pkl is read-only. Pools are keyed by (segment, arm).
  * Dev/eval split: seed 4242, stratified 50/50 by (finest segmentation of the dataset, arm) -- HR uses S16 x arm (which
    refines S6 x arm), LR uses S8 x arm. In each stratum of size n the dev half gets floor(n/2) rows plus one more row
    with probability 1/2 when n is odd (drawn from the same seed). Dev half D is its own finite population.
  * A schedule is generated in full, before any outcome is seen, from a permutation seed and the *labels only*:
      - arrival segments: a uniformly random permutation of the half's segment-label multiset (so each arrival is
        marginally distributed by the table's segment shares w_s and segment totals match the table exactly);
      - QFC allocation within segment = pool proportions: a uniformly random permutation of that segment's arm-label
        multiset is assigned to its arrivals in order. This is reading the RCT log in random order; no pool can be
        exhausted before tau_R = number of rows of the half;
      - optional fixed allocation matrix p[s, a] (A-Ney / A-XY): iid draws from p[s, :], pre-drawn uniforms used for
        exhaustion re-selection (all outcome-free);
      - within-pool record order: independent uniform permutation per pool (child seed stream, also outcome-free).
  * Billing (all methods): every arrival is billed. If the chosen pool is exhausted the method re-chooses among the
    segment's non-exhausted arms by its own rule; if the whole segment is exhausted the arrival is skipped but billed.

r4 pre-lock revision (methodology "Pilot 后锁前修订" R1): layers CR9 (primary, real Criteo Uplift v2.1), CR12 (scale
sensitivity), HR8 (secondary Hillstrom layer). HR8 shares the HR split (S16 x arm). CR9 / CR12 are split stratified by
(CR12 x CR9 joint cell) x arm: CR12 does NOT refine CR9 (f3 is not a CR12 feature; 4 CR12 segments contain two CR9
segments), so the joint cell -- which refines both -- is used; this is the intended "(CR12 segment x arm), refining CR9"
stratification made well-defined. Existing HR6 / HR16 / LR8 behaviour is unchanged.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..streams import frontier as fr
from .base import TruthAccessError

DATA_ROOT = Path(__import__("os").environ.get("DATA_DIR", "data"))
DATA_PATHS = {"hillstrom": DATA_ROOT / "hillstrom" / "tidy.pkl", "lenta": DATA_ROOT / "lenta" / "tidy.pkl",
              "criteo": DATA_ROOT / "criteo_uplift_real" / "tidy.pkl"}
SPLIT_SEED = 4242
CLIP_Q = 0.999

LAYERS = {
    # name: dataset, arms, outcome map (public name -> column), n_min for checkpoints
    "HR6": dict(dataset="hillstrom", A=3, outcomes={"visit": "binary", "conversion": "conversion", "spend": "outcome"},
                n_min=1000, binary=("visit", "conversion")),
    "HR16": dict(dataset="hillstrom", A=3, outcomes={"visit": "binary", "conversion": "conversion", "spend": "outcome"},
                 n_min=1000, binary=("visit", "conversion")),
    "LR8": dict(dataset="lenta", A=2, outcomes={"response_att": "response_att", "response_amt": "response_amt"},
                n_min=5000, binary=("response_att",)),
    # ---- r4 pre-lock revision (R1)
    "HR8": dict(dataset="hillstrom", A=3, outcomes={"visit": "binary", "conversion": "conversion", "spend": "outcome"},
                n_min=1000, binary=("visit", "conversion"), eps_grid=fr.EPS_GRID),
    "CR9": dict(dataset="criteo", A=2, outcomes={"visit": "visit", "conversion": "conversion"},
                n_min=50_000, binary=("visit", "conversion"), eps_grid=fr.CR_EPS_GRID),
    "CR12": dict(dataset="criteo", A=2, outcomes={"visit": "visit", "conversion": "conversion"},
                 n_min=50_000, binary=("visit", "conversion"), eps_grid=fr.CR_EPS_GRID),
}
CLIPPED = {"spend", "response_amt"}


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


_TABLE_CACHE: dict = {}


def load_table(dataset):
    """Read-only load; returns a copy-protected frame (numpy arrays are set non-writeable downstream)."""
    if dataset not in _TABLE_CACHE:
        _TABLE_CACHE[dataset] = pd.read_pickle(DATA_PATHS[dataset])
    return _TABLE_CACHE[dataset]


_LABEL_CACHE: dict = {}


def _criteo_labels(layer, df):
    key = (layer, id(df))
    if key not in _LABEL_CACHE:
        modes = fr.criteo_modes(df)
        if layer == "CR9":
            lab, desc = fr.seg_cr9(df, modes)
        else:
            lab, desc, _ = fr.seg_cr12(df, modes)
        lab.setflags(write=False)
        _LABEL_CACHE[key] = (lab, desc)
    return _LABEL_CACHE[key]


def segment_labels(layer, df):
    """Segment labels on the FULL table (so dev/full halves share label ids) + human-readable descriptions."""
    if layer in ("CR9", "CR12"):
        return _criteo_labels(layer, df)
    if layer == "HR8":
        lab, pairs = fr.seg_hr8(df)
        cn = ("mens_only", "womens_only", "both")
        return lab, [f"hs3={h}|{cn[c]}" for h, c in pairs]
    if layer == "HR6":
        lab = fr.seg_hr6(df)
        desc = [f"hs3={s // 2}|newbie={s % 2}" for s in range(6)]
    elif layer == "HR16":
        lab, pairs = fr.seg_hr16(df)
        cn = ("mens_only", "womens_only", "both")
        desc = [f"hs3={p // 2}|newbie={p % 2}|{cn[c]}" for p, c in pairs]
    elif layer == "LR8":
        edges = fr.lenta_age_edges(df["x_age"].to_numpy())
        lab = fr.seg_lr8(df, edges)
        desc = [f"ageQ{s // 2}|fmt={s % 2}" for s in range(8)]
    else:
        raise KeyError(layer)
    return lab, desc


def split_mask(strata, arm, seed=SPLIT_SEED):
    """Stratified 50/50 dev mask; strata x arm cells processed in sorted order from one RNG stream."""
    rng = np.random.default_rng(seed)
    key = strata.astype(np.int64) * 16 + arm.astype(np.int64)
    dev = np.zeros(len(key), dtype=bool)
    for k in np.unique(key):
        idx = np.flatnonzero(key == k)
        perm = rng.permutation(len(idx))
        n_dev = len(idx) // 2 + (int(rng.integers(2)) if len(idx) % 2 else 0)
        dev[idx[perm[:n_dev]]] = True
    return dev


def split_strata(layer, df):
    if layer in ("HR6", "HR16", "HR8"):
        return segment_labels("HR16", df)[0]
    if layer in ("CR9", "CR12"):     # joint (CR12, CR9) cell refines both CR layers -> identical halves for both
        l12, l9 = segment_labels("CR12", df)[0], segment_labels("CR9", df)[0]
        return np.unique(l12 * 16 + l9, return_inverse=True)[1].astype(np.int64)
    return segment_labels("LR8", df)[0]


# =============================================================================================== environment (truth)
class PoolReplayEnv:
    """Finite population (one half of a real RCT table) partitioned into pools (segment, arm)."""

    def __init__(self, layer, half="dev", split_seed=SPLIT_SEED, df=None):
        spec = LAYERS[layer]
        self.layer, self.half, self.split_seed = layer, half, split_seed
        df = load_table(spec["dataset"]) if df is None else df
        self.A = spec["A"]
        seg_full, self.seg_desc = segment_labels(layer, df)
        self.S = len(self.seg_desc)
        arm_full = df["treatment"].to_numpy().astype(np.int64)
        if half == "full":
            mask = np.ones(len(df), dtype=bool)
        else:
            dev = split_mask(split_strata(layer, df), arm_full, split_seed)
            mask = dev if half == "dev" else ~dev
        self.row_index = np.flatnonzero(mask)                     # rows of the source table in this half
        self.seg = seg_full[mask]
        self.arm = arm_full[mask]
        # outcomes (clipped ones at the FULL-table P99.9, so dev and full share the clip constant)
        self._y = {}
        self.clip = {}
        for name, col in spec["outcomes"].items():
            full = df[col].to_numpy().astype(float)
            if name in CLIPPED:
                self.clip[name] = float(np.quantile(full, CLIP_Q))
                full = np.minimum(full, self.clip[name])
            v = full[mask].copy()
            v.setflags(write=False)
            self._y[name] = v
        self.binary_outcomes = spec["binary"]
        self.N = int(mask.sum())
        self.cell = self.seg * self.A + self.arm
        self.pool_sizes = np.bincount(self.cell, minlength=self.S * self.A).reshape(self.S, self.A)
        self.pool_rows = [np.flatnonzero(self.cell == c) for c in range(self.S * self.A)]   # half-local row ids
        self.seg_sizes = self.pool_sizes.sum(1)
        self.w = self.seg_sizes / self.N
        self.tau_R = self.N
        self.n_min = spec["n_min"]
        self.eps_grid = spec.get("eps_grid", fr.EPS_GRID)
        self.replan_interval = fr.replan_interval(self.tau_R)

    # ---------------------------------------------------------------- truth-only API
    def true_mu(self, outcome):
        """Finite-population cell means mu[s, a] (truth)."""
        s = np.bincount(self.cell, weights=self._y[outcome], minlength=self.S * self.A)
        return (s / self.pool_sizes.ravel()).reshape(self.S, self.A)

    def true_sigma2(self, outcome):
        y = self._y[outcome]
        m = self.true_mu(outcome).ravel()
        ss = np.bincount(self.cell, weights=(y - m[self.cell]) ** 2, minlength=self.S * self.A)
        return (ss / self.pool_sizes.ravel()).reshape(self.S, self.A)

    def true_pooled_mu(self, outcome):
        """Population mean of each arm ignoring segments (used only by the trivial-policy guard)."""
        y = self._y[outcome]
        return np.array([y[self.arm == a].mean() for a in range(self.A)])

    def outcomes_view(self, outcome):
        return self._y[outcome]

    def with_permuted_outcomes(self, seed):
        """Copy whose outcome vectors are shuffled across rows (labels untouched); used by the invariance check."""
        clone = object.__new__(PoolReplayEnv)
        clone.__dict__.update(self.__dict__)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(self.N)
        clone._y = {k: v[perm] for k, v in self._y.items()}
        return clone

    def public_info(self):
        return {"layer": self.layer, "half": self.half, "S": self.S, "A": self.A, "N": self.N,
                "pool_sizes": self.pool_sizes.tolist(), "w": self.w.tolist(), "tau_R": self.tau_R,
                "seg_desc": self.seg_desc, "replan_interval": self.replan_interval}

    def checkpoints(self, K=20):
        return fr.checkpoints(self.n_min, self.tau_R, K)

    # ---------------------------------------------------------------- truth answers
    def truth_answers(self, problems, eps_grid=None):
        eps_grid = self.eps_grid if eps_grid is None else eps_grid
        out = []
        for p in problems:
            mu = self.true_mu(p.outcome)
            if self.A ** self.S <= 10_000:
                ans = fr.solve_enum(self.w, mu, p, eps_grid=eps_grid)
                Jdp, pidp, _ = fr.solve_dp(self.seg_sizes, mu, p)
                ans.dp_agrees = bool(abs(Jdp - ans.J_star) <= 1e-12)
            else:
                Jdp, pidp, cdp = fr.solve_dp(self.seg_sizes, mu, p)
                ans = fr.solve_mitm(self.seg_sizes, mu, p, eps_grid=eps_grid)
                ans.dp_agrees = bool(abs(Jdp - ans.J_star) <= 1e-12)
            ans.dp_pi = pidp
            out.append(ans)
        return out

    def handle(self, schedule, outcome) -> "ReplayHandle":
        return ReplayHandle(ReplayStream(self, schedule, outcome))


# =============================================================================================== schedules
@dataclass
class Schedule:
    perm_seed: int
    seg_seq: np.ndarray          # (tau_R,) arrival segment
    arm_seq: np.ndarray          # (tau_R,) pre-planned arm, -1 if the method is adaptive
    resel_u: np.ndarray          # (tau_R,) uniforms for fixed-allocation exhaustion re-selection
    pool_perm: list              # per pool: permutation of pool records (truth side; outcome-free)
    alloc: str                   # 'pool' | 'fixed' | 'adaptive'
    alloc_p: np.ndarray = None   # (S, A) fixed allocation matrix (alloc == 'fixed')

    def digest(self):
        h = hashlib.sha256()
        for arr in (self.seg_seq, self.arm_seq, self.resel_u):
            h.update(np.ascontiguousarray(arr).tobytes())
        for p in self.pool_perm:
            h.update(np.ascontiguousarray(p).tobytes())
        return h.hexdigest()


def make_schedule(env: PoolReplayEnv, perm_seed: int, alloc=None, adaptive=False) -> Schedule:
    """Pre-generate a complete schedule from labels only (never reads env outcomes).

    alloc=None & adaptive=False -> pool-proportional (QFC);  alloc = (S, A) matrix -> fixed iid allocation;
    adaptive=True -> arrivals + pool orders only (arm chosen online by the method)."""
    ss = np.random.SeedSequence([int(perm_seed), 0x52_34])        # 'R4'
    r_arr, r_alloc, r_pool = (np.random.default_rng(c) for c in ss.spawn(3))
    seg, arm, S, A = env.seg, env.arm, env.S, env.A
    seg_seq = r_arr.permutation(seg).astype(np.int16)
    arm_seq = np.full(env.N, -1, dtype=np.int8)
    resel_u = r_alloc.random(env.N)
    if adaptive:
        kind = "adaptive"
    elif alloc is None:
        kind = "pool"
        for s in range(S):
            pos = np.flatnonzero(seg_seq == s)
            arm_seq[pos] = r_alloc.permutation(arm[seg == s]).astype(np.int8)
    else:
        kind = "fixed"
        P = np.asarray(alloc, dtype=float)
        P = P / P.sum(1, keepdims=True)
        u = r_alloc.random(env.N)
        cdf = np.cumsum(P, 1)
        arm_seq[:] = (u[:, None] > cdf[seg_seq.astype(np.int64)]).sum(1).clip(0, A - 1)
    pool_perm = [r_pool.permutation(int(n)).astype(np.int32) for n in env.pool_sizes.ravel()]
    return Schedule(int(perm_seed), seg_seq, arm_seq, resel_u, pool_perm, kind,
                    None if kind != "fixed" else P)


# =============================================================================================== fast fixed replay
def replay_fixed(env: PoolReplayEnv, sch: Schedule, outcome, checkpoints):
    """Vectorised replay of a pre-planned (non-adaptive) schedule assuming no exhaustion (asserted).

    Returns n[k, c], sum[k, c], sumsq[k, c] at each checkpoint (cells c = s*A + a)."""
    if sch.alloc != "pool":
        raise ValueError("replay_fixed only handles pool-proportional schedules; use ReplayStream for others")
    cell = sch.seg_seq.astype(np.int64) * env.A + sch.arm_seq.astype(np.int64)
    rank = occurrence_rank(cell)
    sizes = env.pool_sizes.ravel()
    if np.any(rank >= sizes[cell]):
        raise RuntimeError("pool exhausted before tau_R under pool-proportional allocation")
    y = env.outcomes_view(outcome)
    rec = np.empty(env.N, dtype=np.int64)
    for c in range(len(sizes)):
        m = cell == c
        rec[m] = env.pool_rows[c][sch.pool_perm[c][rank[m]]]
    obs = y[rec]
    C = env.S * env.A
    K = len(checkpoints)
    n = np.zeros((K, C), dtype=np.int64)
    s1 = np.zeros((K, C))
    s2 = np.zeros((K, C))
    for k, t in enumerate(checkpoints):
        n[k] = np.bincount(cell[:t], minlength=C)
        s1[k] = np.bincount(cell[:t], weights=obs[:t], minlength=C)
        s2[k] = np.bincount(cell[:t], weights=obs[:t] ** 2, minlength=C)
    return n, s1, s2


def occurrence_rank(cell):
    """rank[t] = number of earlier arrivals with the same cell."""
    order = np.argsort(cell, kind="stable")
    sc = cell[order]
    start = np.r_[0, np.flatnonzero(np.diff(sc)) + 1]
    grp_start = np.repeat(start, np.diff(np.r_[start, len(sc)]))
    rank = np.empty(len(cell), dtype=np.int64)
    rank[order] = np.arange(len(sc)) - grp_start
    return rank


# =============================================================================================== generic stream
class ReplayStream:
    """Arrival-by-arrival replay with the common exhaustion / billing rules (truth side)."""

    def __init__(self, env: PoolReplayEnv, sch: Schedule, outcome):
        self._env, self._sch = env, sch
        self._y = env.outcomes_view(outcome) if isinstance(outcome, str) else None
        self._outcomes = [outcome] if isinstance(outcome, str) else list(outcome)
        self.S, self.A, self.tau_R = env.S, env.A, env.tau_R
        self.pool_sizes = env.pool_sizes.copy()
        self.w = env.w.copy()
        self.remaining = env.pool_sizes.copy()
        self.t = 0
        self.billed = 0
        self.n_served = 0
        self.n_skipped = 0
        self.n_reselected = 0

    def current_segment(self):
        return int(self._sch.seg_seq[self.t])

    def planned_arm(self):
        a = int(self._sch.arm_seq[self.t])
        return None if a < 0 else a

    def available_arms(self, s=None):
        s = self.current_segment() if s is None else s
        return [a for a in range(self.A) if self.remaining[s, a] > 0]

    def arrive(self, choose=None):
        """Consume one arrival. `choose(s, avail) -> arm` is called only when the method must pick (adaptive), or to
        re-select after its pre-planned arm is exhausted (fixed allocations use the pre-drawn uniform instead).
        Returns (s, a, y) with a = y = None for a skipped arrival."""
        if self.t >= self.tau_R:
            raise StopIteration("tau_R reached")
        s = self.current_segment()
        u = float(self._sch.resel_u[self.t])
        planned = self.planned_arm()
        self.t += 1
        self.billed += 1
        avail = [a for a in range(self.A) if self.remaining[s, a] > 0]
        if not avail:
            self.n_skipped += 1
            return s, None, None
        if planned is not None:
            a = planned
            if self.remaining[s, a] <= 0:
                self.n_reselected += 1
                if choose is not None:
                    a = int(choose(s, avail))
                else:   # fixed allocation: its own rule = p[s, :] renormalised over the non-exhausted arms
                    p = (self._sch.alloc_p[s, avail] if self._sch.alloc_p is not None else np.ones(len(avail)))
                    p = p / p.sum() if p.sum() > 0 else np.full(len(avail), 1.0 / len(avail))
                    a = avail[min(int((u > np.cumsum(p)).sum()), len(avail) - 1)]
                if a not in avail:
                    raise ValueError(f"re-selected exhausted arm {a} in segment {s}")
        else:
            a = int(choose(s, avail))
            if a not in avail:
                raise ValueError(f"method chose exhausted / invalid arm {a} in segment {s}; available {avail}")
        c = s * self.A + a
        k = int(self.pool_sizes[s, a] - self.remaining[s, a])
        row = self._env.pool_rows[c][self._sch.pool_perm[c][k]]
        self.remaining[s, a] -= 1
        self.n_served += 1
        if self._y is not None:
            return s, a, float(self._y[row])
        return s, a, tuple(float(self._env.outcomes_view(o)[row]) for o in self._outcomes)

    def billing_ok(self):
        return (self.billed == self.t and self.n_served + self.n_skipped == self.billed and
                int((self.pool_sizes - self.remaining).sum()) == self.n_served and bool((self.remaining >= 0).all()))


class ReplayHandle:
    """Learner-facing proxy: arrivals, public pool sizes / remaining counts, billing counters. No truth."""
    _PUBLIC = ("arrive", "current_segment", "planned_arm", "available_arms", "S", "A", "tau_R", "pool_sizes", "w",
               "remaining", "t", "billed", "n_served", "n_skipped", "n_reselected", "billing_ok")

    def __init__(self, stream: ReplayStream):
        object.__setattr__(self, "_ReplayHandle__stream", stream)

    def __getattribute__(self, name):
        if name in ReplayHandle._PUBLIC:
            v = getattr(object.__getattribute__(self, "_ReplayHandle__stream"), name)
            return v.copy() if isinstance(v, np.ndarray) else v
        if name.startswith("__") and name.endswith("__"):
            return object.__getattribute__(self, name)
        raise TruthAccessError(f"learner may not access replay attribute '{name}'")

    def __setattr__(self, name, value):
        raise TruthAccessError("replay handle is read-only")
