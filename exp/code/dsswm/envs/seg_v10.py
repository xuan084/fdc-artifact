"""v10 uplift-score segmentations with S in {16, 32, 64} on the Lenta and X5 DEV halves (NEW file; locked env modules
are imported, never modified).  Truth side: builds the finite population and runs streams for implicit policy classes.

Data discipline (plan/fdc_dp_theory.md s9).
* DEV HALVES ONLY.  Lenta: the LR9 dev half (lenta_v6: covariate-only LR9 strata, split seed 6006).  X5: dev.pkl (the
  blinded hashed split of ingest_v8_fresh).  There is no 'eval' option in this module: any half other than 'dev' raises.
  Lenta: pandas reads the whole pickle (it cannot read columns selectively); eval-half outcomes are dropped right after
  loading and never used, exactly as lenta_v6 documents.
* MODEL SPLIT: inside the dev half, a stratified (by arm) 30 % "model split" (seed MODEL_SPLIT_SEED = 10010) is used
  ONLY to fit the score model; the remaining 70 % of the dev half is the replay population ("replay pool").  The score
  model never sees replay-pool outcomes.
* Score: T-learner, two sklearn HistGradientBoostingClassifier (one per arm, fixed hyper-parameters, random_state 0) on
  the model split; score = p1(x) - p0(x) on the replay pool (covariates only).
* Segments: rank of the score on the replay pool (ties broken by a seeded random permutation, labels only), cut into S
  groups of equal size (+-1 row): segment 0 = lowest predicted uplift.
* Problems ("SR" family, the CR family transposed): arm 0 = control (cost 0), arm 1 = treatment; the cost of treating
  segment s is kappa[s, 1] = round(G S w_s) cost units with G = 8 (8 units = one average segment), integer budgets
  B_q = floor(b_q * sum_s kappa[s, 1]) for b_q = 0.10, 0.15, ..., 0.80 (15 problems).  With equal-size segments every
  kappa[s, 1] = 8, i.e. "treat at most floor(b_q S) segments" (a cardinality-constrained class); the DP code is general.
* checkpoints: K = 20 log-spaced from n_min (Lenta 5,000, X5 2,000, as LR9 / X9) to tau_R = replay-pool size.
"""
from __future__ import annotations

import hashlib
import json
import math
import time

import numpy as np
import pandas as pd

from ..streams import frontier as fr
from ..streams.frontier_runner import StreamEngine, SyntheticPoolEnv
from .pool_replay import DATA_PATHS, make_schedule, sha256_file, split_mask

__all__ = ["SegEnvV10", "seg_problems", "fit_scores", "MODEL_SPLIT_SEED", "MODEL_FRAC", "COST_G", "SR_BUDGETS",
           "run_stream_seg", "true_opt", "policy_value"]

MODEL_SPLIT_SEED = 10010
MODEL_FRAC = 0.30
TIE_SEED = 10011
COST_G = 8
SR_BUDGETS = fr.CR_BUDGETS
X5_ROOT = DATA_PATHS["lenta"].parents[1] / "x5_retailhero"
LENTA_FEATURES = ("x_age", "x_children", "x_main_format", "x_months_from_register", "x_promo_share_15d",
                  "x_food_share_1m", "x_k_var_cheque_3m", "x_mean_discount_depth_15d")
_SCORE_CACHE: dict = {}


# =============================================================================================== data (dev half only)
def _lenta_dev():
    from .lenta_v6 import LENTA_SHA256, LR9_OUTCOME, LR9_SPLIT_SEED, lr9_labels
    if sha256_file(DATA_PATHS["lenta"]) != LENTA_SHA256:
        raise RuntimeError("lenta tidy.pkl sha256 differs from the pinned value")
    df = pd.read_pickle(DATA_PATHS["lenta"])
    seg9, _, _ = lr9_labels(df)
    arm = df["treatment"].to_numpy().astype(np.int64)
    dev = split_mask(seg9, arm, LR9_SPLIT_SEED)
    X = df.loc[dev, list(LENTA_FEATURES)].to_numpy(dtype=float)
    y = df.loc[dev, LR9_OUTCOME].to_numpy(dtype=float)
    a = arm[dev]
    rows = np.flatnonzero(dev)
    del df
    return X, a, y, rows, list(LENTA_FEATURES)


def _x5_dev():
    df = pd.read_pickle(X5_ROOT / "dev.pkl")
    age = df["age"].to_numpy(dtype=float)
    valid = (age >= 14) & (age <= 100)
    g = df["gender"].astype(str)
    t0 = pd.Timestamp("2017-01-01")
    iss = (pd.to_datetime(df["first_issue_date"], errors="coerce") - t0).dt.days.to_numpy(dtype=float)
    red = (pd.to_datetime(df["first_redeem_date"], errors="coerce") - t0).dt.days.to_numpy(dtype=float)
    X = np.column_stack([np.where(valid, age, np.nan), (~valid).astype(float), (g == "F").astype(float),
                         (g == "M").astype(float), iss, red, red - iss, np.isnan(red).astype(float)])
    names = ["age_valid", "age_invalid", "g_F", "g_M", "issue_day", "redeem_day", "redeem_minus_issue", "no_redeem"]
    return X, df["treatment"].to_numpy().astype(np.int64), df["outcome"].to_numpy().astype(float), \
        np.arange(len(df)), names


def model_split(arm, seed=MODEL_SPLIT_SEED, frac=MODEL_FRAC):
    rng = np.random.default_rng(seed)
    m = np.zeros(len(arm), dtype=bool)
    for a in np.unique(arm):
        idx = np.flatnonzero(arm == a)
        m[idx[rng.permutation(len(idx))[:int(round(frac * len(idx)))]]] = True
    return m


def fit_scores(data):
    """T-learner uplift scores on the replay pool; returns dict (cached per data set)."""
    if data in _SCORE_CACHE:
        return _SCORE_CACHE[data]
    from sklearn.ensemble import HistGradientBoostingClassifier
    t0 = time.perf_counter()
    X, arm, y, rows, names = _lenta_dev() if data == "lenta" else _x5_dev()
    msk = model_split(arm)
    models = []
    for a in (0, 1):
        sel = msk & (arm == a)
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                             min_samples_leaf=200, l2_regularization=1.0, random_state=0)
        clf.fit(X[sel], y[sel].astype(int))
        models.append(clf)
    rp = ~msk
    Xr = X[rp]
    score = models[1].predict_proba(Xr)[:, 1] - models[0].predict_proba(Xr)[:, 1]
    out = {"data": data, "features": names, "n_dev": int(len(arm)), "n_model": int(msk.sum()),
           "n_replay": int(rp.sum()), "score": score, "arm": arm[rp], "y": y[rp], "source_rows": rows[rp],
           "model_rows_sha": hashlib.sha256(np.ascontiguousarray(rows[msk]).tobytes()).hexdigest()[:16],
           "fit_sec": round(time.perf_counter() - t0, 1)}
    _SCORE_CACHE[data] = out
    return out


def seg_labels(score, S, seed=TIE_SEED):
    """Equal-size score-quantile segments (0 = lowest score); ties broken by a seeded permutation (labels only)."""
    n = len(score)
    tb = np.random.default_rng(seed).permutation(n)
    order = np.lexsort((tb, score))
    lab = np.empty(n, dtype=np.int64)
    for s, chunk in enumerate(np.array_split(order, S)):
        lab[chunk] = s
    return lab


def seg_problems(w, budgets=SR_BUDGETS, G=COST_G):
    from ..baselines.fdc_dp import SegProblems
    S = len(w)
    k1 = np.rint(G * S * np.asarray(w, float)).astype(np.int64)
    cost = np.stack([np.zeros(S, dtype=np.int64), k1], 1)
    tot = int(k1.sum())
    B = np.array([int(math.floor(b * tot + 1e-9)) for b in budgets], dtype=np.int64)
    return SegProblems(cost=cost, budgets=B, qids=[f"SR|c1|B{b:.2f}" for b in budgets], budget_frac=list(budgets))


# =============================================================================================== environment
class SegEnvV10(SyntheticPoolEnv):
    """Replay population of the v10 segmentation (PoolReplayEnv interface; outcome name 'visit')."""

    def __init__(self, data="lenta", S=16, half="dev"):  # noqa: D401 - no super().__init__
        if half != "dev":
            raise PermissionError("seg_v10 is dev-only: eval halves are not accessible from this module")
        if data not in ("lenta", "x5"):
            raise ValueError(data)
        sc = fit_scores(data)
        self.data, self.layer, self.half, self.split_seed = data, f"{data.upper()}-SR{S}", "dev", MODEL_SPLIT_SEED
        self.S, self.A = int(S), 2
        self.seg = seg_labels(sc["score"], S)
        self.arm = sc["arm"]
        self.seg_desc = [f"q{s}/{S}" for s in range(S)]
        y = sc["y"].copy()
        y.setflags(write=False)
        self._y = {"visit": y}
        self.clip = {}
        self.binary_outcomes = ("visit",)
        self.N = int(len(self.seg))
        self.row_index = sc["source_rows"]
        self.cell = self.seg * 2 + self.arm
        self.pool_sizes = np.bincount(self.cell, minlength=2 * S).reshape(S, 2)
        if (self.pool_sizes <= 0).any():
            raise RuntimeError("empty pool in the segmentation")
        self.pool_rows = [np.flatnonzero(self.cell == c) for c in range(2 * S)]
        self.seg_sizes = self.pool_sizes.sum(1)
        self.w = self.seg_sizes / self.N
        self.tau_R = self.N
        self.n_min = 5000 if data == "lenta" else 2000
        self.eps_grid = ()
        self.replan_interval = fr.replan_interval(self.tau_R)
        self.problems = seg_problems(self.w)
        self.score_meta = {k: sc[k] for k in ("features", "n_dev", "n_model", "n_replay", "model_rows_sha",
                                              "fit_sec")}

    def describe(self):
        mu = self.true_mu("visit")
        return {"layer": self.layer, "S": self.S, "N": self.N, "pool_sizes_min": int(self.pool_sizes.min()),
                "w_range": [float(self.w.min()), float(self.w.max())], **self.score_meta,
                "true_uplift_by_segment": (mu[:, 1] - mu[:, 0]).round(5).tolist(),
                "cost": self.problems.cost[:, 1].tolist(), "budgets": self.problems.budgets.tolist()}


# =============================================================================================== truth helpers
def policy_value(w, mu, pi):
    pi = np.asarray(pi, dtype=np.int64)
    return float((np.asarray(w) * np.asarray(mu)[np.arange(len(pi)), pi]).sum())


def true_opt(env, outcome="visit"):
    """J*_q for every problem (exact DP on the true cell means)."""
    from ..baselines.fdc_dp import dp_argmax
    mu = env.true_mu(outcome)
    vals = np.asarray(env.w)[:, None] * mu
    out = []
    for q in range(env.problems.Q):
        v, _ = dp_argmax(vals, env.problems.cost, int(env.problems.budgets[q]))
        out.append(v)
    return np.array(out), mu


# =============================================================================================== stream (fixed designs)
def run_stream_seg(env, method, perm_seed, ctx, J_star=None, mu_true=None, outcome="visit", n80_k=None):
    """Frozen-design stream for implicit policy classes: same schedule generator (make_schedule, labels only), same
    batched engine (StreamEngine; the HC engine of frontier_runner_v6 for cs_kind 'hc'), same sticky answers.

    Horizon (external reviewer FDC-DP r1 B2): the stream runs until ctx.stop_k problems are certified or tau_R (v2 dev and lock
    v10: stop_k = Q = 15); the endpoint is recorded at n80_k = ceil(0.8 Q) = 12 certified problems from the SAME
    trajectory: N80_pen = the checkpoint where 12/15 are first certified if no false certificate exists by then, else
    tau_R (v9 rule); fwer_event = any false certificate over the WHOLE run.  'n80_lt_tau' (strict N80_pen < tau_R) is
    the success event of the eps-selection rule (the v1 field 'censored' was mislabelled and is gone).  Exhaustion
    fractions of the cells at the N80 checkpoint are recorded (Lenta exhaustion qualification).
    Methods return policy tuples."""
    from ..baselines.frontier_common import FrontierState
    t_start = time.perf_counter()
    if J_star is None or mu_true is None:
        J_star, mu_true = true_opt(env, outcome)
    method.setup(ctx)
    if method.alloc_kind != "fixed":
        raise ValueError("run_stream_seg: fixed allocations only")
    eng_cls = StreamEngine
    if getattr(method, "cs_kind", None) == "hc":
        from ..streams.frontier_runner_v6 import hc_engine_class
        eng_cls = hc_engine_class(method.cs_params(ctx))
    sch = make_schedule(env, perm_seed, alloc=method.alloc_p, adaptive=False)
    eng = eng_cls(env, sch, outcome, delta_cell=ctx.delta / (env.S * env.A), R=ctx.R)
    undecided = np.ones(ctx.Q, dtype=bool)
    decided = [None] * ctx.Q
    cert_k = np.full(ctx.Q, -1, dtype=np.int64)
    false_q = np.zeros(ctx.Q, dtype=bool)
    rows = []
    stop_k_idx = None
    n80_k = int(math.ceil(0.8 * ctx.Q - 1e-9)) if n80_k is None else int(n80_k)
    k80 = None
    exh80 = None
    t_cert = 0.0
    for k, b in enumerate(ctx.checkpoints):
        eng.advance_planned(int(b))
        st = FrontierState(eng.t, eng.n, eng.sum, eng.N, undecided, eng.cs_at, None)
        t0 = time.perf_counter()
        res = method.certify(ctx, st)
        dt = time.perf_counter() - t0
        t_cert += dt
        new = []
        for q, (ok, pi, U) in enumerate(res):
            if ok and undecided[q]:
                undecided[q] = False
                decided[q] = [int(x) for x in pi]
                cert_k[q] = k
                false_q[q] = bool(J_star[q] - policy_value(env.w, mu_true, pi) > ctx.eps + 1e-12)
                new.append(q)
        n_cert = int((~undecided).sum())
        if k80 is None and n_cert >= n80_k:
            k80 = k
            ex = np.asarray(eng.n) >= np.asarray(env.pool_sizes)
            exh80 = {"all": float(ex.mean()), "ctrl": float(ex[:, 0].mean()), "treat": float(ex[:, 1].mean())}
        rows.append({"k": k, "t": int(b), "n_cert": n_cert, "n_false": int(false_q.sum()), "new": new,
                     "sec_cert": round(dt, 4), "billing_ok": eng.billing_ok(),
                     "U": [None if (U is None or not np.isfinite(U)) else float(U) for (_, _, U) in res]})
        if n_cert >= ctx.stop_k:
            stop_k_idx = k
            break
    reached = stop_k_idx is not None
    any_false = bool(false_q.any())
    tau = int(ctx.tau_R)
    false80 = bool(k80 is not None and rows[k80]["n_false"] > 0)
    n80_pen = int(ctx.checkpoints[k80]) if (k80 is not None and not false80) else tau
    summ = {"method": method.name, "validity": method.validity, "perm_seed": int(perm_seed),
            "schedule_digest": sch.digest()[:16], "eps": ctx.eps, "reached_stop": reached, "stop_k": ctx.stop_k,
            "n80_k": n80_k, "k_stop": stop_k_idx, "k80": k80, "N80_pen": n80_pen,
            "N80_raw": int(ctx.checkpoints[k80]) if k80 is not None else tau, "n80_lt_tau": bool(n80_pen < tau),
            "false_by_k80": false80, "exhaustion_at_k80": exh80,
            "N_stop_pen": int(ctx.checkpoints[stop_k_idx]) if (reached and not any_false) else tau,
            "n_cert": int((~undecided).sum()), "n_false": int(false_q.sum()), "fwer_event": any_false,
            "cert_k": cert_k.tolist(), "decided_pi": decided, "billing_ok": all(r["billing_ok"] for r in rows),
            "sec_cert": round(t_cert, 3), "sec_total": round(time.perf_counter() - t_start, 3),
            "n_cert_curve": [r["n_cert"] for r in rows], "sec_cert_by_k": [r["sec_cert"] for r in rows],
            "tau_R": tau}
    return summ, rows


def env_digest(env):
    h = hashlib.sha256(np.ascontiguousarray(env.seg).tobytes())
    h.update(np.ascontiguousarray(env.arm).tobytes())
    h.update(np.ascontiguousarray(env.row_index).tobytes())
    return h.hexdigest()[:16]


__all__ += ["env_digest", "model_split", "seg_labels"]
_ = json  # (json kept for callers that serialise describe())
