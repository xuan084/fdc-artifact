"""H8: one-step in-model (model-based) certifiers on the real-layer frontier (descriptive only; methodology 1.5).

All three certify INSIDE a parametric / learned model of the cell means; none of them is valid when the model is
misspecified (that is what H8 measures on the real RCT pools). Features are public: they are parsed from the segment
descriptions (``seg_desc``, e.g. CR9 'f0nm=0|f2nm=1|f3nm=0|f9nm=0' -> four binary covariates); without parseable
features (toy) the segment one-hot is used.

Cell feature map (shared by H8-JPC1 and H8-MisLid), cell c = (s, a):
    phi(s, a) = [ onehot_S(s) ,  1[a = a'] (1, x_s)  for a' = 1..A-1 ]       (d = S + (A - 1)(1 + F))
i.e. a free baseline per segment plus an arm effect that is ADDITIVE in the segment covariates -- the one-step
analogue of r3's additive SWM (CR9: d = 14 < 18 cells, so the class is misspecified unless uplift is additive).

* H8-JPC1 (r3 JPC, one step, additive logistic): logit mu(s, a) = phi(s, a) . theta, cell-level MLE (Newton, ridge
  1e-3). Confidence set = r3's sequential likelihood-ratio set with a plug-in (prequential) numerator,
  Theta_t = {theta : L(theta_hat) - L(theta) < R_t + log(1/delta)}, R_t = L(theta_hat) - log q_{1:t}, one set for all
  problems and times (Ville). Numerator: the in-class Bayes mixture q = int p_theta pi(d theta), pi = N(0, 10^2 I)
  (Ville-valid like r3's in-class plug-in; r3 started its plug-in from a Bayes predictive), Laplace-approximated:
  R_t = 1/2 log det(I + 10^2 H_L) + ||theta_hat||^2 / (2 10^2). (A checkpoint-level plug-in would predict the first
  50k arrivals with p = 1/2 and inflate R_t to ~3e4 -- a vacuous certifier; recorded as a deviation.)
  sup over Theta_t of the linearised regret (Laplace / delta method):
      U_q = max_{pi'} Delta_hat(pi', pi_hat) + sqrt(2 (R_t + log 1/delta) g^T H^{-1} g),  g = grad_theta Delta.
* H8-Tboot (T-learner + bootstrap interpolation): per arm a gradient-boosted logistic model on segment features
  (100 rounds, shrinkage 0.1, depth-1 trees over the features and their pairwise ANDs = depth-2 interactions,
  Newton leaves with lambda = 1), fitted on cell sufficient statistics; 200 nonparametric bootstrap refits (rows of
  each arm resampled with replacement = multinomial on (segment, outcome)); certificate
      U_q = Quantile_{1 - delta/Q}( max_{pi'} Delta^b(pi', pi_hat) )  (linear interpolation between order statistics).
* H8-MisLid (Reda, Tirinzoni & Degenne 2021, misspecified linear identification, adapted to eps-good frontier
  answers): linear model mu = Phi theta + e, ||e||_inf <= eta. eta is NOT an oracle input: it is estimated online as
  the max absolute residual of the D_N-weighted projection mu_tilde = Phi theta_tilde over sampled cells (the paper's
  real-data proxy is "max absolute error of the fitted linear model"; methodology 1.5 "non-oracle, online residual").
  Stopping (paper: inf_{lambda in Alt(mu_tilde)} ||mu_tilde - lambda||^2_{D_N} > 2 beta) with the alternative relaxed
  conservatively (deviation e free of cost): certify iff for all pi'
      Delta_tilde(pi', pi_hat) + eta ||z||_1 + sqrt(2 beta sigma^2 z^T Phi (Phi^T D_N Phi)^{-1} Phi^T z) <= eps,
  sigma^2 = 1/4 (Bernoulli proxy; the paper assumes unit-variance Gaussian noise), beta = ln((1 + ln(t + 1)) / (delta/Q))
  (the paper's experimental threshold; union over the Q answers). Sampling: the AdaHedge game learner with C-tracking
  of ``combgame_joint`` (nominal variance) -- MisLid's learner is the same game-based scheme.
"""
from __future__ import annotations

import math
import re

import numpy as np

from .combgame_joint import CombGameJoint
from .frontier_common import Method, jhat_all

PRIOR_SD = 10.0

__all__ = ["segment_features", "cell_features", "JPC1Step", "TLearnerBoot", "MisLid1Step", "boost_logistic",
           "fit_logistic"]


# =============================================================================================== features
def segment_features(seg_desc, S):
    """Parse public segment descriptions into a binary covariate matrix (S, F) (no intercept). Numeric 0/1 values ->
    one column; other values -> one-hot of the non-first levels. Unparseable -> (S, 0)."""
    if seg_desc is None or len(seg_desc) != S or all(re.fullmatch(r"s\d+", str(d)) for d in seg_desc):
        return np.zeros((S, 0))
    recs = []
    for d in seg_desc:
        r = {}
        for i, part in enumerate(str(d).split("|")):
            if "=" in part:
                k, v = part.split("=", 1)
            else:
                k, v = f"_k{i}", part
            r[k.strip()] = v.strip()
        recs.append(r)
    keys = sorted({k for r in recs for k in r})
    cols = []
    for k in keys:
        vals = [r.get(k, "") for r in recs]
        lv = sorted(set(vals))
        if len(lv) <= 1:
            continue
        if set(lv) <= {"0", "1"}:
            cols.append(np.array([float(v) for v in vals]))
        else:
            for level in lv[1:]:
                cols.append(np.array([1.0 if v == level else 0.0 for v in vals]))
    if not cols:
        return np.zeros((S, 0))
    return np.stack(cols, 1)


def cell_features(S, A, X):
    """phi(s, a) rows for cells c = s * A + a: (S A, d)."""
    F = X.shape[1]
    d = S + (A - 1) * (1 + F)
    Phi = np.zeros((S * A, d))
    for s in range(S):
        for a in range(A):
            c = s * A + a
            Phi[c, s] = 1.0
            if a >= 1:
                o = S + (a - 1) * (1 + F)
                Phi[c, o] = 1.0
                Phi[c, o + 1:o + 1 + F] = X[s]
    return Phi


def policy_rows(ctx):
    """X_all (P, S A): z_pi[s * A + pi(s)] = w_s."""
    P, S = ctx.pols.shape
    X = np.zeros((P, S * ctx.A))
    X[np.arange(P)[:, None], np.arange(S)[None, :] * ctx.A + ctx.pols] = ctx.w[None, :]
    return X


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def fit_logistic(Phi, n, k, theta0=None, lam=1e-3, iters=60):
    """Penalised cell-level logistic MLE by damped Newton. Returns (theta, p, H)."""
    d = Phi.shape[1]
    th = np.zeros(d) if theta0 is None else theta0.copy()

    def obj(t):
        eta = Phi @ t
        return -(k * eta - n * np.logaddexp(0.0, eta)).sum() + 0.5 * lam * t @ t

    f = obj(th)
    for _ in range(iters):
        p = _sig(Phi @ th)
        g = Phi.T @ (n * p - k) + lam * th
        H = (Phi * (n * p * (1 - p))[:, None]).T @ Phi + lam * np.eye(d)
        step = np.linalg.solve(H, g)
        a = 1.0
        while a > 1e-6:
            tn = th - a * step
            fn = obj(tn)
            if fn <= f + 1e-12:
                break
            a *= 0.5
        th, f_old, f = tn, f, fn
        if abs(f_old - f) <= 1e-10 * (1 + abs(f)):
            break
    p = _sig(Phi @ th)
    H = (Phi * (n * p * (1 - p))[:, None]).T @ Phi + lam * np.eye(d)
    return th, p, H


def _loglik(p, n, k):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float((k * np.log(p) + (n - k) * np.log1p(-p)).sum())


# =============================================================================================== H8-JPC1
class JPC1Step(Method):
    alloc_kind = "pool"
    validity = "model"

    def __init__(self, seg_desc=None, name="H8-JPC1"):
        self.name = name
        self.seg_desc = seg_desc

    def setup(self, ctx):
        self.Phi = cell_features(ctx.S, ctx.A, segment_features(self.seg_desc, ctx.S))
        self.Xall = policy_rows(ctx)
        self.theta = None
        self.trace = []

    def certify(self, ctx, st):
        n = st.n.ravel().astype(float)
        k = st.sum.ravel().astype(float)
        lam = 1.0 / PRIOR_SD ** 2
        th, p, H = fit_logistic(self.Phi, n, k, self.theta, lam=lam)
        sign, logdet = np.linalg.slogdet(np.eye(H.shape[0]) + PRIOR_SD ** 2 * (H - lam * np.eye(H.shape[0])))
        Rt = max(0.5 * logdet + 0.5 * float(th @ th) * lam, 0.0)
        r2 = 2.0 * (Rt + math.log(1.0 / ctx.delta))
        mu = p.reshape(ctx.S, ctx.A)
        J = jhat_all(ctx, mu)
        G = self.Xall @ (self.Phi * (p * (1 - p))[:, None])            # (P, d) grad of J(pi)
        Hinv = np.linalg.inv(H)
        out = []
        for q in range(ctx.Q):
            ih = int(np.argmax(np.where(ctx.feas[q], J, -np.inf)))
            D = G - G[ih]
            quad = np.maximum(((D @ Hinv) * D).sum(1), 0.0)
            U = (J - J[ih]) + np.sqrt(r2 * quad)
            Uq = float(np.max(np.where(ctx.feas[q], U, -np.inf)))
            out.append((bool(Uq <= ctx.eps), ih, Uq))
        self.theta = th
        self.trace.append({"t": int(st.t), "R_t": Rt, "r2": r2})
        return out

    def describe(self):
        d = super().describe()
        d.update({"d": int(self.Phi.shape[1]) if hasattr(self, "Phi") else None, "trace_tail": getattr(self, "trace", [])[-5:]})
        return d


# =============================================================================================== H8-Tboot
def boost_features(X, S):
    if X.shape[1] == 0:
        return np.eye(S)
    cols = [X[:, j] for j in range(X.shape[1])]
    binary = [j for j in range(X.shape[1]) if set(np.unique(X[:, j])) <= {0.0, 1.0}]
    for i in range(len(binary)):
        for j in range(i + 1, len(binary)):
            c = X[:, binary[i]] * X[:, binary[j]]
            if 0 < c.sum() < S:
                cols.append(c)
    F = np.stack(cols, 1)
    keep = [j for j in range(F.shape[1]) if 0 < F[:, j].sum() < S]
    return F[:, keep] if keep else np.eye(S)


def boost_logistic(Fb, n, k, rounds=100, lr=0.1, lam=1.0):
    """Gradient-boosted logistic model with depth-1 trees on binary features Fb (S, J), vectorised over replicates.
    n, k: (R, S) counts / ones. Returns fitted probabilities (R, S)."""
    n = np.asarray(n, dtype=float)
    k = np.asarray(k, dtype=float)
    R, S = n.shape
    tot = n.sum(1)
    base = np.where(tot > 0, (k.sum(1) + 0.5) / (tot + 1.0), 0.5)
    f = np.repeat(np.log(base / (1 - base))[:, None], S, 1)
    right = Fb.astype(bool)                                           # (S, J)
    rr = np.arange(R)
    for _ in range(rounds):
        p = _sig(f)
        g = k - n * p
        h = n * p * (1 - p)
        GR, HR = g @ right, h @ right                                 # (R, J)
        GL, HL = g.sum(1, keepdims=True) - GR, h.sum(1, keepdims=True) - HR
        gain = GL ** 2 / (HL + lam) + GR ** 2 / (HR + lam)
        j = gain.argmax(1)
        vR = GR[rr, j] / (HR[rr, j] + lam)
        vL = GL[rr, j] / (HL[rr, j] + lam)
        f += lr * np.where(right[:, j].T, vR[:, None], vL[:, None])
    return _sig(f)


class TLearnerBoot(Method):
    alloc_kind = "pool"
    validity = "model"

    def __init__(self, seg_desc=None, B=200, rounds=100, lr=0.1, seed=4040, name="H8-Tboot"):
        self.name = name
        self.seg_desc = seg_desc
        self.B, self.rounds, self.lr, self.seed = B, rounds, lr, seed

    def setup(self, ctx):
        self.Fb = boost_features(segment_features(self.seg_desc, ctx.S), ctx.S)

    def certify(self, ctx, st):
        rng = np.random.default_rng([self.seed, int(st.t)])
        S, A = ctx.S, ctx.A
        mu0 = np.zeros((S, A))
        mub = np.zeros((self.B, S, A))
        for a in range(A):
            n_a, k_a = st.n[:, a].astype(float), st.sum[:, a].astype(float)
            Na = int(n_a.sum())
            mu0[:, a] = boost_logistic(self.Fb, n_a[None], k_a[None], self.rounds, self.lr)[0]
            if Na == 0:
                mub[:, :, a] = 0.5
                continue
            probs = np.concatenate([k_a, n_a - k_a]) / Na
            dr = rng.multinomial(Na, probs / probs.sum(), size=self.B)
            kb = dr[:, :S].astype(float)
            mub[:, :, a] = boost_logistic(self.Fb, kb + dr[:, S:], kb, self.rounds, self.lr)
        J0 = jhat_all(ctx, mu0)
        seg = np.arange(S)
        Jb = (ctx.w[None, None, :] * mub[:, seg[None, :], ctx.pols]).sum(-1)      # (B, P)
        lvl = 1.0 - ctx.delta / ctx.Q
        out = []
        for q in range(ctx.Q):
            ih = int(np.argmax(np.where(ctx.feas[q], J0, -np.inf)))
            Db = np.max(np.where(ctx.feas[q][None, :], Jb - Jb[:, [ih]], -np.inf), 1)
            Uq = float(np.quantile(Db, lvl, method="linear"))
            out.append((bool(Uq <= ctx.eps), ih, Uq))
        return out


# =============================================================================================== H8-MisLid
class MisLid1Step(CombGameJoint):
    def __init__(self, seg_desc=None, eps_mis="online", beta="heur", name="H8-MisLid"):
        super().__init__("nominal", name=name)
        self.validity = "model"
        self.seg_desc = seg_desc
        self.eps_mis = eps_mis
        self.beta_kind = beta

    def setup(self, ctx):
        super().setup(ctx)
        self.Phi = cell_features(ctx.S, ctx.A, segment_features(self.seg_desc, ctx.S))
        self.Xall = policy_rows(ctx)
        self.trace = []

    def _beta(self, ctx, t, d):
        dq = ctx.delta / ctx.Q
        if self.beta_kind == "heur":
            return math.log((1 + math.log(t + 1)) / dq)
        L = math.log(1 / dq)               # beta^lin of the paper (eta inserted for eps), sub-Gaussian units
        eta = self._eta if isinstance(self._eta, float) else 0.0
        return 0.5 * (4 * math.sqrt(t) * eta + math.sqrt(2) * math.sqrt(1 + L + (1 + 1 / L) * (d / 2) *
                                                                         math.log(1 + t / (2 * d) * L))) ** 2

    def certify(self, ctx, st):
        n = st.n.ravel().astype(float)
        y = st.mu_hat.ravel()
        Phi = self.Phi
        d = Phi.shape[1]
        M = (Phi * n[:, None]).T @ Phi + 1e-6 * np.eye(d)
        Minv = np.linalg.inv(M)
        th = Minv @ (Phi.T @ (n * y))
        mut = Phi @ th
        live = n > 0
        eta = float(np.max(np.abs(y - mut)[live])) if (self.eps_mis == "online" and live.any()) else \
            (float(self.eps_mis) if self.eps_mis != "online" else 1.0)
        self._eta = eta
        beta = self._beta(ctx, max(st.t, 1), d)
        Jt = self.Xall @ mut
        Gm = Phi @ Minv @ Phi.T
        out = []
        for q in range(ctx.Q):
            ih = int(np.argmax(np.where(ctx.feas[q], Jt, -np.inf)))
            Zd = self.Xall - self.Xall[ih]
            den = np.maximum(((Zd @ Gm) * Zd).sum(1), 0.0)
            U = (Jt - Jt[ih]) + eta * np.abs(Zd).sum(1) + np.sqrt(2 * beta * 0.25 * ctx.R ** 2 * den)
            U[ih] = 0.0
            Uq = float(np.max(np.where(ctx.feas[q], U, -np.inf)))
            out.append((bool(Uq <= ctx.eps), ih, Uq))
        self.trace.append({"t": int(st.t), "eta_hat": eta, "beta": beta})
        return out

    def describe(self):
        d = super().describe()
        d.update({"eps_mis": self.eps_mis, "beta": self.beta_kind, "trace_tail": getattr(self, "trace", [])[-5:]})
        return d
