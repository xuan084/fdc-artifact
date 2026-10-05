"""Peace (Katz-Samuels, Jain, Karnin & Jamieson, NeurIPS 2020), round-4 frontier adaptation, exact over Pi_all.

Shares the round structure / multi-problem adaptation / tracking of ``rage_frontier.PhasedElimination``; only the
design objective, the round size and the elimination scale differ. Constants transcribed from the paper (arXiv
2006.11685, Alg. 1 and Alg. 2; idea/lit_diff_r4.md s3.1).

Gaussian width (enumeration LMO): with A(lambda)^{-1/2} = diag(sigma / sqrt(lambda)), lambda_{s,a} = w_s p_{s,a}, the
process value of policy z for eta ~ N(0, I_{SA}) is h_z(eta) = sum_s sqrt(w_s) sigma_{s,z(s)} eta_{s,z(s)} / sqrt(p_{s,z(s)}),
and  sup_{z,z' in Z}(z - z')^T A^{-1/2} eta = max_z h_z - min_z h_z  (exact over the enumerated active set).
E[.] is estimated with 2000 Monte-Carlo draws (paper experiments: 2000); its MC standard error is reported.

Alg. 1 (fixed confidence, enumeration) -- used by -nominal
  delta_k = delta / (2 k^2)  (x 1/Q, union over the Q problems -- deviation, as for B2 / RAGE);
  tau(lambda; Z_k) = E[sup (z - z')^T A^{-1/2} eta]^2 + 2 log(1/delta_k) max ||z - z'||^2_{A^{-1}}  (the TIS term);
  N_k = ceil(2 (1 + eps_r) tau_k (2^{k+1} / B)^2) v q(eps_r),  eps_r = 1/10 (paper default), q(eps_r) = d / eps_r^2;
  eliminate z if exists z': (z' - z)^T theta_hat_k >= B / 2^{k+1};
  B: the paper allows "an upper bound on max_z Delta_z when one is known" -> B = R = 1 (data-free, exact bound here).
  On the good event |(z - z')^T (theta_hat_k - theta)| <= B / 2^{k+1} for all pairs (sqrt(2 tau / N_k) by TIS), so
  survivors have gap < B / 2^k; eps-good frontier adaptation: certify when B / 2^k <= eps or |Z_q| = 1.
  Design lambda_k = argmin tau: stochastic mirror descent (paper: 1000 iterations, batch 10), warm-started at the
  XY design; the better of {XY, SMD} under a 500-draw estimate is kept. Compute cap: active sets with more than
  1000 policies (HR8 only) get 100 SMD iterations (deviation; affects design quality, never validity).
-fav: plug-in variance + the paper's RECOMMENDED constant alpha = 4 of Alg. 2:
  N_k = alpha ceil(tau'_k log(1/delta_k) (1 + eps_r)) v q(eps_r), delta_k = delta / (2 k^3) (x 1/Q),
  tau'_k = E[max_{z in Z_k} (z~ - z)^T A^{-1/2} eta / (2^{-k} B + theta_hat^T (z~ - z))]^2  (Alg. 2 objective (4),
  restricted to the enumerated active set; z~ = empirical best, theta_hat = cumulative plug-in means);
  elimination / stopping scale as in Alg. 1 (enumeration is available). No validity claimed.
-rect: the -fav schedule as the SAMPLING rule; certificate = per-cell WSR20 CS rectangle. Rigorous.
-nominal is labelled "nominal" (sigma^2 = 1/4 sub-Gaussian proxy; unproven under WoR + adaptive tracking).
If the nominal constants make Peace never stop within tau_R, that is reported as is (censored).
"""
from __future__ import annotations

import math

import numpy as np

from .frontier_common import jhat_all
from .rage_frontier import PhasedElimination, _support, _finalise, pair_signatures, pair_V, xy_design

__all__ = ["PeaceFrontier", "gw_samples", "gaussian_width", "tau_value", "peace_design", "PEACE_ALPHA_FAV",
           "PEACE_ALPHA_THEORY", "N_ETA", "SMD_ITERS", "SMD_BATCH"]

PEACE_ALPHA_FAV = 4.0          # "we recommend using alpha = 4"
PEACE_ALPHA_THEORY = 42941.0   # "suffices though this is wildly pessimistic" (recorded, not used)
PEACE_EPS_ROUND = 0.1
N_ETA = 2000
SMD_ITERS = 1000
SMD_BATCH = 10
SMD_LARGE_SET = 1000           # compute cap (HR8, |Pi| = 6561): active sets larger than this get SMD_ITERS_LARGE
SMD_ITERS_LARGE = 100          # iterations (design quality only; N_k is always evaluated at the design actually played)


def _coef(w, var, p):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(p > 0, np.sqrt(np.asarray(w)[:, None] * var / np.where(p > 0, p, 1.0)), 0.0)


def _h(rows, coef, eta):
    S = rows.shape[1]
    seg = np.arange(S)
    return (eta * coef[None])[:, seg[None, :], rows].sum(-1)            # (n, m)


def gw_samples(rows, w, var, p, eta):
    """sup_{z,z'} (z - z')^T A^{-1/2} eta for each eta (n,): max_z h_z - min_z h_z."""
    h = _h(np.asarray(rows), _coef(w, var, p), eta)
    return h.max(1) - h.min(1)


def gaussian_width(rows, w, var, p, eta):
    """(mean, relative MC standard error) of the Gaussian width over the draws eta (n, S, A)."""
    s = gw_samples(rows, w, var, p, eta)
    m = float(s.mean())
    return m, float(s.std(ddof=1) / math.sqrt(s.size) / m) if m > 0 else 0.0


def tau_value(rows, inc, w, var, p, eta, logterm):
    W, _ = gaussian_width(rows, w, var, p, eta)
    rho = float(pair_V(inc, w, var, p).max()) if inc.size else 0.0
    return W * W + 2.0 * logterm * rho, W, rho


def peace_design(rows, inc, w, var, logterm, rng, iters=SMD_ITERS, batch=SMD_BATCH, p0=None, eta_eval=None):
    """argmin_p tau(p; Z) by stochastic mirror descent (exponentiated gradient per segment), warm start p0."""
    rows = np.asarray(rows, dtype=np.int64)
    w = np.asarray(w, dtype=float)
    var = np.maximum(np.asarray(var, dtype=float), 1e-12)
    S, A = var.shape
    used = _support(inc, S, A)
    if p0 is None:
        p0, _ = xy_design(inc, w, var)
    p = _finalise(p0, used)
    seg = np.arange(S)
    incf = inc.astype(float)
    for it in range(iters):
        eta = rng.standard_normal((batch, S, A))
        coef = _coef(w, var, p)
        h = _h(rows, coef, eta)
        imax, imin = h.argmax(1), h.argmin(1)
        width = h[np.arange(batch), imax] - h[np.arange(batch), imin]
        gW = np.zeros((S, A))
        ce = coef[None] * eta                                           # (b, S, A)
        for b in range(batch):
            gW[seg, rows[imax[b]]] += ce[b, seg, rows[imax[b]]]
            gW[seg, rows[imin[b]]] -= ce[b, seg, rows[imin[b]]]
        gW *= -0.5 / np.maximum(p, 1e-12) / batch
        c = w[:, None] * var / np.maximum(p, 1e-12)
        V = incf @ c.ravel()
        u = int(np.argmax(V))
        grho = -incf[u].reshape(S, A) * c / np.maximum(p, 1e-12)
        g = 2.0 * width.mean() * gW + 2.0 * logterm * grho
        g = np.where(used, g, 0.0)
        sc = np.abs(g).max()
        if sc <= 0:
            break
        p = p * np.exp(-(0.5 / math.sqrt(it + 1.0)) * g / sc)
        p = _finalise(np.where(used, p, 0.0), used)
    if eta_eval is not None:
        t0 = tau_value(rows, inc, w, var, _finalise(p0, used), eta_eval, logterm)[0]
        t1 = tau_value(rows, inc, w, var, p, eta_eval, logterm)[0]
        if t0 <= t1:
            return _finalise(p0, used)
    return p


class PeaceFrontier(PhasedElimination):
    family = "Peace"

    def __init__(self, variant="rect", name=None, smd_iters=SMD_ITERS, n_eta=N_ETA, seed=2020):
        super().__init__(variant, name)
        self.smd_iters = smd_iters
        self.n_eta = n_eta
        self.seed = seed
        self.B = 1.0
        self.gw_log = []

    def setup(self, ctx):
        super().setup(ctx)
        self.B = 1.0 * ctx.R
        self.gw_log = []

    def _delta_k(self, ctx, k):
        if self.variant == "nominal":
            return ctx.delta / (2.0 * k * k) / ctx.Q
        return ctx.delta / (2.0 * k ** 3) / ctx.Q

    def _eta(self, ctx, k, q, n):
        return np.random.default_rng([self.seed, k, q, 7]).standard_normal((n, ctx.S, ctx.A))

    def _design(self, ctx, st, q, rows_idx, var, k):
        rows = ctx.pols[rows_idx]
        inc = pair_signatures(rows, ctx.A)
        L = math.log(1.0 / self._delta_k(ctx, k))
        rng = np.random.default_rng([self.seed, k, q])
        iters = self.smd_iters if rows.shape[0] <= SMD_LARGE_SET else min(self.smd_iters, SMD_ITERS_LARGE)
        return peace_design(rows, inc, ctx.w, var, L, rng, iters=iters, eta_eval=self._eta(ctx, k, q, 500))

    def _round_size(self, ctx, st, q, rows_idx, var, p_mix, k):
        rows = ctx.pols[rows_idx]
        inc = pair_signatures(rows, ctx.A)
        dk = self._delta_k(ctx, k)
        eta = self._eta(ctx, k, q, self.n_eta)
        scale = (2.0 ** (k + 1) / self.B) ** 2
        if self.variant == "nominal":
            tau, W, rho = tau_value(rows, inc, ctx.w, var, p_mix, eta, math.log(1.0 / dk))
            _, se = gaussian_width(rows, ctx.w, var, p_mix, eta)
            self.gw_log.append({"k": k, "q": int(q), "W": W, "W_mc_rel_se": se, "rho": rho, "tau": tau})
            return int(math.ceil(2.0 * (1 + PEACE_EPS_ROUND) * tau * scale))
        # Alg. 2 objective (4) on the active set, recommended alpha = 4
        mu = st.mu_hat
        J = jhat_all(ctx, mu)[rows_idx]
        it = int(np.argmax(J))
        gap = np.maximum(J[it] - J, 0.0)
        h = _h(rows, _coef(ctx.w, var, p_mix), eta)
        ratio = (h[:, [it]] - h) / (self.B * 2.0 ** (-k) + gap)[None, :]
        smp = ratio.max(1)
        taup = float(smp.mean()) ** 2
        W, se = gaussian_width(rows, ctx.w, var, p_mix, eta)
        self.gw_log.append({"k": k, "q": int(q), "W": W, "W_mc_rel_se": se, "tau_alg2": taup,
                            "tau_alg2_mc_rel_se": float(smp.std(ddof=1) / math.sqrt(smp.size) / max(smp.mean(), 1e-300))})
        return int(PEACE_ALPHA_FAV * math.ceil(taup * math.log(1.0 / dk) * (1 + PEACE_EPS_ROUND)))

    def _thr(self, k):
        return 2.0 ** (-(k + 1))          # x B (= R) applied by the caller

    def _stop_scale(self, k):
        return 2.0 ** (-k)

    def _q_min(self, ctx):
        return ctx.S * ctx.A / PEACE_EPS_ROUND ** 2

    def describe(self):
        d = super().describe()
        d.update({"B": self.B, "gw_log_tail": self.gw_log[-30:]})
        return d
