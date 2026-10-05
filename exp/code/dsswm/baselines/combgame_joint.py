"""B2: CombGame-joint-eps (Jourdan, Mutny, Kirschner & Krause, ALT 2021), round-4 frontier adaptation.

Sampling rule (shared by all three variants; game-based with tracking)
---------------------------------------------------------------------
Every batch of ``ctx.replan`` arrivals:
  1. answer: pi_hat_q = argmax J_hat over Pi_{B_q} (enumeration) for every undecided problem q;
  2. best response (nature): pi~_q = argmin_{pi' in Pi_{B_q}, pi' != pi_hat_q} Z_q(pi'),
       Z_q(pi') = (eps - Delta_hat(pi', pi_hat_q))_+^2 / (2 V(pi', pi_hat_q))       (GLR, sign as in r4_lit_g0),
     computed by exact enumeration of Pi_all (CR9: 512, HR8: 6561);
  3. learner gains: the closest-alternative gain of cell c, g_c = sum_q a_{q,c}^2 var_c (eps - Delta_hat_q)_+^2 /
     (2 V_q^2 lambda_c^2) with lambda_c = n_c / t (the gradient of Z_q in the allocation), normalised per segment;
     AdaHedge-style exponential weights (eta_r = sqrt(8 ln A / r)) per segment over its A arms -> target p_{s, .};
     arrivals are exogenous segments, so the learner allocates within segment;
  4. C-tracking: cumulative targets T_{s,a} += p_{s,a} x (segment-s arrivals of the last batch); the target arm of
     segment s is argmax_a (T_{s,a} + p_{s,a} E[batch arrivals of s] - n_{s,a}); forced exploration: if
     min_a n_{s,a} < sqrt(n_s) the least-sampled arm is targeted. Re-selection order: descending deficit.
  The variance used by the sampling rule is 1/4 for the nominal variant and the plug-in mu_hat (1 - mu_hat) for the
  rect / fav variants (sampling rules carry no validity burden).

Variants (methodology R3)
-------------------------
  * B2-rect    -- CombGame sampling; certificate = per-cell WSR20 CS rectangle (``rect_certificate``). Rigorous.
  * B2-nominal -- sigma^2 = 1/4 sub-Gaussian proxy GLR with a threshold frozen BEFORE any data: among the three
                  deterministic functions {beta_dir, beta_KK, beta_CG} the smallest at the median planned arrival
                  count is frozen (``freeze_beta``). Label: "nominal -- valid with replacement, unproven under WoR +
                  adaptive counts" (qfc_lemma s8).
  * B2-fav     -- plug-in variance + the same frozen threshold. No validity claimed (descriptive / most favourable).
  B2-min is deleted (R3).

Thresholds (transcribed in idea/lit_diff_r4.md s3.3-3.4)
  beta_dir           = ln(|Pi|^2 K / delta)        (Gaussian fixed-time tail, union over ordered pairs and K checkpoints)
  beta_KK(n)         = 2 sum_c ln(4 + ln n_c) + d C^{g_G}(ln(1/delta) / d),  d = S A    (joint, all functionals)
  beta_CG(t)         = 2 d0 ln(4 + ln(t Kmax / d0)) + d0 C^{g_G}(ln(Q (|I| - 1) / delta) / d0),  d0 = 2S, Kmax = S
  g_G(l) = 2l - 2l ln(4l) + ln zeta(2l) - 0.5 ln(1 - l),  C^{g}(x) = min_{l in (1/2, 1]} (g(l) + x) / l.
  CombGame's own threshold is single-answer (ln((|I|-1)/delta)); 15 problems share one stream, so the multi-problem
  adaptation takes the union over the Q answers (factor Q inside the log) -- recorded as a deviation.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import zeta

from .frontier_common import Method, glr_certificate, glr_V, jhat_all, pi_hat_indices, plugin_var, rect_certificate

__all__ = ["CombGameJoint", "g_G", "C_gG", "beta_dir", "beta_KK", "beta_CG", "freeze_beta", "VARIANTS"]

VARIANTS = ("rect", "nominal", "fav")


# =============================================================================================== thresholds
def g_G(lam):
    lam = np.asarray(lam, dtype=float)
    return 2 * lam - 2 * lam * np.log(4 * lam) + np.log(zeta(2 * lam, 1)) - 0.5 * np.log1p(-lam)


def C_gG(x):
    """C^{g_G}(x) = min_{lambda in (1/2, 1)} (g_G(lambda) + x) / lambda (grid + bounded refinement)."""
    grid = np.linspace(0.5 + 1e-6, 1 - 1e-9, 20001)
    vals = (g_G(grid) + x) / grid
    i = int(np.argmin(vals))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    r = minimize_scalar(lambda l: float((g_G(l) + x) / l), bounds=(lo, hi), method="bounded",
                        options={"xatol": 1e-12})
    return float(min(vals[i], r.fun))


def beta_dir(P, K, delta):
    return math.log(P * P * K / delta)


def beta_KK(n, delta):
    n = np.asarray(n, dtype=float).ravel()
    d = n.size
    return float(2 * np.sum(np.log(4 + np.log(np.maximum(n, 1.0)))) + d * C_gG(math.log(1 / delta) / d))


def beta_CG(t, S, P, Q, delta):
    d0, Kmax = 2 * S, S
    return float(2 * d0 * math.log(4 + math.log(max(t * Kmax / d0, 1.0))) +
                 d0 * C_gG(math.log(Q * (P - 1) / delta) / d0))


def freeze_beta(ctx):
    """Data-free choice: evaluate the three functions at the median planned arrival count (median of the K
    checkpoints, expected pool-proportional cell counts) and freeze the smallest FUNCTION."""
    t_med = float(np.median(ctx.checkpoints))
    N = ctx.N.astype(float)
    n_exp = t_med * ctx.w[:, None] * N / N.sum(1, keepdims=True)
    K = len(ctx.checkpoints)
    vals = {"dir": beta_dir(ctx.P, K, ctx.delta), "KK": beta_KK(n_exp, ctx.delta),
            "CG": beta_CG(t_med, ctx.S, ctx.P, ctx.Q, ctx.delta)}
    choice = min(vals, key=vals.get)
    return {"choice": choice, "t_med": t_med, "values_at_t_med": vals,
            "C_gG": {"KK_arg": math.log(1 / ctx.delta) / (ctx.S * ctx.A),
                     "CG_arg": math.log(ctx.Q * (ctx.P - 1) / ctx.delta) / (2 * ctx.S)}}


def beta_eval(spec, ctx, st):
    if spec["choice"] == "dir":
        return spec["values_at_t_med"]["dir"]
    if spec["choice"] == "KK":
        return beta_KK(st.n, ctx.delta)
    return beta_CG(max(st.t, 1), ctx.S, ctx.P, ctx.Q, ctx.delta)


# =============================================================================================== method
class CombGameJoint(Method):
    alloc_kind = "adaptive"

    def __init__(self, variant="rect", beta_override=None, name=None):
        """beta_override: harness power control only (e.g. ln(1/delta), no union); never a reported baseline."""
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.variant = variant
        self.name = name or f"B2-{variant}"
        self.validity = {"rect": "rigorous", "nominal": "nominal", "fav": "none"}[variant]
        if beta_override is not None:
            self.validity = "none"
        self.beta_override = beta_override
        self.beta_spec = None

    def setup(self, ctx):
        self.beta_spec = freeze_beta(ctx)
        self._cum_gain = np.zeros((ctx.S, ctx.A))
        self._T = np.zeros((ctx.S, ctx.A))
        self._p = np.full((ctx.S, ctx.A), 1.0 / ctx.A)
        self._last_ns = np.zeros(ctx.S)
        self._round = 0

    def _var(self, ctx, st):
        if self.variant == "nominal":
            return np.full((ctx.S, ctx.A), 0.25 * ctx.R * ctx.R)
        v = plugin_var(st, ctx.R)
        return np.where(st.n > 0, v, 0.25 * ctx.R * ctx.R)

    def plan(self, ctx, st):
        self._round += 1
        n = st.n.astype(float)
        ns = n.sum(1)
        # tracking: credit last batch's segment arrivals with last targets
        self._T += self._p * (ns - self._last_ns)[:, None]
        self._last_ns = ns.copy()
        var = self._var(ctx, st)
        var_s = np.maximum(var, 1e-6)
        mu = st.mu_hat
        J = jhat_all(ctx, mu)
        ihs = pi_hat_indices(ctx, mu)
        t = max(st.t, 1)
        gain = np.zeros((ctx.S, ctx.A))
        seg = np.arange(ctx.S)
        if (n > 0).all():
            for q in np.flatnonzero(st.undecided):
                ih = ihs[q]
                V = glr_V(ctx, st, var_s, ih)
                gap = np.maximum(ctx.eps - (J - J[ih]), 0.0)
                with np.errstate(divide="ignore", invalid="ignore"):
                    Z = np.where(V > 0, gap ** 2 / (2 * V), np.inf)
                Z[ih] = np.inf
                Z = np.where(ctx.feas[q], Z, np.inf)
                br = int(np.argmin(Z))
                if not np.isfinite(Z[br]) or V[br] <= 0:
                    continue
                ph, pc = ctx.pols[ih], ctx.pols[br]
                d = np.flatnonzero(ph != pc)
                lam = n / t
                coef = gap[br] ** 2 / (2 * V[br] ** 2)
                for arm_row in (pc, ph):
                    a = arm_row[d]
                    gain[d, a] += coef * ctx.w[d] ** 2 * var_s[d, a] / (lam[d, a] ** 2) / t
        mx = gain.max(1, keepdims=True)
        g = np.where(mx > 0, gain / np.where(mx > 0, mx, 1.0), 0.0)
        self._cum_gain += g
        eta = math.sqrt(8 * math.log(max(ctx.A, 2)) / self._round)
        z = eta * (self._cum_gain - self._cum_gain.max(1, keepdims=True))
        p = np.exp(z)
        self._p = p / p.sum(1, keepdims=True)
        exp_arr = ctx.w * ctx.replan
        deficit = self._T + self._p * exp_arr[:, None] - n
        pref = np.zeros((ctx.S, ctx.A), dtype=np.int64)
        for s in range(ctx.S):
            order = [int(a) for a in np.argsort(-deficit[s], kind="stable")]
            if n[s].min() < math.sqrt(max(ns[s], 1.0)):
                low = int(np.argmin(n[s]))
                order.remove(low)
                order = [low] + order
            pref[s] = order
        return pref

    def certify(self, ctx, st):
        if self.variant == "rect":
            return rect_certificate(ctx, st)
        beta = self.beta_override if self.beta_override is not None else beta_eval(self.beta_spec, ctx, st)
        if self.variant == "nominal":
            var = np.full((ctx.S, ctx.A), 0.25 * ctx.R * ctx.R)
        else:
            var = plugin_var(st, ctx.R)
        return glr_certificate(ctx, st, var, beta)

    def describe(self):
        d = super().describe()
        d.update({"variant": self.variant, "beta_spec": self.beta_spec})
        return d
