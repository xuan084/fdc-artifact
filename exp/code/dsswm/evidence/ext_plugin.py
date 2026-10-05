"""q^{ext}: predictable online L2-logistic plug-in on the registered out-of-class directions (methodology 4.2).

Two logistic models over the same likelihood factors as the class (dsswm.evidence.mixed_lr.factorize):
  pair outcome y_ij  : logit = w . x_pair
      'cont' (always) intercept, e_alpha_i, e_beta_j, one-hot incentive level b>=1      (continuous in-cell offsets)
      'g_free'        one-hot (i, n_i) for n_i >= 1  (free fatigue shape g(n), per left participant)
                      [if absent: the class direction -n_i e_gamma_i]
      'psi_i'         one-hot (i, b) for b >= 1       (per-person incentive effect psi_i)
      'syn'           one-hot (i, j)                   (pair synergy)
  retention e_p'     : logit = c * y_p (known offset, c public) + w . x_ret
      'cont' intercept, e_tau_p;  'g_free' one-hot shared n_p >= 1 (free g_ret(n)) [if absent: -n_p]
NO time / round-index features: temporal drift (m4) is an unregistered direction by construction.
Predictable: predictions for round s use weights fitted on rounds < s only. Fits are L2-penalised Newton (IRLS)
refits on a geometric schedule (every time the data grew by `refit_frac`), warm started.
Each prediction is a Bernoulli probability clipped to [p_clip, 1 - p_clip], so prod_s q_s is a valid density.
"""
from __future__ import annotations

import math

import numpy as np

REGISTERED_DIRECTIONS = ("cont", "g_free", "psi_i", "syn")


def _log_sigmoid(z):
    return -np.logaddexp(0.0, -z)


class _Logistic:
    def __init__(self, d, l2, refit_frac, newton_iters):
        self.d, self.l2, self.refit_frac, self.newton_iters = d, l2, refit_frac, newton_iters
        self.w = np.zeros(d)
        self.X = np.zeros((256, d))
        self.off = np.zeros(256)
        self.y = np.zeros(256)
        self.n = 0
        self.n_at_fit = 0
        self.n_fits = 0

    def append(self, X, off, y):
        k = len(y)
        if self.n + k > len(self.y):
            cap = max(2 * len(self.y), self.n + k)
            for name in ("X", "off", "y"):
                a = getattr(self, name)
                b = np.zeros((cap,) + a.shape[1:])
                b[: self.n] = a[: self.n]
                setattr(self, name, b)
        self.X[self.n:self.n + k] = X
        self.off[self.n:self.n + k] = off
        self.y[self.n:self.n + k] = y
        self.n += k
        if self.n >= max(self.n_at_fit * (1.0 + self.refit_frac), self.n_at_fit + 1):
            self.fit()

    def fit(self):
        X, off, y = self.X[: self.n], self.off[: self.n], self.y[: self.n]
        w = self.w.copy()
        I = np.eye(self.d) * self.l2
        for _ in range(self.newton_iters):
            z = X @ w + off
            p = 1.0 / (1.0 + np.exp(-z))
            g = X.T @ (p - y) + self.l2 * w
            Hm = (X * (p * (1 - p))[:, None]).T @ X + I
            step = np.linalg.solve(Hm, g)
            w = w - step
            if np.abs(step).max() < 1e-8:
                break
        self.w = w
        self.n_at_fit = self.n
        self.n_fits += 1

    def logprob(self, X, off, y, p_clip):
        z = X @ self.w + off
        p = np.clip(1.0 / (1.0 + np.exp(-z)), p_clip, 1.0 - p_clip)
        return np.where(y > 0.5, np.log(p), np.log1p(-p))


class ExtPlugin:
    def __init__(self, L: int, R: int, nmax: int, nb: int, c: float, directions=REGISTERED_DIRECTIONS,
                 l2: float = 1.0, refit_frac: float = 0.1, newton_iters: int = 4, p_clip: float = 1e-6):
        bad = set(directions) - set(REGISTERED_DIRECTIONS)
        if bad:
            raise ValueError(f"unregistered ext directions {bad}")
        self.L, self.R, self.P, self.nmax, self.nb, self.c = L, R, L + R, nmax, nb, float(c)
        self.directions = tuple(d for d in REGISTERED_DIRECTIONS if d in directions or d == "cont")
        self.p_clip = p_clip
        # pair feature layout
        lay, k = {}, 0
        for name, size in (("icpt", 1), ("alpha", L), ("beta", R), ("lvl", nb - 1)):
            lay[name] = (k, size); k += size
        if "g_free" in self.directions:
            lay["gfree"] = (k, L * nmax); k += L * nmax
        else:
            lay["gamma"] = (k, L); k += L
        if "psi_i" in self.directions:
            lay["psii"] = (k, L * (nb - 1)); k += L * (nb - 1)
        if "syn" in self.directions:
            lay["syn"] = (k, L * R); k += L * R
        self.lay_pair, self.d_pair = lay, k
        lay, k = {}, 0
        for name, size in (("icpt", 1), ("tau", self.P)):
            lay[name] = (k, size); k += size
        if "g_free" in self.directions:
            lay["gret"] = (k, nmax); k += nmax
        else:
            lay["lam"] = (k, 1); k += 1
        self.lay_ret, self.d_ret = lay, k
        self.m_pair = _Logistic(self.d_pair, l2, refit_frac, newton_iters)
        self.m_ret = _Logistic(self.d_ret, l2, refit_frac, newton_iters)

    # ---------------- features ----------------
    def _x_pair(self, i, j, n, b):
        x = np.zeros(self.d_pair)
        lay = self.lay_pair
        x[lay["icpt"][0]] = 1.0
        x[lay["alpha"][0] + i] = 1.0
        x[lay["beta"][0] + j] = 1.0
        if b >= 1:
            x[lay["lvl"][0] + b - 1] = 1.0
        if "gfree" in lay:
            if n >= 1:
                x[lay["gfree"][0] + i * self.nmax + (n - 1)] = 1.0
        else:
            x[lay["gamma"][0] + i] = -float(n)
        if "psii" in lay and b >= 1:
            x[lay["psii"][0] + i * (self.nb - 1) + (b - 1)] = 1.0
        if "syn" in lay:
            x[lay["syn"][0] + i * self.R + j] = 1.0
        return x

    def _x_ret(self, p, n):
        x = np.zeros(self.d_ret)
        lay = self.lay_ret
        x[lay["icpt"][0]] = 1.0
        x[lay["tau"][0] + p] = 1.0
        if "gret" in lay:
            if n >= 1:
                x[lay["gret"][0] + (n - 1)] = 1.0
        else:
            x[lay["lam"][0]] = -float(n)
        return x

    def _design(self, factors):
        Xp, yp, Xr, offr, yr = [], [], [], [], []
        for f in factors:
            if f.kind == "pair":
                Xp.append(self._x_pair(f.i, f.j, f.load, f.level)); yp.append(f.value)
            else:
                Xr.append(self._x_ret(f.p, f.load)); offr.append(self.c * f.y_partner); yr.append(f.value)
        Xp = np.array(Xp).reshape(-1, self.d_pair); Xr = np.array(Xr).reshape(-1, self.d_ret)
        return Xp, np.array(yp, float), Xr, np.array(offr, float), np.array(yr, float)

    # ---------------- predictable numerator ----------------
    def log_pred(self, factors) -> float:
        """log q^{ext}(realised factors) with weights fitted on PAST data only."""
        Xp, yp, Xr, offr, yr = self._design(factors)
        v = 0.0
        if len(yp):
            v += float(self.m_pair.logprob(Xp, np.zeros(len(yp)), yp, self.p_clip).sum())
        if len(yr):
            v += float(self.m_ret.logprob(Xr, offr, yr, self.p_clip).sum())
        return v

    def update(self, factors) -> None:
        Xp, yp, Xr, offr, yr = self._design(factors)
        if len(yp):
            self.m_pair.append(Xp, np.zeros(len(yp)), yp)
        if len(yr):
            self.m_ret.append(Xr, offr, yr)

    def state(self):
        return {"directions": list(self.directions), "n_pair": self.m_pair.n, "n_ret": self.m_ret.n,
                "n_fits": self.m_pair.n_fits + self.m_ret.n_fits}
