"""Falsification layer: whole-sequence mixed numerator over a finite class (methodology section 4.2).

  M_t^mixed(theta) = [ sum_c w_c prod_{s<=t} q^c_s(x_s) ] / prod_{s<=t} p_theta(x_s),   w = (1/3, 1/3, 1/3)
components c (each a predictable density for the next real round, built only from past real rounds):
  pool : in-class plug-in, q_s = p_{MLE_{s-1}} (first round: uniform-prior Bayes predictive) -- identical to
         SeqLRSet(numerator='plugin')
  reg  : regime-indexed in-class plug-in. Every likelihood factor f of a round (pair outcome / retention bit) gets a
         regime r(f) from a public regime function (dsswm.evidence.regimes); it is predicted by the class MLE fitted
         on PAST factors of the same regime only (uniform-prior factor mixture while the regime has no data)
  ext  : online L2-logistic plug-in on the registered out-of-class directions (dsswm.evidence.ext_plugin)
The three numerator SEQUENCES are multiplied over the whole sequence and only then mixed (no per-step mixing), so
  log M_mixed(theta) - log M_pool(theta) = logsumexp_c(log w_c + LN_c) - LN_pool >= log w_pool = -log 3
holds deterministically, and M_mixed(theta*) is a non-negative martingale under theta* (average of three), so by
Ville  P(exists t: theta* not in Theta_t) <= delta.   Theta_t = {theta: log M_mixed(theta) < log(1/delta)}.
Theta_t empty -> MODEL_CONFLICT (the whole class is refuted at zero extra interaction).
'half' variant: components {pool, ext}, weights (1/2, 1/2), cost log 2.

Falsification scope: every certificate carries `falsification_scope()` = registered components + directions.
A plain in-class SeqLRSet has scope {} (empty).

MODEL_CONFLICT fallback: `model_conflict_fallback` runs B2 (residual + empirical-Bernstein whole-trial BAI, dsswm.
baselines.whole_trial_bai.lucb(kind='eb')) on the problem; its steps are charged to the problem by the caller.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from ..baselines.whole_trial_bai import lucb
from ..core.provenance import require_authentic
from .ext_plugin import REGISTERED_DIRECTIONS, ExtPlugin
from .regimes import PublicFactor, load_bucket

COMPONENTS = ("pool", "reg", "ext")


@dataclass(frozen=True)
class Factor:
    kind: str
    i: int
    j: int
    p: int
    load: int
    level: int
    value: int         # realised value (y for 'pair', e' for 'ret')  -- never shown to regime functions
    y_partner: int     # for 'ret': outcome of p's pair this round (revealed earlier in the factorisation)
    col: int           # column of the factor in the propagator log-table LT

    def public(self, problem_id, t) -> PublicFactor:
        return PublicFactor(self.kind, self.i, self.j, self.p, self.load, self.level, problem_id, t)


def factorize(prop, obs):
    """Likelihood factors of one observation in the order of NLPropagator.obs_factors, plus the known constant
    (disengaged participants' return probability, identical under every theta and every numerator)."""
    L, P = prop.L, prop.P
    fac = []
    y_p = [0] * P
    const = 0.0
    for (i, j, b, y, active) in obs.outcomes:
        if active:
            fac.append(Factor("pair", i, j, i, int(obs.loads[i]), int(b), int(y), 0,
                              prop.py_idx(y, i, j, obs.loads[i], b)))
            y_p[i] = y_p[L + j] = y
    for p in range(P):
        e_next = obs.next_engaged[p]
        if obs.engaged[p]:
            fac.append(Factor("ret", -1, -1, p, int(obs.loads[p]), 0, int(e_next), int(y_p[p]),
                              prop.re_idx(p, y_p[p], obs.loads[p], e_next)))
        else:
            const += prop.log_rho if e_next == 1 else prop.log_1mrho
    return fac, const


class MixedLRSet:
    def __init__(self, prop, LT: torch.Tensor, delta: float, components=COMPONENTS, weights=None,
                 regime_fn=load_bucket, ext_directions=REGISTERED_DIRECTIONS, ext_kwargs=None, name=None):
        comps = tuple(components)
        if not set(comps) <= set(COMPONENTS) or "pool" not in comps:
            raise ValueError(f"components must include 'pool' and be a subset of {COMPONENTS}")
        self.prop, self.LT, self.delta = prop, LT, float(delta)
        self.B = LT.shape[0]
        self.components = comps
        w = np.full(len(comps), 1.0 / len(comps)) if weights is None else np.asarray(weights, float)
        assert abs(w.sum() - 1.0) < 1e-12 and (w > 0).all()
        self.log_w = np.log(w)
        self.regime_fn = regime_fn
        self.cum = torch.zeros(self.B, device=LT.device, dtype=LT.dtype)
        self.LN = {c: 0.0 for c in comps}
        self.reg_cum: dict = {}
        self.ext = None
        if "ext" in comps:
            kw = dict(ext_kwargs or {})
            self.ext = ExtPlugin(prop.L, prop.R, prop.nmax, prop.nb, prop.c, directions=ext_directions, **kw)
        self.n_rounds = 0
        self.serials: list[int] = []
        self.name = name or "+".join(comps)
        self.conflict_round = None          # first round index (1-based) at which Theta_t became empty
        self.min_gap_vs_pool = 0.0          # running min of log M_mixed - log M_pool (must stay >= log w_pool)

    # ---------------- per-component predictive ----------------
    def _pool_pred(self, ll_obs: torch.Tensor) -> float:
        if self.n_rounds == 0:
            return float(torch.logsumexp(self.cum + ll_obs, 0) - torch.logsumexp(self.cum, 0))
        return float(ll_obs[int(torch.argmax(self.cum))])

    def _reg_pred_and_update(self, fac, llf: torch.Tensor, problem_id) -> float:
        v = 0.0
        labels = [self.regime_fn(f.public(problem_id, self.n_rounds)) for f in fac]
        # predict all factors of the round from past-only regime MLEs, then update
        for k, r in enumerate(labels):
            c = self.reg_cum.get(r)
            if c is None:
                v += float(torch.logsumexp(llf[:, k], 0) - math.log(self.B))
            else:
                v += float(llf[int(torch.argmax(c)), k])
        for k, r in enumerate(labels):
            c = self.reg_cum.get(r)
            self.reg_cum[r] = llf[:, k].clone() if c is None else c + llf[:, k]
        return v

    # ---------------- update ----------------
    def update(self, obs, problem_id=None) -> None:
        require_authentic(obs)
        fac, const = factorize(self.prop, obs)
        cols = torch.as_tensor([f.col for f in fac], dtype=torch.long, device=self.LT.device)
        llf = self.LT[:, cols] if fac else torch.zeros(self.B, 0, device=self.LT.device, dtype=self.LT.dtype)
        ll_obs = llf.sum(1) + const
        if "pool" in self.LN:
            self.LN["pool"] += self._pool_pred(ll_obs)
        if "reg" in self.LN:
            self.LN["reg"] += self._reg_pred_and_update(fac, llf, problem_id) + const
        if "ext" in self.LN:
            self.LN["ext"] += self.ext.log_pred(fac) + const
            self.ext.update(fac)
        self.cum = self.cum + ll_obs
        self.n_rounds += 1
        self.serials.append(obs.serial)
        gap = self.log_num() - self.LN["pool"]
        self.min_gap_vs_pool = min(self.min_gap_vs_pool, gap)
        if self.conflict_round is None and self.size() == 0:
            self.conflict_round = self.n_rounds

    # ---------------- set ----------------
    def log_num(self) -> float:
        a = np.array([self.LN[c] for c in self.components]) + self.log_w
        m = a.max()
        return float(m + math.log(np.exp(a - m).sum()))

    def log_ratio(self) -> torch.Tensor:
        return self.log_num() - self.cum

    def log_ratio_pool(self) -> torch.Tensor:
        return self.LN["pool"] - self.cum

    def mask(self) -> torch.Tensor:
        return self.log_ratio() < math.log(1.0 / self.delta)

    def size(self) -> int:
        return int(self.mask().sum())

    def is_conflict(self) -> bool:
        return self.size() == 0

    def status(self) -> str | None:
        return "MODEL_CONFLICT" if self.is_conflict() else None

    def mle(self) -> int:
        return int(torch.argmax(self.cum))

    def evidence_margin(self) -> float:
        """log(1/delta) - min_theta log M_mixed(theta): > 0 iff Theta_t non-empty."""
        return float(math.log(1.0 / self.delta) - (self.log_num() - float(self.cum.max())))

    def component_advantage(self) -> dict:
        """LN_c - max_theta cum(theta): how far each numerator beats the best in-class fit (nats)."""
        mx = float(self.cum.max())
        return {c: self.LN[c] - mx for c in self.components}

    def falsification_scope(self) -> dict:
        scope = {"components": [c for c in self.components if c != "pool"]}
        if "reg" in self.components:
            scope["regime"] = getattr(self.regime_fn, "__name__", type(self.regime_fn).__name__)
        if "ext" in self.components:
            scope["ext_directions"] = [d for d in self.ext.directions]
        return scope

    def state_digest(self):
        return (self.cum.detach().cpu().numpy().copy(), dict(self.LN), tuple(self.serials))


def make_half_set(prop, LT, delta, **kw):
    """1/2-1/2 fallback {pool, ext} (cost log 2)."""
    return MixedLRSet(prop, LT, delta, components=("pool", "ext"), name="half", **kw)


def pool_scope() -> dict:
    """Falsification scope of a pure in-class LR set: empty."""
    return {"components": []}


def model_conflict_fallback(sampler, K: int, H: int, eps: float, delta: float, u_max: float, max_steps: int,
                            proxy=None):
    """MODEL_CONFLICT -> B2 (residual + empirical Bernstein whole-trial BAI). The caller charges out['steps']."""
    out = lucb(sampler, K, H, eps, delta, u_max, max_steps, kind="eb", proxy=proxy)
    out["fallback"] = "B2"
    return out
