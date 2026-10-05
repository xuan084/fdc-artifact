"""Round-5 tuned plug-in descriptive comparators (task r5_setup_plugin_variants; methodology s4 / E-cost).

All three methods are plug-in (validity = 'none'): they carry NO finite-sample guarantee and are reported only as
descriptive "price of the guarantee" comparators (never members of the rigorous rival set R). They port the same
design / accounting choices we give FDC to the plug-in family, so that FDC versus the fastest plug-in variant
estimates the cost of rigour rather than the cost of a worse design.

Shared tight union threshold
----------------------------
  beta_tight(ctx) = ln( sum_q |Pi_{B_q}| * K / delta )
the union over every (problem, feasible challenger) pair and the K pre-fixed checkpoints (CR9 dev, eps* = 0.001,
K = 20, delta = 0.05: ~= 14.12). It is computed from the public ctx only (no data), so it is frozen before any draw.
This is the threshold the contrarian offline script (idea/r5_offline/contrarian/offline_ctr.py, global ``TB``) used.

Methods
-------
  * B2-fav-tight  -- CombGame-joint-eps sampling (r4 ``CombGameJoint('fav')``, unchanged) + plug-in GLR certificate
                     with beta = beta_tight (r4 used the frozen beta_dir = ln(|Pi|^2 K / delta)). Offline counterpart:
                     ``mk('B2-tightbeta')``.
  * FIX-bal-fav   -- frozen data-free 50/50 design (control share 0.5 in every segment, rest split equally; the FDC /
                     B4-bal design, ``b4_bal.balanced_alloc``) + plug-in variance GLR (``glr_certificate``) with
                     beta = beta_tight. Offline counterpart: ``mk('FIX0.50-fav-tight')``. ``share=None`` gives the
                     pool-proportional design (offline ``POOL-fav-tight``; identity test only, not registered).
  * Peace-fav-bal -- frozen 50/50 design + the r4 Peace-fav (Alg. 2, alpha = 4, delta_k = delta / (2 k^3) / Q, eps_r =
                     1/10, B = R) round sizes, elimination and stopping, with the round design p_mix fixed to the
                     balanced allocation instead of the SMD design. Because a frozen schedule exposes the state only at
                     the K checkpoints, rounds are checkpoint-aligned: round k starts at the checkpoint at which round
                     k - 1 ended (round 1 starts at t = 0 with the data-free state), and it ends at the first
                     checkpoint t >= t_start + L_k, L_k = max(N_k, q(eps_r), A x replan) rounded up to the re-planning
                     interval (as r4). Elimination uses the round's own samples (n - n0), as r4. No offline
                     counterpart exists (new variant); tested for determinism, invariants and a known-answer toy.
"""
from __future__ import annotations

import math

import numpy as np

from .b4_bal import balanced_alloc
from .combgame_joint import CombGameJoint
from .frontier_common import FrontierState, Method, glr_certificate, plugin_var
from .peace_frontier import PeaceFrontier

__all__ = ["beta_tight", "B2FavTight", "FixBalFav", "PeaceFavBal", "PLUGIN_R5_NAMES"]

PLUGIN_R5_NAMES = ("B2-fav-tight", "FIX-bal-fav", "Peace-fav-bal")


def beta_tight(ctx):
    """ln(sum_q |Pi_{B_q}| K / delta): union over (problem, feasible policy) pairs and the K checkpoints."""
    return math.log(int(ctx.feas.sum()) * len(ctx.checkpoints) / ctx.delta)


# =============================================================================================== B2-fav-tight
class B2FavTight(CombGameJoint):
    """CombGame-joint-eps (fav sampling) + plug-in GLR at beta_tight(ctx). Descriptive only."""

    def __init__(self, name="B2-fav-tight"):
        super().__init__("fav", beta_override=None, name=name)
        self.validity = "none"

    def setup(self, ctx):
        super().setup(ctx)
        self.beta_override = beta_tight(ctx)
        self.validity = "none"

    def describe(self):
        d = super().describe()
        d.update({"beta": self.beta_override, "beta_rule": "ln(sum_q |Pi_Bq| K / delta)", "certificate": "plug-in GLR"})
        return d


# =============================================================================================== FIX-bal-fav
class FixBalFav(Method):
    """Frozen design (balanced 50/50 by default; share=None -> pool proportions) + plug-in variance GLR at
    beta_tight(ctx) (or a fixed ``beta``). Descriptive only."""
    validity = "none"

    def __init__(self, share=0.5, beta=None, name="FIX-bal-fav"):
        self.share = None if share is None else float(share)
        self.beta_fixed = beta
        self.name = name
        self.alloc_kind = "pool" if share is None else "fixed"
        self.alloc_p = None
        self.beta = None

    def setup(self, ctx):
        if self.share is not None:
            self.alloc_p = balanced_alloc(ctx.S, ctx.A, self.share)
        self.beta = beta_tight(ctx) if self.beta_fixed is None else float(self.beta_fixed)

    def certify(self, ctx, st):
        return glr_certificate(ctx, st, plugin_var(st, ctx.R), self.beta)

    def describe(self):
        d = super().describe()
        d.update({"share_control": self.share, "beta": self.beta, "certificate": "plug-in GLR",
                  "design": "frozen fixed shares (data-free)" if self.share is not None else "pool proportions"})
        return d


# =============================================================================================== Peace-fav-bal
class PeaceFavBal(PeaceFrontier):
    """Peace-fav (Alg. 2, alpha = 4) stopping / elimination on a frozen balanced design, checkpoint-aligned rounds."""

    def __init__(self, share=0.5, name="Peace-fav-bal", n_eta=None, seed=2020):
        kw = {} if n_eta is None else {"n_eta": n_eta}
        super().__init__("fav", name=name, seed=seed, **kw)
        self.share = float(share)
        self.alloc_kind = "fixed"
        self.alloc_p = None
        self.validity = "none"

    def setup(self, ctx):
        super().setup(ctx)
        self.alloc_p = balanced_alloc(ctx.S, ctx.A, self.share)
        self.p = self.alloc_p.copy()

    # the round design is the frozen schedule's design (no SMD); round size is evaluated AT that design
    def _design(self, ctx, st, q, rows_idx, var, k):
        return self.alloc_p

    def _zero_state(self, ctx, st):
        z = np.zeros((ctx.S, ctx.A))
        return FrontierState(0, z.astype(np.int64), z, st.N, np.ones(ctx.Q, dtype=bool), lambda n: None)

    def plan(self, ctx, st):  # pragma: no cover - fixed schedule, the harness never calls plan
        raise RuntimeError("Peace-fav-bal uses a frozen schedule (alloc_kind='fixed')")

    def certify(self, ctx, st):
        if self.k == 0:
            self._start_round(ctx, self._zero_state(ctx, st))
        # close every round whose planned end has been reached at this checkpoint (one round per checkpoint at most
        # can close with data; the next starts here), then report the sticky done flags
        if st.t >= self.t_end and not self.done.all():
            self._end_round(ctx, st)
            if not self.done.all():
                self._start_round(ctx, st)
        return super().certify(ctx, st)

    def describe(self):
        d = super().describe()
        d.update({"share_control": self.share, "design": "frozen fixed shares (data-free)",
                  "rounds": "checkpoint-aligned", "alpha": 4.0})
        return d


# =============================================================================================== registration
def _register():
    from ..streams.r5_registry import MethodSpec, register
    register(MethodSpec("B2-fav-tight", "none", "descriptive", False,
                        "CombGame (Jourdan et al. 2021) fav sampling + plug-in GLR, beta = ln(sum_q|Pi_Bq| K/delta)",
                        "baselines/plugin_r5.py", note="tuned plug-in (E-cost); offline: B2-tightbeta"),
             lambda **kw: B2FavTight(**kw))
    register(MethodSpec("FIX-bal-fav", "none", "descriptive", False,
                        "frozen 50/50 design + plug-in variance GLR, beta = ln(sum_q|Pi_Bq| K/delta)",
                        "baselines/plugin_r5.py", {"share": 0.5}, note="tuned plug-in (E-cost); offline: FIX0.50-fav-tight"),
             lambda **kw: FixBalFav(**kw))
    register(MethodSpec("Peace-fav-bal", "none", "descriptive", False,
                        "frozen 50/50 design + Peace Alg. 2 plug-in stopping (alpha = 4), checkpoint-aligned rounds",
                        "baselines/plugin_r5.py", {"share": 0.5}, note="tuned plug-in (E-cost); no offline counterpart"),
             lambda **kw: PeaceFavBal(**kw))


_register()
