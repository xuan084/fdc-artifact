"""Regime functions for the regime-indexed plug-in numerator q^{reg} (methodology section 4.2).

A regime function maps ONE public likelihood factor to a hashable regime label. It may read only the fields of
`PublicFactor` listed in PUBLIC_FIELDS: the factor kind, the participant indices, the participant's observable load
BEFORE the action, the incentive level of the action, the public problem id and the public ledger round index.
It never sees the realised value of the factor, true parameters, or the environment object.
`tests/test_falsify_layer.py` enforces this with an AST check (attribute reads restricted to PUBLIC_FIELDS, no
imports of ground-truth modules, no free names outside an allow-list).

Arms
  load_bucket        primary arm, regime = min(load, 2) in {0, 1, >=2}
  problem_id_regime  secondary arm, regime = public problem id
  PlaceboRegime      placebo: a label in {0, 1, 2} drawn by a fixed per-instance hash of (round index, factor id),
                     with the registered bucket frequencies, independent of the load (fixed before any data).
                     NB a relabelling of the true buckets would be a bijection and have identical power, so the
                     placebo must decouple the label from the load; the frequencies are kept.
"""
from __future__ import annotations

from dataclasses import dataclass

PUBLIC_FIELDS = ("kind", "i", "j", "p", "load", "level", "problem_id", "t")


@dataclass(frozen=True)
class PublicFactor:
    kind: str            # 'pair' (outcome y_ij of an active matched pair) or 'ret' (retention bit of an engaged p)
    i: int               # left participant of the pair (-1 for 'ret')
    j: int               # right participant of the pair (-1 for 'ret')
    p: int               # participant index in 0..P-1 whose load drives the factor (pair: left i; ret: p)
    load: int            # observable load of participant p before the action
    level: int           # incentive level of the pair (0 for 'ret')
    problem_id: object   # public problem id under which the round was collected (None for shared initial data)
    t: int               # public ledger round index (0-based count of evidence rounds)


def load_bucket(pf: PublicFactor):
    return min(int(pf.load), 2)


def problem_id_regime(pf: PublicFactor):
    return pf.problem_id


def _mix64(x: int) -> int:
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
    x = (x ^ (x >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
    return x ^ (x >> 31)


class PlaceboRegime:
    """Load-independent placebo labels with fixed frequencies (registered before data)."""

    def __init__(self, seed: int, freqs=(0.45, 0.35, 0.20)):
        tot = float(sum(freqs))
        self.seed = int(seed)
        self.cuts = []
        acc = 0.0
        for f in freqs[:-1]:
            acc += f / tot
            self.cuts.append(acc)

    def __call__(self, pf: PublicFactor):
        key = (self.seed * 1000003 + pf.t) * 131 + pf.p * 7 + (1 if pf.kind == "pair" else 0) * 3 + pf.j + 1
        u = (_mix64(key & 0xFFFFFFFFFFFFFFFF) >> 11) / float(1 << 53)
        lab = 0
        for c in self.cuts:
            if u >= c:
                lab += 1
        return lab


REGIME_FUNCTIONS = {"load_bucket": load_bucket, "problem_id": problem_id_regime}
