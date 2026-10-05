"""Guarded public entry point for the checkpoint-strength FDC-BF certificate (``FDCBet``).

Usage contract (added in paper revision r6c; the lock-bound module ``fdc_bet.py`` is left byte-identical):

* ``FDCBet.certify`` computes fresh variance boxes and fresh Chernoff widths every time it is called. Its ledger
  ``beta = ln(M K / delta_main)`` and its box level ``delta_var / (2 S A K)`` count exactly the K predeclared
  checkpoints ``ctx.checkpoints``. The legacy method checks only that times do not decrease, so calling it at any
  other time would issue certificates on events the K-event ledger never paid for.
* ``FDCBetGuarded`` is the same certificate with the missing guard: it certifies only at the predeclared checkpoints,
  in strictly increasing order, and raises ``OffGridCertifyError`` for any other time. On the checkpoint grid its
  output is identical to ``FDCBet``'s.
* Arbitrary increasing monitoring times do not authorise fresh widths. Dense (off-grid) monitoring must use
  ``dsswm.baselines.fdc_loc.FDCTimeUniform`` (TU-FDC, Remark 4.2 of the paper): it refreshes boxes and widths only at
  block points and reuses the last block's widths in between, which is what the time-uniform proof covers.
* The implicit-class certificates already carry this guard (``fdc_dp.FDCDP.certify``, ``rect_dp.RectBFDP.certify``;
  dense monitoring via ``FDCDPTimeUniform`` / ``RectBFDPTU``).

New code should construct FDC-BF through ``make_guarded_variant`` (or ``FDCBetGuarded``) rather than ``FDCBet``.
"""
from __future__ import annotations

from .fdc_bet import VARIANTS, FDCBet

__all__ = ["FDCBetGuarded", "OffGridCertifyError", "make_guarded_variant", "DENSE_MONITORING_CLASS"]

DENSE_MONITORING_CLASS = "dsswm.baselines.fdc_loc.FDCTimeUniform"


class OffGridCertifyError(RuntimeError):
    """Raised when a fresh-width certificate is requested outside the predeclared checkpoint grid."""


class FDCBetGuarded(FDCBet):
    """``FDCBet`` that refuses fresh-width certification off the predeclared checkpoint grid."""

    def setup(self, ctx):
        grid = [int(x) for x in ctx.checkpoints]
        if any(b <= a for a, b in zip(grid[:-1], grid[1:])):
            raise ValueError("FDC-BF (guarded): checkpoints must be strictly increasing")
        super().setup(ctx)
        if int(self.ledger["K"]) != len(grid):
            raise ValueError("FDC-BF (guarded): ledger K differs from the number of predeclared checkpoints")
        self._grid_set = set(grid)
        self._last_ck = None

    def certify(self, ctx, st):
        t = int(st.t)
        if t not in self._grid_set:
            raise OffGridCertifyError(
                f"FDC-BF: fresh-width certify at t={t} is off the predeclared checkpoint grid; the K-event ledger "
                f"does not cover it. Use {DENSE_MONITORING_CLASS} (widths frozen at the last block point) for "
                "dense monitoring.")
        if self._last_ck is not None and t <= self._last_ck:
            raise OffGridCertifyError("FDC-BF: checkpoints must be visited in strictly increasing order")
        self._last_ck = t
        return super().certify(ctx, st)

    def describe(self):
        d = super().describe()
        d.update({"entry_point": "guarded (fresh widths only at the K predeclared checkpoints)",
                  "dense_monitoring": DENSE_MONITORING_CLASS})
        return d


def make_guarded_variant(name="FDC-BF"):
    """Guarded counterpart of ``fdc_bet.make_variant``: same configuration and name, same certificates on the grid."""
    kind, box, fpc, rect, split = VARIANTS[name]
    return FDCBetGuarded(kind=kind, var_box=box, fpc=fpc, rect=rect, split=split, name=name)
