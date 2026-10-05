"""B4: uniform A/B (pool-proportional allocation, i.e. the RCT log read in random order) + radius-sum certificate.

Industrial baseline: the allocation is the same pre-generated pool-proportional schedule as QFC; the certificate is the
per-cell WSR20 time-uniform WoR CS rectangle (radius sum) of ``frontier_common.rect_certificate`` -- rigorous.
"""
from __future__ import annotations

from .frontier_common import Method, rect_certificate

__all__ = ["UniformRS"]


class UniformRS(Method):
    name = "B4"
    alloc_kind = "pool"
    validity = "rigorous"

    def certify(self, ctx, st):
        return rect_certificate(ctx, st)
