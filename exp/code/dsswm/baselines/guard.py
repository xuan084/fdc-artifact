"""Guard(c): r3 JPC with a gap guard -- the minimax certificate threshold is tightened from eps to eps / c
(methodology 1.6; c in {1.5, 2, 3}, frozen on the dev block as the smallest c with FWER one-sided CP upper <= 0.05).
Everything else (learner class, SeqLRSet log 1/delta, DDA acquisition, T_max) is the r3 JPC. Scoring stays at eps.
Learner side: no truth access."""
from __future__ import annotations

from .ms_common import JPC, with_eps
from .switched_nl import solve_set_method

GUARD_CS = (1.5, 2.0, 3.0)


class Guard(JPC):
    def __init__(self, pub, c: float):
        super().__init__(pub)
        if c <= 1.0:
            raise ValueError("Guard needs c > 1")
        self.c = float(c)
        self.name = f"Guard_c{c:g}"

    def solve(self, k, sw, lr, handle, rng, tmax):
        res = solve_set_method(with_eps(self.nlpub(), self.pub.eps / self.c), k, sw, lr, handle, rng, tmax,
                               method="JPC")
        res["extra"]["guard_c"] = self.c
        res["extra"]["eps_cert"] = self.pub.eps / self.c
        return res
