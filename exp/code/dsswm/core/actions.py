"""Legal action space: partial matchings x per-pair incentive with budget B."""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np


def enumerate_matchings(L: int, R: int) -> list[tuple[tuple[int, int], ...]]:
    """All partial matchings of the complete L x R bipartite graph (including the empty one)."""
    out = []
    for k in range(0, min(L, R) + 1):
        for lefts in itertools.combinations(range(L), k):
            for rights in itertools.permutations(range(R), k):
                out.append(tuple(sorted(zip(lefts, rights))))
    return out


@dataclass(frozen=True)
class Action:
    pairs: tuple[tuple[int, int], ...]
    incentives: tuple[tuple[int, int, int], ...]  # (i, j, level) for incentivised matched pairs


class ActionSpace:
    """Enumerates legal actions. Budget B = max number of incentivised pairs per round.

    incentive_levels: allowed non-zero incentive levels (e.g. (1,) or (1, 2) for Type-2 problems).
    """

    def __init__(self, L: int, R: int, budget: int = 1, incentive_levels: tuple[int, ...] = (1,)):
        self.L, self.R, self.budget = L, R, budget
        self.incentive_levels = tuple(incentive_levels)
        self.nb = 1 + max(self.incentive_levels)  # number of incentive levels incl. 0
        acts: list[Action] = []
        for m in enumerate_matchings(L, R):
            acts.append(Action(m, ()))
            for k in range(1, min(budget, len(m)) + 1):
                for sub in itertools.combinations(m, k):
                    for lv in itertools.product(self.incentive_levels, repeat=k):
                        acts.append(Action(m, tuple((i, j, l) for (i, j), l in zip(sub, lv))))
        self.actions = acts
        self.n = len(acts)
        self.match = np.zeros((self.n, L, R), dtype=np.int8)
        self.inc = np.zeros((self.n, L, R), dtype=np.int8)
        self._index: dict[tuple, int] = {}
        for a_idx, a in enumerate(acts):
            for i, j in a.pairs:
                self.match[a_idx, i, j] = 1
            for i, j, l in a.incentives:
                self.inc[a_idx, i, j] = l
            self._index[(a.pairs, a.incentives)] = a_idx

    def index(self, pairs, incentives=()) -> int:
        pairs = tuple(sorted(tuple(p) for p in pairs))
        incentives = tuple(sorted(tuple(x) for x in incentives))
        return self._index[(pairs, incentives)]

    def matched_left(self, a_idx: int) -> np.ndarray:
        return self.match[a_idx].any(axis=1)

    def matched_right(self, a_idx: int) -> np.ndarray:
        return self.match[a_idx].any(axis=0)

    def max_level(self, a_idx: int) -> int:
        return int(self.inc[a_idx].max()) if self.n else 0
