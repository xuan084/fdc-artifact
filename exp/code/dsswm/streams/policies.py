"""Eight candidate-policy families. Policies see only the observable state (t, loads, engaged)
and public problem information (affinity proxy, horizon). They never see true parameters."""
from __future__ import annotations

import numpy as np

from ..core.actions import ActionSpace

FAMILIES = ("greedy_affinity", "rotation", "high_load_rest", "front_incentive",
            "uniform_incentive", "late_incentive", "low_engagement_first", "random_legal")


class _MatchingTable:
    def __init__(self, aspace: ActionSpace):
        seen = {}
        for a in aspace.actions:
            seen.setdefault(a.pairs, None)
        self.matchings = list(seen.keys())
        self.sizes = np.array([len(m) for m in self.matchings])

    def best(self, score: np.ndarray, size: int, eligible_l=None, eligible_r=None):
        """Highest total score matching of the largest feasible size <= `size` among eligible participants."""
        for k in range(size, -1, -1):
            best, best_val = None, -np.inf
            for m in self.matchings:
                if len(m) != k:
                    continue
                if eligible_l is not None and any(not eligible_l[i] for i, _ in m):
                    continue
                if eligible_r is not None and any(not eligible_r[j] for _, j in m):
                    continue
                v = sum(score[i, j] for i, j in m)
                if v > best_val + 1e-12:
                    best, best_val = m, v
            if best is not None:
                return best
        return ()


class Policy:
    def __init__(self, family: str, params: dict, aspace: ActionSpace, H: int, aff: np.ndarray, rng_seed: int):
        self.family, self.params, self.aspace, self.H, self.aff = family, dict(params), aspace, H, aff
        self.L, self.R = aspace.L, aspace.R
        self.m = min(self.L, self.R)
        self._mt = _MatchingTable(aspace)
        self.level = int(params.get("level", 1))
        if family == "random_legal":
            r = np.random.default_rng(rng_seed)
            allowed = [k for k in range(aspace.n) if aspace.max_level(k) <= self.level and
                       (self.level == 1 or aspace.max_level(k) != 1 or not params.get("only_b2", False))]
            self._seq = [int(r.choice(allowed)) for _ in range(H)]
        self._cache: dict = {}

    @property
    def name(self):
        p = ",".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"{self.family}({p})"

    def uses_level(self, lv: int) -> bool:
        return self.level >= lv and self.family != "greedy_affinity" and not (
            self.family in ("rotation", "high_load_rest") and self.params.get("inc", "none") == "none")

    # --- helpers
    def _base_matching(self, t, loads, engaged, base):
        if base == "rot":
            off = self.params.get("offset", 0)
            return tuple(sorted((i, (i + t + off) % self.R) for i in range(self.m)))
        return self._mt.best(self.aff, self.m)

    def _top_pair(self, pairs):
        if not pairs:
            return None
        return max(pairs, key=lambda p: self.aff[p[0], p[1]])

    def _idx(self, pairs, inc_pair=None):
        inc = () if inc_pair is None else ((inc_pair[0], inc_pair[1], self.level),)
        return self.aspace.index(pairs, inc)

    def act(self, t: int, loads, engaged) -> int:
        key = (t, tuple(int(x) for x in loads), tuple(int(x) for x in engaged))
        if key in self._cache:
            return self._cache[key]
        a = self._act(t, np.asarray(loads), np.asarray(engaged))
        self._cache[key] = a
        return a

    def _act(self, t, loads, engaged) -> int:
        f, p, H, L = self.family, self.params, self.H, self.L
        if f == "random_legal":
            return self._seq[t % len(self._seq)]
        if f == "greedy_affinity":
            pairs = self._mt.best(self.aff, self.m)
            return self._idx(pairs, self._top_pair(pairs) if p.get("inc") == "top" else None)
        if f == "rotation":
            pairs = self._base_matching(t, loads, engaged, "rot")
            inc = None
            if p.get("inc") == "first" and pairs:
                inc = pairs[0]
            return self._idx(pairs, inc)
        if f == "high_load_rest":
            th = p["thresh"]
            el_l = loads[:L] < th
            el_r = loads[L:] < th
            pairs = self._mt.best(self.aff, self.m, el_l, el_r)
            return self._idx(pairs, self._top_pair(pairs) if p.get("inc") == "top" else None)
        if f in ("front_incentive", "late_incentive", "uniform_incentive"):
            pairs = self._base_matching(t, loads, engaged, p.get("base", "aff"))
            inc = None
            if pairs:
                if f == "front_incentive" and t < p["frac"] * H:
                    inc = self._top_pair(pairs)
                elif f == "late_incentive" and t >= (1 - p["frac"]) * H:
                    inc = self._top_pair(pairs)
                elif f == "uniform_incentive":
                    inc = pairs[t % len(pairs)]
            return self._idx(pairs, inc)
        if f == "low_engagement_first":
            score = -(loads[:L, None] + loads[None, L:]).astype(float) + 0.5 * (engaged[:L, None] + engaged[None, L:]) \
                    + 0.01 * self.aff
            pairs = self._mt.best(score, p["k"])
            inc = None
            if pairs and p.get("inc", True):
                inc = max(pairs, key=lambda q: score[q[0], q[1]])
            return self._idx(pairs, inc)
        raise ValueError(f)


def sample_policy(family: str, rng: np.random.Generator, aspace: ActionSpace, H: int, aff: np.ndarray,
                  nmax: int, level: int = 1) -> Policy:
    m = min(aspace.L, aspace.R)
    if family == "greedy_affinity":
        params = {"inc": str(rng.choice(["none", "top"]))}
    elif family == "rotation":
        params = {"offset": int(rng.integers(0, aspace.R)), "inc": str(rng.choice(["none", "first"]))}
    elif family == "high_load_rest":
        params = {"thresh": int(rng.integers(1, nmax + 1)), "inc": str(rng.choice(["none", "top"]))}
    elif family in ("front_incentive", "late_incentive"):
        params = {"base": str(rng.choice(["aff", "rot"])), "frac": float(rng.choice([0.25, 0.5])),
                  "offset": int(rng.integers(0, aspace.R))}
    elif family == "uniform_incentive":
        params = {"base": str(rng.choice(["aff", "rot"])), "offset": int(rng.integers(0, aspace.R))}
    elif family == "low_engagement_first":
        params = {"k": int(rng.integers(1, m + 1)), "inc": bool(rng.integers(0, 2))}
    elif family == "random_legal":
        params = {"seed": int(rng.integers(0, 2**31 - 1))}
    else:
        raise ValueError(family)
    if level != 1:
        params["level"] = level
    return Policy(family, params, aspace, H, aff, rng_seed=int(params.get("seed", rng.integers(0, 2**31 - 1))))
