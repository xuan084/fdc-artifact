"""Vectorised (torch) candidate-policy families for the 6x6 E2 environment.

Same eight families as streams/policies.py; matchings are chosen greedily (repeated argmax) so the policy is a
cheap batched function of the observable state (t, loads, engaged). Policies see only observable state and public
problem information (affinity proxy `aff`, horizon). Budget B = 1 incentivised pair per round, level 1.
act(t, loads (B, P) long, eng (B, P) long) -> match (B, L, R) bool, inc (B, L, R) long
"""
from __future__ import annotations

import numpy as np
import torch

FAMILIES = ("greedy_affinity", "rotation", "high_load_rest", "front_incentive",
            "uniform_incentive", "late_incentive", "low_engagement_first", "random_legal")


def greedy_matching(score: torch.Tensor, size: int) -> torch.Tensor:
    """score (B, L, R) with -inf for forbidden pairs -> greedy matching (B, L, R) bool of at most `size` pairs."""
    B, Ln, Rn = score.shape
    s = score.clone()
    m = torch.zeros((B, Ln, Rn), dtype=torch.bool, device=score.device)
    ar = torch.arange(B, device=score.device)
    for _ in range(size):
        flat = s.view(B, -1)
        val, idx = flat.max(1)
        ok = torch.isfinite(val)
        i, j = idx // Rn, idx % Rn
        m[ar[ok], i[ok], j[ok]] = True
        s[ar[ok], i[ok], :] = -float("inf")
        s[ar[ok], :, j[ok]] = -float("inf")
    return m


def _greedy_np(aff: np.ndarray, size: int) -> np.ndarray:
    return greedy_matching(torch.as_tensor(aff, dtype=torch.float32)[None], size)[0].numpy()


class E2Policy:
    def __init__(self, family: str, params: dict, H: int, aff: np.ndarray, L: int = 6, R: int = 6, device="cuda"):
        self.family, self.params, self.H, self.L, self.R = family, dict(params), H, L, R
        self.m = min(L, R)
        self.aff_np = np.asarray(aff, float)
        self.dev = device
        self.aff = torch.as_tensor(self.aff_np, dtype=torch.float32, device=device)
        self._aff_match = torch.as_tensor(_greedy_np(self.aff_np, self.m), device=device)
        if family == "random_legal":
            r = np.random.default_rng(int(params["seed"]))
            seq = []
            for _ in range(H):
                k = int(r.integers(3, self.m + 1))
                lefts = r.choice(L, k, replace=False)
                rights = r.choice(R, k, replace=False)
                mt = np.zeros((L, R), bool); mt[lefts, rights] = True
                inc = np.zeros((L, R), np.int64)
                if r.random() < 0.5:
                    q = int(r.integers(k)); inc[lefts[q], rights[q]] = 1
                seq.append((mt, inc))
            self._seq = [(torch.as_tensor(a, device=device), torch.as_tensor(b, device=device)) for a, b in seq]

    @property
    def name(self):
        p = ",".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"{self.family}({p})"

    # -- helpers
    def _rot(self, t):
        off = int(self.params.get("offset", 0))
        mt = torch.zeros((self.L, self.R), dtype=torch.bool, device=self.dev)
        for i in range(self.m):
            mt[i, (i + t + off) % self.R] = True
        return mt

    def _top_inc(self, match: torch.Tensor) -> torch.Tensor:
        """Incentivise the matched pair with the highest affinity (per batch row)."""
        B = match.shape[0]
        s = torch.where(match, self.aff.expand_as(match), torch.full_like(match, -float("inf"), dtype=torch.float32))
        val, idx = s.view(B, -1).max(1)
        inc = torch.zeros((B, self.L * self.R), dtype=torch.long, device=self.dev)
        ok = torch.isfinite(val)
        inc[torch.arange(B, device=self.dev)[ok], idx[ok]] = 1
        return inc.view(B, self.L, self.R)

    def _first_inc(self, match: torch.Tensor) -> torch.Tensor:
        """Incentivise the matched pair with the smallest left index."""
        B = match.shape[0]
        flat = match.view(B, -1).float()
        idx = torch.argmax(flat, 1)
        inc = torch.zeros((B, self.L * self.R), dtype=torch.long, device=self.dev)
        ok = flat.sum(1) > 0
        inc[torch.arange(B, device=self.dev)[ok], idx[ok]] = 1
        return inc.view(B, self.L, self.R)

    def _nth_inc(self, match: torch.Tensor, t: int) -> torch.Tensor:
        """Incentivise the (t mod |matching|)-th matched pair in row-major order."""
        B = match.shape[0]
        flat = match.view(B, -1).long()
        cnt = flat.sum(1)
        k = torch.where(cnt > 0, torch.remainder(torch.full_like(cnt, t), cnt.clamp(min=1)), cnt)
        rank = torch.cumsum(flat, 1) - 1
        inc = (flat.bool() & (rank == k[:, None]) & (cnt[:, None] > 0)).long()
        return inc.view(B, self.L, self.R)

    def _base(self, t, B, base):
        mt = self._rot(t) if base == "rot" else self._aff_match
        return mt.expand(B, self.L, self.R).clone()

    def act(self, t: int, loads: torch.Tensor, eng: torch.Tensor):
        f, p, H, L = self.family, self.params, self.H, self.L
        B = loads.shape[0]
        zeros = torch.zeros((B, self.L, self.R), dtype=torch.long, device=self.dev)
        if f == "random_legal":
            mt, inc = self._seq[t % len(self._seq)]
            return mt.expand(B, L, self.R), inc.expand(B, L, self.R)
        if f == "greedy_affinity":
            mt = self._base(t, B, "aff")
            return mt, (self._top_inc(mt) if p.get("inc") == "top" else zeros)
        if f == "rotation":
            mt = self._base(t, B, "rot")
            return mt, (self._first_inc(mt) if p.get("inc") == "first" else zeros)
        if f == "high_load_rest":
            th = int(p["thresh"])
            el = loads < th
            score = self.aff.expand(B, L, self.R).clone()
            score = score.masked_fill(~(el[:, :L, None] & el[:, None, L:]), -float("inf"))
            mt = greedy_matching(score, self.m)
            return mt, (self._top_inc(mt) if p.get("inc") == "top" else zeros)
        if f in ("front_incentive", "late_incentive", "uniform_incentive"):
            mt = self._base(t, B, p.get("base", "aff"))
            if f == "front_incentive":
                return mt, (self._top_inc(mt) if t < p["frac"] * H else zeros)
            if f == "late_incentive":
                return mt, (self._top_inc(mt) if t >= (1 - p["frac"]) * H else zeros)
            return mt, self._nth_inc(mt, t)
        if f == "low_engagement_first":
            lf, ef = loads.float(), eng.float()
            score = -(lf[:, :L, None] + lf[:, None, L:]) + 0.5 * (ef[:, :L, None] + ef[:, None, L:]) + 0.01 * self.aff
            mt = greedy_matching(score, int(p["k"]))
            if not p.get("inc", True):
                return mt, zeros
            s2 = torch.where(mt, score, torch.full_like(score, -float("inf")))
            val, idx = s2.view(B, -1).max(1)
            inc = torch.zeros((B, L * self.R), dtype=torch.long, device=self.dev)
            ok = torch.isfinite(val)
            inc[torch.arange(B, device=self.dev)[ok], idx[ok]] = 1
            return mt, inc.view(B, L, self.R)
        raise ValueError(f)

    def act_np(self, t, loads, eng):
        mt, inc = self.act(t, torch.as_tensor(np.asarray(loads)[None], device=self.dev, dtype=torch.long),
                           torch.as_tensor(np.asarray(eng)[None], device=self.dev, dtype=torch.long))
        return mt[0].cpu().numpy(), inc[0].cpu().numpy()


def sample_e2_policy(family: str, rng: np.random.Generator, H: int, aff: np.ndarray, nmax: int = 2,
                     device="cuda") -> E2Policy:
    m = 6
    if family == "greedy_affinity":
        params = {"inc": str(rng.choice(["none", "top"]))}
    elif family == "rotation":
        params = {"offset": int(rng.integers(0, 6)), "inc": str(rng.choice(["none", "first"]))}
    elif family == "high_load_rest":
        params = {"thresh": int(rng.integers(1, nmax + 1)), "inc": str(rng.choice(["none", "top"]))}
    elif family in ("front_incentive", "late_incentive"):
        params = {"base": str(rng.choice(["aff", "rot"])), "frac": float(rng.choice([0.25, 0.5])),
                  "offset": int(rng.integers(0, 6))}
    elif family == "uniform_incentive":
        params = {"base": str(rng.choice(["aff", "rot"])), "offset": int(rng.integers(0, 6))}
    elif family == "low_engagement_first":
        params = {"k": int(rng.integers(3, m + 1)), "inc": bool(rng.integers(0, 2))}
    elif family == "random_legal":
        params = {"seed": int(rng.integers(0, 2**31 - 1))}
    else:
        raise ValueError(family)
    return E2Policy(family, params, H, aff, device=device)
