"""E2 ground truth (Tier 2): 6x6, continuous persistent parameters, same mechanism as E1-NL.

Holds theta* (the ONLY place). step() issues authenticated observations; `true_utilities` draws whole-trial
utilities on the platform (used by whole-trial baselines, counted as real env steps by the caller) and
`true_values` estimates J_theta*(pi) with many rollouts for evaluation (reports SE; not an exact truth).
"""
from __future__ import annotations

import numpy as np
import torch

from ..core import provenance
from ..models import e2_model as M


class E2Env:
    kind = "e2"

    def __init__(self, v_true: np.ndarray, seed: int, device="cuda"):
        self._v = np.asarray(v_true, float)
        self.L, self.R, self.P, self.nmax = M.L, M.R, M.P, M.NMAX
        self.rng = np.random.default_rng(seed)
        self._gen = torch.Generator(device=device)
        self._gen.manual_seed(int(seed) + 7)
        self.dev = device
        self.loads = np.zeros(self.P, dtype=np.int64)
        self.engaged = np.ones(self.P, dtype=np.int64)
        self.t = 0
        self.n_steps = 0

    def observable_state(self):
        return self.loads.copy(), self.engaged.copy()

    def step(self, match: np.ndarray, inc: np.ndarray):
        v, L = self._v, self.L
        a, g, tl = v[M.IA:M.IA + L], v[M.IG:M.IG + L], v[M.ITL:M.ITL + L]
        b, tr, psi, lam = v[M.IB:M.IB + self.R], v[M.ITR:M.ITR + self.R], v[M.IPSI], v[M.ILAM]
        tau = np.concatenate([tl, tr])
        yp = np.zeros(self.P, dtype=np.int64)
        outs = []
        for i, j in zip(*np.nonzero(match)):
            i, j = int(i), int(j)
            bb = int(inc[i, j])
            active = bool(self.engaged[i] and self.engaged[L + j])
            y = 0
            if active:
                lg = a[i] + b[j] - g[i] * self.loads[i] + psi * bb
                y = int(self.rng.random() < 1 / (1 + np.exp(-lg)))
                yp[i] = yp[L + j] = y
            outs.append((i, j, bb, y, int(active)))
        p_stay = 1 / (1 + np.exp(-(tau + M.C_KNOWN * yp - lam * self.loads)))
        p = np.where(self.engaged == 1, p_stay, M.RHO_RET)
        ne = (self.rng.random(self.P) < p).astype(np.int64)
        matched = np.concatenate([match.any(1), match.any(0)])
        nl = np.where(matched, np.minimum(self.loads + 1, self.nmax), np.maximum(self.loads - 1, 0)).astype(np.int64)
        obs = provenance.issue(env_kind=self.kind, t=self.t, loads=tuple(int(x) for x in self.loads),
                               engaged=tuple(int(x) for x in self.engaged), action=tuple(outs and [o[:3] for o in outs]),
                               outcomes=tuple(outs), next_loads=tuple(int(x) for x in nl),
                               next_engaged=tuple(int(x) for x in ne))
        self.loads, self.engaged = nl, ne
        self.t += 1
        self.n_steps += 1
        return obs

    # ---- platform whole trials (baselines; caller accounts H env steps per trial) ----
    def true_utilities(self, policy, loads0, eng0, H, n, w, w_ret, c_q) -> np.ndarray:
        out = []
        V = torch.tensor(self._v[None], device=self.dev, dtype=torch.float32)
        for s in range(0, n, 1 << 17):
            m = min(1 << 17, n - s)
            U = torch.rand((m, H, self.L, self.R), device=self.dev, generator=self._gen)
            E = torch.rand((m, H, self.P), device=self.dev, generator=self._gen)
            c = M.rollout_contrib(V, policy, loads0, eng0, H, U, E, w, w_ret, c_q)
            out.append(c.sum(2)[0].cpu().numpy())
        return np.concatenate(out).astype(float)

    # ---- evaluation only ----
    def true_values(self, policies, loads0, eng0, H, w, w_ret, c_q, n=1_000_000, seed=0):
        """MC estimate of J_theta*(pi) for each policy with n rollouts (CRN across policies); returns (J, SE)."""
        gen = torch.Generator(device=self.dev); gen.manual_seed(int(seed))
        V = torch.tensor(self._v[None], device=self.dev, dtype=torch.float32)
        K = len(policies)
        s1 = np.zeros(K); s2 = np.zeros(K); cnt = 0
        ch = 1 << 17
        for s in range(0, n, ch):
            m = min(ch, n - s)
            U = torch.rand((m, H, self.L, self.R), device=self.dev, generator=gen)
            E = torch.rand((m, H, self.P), device=self.dev, generator=gen)
            for k, pol in enumerate(policies):
                u = M.rollout_contrib(V, pol, loads0, eng0, H, U, E, w, w_ret, c_q).sum(2)[0].double()
                s1[k] += float(u.sum()); s2[k] += float((u * u).sum())
            cnt += m
        mu = s1 / cnt
        var = np.maximum(s2 / cnt - mu ** 2, 0.0)
        return mu, np.sqrt(var / cnt)

    def true_vector(self):
        return self._v.copy()
