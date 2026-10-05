"""cand_g: versioned joint evidence (VJE) and detect-reset, for an in-class change at a KNOWN epoch boundary.

Hypothesis space (k = 1): pairs (theta_old, theta_new) in Theta x Theta where theta_new equals theta_old except
possibly on the parameter tuple of ONE participant (left tuple (alpha, gamma, tauL) or right tuple (beta, tauR));
global parameters (psi, lam) are shared across epochs. On the enumerated class this is the (B, V) array
    joint[b, v] = cum_old[b] + cum_new[newidx[b, v]],   V = 1 + sum_s radix_s   (~32 x |Theta| on E1-NL-S)
where column 0 is 'no change' and column (s, d) replaces participant slot s by tuple d.
Numerator: old epoch = in-class plug-in (as SeqLRSet); new epoch = plug-in at the theta_new component of the
joint MLE (predictable). Under the true versioned pair the likelihood ratio is a non-negative martingale over the
whole sequence, so P(true pair ever excluded) <= delta (Ville). Decisions after the boundary use the projection
Theta_new = {theta_new : some alive (theta_old, v) maps to it}.

DetectReset: run the mixed falsification set (dsswm.evidence.mixed_lr); when it becomes empty (MODEL_CONFLICT)
the evidence is cleared and a fresh set starts from the next real round.
Naive reuse is a plain SeqLRSet over all rounds (dsswm.evidence.lr_set).
"""
from __future__ import annotations

import math

import numpy as np
import torch

from ..core.provenance import require_authentic


class VersionedJointLR:
    def __init__(self, ncl, prop, LT: torch.Tensor, delta: float, k: int = 1):
        if k != 1:
            raise NotImplementedError("only k = 1 changed participant is registered")
        self.prop, self.LT, self.delta = prop, LT, float(delta)
        self.B = LT.shape[0]
        radices = list(ncl._radices)
        S = len(radices)
        strides = np.ones(S, dtype=np.int64)
        for s in range(S - 2, -1, -1):
            strides[s] = strides[s + 1] * radices[s + 1]
        idx = np.arange(self.B, dtype=np.int64)
        digits = np.stack([(idx // strides[s]) % radices[s] for s in range(S)], 1)
        cols, slot_of, dig_of = [idx], [-1], [-1]
        for s in range(S - 1):                       # participant slots only (last slot = global params)
            for d in range(radices[s]):
                cols.append(idx + (d - digits[:, s]) * strides[s])
                slot_of.append(s)
                dig_of.append(d)
        self.newidx_np = np.stack(cols, 1)
        self.newidx = torch.as_tensor(self.newidx_np, device=LT.device)
        self.slot_of = np.array(slot_of)
        self.dig_of = np.array(dig_of)
        self.digits = digits
        self.V = self.newidx_np.shape[1]
        self.n_slots = S - 1
        self.cum_old = torch.zeros(self.B, device=LT.device, dtype=LT.dtype)
        self.cum_new = torch.zeros(self.B, device=LT.device, dtype=LT.dtype)
        self.LN_old = 0.0
        self.LN_new = 0.0
        self.epoch = 0
        self.n_old = 0
        self.n_new = 0

    def _ll(self, obs):
        require_authentic(obs)
        return self.prop.loglik(self.LT, [obs])[:, 0]

    def joint(self) -> torch.Tensor:
        return self.cum_old[:, None] + self.cum_new[self.newidx]

    def update(self, obs) -> None:
        ll = self._ll(obs)
        if self.epoch == 0:
            if self.n_old == 0:
                self.LN_old += float(torch.logsumexp(ll, 0) - math.log(self.B))
            else:
                self.LN_old += float(ll[int(torch.argmax(self.cum_old))])
            self.cum_old = self.cum_old + ll
            self.n_old += 1
        else:
            Jt = self.joint()
            flat = int(torch.argmax(Jt))
            th_new = int(self.newidx_np.reshape(-1)[flat])
            self.LN_new += float(ll[th_new])
            self.cum_new = self.cum_new + ll
            self.n_new += 1

    def start_new_epoch(self) -> None:
        assert self.epoch == 0
        self.epoch = 1

    def log_num(self) -> float:
        return self.LN_old + self.LN_new

    def joint_mask(self) -> torch.Tensor:
        return (self.log_num() - self.joint()) < math.log(1.0 / self.delta)

    def new_mask(self) -> torch.Tensor:
        jm = self.joint_mask()
        out = torch.zeros(self.B, dtype=torch.bool, device=self.LT.device)
        out[self.newidx[jm]] = True
        return out

    def old_mask(self) -> torch.Tensor:
        return self.joint_mask().any(1)

    def size(self) -> int:
        return int(self.new_mask().sum())

    def changed_slots_alive(self) -> list[int]:
        """Participant slots s for which some alive hypothesis changes slot s (affected participants)."""
        jm = self.joint_mask().cpu().numpy()
        alive_cols = np.flatnonzero(jm.any(0))
        out = set()
        for c in alive_cols:
            s = int(self.slot_of[c])
            if s < 0:
                continue
            rows = np.flatnonzero(jm[:, c])
            if (self.digits[rows, s] != self.dig_of[c]).any():
                out.add(s)
        return sorted(out)

    def contains(self, theta_old_idx: int, theta_new_idx: int) -> bool:
        jm = self.joint_mask()[theta_old_idx].cpu().numpy()
        return bool(np.any(jm & (self.newidx_np[theta_old_idx] == theta_new_idx)))


class DetectReset:
    """Mixed falsification set with clear-on-alarm. factory() -> fresh MixedLRSet."""

    def __init__(self, factory):
        self.factory = factory
        self.cur = factory()
        self.alarms: list[int] = []
        self.n_rounds = 0

    def update(self, obs, problem_id=None) -> bool:
        self.cur.update(obs, problem_id=problem_id)
        self.n_rounds += 1
        if self.cur.is_conflict():
            self.alarms.append(self.n_rounds)
            self.cur = self.factory()
            return True
        return False

    def mask(self) -> torch.Tensor:
        return self.cur.mask()

    def size(self) -> int:
        return self.cur.size()
