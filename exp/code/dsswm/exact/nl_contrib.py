"""Per-participant attribution of the exact E1-NL value J_theta(pi) (GPU, float64).

J_theta(pi) = c_q * ( sum_t w_t sum_{(i,j) active} E[y_ij,t] + w_ret sum_p E[e_p(H)] )
            = sum_p C_p(theta, pi),
C_p = c_q * ( 0.5 * sum_t w_t sum_{pairs containing p} E[y_pair,t] + w_ret * P(e_p(H) = 1) ).
Each pair outcome is split half/half between its two members, so sum_p C_p == J exactly.
This is a learner-side computation over the model class (it uses only the propagator and class params).
"""
from __future__ import annotations

import numpy as np
import torch

from .nl_propagate import NLPropagator


def _participant_plan(prop: NLPropagator, policy, loads0, engaged0, H: int):
    """Same reachable-state plan as NLPropagator.build_policy_plan, plus per-participant EY gather indices."""
    dev = prop.device
    L, P = prop.L, prop.P
    s0 = prop.codec.encode(loads0, engaged0)
    states = np.array([s0], dtype=np.int64)
    plan = []
    for t in range(H):
        fl, cl, src, dst_codes = [], [], [], []
        eyp = []                     # per state: (P, L*R) gather idx, padded with ey_pad
        for k, code in enumerate(states):
            ld, en = prop.codec.decode(int(code))
            a_idx = policy.act(t, ld, en)
            fidx, const, nxt, _ = prop._struct(int(code), a_idx)
            fl.append(fidx)
            cl.append(const)
            src.append(np.full(len(nxt), k, dtype=np.int64))
            dst_codes.append(nxt)
            a = prop.aspace.actions[a_idx]
            inc = {(i, j): l for i, j, l in a.incentives}
            m = np.full((P, max(prop.L * prop.R, 1)), prop.ey_pad, dtype=np.int64)
            cnt = np.zeros(P, dtype=np.int64)
            for (i, j) in a.pairs:
                if en[i] and en[L + j]:
                    pi = prop.pair_idx(i, j, ld[i], inc.get((i, j), 0))
                    for p in (i, L + j):
                        m[p, cnt[p]] = pi
                        cnt[p] += 1
            eyp.append(m)
        F = max(f.shape[1] for f in fl)
        fl = [np.pad(f, ((0, 0), (0, F - f.shape[1])), constant_values=prop.pad) for f in fl]
        dst_codes = np.concatenate(dst_codes)
        nxt_states, dst = np.unique(dst_codes, return_inverse=True)
        plan.append({
            "fidx": torch.as_tensor(np.concatenate(fl), device=dev),
            "const": torch.as_tensor(np.concatenate(cl), device=dev, dtype=prop.dtype),
            "src": torch.as_tensor(np.concatenate(src), device=dev),
            "dst": torch.as_tensor(dst, device=dev),
            "eyp": torch.as_tensor(np.stack(eyp), device=dev),          # (S_t, P, LR)
            "n_next": len(nxt_states),
        })
        states = nxt_states
    ebits = np.stack([prop.codec.decode(int(c))[1] for c in states]).astype(float)   # (S_H, P)
    return plan, torch.as_tensor(ebits, device=dev, dtype=prop.dtype)


def contrib_table(prop: NLPropagator, params: dict, policies, loads0, engaged0, H: int, utility,
                  chunk: int = 4096) -> np.ndarray:
    """(B, n_policies, P) per-participant contributions; sum over the last axis equals prop.j_table(...)."""
    LT, EY = prop.tables(params)
    B = LT.shape[0]
    w = torch.as_tensor(np.asarray(utility.w, float), device=LT.device, dtype=LT.dtype)
    out = torch.zeros(B, len(policies), prop.P, device=LT.device, dtype=LT.dtype)
    for k, pi in enumerate(policies):
        plan, ebits = _participant_plan(prop, pi, loads0, engaged0, H)
        for s in range(0, B, chunk):
            lt, ey = LT[s:s + chunk], EY[s:s + chunk]
            b = lt.shape[0]
            Pt = torch.ones(b, 1, device=LT.device, dtype=LT.dtype)
            val = torch.zeros(b, prop.P, device=LT.device, dtype=LT.dtype)
            for t, st in enumerate(plan):
                eyp = ey[:, st["eyp"]].sum(-1)                         # (b, S_t, P)
                val = val + 0.5 * w[t] * (Pt[:, :, None] * eyp).sum(1)
                logp = lt[:, st["fidx"]].sum(-1) + st["const"][None]
                W = Pt[:, st["src"]] * torch.exp(logp)
                Pn = torch.zeros(b, st["n_next"], device=LT.device, dtype=LT.dtype)
                Pn.index_add_(1, st["dst"], W)
                Pt = Pn
            val = val + float(utility.w_ret) * (Pt @ ebits)
            out[s:s + chunk, k] = float(utility.c_q) * val
    if LT.is_cuda:
        torch.cuda.synchronize()
    return out.cpu().numpy()
