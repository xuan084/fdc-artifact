"""Exact distribution propagation for E1-NL (GPU, float64).

For a deterministic policy pi(t, loads, engaged) the state distribution P_theta(s_t) is propagated exactly
for every theta in a batch simultaneously. Transitions factorise into
  * active pairs (matched, both engaged): joint (e_i', e_j') after marginalising the shared outcome y_ij,
  * other engaged participants: retention with y = 0,
  * disengaged participants: known return probability rho_ret (constant),
so one (state, action) has 2^P successor rows. Log-probabilities are gathered from per-theta tables.

params (torch, leading batch dim B): alpha (B,L) beta (B,R) gamma (B,L) tauL (B,L) tauR (B,R)
psi_left (B,L,nb) lam (B,), optional syn (B,L,R), g (N,) (fatigue transform, default n) and g_ret (N,)
(load transform in retention, default n).
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch

from ..core.actions import ActionSpace
from ..core.state import StateCodec


class NLPropagator:
    def __init__(self, L: int, R: int, nmax: int, aspace: ActionSpace, c: float, rho_ret: float,
                 device="cuda", dtype=torch.float64, static: bool = False, mem_budget_elems: int = 1.5e8):
        self.L, self.R, self.P, self.nmax, self.N = L, R, L + R, nmax, nmax + 1
        self.aspace, self.c, self.rho_ret = aspace, float(c), float(rho_ret)
        self.nb = aspace.nb
        self.codec = StateCodec(L, R, nmax, engagement=True)
        self.device, self.dtype, self.static = device, dtype, static
        self.mem_budget_elems = int(mem_budget_elems)
        N, nb, P = self.N, self.nb, self.P
        self.U_pair = L * R * N * nb
        self.U_re = P * 2 * N * 2
        self.U_grp = L * R * N * N * nb * 4
        self.off_re = 2 * self.U_pair
        self.off_grp = self.off_re + self.U_re
        self.U = self.off_grp + self.U_grp
        self.pad = self.U
        self.ey_pad = self.U_pair
        self._ebits = ((np.arange(2 ** P)[:, None] >> np.arange(P)[None]) & 1).astype(np.int64)
        self._pow2 = 2 ** np.arange(P)
        self._struct_cache: dict = {}
        self.log_rho = math.log(self.rho_ret)
        self.log_1mrho = math.log1p(-self.rho_ret)

    # ---------------- index helpers ----------------
    def pair_idx(self, i, j, n, b):
        return ((i * self.R + j) * self.N + n) * self.nb + b

    def py_idx(self, y, i, j, n, b):
        return (0 if y == 1 else self.U_pair) + self.pair_idx(i, j, n, b)

    def re_idx(self, p, y, n, e):
        return self.off_re + ((p * 2 + y) * self.N + n) * 2 + e

    def grp_base(self, i, j, ni, nj, b):
        return self.off_grp + ((((i * self.R + j) * self.N + ni) * self.N + nj) * self.nb + b) * 4

    # ---------------- per-theta tables ----------------
    def tables(self, params: dict):
        """Returns LT (B, U+1) log-prob table (last column 0) and EY (B, U_pair+1) mean-outcome table."""
        L, R, N, nb = self.L, self.R, self.N, self.nb
        alpha, beta, gamma = params["alpha"], params["beta"], params["gamma"]
        B = alpha.shape[0]
        dev, dt = alpha.device, alpha.dtype
        g = params.get("g")
        g = torch.arange(N, device=dev, dtype=dt) if g is None else torch.as_tensor(g, device=dev, dtype=dt)
        psi_left = params["psi_left"]
        if psi_left.shape[-1] < nb:
            raise ValueError("model class does not cover the incentive levels of this action space")
        psi_left = psi_left[..., :nb]
        syn = params.get("syn")
        PL = (alpha[:, :, None, None, None] + beta[:, None, :, None, None]
              - gamma[:, :, None, None, None] * g[None, None, None, :, None]
              + psi_left[:, :, None, None, :])
        if syn is not None:
            PL = PL + torch.as_tensor(syn, device=dev, dtype=dt).reshape(-1, L, R)[:, :, :, None, None]
        if self.static:
            pass
        lpy1 = torch.nn.functional.logsigmoid(PL)
        lpy0 = torch.nn.functional.logsigmoid(-PL)
        tau = torch.cat([params["tauL"], params["tauR"]], 1)                      # (B,P)
        yv = torch.arange(2, device=dev, dtype=dt)
        g_ret = params.get("g_ret")      # optional load transform in retention (default n); static classes use const
        nv = torch.arange(N, device=dev, dtype=dt) if g_ret is None else torch.as_tensor(g_ret, device=dev, dtype=dt)
        lam = params["lam"].reshape(-1)
        RL = tau[:, :, None, None] + self.c * yv[None, None, :, None] - lam[:, None, None, None] * nv[None, None, None, :]
        lre = torch.stack([torch.nn.functional.logsigmoid(-RL), torch.nn.functional.logsigmoid(RL)], -1)  # (B,P,y,N,e)
        lpy = torch.stack([lpy0, lpy1], -1)                                         # (B,L,R,N,nb,y)
        # group table [B, i, j, ni, nj, b, y, ei, ej]
        A = lpy[:, :, :, :, None, :, :, None, None]
        Ei = lre[:, :L].permute(0, 1, 3, 2, 4)[:, :, None, :, None, None, :, :, None]
        Ej = lre[:, L:].permute(0, 1, 3, 2, 4)[:, None, :, None, :, None, :, None, :]
        G = torch.logsumexp(A + Ei + Ej, dim=6)                                     # (B,L,R,Ni,Nj,nb,ei,ej)
        LT = torch.cat([lpy1.reshape(B, -1), lpy0.reshape(B, -1), lre.reshape(B, -1), G.reshape(B, -1),
                        torch.zeros(B, 1, device=dev, dtype=dt)], 1)
        EY = torch.cat([torch.sigmoid(PL).reshape(B, -1), torch.zeros(B, 1, device=dev, dtype=dt)], 1)
        assert LT.shape[1] == self.U + 1
        return LT, EY

    # ---------------- (state, action) structure ----------------
    def _struct(self, code: int, a_idx: int):
        key = (code, a_idx)
        s = self._struct_cache.get(key)
        if s is not None:
            return s
        L, P = self.L, self.P
        loads, eng = self.codec.decode(code)
        a = self.aspace.actions[a_idx]
        inc = {(i, j): l for i, j, l in a.incentives}
        E = self._ebits
        nrow = E.shape[0]
        fidx = np.full((nrow, P), self.pad, dtype=np.int64)
        const = np.zeros(nrow)
        eyidx = np.full(P, self.ey_pad, dtype=np.int64)
        grouped = np.zeros(P, dtype=bool)
        col = 0
        for k, (i, j) in enumerate(a.pairs):
            b = inc.get((i, j), 0)
            if eng[i] and eng[L + j]:
                fidx[:, col] = self.grp_base(i, j, loads[i], loads[L + j], b) + E[:, i] * 2 + E[:, L + j]
                eyidx[col] = self.pair_idx(i, j, loads[i], b)
                grouped[i] = grouped[L + j] = True
                col += 1
        for p in range(P):
            if grouped[p]:
                continue
            if eng[p]:
                fidx[:, col] = self.re_idx(p, 0, loads[p], 0) + E[:, p]
                col += 1
            else:
                const += np.where(E[:, p] == 1, self.log_rho, self.log_1mrho)
        if self.static:
            nl = loads.copy()
        else:
            matched = np.zeros(P, dtype=bool)
            for i, j in a.pairs:
                matched[i] = matched[L + j] = True
            nl = np.where(matched, np.minimum(loads + 1, self.nmax), np.maximum(loads - 1, 0))
        lc = int(np.dot(nl, self.codec._pow_n))
        nxt = lc + self.codec.n_load_codes * (E @ self._pow2)
        s = (fidx[:, :max(col, 1)], const, nxt, eyidx[:max(col, 1)])
        self._struct_cache[key] = s
        return s

    def build_policy_plan(self, policy, loads0, engaged0, H: int):
        """Reachable-state propagation plan for one policy (theta-independent)."""
        dev = self.device
        s0 = self.codec.encode(loads0, engaged0)
        states = np.array([s0], dtype=np.int64)
        plan = []
        for t in range(H):
            fl, cl, src, dst_codes, eyl = [], [], [], [], []
            for k, code in enumerate(states):
                ld, en = self.codec.decode(int(code))
                a_idx = policy.act(t, ld, en)
                fidx, const, nxt, ey = self._struct(int(code), a_idx)
                fl.append(fidx)
                cl.append(const)
                src.append(np.full(len(nxt), k, dtype=np.int64))
                dst_codes.append(nxt)
                eyl.append(ey)
            F = max(f.shape[1] for f in fl)
            fl = [np.pad(f, ((0, 0), (0, F - f.shape[1])), constant_values=self.pad) for f in fl]
            Fy = max(len(e) for e in eyl)
            eyl = [np.pad(e, (0, Fy - len(e)), constant_values=self.ey_pad) for e in eyl]
            dst_codes = np.concatenate(dst_codes)
            nxt_states, dst = np.unique(dst_codes, return_inverse=True)
            plan.append({
                "fidx": torch.as_tensor(np.concatenate(fl), device=dev),
                "const": torch.as_tensor(np.concatenate(cl), device=dev, dtype=self.dtype),
                "src": torch.as_tensor(np.concatenate(src), device=dev),
                "dst": torch.as_tensor(dst, device=dev),
                "eyidx": torch.as_tensor(np.stack(eyl), device=dev),
                "n_next": len(nxt_states),
            })
            states = nxt_states
        esum = np.array([self.codec.decode(int(c))[1].sum() for c in states], dtype=float)
        return plan, torch.as_tensor(esum, device=dev, dtype=self.dtype)

    def _run_plan(self, plan, esum, LT, EY, w, w_ret, c_q):
        """LT, EY for one theta chunk -> J (b,) plus per-step expected outcomes."""
        b = LT.shape[0]
        Pt = torch.ones(b, 1, device=LT.device, dtype=LT.dtype)
        val = torch.zeros(b, device=LT.device, dtype=LT.dtype)
        for t, st in enumerate(plan):
            ey = EY[:, st["eyidx"]].sum(-1)                         # (b, S_t)
            val = val + w[t] * (Pt * ey).sum(1)
            logp = LT[:, st["fidx"]].sum(-1) + st["const"][None]    # (b, rows)
            W = Pt[:, st["src"]] * torch.exp(logp)
            Pn = torch.zeros(b, st["n_next"], device=LT.device, dtype=LT.dtype)
            Pn.index_add_(1, st["dst"], W)
            Pt = Pn
        val = val + w_ret * (Pt * esum[None]).sum(1)
        return c_q * val

    def j_table(self, params: dict, policies, loads0, engaged0, H: int, utility, chunk: int | None = None,
                timings: dict | None = None) -> np.ndarray:
        """Exact J (B, n_policies) for every theta in the batch."""
        t0 = time.perf_counter()
        LT, EY = self.tables(params)
        B = LT.shape[0]
        plans = [self.build_policy_plan(pi, loads0, engaged0, H) for pi in policies]
        t1 = time.perf_counter()
        out = torch.zeros(B, len(policies), device=LT.device, dtype=LT.dtype)
        for k, (plan, esum) in enumerate(plans):
            max_rows = max(int(st["fidx"].shape[0] * st["fidx"].shape[1]) for st in plan)
            ch = chunk or max(64, int(self.mem_budget_elems // max(max_rows, 1)))
            for s in range(0, B, ch):
                out[s:s + ch, k] = self._run_plan(plan, esum, LT[s:s + ch], EY[s:s + ch],
                                                  utility.w, utility.w_ret, utility.c_q)
        if LT.is_cuda:
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        if timings is not None:
            timings["plan_s"] = timings.get("plan_s", 0.0) + (t1 - t0)
            timings["gpu_s"] = timings.get("gpu_s", 0.0) + (t2 - t1)
            timings["max_states"] = max(timings.get("max_states", 0), max(st["n_next"] for pl, _ in plans for st in pl))
        return out.cpu().numpy()

    # ---------------- likelihood of real observations ----------------
    def obs_factors(self, obs):
        """Gather indices into LT and an additive constant for one observation."""
        L, P = self.L, self.P
        idx = []
        const = 0.0
        y_p = [0] * P
        for (i, j, b, y, active) in obs.outcomes:
            if active:
                idx.append(self.py_idx(y, i, j, obs.loads[i], b))
                y_p[i] = y_p[L + j] = y
        for p in range(P):
            e_next = obs.next_engaged[p]
            if obs.engaged[p]:
                idx.append(self.re_idx(p, y_p[p], obs.loads[p], e_next))
            else:
                const += self.log_rho if e_next == 1 else self.log_1mrho
        return idx, const

    def loglik(self, LT: torch.Tensor, observations) -> torch.Tensor:
        """(B, n_obs) per-observation log-likelihood under every theta."""
        if not observations:
            return torch.zeros(LT.shape[0], 0, device=LT.device, dtype=LT.dtype)
        fac = [self.obs_factors(o) for o in observations]
        F = max(len(f[0]) for f in fac)
        idx = np.full((len(fac), F), self.pad, dtype=np.int64)
        for k, (ii, _) in enumerate(fac):
            idx[k, :len(ii)] = ii
        const = torch.as_tensor([f[1] for f in fac], device=LT.device, dtype=LT.dtype)
        return LT[:, torch.as_tensor(idx, device=LT.device)].sum(-1) + const[None]


def params_to_torch(p: dict, device="cuda", dtype=torch.float64) -> dict:
    """Single parameter dict (numpy, no batch dim) -> batched torch dict with B = 1."""
    out = {}
    for k in ("alpha", "beta", "gamma", "tauL", "tauR"):
        out[k] = torch.as_tensor(np.asarray(p[k], float)[None], device=device, dtype=dtype)
    out["psi_left"] = torch.as_tensor(np.asarray(p["psi_left"], float)[None], device=device, dtype=dtype)
    out["lam"] = torch.as_tensor(np.asarray([p["lam"]], float), device=device, dtype=dtype)
    if p.get("syn") is not None:
        out["syn"] = torch.as_tensor(np.asarray(p["syn"], float)[None], device=device, dtype=dtype)
    if p.get("g") is not None:
        out["g"] = torch.as_tensor(np.asarray(p["g"], float), device=device, dtype=dtype)
    return out
