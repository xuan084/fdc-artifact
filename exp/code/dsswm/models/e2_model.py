"""E2 (Tier 2) learner-side model: 6x6 outcome-dependent dynamics with CONTINUOUS persistent parameters.

Parameter vector v (d = 32):
    alpha_0..5 | gamma_0..5 | tauL_0..5 | beta_0..5 | tauR_0..5 | psi | lam
Outcome  : y_ij ~ Bern(sigmoid(alpha_i + beta_j - gamma_i n_i + psi b_ij))   (both engaged, matched)
Retention: engaged p:  e_p' ~ Bern(sigmoid(tau_p + c y_p - lam n_p));  disengaged p: e_p' ~ Bern(rho_ret)
Loads    : known rule (matched +1 capped at nmax, unmatched -1 floored at 0).
c, rho_ret, nmax are public constants.

This module is learner-side: it contains a *simulator* that takes an arbitrary parameter batch V (M, d)
(model rollouts), the negative log-likelihood of authenticated observations, and the expected Fisher information.
It never sees ground-truth parameters (the harness calls the same simulator with theta* for evaluation only).
"""
from __future__ import annotations

import math

import numpy as np
import torch

L = R = 6
P = L + R
NMAX = 2
C_KNOWN = 1.0
RHO_RET = 0.3
D = 3 * L + 2 * R + 2
IA, IG, ITL, IB, ITR, IPSI, ILAM = 0, L, 2 * L, 3 * L, 3 * L + R, 3 * L + 2 * R, 3 * L + 2 * R + 1

# model-class box (public); the generator draws theta* strictly inside it
LO = np.array([-2.0] * L + [0.0] * L + [0.0] * L + [-1.5] * R + [0.0] * R + [0.0, 0.0])
HI = np.array([2.0] * L + [1.0] * L + [2.0] * L + [1.5] * R + [2.0] * R + [2.0, 1.0])

# super-blocks used by the copy dual / BnB (parameter indices). Participants own their own parameters; the global
# block (psi, lam) is shared.
SUPER_BLOCKS = [
    [IA + i for i in (0, 1, 2)] + [IG + i for i in (0, 1, 2)] + [ITL + i for i in (0, 1, 2)],
    [IA + i for i in (3, 4, 5)] + [IG + i for i in (3, 4, 5)] + [ITL + i for i in (3, 4, 5)],
    [IB + j for j in (0, 1, 2)] + [ITR + j for j in (0, 1, 2)],
    [IB + j for j in (3, 4, 5)] + [ITR + j for j in (3, 4, 5)],
    [IPSI, ILAM],
]


def unpack(V: torch.Tensor):
    return (V[:, IA:IA + L], V[:, IG:IG + L], V[:, ITL:ITL + L], V[:, IB:IB + R], V[:, ITR:ITR + R],
            V[:, IPSI], V[:, ILAM])


# ------------------------------------------------------------------ batched Monte-Carlo simulator
@torch.no_grad()
def rollout_contrib(V: torch.Tensor, policy, loads0, eng0, H: int, Urand: torch.Tensor, Erand: torch.Tensor,
                    w: np.ndarray, w_ret: float, c_q: float, max_batch: int = 1 << 18) -> torch.Tensor:
    """Per-episode, per-participant utility contributions (M, N, P) for parameter batch V (M, d).

    Urand (N, H, L, R), Erand (N, H, P): uniforms shared across all members (common random numbers).
    Pair outcome y_ij contributes c_q w_t / 2 to i and to j; final engagement e_p(H) contributes c_q w_ret to p.
    Sum over P gives the trajectory utility U_q.
    """
    M = V.shape[0]
    N = Urand.shape[0]
    dev = V.device
    chunk = max(1, max_batch // N)
    out = torch.empty((M, N, P), device=dev, dtype=torch.float32)
    l0 = torch.as_tensor(np.asarray(loads0), device=dev, dtype=torch.long)
    e0 = torch.as_tensor(np.asarray(eng0), device=dev, dtype=torch.long)
    wt = [float(x) for x in w]
    for s in range(0, M, chunk):
        Vc = V[s:s + chunk].float()
        m = Vc.shape[0]
        a, g, tl, b, tr, psi, lam = unpack(Vc)
        tau = torch.cat([tl, tr], 1)
        loads = l0.expand(m, N, P).clone()
        eng = e0.expand(m, N, P).clone()
        contrib = torch.zeros((m, N, P), device=dev)
        for t in range(H):
            match, inc = policy.act(t, loads.view(-1, P), eng.view(-1, P))
            match = match.view(m, N, L, R)
            inc = inc.view(m, N, L, R)
            nL = loads[:, :, :L].float()
            logit = (a[:, None, :, None] + b[:, None, None, :] - g[:, None, :, None] * nL[..., None]
                     + psi[:, None, None, None] * inc.float())
            engb = eng.bool()
            active = match & engb[:, :, :L, None] & engb[:, :, None, L:]
            y = active & (Urand[None, :, t] < torch.sigmoid(logit))
            yf = y.float()
            contrib[:, :, :L] += (0.5 * c_q * wt[t]) * yf.sum(3)
            contrib[:, :, L:] += (0.5 * c_q * wt[t]) * yf.sum(2)
            yp = torch.cat([y.any(3), y.any(2)], 2).float()
            p_stay = torch.sigmoid(tau[:, None, :] + C_KNOWN * yp - lam[:, None, None] * loads.float())
            p = torch.where(engb, p_stay, torch.full_like(p_stay, RHO_RET))
            eng = (Erand[None, :, t] < p).long()
            matched = torch.cat([match.any(3), match.any(2)], 2)
            loads = torch.where(matched, torch.clamp(loads + 1, max=NMAX), torch.clamp(loads - 1, min=0))
        contrib += (c_q * w_ret) * eng.float()
        out[s:s + m] = contrib
    return out


def rollout_scorefn(v: torch.Tensor, policy, loads0, eng0, H: int, Urand, Erand, w, w_ret, c_q):
    """Single parameter vector v (d,) with requires_grad: returns per-episode utility U (N,) (detached) and
    log-probability of each sampled trajectory under v (N,) (differentiable) -> score-function gradients."""
    N = Urand.shape[0]
    dev = v.device
    V = v[None].float()
    a, g, tl, b, tr, psi, lam = unpack(V)
    tau = torch.cat([tl, tr], 1)
    loads = torch.as_tensor(np.asarray(loads0), device=dev, dtype=torch.long).expand(N, P).clone()
    eng = torch.as_tensor(np.asarray(eng0), device=dev, dtype=torch.long).expand(N, P).clone()
    U = torch.zeros(N, device=dev)
    logp = torch.zeros(N, device=dev)
    for t in range(H):
        with torch.no_grad():
            match, inc = policy.act(t, loads, eng)
        nL = loads[:, :L].float()
        logit = a[:, :, None] + b[:, None, :] - g[:, :, None] * nL[..., None] + psi[:, None, None] * inc.float()
        pr = torch.sigmoid(logit)
        engb = eng.bool()
        active = match & engb[:, :L, None] & engb[:, None, L:]
        with torch.no_grad():
            y = active & (Urand[:, t] < pr)
        yf = y.float()
        lp_y = torch.where(y, torch.nn.functional.logsigmoid(logit), torch.nn.functional.logsigmoid(-logit))
        logp = logp + (lp_y * active.float()).sum((1, 2))
        U = U + c_q * float(w[t]) * yf.sum((1, 2))
        yp = torch.cat([y.any(2), y.any(1)], 1).float()
        lg = tau + C_KNOWN * yp - lam[:, None] * loads.float()
        with torch.no_grad():
            p = torch.where(engb, torch.sigmoid(lg), torch.full_like(lg, RHO_RET))
            new_eng = (Erand[:, t] < p)
        lp_e = torch.where(new_eng, torch.nn.functional.logsigmoid(lg), torch.nn.functional.logsigmoid(-lg))
        logp = logp + (lp_e * engb.float()).sum(1)
        matched = torch.cat([match.any(2), match.any(1)], 1)
        loads = torch.where(matched, torch.clamp(loads + 1, max=NMAX), torch.clamp(loads - 1, min=0))
        eng = new_eng.long()
    U = U + c_q * w_ret * eng.float().sum(1)
    return U.detach(), logp


def values_and_grads(v_hat: np.ndarray, policies, loads0, eng0, H, w, w_ret, c_q, N: int, gen: torch.Generator,
                     device="cuda"):
    """MC values J (K,), score-function gradients G (K, d) and per-episode utilities (K, N) at v_hat (CRN)."""
    Urand = torch.rand((N, H, L, R), device=device, generator=gen)
    Erand = torch.rand((N, H, P), device=device, generator=gen)
    Js, Gs, Us = [], [], []
    for pol in policies:
        v = torch.tensor(v_hat, dtype=torch.float32, device=device, requires_grad=True)
        U, logp = rollout_scorefn(v, pol, loads0, eng0, H, Urand, Erand, w, w_ret, c_q)
        base = U.mean()
        (gr,) = torch.autograd.grad(((U - base) * logp).mean(), v)
        Js.append(float(base)); Gs.append(gr.detach().cpu().numpy().astype(float)); Us.append(U)
    return np.array(Js), np.stack(Gs), torch.stack(Us)


# ------------------------------------------------------------------ evidence: likelihood of authenticated rounds
class E2Data:
    """Accumulates authenticated observations into flat arrays for the likelihood."""

    def __init__(self):
        self.o_i, self.o_j, self.o_n, self.o_b, self.o_y = [], [], [], [], []
        self.r_p, self.r_y, self.r_n, self.r_e = [], [], [], []
        self.rounds = []          # (loads, engaged, match(6x6), inc(6x6)) for Fisher information
        self.n_rounds = 0

    def add(self, obs):
        from ..core.provenance import require_authentic
        require_authentic(obs)
        loads = np.asarray(obs.loads)
        eng = np.asarray(obs.engaged)
        ne = np.asarray(obs.next_engaged)
        yp = np.zeros(P, dtype=np.int64)
        active = np.zeros(P, dtype=bool)
        match = np.zeros((L, R), dtype=bool)
        inc = np.zeros((L, R), dtype=np.int64)
        for i, j, b, y, act in obs.outcomes:
            match[i, j] = True
            inc[i, j] = b
            if act:
                self.o_i.append(i); self.o_j.append(j); self.o_n.append(int(loads[i])); self.o_b.append(b)
                self.o_y.append(y)
                yp[i] = y; yp[L + j] = y
                active[i] = active[L + j] = True
        for p in range(P):
            if eng[p]:
                self.r_p.append(p); self.r_y.append(int(yp[p])); self.r_n.append(int(loads[p]))
                self.r_e.append(int(ne[p]))
        self.rounds.append((loads.copy(), eng.copy(), match, inc))
        self.n_rounds += 1
        self._t = None

    def tensors(self, device):
        if getattr(self, "_t", None) is None or self._t[0] != device:
            f = lambda x, dt=torch.float32: torch.as_tensor(np.asarray(x), device=device, dtype=dt)
            self._t = (device, dict(oi=f(self.o_i, torch.long), oj=f(self.o_j, torch.long), on=f(self.o_n),
                                    ob=f(self.o_b), oy=f(self.o_y), rp=f(self.r_p, torch.long), ry=f(self.r_y),
                                    rn=f(self.r_n), re=f(self.r_e)))
        return self._t[1]


def nll_batch(V: torch.Tensor, data: E2Data) -> torch.Tensor:
    """Negative log-likelihood (M,) of all authenticated rounds for each row of V (M, d) (float64 for LR tests)."""
    T = data.tensors(V.device)
    V = V.double()
    a, g, tl, b, tr, psi, lam = unpack(V)
    tau = torch.cat([tl, tr], 1)
    out = torch.zeros(V.shape[0], device=V.device, dtype=torch.float64)
    if len(data.o_i):
        lo = (a[:, T["oi"]] + b[:, T["oj"]] - g[:, T["oi"]] * T["on"].double() + psi[:, None] * T["ob"].double())
        out = out + torch.nn.functional.binary_cross_entropy_with_logits(
            lo, T["oy"].double().expand_as(lo), reduction="none").sum(1)
    if len(data.r_p):
        lr = tau[:, T["rp"]] + C_KNOWN * T["ry"].double() - lam[:, None] * T["rn"].double()
        out = out + torch.nn.functional.binary_cross_entropy_with_logits(
            lr, T["re"].double().expand_as(lr), reduction="none").sum(1)
    return out


def fit_mle(data: E2Data, device="cuda", init=None, ridge=1e-3, iters=200) -> tuple[np.ndarray, float]:
    """Box-constrained MLE via a sigmoid reparametrisation + L-BFGS; tiny ridge towards the box centre for
    unidentified directions. Returns (v_hat, NLL(v_hat))."""
    lo = torch.tensor(LO, device=device, dtype=torch.float64)
    hi = torch.tensor(HI, device=device, dtype=torch.float64)
    ctr = 0.5 * (lo + hi)
    v0 = torch.tensor(init if init is not None else 0.5 * (LO + HI), device=device, dtype=torch.float64)
    u0 = ((v0 - lo) / (hi - lo)).clamp(1e-4, 1 - 1e-4)
    z = torch.log(u0 / (1 - u0)).clone().requires_grad_(True)
    opt = torch.optim.LBFGS([z], lr=1.0, max_iter=iters, tolerance_grad=1e-9, tolerance_change=1e-12,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        v = lo + (hi - lo) * torch.sigmoid(z)
        loss = nll_batch(v[None], data)[0] + ridge * ((v - ctr) ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        v = lo + (hi - lo) * torch.sigmoid(z)
        nll = float(nll_batch(v[None], data)[0])
    return v.cpu().numpy(), nll


def fisher_terms(v: np.ndarray, loads, eng, match, inc) -> np.ndarray:
    """Expected Fisher information (d, d) of one round at state (loads, eng) under action (match, inc)."""
    a, g, tl = v[IA:IA + L], v[IG:IG + L], v[ITL:ITL + L]
    b, tr, psi, lam = v[IB:IB + R], v[ITR:ITR + R], v[IPSI], v[ILAM]
    tau = np.concatenate([tl, tr])
    F = np.zeros((D, D))
    py = np.zeros(P)
    act = np.zeros(P, bool)
    for i, j in zip(*np.nonzero(match)):
        if eng[i] and eng[L + j]:
            lg = a[i] + b[j] - g[i] * loads[i] + psi * inc[i, j]
            p = 1 / (1 + math.exp(-lg))
            gy = np.zeros(D); gy[IA + i] = 1; gy[IB + j] = 1; gy[IG + i] = -loads[i]; gy[IPSI] = inc[i, j]
            F += p * (1 - p) * np.outer(gy, gy)
            py[i] = py[L + j] = p
            act[i] = act[L + j] = True
    for pp in range(P):
        if not eng[pp]:
            continue
        ge = np.zeros(D)
        ge[(ITL + pp) if pp < L else (ITR + pp - L)] = 1
        ge[ILAM] = -loads[pp]
        q0 = 1 / (1 + math.exp(-(tau[pp] - lam * loads[pp])))
        if act[pp]:
            q1 = 1 / (1 + math.exp(-(tau[pp] + C_KNOWN - lam * loads[pp])))
            wgt = (1 - py[pp]) * q0 * (1 - q0) + py[pp] * q1 * (1 - q1)
        else:
            wgt = q0 * (1 - q0)
        F += wgt * np.outer(ge, ge)
    return F


def fisher_total(v: np.ndarray, data: E2Data, lam0: float = 1.0) -> np.ndarray:
    V = lam0 * np.eye(D)
    for loads, eng, match, inc in data.rounds:
        V += fisher_terms(v, loads, eng, match, inc)
    return V


# ------------------------------------------------------------------ one-round KL (DDA) between two models
def _bern_kl(p, q):
    p = np.clip(p, 1e-9, 1 - 1e-9); q = np.clip(q, 1e-9, 1 - 1e-9)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def round_kl(v_ref: np.ndarray, V_alt: np.ndarray, loads, eng, matches: np.ndarray, incs: np.ndarray) -> np.ndarray:
    """KL(P_ref(.|s,a) || P_alt(.|s,a)) for candidate actions (A,) x alternative models (M,) -> (A, M).
    matches/incs: (A, L, R). Outcome KL over active pairs + retention KL averaged over y under the reference."""
    sig = lambda x: 1 / (1 + np.exp(-x))
    loads = np.asarray(loads, float); engb = np.asarray(eng).astype(bool)

    def parts(v):     # v (M, d)
        return (v[:, IA:IA + L], v[:, IG:IG + L], np.concatenate([v[:, ITL:ITL + L], v[:, ITR:ITR + R]], 1),
                v[:, IB:IB + R], v[:, IPSI], v[:, ILAM])

    ar, gr, tr_, br, pr_, lr_ = parts(v_ref[None])
    aa, ga, ta, ba, pa, la = parts(V_alt)
    A = matches.shape[0]
    active = matches.astype(bool) & engb[None, :L, None] & engb[None, None, L:]          # (A, L, R)
    lg_r = ar[0][None, :, None] + br[0][None, None, :] - gr[0][None, :, None] * loads[None, :L, None] \
        + pr_[0] * incs                                                                   # (A, L, R)
    lg_a = aa[:, None, :, None] + ba[:, None, None, :] - ga[:, None, :, None] * loads[None, None, :L, None] \
        + pa[:, None, None, None] * incs[None]                                            # (M, A, L, R)
    p_r = sig(lg_r)
    kl_y = (_bern_kl(p_r[None], sig(lg_a)) * active[None]).sum((2, 3))                    # (M, A)
    # P(y_p = 1) under ref for participants in active pairs
    py = np.concatenate([(p_r * active).sum(2), (p_r * active).sum(1)], 1)               # (A, P)
    act_p = np.concatenate([active.any(2), active.any(1)], 1)                             # (A, P)
    q0r = sig(tr_[0] - lr_[0] * loads)                                                    # (P,)
    q1r = sig(tr_[0] + C_KNOWN - lr_[0] * loads)
    q0a = sig(ta - la[:, None] * loads[None])                                             # (M, P)
    q1a = sig(ta + C_KNOWN - la[:, None] * loads[None])
    k0 = _bern_kl(q0r[None], q0a) * engb[None]                                            # (M, P)
    k1 = _bern_kl(q1r[None], q1a) * engb[None]
    kl_e = ((1 - py[None]) * k0[:, None] + py[None] * k1[:, None]) * act_p[None] + k0[:, None] * (~act_p[None])
    return (kl_y + kl_e.sum(2)).T                                                         # (A, M)
