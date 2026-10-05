"""NL-layer methods wired into the five-level reuse switch (round 3). Learner side: no truth access.

Each solver handles ONE problem on the evidence set handed out by `ReuseSwitch.begin_problem` and records every new
platform round through `ReuseSwitch.record` (the only billed path). Certification is attempted only when
`switch.may_certify()` (ev(m): at least m new steps on this problem); when the certificate already holds but more new
steps are required, the method keeps acquiring with its own rule, using the largest-regret alive models (ignoring the
eps cut) as the blocking set.

  JPC          SeqLRSet (UI threshold log 1/delta) + exact minimax certificate + DDA probes
  LR-chi2-grid same grid, same certificate and DDA design, Wald-type fixed-n threshold 0.5 chi2_{d_eff,1-delta}
               (lr_set threshold='chi2_deff', d_eff locked in round 2 = 2.98); NOT anytime-valid (isolates the threshold)
  B3           GLM linearised Wald ellipsoid at the grid MLE + greedy transductive design (round-0/1 B3); its design
               history is rebuilt from the rows inside the active evidence set (real + replay), so every arm changes
               B3's evidence exactly as it changes JPC's
  B8           full identification (width_J(Theta_t) < tau = eps/2) on the LR set, DDA towards the widest models,
               then the minimax certificate

Offset-null problems (streams.gap_quota) carry a public per-policy offset: J'_theta(pi) = (J_theta(pi) + b_pi) / s with
s = 1 + max b. The class J table passed in `PublicNL.J` is already transformed; B3's linearised values and gradients
are transformed with the same affine map.
"""
from __future__ import annotations

import math
import time

import numpy as np

from ..acquire.nl_kl_dda import IncidenceIndex, class_prob_tables, dda_choose
from ..certify.minimax_enum import certify_minimax, group_ids, observable_signature, regret_matrix, trichotomy
from ..evidence.lr_set import SeqLRSet
from .glm_linearised import (certify_linear, choose_design_action, class_theta_vecs, fisher_rounds, value_and_grad,
                             wald_beta)

METHODS = ("JPC", "LR-chi2-grid", "B3", "B8")
D_EFF_LOCKED_R2 = 2.98


class PublicNL:
    """Truth-free per-instance objects shared by all methods/arms: class tables, J tables of the stream problems."""

    def __init__(self, prop, ncl, problems, J_tables, c_known, nmax, eps, delta, top_m=5):
        self.prop, self.ncl = prop, ncl
        self.problems = problems
        self.J = [np.asarray(J, float) for J in J_tables]
        self.Reg = [regret_matrix(J) for J in self.J]
        self.eps, self.delta, self.top_m = eps, delta, top_m
        self.log_thr = math.log(1.0 / delta)
        self.c_known, self.nmax = c_known, nmax
        self.aspace = prop.aspace
        self.params = ncl.torch_params()
        self.LT = prop.tables(self.params)[0]
        self.LT_np = self.LT.numpy()
        nb = ncl.nb
        self.py, self.pe = class_prob_tables(ncl.np_params, c_known, nmax, nb)
        self.inc = IncidenceIndex(prop.codec, prop.aspace, nmax)
        self.legal = np.arange(prop.aspace.n)
        self.gid = group_ids(observable_signature(self.py, self.pe, max_level=1))
        self.all_single = len(np.unique(self.gid)) == ncl.B
        self.vecs = class_theta_vecs(ncl.np_params)

    def with_problems(self, problems, J_tables) -> "PublicNL":
        """Same class objects (shared, read-only), different problem list / J tables (a stream or a stream kind)."""
        import copy
        out = copy.copy(self)
        out.problems = list(problems)
        out.J = [np.asarray(J, float) for J in J_tables]
        out.Reg = [regret_matrix(J) for J in out.J]
        return out

    def make_set_factory(self, method: str, d_eff: float = D_EFF_LOCKED_R2):
        if method == "LR-chi2-grid":
            return lambda: SeqLRSet(self.prop, self.LT, self.delta, threshold="chi2_deff", d_eff=d_eff)
        return lambda: SeqLRSet(self.prop, self.LT, self.delta)


def _forced_blocking(Reg, mask, pi, top_m):
    idx = np.flatnonzero(mask)
    if len(idx) == 0 or pi is None:
        return []
    r = Reg[idx, pi]
    order = np.argsort(-r)[:top_m]
    return [int(idx[o]) for o in order if r[o] > 0]


def _status_value(cert):
    return cert["status"].value


# ------------------------------------------------------------------------------------------------ set methods
def solve_set_method(pub: PublicNL, k: int, sw, lr, handle, rng, tmax: int, method: str = "JPC", tau_b8=None):
    """JPC / LR-chi2-grid / B8 on problem k. Returns a dict (status, pi, steps, extra)."""
    Reg, J = pub.Reg[k], pub.J[k]
    eps, top_m = pub.eps, pub.top_m
    tau = (eps / 2.0) if tau_b8 is None else tau_b8
    t0 = time.perf_counter()
    status, cert, width, n_forced, n_explore = None, None, None, 0, 0
    size0 = lr.size()
    first_cert_step = None
    while True:
        mask = lr.mask().numpy()
        if not mask.any():
            status = "MODEL_CONFLICT"
            cert = certify_minimax(Reg, mask, eps, top_m)
            break
        if method == "B8":
            idx = np.flatnonzero(mask)
            Jm = J[idx]
            width = float((Jm.max(0) - Jm.min(0)).max())
            identified = width < tau
            cert = certify_minimax(Reg, mask, eps, top_m)
            ok = identified
        else:
            cert = certify_minimax(Reg, mask, eps, top_m)
            ok = _status_value(cert) == "CERTIFIED"
        if ok and first_cert_step is None:
            first_cert_step = sw.new_steps
        if ok and sw.may_certify():
            if method == "B8":
                status = "CERTIFIED" if _status_value(cert) == "CERTIFIED" else "IDENTIFIED_UNCERTIFIED"
            else:
                status = "CERTIFIED"
            break
        if not ok and not pub.all_single and sw.new_steps % 10 == 0:
            amb, _, _ = trichotomy(Reg, mask, pub.gid, eps)
            if amb:
                status = "OUT_OF_SCOPE"
                break
        if sw.new_steps >= tmax:
            status = "NEED_DATA"
            break
        kh = lr.mle()
        if ok:                                   # ev(m): certificate holds, more new steps required
            blk = _forced_blocking(Reg, mask, cert["pi"], top_m)
            n_forced += 1
        elif method == "B8":
            idx = np.flatnonzero(mask)
            dist = np.abs(J[idx] - J[kh][None]).max(1)
            order = np.argsort(-dist)
            blk = [int(idx[o]) for o in order[:top_m] if dist[o] > tau / 2] or [int(idx[o]) for o in order[:top_m]]
        else:
            blk = cert["blocking"]
        margins = pub.log_thr - lr.log_ratio().numpy()[blk] if method != "LR-chi2-grid" else \
            np.maximum(lr.cutoff() - lr.statistic().numpy()[blk], 1e-6)
        code = pub.prop.codec.encode(*handle.observable_state())
        a, info = dda_choose(code, pub.py[kh:kh + 1], pub.pe[kh:kh + 1], pub.LT_np[kh], pub.py[blk], pub.pe[blk],
                             margins, pub.inc, pub.prop, pub.legal, rng)
        n_explore += int(info["mode"] != "dda")
        sw.record(handle.step(a))
    mask = lr.mask().numpy()
    pi = cert["pi"] if status in ("CERTIFIED", "IDENTIFIED_UNCERTIFIED") else None
    if method == "B8" and status == "CERTIFIED":
        pi = cert["pi"]
    return {"status": status, "pi": pi, "steps": sw.new_steps,
            "extra": {"set_size_start": int(size0), "set_size": int(mask.sum()), "r_bar_end": float(cert["r_bar"]),
                      "first_cert_step": first_cert_step, "forced_steps": n_forced,
                      "id_width_end": width, "wall_clock_s": time.perf_counter() - t0,
                      "cutoff": float(lr.cutoff()), "explore_or_zero_info_steps": n_explore}}


# ------------------------------------------------------------------------------------------------ B3
def _hist_from_rows(codec, obs_list):
    hist = {}
    for o in obs_list:
        key = (codec.encode(o.loads, o.engaged), o.action)
        hist[key] = hist.get(key, 0) + 1
    return hist


def solve_b3(pub: PublicNL, k: int, sw, lr, handle, rng, tmax: int, q):
    """B3 on problem k with the GLM ellipsoid built from the rows of the active evidence set."""
    t0 = time.perf_counter()
    prop, codec = pub.prop, pub.prop.codec
    d = pub.vecs.shape[1]
    beta = wald_beta(d, pub.delta)
    hist = _hist_from_rows(codec, lr.obs)
    b_off = np.asarray(q.meta.get("offset_b", np.zeros(len(q.policies))), float)
    s_off = float(q.meta.get("offset_scale", 1.0))
    U = type("U", (), {"w": q.utility.w, "w_ret": q.utility.w_ret, "c_q": q.utility.c_q})()
    plans = [prop.build_policy_plan(p, q.loads0, q.engaged0, q.H) for p in q.policies]

    def build_V(kh):
        v = pub.vecs[kh]
        V = np.eye(d)
        by_code = {}
        for (code, a), n in hist.items():
            by_code.setdefault(code, []).append((a, n))
        for code, lst in by_code.items():
            F = fisher_rounds(v, codec, prop.aspace, code, [a for a, _ in lst], pub.c_known, 2, 2)
            V += np.einsum("k,kde->de", np.array([n for _, n in lst], float), F)
        return V

    grad_cache, kh_prev, V, status, lc, lin_err, n_forced = {}, None, None, None, None, None, 0
    first_cert_step = None
    while True:
        kh = lr.mle()
        v = pub.vecs[kh]
        if kh != kh_prev:
            V = build_V(kh)
            kh_prev = kh
        if kh not in grad_cache:
            Jlin, G = value_and_grad(prop, plans, v, U, 2, 2)
            Jlin, G = (Jlin + b_off) / s_off, G / s_off
            grad_cache[kh] = (Jlin, G)
            if lin_err is None:
                lin_err = float(np.abs(Jlin - pub.J[k][kh]).max())
        Jlin, G = grad_cache[kh]
        lc = certify_linear(pub.J[k][kh], G, np.linalg.inv(V), beta, pub.eps)
        ok = bool(lc["certified"])
        if ok and first_cert_step is None:
            first_cert_step = sw.new_steps
        if ok and sw.may_certify():
            status = "CERTIFIED"
            break
        if sw.new_steps >= tmax:
            status = "NEED_DATA"
            break
        code = codec.encode(*handle.observable_state())
        Fa = fisher_rounds(v, codec, prop.aspace, code, pub.legal, pub.c_known, 2, 2)
        if ok:
            blkD = np.delete(lc["D"], lc["pi"], axis=0)
            n_forced += 1
        else:
            blkD = lc["D"][lc["ub"] > pub.eps]
        a = choose_design_action(Fa, V, blkD, pub.legal, rng)
        obs = handle.step(a)
        sw.record(obs)
        hist[(code, a)] = hist.get((code, a), 0) + 1
        V = V + Fa[list(pub.legal).index(a)]
    return {"status": status, "pi": lc["pi"] if status == "CERTIFIED" else None, "steps": sw.new_steps,
            "extra": {"wald_beta": beta, "lin_value_err": lin_err, "r_bar_end": float(lc["r_bar"]),
                      "first_cert_step": first_cert_step, "forced_steps": n_forced, "set_size": int(lr.size()),
                      "wall_clock_s": time.perf_counter() - t0}}


def solve(pub: PublicNL, method: str, k: int, sw, lr, handle, rng, tmax: int):
    q = pub.problems[k]
    if method == "B3":
        return solve_b3(pub, k, sw, lr, handle, rng, tmax, q)
    if method in ("JPC", "LR-chi2-grid", "B8"):
        return solve_set_method(pub, k, sw, lr, handle, rng, tmax, method=method)
    raise ValueError(method)
