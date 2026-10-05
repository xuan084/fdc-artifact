"""Round-3 E1-Lin problem streams + cell runner (harness side: this module CREATES ground truth and reads it to place
and score problems; learner code must never import it). Methodology section 2.2.

Instance (statistical unit, seed s): theta* drawn exactly as streams.generator.make_lin_instance (rng [s, 1], the
LIN_DEFAULTS continuous prior box), E1-Lin 3x3, sigma = 1.5, incentive levels (1, 2) with legal probes b <= 1,
n0 = 20 shared initial rounds (rng [s, 2], level <= 1).
Pool (sequential, rng-addressed, deterministic):
  fresh    candidate j: rng [s, 53, j] -> H in {4, 6, 8}, loads0, affinity, 4-8 level-1 generator policies, a utility
           without retention (y_scale 2.4). Policies with identical per-step feature paths are de-duplicated
           (first kept); a candidate needs >= 2 distinct policies.
  twin     of an already ACCEPTED problem p, try r: rng [s, 54, j_p, r] -> same (s0, Pi) (same policy objects, same
           loads0), and either new utility weights ('weights') or a horizon shortened by 2 with new weights
           ('horizon', only if H_p >= 4). A twin whose candidates have coinciding feature paths is rejected.
  draw t:  if both kinds still need problems, a twin is tried with probability 0.4 (rng [s, 52] decides; the parent is
           uniform over accepted problems), otherwise a fresh candidate.
Quota (eps_Lin = 0.05): tie Delta < eps, near eps <= Delta < 2 eps, clear Delta >= 2 eps, 5 each (Delta = true
top-2 gap of J = Z theta*). A candidate is accepted iff its layer still has room and its kind (fresh / twin) still
has room. The number of twins is N_SHARED(s) = 4 + (s mod 2) (shared fraction 4/15 or 5/15, mean 0.30 over
consecutive seeds). A stream that is not full after MAX_DRAWS pool draws makes the instance `quota_fail`
(replaced by gap_quota.fill_instances with the next reserve seed).
Streams: stream t in {0, 1, 2}: permutation rng [s, 57, t] of the accepted problems, noise seed 42 / 123 / 456; the
sibling platform of the `vol` arm uses noise seed + 7919, same theta*, same problems. `shared` of problem k = some
earlier problem of THIS stream order has the same (s0, Pi) (so the stream-level shared count equals N_SHARED).
Public decision direction D_q = z_a - z_b for the top-2 candidates at the prior-box centre theta_bar (no truth;
exact on the Lin class); per problem: cos_max = max_prev |cos(D_q, D_prev)|, cos_span = ||P_span(prev) D_q|| / ||D_q||.
Lin-Static: StaticLinEnv (gamma = 0, loads frozen) with the static class (load feature removed, parameter count
matched, d = 11). Putting the dynamic stream's problems on the static platform makes almost all of them static ties
(pilot: 14-15 of 15), so Lin-Static gets ITS OWN quota selection under the static truth (same rule, same 5/5/5 and
twin counts, separate rng streams); `stream(..., static=True, same_problems=True)` keeps the diagnostic variant.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..baselines.lin_rage import (EPS_LIN, LIN_TAUS, METHOD_CODE, S_LIN, T_MAX_LIN, PublicLin, LinGeom, solve)
from ..envs.lin import LinEnv, StaticLinEnv
from ..evidence.reuse_switch import (SHADOW_SEED_OFFSET, BillingError, ReuseSwitch, ShadowLedger, parse_arm,
                                     score_truncations)
from ..models.lin_class import LinClass
from .generator import LIN_DEFAULTS, Problem, _initial_data, _sample_policies
from .utilities import sample_utility

QUOTA = 5
LAYERS = ("tie", "near", "clear")
K_STREAM = 3 * QUOTA
MAX_DRAWS = 4000
P_TWIN = 0.4
STREAM_NOISE = (42, 123, 456)
THETA_BAR = None   # set below from the public prior box


def n_shared(seed: int) -> int:
    return 4 + (int(seed) % 2)


def gap_layer(delta: float, eps: float = EPS_LIN) -> str:
    if delta < eps:
        return "tie"
    if delta < 2 * eps:
        return "near"
    return "clear"


def top2_gap(j) -> float:
    s = np.sort(np.asarray(j, float))[::-1]
    return float(s[0] - s[1])


def prior_centre(lc: LinClass, prior=LIN_DEFAULTS["prior"]) -> np.ndarray:
    c = np.zeros(lc.d)
    for k, n in enumerate(lc.names):
        key = "psi2" if n == "psi2" else ("psi" if n == "psi1" else n.rstrip("0123456789"))
        lo, hi = prior[key]
        c[k] = 0.5 * (lo + hi)
    return c


def lin_truth(seed: int, cfg=LIN_DEFAULTS) -> dict:
    """Identical draw to make_lin_instance(seed) (rng [seed, 1])."""
    rng = np.random.default_rng([seed, 1])
    L, R, pr = cfg["L"], cfg["R"], cfg["prior"]
    alpha = rng.uniform(*pr["alpha"], L)
    beta = rng.uniform(*pr["beta"], R)
    gamma = rng.uniform(*pr["gamma"], L)
    psi = np.array([rng.uniform(*pr["psi"]), rng.uniform(*pr["psi2"])])
    return {"alpha": alpha, "beta": beta, "gamma": gamma, "psi": psi}


def make_env(seed: int, truth: dict, noise_seed: int, static: bool = False, cfg=LIN_DEFAULTS):
    Env = StaticLinEnv if static else LinEnv
    return Env(truth["alpha"], truth["beta"], truth["gamma"], truth["psi"], sigma=cfg["sigma"], L=cfg["L"],
               R=cfg["R"], nmax=cfg["nmax"], budget=cfg["budget"], incentive_levels=(1, 2),
               seed=int(seed) * 1000 + int(noise_seed))


@dataclass
class LinCandidate:
    j: int                    # pool draw index
    problem: Problem
    Zt: np.ndarray            # (n_pol, H, d) dynamic class
    J: np.ndarray             # harness: true values
    delta: float
    layer: str
    kind: str                 # 'fresh' | 'twin'
    group: int                # root fresh draw index (same (s0, Pi) <=> same group)
    D: np.ndarray | None = None


def _zt(lc, policies, loads0, H, util, aspace):
    return np.stack([lc.z_per_step(p, loads0, H, util, aspace) for p in policies])


def _distinct_paths(Zt, tol=1e-12):
    keep = []
    for a in range(len(Zt)):
        if all(np.abs(Zt[a] - Zt[b]).max() > tol for b in keep):
            keep.append(a)
    return keep


class LinStreamBuilder:
    def __init__(self, seed: int, eps: float = EPS_LIN, max_draws: int = MAX_DRAWS, quota: int = QUOTA,
                 cfg=LIN_DEFAULTS):
        self.seed, self.eps, self.max_draws, self.quota, self.cfg = int(seed), eps, max_draws, quota, cfg
        self.truth = lin_truth(seed, cfg)
        self._env0 = make_env(seed, self.truth, STREAM_NOISE[0])
        self.aspace = self._env0.aspace
        self.theta = self._env0.true_theta_vector()
        self.theta_static = make_env(seed, self.truth, STREAM_NOISE[0], static=True).true_theta_vector()
        self.lc = LinClass(cfg["L"], cfg["R"], self.aspace.nb, nmax=cfg["nmax"], static=False)
        self.lcs = LinClass(cfg["L"], cfg["R"], self.aspace.nb, nmax=cfg["nmax"], static=True)
        self.theta_bar = prior_centre(self.lc, cfg["prior"])
        self.n_shared = n_shared(seed)
        self.n_draws_fresh = 0
        self._sel = {}
        self._geom = {}

    # ------------------------------------------------------------------------------ pool
    def _fresh(self, j: int, static: bool = False):
        cfg, L, R = self.cfg, self.cfg["L"], self.cfg["R"]
        prng = np.random.default_rng([self.seed, 53, j])
        H = int(prng.choice(cfg["H_choices"]))
        loads0 = prng.integers(0, cfg["nmax"] + 1, L + R)
        aff = prng.standard_normal((L, R))
        pols = _sample_policies(prng, self.aspace, H, aff, cfg["nmax"], cfg["n_pol"])
        util = sample_utility(prng, H, L, R, with_retention=False, y_scale=cfg["y_scale"])
        Zt = _zt(self.lcs if static else self.lc, pols, loads0, H, util, self.aspace)
        keep = _distinct_paths(Zt)
        if len(keep) < 2:
            return None
        pols = [pols[a] for a in keep]
        q = Problem(f"lin3{'s' if static else 'r'}{self.seed}_f{j}", 1, H, loads0, np.ones(L + R, dtype=np.int64), pols, util, aff,
                    meta={"draw": j, "kind": "fresh", "twin_of": None, "twin_variant": None})
        return self._cand(j, q, Zt[keep], "fresh", j, static)

    def _twin(self, parent: LinCandidate, r: int, j: int, static: bool = False):
        cfg, L, R = self.cfg, self.cfg["L"], self.cfg["R"]
        prng = np.random.default_rng([self.seed, 54 + 100 * int(static), parent.group, parent.j, r])
        p = parent.problem
        variant = "horizon" if (p.H >= 4 and prng.random() < 0.5) else "weights"
        H = p.H - 2 if variant == "horizon" else p.H
        util = sample_utility(prng, H, L, R, with_retention=False, y_scale=cfg["y_scale"])
        Zt = _zt(self.lcs if static else self.lc, p.policies, p.loads0, H, util, self.aspace)
        if len(_distinct_paths(Zt)) < len(p.policies):
            return None
        q = Problem(f"lin3{'s' if static else 'r'}{self.seed}_f{parent.group}_t{j}", 1, H, p.loads0.copy(), p.engaged0.copy(),
                    list(p.policies), util, p.aff, meta={"draw": j, "kind": "twin", "twin_of": p.pid,
                                                         "twin_variant": variant})
        return self._cand(j, q, Zt, "twin", parent.group, static)

    def _cand(self, j, q, Zt, kind, group, static=False):
        J = Zt.sum(1) @ (self.theta_static if static else self.theta)
        d = top2_gap(J)
        return LinCandidate(j, q, Zt, J, d, gap_layer(d, self.eps), kind, group)

    # ------------------------------------------------------------------------------ selection
    def select(self, static: bool = False) -> dict:
        """Quota selection under the dynamic truth (static=False) or, for Lin-Static, its OWN quota under the static
        truth (fresh pool rng [s, 53, 10^6 + j], twins rng [s, 154, ...], coin rng [s, 152]); the same rule otherwise."""
        if static in self._sel:
            return self._sel[static]
        need_kind = {"fresh": K_STREAM - self.n_shared, "twin": self.n_shared}
        have_kind = {"fresh": 0, "twin": 0}
        layer_cnt = {l: 0 for l in LAYERS}
        chosen: list[LinCandidate] = []
        coin = np.random.default_rng([self.seed, 52 + 100 * int(static)])
        n_fresh_draw, n_twin_try, draws, rejected = 0, 0, 0, {"layer_full": 0, "invalid": 0, "kind_full": 0}
        while len(chosen) < K_STREAM and draws < self.max_draws:
            draws += 1
            u, pr = coin.random(), coin.random()
            want_twin = (have_kind["twin"] < need_kind["twin"] and chosen and
                         (have_kind["fresh"] >= need_kind["fresh"] or u < P_TWIN))
            if want_twin:
                parent = chosen[int(pr * len(chosen))]
                c = self._twin(parent, n_twin_try, draws, static)
                n_twin_try += 1
            else:
                if have_kind["fresh"] >= need_kind["fresh"]:
                    rejected["kind_full"] += 1
                    continue
                c = self._fresh(n_fresh_draw + (10 ** 6 if static else 0), static)
                n_fresh_draw += 1
            if c is None:
                rejected["invalid"] += 1
                continue
            if layer_cnt[c.layer] >= self.quota:
                rejected["layer_full"] += 1
                continue
            chosen.append(c)
            layer_cnt[c.layer] += 1
            have_kind[c.kind] += 1
        fail = len(chosen) < K_STREAM
        self._sel[static] = {"quota_fail": fail, "n_draws": draws, "n_fresh_draws": n_fresh_draw, "n_twin_tries": n_twin_try,
                     "rejected": rejected, "chosen": chosen, "layer_counts": layer_cnt, "kind_counts": have_kind,
                     "n_shared_target": self.n_shared, "static": static}
        return self._sel[static]

    # ------------------------------------------------------------------------------ public directions
    def decision_direction(self, c: LinCandidate) -> np.ndarray:
        if c.D is None:
            Z = c.Zt.sum(1)
            v = Z @ self.theta_bar
            o = np.argsort(-v)
            c.D = Z[o[0]] - Z[o[1]]
        return c.D

    def overlap(self, cands) -> list:
        out, prev = [], []
        for c in cands:
            D = self.decision_direction(c)
            nd = float(np.linalg.norm(D))
            u = D / nd if nd > 0 else D
            if not prev:
                out.append({"cos_max": None, "cos_span": None})
            else:
                P = np.stack(prev, 1)
                Qm, _ = np.linalg.qr(P)
                out.append({"cos_max": max(abs(float(u @ p)) for p in prev),
                            "cos_span": float(np.linalg.norm(Qm @ (Qm.T @ u)))})
            prev.append(u)
        return out

    # ------------------------------------------------------------------------------ streams
    def order(self, stream: int, static: bool = False):
        sel = self.select(static)
        return list(np.random.default_rng([self.seed, 57, stream]).permutation(len(sel["chosen"])))

    def stream(self, stream: int, sibling: bool = False, static: bool = False, n_problems: int | None = None,
               same_problems: bool = False):
        """static=True: Lin-Static stream with its own static quota; same_problems=True (diagnostic) instead puts
        the DYNAMIC stream's problems on the static platform (almost all of them are static ties)."""
        sk = static and not same_problems
        sel = self.select(sk)
        if sel["quota_fail"]:
            raise RuntimeError(f"instance {self.seed}: quota_fail (Lin stream, static={sk})")
        cands = [sel["chosen"][i] for i in self.order(stream, sk)]
        ov = self.overlap(cands)
        seen, meta = set(), []
        lc = self.lcs if static else self.lc
        th = self.theta_static if static else self.theta
        J_true = []
        for c, o in zip(cands, ov):
            q = c.problem
            if static and same_problems:
                Jq = _zt(self.lcs, q.policies, q.loads0, q.H, q.utility, self.aspace).sum(1) @ th
            else:
                Jq = c.J
            J_true.append(np.asarray(Jq, float))
            g = top2_gap(Jq)
            meta.append({"pid": q.pid, "draw": c.j, "kind_pool": c.kind, "group": c.group,
                         "twin_of": q.meta["twin_of"], "twin_variant": q.meta["twin_variant"],
                         "shared": c.group in seen, "true_gap": g, "gap_layer": gap_layer(g, self.eps),
                         "true_gap_dyn": c.delta, "gap_layer_dyn": c.layer, "pi_star": int(np.argmax(Jq)),
                         "cos_max": o["cos_max"], "cos_span": o["cos_span"]})
            seen.add(c.group)
        n = n_problems or len(cands)
        noise = STREAM_NOISE[stream] + (SHADOW_SEED_OFFSET if sibling else 0)
        env = make_env(self.seed, self.truth, noise, static=static)
        init = _initial_data(env, self.cfg["n0"], np.random.default_rng([self.seed, 2]))
        return LinStream(seed=self.seed, stream=stream, static=static, noise_seed=noise, sibling=sibling, env=env,
                         init_obs=init, problems=[c.problem for c in cands][:n], J=J_true[:n], harness=meta[:n],
                         lc=lc, theta=th.copy())

    def geom(self, static: bool):
        if static not in self._geom:
            self._geom[static] = LinGeom(self.lcs if static else self.lc, self.aspace, 1)
        return self._geom[static]

    def shared_fraction(self, stream: int = 0, static: bool = False) -> float:
        st_meta = self.stream_meta(stream, static)
        return sum(m["shared"] for m in st_meta) / len(st_meta)

    def stream_meta(self, stream: int, static: bool = False):
        sel = self.select(static)
        cands = [sel["chosen"][i] for i in self.order(stream, static)]
        seen, out = set(), []
        for c in cands:
            out.append({"pid": c.problem.pid, "shared": c.group in seen, "layer": c.layer, "kind": c.kind})
            seen.add(c.group)
        return out


@dataclass
class LinStream:
    seed: int
    stream: int
    static: bool
    noise_seed: int
    sibling: bool
    env: object                     # ground truth: harness only
    init_obs: list
    problems: list                  # public Problem objects
    J: list                         # harness: true values per problem
    harness: list                   # per problem: gap / layer / sharing / overlap (scoring only)
    lc: LinClass
    theta: np.ndarray = field(default=None)


# ================================================================================================ cell runner
def _rng(seed, noise, method):
    return np.random.default_rng([int(seed), int(noise), METHOD_CODE[method]])


def _public(builder, st):
    return PublicLin(st.lc, builder.aspace, st.problems, sigma=st.env.handle().known_constants()["sigma"],
                     eps=builder.eps, S=S_LIN, geom=builder.geom(st.static))


def sibling_run(builder, stream, method, tmax, static, n_problems):
    st = builder.stream(stream, sibling=True, static=static, n_problems=n_problems)
    env = st.env
    h = env.handle()
    pub = _public(builder, st)
    sw = ReuseSwitch("full", pub.make_set_factory(method), st.init_obs)
    rng = _rng(builder.seed, st.noise_seed, method)
    statuses = []
    for k, q in enumerate(st.problems):
        lr = sw.begin_problem(k, q.pid)
        res = solve(pub, method, k, sw, lr, h, rng, tmax)
        sw.end_problem(env.n_steps)
        statuses.append(res["status"])
    rows = [(o, pid) for o, pid in sw.real_ledger if pid is not None]
    pad_rng = np.random.default_rng([int(builder.seed), int(st.noise_seed), 7919])
    legal = pub.geom.legal

    def pad_fn(n):
        return [h.step(int(pad_rng.choice(legal))) for _ in range(n)]

    return rows, env, pad_fn, {"sibling_noise_seed": st.noise_seed, "sibling_steps": len(rows),
                               "sibling_statuses": statuses}


def run_lin_cell(builder: LinStreamBuilder, stream: int, method: str, arm: str, tmax: int = T_MAX_LIN,
                 static: bool = False, n_problems=None, orth_provider=None, sibling_cache=None, samples=None,
                 taus=LIN_TAUS) -> list:
    """All problems of one Lin stream for one (method, arm). Returns scored rows (harness)."""
    st = builder.stream(stream, static=static, n_problems=n_problems)
    pub = _public(builder, st)
    env = st.env
    h = env.handle()
    base, _ = parse_arm(arm)
    shadow, sib_info = None, {}
    if base == "vol":
        key = (builder.seed, stream, static, method, n_problems, tmax)
        if sibling_cache is not None and key in sibling_cache:
            shadow, _, sib_info = sibling_cache[key]
        else:
            rows_s, env_s, pad_fn, sib_info = sibling_run(builder, stream, method, tmax, static, n_problems)
            shadow = ShadowLedger(rows_s, [q.pid for q in st.problems], pad_fn=pad_fn)
            if sibling_cache is not None:
                sibling_cache[key] = (shadow, env_s, sib_info)
    sw = ReuseSwitch(arm, pub.make_set_factory(method), st.init_obs, shadow=shadow, orth_provider=orth_provider)
    rng = _rng(builder.seed, st.noise_seed, method)
    out = []
    for k, q in enumerate(st.problems):
        t0 = time.perf_counter()
        hm = st.harness[k]
        resets0 = env.n_resets
        lr = sw.begin_problem(k, q.pid, context={"stream": stream, "layer": "Lin-Static" if static else "E1-Lin"})
        billing_ok, err = True, None
        if lr is None:
            res = {"status": "ORTH_INFEASIBLE", "pi": None, "steps": 0, "extra": {}}
        else:
            res = solve(pub, method, k, sw, lr, h, rng, tmax)
        try:
            acc = sw.end_problem(env.n_steps)
        except BillingError as e:
            billing_ok, err = False, str(e)
            acc = sw.accounts[-1]
            sw._k = sw._pid = None
            sw._new = 0
        Jt = st.J[k]
        pi = res["pi"]
        cert = res["status"] == "CERTIFIED"
        regret = float(Jt.max() - Jt[pi]) if pi is not None else None
        row = {"instance": builder.seed, "stream": stream, "kind": "lin_quota", "noise_seed": st.noise_seed,
               "problem_index": k, "pid": q.pid, "method": method, "arm": arm,
               "layer": "Lin-Static" if static else "E1-Lin", "gap_layer": hm["gap_layer"], "true_gap": hm["true_gap"],
               "gap_layer_dyn": hm["gap_layer_dyn"], "shared": hm["shared"], "twin_of": hm["twin_of"],
               "twin_variant": hm["twin_variant"], "cos_max": hm["cos_max"], "cos_span": hm["cos_span"],
               "H": q.H, "n_policies": len(q.policies),
               "status": res["status"], "new_env_steps": int(res["steps"]), "replay_steps": int(acc.replay_steps),
               "replay_pad_steps": int(acc.replay_pad_steps), "n_rounds_total": int(acc.n_rounds_total),
               "n_rounds_billed": int(acc.n_rounds_billed), "env_n_steps": int(env.n_steps),
               "n_resets": int(env.n_resets - resets0), "billing_ok": billing_ok, "billing_error": err,
               "censored": not cert, "zero_cost": bool(cert and res["steps"] == 0), "certified_policy": pi,
               "true_regret": regret, "false_cert": bool(cert and regret is not None and regret > pub.eps),
               "orth_feasible": acc.orth_feasible, "prov_counts": acc.prov_counts, "eps": pub.eps,
               "delta": pub.delta, "tmax": tmax, "truncated_at_tmax": bool(res["status"] == "NEED_DATA"
                                                                           and res["steps"] >= tmax),
               "wall_clock_s": time.perf_counter() - t0}
        row.update(score_truncations(int(res["steps"]), cert, taus))
        row.update({f"x_{a}": b for a, b in res["extra"].items() if a != "wall_clock_s"})
        out.append(row)
        if samples is not None and k < 2:
            samples.append({"instance": builder.seed, "stream": stream, "static": static, "method": method,
                            "arm": arm, "problem": q.public_dict(), "J_true": [round(float(x), 6) for x in Jt],
                            "row": {kk: row[kk] for kk in ("status", "new_env_steps", "replay_steps", "gap_layer",
                                                           "shared", "certified_policy", "true_regret",
                                                           "prov_counts")}})
    if base == "vol":
        assert env.n_steps == sw.n_rounds_billed
        for r in out:
            r["sibling_steps"] = sib_info.get("sibling_steps")
            r["sibling_pad_pool"] = len(shadow.pad_pool)
    return out
