"""Round-3 harness: run one (instance, stream, kind, method, arm) through the reuse switch and score it.
Harness side (reads theta* for scoring and builds the sibling platform); learner code never imports it.

CRN: every (method, arm) cell of one (instance, stream, kind) runs on a fresh copy of the same platform (same noise
seed, same initial data, same problem order) and uses the acquisition rng [seed, noise_seed, method_code], so arms of
one method differ only in the evidence the switch hands out.
vol arm: the sibling platform (noise seed + 7919, same theta*, same problems) is run by the SAME method in the `full`
arm over the whole stream; its new-data rows tagged by problem id form the shadow ledger. If the sibling prefix is
shorter than N_k, it is padded with uniform random legal steps on the same sibling platform after its stream ended
(rng [seed, noise_sib, 7919]); padding rows are tagged replay_vol_pad and counted.
"""
from __future__ import annotations

import time

import numpy as np

from ..baselines.switched_nl import PublicNL, solve
from ..evidence.reuse_switch import (BillingError, ReuseSwitch, ShadowLedger, parse_arm, score_truncations, T_MAX,
                                     TAUS)

METHOD_CODE = {"JPC": 101, "B3": 303, "B8": 808, "LR-chi2-grid": 2100}


def _rng(seed, noise, method):
    return np.random.default_rng([int(seed), int(noise), METHOD_CODE[method]])


def sibling_run(builder, pub: PublicNL, stream: int, kind: str, method: str, tmax: int, n_problems=None):
    """Same method, `full` arm, on the sibling platform. Returns (rows [(obs, pid)], env, pad_fn, summary)."""
    st = builder.stream(stream, kind, sibling=True, n_problems=n_problems)
    env = st.env
    h = env.handle()
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
    legal = np.arange(env.aspace.n)

    def pad_fn(n):
        return [h.step(int(pad_rng.choice(legal))) for _ in range(n)]

    return rows, env, pad_fn, {"sibling_noise_seed": st.noise_seed, "sibling_steps": len(rows),
                               "sibling_statuses": statuses}


def run_cell(builder, pub_base: PublicNL, stream: int, kind: str, method: str, arm: str, tmax: int = T_MAX,
             n_problems=None, orth_provider=None, sibling_cache=None, overlap=None, samples=None) -> list:
    """All problems of one stream for one (method, arm). Returns scored result rows (harness)."""
    st = builder.stream(stream, kind, n_problems=n_problems)
    pub = pub_base.with_problems(st.problems, st.J)
    env = st.env
    h = env.handle()
    base, m = parse_arm(arm)
    shadow, sib_info = None, {}
    if base == "vol":
        key = (builder.seed, stream, kind, method, n_problems)
        if sibling_cache is not None and key in sibling_cache:
            shadow, env_s, sib_info = sibling_cache[key]
        else:
            rows_s, env_s, pad_fn, sib_info = sibling_run(builder, pub, stream, kind, method, tmax, n_problems)
            shadow = ShadowLedger(rows_s, [q.pid for q in st.problems], pad_fn=pad_fn)
            if sibling_cache is not None:
                sibling_cache[key] = (shadow, env_s, sib_info)
    sw = ReuseSwitch(arm, pub.make_set_factory(method), st.init_obs, shadow=shadow, orth_provider=orth_provider)
    rng = _rng(builder.seed, st.noise_seed, method)
    out = []
    for k, q in enumerate(st.problems):
        t0 = time.perf_counter()
        hm = st.harness[k]
        lr = sw.begin_problem(k, q.pid, context={"stream": stream, "kind": kind})
        billing_ok, err = True, None
        if lr is None:
            res = {"status": "ORTH_INFEASIBLE", "pi": None, "steps": 0, "extra": {}}
        else:
            res = solve(pub, method, k, sw, lr, h, rng, tmax)
        try:
            acc = sw.end_problem(env.n_steps)
        except BillingError as e:                               # recorded, never silently dropped
            billing_ok, err = False, str(e)
            acc = sw.accounts[-1]
            sw._k = sw._pid = None
            sw._new = 0
        J = st.J[k]
        Jt = J[st.theta_index]
        pi = res["pi"]
        cert = res["status"] == "CERTIFIED"
        regret = float(Jt.max() - Jt[pi]) if pi is not None else None
        row = {"instance": builder.seed, "stream": stream, "kind": kind, "noise_seed": st.noise_seed,
               "problem_index": k, "pid": q.pid, "method": method, "arm": arm, "layer": "NL-R0",
               "gap_layer": hm["gap_layer"], "true_gap": hm["true_gap"], "true_gap_orig": hm["true_gap_orig"],
               "status": res["status"], "new_env_steps": int(res["steps"]), "replay_steps": int(acc.replay_steps),
               "replay_pad_steps": int(acc.replay_pad_steps), "n_rounds_total": int(acc.n_rounds_total),
               "n_rounds_billed": int(acc.n_rounds_billed), "env_n_steps": int(env.n_steps),
               "billing_ok": billing_ok, "billing_error": err,
               "censored": not cert, "zero_cost": bool(cert and res["steps"] == 0), "certified_policy": pi,
               "true_regret": regret, "false_cert": bool(cert and regret is not None and regret > pub.eps),
               "orth_feasible": acc.orth_feasible, "prov_counts": acc.prov_counts,
               "cos_overlap": None if overlap is None else overlap[k], "eps": pub.eps, "delta": pub.delta,
               "tmax": tmax, "wall_clock_s": time.perf_counter() - t0}
        row.update(score_truncations(int(res["steps"]), cert, TAUS))
        row.update({f"x_{a}": b for a, b in res["extra"].items() if a != "wall_clock_s"})
        out.append(row)
        if samples is not None and k < 2:
            samples.append({"instance": builder.seed, "stream": stream, "kind": kind, "method": method, "arm": arm,
                            "problem": q.public_dict(), "J_true": [round(float(x), 6) for x in Jt],
                            "row": {kk: row[kk] for kk in ("status", "new_env_steps", "replay_steps",
                                                           "certified_policy", "true_regret", "prov_counts")}})
    if base == "vol":
        # the shadow ledger must not have touched the real platform; sibling steps are accounted separately
        assert env.n_steps == sw.n_rounds_billed
        for r in out:
            r["sibling_steps"] = sib_info.get("sibling_steps")
            r["sibling_pad_pool"] = len(shadow.pad_pool)
    return out
