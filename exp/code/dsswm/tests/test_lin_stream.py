"""Round-3 E1-Lin stream, Lin baselines (RAGE / XY-static / G-opt / B1eb), JPC-Lin, Lin-Static (r3_setup_lin_stream)."""
from collections import Counter

import numpy as np
import pytest

from dsswm.baselines.lin_rage import EPS_LIN, LIN_TAUS, METHODS, S_LIN, PublicLin, TrialStore
from dsswm.certify.enum_exact import sup_linear_enum
from dsswm.certify.lin_closed import certify_lin
from dsswm.envs.lin import StaticLinEnv
from dsswm.evidence.ellipsoid import EllipsoidSet
from dsswm.streams.generator import LIN_DEFAULTS, make_lin_instance
from dsswm.streams.lin_stream import K_STREAM, LinStreamBuilder, gap_layer, n_shared, run_lin_cell, top2_gap

DEV = list(range(740, 750))
ARMS = ("off", "vol", "orth", "ev1", "ev(5)", "full")


@pytest.fixture(scope="module")
def b740():
    return LinStreamBuilder(740)


def _sample_in_ellipsoid(ell, n, rng):
    L = np.linalg.cholesky(np.linalg.inv(ell.V))
    u = rng.standard_normal((n, ell.d))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    r = rng.random(n) ** (1 / ell.d)
    return ell.theta_hat() + ell.sqrt_beta() * (u * r[:, None]) @ L.T


def closed_form_error(builder, stream=0, n_extra=200, seed=0):
    """max |closed-form sup - enumerated sup| over every pairwise candidate difference of every stream problem."""
    st = builder.stream(stream)
    ell = EllipsoidSet(st.lc, sigma=LIN_DEFAULTS["sigma"], delta=0.05, S=S_LIN)
    for o in st.init_obs:
        ell.update(o)
    h = st.env.handle()
    rng = np.random.default_rng(seed)
    legal = [a for a in range(h.aspace.n) if h.aspace.max_level(a) <= 1]
    for _ in range(n_extra):
        ell.update(h.step(int(rng.choice(legal))))
    pub = PublicLin(st.lc, builder.aspace, st.problems, sigma=LIN_DEFAULTS["sigma"])
    err, err_cert = 0.0, 0.0
    w, U = np.linalg.eigh(ell.V)
    for Z in pub.Z:
        D = (Z[None] - Z[:, None]).reshape(-1, Z.shape[1])
        D = D[np.linalg.norm(D, axis=1) > 0]
        closed = ell.sup_linear(D)
        pts = np.vstack([_sample_in_ellipsoid(ell, 2000, rng)] + [ell.argsup_linear(d)[None] for d in D])
        assert all(ell.contains(p) for p in pts[-len(D):])
        enum = sup_linear_enum(pts, D)
        indep = D @ ell.theta_hat() + ell.sqrt_beta() * np.linalg.norm((D @ U) / np.sqrt(w), axis=1)
        err = max(err, float(np.abs(enum - closed).max()), float(np.abs(indep - closed).max()))
        c = certify_lin(Z, ell, EPS_LIN)
        Dk = Z - Z[c["pi_hat"]]
        ptsk = np.vstack([pts] + [ell.argsup_linear(d)[None] for d in Dk if np.abs(d).max() > 0])
        en = sup_linear_enum(ptsk, Dk)
        en[c["pi_hat"]] = 0.0
        err_cert = max(err_cert, abs(max(float(en.max()), 0.0) - c["r_bar"]))
    return max(err, err_cert)


# ------------------------------------------------------------------------------------------------ closed form
def test_ellipsoid_closed_form_equals_enumeration(b740):
    assert closed_form_error(b740) <= 1e-9


# ------------------------------------------------------------------------------------------------ stream
def test_quota_and_shared_fraction():
    fr, n_fail = [], 0
    for s in DEV:
        b = LinStreamBuilder(s)
        sel = b.select()
        if sel["quota_fail"]:                         # replaced by the next reserve seed (fill_instances)
            n_fail += 1
            assert sum(sel["layer_counts"].values()) < K_STREAM and sel["n_draws"] == 4000
            continue
        assert sel["layer_counts"] == {"tie": 5, "near": 5, "clear": 5}
        for c in sel["chosen"]:
            assert gap_layer(top2_gap(c.Zt.sum(1) @ b.theta)) == c.layer
        for t in range(3):
            st = b.stream(t)
            assert len(st.problems) == K_STREAM
            sh = [m["shared"] for m in st.harness]
            assert sum(sh) == n_shared(s)
            fr.append(sum(sh) / K_STREAM)
            for k, m in enumerate(st.harness):          # shared <=> same (s0, Pi) as an EARLIER problem
                q = st.problems[k]
                prev = [p for p in st.problems[:k] if np.array_equal(p.loads0, q.loads0)
                        and [id(x) for x in p.policies] == [id(x) for x in q.policies]]
                assert m["shared"] == bool(prev)
            assert [m["gap_layer"] for m in st.harness].count("tie") == 5
    assert n_fail <= 3
    assert abs(np.mean(fr) - 0.30) <= 0.05
    assert all(abs(f - 0.30) <= 0.05 for f in fr)
    from dsswm.streams.gap_quota import fill_instances
    seeds, log = fill_instances(DEV, lambda s: not LinStreamBuilder(s).select()["quota_fail"], reserve_start=750)
    assert len(seeds) == len(DEV) and len(log) == n_fail


def test_stream_is_deterministic_and_crn(b740):
    b2 = LinStreamBuilder(740)
    s1, s2 = b740.stream(1), b2.stream(1)
    assert [q.pid for q in s1.problems] == [q.pid for q in s2.problems]
    assert [o.outcomes for o in s1.init_obs] == [o.outcomes for o in s2.init_obs]
    sib = b740.stream(1, sibling=True)
    assert sib.noise_seed == s1.noise_seed + 7919
    assert np.allclose(sib.theta, s1.theta)
    assert [o.outcomes for o in sib.init_obs] != [o.outcomes for o in s1.init_obs]


def test_theta_matches_round0_generator():
    for s in (740, 10200):
        assert np.allclose(LinStreamBuilder(s).theta, make_lin_instance(s).env.true_theta_vector())


def test_overlap_records(b740):
    st = b740.stream(0)
    assert st.harness[0]["cos_max"] is None and st.harness[0]["cos_span"] is None
    for m in st.harness[1:]:
        assert 0.0 <= m["cos_max"] <= m["cos_span"] + 1e-12 <= 1.0 + 1e-9


# ------------------------------------------------------------------------------------------------ billing, every arm
def _orth_provider(builder):
    sib = builder.stream(0, sibling=True)
    h = sib.env.handle()
    rng = np.random.default_rng(3)

    def prov(k, pid, n_rows, ctx):
        if k == 2:
            return None                                     # infeasible hook path
        return [h.step(int(rng.integers(h.aspace.n))) for _ in range(n_rows)]
    return prov


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("arm", ARMS)
def test_billing_every_method_and_arm(b740, method, arm):
    rows = run_lin_cell(b740, 0, method, arm, tmax=300, n_problems=3,
                        orth_provider=_orth_provider(b740) if arm == "orth" else None)
    n0 = LIN_DEFAULTS["n0"]
    assert len(rows) == 3
    cum = 0
    for r in rows:
        assert r["billing_ok"], r["billing_error"]
        assert r["env_n_steps"] == r["n_rounds_billed"]
        assert r["new_env_steps"] <= 300
        cum += r["new_env_steps"]
        assert r["n_rounds_billed"] == n0 + cum
        if arm == "off":
            assert r["replay_steps"] == 0 and r["n_rounds_total"] == n0 + r["new_env_steps"]
        elif arm == "vol" or (arm == "orth" and r["status"] != "ORTH_INFEASIBLE"):
            assert r["replay_steps"] == r["n_rounds_billed"] - n0 - r["new_env_steps"]
        elif arm in ("ev1", "ev(5)", "full"):
            assert r["replay_steps"] == 0 and r["n_rounds_total"] == r["n_rounds_billed"]
        if arm.startswith("ev") and r["status"] == "CERTIFIED":
            m = 1 if arm == "ev1" else 5
            assert r["new_env_steps"] >= m
        if arm == "orth" and r["problem_index"] == 2:
            assert r["status"] == "ORTH_INFEASIBLE" and r["new_env_steps"] == 0
        if method == "B1eb":
            assert r["new_env_steps"] % r["H"] == 0
        else:
            assert r["n_resets"] == 0
    if arm == "vol":
        assert rows[0]["replay_steps"] == 0 and rows[-1]["sibling_steps"] is not None


def test_tmax_truncation(b740):
    rows = run_lin_cell(b740, 0, "G-opt", "off", tmax=40, n_problems=3)
    for r in rows:
        if r["status"] != "CERTIFIED":
            assert r["status"] == "NEED_DATA" and r["new_env_steps"] == 40 and r["truncated_at_tmax"]
            assert all(r[f"cost_tau{t}"] == t for t in LIN_TAUS)
    assert any(r["truncated_at_tmax"] for r in rows)
    rb = run_lin_cell(b740, 0, "B1eb", "off", tmax=50, n_problems=2)
    assert all(r["new_env_steps"] <= 50 for r in rb)


def test_cell_is_reproducible(b740):
    a = run_lin_cell(b740, 0, "JPC-Lin", "ev1", tmax=500, n_problems=2)
    b = run_lin_cell(LinStreamBuilder(740), 0, "JPC-Lin", "ev1", tmax=500, n_problems=2)
    assert [(r["status"], r["new_env_steps"], r["certified_policy"]) for r in a] == \
        [(r["status"], r["new_env_steps"], r["certified_policy"]) for r in b]


def test_b1eb_reuses_same_s0_pi_trials():
    """(s0, Pi) reuse only under ledger arms: a shared problem sees trials from its earlier twin."""
    for s in DEV:
        b = LinStreamBuilder(s)
        if b.select()["quota_fail"]:
            continue
        st = b.stream(0)
        # a trial of length H' >= H re-scores a horizon-H twin (prefix); a shorter earlier twin cannot be reused
        ks = [k for k, m in enumerate(st.harness) if m["shared"] and any(
            st.harness[j]["group"] == m["group"] and st.problems[j].H >= st.problems[k].H for j in range(k))]
        if ks and ks[0] <= 7:
            n = ks[0] + 1
            full = run_lin_cell(b, 0, "B1eb", "full", tmax=4000, n_problems=n)
            off = run_lin_cell(b, 0, "B1eb", "off", tmax=4000, n_problems=n)
            assert all(r["x_n_reused_trials"] == 0 for r in off)
            assert full[-1]["x_n_reused_trials"] > 0
            return
    pytest.skip("no early shared problem on dev seeds")


def test_trialstore_ignores_untagged_rows():
    inst = make_lin_instance(5, K=0)
    ts = TrialStore()
    for o in inst.init_obs:
        ts.update(o)
    assert ts.n_rounds == len(inst.init_obs) and ts.trials() == []


# ------------------------------------------------------------------------------------------------ Lin-Static
def test_lin_static_stream(b740):
    st = b740.stream(0, static=True)
    assert isinstance(st.env, StaticLinEnv) and st.lc.static and st.lc.d == b740.lc.d
    assert Counter(m["gap_layer"] for m in st.harness) == {"tie": 5, "near": 5, "clear": 5}   # own static quota
    for m, J in zip(st.harness, st.J):
        assert abs(m["true_gap"] - top2_gap(J)) < 1e-15
    assert sum(m["shared"] for m in st.harness) == n_shared(740)
    diag = b740.stream(0, static=True, same_problems=True)
    assert [q.pid for q in diag.problems] == [q.pid for q in b740.stream(0).problems]
    assert not set(q.pid for q in diag.problems) & set(q.pid for q in st.problems)
    gsl = st.lc.slices()["gamma"]
    pub = PublicLin(st.lc, b740.aspace, st.problems, sigma=1.5)
    assert all(np.abs(Z[:, gsl]).max() == 0 for Z in pub.Z)
    for arm in ("off", "ev1", "full"):
        rows = run_lin_cell(b740, 0, "JPC-Lin", arm, static=True, tmax=300, n_problems=2)
        assert all(r["billing_ok"] and r["layer"] == "Lin-Static" for r in rows)


def test_no_truth_in_learner_module():
    import ast
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "baselines" / "lin_rage.py").read_text()
    names = [n.module or "" for n in ast.walk(ast.parse(src)) if isinstance(n, ast.ImportFrom)]
    assert not any("envs" in m or "lin_stream" in m or "generator" in m for m in names)
