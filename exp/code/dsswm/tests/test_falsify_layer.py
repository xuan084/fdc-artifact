"""r2_setup_falsify unit tests: mixed numerator, regimes (public-only), ext plug-in, replay, m4, e-process,
Lin OOS family, versioned evidence."""
import ast
import dataclasses
import inspect
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from dsswm.core.provenance import EvidenceBoundaryError, Observation
from dsswm.evidence import regimes
from dsswm.evidence.ext_plugin import REGISTERED_DIRECTIONS, ExtPlugin
from dsswm.evidence.lr_set import SeqLRSet
from dsswm.evidence.mixed_lr import MixedLRSet, factorize, make_half_set, model_conflict_fallback
from dsswm.evidence.regimes import PUBLIC_FIELDS, PlaceboRegime, PublicFactor
from dsswm.evidence.replay import replay_check
from dsswm.evidence.versioned import DetectReset, VersionedJointLR
from dsswm.exact.nl_propagate import NLPropagator
from dsswm.models.nl_class import NLClass
from dsswm.streams.generator import DEFAULT_NL_S_GRID, make_nl_instance

torch.set_num_threads(1)
DELTA = 0.05


@pytest.fixture(scope="module")
def cls():
    ncl = NLClass(DEFAULT_NL_S_GRID, device="cpu")
    return ncl, ncl.torch_params()


def _setup(cls, seed, kappa="natural", K=3, noise=42):
    ncl, P = cls
    inst = make_nl_instance(seed, noise_seed=noise, nl_class=ncl, kappa_mode=kappa, K=K)
    prop = NLPropagator(2, 2, 2, inst.env.aspace, 1.0, 0.3, device="cpu")
    LT, EY = prop.tables(P)
    return inst, prop, LT, EY


def _stream(inst, n, seed):
    h = inst.env.handle()
    rng = np.random.default_rng(seed)
    obs = list(inst.init_obs)
    for _ in range(n):
        obs.append(h.step(int(rng.integers(h.aspace.n))))
    return obs


# ---------------------------------------------------------------- mixed numerator
def test_factorize_matches_propagator(cls):
    inst, prop, LT, _ = _setup(cls, 600)
    for o in _stream(inst, 50, 1):
        fac, const = factorize(prop, o)
        idx, c2 = prop.obs_factors(o)
        assert [f.col for f in fac] == idx and abs(const - c2) < 1e-12


def test_mixed_cost_bounds_and_pool_identity(cls):
    inst, prop, LT, _ = _setup(cls, 601, kappa="high")
    ms = MixedLRSet(prop, LT, DELTA)
    hs = make_half_set(prop, LT, DELTA)
    sl = SeqLRSet(prop, LT, DELTA)
    for s, o in enumerate(_stream(inst, 300, 2)):
        ms.update(o, problem_id=s // 40)
        hs.update(o, problem_id=s // 40)
        sl.update(o)
        assert ms.log_num() - ms.LN["pool"] >= -math.log(3) - 1e-12
        assert hs.log_num() - hs.LN["pool"] >= -math.log(2) - 1e-12
        assert abs(ms.LN["pool"] - sl.log_num) < 1e-8
    # log M_mixed(theta) - log M_pool(theta) >= -log 3 for every theta
    assert float((ms.log_ratio() - sl.log_ratio()).min()) >= -math.log(3) - 1e-9
    # mixed set contains the pool set shrunk by log 3 margin: theta in pool set at delta/3 => in mixed set at delta
    pool3 = sl.log_ratio() < math.log(1 / DELTA) - math.log(3)
    assert bool((~pool3 | ms.mask()).all())
    assert ms.falsification_scope()["components"] == ["reg", "ext"]


def test_inclass_no_false_alarm_small(cls):
    """Smoke version of the 1000-stream false-alarm test (the full count is in the pilot run)."""
    conflicts, covered = 0, 0
    for k in range(12):
        inst, prop, LT, _ = _setup(cls, 610 + k, noise=42 + k)
        ms = MixedLRSet(prop, LT, DELTA)
        for s, o in enumerate(_stream(inst, 200, 100 + k)):
            ms.update(o, problem_id=s // 40)
        conflicts += ms.conflict_round is not None
        covered += bool(ms.mask()[inst.truth["theta_index"]])
    assert conflicts <= 1 and covered >= 11


def test_rejects_unauthentic_observation(cls):
    inst, prop, LT, _ = _setup(cls, 600)
    o = inst.init_obs[0]
    fake = Observation(**{f.name: getattr(o, f.name) for f in dataclasses.fields(o) if f.name != "serial"},
                       serial=10 ** 9)
    ms = MixedLRSet(prop, LT, DELTA)
    with pytest.raises(EvidenceBoundaryError):
        ms.update(fake)


def test_static_learner_refuted_on_kappa_high(cls):
    """Power sanity: a static learner (gamma = lam = 0) on a kappa_high truth is refuted at zero cost."""
    ncl, P = cls
    inst, prop, _, _ = _setup(cls, 601, kappa="high")
    Ps = {k: v.clone() for k, v in P.items()}
    Ps["gamma"] = Ps["gamma"] * 0
    Ps["lam"] = Ps["lam"] * 0
    LTs, _ = prop.tables(Ps)
    ms = MixedLRSet(prop, LTs, DELTA)
    for s, o in enumerate(_stream(inst, 600, 1)):
        ms.update(o, problem_id=s // 40)
    assert ms.is_conflict()


def test_model_conflict_fallback_b2():
    rng = np.random.default_rng(0)
    mus = np.array([0.2, 0.5, 0.45])
    out = model_conflict_fallback(lambda k, n: np.clip(mus[k] + 0.1 * rng.standard_normal(n), 0, 1), 3, 4, 0.1,
                                  0.05, 1.0, 10 ** 6)
    assert out["fallback"] == "B2" and out["status"] == "CERTIFIED" and out["pi"] in (1, 2) and out["steps"] > 0


# ---------------------------------------------------------------- regimes: public information only
ALLOWED_NAMES = {"min", "int", "float", "self", "pf", "lab", "c", "u", "key", "_mix64", "x", "tot", "f", "acc",
                 "freqs", "seed", "PublicFactor", "dataclass"}


def _regime_nodes():
    src = Path(regimes.__file__).read_text()
    tree = ast.parse(src)
    fns = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in ("load_bucket", "problem_id_regime", "__call__"):
            fns.append(node)
    return tree, fns


def test_regime_functions_read_public_fields_only():
    tree, fns = _regime_nodes()
    assert len(fns) == 3
    for fn in fns:
        for node in ast.walk(fn):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "pf":
                assert node.attr in PUBLIC_FIELDS, f"{fn.name} reads non-public field {node.attr}"
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
                assert node.attr in ("seed", "cuts"), node.attr
            if isinstance(node, ast.Name):
                assert node.id in ALLOWED_NAMES, f"{fn.name} uses free name {node.id}"
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module]
            assert all(m in ("__future__", "dataclasses") for m in mods), mods
    assert tuple(f.name for f in dataclasses.fields(PublicFactor)) == PUBLIC_FIELDS
    assert "value" not in PUBLIC_FIELDS and "y_partner" not in PUBLIC_FIELDS


def test_placebo_frequencies_and_load_independence():
    pr = PlaceboRegime(seed=7, freqs=(0.5, 0.3, 0.2))
    labs = {0: [], 2: []}
    for t in range(20000):
        for load in (0, 2):
            labs[load].append(pr(PublicFactor("ret", -1, -1, 1, load, 0, None, t)))
    for load in (0, 2):
        fr = np.bincount(labs[load], minlength=3) / len(labs[load])
        assert np.allclose(fr, [0.5, 0.3, 0.2], atol=0.02)
    assert labs[0] == labs[2]           # the label does not depend on the load at all


# ---------------------------------------------------------------- ext plug-in
def test_ext_valid_density_and_predictable(cls):
    inst, prop, _, _ = _setup(cls, 602)
    ext = ExtPlugin(2, 2, 2, 2, 1.0)
    for o in _stream(inst, 120, 3):
        fac, _ = factorize(prop, o)
        a = ext.log_pred(fac)
        assert ext.log_pred(fac) == a                       # prediction does not move before update
        for f in fac:                                       # each factor's two outcomes sum to 1
            f1 = dataclasses.replace(f, value=1)
            f0 = dataclasses.replace(f, value=0)
            assert abs(math.exp(ext.log_pred([f1])) + math.exp(ext.log_pred([f0])) - 1) < 1e-9
        ext.update(fac)
    assert "time" not in REGISTERED_DIRECTIONS
    assert "t" not in inspect.signature(ExtPlugin._x_pair).parameters


def test_ext_detects_pair_synergy(cls):
    from dsswm.envs.nl import NLEnv
    ncl, P = cls
    inst, prop, LT, _ = _setup(cls, 603)
    t = inst.truth
    syn = 2.5 * np.array([[1.0, -1.0], [-1.0, 1.0]])
    env = NLEnv(t["alpha"], t["beta"], t["gamma"], t["tauL"], t["tauR"], [t["psi"]], t["lam"], seed=5, syn=syn)
    h = env.handle()
    ms = MixedLRSet(prop, LT, DELTA, components=("pool", "ext"))
    rng = np.random.default_rng(4)
    for s in range(1500):
        ms.update(h.step(int(rng.integers(h.aspace.n))))
        if ms.is_conflict():
            break
    assert ms.is_conflict()


# ---------------------------------------------------------------- replay
def test_replay_matches_running_set_and_no_interaction(cls):
    inst, prop, LT, _ = _setup(cls, 604)
    h = inst.env.handle()
    rng = np.random.default_rng(5)
    ledger = [(o, None) for o in inst.init_obs]
    ms = MixedLRSet(prop, LT, DELTA)
    for o, pid in ledger:
        ms.update(o, problem_id=pid)
    for s in range(150):
        o = h.step(int(rng.integers(h.aspace.n)))
        ledger.append((o, f"q{s // 50}"))
        ms.update(o, problem_id=f"q{s // 50}")
    r = replay_check(prop, LT, ledger, "q3", DELTA, env_handle=h)
    assert r["n_replayed"] == len(ledger) and r["conflict"] == ms.is_conflict()
    assert torch.equal(r["mask"], ms.mask())
    r2 = replay_check(prop, LT, ledger, "q3", DELTA, regime_fn=regimes.problem_id_regime, env_handle=h)
    assert r2["scope"]["regime"] == "problem_id_regime"


# ---------------------------------------------------------------- m4
def test_m4_drift_env():
    from dsswm.envs.mis import DriftNLEnv
    env = DriftNLEnv([0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [1, 1], [1, 1], [0.5], 0.3, drift_s=2.0, t_ref=100.0)
    h = env.handle()
    for _ in range(50):
        h.step(0)
    assert np.allclose(env.true_params_at(50)["alpha"], [1.0, 2.0])
    assert np.allclose(env._alpha, [0.0 + 2.0 * 49 / 100, 1.0 + 2.0 * 49 / 100])


def test_m4_calibration_reachable(cls):
    from dsswm.envs.mis import calibrate_m4, m4_schedule
    from dsswm.exact.nl_propagate import params_to_torch
    from dsswm.streams.generator import nl_class_max_jtable
    ncl, P = cls
    inst, prop, _, _ = _setup(cls, 605, kappa="high", K=3)
    J = [nl_class_max_jtable(q, prop, P) for q in inst.problems]
    base = inst.env.true_params()
    plans = [[prop.build_policy_plan(p, q.loads0, q.engaged0, q.H) for p in q.policies] for q in inst.problems]

    def J_true(k, shift):
        tp = dict(base)
        tp["alpha"] = base["alpha"] + shift
        LTt, EYt = prop.tables(params_to_torch(tp, device="cpu"))
        u = inst.problems[k].utility
        return np.array([float(prop._run_plan(pl, es, LTt, EYt, u.w, u.w_ret, u.c_q)[0]) for pl, es in plans[k]])

    cal = calibrate_m4(J, J_true, 2 * 0.02, m4_schedule(20, 3))
    assert cal["reached"] and cal["s"] > 0 and abs(cal["eta_min"] - 0.04) < 1e-3


# ---------------------------------------------------------------- e-process
def test_betting_eprocess_null_and_power():
    from dsswm.audit.eprocess import TwoSidedEProcess
    rng = np.random.default_rng(0)
    alarms = 0
    for _ in range(200):
        ep = TwoSidedEProcess(0.02, -1.0, 1.0)
        le = max(ep.update(float(x)) for x in rng.uniform(-1, 1, 400))
        alarms += le >= math.log(1 / DELTA)
    assert alarms <= 10
    ep = TwoSidedEProcess(0.02, -1.0, 1.0)
    le = max(ep.update(float(np.clip(x, -1, 1))) for x in rng.uniform(-1, 1, 3000) + 0.3)
    assert le >= math.log(1 / DELTA)


def test_eprocess_values_match_exact_J(cls):
    from dsswm.audit.eprocess import EProcessAudit, policy_value_tables
    from dsswm.streams.generator import nl_class_max_jtable
    ncl, P = cls
    inst, prop, LT, EY = _setup(cls, 606, K=1)
    q = inst.problems[0]
    J = nl_class_max_jtable(q, prop, P)
    ti = inst.truth["theta_index"]
    LTr, EYr = LT[ti].numpy(), EY[ti].numpy()
    s0 = prop.codec.encode(q.loads0, q.engaged0)
    for k, pi in enumerate(q.policies[:3]):
        V = policy_value_tables(prop, LTr, EYr, pi, q.H, q.utility)
        assert abs(V[0, s0] - J[ti, k]) < 1e-9
    aud = EProcessAudit(prop, LTr, EYr, q.policies[:2], q.H, q.utility, eta=0.04, delta_audit=0.05)
    res = aud.run(inst.env.handle(), 400)                    # model == truth: no alarm expected
    assert res["status"] == "PASS" and res["steps"] == 400


# ---------------------------------------------------------------- Lin OOS family
def test_lin_oos_truth_lp_and_families():
    from dsswm.certify.lin_closed import range_projector
    from dsswm.models.lin_class import LinClass
    from dsswm.streams.lin_oos import make_oos_instance, prior_box, truth_vstar
    for fam in ("squeeze", "newpart"):
        inst, leg, prior = make_oos_instance(fam, 600, K=3)
        env = inst.env
        assert all(leg.is_legal(env.aspace, o.action) for o in inst.init_obs)
        lc = LinClass(env.L, env.R, env.aspace.nb, nmax=env.nmax)
        lo, hi = prior_box(lc, prior)
        theta = env.true_theta_vector()
        PR = range_projector(leg.library(lc))
        q = inst.problems[0]
        Z = np.stack([lc.z(p, q.loads0, q.H, q.utility, env.aspace) for p in q.policies])
        tl = truth_vstar(Z, theta, PR, lo, hi, 0.05, return_argmax=True)
        for (a, b), x in tl["argmax"].items():          # LP optimisers are feasible: in box and on the fibre
            assert np.all(x >= lo - 1e-7) and np.all(x <= hi + 1e-7)
            assert np.abs(PR @ (x - theta)).max() < 1e-6
            assert abs((Z[b] - Z[a]) @ x - tl["W"][a, b]) < 1e-7
        # theta* itself is feasible, so V* >= the true minimax regret of theta* alone
        Jt = Z @ theta
        assert tl["V_star"] >= min(Jt.max() - Jt) - 1e-9
    if True:
        _, leg, _ = make_oos_instance("newpart", 600, K=1)
        assert leg.excluded_right == (3,)


# ---------------------------------------------------------------- versioned evidence
def test_versioned_index_mapping(cls):
    ncl, P = cls
    inst, prop, LT, _ = _setup(cls, 607)
    vj = VersionedJointLR(ncl, prop, LT, DELTA)
    assert vj.V == 1 + 2 * 12 + 2 * 4
    rng = np.random.default_rng(0)
    names = ("alpha", "gamma", "tauL")
    for _ in range(50):
        b = int(rng.integers(vj.B))
        c = int(rng.integers(1, vj.V))
        nb_ = int(vj.newidx_np[b, c])
        s = int(vj.slot_of[c])
        pa, pb = ncl.params_at(b), ncl.params_at(nb_)
        for key in ("alpha", "gamma", "tauL"):
            diff = np.flatnonzero(pa[key] != pb[key])
            assert set(diff.tolist()) <= ({s} if s < 2 else set())
        for key in ("beta", "tauR"):
            diff = np.flatnonzero(pa[key] != pb[key])
            assert set(diff.tolist()) <= ({s - 2} if s >= 2 else set())
        assert pa["lam"] == pb["lam"] and np.all(pa["psi_left"] == pb["psi_left"])
    assert names


def test_versioned_covers_true_change_and_detect_reset(cls):
    from dsswm.envs.nl import NLEnv
    ncl, P = cls
    inst, prop, LT, _ = _setup(cls, 608, kappa="high")
    t = inst.truth
    vj = VersionedJointLR(ncl, prop, LT, DELTA)
    for o in _stream(inst, 300, 9):
        vj.update(o)
    vj.start_new_epoch()
    # change right participant 1's beta (in-class, k = 1)
    beta_new = list(t["beta"])
    beta_new[1] = -beta_new[1]
    env2 = NLEnv(t["alpha"], beta_new, t["gamma"], t["tauL"], t["tauR"], [t["psi"]], t["lam"], seed=99)
    h2 = env2.handle()
    rng = np.random.default_rng(10)
    dr = DetectReset(lambda: MixedLRSet(prop, LT, DELTA))
    for _ in range(300):
        o = h2.step(int(rng.integers(h2.aspace.n)))
        vj.update(o)
        dr.update(o)
    ti_old = t["theta_index"]
    ti_new = ncl.index_of(tuple(t["alpha"]), tuple(beta_new), tuple(t["gamma"]), tuple(t["tauL"]),
                          tuple(t["tauR"]), t["psi"], t["lam"])
    assert vj.contains(ti_old, ti_new)
    assert bool(vj.new_mask()[ti_new])
    assert vj.size() >= 1
    assert dr.n_rounds == 300
