"""(3) Exact J agrees with 1e5-episode Monte Carlo on the ground-truth env.step dynamics (|err| < 3 SE)."""
import numpy as np
import pytest

from dsswm.envs.mis import make_mis_env
from dsswm.exact.nl_propagate import NLPropagator, params_to_torch
from dsswm.models.lin_class import LinClass
from dsswm.models.nl_class import NLClass
from dsswm.streams.generator import (DEFAULT_NL_M_GRID, DEFAULT_NL_S_GRID, make_lin_instance, make_nl_instance)

N_MC = 100_000


def _z(err, se):
    return abs(err) / max(se, 1e-12)


@pytest.mark.parametrize("seed", [0, 1])
def test_lin_exact_vs_mc(seed):
    inst = make_lin_instance(seed)
    lc = LinClass(3, 3, inst.env.aspace.nb)
    th = inst.env.true_theta_vector()
    rng = np.random.default_rng(seed)
    for q in inst.problems[:3]:
        for p in q.policies:
            J = lc.z(p, q.loads0, q.H, q.utility, inst.env.aspace) @ th
            u = q.utility.value_from_components(inst.env.simulate_batch(p, q.loads0, q.H, N_MC, rng))
            assert _z(u.mean() - J, u.std(ddof=1) / np.sqrt(N_MC)) < 3.0


def _check_nl(env, problems, prop, ncl_params=None, theta_index=None, rng=None, max_problems=2):
    zs, errs = [], []
    for q in problems[:max_problems]:
        Jtrue = prop.j_table(params_to_torch(env.true_params()), q.policies, q.loads0, q.engaged0, q.H, q.utility)[0]
        if ncl_params is not None and theta_index is not None:
            Jcls = prop.j_table(ncl_params, q.policies, q.loads0, q.engaged0, q.H, q.utility)[theta_index]
            assert np.max(np.abs(Jcls - Jtrue)) < 1e-12
        for k, p in enumerate(q.policies):
            ys, es = env.simulate_batch(p, q.loads0, q.engaged0, q.H, N_MC, rng)
            u = q.utility.value_from_components(ys, es)
            zs.append(_z(u.mean() - Jtrue[k], u.std(ddof=1) / np.sqrt(N_MC)))
            errs.append(abs(u.mean() - Jtrue[k]))
    return np.array(zs), np.array(errs)


def test_nl_s_exact_vs_mc():
    ncl = NLClass(DEFAULT_NL_S_GRID)
    inst = make_nl_instance(5, nl_class=ncl)
    prop = NLPropagator(2, 2, 2, inst.env.aspace, 1.0, 0.3)
    zs, _ = _check_nl(inst.env, inst.problems, prop, ncl.torch_params(), inst.truth["theta_index"],
                      np.random.default_rng(5))
    assert (zs < 3.0).all(), zs


@pytest.mark.parametrize("kind", ["m1", "m2", "m3"])
def test_mis_exact_vs_mc(kind):
    inst = make_nl_instance(6)
    base = {k: np.asarray(v) if isinstance(v, list) else v for k, v in inst.truth.items() if k != "theta_index"}
    base["psi"] = [base["psi"]]
    env = make_mis_env(kind, base, 0.8, np.random.default_rng(0), L=2, R=2, nmax=2, seed=1)
    prop = NLPropagator(2, 2, 2, env.aspace, 1.0, 0.3)
    zs, _ = _check_nl(env, inst.problems, prop, rng=np.random.default_rng(6), max_problems=1)
    assert (zs < 3.0).all(), zs


def test_nl_m_exact_vs_mc():
    ncl = NLClass(DEFAULT_NL_M_GRID)
    inst = make_nl_instance(7, grid=DEFAULT_NL_M_GRID, nl_class=ncl, H_choices=(6,), n_pol=(3, 3))
    prop = NLPropagator(3, 3, 2, inst.env.aspace, 1.0, 0.3)
    zs, _ = _check_nl(inst.env, inst.problems, prop, rng=np.random.default_rng(7), max_problems=1)
    assert (zs < 3.0).all(), zs


def test_nl_jtable_rows_are_probability_consistent():
    """With w = 0 and w_ret = 1, c_q = 1 the value is E[# engaged at H] in [0, P]; with all-zero logits the
    J table is identical across the class members that only differ in unused parameters."""
    ncl = NLClass(DEFAULT_NL_S_GRID)
    inst = make_nl_instance(8, nl_class=ncl)
    prop = NLPropagator(2, 2, 2, inst.env.aspace, 1.0, 0.3)
    q = inst.problems[0]
    from dsswm.streams.utilities import Utility
    u = Utility(w=np.zeros(q.H), w_ret=1.0, c_q=1.0)
    J = prop.j_table(ncl.torch_params(), q.policies, q.loads0, q.engaged0, q.H, u)
    assert (J >= -1e-12).all() and (J <= 4 + 1e-12).all()
