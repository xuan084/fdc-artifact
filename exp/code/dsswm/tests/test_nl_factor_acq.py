"""Factorised observation model used by the E1-NL acquisition rules (DDA / IG / Fisher designs)."""
import numpy as np
import torch
from scipy.special import logsumexp

from dsswm.acquire.nl_factor_acq import FactorModel, perstep_jtable, theta_vector, vector_to_params
from dsswm.core.actions import ActionSpace
from dsswm.exact.nl_propagate import NLPropagator
from dsswm.models.nl_class import NLClass, NLGrid
from dsswm.streams.generator import make_nl_instance
from dsswm.streams.utilities import Utility

GRID = NLGrid(L=2, R=2, alpha=(-1.0, 1.0), gamma=(0.0, 0.75), tauL=(0.5,), beta=(0.0,), tauR=(0.5, 1.5),
              psi=(0.0, 1.0), lam=(0.0, 0.5))


def _setup():
    ncl = NLClass(GRID, device="cpu")
    prop = NLPropagator(2, 2, 2, ActionSpace(2, 2, 1, (1,)), 1.0, 0.3, device="cpu")
    params = ncl.torch_params()
    LT, _ = prop.tables(params)
    fm = FactorModel(prop)
    return ncl, prop, params, LT, fm


def test_rows_normalised_and_match_real_loglik():
    ncl, prop, params, LT, fm = _setup()
    FT = fm.factor_table(LT.numpy())
    for k in (0, ncl.B - 1):
        assert np.abs(logsumexp(fm.row_logp(FT[k]), 1)).max() < 1e-10
    inst = make_nl_instance(3, grid=GRID, nl_class=ncl)
    for o in inst.init_obs:
        r = fm.row(prop.codec.encode(o.loads, o.engaged), o.action)
        ll = prop.loglik(LT, [o])[:, 0].numpy()
        lp = FT.reshape(FT.shape[0], -1)[:, fm.FLAT[r]].sum(-1)
        nxt = prop.codec.encode(o.next_loads, o.next_engaged)
        errs = [np.abs(lp[:, i] - ll).max() for i in range(fm.NOUT[r]) if fm.NEXT[r, i] == nxt]
        assert min(errs) < 1e-9


def test_fisher_closed_form_matches_numeric():
    ncl, prop, _, _, fm = _setup()
    v = theta_vector(ncl.params_at(5))
    d, h = len(v), 1e-5
    V = np.vstack([v] + [v + h * np.eye(d)[k] for k in range(d)] + [v - h * np.eye(d)[k] for k in range(d)])
    p = {k: torch.as_tensor(x) for k, x in vector_to_params(V, 2, 2, 2).items()}
    F = fm.factor_table(prop.tables(p)[0].numpy())
    sc = (F[1:d + 1] - F[d + 1:]) / (2 * h)
    pr = np.exp(F[0])
    sc = np.where(pr[None] > 0, sc, 0)
    num = np.einsum("fo,afo,bfo->fab", pr, sc, sc)
    assert np.abs(num - fm.factor_fisher(F[0])).max() < 1e-7


def test_kl_nonnegative_and_zero_on_self():
    _, _, _, LT, fm = _setup()
    FT = fm.factor_table(LT.numpy())
    kl = fm.row_kl(FT[0], FT[[0, 1, 7]])
    assert np.abs(kl[0]).max() < 1e-12 and kl.min() > -1e-12


def test_perstep_sums_to_jtable():
    ncl, prop, params, _, _ = _setup()
    q = make_nl_instance(4, grid=GRID, nl_class=ncl).problems[0]
    pj = perstep_jtable(prop, params, q.policies, q.loads0, q.engaged0, q.H, q.utility.w, q.utility.w_ret)
    raw = prop.j_table(params, q.policies, q.loads0, q.engaged0, q.H, Utility(q.utility.w, q.utility.w_ret, 1.0))
    assert np.abs(pj.sum(-1) - raw).max() < 1e-10
