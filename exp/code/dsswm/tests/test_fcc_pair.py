"""Pair-difference (PDL) certificate for FCC: validity against an independent forward evaluator of the SHARED-parameter
model, exact cancellation for identical policies, never wider than the independent difference rule."""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from dsswm.certify.fcc import joint_plan, pair_difference_bounds
from dsswm.tests.test_fcc import ConstPolicy, TablePolicy, forward_J, random_instance  # noqa: F401


class PerturbedPolicy:
    """Copy of `base` except at a hash-selected fraction `frac` of (t, state) keys (shares many actions with base)."""

    def __init__(self, base, seed, frac=0.3):
        self.base, self.seed, self.frac = base, seed, frac
        self.alt = TablePolicy(base.aspace, seed + 99991, max_level=1)

    def act(self, t, loads, engaged):
        key = (self.seed, t, tuple(int(x) for x in loads), tuple(int(x) for x in engaged))
        u = (abs(hash(key)) % 10007) / 10007.0
        return self.alt.act(t, loads, engaged) if u < self.frac else self.base.act(t, loads, engaged)


def _check(seed, frac, n_points=3000):
    solver, pol, loads0, eng0, H, util, lo, hi, rng = random_instance(seed)
    pb = PerturbedPolicy(pol, seed, frac)
    dl, dh = pair_difference_bounds(solver, pol, pb, loads0, eng0, H, util, lo, hi)
    pa_plan = solver.plan(pol, loads0, eng0, H)
    pb_plan = solver.plan(pb, loads0, eng0, H)
    al, ah = solver.solve(pa_plan, util.w, util.w_ret, util.c_q, lo, hi)
    bl, bh = solver.solve(pb_plan, util.w, util.w_ret, util.c_q, lo, hi)
    pts = lo + (hi - lo) * rng.random((n_points, solver.cells.n))
    vert = np.where(rng.random((1024, solver.cells.n)) < 0.5, lo, hi)
    th = np.concatenate([pts, vert])
    D = forward_J(solver, pol, loads0, eng0, H, util, th) - forward_J(solver, pb, loads0, eng0, H, util, th)
    return dl, dh, D, (al - bh, ah - bl)


@pytest.mark.parametrize("block", range(5))
def test_pair_bound_valid_random(block):
    for s in range(block * 20, block * 20 + 20):
        for frac in (0.0, 0.3, 1.0):
            dl, dh, D, (il, ih) = _check(s, frac)
            assert dl <= D.min() and dh >= D.max(), (s, frac, dl, D.min(), D.max(), dh)
            assert dh <= ih + 1e-12 and dl >= il - 1e-12


def test_identical_policies_cancel_exactly():
    for s in range(20):
        dl, dh, D, _ = _check(s, 0.0, n_points=200)
        assert abs(dl) < 1e-9 and abs(dh) < 1e-9
        assert np.abs(D).max() < 1e-12


def test_joint_plan_same_flags():
    solver, pol, loads0, eng0, H, util, lo, hi, rng = random_instance(3)
    jp = joint_plan(solver, pol, pol, loads0, eng0, H)
    assert all(bool(x.all()) for x in jp.same)
    assert len(jp.states) == H + 1


def test_decide_pdl_valid_and_no_wider_than_indep():
    from types import SimpleNamespace

    from dsswm.certify.fcc import FCCCertifier, decide_pdl
    for s in range(10):
        solver, pol, loads0, eng0, H, util, lo, hi, rng = random_instance(100 + s)
        pols = [pol, PerturbedPolicy(pol, s, 0.3), PerturbedPolicy(pol, s + 50, 0.6)]
        prob = SimpleNamespace(policies=pols, loads0=loads0, engaged0=eng0, H=H, utility=util)
        st, k, e, D = decide_pdl(solver, prob, lo, hi, eps=0.0)
        th = lo + (hi - lo) * rng.random((500, solver.cells.n))
        J = np.stack([forward_J(solver, p, loads0, eng0, H, util, th) for p in pols])
        assert ((J.max(0) - J[k]) <= e + 1e-12).all()
        b = np.array([solver.solve(solver.plan(p, loads0, eng0, H), util.w, util.w_ret, util.c_q, lo, hi)
                      for p in pols])
        kk = int(np.argmax(b[:, 0]))
        e_ind = float(np.delete(b[:, 1], kk).max() - b[kk, 0])
        assert e <= e_ind + 1e-12
        assert FCCCertifier.decide(b, e_ind + 1e-9)[0] == "CERTIFIED"
