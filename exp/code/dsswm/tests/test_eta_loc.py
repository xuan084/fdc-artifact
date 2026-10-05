"""eta_loc soundness: 1e3 random theta drawn uniformly in random cells, |J_theta - J_g| <= eta_loc(g), 0 violations.
Also: Voronoi cells partition the box, JPC_infl reduces to JPC at eta = 0, and the inflation is applied."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch

from dsswm.certify.eta_loc import build_plans, eta_loc, j_values
from dsswm.certify.minimax_enum import certify_minimax, regret_matrix
from dsswm.certify.minimax_infl import certify_minimax_infl
from dsswm.certify.status import Status
from dsswm.exact.nl_propagate import NLPropagator
from dsswm.models.grid_ladder import cell_boxes, cell_index_of, class_vectors, ladder_grid, sample_in_cells
from dsswm.models.nl_class import NLClass
from dsswm.streams.offgrid import make_offgrid_instance

DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_DRAWS = 1000


def _setup(seed=600):
    inst = make_offgrid_instance(seed, "R1")
    prop = NLPropagator(2, 2, 2, inst.env.aspace, 1.0, 0.3, device=DEV)
    return inst, prop


def test_eta_loc_zero_violations_1e3():
    inst, prop = _setup()
    rng = np.random.default_rng(42)
    viol, total, worst = 0, 0, 0.0
    cases = [(1, inst.problems[0]), (1, inst.problems[1]), (2, inst.problems[2]), (2, inst.problems[3])]
    per = N_DRAWS // len(cases)
    for f, q in cases:
        ncl = NLClass(ladder_grid(f), device=DEV)
        eta, J, _, _ = eta_loc(prop, ncl, q)
        idx = rng.integers(ncl.B, size=per)
        th = sample_in_cells(ncl, idx, rng)
        # include cell vertices-ish points: push half the draws to the cell boundary on a random coordinate
        lo, hi = cell_boxes(ncl)
        half = per // 2
        c = rng.integers(th.shape[1], size=half)
        th[np.arange(half), c] = np.where(rng.random(half) < 0.5, lo[idx[:half], c], hi[idx[:half], c])
        Jt = j_values(prop, build_plans(prop, q), th, q.utility.w, q.utility.w_ret)
        err = np.abs(Jt - J[idx]).max(1)
        viol += int((err > eta[idx] + 1e-12).sum())
        total += per
        worst = max(worst, float((err / eta[idx]).max()))
    assert total == N_DRAWS
    assert viol == 0, f"{viol} violations, worst err/eta = {worst}"


def test_cells_partition_and_contain_grid_points():
    for f in (0.5, 1, 2, 4):
        ncl = NLClass(ladder_grid(f), device="cpu")
        V = class_vectors(ncl)
        lo, hi = cell_boxes(ncl)
        assert (lo <= V).all() and (V <= hi).all()
        vol = np.prod(hi - lo, 1).sum()
        box = np.prod(hi.max(0) - lo.min(0))
        assert abs(vol - box) < 1e-9 * box
        rng = np.random.default_rng(f.__hash__() % 1000)
        idx = rng.integers(ncl.B, size=200)
        th = sample_in_cells(ncl, idx, rng)
        assert (cell_index_of(th, ncl) == idx).all()


def test_ladder_sizes():
    assert [NLClass(ladder_grid(f), device="cpu").B for f in (0.5, 1, 2, 4)] == [4096, 13824, 64000, 373248]


def test_infl_reduces_and_inflates():
    rng = np.random.default_rng(0)
    J = rng.random((50, 4)) * 0.01
    J[:, 0] += 0.5
    Reg = regret_matrix(J)
    m = np.ones(50, bool)
    a = certify_minimax(Reg, m, 0.02)
    b = certify_minimax_infl(Reg, m, np.zeros(50), 0.02)
    assert a["status"] == b["status"] == Status.CERTIFIED and abs(a["r_bar"] - b["r_bar"]) < 1e-15
    c = certify_minimax_infl(Reg, m, np.full(50, 0.011), 0.02)
    assert c["status"] == Status.NEED_DATA and abs(c["r_bar"] - (a["r_bar"] + 0.022)) < 1e-12


def test_eta_loc_diff_zero_violations():
    from dsswm.certify.eta_loc import eta_loc_diff
    inst, prop = _setup(601)
    rng = np.random.default_rng(7)
    q = inst.problems[0]
    ncl = NLClass(ladder_grid(1), device=DEV)
    eta, J = eta_loc_diff(prop, ncl, q)
    idx = rng.integers(ncl.B, size=300)
    th = sample_in_cells(ncl, idx, rng)
    Jt = j_values(prop, build_plans(prop, q), th, q.utility.w, q.utility.w_ret)
    Dt = Jt[:, :, None] - Jt[:, None, :]
    Dg = J[idx][:, :, None] - J[idx][:, None, :]
    err = np.abs(Dt - Dg).reshape(len(idx), -1).max(1)
    assert (err <= eta[idx] + 1e-12).all()
