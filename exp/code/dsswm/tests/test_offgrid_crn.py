"""Off-grid ladder: R0 is exactly the nearest G_1 point of R1; CRN pairing (same seed -> same initial data);
twin / dose construction; streams; problem stream identical to the round-0 generator."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np

from dsswm.streams.generator import DEFAULT_NL_S_GRID, generator_hash, make_nl_instance
from dsswm.streams.offgrid import (K_STREAM, R1_BOX, make_offgrid_instance, sample_r1_truth, snap_to_grid,
                                   stream_perm, truth_for)

G = DEFAULT_NL_S_GRID
SEEDS = range(600, 640)


def _vals(name):
    return {"alpha": G.alpha, "beta": G.beta, "tauL": G.tauL, "tauR": G.tauR, "gamma": G.gamma, "lam": G.lam,
            "psi": G.psi}[name]


def test_r0_is_nearest_grid_point_of_r1():
    for s in SEEDS:
        r1 = sample_r1_truth(s)
        r0 = snap_to_grid(r1)
        for k in r1:
            x, y = np.atleast_1d(r1[k]), np.atleast_1d(r0[k])
            v = np.asarray(_vals(k))
            d = np.abs(x[:, None] - v[None])
            assert np.allclose(y, v[d.argmin(1)])
            assert np.all(np.abs(x - y) <= d.min(1) + 1e-15)
        # kappa_high support of round 0
        assert np.all(r0["gamma"] == 0.75) and r0["lam"] == 0.5 and r0["psi"] in (0.75, 1.5)
        for k, (lo, hi) in [("alpha", R1_BOX["alpha"]), ("gamma", R1_BOX["gamma"]), ("psi", R1_BOX["psi"])]:
            assert np.all((np.atleast_1d(r1[k]) >= lo) & (np.atleast_1d(r1[k]) <= hi))


def _obs_key(o):
    return (o.t, o.loads, o.engaged, o.action, o.outcomes, o.next_loads, o.next_engaged)


def test_same_seed_same_initial_data_and_crn():
    for s in list(SEEDS)[:10]:
        a, b = make_offgrid_instance(s, "R0"), make_offgrid_instance(s, "R0")
        assert [_obs_key(o) for o in a.init_obs] == [_obs_key(o) for o in b.init_obs]
        r1 = make_offgrid_instance(s, "R1")
        assert [o.action for o in a.init_obs] == [o.action for o in r1.init_obs]     # CRN: same design
        assert a.truth["sha256"] != r1.truth["sha256"]
        assert np.allclose(a.env.true_params()["alpha"], snap_to_grid(sample_r1_truth(s))["alpha"])
        assert np.allclose(r1.env.true_params()["alpha"], sample_r1_truth(s)["alpha"])
        for qa, qb in zip(a.problems, r1.problems):
            assert qa.pid == qb.pid and [p.name for p in qa.policies] == [p.name for p in qb.policies]


def test_twin_and_dose():
    for s in list(SEEDS)[:10]:
        r0, tw = truth_for(s, "R0"), truth_for(s, "R0", twin=True)
        assert np.all(tw["gamma"] == 0) and tw["lam"] == 0
        for k in ("alpha", "beta", "tauL", "tauR", "psi"):
            assert np.allclose(r0[k], tw[k])
        h = truth_for(s, "R0", dose=0.5)
        assert np.allclose(h["gamma"], 0.5 * r0["gamma"]) and abs(h["lam"] - 0.5 * r0["lam"]) < 1e-15
        inst = make_offgrid_instance(s, "R0", twin=True)
        assert inst.truth["dose"] == 0.0 and np.all(inst.env.true_params()["gamma"] == 0)


def test_streams_and_round0_problem_stream():
    assert generator_hash() == "794acc9d48ff8c75"
    for s in list(SEEDS)[:5]:
        assert np.array_equal(stream_perm(s, 0), np.arange(K_STREAM))
        for k in (1, 2):
            assert sorted(stream_perm(s, k)) == list(range(K_STREAM))
        old = make_nl_instance(s, kappa_mode="high")
        base = make_offgrid_instance(s, "R1", stream=0)
        assert len(base.problems) == K_STREAM
        for qo, qn in zip(old.problems, base.problems[:10]):
            assert qo.pid == qn.pid and qo.H == qn.H and [p.name for p in qo.policies] == [p.name for p in qn.policies]
            assert np.allclose(qo.utility.w, qn.utility.w)
        st1 = make_offgrid_instance(s, "R1", stream=1)
        assert st1.noise_seed == 123 and [q.pid for q in st1.problems] == [base.problems[i].pid for i in stream_perm(s, 1)]
        # different noise seed -> different initial outcomes but same actions
        assert [o.action for o in st1.init_obs] == [o.action for o in base.init_obs]
