import numpy as np

from dsswm.evidence.lr_set import SeqLRSet
from dsswm.exact.nl_propagate import NLPropagator
from dsswm.models.nl_class import NLClass, NLGrid
from dsswm.streams.generator import make_nl_instance

SMALL = NLGrid(alpha=(-1.0, 1.0), gamma=(0.0, 0.75), tauL=(1.0,), beta=(0.0,), tauR=(0.5, 1.5), psi=(0.0, 1.0), lam=(0.0, 0.5))


def test_lr_set_keeps_truth_and_shrinks():
    ncl = NLClass(SMALL)
    covered, shrink = 0, 0
    for seed in range(10):
        inst = make_nl_instance(seed, grid=SMALL, nl_class=ncl)
        prop = NLPropagator(2, 2, 2, inst.env.aspace, 1.0, 0.3)
        LT, _ = prop.tables(ncl.torch_params())
        lr = SeqLRSet(prop, LT, 0.05)
        for o in inst.init_obs:
            lr.update(o)
        rng = np.random.default_rng(seed)
        for _ in range(150):
            lr.update(inst.env.step(int(rng.integers(inst.env.aspace.n))))
        covered += bool(lr.mask()[inst.truth["theta_index"]])
        shrink += lr.size() < ncl.B
    assert covered >= 9 and shrink == 10


def test_loglik_matches_env_probabilities():
    """Per-observation likelihood under theta* equals the product of env transition probabilities."""
    from dsswm.core.dynamics import sigmoid
    ncl = NLClass(SMALL)
    inst = make_nl_instance(1, grid=SMALL, nl_class=ncl)
    prop = NLPropagator(2, 2, 2, inst.env.aspace, 1.0, 0.3)
    LT, _ = prop.tables(ncl.torch_params())
    ll = prop.loglik(LT, inst.init_obs)[inst.truth["theta_index"]].cpu().numpy()
    tp = inst.env.true_params()
    tau = np.concatenate([tp["tauL"], tp["tauR"]])
    for o, l in zip(inst.init_obs, ll):
        lp = 0.0
        yp = np.zeros(4)
        for i, j, b, y, act in o.outcomes:
            if act:
                pr = sigmoid(tp["alpha"][i] + tp["beta"][j] - tp["gamma"][i] * o.loads[i] + tp["psi_left"][i, b])
                lp += np.log(pr if y else 1 - pr)
                yp[i] = yp[2 + j] = y
        for p in range(4):
            if o.engaged[p]:
                pr = sigmoid(tau[p] + 1.0 * yp[p] - tp["lam"] * o.loads[p])
            else:
                pr = 0.3
            lp += np.log(pr if o.next_engaged[p] else 1 - pr)
        assert abs(lp - l) < 1e-10
