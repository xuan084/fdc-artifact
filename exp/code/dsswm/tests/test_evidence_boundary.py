"""(1) Theta_t is built only from env.step() outputs; model rollouts never enter the evidence."""
import dataclasses

import numpy as np
import pytest

from dsswm.core.provenance import EvidenceBoundaryError, Observation
from dsswm.evidence.ellipsoid import EllipsoidSet
from dsswm.evidence.ledger import EvidenceLedger
from dsswm.evidence.lr_set import SeqLRSet
from dsswm.exact.nl_propagate import NLPropagator
from dsswm.models.lin_class import LinClass
from dsswm.models.nl_class import NLClass, NLGrid
from dsswm.streams.generator import make_lin_instance, make_nl_instance

SMALL = NLGrid(alpha=(-1.0, 1.0), gamma=(0.0, 0.75), tauL=(1.0,), beta=(0.0,), tauR=(1.0,), psi=(0.0, 1.0), lam=(0.0, 0.5))


def _fake_from(obs, **chg):
    d = {f.name: getattr(obs, f.name) for f in dataclasses.fields(obs)}
    d.update(chg)
    return Observation(**d)


def test_lin_rejects_rollouts_and_tampering():
    inst = make_lin_instance(1)
    lc = LinClass(3, 3, inst.env.aspace.nb)
    ell = EllipsoidSet(lc, sigma=1.5, delta=0.05, S=3.0)
    led = EvidenceLedger([ell])
    for o in inst.init_obs:
        led.record(o, "initial")
    before = ell.state_digest()
    # a model rollout: an Observation constructed by the learner/simulator, never issued by env.step()
    o = inst.init_obs[0]
    fake = _fake_from(o, serial=10**9, outcomes=tuple((i, j, b, 9.9) for i, j, b, _ in o.outcomes))
    with pytest.raises(EvidenceBoundaryError):
        ell.update(fake)
    with pytest.raises(EvidenceBoundaryError):
        led.record(fake, "rollout")
    # tampering with an authentic serial number is detected by the payload digest
    tam = _fake_from(o, outcomes=tuple((i, j, b, y + 1.0) for i, j, b, y in o.outcomes))
    with pytest.raises(EvidenceBoundaryError):
        ell.update(tam)
    # planning computations (z vectors = model rollouts of policies) leave the evidence untouched
    for q in inst.problems[:3]:
        for p in q.policies:
            lc.z(p, q.loads0, q.H, q.utility, inst.env.aspace)
    after = ell.state_digest()
    assert np.array_equal(before[0], after[0]) and np.array_equal(before[1], after[1]) and before[2:] == after[2:]
    assert ell.n_rounds == len(inst.init_obs)
    # authentic observations still accepted
    ell.update(inst.env.step(0))


def test_nl_rejects_rollouts_and_jtables_do_not_touch_evidence():
    ncl = NLClass(SMALL)
    inst = make_nl_instance(3, grid=SMALL, nl_class=ncl)
    prop = NLPropagator(2, 2, 2, inst.env.aspace, 1.0, 0.3)
    params = ncl.torch_params()
    LT, _ = prop.tables(params)
    lr = SeqLRSet(prop, LT, 0.05)
    for o in inst.init_obs:
        lr.update(o)
    before = lr.state_digest()
    for q in inst.problems[:3]:
        prop.j_table(params, q.policies, q.loads0, q.engaged0, q.H, q.utility)
        # MC rollouts from a *model* (here: an env-like simulator built from a class member) are not evidence
    o = inst.init_obs[1]
    fake = _fake_from(o, serial=10**9 + 1)
    with pytest.raises(EvidenceBoundaryError):
        lr.update(fake)
    after = lr.state_digest()
    assert np.array_equal(before[0], after[0]) and before[1:] == after[1:]


def test_simulated_env_observations_are_not_confused_with_platform_steps():
    """Observations issued by a *different* env object are authentic platform data for that env only.
    The ledger tags them; rollouts produced by simulate_batch() never produce Observation objects at all."""
    inst = make_nl_instance(4, grid=SMALL)
    q = inst.problems[0]
    out = inst.env.simulate_batch(q.policies[0], q.loads0, q.engaged0, q.H, 10, np.random.default_rng(0))
    assert not any(isinstance(x, Observation) for x in out)
    n0 = inst.env.n_steps
    inst.env.simulate_batch(q.policies[0], q.loads0, q.engaged0, q.H, 10, np.random.default_rng(0))
    assert inst.env.n_steps == n0   # MC evaluation does not consume or log platform rounds
