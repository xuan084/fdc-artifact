"""Round-1 baselines and threshold option: B12 TaS coverage / correctness, LR chi2_deff threshold, B3-UI radius,
B11 == B5-TaskDirected criterion."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from scipy.stats import chi2

from dsswm.baselines.b3_ui import ui_radius
from dsswm.baselines.b11_taskfisher import task_directed_gain
from dsswm.baselines.b12_tas import B12TaS
from dsswm.evidence.lr_set import SeqLRSet, participation_ratio
from dsswm.exact.nl_propagate import NLPropagator
from dsswm.models.nl_class import NLClass, NLGrid
from dsswm.streams.generator import make_nl_instance

SMALL = NLGrid(alpha=(-1.0, 1.0), gamma=(0.0, 0.75), tauL=(1.0,), beta=(0.0,), tauR=(0.5, 1.5), psi=(0.0, 1.0),
               lam=(0.0, 0.5))


def test_b12_covers_truth_and_certifies_correctly():
    rng = np.random.default_rng(42)
    J = 0.3 + 0.4 * rng.random((300, 5))
    covered, wrong, cert = 0, 0, 0
    for r in range(40):
        star = int(rng.integers(300))
        srng = np.random.default_rng(1000 + r)
        sampler = lambda k, n: (srng.random(n) < J[star, k]).astype(float)  # noqa: E731
        b = B12TaS(J, u_max=1.0, delta=0.05, eps=0.05)
        out = b.run(sampler, H=6, max_steps=200000)
        covered += bool(b.mask()[star])
        if out["status"] == "CERTIFIED":
            cert += 1
            wrong += J[star].max() - J[star, out["pi"]] > 0.05
    assert covered >= 36 and wrong <= 2 and cert >= 30


def test_b12_prior_carry_over_shrinks_set():
    rng = np.random.default_rng(0)
    J = rng.random((100, 3))
    b = B12TaS(J, 1.0, 0.05, 0.01)
    b.update(0, (rng.random(400) < J[7, 0]).astype(float))
    lm = b.logM()
    b2 = B12TaS(J, 1.0, 0.05, 0.01, prior_logM=lm)
    assert b2.mask().sum() == b.mask().sum() < 100 and b2.mask()[7]


def test_lr_chi2_deff_threshold():
    ncl = NLClass(SMALL)
    inst = make_nl_instance(3, grid=SMALL, nl_class=ncl)
    prop = NLPropagator(2, 2, 2, inst.env.aspace, 1.0, 0.3)
    LT, _ = prop.tables(ncl.torch_params())
    ui = SeqLRSet(prop, LT, 0.05)
    cs = SeqLRSet(prop, LT, 0.05, threshold="chi2_deff", d_eff=6)
    for o in inst.init_obs:
        ui.update(o)
        cs.update(o)
    assert abs(ui.cutoff() - math.log(20)) < 1e-12
    assert abs(cs.cutoff() - 0.5 * chi2.ppf(0.95, 6)) < 1e-12
    assert bool(cs.mask()[cs.mle()])
    assert np.allclose(ui.cum.cpu().numpy(), cs.cum.cpu().numpy())
    assert abs(participation_ratio(np.eye(7)) - 7) < 1e-12 and abs(participation_ratio(np.diag([1, 0, 0])) - 1) < 1e-12


def test_ui_radius():
    assert abs(ui_radius(-10.0, -10.0, 0.05) - math.sqrt(2 * math.log(20))) < 1e-12
    assert ui_radius(0.0, -5.0, 0.05) == 0.0          # numerator beats MLE by > log(1/delta): empty set


def test_b11_matches_b5_task_directed_formula():
    rng = np.random.default_rng(1)
    d, K, n = 6, 5, 7
    A = rng.standard_normal((d, d)); Vd = A @ A.T + np.eye(d)
    Fel = np.stack([(lambda B: B @ B.T)(rng.standard_normal((d, 2))) for _ in range(n)])
    G = rng.standard_normal((K, d)); Jh = rng.random(K); kh = int(np.argmax(Jh)); eps = 0.02
    # transcription of run_nl_acquisition_factorial.decide_fisher (crit == "T", gapnorm=True)
    others = [k for k in range(K) if k != kh]
    Gd = (G[others] - G[kh][None]) / ((Jh[kh] - Jh[others]) + eps)[:, None]
    f0 = np.einsum("kd,de,ke->k", Gd, np.linalg.inv(Vd), Gd).max()
    f1 = np.einsum("kd,nde,ke->nk", Gd, np.linalg.inv(Vd[None] + Fel), Gd).max(1)
    assert np.allclose(task_directed_gain(Vd, Fel, G, Jh, kh, True, eps), f0 - f1)
