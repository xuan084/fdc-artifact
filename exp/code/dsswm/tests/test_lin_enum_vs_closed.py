"""(4) On the T-Lin subclass, enumerated sup over the ellipsoid equals the closed form <d, th> + sqrt(beta)||d||_{V^-1}."""
import numpy as np
from scipy.optimize import minimize

from dsswm.certify.enum_exact import sup_linear_enum
from dsswm.certify.lin_closed import certify_lin
from dsswm.evidence.ellipsoid import EllipsoidSet
from dsswm.models.lin_class import LinClass
from dsswm.streams.generator import make_lin_instance


def _ellipsoid(seed=2):
    inst = make_lin_instance(seed)
    lc = LinClass(3, 3, inst.env.aspace.nb)
    ell = EllipsoidSet(lc, sigma=1.5, delta=0.05, S=3.0)
    for o in inst.init_obs:
        ell.update(o)
    for _ in range(30):
        ell.update(inst.env.step(int(np.random.default_rng(len(ell.serials)).integers(97))))
    return inst, lc, ell


def _sample_in_ellipsoid(ell, n, rng):
    L = np.linalg.cholesky(np.linalg.inv(ell.V))
    u = rng.standard_normal((n, ell.d))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    r = rng.random(n) ** (1 / ell.d)
    return ell.theta_hat() + ell.sqrt_beta() * (u * r[:, None]) @ L.T


def test_enum_equals_closed_form():
    inst, lc, ell = _ellipsoid()
    rng = np.random.default_rng(0)
    q = inst.problems[0]
    Z = np.array([lc.z(p, q.loads0, q.H, q.utility, inst.env.aspace) for p in q.policies])
    D = Z[None] - Z[:, None]
    D = D.reshape(-1, lc.d)
    D = D[np.linalg.norm(D, axis=1) > 0]
    D = np.vstack([D, rng.standard_normal((50, lc.d))])
    closed = ell.sup_linear(D)
    # enumerated finite subclass: interior samples + boundary maximisers of every direction
    pts = np.vstack([_sample_in_ellipsoid(ell, 20000, rng)] + [ell.argsup_linear(d)[None] for d in D])
    assert all(ell.contains(p) for p in pts[-len(D):])
    enum = sup_linear_enum(pts, D)
    assert np.max(np.abs(enum - closed)) < 1e-9
    # interior-only enumeration is a lower bound that never exceeds the closed form
    inner = sup_linear_enum(pts[:-len(D)], D)
    assert (inner <= closed + 1e-12).all()


def test_closed_form_matches_numerical_optimum():
    _, lc, ell = _ellipsoid(3)
    rng = np.random.default_rng(1)
    th, V, r2 = ell.theta_hat(), ell.V, ell.sqrt_beta() ** 2
    for _ in range(5):
        d = rng.standard_normal(lc.d)
        cons = {"type": "ineq", "fun": lambda x: r2 - (x - th) @ V @ (x - th)}
        res = minimize(lambda x: -d @ x, th, constraints=[cons], method="SLSQP", options={"ftol": 1e-12, "maxiter": 500})
        assert abs(-res.fun - ell.sup_linear(d)[0]) < 1e-6 * max(1, abs(res.fun))


def test_certify_lin_regret_consistency():
    inst, lc, ell = _ellipsoid(4)
    q = inst.problems[1]
    Z = np.array([lc.z(p, q.loads0, q.H, q.utility, inst.env.aspace) for p in q.policies])
    out = certify_lin(Z, ell, eps=0.05)
    pts = _sample_in_ellipsoid(ell, 5000, np.random.default_rng(2))
    emp = ((pts @ Z.T).max(1) - pts @ Z[out["pi_hat"]]).max()
    assert emp <= out["r_bar"] + 1e-12
