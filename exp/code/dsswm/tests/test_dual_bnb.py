import numpy as np

from dsswm.certify.dual_bnb import (bnb_bound, block_digits, dual_bound, kappa_participant_set, lin_copy_dual,
                                    lin_mu_dual)


def _random_problem(rng, radices=(3, 3, 2, 2, 2), P=4, keep=0.6):
    B = int(np.prod(radices))
    dig = block_digits(list(radices), B)
    # non-separable decision gap with cross-block interactions
    D = rng.standard_normal((B, P)) * 0.1
    for p in range(P):
        D[:, p] += 0.3 * np.sin(dig[:, p] + 2 * dig[:, (p + 1) % len(radices)]) * dig[:, -1]
    mask = rng.random(B) < keep
    mask[0] = True
    return D[mask], dig[mask], list(radices)


def test_dual_and_bnb_are_upper_bounds():
    rng = np.random.default_rng(0)
    for _ in range(40):
        D, dg, rad = _random_problem(rng)
        exact = D.sum(1).max()
        d = dual_bound(D, dg, rad)
        assert d["bound"] >= exact - 1e-9
        assert d["rect"] >= d["bound"] - 1e-9 or d["rect"] >= exact - 1e-9
        assert d["incumbent"] <= exact + 1e-9
        for hubs in ([0], [0, 1]):
            b = bnb_bound(D, dg, rad, hubs, d["incumbent"])
            assert b["bound"] >= exact - 1e-9


def test_bnb_on_all_blocks_is_exact():
    rng = np.random.default_rng(1)
    D, dg, rad = _random_problem(rng, radices=(2, 2, 2), P=3)
    exact = D.sum(1).max()
    b = bnb_bound(D, dg, rad, [0, 1, 2], -np.inf)
    assert abs(b["bound"] - exact) < 1e-9


def test_tlin_duals_equal_closed_form():
    rng = np.random.default_rng(2)
    for _ in range(20):
        dim, P = 7, 4
        A = rng.standard_normal((30, dim))
        V = np.eye(dim) + A.T @ A
        th = rng.standard_normal(dim)
        parts = rng.standard_normal((P, dim))
        d = parts.sum(0)
        beta = 3.0
        closed = d @ th + np.sqrt(beta) * np.sqrt(d @ np.linalg.solve(V, d))
        assert abs(lin_mu_dual(d, th, V, beta) - closed) < 1e-8
        cp = lin_copy_dual(parts, th, V, beta)
        assert abs(cp["opt_numeric"] - closed) < 1e-6
        assert cp["rect_nu0"] >= closed - 1e-12


def test_kappa_set_matches_ellipsoid_kappa_on_dense_ellipsoid_sample():
    # kappa_set on the boundary of an ellipsoid approaches sum ||d_p|| / ||d|| for linear Delta
    rng = np.random.default_rng(3)
    dim, P = 3, 2
    V = np.diag([1.0, 2.0, 4.0])
    parts = rng.standard_normal((P, dim))
    U = rng.standard_normal((200000, dim))
    U /= np.sqrt(np.einsum("ij,jk,ik->i", U, V, U))[:, None]
    D = np.stack([U @ parts[p] for p in range(P)], 1)
    Vi = np.linalg.inv(V)
    kap = sum(np.sqrt(p @ Vi @ p) for p in parts) / np.sqrt(parts.sum(0) @ Vi @ parts.sum(0))
    assert abs(kappa_participant_set(D) - kap) < 2e-2
