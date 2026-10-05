"""KL decomposition used by DDA / rho* equals brute-force KL; minimax certificate is sound on the set."""
import itertools

import numpy as np

from dsswm.acquire.nl_kl_dda import IncidenceIndex, class_prob_tables, kl_features
from dsswm.certify.minimax_enum import certify_minimax, regret_matrix
from dsswm.core.actions import ActionSpace
from dsswm.core.state import StateCodec
from dsswm.models.nl_class import NLClass, NLGrid

GRID = NLGrid(L=2, R=2, alpha=(-1.5, 1.5), gamma=(0.0, 0.75), tauL=(0.5,), beta=(-1.0, 1.0), tauR=(1.5,),
              psi=(0.0, 1.5), lam=(0.0, 0.5))


def _brute_kl(py, pe, a, b, loads, eng, act, rho=0.3):
    inc = {(i, j): l for i, j, l in act.incentives}
    active = [(i, j) for i, j in act.pairs if eng[i] and eng[2 + j]]

    def logp(th, ys, es):
        lp, yp = 0.0, [0] * 4
        for k, (i, j) in enumerate(active):
            p = py[th, i, j, loads[i], inc.get((i, j), 0)]
            lp += np.log(p if ys[k] else 1 - p)
            yp[i] = yp[2 + j] = ys[k]
        for p_ in range(4):
            if eng[p_]:
                q = pe[th, p_, yp[p_], loads[p_]]
                lp += np.log(q if es[p_] else 1 - q)
            else:
                lp += np.log(rho if es[p_] else 1 - rho)
        return lp

    kl = 0.0
    for ys in itertools.product([0, 1], repeat=len(active)):
        for es in itertools.product([0, 1], repeat=4):
            la, lb = logp(a, ys, es), logp(b, ys, es)
            kl += np.exp(la) * (la - lb)
    return kl


def test_kl_decomposition_matches_bruteforce():
    ncl = NLClass(GRID, device="cpu")
    aspace = ActionSpace(2, 2, 1, (1,))
    codec = StateCodec(2, 2, 2, True)
    py, pe = class_prob_tables(ncl.np_params, 1.0, 2, aspace.nb)
    inc = IncidenceIndex(codec, aspace, 2)
    rng = np.random.default_rng(0)
    for _ in range(25):
        a, b = rng.integers(ncl.B, size=2)
        code, act = int(rng.integers(codec.size)), int(rng.integers(aspace.n))
        F = kl_features(py[a:a + 1], pe[a:a + 1], py[b:b + 1], pe[b:b + 1])
        dec = float(inc.rows_for_state(code)[act] @ F[:, 0])
        loads, eng = codec.decode(code)
        assert abs(dec - _brute_kl(py, pe, a, b, loads, eng, aspace.actions[act])) < 1e-9


def test_minimax_certificate_is_sound():
    rng = np.random.default_rng(1)
    for _ in range(200):
        J = rng.random((50, 6))
        Reg = regret_matrix(J)
        mask = rng.random(50) < 0.3
        mask[0] = True                     # truth index 0 inside the set
        c = certify_minimax(Reg, mask, eps=0.3)
        if c["status"].value == "CERTIFIED":
            assert Reg[0, c["pi"]] <= 0.3 + 1e-12
        assert c["r_bar"] >= Reg[0, c["pi"]] - 1e-12
