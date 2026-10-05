import numpy as np

from dsswm.core.actions import ActionSpace, enumerate_matchings
from dsswm.core.state import StateCodec
from dsswm.stats.cp import clopper_pearson
from dsswm.stats.km_rmst import kaplan_meier, rmst


def test_matching_and_action_counts():
    assert len(enumerate_matchings(3, 3)) == 34
    assert ActionSpace(3, 3, 1, (1,)).n == 97
    assert len(enumerate_matchings(2, 2)) == 7
    assert ActionSpace(2, 2, 1, (1,)).n == 1 + 4 * 2 + 2 * 3


def test_state_codec_roundtrip():
    c = StateCodec(2, 2, 2, True)
    rng = np.random.default_rng(0)
    for _ in range(100):
        ld = rng.integers(0, 3, 4); e = rng.integers(0, 2, 4)
        l2, e2 = c.decode(c.encode(ld, e))
        assert (l2 == ld).all() and (e2 == e).all()


def test_km_rmst_cp():
    t, s = kaplan_meier([1, 2, 3, 4], [False, False, False, False])
    assert np.allclose(s, [0.75, 0.5, 0.25, 0.0])
    assert abs(rmst([1, 2, 3, 4], [False] * 4, 10) - 2.5) < 1e-12
    assert abs(rmst([5, 5], [True, True], 4) - 4.0) < 1e-12
    lo, hi = clopper_pearson(0, 100)
    assert lo == 0.0 and 0.03 < hi < 0.04
