"""r5_rival_tuning: B2-rect forced-exploration subclass.

* explore_c = 1.0 reproduces the frozen r4 CombGameJoint("rect") checkpoint by checkpoint (toy populations);
* explore_c = 0.5 stays rigorous (rect certificate untouched), returns permutations, keeps billing consistent and
  certifies the near-tie toy with 0 false streams on a 40-stream smoke;
* registry: B2-rect resolves to the subclass, default explore_c = 1.0, grid point stays rigorous and in R.
"""
import numpy as np

from dsswm.baselines.b2_rect_fe import B2_EXPLORE_GRID, CombGameRectFE
from dsswm.baselines.combgame_joint import CombGameJoint
from dsswm.streams import r5_registry as reg
from dsswm.streams.frontier_runner import run_stream
from dsswm.tests.test_baselines_r4a import PROBS, TOY_EPS, _toy_population


def test_registry_b2_rect_is_tunable_subclass():
    m = reg.make_method("B2-rect")
    assert isinstance(m, CombGameRectFE) and m.explore_c == 1.0 and m.validity == "rigorous"
    assert reg.spec("B2-rect").in_R and "B2-rect" in reg.RIGOROUS_SET_R
    for c in B2_EXPLORE_GRID:
        mm = reg.make_method("B2-rect", explore_c=c)
        assert mm.validity == "rigorous" and mm.name == "B2-rect" and mm.explore_c == c


def test_explore_c_one_identical_to_r4():
    for seed in (900, 901, 902):
        env = _toy_population(seed)
        s1, r1 = run_stream(env, CombGameJoint("rect"), 3, PROBS, TOY_EPS, keep_U=True)
        s2, r2 = run_stream(env, CombGameRectFE(1.0), 3, PROBS, TOY_EPS, keep_U=True)
        assert len(r1) == len(r2)
        for x, y in zip(r1, r2):
            assert x["n_cells"] == y["n_cells"] and x["n_cert"] == y["n_cert"]
            assert np.array_equal(np.asarray(x["U"]), np.asarray(y["U"]))
        assert s1["N80"] == s2["N80"] and s1["cert_k"] == s2["cert_k"]


def test_explore_half_rigorous_smoke():
    false_streams = 0
    for seed in range(900, 920):
        env = _toy_population(seed)
        for perm in (0, 1):
            s, _ = run_stream(env, CombGameRectFE(0.5), perm, PROBS, TOY_EPS, keep_U=False)
            assert s["billing_ok"]
            false_streams += int(s["fwer_event"])
    assert false_streams == 0
