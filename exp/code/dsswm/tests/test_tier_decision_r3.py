import math

from dsswm.stats.tier_decision_r3 import decide_tier_r3

BASE = dict(A1_NL_ci_upper=math.log(0.4), A1_pass_NL=True, A1_pass_Lin=True, A2_pass_NL=True, A2_pass_Lin=True,
            clear_ratio_ci_upper=0.9, saving_null=0.1, saving_main=0.5, C1_pass=True, Q1_pass=True)


def test_tier_a():
    assert decide_tier_r3(BASE)["tier"] == "A"


def test_clear_switch_c():
    assert decide_tier_r3({**BASE, "clear_ratio_ci_upper": 1.0})["tier"] == "C"


def test_zero_effect_c():
    assert decide_tier_r3({**BASE, "saving_null": 0.5})["tier"] == "C"


def test_a1_falsified_c():
    assert decide_tier_r3({**BASE, "A1_NL_ci_upper": math.log(0.8)})["tier"] == "C"


def test_tier_b_one_layer():
    out = decide_tier_r3({**BASE, "A2_pass_Lin": False})
    assert out["tier"] == "B"


def test_tier_b_c1_fail():
    assert decide_tier_r3({**BASE, "C1_pass": False})["tier"] == "B"


def test_q_wording_orthogonal():
    assert "conditional" in decide_tier_r3({**BASE, "Q1_pass": False})["q_wording"]
    assert decide_tier_r3({**BASE, "Q1_pass": False})["tier"] == "A"


def test_missing_inputs_reported():
    out = decide_tier_r3({})
    assert out["tier"] == "B" and "clear_ratio_ci_upper" in out["not_evaluable_inputs"]
