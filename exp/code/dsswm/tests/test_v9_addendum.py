"""Tests for the lock-v9 addendum machinery (rect_tu_v9, x5_v9 / hillstrom eval gates, data_v9, prereg_v9, v9_analysis,
v9_replica, run_r5s_v9 task registry)."""
from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from dsswm.baselines.fdc_loc import dense_grid
from dsswm.baselines.pjc_bf import RectCkBF
from dsswm.baselines.rect_tu_v9 import RectCkBFTU
from dsswm.envs.data_v9 import DATA_FILES_V9, LAYER_DATA_V9
from dsswm.stats import prereg, prereg_v9
from dsswm.stats import v9_analysis as VA
from dsswm.stats import v9_replica as R
from dsswm.streams.frontier_runner import build_ctx
from dsswm.streams.frontier_runner_v6 import run_stream_v6
from dsswm.tests.test_baselines_r4a import PROBS, TOY_EPS, _toy_population

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


@pytest.fixture(scope="module")
def toy():
    env = _toy_population(901)
    return env, build_ctx(env, PROBS, TOY_EPS)


def _run(env, m, seed, ctx):
    return run_stream_v6(env, m, seed, PROBS, TOY_EPS, ctx=ctx, keep_U=True)


# --------------------------------------------------------------------------------------------- RECT-ck-BF-TU
def test_rect_tu_on_block_grid_equals_rect_ck_bf_box(toy):
    env, ctx = toy
    for seed in (901000, 901001, 901002):
        s1, r1 = _run(env, RectCkBF(box=True), seed, ctx)
        s2, r2 = _run(env, RectCkBFTU(ctx.checkpoints, box=True), seed, ctx)
        assert s1["N80"] == s2["N80"] and s1["cert_k"] == s2["cert_k"]
        for a, b in zip(r1, r2):
            assert np.array_equal(a["U"], b["U"])


def test_rect_tu_without_box_equals_rect_ck_bf(toy):
    env, ctx = toy
    s1, r1 = _run(env, RectCkBF(box=False), 901003, ctx)
    s2, r2 = _run(env, RectCkBFTU(ctx.checkpoints, box=False), 901003, ctx)
    assert s1["cert_k"] == s2["cert_k"] and all(np.array_equal(a["U"], b["U"]) for a, b in zip(r1, r2))


def test_rect_tu_dense_uses_block_ledger_and_stale_radii(toy):
    env, ctx = toy
    c2 = copy.copy(ctx)
    c2.checkpoints = dense_grid(ctx.checkpoints, 4)
    m = RectCkBFTU(ctx.checkpoints)
    _run(env, m, 901004, c2)
    K, SA = len(ctx.checkpoints), ctx.S * ctx.A
    assert m.ledger["K"] == K and m.ledger["K_eval"] == len(c2.checkpoints)
    assert math.isclose(m.ledger["beta_c"], math.log(2 * SA * K / 0.045))
    assert math.isclose(m.ledger["alpha_side_var"], 0.005 / (2 * SA * K))
    assert m.n_stale_evals > 0
    with pytest.raises(RuntimeError):
        _run(env, RectCkBFTU(ctx.checkpoints + 1), 901005, c2)


def test_rect_tu_intervals_contain_truth_on_dense_grid(toy):
    """Coverage sanity on the toy: the running-intersection rectangle contains mu at every dense evaluation time."""
    env, ctx = toy
    c2 = copy.copy(ctx)
    c2.checkpoints = dense_grid(ctx.checkpoints, 4)
    mu = env.true_mu("visit")
    for seed in range(901010, 901016):
        m = RectCkBFTU(ctx.checkpoints)
        _run(env, m, seed, c2)
        assert np.all(m._lo <= mu + 1e-12) and np.all(m._hi >= mu - 1e-12)


# --------------------------------------------------------------------------------------------- data / gates
def test_data_v9_binding():
    assert set(DATA_FILES_V9) >= {"criteo_tidy", "criteo_gz", "x5_dev", "x5_eval_labels", "x5_eval_outcome",
                                  "x5_provenance", "hillstrom_tidy"}
    assert LAYER_DATA_V9["HCZ6"] == ("hillstrom_tidy",) and LAYER_DATA_V9["CR9"] == ("criteo_tidy", "criteo_gz")


def test_eval_halves_are_gated():
    from dsswm.envs.hillstrom_v9 import HillstromV9Env
    from dsswm.envs.x5_v8 import X5LayerEnv
    from dsswm.envs.x5_v9 import X5V9Env
    with pytest.raises(PermissionError):
        X5V9Env("eval")
    with pytest.raises(PermissionError):
        X5LayerEnv("eval", eval_task_id="v9b_full_d")       # the locked v8 gate never accepts a v9 task
    if not prereg_v9.ADDENDUM_PATH.exists():
        with pytest.raises(PermissionError):
            X5V9Env("eval", eval_task_id="v9b_full_d")
        with pytest.raises(PermissionError):
            HillstromV9Env("CZ6", "eval", eval_task_id="v9c_full_b")
    with pytest.raises(PermissionError):
        X5V9Env("eval", eval_task_id="v8a_full_a")
    d = X5V9Env("dev")
    assert d.half == "dev" and d.N == 99646 and d.pool_sizes.shape == (9, 2)


def test_draft_never_authorises():
    ok, why = prereg_v9.addendum_gate("v9a_full_d", path=prereg_v9.DRAFT_PATH)
    assert not ok
    if not prereg_v9.ADDENDUM_PATH.exists():
        ok, why = prereg_v9.addendum_gate("v9a_full_d")
        assert not ok


def test_prereg_v9_schema_refuses_missing_keys():
    with pytest.raises(RuntimeError):
        prereg_v9._schema({"version": 1}, require_commit=False)
    add = {k: {"x": 1} for k in prereg_v9.V9_REQUIRED_KEYS}
    add["data_sha256"] = {"criteo_tidy": {}}
    with pytest.raises(RuntimeError):
        prereg_v9._schema(add, require_commit=False)
    add["data_sha256"] = {n: {} for n in DATA_FILES_V9}
    add["git_commit"] = "zz"
    prereg_v9._schema(add, require_commit=False)
    with pytest.raises(RuntimeError):
        prereg_v9._schema(add, require_commit=True)


def test_v8_input_drift_check(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    v8 = {"input_sha256": {"a.txt": prereg._sha_file(tmp_path / "a.txt")}}
    prereg_v9.v8_input_drift_check(v8, {}, ws_root=tmp_path)
    (tmp_path / "a.txt").write_text("y")
    with pytest.raises(RuntimeError):
        prereg_v9.v8_input_drift_check(v8, {}, ws_root=tmp_path)


# --------------------------------------------------------------------------------------------- decisions
def _p(ub):
    return {"ub95_one_sided": ub, "geomean_ratio": ub * 0.95}


def test_decide_confirmatory_A():
    ok = VA.decide_confirmatory("A", {"HC-WoR@D": _p(0.74), "RECT-ck-BF-TU@D": _p(0.6)}, {}, 0, {"status": "pass"})
    assert ok["verdict"] == "positive_result_achieved" and ok["governing_rival"] == "HC-WoR@D"
    bad = VA.decide_confirmatory("A", {"HC-WoR@D": _p(0.86), "RECT-ck-BF-TU@D": _p(0.6)}, {}, 0, {"status": "pass"})
    assert bad["failing_components"] == ["rival_superiority_failed"] and bad["rival_sub_label"] == "faster_below_1"
    bad = VA.decide_confirmatory("A", {"HC-WoR@D": _p(1.1), "RECT-ck-BF-TU@D": _p(0.6)}, {}, 1, {"status": "fail"})
    assert bad["failing_components"] == ["rival_superiority_failed", "validity_failure", "replica_fail"]
    assert bad["rival_sub_label"] == "not_faster"
    with pytest.raises(ValueError):
        VA.decide_confirmatory("A", {"HC-WoR@D": _p(0.5)}, {}, 0, {"status": "pass"})
    with pytest.raises(ValueError):
        VA.decide_confirmatory("A", {"HC-WoR@D": _p(0.5), "RECT-ck-BF-TU@D": _p(0.5)}, {}, 0, {"status": None})


def test_decide_confirmatory_B_needs_ni():
    r = {"HC-WoR@D": _p(0.40), "RECT-ck-BF-TU@D": _p(0.30)}
    with pytest.raises(ValueError):
        VA.decide_confirmatory("B", r, {}, 0, {"status": "pass"})
    with pytest.raises(ValueError):
        VA.decide_confirmatory("B", r, {"PJC-local@D": _p(1.0)}, 0, {"status": "pass"})
    ok = VA.decide_confirmatory("B", r, {"PJC-local@D": _p(1.04), "TU-PJC@D": _p(1.02)}, 0, {"status": "pass"})
    assert ok["verdict"] == "positive_result_achieved"
    bad = VA.decide_confirmatory("B", r, {"PJC-local@D": _p(1.0), "TU-PJC@D": _p(1.05)}, 0, {"status": "pass"})
    assert bad["failing_components"] == ["joint_rival_not_inferior_failed"]
    assert bad["governing_NI_rival"] == "TU-PJC@D"
    bad = VA.decide_confirmatory("B", {"HC-WoR@D": _p(0.50), "RECT-ck-BF-TU@D": _p(0.3)},
                                 {"PJC-local@D": _p(1.0), "TU-PJC@D": _p(1.0)}, 0, {"status": "pass"})
    assert bad["failing_components"] == ["rival_superiority_failed"]
    j = VA.decide_joint(ok, bad)
    assert j["verdict"] == "not_positive_on_both"


def test_word_block_c():
    w = VA.word_block_c({"a": _p(0.9), "b": _p(0.94)}, 0)
    assert w["wording"] == "faster_descriptive"
    assert VA.word_block_c({"a": _p(0.9), "b": _p(0.95)}, 0)["wording"] == "no_speed_advantage"
    assert VA.word_block_c({"a": _p(0.5), "b": _p(0.5)}, 1)["wording"] == "no_speed_advantage"


def test_groups_for():
    frozen = {"CR": {"RECT-ck-HG": {"plan": "0.5"}, "HC-WoR": {"plan": "0.5"}},
              "X9": {"RECT-ck-HG": {"plan": "0.5"}, "HC-WoR": {"plan": "ney"}}}
    gA = R.groups_for("v9a_full_d", frozen)[0]
    assert "HC-WoR" in gA and "RECT-ck-HG" in gA and "TU-FDC" in gA and "PJC-local" not in gA
    gB = R.groups_for("v9b_full_d", frozen)[0]
    assert "HC-WoR" not in gB and "RECT-ck-HG" in gB
    assert R.groups_for("v9c_full_b", frozen)[0][0] == "FDC-BF"


# --------------------------------------------------------------------------------------------- runner registry
def test_runner_tasks_and_seeds():
    import run_r5s_v9 as RV
    full = {t: T for t, T in RV.TASKS.items() if T["half"] == "eval"}
    assert set(full) == {"v9a_full_d", "v9a_full_k", "v9a_full_x", "v9b_full_d", "v9b_full_k", "v9b_full_x",
                         "v9c_full_b", "v9c_full_a"}
    assert full["v9a_full_d"]["seeds"] == list(range(37000, 37200))
    assert full["v9b_full_k"]["seeds"] == list(range(37200, 37400))
    assert full["v9c_full_b"]["seeds"] == list(range(37400, 37600))
    assert full["v9c_full_a"]["seeds"] == list(range(37600, 37800))
    for t, T in RV.TASKS.items():
        if T["half"] == "dev":
            assert all(950 <= s <= 999 for s in T["seeds"])
    assert full["v9a_full_d"]["grid"] == "D" and full["v9a_full_k"]["grid"] == "K"
    for blk in ("A", "B"):
        sp = VA.CONF_SPEC[blk]
        for t, (g, ms) in sp["tasks"].items():
            assert RV.TASKS[t]["grid"] == g and tuple(RV.TASKS[t]["methods"]) == tuple(ms)
        assert set(sp["rivals"]) <= {f"{m}@D" for m in VA.DENSE_METHODS}
        assert VA.PRIMARY in {f"{m}@D" for m in VA.DENSE_METHODS}


# --------------------------------------------------------------------------------------------- external reviewer v9 r1 additions
def _pjc_toy_kw():
    return {"mode": "local", "boundaries": [], "rule": "half", "rect": False, "pick": "PJC-BF[local,b=,half]"}


def test_tu_pjc_on_block_grid_equals_pjc_local(toy):
    from dsswm.baselines.pjc_bf import PJCBF
    from dsswm.baselines.rect_tu_v9 import make_tu_pjc
    env, ctx = toy
    kw = _pjc_toy_kw()
    for seed in (901020, 901021):
        k2 = dict(kw)
        k2.pop("pick")
        k2["boundaries"] = ()
        s1, r1 = _run(env, PJCBF(**k2, name="PJC"), seed, ctx)
        s2, r2 = _run(env, make_tu_pjc(ctx.checkpoints, **kw), seed, ctx)
        assert s1["cert_k"] == s2["cert_k"] and s1["N80"] == s2["N80"]
        for a, b in zip(r1, r2):
            assert np.array_equal(a["U"], b["U"])


def test_tu_pjc_dense_block_ledger_and_validity(toy):
    from dsswm.baselines.rect_tu_v9 import make_tu_pjc
    env, ctx = toy
    c2 = copy.copy(ctx)
    c2.checkpoints = dense_grid(ctx.checkpoints, 4)
    m = make_tu_pjc(ctx.checkpoints, **_pjc_toy_kw())
    s, _ = _run(env, m, 901022, c2)
    assert m.ledger["K"] == len(ctx.checkpoints) and m.ledger["K_eval"] == len(c2.checkpoints)
    assert m.n_stale_evals > 0
    with pytest.raises(ValueError):
        make_tu_pjc(ctx.checkpoints, mode="local", boundaries=(2,), rule="half")


def test_word_block_c_rejects_bad_input():
    with pytest.raises(ValueError):
        VA.word_block_c({}, 0)
    with pytest.raises(ValueError):
        VA.word_block_c({"a": _p(-math.inf)}, 0)
    with pytest.raises(ValueError):
        VA.word_block_c({"a": _p(0.5)}, 0, rectangles=("a", "b"))
    with pytest.raises(ValueError):
        VA.word_block_c({"a": _p(0.5)}, -1)


def test_check_cross_grid():
    base = {"eps": 0.02, "schedule_digest": "x", "block_counts": {"10": [1, 2], "20": [3, 4]}}
    rows = [dict(base, method="TU-FDC@D", seed=1), dict(base, method="FDC-BF@K", seed=1)]
    assert R.check_cross_grid(rows, [1], [0.02], (("TU-FDC@D", "FDC-BF@K"),))["pass"]
    rows[1] = dict(base, method="FDC-BF@K", seed=1, block_counts={"10": [1, 2], "20": [3, 5]})
    assert not R.check_cross_grid(rows, [1], [0.02], (("TU-FDC@D", "FDC-BF@K"),))["pass"]
    rows[1] = dict(base, method="FDC-BF@K", seed=1, schedule_digest="y")
    assert not R.check_cross_grid(rows, [1], [0.02], (("TU-FDC@D", "FDC-BF@K"),))["pass"]
