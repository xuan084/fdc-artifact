"""Tests for the lock-v8 addendum machinery (prereg_v8, x5_v8 eval gate, data_v8, v8_analysis, v8_replica, runner)."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

from dsswm.envs.data_v8 import DATA_FILES_V8, LAYER_DATA_V8, x5_provenance_check
from dsswm.stats import prereg, prereg_v7, prereg_v8
from dsswm.stats import v8_analysis as VA
from dsswm.stats import v8_replica as R

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# --------------------------------------------------------------------------------------------- frozen FDC-BF
def test_fdc_bf_unchanged():
    from dsswm.baselines.fdc_bet import VARIANTS, make_variant
    m = make_variant("FDC-BF")
    assert VARIANTS["FDC-BF"] == ("bennett", "HG", True, False, (0.045, 0.005))
    assert (m.validity, m.alloc_kind) == ("rigorous", "fixed")


# --------------------------------------------------------------------------------------------- data / X5 gate
def test_data_v8_binding():
    assert set(DATA_FILES_V8) >= {"criteo_tidy", "criteo_gz", "lenta_tidy", "x5_dev", "x5_eval_labels",
                                  "x5_eval_outcome"}
    assert set(LAYER_DATA_V8["X9"]) == {"x5_dev", "x5_eval_labels", "x5_eval_outcome", "x5_provenance"}
    x5_provenance_check()


def test_x5_eval_is_gated():
    from dsswm.envs.x5_v8 import X5LayerEnv
    with pytest.raises(PermissionError):
        X5LayerEnv("eval")
    if not prereg_v8.ADDENDUM_PATH.exists():
        with pytest.raises(PermissionError):
            X5LayerEnv("eval", eval_task_id="v8a_full_a")
    with pytest.raises(PermissionError):
        X5LayerEnv("eval", eval_task_id="v7a_full_b")
    d = X5LayerEnv("dev")
    assert d.half == "dev" and d.N == 99646 and d.pool_sizes.shape == (9, 2)


# --------------------------------------------------------------------------------------------- prereg v8 gate
def _fake_v8(v7):
    add = {"version": "5+v8-addendum", "status": "locked",
           "addendum_to": {"version": 5, "sha256": v7["addendum_to"]["sha256"]},
           "v6_addendum": {"sha256": v7["v6_addendum"]["sha256"]}, "v7_addendum": {"sha256": v7["sha256"]},
           "v7_known_input_drift": {}, "statement": "x", "primary_method": {}, "blocks": {}, "seed_manifest": {},
           "replica_check": {}, "eval_tasks": {"v8a_full_a": {}}, "frozen_configs": {"X9": {}},
           "input_sha256": {"plan/prereg_lock.json": prereg._sha_file(prereg.LOCK_PATH)},
           "code_sha256": {"run_r5s_v8.py": "0" * 64}, "git_commit": "a" * 40,
           "data_sha256": {n: {"path": p, "sha256": "0" * 64} for n, p in DATA_FILES_V8.items()}}
    add["sha256"] = prereg.canonical_hash(add)
    return add


def _kw():
    return dict(verify_code=False, verify_inputs=False, verify_data=False)


def test_prereg_v8_gate_logic():
    v7 = prereg.load_lock(prereg_v7.ADDENDUM_PATH)
    add = _fake_v8(v7)
    assert prereg_v8.check_addendum(add, "v8a_full_a", v7_add=v7, **_kw()) is add
    with pytest.raises(RuntimeError):
        prereg_v8.check_addendum(add, "v8x_unknown", v7_add=v7, **_kw())
    for mut in ("status", "v5", "v6", "v7", "data", "tamper"):
        bad = copy.deepcopy(add)
        if mut == "status":
            bad["status"] = "draft"
        elif mut == "v5":
            bad["addendum_to"]["sha256"] = "0" * 64
        elif mut == "v6":
            bad["v6_addendum"]["sha256"] = "0" * 64
        elif mut == "v7":
            bad["v7_addendum"]["sha256"] = "0" * 64
        elif mut == "data":
            bad["data_sha256"].pop("x5_eval_outcome")
        if mut != "tamper":
            bad["sha256"] = prereg.canonical_hash(bad)
        else:
            bad["eval_tasks"]["v8c_full"] = {}
        with pytest.raises(RuntimeError):
            prereg_v8.check_addendum(bad, "v8a_full_a", v7_add=v7, **_kw())


def test_v7_input_drift_must_match_disclosure(tmp_path):
    v7 = {"input_sha256": {"a.txt": "0" * 64}}
    (tmp_path / "a.txt").write_text("x")
    cur = prereg._sha_file(tmp_path / "a.txt")
    with pytest.raises(RuntimeError):
        prereg_v8.v7_input_drift_check(v7, {}, ws_root=tmp_path)
    prereg_v8.v7_input_drift_check(v7, {"a.txt": {"locked_sha256": "0" * 64, "current_sha256": cur}},
                                   ws_root=tmp_path)
    with pytest.raises(RuntimeError):
        prereg_v8.v7_input_drift_check(v7, {"a.txt": {"locked_sha256": "0" * 64, "current_sha256": "1" * 64}},
                                       ws_root=tmp_path)


def test_draft_never_authorises_and_older_gates_reject_v8_tasks():
    ok, _ = prereg_v8.addendum_gate("v8a_full_a", path=prereg_v8.DRAFT_PATH, verify_code=False)
    assert not ok
    ok, _ = prereg_v7.addendum_gate("v8a_full_a", verify_code=False, verify_data=False)
    assert not ok


# --------------------------------------------------------------------------------------------- decision rule
def _ub(u):
    return {"ub95_one_sided": u}


def test_decide_block_a():
    F = {m: _ub(0.4) for m in VA.F_FAMILY}
    G = {m: _ub(0.98) for m in VA.G_FAMILY}
    d = VA.decide_block_a(F, G, 0, {"status": "pass"})
    assert d["verdict"] == "positive_result_achieved" and not d["failing_components"]
    d = VA.decide_block_a(dict(F, **{"HC-WoR": _ub(0.60)}), G, 0, {"status": "pass"})
    assert d["failing_components"] == ["rectangle_superiority_failed"] and d["rectangle_sub_label"] == "faster_below_1"
    d = VA.decide_block_a(dict(F, **{"HC-WoR": _ub(1.2)}), G, 0, {"status": "pass"})
    assert d["rectangle_sub_label"] == "not_faster"
    d = VA.decide_block_a(F, dict(G, **{"PJC-menu": _ub(1.05)}), 0, {"status": "pass"})
    assert d["failing_components"] == ["joint_adaptive_rival_not_inferior_failed"]
    d = VA.decide_block_a(F, G, 1, {"status": "fail"})
    assert d["failing_components"] == ["validity_failure", "replica_fail"]
    with pytest.raises(ValueError):
        VA.decide_block_a({"RECT-ck-HG": _ub(0.4)}, G, 0, {"status": "pass"})
    with pytest.raises(ValueError):
        VA.decide_block_a(F, G, 0, {"status": None})
    assert VA.F_THRESHOLD == 0.60 and VA.G_MARGIN == 1.05 and VA.DECISION_EPS == 0.02


def test_build_matrix_by_eps_refuses_gaps():
    rows = [{"method": "a", "eps": 0.02, "seed": s, "N80_pen": 10} for s in (1, 2)]
    VA.build_matrix_by_eps(rows, {0.02: ("a",)}, (1, 2), "N80_pen")
    with pytest.raises(ValueError):
        VA.build_matrix_by_eps(rows[:1], {0.02: ("a",)}, (1, 2), "N80_pen")
    with pytest.raises(ValueError):
        VA.build_matrix_by_eps(rows + [rows[0]], {0.02: ("a",)}, (1, 2), "N80_pen")


# --------------------------------------------------------------------------------------------- replica rules
def test_r2b_and_groups():
    rows = [{"method": m, "seed": 1, "eps": 0.02, "arrival_digest": "x"} for m in ("a", "b")]
    assert R.check_r2b(rows, [1], [0.02], ["a", "b"])["pass"]
    rows[1]["arrival_digest"] = "y"
    assert not R.check_r2b(rows, [1], [0.02], ["a", "b"])["pass"]
    g = R.x9_groups("0.5", "0.4")
    assert "RECT-ck-HG" in g[0] and "HC-WoR" not in g[0] and g[1] == ("PJC-local", "PJC-menu")
    with pytest.raises(ValueError):
        R.groups_for("v8a_full_a", {})
    with pytest.raises(ValueError):
        R.groups_for("v9_unknown")


# --------------------------------------------------------------------------------------------- runner
def test_runner_tasks_and_make():
    import run_r5s_v8 as RV
    for t in ("v8a_full_a", "v8a_full_b"):
        assert RV.TASKS[t]["seeds"] == list(range(35000, 35200)) and RV.TASKS[t]["layer"] == "X9"
    assert set(RV.TASKS["v8a_full_a"]["methods"]) == set(VA.A_CORE)
    assert set(RV.TASKS["v8a_full_b"]["methods"]) == set(VA.A_DESCR)
    assert set(RV.TASKS["v8b_full_a"]["methods"]) | set(RV.TASKS["v8b_full_b"]["methods"]) == set(VA.B_METHODS)
    assert RV.TASKS["v8b_full_a"]["seeds"] == list(range(35200, 35400))
    assert RV.TASKS["v8c_full"]["seeds"] == list(range(33000, 33200))
    for t, T in RV.TASKS.items():
        if T["half"] == "dev":
            assert all(900 <= s < 1000 for s in T["seeds"])
    assert set(VA.F_FAMILY) | set(VA.G_FAMILY) <= set(VA.A_CORE)
    from dsswm.baselines.pjc_bf import PJCBF, RectCkBF, RectCkHGPlan
    from dsswm.baselines.rect_v6 import RectCkHG
    assert type(RV.make("RECT-ck-HG", {"plan": "0.5"})) is RectCkHG
    assert isinstance(RV.make("RECT-ck-HG", {"plan": "0.4"}), RectCkHGPlan)
    assert isinstance(RV.make("RECT-ck-HG", {"plan": "ney", "alloc": [[0.5, 0.5]] * 9}), RectCkHGPlan)
    m = RV.make("PJC-menu", {"mode": "menu", "boundaries": [2], "rule": "proj", "menu": [0.4, 0.5], "pick": "x"})
    assert isinstance(m, PJCBF) and m.boundaries == (2,) and m.menu == (0.4, 0.5) and m.name == "PJC-menu"
    assert isinstance(RV.make("RECT-ck-BF+box", {}), RectCkBF)


def test_arrival_digest_is_allocation_free():
    from dsswm.envs.pool_replay import make_schedule
    from dsswm.envs.x5_v8 import X5LayerEnv
    env = X5LayerEnv("dev")
    a = make_schedule(env, 950, adaptive=True)
    b = make_schedule(env, 950, alloc=np.full((9, 2), 0.5))
    assert np.array_equal(a.seg_seq, b.seg_seq)
    assert all(np.array_equal(x, y) for x, y in zip(a.pool_perm, b.pool_perm))


def test_arrival_digest_comes_from_the_consumed_schedule():
    import run_r5s_v8 as RV
    from dsswm.envs.pool_replay import make_schedule
    RV.init_env("X9", "dev", (0.02,))
    r = RV.job(("FDC-BF", {}, 950, 0.02, 15, "c", None, "d"))
    assert r["error"] is None
    assert r["arrival_digest"] == RV.schedule_arrival_digest(make_schedule(RV._ENV["env"], 950, adaptive=True))
    r2 = RV.job(("PJC-local", {"mode": "local", "boundaries": [], "rule": "half", "rect": False}, 950, 0.02, 15,
                 "c", None, "d"))
    assert r2["error"] is None and r2["arrival_digest"] == r["arrival_digest"]
    # fault injection: a run on another seed must show a different digest
    r3 = RV.job(("FDC-BF", {}, 951, 0.02, 15, "c", None, "d"))
    assert r3["arrival_digest"] != r["arrival_digest"]
    from dsswm.streams import frontier_runner as FR
    assert FR.make_schedule is make_schedule          # wrapper removed after the run
