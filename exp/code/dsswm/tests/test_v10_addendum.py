"""Tests for the lock-v10 addendum machinery (seg_v10_eval gate + frozen segmentation, prereg_v10, v10_analysis,
v10_replica, run_v10 task registry, TU wrappers on the block grid)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

from dsswm.baselines.fdc_dp import FDCDP, FDCDPTimeUniform, make_seg_ctx
from dsswm.baselines.rect_dp import RectBFDP, RectBFDPTU
from dsswm.envs import seg_v10_eval as SE
from dsswm.envs.seg_v10 import run_stream_seg, seg_problems, true_opt
from dsswm.stats import prereg, prereg_v10
from dsswm.stats import v10_analysis as VA
from dsswm.stats import v10_replica as R
from dsswm.streams.frontier_runner import SyntheticPoolEnv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# --------------------------------------------------------------------------------------------- eval gate
def test_eval_refused_without_lock():
    with pytest.raises(PermissionError):
        SE.SegEnvV10Frozen("x5", 16, "eval")                               # no task id
    if not prereg_v10.ADDENDUM_PATH.exists():
        for data, t in (("x5", "v10a_full_s16"), ("lenta", "v10b_full_s16")):
            with pytest.raises(PermissionError):
                SE.SegEnvV10Frozen(data, 16, "eval", eval_task_id=t)
        ok, why = prereg_v10.addendum_gate("v10a_full_s16")
        assert not ok


def test_draft_never_authorises():
    ok, why = prereg_v10.addendum_gate("v10a_full_s16", path=prereg_v10.DRAFT_PATH)
    assert not ok


def test_gate_runs_before_any_eval_read(monkeypatch):
    def boom(data):
        raise AssertionError("eval rows read before the gate")
    monkeypatch.setattr(SE, "_read_eval_rows_raw", boom)
    with pytest.raises(PermissionError):
        SE.SegEnvV10Frozen("x5", 16, "eval", eval_task_id="v10a_full_s16", gate_path="/nonexistent/lock.json")


def test_eval_gate_checks_registered_layer(monkeypatch):
    fake = {"eval_tasks": {"t1": {"layer": "LENTA-SR16"}}}
    monkeypatch.setattr(prereg_v10, "addendum_gate", lambda tid, path=None: (True, fake))
    assert SE.eval_gate("t1", "LENTA-SR16") is fake
    with pytest.raises(PermissionError):
        SE.eval_gate("t1", "X5-SR16")
    with pytest.raises(PermissionError):
        SE.eval_gate("t2", "LENTA-SR16")
    monkeypatch.setattr(prereg_v10, "addendum_gate", lambda tid, path=None: (False, "no lock"))
    with pytest.raises(PermissionError):
        SE.eval_gate("t1", "LENTA-SR16")


def test_eval_path_on_synthetic_rows(monkeypatch):
    """The eval code path end to end on SYNTHETIC rows: gate (faked) -> read (faked) -> frozen models + frozen cut
    points + frozen problems (no refit, no re-cut, no eval-weight costs)."""
    fz = SE.load_frozen_seg()
    rng = np.random.default_rng(0)
    n = 20_000
    X = np.column_stack([rng.uniform(18, 80, n), np.zeros(n), rng.integers(0, 2, n), rng.integers(0, 2, n),
                         rng.uniform(0, 800, n), rng.uniform(0, 900, n), rng.uniform(-100, 300, n),
                         rng.integers(0, 2, n)]).astype(float)
    arm = rng.integers(0, 2, n)
    y = rng.integers(0, 2, n).astype(float)
    calls = []
    monkeypatch.setattr(SE, "eval_gate", lambda tid, layer, gate_path=None: calls.append(("gate", tid, layer)))
    monkeypatch.setattr(SE, "_read_eval_rows_raw", lambda data: calls.append(("read", data)) or (X, arm, y, np.arange(n)))
    env = SE.SegEnvV10Frozen("x5", 16, "eval", eval_task_id="v10a_full_s16")
    assert calls == [("gate", "v10a_full_s16", "X5-SR16"), ("read", "x5")]
    models = SE._load_models(fz, "x5")
    score = SE._score(models, X)
    cuts = [float.fromhex(c) for c in fz["data"]["x5"]["S"]["16"]["cuts_hex"]]
    assert np.array_equal(env.seg, np.searchsorted(cuts, score, side="right"))
    sp = SE.frozen_problems(fz, "x5", 16)
    assert np.array_equal(env.problems.cost, sp.cost) and np.array_equal(env.problems.budgets, sp.budgets)
    assert (env.problems.cost[:, 1] == 8).all() and env.half == "eval" and env.N == n and env.tau_R == n
    assert env.pool_sizes.sum() == n


# --------------------------------------------------------------------------------------------- frozen segmentation
def test_frozen_files_and_hashes():
    fz = SE.load_frozen_seg()
    for d in ("x5", "lenta"):
        v = fz["data"][d]
        assert prereg._sha_file(SE.FROZEN_DIR / v["model_pickle"]) == v["model_pickle_sha256"]
        for S in ("16", "32", "64"):
            e = v["S"][S]
            c = [float.fromhex(x) for x in e["cuts_hex"]]
            assert len(c) == int(S) - 1 and np.all(np.diff(c) > 0)
            assert e["cost"] == [[0, 8]] * int(S)
    with pytest.raises(RuntimeError):
        SE.load_frozen_seg(expect_sha256="0" * 64)


@pytest.mark.parametrize("S", [16, 64])
def test_dev_env_reproduces_frozen_scores_and_seg_v10(S):
    from dsswm.envs.seg_v10 import SegEnvV10
    fz = SE.load_frozen_seg()
    e = SE.SegEnvV10Frozen("x5", S, "dev")              # raises if the frozen model does not reproduce dev scores
    ref = SegEnvV10("x5", S, "dev")
    assert e.N == ref.N and np.array_equal(e.arm, ref.arm) and np.array_equal(e.row_index, ref.row_index)
    assert int((e.seg != ref.seg).sum()) == fz["data"]["x5"]["S"][str(S)]["dev_rows_reassigned_vs_seg_v10"]
    assert np.array_equal(e.problems.cost, ref.problems.cost) and np.array_equal(e.problems.budgets,
                                                                                 ref.problems.budgets)


def test_x5_features_identical_to_seg_v10():
    import pandas as pd
    from dsswm.envs import seg_v10 as SV
    X, _, _, _, _ = SV._x5_dev()
    assert np.array_equal(X, SE.x5_features(pd.read_pickle(SV.X5_ROOT / "dev.pkl")), equal_nan=True)


def test_disjointness_model_replay_and_lenta_eval():
    from dsswm.envs import seg_v10 as SV
    X, arm, y, rows, _ = SV._lenta_dev()
    msk = SV.model_split(arm)
    assert not (set(rows[msk]) & set(rows[~msk]))
    ev = SE.lenta_eval_mask()
    assert not (set(np.flatnonzero(ev)) & set(rows))                       # eval rows never in the dev half
    assert int(ev.sum()) + len(rows) == len(ev)


def test_assign_segments_rule():
    assert SE.assign_segments([0.0, 1.0, 1.5, 2.0, 3.0], [1.0, 2.0]).tolist() == [0, 1, 1, 2, 2]
    with pytest.raises(RuntimeError):
        SE.assign_segments([0.0], [1.0, 1.0])


def test_data_binding():
    assert set(SE.DATA_FILES_V10) == {"lenta_tidy", "x5_dev", "x5_eval_labels", "x5_eval_outcome", "x5_provenance"}
    assert SE.LAYER_DATA_V10["lenta"] == ("lenta_tidy",)


# --------------------------------------------------------------------------------------------- prereg
def test_prereg_v10_schema_refuses_missing_keys():
    with pytest.raises(RuntimeError):
        prereg_v10._schema({"version": 1}, require_commit=False)
    add = {k: {"x": 1} for k in prereg_v10.V10_REQUIRED_KEYS}
    add["data_sha256"] = {"lenta_tidy": {}}
    with pytest.raises(RuntimeError):
        prereg_v10._schema(add, require_commit=False)
    add["data_sha256"] = {n: {} for n in SE.DATA_FILES_V10}
    add["git_commit"] = "zz"
    prereg_v10._schema(add, require_commit=False)
    with pytest.raises(RuntimeError):
        prereg_v10._schema(add, require_commit=True)


def test_check_addendum_refuses_draft_status():
    with pytest.raises(RuntimeError):
        prereg_v10.check_addendum({"status": "draft"})


def test_v9_input_drift_check(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    v9 = {"input_sha256": {"a.txt": prereg._sha_file(tmp_path / "a.txt")}}
    prereg_v10.v9_input_drift_check(v9, {}, ws_root=tmp_path)
    (tmp_path / "a.txt").write_text("y")
    with pytest.raises(RuntimeError):
        prereg_v10.v9_input_drift_check(v9, {}, ws_root=tmp_path)


# --------------------------------------------------------------------------------------------- decisions
def _p(ub):
    return {"ub95_one_sided": ub, "geomean_ratio": ub * 0.97}


def test_decide_block_iut():
    cells = VA.CELLS["B"]
    ok = VA.decide_block("B", {t: {"ratio": _p(0.7), "primary_false_streams": 0} for t in cells}, {"status": "pass"})
    assert ok["verdict"] == "positive_result_achieved" and all(c["pass"] for c in ok["cells"].values())
    one = dict({t: {"ratio": _p(0.7), "primary_false_streams": 0} for t in cells})
    t0 = next(iter(cells))
    one[t0] = {"ratio": _p(0.80), "primary_false_streams": 0}
    bad = VA.decide_block("B", one, {"status": "pass"})
    assert bad["verdict"] == "positive_result_not_achieved" and bad["failing_components"] == \
        [f"{t0}:rival_superiority_failed"] and bad["cells"][t0]["sub_label"] == "faster_below_1"
    one[t0] = {"ratio": _p(0.5), "primary_false_streams": 1}
    bad = VA.decide_block("B", one, {"status": "fail"})
    assert bad["failing_components"] == [f"{t0}:validity_failure", "replica_fail"]
    with pytest.raises(ValueError):
        VA.decide_block("B", {t0: {"ratio": _p(0.5), "primary_false_streams": 0}}, {"status": "pass"})
    with pytest.raises(ValueError):
        VA.decide_block("B", {t: {"ratio": _p(0.5), "primary_false_streams": 0} for t in cells}, {"status": None})
    with pytest.raises(ValueError):
        VA.decide_block("B", {t: {"ratio": _p(-math.inf), "primary_false_streams": 0} for t in cells},
                        {"status": "pass"})
    with pytest.raises(ValueError):
        VA.decide_block("B", {t: {"ratio": _p(0.5), "primary_false_streams": True} for t in cells},
                        {"status": "pass"})
    j = VA.decide_joint(ok, bad)
    assert j["verdict"] == "no_joint_claim"
    assert VA.decide_joint(ok, ok)["verdict"] == "positive_on_both_blocks"


def _fake_rows(seeds, eps, n80):
    rows = []
    for m, base in n80.items():
        for i, s in enumerate(seeds):
            v = base * (1 + 0.01 * (i % 5))
            rows.append({"method": m, "seed": s, "eps": eps, "N80_pen": v, "tau_R": 1000, "fwer_event": False,
                         "false_by_k80": False, "n80_lt_tau": v < 1000, "reached_stop": True, "validity": "rigorous",
                         "exhaustion_at_k80": {"all": 0.1, "ctrl": 0.2, "treat": 0.0},
                         "schedule_digest": "s", "arrival_digest": f"a{s}"})
    return rows


def test_analyse_block_on_synthetic_rows():
    cells = {"c1": {"data": "x5", "S": 16, "eps": 0.03}, "c2": {"data": "x5", "S": 32, "eps": 0.04}}
    seeds = list(range(30))
    rows = {"c1": _fake_rows(seeds, 0.03, {VA.PRIMARY: 300, VA.RIVAL: 600, "HC-WoR-DP": 600, "FDC-DP(a)": 300}),
            "c2": _fake_rows(seeds, 0.04, {VA.PRIMARY: 300, VA.RIVAL: 350, "HC-WoR-DP": 600, "FDC-DP(a)": 300})}
    res = VA.analyse_block("A", rows, {"status": "pass"}, seeds=seeds, cells=cells, B=2000)
    d = res["decision"]
    assert d["cells"]["c1"]["pass"] and not d["cells"]["c2"]["pass"] and d["verdict"] == "positive_result_not_achieved"
    assert res["cells"]["c1"]["methods"][VA.PRIMARY]["exhaustion_at_k80_mean"]["ctrl"] == pytest.approx(0.2)


# --------------------------------------------------------------------------------------------- replica
def test_replica_groups_and_checks():
    assert R.groups_for("v10a_full_s16") == (VA.METHODS,)
    seeds = list(range(12))
    rows = _fake_rows(seeds, 0.03, {m: 300 for m in VA.METHODS})
    rep = [r for r in rows if r["seed"] < 10]
    ch = R.compute_checks("t", False, rows, rep, VA.METHODS, (0.03,), seeds)
    assert all(c["pass"] for c in ch)
    rows[0] = dict(rows[0], schedule_digest="other")
    ch = R.compute_checks("t", False, rows, rep, VA.METHODS, (0.03,), seeds)
    assert not all(c["pass"] for c in ch)


# --------------------------------------------------------------------------------------------- TU on the block grid
def _toy(seed=911, S=6):
    rng = np.random.default_rng(seed)
    sizes = rng.integers(600, 2000, size=(S, 2))
    base = rng.uniform(0.2, 0.6, size=S)
    mu = np.stack([base, np.clip(base + rng.normal(0.0, 0.08, size=S), 0.01, 0.99)], 1)
    env = SyntheticPoolEnv(sizes, mu, seed=seed, n_min=200)
    env.problems = seg_problems(env.w, budgets=(0.2, 0.35, 0.5, 0.65, 0.8))
    return env


def test_tu_methods_identical_on_block_grid():
    env = _toy()
    Js, mu = true_opt(env)
    ctx = make_seg_ctx(env.w, env.pool_sizes, env.problems, 0.02, 0.05, env.checkpoints(20), env.tau_R,
                       stop_frac=1.0)
    for seed in (1, 2, 3):
        a, ra = run_stream_seg(env, FDCDP("b"), seed, ctx, Js, mu)
        b, rb = run_stream_seg(env, FDCDPTimeUniform(ctx.checkpoints, scheme="b"), seed, ctx, Js, mu)
        assert a["N80_pen"] == b["N80_pen"] and a["cert_k"] == b["cert_k"] and [x["U"] for x in ra] == \
            [x["U"] for x in rb]
        c, rc = run_stream_seg(env, RectBFDP(), seed, ctx, Js, mu)
        d, rd = run_stream_seg(env, RectBFDPTU(ctx.checkpoints), seed, ctx, Js, mu)
        assert c["N80_pen"] == d["N80_pen"] and c["cert_k"] == d["cert_k"] and [x["U"] for x in rc] == \
            [x["U"] for x in rd]


# --------------------------------------------------------------------------------------------- runner registry
def test_runner_tasks_and_seeds():
    import run_v10 as RV
    full = {t: T for t, T in RV.TASKS.items() if T["half"] == "eval"}
    assert set(full) == set(VA.CELLS["A"]) | set(VA.CELLS["B"]) | set(VA.DESC_CELLS)
    for t, T in full.items():
        blk = T["block"]
        assert T["seeds"] == list(VA.BLOCK_SEEDS[blk]) and T["methods"] == list(VA.METHODS)
        assert T["layer"] == SE.layer_name(T["data"], T["S"])
    assert VA.BLOCK_SEEDS["A"] == tuple(range(38000, 38200)) and VA.BLOCK_SEEDS["B"] == tuple(range(38200, 38400))
    assert all(c["data"] == "x5" for c in VA.CELLS["A"].values())
    assert all(c["data"] == "lenta" for c in VA.CELLS["B"].values())
    for t, T in RV.TASKS.items():
        if T["half"] == "dev":
            assert all(950 <= s <= 999 for s in T["seeds"]) and t.replace("_pilot_", "_full_") in full
    assert RV.STOP_FRAC == 1.0 and RV.N80_K == 12 and RV.K_GRID == 20


def test_public_read_eval_rows_is_gated(monkeypatch):
    """external reviewer v10 lock r1 F1: the exported reader must call eval_gate before any read."""
    from dsswm.envs import seg_v10_eval as SE
    calls = []

    def deny(*a, **k):
        calls.append("gate")
        raise RuntimeError("denied")

    def boom(data):
        raise AssertionError("eval rows read without gate")

    monkeypatch.setattr(SE, "eval_gate", deny)
    monkeypatch.setattr(SE, "_read_eval_rows_raw", boom)
    import pytest
    for data in ("x5", "lenta"):
        with pytest.raises(RuntimeError, match="denied"):
            SE.read_eval_rows(data, "v10a_full_s16", "x")
    assert calls == ["gate", "gate"]
    import inspect
    assert list(inspect.signature(SE.read_eval_rows).parameters)[:3] == ["data", "eval_task_id", "layer"]


def test_read_eval_rows_binds_table_to_layer(monkeypatch):
    """external reviewer v10 lock r2: an authorised task for one layer cannot read the other table."""
    from dsswm.envs import seg_v10_eval as SE
    import pytest
    monkeypatch.setattr(SE, "eval_gate", lambda *a, **k: {"ok": True})
    reads = []
    monkeypatch.setattr(SE, "_read_eval_rows_raw", lambda data: reads.append(data) or "rows")
    with pytest.raises(PermissionError):
        SE.read_eval_rows("lenta", "v10a_full_s16", SE.layer_name("x5", 16))
    with pytest.raises(PermissionError):
        SE.read_eval_rows("x5", "v10b_full_s16", SE.layer_name("lenta", 16))
    with pytest.raises(PermissionError):
        SE.read_eval_rows("criteo", "v10a_full_s16", SE.layer_name("x5", 16))
    assert reads == []
    assert SE.read_eval_rows("x5", "v10a_full_s16", SE.layer_name("x5", 16)) == "rows"
    assert SE.read_eval_rows("lenta", "v10b_full_s64", SE.layer_name("lenta", 64)) == "rows"
    assert reads == ["x5", "lenta"]
