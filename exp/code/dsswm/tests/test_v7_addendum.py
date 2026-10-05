"""Tests for the lock-v7 addendum machinery (prereg_v7, lenta_v7, v7_analysis, v7_replica, run_r5s_v7 helpers)."""
from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from dsswm.envs.data_v6 import DATA_FILES
from dsswm.stats import prereg, prereg_v6, prereg_v7
from dsswm.stats import v7_analysis as VA
from dsswm.stats import v7_replica as R

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# --------------------------------------------------------------------------------------------- frozen FDC-BF
def test_fdc_bf_hyperparameters_frozen():
    from dsswm.baselines.fdc_bet import _LAMBDA_GRID, VARIANTS, make_variant
    m = make_variant("FDC-BF")
    assert (m.kind, m.var_box, m.fpc, m.rect, m.split) == ("bennett", "HG", True, False, (0.045, 0.005))
    assert VARIANTS["FDC-BF"] == ("bennett", "HG", True, False, (0.045, 0.005))
    assert len(_LAMBDA_GRID) == 161
    assert math.isclose(_LAMBDA_GRID[0], 1 / 200) and math.isclose(_LAMBDA_GRID[-1], 10.0)
    assert m.validity == "rigorous" and m.alloc_kind == "fixed"


def test_runner_make_and_methods():
    import run_r5s_v7 as RV
    assert RV.make("FDC-BF", {}).name == "FDC-BF"
    mr = RV.make("FDC-MR[front3]", {})
    assert mr.rho_name == "front3" and mr.kind == "min" and mr.rect and mr.partition
    assert set(RV.TASKS["v7a_full_a"]["methods"]) | set(RV.TASKS["v7a_full_b"]["methods"]) == set(VA.A_METHODS)
    assert set(RV.TASKS["v7b_full_a"]["methods"]) | set(RV.TASKS["v7b_full_b"]["methods"]) == set(VA.B_METHODS)
    for t in ("v7a_full_a", "v7a_full_b"):
        assert RV.TASKS[t]["seeds"] == list(range(33000, 33200)) and RV.TASKS[t]["stop_k"] == 15
    for t in ("v7b_full_a", "v7b_full_b"):
        assert RV.TASKS[t]["seeds"] == list(range(34000, 34200))
    for t in ("v7a_pilot", "v7b_pilot"):
        assert all(900 <= s < 1000 for s in RV.TASKS[t]["seeds"])
    assert set(R.GROUP_A) <= set(RV.TASKS["v7a_full_b"]["methods"])
    assert set(R.GROUP_B) <= set(RV.TASKS["v7b_full_b"]["methods"])


def test_x_rank():
    import run_r5s_v7 as RV
    ck = np.array([10, 100, 1000])
    U = [[1.0, 1.0, 1.0], [0.5, 0.01, 1.0], [0.001, 0.001, 0.001]]
    assert RV.x_rank(U, ck, 0.01, 1) == pytest.approx(100.0)
    v = RV.x_rank(U, ck, 0.01, 2)
    assert 100 < v <= 1000
    assert RV.x_rank(U[:2], ck, 0.01, 2) is None


# --------------------------------------------------------------------------------------------- prereg v7 gate
def _fake_v7(v5, v6):
    add = {"version": "5+v7-addendum", "status": "locked", "addendum_to": {"version": 5, "sha256": v5["sha256"]},
           "v6_addendum": {"sha256": v6["sha256"]}, "statement": "x", "primary_method": {}, "blocks": {},
           "seed_manifest": {}, "replica_check": {}, "eval_tasks": {"v7a_full_b": {}}, "frozen_configs": {"CR9": {}},
           "input_sha256": {"plan/prereg_lock.json": prereg._sha_file(prereg.LOCK_PATH)},
           "code_sha256": {"run_r5s_v7.py": "0" * 64}, "git_commit": "a" * 40,
           "data_sha256": {n: {"path": p, "sha256": "0" * 64} for n, p in DATA_FILES.items()}}
    add["sha256"] = prereg.canonical_hash(add)
    return add


def _kw():
    return dict(verify_code=False, verify_data=False)


def test_prereg_v7_gate_logic():
    v5 = prereg.load_lock(prereg.lock_path(5))
    v6 = prereg.load_lock(prereg_v6.ADDENDUM_PATH)
    add = _fake_v7(v5, v6)
    assert prereg_v7.check_addendum(add, "v7a_full_b", v5_lock=v5, v6_add=v6, **_kw()) is add
    with pytest.raises(RuntimeError):
        prereg_v7.check_addendum(add, "v7x_unknown", v5_lock=v5, v6_add=v6, **_kw())
    for mut in ("status", "v5", "v6", "tamper"):
        bad = copy.deepcopy(add)
        if mut == "status":
            bad["status"] = "draft"
        elif mut == "v5":
            bad["addendum_to"]["sha256"] = "0" * 64
        elif mut == "v6":
            bad["v6_addendum"]["sha256"] = "0" * 64
        if mut != "tamper":
            bad["sha256"] = prereg.canonical_hash(bad)
        else:
            bad["eval_tasks"]["v7b_full_b"] = {}
        with pytest.raises(RuntimeError):
            prereg_v7.check_addendum(bad, "v7a_full_b", v5_lock=v5, v6_add=v6, **_kw())


def test_v7_draft_never_authorises_and_v6_gate_rejects_v7_tasks():
    ok, _ = prereg_v7.addendum_gate("v7a_full_b", path=prereg_v7.DRAFT_PATH, verify_code=False)
    assert not ok
    ok, _ = prereg_v6.addendum_gate("v7a_full_b", verify_code=False, verify_data=False)
    assert not ok


def test_lr9_v7_eval_guard():
    from dsswm.envs.base import TruthAccessError
    from dsswm.envs.lenta_v6 import LentaLayerEnv
    from dsswm.envs.lenta_v7 import LentaLayerEnvV7
    ev = LentaLayerEnvV7("eval")
    assert ev.structure_only
    with pytest.raises(TruthAccessError):
        ev.true_mu()
    with pytest.raises(TruthAccessError):
        LentaLayerEnvV7("eval", eval_task_id="v6b_full_b")        # v7 gate: not a v7 task / no v7 lock
    if not prereg_v7.ADDENDUM_PATH.exists():
        with pytest.raises(TruthAccessError):
            LentaLayerEnvV7("eval", eval_task_id="v7b_full_b")
    dev7, dev6 = LentaLayerEnvV7("dev"), LentaLayerEnv("dev")
    assert np.array_equal(dev7.true_mu(), dev6.true_mu()) and np.array_equal(dev7.pool_sizes, dev6.pool_sizes)
    assert np.array_equal(ev.pool_sizes, LentaLayerEnv("eval").pool_sizes)


# --------------------------------------------------------------------------------------------- decision rule
def _pr(u):
    return {"ub95_one_sided": u}


def test_decide_block_a():
    ok = VA.decide_block_a({m: _pr(0.7) for m in VA.A_FAMILY}, 0, {"status": "pass"})
    assert ok["verdict"] == "positive_result_achieved" and not ok["failing_components"]
    d = VA.decide_block_a({"RECT-ck-HG": _pr(0.7), "RECT-ck-HG-live": _pr(0.7), "HC-WoR": _pr(0.80)}, 0,
                          {"status": "pass"})
    assert d["verdict"] == "positive_result_not_achieved" and d["failing_components"] == ["faster_below_1"]
    assert d["governing_rival"] == "HC-WoR"
    d = VA.decide_block_a({m: _pr(1.2) for m in VA.A_FAMILY}, 0, {"status": "pass"})
    assert d["failing_components"] == ["not_faster"]
    d = VA.decide_block_a({m: _pr(0.5) for m in VA.A_FAMILY}, 1, {"status": "pass"})
    assert d["failing_components"] == ["validity_failure"]
    d = VA.decide_block_a({m: _pr(0.5) for m in VA.A_FAMILY}, 0, {"status": "fail"})
    assert d["failing_components"] == ["replica_fail"]
    for bad in ({m: _pr(0.5) for m in VA.A_FAMILY[:2]}, {**{m: _pr(0.5) for m in VA.A_FAMILY}, "X": _pr(0.1)},
                {m: _pr(float("nan")) for m in VA.A_FAMILY}):
        with pytest.raises(ValueError):
            VA.decide_block_a(bad, 0, {"status": "pass"})
    with pytest.raises(ValueError):
        VA.decide_block_a({m: _pr(0.5) for m in VA.A_FAMILY}, 0, {"status": None})


def _rows(methods, seeds, eps=(0.001,), fn=None, tau=1000):
    out = []
    for m in methods:
        for e in eps:
            for s in seeds:
                v = 100.0 if fn is None else fn(m, s)
                out.append({"method": m, "seed": s, "eps": e, "N80_pen": v, "x12": v, "N100_pen": 2 * v,
                            "fwer_event": False, "n_false_at_k80": 0, "tau_R": tau, "schedule_digest": "d"})
    return out


def test_analyse_block_a_end_to_end():
    seeds = tuple(range(20))
    spec = dict(VA.BLOCK_SPEC["A"], seeds=seeds)
    rows = _rows(VA.A_METHODS, seeds, fn=lambda m, s: 60.0 + (s % 3) if m == "FDC-BF" else 100.0)
    res = VA.analyse_block("A", rows, {"status": "pass"}, spec=spec, B=500)
    assert res["decision"]["verdict"] == "positive_result_achieved"
    assert res["decision"]["UB_star"] < 0.8
    assert "FDC-MR[front3]|N100_pen" in res["per_eps"]["0.001"]
    rows[0]["fwer_event"] = True                    # FDC-BF row is first
    assert rows[0]["method"] == "FDC-BF"
    res = VA.analyse_block("A", rows, {"status": "pass"}, spec=spec, B=500)
    assert res["decision"]["failing_components"] == ["validity_failure"]
    with pytest.raises(ValueError):
        VA.analyse_block("A", rows[1:], {"status": "pass"}, spec=spec, B=500)


# --------------------------------------------------------------------------------------------- replica
def test_v7_replica_reconcile_and_tamper():
    methods = list(R.GROUP_A)
    seeds = list(range(12))
    rows = _rows(methods, seeds)
    rep_rows = [dict(r) for r in rows if r["seed"] < R.R1_N_SEEDS]
    rep = R.build_report("v7a_full_b", True, rows, rep_rows, methods, (0.001,), seeds, "f" * 64, {"x": "y"}, "d")
    assert rep["status"] == "pass"
    from dsswm.stats.v6_replica import content_hash
    sealed = content_hash(rows)
    assert R.validate_report(rep, "v7a_full_b", True, rows, rep_rows, methods, (0.001,), seeds, "f" * 64,
                             {"x": "y"}, "d", sealed_content_sha256=sealed) == "pass"
    bad = copy.deepcopy(rep_rows)
    bad[0]["N80_pen"] = 1.0
    with pytest.raises(ValueError):                 # report no longer reconciles
        R.validate_report(rep, "v7a_full_b", True, rows, bad, methods, (0.001,), seeds, "f" * 64, {"x": "y"}, "d",
                          sealed_content_sha256=sealed)
    rep2 = R.build_report("v7a_full_b", True, rows, bad, methods, (0.001,), seeds, "f" * 64, {"x": "y"}, "d")
    assert rep2["status"] == "fail"
    dig = copy.deepcopy(rows)
    dig[5]["schedule_digest"] = "other"
    rep3 = R.build_report("v7a_full_b", True, dig, [dict(r) for r in dig if r["seed"] < 10], methods, (0.001,),
                          seeds, "f" * 64, {"x": "y"}, "d")
    assert rep3["status"] == "fail"                 # R2 design identity
    edited = copy.deepcopy(rows)
    edited[3]["N80_pen"] = 1.0
    with pytest.raises(ValueError):                 # sealed hash
        R.validate_report(rep, "v7a_full_b", True, edited, rep_rows, methods, (0.001,), seeds, "f" * 64,
                          {"x": "y"}, "d", sealed_content_sha256=sealed)


# --------------------------------------------------------------------------------------------- external reviewer v7 review fixes
@pytest.fixture
def git_repo(tmp_path):
    import subprocess as sp
    root = tmp_path / "repo"
    (root / "seals").mkdir(parents=True)
    for args in (["init", "-q"], ["config", "user.email", "t@t"], ["config", "user.name", "t"],
                 ["commit", "-q", "--allow-empty", "-m", "init"]):
        sp.run(["git", *args], cwd=root, check=True, capture_output=True)
    return root


_SM, _SS = list(R.GROUP_A), list(range(12))


def _seal_rows(git_repo, rows):
    from dsswm.stats.v7_seal import make_seal, write_and_commit_seal
    return write_and_commit_seal(make_seal("v7a_full_b", rows, _SM, (0.001,), _SS, {"x": "1"}, "dd", "f" * 64),
                                 git_repo / "seals")


def _verify(git_repo, rows):
    from dsswm.stats.v7_seal import verify_seal
    return verify_seal(git_repo / "seals", "v7a_full_b", rows, _SM, (0.001,), _SS, {"x": "1"}, "dd", "f" * 64)


def test_v7_seal_honest_passes_and_reseal_refused(git_repo):
    rows = _rows(_SM, _SS)
    sha = _seal_rows(git_repo, rows)
    assert _verify(git_repo, rows)["seal_commit"] == sha
    edited = [dict(r, N80_pen=1.0) if (r["seed"] == 11 and r["method"] == "FDC-BF") else r for r in rows]
    with pytest.raises(RuntimeError):                          # external reviewer counterexample: reseal of an edited row
        _seal_rows(git_repo, edited)
    with pytest.raises(ValueError):
        _verify(git_repo, edited)


def test_v7_seal_recommitted_after_edit_refused(git_repo):
    """Attacker edits a row outside R1, deletes the seal files and re-seals with ordinary new commits (no history
    rewrite): the seal file now has 2 commits in its history -> refused."""
    import subprocess as sp
    rows = _rows(_SM, _SS)
    _seal_rows(git_repo, rows)
    edited = [dict(r, N80_pen=1.0) if (r["seed"] == 11 and r["method"] == "FDC-BF") else r for r in rows]
    for f in ("v7a_full_b.seal.json", "v7a_full_b.seal_ref.json"):
        (git_repo / "seals" / f).unlink()
    _seal_rows(git_repo, edited)
    with pytest.raises(ValueError, match="single original commit"):
        _verify(git_repo, edited)


def test_lock_history_anchor(git_repo):
    import subprocess as sp
    p = git_repo / "lock.json"
    p.write_text('{"a": 1}')
    with pytest.raises(RuntimeError):                          # not committed
        prereg_v7.lock_history_check(p)
    sp.run(["git", "add", "lock.json"], cwd=git_repo, check=True)
    sp.run(["git", "commit", "-q", "-m", "lock"], cwd=git_repo, check=True)
    assert len(prereg_v7.lock_history_check(p)) == 40
    p.write_text('{"a": 2}')
    with pytest.raises(RuntimeError):                          # edited working copy
        prereg_v7.lock_history_check(p)
    sp.run(["git", "commit", "-q", "-am", "edit"], cwd=git_repo, check=True)
    with pytest.raises(RuntimeError):                          # re-committed lock
        prereg_v7.lock_history_check(p)
